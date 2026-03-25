import math
import torch
import numpy as np
import odak
import os
import imageio
import pytorch3d
import torch.nn.functional as F
from typing import Tuple, Optional
from pytorch3d.ops.knn import knn_points
from pytorch3d.renderer.cameras import PerspectiveCameras
from utils import console_only_print

# Import the CUDA implementation
try:
    from cuda_prop.cov3d_cuda.python_import import (
        compute_cov3d_cuda, compute_jacobian_cuda, compute_cov2d_cuda,
        compute_means2d_cuda, invert_cov2d_cuda, splat_tile_cuda,
    )
    USE_COV3D_CUDA = True
    USE_JACOBIAN_CUDA = True
    USE_COV2D_CUDA = True
    USE_MEANS2D_CUDA = True
    USE_INVCOV2D_CUDA = True
    USE_SPLAT_TILE_CUDA = True
    print("Using CUDA implementation for 3D/2D covariance and related operations")
except ImportError as e:
    print(f"Fail {e}")
    USE_COV3D_CUDA = False
    USE_JACOBIAN_CUDA = False
    USE_COV2D_CUDA = False
    USE_MEANS2D_CUDA = False
    USE_INVCOV2D_CUDA = False
    USE_SPLAT_TILE_CUDA = False
    print("Using PyTorch implementation for 3D/2D covariance and related operations")

from cuda_prop import BandlimitedPropagation, compute_bmm_cuda, sum_last_dim_cuda, element_wise_multiplication_cuda
from argparse import Namespace
import sys

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, temperature=0.001):
        # Save temperature for backward pass
        ctx.save_for_backward(input)
        ctx.temperature = temperature
        indices = torch.argmax(input, dim=1) # one-hot
        return F.one_hot(indices, num_classes=input.size(1)).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temperature = ctx.temperature
        # Compute softmax gradients for backward pass
        soft_probs = F.softmax(input / temperature, dim=1)
        return grad_output * soft_probs, None  # None for temperature gradient

class StraightThroughEstimator(torch.nn.Module):
    def __init__(self, temperature=0.001):
        super(StraightThroughEstimator, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        return STEFunction.apply(x, self.temperature)

class Gaussians(torch.nn.Module):

    def __init__(
        self, init_type: str, device: str, load_path: Optional[str] = None,
        num_points: Optional[int] = None, args_prop: Namespace = None,
        pointcloud_data: Optional[dict] = None, generate_dense_point = False,
        densepoint_scatter = 0.01, img_size = None
    ):
        super(Gaussians, self).__init__()

        self.device = device
        self.num_planes = args_prop.num_planes
        self.generate_dense_point = generate_dense_point
        self.densepoint_scatter = densepoint_scatter
        self.NEAR_PLANE = 1.
        self.FAR_PLANE = 1000.0

        if init_type == "gaussians":
            if load_path is None:
                raise ValueError
            torch.cuda.empty_cache()
            data = self._load_gaussians(load_path)

        elif init_type == "random":
            if num_points is None:
                raise ValueError

            data = self._load_random(num_points, img_size)
        elif init_type == "point":
            if pointcloud_data is None:
                raise ValueError("Point cloud data is required for 'point' initialization")
            self.is_outdoor = args_prop.is_outdoor
            data = self._load_point(pointcloud_data)
        else:
            raise ValueError(f"Invalid init_type: {init_type}")


        # Instead of creating plain tensors, register them as nn.Parameter
        self.register_parameter('pre_act_quats', torch.nn.Parameter(data["pre_act_quats"], requires_grad=False))
        self.register_parameter('means', torch.nn.Parameter(data["means"], requires_grad=False))
        self.register_parameter('pre_act_scales', torch.nn.Parameter(data["pre_act_scales"], requires_grad=False))
        self.register_parameter('colours', torch.nn.Parameter(data["colours"], requires_grad=False))
        self.register_parameter('pre_act_phase', torch.nn.Parameter(data["pre_act_phase"], requires_grad=False))
        self.register_parameter('pre_act_opacities', torch.nn.Parameter(data["pre_act_opacities"], requires_grad=False))
        self.register_parameter('pre_act_plane_assignment', torch.nn.Parameter(data["pre_act_plane_assignment"], requires_grad=False))
        self.to_cuda()

    def to_cuda(self):
        # Convert all parameters to CUDA
        self.to(self.device)

    def __len__(self):
        return len(self.means)

    def _load_gaussians(self, ply_path: str):
        if ply_path.endswith('.pth'):
            checkpoint = torch.load(ply_path, map_location='cpu', weights_only=False)

            data = {
                "pre_act_quats": checkpoint["pre_act_quats"].clone().detach().to(torch.float32).contiguous(),
                "means": checkpoint["means"].clone().detach().to(torch.float32).contiguous(),
                "pre_act_scales": checkpoint["pre_act_scales"].clone().detach().to(torch.float32).contiguous(),
                "colours": checkpoint["colours"].clone().detach().to(torch.float32).contiguous(),
                "pre_act_phase": checkpoint["pre_act_phase"].clone().detach().to(torch.float32).contiguous(),
                "pre_act_opacities": checkpoint["pre_act_opacities"].clone().detach().to(torch.float32).contiguous(),
                "pre_act_plane_assignment": checkpoint["pre_act_plane_assignment"].clone().detach().to(torch.float32).contiguous()
            }

            num = len(data["means"])
            print(f"Loaded Gaussians {num} from checkpoint: {ply_path}")
            return data

    def _load_random(self, num_points: int, image_size=None):
        data = dict()

        # Option 1. nerf_synthetic dataset, randomly sample means in normal distribution
        # means = torch.randn((num_points, 3)).to(torch.float32) * 0.4  # (N, 3)
        # Option 2. nerf_synthetic dataset, uniformly sample means over entire scene
        means = (torch.rand((num_points, 3)) * 2 - 1).to(torch.float32) * 15.7
        # Option 2A. nerf_synthetic dataset, uniformly sample means over entire scene but just a plane, choi etal baseline
        # means = (torch.rand((num_points, 3)) * 2 - 1).to(torch.float32)
        # # means = torch.randn((num_points, 3)).to(torch.float32)
        # means[:, :2] *= 85.4  # x and y: wide range
        # means[:, 2] *= 0.1    # z: very narrow range, making it a flat plane
        # Option 3. llff dataset fern, uniformly sample means over entire scene
        # means = (torch.rand((num_points, 3)) * 2 - 1).to(torch.float32) * 1.9
        # Option 4. tandt dataset train, uniformly sample means over entire scene
        # means = (torch.rand((num_points, 3)) * 2 - 1)* 1.5
        data["means"] = means.to(torch.float32)

        data["colours"] = torch.rand((num_points, 3), dtype=torch.float32)  # (N, 3)
        quats_norm = torch.randn((num_points, 4), dtype=torch.float32)
        quats_norm = F.normalize(quats_norm, dim=1)  # Normalize to unit quaternions
        quats = torch.zeros((num_points, 4), dtype=torch.float32)  # (N, 4)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats + quats_norm * 0.01  # (N, 4)
        data["pre_act_scales"] = torch.log((torch.rand((num_points, 1), dtype=torch.float32) + 1e-6) * 0.01)
        data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)
        data["pre_act_phase"] = torch.randn((num_points, 3), dtype=torch.float32) # * 2 * odak.pi
        data["pre_act_opacities"] = torch.ones((num_points,), dtype=torch.float32)  # (N,)
        data["pre_act_plane_assignment"] = torch.randn((num_points, self.num_planes), dtype=torch.float32) * 10.0

        print(f"Loaded Randomly {num_points} gaussians with image size {image_size if image_size else 'default'}")
        return data

    def visualize_point_cloud(self, positions, colors, save_path="point_cloud_viz.png"):
        """
        Visualize the point cloud in 3D with matplotlib from 4 different perspectives.

        Args:
            positions: Tensor of point positions (N, 3)
            colors: Tensor of point colors (N, 3)
            save_path: Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            from itertools import product, combinations

            # Convert tensors to numpy arrays
            positions_np = positions.detach().cpu().numpy()
            colors_np = colors.detach().cpu().numpy()
            total_points = positions_np.shape[0]

            if total_points > 100000:
                # Calculate sampling rate
                sample_size = 100000
                sample_indices = np.random.choice(total_points, sample_size, replace=False)

                # Apply sampling
                positions_np = positions_np[sample_indices]
                colors_np = colors_np[sample_indices]

                print(f"Sampled point cloud from {total_points} to {sample_size} points for visualization")

            # Create figure with 2x2 subplots for different perspectives
            fig = plt.figure(figsize=(8, 8))

            # Define 4 different viewing angles
            view_angles = [
                (30, 30),   # Perspective 1
                (0, 0),     # Front view
                (0, 90),    # Side view
                (90, 0),    # Top view
            ]

            # Calculate bounds once for all subplots
            min_bounds = positions_np.min(axis=0)
            max_bounds = positions_np.max(axis=0)

            # Use the maximum range for all dimensions
            max_range = max(max_bounds - min_bounds)

            # Calculate midpoints
            mid_x = (max_bounds[0] + min_bounds[0]) * 0.5
            mid_y = (max_bounds[1] + min_bounds[1]) * 0.5
            mid_z = (max_bounds[2] + min_bounds[2]) * 0.5

            # Create each subplot with different perspective
            for i, (elev, azim) in enumerate(view_angles):
                ax = fig.add_subplot(2, 2, i+1, projection='3d')
                ax.scatter(
                    positions_np[:, 0], positions_np[:, 1], positions_np[:, 2],
                    c=colors_np, s=0.01, alpha=0.5
                )

                # Set view angle for this subplot
                ax.view_init(elev=elev, azim=azim)

                # Set equal aspect ratio using the max range for all axes
                ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
                ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
                ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

                # Set labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # Set title based on the perspective
                if i == 0:
                    ax.set_title(f'Isometric View ({len(positions_np)} points)')
                elif i == 1:
                    ax.set_title('Front View')
                elif i == 2:
                    ax.set_title('Side View')
                else:
                    ax.set_title('Top View')

            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"Point cloud visualization with 4 perspectives saved to {save_path}")

        except ImportError:
            print("Matplotlib or numpy not available for visualization")
        except Exception as e:
            print(f"Error visualizing point cloud: {e}")

    def _load_point(self, pointcloud_data: dict) -> dict:
        positions = pointcloud_data['positions']
        colors = pointcloud_data['colors']
        data = {}

        centre = positions.mean(dim=0, keepdim=True)
        distances = torch.norm(positions - centre, dim=1)

        if self.is_outdoor:
            num_points = positions.shape[0]
            num_points_to_keep = int(num_points * 0.98)
            sorted_indices = torch.argsort(distances)
            keep_indices = sorted_indices[:num_points_to_keep]
            print(f"Keeping {num_points_to_keep} points from {num_points} points")
            positions = positions[keep_indices]
            colors = colors[keep_indices]

        # -------------------------------------------------
        # Optional densification of the foreground geometry
        # -------------------------------------------------
        if self.generate_dense_point > 0:
            orig_positions = positions
            orig_colors = colors
            for _ in range(self.generate_dense_point):
                offset = torch.randn_like(orig_positions) * self.densepoint_scatter
                positions = torch.cat([positions, orig_positions + offset], dim=0)
                colors = torch.cat([colors, orig_colors], dim=0)

        if self.is_outdoor:
            divide = self.generate_dense_point if self.generate_dense_point > 0 else 1
            print(divide)
            divide = 1
            print(len(positions))
            bg_count = int(positions.shape[0] * (0.8 / divide))  # N % of (densified) cloud
            # bg_count = 50000
            print(f"randomize {bg_count} points for outdoor scene")
            centre = positions.mean(dim=0, keepdim=True)
            centred = positions - centre
            max_dist = torch.norm(centred, dim=1).max().item()

            # Compute principal directions of the pointcloud to orient the hemisphere correctly
            # Calculate covariance matrix of centered points
            cov = torch.matmul(centred.T, centred) / centred.shape[0]
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort eigenvectors by eigenvalues in descending order
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]

            # Use eigenvectors to define a coordinate system aligned with the point cloud
            # First eigenvector is the main axis, second is up direction, third is side direction
            main_axis = eigenvectors[:, 0]
            up_direction = eigenvectors[:, 1]
            side_direction = eigenvectors[:, 2]

            # Define our pole exclusion threshold
            pole_threshold = 0.35  # Adjust this value as needed
            valid_directions = []
            batch_size = bg_count * 3  # Sample in batches for efficiency

            while len(valid_directions) < bg_count:
                # Sample a batch of directions in a standard hemisphere
                directions_batch = F.normalize(torch.randn(batch_size, 3, device=positions.device), dim=1)

                # Apply the filter to exclude points near the poles
                valid_mask = torch.abs(directions_batch[:, 2]) <= pole_threshold
                valid_batch = directions_batch[valid_mask]

                # Add valid points to our collection
                valid_directions.append(valid_batch)
                all_valid = torch.cat(valid_directions, dim=0)

                # If we have enough valid points, truncate to exact count needed
                if all_valid.size(0) >= bg_count:
                    directions_standard = all_valid[:bg_count]
                    break

            # Transform directions from standard hemisphere to point cloud-aligned hemisphere
            # Create rotation matrix from standard basis to eigenvector basis
            rotation_matrix = torch.stack([main_axis, up_direction, side_direction], dim=1)

            # Apply rotation to transform hemisphere directions
            directions = torch.matmul(directions_standard, rotation_matrix.T)

            # Sample radii in an annulus outside the main cloud
            radii = torch.empty(bg_count, device=positions.device).uniform_(
                max_dist * 0.5, max_dist * 0.8
            )

            bg_positions = directions * radii.unsqueeze(1) + centre

            # colour the background with the palette of the farthest 10 % points
            dist = torch.norm(centred, dim=1)
            far_mask = dist >= torch.quantile(dist, 0.9)
            candidate_colors = colors[far_mask] if far_mask.any() else colors
            rand_idx = torch.randint(0, candidate_colors.shape[0], (bg_count,),
                                    device=positions.device)
            # bg_colors = candidate_colors[rand_idx]
            bg_colors = torch.rand((bg_count, 3), dtype=torch.float32).to(positions.device)
            positions = torch.cat([positions, bg_positions.to(positions.device)], dim=0)
            colors = torch.cat([colors, bg_colors], dim=0)
            # positions, colors = bg_positions, bg_colors
        # -------------------------------------------------
        # Visualise and pack everything into the output dict
        # -------------------------------------------------
        vis = False
        if vis:
            self.visualize_point_cloud(positions, colors, save_path="initial_pointcloud.png")
        total_points = positions.shape[0]
        print(f"Total points in original point cloud: {total_points}")

        data["means"] = positions.to(torch.float32).contiguous()
        data["colours"] = colors.to(torch.float32).contiguous()

        quats_norm = F.normalize(torch.randn((total_points, 4), dtype=torch.float32), dim=1)
        quats = torch.zeros((total_points, 4), dtype=torch.float32)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats + quats_norm * 0.01

        scales = torch.log((torch.rand((total_points, 1), dtype=torch.float32) + 1e-6) * 0.01)
        data["pre_act_scales"] = scales.repeat(1, 3)
        data["pre_act_phase"] = torch.randn((total_points, 3), dtype=torch.float32)
        data["pre_act_opacities"] = torch.ones(total_points, dtype=torch.float32)
        data["pre_act_plane_assignment"] = torch.randn(
            (total_points, self.num_planes), dtype=torch.float32
        ) * 10.0

        print(f"Initialized {total_points} Gaussians from point cloud data")
        return data

    def check_if_trainable(self):
        attrs = ["means", "pre_act_scales", "colours", "pre_act_phase", "pre_act_opacities", "pre_act_plane_assignment"]
        attrs += ["pre_act_quats"]

        for attr in attrs:
            param = getattr(self, attr)
            if not getattr(param, "requires_grad", False):
                raise Exception("Please use function make_trainable to make parameters trainable")

    def compute_cov_3D(self, quats: torch.Tensor, scales: torch.Tensor):
        # Use CUDA implementation if available
        if USE_COV3D_CUDA:
            cov_3D = compute_cov3d_cuda(quats, scales)
        else:
            # Original PyTorch implementation
            Is = torch.eye(scales.size(1)).to(scales.device)
            scale_mats = (scales.unsqueeze(2).expand(*scales.size(), scales.size(1))) * Is
            rots = pytorch3d.transforms.quaternion_to_matrix(quats)
            cov_3D = torch.matmul(rots, scale_mats)
            cov_3D = torch.matmul(cov_3D, torch.transpose(scale_mats, 1, 2))
            cov_3D = torch.matmul(cov_3D, torch.transpose(rots, 1, 2))
            del Is, scale_mats, rots
        return cov_3D

    def _compute_jacobian(self, cam_means_3D: torch.Tensor, fx, fy, img_size: Tuple):
        """
        Compute the Jacobian matrix for projection from 3D to 2D with near/far plane clipping.
        """
        W, H = img_size

        # Use CUDA implementation if available
        if USE_JACOBIAN_CUDA:
            J = compute_jacobian_cuda(cam_means_3D, fx, fy, img_size, self.NEAR_PLANE, self.FAR_PLANE)
        else:
            # Original PyTorch implementation with clipping planes
            half_tan_fov_x = 0.5 * W / fx
            half_tan_fov_y = 0.5 * H / fy

            means_view_space = cam_means_3D

            tx = means_view_space[:, 0]
            ty = means_view_space[:, 1]
            tz = means_view_space[:, 2]
            tz2 = tz*tz

            # Clipping plane check - zero out Jacobian for points outside the clipping planes
            clipping_mask = (tz > self.NEAR_PLANE) & (tz < self.FAR_PLANE)

            lim_x = 1.3 * half_tan_fov_x
            lim_y = 1.3 * half_tan_fov_y

            # Clamp points to view frustum
            tx = torch.clamp(tx/tz, -lim_x, lim_x) * tz
            ty = torch.clamp(ty/tz, -lim_y, lim_y) * tz

            J = torch.zeros((len(tx), 2, 3), device=self.device)

            # Calculate Jacobian entries
            J[:, 0, 0] = fx / tz
            J[:, 1, 1] = fy / tz
            J[:, 0, 2] = -(fx * tx) / tz2
            J[:, 1, 2] = -(fy * ty) / tz2

            # Apply clipping mask to zero out entries outside clipping planes
            clipping_mask = clipping_mask.to(torch.float32).view(-1, 1, 1)
            J = J * clipping_mask

            del means_view_space, tx, ty, tz, tz2

        return J  # (N, 2, 3)

    def compute_cov_2D(
        self, cam_means_3D: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
        fx, fy, R, img_size: Tuple
    ):
        """
        Computes the covariance matrices of 2D Gaussians using equation (5) of the 3D
        Gaussian Splatting paper.

        Link: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf
        """
        # Use CUDA implementation if available
        if USE_COV2D_CUDA:
            # Extract a single view matrix with shape (3, 3) from R
            view_matrix = R[0] if R.dim() == 3 else R
            cov_2D = compute_cov2d_cuda(cam_means_3D, quats, scales, view_matrix, fx, fy, img_size, self.NEAR_PLANE, self.FAR_PLANE)
        else:
            # Original PyTorch implementation
            J = self._compute_jacobian(cam_means_3D, fx, fy, img_size)
            N = J.shape[0]

            W = R.repeat(N, 1, 1)

            cov_3D = self.compute_cov_3D(quats, scales)  # (N, 3, 3)

            cov_2D = torch.matmul(J, W)
            cov_2D = torch.matmul(cov_2D, cov_3D)
            cov_2D = torch.matmul(cov_2D, torch.transpose(W, 1, 2))
            cov_2D = torch.matmul(cov_2D, torch.transpose(J, 1, 2))

            cov_2D[:, 0, 0] += 0.3
            cov_2D[:, 1, 1] += 0.3
            del J, W, cov_3D

        return cov_2D

    def compute_means_2D(self, cam_means_3D: torch.Tensor, fx, fy, px, py):
        """
        Projects 3D points to 2D with near/far plane clipping.
        """

        # Use CUDA implementation if available
        if USE_MEANS2D_CUDA:
            means_2D = compute_means2d_cuda(cam_means_3D, fx, fy, px, py, self.NEAR_PLANE, self.FAR_PLANE)
        else:
            # Original PyTorch implementation with clipping planes
            clipping_mask = (cam_means_3D[:, 2] > self.NEAR_PLANE) & (cam_means_3D[:, 2] < self.FAR_PLANE)

            # Compute inverse z for perspective division
            inv_z = 1.0 / cam_means_3D[:, 2].unsqueeze(1)
            cam_means_3D_xy = -cam_means_3D[:, :2] * inv_z

            # Combine operations for projection
            means_2D = torch.empty((cam_means_3D.shape[0], 2), device=cam_means_3D.device)
            means_2D[:, 0] = fx * cam_means_3D_xy[:, 0] + px
            means_2D[:, 1] = fy * cam_means_3D_xy[:, 1] + py

            # Set points outside clipping planes to a large value (effectively discarding them)
            large_value = 1e6
            means_2D[~clipping_mask] = large_value

            del cam_means_3D_xy, inv_z
        return means_2D

    # Update the invert_cov_2D method in Gaussians class
    @staticmethod
    def invert_cov_2D(cov_2D: torch.Tensor):
        # Use CUDA implementation if available
        if USE_INVCOV2D_CUDA:
            cov_2D_inverse = invert_cov2d_cuda(cov_2D)
        else:
            # Original PyTorch implementation
            determinants = cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
            determinants = determinants[:, None, None]  # (N, 1, 1)

            cov_2D_inverse = torch.zeros_like(cov_2D)  # (N, 2, 2)
            cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
            cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
            cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
            cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]

            cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse
            del determinants
        return cov_2D_inverse

    @staticmethod
    def calculate_gaussian_bounds(means_2D, cov_2D, img_size, confidence=3.0):
        """
        Calculate the bounding boxes for each Gaussian based on covariance

        Args:
            means_2D: 2D positions of Gaussians, tensor of shape (N, 2)
            cov_2D: 2D covariance matrices, tensor of shape (N, 2, 2)
            img_size: image dimensions (width, height)
            confidence: number of standard deviations to include in bounding box

        Returns:
            bounds: tensor of shape (N, 4) with [min_x, min_y, max_x, max_y] for each Gaussian
        """
        # Extract variances (diagonal elements of covariance matrices)
        var_x = cov_2D[:, 0, 0]  # σ²x, shape (N,)
        var_y = cov_2D[:, 1, 1]  # σ²y, shape (N,)

        # Calculate standard deviations
        std_x = torch.sqrt(var_x)  # σx
        std_y = torch.sqrt(var_y)  # σy

        # Calculate bounds (confidence * standard deviation in each direction)
        radius_x = confidence * std_x
        radius_y = confidence * std_y

        # Create bounding boxes [min_x, min_y, max_x, max_y]
        min_x = means_2D[:, 0] - radius_x
        min_y = means_2D[:, 1] - radius_y
        max_x = means_2D[:, 0] + radius_x
        max_y = means_2D[:, 1] + radius_y

        # Clamp bounds to image dimensions
        W, H = img_size
        min_x = torch.clamp(min_x, 0, W-1)
        min_y = torch.clamp(min_y, 0, H-1)
        max_x = torch.clamp(max_x, 0, W-1)
        max_y = torch.clamp(max_y, 0, H-1)

        bounds = torch.stack([min_x, min_y, max_x, max_y], dim=1)
        return bounds

    @staticmethod
    def apply_activations(pre_act_quats, pre_act_scales, pre_act_phase=None, pre_act_opacities=None, pre_act_plane_assignment=None,
                          step=None, max_step=None):
        # Convert logscales to scales
        scales = torch.exp(pre_act_scales)

        # Normalize quaternions
        quats = torch.nn.functional.normalize(pre_act_quats)
        phase = pre_act_phase % (2.0 * odak.pi)
        opacities = torch.sigmoid(pre_act_opacities)

        ste = StraightThroughEstimator()
        plane_probs = ste(pre_act_plane_assignment)

        return quats, scales, phase, opacities, plane_probs

    def save_gaussians(self, save_path: str):
        state_dict = {
            'pre_act_quats': self.pre_act_quats.cpu(),
            'means': self.means.cpu(),
            'pre_act_scales': self.pre_act_scales.cpu(),
            'colours': self.colours.cpu(),
            'pre_act_phase': self.pre_act_phase.cpu(),
            # 'spherical_harmonics': self.spherical_harmonics.cpu(),
            # 'zernike_coeffs': self.zernike_coeffs.cpu(),
            'pre_act_opacities': self.pre_act_opacities.cpu(),
            'pre_act_plane_assignment': self.pre_act_plane_assignment.cpu()
        }

        torch.save(state_dict, save_path)
        print(f"Gaussians saved to {save_path}")

    def density_control(
            self,
            grad_threshold: float = 0.000005,
            opacity_threshold: float = 0.001,
            small_scale_threshold: float = 0.01,
            large_scale_threshold: float = 0.05,
            split_factor: float = 2,
            clone_offset_factor: float = 0.05,
        ):
            """
            Adaptive control of Gaussians as described in the paper.
            Two main operations:
            1. Clone small-scale Gaussians in under-reconstructed regions (high gradient, small scale)
            2. Split large Gaussians in over-reconstructed regions (high gradient, large scale)

            Also prunes Gaussians with opacity below a threshold.
            Revising Densification in Gaussian Splatting ECCV 2024

            Args:
                grad_threshold: Minimum gradient magnitude to trigger clone/split
                opacity_threshold: Gaussians with opacity below this threshold are pruned
                small_scale_threshold: Maximum scale to be considered "small" for cloning
                large_scale_threshold: Minimum scale to be considered "large" for splitting
                split_factor: Factor by which to divide scale when splitting
                clone_offset_factor: Factor determining how far to offset cloned Gaussians
            """
            # Check if gradient computation has been performed
            if self.means.grad is None:
                print("[adaptive_control] No .grad on self.means. Call backward() first.")
                return
            self.is_isotropic = False
            # Get the magnitude of position gradients and current scales
            grad_means = self.means.grad  # shape: (N, 3)
            grad_mag = grad_means.norm(dim=1)  # shape: (N,)
            raw_scales = torch.exp(self.pre_act_scales)  # shape: (N, 3) or (N, 1)
            avg_scale = raw_scales.mean(dim=1)  # shape: (N,)

            # Get current opacities for pruning
            opacities = torch.sigmoid(self.pre_act_opacities)

            orig_count = len(self.means)

            # Create masks for different operations
            large_grad_mask = grad_mag >= grad_threshold
            clone_mask = large_grad_mask & (avg_scale < small_scale_threshold)
            split_mask = large_grad_mask & (avg_scale > large_scale_threshold)
            prune_mask = opacities < opacity_threshold

            # Final keep mask (don't keep pruned Gaussians, but do keep those that will be cloned/split)
            keep_mask = ~prune_mask

            # Make sure masks are on the same device as tensors
            device = self.means.device
            keep_mask = keep_mask.to(device)
            clone_mask = clone_mask.to(device)
            split_mask = split_mask.to(device)
            prune_mask = prune_mask.to(device)

            num_keep = keep_mask.sum().item()
            num_clone = clone_mask.sum().item()
            num_split = split_mask.sum().item()
            num_prune = prune_mask.sum().item()

            print("---------------------  ")
            print(f"  Adaptive Control: grad_threshold={grad_threshold}")
            print(f"  Large gradient Gaussians: {large_grad_mask.sum().item()}")
            print(f"  Small scale (clone candidates): {clone_mask.sum().item()}")
            print(f"  Large scale (split candidates): {split_mask.sum().item()}")
            print(f"  Low opacity (pruned): {prune_mask.sum().item()}")

            # Extract data for Gaussians we're keeping
            means_keep = self.means[keep_mask]
            scales_keep = self.pre_act_scales[keep_mask]
            quats_keep = self.pre_act_quats[keep_mask]
            phase_keep = self.pre_act_phase[keep_mask]
            colours_keep = self.colours[keep_mask]
            pre_act_opacities_keep = self.pre_act_opacities[keep_mask]
            pre_act_plane_assignment_keep = self.pre_act_plane_assignment[keep_mask]

            # We also store old grads for inheritance:
            grad_means_keep = self.means.grad[keep_mask]
            grad_scales_keep = self.pre_act_scales.grad[keep_mask]
            grad_quats_keep = self.pre_act_quats.grad[keep_mask] if not self.is_isotropic else None
            grad_phase_keep = self.pre_act_phase.grad[keep_mask]
            grad_colours_keep = self.colours.grad[keep_mask]
            grad_pre_act_opacities_keep = self.pre_act_opacities.grad[keep_mask]

            if self.pre_act_plane_assignment.grad is not None:
                grad_pre_act_plane_assignment_keep = self.pre_act_plane_assignment.grad[keep_mask]
            else:
                grad_pre_act_plane_assignment_keep = torch.zeros_like(self.pre_act_plane_assignment)[keep_mask]

            # HANDLE CLONING (under-reconstruction case)
            # Find Gaussians in under-reconstructed regions (high gradient but small scale)
            clone_idx = torch.where(clone_mask & keep_mask)[0]  # only clone from kept Gaussians

            # Initialize variables for cloning
            means_clone = torch.empty((0, 3), device=device, dtype=self.means.dtype)
            scales_clone = torch.empty((0, self.pre_act_scales.size(1)), device=device, dtype=self.pre_act_scales.dtype)
            quats_clone = torch.empty((0, 4), device=device, dtype=self.pre_act_quats.dtype)
            phase_clone = torch.empty((0, 3), device=device, dtype=self.pre_act_phase.dtype)
            colours_clone = torch.empty((0, 3), device=device, dtype=self.colours.dtype)
            pre_act_opacities_clone = torch.empty((0,), device=device, dtype=self.pre_act_opacities.dtype)
            pre_act_plane_assignment_clone = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.dtype)

            # Also for grads
            grad_means_clone = torch.empty((0, 3), device=device, dtype=self.means.grad.dtype)
            grad_scales_clone = torch.empty((0, self.pre_act_scales.size(1)), device=device, dtype=self.pre_act_scales.grad.dtype)
            grad_quats_clone = None if self.is_isotropic else torch.empty((0, 4), device=device, dtype=self.pre_act_quats.grad.dtype)
            grad_phase_clone = torch.empty((0, 3), device=device, dtype=self.pre_act_phase.grad.dtype)
            grad_colours_clone = torch.empty((0, 3), device=device, dtype=self.colours.grad.dtype)

            # Handle potential None gradients for clone variables
            if self.pre_act_opacities.grad is not None:
                grad_pre_act_opacities_clone = torch.empty((0,), device=device, dtype=self.pre_act_opacities.grad.dtype)
            else:
                grad_pre_act_opacities_clone = torch.empty((0,), device=device, dtype=self.pre_act_opacities.dtype)

            if self.pre_act_plane_assignment.grad is not None:
                grad_pre_act_plane_assignment_clone = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.grad.dtype)
            else:
                grad_pre_act_plane_assignment_clone = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.dtype)

            # If we have clone candidates, process them
            if len(clone_idx) > 0:
                means_clone = self.means[clone_idx]
                scales_clone = self.pre_act_scales[clone_idx]
                quats_clone = self.pre_act_quats[clone_idx]
                phase_clone = self.pre_act_phase[clone_idx]
                colours_clone = self.colours[clone_idx]

                # Get original pre-activation opacities
                original_pre_act_opacities = self.pre_act_opacities[clone_idx]

                # Get original sigmoid opacities
                original_opacities = torch.sigmoid(original_pre_act_opacities)

                # Apply opacity correction formula: α̂ = 1 - √(1 - α)
                corrected_opacities = 1.0 - torch.sqrt(1.0 - original_opacities)

                # Convert back to pre-activation space (inverse of sigmoid)
                epsilon = 1e-6  # for numerical stability
                corrected_pre_act_opacities = torch.log(
                    corrected_opacities / (1.0 - corrected_opacities + epsilon) + epsilon
                )

                # Use the corrected opacities for clones
                pre_act_opacities_clone = corrected_pre_act_opacities
                pre_act_plane_assignment_clone = self.pre_act_plane_assignment[clone_idx]

                # Offset new means along the gradient direction
                grad_clone = grad_means[clone_idx]
                grad_norms = grad_clone.norm(dim=1, keepdim=True).clamp(min=1e-8)
                offset = (clone_offset_factor * grad_clone / grad_norms)
                means_clone_new = means_clone + offset

                # Free memory
                del grad_clone, grad_norms, offset, original_opacities, corrected_opacities

                # Inherit old grads
                grad_means_clone = grad_means[clone_idx]
                grad_scales_clone = self.pre_act_scales.grad[clone_idx]
                grad_quats_clone = self.pre_act_quats.grad[clone_idx] if not self.is_isotropic else None
                grad_phase_clone = self.pre_act_phase.grad[clone_idx]
                grad_colours_clone = self.colours.grad[clone_idx]

                # Handle potential None gradients
                if self.pre_act_opacities.grad is not None:
                    grad_pre_act_opacities_clone = self.pre_act_opacities.grad[clone_idx]
                else:
                    grad_pre_act_opacities_clone = torch.zeros_like(self.pre_act_opacities)[clone_idx]

                if self.pre_act_plane_assignment.grad is not None:
                    grad_pre_act_plane_assignment_clone = self.pre_act_plane_assignment.grad[clone_idx]
                else:
                    grad_pre_act_plane_assignment_clone = torch.zeros_like(self.pre_act_plane_assignment)[clone_idx]

                # Reassign means_clone to the new offset values
                means_clone = means_clone_new

            # HANDLE SPLITTING (over-reconstruction case)
            # Find Gaussians in over-reconstructed regions (high gradient and large scale)
            split_idx = torch.where(split_mask & keep_mask)[0]  # only split from kept Gaussians

            # Initialize variables for splitting
            means_split_1 = torch.empty((0, 3), device=device, dtype=self.means.dtype)
            means_split_2 = torch.empty((0, 3), device=device, dtype=self.means.dtype)
            scales_split_1 = torch.empty((0, self.pre_act_scales.size(1)), device=device, dtype=self.pre_act_scales.dtype)
            scales_split_2 = torch.empty((0, self.pre_act_scales.size(1)), device=device, dtype=self.pre_act_scales.dtype)
            quats_split_1 = torch.empty((0, 4), device=device, dtype=self.pre_act_quats.dtype)
            quats_split_2 = torch.empty((0, 4), device=device, dtype=self.pre_act_quats.dtype)
            phase_split_1 = torch.empty((0, 3), device=device, dtype=self.pre_act_phase.dtype)
            phase_split_2 = torch.empty((0, 3), device=device, dtype=self.pre_act_phase.dtype)
            colours_split_1 = torch.empty((0, 3), device=device, dtype=self.colours.dtype)
            colours_split_2 = torch.empty((0, 3), device=device, dtype=self.colours.dtype)
            pre_act_opacities_split_1 = torch.empty((0,), device=device, dtype=self.pre_act_opacities.dtype)
            pre_act_opacities_split_2 = torch.empty((0,), device=device, dtype=self.pre_act_opacities.dtype)
            pre_act_plane_assignment_split_1 = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.dtype)
            pre_act_plane_assignment_split_2 = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.dtype)

            # Also for grads
            grad_means_split_1 = torch.empty((0, 3), device=device, dtype=self.means.grad.dtype)
            grad_means_split_2 = torch.empty((0, 3), device=device, dtype=self.means.grad.dtype)
            grad_scales_split_1 = torch.empty((0, self.pre_act_scales.size(1)), device=device, dtype=self.pre_act_scales.grad.dtype)
            grad_scales_split_2 = torch.empty((0, self.pre_act_scales.size(1)), device=device, dtype=self.pre_act_scales.grad.dtype)
            grad_quats_split_1 = None if self.is_isotropic else torch.empty((0, 4), device=device, dtype=self.pre_act_quats.grad.dtype)
            grad_quats_split_2 = None if self.is_isotropic else torch.empty((0, 4), device=device, dtype=self.pre_act_quats.grad.dtype)
            grad_phase_split_1 = torch.empty((0, 3), device=device, dtype=self.pre_act_phase.grad.dtype)
            grad_phase_split_2 = torch.empty((0, 3), device=device, dtype=self.pre_act_phase.grad.dtype)
            grad_colours_split_1 = torch.empty((0, 3), device=device, dtype=self.colours.grad.dtype)
            grad_colours_split_2 = torch.empty((0, 3), device=device, dtype=self.colours.grad.dtype)

            # Handle potential None gradients for split variables
            if self.pre_act_opacities.grad is not None:
                grad_pre_act_opacities_split_1 = torch.empty((0,), device=device, dtype=self.pre_act_opacities.grad.dtype)
                grad_pre_act_opacities_split_2 = torch.empty((0,), device=device, dtype=self.pre_act_opacities.grad.dtype)
            else:
                grad_pre_act_opacities_split_1 = torch.empty((0,), device=device, dtype=self.pre_act_opacities.dtype)
                grad_pre_act_opacities_split_2 = torch.empty((0,), device=device, dtype=self.pre_act_opacities.dtype)

            if self.pre_act_plane_assignment.grad is not None:
                grad_pre_act_plane_assignment_split_1 = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.grad.dtype)
                grad_pre_act_plane_assignment_split_2 = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.grad.dtype)
            else:
                grad_pre_act_plane_assignment_split_1 = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.dtype)
                grad_pre_act_plane_assignment_split_2 = torch.empty((0, self.num_planes), device=device, dtype=self.pre_act_plane_assignment.dtype)

            # If we have split candidates, process them
            if len(split_idx) > 0:
                means_split = self.means[split_idx]
                scales_split = self.pre_act_scales[split_idx]
                quats_split = self.pre_act_quats[split_idx]
                phase_split = self.pre_act_phase[split_idx]
                colours_split = self.colours[split_idx]
                pre_act_opacities_split = self.pre_act_opacities[split_idx]
                pre_act_plane_assignment_split = self.pre_act_plane_assignment[split_idx]

                grad_means_split = grad_means[split_idx]
                grad_scales_split = self.pre_act_scales.grad[split_idx]
                grad_quats_split = self.pre_act_quats.grad[split_idx] if not self.is_isotropic else None
                grad_phase_split = self.pre_act_phase.grad[split_idx]
                grad_colours_split = self.colours.grad[split_idx]

                # Handle potential None gradients
                if self.pre_act_opacities.grad is not None:
                    grad_pre_act_opacities_split = self.pre_act_opacities.grad[split_idx]
                else:
                    grad_pre_act_opacities_split = torch.zeros_like(self.pre_act_opacities)[split_idx]

                if self.pre_act_plane_assignment.grad is not None:
                    grad_pre_act_plane_assignment_split = self.pre_act_plane_assignment.grad[split_idx]
                else:
                    grad_pre_act_plane_assignment_split = torch.zeros_like(self.pre_act_plane_assignment)[split_idx]

                # Reduce the scale for the split Gaussians
                scale_down = scales_split - math.log(split_factor)

                if self.is_isotropic:
                    # shape => (Nsplit,1)
                    stdev = torch.exp(scales_split).view(-1,1) * 0.5
                else:
                    # shape => (Nsplit,)
                    stdev = torch.exp(scales_split).mean(dim=1, keepdim=True) * 0.5

                # Build rotation from quaternions to rotate the random offsets
                R_split = pytorch3d.transforms.quaternion_to_matrix(
                    F.normalize(quats_split, dim=1)
                )  # shape => (Nsplit,3,3)

                # Create new positions for the two children Gaussians
                # Child 1: with positive offset
                noise_1 = torch.randn_like(means_split) * stdev
                noise_1 = torch.bmm(R_split, noise_1.unsqueeze(-1)).squeeze(-1)
                means_split_1 = means_split + noise_1

                # Child 2: with negative offset
                noise_2 = -noise_1  # Use the opposite direction
                means_split_2 = means_split + noise_2

                # New scales for both children
                scales_split_1 = scale_down.clone()
                scales_split_2 = scale_down.clone()

                # Same orientation, colour, phase, etc. for both children
                quats_split_1 = quats_split.clone()
                quats_split_2 = quats_split.clone()

                phase_split_1 = phase_split.clone()
                phase_split_2 = phase_split.clone()

                colours_split_1 = colours_split.clone()
                colours_split_2 = colours_split.clone()

                # For splitting, we keep the original opacity as stated in the paper:
                # "For this reason, we stick to the standard rule of preserving the opacity of a primitive we split."
                pre_act_opacities_split_1 = pre_act_opacities_split.clone()
                pre_act_opacities_split_2 = pre_act_opacities_split.clone()

                pre_act_plane_assignment_split_1 = pre_act_plane_assignment_split.clone()
                pre_act_plane_assignment_split_2 = pre_act_plane_assignment_split.clone()

                # Gradient inheritance
                grad_means_split_1 = grad_means_split.clone() * 0.5
                grad_means_split_2 = grad_means_split.clone() * 0.5

                grad_scales_split_1 = grad_scales_split.clone() * 0.5
                grad_scales_split_2 = grad_scales_split.clone() * 0.5

                grad_quats_split_1 = grad_quats_split.clone() * 0.5 if not self.is_isotropic else None
                grad_quats_split_2 = grad_quats_split.clone() * 0.5 if not self.is_isotropic else None

                grad_phase_split_1 = grad_phase_split.clone() * 0.5
                grad_phase_split_2 = grad_phase_split.clone() * 0.5

                grad_colours_split_1 = grad_colours_split.clone() * 0.5
                grad_colours_split_2 = grad_colours_split.clone() * 0.5

                # Handle potential None gradients for split
                if self.pre_act_opacities.grad is not None:
                    grad_pre_act_opacities_split_1 = grad_pre_act_opacities_split.clone() * 0.5
                    grad_pre_act_opacities_split_2 = grad_pre_act_opacities_split.clone() * 0.5
                else:
                    grad_pre_act_opacities_split_1 = torch.zeros_like(pre_act_opacities_split_1)
                    grad_pre_act_opacities_split_2 = torch.zeros_like(pre_act_opacities_split_2)

                if self.pre_act_plane_assignment.grad is not None:
                    grad_pre_act_plane_assignment_split_1 = grad_pre_act_plane_assignment_split.clone() * 0.5
                    grad_pre_act_plane_assignment_split_2 = grad_pre_act_plane_assignment_split.clone() * 0.5
                else:
                    grad_pre_act_plane_assignment_split_1 = torch.zeros_like(pre_act_plane_assignment_split_1)
                    grad_pre_act_plane_assignment_split_2 = torch.zeros_like(pre_act_plane_assignment_split_2)

            # COMBINE ALL GAUSSIANS
            # Concatenate the kept, cloned, and split Gaussians
            new_means = torch.cat(
                [means_keep, means_clone, means_split_1, means_split_2], dim=0
            )
            new_scales = torch.cat(
                [scales_keep, scales_clone, scales_split_1, scales_split_2], dim=0
            )
            new_quats = torch.cat(
                [quats_keep, quats_clone, quats_split_1, quats_split_2], dim=0
            )
            new_phase = torch.cat(
                [phase_keep, phase_clone, phase_split_1, phase_split_2], dim=0
            )
            new_colours = torch.cat(
                [colours_keep, colours_clone, colours_split_1, colours_split_2], dim=0
            )
            new_pre_act_opacities = torch.cat(
                [pre_act_opacities_keep, pre_act_opacities_clone,
                pre_act_opacities_split_1, pre_act_opacities_split_2], dim=0
            )
            new_pre_act_plane_assignment = torch.cat(
                [pre_act_plane_assignment_keep, pre_act_plane_assignment_clone,
                pre_act_plane_assignment_split_1, pre_act_plane_assignment_split_2], dim=0
            )

            # Also reassign gradients
            new_grad_means = torch.cat(
                [grad_means_keep, grad_means_clone, grad_means_split_1, grad_means_split_2], dim=0
            )
            new_grad_scales = torch.cat(
                [grad_scales_keep, grad_scales_clone, grad_scales_split_1, grad_scales_split_2], dim=0
            )

            new_grad_phase = torch.cat(
                [grad_phase_keep, grad_phase_clone, grad_phase_split_1, grad_phase_split_2], dim=0
            )
            new_grad_colours = torch.cat(
                [grad_colours_keep, grad_colours_clone, grad_colours_split_1, grad_colours_split_2], dim=0
            )
            new_grad_pre_act_opacities = torch.cat(
                [grad_pre_act_opacities_keep, grad_pre_act_opacities_clone,
                grad_pre_act_opacities_split_1, grad_pre_act_opacities_split_2], dim=0
            )
            new_grad_pre_act_plane_assignment = torch.cat(
                [grad_pre_act_plane_assignment_keep, grad_pre_act_plane_assignment_clone,
                grad_pre_act_plane_assignment_split_1, grad_pre_act_plane_assignment_split_2], dim=0
            )

            if not self.is_isotropic:
                new_grad_quats = torch.cat(
                    [grad_quats_keep, grad_quats_clone, grad_quats_split_1, grad_quats_split_2], dim=0
                )
            else:
                new_grad_quats = None

            # Update the model parameters
            self.means = torch.nn.Parameter(new_means)
            self.pre_act_scales = torch.nn.Parameter(new_scales)
            self.pre_act_quats = torch.nn.Parameter(new_quats)
            self.pre_act_phase = torch.nn.Parameter(new_phase)
            self.colours = torch.nn.Parameter(new_colours)
            self.pre_act_opacities = torch.nn.Parameter(new_pre_act_opacities)
            self.pre_act_plane_assignment = torch.nn.Parameter(new_pre_act_plane_assignment)

            # Update the gradients
            self.means.grad = new_grad_means.clone()
            self.pre_act_scales.grad = new_grad_scales.clone()
            if not self.is_isotropic:
                self.pre_act_quats.grad = new_grad_quats.clone()
            self.pre_act_phase.grad = new_grad_phase.clone()
            self.colours.grad = new_grad_colours.clone()
            self.pre_act_opacities.grad = new_grad_pre_act_opacities.clone()
            self.pre_act_plane_assignment.grad = new_grad_pre_act_plane_assignment.clone()

            # Move to CUDA if needed
            if self.device.startswith("cuda"):
                self.to_cuda()

            final_count = len(self.means)
            added_count = final_count - orig_count
            print("---------------------  ")
            print(f"  Adaptive Control Completed. Gaussians count = {len(self.means)}")
            print(f"  Kept: {num_keep} | Pruned: {num_prune} | Cloned: {num_clone} | Split: {num_split * 2}")
            print(f"  Added: {added_count} new gaussians")
            print(f"  Final count: {final_count}")
            print("--------------------- ")

            torch.cuda.empty_cache()  # Explicitly call garbage collection
            return final_count

    def opacity_regularization(self, decrease_amount=0.001):
        """
        Alternative to opacity reset that gradually decreases opacity by a fixed amount.
        This provides a smoother approach to opacity regularization than hard resets.

        Revising Densification in Gaussian Splatting ECCV 2024

        Args:
            decrease_amount: Amount to decrease opacity by (default: 0.001)
                            This is the value mentioned in the paper

        Returns:
            updated_count: Number of Gaussians affected by the regularization
        """
        # Get current opacities
        opacities = torch.sigmoid(self.pre_act_opacities)
        denominator = opacities * (1 - opacities)
        denominator = torch.clamp(denominator, min=1e-6)
        # Calculate how much to decrease pre_act_opacities by
        delta = decrease_amount / denominator
        # Apply the decrease to pre_act_opacities
        self.pre_act_opacities.data = self.pre_act_opacities.data - delta
        # Count the affected Gaussians (those not already near zero opacity)
        affected_count = (opacities > decrease_amount).sum().item()

        print(f"Opacity regularization applied: decreased all opacities by ~{decrease_amount}")
        print(f"Affected {affected_count} out of {len(opacities)} Gaussians")
        sys.stdout.flush()
        return affected_count

class Scene:

    def __init__(self, gaussians: Gaussians, args_prop):
        self.gaussians = gaussians
        self.args_prop = args_prop
        self.device = self.gaussians.device
        self.wavelengths = torch.tensor(args_prop.wavelengths, dtype=torch.float32, device=self.device)
        self.mean_2D_for_planeprob = None
    def __repr__(self):
        return f"<Scene with {len(self.gaussians)} Gaussians>"

    def compute_transmittance(self, alphas: torch.Tensor):
        _, H, W = alphas.shape

        # Only use original implementation
        S = torch.ones((1, H, W), device=alphas.device, dtype=alphas.dtype)

        one_minus_alphas = 1.0 - alphas
        one_minus_alphas = torch.concat((S, one_minus_alphas), dim=0)  # (N+1, H, W)

        transmittance = torch.cumprod(one_minus_alphas, dim=0)[:-1]
        # Post processing for numerical stability
        transmittance = torch.where(transmittance < 1e-4, 0.0, transmittance)  # (N, H, W)

        return transmittance

    def compute_depth_values(self, camera: PerspectiveCameras):
        means_3D = self.gaussians.means  # (N, 3)
        R = camera.R[0]  # (3, 3)
        T = camera.T[0]  # (3,)
        # X_cam = X_world @ R + T
        means_cam = means_3D @ R + T  # (N, 3)
        z_vals = means_cam[:, -1]

        return z_vals

    def calculate_gaussian_directions(self, means_3D, camera):

        N = means_3D.shape[0]
        camera_centers = camera.get_camera_center().repeat(N, 1)
        gaussian_dirs = means_3D - camera_centers  # (N, 3)
        gaussian_dirs = F.normalize(gaussian_dirs)
        return gaussian_dirs

    def get_idxs_to_filter_and_sort(self, z_vals: torch.Tensor):
        sorted, indices = torch.sort(z_vals)
        mask = sorted >= 0
        idxs = torch.masked_select(indices, mask).to(torch.int64)
        return idxs


    def splat(self, camera: PerspectiveCameras, means_3D: torch.Tensor, z_vals: torch.Tensor,
            quats: torch.Tensor, scales: torch.Tensor, colours: torch.Tensor,
            phase: torch.Tensor, opacities: torch.Tensor, plane_probs: torch.Tensor, wavelengths: torch.Tensor,
            img_size: Tuple = (256, 256),
            tile_size: Tuple = (-1, -1), render_using_python = False):
        """
        Multi-channel wave-based rendering by summing each color channel separately.
        """
        W, H = img_size
        device = means_3D.device
        num_planes = plane_probs.shape[1]

        if isinstance(wavelengths, list):
            wavelengths = torch.tensor(wavelengths, device=device, dtype=torch.float32)

        # Get camera parameters
        R = camera.R
        T = camera.T
        fx, fy = camera.focal_length.flatten()
        px, py = camera.principal_point.flatten()

        # Extract the view matrix (3x3) from R
        view_matrix = R[0] if R.dim() == 3 else R

        # Set default tile size if not specified
        if tile_size[0] <= 0 or tile_size[1] <= 0:
            tile_size = (64, 64)  # Default tile size for CUDA implementation

        num_channels = len(wavelengths)

        view_transform = camera.get_world_to_view_transform()
        cam_means_3D = view_transform.transform_points(means_3D)
        # print("cam_means_3D: ", cam_means_3D)
        # print("z_vals: ", z_vals)
        # print("quats: ", quats)
        # print("scales: ", scales)
        # print("colours: ", colours)
        # print("phase: ", phase)
        # print("opacities: ", opacities)
        # print("plane_probs: ", plane_probs)
        # print("fx fy: ", fx, fy)
        # print("px py: ", px, py)
        # print("plane_probs: ", plane_probs)
        if not render_using_python:
            # Use the CUDA implementation for parallel tile processing if available
            if USE_SPLAT_TILE_CUDA:
                # Call the CUDA function that processes all tiles in parallel
                # print("\n----- CUDA Implementation -----")
                cuda_plane_fields, visible_indices = splat_tile_cuda(
                    cam_means_3D=cam_means_3D,
                    z_vals=z_vals,
                    quats=quats,
                    scales=scales,
                    colours=colours,
                    phase=phase,
                    opacities=opacities,
                    plane_probs=plane_probs,
                    fx=fx, fy=fy, px=px, py=py,
                    view_matrix=view_matrix,
                    img_size=img_size,
                    near_plane=self.gaussians.NEAR_PLANE,
                    far_plane=self.gaussians.FAR_PLANE
                )
                self.visible_indices = visible_indices
        else:
            tile_size = (64, 64)
            # Fall back to the Python implementation with for-loops if CUDA is not available
            means_2D = self.gaussians.compute_means_2D(cam_means_3D, fx, fy, px, py)
            self.mean_2D_for_planeprob = means_2D
            # Calculate gaussian bounds for adaptive selection
            cov_2D = self.gaussians.compute_cov_2D(cam_means_3D, quats, scales, fx, fy, R, img_size)
            gaussian_bounds = self.gaussians.calculate_gaussian_bounds(means_2D, cov_2D, img_size)
            python_plane_fields = torch.zeros((num_planes, num_channels, H, W), dtype=torch.complex64, device=device)

            # Get tile sizes
            tile_w, tile_h = tile_size
            x_tiles = math.ceil(W / tile_w)
            y_tiles = math.ceil(H / tile_h)
            for y_idx in range(y_tiles):
                for x_idx in range(x_tiles):
                    x = x_idx * tile_w
                    y = y_idx * tile_h
                    # Determine actual tile size to handle edge cases properly
                    actual_tile_w = min(tile_w, W - x)
                    actual_tile_h = min(tile_h, H - y)
                    # Find Gaussians that affect this tile (adaptive selection)
                    x_min, y_min = x, y
                    x_max = x + actual_tile_w - 1
                    y_max = y + actual_tile_h - 1

                    in_x_range = (gaussian_bounds[:, 0] <= x_max) & (gaussian_bounds[:, 2] >= x_min)
                    in_y_range = (gaussian_bounds[:, 1] <= y_max) & (gaussian_bounds[:, 3] >= y_min)
                    gaussian_indices = torch.where(in_x_range & in_y_range)[0]
                    tile_plane_fields = self.splat_tile_python(
                        R, T, fx, fy, px, py, cam_means_3D, z_vals, quats, scales, colours, phase, opacities,
                        plane_probs, x, y, (actual_tile_w, actual_tile_h), gaussian_indices, img_size, wavelengths
                    )  # (num_planes, num_channels, actual_tile_h, actual_tile_w)
                    sys.stdout.flush()
                    python_plane_fields[:, :, y:y+actual_tile_h, x:x+actual_tile_w] += tile_plane_fields

        if render_using_python:
            plane_fields = python_plane_fields
        else:
            plane_fields = cuda_plane_fields

        # # Compare full matrices
        # if torch.allclose(cuda_plane_fields, python_plane_fields, rtol=1e-3, atol=1e-3):
        #     print("✓ CUDA and Python outputs match within tolerance!")
        # else:
        #     print("✗ CUDA and Python outputs differ!")
        #     diff = torch.abs(cuda_plane_fields - python_plane_fields)
        #     max_diff = torch.max(diff)
        #     print(f"  Max difference: {max_diff}")
        #     max_idx = torch.argmax(diff.flatten())
        #     max_idx_unraveled = np.unravel_index(max_idx.item(), diff.shape)
        #     print(f"  Location of max difference: {max_idx_unraveled}")

        # Compute the final hologram for each plane
        hologram_complex_planes = []
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()
        for p in range(num_planes):
            plane_hologram = []
            for c, plane_field_c in enumerate(plane_fields[p]):
                wavelength_val = float(wavelengths[c].cpu().item())
                hologram_complex_c = BandlimitedPropagation(
                    plane_field_c,
                    wavelength=wavelength_val,
                    pixel_pitch=self.args_prop.pixel_pitch,
                    distance=-self.args_prop.distances[p],  # Use plane-specific distance
                    size=self.args_prop.pad_size,
                    aperture_size=self.args_prop.aperture_size,
                    use_cuda=True
                )
                plane_hologram.append(hologram_complex_c)
            hologram_complex_planes.append(torch.stack(plane_hologram, dim=0))
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # with console_only_print():
        #     print(f"--band cuda {curr_time}")
        # Combine holograms from different planes
        hologram_complex = sum(hologram_complex_planes)

        return hologram_complex, plane_fields

    def splat_tile_python(self, R, T, fx, fy, px, py, cam_means_3D, z_vals, quats, scales, colours, phase, opacities,
                plane_probs, tile_x, tile_y, tile_size, gaussian_indices, img_size, wavelengths
        ):
        """
        Process a single tile of the image for multiple planes
        With hard assignment of gaussians to planes
        Handles arbitrary tile sizes at image boundaries

        Args:
            tile_size: The actual size of this specific tile (width, height)
        """
        device = cam_means_3D.device
        W, H = img_size
        tile_w, tile_h = tile_size
        num_planes = plane_probs.shape[1]

        # Initialize output tensors for each plane
        tile_plane_fields = []
        for _ in range(num_planes):
            tile_plane_fields.append(torch.zeros((len(wavelengths), tile_h, tile_w),
                                                device=device, dtype=torch.complex64))
        # Skip early if no Gaussians affect this tile
        if gaussian_indices.numel() == 0:
            return torch.stack(tile_plane_fields, dim=0)

        # Create coordinates for this tile only
        xs, ys = torch.meshgrid(
            torch.arange(tile_x, tile_x + tile_w, device=device),
            torch.arange(tile_y, tile_y + tile_h, device=device),
            indexing="xy"
        )

        points_2D = torch.stack([xs.flatten(), ys.flatten()], dim=1)  # (tile_w*tile_h, 2)
        # Subset the Gaussian data to only those affecting this tile
        tile_means_3D = cam_means_3D[gaussian_indices]
        valid_mask = (tile_means_3D[:, 2] > self.gaussians.NEAR_PLANE) & (tile_means_3D[:, 2] < self.gaussians.FAR_PLANE)
        if not valid_mask.any():
            return torch.stack(tile_plane_fields, dim=0)
        tile_means_3D = tile_means_3D[valid_mask]

        tile_means_2D = self.gaussians.compute_means_2D(tile_means_3D, fx, fy, px, py)  # (tile_N, 2)

        # Get gaussian indices for each plane to avoid processing unnecessary Gaussians
        tile_plane_probs = plane_probs[gaussian_indices]  # (tile_N, num_planes)

        # Process points against Gaussians
        tile_means_2D = tile_means_2D.unsqueeze(1)  # (tile_N, 1, 2)

        # Calculate the difference between each point and each Gaussian mean
        diff = points_2D.unsqueeze(0) - tile_means_2D  # (tile_N, tile_w*tile_h, 2)

        # Compute 2D covariance and its inverse
        tile_cov_2D = self.gaussians.compute_cov_2D(
            tile_means_3D, quats[gaussian_indices], scales[gaussian_indices], fx, fy, R, img_size
        )  # (tile_N, 2, 2)
        cov_inv = self.gaussians.invert_cov_2D(tile_cov_2D)  # (tile_N, 2, 2)

        # Compute the Mahalanobis distance
        term = compute_bmm_cuda(diff, cov_inv)  # (tile_N, tile_w*tile_h, 2)
        # Sum over the last dimension to get the exponent term
        term = sum_last_dim_cuda(element_wise_multiplication_cuda(term, diff))
        term = term.view(-1, tile_h, tile_w)  # (tile_N, tile_h, tile_w)

        # Calculate the Gaussian values
        gauss_exp = torch.exp(-0.5 * term)
        tile_opacities = opacities[gaussian_indices].view(-1, 1, 1)
        base_alphas = tile_opacities * gauss_exp

        for plane_idx in range(num_planes):
            # Get Gaussians assigned to this plane (binary mask)
            plane_mask = tile_plane_probs[:, plane_idx].view(-1, 1, 1)
            plane_alphas = base_alphas * plane_mask
            # Make sure the tensor is properly laid out for the CUDA kernel
            plane_alphas_reshaped = plane_alphas.reshape(-1, tile_h, tile_w)
            transmittance = self.compute_transmittance(plane_alphas_reshaped)  # (tile_N, tile_h, tile_w)
            for c in range(len(wavelengths)):
                colours_c = colours[gaussian_indices, c].view(-1, 1, 1)
                phase_c = phase[gaussian_indices, c].view(-1, 1, 1)
                # amplitude = colours_c * plane_alphas * transmittance
                # phase = phase_c * gauss_exp
                # complex_contribution = amplitude * (torch.cos(phase) + 1j * torch.sin(phase))
                # tile_plane_fields[plane_idx][c] = torch.sum(complex_contribution, dim=0)
                tile_plane_fields[plane_idx][c] = torch.sum(
                    colours_c * plane_alphas * transmittance * torch.exp(1j * phase_c),
                    dim=0
                )
        result = torch.stack(tile_plane_fields, dim=0)  # (num_planes, 3, tile_h, tile_w)
        return result

    def render(
        self, camera: PerspectiveCameras,
        img_size: Tuple = (-1, -1),
        bg_colour: Tuple = (0.0, 0.0, 0.0),
        tile_size: Tuple = (64, 64),
        step = -1, max_step = -1, render_using_python = False
    ):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()
        bg_colour_ = torch.tensor(bg_colour)[None, None, :] # (1, 1, 3)

        # Globally sort gaussians according to their depth value
        z_vals = self.compute_depth_values(camera)
        # Filter based on near/far planes first
        view_transform = camera.get_world_to_view_transform()
        cam_means_3D = view_transform.transform_points(self.gaussians.means)
        visible_mask = (cam_means_3D[:, 2] > self.gaussians.NEAR_PLANE) & (cam_means_3D[:, 2] < self.gaussians.FAR_PLANE)

        # Only keep points within near/far planes
        valid_indices = torch.where(visible_mask)[0]

        # idxs = self.get_idxs_to_filter_and_sort(z_vals)
        idxs = self.get_idxs_to_filter_and_sort(z_vals[valid_indices])
        idxs = valid_indices[idxs]
        pre_act_quats = self.gaussians.pre_act_quats[idxs]
        pre_act_scales = self.gaussians.pre_act_scales[idxs]
        pre_act_phase = self.gaussians.pre_act_phase[idxs]
        pre_act_opacities = self.gaussians.pre_act_opacities[idxs]
        pre_act_plane_assignment = self.gaussians.pre_act_plane_assignment[idxs]

        z_vals = z_vals[idxs]
        means_3D = self.gaussians.means[idxs]
        colours = self.gaussians.colours[idxs]

        # Apply activations
        quats, scales, phase, opacities, plane_probs = self.gaussians.apply_activations(
            pre_act_quats, pre_act_scales, pre_act_phase, pre_act_opacities, pre_act_plane_assignment, step, max_step
        )
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print(f"render before splat {curr_time}")

        wavelengths = self.wavelengths
        hologram_complex, plane_field = self.splat(
            camera, means_3D, z_vals, quats, scales,
            colours,
            phase, opacities, plane_probs, wavelengths,
            img_size, tile_size, render_using_python
        )

        return hologram_complex, plane_field
