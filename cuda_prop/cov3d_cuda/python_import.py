# First, fix the missing imports
import os
import sys
import subprocess
from torch.autograd import Function
import torch

# Try to import pre-compiled module or build it
print("Building cov3d_cuda module...")

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Run build command
build_command = [sys.executable, "setup.py", "install"]
subprocess.check_call(build_command, cwd=current_dir)

# Now try importing
try:
    from cov3d_cuda import (
        compute_cov3d_forward, compute_cov3d_backward,
        compute_jacobian_forward, compute_jacobian_backward,
        compute_cov2d_forward, compute_cov2d_backward,
        compute_means2d_forward, compute_means2d_backward,
        invert_cov2d_forward, invert_cov2d_backward,
        fusedssim, fusedssim_backward, 
        Rasterizer, adamUpdate
    )
    print("Successfully built and imported cov3d_cuda module.")
except ImportError as e:
    print(f"Fail: {e}")
    print("Failed to build or import cov3d_cuda module.")
    raise

# Create a global Rasterizer instance (or you could create it in SplatTileCuda)
cuda_rasterizer = None
def get_rasterizer():
    global cuda_rasterizer
    if cuda_rasterizer is None:
        cuda_rasterizer = Rasterizer()
    return cuda_rasterizer

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

class Compute3DCovarianceFunction(Function):
    @staticmethod
    def forward(ctx, quats, scales):
        """
        Forward pass for computing 3D covariance matrices.
        
        Args:
            quats: Rotation quaternions (N, 4)
            scales: Scaling factors (N, 3)
        
        Returns:
            cov3d: 3D covariance matrices (N, 3, 3)
        """
        # Make sure inputs are contiguous and on CUDA
        quats = quats.contiguous()
        scales = scales.contiguous()
        
        # Check device
        if not quats.is_cuda or not scales.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        
        # Save inputs for backward
        ctx.save_for_backward(quats, scales)
        
        # Call CUDA implementation
        return compute_cov3d_forward(quats, scales)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for computing gradients.
        
        Args:
            grad_output: Gradient of loss with respect to output (N, 3, 3)
        
        Returns:
            grad_quats: Gradient of loss with respect to quaternions (N, 4)
            grad_scales: Gradient of loss with respect to scales (N, 3)
        """
        quats, scales = ctx.saved_tensors
        
        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA implementation
        grad_quats, grad_scales = compute_cov3d_backward(grad_output, quats, scales)
        
        return grad_quats, grad_scales

class ComputeJacobianFunction(Function):
    @staticmethod
    def forward(ctx, cam_means_3D, fx, fy, img_size, near_plane, far_plane):
        """
        Forward pass for computing projection Jacobian matrices.
        """
        # Make sure inputs are contiguous and on CUDA
        cam_means_3D = cam_means_3D.contiguous()
        
        # Check device
        if not cam_means_3D.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        
        # Save inputs for backward
        ctx.save_for_backward(cam_means_3D)
        ctx.fx = fx
        ctx.fy = fy
        ctx.width, ctx.height = img_size
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        
        # Call CUDA implementation
        return compute_jacobian_forward(cam_means_3D, float(fx), float(fy), 
                                       int(img_size[0]), int(img_size[1]),
                                       float(near_plane), float(far_plane))
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for computing gradients.
        """
        cam_means_3D, = ctx.saved_tensors
        
        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA implementation
        grad_cam_means_3D = compute_jacobian_backward(
            grad_output, cam_means_3D, ctx.fx, ctx.fy, ctx.width, ctx.height,
            ctx.near_plane, ctx.far_plane
        )
        
        return grad_cam_means_3D, None, None, None, None, None

class ComputeCov2DFunction(Function):
    @staticmethod
    def forward(ctx, cam_means_3D, quats, scales, view_matrix, fx, fy, img_size, near_plane, far_plane):
        """
        Forward pass for computing 2D covariance matrices.
        """
        # Make sure inputs are contiguous and on CUDA
        cam_means_3D = cam_means_3D.contiguous()
        quats = quats.contiguous()
        scales = scales.contiguous()
        view_matrix = view_matrix.contiguous()
        
        # Check device
        if not cam_means_3D.is_cuda or not quats.is_cuda or not scales.is_cuda or not view_matrix.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        
        # Call CUDA implementation which now returns intermediate values
        width, height = img_size
        # The function now returns multiple tensors
        result = compute_cov2d_forward(cam_means_3D, quats, scales, view_matrix, 
                                     float(fx), float(fy), int(width), int(height),
                                     float(near_plane), float(far_plane))
        
        # Unpack the results - first one is the 2D covariance, rest are intermediate values
        cov2d, J, cov3D, W, JW = result
        
        # Save inputs and intermediate tensors for backward
        ctx.save_for_backward(cam_means_3D, quats, scales, view_matrix, J, cov3D, W, JW)
        ctx.fx = fx
        ctx.fy = fy
        ctx.img_size = img_size
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        
        return cov2d
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for computing gradients.
        """
        # Retrieve saved tensors including intermediate values
        cam_means_3D, quats, scales, view_matrix, J, cov3D, W, JW = ctx.saved_tensors
        
        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA implementation with the intermediate values
        width, height = ctx.img_size
        grad_cam_means_3D, grad_quats, grad_scales = compute_cov2d_backward(
            grad_output, cam_means_3D, quats, scales, view_matrix,
            J, cov3D, W, JW,  # Pass the intermediate tensors
            ctx.fx, ctx.fy, width, height, ctx.near_plane, ctx.far_plane
        )
        
        # We don't compute gradients for view_matrix, fx, fy, img_size, and clipping planes
        grad_view_matrix = None
        
        return grad_cam_means_3D, grad_quats, grad_scales, grad_view_matrix, None, None, None, None, None

class ComputeMeans2DFunction(Function):
    @staticmethod
    def forward(ctx, cam_means_3D, fx, fy, px, py, near_plane, far_plane):
        """
        Forward pass for computing 2D means from 3D points.
        """
        # Make sure inputs are contiguous and on CUDA
        cam_means_3D = cam_means_3D.contiguous()
        
        # Check device
        if not cam_means_3D.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        
        # Save inputs for backward
        ctx.save_for_backward(cam_means_3D)
        ctx.fx = fx
        ctx.fy = fy
        ctx.px = px
        ctx.py = py
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        
        # Call CUDA implementation
        return compute_means2d_forward(cam_means_3D, float(fx), float(fy), float(px), float(py),
                                      float(near_plane), float(far_plane))
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for computing gradients.
        """
        cam_means_3D, = ctx.saved_tensors
        
        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA implementation
        grad_cam_means_3D = compute_means2d_backward(grad_output, cam_means_3D, 
                                                   ctx.fx, ctx.fy, ctx.px, ctx.py,
                                                   ctx.near_plane, ctx.far_plane)
        
        return grad_cam_means_3D, None, None, None, None, None, None

class InvertCov2DFunction(Function):
    @staticmethod
    def forward(ctx, cov_2D):
        # Make sure cov_2D is contiguous and on CUDA
        cov_2D = cov_2D.contiguous()
        
        # Check device
        if not cov_2D.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")
        
        # Save for backward
        ctx.save_for_backward(cov_2D)
        
        # Call CUDA implementation
        return invert_cov2d_forward(cov_2D)
    
    @staticmethod
    def backward(ctx, grad_output):
        cov_2D, = ctx.saved_tensors
        
        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA implementation
        grad_cov_2D = invert_cov2d_backward(grad_output, cov_2D)
        
        return grad_cov_2D

class SplatTileCuda(Function):
    @staticmethod
    def forward(ctx, cam_means_3D, z_vals, quats, scales, colours, phase, opacities, plane_probs,
                fx, fy, px, py, view_matrix, img_size, near_plane, far_plane, tile_size=(16, 16)):
        # Save inputs for backward pass
        ctx.save_for_backward(cam_means_3D, z_vals, quats, scales, colours, phase, opacities,
                              plane_probs, view_matrix)
        ctx.fx = fx
        ctx.fy = fy
        ctx.px = px 
        ctx.py = py
        ctx.img_size = img_size
        ctx.tile_size = tile_size
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        
        # Get the rasterizer instance
        rasterizer = get_rasterizer()
        
        # Call the forward method directly
        output, forward_info = rasterizer.forward(
            cam_means_3D, z_vals, quats, scales, colours, phase, opacities, plane_probs,
            fx, fy, px, py, view_matrix, img_size, tile_size, near_plane, far_plane
        )
        
        # Save forward_info for backward pass
        # forward_info is a tuple of (final_Ts, n_contrib, point_list, ranges, visible_indices)
        visible_indices = forward_info[-1]
        ctx.forward_info = forward_info
        
        return output, visible_indices
        
    @staticmethod
    def backward(ctx, grad_output, grad_visible_indices=None):
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Retrieve saved tensors
        cam_means_3D, z_vals, quats, scales, colours, phase, opacities, plane_probs, view_matrix = ctx.saved_tensors
        
        # Unpack forward info
        final_Ts, n_contrib, point_list, ranges, visible_indices = ctx.forward_info
        
        # Get the rasterizer instance
        rasterizer = get_rasterizer()
        
        # Call the backward method directly
        grad_cam_means_3D, grad_z_vals, grad_quats, grad_scales, grad_colours, \
        grad_phase, grad_opacities, grad_plane_probs = rasterizer.backward(
            grad_output,
            cam_means_3D, z_vals, quats, scales, colours, phase, opacities, plane_probs,
            ctx.fx, ctx.fy, ctx.px, ctx.py, view_matrix,
            ctx.img_size, ctx.tile_size,
            final_Ts, n_contrib, point_list, ranges,
            ctx.near_plane, ctx.far_plane
        )
        # Return gradients in the same order as inputs to forward
        return grad_cam_means_3D, grad_z_vals, grad_quats, grad_scales, grad_colours, \
            grad_phase, grad_opacities, grad_plane_probs, \
            None, None, None, None, None, None, None, None, None

# Wrapper functions to maintain backward compatibility
def compute_cov3d_cuda(quats, scales):
    """
    Compute 3D covariance matrices using CUDA implementation.
    
    Args:
        quats: Rotation quaternions (N, 4)
        scales: Scaling factors (N, 3)
    
    Returns:
        cov3d: 3D covariance matrices (N, 3, 3)
    """
    return Compute3DCovarianceFunction.apply(quats, scales)

def compute_jacobian_cuda(cam_means_3D, fx, fy, img_size, near_plane, far_plane):
    """
    Compute projection Jacobian matrices using CUDA implementation.
    
    Args:
        cam_means_3D: 3D points in camera space (N, 3)
        fx, fy: Focal lengths
        img_size: Image dimensions (width, height)
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
    
    Returns:
        jacobian: Jacobian matrices (N, 2, 3)
    """
    return ComputeJacobianFunction.apply(cam_means_3D, fx, fy, img_size, near_plane, far_plane)

def compute_cov2d_cuda(cam_means_3D, quats, scales, view_matrix, fx, fy, img_size, near_plane, far_plane):
    """
    Compute 2D covariance matrices using CUDA implementation.
    
    Args:
        cam_means_3D: 3D points in camera space (N, 3)
        quats: Rotation quaternions (N, 4)
        scales: Scaling factors (N, 3) or (N, 1)
        view_matrix: Camera view matrix (3, 3)
        fx, fy: Focal lengths
        img_size: Image dimensions (width, height)
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
    
    Returns:
        cov2d: 2D covariance matrices (N, 2, 2)
    """
    return ComputeCov2DFunction.apply(cam_means_3D, quats, scales, view_matrix, fx, fy, img_size, near_plane, far_plane)

def compute_means2d_cuda(cam_means_3D, fx, fy, px, py, near_plane, far_plane):
    """
    Compute 2D means from 3D camera-space points using CUDA implementation.
    
    Args:
        cam_means_3D: 3D points in camera space (N, 3)
        fx, fy: Focal lengths
        px, py: Principal point coordinates
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
    
    Returns:
        means_2D: 2D image coordinates (N, 2)
    """
    return ComputeMeans2DFunction.apply(cam_means_3D, fx, fy, px, py, near_plane, far_plane)

def invert_cov2d_cuda(cov_2D):
    """
    Invert 2D covariance matrices using CUDA implementation.
    
    Args:
        cov_2D: 2D covariance matrices (N, 2, 2)
    
    Returns:
        cov_2D_inverse: Inverted covariance matrices (N, 2, 2)
    """
    return InvertCov2DFunction.apply(cov_2D)

def splat_tile_cuda(cam_means_3D, z_vals, quats, scales, colours, phase, opacities, plane_probs,
                   fx, fy, px, py, view_matrix, img_size, near_plane, far_plane, tile_size=(16, 16)):
    """
    Perform tile-based Gaussian splatting using CUDA.
    
    Args:
        cam_means_3D: Camera-space 3D points (N, 3)
        z_vals: Depth values (N,)
        quats: Quaternion rotations (N, 4)
        scales: Scaling factors (N, 3) or (N, 1)
        colours: Color values (N, C)
        phase: Phase values (N, C)
        opacities: Opacity values (N,)
        plane_probs: Probability distribution over planes (N, P)
        fx, fy: Focal lengths
        px, py: Principal point
        view_matrix: Camera view matrix (3, 3)
        img_size: Image dimensions (width, height)
        near_plane: Near clipping plane distance
        far_plane: Far clipping plane distance
        tile_size: Tile dimensions (width, height)
    
    Returns:
        Complex tensor of shape (P, C, H, W) representing the field
    """
    return SplatTileCuda.apply(cam_means_3D, z_vals, quats, scales, colours, phase, opacities, plane_probs,
                              fx, fy, px, py, view_matrix, img_size, near_plane, far_plane, tile_size)

C1 = 0.01 ** 2
C2 = 0.03 ** 2

def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def sparse_adam_update(param:torch.Tensor, grad:torch.Tensor, exp_avg:torch.Tensor, exp_avg_sq:torch.Tensor, visible_chunk:torch.Tensor, 
                       lr:float, b1:float, b2:float, eps:float):
    adamUpdate(param,grad,exp_avg,exp_avg_sq,visible_chunk,lr,b1,b2,eps)
    return