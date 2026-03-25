import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import imageio
import numpy as np
import matplotlib.pyplot as plt
import odak
import math
import sys
import random
from torch.nn.functional import mse_loss
from pytorch3d.ops.knn import knn_points
from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
from pytorch_msssim import SSIM, ms_ssim
from cuda_prop.cov3d_cuda.python_import import fast_ssim

CMAP_JET = plt.get_cmap("jet")
CMAP_MIN_NORM, CMAP_MAX_NORM = 5.0, 7.0

def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch


@contextlib.contextmanager
def console_only_print():
    original_stdout = sys.stdout
    if original_stdout != sys.__stdout__:
        sys.stdout = sys.__stdout__
    try:
        yield
    finally:
        sys.stdout = original_stdout

def total_variation_loss_difference(pred, target):
    """
    Function for calculating the difference between total variation of prediction and target images.
    
    Parameters
    ----------
    pred          : torch.tensor
                    Predicted frame [B x C x H x W] or [C x H x W] or [H x W].
    target        : torch.tensor
                    Target frame with same dimensions as pred.
    Returns
    -------
    loss          : float
                    Absolute difference between TV losses.
    """
    # Ensure both inputs have the same shape and dimensionality
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
    
    # Calculate TV loss for prediction
    pred_diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_diff_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    # Calculate TV loss for target
    target_diff_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_diff_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = mse_loss(pred_diff_x, target_diff_x)
    loss_y = mse_loss(pred_diff_y, target_diff_y)
    loss = loss_x + loss_y
    return loss


def multi_scale_total_variation_loss_difference(pred, target, levels=3):
    """
    Function for calculating the difference between total variation of prediction and target 
    using multi-scale approach.
    
    Parameters
    ----------
    pred          : torch.tensor
                    Predicted frame [B x C x H x W] or [C x H x W] or [H x W].
    target        : torch.tensor
                    Target frame with same dimensions as pred.
    levels        : int
                    Number of levels in the image pyramid.
    Returns
    -------
    loss          : float
                    Sum of absolute differences between TV losses across scales.
    """
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
        
    if len(target.shape) == 2:
        target = target.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
        
    scale = torch.nn.Upsample(scale_factor=0.5, mode='nearest')
    pred_level = pred
    target_level = target
    loss = 0
    
    for i in range(levels):
        if i != 0:
            pred_level = scale(pred_level)
            target_level = scale(target_level)
            
        # Calculate TV loss difference at this level
        loss += total_variation_loss_difference(pred_level, target_level) 
        
    return loss

def GaussianLoss(pred, target, lambda_ssim=0.025, lambda_l2=10):
    """
    Calculate combined Gaussian loss with SSIM and L2 components.
    
    Args:
        pred: Predicted image [3, H, W] or [B, 3, H, W]
        target: Target image [3, H, W, 3] or [B, 3, H, W]
        lambda_ssim: Weight for SSIM loss component (default: 0.02)
        lambda_l2: Weight for L2 loss component (default: 10)
        
    Returns:
        total_loss: Combined weighted loss
    """
    # Add batch dimension if not present
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
    # Calculate losses
    # l2_loss = F.mse_loss(pred, target) * 2.
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    ssim_loss = (1 - ssim(pred, target)) * 0.2 # Convert to loss # ms_ssim(pred, target, data_range=1.0)
    # ssim_loss = (1 - fast_ssim(pred, target)) * 0.2
    # tv_loss = odak.learn.tools.multi_scale_total_variation_loss(pred, 3) * 0.1
    # tv_loss = multi_scale_total_variation_loss_difference(pred, target, levels=3) * 5e-2
    
    # print("ssim: ", lambda_ssim * ssim_loss)
    # print("l2: ", lambda_l2 * l2_loss)
    # print("tv_loss: ", tv_loss)
    
    # Combined loss
    # total_loss = lambda_l2 * l2_loss + lambda_ssim * ssim_loss + tv_loss
    total_loss = lambda_ssim * ssim_loss #+ tv_loss    
    return total_loss
    
def plane_assignment_loss(scene, quantized_depth, num_planes, weight=0.1):
    """
    Balanced contrastive loss for plane assignment with appropriate scaling.
    
    Args:
        scene: Scene object containing gaussians
        quantized_depth: Quantized depth mask (1, H, W)
        num_planes: Number of planes
        weight: Weight factor to balance this loss with others
    """
    _, H, W = quantized_depth.shape
    pre_act_plane_assignment = scene.gaussians.pre_act_plane_assignment
    means_2D = scene.mean_2D_for_planeprob
    
    # Round and clamp 2D positions to valid pixel coordinates
    pixel_x = torch.clamp(means_2D[:, 0].round().long(), 0, W-1)
    pixel_y = torch.clamp(means_2D[:, 1].round().long(), 0, H-1)
    
    # Sample the quantized depth at each Gaussian's projected position
    # Normalize depth values to [0, 1] range
    depth_values = quantized_depth[0, pixel_y, pixel_x] / 255.0  # (N,)

    # Map normalized depth to target plane indices
    target_plane_idx = (depth_values * (num_planes - 1)).round().long()
    
    # cross-entropy loss
    loss = weight * F.cross_entropy(pre_act_plane_assignment, target_plane_idx)
    
    return loss

def colours_from_spherical_harmonics(spherical_harmonics, gaussian_dirs):
    # Each SH basis has 3 coefficients (R,G,B). Chunk them out:
    c0  = spherical_harmonics[:,  0:  3]
    c1  = spherical_harmonics[:,  3:  6]
    c2  = spherical_harmonics[:,  6:  9]
    c3  = spherical_harmonics[:,  9: 12]
    c4  = spherical_harmonics[:, 12: 15]
    c5  = spherical_harmonics[:, 15: 18]
    c6  = spherical_harmonics[:, 18: 21]
    c7  = spherical_harmonics[:, 21: 24]
    c8  = spherical_harmonics[:, 24: 27]
    c9  = spherical_harmonics[:, 27: 30]
    c10 = spherical_harmonics[:, 30: 33]
    c11 = spherical_harmonics[:, 33: 36]
    c12 = spherical_harmonics[:, 36: 39]
    c13 = spherical_harmonics[:, 39: 42]
    c14 = spherical_harmonics[:, 42: 45]
    c15 = spherical_harmonics[:, 45: 48]

    # Directions
    x = gaussian_dirs[:, 0:1]  # shape (N,1)
    y = gaussian_dirs[:, 1:2]  # Fixed slicing
    z = gaussian_dirs[:, 2:3]  # Fixed slicing

    # Polynomial expansions for real SH up to order 3 (removing normalization constants)
    # Avoid in-place operations by using + instead of +=
    # L=0
    color = c0.clone()  # Start with a clone to avoid modifying the original tensor

    # L=1
    color = color + c1 * y
    color = color + c2 * z
    color = color + c3 * x

    # L=2
    color = color + c4 * (x * y)
    color = color + c5 * (y * z)
    color = color + c6 * (2.0 * z**2 - x**2 - y**2)
    color = color + c7 * (x * z)
    color = color + c8 * (x**2 - y**2)

    # L=3
    color = color + c9  * (y * (3.0 * x**2 - y**2))
    color = color + c10 * (x * y * z)
    color = color + c11 * (y * (4.0 * z**2 - x**2 - y**2))
    color = color + c12 * (z * (2.0 * z**2 - 3.0 * x**2 - 3.0 * y**2))
    color = color + c13 * (x * (4.0 * z**2 - x**2 - y**2))
    color = color + c14 * (z * (x**2 - y**2))
    color = color + c15 * (x * (x**2 - 3.0 * y**2))

    # Non-inplace clamping
    color = torch.clamp(color, 0.0, 1.0)
    return color

 
def multiplane_loss(target_image, target_depth, args_prop):
    # loss_function = odak.learn.wave.multiplane_loss(
    from .propagator import multiplane_loss_odak
    loss_function = multiplane_loss_odak(
                        target_image = target_image,
                        target_depth = target_depth,
                        target_blur_size = 20,
                        number_of_planes = args_prop.num_planes,
                        blur_ratio = 8,
                        weights = [1.0, 1.0, 1.0, 0.0],
                        scheme = "defocus",
                        reduction = "mean",
                        split_ratio = args_prop.split_ratio,  
                        device = "cuda"
    )

    targets, mask, quantized_depth = loss_function.get_targets()
    
    # # below code is for front masking used only in the experiment of MIP360 to prove the point of motion parallax,
    # # to use the this code, please turn the number of the plane to be 1 and volume depth to be 0
    # loss_function2 = multiplane_loss_odak(
    #                     target_image = target_image,
    #                     target_depth = target_depth,
    #                     target_blur_size = 20,
    #                     number_of_planes = 2,
    #                     blur_ratio = 8,
    #                     weights = [1.0, 1.0, 1.0, 0.0],
    #                     scheme = "defocus",
    #                     reduction = "mean",
    #                     split_ratio = args_prop.split_ratio,  
    #                     device = "cuda"
    # )

    # targets2, mask2, quantized_depth2 = loss_function2.get_targets()
    # # print("mask: ", mask.size())
    # # for i, dep in enumerate(mask2):
    # #     odak.learn.tools.save_image(
    # #         f"./mask2{i}.png",
    # #         dep,
    # #         cmin=0.,
    # #         cmax=1.
    # #     )
    # targets = targets * mask2[1]  
    # # odak.learn.tools.save_image( 
    # #     f"./targets.png",
    # #     targets,
    # #     cmin=0.,
    # #     cmax=1.
    # # )          
    return targets, loss_function, mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")
    
def ndc_to_screen_camera(camera, img_size):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )
    