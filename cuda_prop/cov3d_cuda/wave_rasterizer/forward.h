#pragma once

#include <torch/extension.h>
#include <vector>

namespace cov3d_cuda {

/**
 * Forward pass for computing 3D covariance matrices from quaternions and scales.
 * 
 * @param quats Rotation quaternions (N, 4)
 * @param scales Scaling factors (N, 3)
 * @return torch::Tensor Covariance matrices (N, 3, 3)
 */
torch::Tensor compute_cov3d_forward(
    const torch::Tensor& quats,
    const torch::Tensor& scales);

// Update the jacobian function signature
torch::Tensor compute_jacobian_forward(
    torch::Tensor cam_means_3D,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane);

// Modified: Return intermediate values for backward pass
std::vector<torch::Tensor> compute_cov2d_forward(
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& view_matrix,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane);

// Update the 2D means function signature
torch::Tensor compute_means2d_forward(
    const torch::Tensor& cam_means_3D,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane);

// New function for inverting 2D covariance matrices
torch::Tensor invert_cov2d_forward(
    const torch::Tensor& cov_2D);

} // namespace cov3d_cuda