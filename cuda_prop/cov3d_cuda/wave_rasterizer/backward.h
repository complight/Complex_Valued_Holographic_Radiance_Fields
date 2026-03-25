#pragma once

#include <torch/extension.h>
#include <vector>

namespace cov3d_cuda {

/**
 * Backward pass for computing gradients of 3D covariance matrices.
 * 
 * @param grad_cov3d Gradient of loss with respect to output covariances (N, 3, 3)
 * @param quats Input quaternions (N, 4)
 * @param scales Input scales (N, 3)
 * @return std::vector<torch::Tensor> Gradients for quaternions and scales
 */
std::vector<torch::Tensor> compute_cov3d_backward(
    const torch::Tensor& grad_cov3d,
    const torch::Tensor& quats,
    const torch::Tensor& scales);

// Update the jacobian backward function signature
torch::Tensor compute_jacobian_backward(
    torch::Tensor grad_output,
    torch::Tensor cam_means_3D,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane);

// Modified: Accept intermediate values from forward pass
std::vector<torch::Tensor> compute_cov2d_backward(
    const torch::Tensor& grad_cov2d,
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& view_matrix,
    const torch::Tensor& J,          // Add Jacobian
    const torch::Tensor& cov3D,      // Add 3D covariance
    const torch::Tensor& W,          // Add repeated view matrix
    const torch::Tensor& JW,         // Add J*W product
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane);

// Update the 2D means backward function signature
torch::Tensor compute_means2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& cam_means_3D,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane);

// Backward pass for invert_cov2d
torch::Tensor invert_cov2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& cov_2D);

} // namespace cov3d_cuda