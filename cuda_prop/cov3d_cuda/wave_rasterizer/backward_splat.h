#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace cov3d_cuda {
namespace BACKWARD_SPLAT {

// Main backward render function (similar to BACKWARD::render in official code)
void render(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    const float2* means_2D,
    const float4* cov_2D,
    const float* z_vals,
    const float* colours,
    const float* phase,
    const float* opacities,
    const float* plane_probs,
    const float* final_Ts,
    const uint32_t* n_contrib,
    const float* grad_output_real,
    const float* grad_output_imag,
    float* grad_means_2D,
    float* grad_cov_2D,
    float* grad_z_vals,
    float* grad_colours,
    float* grad_phase,
    float* grad_opacities,
    float* grad_plane_probs,
    int N, int num_planes, int num_channels,
    int W, int H,
    float near_plane, float far_plane);

// Modified: Accept intermediate values from forward pass
void preprocess(
    float fx, float fy,
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& view_matrix,
    const torch::Tensor& grad_means_2D,
    const torch::Tensor& grad_cov_2D,
    torch::Tensor& grad_means_3D,
    torch::Tensor& grad_quats,
    torch::Tensor& grad_scales,
    int W, int H,
    float near_plane, float far_plane,
    const torch::Tensor& J,        // Jacobian from forward
    const torch::Tensor& cov3D,    // 3D covariance from forward
    const torch::Tensor& W_mat,    // Repeated view matrix from forward
    const torch::Tensor& JW        // J*W from forward
);

} // namespace BACKWARD_SPLAT
} // namespace cov3d_cuda