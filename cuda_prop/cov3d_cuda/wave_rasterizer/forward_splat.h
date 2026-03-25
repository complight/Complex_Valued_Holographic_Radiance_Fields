/*
 * Forward pass for wave-based 3D Gaussian Splatting
 * Optimized for performance matching the official implementation structure
 */

 #pragma once

 #include <cuda.h>
 #include <cuda_runtime.h>
 
 namespace cov3d_cuda {
 
 // Forward declarations of kernel functions
 __global__ void preprocessGaussiansKernel(
     int N,
     const float2* __restrict__ means_2D,
     const float4* __restrict__ cov_2D,
     const float* __restrict__ z_vals,
     int* __restrict__ radii,
     uint32_t* __restrict__ tiles_touched,
     dim3 grid,
     bool prefiltered,
     bool antialiasing,
     float near_plane,
     float far_plane);
 
 __global__ void renderTileKernel(
     const uint2* __restrict__ tile_ranges,
     const uint32_t* __restrict__ gaussian_indices_sorted,
     const float2* __restrict__ means_2D,
     const float4* __restrict__ cov_2D,
     const float* __restrict__ z_vals,
     const float* __restrict__ colours,
     const float* __restrict__ phase,
     const float* __restrict__ opacities,
     const float* __restrict__ plane_probs,
     float* __restrict__ output_real,
     float* __restrict__ output_imag,
     float* __restrict__ final_Ts,
     uint32_t* __restrict__ n_contrib,
     int W, int H, int N,
     int num_planes, int channels,
     float near_plane, float far_plane);
 
 // Namespace for forward pass functions (similar to FORWARD in official code)
 namespace FORWARD_SPLAT {
 
 // Preprocessing function (similar to FORWARD::preprocess in official code)
 void preprocess(
     int N,
     const float2* means_2D,
     const float4* cov_2D,
     const float* z_vals,
     int* radii,
     uint32_t* tiles_touched,
     const dim3 grid,
     bool prefiltered,
     bool antialiasing,
     float near_plane,
     float far_plane);
 
 // Rendering function (similar to FORWARD::render in official code)
 void render(
     const dim3 grid, dim3 block,
     const uint2* tile_ranges,
     const uint32_t* point_list,
     const float2* means_2D,
     const float4* cov_2D,
     const float* z_vals,
     const float* colours,
     const float* phase,
     const float* opacities,
     const float* plane_probs,
     float* output_real,
     float* output_imag,
     float* final_Ts,
     uint32_t* n_contrib,
     int W, int H, int N,
     int num_planes, int channels,
     float near_plane, float far_plane);
 
 } // namespace FORWARD_SPLAT
 
 } // namespace cov3d_cuda