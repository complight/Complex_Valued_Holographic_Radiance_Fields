/*
* Main rasterizer implementation for wave-based 3D Gaussian Splatting
*/

#include "rasterizer_splat.h"
#include "auxiliary.h"
#include "forward_splat.h"
#include "backward_splat.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <iostream>  // For error reporting

namespace cg = cooperative_groups;

namespace cov3d_cuda {

// Helper function to find the next-highest bit of the MSB
__host__ uint32_t getHigherMsb(uint32_t n) {
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int N,
    const float* orig_points,
    const float* viewmatrix,
    const float* projmatrix,
    bool* present,
    float near_plane,
    float far_plane)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    float3 p_view;
    // Check if point is within clipping planes
    bool in_view = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
    // Also check near and far planes
    if (in_view) {
        in_view = (p_view.z > near_plane && p_view.z < far_plane);
    }
    present[idx] = in_view;
}

// Structure method implementations
GeometryState GeometryState::fromChunk(char*& chunk, size_t N) {
    GeometryState state;
    obtain(chunk, state.means_2D, N, 128);
    obtain(chunk, state.cov_2D, N, 128);
    obtain(chunk, state.z_vals, N, 128);
    obtain(chunk, state.radii, N, 128);
    obtain(chunk, state.tiles_touched, N, 128);
    obtain(chunk, state.point_offsets, N, 128);
    
    cub::DeviceScan::InclusiveSum(nullptr, state.scan_size, state.tiles_touched, state.tiles_touched, N);
    obtain(chunk, state.scanning_space, state.scan_size, 128);
    
    return state;
}

BinningState BinningState::fromChunk(char*& chunk, size_t total_pairs) {
    BinningState state;
    obtain(chunk, state.keys_unsorted, total_pairs, 128);
    obtain(chunk, state.values_unsorted, total_pairs, 128);
    obtain(chunk, state.keys_sorted, total_pairs, 128);
    obtain(chunk, state.values_sorted, total_pairs, 128);
    
    cub::DeviceRadixSort::SortPairs(
        nullptr, state.sorting_size,
        state.keys_unsorted, state.keys_sorted,
        state.values_unsorted, state.values_sorted, total_pairs);
    obtain(chunk, state.sorting_space, state.sorting_size, 128);
    
    return state;
}

ImageState ImageState::fromChunk(char*& chunk, size_t num_tiles, size_t num_pixels, int num_planes) {
    ImageState state;
    obtain(chunk, state.ranges, num_tiles, 128);
    obtain(chunk, state.final_Ts, num_pixels * num_planes, 128);
    obtain(chunk, state.n_contrib, num_pixels * num_planes, 128);
    
    return state;
}

// Kernel to fill an array with ones
__global__ void fillOnesKernel(float* data, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

// Generate key-value pairs for tile-Gaussian associations with depth-based sorting for back-to-front rendering
__global__ void duplicateWithKeysKernel(
    int N,
    const float2* __restrict__ means_2D,
    const float* __restrict__ z_vals,
    const int* __restrict__ radii,
    const uint32_t* __restrict__ point_offsets,
    uint64_t* __restrict__ keys_unsorted,
    uint32_t* __restrict__ values_unsorted,
    dim3 grid,
    float near_plane,
    float far_plane)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    
    // Skip if radius is 0 (culled)
    const int radius = radii[idx];
    if (radius == 0)
        return;
    
    // Skip if outside clipping planes
    const float z = z_vals[idx];
    if (z <= near_plane || z >= far_plane)
        return;
    
    // Find this Gaussian's offset for writing
    const uint32_t offset = (idx == 0) ? 0 : point_offsets[idx - 1];
    
    // Compute affected tiles
    uint2 rect_min, rect_max;
    getRect(means_2D[idx], radius, rect_min, rect_max, grid);
    
    // Get depth for sorting (invert for back-to-front sorting)
    // Use bitwise operation to invert IEEE float bit pattern for proper sorting
    // This ensures farthest Gaussians get processed first
    const float depth = z_vals[idx];
    uint32_t depth_bits = __float_as_uint(depth);
    
    // If depth is positive, flip the sign bit to sort back-to-front
    // If depth is negative, flip all except sign bit
    if ((depth_bits & 0x80000000) == 0)
        depth_bits = depth_bits | 0x80000000;  // Set sign bit
    else
        depth_bits = depth_bits & 0x7FFFFFFF;  // Clear sign bit
    
    // Generate key-value pairs for each affected tile
    uint32_t current_offset = offset;
    for (uint32_t y = rect_min.y; y < rect_max.y; y++) {
        for (uint32_t x = rect_min.x; x < rect_max.x; x++) {
            // Create key with tile ID in upper 32 bits, depth in lower 32 bits
            uint64_t key = (uint64_t)(y * grid.x + x) << 32;
            key |= depth_bits;
            
            // Store key-value pair
            keys_unsorted[current_offset] = key;
            values_unsorted[current_offset] = idx;
            current_offset++;
        }
    }
}

// Identify tile ranges kernel
__global__ void identifyTileRangesKernel(
    int num_pairs,
    const uint64_t* __restrict__ keys_sorted,
    uint2* __restrict__ ranges)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs)
        return;
    
    // Extract tile ID from key
    const uint32_t tile_id = keys_sorted[idx] >> 32;
    
    // Mark start and end of tile ranges
    if (idx == 0) {
        ranges[tile_id].x = idx;
    } else if (tile_id != (keys_sorted[idx-1] >> 32)) {
        const uint32_t prev_tile = keys_sorted[idx-1] >> 32;
        ranges[prev_tile].y = idx;
        ranges[tile_id].x = idx;
    }
    
    if (idx == num_pairs - 1) {
        ranges[tile_id].y = idx + 1;
    }
}

} // namespace cov3d_cuda

// Implementation of the memory manager methods
namespace CudaRasterizer {

char* CudaMemoryManager::getGeometryBuffer(size_t size) {
    // If current buffer is too small, free it and allocate a new one
    if (geom_buffer != nullptr && geom_buffer_size < size) {
        cudaFree(geom_buffer);
        geom_buffer = nullptr;
    }
    
    // If no buffer exists, allocate a new one
    if (geom_buffer == nullptr) {
        cudaError_t err = cudaMalloc(&geom_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate geometry buffer: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        geom_buffer_size = size;
    }
    
    return geom_buffer;
}

char* CudaMemoryManager::getBinningBuffer(size_t size) {
    // If current buffer is too small, free it and allocate a new one
    if (binning_buffer != nullptr && binning_buffer_size < size) {
        cudaFree(binning_buffer);
        binning_buffer = nullptr;
    }
    
    // If no buffer exists, allocate a new one
    if (binning_buffer == nullptr) {
        cudaError_t err = cudaMalloc(&binning_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate binning buffer: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        binning_buffer_size = size;
    }
    
    return binning_buffer;
}

char* CudaMemoryManager::getImageBuffer(size_t size) {
    // If current buffer is too small, free it and allocate a new one
    if (img_buffer != nullptr && img_buffer_size < size) {
        cudaFree(img_buffer);
        img_buffer = nullptr;
    }
    
    // If no buffer exists, allocate a new one
    if (img_buffer == nullptr) {
        cudaError_t err = cudaMalloc(&img_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate image buffer: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        img_buffer_size = size;
    }
    
    return img_buffer;
}

void CudaMemoryManager::freeAll() {
    if (geom_buffer != nullptr) {
        cudaFree(geom_buffer);
        geom_buffer = nullptr;
        geom_buffer_size = 0;
    }
    
    if (binning_buffer != nullptr) {
        cudaFree(binning_buffer);
        binning_buffer = nullptr;
        binning_buffer_size = 0;
    }
    
    if (img_buffer != nullptr) {
        cudaFree(img_buffer);
        img_buffer = nullptr;
        img_buffer_size = 0;
    }
}

// Move markVisible function out of the Rasterizer class and make it a standalone function
void markVisible(
    int N,
    float* means3D,
    float* viewmatrix,
    float* projmatrix,
    bool* present,
    float near_plane,
    float far_plane)
{
    // Clear visibility buffer first
    cudaMemset(present, 0, N * sizeof(bool));
    
    // Use the existing checkFrustum kernel to mark visible Gaussians
    cov3d_cuda::checkFrustum<<<(N + 255) / 256, 256>>>(
        N,
        means3D,
        viewmatrix, projmatrix,
        present,
        near_plane, far_plane);
    
    // Wait for completion
    cudaDeviceSynchronize();
}

Rasterizer::Rasterizer() {}

Rasterizer::~Rasterizer() {
    // Free CUDA memory when the rasterizer is destroyed
    CudaMemoryManager::getInstance().freeAll();
}

// Forward pass implementation
std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
Rasterizer::forward(
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& z_vals,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& colours,
    const torch::Tensor& phase,
    const torch::Tensor& opacities,
    const torch::Tensor& plane_probs,
    float fx, float fy,
    float px, float py,
    const torch::Tensor& view_matrix,
    std::tuple<int, int> img_size,
    std::tuple<int, int> tile_size,
    float near_plane,
    float far_plane)
{
    // Get CUDA stream and device
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int device_id = cam_means_3D.device().index();
    at::cuda::CUDAGuard device_guard(device_id);
    
    // Get dimensions
    const int N = cam_means_3D.size(0);
    const int num_planes = plane_probs.size(1);
    const int channels = colours.size(1);
    const int W = std::get<0>(img_size);
    const int H = std::get<1>(img_size);
    
    // Configure tile grid
    const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
    const int num_tiles = tile_grid.x * tile_grid.y;
    
    // Create output tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(cam_means_3D.device());
    
    auto output_real = torch::zeros({num_planes, channels, H, W}, options);
    auto output_imag = torch::zeros({num_planes, channels, H, W}, options);
    
    // Create visibility buffer
    auto visibility = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(cam_means_3D.device()));
    
    // Mark visible Gaussians (frustum culling)
    markVisible(
        N,
        cam_means_3D.data_ptr<float>(),
        view_matrix.data_ptr<float>(),
        view_matrix.data_ptr<float>(),  // Using view matrix as projmatrix for simplicity
        visibility.data_ptr<bool>(),
        near_plane, far_plane
    );
    
    // Convert visibility boolean tensor to indices using nonzero()
    auto visible_indices = visibility.nonzero().squeeze(1);
    
    // Compute 2D means and covariances with clipping planes
    auto means_2D_tensor = cov3d_cuda::compute_means2d_forward(cam_means_3D, fx, fy, px, py, near_plane, far_plane);
    
    auto view_transform = view_matrix;
    if (view_matrix.dim() == 3) {
        view_transform = view_matrix[0];
    }
    
    // Use the updated compute_cov2d_forward function that returns intermediate values
    auto cov2d_results = cov3d_cuda::compute_cov2d_forward(cam_means_3D, quats, scales, view_transform, fx, fy, W, H, near_plane, far_plane);
    auto cov_2D_tensor = cov2d_results[0];
    
    // Store intermediate tensors as member variables for backward pass
    this->J_tensor = cov2d_results[1];
    this->cov3D_tensor = cov2d_results[2];
    this->W_tensor = cov2d_results[3];
    this->JW_tensor = cov2d_results[4];
    
    auto cov_2D_flat = cov_2D_tensor.view({N, 4});
    
    // Allocate memory using memory manager
    size_t geom_chunk_size = required<cov3d_cuda::GeometryState>(N);
    char* geom_chunk = geometryBuffer(geom_chunk_size);
    cov3d_cuda::GeometryState geom = cov3d_cuda::GeometryState::fromChunk(geom_chunk, N);
    
    // Copy data to device buffers
    cudaMemcpyAsync(geom.means_2D, means_2D_tensor.data_ptr<float>(), N * sizeof(float2), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(geom.cov_2D, cov_2D_flat.data_ptr<float>(), N * sizeof(float4), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(geom.z_vals, z_vals.data_ptr<float>(), N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    // Preprocess Gaussians with clipping planes
    cov3d_cuda::FORWARD_SPLAT::preprocess(
        N,
        geom.means_2D,
        geom.cov_2D,
        geom.z_vals,
        geom.radii,
        geom.tiles_touched,
        tile_grid,
        false,  // prefiltered
        true,   // antialiasing is always on as requested
        near_plane, far_plane
    );
    
    // Compute prefix sum
    cub::DeviceScan::InclusiveSum(geom.scanning_space, geom.scan_size, geom.tiles_touched, geom.point_offsets, N, stream);
    
    // Get total number of key-value pairs
    int total_pairs = 0;
    cudaMemcpyAsync(&total_pairs, geom.point_offsets + (N - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (total_pairs == 0) {
        // No Gaussians to render, return early
        auto output = torch::complex(output_real, output_imag);
        auto final_Ts = torch::ones({num_planes, H, W}, options);
        auto n_contrib = torch::zeros({num_planes, H, W}, torch::TensorOptions().dtype(torch::kInt32).device(cam_means_3D.device()));
        auto ranges = torch::zeros({num_tiles, 2}, torch::TensorOptions().dtype(torch::kInt32).device(cam_means_3D.device()));
        auto point_list = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(cam_means_3D.device()));
        
        // Return empty visible indices tensor
        auto empty_visible = torch::zeros({0}, torch::TensorOptions().dtype(torch::kLong).device(cam_means_3D.device()));
        
        return std::make_tuple(
            output, 
            std::make_tuple(final_Ts, n_contrib, point_list, ranges, empty_visible)
        );
    }
    
    // Allocate binning state using memory manager
    size_t binning_chunk_size = required<cov3d_cuda::BinningState>(total_pairs);
    char* binning_chunk = binningBuffer(binning_chunk_size);
    cov3d_cuda::BinningState binning = cov3d_cuda::BinningState::fromChunk(binning_chunk, total_pairs);
    
    // Create key-value pairs with proper depth sorting, using clipping planes
    const dim3 preproc_block(256);
    const dim3 preproc_grid((N + preproc_block.x - 1) / preproc_block.x);
    
    cov3d_cuda::duplicateWithKeysKernel<<<preproc_grid, preproc_block, 0, stream>>>(
        N,
        geom.means_2D,
        geom.z_vals,
        geom.radii,
        geom.point_offsets,
        binning.keys_unsorted,
        binning.values_unsorted,
        tile_grid,
        near_plane, far_plane
    );
    
    // Sort key-value pairs
    int bit = cov3d_cuda::getHigherMsb(tile_grid.x * tile_grid.y);
    
    cub::DeviceRadixSort::SortPairs(
        binning.sorting_space,
        binning.sorting_size,
        binning.keys_unsorted, binning.keys_sorted,
        binning.values_unsorted, binning.values_sorted,
        total_pairs, 0, 32 + bit, stream);
    
    // Allocate image state using memory manager
    size_t img_chunk_size = required<cov3d_cuda::ImageState>(num_tiles, H * W, num_planes);
    char* img_chunk = imageBuffer(img_chunk_size);
    cov3d_cuda::ImageState img = cov3d_cuda::ImageState::fromChunk(img_chunk, num_tiles, H * W, num_planes);
    
    // Reset range values
    cudaMemsetAsync(img.ranges, 0, num_tiles * sizeof(uint2), stream);
    
    // Identify ranges for each tile
    cov3d_cuda::identifyTileRangesKernel<<<(total_pairs + 255) / 256, 256, 0, stream>>>(
        total_pairs,
        binning.keys_sorted,
        img.ranges
    );
    
    // Initialize transmittance and contribution count
    cudaMemsetAsync(img.n_contrib, 0, num_planes * H * W * sizeof(uint32_t), stream);
    
    // Fill final_Ts with ones
    const int fill_block_size = 256;
    const int fill_grid_size = (num_planes * H * W + fill_block_size - 1) / fill_block_size;
    
    cov3d_cuda::fillOnesKernel<<<fill_grid_size, fill_block_size, 0, stream>>>(
        img.final_Ts, 
        num_planes * H * W
    );
    
    // Render tiles using clipping planes
    const dim3 render_block(BLOCK_X, BLOCK_Y);
    
    cov3d_cuda::FORWARD_SPLAT::render(
        tile_grid, render_block,
        img.ranges,
        binning.values_sorted,
        geom.means_2D,
        geom.cov_2D,
        geom.z_vals,
        colours.data_ptr<float>(),
        phase.data_ptr<float>(),
        opacities.data_ptr<float>(),
        plane_probs.data_ptr<float>(),
        output_real.data_ptr<float>(),
        output_imag.data_ptr<float>(),
        img.final_Ts,
        img.n_contrib,
        W, H, N,
        num_planes, channels,
        near_plane, far_plane
    );
    
    // Create output
    auto output = torch::complex(output_real, output_imag);
    
    // Create tensors for backward pass
    auto final_Ts_tensor = torch::from_blob(img.final_Ts, {num_planes, H, W}, options).clone();
    auto n_contrib_tensor = torch::from_blob(img.n_contrib, {num_planes, H, W}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(cam_means_3D.device())).clone();
    auto point_list_tensor = torch::from_blob(binning.values_sorted, {total_pairs}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(cam_means_3D.device())).clone();
    auto ranges_tensor = torch::from_blob(img.ranges, {num_tiles, 2}, 
                                       torch::TensorOptions().dtype(torch::kInt32).device(cam_means_3D.device())).clone();
    
    return std::make_tuple(
        output, 
        std::make_tuple(final_Ts_tensor, n_contrib_tensor, point_list_tensor, ranges_tensor, visible_indices)
    );
}

// Backward pass implementation
std::vector<torch::Tensor> Rasterizer::backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& z_vals,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& colours,
    const torch::Tensor& phase,
    const torch::Tensor& opacities,
    const torch::Tensor& plane_probs,
    float fx, float fy,
    float px, float py,
    const torch::Tensor& view_matrix,
    std::tuple<int, int> img_size,
    std::tuple<int, int> tile_size,
    const torch::Tensor& final_Ts,
    const torch::Tensor& n_contrib,
    const torch::Tensor& point_list,
    const torch::Tensor& ranges,
    float near_plane,
    float far_plane)
{
    // Get CUDA stream and device
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int device_id = cam_means_3D.device().index();
    at::cuda::CUDAGuard device_guard(device_id);
    
    // Get dimensions
    const int N = cam_means_3D.size(0);
    const int num_planes = plane_probs.size(1);
    const int channels = colours.size(1);
    const int W = std::get<0>(img_size);
    const int H = std::get<1>(img_size);
    
    // Create gradient tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(cam_means_3D.device());
    
    auto grad_cam_means_3D = torch::zeros({N, 3}, options);
    auto grad_z_vals = torch::zeros({N}, options);
    auto grad_quats = torch::zeros({N, 4}, options);
    auto grad_scales = torch::zeros({N, 3}, options);
    auto grad_colours = torch::zeros({N, channels}, options);
    auto grad_phase = torch::zeros({N, channels}, options);
    auto grad_opacities = torch::zeros({N}, options);
    auto grad_plane_probs = torch::zeros({N, num_planes}, options);
    
    // Create intermediate 2D gradients
    auto grad_means_2D = torch::zeros({N, 2}, options);
    auto grad_cov_2D = torch::zeros({N, 4}, options);
    
    // Split complex grad_output
    auto grad_output_real = torch::real(grad_output).contiguous();
    auto grad_output_imag = torch::imag(grad_output).contiguous();
    
    // Compute 2D means and covariances with clipping planes
    auto means_2D_tensor = cov3d_cuda::compute_means2d_forward(cam_means_3D, fx, fy, px, py, near_plane, far_plane);
    
    auto view_transform = view_matrix;
    if (view_matrix.dim() == 3) {
        view_transform = view_matrix[0];
    }
    
    // Use the stored intermediate values from forward pass
    // Recompute the 2D covariance instead of incorrectly using cov3D_tensor
    auto cov2d_results = cov3d_cuda::compute_cov2d_forward(cam_means_3D, quats, scales, view_transform, fx, fy, W, H, near_plane, far_plane);
    auto cov_2D_tensor = cov2d_results[0]; // Get the 2D covariance from results
    auto cov_2D_flat = cov_2D_tensor.view({N, 4}); // Reshape it correctly
    
    // Configure kernel dimensions
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    
    // Launch backward render kernel with clipping planes
    cov3d_cuda::BACKWARD_SPLAT::render(
        grid, block,
        (uint2*)ranges.data_ptr<int>(),
        (uint32_t*)point_list.data_ptr<int>(),
        (float2*)means_2D_tensor.data_ptr<float>(),
        (float4*)cov_2D_flat.data_ptr<float>(),
        z_vals.data_ptr<float>(),
        colours.data_ptr<float>(),
        phase.data_ptr<float>(),
        opacities.data_ptr<float>(),
        plane_probs.data_ptr<float>(),
        final_Ts.data_ptr<float>(),
        (uint32_t*)n_contrib.data_ptr<int>(),
        grad_output_real.data_ptr<float>(),
        grad_output_imag.data_ptr<float>(),
        grad_means_2D.data_ptr<float>(),
        grad_cov_2D.data_ptr<float>(),
        grad_z_vals.data_ptr<float>(),
        grad_colours.data_ptr<float>(),
        grad_phase.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        grad_plane_probs.data_ptr<float>(),
        N, num_planes, channels,
        W, H,
        near_plane, far_plane
    );
    
    // Transform 2D gradients to 3D parameters with clipping planes
    cudaStreamSynchronize(stream);
    
    cov3d_cuda::BACKWARD_SPLAT::preprocess(
        fx, fy,
        cam_means_3D,
        quats,
        scales,
        view_transform,
        grad_means_2D,
        grad_cov_2D,
        grad_cam_means_3D,
        grad_quats,
        grad_scales,
        W, H,
        near_plane, far_plane,
        this->J_tensor,      
        this->cov3D_tensor,  
        this->W_tensor,      
        this->JW_tensor      
    );
    
    // Return gradients
    return {
        grad_cam_means_3D,
        grad_z_vals,
        grad_quats,
        grad_scales,
        grad_colours,
        grad_phase,
        grad_opacities,
        grad_plane_probs
    };
}

} // namespace CudaRasterizer