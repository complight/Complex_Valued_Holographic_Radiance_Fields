#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Memory-optimized kernel with shared memory for covariance matrix
template <typename scalar_t>
__global__ void batch_matrix_multiplication_forward_kernel(
    const scalar_t* __restrict__ diff,     // Input tensor (N, H*W, 2)
    const scalar_t* __restrict__ cov_inv,  // Covariance inverse (N, 2, 2)
    scalar_t* __restrict__ output,         // Output tensor (N, H*W, 2)
    int batch_size,                        // N
    int hw_size                            // H*W
) {
    // Calculate batch index
    const int batch_idx = blockIdx.x;
    
    // Each thread processes one hw position
    const int tid = threadIdx.x;
    const int thread_stride = blockDim.x;
    
    // Shared memory for covariance matrix (one per batch)
    __shared__ scalar_t shared_cov[4];
    
    // First thread in block loads the covariance matrix to shared memory
    if (tid < 4) {
        shared_cov[tid] = cov_inv[batch_idx * 4 + tid];
    }
    
    // Make sure all threads have access to the shared memory
    __syncthreads();
    
    // Load from shared memory
    const scalar_t cov00 = shared_cov[0];
    const scalar_t cov01 = shared_cov[1];
    const scalar_t cov10 = shared_cov[2];
    const scalar_t cov11 = shared_cov[3];
    
    // Process positions with thread striding pattern
    for (int hw_idx = tid; hw_idx < hw_size; hw_idx += thread_stride) {
        // Calculate input index
        const int in_idx = batch_idx * (hw_size * 2) + hw_idx * 2;
        
        // Load input values
        const scalar_t diff_x = diff[in_idx];
        const scalar_t diff_y = diff[in_idx + 1];
        
        // Perform matrix multiplication
        const scalar_t result_x = diff_x * cov00 + diff_y * cov10;
        const scalar_t result_y = diff_x * cov01 + diff_y * cov11;
        
        // Write to output
        const int out_idx = batch_idx * (hw_size * 2) + hw_idx * 2;
        output[out_idx] = result_x;
        output[out_idx + 1] = result_y;
    }
}

// Optimized backward kernel with warp-level reduction and shared memory
template <typename scalar_t>
__global__ void batch_matrix_multiplication_backward_kernel(
    const scalar_t* __restrict__ grad_output,  // Gradient from output (N, H*W, 2)
    const scalar_t* __restrict__ diff,         // Original input (N, H*W, 2)
    const scalar_t* __restrict__ cov_inv,      // Original covariance inverse (N, 2, 2)
    scalar_t* __restrict__ grad_diff,          // Gradient for diff (N, H*W, 2)
    scalar_t* __restrict__ grad_cov_inv,       // Gradient for cov_inv (N, 2, 2)
    int batch_size,                            // N
    int hw_size                                // H*W
) {
    // Calculate batch index
    const int batch_idx = blockIdx.x;
    
    // Each thread processes one hw position
    const int tid = threadIdx.x;
    const int thread_stride = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for covariance matrix and per-warp gradients
    __shared__ scalar_t shared_cov[4];
    __shared__ scalar_t warp_grad_cov[4 * 8]; // Support up to 8 warps
    
    // First thread in block loads the covariance matrix to shared memory
    if (tid < 4) {
        shared_cov[tid] = cov_inv[batch_idx * 4 + tid];
    }
    
    // Initialize warp-level gradient accumulation
    if (lane_id == 0 && warp_id < 8) {
        for (int i = 0; i < 4; i++) {
            warp_grad_cov[warp_id * 4 + i] = 0;
        }
    }
    
    // Make sure all threads have access to the shared memory
    __syncthreads();
    
    // Load from shared memory
    const scalar_t cov00 = shared_cov[0];
    const scalar_t cov01 = shared_cov[1];
    const scalar_t cov10 = shared_cov[2];
    const scalar_t cov11 = shared_cov[3];
    
    // Thread local accumulators for gradient of cov_inv - register usage optimization
    scalar_t grad_cov00 = 0;
    scalar_t grad_cov01 = 0;
    scalar_t grad_cov10 = 0;
    scalar_t grad_cov11 = 0;
    
    // Process positions with thread striding pattern
    for (int hw_idx = tid; hw_idx < hw_size; hw_idx += thread_stride) {
        // Calculate indices
        const int in_idx = batch_idx * (hw_size * 2) + hw_idx * 2;
        
        // Load values
        const scalar_t diff_x = diff[in_idx];
        const scalar_t diff_y = diff[in_idx + 1];
        const scalar_t grad_x = grad_output[in_idx];
        const scalar_t grad_y = grad_output[in_idx + 1];
        
        // Compute gradient for diff
        grad_diff[in_idx] = grad_x * cov00 + grad_y * cov10;
        grad_diff[in_idx + 1] = grad_x * cov01 + grad_y * cov11;
        
        // Accumulate gradients for cov_inv
        grad_cov00 += diff_x * grad_x;
        grad_cov01 += diff_x * grad_y;
        grad_cov10 += diff_y * grad_x;
        grad_cov11 += diff_y * grad_y;
    }
    
    // Use warp-level reduction to combine gradients
    // This reduces memory pressure from atomic operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        grad_cov00 += __shfl_down_sync(0xffffffff, grad_cov00, offset);
        grad_cov01 += __shfl_down_sync(0xffffffff, grad_cov01, offset);
        grad_cov10 += __shfl_down_sync(0xffffffff, grad_cov10, offset);
        grad_cov11 += __shfl_down_sync(0xffffffff, grad_cov11, offset);
    }
    
    // First thread in each warp accumulates to the warp's shared memory
    if (lane_id == 0 && warp_id < 8) {
        warp_grad_cov[warp_id * 4 + 0] = grad_cov00;
        warp_grad_cov[warp_id * 4 + 1] = grad_cov01;
        warp_grad_cov[warp_id * 4 + 2] = grad_cov10;
        warp_grad_cov[warp_id * 4 + 3] = grad_cov11;
    }
    
    // Make sure all threads have updated the shared memory
    __syncthreads();
    
    // Single reduction across all warps - only done by first warp
    if (warp_id == 0 && lane_id < 4) {
        scalar_t sum = 0;
        for (int w = 0; w < 8; w++) {
            sum += warp_grad_cov[w * 4 + lane_id];
        }
        
        // Single atomic operation per gradient element
        atomicAdd(&grad_cov_inv[batch_idx * 4 + lane_id], sum);
    }
}

// Optimized forward CUDA implementation
void batch_matrix_multiplication_forward_cuda(
    const torch::Tensor& diff,
    const torch::Tensor& cov_inv,
    torch::Tensor& output) {
    
    // Get dimensions
    const int batch_size = diff.size(0);
    const int hw_size = diff.size(1);
    
    // Determine optimal thread count based on hardware capability
    const int threads_per_block = std::min(256, hw_size);
    
    AT_DISPATCH_FLOATING_TYPES(diff.scalar_type(), "batch_matrix_multiplication_forward", ([&] {
        batch_matrix_multiplication_forward_kernel<scalar_t><<<batch_size, threads_per_block>>>(
            diff.data_ptr<scalar_t>(),
            cov_inv.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hw_size);
    }));
}

// Optimized backward CUDA implementation
void batch_matrix_multiplication_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& diff,
    const torch::Tensor& cov_inv,
    torch::Tensor& grad_diff,
    torch::Tensor& grad_cov_inv) {
    
    // Get dimensions
    const int batch_size = diff.size(0);
    const int hw_size = diff.size(1);
    
    // Determine optimal thread count based on hardware capability and workload
    const int threads_per_block = std::min(256, hw_size);
    
    AT_DISPATCH_FLOATING_TYPES(diff.scalar_type(), "batch_matrix_multiplication_backward", ([&] {
        batch_matrix_multiplication_backward_kernel<scalar_t><<<batch_size, threads_per_block>>>(
            grad_output.data_ptr<scalar_t>(),
            diff.data_ptr<scalar_t>(),
            cov_inv.data_ptr<scalar_t>(),
            grad_diff.data_ptr<scalar_t>(),
            grad_cov_inv.data_ptr<scalar_t>(),
            batch_size,
            hw_size);
    }));
}