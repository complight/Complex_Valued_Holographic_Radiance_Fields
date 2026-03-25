#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Forward kernel to sum along the last dimension - optimized version
template <typename scalar_t>
__global__ void sum_last_dim_forward_kernel(
    const scalar_t* __restrict__ input,   // Input tensor (N, HW, 2)
    scalar_t* __restrict__ output,        // Output tensor (N, HW)
    int batch_size,                       // N
    int hw_size                           // HW
) {
    // Calculate batch index - process multiple batches per block for better occupancy
    const int batch_idx = blockIdx.x;
    
    // Each thread processes multiple HW positions to increase arithmetic intensity
    const int tid = threadIdx.x;
    const int thread_stride = blockDim.x;
    
    // Process positions with thread striding pattern
    for (int hw_idx = tid; hw_idx < hw_size; hw_idx += thread_stride) {
        // Calculate input base index - uses direct computation to minimize register usage
        const int in_base_idx = batch_idx * (hw_size * 2) + hw_idx * 2;
        
        // Load both values at once if architecture supports it
        // This reduces memory transactions and increases memory coalescing
        #if __CUDA_ARCH__ >= 350
            // Use vector load for architectures that support it (compute capability >= 3.5)
            float2 vals = *reinterpret_cast<const float2*>(&input[in_base_idx]);
            const scalar_t val0 = vals.x;
            const scalar_t val1 = vals.y;
        #else
            // Fallback to scalar loads
            const scalar_t val0 = input[in_base_idx];
            const scalar_t val1 = input[in_base_idx + 1];
        #endif
        
        // Compute sum and write to output in a single operation
        output[batch_idx * hw_size + hw_idx] = val0 + val1;
    }
}

// Backward kernel to distribute gradients - optimized version
template <typename scalar_t>
__global__ void sum_last_dim_backward_kernel(
    const scalar_t* __restrict__ grad_output,  // Gradient from output (N, HW)
    scalar_t* __restrict__ grad_input,         // Gradient for input (N, HW, 2)
    int batch_size,                            // N
    int hw_size                                // HW
) {
    // Calculate batch index - process multiple batches per block for better occupancy
    const int batch_idx = blockIdx.x;
    
    // Each thread processes multiple HW positions
    const int tid = threadIdx.x;
    const int thread_stride = blockDim.x;
    
    // Process positions with thread striding pattern
    for (int hw_idx = tid; hw_idx < hw_size; hw_idx += thread_stride) {
        // Get gradient value - direct indexing for minimal register usage
        const scalar_t grad = grad_output[batch_idx * hw_size + hw_idx];
        
        // Calculate input base index
        const int in_base_idx = batch_idx * (hw_size * 2) + hw_idx * 2;
        
        // Use vector write if architecture supports it
        #if __CUDA_ARCH__ >= 350
            // Vector write for better memory throughput
            *reinterpret_cast<float2*>(&grad_input[in_base_idx]) = make_float2(grad, grad);
        #else
            // Fallback to scalar writes
            grad_input[in_base_idx] = grad;
            grad_input[in_base_idx + 1] = grad;
        #endif
    }
}

// Optimized C++ interface for forward operation
void sum_last_dim_forward_cuda(
    const torch::Tensor& input,
    torch::Tensor& output) {
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int hw_size = input.size(1);
    
    const int threads_per_block = 256;  // Good balance for all GPU architectures
    
    // Use a 1D grid for better occupancy and simplified indexing
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_last_dim_forward", ([&] {
        sum_last_dim_forward_kernel<scalar_t><<<batch_size, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hw_size);
    }));
}

// Optimized C++ interface for backward operation
void sum_last_dim_backward_cuda(
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input) {
    
    // Get dimensions
    const int batch_size = grad_output.size(0);
    const int hw_size = grad_output.size(1);
    
    // Configure CUDA kernel - optimize for occupancy
    const int threads_per_block = 256;
    
    // Use a 1D grid for better occupancy
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "sum_last_dim_backward", ([&] {
        sum_last_dim_backward_kernel<scalar_t><<<batch_size, threads_per_block>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            batch_size,
            hw_size);
    }));
}