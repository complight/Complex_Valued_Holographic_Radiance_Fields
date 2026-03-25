#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Forward kernel for element-wise multiplication - optimized version
template <typename scalar_t>
__global__ void element_wise_multiplication_forward_kernel(
    const scalar_t* __restrict__ input1,  // First input tensor (N, HW, 2)
    const scalar_t* __restrict__ input2,  // Second input tensor (N, HW, 2)
    scalar_t* __restrict__ output,        // Output tensor (N, HW, 2)
    int batch_size,                       // N
    int hw_size                           // HW
) {
    // Calculate batch index
    const int batch_idx = blockIdx.x;
    
    // Each thread processes multiple HW positions to increase arithmetic intensity
    const int tid = threadIdx.x;
    const int thread_stride = blockDim.x;
    
    // Process positions with thread striding pattern
    for (int hw_idx = tid; hw_idx < hw_size; hw_idx += thread_stride) {
        // Calculate input base indices - direct computation to minimize register usage
        const int idx = batch_idx * (hw_size * 2) + hw_idx * 2;
        
        // Load values from inputs
        #if __CUDA_ARCH__ >= 350
            // Use vector loads for better memory throughput on supported architectures
            float2 vals1 = *reinterpret_cast<const float2*>(&input1[idx]);
            float2 vals2 = *reinterpret_cast<const float2*>(&input2[idx]);
            
            // Perform element-wise multiplication
            float2 result;
            result.x = vals1.x * vals2.x;
            result.y = vals1.y * vals2.y;
            
            // Write result using vector write
            *reinterpret_cast<float2*>(&output[idx]) = result;
        #else
            // Fallback to scalar operations for older architectures
            // Load values
            const scalar_t val1_x = input1[idx];
            const scalar_t val1_y = input1[idx + 1];
            const scalar_t val2_x = input2[idx];
            const scalar_t val2_y = input2[idx + 1];
            
            // Perform element-wise multiplication
            output[idx] = val1_x * val2_x;
            output[idx + 1] = val1_y * val2_y;
        #endif
    }
}

// Backward kernel for element-wise multiplication - optimized version
template <typename scalar_t>
__global__ void element_wise_multiplication_backward_kernel(
    const scalar_t* __restrict__ grad_output,  // Gradient from output (N, HW, 2)
    const scalar_t* __restrict__ input1,       // First input tensor (N, HW, 2)
    const scalar_t* __restrict__ input2,       // Second input tensor (N, HW, 2)
    scalar_t* __restrict__ grad_input1,        // Gradient for input1 (N, HW, 2)
    scalar_t* __restrict__ grad_input2,        // Gradient for input2 (N, HW, 2)
    int batch_size,                            // N
    int hw_size                                // HW
) {
    // Calculate batch index
    const int batch_idx = blockIdx.x;
    
    // Each thread processes multiple HW positions
    const int tid = threadIdx.x;
    const int thread_stride = blockDim.x;
    
    // Process positions with thread striding pattern
    for (int hw_idx = tid; hw_idx < hw_size; hw_idx += thread_stride) {
        // Calculate index
        const int idx = batch_idx * (hw_size * 2) + hw_idx * 2;
        
        // Load values using the most efficient memory access pattern
        #if __CUDA_ARCH__ >= 350
            // Vector loads for better memory throughput
            float2 grad_vals = *reinterpret_cast<const float2*>(&grad_output[idx]);
            float2 vals1 = *reinterpret_cast<const float2*>(&input1[idx]);
            float2 vals2 = *reinterpret_cast<const float2*>(&input2[idx]);
            
            // Compute gradients
            float2 grad1, grad2;
            grad1.x = grad_vals.x * vals2.x;
            grad1.y = grad_vals.y * vals2.y;
            grad2.x = grad_vals.x * vals1.x;
            grad2.y = grad_vals.y * vals1.y;
            
            // Write results using vector writes
            *reinterpret_cast<float2*>(&grad_input1[idx]) = grad1;
            *reinterpret_cast<float2*>(&grad_input2[idx]) = grad2;
        #else
            // Scalar operations for older architectures
            const scalar_t grad_x = grad_output[idx];
            const scalar_t grad_y = grad_output[idx + 1];
            const scalar_t val1_x = input1[idx];
            const scalar_t val1_y = input1[idx + 1];
            const scalar_t val2_x = input2[idx];
            const scalar_t val2_y = input2[idx + 1];
            
            // Gradients for input1: grad_output * input2
            grad_input1[idx] = grad_x * val2_x;
            grad_input1[idx + 1] = grad_y * val2_y;
            
            // Gradients for input2: grad_output * input1
            grad_input2[idx] = grad_x * val1_x;
            grad_input2[idx + 1] = grad_y * val1_y;
        #endif
    }
}

// Optimized CUDA implementation for forward operation
void element_wise_multiplication_forward_cuda(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    torch::Tensor& output) {
    
    // Get dimensions
    const int batch_size = input1.size(0);
    const int hw_size = input1.size(1);
    
    // Thread configuration - optimal for most GPUs
    const int threads_per_block = 256;
    
    // Launch kernel with one block per batch
    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "element_wise_multiplication_forward", ([&] {
        element_wise_multiplication_forward_kernel<scalar_t><<<batch_size, threads_per_block>>>(
            input1.data_ptr<scalar_t>(),
            input2.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hw_size);
    }));
}

// Optimized CUDA implementation for backward operation
void element_wise_multiplication_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    torch::Tensor& grad_input1,
    torch::Tensor& grad_input2) {
    
    // Get dimensions
    const int batch_size = input1.size(0);
    const int hw_size = input1.size(1);
    
    // Thread configuration - optimal for most GPUs
    const int threads_per_block = 256;
    
    // Launch kernel with one block per batch for optimal occupancy
    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "element_wise_multiplication_backward", ([&] {
        element_wise_multiplication_backward_kernel<scalar_t><<<batch_size, threads_per_block>>>(
            grad_output.data_ptr<scalar_t>(),
            input1.data_ptr<scalar_t>(),
            input2.data_ptr<scalar_t>(),
            grad_input1.data_ptr<scalar_t>(),
            grad_input2.data_ptr<scalar_t>(),
            batch_size,
            hw_size);
    }));
}