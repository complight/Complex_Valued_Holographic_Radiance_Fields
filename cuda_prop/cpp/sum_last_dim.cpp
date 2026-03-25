#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA functions for sum_last_dim
void sum_last_dim_forward_cuda(
    const torch::Tensor& input,
    torch::Tensor& output);

void sum_last_dim_backward_cuda(
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input);

// Memory-optimized C++ interface for sum_last_dim forward operation
torch::Tensor sum_last_dim_forward(
    const torch::Tensor& input) 
{
    // Validate input shape - early return if invalid to avoid unnecessary allocations
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 2,
               "input must have shape (N, HW, 2), but got ", input.sizes());
    
    // Get dimensions
    int batch_size = input.size(0);
    int hw_size = input.size(1);
    
    // For small tensors, PyTorch's implementation might be faster due to CUDA launch overhead
    // Choose a reasonable threshold based on testing
    const int THRESHOLD = 1024; // Adjust this based on benchmarking

    // Use existing memory for output if possible and ensure contiguity in one step
    // This avoids unnecessary memory allocations and copies
    auto input_contig = input.contiguous();
    
    // Create output tensor with optimal memory layout (contiguous)
    auto output = torch::empty({batch_size, hw_size}, 
                              torch::TensorOptions()
                               .dtype(input.dtype())
                               .device(input.device())
                               .memory_format(torch::MemoryFormat::Contiguous));
    
    sum_last_dim_forward_cuda(input_contig, output);
    
    return output;
}

// Memory-optimized C++ interface for sum_last_dim backward operation
torch::Tensor sum_last_dim_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input) 
{
    // Validate shapes - early return if invalid
    TORCH_CHECK(grad_output.dim() == 2,
               "grad_output must have shape (N, HW), but got ", grad_output.sizes());
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 2,
               "input must have shape (N, HW, 2), but got ", input.sizes());
    TORCH_CHECK(grad_output.size(0) == input.size(0) && grad_output.size(1) == input.size(1),
               "grad_output and input batch and sequence dimensions must match");
    
    // Get dimensions
    int batch_size = input.size(0);
    int hw_size = input.size(1);
    
    // For small tensors, PyTorch's implementation might be faster
    const int THRESHOLD = 1024; // Adjust based on benchmarking
    
    // Make sure grad_output is contiguous for optimal memory access pattern
    auto grad_output_contig = grad_output.contiguous();
    
    // Create gradient tensor with optimal memory layout
    auto grad_input = torch::zeros({batch_size, hw_size, 2},
                                  torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device())
                                   .memory_format(torch::MemoryFormat::Contiguous));
    
    sum_last_dim_backward_cuda(grad_output_contig, grad_input);
    
    return grad_input;
}