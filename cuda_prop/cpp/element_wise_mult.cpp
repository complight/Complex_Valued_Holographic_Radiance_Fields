#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA functions for element-wise multiplication
void element_wise_multiplication_forward_cuda(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    torch::Tensor& output);

void element_wise_multiplication_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    torch::Tensor& grad_input1,
    torch::Tensor& grad_input2);

// Memory-optimized C++ interface for element-wise multiplication forward operation
torch::Tensor element_wise_multiplication_forward(
    const torch::Tensor& input1,
    const torch::Tensor& input2) 
{
    // Validate input shapes - early return if invalid to avoid unnecessary allocations
    TORCH_CHECK(input1.dim() == 3 && input1.size(2) == 2,
               "input1 must have shape (N, HW, 2), but got ", input1.sizes());
    TORCH_CHECK(input2.dim() == 3 && input2.size(2) == 2,
               "input2 must have shape (N, HW, 2), but got ", input2.sizes());
    TORCH_CHECK(input1.size(0) == input2.size(0) && input1.size(1) == input2.size(1),
               "input1 and input2 batch and sequence dimensions must match");
    
    // Get dimensions
    int batch_size = input1.size(0);
    int hw_size = input1.size(1);
    
    // Ensure inputs are contiguous for optimal memory access
    auto input1_contig = input1.contiguous();
    auto input2_contig = input2.contiguous();
    
    // Create output tensor with optimal memory layout (contiguous)
    auto output = torch::empty({batch_size, hw_size, 2}, 
                              torch::TensorOptions()
                               .dtype(input1.dtype())
                               .device(input1.device())
                               .memory_format(torch::MemoryFormat::Contiguous));
    
    element_wise_multiplication_forward_cuda(input1_contig, input2_contig, output);
    
    return output;
}

// Memory-optimized C++ interface for element-wise multiplication backward operation
std::vector<torch::Tensor> element_wise_multiplication_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input1,
    const torch::Tensor& input2) 
{
    // Validate shapes - early return if invalid
    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(2) == 2,
               "grad_output must have shape (N, HW, 2), but got ", grad_output.sizes());
    TORCH_CHECK(input1.dim() == 3 && input1.size(2) == 2,
               "input1 must have shape (N, HW, 2), but got ", input1.sizes());
    TORCH_CHECK(input2.dim() == 3 && input2.size(2) == 2,
               "input2 must have shape (N, HW, 2), but got ", input2.sizes());
    TORCH_CHECK(grad_output.size(0) == input1.size(0) && grad_output.size(1) == input1.size(1),
               "grad_output and input dimensions must match");
    
    // Get dimensions
    int batch_size = input1.size(0);
    int hw_size = input1.size(1);
    
    // Make sure tensors are contiguous for optimal memory access
    auto grad_output_contig = grad_output.contiguous();
    auto input1_contig = input1.contiguous();
    auto input2_contig = input2.contiguous();
    
    // Create gradient tensors with optimal memory layout
    auto grad_input1 = torch::zeros({batch_size, hw_size, 2},
                                  torch::TensorOptions()
                                   .dtype(input1.dtype())
                                   .device(input1.device())
                                   .memory_format(torch::MemoryFormat::Contiguous));
    
    auto grad_input2 = torch::zeros({batch_size, hw_size, 2},
                                  torch::TensorOptions()
                                   .dtype(input2.dtype())
                                   .device(input2.device())
                                   .memory_format(torch::MemoryFormat::Contiguous));
    
    element_wise_multiplication_backward_cuda(grad_output_contig, input1_contig, input2_contig, grad_input1, grad_input2);
    
    return {grad_input1, grad_input2};
}