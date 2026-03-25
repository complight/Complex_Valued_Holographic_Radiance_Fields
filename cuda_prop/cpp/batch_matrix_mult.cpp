#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA functions
void batch_matrix_multiplication_forward_cuda(
    const torch::Tensor& diff,         // (N, H*W, 2)
    const torch::Tensor& cov_inv,      // (N, 2, 2)
    torch::Tensor& output);            // Output will be (N, H*W, 2)

void batch_matrix_multiplication_backward_cuda(
    const torch::Tensor& grad_output,  // (N, H*W, 2)
    const torch::Tensor& diff,         // (N, H*W, 2)
    const torch::Tensor& cov_inv,      // (N, 2, 2)
    torch::Tensor& grad_diff,          // (N, H*W, 2)
    torch::Tensor& grad_cov_inv);      // (N, 2, 2)

// Direct memory mapping - avoids unnecessary allocations
torch::Tensor batch_matrix_multiplication_forward(
    const torch::Tensor& diff,          // (N, H*W, 2)
    const torch::Tensor& cov_inv)       // (N, 2, 2)
{
    // Memory optimization: use torch::empty directly with options()
    // This avoids creating unnecessary temporary tensors or using extra flags
    auto output = torch::empty({diff.size(0), diff.size(1), 2}, 
                              diff.options());
    
    // Avoid creating contiguous copies by passing tensors directly to CUDA
    batch_matrix_multiplication_forward_cuda(diff, cov_inv, output);
    
    return output;
}

std::vector<torch::Tensor> batch_matrix_multiplication_backward(
    const torch::Tensor& grad_output,    // (N, H*W, 2)
    const torch::Tensor& diff,           // (N, H*W, 2)
    const torch::Tensor& cov_inv)        // (N, 2, 2)
{
    // Memory optimization: specify device and dtype explicitly from input
    auto grad_diff = torch::zeros({diff.size(0), diff.size(1), 2}, diff.options());
    auto grad_cov_inv = torch::zeros({cov_inv.size(0), 2, 2}, cov_inv.options());
    
    // Avoid creating contiguous copies by passing tensors directly to CUDA
    batch_matrix_multiplication_backward_cuda(
        grad_output, 
        diff, 
        cov_inv,
        grad_diff, 
        grad_cov_inv);
    
    return {grad_diff, grad_cov_inv};
}