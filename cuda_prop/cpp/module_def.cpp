#include <torch/extension.h>

// Function declarations from other files with correct signatures
std::vector<torch::Tensor> bandlimited_propagation_forward(
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    const torch::Tensor& fx,
    const torch::Tensor& fy,
    float wavelength,
    float distance,
    float aperture_size, 
    float pixel_pitch);

std::vector<torch::Tensor> bandlimited_propagation_backward(
    const torch::Tensor& grad_output_real,
    const torch::Tensor& grad_output_imag,
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    const torch::Tensor& fx,
    const torch::Tensor& fy,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch);

torch::Tensor batch_matrix_multiplication_forward(
    const torch::Tensor& diff,
    const torch::Tensor& cov_inv);

std::vector<torch::Tensor> batch_matrix_multiplication_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& diff,
    const torch::Tensor& cov_inv);

torch::Tensor sum_last_dim_forward(
    const torch::Tensor& input);

torch::Tensor sum_last_dim_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input);

torch::Tensor element_wise_multiplication_forward(
    const torch::Tensor& input1,
    const torch::Tensor& input2);

std::vector<torch::Tensor> element_wise_multiplication_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input1,
    const torch::Tensor& input2);

    
// === PYBIND11 module definition ===
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("bandlimited_propagation_forward", &bandlimited_propagation_forward,
        "Bandlimited propagation forward (CUDA)");
    m.def("bandlimited_propagation_backward", &bandlimited_propagation_backward,
        "Bandlimited propagation backward (CUDA)");

    m.def("batch_matrix_multiplication_forward", &batch_matrix_multiplication_forward,
          "Batch matrix multiplication forward (CUDA)");
    
    m.def("batch_matrix_multiplication_backward", &batch_matrix_multiplication_backward,
          "Batch matrix multiplication backward (CUDA)");
    
    m.def("sum_last_dim_forward", &sum_last_dim_forward,
          "Sum last dimension forward (CUDA)");
    
    m.def("sum_last_dim_backward", &sum_last_dim_backward,
          "Sum last dimension backward (CUDA)");
    
    m.def("element_wise_multiplication_forward", &element_wise_multiplication_forward,
          "Element-wise multiplication forward (CUDA)");
    
    m.def("element_wise_multiplication_backward", &element_wise_multiplication_backward,
          "Element-wise multiplication backward (CUDA)");
}