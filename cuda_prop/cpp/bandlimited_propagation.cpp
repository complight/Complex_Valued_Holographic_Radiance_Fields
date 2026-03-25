#include <torch/extension.h>
#include <vector>
#include <complex>

// === Forward declarations of the CUDA functions ===
void bandlimited_propagation_forward_cuda(
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    torch::Tensor& output_real,
    torch::Tensor& output_imag,
    const torch::Tensor& fx,
    const torch::Tensor& fy,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch);

void bandlimited_propagation_backward_cuda(
    const torch::Tensor& grad_output_real,
    const torch::Tensor& grad_output_imag,
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    torch::Tensor& grad_field_real,
    torch::Tensor& grad_field_imag,
    const torch::Tensor& fx,
    const torch::Tensor& fy,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch);


// === C++ interface for bandlimited_propagation_forward ===
std::vector<torch::Tensor> bandlimited_propagation_forward(
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    const torch::Tensor& fx,
    const torch::Tensor& fy,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch) 
{
    auto output_real = torch::zeros_like(field_f_real);
    auto output_imag = torch::zeros_like(field_f_imag);

    bandlimited_propagation_forward_cuda(
        field_f_real, field_f_imag,
        output_real, output_imag,
        fx, fy, wavelength, distance, aperture_size, pixel_pitch
    );

    return {output_real, output_imag};
}

// === C++ interface for bandlimited_propagation_backward ===
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
    float pixel_pitch) 
{
    auto grad_field_real = torch::zeros_like(field_f_real);
    auto grad_field_imag = torch::zeros_like(field_f_imag);

    bandlimited_propagation_backward_cuda(
        grad_output_real, grad_output_imag,
        field_f_real, field_f_imag,
        grad_field_real, grad_field_imag,
        fx, fy, wavelength, distance, aperture_size, pixel_pitch
    );

    return {grad_field_real, grad_field_imag};
}