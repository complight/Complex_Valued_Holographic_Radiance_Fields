#include <torch/extension.h>
#include "conv.h"
#include "wave_rasterizer/forward.h"
#include "wave_rasterizer/backward.h"
#include "wave_rasterizer/forward_splat.h"
#include "wave_rasterizer/backward_splat.h"
#include "wave_rasterizer/rasterizer_splat.h"

// Make sure to define the namespace for pybind11
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original forward functions
    m.def("compute_cov3d_forward", &cov3d_cuda::compute_cov3d_forward, "3D Covariance Computation (CUDA)");
    m.def("compute_jacobian_forward", &cov3d_cuda::compute_jacobian_forward, "Jacobian Computation (CUDA)");
    m.def("compute_cov2d_forward", &cov3d_cuda::compute_cov2d_forward, "2D Covariance Computation (CUDA)");
    m.def("compute_means2d_forward", &cov3d_cuda::compute_means2d_forward, "2D Means Projection (CUDA)");
    m.def("invert_cov2d_forward", &cov3d_cuda::invert_cov2d_forward, "2D Covariance Matrix Inversion (CUDA)");
    
    // Original backward functions
    m.def("compute_cov3d_backward", &cov3d_cuda::compute_cov3d_backward, "3D Covariance Gradient Computation (CUDA)");
    m.def("compute_jacobian_backward", &cov3d_cuda::compute_jacobian_backward, "Jacobian Gradient Computation (CUDA)");
    m.def("compute_cov2d_backward", &cov3d_cuda::compute_cov2d_backward, "2D Covariance Gradient Computation (CUDA)");
    m.def("compute_means2d_backward", &cov3d_cuda::compute_means2d_backward, "2D Means Projection Gradient (CUDA)");
    m.def("invert_cov2d_backward", &cov3d_cuda::invert_cov2d_backward, "2D Covariance Matrix Inversion Gradient (CUDA)");
    m.def("fusedssim", &fusedssim);
    m.def("fusedssim_backward", &fusedssim_backward);
    m.def("adamUpdate", &adamUpdate);
    
    // Direct binding of the Rasterizer class
    py::class_<CudaRasterizer::Rasterizer>(m, "Rasterizer")
        .def(py::init<>())
        .def("forward", &CudaRasterizer::Rasterizer::forward,
             "Forward pass for tile-based Gaussian splatting")
        .def("backward", &CudaRasterizer::Rasterizer::backward,
             "Backward pass for tile-based Gaussian splatting");
}