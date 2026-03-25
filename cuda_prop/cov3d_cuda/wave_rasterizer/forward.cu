#include "forward.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace cov3d_cuda {

// CUDA kernel for computing covariance matrices
template <typename scalar_t>
__global__ void compute_cov3d_cuda_kernel(
    const scalar_t* __restrict__ quats,      // Shape: (N, 4)
    const scalar_t* __restrict__ scales,     // Shape: (N, 3)
    scalar_t* __restrict__ cov3d,            // Output: (N, 3, 3)
    const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Get quaternion and scale for this Gaussian
    const scalar_t* quat = &quats[n * 4];
    const scalar_t* scale = &scales[n * 3];

    // Normalize quaternion
    float r = quat[0]; // real part
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];
    const float norm = sqrtf(r*r + x*x + y*y + z*z);
    r /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    // Create scaling matrix
    glm::mat3 S(1.0f);
    S[0][0] = scale[0];
    S[1][1] = scale[1];
    S[2][2] = scale[2];

    // Compute rotation matrix from normalized quaternion
    glm::mat3 R(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    // Compute SR = S * R
    glm::mat3 SR = S * R;

    // Compute covariance matrix: Σ = (SR)^T * SR
    // This is equivalent to Σ = R^T * S^T * S * R since S is diagonal
    glm::mat3 Sigma = glm::transpose(SR) * SR;

    // Write output in row-major order
    scalar_t* out_cov = &cov3d[n * 9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out_cov[i * 3 + j] = Sigma[i][j];
        }
    }
}

// Add the new Jacobian computation kernel
__global__ void compute_jacobian_kernel(
    const float* __restrict__ cam_means_3D,
    float* __restrict__ jacobian,
    int N,
    float fx, float fy,
    float half_tan_fov_x, float half_tan_fov_y,
    float near_plane, float far_plane)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Extract point coordinates
    const float tx = cam_means_3D[idx * 3 + 0];
    const float ty = cam_means_3D[idx * 3 + 1];
    const float tz = cam_means_3D[idx * 3 + 2];
    
    // Apply clipping plane check
    if (tz <= near_plane || tz >= far_plane) {
        // Zero out the Jacobian for points outside clipping planes
        jacobian[idx * 6 + 0] = 0.0f;
        jacobian[idx * 6 + 1] = 0.0f;
        jacobian[idx * 6 + 2] = 0.0f;
        jacobian[idx * 6 + 3] = 0.0f;
        jacobian[idx * 6 + 4] = 0.0f;
        jacobian[idx * 6 + 5] = 0.0f;
        return;
    }
    
    const float tz2 = tz * tz;
    
    // Compute limits for clamping
    const float lim_x = 1.3f * half_tan_fov_x;
    const float lim_y = 1.3f * half_tan_fov_y;
    
    // Clamp tx/tz and ty/tz within limits
    const float tx_tz = tx / tz;
    const float ty_tz = ty / tz;
    
    const float clamped_tx_tz = fminf(fmaxf(tx_tz, -lim_x), lim_x);
    const float clamped_ty_tz = fminf(fmaxf(ty_tz, -lim_y), lim_y);
    
    // Recompute tx and ty with clamped values
    const float clamped_tx = clamped_tx_tz * tz;
    const float clamped_ty = clamped_ty_tz * tz;
    
    // Compute Jacobian entries
    // J = [[fx/tz, 0, -fx*tx/tz^2], 
    //      [0, fy/tz, -fy*ty/tz^2]]
    
    // First row [fx/tz, 0, -fx*tx/tz^2]
    jacobian[idx * 6 + 0] = fx / tz;
    jacobian[idx * 6 + 1] = 0.0f;
    jacobian[idx * 6 + 2] = -(fx * clamped_tx) / tz2;
    
    // Second row [0, fy/tz, -fy*ty/tz^2]
    jacobian[idx * 6 + 3] = 0.0f;
    jacobian[idx * 6 + 4] = fy / tz;
    jacobian[idx * 6 + 5] = -(fy * clamped_ty) / tz2;
}

// Kernel for adding a constant to diagonal elements of 2D covariance matrices
__global__ void add_diagonal_offset_kernel(
    float* __restrict__ cov2d,
    float diagonal_offset,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Add the offset to the diagonal elements
    cov2d[idx * 4 + 0] += diagonal_offset; // (0,0)
    cov2d[idx * 4 + 3] += diagonal_offset; // (1,1)
}

torch::Tensor compute_cov3d_forward(
    const torch::Tensor& quats,
    const torch::Tensor& scales)
{
    // Check inputs
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, 
        "quats must have shape (N, 4)");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, 
        "scales must have shape (N, 3)");
    TORCH_CHECK(quats.size(0) == scales.size(0), 
        "quats and scales must have the same batch size");
    TORCH_CHECK(quats.is_cuda() && scales.is_cuda(), 
        "inputs must be CUDA tensors");
    
    const int N = quats.size(0);
    
    // Allocate output covariance matrices
    torch::Tensor cov3d = torch::empty({N, 3, 3}, 
        torch::TensorOptions()
            .dtype(quats.dtype())
            .device(quats.device()));
    
    // Configure kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(quats.scalar_type(), "compute_cov3d_forward_cuda", ([&] {
        compute_cov3d_cuda_kernel<scalar_t><<<blocks, threads>>>(
            quats.contiguous().data_ptr<scalar_t>(),
            scales.contiguous().data_ptr<scalar_t>(),
            cov3d.contiguous().data_ptr<scalar_t>(),
            N
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return cov3d;
}

torch::Tensor compute_jacobian_forward(
    torch::Tensor cam_means_3D,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane)
{
    // Input validation
    TORCH_CHECK(cam_means_3D.is_cuda(), "cam_means_3D must be a CUDA tensor");
    TORCH_CHECK(cam_means_3D.dim() == 2 && cam_means_3D.size(1) == 3, 
                "cam_means_3D must have shape [N, 3]");
    TORCH_CHECK(cam_means_3D.scalar_type() == torch::kFloat, 
                "cam_means_3D must be a float tensor");
                
    const int N = cam_means_3D.size(0);
    
    // Calculate field of view parameters
    const float half_tan_fov_x = 0.5f * width / fx;
    const float half_tan_fov_y = 0.5f * height / fy;
    
    // Create output tensor for Jacobian matrices
    auto options = torch::TensorOptions()
                   .dtype(torch::kFloat)
                   .device(cam_means_3D.device());
    auto jacobian = torch::empty({N, 2, 3}, options);
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    compute_jacobian_kernel<<<blocks, threads>>>(
        cam_means_3D.data_ptr<float>(),
        jacobian.data_ptr<float>(),
        N, fx, fy, half_tan_fov_x, half_tan_fov_y,
        near_plane, far_plane);
    
    return jacobian;
}

std::vector<torch::Tensor> compute_cov2d_forward(
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& view_matrix,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane)
{
    // Check inputs
    TORCH_CHECK(cam_means_3D.dim() == 2 && cam_means_3D.size(1) == 3, 
        "cam_means_3D must have shape (N, 3)");
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, 
        "quats must have shape (N, 4)");
    TORCH_CHECK(scales.dim() == 2 && (scales.size(1) == 3 || scales.size(1) == 1), 
        "scales must have shape (N, 3) or (N, 1)");
    TORCH_CHECK(view_matrix.dim() == 2 && view_matrix.size(0) == 3 && view_matrix.size(1) == 3, 
        "view_matrix must have shape (3, 3)");
    TORCH_CHECK(cam_means_3D.size(0) == quats.size(0) && quats.size(0) == scales.size(0), 
        "cam_means_3D, quats, and scales must have the same batch size");
    TORCH_CHECK(cam_means_3D.is_cuda() && quats.is_cuda() && scales.is_cuda() && view_matrix.is_cuda(), 
        "All inputs must be CUDA tensors");
    
    const int N = cam_means_3D.size(0);
    
    // Get device from input
    auto device = cam_means_3D.device();
    auto options = torch::TensorOptions().dtype(cam_means_3D.dtype()).device(device);
    
    // Step 1: Compute the Jacobian matrix using the existing function with clipping planes
    auto J = compute_jacobian_forward(cam_means_3D, fx, fy, width, height, near_plane, far_plane);
    
    // Step 2: Compute the 3D covariance matrix using the existing function
    auto cov3D = compute_cov3d_forward(quats, scales);
    
    // Step 3: Repeat the view_matrix for each point
    auto W = view_matrix.unsqueeze(0).repeat({N, 1, 1});
    
    // Step 4: Compute the 2D covariance matrix
    // cov_2D = torch.matmul(J, W)
    // cov_2D = torch.matmul(cov_2D, cov_3D)
    // cov_2D = torch.matmul(cov_2D, torch.transpose(W, 1, 2))
    // cov_2D = torch.matmul(cov_2D, torch.transpose(J, 1, 2))
    
    auto JW = torch::matmul(J, W);
    auto JWcov3D = torch::matmul(JW, cov3D);
    auto JWcov3DWt = torch::matmul(JWcov3D, W.transpose(1, 2));
    auto cov2D = torch::matmul(JWcov3DWt, J.transpose(1, 2));
    
    // Step 5: Add the constant offset to diagonal elements
    // In PyTorch this would be:
    // cov_2D[:, 0, 0] += 0.3
    // cov_2D[:, 1, 1] += 0.3
    
    // Reshape to [N, 4] for the kernel
    auto cov2D_flat = cov2D.view({N, 4});
    
    // Configure kernel parameters
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // Add offset using CUDA kernel
    add_diagonal_offset_kernel<<<blocks, threads>>>(
        cov2D_flat.data_ptr<float>(), 
        0.3f,  // diagonal offset
        N
    );
    
    // Reshape back to [N, 2, 2]
    cov2D = cov2D_flat.view({N, 2, 2});
    
    // Return both the result and intermediate tensors for backward pass
    return {cov2D, J, cov3D, W, JW};
}

// Kernel for projecting 3D points to 2D image points
__global__ void compute_means2d_kernel(
    const float* __restrict__ cam_means_3D,
    float* __restrict__ means_2D,
    int N,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Extract coordinates
    const float x = cam_means_3D[idx * 3 + 0];
    const float y = cam_means_3D[idx * 3 + 1];
    const float z = cam_means_3D[idx * 3 + 2];
    
    // Check if point is within clipping planes
    if (z <= near_plane || z >= far_plane) {
        // Set points outside clipping planes to a large value
        means_2D[idx * 2 + 0] = 1e6f;
        means_2D[idx * 2 + 1] = 1e6f;
        return;
    }
    
    // Perspective division
    const float inv_z = 1.0f / z;
    // Note the negative sign matching the PyTorch implementation
    const float px_div = -x * inv_z;
    const float py_div = -y * inv_z;
    
    // Apply camera intrinsics
    means_2D[idx * 2 + 0] = fx * px_div + px;
    means_2D[idx * 2 + 1] = fy * py_div + py;
}

// Kernel for inverting 2x2 covariance matrices
__global__ void invert_cov2d_kernel(
    const float* __restrict__ cov_2D,
    float* __restrict__ cov_2D_inverse,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Get covariance matrix elements from flattened input
    const float a = cov_2D[idx * 4 + 0]; // (0,0)
    const float b = cov_2D[idx * 4 + 1]; // (0,1)
    const float c = cov_2D[idx * 4 + 2]; // (1,0)
    const float d = cov_2D[idx * 4 + 3]; // (1,1)
    
    // Compute determinant
    const float det = a * d - b * c;
    const float inv_det = 1.0f / det;
    
    // Set inverse matrix elements
    cov_2D_inverse[idx * 4 + 0] = d * inv_det;      // (0,0)
    cov_2D_inverse[idx * 4 + 1] = -b * inv_det;     // (0,1)
    cov_2D_inverse[idx * 4 + 2] = -c * inv_det;     // (1,0)
    cov_2D_inverse[idx * 4 + 3] = a * inv_det;      // (1,1)
}

torch::Tensor compute_means2d_forward(
    const torch::Tensor& cam_means_3D,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane)
{
    // Input validation
    TORCH_CHECK(cam_means_3D.is_cuda(), "cam_means_3D must be a CUDA tensor");
    TORCH_CHECK(cam_means_3D.dim() == 2 && cam_means_3D.size(1) == 3, 
                "cam_means_3D must have shape [N, 3]");
    
    const int N = cam_means_3D.size(0);
    
    // Create output tensor
    auto options = torch::TensorOptions()
                   .dtype(cam_means_3D.dtype())
                   .device(cam_means_3D.device());
    auto means_2D = torch::empty({N, 2}, options);
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    compute_means2d_kernel<<<blocks, threads>>>(
        cam_means_3D.data_ptr<float>(),
        means_2D.data_ptr<float>(),
        N, fx, fy, px, py,
        near_plane, far_plane);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return means_2D;
}

torch::Tensor invert_cov2d_forward(
    const torch::Tensor& cov_2D)
{
    // Input validation
    TORCH_CHECK(cov_2D.is_cuda(), "cov_2D must be a CUDA tensor");
    TORCH_CHECK(cov_2D.dim() == 3 && cov_2D.size(1) == 2 && cov_2D.size(2) == 2,
                "cov_2D must have shape [N, 2, 2]");
    
    const int N = cov_2D.size(0);
    
    // Create output tensor
    auto options = torch::TensorOptions()
                   .dtype(cov_2D.dtype())
                   .device(cov_2D.device());
    auto cov_2D_inverse = torch::empty({N, 2, 2}, options);
    
    // Reshape input and output for kernel (flatten last two dimensions)
    auto cov_2D_flat = cov_2D.reshape({N, 4});
    auto cov_2D_inverse_flat = cov_2D_inverse.reshape({N, 4});
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    invert_cov2d_kernel<<<blocks, threads>>>(
        cov_2D_flat.data_ptr<float>(),
        cov_2D_inverse_flat.data_ptr<float>(),
        N);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return cov_2D_inverse;
}

} // namespace cov3d_cuda