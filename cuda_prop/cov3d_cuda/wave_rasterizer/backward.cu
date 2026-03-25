#include "backward.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace cov3d_cuda {

// Forward declarations for functions from forward.cu
torch::Tensor compute_jacobian_forward(
    torch::Tensor cam_means_3D,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane);

torch::Tensor compute_cov3d_forward(
    const torch::Tensor& quats,
    const torch::Tensor& scales);

// Helper function to compute derivatives of rotation matrix w.r.t quaternion parameters
template <typename scalar_t>
__device__ void compute_dR_dq(
    const scalar_t r, const scalar_t x, const scalar_t y, const scalar_t z,
    glm::mat3& dR_dr, glm::mat3& dR_dx, glm::mat3& dR_dy, glm::mat3& dR_dz)  // Changed to references
{
    // Initialize matrices to zero
    dR_dr = glm::mat3(0.0f);
    dR_dx = glm::mat3(0.0f);
    dR_dy = glm::mat3(0.0f);
    dR_dz = glm::mat3(0.0f);
    
    // dR/dr
    dR_dr[0][0] = 0.f;
    dR_dr[0][1] = -2.f * z;
    dR_dr[0][2] = 2.f * y;
    dR_dr[1][0] = 2.f * z;
    dR_dr[1][1] = 0.f;
    dR_dr[1][2] = -2.f * x;
    dR_dr[2][0] = -2.f * y;
    dR_dr[2][1] = 2.f * x;
    dR_dr[2][2] = 0.f;
    
    // dR/dx
    dR_dx[0][0] = 0.f;
    dR_dx[0][1] = 2.f * y;
    dR_dx[0][2] = 2.f * z;
    dR_dx[1][0] = 2.f * y;
    dR_dx[1][1] = -4.f * x;
    dR_dx[1][2] = -2.f * r;
    dR_dx[2][0] = 2.f * z;
    dR_dx[2][1] = 2.f * r;
    dR_dx[2][2] = -4.f * x;
    
    // dR/dy
    dR_dy[0][0] = -4.f * y;
    dR_dy[0][1] = 2.f * x;
    dR_dy[0][2] = 2.f * r;
    dR_dy[1][0] = 2.f * x;
    dR_dy[1][1] = 0.f;
    dR_dy[1][2] = 2.f * z;
    dR_dy[2][0] = -2.f * r;
    dR_dy[2][1] = 2.f * z;
    dR_dy[2][2] = -4.f * y;
    
    // dR/dz
    dR_dz[0][0] = -4.f * z;
    dR_dz[0][1] = -2.f * r;
    dR_dz[0][2] = 2.f * x;
    dR_dz[1][0] = 2.f * r;
    dR_dz[1][1] = -4.f * z;
    dR_dz[1][2] = 2.f * y;
    dR_dz[2][0] = 2.f * x;
    dR_dz[2][1] = 2.f * y;
    dR_dz[2][2] = 0.f;
}

// CUDA kernel for backward pass
template <typename scalar_t>
__global__ void compute_cov3d_backward_cuda_kernel(
    const scalar_t* __restrict__ grad_cov3d,  // Shape: (N, 3, 3)
    const scalar_t* __restrict__ quats,       // Shape: (N, 4)
    const scalar_t* __restrict__ scales,      // Shape: (N, 3)
    scalar_t* __restrict__ grad_quats,        // Output: (N, 4)
    scalar_t* __restrict__ grad_scales,       // Output: (N, 3)
    const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Get quaternion and scale for this Gaussian
    const scalar_t* quat = &quats[n * 4];
    const scalar_t* scale = &scales[n * 3];
    const scalar_t* grad_cov = &grad_cov3d[n * 9];
    
    // Reshape grad_cov to 3x3 matrix
    glm::mat3 G;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            G[i][j] = grad_cov[i * 3 + j];
        }
    }

    // Normalize quaternion
    float r = quat[0];
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

    // Compute rotation matrix
    glm::mat3 R(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    // Initialize gradient accumulators
    scalar_t grad_r = 0.0f;
    scalar_t grad_x = 0.0f;
    scalar_t grad_y = 0.0f;
    scalar_t grad_z = 0.0f;
    glm::vec3 grad_s(0.0f);

    // Calculate derivatives of R with respect to quaternion components
    glm::mat3 dR_dr, dR_dx, dR_dy, dR_dz;
    compute_dR_dq(r, x, y, z, dR_dr, dR_dx, dR_dy, dR_dz);
    
    // Compute gradients for scales directly
    // dL/dS = 2 * R^T * G * R * S
    glm::mat3 temp = 2.0f * glm::transpose(R) * G * R;
    grad_s[0] = temp[0][0] * scale[0];
    grad_s[1] = temp[1][1] * scale[1];
    grad_s[2] = temp[2][2] * scale[2];
    
    // Compute gradients for quaternions through chain rule
    // We need to compute: dL/dq = dL/dΣ : dΣ/dq
    // where : is the Frobenius inner product
    
    // For each quaternion component q_i
    // dL/dq_i = ∑_jk (dL/dΣ_jk * dΣ_jk/dq_i)
    // dΣ_jk/dq_i = [S * dR/dq_i]^T * (S * R) + (S * R)^T * [S * dR/dq_i]
    
    glm::mat3 SR = S * R;
    
    // Compute for each quaternion component
    glm::mat3 S_dR_dr = S * dR_dr;
    glm::mat3 S_dR_dx = S * dR_dx;
    glm::mat3 S_dR_dy = S * dR_dy;
    glm::mat3 S_dR_dz = S * dR_dz;
    
    // For each component, compute dΣ/dq_i and accumulate gradient
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // Contribution for r
            grad_r += G[i][j] * (
                glm::transpose(S_dR_dr)[i][j] * SR[i][j] + 
                glm::transpose(SR)[i][j] * S_dR_dr[i][j]
            );
            
            // Contribution for x
            grad_x += G[i][j] * (
                glm::transpose(S_dR_dx)[i][j] * SR[i][j] + 
                glm::transpose(SR)[i][j] * S_dR_dx[i][j]
            );
            
            // Contribution for y
            grad_y += G[i][j] * (
                glm::transpose(S_dR_dy)[i][j] * SR[i][j] + 
                glm::transpose(SR)[i][j] * S_dR_dy[i][j]
            );
            
            // Contribution for z
            grad_z += G[i][j] * (
                glm::transpose(S_dR_dz)[i][j] * SR[i][j] + 
                glm::transpose(SR)[i][j] * S_dR_dz[i][j]
            );
        }
    }
    
    // Write output gradients
    grad_quats[n * 4 + 0] = grad_r;
    grad_quats[n * 4 + 1] = grad_x;
    grad_quats[n * 4 + 2] = grad_y;
    grad_quats[n * 4 + 3] = grad_z;
    
    grad_scales[n * 3 + 0] = grad_s[0];
    grad_scales[n * 3 + 1] = grad_s[1];
    grad_scales[n * 3 + 2] = grad_s[2];
}

// Kernel for computing gradients with respect to cam_means_3D
__global__ void compute_jacobian_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ cam_means_3D,
    float* __restrict__ grad_cam_means_3D,
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
    
    // Apply clipping plane check - zero out gradients outside clipping planes
    if (tz <= near_plane || tz >= far_plane) {
        grad_cam_means_3D[idx * 3 + 0] = 0.0f;
        grad_cam_means_3D[idx * 3 + 1] = 0.0f;
        grad_cam_means_3D[idx * 3 + 2] = 0.0f;
        return;
    }
    
    const float tz2 = tz * tz;
    const float tz3 = tz2 * tz;
    
    // Compute limits for clamping
    const float lim_x = 1.3f * half_tan_fov_x;
    const float lim_y = 1.3f * half_tan_fov_y;
    
    // Calculate tx/tz and ty/tz
    const float tx_tz = tx / tz;
    const float ty_tz = ty / tz;
    
    // Check if values are clamped and set flag for gradient computation
    const float x_grad_mul = (tx_tz < -lim_x || tx_tz > lim_x) ? 0.0f : 1.0f;
    const float y_grad_mul = (ty_tz < -lim_y || ty_tz > lim_y) ? 0.0f : 1.0f;
    
    // Get gradients from grad_output (shape [N, 2, 3])
    // J is shaped [N, 2, 3], so grad_output has the same shape
    const float dL_dJ00 = grad_output[idx * 6 + 0]; // dL/dJ[0,0]
    // Note: dL_dJ01 not used
    const float dL_dJ02 = grad_output[idx * 6 + 2]; // dL/dJ[0,2]
    // Note: dL_dJ10 not used
    const float dL_dJ11 = grad_output[idx * 6 + 4]; // dL/dJ[1,1]
    const float dL_dJ12 = grad_output[idx * 6 + 5]; // dL/dJ[1,2]
    
    // Compute gradients for tx, ty, tz
    // Gradients through J[0,0] = fx/tz
    const float dJ00_dtz = -fx / tz2;
    
    // Gradients through J[0,2] = -fx*tx/tz^2
    const float dJ02_dtx = x_grad_mul * (-fx / tz2);
    const float dJ02_dtz = x_grad_mul * (2.0f * fx * tx / tz3);
    
    // Gradients through J[1,1] = fy/tz
    const float dJ11_dtz = -fy / tz2;
    
    // Gradients through J[1,2] = -fy*ty/tz^2
    const float dJ12_dty = y_grad_mul * (-fy / tz2);
    const float dJ12_dtz = y_grad_mul * (2.0f * fy * ty / tz3);
    
    // Compute final gradients by chain rule
    grad_cam_means_3D[idx * 3 + 0] = dL_dJ02 * dJ02_dtx;
    grad_cam_means_3D[idx * 3 + 1] = dL_dJ12 * dJ12_dty;
    grad_cam_means_3D[idx * 3 + 2] = dL_dJ00 * dJ00_dtz + 
                                     dL_dJ02 * dJ02_dtz + 
                                     dL_dJ11 * dJ11_dtz + 
                                     dL_dJ12 * dJ12_dtz;
}

std::vector<torch::Tensor> compute_cov3d_backward(
    const torch::Tensor& grad_cov3d,
    const torch::Tensor& quats,
    const torch::Tensor& scales)
{
    // Check inputs
    TORCH_CHECK(grad_cov3d.dim() == 3 && grad_cov3d.size(1) == 3 && grad_cov3d.size(2) == 3,
        "grad_cov3d must have shape (N, 3, 3)");
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4,
        "quats must have shape (N, 4)");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3,
        "scales must have shape (N, 3)");
    TORCH_CHECK(grad_cov3d.size(0) == quats.size(0) && quats.size(0) == scales.size(0),
        "All inputs must have the same batch size");
    TORCH_CHECK(grad_cov3d.is_cuda() && quats.is_cuda() && scales.is_cuda(),
        "All inputs must be CUDA tensors");
    
    const int N = quats.size(0);
    
    // Allocate output gradients
    torch::Tensor grad_quats = torch::zeros_like(quats);
    torch::Tensor grad_scales = torch::zeros_like(scales);
    
    // Configure kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(quats.scalar_type(), "compute_cov3d_backward_cuda", ([&] {
        compute_cov3d_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_cov3d.contiguous().view({N, 9}).data_ptr<scalar_t>(),
            quats.contiguous().data_ptr<scalar_t>(),
            scales.contiguous().data_ptr<scalar_t>(),
            grad_quats.data_ptr<scalar_t>(),
            grad_scales.data_ptr<scalar_t>(),
            N
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return {grad_quats, grad_scales};
}

torch::Tensor compute_jacobian_backward(
    torch::Tensor grad_output,
    torch::Tensor cam_means_3D,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane)
{
    // Input validation
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(cam_means_3D.is_cuda(), "cam_means_3D must be a CUDA tensor");
    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(1) == 2 && grad_output.size(2) == 3,
                "grad_output must have shape [N, 2, 3]");
    TORCH_CHECK(cam_means_3D.dim() == 2 && cam_means_3D.size(1) == 3,
                "cam_means_3D must have shape [N, 3]");
    
    const int N = cam_means_3D.size(0);
    
    // Calculate field of view parameters
    const float half_tan_fov_x = 0.5f * width / fx;
    const float half_tan_fov_y = 0.5f * height / fy;
    
    // Create output tensor for gradients
    auto options = torch::TensorOptions()
                   .dtype(torch::kFloat)
                   .device(cam_means_3D.device());
    auto grad_cam_means_3D = torch::zeros({N, 3}, options);
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    compute_jacobian_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        cam_means_3D.data_ptr<float>(),
        grad_cam_means_3D.data_ptr<float>(),
        N, fx, fy, half_tan_fov_x, half_tan_fov_y,
        near_plane, far_plane);
    
    return grad_cam_means_3D;
}

std::vector<torch::Tensor> compute_cov2d_backward(
    const torch::Tensor& grad_cov2d,
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& view_matrix,
    const torch::Tensor& J,         
    const torch::Tensor& cov3D,     
    const torch::Tensor& W,         
    const torch::Tensor& JW,       
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane)
{
    // Check inputs
    TORCH_CHECK(grad_cov2d.dim() == 3 && grad_cov2d.size(1) == 2 && grad_cov2d.size(2) == 2,
        "grad_cov2d must have shape (N, 2, 2)");
    TORCH_CHECK(cam_means_3D.dim() == 2 && cam_means_3D.size(1) == 3, 
        "cam_means_3D must have shape (N, 3)");
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, 
        "quats must have shape (N, 4)");
    TORCH_CHECK(scales.dim() == 2 && (scales.size(1) == 3 || scales.size(1) == 1), 
        "scales must have shape (N, 3) or (N, 1)");
    TORCH_CHECK(view_matrix.dim() == 2 && view_matrix.size(0) == 3 && view_matrix.size(1) == 3, 
        "view_matrix must have shape (3, 3)");
    // Check new intermediate tensors
    TORCH_CHECK(J.dim() == 3 && J.size(1) == 2 && J.size(2) == 3,
        "J must have shape (N, 2, 3)");
    TORCH_CHECK(cov3D.dim() == 3 && cov3D.size(1) == 3 && cov3D.size(2) == 3,
        "cov3D must have shape (N, 3, 3)");
    TORCH_CHECK(W.dim() == 3 && W.size(1) == 3 && W.size(2) == 3,
        "W must have shape (N, 3, 3)");
    TORCH_CHECK(JW.dim() == 3 && JW.size(1) == 2 && JW.size(2) == 3,
        "JW must have shape (N, 2, 3)");
    
    TORCH_CHECK(cam_means_3D.size(0) == quats.size(0) && 
                quats.size(0) == scales.size(0) && 
                grad_cov2d.size(0) == cam_means_3D.size(0), 
        "All inputs must have the same batch size in first dimension");
    TORCH_CHECK(grad_cov2d.is_cuda() && cam_means_3D.is_cuda() && 
                quats.is_cuda() && scales.is_cuda() && view_matrix.is_cuda() &&
                J.is_cuda() && cov3D.is_cuda() && W.is_cuda() && JW.is_cuda(), 
        "All inputs must be CUDA tensors");
    
    const int N = cam_means_3D.size(0);
    
    // Get device from input
    auto device = cam_means_3D.device();
    
    // Step 0: Make sure diagonal offset gradients are accounted for (no changes needed)
    
    // Step 1: Use the provided intermediate tensors instead of recomputing
    
    // Step 2: Backpropagate through the matrix multiplications
    // From forward: cov2D = J * W * cov3D * W^T * J^T
    
    // For clarity, we'll name intermediate results:
    // A = J * W (this is JW that we already have)
    // B = A * cov3D
    // C = B * W^T
    // cov2D = C * J^T
    
    // Compute dL/dC from dL/dcov2D
    auto dL_dC = torch::matmul(grad_cov2d, J);
    
    // Compute dL/dB from dL/dC
    auto dL_dB = torch::matmul(dL_dC, W);
    
    // Compute dL/dcov3D from dL/dB
    // Use the precomputed JW instead of recomputing J*W
    auto dL_dcov3D = torch::matmul(JW.transpose(1, 2), torch::matmul(grad_cov2d, JW));
    
    // Compute dL/dA from dL/dB
    auto dL_dA = torch::matmul(dL_dB, cov3D.transpose(1, 2));
    
    // Compute dL/dJ from dL/dA
    auto dL_dJ = torch::matmul(dL_dA, W.transpose(1, 2));
    
    // Step 3: Compute gradients using existing functions
    auto grad_cam_means_3D = compute_jacobian_backward(dL_dJ, cam_means_3D, fx, fy, width, height, near_plane, far_plane);
    auto grad_quats_scales = compute_cov3d_backward(dL_dcov3D, quats, scales);
    
    auto grad_quats = grad_quats_scales[0];
    auto grad_scales = grad_quats_scales[1];
    
    // Return gradients for all inputs
    return {grad_cam_means_3D, grad_quats, grad_scales};
}

// Kernel for computing gradients of 2D means projection
__global__ void compute_means2d_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ cam_means_3D,
    float* __restrict__ grad_cam_means_3D,
    int N,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Extract z coordinate and check clipping planes
    const float z = cam_means_3D[idx * 3 + 2];
    
    // Zero out gradients for points outside clipping planes
    if (z <= near_plane || z >= far_plane) {
        grad_cam_means_3D[idx * 3 + 0] = 0.0f;
        grad_cam_means_3D[idx * 3 + 1] = 0.0f;
        grad_cam_means_3D[idx * 3 + 2] = 0.0f;
        return;
    }
    
    const float z2 = z * z;
    
    // Get gradients from output
    const float grad_x = grad_output[idx * 2 + 0]; // dL/dx'
    const float grad_y = grad_output[idx * 2 + 1]; // dL/dy'
    
    // Extract original coordinates
    const float x = cam_means_3D[idx * 3 + 0];
    const float y = cam_means_3D[idx * 3 + 1];
    
    // Compute gradients for input coordinates
    // For x: d(fx * (-x/z) + px)/dx = -fx/z
    grad_cam_means_3D[idx * 3 + 0] = -fx * grad_x / z;
    
    // For y: d(fy * (-y/z) + py)/dy = -fy/z
    grad_cam_means_3D[idx * 3 + 1] = -fy * grad_y / z;
    
    // For z:
    // d(fx * (-x/z) + px)/dz = fx*x/z^2
    // d(fy * (-y/z) + py)/dz = fy*y/z^2
    grad_cam_means_3D[idx * 3 + 2] = (fx * x * grad_x / z2) + (fy * y * grad_y / z2);
}

// Kernel for computing gradients of covariance matrix inversion
__global__ void invert_cov2d_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ cov_2D,
    float* __restrict__ grad_cov_2D,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Load original matrix
    const float a = cov_2D[idx * 4 + 0]; // (0,0)
    const float b = cov_2D[idx * 4 + 1]; // (0,1)
    const float c = cov_2D[idx * 4 + 2]; // (1,0)
    const float d = cov_2D[idx * 4 + 3]; // (1,1)
    
    // Compute determinant
    const float det = a * d - b * c;
    const float inv_det = 1.0f / det;
    
    // Compute inverse (same as forward pass)
    const float inv_a = d * inv_det;      // (0,0)
    const float inv_b = -b * inv_det;     // (0,1)
    const float inv_c = -c * inv_det;     // (1,0)
    const float inv_d = a * inv_det;      // (1,1)
    
    // Load gradient of loss with respect to inverse
    const float grad_inv_a = grad_output[idx * 4 + 0]; // dL/dA^(-1)_00
    const float grad_inv_b = grad_output[idx * 4 + 1]; // dL/dA^(-1)_01
    const float grad_inv_c = grad_output[idx * 4 + 2]; // dL/dA^(-1)_10
    const float grad_inv_d = grad_output[idx * 4 + 3]; // dL/dA^(-1)_11
    
    // Gradient of A^(-1) with respect to A is -A^(-1) * dL/dA^(-1) * A^(-1)
    // First, compute inner product: dL/dA^(-1) * A^(-1)
    const float temp_a = grad_inv_a * inv_a + grad_inv_b * inv_c;
    const float temp_b = grad_inv_a * inv_b + grad_inv_b * inv_d;
    const float temp_c = grad_inv_c * inv_a + grad_inv_d * inv_c;
    const float temp_d = grad_inv_c * inv_b + grad_inv_d * inv_d;
    
    // Then multiply by -A^(-1) to get final gradients
    grad_cov_2D[idx * 4 + 0] = -(inv_a * temp_a + inv_b * temp_c);
    grad_cov_2D[idx * 4 + 1] = -(inv_a * temp_b + inv_b * temp_d);
    grad_cov_2D[idx * 4 + 2] = -(inv_c * temp_a + inv_d * temp_c);
    grad_cov_2D[idx * 4 + 3] = -(inv_c * temp_b + inv_d * temp_d);
}

torch::Tensor compute_means2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& cam_means_3D,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane)
{
    // Input validation
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(cam_means_3D.is_cuda(), "cam_means_3D must be a CUDA tensor");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(1) == 2,
                "grad_output must have shape [N, 2]");
    TORCH_CHECK(cam_means_3D.dim() == 2 && cam_means_3D.size(1) == 3,
                "cam_means_3D must have shape [N, 3]");
    TORCH_CHECK(grad_output.size(0) == cam_means_3D.size(0),
                "grad_output and cam_means_3D must have the same batch size");
    
    const int N = grad_output.size(0);
    
    // Create output tensor
    auto grad_cam_means_3D = torch::zeros_like(cam_means_3D);
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    compute_means2d_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        cam_means_3D.data_ptr<float>(),
        grad_cam_means_3D.data_ptr<float>(),
        N, fx, fy, px, py,
        near_plane, far_plane);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return grad_cam_means_3D;
}

torch::Tensor invert_cov2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& cov_2D)
{
    // Input validation
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(cov_2D.is_cuda(), "cov_2D must be a CUDA tensor");
    TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(1) == 2 && grad_output.size(2) == 2,
                "grad_output must have shape [N, 2, 2]");
    TORCH_CHECK(cov_2D.dim() == 3 && cov_2D.size(1) == 2 && cov_2D.size(2) == 2,
                "cov_2D must have shape [N, 2, 2]");
    TORCH_CHECK(grad_output.size(0) == cov_2D.size(0),
                "grad_output and cov_2D must have the same batch size");
    
    const int N = grad_output.size(0);
    
    // Create output tensor
    auto grad_cov_2D = torch::zeros_like(cov_2D);
    
    // Reshape tensors for kernel
    auto grad_output_flat = grad_output.reshape({N, 4});
    auto cov_2D_flat = cov_2D.reshape({N, 4});
    auto grad_cov_2D_flat = grad_cov_2D.reshape({N, 4});
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    invert_cov2d_backward_kernel<<<blocks, threads>>>(
        grad_output_flat.data_ptr<float>(),
        cov_2D_flat.data_ptr<float>(),
        grad_cov_2D_flat.data_ptr<float>(),
        N);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    return grad_cov_2D;
}

} // namespace cov3d_cuda