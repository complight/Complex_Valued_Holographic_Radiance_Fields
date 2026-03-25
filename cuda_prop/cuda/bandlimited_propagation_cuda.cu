#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuComplex.h>

// Helper device function to create a circular binary mask
template <typename scalar_t>
__device__ __forceinline__ scalar_t circular_mask(scalar_t x, scalar_t y, scalar_t radius) {
    scalar_t distance_squared = x*x + y*y;
    return (distance_squared < radius*radius) ? 1.0f : 0.0f;
}

// Type-specific sincos wrappers
template <typename scalar_t>
__device__ __forceinline__ void sincos_wrapper(scalar_t phase, scalar_t* sin_val, scalar_t* cos_val);

// Specialization for float
template <>
__device__ __forceinline__ void sincos_wrapper<float>(float phase, float* sin_val, float* cos_val) {
    sincosf(phase, sin_val, cos_val);
}

// Specialization for double
template <>
__device__ __forceinline__ void sincos_wrapper<double>(double phase, double* sin_val, double* cos_val) {
    sincos(phase, sin_val, cos_val);
}

template <typename scalar_t>
__global__ void bandlimited_propagation_forward_kernel(
    const scalar_t* __restrict__ field_f_real,
    const scalar_t* __restrict__ field_f_imag,
    scalar_t* __restrict__ output_real,
    scalar_t* __restrict__ output_imag,
    const scalar_t* __restrict__ fx_grid,
    const scalar_t* __restrict__ fy_grid,
    scalar_t wavelength,
    scalar_t distance,
    int nx,
    int ny,
    scalar_t aperture_size,
    scalar_t pixel_pitch,
    scalar_t fx_max,
    scalar_t fy_max,
    scalar_t offset_x,
    scalar_t offset_y,
    scalar_t k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < nx && idy < ny) {
        // Use column-major indexing for better memory coalescing
        const int index = idy * nx + idx;
        
        const scalar_t fx_val = fx_grid[index];
        const scalar_t fy_val = fy_grid[index];
        
        bool within_bandlimit = (fabsf(fx_val) < fx_max) && (fabsf(fy_val) < fy_max);
        scalar_t kz_squared = k*k - (2*M_PI*fx_val)*(2*M_PI*fx_val) - (2*M_PI*fy_val)*(2*M_PI*fy_val);
        
        cuComplex input = make_cuComplex(field_f_real[index], field_f_imag[index]);
        cuComplex transfer = make_cuComplex(0.0f, 0.0f);
        
        if (kz_squared > 0.0f && within_bandlimit) {
            scalar_t kz = sqrtf(kz_squared);
            scalar_t phase = distance * kz;
            
            scalar_t sin_val, cos_val;
            sincos_wrapper(phase, &sin_val, &cos_val);
            transfer = make_cuComplex(cos_val, sin_val);
        }
        
        scalar_t aperture_value = 1.0f;
        if (aperture_size > 0.0f) {
            // Adjusted coordinates to match Python's grid
            scalar_t x = (scalar_t)idx - offset_x;
            scalar_t y = (scalar_t)idy - offset_y;
            aperture_value = circular_mask(x, y, aperture_size);
        }
        
        cuComplex result = cuCmulf(input, transfer);
        output_real[index] = cuCrealf(result) * aperture_value;
        output_imag[index] = cuCimagf(result) * aperture_value;
    }
}

template <typename scalar_t>
__global__ void bandlimited_propagation_backward_kernel(
    const scalar_t* __restrict__ grad_output_real,
    const scalar_t* __restrict__ grad_output_imag,
    const scalar_t* __restrict__ field_f_real,
    const scalar_t* __restrict__ field_f_imag,
    scalar_t* __restrict__ grad_field_real,
    scalar_t* __restrict__ grad_field_imag,
    const scalar_t* __restrict__ fx_grid,
    const scalar_t* __restrict__ fy_grid,
    scalar_t wavelength,
    scalar_t distance,
    int nx,
    int ny,
    scalar_t aperture_size,
    scalar_t pixel_pitch,
    scalar_t fx_max,
    scalar_t fy_max,
    scalar_t offset_x,
    scalar_t offset_y,
    scalar_t k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < nx && idy < ny) {
        // Use column-major indexing for better memory coalescing
        const int index = idy * nx + idx;
        
        const scalar_t fx_val = fx_grid[index];
        const scalar_t fy_val = fy_grid[index];
        
        bool within_bandlimit = (fabsf(fx_val) < fx_max) && (fabsf(fy_val) < fy_max);
        scalar_t kz_squared = k*k - (2*M_PI*fx_val)*(2*M_PI*fx_val) - (2*M_PI*fy_val)*(2*M_PI*fy_val);
        
        cuComplex grad = make_cuComplex(grad_output_real[index], grad_output_imag[index]);
        cuComplex transfer_conj = make_cuComplex(0.0f, 0.0f);
        
        if (kz_squared > 0.0f && within_bandlimit) {
            scalar_t kz = sqrtf(kz_squared);
            scalar_t phase = -distance * kz;
            
            scalar_t sin_val, cos_val;
            sincos_wrapper(phase, &sin_val, &cos_val);
            transfer_conj = make_cuComplex(cos_val, sin_val);
        }
        
        scalar_t aperture_value = 1.0f;
        if (aperture_size > 0.0f) {
            // Adjusted coordinates to match Python's grid
            scalar_t x = (scalar_t)idx - offset_x;
            scalar_t y = (scalar_t)idy - offset_y;
            aperture_value = circular_mask(x, y, aperture_size);
        }
        
        cuComplex result = cuCmulf(grad, transfer_conj);
        grad_field_real[index] = cuCrealf(result) * aperture_value;
        grad_field_imag[index] = cuCimagf(result) * aperture_value;
    }
}

void bandlimited_propagation_forward_cuda(
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    torch::Tensor& output_real,
    torch::Tensor& output_imag,
    const torch::Tensor& fx_grid,
    const torch::Tensor& fy_grid,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch) {
    
    const int nx = field_f_real.size(0);
    const int ny = field_f_real.size(1);
    
    // Pre-compute constants used by all threads
    const float k = 2.0f * M_PI / wavelength;
    const float x_size = pixel_pitch * nx;
    const float y_size = pixel_pitch * ny;
    
    const float fx_max = 1.0f / sqrtf((2.0f * distance * (1.0f / x_size)) * (2.0f * distance * (1.0f / x_size)) + 1.0f) / wavelength;
    const float fy_max = 1.0f / sqrtf((2.0f * distance * (1.0f / y_size)) * (2.0f * distance * (1.0f / y_size)) + 1.0f) / wavelength;
    
    const float offset_x = nx / 2.0f - 0.5f;
    const float offset_y = ny / 2.0f - 0.5f;
    
    // Optimize thread configuration based on problem size
    int block_size_x = 16;
    int block_size_y = 16;
    
    // For very large images, use more threads per block
    if (nx > 1024 && ny > 1024) {
        block_size_x = 32;
        block_size_y = 8;
    }
    
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(field_f_real.scalar_type(), "bandlimited_propagation_forward", ([&] {
        bandlimited_propagation_forward_kernel<scalar_t><<<blocks, threads>>>(
            field_f_real.data_ptr<scalar_t>(),
            field_f_imag.data_ptr<scalar_t>(),
            output_real.data_ptr<scalar_t>(),
            output_imag.data_ptr<scalar_t>(),
            fx_grid.data_ptr<scalar_t>(),
            fy_grid.data_ptr<scalar_t>(),
            static_cast<scalar_t>(wavelength),
            static_cast<scalar_t>(distance),
            nx,
            ny,
            static_cast<scalar_t>(aperture_size),
            static_cast<scalar_t>(pixel_pitch),
            static_cast<scalar_t>(fx_max),
            static_cast<scalar_t>(fy_max),
            static_cast<scalar_t>(offset_x),
            static_cast<scalar_t>(offset_y),
            static_cast<scalar_t>(k));
    }));
}

void bandlimited_propagation_backward_cuda(
    const torch::Tensor& grad_output_real,
    const torch::Tensor& grad_output_imag,
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    torch::Tensor& grad_field_real,
    torch::Tensor& grad_field_imag,
    const torch::Tensor& fx_grid,
    const torch::Tensor& fy_grid,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch) {
    
    const int nx = grad_output_real.size(0);
    const int ny = grad_output_real.size(1);
    
    // Pre-compute constants used by all threads
    const float k = 2.0f * M_PI / wavelength;
    const float x_size = pixel_pitch * nx;
    const float y_size = pixel_pitch * ny;
    
    const float fx_max = 1.0f / sqrtf((2.0f * distance * (1.0f / x_size)) * (2.0f * distance * (1.0f / x_size)) + 1.0f) / wavelength;
    const float fy_max = 1.0f / sqrtf((2.0f * distance * (1.0f / y_size)) * (2.0f * distance * (1.0f / y_size)) + 1.0f) / wavelength;
    
    const float offset_x = nx / 2.0f - 0.5f;
    const float offset_y = ny / 2.0f - 0.5f;
    
    // Optimize thread configuration based on problem size
    int block_size_x = 16;
    int block_size_y = 16;
    
    // For very large images, use more threads per block
    if (nx > 1024 && ny > 1024) {
        block_size_x = 32;
        block_size_y = 8;
    }
    
    dim3 threads(block_size_x, block_size_y);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(grad_output_real.scalar_type(), "bandlimited_propagation_backward", ([&] {
        bandlimited_propagation_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output_real.data_ptr<scalar_t>(),
            grad_output_imag.data_ptr<scalar_t>(),
            field_f_real.data_ptr<scalar_t>(),
            field_f_imag.data_ptr<scalar_t>(),
            grad_field_real.data_ptr<scalar_t>(),
            grad_field_imag.data_ptr<scalar_t>(),
            fx_grid.data_ptr<scalar_t>(),
            fy_grid.data_ptr<scalar_t>(),
            static_cast<scalar_t>(wavelength),
            static_cast<scalar_t>(distance),
            nx,
            ny,
            static_cast<scalar_t>(aperture_size),
            static_cast<scalar_t>(pixel_pitch),
            static_cast<scalar_t>(fx_max),
            static_cast<scalar_t>(fy_max),
            static_cast<scalar_t>(offset_x),
            static_cast<scalar_t>(offset_y),
            static_cast<scalar_t>(k));
    }));
}