/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}


__device__ __forceinline__ float calculate_attenuation(
    float z, float z_min, float z_max, float a_min, float a_max)
{
    // Ensure z_range is never zero to avoid division by zero
    float z_range = fmaxf(z_max - z_min, 1e-6f);
    float inv_z_range = 1.0f / z_range;
    float inv_a_range = 1.0f / fmaxf(a_max - a_min, 1e-6f);
    
    // Normalize z to [0, 1] range
    float z_normalized = (z - z_min) * inv_z_range;
    
    // Calculate attenuation exactly as in Python: 1/(1+z)^2
    float raw_atten = 1.0f / ((1.0f + z_normalized) * (1.0f + z_normalized));
    
    // Normalize attenuation to [0, 1] range
    float result = (raw_atten - a_min) * inv_a_range;
    
    // Clamp result to valid range
    return fmaxf(0.0f, fminf(1.0f, result));
}

// Calculate gradient of attenuation with respect to z
__device__ __forceinline__ float calculate_attenuation_grad(
    float z, float z_min, float z_max, float a_min, float a_max)
{
    // Ensure ranges are never zero
    float z_range = fmaxf(z_max - z_min, 1e-6f);
    float inv_z_range = 1.0f / z_range;
    float inv_a_range = 1.0f / fmaxf(a_max - a_min, 1e-6f);
    
    // Normalized z in [0, 1] range
    float z_normalized = (z - z_min) * inv_z_range;
    
    // Derivative of 1/(1+z)^2 with respect to z_normalized is -2/(1+z)^3
    // Chain rule: d(atten)/d(z) = d(atten)/d(z_normalized) * d(z_normalized)/d(z)
    // d(z_normalized)/d(z) = inv_z_range
    float power = -2.0f / powf(1.0f + z_normalized, 3.0f);
    
    // Apply chain rule and normalization
    return power * inv_z_range * inv_a_range;
}

// Helper function to compute 2D covariance inverse
__forceinline__ __device__ void compute_cov2d_inverse(
    const float* cov_2D, int g_idx,
    float& det, float& inv00, float& inv01, float& inv10, float& inv11)
{
    float cov00 = cov_2D[g_idx * 4 + 0];
    float cov01 = cov_2D[g_idx * 4 + 1];
    float cov10 = cov_2D[g_idx * 4 + 2];
    float cov11 = cov_2D[g_idx * 4 + 3];
    
    det = cov00 * cov11 - cov01 * cov10;
    if (fabsf(det) < DETERMINANT_CUTOFF)
    {
        inv00 = inv01 = inv10 = inv11 = 0.0f;
        return;
    }
    
    float inv_det = 1.0f / det;
    inv00 = cov11 * inv_det;
    inv01 = -cov01 * inv_det;
    inv10 = -cov10 * inv_det;
    inv11 = cov00 * inv_det;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
    // Make sure radius is at least 1 to ensure we have at least one tile
    int radius = max(1, max_radius);
    
    // Convert point to tile coordinates, ensuring at least one tile is covered
    rect_min = {
        min(grid.x - 1, max((uint32_t)0, (uint32_t)((p.x - radius) / BLOCK_X))),
        min(grid.y - 1, max((uint32_t)0, (uint32_t)((p.y - radius) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((uint32_t)1, (uint32_t)((p.x + radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((uint32_t)1, (uint32_t)((p.y + radius + BLOCK_Y - 1) / BLOCK_Y)))
    };
    
    // Make sure we get at least one tile
    if (rect_max.x <= rect_min.x) rect_max.x = min(grid.x, rect_min.x + 1);
    if (rect_max.y <= rect_min.y) rect_max.y = min(grid.y, rect_min.y + 1);
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif