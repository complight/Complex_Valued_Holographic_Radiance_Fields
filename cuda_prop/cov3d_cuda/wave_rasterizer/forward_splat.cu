/*
* Forward pass for wave-based 3D Gaussian Splatting
*/

#include "forward_splat.h"
#include "auxiliary.h"
#include "config.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace cov3d_cuda {

__global__ void preprocessGaussiansKernel(
    int N,
    const float2* __restrict__ means_2D,
    const float4* __restrict__ cov_2D,
    const float* __restrict__ z_vals,
    int* __restrict__ radii,
    uint32_t* __restrict__ tiles_touched,
    dim3 grid,
    bool prefiltered,
    bool antialiasing,
    float near_plane,
    float far_plane)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    
    // Initialize radius and tiles_touched with a clear 0
    radii[idx] = 0;
    tiles_touched[idx] = 0;
    
    // Get depth value and check clipping planes
    const float z = z_vals[idx];
    if (z <= near_plane || z >= far_plane)
        return;
    
    // Get 2D position and covariance of the Gaussian
    const float2 pos = means_2D[idx];
    const float4 cov = cov_2D[idx];
    
    // Extract original covariance components
    float cov_x = cov.x;  // cov_xx
    float cov_y = cov.y;  // cov_xy
    float cov_z = cov.z;  // cov_yx
    float cov_w = cov.w;  // cov_yy
    
    // Compute determinant before adding anti-aliasing
    float det_cov = cov_x * cov_w - cov_y * cov_z;
    
    // Check for numerical stability
    if (fabsf(det_cov) < 1e-10f)
        return;
        
    if (antialiasing) {
        const float reg_value = 0.3f;
        cov_x += reg_value;
        cov_w += reg_value;
    
        // Compute the new determinant
        det_cov = cov_x * cov_w - cov_y * cov_z;
    }
    
    // Compute the final determinant for radius calculation
    const float det = det_cov;
    
    if (fabsf(det) < 1e-8f)
        return;
    
    // IMPROVED: Compute extent in screen space using eigenvalues of covariance matrix
    // We use a smaller minimum value for the discriminant to better preserve elliptical shapes
    const float mid = 0.5f * (cov_x + cov_w);
    const float disc = fmaxf(0.1f, mid * mid - det); 
    const float lambda1 = mid + sqrtf(disc);
    const float lambda2 = mid - sqrtf(disc);
    
    // Calculate separate radii for X and Y to properly handle elliptical Gaussians
    const float radius_x = ceilf(3.0f * sqrtf(fmaxf(lambda1, lambda2)));
    const float radius_y = radius_x; // For backward compatibility with getRect
    
    // Store radius (using max for compatibility with current getRect implementation)
    // Ideally, we'd update getRect to handle separate x/y radii
    const float my_radius = fmaxf(radius_x, radius_y);
    radii[idx] = my_radius;
    
    // Calculate affected tiles
    uint2 rect_min, rect_max;
    getRect(pos, my_radius, rect_min, rect_max, grid);
    
    // Count affected tiles
    const uint32_t count = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    tiles_touched[idx] = count;
}

// Main tile-based rendering kernel (front-to-back processing)
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderTileKernel(
    const uint2* __restrict__ tile_ranges,
    const uint32_t* __restrict__ gaussian_indices_sorted,
    const float2* __restrict__ means_2D,
    const float4* __restrict__ cov_2D,
    const float* __restrict__ z_vals,
    const float* __restrict__ colours,
    const float* __restrict__ phase,
    const float* __restrict__ opacities,
    const float* __restrict__ plane_probs,
    float* __restrict__ output_real,
    float* __restrict__ output_imag,
    float* __restrict__ final_Ts,
    uint32_t* __restrict__ n_contrib,
    int W, int H, int N,
    int num_planes, int channels,
    float near_plane, float far_plane)
{
    // Identify current tile
    const uint32_t tile_x = blockIdx.x;
    const uint32_t tile_y = blockIdx.y;
    const uint32_t horizontal_tiles = (W + BLOCK_X - 1) / BLOCK_X;
    const uint32_t tile_id = tile_y * horizontal_tiles + tile_x;
    
    // Calculate pixel coordinates for this thread
    const uint32_t pix_x = tile_x * BLOCK_X + threadIdx.x;
    const uint32_t pix_y = tile_y * BLOCK_Y + threadIdx.y;
    const bool inside = pix_x < W && pix_y < H;
    
    // Get range of Gaussians for this tile
    const uint2 range = tile_ranges[tile_id];
    const int toDo = range.y - range.x;
    
    // Skip empty tiles
    if (toDo <= 0) return;
    
    // Convert to float coordinates for distance calculations
    const float2 pixf = { (float)pix_x, (float)pix_y };
    
    // Process each pixel in tile if it's inside the image
    if (inside) {
        const uint32_t pix_id = pix_y * W + pix_x;
        
        // Process each plane
        for (int p = 0; p < num_planes; p++) {
            const int plane_offset = p * H * W;
            
            // Initialize transmittance
            float T = 1.0f;
            
            // Track position and last contributing position instead of count
            uint32_t position = 0;
            uint32_t last_position = 0;
            
            // Initialize output for each channel
            float accum_real[CHANNELS] = {0.0f};
            float accum_imag[CHANNELS] = {0.0f};
            
            // Compute number of rounds needed to process all Gaussians
            const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
            
            // Flag to track if we're done with rendering (early termination)
            bool done = false;
            
            // Shared memory for batch processing
            __shared__ uint32_t collected_ids[BLOCK_SIZE];
            __shared__ float2 collected_means[BLOCK_SIZE];
            __shared__ float4 collected_covs[BLOCK_SIZE];
            __shared__ float collected_opacities[BLOCK_SIZE];
            __shared__ float collected_z_vals[BLOCK_SIZE];
            
            for (int i = 0; i < rounds; i++) {
                __syncthreads();
                
                // Check if we can terminate early (if all threads are done)
                int num_done = __syncthreads_count(done);
                if (num_done == BLOCK_X * BLOCK_Y)
                    break;
                
                // Collaboratively load a batch of data into shared memory
                const int progress = i * BLOCK_SIZE + threadIdx.x + threadIdx.y * blockDim.x;
                if (range.x + progress < range.y) {
                    const uint32_t g_idx = gaussian_indices_sorted[range.x + progress];
                    collected_ids[threadIdx.x + threadIdx.y * blockDim.x] = g_idx;
                    collected_means[threadIdx.x + threadIdx.y * blockDim.x] = means_2D[g_idx];
                    collected_covs[threadIdx.x + threadIdx.y * blockDim.x] = cov_2D[g_idx];
                    collected_opacities[threadIdx.x + threadIdx.y * blockDim.x] = opacities[g_idx];
                    collected_z_vals[threadIdx.x + threadIdx.y * blockDim.x] = z_vals[g_idx];
                }
                __syncthreads();
                
                const int batch_size = min(BLOCK_SIZE, toDo - i * BLOCK_SIZE);
                for (int j = 0; j < batch_size && !done; j++) {
                    // Increment position counter for EVERY Gaussian processed
                    position++;
                    
                    // Get Gaussian ID
                    const uint32_t g_idx = collected_ids[j];
                    
                    // Check clipping planes - skip if outside
                    const float z = collected_z_vals[j];
                    if (z <= near_plane || z >= far_plane)
                        continue;
                    
                    const float plane_weight = plane_probs[g_idx * num_planes + p];
                    if (plane_weight < PLANE_PROBABILITY_CUTOFF)
                        continue;
                    
                    // Calculate distance to Gaussian center
                    const float2 xy = collected_means[j];
                    const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                    
                    // Get covariance and compute power
                    const float4 cov = collected_covs[j];
                    const float det = cov.x * cov.w - cov.y * cov.z;
                    
                    const float det_inv = 1.0f / det;
                    const float inv00 = cov.w * det_inv;
                    const float inv01 = -cov.y * det_inv;
                    const float inv10 = -cov.z * det_inv;
                    const float inv11 = cov.x * det_inv;
                    
                    // Compute quadratic form for Gaussian exponent (improved stability)
                    const float power = -0.5f * (d.x * (inv00 * d.x + inv10 * d.y) +
                                                d.y * (inv01 * d.x + inv11 * d.y));
                    if (power > 0.0f)
                        continue;

                    // Compute alpha
                    const float g_opacity = collected_opacities[j];
                    const float gauss_exp = expf(power); // Use fast math
                    
                    const float alpha = fminf(MAX_ALPHA_VALUE, g_opacity * gauss_exp * plane_weight);
                    // if (alpha < 1.0f / 255.0f)
                    //     continue;
                    // Test updated transmittance (for early termination)
                    float test_T = T * (1.0f - alpha);
                    // if (test_T < 0.95f && alpha < 1.0f / 255.0f)
                    //     continue;

                    if (test_T < 0.0001f) {
                        done = true;
                    }
                    
                    // Process all channels with explicit loop unrolling
                    #pragma unroll
                    for (int c = 0; c < CHANNELS; c++) {
                        const float color = colours[g_idx * channels + c];
                        const float ph = phase[g_idx * channels + c];
                        
                        // Compute field contribution for wave-based rendering
                        const float scale = color * alpha * T;
                        
                        // Use sincos for more efficient computation
                        float sin_val, cos_val;
                        __sincosf(ph, &sin_val, &cos_val);
                        
                        // Accumulate
                        accum_real[c] += scale * cos_val;
                        accum_imag[c] += scale * sin_val;
                    }
                    
                    T = test_T;
                    
                    // Store position of last contributing Gaussian
                    last_position = position;
                    
                    if (done) {
                        break;
                    }
                }
            }
            
            // Write results to global memory for each channel with unrolled loop
            #pragma unroll
            for (int c = 0; c < CHANNELS; c++) {
                const int idx = (p * channels * H * W) + (c * H * W) + pix_id;
                output_real[idx] = accum_real[c];
                output_imag[idx] = accum_imag[c];
            }
            
            // Store final transmittance and POSITION of last contributor (not count)
            final_Ts[plane_offset + pix_id] = T;
            n_contrib[plane_offset + pix_id] = last_position;
        }
    }
}
void FORWARD_SPLAT::preprocess(
    int N,
    const float2* means_2D,
    const float4* cov_2D,
    const float* z_vals,
    int* radii,
    uint32_t* tiles_touched,
    const dim3 grid,
    bool prefiltered,
    bool antialiasing,
    float near_plane,
    float far_plane)
{
    const int block_size = 256;
    const dim3 blocks((N + block_size - 1) / block_size);
    
    preprocessGaussiansKernel<<<blocks, block_size>>>(
        N,
        means_2D,
        cov_2D,
        z_vals,
        radii,
        tiles_touched,
        grid,
        prefiltered,
        antialiasing,
        near_plane,
        far_plane
    );
}

// Public API for forward pass
void FORWARD_SPLAT::render(
    const dim3 grid, dim3 block,
    const uint2* tile_ranges,
    const uint32_t* point_list,
    const float2* means_2D,
    const float4* cov_2D,
    const float* z_vals,
    const float* colours,
    const float* phase,
    const float* opacities,
    const float* plane_probs,
    float* output_real,
    float* output_imag,
    float* final_Ts,
    uint32_t* n_contrib,
    int W, int H, int N,
    int num_planes, int channels,
    float near_plane,
    float far_plane)
{
    switch (channels) {
        case 1:
            renderTileKernel<1><<<grid, block>>>(
                tile_ranges,
                point_list,
                means_2D,
                cov_2D,
                z_vals,
                colours,
                phase,
                opacities,
                plane_probs,
                output_real,
                output_imag,
                final_Ts,
                n_contrib,
                W, H, N,
                num_planes, channels,
                near_plane, far_plane);
            break;
        case 3:
            renderTileKernel<3><<<grid, block>>>(
                tile_ranges,
                point_list,
                means_2D,
                cov_2D,
                z_vals,
                colours,
                phase,
                opacities,
                plane_probs,
                output_real,
                output_imag,
                final_Ts,
                n_contrib,
                W, H, N,
                num_planes, channels,
                near_plane, far_plane);
            break;
        default:
            if (channels <= MAX_CHANNELS) {
                renderTileKernel<MAX_CHANNELS><<<grid, block>>>(
                    tile_ranges,
                    point_list,
                    means_2D,
                    cov_2D,
                    z_vals,
                    colours,
                    phase,
                    opacities,
                    plane_probs,
                    output_real,
                    output_imag,
                    final_Ts,
                    n_contrib,
                    W, H, N,
                    num_planes, channels,
                    near_plane, far_plane);
            } else {
                printf("Error: Number of channels exceeds MAX_CHANNELS\n");
            }
            break;
    }
}

} // namespace cov3d_cuda