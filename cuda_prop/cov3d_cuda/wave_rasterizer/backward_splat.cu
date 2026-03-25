/*
* Backward pass for wave-based 3D Gaussian Splatting
* Tile-based implementation
*/

#include "backward_splat.h"
#include "config.h"
#include "auxiliary.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>  // For printf debugging

namespace cg = cooperative_groups;

namespace cov3d_cuda {

// Forward declarations for functions defined elsewhere
extern torch::Tensor compute_means2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& cam_means_3D,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane);
// 
extern std::vector<torch::Tensor> compute_cov2d_backward(
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
    float near_plane, float far_plane);


// covariance gradient calculation
__device__ void calculate_cov_gradient(
    float dx, float dy, float dL_dpower,
    float inv00, float inv01, float inv10, float inv11, float det,
    float& dL_dcov00, float& dL_dcov01, float& dL_dcov10, float& dL_dcov11)
{
    // 1) Gradient w.r.t. inverse-covariance entries from the exponent:
    float dL_dinv00 = -0.5f * dL_dpower * dx * dx;
    float dL_dinv01 = -0.5f * dL_dpower * dx * dy;
    float dL_dinv10 = -0.5f * dL_dpower * dy * dx;
    float dL_dinv11 = -0.5f * dL_dpower * dy * dy;

    // 2) Use full matrix derivative: dA = -A^{-1} dA^{-1} A^{-1}
    //    i.e. dL/dA = –A^{-T} (dL/dA^{-1}) A^{-T}

    // Precompute A^{-T} = (A^{-1})^T
    float bt00 = inv00, bt01 = inv10;
    float bt10 = inv01, bt11 = inv11;

    // tmp = (dL/dinv) ⋅ A^{-T}
    float tmp00 = bt00 * dL_dinv00 + bt01 * dL_dinv10;
    float tmp01 = bt00 * dL_dinv01 + bt01 * dL_dinv11;
    float tmp10 = bt10 * dL_dinv00 + bt11 * dL_dinv10;
    float tmp11 = bt10 * dL_dinv01 + bt11 * dL_dinv11;

    // dL/dcov = –A^{-T} ⋅ tmp
    dL_dcov00 = -(bt00 * tmp00 + bt01 * tmp10);
    dL_dcov01 = -(bt00 * tmp01 + bt01 * tmp11);
    dL_dcov10 = -(bt10 * tmp00 + bt11 * tmp10);
    dL_dcov11 = -(bt10 * tmp01 + bt11 * tmp11);
}

namespace BACKWARD_SPLAT {

// Main backward kernel using tile-based approach for wave-based 3DGS
// Aligned with official implementation's processing order
template <int MAX_CH>
__global__ void renderBackwardTileKernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float2* __restrict__ means_2D,
    const float4* __restrict__ cov_2D,
    const float* __restrict__ z_vals,
    const float* __restrict__ colours,
    const float* __restrict__ phase,
    const float* __restrict__ opacities,
    const float* __restrict__ plane_probs,
    const float* __restrict__ final_Ts,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ grad_output_real,
    const float* __restrict__ grad_output_imag,
    float* __restrict__ grad_means_2D,
    float* __restrict__ grad_cov_2D,
    float* __restrict__ grad_z_vals,
    float* __restrict__ grad_colours,
    float* __restrict__ grad_phase,
    float* __restrict__ grad_opacities,
    float* __restrict__ grad_plane_probs,
    int N, int num_planes, int num_channels,
    int W, int H,
    float near_plane, float far_plane)
{
    // Use the same block/thread organization as official 3DGS
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    // Simplified shared memory layout borrowed from official 3DGS
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_cov[BLOCK_SIZE];
    __shared__ float collected_opacities[BLOCK_SIZE];
    __shared__ float collected_colors[MAX_CH * BLOCK_SIZE];
    __shared__ float collected_phases[MAX_CH * BLOCK_SIZE];
    __shared__ float collected_plane_probs[BLOCK_SIZE];

    // Process each plane separately
    for (int p = 0; p < num_planes; p++) {
        int plane_offset = p * H * W;
        
        // Load gradients for this pixel and plane
        float dL_dpixel_real[MAX_CH];
        float dL_dpixel_imag[MAX_CH];
        if (inside) {
            for (int i = 0; i < num_channels; i++) {
                int idx = (p * num_channels * H * W) + (i * H * W) + pix_id;
                dL_dpixel_real[i] = grad_output_real[idx];
                dL_dpixel_imag[i] = grad_output_imag[idx];
            }
        }

        // Get final transmittance and contributor count for this plane
        const float T_final = inside ? final_Ts[plane_offset + pix_id] : 0;
        float T = T_final;

        uint32_t contributor = toDo;
        const int last_contributor = inside ? n_contrib[plane_offset + pix_id] : 0;

        // Accumulation arrays borrowed from official 3DGS pattern
        float accum_rec_real[MAX_CH] = {0};
        float accum_rec_imag[MAX_CH] = {0};
        float last_alpha = 0;
        float last_color_real[MAX_CH] = {0};
        float last_color_imag[MAX_CH] = {0};

        // Gradient computation parameters borrowed from official 3DGS
        const float ddelx_dx = 0.5f * W;
        const float ddely_dy = 0.5f * H;

        // Main rendering loop with official 3DGS structure
        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            // Load data into shared memory - borrowed loading pattern
            block.sync();
            const int progress = i * BLOCK_SIZE + block.thread_rank();
            if (range.x + progress < range.y) {
                const int coll_id = point_list[range.y - progress - 1];
                collected_id[block.thread_rank()] = coll_id;
                collected_xy[block.thread_rank()] = means_2D[coll_id];
                collected_cov[block.thread_rank()] = cov_2D[coll_id];
                collected_opacities[block.thread_rank()] = opacities[coll_id];
                collected_plane_probs[block.thread_rank()] = plane_probs[coll_id * num_planes + p];
                
                // Load colors and phases
                for (int ch = 0; ch < min(num_channels, MAX_CH); ch++) {
                    collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colours[coll_id * num_channels + ch];
                    collected_phases[ch * BLOCK_SIZE + block.thread_rank()] = phase[coll_id * num_channels + ch];
                }
            }
            block.sync();

            // Process Gaussians in this batch
            for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
                contributor--;
                if (contributor >= last_contributor)
                    continue;

                // Get Gaussian data
                const int global_id = collected_id[j];
                const float2 xy = collected_xy[j];
                const float4 cov = collected_cov[j];
                const float opacity = collected_opacities[j];
                const float plane_prob = collected_plane_probs[j];
                
                if (plane_prob < PLANE_PROBABILITY_CUTOFF)
                    continue;

                // Compute blending values exactly like official 3DGS
                const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                
                // Convert cov4 to conic (inverse covariance)
                float det = cov.x * cov.w - cov.y * cov.z;
                if (det == 0.0f) continue;
                
                float det_inv = 1.0f / det;
                float conic_x = cov.w * det_inv;
                float conic_y = -cov.y * det_inv;
                float conic_z = cov.x * det_inv;
                
                const float power = -0.5f * (conic_x * d.x * d.x + conic_z * d.y * d.y) - conic_y * d.x * d.y;
                if (power > 0.0f)
                    continue;

                const float G = expf(power);
                const float alpha = min(0.99f, opacity * G * plane_prob);
                if (alpha < 1.0f / 255.0f)
                    continue;

                // Update transmittance exactly like official 3DGS
                T = T / (1.0f - alpha);
                const float dchannel_dcolor = alpha * T;

                // Gradient computation borrowed from official 3DGS pattern
                float dL_dalpha = 0.0f;
                
                for (int ch = 0; ch < min(num_channels, MAX_CH); ch++) {
                    const float color = collected_colors[ch * BLOCK_SIZE + j];
                    const float ph = collected_phases[ch * BLOCK_SIZE + j];
                    
                    // Use fast trig functions
                    float cos_ph, sin_ph;
                    __sincosf(ph, &sin_ph, &cos_ph);
                    
                    const float color_real = color * cos_ph;
                    const float color_imag = color * sin_ph;

                    // Update accumulated values using official 3DGS pattern
                    accum_rec_real[ch] = last_alpha * last_color_real[ch] + (1.0f - last_alpha) * accum_rec_real[ch];
                    accum_rec_imag[ch] = last_alpha * last_color_imag[ch] + (1.0f - last_alpha) * accum_rec_imag[ch];
                    last_color_real[ch] = color_real;
                    last_color_imag[ch] = color_imag;

                    const float dL_dchannel_real = dL_dpixel_real[ch];
                    const float dL_dchannel_imag = dL_dpixel_imag[ch];
                    
                    dL_dalpha += (color_real - accum_rec_real[ch]) * dL_dchannel_real;
                    dL_dalpha += (color_imag - accum_rec_imag[ch]) * dL_dchannel_imag;

                    // Gradient w.r.t. color
                    float dL_dcolor = dchannel_dcolor * (cos_ph * dL_dchannel_real + sin_ph * dL_dchannel_imag);
                    atomicAdd(&grad_colours[global_id * num_channels + ch], dL_dcolor);

                    // Gradient w.r.t. phase
                    float dL_dphase = dchannel_dcolor * color * (-sin_ph * dL_dchannel_real + cos_ph * dL_dchannel_imag);
                    atomicAdd(&grad_phase[global_id * num_channels + ch], dL_dphase);
                }

                dL_dalpha *= T;
                last_alpha = alpha;

                // Background contribution (assuming zero background)
                float bg_dot_dpixel = 0.0f;
                dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_dpixel;

                // Compute gradients w.r.t. geometry - borrowed from official 3DGS
                const float dL_dG = opacity * plane_prob * dL_dalpha;
                const float gdx = G * d.x;
                const float gdy = G * d.y;
                const float dG_ddelx = -gdx * conic_x - gdy * conic_y;
                const float dG_ddely = -gdy * conic_z - gdx * conic_y;

                // Update gradients w.r.t. 2D mean position
                atomicAdd(&grad_means_2D[global_id * 2], dL_dG * dG_ddelx * ddelx_dx);
                atomicAdd(&grad_means_2D[global_id * 2 + 1], dL_dG * dG_ddely * ddely_dy);

                // Update gradients w.r.t. 2D covariance using conic gradients
                float dL_dconic_x = -0.5f * gdx * d.x * dL_dG;
                float dL_dconic_y = -0.5f * gdx * d.y * dL_dG;
                float dL_dconic_z = -0.5f * gdy * d.y * dL_dG;

                // Convert conic gradients back to covariance gradients
                // Using chain rule: dL/dcov = dL/dconic * dconic/dcov
                float dL_dcov_xx = dL_dconic_x * (cov.w * det_inv * det_inv) * cov.w + 
                                  dL_dconic_z * (-cov.y * det_inv * det_inv) * (-cov.y);
                float dL_dcov_xy = dL_dconic_x * (cov.w * det_inv * det_inv) * (-cov.y) + 
                                  dL_dconic_y * (-det_inv) + 
                                  dL_dconic_z * (-cov.y * det_inv * det_inv) * cov.x;
                float dL_dcov_yy = dL_dconic_x * (-cov.y * det_inv * det_inv) * (-cov.y) + 
                                  dL_dconic_z * (cov.x * det_inv * det_inv) * cov.x;

                atomicAdd(&grad_cov_2D[global_id * 4], dL_dcov_xx);
                atomicAdd(&grad_cov_2D[global_id * 4 + 1], dL_dcov_xy);
                atomicAdd(&grad_cov_2D[global_id * 4 + 2], dL_dcov_xy); // symmetric
                atomicAdd(&grad_cov_2D[global_id * 4 + 3], dL_dcov_yy);

                // Update gradients w.r.t. opacity and plane probability
                atomicAdd(&grad_opacities[global_id], G * plane_prob * dL_dalpha);
                atomicAdd(&grad_plane_probs[global_id * num_planes + p], opacity * G * dL_dalpha);
            }
        }
        
        // Reset for next plane
        toDo = range.y - range.x;
    }
}

// Main rendering function wrapper
void render(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    const float2* means_2D,
    const float4* cov_2D,
    const float* z_vals,
    const float* colours,
    const float* phase,
    const float* opacities,
    const float* plane_probs,
    const float* final_Ts,
    const uint32_t* n_contrib,
    const float* grad_output_real,
    const float* grad_output_imag,
    float* grad_means_2D,
    float* grad_cov_2D,
    float* grad_z_vals,
    float* grad_colours,
    float* grad_phase,
    float* grad_opacities,
    float* grad_plane_probs,
    int N, int num_planes, int num_channels,
    int W, int H,
    float near_plane, float far_plane)
{
    // Use the same launch configuration as official 3DGS
    switch (num_channels) {
        case 1:
            renderBackwardTileKernel<1><<<grid, block>>>(
                ranges, point_list, means_2D, cov_2D, z_vals, colours, phase, opacities, plane_probs,
                final_Ts, n_contrib, grad_output_real, grad_output_imag,
                grad_means_2D, grad_cov_2D, grad_z_vals, grad_colours, grad_phase, grad_opacities, grad_plane_probs,
                N, num_planes, num_channels, W, H, near_plane, far_plane);
            break;
        case 3:
            renderBackwardTileKernel<3><<<grid, block>>>(
                ranges, point_list, means_2D, cov_2D, z_vals, colours, phase, opacities, plane_probs,
                final_Ts, n_contrib, grad_output_real, grad_output_imag,
                grad_means_2D, grad_cov_2D, grad_z_vals, grad_colours, grad_phase, grad_opacities, grad_plane_probs,
                N, num_planes, num_channels, W, H, near_plane, far_plane);
            break;
        default:
            if (num_channels <= MAX_CHANNELS) {
                renderBackwardTileKernel<MAX_CHANNELS><<<grid, block>>>(
                    ranges, point_list, means_2D, cov_2D, z_vals, colours, phase, opacities, plane_probs,
                    final_Ts, n_contrib, grad_output_real, grad_output_imag,
                    grad_means_2D, grad_cov_2D, grad_z_vals, grad_colours, grad_phase, grad_opacities, grad_plane_probs,
                    N, num_planes, num_channels, W, H, near_plane, far_plane);
            }
            break;
    }
}

void preprocess(
    float fx, float fy,
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& view_matrix,
    const torch::Tensor& grad_means_2D,
    const torch::Tensor& grad_cov_2D,
    torch::Tensor& grad_means_3D,
    torch::Tensor& grad_quats,
    torch::Tensor& grad_scales,
    int W, int H,
    float near_plane, float far_plane,
    const torch::Tensor& J,
    const torch::Tensor& cov3D,
    const torch::Tensor& W_mat,
    const torch::Tensor& JW)
{
    // Use the compute_means2d_backward function for the means gradient propagation
    auto view_transform = view_matrix;
    if (view_matrix.dim() == 3) {
        view_transform = view_matrix[0];
    }
    
    // Get principal point, default to image center if not available
    float px = W * 0.5f;
    float py = H * 0.5f;
    
    auto grad_means_3D_from_2D = compute_means2d_backward(grad_means_2D, cam_means_3D, fx, fy, px, py, near_plane, far_plane);
    grad_means_3D.add_(grad_means_3D_from_2D);
    
    // Ensure proper shape for covariance gradients with explicit dimension
    auto N = grad_cov_2D.size(0);
    auto grad_cov_2D_reshaped = grad_cov_2D.view({N, 2, 2});
    
    // Use saved intermediate values from forward pass
    auto cov_grads = compute_cov2d_backward(
        grad_cov_2D_reshaped,
        cam_means_3D,
        quats,
        scales,
        view_transform,
        J,          
        cov3D,      
        W_mat,      
        JW,         
        fx, fy,
        W, H,
        near_plane, far_plane
    );
    grad_means_3D.add_(cov_grads[0]);
    grad_quats.add_(cov_grads[1]);
    grad_scales.add_(cov_grads[2]);
}

} // namespace BACKWARD_SPLAT
} // namespace cov3d_cuda