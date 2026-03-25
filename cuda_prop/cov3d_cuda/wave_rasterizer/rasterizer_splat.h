#pragma once

#include <torch/extension.h>
#include <tuple>
#include <vector>
#include <cuda_runtime.h>
#include <functional>

// Helper function to obtain memory from chunk
template<class T>
void obtain(char*& chunk, T*& ptr, size_t N, int alignment = 1) {
    size_t remainder = (size_t)chunk % alignment;
    if (remainder != 0)
        chunk += alignment - remainder;
    ptr = (T*)chunk;
    chunk += N * sizeof(T);
}

// Special case for void* to avoid sizeof(void) which is invalid
// Adding 'inline' keyword to prevent multiple definition errors
template<>
inline void obtain<void>(char*& chunk, void*& ptr, size_t N, int alignment) {
    size_t remainder = (size_t)chunk % alignment;
    if (remainder != 0)
        chunk += alignment - remainder;
    ptr = (void*)chunk;
    chunk += N; // Just use N directly as size in bytes for void*
}

// Helper function to calculate required memory size
template<class T>
size_t required(size_t N) {
    return N * sizeof(T) + 128;
}

template<class T>
size_t required(size_t N, size_t M, size_t P = 1) {
    return N * M * P * sizeof(T) + 128;
}

namespace cov3d_cuda {

// Helper function declaration
__host__ uint32_t getHigherMsb(uint32_t n);

// Forward declarations for functions from other files
torch::Tensor compute_means2d_forward(
    const torch::Tensor& cam_means_3D,
    float fx, float fy,
    float px, float py,
    float near_plane, float far_plane);

// Modified: Use forward declaration that matches the one in forward.h
std::vector<torch::Tensor> compute_cov2d_forward(
    const torch::Tensor& cam_means_3D,
    const torch::Tensor& quats,
    const torch::Tensor& scales,
    const torch::Tensor& view_matrix,
    float fx, float fy,
    int width, int height,
    float near_plane, float far_plane);

// Structure declarations
struct GeometryState {
    float2* means_2D = nullptr;
    float4* cov_2D = nullptr;
    float* z_vals = nullptr;
    int* radii = nullptr;
    uint32_t* tiles_touched = nullptr;
    uint32_t* point_offsets = nullptr;
    void* scanning_space = nullptr;
    size_t scan_size = 0;
    
    static GeometryState fromChunk(char*& chunk, size_t N);
};

struct BinningState {
    uint64_t* keys_unsorted = nullptr;
    uint32_t* values_unsorted = nullptr;
    uint64_t* keys_sorted = nullptr;
    uint32_t* values_sorted = nullptr;
    void* sorting_space = nullptr;
    size_t sorting_size = 0;
    
    static BinningState fromChunk(char*& chunk, size_t total_pairs);
};

struct ImageState {
    uint2* ranges = nullptr;
    float* final_Ts = nullptr;
    uint32_t* n_contrib = nullptr;
    
    static ImageState fromChunk(char*& chunk, size_t num_tiles, size_t num_pixels, int num_planes);
};

// Forward declarations for namespaces with render and processing functions
namespace FORWARD_SPLAT {
    // Add preprocess function declaration to match our code reorganization
    void preprocess(
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
        float far_plane);
        
    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
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
        int width, int height, int num_gaussians,
        int num_planes, int channels,
        float near_plane, float far_plane);
}

namespace BACKWARD_SPLAT {
    void render(
        dim3 grid, dim3 block,
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
        int num_gaussians, int num_planes, int channels,
        int width, int height,
        float near_plane, float far_plane);
        
    void preprocess(
        float fx, float fy,
        const torch::Tensor& cam_means_3D,
        const torch::Tensor& quats,
        const torch::Tensor& scales,
        const torch::Tensor& view_matrix,
        const torch::Tensor& grad_means_2D,
        const torch::Tensor& grad_cov_2D,
        torch::Tensor& grad_cam_means_3D,
        torch::Tensor& grad_quats,
        torch::Tensor& grad_scales,
        int width, int height,
        float near_plane, float far_plane);
}

} // namespace cov3d_cuda

// Forward declaration for markVisible which was moved outside the class
void markVisible(
    int N,
    float* means3D,
    float* viewmatrix,
    float* projmatrix,
    bool* present,
    float near_plane,
    float far_plane);

namespace CudaRasterizer {

/**
 * Memory manager for CUDA buffers
 * Ensures proper memory management and prevents leaks
 */
class CudaMemoryManager {
public:
    static CudaMemoryManager& getInstance() {
        static CudaMemoryManager instance;
        return instance;
    }

    // Get or allocate a geometry buffer
    char* getGeometryBuffer(size_t size);
    
    // Get or allocate a binning buffer
    char* getBinningBuffer(size_t size);
    
    // Get or allocate an image buffer
    char* getImageBuffer(size_t size);
    
    // Free all allocated memory
    void freeAll();
    
    ~CudaMemoryManager() {
        freeAll();
    }

private:
    CudaMemoryManager() = default;
    
    // Prevent copies
    CudaMemoryManager(const CudaMemoryManager&) = delete;
    CudaMemoryManager& operator=(const CudaMemoryManager&) = delete;
    
    char* geom_buffer = nullptr;
    size_t geom_buffer_size = 0;
    
    char* binning_buffer = nullptr;
    size_t binning_buffer_size = 0;
    
    char* img_buffer = nullptr;
    size_t img_buffer_size = 0;
};

// Helper functions that use the memory manager
inline char* geometryBuffer(size_t size) {
    return CudaMemoryManager::getInstance().getGeometryBuffer(size);
}

inline char* binningBuffer(size_t size) {
    return CudaMemoryManager::getInstance().getBinningBuffer(size);
}

inline char* imageBuffer(size_t size) {
    return CudaMemoryManager::getInstance().getImageBuffer(size);
}

// Call this function to free all memory
inline void freeRasterizerBuffers() {
    CudaMemoryManager::getInstance().freeAll();
}

class Rasterizer {
public:
    Rasterizer();
    ~Rasterizer();

    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
    forward(
        const torch::Tensor& cam_means_3D,
        const torch::Tensor& z_vals,
        const torch::Tensor& quats,
        const torch::Tensor& scales,
        const torch::Tensor& colours,
        const torch::Tensor& phase,
        const torch::Tensor& opacities,
        const torch::Tensor& plane_probs,
        float fx, float fy,
        float px, float py,
        const torch::Tensor& view_matrix,
        std::tuple<int, int> img_size,
        std::tuple<int, int> tile_size,
        float near_plane,
        float far_plane);
    
    std::vector<torch::Tensor> backward(
        const torch::Tensor& grad_output,
        const torch::Tensor& cam_means_3D,
        const torch::Tensor& z_vals,
        const torch::Tensor& quats,
        const torch::Tensor& scales,
        const torch::Tensor& colours,
        const torch::Tensor& phase,
        const torch::Tensor& opacities,
        const torch::Tensor& plane_probs,
        float fx, float fy,
        float px, float py,
        const torch::Tensor& view_matrix,
        std::tuple<int, int> img_size,
        std::tuple<int, int> tile_size,
        const torch::Tensor& final_Ts,
        const torch::Tensor& n_contrib,
        const torch::Tensor& point_list,
        const torch::Tensor& ranges,
        float near_plane,
        float far_plane);
            
    private:
        // Intermediate tensors for backward pass
        torch::Tensor J_tensor;
        torch::Tensor cov3D_tensor;
        torch::Tensor W_tensor;
        torch::Tensor JW_tensor;
};

} // namespace CudaRasterizer