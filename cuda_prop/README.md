Custom CUDA extensions for wave optics simulation, providing CUDA-based bandlimited angular spectrum propagation and supporting linear algebra operations.

## cov3d_cuda

A separate sub-package implementing a tile-based, wave-optics-aware rasterizer with custom CUDA forward/backward passes. See [`cov3d_cuda/README.md`](cov3d_cuda/README.md) for details.

## Structure

```
cuda_prop/
├── __init__.py                  # Public API exports
├── python_function.py           # PyTorch autograd wrappers for all CUDA ops
├── build_cuda_extension.py      # JIT compilation via torch.utils.cpp_extension.load
├── cpp/                         # C++ binding layer (torch extension interface)
│   ├── module_def.cpp           # PYBIND11 module definition (registers all ops)
│   ├── bandlimited_propagation.cpp
│   ├── batch_matrix_mult.cpp
│   ├── sum_last_dim.cpp
│   └── element_wise_mult.cpp
├── cuda/                        # CUDA kernel implementations
│   ├── bandlimited_propagation_cuda.cu
│   ├── batch_matrix_mult_cuda.cu
│   ├── sum_last_dim_cuda.cu
│   └── element_wise_mult_cuda.cu
└── cov3d_cuda/                  # Wave-based 3D Gaussian splatting rasterizer (see its own README)
```

## Operations

### Bandlimited Angular Spectrum Propagation
Core operation for simulating coherent light propagation between parallel planes. Applies a bandlimited transfer function in the frequency domain with configurable wavelength, propagation distance, and circular aperture mask. Supports both a pure-PyTorch fallback and CUDA-accelerated path.

### Batch Matrix Multiplication (Test code for Basic Matrix Operations)
Batched multiplication of per-point 2D vectors by per-point 2×2 matrices: `(N, H*W, 2) × (N, 2, 2) → (N, H*W, 2)`. Uses shared memory to broadcast the 2×2 matrix across all spatial positions within each batch element.

### Sum Last Dimension (Test code for Basic Matrix Operations)
Reduces `(N, H*W, 2) → (N, H*W)` by summing over the last dimension. Uses vectorized `float2` loads/stores for memory throughput.

### Element-Wise Multiplication (Test code for Basic Matrix Operations)
Per-element product of two `(N, H*W, 2)` tensors. Uses vectorized `float2` loads/stores.

## Public API (`cuda_prop`)

```python
from cuda_prop import (
    BandlimitedPropagation,            
    compute_bmm_cuda,                 
    sum_last_dim_cuda,                
    element_wise_multiplication_cuda, 
)
```
All functions support `torch.autograd` (custom forward/backward CUDA kernels).

