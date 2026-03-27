# cuda_prop

CUDA-accelerated wave-optics propagation and differentiable wave rasterizer for Complex-Valued Holographic Radiance Fields.

---

## Purpose

`cuda_prop` is a Python package that provides two complementary CUDA extensions used during training and rendering:

### 1. `cuda_modules` — Propagation kernels

A JIT-compiled extension (`build_cuda_extension.py`) that implements the following differentiable CUDA operations (each with an explicit forward and backward pass):

| Function | Description |
|---|---|
| `BandlimitedPropagation` | Bandlimited Angular Spectrum (BLAS) wave-field propagation. Applies a frequency-domain transfer function with band-limiting to diffract a complex field by a given distance. Supports both CUDA and pure-PyTorch paths. |
| `compute_bmm_cuda` | Batched matrix–vector products of the form `diff^T @ cov_inv @ diff`, used when evaluating Gaussian opacity in the rasterizer. |
| `sum_last_dim_cuda` | Reduces a `(N, HW, 2)` tensor along its last dimension, treating the two values as real/imaginary parts. |
| `element_wise_multiplication_cuda` | Element-wise complex multiplication of two `(N, HW, 2)` tensors. |

These are exposed through `python_function.py` as standard `torch.autograd.Function` subclasses and re-exported from the package `__init__.py`.

### 2. `cov3d_cuda` — Wave rasterizer

A separately compiled extension (`cov3d_cuda/`) that implements tile-based Gaussian splatting adapted for wave optics. It exposes:

| Function | Description |
|---|---|
| `SplatTileCuda` | Full tile-based rasterizer: renders a complex-valued field `(P, C, H, W)` from 3D Gaussian primitives, returning amplitude and phase per depth plane. |
| `Compute3DCovarianceFunction` | Computes 3D covariance matrices from quaternions and scales. |
| `ComputeJacobianFunction` | Computes projection Jacobians for each Gaussian. |
| `ComputeCov2DFunction` | Projects 3D covariances to 2D screen space. |
| `ComputeMeans2DFunction` | Projects 3D Gaussian centres to 2D pixel coordinates. |
| `InvertCov2DFunction` | Inverts 2D covariance matrices. |
| `FusedSSIMMap` | Fused SSIM computation used in the training loss. |
| `sparse_adam_update` | Sparse Adam optimiser step, operating only on visible Gaussians. |

The `cov3d_cuda` build also downloads the **GLM** header-only math library automatically via `setup_glm.py` into `cov3d_cuda/third_party/glm/`.

---

## Directory Structure

```
cuda_prop/
├── __init__.py                  # Package entry point; exports BandlimitedPropagation and helper ops
├── python_function.py           # PyTorch autograd wrappers for cuda_modules
├── build_cuda_extension.py      # JIT loader / compile script for cuda_modules
├── cpp/                         # C++ dispatch layer (pybind11 bindings)
│   ├── bandlimited_propagation.cpp
│   ├── batch_matrix_mult.cpp
│   ├── element_wise_mult.cpp
│   ├── sum_last_dim.cpp
│   └── module_def.cpp           # PYBIND11_MODULE definition
├── cuda/                        # CUDA kernel implementations (.cu)
│   ├── bandlimited_propagation_cuda.cu
│   ├── batch_matrix_mult_cuda.cu
│   ├── element_wise_mult_cuda.cu
│   └── sum_last_dim_cuda.cu
└── cov3d_cuda/                  # Wave rasterizer sub-package
    ├── install.py               # Top-level build entry point (downloads GLM, runs setup.py)
    ├── setup.py                 # setuptools build script for cov3d_cuda extension
    ├── setup_glm.py             # Downloads the GLM library into third_party/
    ├── python_import.py         # Python autograd wrappers for cov3d_cuda
    ├── binding.cpp              # C++ pybind11 bindings for the rasterizer
    ├── conv.cu / conv.h         # Convolution helpers
    ├── spadam.cu                # Sparse Adam kernel
    ├── wave_rasterizer/         # Tile-based wave rasterizer (forward + backward CUDA passes)
    └── third_party/glm/         # GLM math library (auto-downloaded at build time)
```

---

## Dependencies

### Runtime

| Dependency | Version | Role |
|---|---|---|
| Python | 3.11 | Language runtime |
| PyTorch | 2.4.0+cu121 | Tensor operations, autograd, `torch.utils.cpp_extension` |
| CUDA Toolkit | 12.1 | Kernel compilation and runtime (`libcudart`) |
| [odak](https://github.com/kaanaksit/odak) | 0.2.6 | `zero_pad`, `crop_center`, `circular_binary_mask` used in `python_function.py` |
| GLM | 0.9.9.8 (header-only) | Vector/matrix math inside `cov3d_cuda` kernels (auto-downloaded) |

### Build-time

| Dependency | Role |
|---|---|
| `ninja` | Fast parallel compilation via `torch.utils.cpp_extension` |
| `gcc` / `g++` 12 | C++ compiler matching the conda environment |
| `nvcc` (CUDA 12.1) | CUDA kernel compiler |

All Python dependencies are installed as part of the top-level `environment.yml` (conda environment `compval`).

---

## Building

See the [top-level README](../README.md#3-compile-cuda-extensions) for full build instructions. In short:

```bash
# Step 1 — Wave rasterizer
cd cuda_prop/cov3d_cuda && python install.py
cd ../..

# Step 2 — Propagation kernels
cd cuda_prop && python build_cuda_extension.py
cd ..
```

Compiled artefacts are written to `cuda_prop/compile/` and `cuda_prop/cov3d_cuda/` respectively.

By default both extensions target `sm_80` (A100) and `sm_86` (RTX 3090 / A6000).  
To add further architectures, uncomment the corresponding lines in `build_cuda_extension.py` and `cov3d_cuda/setup.py`.
