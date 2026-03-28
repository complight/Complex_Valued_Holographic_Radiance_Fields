<div align="center">

# Complex-Valued Holographic Radiance Fields

[![DOI](https://img.shields.io/badge/DOI-10.1145/3804450-green)](https://dl.acm.org/doi/10.1145/3804450?__cf_chl_tk=jludhiKpsnKdnHZYXsiGIMuxG8Ohz.qAgsrMvgG00JY-1774383449-1.0.1.1-YuTd0hsYfbfw17evdGlJe8aKtuk_gm9XlSC33l96L0g)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/complight/Complex_Valued_Holographic_Radiance_Fields)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b)](https://arxiv.org/abs/2506.08350)
[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://complightlab.com/publications/complex_valued_holographic_radiance_fields/)

**ACM Transactions on Graphics (presented in SIGGRAPH 2026)**

[Yicheng Zhan](https://albertgary.github.io/)<sup>1</sup> · [Dong-Ha Shin](https://dhsh.in/)<sup>2</sup> · [Seung-Hwan Baek](https://www.shbaek.com/)<sup>2</sup> · [Kaan Akşit](https://kaanaksit.com/)<sup>1</sup>

<sup>1</sup> University College London (UCL) &emsp; <sup>2</sup> Pohang University of Science and Technology (POSTECH)

</div>



## Overview

We introduce *Complex-Valued Holographic Radiance Fields*, a new variant of 3D Gaussian Splatting that utilizes complex-valued Gaussian primitives to model both amplitude and phase information from 3D scenes. Our method is transformation-free and follows the geometry of 3D scenes in terms of phase and amplitude, regardless of the viewpoints. We enable efficient rendering and optimization of our method by rasterizing optical waves emitted from a 3D scene through a differentiable wave-optics rasterizer.



## Installation

### 1. Create Conda Environment

```bash
https://github.com/complight/Complex_Valued_Holographic_Radiance_Fields.git
cd Complex_Valued_Holographic_Radiance_Fields
conda env create -f environment.yml
conda activate compval
```

This process typically takes about *30 minutes* and installs *Python 3.11.10*, *PyTorch 2.4.0+cu121*, and all required dependencies. Verify the installation after completion:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.4.0+cu121 True
```

### 2. Install PyTorch3D


```bash
pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

If the command above fails, install our tested version via `python setup.py install` from [PyTorch3D Archive](https://drive.google.com/file/d/10Ga31XCOYdEUp6yR5LX3-mAmPR7I7eZ6/view?usp=sharing).
<details>
<summary><b>Troubleshooting: <code>CUDA Toolkit 12.1</code> and <code>TORCH_CUDA_ARCH_LIST</code> related errors</b></summary>

PyTorch3D is expected to work with CUDA 12.1. If your current CUDA version is incompatible, run:
```bash
conda remove -y cuda-toolkit cuda-nvcc cuda-compiler cuda-version
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit cuda-nvcc cuda-compiler
conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y
unset TORCH_CUDA_ARCH_LIST
export TORCH_CUDA_ARCH_LIST="8.0"
```
This reinstalls CUDA 12.1 and the required C/C++ compilers inside the current conda environment.
`TORCH_CUDA_ARCH_LIST` indicates the CUDA compute capability of the target GPU, for NVIDIA A100 GPUs, this value should be `8.0`.

</details>

If none of the above solutions work, please refer to the official [PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details.

### 3. Compile CUDA Extensions

Run from the project root:

```bash
# Step 1: Compile the wave rasterizer (cov3d_cuda)
cd cuda_prop/cov3d_cuda && python install.py
cd ../..

# Step 2: Compile the propagation kernels (cuda_modules)
cd cuda_prop && python build_cuda_extension.py
cd ..
```

The wave rasterizer compilation automatically downloads the GLM library. Both extensions are compiled with `sm_80` (A100) and `sm_86` (RTX 3090 / A6000) architectures by default.
To enable additional GPU architectures, uncomment the corresponding lines in `cuda_prop/cov3d_cuda/setup.py` and `cuda_prop/build_cuda_extension.py`:

```python
# '--generate-code=arch=compute_89,code=sm_89',  # RTX 4090
# '--generate-code=arch=compute_90,code=sm_90'   # H100
```

<details>
<summary><b>Troubleshooting: <code>cannot find -lcudart</code> linker error</b></summary>
<br>

If Step 2 fails with the following error:

```
/usr/bin/ld: cannot find -lcudart
collect2: error: ld returned 1 exit status
```

or

```
ImportError: libcudart.so.11.0: cannot open shared object file: No such file or directory
```

this means the linker cannot locate `libcudart.so` on your system. To fix this, first find where the library is installed:

```bash
find /usr/local -name "libcudart.so" 2>/dev/null
```

Then open `cuda_prop/build_cuda_extension.py` and add an `extra_ldflags` argument to the `load()` call, replacing `/path/to/your/cuda/lib` with the directory from the previous step:

```python
    return load(
        name="cuda_modules",
        sources=sources,
        extra_cflags=['-O3'],
        extra_cuda_cflags=extra_cuda_flags,
        extra_ldflags=['-L/path/to/your/cuda/lib', '-lcudart'],
        build_directory=compile_dir,
        verbose=True,
        is_python_module=True
    )
```

For example, if `find` returns `/usr/local/cuda-12.1/lib64/libcudart.so`, the flag should be `'-L/usr/local/cuda-12.1/lib64'`.

</details>



## Dataset

**Option 1 — Automatic download (Default)**

```bash
bash download.sh
```

This downloads both datasets via `gdown`.

**Option 2 — Manual download**

Download the [datasets](https://drive.google.com/drive/folders/18snBWtC0GSFmoHCuh19cUFXjTUlYef46?usp=sharing) and unzip them under the `./Complex_Valued_Holographic_Radiance_Fields/data/` directory.

The expected folder structure is:

```
data/
├── colmap/                       # NeRF Synthetic (processed with COLMAP)
│   ├── colmap_chair/
│   ├── colmap_lego/
│   ├── colmap_drum/
│   ├── colmap_ship/
│   └── ...
└── nerf_llff_data/               # LLFF Forward-Facing Scenes
    ├── fern/
    ├── trex/
    ├── room/
    └── ...
```



## Light Propagation Parameters

The light propagator is based on [odak](https://github.com/kaanaksit/odak) and uses the following default parameters, defined in the `args_prop` namespace inside both `train.py` and `render_synthetic.py`:

| Parameter | Default Value | Description |
|---|----|---|
| `wavelengths` | `[639,532,473]e-9` | RGB laser wavelengths in meters (red, green, blue) |
| `pixel_pitch` | `3.74e-6` | SLM pixel pitch in meters |
| `volume_depth` | `4e-3` | Total depth of the reconstruction volume in meters |
| `d_val` | `-2e-3` | Image location offset (center of the volume) in meters |
| `pad_size` | `[800,800]` | Zero-padding size for the propagation kernel |
| `aperture_size` | `800` | Aperture size in pixels (typically max values in pad_size) |
| `num_planes` | `2` | Number of depth planes (volume_depth needs to be 0 if num_planes = 1) |

**Important:** the propagator settings in `render_synthetic.py` must match those used in `train.py` so that the reconstructions are focused correctly.



## Training

Training scripts for each dataset are provided under `script/`. For example, to train on the NeRF Synthetic chair scene:

```bash
bash script/train_colmap_chair.sh
```

Or run `train.py` directly with custom arguments:

```bash
python train.py \
    --lr 0.01 \
    --load_point \
    --dataset_name "colmap_chair" \
    --dataset_type "colmap" \
    --split_ratio 2.2 \
    --extra_scale 550 \
    --densify_every 300 \
    --generate_dense_point 3 \
    --grad_threshold 0.0005 \
    --densepoint_scatter 1 \
    --img_size 800 800 \
    --num_itrs 20000
```

Checkpoints and training logs are saved to `./result/checkpoints/` and `./result/log.txt`.


## Evaluation
We provide the [pretrained weights](https://drive.google.com/file/d/1XExwIFQPtS19O6dzjWaF3wjNA3vKtQl5/view?usp=drive_link) for direct rendering and evaluation.

Use `render_synthetic.py` to render novel views and compute metrics (PSNR, SSIM, LPIPS) from a trained checkpoint.
The `--train_script` argument is **required** as it points to the same `.sh` file used during training so that dataset parameters are automatically extracted and aligned with the training configuration.

```bash
python render_synthetic.py \
    --train_script ./trained_weights/train_colmap_chair.sh \
    --load_checkpoint "./trained_weights/best_gaussians_10000_chair800.pth" \
    --output_format both
```

The following parameters are automatically read from the training script and do **not** need to be specified manually: `--dataset_name`, `--dataset_type`, `--split_ratio`, `--extra_scale`, `--img_size`.

Rendered outputs (per-plane amplitude, phase, and reconstructed intensities) are saved to `./output/` in mp4 and gif.



## Hardware Requirements

All experiments in the paper are conducted on a NVIDIA A100 80GB GPU.

- **800×800 resolution:** A100 80GB is recommended.
- If you encounter GPU out-of-memory errors, reduce the resolution via `--img_size 512 512`.



## Citation

```bibtex
@article{zhan2025complexvalued,
  author = {Zhan, Yicheng and Shin, Dong-Ha and Baek, Seung-Hwan and Ak\c{s}it, Kaan},
  title = {Complex-Valued Holographic Radiance Fields},
  year = {2026},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  issn = {0730-0301},
  journal = {ACM Transactions on Graphics (Presented in SIGGRAPH 2026)},
  month = mar,
  note = {},
  keywords = {Novel View Synthesis, Radiance Fields, 3D Gaussians, Computer-Generated Holography},
  location = {Los Angeles, California, USA},
  doi = {10.1145/3804450},
  url = {https://doi.org/10.1145/3804450},
}
```



## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## Contact

For questions about the code or methodology:

**Yicheng Zhan:** yicheng_zhan2001@outlook.com
