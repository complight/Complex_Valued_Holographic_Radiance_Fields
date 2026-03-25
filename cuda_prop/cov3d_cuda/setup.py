from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

'''
This file is used in install.py to build the CUDA extension for the cov3d_cuda module.
'''
# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the local GLM library installed by setup_glm.py
local_glm_path = os.path.join(current_dir, 'third_party', 'glm')

# Alternative paths to try
glm_paths = [
    local_glm_path,  # First check our local copy
    os.path.abspath(os.path.join(current_dir, '../../../third_party/glm')),
    os.path.abspath(os.path.join(current_dir, '../../third_party/glm')),
    os.path.abspath(os.path.join(current_dir, '../../../diff-gaussian-rasterization/third_party/glm'))
]

# Find a valid GLM path
glm_path = None
for path in glm_paths:
    if os.path.exists(path) and os.path.exists(os.path.join(path, 'glm', 'glm.hpp')):
        glm_path = path
        break

if not glm_path:
    raise RuntimeError(
        "GLM library not found. Make sure the GLM library is installed in one of these locations:\n" +
        "\n".join(glm_paths)
    )

print(f"Using GLM path: {glm_path}")

nvcc_args = ['-O3', 
            '--threads=16',
             '-DTORCH_USE_CUDA_DSA',
             '-diag-suppress=20012', 
             '--generate-code=arch=compute_80,code=sm_80',  # A100
             '--generate-code=arch=compute_86,code=sm_86',  # 3090 / A6000
            #  '--generate-code=arch=compute_89,code=sm_89',  # 4090
            #  '--generate-code=arch=compute_90,code=sm_90'   # H100
            
             ]

setup(
    name='cov3d_cuda',
    ext_modules=[
        CUDAExtension(
            name='cov3d_cuda',
            sources=[
                os.path.join(current_dir, 'binding.cpp'), 
                os.path.join(current_dir, 'wave_rasterizer', 'forward.cu'), 
                os.path.join(current_dir, 'wave_rasterizer', 'backward.cu'),
                os.path.join(current_dir, 'wave_rasterizer', 'forward_splat.cu'),
                os.path.join(current_dir, 'wave_rasterizer', 'backward_splat.cu'),
                os.path.join(current_dir, 'wave_rasterizer', 'rasterizer_splat.cu'),
                os.path.join(current_dir, "conv.cu"),
                            
            ],
            include_dirs=[glm_path],  # Add GLM include path
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_args,  # Updated to use both architectures
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
