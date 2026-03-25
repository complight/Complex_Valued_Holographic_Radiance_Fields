import math
import torch
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple
import odak
import os
from odak.learn.tools import zero_pad, crop_center, circular_binary_mask

try:
    import sys
    import os
    # Add compile directory to Python path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    compile_dir = os.path.join(current_dir, 'compile')
    if compile_dir not in sys.path:
        sys.path.append(compile_dir)
    
    from cuda_modules import *
    print("Successfully loaded pre-compiled CUDA module.")
    # Keep a reference to the module for debugging
    cuda_module = sys.modules['cuda_modules']
except ImportError:
    print("Pre-compiled CUDA module not found. Compiling now...")
    from .build_cuda_extension import build_cuda_module
    cuda_module = build_cuda_module()
    print("CUDA module compilation completed.")

def calculate_padding(original_size: int, target_size: int) -> Tuple[int, int]:
    pad_size = target_size - original_size
    pad_left = pad_size // 2
    pad_right = pad_size - pad_left
    return (pad_left, pad_right)

def bandlimited_angular_spectrum_propagation(field, wavelength, pixel_pitch, distance, size, aperture_size = -1):
    size = [i * 2 for i in size]
    # Zero pad the input field
    field = odak.learn.tools.zero_pad(field, size)
    aperture = circular_binary_mask(
                                        size[0],
                                        size[1],
                                        aperture_size,
                                    ).to(field.device) * 1.
    # Compute 2D Fourier transform
    field_f = torch.fft.fftshift(torch.fft.fft2(field))

    # Calculate spatial frequencies
    Nx, Ny = field.shape
    fx = torch.fft.fftshift(torch.fft.fftfreq(Nx, d=pixel_pitch)).to(field.device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(Ny, d=pixel_pitch)).to(field.device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    # Convert scalar values to torch tensors
    x = torch.tensor(pixel_pitch * float(Nx), device=field.device)
    y = torch.tensor(pixel_pitch * float(Ny), device=field.device)
    distance = torch.tensor(distance, device=field.device)
    wavelength = torch.tensor(wavelength, device=field.device)
    
    # Calculate bandlimits
    fx_max = 1 / torch.sqrt((2 * distance * (1 / x))**2 + 1) / wavelength
    fy_max = 1 / torch.sqrt((2 * distance * (1 / y))**2 + 1) / wavelength
    bandlimit_mask = ((torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max))
    
    # Compute transfer function
    k = 2 * torch.pi / wavelength
    kz = torch.sqrt(k**2 - (2*torch.pi*FX)**2 - (2*torch.pi*FY)**2)
    kz = torch.where(torch.isnan(kz), 0, kz)
    
    # Create transfer function
    H = torch.exp(1j * distance * kz) * bandlimit_mask
    
    # Apply transfer function in frequency domain
    field_propagated_f = field_f * H  * aperture
    
    # Inverse Fourier transform to get propagated field
    field_propagated = torch.fft.ifft2(torch.fft.ifftshift(field_propagated_f))
    field_propagated = crop_center(field_propagated)
    return field_propagated


class BatchMatrixMultiplicationFunction(Function):
    @staticmethod
    def forward(ctx, diff: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
        diff = diff.contiguous()
        cov_inv = cov_inv.contiguous()
        # Save inputs for backward pass
        ctx.save_for_backward(diff, cov_inv)
        
        # Call CUDA implementation
        output = cuda_module.batch_matrix_multiplication_forward(
            diff, cov_inv)
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get saved tensors
        diff, cov_inv = ctx.saved_tensors
        
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA implementation for backward pass
        grad_diff, grad_cov_inv = cuda_module.batch_matrix_multiplication_backward(
            grad_output, diff, cov_inv)
        
        return grad_diff, grad_cov_inv

class BandlimitedPropagationFunction(Function):
    @staticmethod
    def forward(ctx, field: torch.Tensor, wavelength: float, pixel_pitch: float, 
                distance: float, size: Tuple[int, int], aperture_size: float = -1.0) -> torch.Tensor:
        # Ensure input is contiguous and on the correct device
        field = field.contiguous()
        
        # Save original field and parameters for backward pass
        ctx.save_for_backward(field)
        ctx.wavelength = wavelength
        ctx.pixel_pitch = pixel_pitch
        ctx.distance = distance
        ctx.input_size = field.shape[-2:]  # Save original input size for backward
        ctx.target_size = size
        ctx.aperture_size = aperture_size
        
        # Double the size, matching the Python implementation
        doubled_size = [i * 2 for i in size]
        
        # Zero pad the input field to the doubled size
        padded_field = odak.learn.tools.zero_pad(field, doubled_size).contiguous()
        
        # FFT operations with explicit contiguous calls
        field_f = torch.fft.fft2(padded_field)
        field_f = torch.fft.fftshift(field_f).contiguous()
        
        # Calculate frequencies
        nx, ny = doubled_size
        fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=pixel_pitch)).to(field.device)
        fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=pixel_pitch)).to(field.device)
        
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        FX = FX.contiguous()
        FY = FY.contiguous()
        
        # Separate real and imaginary parts
        real = field_f.real.contiguous()
        imag = field_f.imag.contiguous()
        
        # CUDA operation with aperture_size
        output_real, output_imag = cuda_module.bandlimited_propagation_forward(
            real, imag, FX, FY, wavelength, distance, aperture_size, pixel_pitch)
        
        # Ensure outputs are contiguous
        output = torch.complex(output_real.contiguous(), 
                             output_imag.contiguous())
        
        # Inverse operations with explicit contiguous calls
        output = torch.fft.ifftshift(output)
        propagated_field = torch.fft.ifft2(output).contiguous()
        
        # Crop back to original size
        propagated_field = odak.learn.tools.crop_center(propagated_field, size)
        
        return propagated_field

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None, None]:
        field, = ctx.saved_tensors
        wavelength = ctx.wavelength
        pixel_pitch = ctx.pixel_pitch
        distance = ctx.distance
        input_size = ctx.input_size
        target_size = ctx.target_size
        aperture_size = ctx.aperture_size
        
        # Double the target size for processing
        doubled_size = [i * 2 for i in target_size]
        
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        padded_grad = odak.learn.tools.zero_pad(grad_output, doubled_size).contiguous()
        padded_field = odak.learn.tools.zero_pad(field, doubled_size).contiguous()
        
        # FFT with explicit contiguous calls
        grad_f = torch.fft.fft2(padded_grad)
        grad_f = torch.fft.fftshift(grad_f).contiguous()
        
        field_f = torch.fft.fft2(padded_field)
        field_f = torch.fft.fftshift(field_f).contiguous()
        
        # Calculate frequencies
        nx, ny = doubled_size
        fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=pixel_pitch)).to(field.device)
        fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=pixel_pitch)).to(field.device)
        
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        FX = FX.contiguous()
        FY = FY.contiguous()
        
        # Prepare inputs for CUDA with size checks
        grad_real = grad_f.real.contiguous()
        grad_imag = grad_f.imag.contiguous()
        field_real = field_f.real.contiguous()
        field_imag = field_f.imag.contiguous()
        
        # CUDA operation with aperture_size
        grad_field_real, grad_field_imag = cuda_module.bandlimited_propagation_backward(
            grad_real, grad_imag,
            field_real, field_imag,
            FX, FY,
            wavelength, distance, aperture_size, pixel_pitch)
        
        # Combine with size check
        grad_field = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(grad_field_real, grad_field_imag))).contiguous()
        grad_field = odak.learn.tools.crop_center(grad_field, input_size)

        
        return grad_field.contiguous(), None, None, None, None, None

def BandlimitedPropagation(field: torch.Tensor, wavelength: float, pixel_pitch: float, 
                          distance: float, size: Tuple[int, int], use_cuda: bool, aperture_size: float = -1.0) -> torch.Tensor:
    # Both implementations now handle the doubling internally
    if use_cuda:
        return BandlimitedPropagationFunction.apply(field, wavelength, pixel_pitch, distance, size, aperture_size)
    else:
        return bandlimited_angular_spectrum_propagation(field, wavelength, pixel_pitch, distance, size, aperture_size)

def compute_bmm_cuda(diff: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    return BatchMatrixMultiplicationFunction.apply(diff, cov_inv)

class SumLastDimFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = input.shape
        ctx.device = input.device
        ctx.dtype = input.dtype
        
        output = cuda_module.sum_last_dim_forward(input)
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        batch_size, hw_size, _ = ctx.input_shape
        
        dummy_input = torch.empty(ctx.input_shape, device=ctx.device, dtype=ctx.dtype)
        grad_input = cuda_module.sum_last_dim_backward(
            grad_output, dummy_input)
        
        return grad_input

def sum_last_dim_cuda(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3 or x.shape[-1] != 2:
        raise ValueError(f"Input must be (N, HW, 2), but got {x.shape}")

    return SumLastDimFunction.apply(x.contiguous())

class ElementWiseMultiplicationFunction(Function):
    @staticmethod
    def forward(ctx, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        
        # Save inputs for backward pass
        ctx.save_for_backward(input1, input2)
        
        # Call CUDA implementation
        return cuda_module.element_wise_multiplication_forward(
            input1, input2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get saved tensors
        input1, input2 = ctx.saved_tensors
        
        # Ensure grad_output is contiguous
        grad_output = grad_output.contiguous()
        
        # Call CUDA implementation for backward pass
        grad_input1, grad_input2 = cuda_module.element_wise_multiplication_backward(
            grad_output, input1, input2)
        
        return grad_input1, grad_input2

def element_wise_multiplication_cuda(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    assert input1.dim() == 3 and input1.shape[-1] == 2, "input1 must be (N, HW, 2)"
    assert input2.dim() == 3 and input2.shape[-1] == 2, "input2 must be (N, HW, 2)"
    assert input1.shape[0] == input2.shape[0] and input1.shape[1] == input2.shape[1], "input1 and input2 dimensions must match"
    
    # Apply the custom CUDA function
    return ElementWiseMultiplicationFunction.apply(input1, input2)
