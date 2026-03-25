import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import odak


BASE_FONT_SIZE = 16
FIG_WIDTH = 8
FIG_HEIGHT = 6


def analyze_phase_statistics(phase_map, iteration, save_dir):
    """
    Analyze and visualize phase distribution characteristics.
    
    Args:
        phase_map: torch.Tensor of shape (C, H, W) or (H, W) containing phase values [0, 2π]
        iteration: int, current training iteration
        save_dir: str, directory to save the analysis plot
    """
    if phase_map.dim() == 3:
        phase_map = phase_map[0]
    
    phase_np = phase_map.detach().cpu().numpy()
    H, W = phase_np.shape
    
    # 1. Phase distribution histogram
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1.hist(phase_np.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Phase (rad)', fontsize=BASE_FONT_SIZE)
    ax1.set_ylabel('Count', fontsize=BASE_FONT_SIZE)
    ax1.set_title(f'Phase Distribution', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    hist, _ = np.histogram(phase_np.flatten(), bins=100, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist + 1e-10)) * (2*np.pi/100)
    max_entropy = np.log(2*np.pi)
    
    plt.savefig(f"{save_dir}/phase_distribution_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Phase gradient magnitude (smoothness measure)
    phase_grad_y = np.diff(phase_np, axis=0, prepend=phase_np[0:1, :])
    phase_grad_x = np.diff(phase_np, axis=1, prepend=phase_np[:, 0:1])
    phase_grad_y = np.angle(np.exp(1j * phase_grad_y))
    phase_grad_x = np.angle(np.exp(1j * phase_grad_x))
    grad_magnitude = np.sqrt(phase_grad_y**2 + phase_grad_x**2)
    
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im2 = ax2.imshow(grad_magnitude, cmap='hot', aspect='equal')
    ax2.set_xlabel('X (pixels)', fontsize=BASE_FONT_SIZE)
    ax2.set_ylabel('Y (pixels)', fontsize=BASE_FONT_SIZE)
    ax2.set_title(f'Phase Gradient Magnitude', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax2.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    plt.savefig(f"{save_dir}/gradient_magnitude_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Phase map visualization
    fig3, ax3 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im3 = ax3.imshow(phase_np, cmap='twilight', aspect='equal', vmin=0, vmax=2*np.pi)
    ax3.set_xlabel('X (pixels)', fontsize=BASE_FONT_SIZE)
    ax3.set_ylabel('Y (pixels)', fontsize=BASE_FONT_SIZE)
    ax3.set_title(f'Phase Map', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax3.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Phase (rad)')
    cbar3.ax.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    cbar3.set_label('Phase (rad)', fontsize=BASE_FONT_SIZE)
    
    plt.savefig(f"{save_dir}/phase_map_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    return 0


def analyze_amplitude_statistics(amplitude_map, iteration, save_dir):
    """
    Analyze and visualize amplitude distribution characteristics.
    
    Args:
        amplitude_map: torch.Tensor of shape (C, H, W) or (H, W) containing amplitude values
        iteration: int, current training iteration
        save_dir: str, directory to save the analysis plot
    """
    if amplitude_map.dim() == 3:
        amplitude_map = amplitude_map[0]
    
    amplitude_np = amplitude_map.detach().cpu().numpy()
    H, W = amplitude_np.shape
    
    # 1. Amplitude distribution histogram
    amplitude_flat = amplitude_np.flatten()
    
    amplitude_min = np.percentile(amplitude_flat, 0)
    amplitude_max = np.percentile(amplitude_flat, 90)
    
    bins = np.linspace(amplitude_min, amplitude_max, 250)
    
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1.hist(amplitude_flat, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Amplitude', fontsize=BASE_FONT_SIZE)
    ax1.set_ylabel('Count', fontsize=BASE_FONT_SIZE)
    ax1.set_title(f'Amplitude Distribution', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax1.set_xlim(amplitude_min, amplitude_max)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    plt.savefig(f"{save_dir}/amplitude_distribution_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Amplitude gradient magnitude (smoothness measure)
    amplitude_grad_y = np.diff(amplitude_np, axis=0, prepend=amplitude_np[0:1, :])
    amplitude_grad_x = np.diff(amplitude_np, axis=1, prepend=amplitude_np[:, 0:1])
    grad_magnitude = np.sqrt(amplitude_grad_y**2 + amplitude_grad_x**2)
    
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im2 = ax2.imshow(grad_magnitude, cmap='hot', aspect='equal')
    ax2.set_xlabel('X (pixels)', fontsize=BASE_FONT_SIZE)
    ax2.set_ylabel('Y (pixels)', fontsize=BASE_FONT_SIZE)
    ax2.set_title(f'Amplitude Gradient Magnitude', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax2.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    plt.savefig(f"{save_dir}/amplitude_gradient_magnitude_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Amplitude map visualization
    fig3, ax3 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im3 = ax3.imshow(amplitude_np, cmap='gray', aspect='equal')
    ax3.set_xlabel('X (pixels)', fontsize=BASE_FONT_SIZE)
    ax3.set_ylabel('Y (pixels)', fontsize=BASE_FONT_SIZE)
    ax3.set_title(f'Amplitude Map', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax3.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Amplitude')
    cbar3.ax.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    cbar3.set_label('Amplitude', fontsize=BASE_FONT_SIZE)
    
    plt.savefig(f"{save_dir}/amplitude_map_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    return 0


def analyze_complex_field_spectrum(amplitude_map, phase_map, iteration, save_dir):
    """
    Analyze and visualize the complex field spectrum (unified amplitude and phase analysis).
    
    Args:
        amplitude_map: torch.Tensor of shape (C, H, W) or (H, W) containing amplitude values
        phase_map: torch.Tensor of shape (C, H, W) or (H, W) containing phase values [0, 2π]
        iteration: int, current training iteration
        save_dir: str, directory to save the analysis plot
    """
    if amplitude_map.dim() == 3:
        amplitude_map = amplitude_map[0]
    if phase_map.dim() == 3:
        phase_map = phase_map[0]
    
    amplitude_np = amplitude_map.detach().cpu().numpy()
    phase_np = phase_map.detach().cpu().numpy()
    H, W = amplitude_np.shape
    
    complex_field = amplitude_np * np.exp(1j * phase_np)
    
    # 1. Spatial frequency spectrum (2D FFT)
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fft_complex = np.fft.fftshift(np.fft.fft2(complex_field))
    magnitude_spectrum = np.log(np.abs(fft_complex) + 1)
    im1 = ax1.imshow(magnitude_spectrum, cmap='viridis', aspect='equal')
    ax1.set_xlabel('Frequency X', fontsize=BASE_FONT_SIZE)
    ax1.set_ylabel('Frequency Y', fontsize=BASE_FONT_SIZE)
    ax1.set_title(f'Complex Field Frequency Spectrum', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax1.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    plt.savefig(f"{save_dir}/complex_frequency_spectrum_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Radial frequency spectrum
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    center_y, center_x = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    
    radial_profile = np.bincount(r.ravel(), weights=magnitude_spectrum.ravel()) / np.bincount(r.ravel())
    ax2.plot(radial_profile, color='steelblue', linewidth=2)
    ax2.set_xlabel('Radial Frequency (pixels)', fontsize=BASE_FONT_SIZE)
    ax2.set_ylabel('Magnitude', fontsize=BASE_FONT_SIZE)
    ax2.set_title(f'Complex Field Radial Frequency Profile', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    plt.savefig(f"{save_dir}/complex_radial_frequency_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    return 0


def analyze_gaussian_distributions(gaussians, args_prop, iteration, save_dir):
    """
    Analyze and visualize the learned Gaussian distributions in 3D space.
    
    Args:
        gaussians: Gaussians object containing the learned parameters
        args_prop: Namespace with propagation parameters
        iteration: int, current training iteration
        save_dir: str, directory to save the analysis plot
    """
    means = gaussians.means.detach().cpu().numpy()
    scales_raw = torch.exp(gaussians.pre_act_scales).detach().cpu().numpy()
    opacities_raw = torch.sigmoid(gaussians.pre_act_opacities).detach().cpu().numpy().flatten()
    
    N = len(means)
    
    # 1. Depth (Z) distribution
    depth_values = means[:, 2]
    depth_mean = np.mean(depth_values)
    depth_std = np.std(depth_values)
    depth_mask = np.abs(depth_values - depth_mean) <= 3 * depth_std
    depth_filtered = depth_values[depth_mask]
    opacities_filtered = opacities_raw[depth_mask]
    
    fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax1.hist(depth_filtered, bins=100, weights=opacities_filtered, alpha=0.7, 
             color='steelblue', edgecolor='black', density=True)
    
    if args_prop.num_planes > 1:
        for i, dist in enumerate(args_prop.distances):
            ax1.axvline(x=dist.item() if torch.is_tensor(dist) else dist, 
                       color='red', linestyle='--', linewidth=1.5, 
                       label=f'Plane {i+1}' if i == 0 else None)
        ax1.legend(fontsize=BASE_FONT_SIZE*0.8)
    
    ax1.set_xlabel('Z (depth)', fontsize=BASE_FONT_SIZE)
    ax1.set_ylabel('Density', fontsize=BASE_FONT_SIZE)
    ax1.set_title(f'Depth Distribution', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax1.set_xlim(depth_filtered.min(), depth_filtered.max())
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    plt.savefig(f"{save_dir}/depth_distribution_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Opacity distribution
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax2.hist(opacities_raw, bins=100, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Opacity', fontsize=BASE_FONT_SIZE)
    ax2.set_ylabel('Count', fontsize=BASE_FONT_SIZE)
    ax2.set_title(f'Opacity Distribution', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    mean_opacity = np.mean(opacities_raw)
    ax2.axvline(x=mean_opacity, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_opacity:.3f}')
    ax2.legend(fontsize=BASE_FONT_SIZE*0.8)
    
    plt.savefig(f"{save_dir}/opacity_distribution_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Scale distribution - cover 95% of Gaussians with 50 intervals
    mean_scales = np.mean(scales_raw, axis=1)
    
    scale_min = np.percentile(mean_scales, 0)
    scale_max = np.percentile(mean_scales, 90)
    
    bins = np.linspace(scale_min, scale_max, 250)
    
    fig3, ax3 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax3.hist(mean_scales, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Mean Scale', fontsize=BASE_FONT_SIZE)
    ax3.set_ylabel('Count', fontsize=BASE_FONT_SIZE)
    ax3.set_title(f'Gaussian Scale Distribution', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
    ax3.set_xlim(scale_min, scale_max)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=BASE_FONT_SIZE*0.9)
    
    plt.savefig(f"{save_dir}/scale_distribution_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Plane assignment analysis (if multi-plane)
    fig4, ax4 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    if args_prop.num_planes > 1 and hasattr(gaussians, 'pre_act_plane_assignment'):
        plane_probs = torch.softmax(gaussians.pre_act_plane_assignment, dim=-1).detach().cpu().numpy()
        plane_assignments = np.argmax(plane_probs, axis=1)
        
        unique, counts = np.unique(plane_assignments, return_counts=True)
        ax4.bar(unique, counts, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.set_xlabel('Plane Index', fontsize=BASE_FONT_SIZE)
        ax4.set_ylabel('Number of Gaussians', fontsize=BASE_FONT_SIZE)
        ax4.set_title(f'Gaussians per Plane', fontsize=BASE_FONT_SIZE*1.1, fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')
        ax4.tick_params(labelsize=BASE_FONT_SIZE*0.9)
        max_count = counts.max()
        ax4.set_ylim(0, max_count * 1.15)
        for i, (plane_idx, count) in enumerate(zip(unique, counts)):
            ax4.text(plane_idx, count, f'{count}\n({count/N*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=BASE_FONT_SIZE*0.8)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'Single Plane\nConfiguration', 
                ha='center', va='center', fontsize=BASE_FONT_SIZE*1.2, transform=ax4.transAxes)
    
    plt.savefig(f"{save_dir}/plane_assignment_{iteration:06d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    depth_std_full = np.std(means[:, 2])
    surface_concentration = 1.0 / (1.0 + depth_std_full * 100)
    
    return 0