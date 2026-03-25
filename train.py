import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import argparse
import numpy as np
import odak
import imageio.v2 as imageio
import lpips  # Added LPIPS import
from argparse import Namespace
from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from utils import trivial_collate, set_seed, ndc_to_screen_camera, \
                  get_colmap_datasets, get_tandt_datasets, GaussianLoss, \
                  Adan, SparseGaussianAdam, setup_optimizer, propagator, console_only_print, multiplane_loss, plane_assignment_loss
from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import sys


result_dir = os.path.join("./result")
checkpoint_dir = os.path.join(result_dir, "checkpoints")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
logsave = result_dir

log_debug = True

if log_debug:
    sys.stdout = open(os.path.join(logsave, "log.txt"), "w")

set_seed(100)

def DPAC(amplitudes, phases, three_pi=False, mean_adjust=True):
    """converts amplitude and phase to double phase coding

    amplitudes:  per-pixel amplitudes of the complex field
    phases:  per-pixel phases of the complex field
    three_pi:  if True, outputs values in a 3pi range, instead of 2pi
    mean_adjust:  if True, centers the phases in the range of interest
    """
    # normalize
    amplitudes = amplitudes / amplitudes.max()
    phases_a = phases - torch.arccos(amplitudes)
    phases_b = phases + torch.arccos(amplitudes)

    phases_out = phases_a
    phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
    phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

    if three_pi:
        max_phase = 3 * odak.pi
    else:
        max_phase = 2 * odak.pi

    if mean_adjust:
        phases_out -= phases_out.mean()

    return (phases_out + max_phase / 2) % max_phase - max_phase / 2

# Function to process and visualize dataset images
def process_dataset_images(dataset, indices, has_depth):
    images = []
    for i in indices:
        img = (dataset[i]["image"]*255.0).cpu().numpy().astype(np.uint8)

        if has_depth:
            # Depth visualization
            depth = dataset[i]["depth"].cpu().numpy()
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_viz = np.repeat((depth_norm * 255.0).astype(np.uint8), 3, axis=2)

            # Get targets
            img_gpu = dataset[i]["image"].cuda()
            depth_gpu = dataset[i]["depth"].cuda()
            targets, _, _ = multiplane_loss(
                target_image=img_gpu.permute(2, 0, 1),
                target_depth=depth_gpu.permute(2, 0, 1),
                args_prop=args_prop
            )

            # Prepare target visualizations
            target_viz = [(target.detach().cpu().numpy() * 255.0).astype(np.uint8) for target in targets]
            target_viz = [np.transpose(t, (1, 2, 0)) for t in target_viz]

            # Combine all visualizations
            img = np.concatenate([img, depth_viz] + target_viz, axis=1)

        images.append(img)
    return images

def make_trainable(gaussians):

    gaussians.pre_act_quats.requires_grad_()
    gaussians.means.requires_grad_()
    gaussians.pre_act_scales.requires_grad_()
    gaussians.colours.requires_grad_()
    gaussians.pre_act_phase.requires_grad_()
    # gaussians.spherical_harmonics.requires_grad_()
    gaussians.pre_act_opacities.requires_grad_()
    gaussians.pre_act_plane_assignment.requires_grad_()
    # gaussians.zernike_coeffs.requires_grad_()

def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target tensors."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    return peak_signal_noise_ratio(target_np, pred_np)

def run_training(args, args_prop):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    # Width x Height
    img_size = args.img_size

    # Initialize LPIPS metric
    lpips_fn = lpips.LPIPS(net='vgg').cuda()  # Added LPIPS initialization

    # Choose dataset loader based on dataset_type
    if args.dataset_type == "colmap" or args.dataset_type == "mip360" or args.dataset_type == "nerf_llff_data":
        # Import COLMAP dataset loader
        train_dataset, val_dataset, test_dataset, pointcloud_data = get_colmap_datasets(
            dataset_type=args.dataset_type,
            dataset_name=args.dataset_name,
            image_size=img_size,
            args = args,
            device=args.device
        )
    elif args.dataset_type == "tandt":
        # Import TandT dataset loader
        train_dataset, val_dataset, test_dataset, pointcloud_data = get_tandt_datasets(
            dataset_name=args.dataset_name,
            image_size=img_size,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Detect if we have a validation set with images (COLMAP, NeRF) or just cameras (LLFF)
    val_has_images = len(val_dataset) > 0 and "image" in val_dataset[0]

    # Preparing some code for visualization
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]
    val_viz_idxs = np.linspace(10, len(val_dataset)-1, 5).astype(np.int32)[:4] if len(val_dataset) > 0 else []

    print(f"viz_idxs {viz_idxs}")
    print(f"val_viz_idxs {val_viz_idxs}")
    all_viz_imgs = []

    # Check if depth is available
    has_train_depth = len(train_dataset) > 0 and "depth" in train_dataset[0]
    has_val_depth = val_has_images and len(val_dataset) > 0 and "depth" in val_dataset[0]

    # Process training and validation images
    all_viz_imgs.extend(process_dataset_images(train_dataset, viz_idxs, has_train_depth))

    if val_has_images:
        all_viz_imgs.extend(process_dataset_images(val_dataset, val_viz_idxs, has_val_depth))

    if all_viz_imgs:
        gt_viz_img = torch.cat([torch.from_numpy(img) for img in all_viz_imgs], dim=0)
        gt_viz_img = gt_viz_img.permute(2, 0, 1)
        odak.learn.tools.save_image(
            f"{result_dir}/gt_viz_img.png",
            gt_viz_img,
            cmin=0.,
            cmax=255.
        )

    # Process cameras
    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"], img_size).cuda() for i in viz_idxs]
    if val_viz_idxs.any():
        val_viz_cameras = [ndc_to_screen_camera(val_dataset[i]["camera"], img_size).cuda() for i in val_viz_idxs]
        viz_cameras.extend(val_viz_cameras)

    if args.load_checkpoint:
        gaussians = Gaussians(
            init_type="gaussians",
            device=args.device,
            load_path=args.load_checkpoint,
            args_prop=args_prop
        )
    elif args.load_point:
        gaussians = Gaussians(
            init_type="point",
            device=args.device,
            args_prop=args_prop,
            pointcloud_data=pointcloud_data,
            generate_dense_point=args.generate_dense_point,
            densepoint_scatter=args.densepoint_scatter
        )
    else:
        gaussians = Gaussians(
            # chair = 150000, lego = 250000
            # num_points=150000,
            num_points=100000,
            init_type="random",
            device=args.device,
            args_prop=args_prop,
            img_size = img_size
        )

    scene = Scene(gaussians, args_prop)
    make_trainable(gaussians)
    optimizer, scheduler, parameters = setup_optimizer(gaussians, args.num_itrs, args.lr)

    # Training loop
    running_losses = []
    running_ssim_losses = []
    running_psnrs = []  # Keep track of PSNRs for moving average
    best_psnr = 0.0
    pbar = tqdm(range(args.num_itrs), desc='Training')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for itr in pbar:
        # print(f"---------------------- {itr} --------------------")
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        gt_img = gt_img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        C, H, W = gt_img.size()
        camera = ndc_to_screen_camera(data[0]["camera"], img_size).cuda()

        # Initialize targets for multiplane supervision
        has_depth = "depth" in data[0]
        loss_ssim = GaussianLoss

        if not has_depth and args_prop.num_planes > 1:
            raise ValueError("Depth is required for supervision more than 1 plane.")
        if has_depth:
            depth = data[0]["depth"].cuda()
            depth = depth.permute(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        else:
            H, W = gt_img.shape[1], gt_img.shape[2]
            depth = torch.zeros((1, H, W), device='cuda')

        targets, loss_function, mask = multiplane_loss(
            target_image=gt_img,
            target_depth=depth,
            args_prop=args_prop
        )

        optimizer.zero_grad()
        starter.record()
        hologram_complex, plane_fields = scene.render(
            camera,
            img_size=img_size,
            bg_colour=(0.0, 0.0, 0.0),
            step=itr,
            max_step=args.num_itrs
        )
        # if itr == 201:
        #     return None
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        with console_only_print():
            print(f"--forward {curr_time}")
        # return None
        starter.record()
        phase_map = odak.learn.wave.calculate_phase(hologram_complex) % (2 * odak.pi)
        amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)
        phase_map = phase_map - phase_map.mean()
        # phase_only = DPAC(amplitude, phase_map)
        reconstruction_intensities_sum = propagator.reconstruct(phase_map, amplitude=amplitude, no_grad = False)
        # reconstruction_intensities_sum_phaseonly = propagator.reconstruct(phase_only, amplitude=None, no_grad = False)

        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        with console_only_print():
            print(f"recon {curr_time}")

        loss = 0.

        reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim = 0)
        for idx, (reconstruction_intensity, target) in enumerate(zip(reconstruction_intensities, targets)):
            pred_cropped = odak.learn.tools.crop_center(reconstruction_intensity, size=(H, W))
            pred_cropped = torch.clamp(pred_cropped, min=0.0, max=1.0)
            loss += loss_function(pred_cropped, target, idx)
            ssim_loss = loss_ssim(pred_cropped, target)
            loss += ssim_loss
            if idx == 0:
                current_psnr = calculate_psnr(pred_cropped, target)
                running_psnrs.append(current_psnr)
        # plane_loss = plane_assignment_loss(scene, quantized_depth, args_prop.num_planes)
        # loss += plane_loss
        # print(f"plane_loss: {plane_loss}")
        del reconstruction_intensities, targets

        running_losses.append(loss.item())
        running_ssim_losses.append(ssim_loss.item())
        mean_loss = sum(running_losses[-50:]) / min(len(running_losses), 50)  # Moving average of last 50 losses
        mean_ssim = sum(running_ssim_losses[-50:]) / min(len(running_ssim_losses), 50)
        mean_psnr = sum(running_psnrs[-50:]) / min(len(running_psnrs), 50)

        starter.record()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters, 1.0)

        if isinstance(optimizer, SparseGaussianAdam):
            optimizer.step(scene.visible_indices)
        else:
            optimizer.step()
        scheduler.step()

        if args_prop.densify_every != -1 and itr % args.eval_freq != 0 and itr % args_prop.densify_every == 0 and itr >= args_prop.start_densify_step and itr <= args_prop.end_densify_step:
            gaussians.density_control(args.grad_threshold)
            gaussians.opacity_regularization()
            make_trainable(gaussians)
            optimizer, scheduler, parameters = setup_optimizer(
                gaussians,
                args.num_itrs,
                args.lr,
                current_iter=itr,
                prev_optimizer=optimizer,
                prev_scheduler=scheduler
            )

        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        with console_only_print():
            print(f"back {curr_time}")

        pbar.set_postfix({
            'Loss': f'{mean_loss:.6f}',
            'PSNR': f'{mean_psnr:.2f}',
            's': f'{mean_ssim:.6f}',
            'G': f'{len(scene.gaussians)}'
        })

        optimizer.zero_grad()

        if itr % 100 == 0:
            torch.cuda.empty_cache()

        # if args.viz_freq != -1 and itr % args.viz_freq == 0:
        if args.viz_freq != -1 and itr % args.viz_freq == 0:
            with torch.no_grad():
                # Lists to store visualizations for each plane
                recon_list = [[] for _ in range(args_prop.num_planes)]
                raw_phase_list = [[] for _ in range(args_prop.num_planes)]
                raw_amp_list = [[] for _ in range(args_prop.num_planes)]
                phase_list = [[] for _ in range(args_prop.num_planes)]
                amp_list = [[] for _ in range(args_prop.num_planes)]

                for cam in viz_cameras:
                    hologram_complex, plane_fields = scene.render(
                        cam,
                        img_size=img_size,
                        bg_colour=(0.0, 0.0, 0.0)
                    )
                    phase_map = odak.learn.wave.calculate_phase(hologram_complex) % (2 * odak.pi)
                    amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)
                    # phase_only = DPAC(amplitude, phase_map)
                    reconstruction_intensities_sum = propagator.reconstruct(phase_map, amplitude=amplitude, no_grad = False)
                    # reconstruction_intensities_sum = propagator.reconstruct(phase_only, amplitude=None, no_grad = False)
                    reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)

                    # Process each plane separately
                    for plane_idx in range(args_prop.num_planes):
                        # Get plane-specific field
                        plane_field = plane_fields[plane_idx]

                        # Get plane-specific reconstruction
                        reconstruction_intensity = reconstruction_intensities[plane_idx]
                        pred_img = torch.clamp(reconstruction_intensity, min=0.0, max=1.0)

                        # Store visualizations for this plane
                        raw_phase_list[plane_idx].append(odak.learn.wave.calculate_phase(plane_field) % (2 * odak.pi))
                        raw_amp_list[plane_idx].append(odak.learn.wave.calculate_amplitude(plane_field))
                        recon_list[plane_idx].append(odak.learn.tools.crop_center(pred_img, size=(H, W)))
                        if plane_idx == 0:
                            phase_list[plane_idx].append(odak.learn.tools.crop_center(phase_map, size=(H, W)).squeeze(0))
                            amp_list[plane_idx].append(odak.learn.tools.crop_center(amplitude, size=(H, W)).squeeze(0))

                    del hologram_complex, plane_fields, phase_map, amplitude, reconstruction_intensities_sum
                    del reconstruction_intensities, pred_img

                # Save combined images for each plane
                for plane_idx in range(args_prop.num_planes):
                    # Create suffix for multi-plane files
                    plane_suffix = f"_{plane_idx+1}" if args_prop.num_planes > 1 else ""
                    combined_image = torch.hstack(recon_list[plane_idx])
                    odak.learn.tools.save_image(
                        f"{result_dir}/frame_{itr:06d}{plane_suffix}.png", combined_image, cmin=0., cmax=1.0)
                    combined_raw_phase = torch.hstack(raw_phase_list[plane_idx])
                    odak.learn.tools.save_image(
                        f"{result_dir}/raw_phase_{itr:06d}{plane_suffix}.png", combined_raw_phase, cmin=0., cmax=2 * odak.pi)
                    combined_raw_amp = torch.hstack(raw_amp_list[plane_idx])
                    odak.learn.tools.save_image(
                        f"{result_dir}/raw_amp_{itr:06d}{plane_suffix}.png", combined_raw_amp, cmin=0., cmax=combined_raw_amp.max())
                    if plane_idx == 0:
                        combined_phase_map = torch.hstack(phase_list[plane_idx])
                        odak.learn.tools.save_image(
                            f"{result_dir}/phase_{itr:06d}.png", combined_phase_map, cmin=0., cmax=2 * odak.pi)

                        combined_amplitude = torch.hstack(amp_list[plane_idx])
                        odak.learn.tools.save_image(
                            f"{result_dir}/amp_{itr:06d}.png", combined_amplitude, cmin=0., cmax=1.)

                # Clean up to free memory
                del raw_phase_list, raw_amp_list, phase_list, amp_list, recon_list

        # if itr == 0 or itr % args.eval_freq == 0:
        if itr != 0 and itr % args.eval_freq == 0:
            # Skip evaluation metrics for LLFF dataset if validation set doesn't have images
            if args.dataset_type == "nerf_llff_data":
                print(f"Evaluation for LLFF dataset at iteration {itr}")
                # Save model checkpoint
                latest_model_path = os.path.join(checkpoint_dir, f"latest_gaussians_{itr}.pth")
                gaussians.save_gaussians(latest_model_path)
                print(f"Saved model checkpoint at iteration {itr}")

                # Calculate metrics using the TRAIN dataset
                psnr_vals, ssim_vals, lpips_vals = [], [], []  # Added lpips_vals list

                print("Calculating metrics using training data...")
                for train_data in tqdm(train_loader, desc="Running Evaluation on Training Data"):
                    # Move inputs to GPU only when needed
                    gt_img = train_data[0]["image"].cuda()
                    gt_img = gt_img.permute(2, 0, 1)
                    camera = ndc_to_screen_camera(train_data[0]["camera"], img_size).cuda()
                    has_depth = "depth" in train_data[0]
                    targets = None

                    if has_depth:
                        depth = train_data[0]["depth"].cuda()
                        depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
                        targets, _, _ = multiplane_loss(
                            target_image=gt_img,
                            target_depth=depth,
                            args_prop=args_prop
                        )
                    else:
                        targets = [gt_img]

                    with torch.no_grad():
                        hologram_complex, plane_fields = scene.render(
                            camera,
                            img_size=img_size,
                            bg_colour=(0.0, 0.0, 0.0),
                        )
                        phase_map = odak.learn.wave.calculate_phase(hologram_complex).unsqueeze(0) % (2 * odak.pi)
                        amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)
                        # phase_only = DPAC(amplitude, phase_map)
                        reconstruction_intensities_sum = propagator.reconstruct(phase_map, amplitude=amplitude, no_grad = False)
                        # reconstruction_intensities_sum = propagator.reconstruct(phase_only, amplitude=None, no_grad = False)
                        reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)

                        # Use the primary plane for comparison (default to 0 for single plane or 1 for multi-plane)
                        primary_plane_idx = min(1, args_prop.num_planes - 1)
                        primary_recon = reconstruction_intensities[primary_plane_idx]
                        primary_recon = torch.clamp(primary_recon, min=0.0, max=1.0)
                        primary_recon = odak.learn.tools.crop_center(primary_recon, size=(H, W))

                        # Move to CPU for metrics calculation
                        primary_recon_cpu = primary_recon.detach().cpu().numpy()
                        target_npy = targets[primary_plane_idx].detach().cpu().numpy()

                        # Calculate metrics
                        psnr = peak_signal_noise_ratio(primary_recon_cpu, target_npy)
                        ssim = structural_similarity(primary_recon_cpu, target_npy, channel_axis=0, data_range=1.0)

                        # Calculate LPIPS (need to convert back to tensor and normalize)
                        primary_recon_tensor = torch.from_numpy(primary_recon_cpu).cuda()
                        target_tensor = torch.from_numpy(target_npy).cuda()

                        # LPIPS expects inputs in [-1, 1] range
                        primary_recon_norm = 2 * primary_recon_tensor - 1
                        target_norm = 2 * target_tensor - 1

                        # Add channel dimension if needed and ensure correct format
                        if primary_recon_norm.dim() == 2:
                            primary_recon_norm = primary_recon_norm.unsqueeze(0)
                        if target_norm.dim() == 2:
                            target_norm = target_norm.unsqueeze(0)

                        # LPIPS expects batch dimension
                        primary_recon_norm = primary_recon_norm.unsqueeze(0)
                        target_norm = target_norm.unsqueeze(0)

                        lpips_val = lpips_fn(primary_recon_norm, target_norm).item()

                        psnr_vals.append(psnr)
                        ssim_vals.append(ssim)
                        lpips_vals.append(lpips_val)

                        # Clear GPU memory
                        del hologram_complex, plane_fields, phase_map, amplitude, reconstruction_intensities_sum
                        del reconstruction_intensities, gt_img, primary_recon, primary_recon_cpu, target_npy
                        del primary_recon_tensor, target_tensor, primary_recon_norm, target_norm

                # Calculate and report metrics from training data
                mean_psnr = np.mean(psnr_vals)
                mean_ssim = np.mean(ssim_vals)
                mean_lpips = np.mean(lpips_vals)
                print(f"iter: {itr}")
                print(f"[*] Evaluation on training data --- Mean PSNR: {mean_psnr:.3f}")
                print(f"[*] Evaluation on training data --- Mean SSIM: {mean_ssim:.3f}")
                print(f"[*] Evaluation on training data --- Mean LPIPS: {mean_lpips:.3f}")

                # Save model checkpoints based on performance
                if mean_psnr > best_psnr:
                    best_psnr = mean_psnr
                    best_model_path = os.path.join(checkpoint_dir, f"best_gaussians_{itr}.pth")
                    gaussians.save_gaussians(best_model_path)
                    print(f"Saved BEST model with PSNR {best_psnr:.3f}")

                # Generate visual output for novel view synthesis (using validation cameras)
                with torch.no_grad():
                    # Store reconstructions on CPU
                    recon_list = [[] for _ in range(args_prop.num_planes)]
                    for val_data in tqdm(val_loader, desc="Generating novel view visualizations"):
                        camera = ndc_to_screen_camera(val_data[0]["camera"], img_size).cuda()
                        # Render the validation view without metrics
                        hologram_complex, plane_fields = scene.render(
                            camera,
                            img_size=img_size,
                            bg_colour=(0.0, 0.0, 0.0),
                        )
                        phase_map = odak.learn.wave.calculate_phase(hologram_complex).unsqueeze(0) % (2 * odak.pi)
                        amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)
                        # phase_only = DPAC(amplitude, phase_map)
                        reconstruction_intensities_sum = propagator.reconstruct(phase_map, amplitude=amplitude, no_grad = False)
                        # reconstruction_intensities_sum = propagator.reconstruct(phase_only, amplitude=None, no_grad = False)
                        reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)

                        # Process each plane and move to CPU immediately
                        for plane_idx in range(args_prop.num_planes):
                            reconstruction_intensity = reconstruction_intensities[plane_idx]
                            reconstruction_intensity = torch.clamp(reconstruction_intensity, min=0.0, max=1.0)
                            reconstruction_intensity = odak.learn.tools.crop_center(reconstruction_intensity, size=(H, W))
                            # Move to CPU immediately to free GPU memory
                            recon_list[plane_idx].append(reconstruction_intensity.detach().cpu())

                        # Clear GPU memory
                        del hologram_complex, plane_fields, phase_map, amplitude, reconstruction_intensities_sum, reconstruction_intensities

                    # Save GIFs for each plane - using same naming convention as other datasets
                    for plane_idx in tqdm(range(args_prop.num_planes), desc="GIF Generation"):
                        plane_suffix = f"_{plane_idx+1}" if args_prop.num_planes > 1 else ""
                        # Convert tensor frames to numpy arrays for GIF creation
                        gif_frames = []
                        # Process in smaller batches to avoid memory spikes
                        batch_size = min(10, len(recon_list[plane_idx]))
                        for batch_start in range(0, len(recon_list[plane_idx]), batch_size):
                            batch_end = min(batch_start + batch_size, len(recon_list[plane_idx]))
                            batch_frames = recon_list[plane_idx][batch_start:batch_end]
                            for frame in batch_frames:
                                np_frame = frame.numpy()
                                np_frame = np.transpose(np_frame, (1, 2, 0)) if np_frame.ndim == 3 else np_frame
                                np_frame = (np_frame * 255).astype(np.uint8)
                                gif_frames.append(np_frame)
                            # Clear batch memory
                            del batch_frames

                        # Save GIF with the same naming convention as other datasets
                        imageio.mimsave(f"{result_dir}/eval_{itr}{plane_suffix}.gif", gif_frames, duration=100, loop=0)

                    # Clean up
                    del psnr_vals, ssim_vals, lpips_vals, mean_psnr, mean_ssim, mean_lpips
                    sys.stdout.flush()
                    torch.cuda.empty_cache()

            else:
                # Regular evaluation for datasets with validation images (COLMAP, NeRF)
                psnr_vals, ssim_vals, lpips_vals = [], [], []  # Added lpips_vals list
                # Store reconstructions on CPU
                recon_list = [[] for _ in range(args_prop.num_planes)]

                for val_data in tqdm(val_loader, desc="Running Evaluation"):
                    # Move inputs to GPU only when needed
                    gt_img = val_data[0]["image"].cuda()
                    gt_img = gt_img.permute(2, 0, 1)
                    camera = ndc_to_screen_camera(val_data[0]["camera"], img_size).cuda()
                    has_depth = "depth" in val_data[0]
                    targets = None

                    if has_depth:
                        depth = val_data[0]["depth"].cuda()
                        depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
                        targets, _, _ = multiplane_loss(
                            target_image=gt_img,
                            target_depth=depth,
                            args_prop=args_prop
                        )
                    else:
                        targets = [gt_img]

                    with torch.no_grad():
                        hologram_complex, plane_fields = scene.render(
                            camera,
                            img_size=img_size,
                            bg_colour=(0.0, 0.0, 0.0),
                        )
                        phase_map = odak.learn.wave.calculate_phase(hologram_complex).unsqueeze(0) % (2 * odak.pi)
                        amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)
                        # phase_only = DPAC(amplitude, phase_map)
                        reconstruction_intensities_sum = propagator.reconstruct(phase_map, amplitude=amplitude, no_grad = False)
                        # reconstruction_intensities_sum = propagator.reconstruct(phase_only, amplitude=None, no_grad = False)
                        reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)

                        # Move plane reconstructions to CPU immediately
                        for plane_idx in range(args_prop.num_planes):
                            reconstruction_intensity = reconstruction_intensities[plane_idx]
                            reconstruction_intensity = torch.clamp(reconstruction_intensity, min=0.0, max=1.0)
                            reconstruction_intensity = odak.learn.tools.crop_center(reconstruction_intensity, size=(H, W))
                            # Move to CPU immediately
                            recon_list[plane_idx].append(reconstruction_intensity.detach().cpu())

                        # Use the primary plane for comparison (default to 0 for single plane or 1 for multi-plane)
                        primary_plane_idx = min(1, args_prop.num_planes - 1)
                        primary_recon = reconstruction_intensities[primary_plane_idx]
                        primary_recon = torch.clamp(primary_recon, min=0.0, max=1.0)
                        primary_recon = odak.learn.tools.crop_center(primary_recon, size=(H, W))

                        # Move to CPU for metrics calculation
                        primary_recon_cpu = primary_recon.detach().cpu().numpy()
                        target_npy = targets[primary_plane_idx].detach().cpu().numpy()

                        # Calculate metrics
                        psnr = peak_signal_noise_ratio(primary_recon_cpu, target_npy)
                        ssim = structural_similarity(primary_recon_cpu, target_npy, channel_axis=0, data_range=1.0)

                        # Calculate LPIPS (need to convert back to tensor and normalize)
                        primary_recon_tensor = torch.from_numpy(primary_recon_cpu).cuda()
                        target_tensor = torch.from_numpy(target_npy).cuda()

                        # LPIPS expects inputs in [-1, 1] range
                        primary_recon_norm = 2 * primary_recon_tensor - 1
                        target_norm = 2 * target_tensor - 1

                        # Add channel dimension if needed and ensure correct format
                        if primary_recon_norm.dim() == 2:
                            primary_recon_norm = primary_recon_norm.unsqueeze(0)
                        if target_norm.dim() == 2:
                            target_norm = target_norm.unsqueeze(0)

                        # LPIPS expects batch dimension
                        primary_recon_norm = primary_recon_norm.unsqueeze(0)
                        target_norm = target_norm.unsqueeze(0)

                        lpips_val = lpips_fn(primary_recon_norm, target_norm).item()

                        psnr_vals.append(psnr)
                        ssim_vals.append(ssim)
                        lpips_vals.append(lpips_val)

                        # Clear GPU memory
                        del hologram_complex, plane_fields, phase_map, amplitude, reconstruction_intensities_sum
                        del reconstruction_intensities, gt_img, primary_recon, primary_recon_cpu, target_npy
                        del primary_recon_tensor, target_tensor, primary_recon_norm, target_norm

                        if (val_loader.dataset.__len__() > 50) and (len(val_loader) % 10 == 0):
                            # Periodically clear CUDA cache for large datasets
                            torch.cuda.empty_cache()

                # Save GIFs for each plane
                for plane_idx in tqdm(range(args_prop.num_planes), desc="GIF Generation"):
                    plane_suffix = f"_{plane_idx+1}" if args_prop.num_planes > 1 else ""

                    # Process in smaller batches to avoid memory spikes
                    gif_frames = []
                    batch_size = min(10, len(recon_list[plane_idx]))

                    for batch_start in range(0, len(recon_list[plane_idx]), batch_size):
                        batch_end = min(batch_start + batch_size, len(recon_list[plane_idx]))
                        batch_frames = recon_list[plane_idx][batch_start:batch_end]

                        for frame in batch_frames:
                            np_frame = frame.numpy()
                            np_frame = np.transpose(np_frame, (1, 2, 0)) if np_frame.ndim == 3 else np_frame
                            np_frame = (np_frame * 255).astype(np.uint8)
                            gif_frames.append(np_frame)

                        # Clear batch memory
                        del batch_frames

                    # Save GIF
                    imageio.mimsave(f"{result_dir}/eval_{itr}{plane_suffix}.gif", gif_frames, duration=100, loop=0)

                # Calculate and report metrics
                mean_psnr = np.mean(psnr_vals)
                mean_ssim = np.mean(ssim_vals)
                mean_lpips = np.mean(lpips_vals)  # Added mean LPIPS calculation
                print(f"iter: {itr}")
                print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
                print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")
                print(f"[*] Evaluation --- Mean LPIPS: {mean_lpips:.3f}")  # Added LPIPS reporting

                # Save model checkpoints based on performance
                if mean_psnr > best_psnr:
                    best_psnr = mean_psnr
                    best_model_path = os.path.join(checkpoint_dir, f"best_gaussians_{itr}.pth")
                    gaussians.save_gaussians(best_model_path)
                    print(f"Saved BEST model with PSNR {best_psnr:.3f}")
                else:
                    latest_model_path = os.path.join(checkpoint_dir, f"latest_gaussians_{itr}.pth")
                    gaussians.save_gaussians(latest_model_path)
                    print(f"Saved latest model with PSNR {mean_psnr:.3f}")

                # Clean up
                del psnr_vals, ssim_vals, lpips_vals, mean_psnr, mean_ssim, mean_lpips
                sys.stdout.flush()
                torch.cuda.empty_cache()

    ############################################################################################
    # For the final evaluation at the end of training
    print("[*] Training Completed.")
    final_model_path = os.path.join(checkpoint_dir, f"final_gaussians_{itr}.pth")
    gaussians.save_gaussians(final_model_path)

    if log_debug:
        sys.stdout.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--dataset_name", default="colmap_chair", type=str,
        help="Name of the dataset to use (folder name under data_path). For example, \
        'materials' and 'flower' for nerf dataset, 'chicken' for COLMAP customized dataset"
    )
    parser.add_argument(
        "--dataset_type", default="colmap", type=str,
        choices=[ "colmap", "nerf_llff_data", "tandt", "mip360"],
        help="Type of dataset to load: 'nerf_llff_data' for original NeRF datasets, \
        'colmap' for COLMAP nerf_synthetic datasets, 'tandt' for Tanks and Temples datasets, or 'mip360' for Mip-NeRF 360 datasets."
    )
    parser.add_argument(
        "--load_point", action="store_true",
        help="Whether to load point cloud data. Set this flag to enable loading."
    )
    parser.add_argument(
        "--load_point_path", default=None, type=str,
        help="Path to the point cloud loading if not point3D.txt"
    )
    parser.add_argument(
        "--generate_dense_point", default=0, type=int,
        help="N times the point cloud to densify. Set the number to enable N multiply."
    )
    parser.add_argument(
        "--densepoint_scatter", default=0.01, type=float,
        help="scatter densepoint ratio for _load_point"
    )
    parser.add_argument(
        "--lr", default=0.01, type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--num_itrs", default=20000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=400, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument(
        "--eval_freq", default=5000, type=int,
        help="Frequency with evaluation process."
    )
    parser.add_argument(
        "--load_checkpoint",
        # default="/hy-tmp/echoRealm/3DGS_pytorch/result/checkpoints/best_gaussians_12500_lego_576_densify1.pth",
        default = None,
        # default = "/hy-tmp/echoRealm/3DGS_pytorch/result/checkpoints/best_gaussians_10000_chair1.pth",
        type=str,
        help="Path to a .pth file to load pre-trained Gaussians from."
    )
    parser.add_argument(
        "--train_mode", default="combined", type=str, choices=["default", "combined"],
        help="Training mode: 'default' (train/val/test) or 'combined' ((train+val)/test). \
            (only used for nerf_synthetic datasets)."
    )
    parser.add_argument(
        "--split_ratio", default=1.0, type=float,
        help="split_ratio for depth in  multiplane loss "
    )
    parser.add_argument(
        "--img_size", nargs=2, default=[800, 800], type=int,
        help="resolution"
    )
    parser.add_argument(
        "--densify_every", default=-1, type=int,
        help="desify the gaussians for model.py"
    )
    parser.add_argument(
        "--start_densify_step", default=3000, type=int,
        help="start_densify_step for train.py"
    )
    parser.add_argument(
        "--end_densify_step", default=15000, type=int,
        help="end_densify_step for train.py"
    )
    parser.add_argument(
        "--extra_scale", default=1, type=float,
        help="extra_scale for dataloader"
    )
    parser.add_argument(
        "--is_outdoor", action="store_true",
        help="is_outdoor for point cloud loading"
    )
    parser.add_argument(
        "--grad_threshold", default=0.0005, type=float,
        help="grad_threshold for density_control in model.py. \
        1.0 keeps original density, >1.0 makes points more spread out."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args_prop = Namespace(
        densify_every = args.densify_every,
        start_densify_step = args.start_densify_step,
        end_densify_step = args.end_densify_step,
        wavelengths=[639e-9, 532e-9, 473e-9],
        pixel_pitch=3.74e-6,
        volume_depth=4e-3,
        # volume_depth=0,
        d_val=-2e-3,
        pad_size=[max(args.img_size[0], args.img_size[1])] * 2,
        aperture_size = max(args.img_size[0], args.img_size[1]),
        num_planes = 2,
        split_ratio = args.split_ratio
    )
    args_prop.is_outdoor = args.is_outdoor
    assert (args_prop.volume_depth == 0 and args_prop.num_planes == 1) or (args_prop.volume_depth > 0 and args_prop.num_planes > 1), \
        "Invalid depth configuration: If volume_depth is 0, num_planes must be 1; \
        if volume_depth > 0, num_planes must be greater than 1."

    if args_prop.num_planes > 1:
        args_prop.distances = torch.linspace(
            -args_prop.volume_depth / 2.,
            args_prop.volume_depth / 2.,
            args_prop.num_planes
        ) + args_prop.d_val
    else:
        args_prop.distances = [args_prop.d_val]

    print("distance: ", args_prop.distances)
    propagator = propagator(
        resolution = args_prop.pad_size,
        wavelengths = args_prop.wavelengths,
        pixel_pitch = args_prop.pixel_pitch,
        number_of_frames = 3,
        number_of_depth_layers = args_prop.num_planes,
        volume_depth = args_prop.volume_depth,
        image_location_offset = args_prop.d_val,
        propagation_type = 'Bandlimited Angular Spectrum',
        propagator_type = 'forward',
        laser_channel_power = torch.eye(3),
        aperture_size = args_prop.aperture_size,
        aperture = None,
        method = 'conventional',
        device = "cuda"
    )
    print("args.load_point: ", args.load_point)

    run_training(args, args_prop)
