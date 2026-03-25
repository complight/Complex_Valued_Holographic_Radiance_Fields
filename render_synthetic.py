import argparse
import glob
import os
import shlex
import shutil
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import cv2
import imageio
import imageio.v3 as iio
import lpips
import numpy as np
import odak
import pytorch3d
import torch
import torch.nn.functional as F
from model import Gaussians, Scene
from pytorch3d.renderer.cameras import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from utils import (
    analyze_amplitude_statistics,
    analyze_complex_field_spectrum,
    analyze_gaussian_distributions,
    analyze_phase_statistics,
    get_colmap_datasets,
    get_tandt_datasets,
    multiplane_loss,
    ndc_to_screen_camera,
    propagator,
    trivial_collate,
)


def _to_numpy_image(tensor, normalize=True, global_min=None, global_max=None):
    """Convert a tensor (GPU or CPU) to a uint8 numpy image.

    Handles optional global normalization and CHW->HWC transposition.
    """
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        tensor = tensor.detach().cpu()

    if normalize:
        lo = global_min if global_min is not None else tensor.min().item()
        hi = global_max if global_max is not None else tensor.max().item()
        img_np = ((tensor - lo) / max(hi - lo, 1e-12)).numpy() * 255
    else:
        img_np = (
            tensor.numpy() * 255 if isinstance(tensor, torch.Tensor) else tensor * 255
        )

    # CHW -> HWC when channel dim is smallest and leading
    if (
        len(img_np.shape) == 3
        and img_np.shape[0] < img_np.shape[1]
        and img_np.shape[0] < img_np.shape[2]
    ):
        img_np = np.transpose(img_np, (1, 2, 0))

    return img_np.astype(np.uint8)


def numpy_to_bgr(img_np):
    """Convert a numpy image (grayscale or RGB) to BGR for cv2.VideoWriter."""
    if len(img_np.shape) == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 3:
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np


def create_video_writer(path, fps, width, height):
    """Create a cv2.VideoWriter with the given parameters."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def ffmpeg_reencode(input_path, output_path, quality="superhigh"):
    """Re-encode a video with ffmpeg for target bitrate, replacing in-place if needed."""
    bitrate = {
        "superhigh": 30000000,
        "high": 20000000,
        "medium": 10000000,
        "low": 5000000,
    }.get(quality, 10000000)

    if input_path == output_path:
        tmp = input_path + "_tmp.mp4"
        os.rename(input_path, tmp)
        input_path = tmp

    cmd = (
        f"ffmpeg -y -i {shlex.quote(input_path)} "
        f"-b:v {bitrate} -maxrate {bitrate} -bufsize {bitrate // 2} "
        f"{shlex.quote(output_path)}"
    )
    os.system(cmd)

    if input_path.endswith("_tmp.mp4") and os.path.exists(input_path):
        os.remove(input_path)


def images_to_video(image_pattern, output_path, fps=30, quality="medium"):
    """
    Convert a sequence of images matching a glob pattern to an MP4 video.

    Args:
        image_pattern (str): Glob pattern to match images (e.g., "folder/image*.png")
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        quality (str): Video quality - 'superhigh', 'high', 'medium', or 'low'

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        images = sorted(glob.glob(image_pattern))
        if not images:
            print(f"No images found matching pattern: {image_pattern}")
            return False

        frame = cv2.imread(images[0])
        if frame is None:
            print(f"Failed to read first image: {images[0]}")
            return False

        height, width, layers = frame.shape
        temp_output = output_path + "_temp.mp4"
        video = create_video_writer(temp_output, fps, width, height)

        total_images = len(images)
        for i, image_path in enumerate(images, 1):
            frame = cv2.imread(image_path)
            if frame is not None:
                video.write(frame)
                print(
                    f"Processing image {i}/{total_images}: {os.path.basename(image_path)}",
                    end="\r",
                )
            else:
                print(f"\nWarning: Could not read image {image_path}")

        video.release()
        ffmpeg_reencode(temp_output, output_path, quality=quality)
        if os.path.exists(temp_output):
            os.remove(temp_output)

        print(f"\nVideo successfully created at: {output_path}")
        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False


def parse_train_script(script_path):
    """Parse a training shell script to extract train.py arguments.

    This ensures render_synthetic.py uses the same dataset and propagation
    parameters that were used during training, so reconstructions are
    focused correctly.

    Args:
        script_path (str): Path to the training .sh file (e.g., script/train_colmap_chair.sh)

    Returns:
        dict: Parsed argument name -> value mapping
    """
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Training script not found: {script_path}")

    with open(script_path, "r") as f:
        content = f.read()

    # Remove line continuations and join into single command
    content = content.replace("\\\n", " ")

    # Find the line containing train.py
    cmd_line = None
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line.startswith("#") and "train.py" in line:
            cmd_line = line
            break

    if cmd_line is None:
        raise ValueError(f"Could not find 'train.py' command in {script_path}")

    tokens = shlex.split(cmd_line)

    # Find the index of 'train.py' (could be 'python train.py' or a path)
    train_idx = next(
        (i for i, t in enumerate(tokens) if t == "train.py" or t.endswith("/train.py")),
        None,
    )
    if train_idx is None:
        raise ValueError(f"Could not locate 'train.py' token in: {cmd_line}")

    # Parse tokens after train.py into a dict
    args_dict = {}
    i = train_idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:]
            values = []
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("--"):
                values.append(tokens[j])
                j += 1
            if len(values) == 0:
                args_dict[key] = True  # flag argument (e.g. --load_point)
            elif len(values) == 1:
                args_dict[key] = values[0]
            else:
                args_dict[key] = values  # multi-value (e.g. --img_size 960 640)
            i = j
        else:
            i += 1

    return args_dict


def apply_train_script_args(args, train_args):
    """Override render args with values parsed from the training script.

    Only overrides parameters that must align between training and rendering
    for correct reconstruction focus.

    Args:
        args: argparse Namespace from render_synthetic.py
        train_args: dict returned by parse_train_script()
    """
    # Parameters that MUST align between training and rendering
    alignment_keys = {
        "dataset_name": str,
        "dataset_type": str,
        "split_ratio": float,
        "extra_scale": float,
    }

    applied = []
    for key, cast_fn in alignment_keys.items():
        if key in train_args:
            value = cast_fn(train_args[key])
            setattr(args, key, value)
            applied.append(f"  {key} = {value}")

    # img_size needs special handling (multi-value)
    if "img_size" in train_args:
        raw = train_args["img_size"]
        args.img_size = (
            [int(x) for x in raw] if isinstance(raw, list) else [int(raw), int(raw)]
        )
        applied.append(f"  img_size = {args.img_size}")

    print(f"[*] Applied parameters from training script ({args.train_script}):")
    for line in applied:
        print(line)

    return args


def _global_minmax(tensor_list):
    """Return (min, max) across a list of CPU tensors."""
    return (
        min(t.min().item() for t in tensor_list),
        max(t.max().item() for t in tensor_list),
    )


def _make_gif(path, tensor_list, duration, global_min, global_max):
    """Convert a list of CPU tensors to a GIF with consistent normalization."""
    imgs = np.stack(
        [
            _to_numpy_image(t, global_min=global_min, global_max=global_max)
            for t in tensor_list
        ]
    )
    imageio.mimwrite(path, imgs, duration=duration, loop=0)
    return imgs


def render_dataset_views(args, args_prop):
    """Render views from validation and test datasets"""
    W, H = args.img_size[0], args.img_size[1]
    img_size = [W, H]
    num_planes = args_prop.num_planes

    # Initialize LPIPS metric
    lpips_fn = lpips.LPIPS(net="vgg").cuda()

    # Create output directories
    base_dir = os.path.join(args.out_path, f"{args.dataset_name}_renders")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create plane-specific directories if multi-plane and saving frames
    if num_planes > 1 and args.save_individual_frames:
        for d in [val_dir, test_dir]:
            for i in range(num_planes):
                os.makedirs(os.path.join(d, f"plane_{i+1}"), exist_ok=True)

    # Load trained gaussians
    print(f"Loading Gaussians from {args.load_checkpoint}")
    gaussians = Gaussians(
        init_type="gaussians",
        device=args.device,
        load_path=args.load_checkpoint,
        args_prop=args_prop,
    )

    # Create scene
    scene = Scene(gaussians, args_prop)

    # Get datasets based on dataset type
    print(f"Loading {args.dataset_name} datasets...")
    if args.dataset_type in ("colmap", "mip360", "nerf_llff_data"):
        train_dataset, val_dataset, test_dataset, _ = get_colmap_datasets(
            dataset_type=args.dataset_type,
            dataset_name=args.dataset_name,
            image_size=img_size,
            args=args,
            device=args.device,
        )
    elif args.dataset_type == "tandt":
        train_dataset, val_dataset, test_dataset, _ = get_tandt_datasets(
            dataset_name=args.dataset_name,
            image_size=img_size,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Open metrics file
    if args.report_metrics:
        metrics_file = open(
            os.path.join(args.out_path, f"eval_metrics_{args.dataset_name}.txt"), "w"
        )
        metrics_file.write(f"Evaluation metrics for {args.dataset_name}\n")
        metrics_file.write(f"Model checkpoint: {args.load_checkpoint}\n")
        metrics_file.write("-" * 80 + "\n")

    io_executor = ThreadPoolExecutor(max_workers=4)
    io_futures = []

    def _write_png(path, img_np):
        """Write a PNG in a background thread."""
        imageio.imwrite(path, img_np)

    def _flush_io(force_all=False):
        """Drain pending I/O futures to bound memory."""
        nonlocal io_futures
        if force_all:
            for f in io_futures:
                f.result()
            io_futures.clear()
        elif len(io_futures) > 64:
            for f in io_futures[:32]:
                f.result()
            io_futures = io_futures[32:]

    def _submit_png(path, img_np):
        io_futures.append(io_executor.submit(_write_png, path, img_np))

    # Function to render and save images for a dataset
    def process_dataset(dataset, output_dir, prefix):
        if dataset is None or len(dataset) == 0:
            print(f"No {prefix} dataset available")
            return None

        print(f"Processing {prefix} dataset with {len(dataset)} views...")

        # Check if dataset has ground truth images
        has_gt_images = len(dataset) > 0 and "image" in dataset[0]

        # For LLFF, we may have cameras but no ground truth images
        if args.dataset_type == "nerf_llff_data" and not has_gt_images:
            print(f"LLFF dataset detected with cameras but no ground truth images")

        # Precompute output naming
        base_name = os.path.splitext(os.path.basename(args.load_checkpoint))[0]
        dataset_prefix = f"{args.dataset_name}_{prefix}_{base_name}"
        duration = 1000.0 / args.fps

        want_gif = args.output_format in ("gif", "both")
        want_mp4 = args.output_format in ("mp4", "both")

        def _plane_suffix(idx):
            return f"_plane{idx+1}" if num_planes > 1 else ""

        # --- Set up direct VideoWriters for MP4 (no PNG intermediary) ---
        video_writers = {}
        if want_mp4:
            for pi in range(num_planes):
                path = os.path.join(
                    args.out_path,
                    f"{dataset_prefix}{_plane_suffix(pi)}_reconstruction_raw.mp4",
                )
                video_writers[pi] = create_video_writer(path, args.fps, W, H)

        # --- Lists for GIF (need global normalization, so we store CPU tensors) ---
        if want_gif:
            recon_lists = [[] for _ in range(num_planes)]
            raw_phase_lists = [[] for _ in range(num_planes)]
            raw_amp_lists = [[] for _ in range(num_planes)]
            phase_list, amp_list = [], []

        # Lists to store metrics if ground truth is available - now per plane
        if has_gt_images and args.report_metrics:
            psnr_vals = [[] for _ in range(num_planes)]
            ssim_vals = [[] for _ in range(num_planes)]
            lpips_vals = [[] for _ in range(num_planes)]

            metrics_file.write(f"\n{prefix.upper()} Dataset Metrics:\n")
            # Create header with plane information
            header = "View ID"
            for pi in range(num_planes):
                header += f" | Plane{pi+1}_PSNR | Plane{pi+1}_SSIM | Plane{pi+1}_LPIPS"
            header += " | Avg_PSNR | Avg_SSIM | Avg_LPIPS\n"
            metrics_file.write(header)
            metrics_file.write("-" * len(header) + "\n")

        # Process each frame
        # for i in [0,5,10,15, 120]:
        for i in tqdm(range(len(dataset)), desc=f"Rendering {prefix} views"):
            # if i != 1:
            #     break
            # Get camera from dataset
            camera = dataset[i]["camera"]

            # Convert to screen space if in NDC
            if hasattr(camera, "in_ndc") and camera.in_ndc():
                camera = ndc_to_screen_camera(camera, img_size)

            # Move camera to correct device
            camera = camera.to(args.device)

            # Process ground truth targets if available
            targets = None
            if has_gt_images and args.report_metrics:
                gt_img = (
                    dataset[i]["image"].cuda().permute(2, 0, 1)
                )  # (H, W, C) -> (C, H, W)
                has_depth = "depth" in dataset[i]

                if has_depth:
                    depth = (
                        dataset[i]["depth"].cuda().permute(2, 0, 1)
                    )  # (H, W, 1) -> (1, H, W)
                    targets, _, _ = multiplane_loss(
                        target_image=gt_img, target_depth=depth, args_prop=args_prop
                    )
                else:
                    # For single plane, use the original image as target
                    targets = [gt_img]

            with torch.no_grad():
                # ============================================================
                # PHASE 1: All GPU computation — no CPU transfers yet
                # ============================================================

                # Render the scene
                hologram_complex, plane_fields = scene.render(
                    camera, img_size=img_size, bg_colour=(0.0, 0.0, 0.0)
                )

                # Calculate phase and amplitude
                phase_map = odak.learn.wave.calculate_phase(hologram_complex) % (
                    2 * odak.pi
                )
                amplitude = odak.learn.wave.calculate_amplitude(hologram_complex) ** 2
                if args.report_distribution:
                    print(
                        "[OPTIONAL] Analyzing phase and gaussian statistics... This is optional and takes a lot of time per step!"
                    )
                    analysis_dir = os.path.join(output_dir, "analysis")
                    os.makedirs(analysis_dir, exist_ok=True)
                    analyze_phase_statistics(phase_map, i, analysis_dir)
                    analyze_amplitude_statistics(amplitude, i, analysis_dir)
                    analyze_complex_field_spectrum(
                        amplitude, phase_map, i, analysis_dir
                    )
                    analyze_gaussian_distributions(
                        gaussians, args_prop, i, analysis_dir
                    )

                # Reconstruct image
                reconstruction_intensities_sum = propagator.reconstruct(
                    phase_map, amplitude=amplitude, no_grad=False
                )
                reconstruction_intensities = torch.sum(
                    reconstruction_intensities_sum, dim=0
                )

                # Process hologram phase and amplitude
                phase_cropped = odak.learn.tools.crop_center(
                    phase_map, size=(H, W)
                ).squeeze(0)
                amp_cropped = odak.learn.tools.crop_center(
                    amplitude, size=(H, W)
                ).squeeze(0)

                # Prepare all per-plane results on GPU before any CPU transfer
                plane_results_gpu = []
                for pi in range(num_planes):
                    recon_i = torch.clamp(
                        reconstruction_intensities[pi], min=0.0, max=1.0
                    )
                    recon_i = odak.learn.tools.crop_center(recon_i, size=(H, W))
                    raw_ph = odak.learn.wave.calculate_phase(plane_fields[pi]) % (
                        2 * odak.pi
                    )
                    raw_am = odak.learn.wave.calculate_amplitude(plane_fields[pi]) ** 2
                    plane_results_gpu.append((recon_i, raw_ph, raw_am))

                if has_gt_images and targets is not None and args.report_metrics:
                    current_frame_metrics = []
                    for pi in range(num_planes):
                        plane_recon = plane_results_gpu[pi][0]
                        target = targets[pi]

                        plane_recon_cpu = plane_recon.detach().cpu().numpy()
                        target_cpu = target.detach().cpu().numpy()

                        psnr = peak_signal_noise_ratio(
                            target_cpu, plane_recon_cpu, data_range=1.0
                        )
                        ssim = structural_similarity(
                            target_cpu, plane_recon_cpu, channel_axis=0, data_range=1.0
                        )

                        # LPIPS expects inputs in [-1, 1] range
                        lpips_val = lpips_fn(
                            (2 * plane_recon - 1).unsqueeze(0),
                            (2 * target - 1).unsqueeze(0),
                        ).item()

                        psnr_vals[pi].append(psnr)
                        ssim_vals[pi].append(ssim)
                        lpips_vals[pi].append(lpips_val)
                        current_frame_metrics.append((psnr, ssim, lpips_val))
                        del target

                    avg_m = [
                        np.mean([m[k] for m in current_frame_metrics]) for k in range(3)
                    ]
                    metrics_line = f"{i:7d}"
                    for pm in current_frame_metrics:
                        metrics_line += f" | {pm[0]:6.3f} | {pm[1]:6.3f} | {pm[2]:6.3f}"
                    metrics_line += (
                        f" | {avg_m[0]:6.3f} | {avg_m[1]:6.3f} | {avg_m[2]:6.3f}\n"
                    )
                    metrics_file.write(metrics_line)
                    metrics_file.flush()

                torch.cuda.synchronize()

                phase_cpu = phase_cropped.cpu()
                amp_cpu = amp_cropped.cpu()
                plane_cpu_results = [
                    (rg.cpu(), rpg.cpu(), rag.cpu())
                    for rg, rpg, rag in plane_results_gpu
                ]

                # Free GPU memory immediately after transfer
                del hologram_complex, plane_fields, phase_map, amplitude
                del reconstruction_intensities_sum, reconstruction_intensities
                del plane_results_gpu, phase_cropped, amp_cropped

                # Store for GIF (CPU tensors, no conversion needed yet)
                if want_gif:
                    phase_list.append(phase_cpu)
                    amp_list.append(amp_cpu)

                # Save hologram PNGs async
                if args.save_individual_frames:
                    _submit_png(
                        os.path.join(output_dir, f"phase_{i:03d}.png"),
                        _to_numpy_image(phase_cpu),
                    )
                    _submit_png(
                        os.path.join(output_dir, f"amp_{i:03d}.png"),
                        _to_numpy_image(amp_cpu),
                    )

                for pi in range(num_planes):
                    recon_cpu, raw_phase_cpu, raw_amp_cpu = plane_cpu_results[pi]

                    # Store for GIF
                    if want_gif:
                        recon_lists[pi].append(recon_cpu)
                        raw_phase_lists[pi].append(raw_phase_cpu)
                        raw_amp_lists[pi].append(raw_amp_cpu)

                    # Convert once, reuse for both MP4 and PNG
                    recon_np = _to_numpy_image(recon_cpu)

                    # Write to MP4 VideoWriter (sequential, but fast — no PNG compression)
                    if want_mp4:
                        video_writers[pi].write(numpy_to_bgr(recon_np))

                    # Save individual frame PNGs async
                    if args.save_individual_frames:
                        plane_dir = (
                            os.path.join(output_dir, f"plane_{pi+1}")
                            if num_planes > 1
                            else output_dir
                        )
                        _submit_png(
                            os.path.join(plane_dir, f"recon_{i:03d}_plane{pi+1}.png"),
                            recon_np.copy(),
                        )
                        _submit_png(
                            os.path.join(
                                plane_dir, f"raw_phase_{i:03d}_plane{pi+1}.png"
                            ),
                            _to_numpy_image(raw_phase_cpu),
                        )
                        _submit_png(
                            os.path.join(plane_dir, f"raw_amp_{i:03d}_plane{pi+1}.png"),
                            _to_numpy_image(raw_amp_cpu),
                        )

                # Throttle: if too many pending I/O tasks, wait for oldest to finish
                # This prevents unbounded memory growth from queued numpy arrays
                _flush_io()

        # Wait for all remaining I/O to complete
        _flush_io(force_all=True)

        # --- Finalize MP4: release writers, then re-encode for quality ---
        if want_mp4:
            print(f"Finalizing {prefix} MP4 videos...")
            for writer in video_writers.values():
                writer.release()

            for pi in range(num_planes):
                ps = _plane_suffix(pi)
                raw_path = os.path.join(
                    args.out_path, f"{dataset_prefix}{ps}_reconstruction_raw.mp4"
                )
                final_path = os.path.join(
                    args.out_path, f"{dataset_prefix}{ps}_reconstruction.mp4"
                )
                ffmpeg_reencode(raw_path, final_path, quality=args.video_quality)
                if os.path.exists(raw_path) and raw_path != final_path:
                    os.remove(raw_path)

            print(f"{prefix} MP4 videos saved successfully")

        # Write average metrics if ground truth was available
        if has_gt_images and args.report_metrics:
            metrics_file.write("-" * (15 * (num_planes * 3 + 3) + 10) + "\n")

            # Per-plane averages
            plane_avgs = [
                (
                    np.mean(psnr_vals[pi]),
                    np.mean(ssim_vals[pi]),
                    np.mean(lpips_vals[pi]),
                )
                for pi in range(num_planes)
            ]

            avg_line = "Average"
            for pa in plane_avgs:
                avg_line += f" | {pa[0]:6.3f} | {pa[1]:6.3f} | {pa[2]:6.3f}"
            overall = tuple(np.mean([pa[k] for pa in plane_avgs]) for k in range(3))
            avg_line += (
                f" | {overall[0]:6.3f} | {overall[1]:6.3f} | {overall[2]:6.3f}\n"
            )
            metrics_file.write(avg_line)

            # Standard deviations
            std_line = "Std Dev"
            for pi in range(num_planes):
                std_line += f" | {np.std(psnr_vals[pi]):6.3f} | {np.std(ssim_vals[pi]):6.3f} | {np.std(lpips_vals[pi]):6.3f}"
            all_psnr = [v for pv in psnr_vals for v in pv]
            all_ssim = [v for sv in ssim_vals for v in sv]
            all_lpips = [v for lv in lpips_vals for v in lv]
            std_line += f" | {np.std(all_psnr):6.3f} | {np.std(all_ssim):6.3f} | {np.std(all_lpips):6.3f}\n"
            metrics_file.write(std_line)
            metrics_file.flush()

            print(f"\n{prefix} Metrics Summary:")
            for pi in range(num_planes):
                print(
                    f"Plane {pi+1} - Average PSNR: {plane_avgs[pi][0]:.3f}, SSIM: {plane_avgs[pi][1]:.3f}, LPIPS: {plane_avgs[pi][2]:.3f}"
                )
            print(
                f"Overall Average - PSNR: {overall[0]:.3f}, SSIM: {overall[1]:.3f}, LPIPS: {overall[2]:.3f}"
            )

        # Create GIFs if requested
        if want_gif and len(recon_lists[0]) > 0:
            print(f"Creating {prefix} GIFs...")

            # Save hologram GIFs with global normalization
            ph_lo, ph_hi = _global_minmax(phase_list)
            am_lo, am_hi = _global_minmax(amp_list)

            print("Creating phase GIF...")
            _make_gif(
                os.path.join(args.out_path, f"{dataset_prefix}_phase.gif"),
                phase_list,
                duration,
                ph_lo,
                ph_hi,
            )
            print("Creating amplitude GIF...")
            _make_gif(
                os.path.join(args.out_path, f"{dataset_prefix}_amplitude.gif"),
                amp_list,
                duration,
                am_lo,
                am_hi,
            )

            # Save GIFs for each plane
            for pi in range(num_planes):
                print(f"Creating GIFs for plane {pi+1}...")
                ps = _plane_suffix(pi)

                if len(recon_lists[pi]) == 0:
                    print(f"No data for plane {pi+1}, skipping")
                    continue

                for label, tlist in [
                    ("reconstruction", recon_lists[pi]),
                    ("raw_phase", raw_phase_lists[pi]),
                    ("raw_amplitude", raw_amp_lists[pi]),
                ]:
                    lo, hi = _global_minmax(tlist)
                    print(f"  Saving {label} GIF for plane {pi+1}")
                    _make_gif(
                        os.path.join(
                            args.out_path, f"{dataset_prefix}{ps}_{label}.gif"
                        ),
                        tlist,
                        duration,
                        lo,
                        hi,
                    )

            # Final cleanup
            del recon_lists, raw_phase_lists, raw_amp_lists
            print(f"{prefix} GIFs saved successfully")

    val_renders = process_dataset(val_dataset, val_dir, "val")
    # test_renders = process_dataset(test_dataset, test_dir, "test")

    if args.report_metrics:
        # Close metrics file
        metrics_file.close()

    # Shut down the I/O thread pool
    io_executor.shutdown(wait=True)

    torch.cuda.empty_cache()
    print("Rendering complete!")
    print(f"Metrics saved to: {os.path.join(args.out_path)}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_script",
        required=True,
        type=str,
        help="Path to the training .sh script used for this model "
        "(e.g., script/train_colmap_chair.sh). "
        "Parameters such as dataset_name, dataset_type, split_ratio, "
        "extra_scale, and img_size are automatically extracted from this "
        "script to ensure the rendering propagator is aligned with training.",
    )
    parser.add_argument(
        "--out_path",
        default="./output",
        type=str,
        help="Path to the directory where output should be saved to.",
    )
    parser.add_argument(
        "--dataset_name",
        default="colmap_chair",
        type=str,
        help="Name of the dataset to use for camera parameters. "
        "(Overridden by --train_script if provided there.)",
    )
    parser.add_argument(
        "--load_point_path",
        default=None,
        type=str,
        help="Path to the point cloud loading if not point3D.txt",
    )
    parser.add_argument(
        "--dataset_type",
        default="colmap",
        type=str,
        choices=["colmap", "nerf_llff_data", "mip360", "tandt"],
        help="Type of dataset to load: 'nerf_llff_data' for original NeRF datasets, "
        "'colmap' for COLMAP nerf_synthetic datasets, 'mip360' for Mip-NeRF 360 datasets, or 'tandt' for Tanks and Temples. "
        "(Overridden by --train_script if provided there.)",
    )
    parser.add_argument(
        "--train_mode",
        default="default",
        type=str,
        choices=["default", "combined"],
        help="Training mode: 'default' (train/val/test) or 'combined' ((train+val)/test).",
    )
    parser.add_argument(
        "--extra_scale",
        default=550,
        type=float,
        help="extra_scale for dataloader. "
        "(Overridden by --train_script if provided there.)",
    )
    parser.add_argument(
        "--split_ratio",
        default=2.2,
        type=float,
        help="split_ratio for depth in multiplane loss. "
        "(Overridden by --train_script if provided there.)",
    )
    parser.add_argument(
        "--img_size",
        nargs=2,
        default=[800, 800],
        type=int,
        help="resolution. " "(Overridden by --train_script if provided there.)",
    )
    parser.add_argument(
        "--fps",
        default=10,
        type=int,
        help="Frames per second for the output GIFs and videos.",
    )
    parser.add_argument(
        "--load_checkpoint",
        # default="/hy-tmp/echoRealm/3DGS_pytorch/result/checkpoints/final_gaussians_19999_lego_good.pth",
        default="/hy-tmp/echoRealm/3DGS_pytorch/best_gaussians_10000_chair800.pth",
        type=str,
        help="Path to a .pth file to load pre-trained Gaussians from.",
    )
    parser.add_argument(
        "--save_individual_frames",
        default=True,
        help="Save individual frame images permanently.",
    )
    parser.add_argument(
        "--report_metrics", action="store_true", help="Record PSNR/SSIM/LPIPS."
    )
    parser.add_argument(
        "--report_distribution",
        action="store_true",
        help="Record phase and gaussian distribution.",
    )
    parser.add_argument(
        "--output_format",
        default="mp4",
        type=str,
        choices=["gif", "mp4", "both"],
        help="Output format for rendered animations: 'gif', 'mp4', or 'both'.",
    )
    parser.add_argument(
        "--video_quality",
        default="superhigh",
        type=str,
        choices=["low", "medium", "high", "superhigh"],
        help="Quality of MP4 video output: 'low' (5Mbps), 'medium' (10Mbps), 'high' (20Mbps), or 'superhigh' (30Mbps).",
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Parse training script and apply aligned parameters
    train_args = parse_train_script(args.train_script)
    args = apply_train_script_args(args, train_args)

    return args


if __name__ == "__main__":
    args = get_args()
    args_prop = Namespace(
        wavelengths=[639e-9, 532e-9, 473e-9],
        pixel_pitch=3.74e-6,
        volume_depth=4e-3,
        d_val=-2e-3,
        pad_size=[max(args.img_size[0], args.img_size[1])] * 2,
        aperture_size=max(args.img_size[0], args.img_size[1]),
        num_planes=2,
        split_ratio=args.split_ratio,
    )

    # Calculate distance values based on volume depth
    if args_prop.num_planes > 1:
        args_prop.distances = (
            torch.linspace(
                -args_prop.volume_depth / 2.0,
                args_prop.volume_depth / 2.0,
                args_prop.num_planes,
            )
            + args_prop.d_val
        )
    else:
        args_prop.distances = [args_prop.d_val]

    print("distance: ", args_prop.distances)
    print(f"Number of planes: {args_prop.num_planes}")

    device = args.device

    # Initialize propagator with support for multiple planes
    propagator = propagator(
        resolution=args_prop.pad_size,
        wavelengths=args_prop.wavelengths,
        pixel_pitch=args_prop.pixel_pitch,
        number_of_frames=3,
        number_of_depth_layers=args_prop.num_planes,
        volume_depth=args_prop.volume_depth,
        image_location_offset=args_prop.d_val,
        propagation_type="Bandlimited Angular Spectrum",
        propagator_type="forward",
        laser_channel_power=torch.eye(3),
        aperture_size=args_prop.aperture_size,
        aperture=None,
        method="conventional",
        device=device,
    )

    # Run rendering
    render_dataset_views(args, args_prop)
