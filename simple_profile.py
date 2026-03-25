"""
GPU-ONLY Averaged Per-Step Profiler (Gaussian Render + Propagator Reconstruct)

What you get:
  1) Averaged GPU "icicle-style" trace PNG (stitched timeline; GPU durations; professional layout)
  2) Averaged GPU breakdown PER STEP (GAUSSIAN_RENDER / PROPAGATOR_RECONSTRUCT):
       - avg_gpu_breakdown_by_step.csv
       - avg_gpu_breakdown_by_step.txt
     This is computed by:
       - using CPU record_function scopes ONLY as time windows
       - collecting ALL CUDA events overlapping each window
       - aggregating by CUDA event name using SELF time (excluding child operations)
       - averaging across iterations

Usage:
  python simple_profile.py --dataset_name colmap_chair --dataset_type colmap --num_iterations 50

Outputs: ./result/profiling/
  - trace_icicle_avg_gpu.png
  - avg_gpu_breakdown_by_step.csv
  - avg_gpu_breakdown_by_step.txt
  - detailed_events_filtered.csv
  - detailed_events_filtered.txt

IMPORTANT:
  - NO torch.cuda.synchronize() inside the loop
  - NO torch.cuda.empty_cache() per iteration
  - For *higher-level* "SPLATTING"/"FORWARD_RECORDING" in the icicle,
    add nested record_function scopes in Scene.render(...) and reconstruct(...).
  - For kernel-level detail (what you asked): use avg_gpu_breakdown_by_step.* (GPU-only, accurate names).
  - Uses self_device_time to avoid double-counting parent/child operations
"""

import os
import csv
import argparse
from argparse import Namespace
from collections import defaultdict

import torch
import torch.profiler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from train import get_colmap_datasets, get_tandt_datasets, make_trainable, ndc_to_screen_camera
from train import multiplane_loss, propagator, GaussianLoss
from model import Gaussians, Scene
from utils import trivial_collate, setup_optimizer, SparseGaussianAdam
from torch.utils.data import DataLoader
import odak


# =========================
#   Filtered key_averages export (reference)
# =========================

def export_profiler_events_filtered(prof, output_path, include_keys=("GAUSSIAN_RENDER", "PROPAGATOR_RECONSTRUCT")):
    """
    Export key_averages() filtered to only rows whose name contains any include_keys substring.
    Note: this is not the per-step kernel breakdown; it's just a convenient reference table.
    """
    events = prof.key_averages()

    event_list = []
    for evt in events:
        name = evt.key
        if not any(k in name for k in include_keys):
            continue

        cuda_time_us = 0
        if hasattr(evt, "device_time_total"):
            cuda_time_us = evt.device_time_total
        elif hasattr(evt, "self_device_time_total"):
            cuda_time_us = evt.self_device_time_total
        elif hasattr(evt, "cuda_time"):
            cuda_time_us = (evt.cuda_time * evt.count) if hasattr(evt, "count") else evt.cuda_time

        cuda_time_avg_us = 0
        if hasattr(evt, "device_time"):
            cuda_time_avg_us = evt.device_time
        elif hasattr(evt, "self_device_time"):
            cuda_time_avg_us = evt.self_device_time
        elif hasattr(evt, "cuda_time"):
            cuda_time_avg_us = evt.cuda_time

        event_list.append({
            "name": name,
            "cuda_time": cuda_time_us / 1000.0,
            "cuda_time_avg": cuda_time_avg_us / 1000.0,
            "calls": evt.count,
        })

    event_list_sorted = sorted(event_list, key=lambda x: x["cuda_time"], reverse=True)

    csv_path = output_path.replace(".txt", ".csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["name", "calls", "cuda_time_total_ms", "cuda_time_avg_ms"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for evt in event_list_sorted:
            writer.writerow({
                "name": evt["name"],
                "calls": evt["calls"],
                "cuda_time_total_ms": f"{evt['cuda_time']:.4f}",
                "cuda_time_avg_ms": f"{evt['cuda_time_avg']:.4f}",
            })

    with open(output_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("FILTERED PROFILER EVENTS (GPU only; Sorted by CUDA Time)\n")
        f.write(f"Included keys: {include_keys}\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"{'Operation':<85} {'Calls':>8} {'CUDA(ms)':>14} {'Avg CUDA':>14}\n")
        f.write("-" * 120 + "\n")
        for evt in event_list_sorted:
            f.write(f"{evt['name']:<85} {evt['calls']:>8} {evt['cuda_time']:>14.4f} {evt['cuda_time_avg']:>14.4f}\n")

    return csv_path, output_path, event_list_sorted


# =========================
#   GPU-only averaged per-step breakdown (kernel-level)
# =========================

def export_avg_gpu_breakdown_by_step(
    prof,
    output_dir,
    iteration_prefix="ITER_",
    steps=("GAUSSIAN_RENDER", "PROPAGATOR_RECONSTRUCT"),
    top_k=80,
    min_kernel_ms=0.02,
):
    """
    GPU-only averaged breakdown per step.

    For each iteration:
      - find step scope events
      - recursively collect all CUDA kernels in the event tree
      - sum GPU time by CUDA event name using SELF time (excludes children)
    Then average across iterations.

    Writes:
      - avg_gpu_breakdown_by_step.csv
      - avg_gpu_breakdown_by_step.txt
    """

    def _name(e):
        return getattr(e, "name", None) or getattr(e, "key", None) or str(e)

    def _children(e):
        ch = getattr(e, "cpu_children", None)
        if ch is None:
            ch = getattr(e, "children", None)
        return ch or []

    def _start_us(e):
        tr = getattr(e, "time_range", None)
        return float(getattr(tr, "start", 0.0)) if tr is not None else 0.0

    def _cuda_self_ms(e):
        if hasattr(e, "self_device_time_total"):
            return float(e.self_device_time_total) / 1000.0
        if hasattr(e, "self_cuda_time_total"):
            return float(e.self_cuda_time_total) / 1000.0
        return 0.0

    def _is_cuda_kernel(nm):
        """Filter out CPU profiling scopes; keep only actual CUDA kernels/operations."""
        if nm.startswith("ITER_"):
            return False
        if nm.startswith("ProfilerStep"):
            return False
        if nm in ("GAUSSIAN_RENDER", "PROPAGATOR_RECONSTRUCT", "TOP"):
            return False
        if nm.isupper() and len(nm) < 50 and not any(x in nm for x in ["CUDA", "void", "kernel", "Kernel", "cutlass", "aten", "ampere", "Memcpy", "regular_fft", "gemmSN", "BandlimitedPropagationFunction", "STEFunction"]):
            return False
        return True

    def collect_kernels_recursive(evt, kernel_ms):
        """Recursively collect all CUDA kernels under this event using self time."""
        nm = _name(evt)
        ms = _cuda_self_ms(evt)
        
        # If this event has self GPU time and is a kernel, count it
        if ms >= min_kernel_ms and _is_cuda_kernel(nm):
            kernel_ms[nm] += ms
        
        # Recurse into children
        for child in _children(evt):
            collect_kernels_recursive(child, kernel_ms)

    events = prof.events()

    iter_scopes = [e for e in events if _name(e).startswith(iteration_prefix)]
    iter_scopes.sort(key=_start_us)
    if not iter_scopes:
        raise RuntimeError(f"No iteration scopes found. Wrap each iter in record_function('{iteration_prefix}<i>').")

    def find_step_in_iter(iter_evt, step_name):
        """Find the step scope within this iteration by traversing children."""
        stack = [iter_evt]
        while stack:
            evt = stack.pop()
            if _name(evt) == step_name:
                return evt
            stack.extend(_children(evt))
        return None

    per_step_per_iter = {s: [] for s in steps}
    used_iters = 0

    for it in iter_scopes:
        it_used = False
        for step in steps:
            step_evt = find_step_in_iter(it, step)
            if step_evt is None:
                continue

            kernel_ms = defaultdict(float)
            collect_kernels_recursive(step_evt, kernel_ms)

            if kernel_ms:
                per_step_per_iter[step].append(kernel_ms)
                it_used = True

        if it_used:
            used_iters += 1

    if used_iters == 0:
        raise RuntimeError("No iterations contained the requested step scopes. Check your scope names and profiler schedule.")

    out_csv = os.path.join(output_dir, "avg_gpu_breakdown_by_step.csv")
    out_txt = os.path.join(output_dir, "avg_gpu_breakdown_by_step.txt")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "cuda_event_name", "avg_ms_per_iter", "avg_step_total_ms", "percent_of_step"])

        for step in steps:
            sum_ms = defaultdict(float)
            n = max(1, len(per_step_per_iter[step]))
            for d in per_step_per_iter[step]:
                for k, v in d.items():
                    sum_ms[k] += v

            avg_ms = {k: v / n for k, v in sum_ms.items()}
            total_step = sum(avg_ms.values())

            items = sorted(avg_ms.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            for k, v in items:
                pct = (v / total_step * 100.0) if total_step > 0 else 0.0
                w.writerow([step, k, f"{v:.6f}", f"{total_step:.6f}", f"{pct:.3f}"])

    with open(out_txt, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("GPU-ONLY AVERAGED PER-STEP BREAKDOWN (kernel-level, hierarchical traversal)\n")
        f.write(f"Averaged over {used_iters} iterations\n")
        f.write(f"Min kernel event included: {min_kernel_ms} ms\n")
        f.write("Note: Uses hierarchical parent-child traversal with self_device_time to avoid double-counting\n")
        f.write("=" * 120 + "\n\n")

        for step in steps:
            sum_ms = defaultdict(float)
            n = max(1, len(per_step_per_iter[step]))
            for d in per_step_per_iter[step]:
                for k, v in d.items():
                    sum_ms[k] += v
            avg_ms = {k: v / n for k, v in sum_ms.items()}
            total_step = sum(avg_ms.values())

            f.write("-" * 120 + "\n")
            f.write(f"STEP: {step}\n")
            f.write(f"Avg total GPU time in step: {total_step:.6f} ms\n")
            f.write(f"{'CUDA Event Name':<90} {'Avg ms/iter':>14} {'% Step':>10}\n")
            f.write("-" * 120 + "\n")

            for name, ms in sorted(avg_ms.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
                pct = (ms / total_step * 100.0) if total_step > 0 else 0.0
                f.write(f"{name:<90} {ms:>14.6f} {pct:>9.3f}%\n")

    return out_csv, out_txt


# =========================
#   GPU "icicle-style" stitched timeline (GPU durations)
# =========================

def generate_avg_icicle_trace_png_gpu(
    prof,
    output_png_path,
    iteration_prefix="ITER_",
    include_roots=("GAUSSIAN_RENDER", "PROPAGATOR_RECONSTRUCT"),
    min_block_ms=0.05,
):
    """
    Professional averaged icicle (GPU durations; stitched timeline; clean).
    This shows USER scopes only (uppercase or ITER_). It does NOT show every kernel.
    For kernel-level details, use export_avg_gpu_breakdown_by_step(...).
    """

    def _name(e):
        return getattr(e, "name", None) or getattr(e, "key", None) or str(e)

    def _children(e):
        ch = getattr(e, "cpu_children", None)
        if ch is None:
            ch = getattr(e, "children", None)
        return ch or []

    def _cpu_start_us(e):
        tr = getattr(e, "time_range", None)
        return float(getattr(tr, "start", 0.0)) if tr is not None else 0.0

    def _cuda_total_ms(e):
        if hasattr(e, "device_time_total"):
            return float(e.device_time_total) / 1000.0
        if hasattr(e, "cuda_time_total"):
            return float(e.cuda_time_total) / 1000.0
        if hasattr(e, "self_device_time_total"):
            return float(e.self_device_time_total) / 1000.0
        return 0.0

    def _is_user_scope(nm: str) -> bool:
        if nm.startswith("ITER_"):
            return True
        return (nm.upper() == nm) and (len(nm) >= 2)

    class Node:
        __slots__ = ("name", "path", "ms", "children")
        def __init__(self, name, path, ms):
            self.name = name
            self.path = path
            self.ms = ms
            self.children = []

    def build_tree_from_event(root_evt, prefix_path=""):
        nm = _name(root_evt)
        ms = _cuda_total_ms(root_evt)
        path = f"{prefix_path}/{nm}" if prefix_path else nm
        node = Node(nm, path, ms)

        kids = []
        for c in _children(root_evt):
            cnm = _name(c)
            if not _is_user_scope(cnm):
                continue
            cms = _cuda_total_ms(c)
            if cms < min_block_ms:
                continue
            kids.append(c)
        kids.sort(key=_cpu_start_us)

        for c in kids:
            node.children.append(build_tree_from_event(c, path))
        return node

    def find_roots_in_iter(iter_evt, events):
        roots = []
        stack = [iter_evt]
        while stack:
            n = stack.pop()
            nm = _name(n)
            if nm in include_roots:
                roots.append(n)
            stack.extend(_children(n))
        roots.sort(key=_cpu_start_us)
        return roots

    events = prof.events()
    iter_evts = [e for e in events if _name(e).startswith(iteration_prefix)]
    iter_evts.sort(key=_cpu_start_us)
    if not iter_evts:
        raise RuntimeError(f"No iteration scopes found. Wrap each iter in record_function('{iteration_prefix}<i>').")

    per_iter = []
    for it in iter_evts:
        roots = find_roots_in_iter(it, events)
        if not roots:
            continue
        top = Node("TOP", "TOP", 0.0)
        for r in roots:
            top.children.append(build_tree_from_event(r, "TOP"))
        per_iter.append(top)

    if not per_iter:
        raise RuntimeError("No usable iterations contained requested roots (for icicle).")

    acc = defaultdict(list)
    name_map = {}

    def walk(n):
        acc[n.path].append(n.ms)
        name_map[n.path] = n.name
        for c in n.children:
            walk(c)

    for t in per_iter:
        walk(t)

    avg_ms = {p: sum(v) / len(v) for p, v in acc.items()}

    def rebuild(n):
        p = n.path
        new = Node(name_map.get(p, n.name), p, avg_ms.get(p, n.ms))
        for c in n.children:
            if c.path in avg_ms:
                new.children.append(rebuild(c))
        return new

    avg_tree = rebuild(per_iter[0])

    segments = []

    def layout(node, depth, x0):
        w = max(0.0, node.ms)
        segments.append((node.name, node.path, depth, x0, w))

        child_ms = [c.ms for c in node.children if c.ms >= min_block_ms]
        sum_child = sum(child_ms)
        scale = 1.0
        if sum_child > 1e-9 and w > 1e-9 and sum_child > w:
            scale = w / sum_child

        x = x0
        for c in node.children:
            cw = c.ms * scale
            if cw < min_block_ms:
                continue
            layout(c, depth + 1, x)
            x += cw

    x_cursor = 0.0
    for r in avg_tree.children:
        layout(r, 0, x_cursor)
        x_cursor += max(0.0, r.ms)

    if not segments:
        raise RuntimeError("No segments to draw in icicle. Lower min_block_ms or add more user scopes.")

    total_ms = max(x + w for (_, _, _, x, w) in segments)
    max_depth = max(d for (_, _, d, _, _) in segments)

    fig_w = 18
    fig_h = max(6, 1.15 * (max_depth + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.tab20c

    def color_for(path):
        h = 0
        for ch in path:
            h = (h * 131 + ord(ch)) % 104729
        return cmap((h % 256) / 255.0)

    row_h = 0.85
    row_gap = 0.18
    y_top = (max_depth + 1) * (row_h + row_gap)

    for name, path, depth, x, w in sorted(segments, key=lambda t: (t[2], t[3])):
        y = y_top - depth * (row_h + row_gap)

        rect = patches.Rectangle(
            (x, y - row_h),
            w,
            row_h,
            linewidth=1.0,
            edgecolor="white",
            facecolor=color_for(path),
            alpha=0.96,
        )
        ax.add_patch(rect)

        if total_ms > 0 and (w / total_ms) >= 0.04:
            label = name if len(name) <= 44 else (name[:41] + "...")
            ax.text(
                x + 0.5 * w,
                y - 0.5 * row_h,
                f"{label}\n{w:.3f} ms",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="black", alpha=0.18, edgecolor="none"),
            )

    ax.set_xlim(0, total_ms * 1.02)
    ax.set_ylim(0, y_top + 0.9)
    ax.set_xlabel("Time (ms)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Averaged Icicle Trace (GPU, stitched)\nOnly: {', '.join(include_roots)} | Averaged over {len(per_iter)} iterations",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.22, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================
#   Main profiling loop
# =========================

def profile_training(args, args_prop, wave_propagator, num_iterations=50):
    img_size = args.img_size

    print("Loading dataset...")
    if args.dataset_type in ["colmap", "mip360", "nerf_llff_data"]:
        train_dataset, _, _, pointcloud_data = get_colmap_datasets(
            dataset_type=args.dataset_type,
            dataset_name=args.dataset_name,
            image_size=img_size,
            args=args,
            device=args.device
        )
    elif args.dataset_type == "tandt":
        train_dataset, _, _, pointcloud_data = get_tandt_datasets(
            dataset_name=args.dataset_name,
            image_size=img_size,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    print("Creating dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=trivial_collate
    )

    print("Initializing model...")
    if args.load_checkpoint:
        gaussians = Gaussians(init_type="gaussians", device=args.device, load_path=args.load_checkpoint, args_prop=args_prop)
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
            num_points=100000,
            init_type="random",
            device=args.device,
            args_prop=args_prop,
            img_size=img_size
        )

    print("Creating scene and optimizer...")
    scene = Scene(gaussians, args_prop)
    make_trainable(gaussians)
    optimizer, scheduler, parameters = setup_optimizer(gaussians, num_iterations, args.lr)

    output_dir = "./result/profiling"
    os.makedirs(output_dir, exist_ok=True)

    wait_steps = 5
    warmup_steps = 5
    active_steps = num_iterations
    total_steps = wait_steps + warmup_steps + active_steps

    print(f"\nRunning {total_steps} iterations: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}")
    print("COUNT ONLY: GAUSSIAN_RENDER + PROPAGATOR_RECONSTRUCT (GPU-only breakdown will be exported)")
    print("Using hierarchical traversal with SELF device time to avoid double-counting")

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    )

    train_itr = iter(train_loader)

    with prof:
        for itr in range(total_steps):
            with torch.profiler.record_function(f"ITER_{itr}"):
                try:
                    data = next(train_itr)
                except StopIteration:
                    train_itr = iter(train_loader)
                    data = next(train_itr)

                gt_img = data[0]["image"].cuda().permute(2, 0, 1)
                C, H, W = gt_img.size()
                camera = ndc_to_screen_camera(data[0]["camera"], img_size).cuda()
                depth = data[0]["depth"].cuda().permute(2, 0, 1) if "depth" in data[0] else torch.zeros((1, H, W), device="cuda")
                targets, loss_function, _ = multiplane_loss(gt_img, depth, args_prop)

                optimizer.zero_grad(set_to_none=True)

                with torch.profiler.record_function("Complex-valued Splatting + Forward Recording"):
                    hologram_complex, plane_fields = scene.render(
                        camera,
                        img_size=img_size,
                        bg_colour=(0.0, 0.0, 0.0),
                        step=itr,
                        max_step=total_steps
                    )

                phase_map = odak.learn.wave.calculate_phase(hologram_complex) % (2 * odak.pi)
                amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)
                phase_map = phase_map - phase_map.mean()

                with torch.profiler.record_function("Inverse Propagation"):
                    reconstruction_intensities_sum = wave_propagator.reconstruct(
                        phase_map,
                        amplitude=amplitude,
                        no_grad=False
                    )

                reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)

                loss = 0.0
                for idx, (recon, target) in enumerate(zip(reconstruction_intensities, targets)):
                    pred = torch.clamp(odak.learn.tools.crop_center(recon, size=(H, W)), 0.0, 1.0)
                    loss = loss + loss_function(pred, target, idx) + GaussianLoss(pred, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)

                if isinstance(optimizer, SparseGaussianAdam):
                    optimizer.step(scene.visible_indices)
                else:
                    optimizer.step()
                scheduler.step()

                del hologram_complex, plane_fields, phase_map, amplitude, reconstruction_intensities_sum, reconstruction_intensities

            prof.step()

    export_profiler_events_filtered(
        prof,
        f"{output_dir}/detailed_events_filtered.txt",
        include_keys=("GAUSSIAN_RENDER", "PROPAGATOR_RECONSTRUCT")
    )

    icicle_path = f"{output_dir}/trace_icicle_avg_gpu.png"
    generate_avg_icicle_trace_png_gpu(
        prof,
        icicle_path,
        iteration_prefix="ITER_",
        include_roots=("GAUSSIAN_RENDER", "PROPAGATOR_RECONSTRUCT"),
        min_block_ms=0.05
    )
    print(f"\nAveraged GPU icicle trace saved to:\n  - {icicle_path}")

    out_csv, out_txt = export_avg_gpu_breakdown_by_step(
        prof,
        output_dir,
        iteration_prefix="ITER_",
        steps=("GAUSSIAN_RENDER", "PROPAGATOR_RECONSTRUCT"),
        top_k=120,
        min_kernel_ms=0.02
    )
    print(f"\nAveraged GPU per-step breakdown saved to:\n  - {out_csv}\n  - {out_txt}")

    return prof, output_dir


# =========================
#   Entrypoint
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="colmap_chair", type=str)
    parser.add_argument("--dataset_type", default="colmap", type=str)
    parser.add_argument("--img_size", nargs=2, default=[576, 576], type=int)
    parser.add_argument("--load_checkpoint", default=None, type=str)
    parser.add_argument("--load_point", action="store_true", default=True)
    parser.add_argument("--load_point_path", default=None, type=str, help="Path to the point cloud loading if not point3D.txt")
    parser.add_argument("--generate_dense_point", default=3, type=int)
    parser.add_argument("--densepoint_scatter", default=1.0, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--split_ratio", default=2.2, type=float)
    parser.add_argument("--extra_scale", default=550, type=float)
    parser.add_argument("--grad_threshold", default=0.0005, type=float)
    parser.add_argument("--densify_every", default=300, type=int)
    parser.add_argument("--is_outdoor", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_iterations", default=50, type=int)
    args = parser.parse_args()

    densify_every = -1

    print(f"\n{'='*100}")
    print("PROFILING CONFIGURATION")
    print(f"{'='*100}")
    print(f"  Dataset: {args.dataset_name} ({args.dataset_type})")
    print(f"  Image size: {args.img_size}")
    print(f"  Load point cloud: {args.load_point}")
    print(f"  Dense point multiplier: {args.generate_dense_point}")
    print(f"  Dense point scatter: {args.densepoint_scatter}")
    print(f"  Split ratio: {args.split_ratio}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Densification: DISABLED")
    print(f"  Active iterations to profile: {args.num_iterations}")
    print(f"  GPU-ONLY per-step breakdown: ENABLED (hierarchical traversal with SELF time)")
    print(f"{'='*100}\n")

    args_prop = Namespace(
        densify_every=densify_every,
        start_densify_step=3000,
        end_densify_step=15000,
        wavelengths=[639e-9, 532e-9, 473e-9],
        pixel_pitch=3.74e-6,
        volume_depth=4e-3,
        d_val=-2e-3,
        pad_size=[800, 800],
        aperture_size=800,
        num_planes=2,
        split_ratio=args.split_ratio,
        is_outdoor=args.is_outdoor
    )

    if args_prop.num_planes > 1:
        args_prop.distances = torch.linspace(
            -args_prop.volume_depth / 2.0,
            args_prop.volume_depth / 2.0,
            args_prop.num_planes
        ) + args_prop.d_val
    else:
        args_prop.distances = [args_prop.d_val]

    print("Creating propagator...")
    wave_propagator = propagator(
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
        device="cuda"
    )
    print("Propagator created\n")

    prof, output_dir = profile_training(args, args_prop, wave_propagator, args.num_iterations)

    print("\n" + "=" * 100)
    print("PROFILING COMPLETED")
    print("=" * 100)
    print("\nGenerated files:")
    print(f"  1. {output_dir}/trace_icicle_avg_gpu.png")
    print(f"  2. {output_dir}/avg_gpu_breakdown_by_step.csv")
    print(f"  3. {output_dir}/avg_gpu_breakdown_by_step.txt")
    print(f"  4. {output_dir}/detailed_events_filtered.csv")
    print(f"  5. {output_dir}/detailed_events_filtered.txt")
    print("=" * 100)