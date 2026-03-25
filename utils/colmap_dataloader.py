import os
import numpy as np
import torch
from PIL import Image
import math
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras
from typing import List, Tuple, Optional
import concurrent.futures
from functools import partial

class ListDataset(Dataset):
    """A simple dataset made of a list of entries."""
    def __init__(self, entries: List) -> None:
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def normalize_pointcloud(points, extra_scale, dataset_type=None):
    """Normalize pointcloud and return transformation parameters"""
    print('Points before normalization:', points.max(0)[0], points.min(0)[0], points.mean(0))
    
    if dataset_type == "nerf_llff_data":
        # For LLFF: Only scale, no centering
        # Scale points around their current center (no translation)
        center = torch.zeros(1, 3, device=points.device)  # Dummy center for compatibility
        scale = 1.0 / extra_scale  # Use extra_scale directly as scaling factor
        
        # Scale points without centering
        positions = points * scale
        
        print(f'LLFF: Scaling points by {scale} (1/{extra_scale})')
    else:
        # Original normalization for other datasets
        center = points.mean(0, keepdim=True)
        points_centered = points - center
        
        target_range = 1-(-1)
        max_dimension = torch.max(points_centered.abs()).item() * 2
        scale = target_range / max_dimension
        
        positions = points_centered * scale * extra_scale
    
    print('Points after normalization:', positions.max(0)[0], positions.min(0)[0], positions.mean(0))
    print('Center:', center, 'Scale:', scale)
    
    return positions, center, scale

def adjust_camera_for_normalization(camera, center, scale, dataset_type=None):
    """Adjust camera transformation for normalized point cloud"""
    w2c = camera['w2c'].clone()
    c2w = torch.inverse(w2c)
    
    R_c2w = c2w[:3, :3]
    T_c2w = c2w[:3, 3]
    
    T_c2w_normalized = (T_c2w - center[0]) / scale
    
    c2w_normalized = c2w.clone()
    c2w_normalized[:3, 3] = T_c2w_normalized
    
    w2c_normalized = torch.inverse(c2w_normalized)
    
    camera['w2c'] = w2c_normalized
    camera['T'] = w2c_normalized[:3, 3]
    
    return camera

def load_colmap_pointcloud(points3D_path: str, device="cuda") -> dict:
    """Load point cloud data from COLMAP points3D.txt file."""
    points3D = {}
    with open(points3D_path, "r") as fid:
        for line in fid:
            if line.startswith("#"):
                continue
            line = line.strip().split()
            if len(line) < 7:
                continue
            point_id = int(line[0])
            xyz = np.array([float(line[1]), float(line[2]), float(line[3])])
            rgb = np.array([int(line[4]), int(line[5]), int(line[6])])
            points3D[point_id] = (xyz, rgb)
    
    positions = []
    colors = []
    for point_id in points3D:
        xyz, color = points3D[point_id]
        positions.append(xyz)
        colors.append(color)
    
    if len(positions) == 0:
        raise ValueError("No points found in the points3D.txt file")
    
    positions = torch.tensor(np.array(positions), dtype=torch.float32, device=device)
    colors = torch.tensor(np.array(colors), dtype=torch.float32, device=device) / 255.0
    
    return {
        'positions': positions,
        'colors': colors
    }

def read_cameras_from_text(camera_file: str, images_file: str, device="cuda"):
    """Read camera parameters from COLMAP text files."""
    cameras = {}
    with open(camera_file, "r") as fid:
        for line in fid:
            if line.startswith("#"):
                continue
            line = line.strip().split()
            if len(line) < 4:
                continue
            camera_id = int(line[0])
            model = line[1]
            width = int(line[2])
            height = int(line[3])
            params = np.array([float(x) for x in line[4:]])
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    
    images = {}
    with open(images_file, "r") as fid:
        lines = fid.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            
            parts = line.split()
            if len(parts) < 10:
                i += 1
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            img_name = parts[9]
            
            i += 2  # Skip the current line and the next keypoints line
            
            q = np.array([qw, qx, qy, qz])
            R = quaternion_to_rotation_matrix(q)
            T = np.array([tx, ty, tz])
            
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            
            camera = cameras[camera_id]
            if camera["model"] == "SIMPLE_PINHOLE":
                f, cx, cy = camera["params"]
                intrinsics = np.array([f, f, cx, cy])
            elif camera["model"] == "PINHOLE":
                fx, fy, cx, cy = camera["params"]
                intrinsics = np.array([fx, fy, cx, cy])
            else:
                print(f"Warning: Unsupported camera model: {camera['model']}, trying to continue...")
                if len(camera["params"]) >= 4:
                    intrinsics = camera["params"][:4]
                else:
                    intrinsics = np.array([camera["width"], camera["width"], 
                                           camera["width"]/2, camera["height"]/2])

            images[image_id] = {
                "img_name": img_name,
                "w2c": torch.tensor(w2c, dtype=torch.float32, device=device),
                "R": torch.tensor(R, dtype=torch.float32, device=device),
                "T": torch.tensor(T, dtype=torch.float32, device=device),
                "H": camera["height"],
                "W": camera["width"],
                "intrinsics": torch.tensor(intrinsics, dtype=torch.float32, device=device),
                "camera_id": camera_id
            }
    
    return list(images.values())

def extract_index_info(filename):
    """Extract index and its format from filename"""
    match = re.search(r'(\d+)\.', filename)
    if match:
        index_str = match.group(1)
        return int(index_str), len(index_str)
    return -1, 0

def find_depth_for_image(img_file, depth_files_list, dataset_type="colmap"):
    """Find matching depth file for a given image file"""
    img_idx, idx_length = extract_index_info(img_file)
    if img_idx < 0:
        return None
    if dataset_type == "colmap":
        img_prefix = "train" if "train" in img_file.lower() else "val"
        idx_format = f"{img_idx:0{idx_length}d}"
        depth_pattern = f"Depth_{img_prefix}_{idx_format}"
        
        for depth_file in depth_files_list:
            if depth_pattern in depth_file:
                return depth_file
                
        for depth_file in depth_files_list:
            if f"Depth_{img_prefix}" in depth_file and f"_{img_idx}." in depth_file:
                return depth_file
    elif dataset_type == "mip360":
        # Get the base name without extension
        img_base = os.path.splitext(img_file)[0]
        
        # Try to find exact match first (same extension)
        depth_pattern = f"d_{img_file}"
        for depth_file in depth_files_list:
            if depth_file == depth_pattern:
                return depth_file
        
        # If no exact match, try to find with the same base name but potentially different extension
        for depth_file in depth_files_list:
            if depth_file.startswith(f"d_{img_base}."):
                return depth_file
    elif dataset_type == "nerf_llff_data":
        # Get the base name without extension
        img_base = os.path.splitext(img_file)[0]
        
        # Try to find exact match first (same extension)
        depth_pattern = f"d_{img_file}"
        for depth_file in depth_files_list:
            if depth_file == depth_pattern:
                return depth_file
        
        # If no exact match, try to find with the same base name but potentially different extension
        for depth_file in depth_files_list:
            if depth_file.startswith(f"d_{img_base}."):
                return depth_file

    return None

def process_frame(img_file, depth_files, camera_params, images_dir, depth_dir, 
                 image_size, device, load_depth, dataset_type):
    """Process a single frame - can be run in parallel"""
    W_new, H_new = image_size
    img_idx, _ = extract_index_info(img_file)
    
    if img_idx < 0:
        return None
    
    matching_depth = None
    if load_depth:
        matching_depth = find_depth_for_image(img_file, depth_files, dataset_type)
    
    frame = None
    for f in camera_params:
        frame_img_path = f['img_name']
        if os.path.basename(frame_img_path) == img_file:
            frame = f
            break
    
    if frame is None:
        return None
    
    img_path = os.path.join(images_dir, img_file)
    if not os.path.exists(img_path):
        return None
    
    depth_path = None
    if load_depth and matching_depth is not None:
        depth_path = os.path.join(depth_dir, matching_depth)
        if not os.path.exists(depth_path):
            depth_path = None
    
    c2w = torch.inverse(frame['w2c'])
    R, T = c2w[:3, :3], c2w[:3, 3:]
    
    R_luf = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)
    
    new_c2w = torch.cat([R_luf, T], 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.tensor([[0,0,0,1]], device=device)), 0))
    R_w2c, T_w2c = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
    
    Rs = R_w2c.cpu().numpy()
    Ts = T_w2c.cpu().numpy()
    
    if 'intrinsics' in frame:
        intrinsics = frame['intrinsics'].cpu().numpy()
        focal_x = intrinsics[0]
        focal_y = intrinsics[1]
        cx = intrinsics[2]
        cy = intrinsics[3]
        width = frame.get('W', W_new)
        height = frame.get('H', H_new)
        
        focal_length_x = (focal_x / width) * 2.0
        focal_length_y = (focal_y / height) * 2.0
        focal_length = (focal_length_x + focal_length_y) / 2.0
        
        principal_point_x = (cx - width / 2.0) / (width / 2.0)
        principal_point_y = (cy - height / 2.0) / (height / 2.0)
        
        focal_lengths = focal_length
        principal_points = [principal_point_x, principal_point_y]
    else:
        focal_lengths = 2.0
        principal_points = [0.0, 0.0]
    
    return {
        'img_path': img_path,
        'depth_path': depth_path,
        'R': Rs,
        'T': Ts,
        'focal_length': focal_lengths,
        'principal_point': principal_points
    }

def load_image(frame_data, image_size):
    """Load a single image - can be run in parallel"""
    W_new, H_new = image_size
    img_path = frame_data['img_path']
    
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((W_new, H_new), Image.BILINEAR)
        img_array = np.array(img)
        return torch.FloatTensor(img_array) / 255.0
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return torch.zeros((H_new, W_new, 3), dtype=torch.float32)

def load_depth_map(frame_data, image_size):
    """Load a single depth map - can be run in parallel"""
    W_new, H_new = image_size
    depth_path = frame_data['depth_path']
    
    if depth_path is None:
        return torch.zeros((H_new, W_new, 1), dtype=torch.float32)
    
    try:
        depth_img = Image.open(depth_path).convert('L')
        depth_img = depth_img.resize((W_new, H_new), Image.BILINEAR)
        depth_array = np.array(depth_img)
        depth_tensor = torch.FloatTensor(depth_array) / 255.0
        depth_tensor = depth_tensor.unsqueeze(2)
        return depth_tensor
    except Exception as e:
        print(f"Error processing depth {depth_path}: {e}")
        return torch.zeros((H_new, W_new, 1), dtype=torch.float32)

# New helper functions for spiral path generation
def normalize(x):
    """Normalize a vector."""
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    """
    Construct a view matrix from viewing direction, up vector, and position.
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    """
    Generate a spiral path for rendering novel views.
    
    Args:
        c2w: Camera-to-world matrix (3x4 or 4x4)
        up: Up vector
        rads: Radius of the spiral for x, y, z
        focal: Focal length
        zdelta: Z offset
        zrate: Z rate of change
        rots: Number of rotations
        N: Number of views to generate
        
    Returns:
        List of camera-to-world matrices along the spiral
    """
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    # Extract focal length if not provided
    if c2w.shape[1] >= 5:
        # Assuming hwf is stored in the last column
        hwf = c2w[:, 4:5]
    else:
        # If hwf not included, just use a placeholder
        hwf = np.array([[0], [0], [focal]])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # Compute position on spiral
        c = np.dot(c2w[:3, :4], np.array([
            np.cos(theta), 
            -np.sin(theta), 
            -np.sin(theta * zrate), 
            1.
        ]) * rads)
        
        # Compute viewing direction
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        
        # Create view matrix
        pose = viewmatrix(z, up, c)
        
        # Concatenate with hwf if available
        if hwf is not None and hwf.size > 0:
            pose = np.concatenate([pose, hwf], 1)
            
        render_poses.append(pose)
    
    return render_poses

def generate_spiral_poses_for_llff(dataset_name, camera_params, n_views, device="cuda"):
    """Generate spiral camera poses for LLFF dataset that stay within training camera bounds."""
    print(f"Generating bounded spiral path with {n_views} views for LLFF validation...")
    
    # Get all camera-to-world poses
    c2ws = []
    for cam in camera_params:
        w2c = cam['w2c'].cpu().numpy()
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)
    
    c2ws = np.stack(c2ws, 0)
    
    # Extract camera positions
    positions = np.stack([c2w[:3, 3] for c2w in c2ws], 0)
    center = positions.mean(0)
    
    # Calculate the min and max distances along each axis
    min_positions = np.min(positions, axis=0)
    max_positions = np.max(positions, axis=0)
    
    # Get average rotation from center camera (or camera closest to center)
    # Find camera closest to the center
    distances = np.linalg.norm(positions - center, axis=1)
    center_cam_idx = np.argmin(distances)
    center_cam = c2ws[center_cam_idx].copy()
    
    # Create a new c2w matrix with the center position and center camera's rotation
    avg_c2w = center_cam.copy()
    avg_c2w[:3, 3] = center  # Set position to the true center
    
    # Get focal length (average from all cameras)
    focal_lengths = []
    for cam in camera_params:
        if 'intrinsics' in cam:
            fx = cam['intrinsics'][0].item()
            fy = cam['intrinsics'][1].item()
            focal_lengths.append((fx + fy) / 2)
    
    focal = np.mean(focal_lengths) if focal_lengths else 1000.0
    
    # Calculate radii based on the distance from center to the furthest point in each direction
    # Use a more balanced approach to ensure the spiral is centered
    if dataset_name == "trex":
        safety_factor_x = 0.05
        safety_factor_y = 0.3
    elif dataset_name == "flower":
        safety_factor_x = 0.6
        safety_factor_y = 0.6
    else:
        safety_factor_x = 0.6
        safety_factor_y = 0.6
        
    radii = np.minimum(
        center - min_positions,
        max_positions - center
    )
    
    radii[0] *= safety_factor_x  # x-axis
    radii[1] *= safety_factor_y  # y-axis
    radii[2] *= 0 

    # Ensure minimum radius in each dimension
    min_radius = np.max(np.max(positions, axis=0) - np.min(positions, axis=0)) * 0.05
    radii = np.maximum(radii, min_radius)
    
    print(f"Spiral center: {center}")
    print(f"Spiral radii: {radii}")
    
    # Set up parameters for spiral
    up = normalize(avg_c2w[:3, 1])  # Use y-axis as up vector
    rots = 2  # Number of rotations
    zrate = 0  # Z rotation rate (adjust for more vertical variation)
    
    # Generate spiral path
    render_poses = render_path_spiral(
        avg_c2w, up, radii, focal, 
        zdelta=0.0, zrate=zrate, rots=rots, N=n_views
    )
    
    # Convert to w2c format compatible with dataset
    spiral_cameras = []
    for i, pose in enumerate(render_poses):
        # Extract the 3x4 part if bigger
        if pose.shape[1] > 4:
            pose = pose[:3, :4]
        
        # Convert to 4x4 c2w matrix
        c2w = np.eye(4)
        c2w[:3, :4] = pose
        
        # Convert to w2c
        w2c = np.linalg.inv(c2w)
        
        # Get R and T from w2c
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        # Use intrinsics from first camera
        intrinsics = camera_params[0]['intrinsics'].clone()
        H = camera_params[0]['H']
        W = camera_params[0]['W']
        
        cam = {
            "img_name": f"spiral_{i:03d}.png",
            "w2c": torch.tensor(w2c, dtype=torch.float32, device=device),
            "R": torch.tensor(R, dtype=torch.float32, device=device),
            "T": torch.tensor(T, dtype=torch.float32, device=device),
            "H": H,
            "W": W,
            "intrinsics": intrinsics,
            "camera_id": -1
        }
        
        spiral_cameras.append(cam)
    
    print(f"Generated {len(spiral_cameras)} spiral camera poses within training camera bounds")
    return spiral_cameras

def process_spiral_data(dataset_name, camera_params, image_size, device, num_views=-1):
    """
    Process spiral camera poses for validation.
    
    Args:
        camera_params: Original camera parameters
        image_size: Tuple of (W, H)
        device: Device to use for calculations
        num_views: Number of views in the spiral
        
    Returns:
        Dataset entries for spiral poses
    """
    W_new, H_new = image_size
    
    # Generate spiral poses
    spiral_cameras = generate_spiral_poses_for_llff(dataset_name, camera_params, n_views=num_views, device=device)
    
    entries = []
    cameras_data = {
        'focal_length': [],
        'principal_point': [],
        'R': [],
        'T': []
    }
    
    # Process each spiral camera pose
    for i, cam in enumerate(spiral_cameras):
        c2w = torch.inverse(cam['w2c'])
        R, T = c2w[:3, :3], c2w[:3, 3:]
        
        # Convert to left-up-forward coordinate system
        R_luf = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)
        
        new_c2w = torch.cat([R_luf, T], 1)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.tensor([[0,0,0,1]], device=device)), 0))
        R_w2c, T_w2c = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
        
        if 'intrinsics' in cam:
            intrinsics = cam['intrinsics'].cpu().numpy()
            focal_x = intrinsics[0]
            focal_y = intrinsics[1]
            cx = intrinsics[2]
            cy = intrinsics[3]
            width = cam.get('W', W_new)
            height = cam.get('H', H_new)
            
            focal_length_x = (focal_x / width) * 2.0
            focal_length_y = (focal_y / height) * 2.0
            focal_length = (focal_length_x + focal_length_y) / 2.0
            
            principal_point_x = (cx - width / 2.0) / (width / 2.0)
            principal_point_y = (cy - height / 2.0) / (height / 2.0)
            
            cameras_data['focal_length'].append([focal_length, focal_length])
            cameras_data['principal_point'].append([principal_point_x, principal_point_y])
        else:
            cameras_data['focal_length'].append([2.0, 2.0])
            cameras_data['principal_point'].append([0.0, 0.0])
        
        cameras_data['R'].append(R_w2c.cpu().numpy())
        cameras_data['T'].append(T_w2c.cpu().numpy())
    
    # Convert to tensors
    for k in cameras_data:
        if k == 'focal_length':
            cameras_data[k] = torch.tensor(cameras_data[k], dtype=torch.float32)
        elif k == 'principal_point':
            cameras_data[k] = torch.tensor(cameras_data[k], dtype=torch.float32)
        else:
            cameras_data[k] = torch.tensor(np.stack(cameras_data[k]), dtype=torch.float32)
    
    # Create camera objects
    num_frames = len(spiral_cameras)
    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None].to(device) for k, v in cameras_data.items()}
        )
        for cami in range(num_frames)
    ]
    
    # Create dataset entries with dummy images (since we only need camera poses for rendering)
    dummy_image = torch.zeros((H_new, W_new, 3), dtype=torch.float32)
    dummy_depth = torch.zeros((H_new, W_new, 1), dtype=torch.float32)
    
    for i in range(num_frames):
        entry = {
            "image": dummy_image,
            "camera": cameras[i],
            "camera_idx": int(i),
            "depth": dummy_depth
        }
        entries.append(entry)
    
    print(f"Created spiral dataset with {len(entries)} entries")
    
    return entries, [], cameras

def process_split_data(image_files, depth_files, camera_params, images_dir, depth_dir, 
                      image_size, device, load_depth, dataset_type, dataset_name, is_validation=False, num_workers=16, 
                      use_spiral_for_val=False, spiral_views=-1):
    """Process frames for train or val split using parallel processing"""
    split_type = "validation" if is_validation else "training"
    
    # For validation in LLFF dataset with spiral path option
    if is_validation and use_spiral_for_val and dataset_type == "nerf_llff_data":
        print(f"Using spiral path for {split_type} in LLFF dataset")
        return process_spiral_data(dataset_name, camera_params, image_size, device, num_views=spiral_views)
    
    split_images = image_files if is_validation else image_files
    split_depths = depth_files if is_validation else depth_files if load_depth else []
    
    if not split_images:
        raise ValueError(f"No {split_type} images found")
    
    print(f"\nExample {split_type} image filenames:")
    for img in split_images[:3]:
        idx, idx_len = extract_index_info(img)
        print(f"  - {img} (index: {idx}, length: {idx_len})")
        if load_depth:
            depth_file = find_depth_for_image(img, split_depths, dataset_type)
            print(f"    Matching depth: {depth_file}")
    
    # Process frames in parallel
    process_frame_partial = partial(
        process_frame, 
        depth_files=split_depths, 
        camera_params=camera_params,
        images_dir=images_dir,
        depth_dir=depth_dir,
        image_size=image_size,
        device=device,
        load_depth=load_depth,
        dataset_type=dataset_type
    )
    
    print(f"Processing {split_type} set with {num_workers} workers...")
    
    # Create a dictionary to store results indexed by original position
    frame_results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks with their original index
        future_to_idx = {executor.submit(process_frame_partial, img_file): i 
                        for i, img_file in enumerate(split_images)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), 
                          total=len(split_images), desc=f'Processing {split_type} frames'):
            original_idx = future_to_idx[future]
            try:
                frame_data = future.result()
                if frame_data is not None:
                    frame_results[original_idx] = frame_data
            except Exception as e:
                print(f"Error processing image at index {original_idx}: {e}")
    
    # Sort results by original index to maintain sequential order
    valid_frames = []
    for i in sorted(frame_results.keys()):
        valid_frames.append(frame_results[i])
    
    if len(valid_frames) == 0:
        raise ValueError(f"No matching frames found for {split_type} images")
    
    print(f"Successfully processed {len(valid_frames)} frames in sequential order")
    
    # Prepare focal lengths and principal points
    focal_lengths = [frame['focal_length'] for frame in valid_frames]
    principal_points = [frame['principal_point'] for frame in valid_frames]
    Rs = [frame['R'] for frame in valid_frames]
    Ts = [frame['T'] for frame in valid_frames]
    
    focal_length_tensor = torch.tensor(focal_lengths, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    
    cameras_data = {
        'focal_length': focal_length_tensor,
        'principal_point': torch.tensor(principal_points, dtype=torch.float32),
        'R': torch.tensor(np.stack(Rs), dtype=torch.float32),
        'T': torch.tensor(np.stack(Ts), dtype=torch.float32)
    }
    
    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    
    # Load images in parallel (this part already maintains order correctly)
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_frame = {executor.submit(load_image, frame, image_size): i 
                          for i, frame in enumerate(valid_frames)}
        
        # Pre-allocate list with None values
        images = [None] * len(valid_frames)
        
        for future in tqdm(concurrent.futures.as_completed(future_to_frame), 
                          total=len(valid_frames), desc=f'Loading {split_type} images'):
            idx = future_to_frame[future]
            try:
                img = future.result()
                images[idx] = img
                if idx < 3:
                    print(f"Loaded image: {valid_frames[idx]['img_path']}")
            except Exception as e:
                print(f"Error in image loading for index {idx}: {e}")
                images[idx] = torch.zeros((image_size[1], image_size[0], 3), dtype=torch.float32)
                
    if load_depth:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_frame = {executor.submit(load_depth_map, frame, image_size): i 
                              for i, frame in enumerate(valid_frames)}
            
            # Pre-allocate list with None values
            depths = [None] * len(valid_frames)
            
            for future in tqdm(concurrent.futures.as_completed(future_to_frame), 
                              total=len(valid_frames), desc=f'Loading {split_type} depths'):
                idx = future_to_frame[future]
                try:
                    depth = future.result()
                    depths[idx] = depth
                    if idx < 3:
                        print(f"Loaded depth: {valid_frames[idx]['depth_path']}")
                except Exception as e:
                    print(f"Error in depth loading for index {idx}: {e}")
                    depths[idx] = torch.zeros((image_size[1], image_size[0], 1), dtype=torch.float32)
    
    images = torch.stack(images)
    if load_depth and depths:
        depths = torch.stack(depths)
    
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels
    
    # Create camera objects
    num_frames = len(valid_frames)
    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None].to(device) for k, v in cameras_data.items()}
        )
        for cami in range(num_frames)
    ]
    
    # Create dataset entries
    entries = []
    for i in range(num_frames):
        entry = {
            "image": images[i], 
            "camera": cameras[i], 
            "camera_idx": int(i)
        }
        if load_depth and depths is not None and i < len(depths):
            entry["depth"] = depths[i]
            
        entries.append(entry)
    
    print(f"Created dataset with {len(entries)} entries in sequential order")
    
    return entries, images, cameras


def get_colmap_datasets(
    dataset_type: str,
    dataset_name: str,
    image_size: Tuple[int, int],
    data_root: str = "./data/",
    load_depth: bool = True,
    load_pointcloud: bool = True, 
    use_spiral_for_llff_val: bool = False,
    spiral_views: int = 120, 
    args = None, 
    num_workers: int = 16, 
    device: str = "cuda"
) -> Tuple[Dataset, Dataset, Dataset, Optional[dict]]:
    """Loads a dataset directly from COLMAP data text files."""
    dataset_dir = os.path.join(data_root, dataset_type, dataset_name)
    load_point_path=args.load_point_path
    if dataset_type == "colmap":
        colmap_sparse_dir = os.path.join(dataset_dir, "colmap", "sparse", "colmap_text")
        images_dir = os.path.join(dataset_dir, "image")
        depth_dir = os.path.join(dataset_dir, "depth") if load_depth else None
    elif dataset_type == "mip360":
        # colmap_sparse_dir = os.path.join(dataset_dir, "sparse", "colmap_text")
        # colmap_sparse_dir = os.path.join(dataset_dir, "vggt_sparse", "colmap_text")
        colmap_sparse_dir = os.path.join(dataset_dir, "sparse_spiral", "colmap_text")
        images_dir = os.path.join(dataset_dir, "images_spiral")
        depth_dir = os.path.join(dataset_dir, "depth_spiral") if load_depth else None
    elif dataset_type == "nerf_llff_data":
        # colmap_sparse_dir = os.path.join(dataset_dir, "sparse", "colmap_text")
        # images_dir = os.path.join(dataset_dir, "images_4")
        colmap_sparse_dir = os.path.join(dataset_dir, "sparse_path_all", "colmap_text")
        images_dir = os.path.join(dataset_dir, "imgs_path_all")
        depth_dir = os.path.join(dataset_dir, "depth_path_all") if load_depth else None
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")     
    cameras_file = os.path.join(colmap_sparse_dir, "cameras.txt")
    images_file = os.path.join(colmap_sparse_dir, "images.txt")
    points3D_file = os.path.join(colmap_sparse_dir, "points3D.txt")

    
    print(f"Loading COLMAP dataset {dataset_name}, image size={str(image_size)} ...")
    print(f"Using cameras file: {cameras_file}")
    print(f"Using images file: {images_file}")
    print(f"Using points3D file: {points3D_file}")
    print(f"Using images directory: {images_dir}")
    print(f"Using {num_workers} workers for parallel processing")
    
    if load_depth:
        print(f"Using depth directory: {depth_dir}")
    
    if dataset_type == "nerf_llff_data" and use_spiral_for_llff_val:
        print(f"LLFF dataset: Will generate spiral camera path with {spiral_views} views for validation")
    
    if not os.path.exists(cameras_file):
        raise FileNotFoundError(f"Cameras file not found: {cameras_file}")
    if not os.path.exists(images_file):
        raise FileNotFoundError(f"Images file not found: {images_file}")
    if not os.path.exists(points3D_file):
        raise FileNotFoundError(f"Points3D file not found: {points3D_file}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if load_depth and not os.path.exists(depth_dir):
        print(f"Warning: Depth directory not found: {depth_dir}. Will proceed without depth images.")
        load_depth = False
    
    pointcloud_data = None
    if load_pointcloud:
        if not os.path.exists(points3D_file):
            print(f"Warning: Points3D file not found: {points3D_file}. Will proceed without point cloud data.")
        else:
            pointcloud_data = load_colmap_pointcloud(points3D_file, device)
            print(f"Loaded {len(pointcloud_data['positions'])} points from point cloud data.")
    
    camera_params = read_cameras_from_text(cameras_file, images_file, device)
    print(f"Found {len(camera_params)} cameras in COLMAP files")

    points, center, scale = normalize_pointcloud(pointcloud_data["positions"], args.extra_scale, dataset_type=dataset_type)
    pointcloud_data["positions"] = points
    for i, camera in enumerate(camera_params):
        camera_params[i] = adjust_camera_for_normalization(camera, center, scale, dataset_type=dataset_type)

    image_files = []
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} image files in {images_dir}")
    
    if dataset_type == "colmap":
        train_images = sorted([f for f in image_files if 'train' in f.lower()])
        val_images = sorted([f for f in image_files if 'val' in f.lower()])
        # train_images = ['Panoramic_val_0059.png']
    elif dataset_type == "mip360":
        num = 40
        all_images = sorted(image_files)
        # Select every 5th image for training
        train_images = [img for i, img in enumerate(all_images) if i % 2 == 0][:num]
        # Use all images for validation
        val_images = [img for i, img in enumerate(all_images) if i % 2 == 0][:num]
        print(f"MIP360: Selected {len(train_images)} images for training out of {len(all_images)} total")
    elif dataset_type == "nerf_llff_data":
        # train_images = sorted(image_files)
        # val_images = sorted(image_files)
        all_images = sorted(image_files)
        # Power-of-2 sampling (32 images)
        sample_rate = 32
        step = len(all_images) / sample_rate
        train_images = [all_images[int(i * step)] for i in range(sample_rate)]
        val_images = sorted(image_files)
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    depth_files = []
    if load_depth:
        if os.path.exists(depth_dir):
            depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(depth_files)} depth files in {depth_dir}")
        
        if dataset_type == "colmap":
            train_depths = sorted([f for f in depth_files if 'train' in f.lower()])
            val_depths = sorted([f for f in depth_files if 'val' in f.lower()])
        elif dataset_type == "mip360":
            all_depths = sorted(depth_files)
            # Select every 5th depth for training to match images
            train_depths = [d for i, d in enumerate(all_depths) if i % 2 == 0][:num]
            # Use all depths for validation
            val_depths = [img for i, img in enumerate(all_images) if i % 2 == 0][:num]
            # train_depths = ['Depth_val_0059.png']
        elif dataset_type == "nerf_llff_data":
            # train_depths = sorted(depth_files)
            # val_depths = sorted(depth_files)   
            all_depths = sorted(depth_files)
            # Power-of-2 sampling (32 images)
            sample_rate = 32
            step = len(all_images) / sample_rate
            train_depths = [all_depths[int(i * step)] for i in range(sample_rate)]
            val_depths = sorted(depth_files)
        print(f"Found {len(train_depths)} training depth maps and {len(val_depths)} validation depth maps")
    num = None
    train_images = train_images[:num]
    train_depths = depth_files[:num]
    # num = None
    val_images = val_images[:num]
    val_depths = depth_files[:num]
    
    train_entries, train_images, train_cameras = process_split_data(
        train_images, train_depths if load_depth else [],
        camera_params, images_dir, depth_dir,
        image_size, device, load_depth, dataset_type, dataset_name, 
        is_validation=False, num_workers=num_workers
    )
    
    val_entries, val_images, val_cameras = process_split_data(
        val_images, val_depths if load_depth else [],
        camera_params, images_dir, depth_dir,
        image_size, device, load_depth, dataset_type, dataset_name, 
        is_validation=True, num_workers=num_workers,
        use_spiral_for_val=use_spiral_for_llff_val, 
        spiral_views=spiral_views
    )
    
    train_dataset = ListDataset(train_entries)
    val_dataset = ListDataset(val_entries)
    test_dataset = ListDataset(val_entries)  # Test dataset uses the same entries as validation
    
    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)} (test=val)")
    
    # Check if depth is included in the dataset entries
    has_depth = False
    if len(train_dataset) > 0:
        has_depth = "depth" in train_dataset[0]
    print(f"Dataset includes depth information: {has_depth}")
    
    # Check if spiral path was generated for validation
    if dataset_type == "nerf_llff_data" and use_spiral_for_llff_val and len(val_dataset) > 0:
        has_spiral = "is_spiral" in val_dataset[0]
        print(f"Validation dataset uses spiral camera path: {has_spiral}")
    if load_point_path:
        print("[IMPORTANT] -> load_point_path: ", load_point_path, " to point cloud instead of the Point3D.txt")
        pointcloud_data = torch.load(load_point_path, map_location='cpu')
    return train_dataset, val_dataset, test_dataset, pointcloud_data