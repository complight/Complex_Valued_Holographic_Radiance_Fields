import os
import json
import numpy as np
import torch
from PIL import Image
import math
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras
from typing import List, Tuple, Optional

class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """
    def __init__(self, entries: List) -> None:
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]
    
def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    """
    Calculate the average pose from a set of camera poses.
    """
    # Average the translation
    center = np.mean(poses[:, :3, 3], axis=0)
    # Average the direction
    vec2 = normalize(np.sum(poses[:, :3, 2], axis=0))
    # Average the up vector
    up = np.sum(poses[:, :3, 1], axis=0)
    # Create view matrix
    c2w = viewmatrix(vec2, up, center)
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    """
    Generate a spiral path of camera poses for novel view synthesis.
    """
    render_poses = []
    # Note: Input rads should be length 3, we extend it to length 4
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    
    return render_poses
    
def get_llff_datasets(
    dataset_name: str,
    image_size: Tuple[int, int],
    data_root: str = "./data/nerf_llff_data",
    device: str = "cuda",
    load_depth: bool = True,
    generate_val: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads a dataset from LLFF format data.
    
    Args:
        dataset_name: The name of the dataset (e.g., 'flower').
        image_size: A tuple (height, width) denoting the target image size.
        data_root: The root folder containing LLFF data (default: "./data/nerf_llff_data").
        device: Device to place the camera tensors on.
        load_depth: Whether to load depth images (default: True).
        generate_val: Whether to generate interpolated validation/test poses (default: True).
                     If False, uses the training set for validation and test.
        
    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object (using rendered poses if generate_val=True).
        test_dataset: The testing dataset object (same as validation).
    """
    # Build paths
    dataset_dir = os.path.join(data_root, dataset_name)
    transforms_json_path = os.path.join(dataset_dir, "transforms.json")
    images_dir = os.path.join(dataset_dir, "images")
    depth_dir = os.path.join(dataset_dir, "depth")
    
    print(f"Loading LLFF dataset {dataset_name}, image size={str(image_size)} ...")
    print(f"Using transforms file: {transforms_json_path}")
    print(f"Using images directory: {images_dir}")
    
    if load_depth:
        print(f"Using depth directory: {depth_dir}")
    
    # Check if directories exist
    if not os.path.exists(transforms_json_path):
        raise FileNotFoundError(f"Transforms file not found: {transforms_json_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if load_depth and not os.path.exists(depth_dir):
        print(f"Warning: Depth directory not found: {depth_dir}. Will proceed without depth images.")
        load_depth = False
    
    # Load transforms from JSON file
    with open(transforms_json_path, 'r') as f:
        data = json.load(f)
    
    # Get all frames
    all_frames = data['frames']
    print(f"Found {len(all_frames)} frames in transforms.json")
    
    # List all image files in the image directory
    image_files = []
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} image files in {images_dir}")
    
    # Use all images for train, val, and test as specified
    # Sort images to ensure consistent order
    image_files = sorted(image_files)
    train_images = image_files
    
    print(f"Using all {len(image_files)} images for training")
    
    # List all depth files if needed
    if load_depth:
        depth_files = []
        if os.path.exists(depth_dir):
            depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith('.png')]
            print(f"Found {len(depth_files)} depth files in {depth_dir}")
    
    # Function to find matching depth file for a given image file
    def find_depth_for_image(img_file):
        # Extract the base name without extension
        img_base = os.path.splitext(img_file)[0]
        # Construct depth file name: 'd_' + image_name + '.png'
        depth_file = f"d_{img_base}.png"
        
        # Check if the exact name exists
        if os.path.exists(os.path.join(depth_dir, depth_file)):
            return depth_file
        
        # Try alternative: just the base name with .png
        depth_file = f"{img_base}.png"
        if os.path.exists(os.path.join(depth_dir, depth_file)):
            return depth_file
            
        # If the image has a JPG extension, try looking for PNG depth
        if img_file.lower().endswith(('.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            depth_file = f"d_{base_name}.png"
            if os.path.exists(os.path.join(depth_dir, depth_file)):
                return depth_file
        
        return None
    
    # Function to extract frame index from filename
    def extract_frame_index(filename):
        # Try to extract a numeric index from the filename
        base_name = os.path.splitext(filename)[0]
        numbers = re.findall(r'\d+', base_name)
        if numbers:
            # Use the last sequence of numbers as the index
            return int(numbers[-1])
        return -1
    
    # Function to find matching frame from transforms.json for a given image file
    def find_matching_frame(img_file):
        # First try exact filename match
        for frame in all_frames:
            frame_file = os.path.basename(frame['file_path'])
            if frame_file == img_file:
                return frame
        
        # If no exact match, try matching by frame index
        img_index = extract_frame_index(img_file)
        if img_index >= 0:
            for frame in all_frames:
                frame_file = os.path.basename(frame['file_path'])
                frame_index = extract_frame_index(frame_file)
                if frame_index == img_index:
                    return frame
        
        # If still no match, use filename substring matching
        img_base = os.path.splitext(img_file)[0]
        for frame in all_frames:
            frame_file = os.path.basename(frame['file_path'])
            if img_base in frame_file or frame_file in img_file:
                return frame
        
        return None
    
    # Function to process frames for train split
    def process_train_split():
        # Create split-specific data structures
        Rs, Ts = [], []
        focal_lengths = []
        image_paths = []
        depth_paths = []
        
        W_new, H_new = image_size
        
        # Get the image list for this split
        split_images = train_images
        
        # Check if we have images
        if not split_images:
            raise ValueError(f"No training images found")
        
        # Print some examples of filenames for debugging
        print(f"\nExample training image filenames:")
        for img in split_images[:3]:
            print(f"  - {img}")
            if load_depth:
                depth_file = find_depth_for_image(img)
                print(f"    Matching depth: {depth_file}")
        
        # For each frame in the split, find matching camera parameters
        for img_file in tqdm(split_images, desc='Processing training set'):
            # Find matching depth file if needed
            matching_depth = None
            if load_depth:
                matching_depth = find_depth_for_image(img_file)
                if matching_depth is None:
                    print(f"Warning: No matching depth found for {img_file}")
            
            # Find matching frame in transforms.json
            frame = find_matching_frame(img_file)
            
            # If we can't find a matching frame, report and skip
            if frame is None:
                print(f"Warning: No frame found in transforms.json for image {img_file}")
                continue
            
            # Add image path
            img_path = os.path.join(images_dir, img_file)
            if not os.path.exists(img_path):
                print(f"Warning: Image file doesn't exist: {img_path}")
                continue
                
            image_paths.append(img_path)
            
            # Add depth path if available
            if load_depth and matching_depth is not None:
                depth_path = os.path.join(depth_dir, matching_depth)
                if not os.path.exists(depth_path):
                    print(f"Warning: Depth file doesn't exist: {depth_path}")
                    depth_paths.append(None)
                else:
                    depth_paths.append(depth_path)
            else:
                depth_paths.append(None)
            
            # Extract camera parameters
            c2w = np.array(frame['transform_matrix'])
            
            # Handle camera coordinate system conversion if necessary
            # COLMAP uses different coordinate system than what's expected
            # Flip axis so it matches the expected format
            c2w[:3, [0, 2]] *= -1  
            
            R = c2w[:3, :3]
            T = c2w[:3, 3]
            
            # Convert to world->camera transformation
            R_w2c = R.T
            T_w2c = -R_w2c @ T
            
            Rs.append(R_w2c)
            Ts.append(T_w2c)
            
            # Get focal length (normalize by width as in the original code)
            if 'fl_x' in frame and 'fl_y' in frame:
                focal_x = frame['fl_x']
                focal_y = frame['fl_y']
                width = frame.get('w', 1024.0)  # Default to 1024 if not provided
                focal_length = (focal_x / width) * 2.0  # Normalize by image width and multiply by 2
            elif 'camera_angle_x' in frame:
                # If camera_angle_x is provided, use it to calculate focal length
                camera_angle_x = frame['camera_angle_x']
                focal_length = 0.5 / math.tan(0.5 * camera_angle_x) * 2.0  # Normalized format
            else:
                # Check if camera.angle_x is in the root data
                if 'camera_angle_x' in data:
                    camera_angle_x = data['camera_angle_x']
                    focal_length = 0.5 / math.tan(0.5 * camera_angle_x) * 2.0
                else:
                    # Fallback to a default value
                    focal_length = 2.0  # Default normalized focal length
                
            focal_lengths.append(focal_length)
        
        if len(Rs) == 0:
            raise ValueError(f"No matching frames found for training images")
        
        # Create camera parameters
        num_frames = len(Rs)
        focal_length_tensor = torch.tensor(focal_lengths, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
        
        # Prepare camera data
        cameras_data = {
            'focal_length': focal_length_tensor,
            'principal_point': torch.zeros((num_frames, 2), dtype=torch.float32),  # Center of the image
            'R': torch.tensor(np.stack(Rs), dtype=torch.float32),
            'T': torch.tensor(np.stack(Ts), dtype=torch.float32)
        }
        
        # Process images directly
        _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None  # The dataset image may be very large
        
        images = []
        depths = []
        
        # Process RGB images
        for img_path in tqdm(image_paths, desc='Loading training images'):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((W_new, H_new), Image.BILINEAR)
                img_array = np.array(img)
                images.append(torch.FloatTensor(img_array) / 255.0)
                if len(images) <= 3:  # Print first few for debugging
                    print(f"Loaded image: {img_path}")
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Use a black image as placeholder
                images.append(torch.zeros((H_new, W_new, 3), dtype=torch.float32))
        
        # Process depth images if available
        if load_depth:
            for depth_path in tqdm(depth_paths, desc='Loading training depths'):
                if depth_path is None:
                    # Use zero tensor as placeholder when depth is not available
                    depths.append(torch.zeros((H_new, W_new, 1), dtype=torch.float32))
                    continue
                
                try:
                    depth_img = Image.open(depth_path).convert('L')  # Convert to grayscale
                    depth_img = depth_img.resize((W_new, H_new), Image.BILINEAR)
                    depth_array = np.array(depth_img)
                    # Normalize depth to 0-1 range
                    depth_tensor = torch.FloatTensor(depth_array) / 255.0
                    # Add channel dimension
                    depth_tensor = depth_tensor.unsqueeze(2)
                    depths.append(depth_tensor)
                    if len(depths) <= 3:  # Print first few for debugging
                        print(f"Loaded depth: {depth_path}")
                
                except Exception as e:
                    print(f"Error processing depth {depth_path}: {e}")
                    # Use zero tensor as placeholder
                    depths.append(torch.zeros((H_new, W_new, 1), dtype=torch.float32))
        
        # Stack all images
        images = torch.stack(images)
        if load_depth and depths:
            depths = torch.stack(depths)
        
        Image.MAX_IMAGE_PIXELS = _image_max_image_pixels
        
        # Create camera objects
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
            
            # Add depth if available
            if load_depth and depths is not None and i < len(depths):
                entry["depth"] = depths[i]
                
            entries.append(entry)
        
        print(f"Created training dataset with {len(entries)} entries")
        
        # Return all components needed for creating both training and validation datasets
        return entries, images, cameras, Rs, Ts, focal_length_tensor, focal_lengths
    
    # Process training split
    train_entries, train_images, train_cameras, Rs, Ts, focal_length_tensor, focal_lengths = process_train_split()
    
    # Generate rendered poses for validation and test sets
    def generate_render_poses(Rs, Ts, focal_lengths):
        W_new, H_new = image_size
        
        # Stack and convert camera poses to format needed for poses_avg
        c2ws = []
        for i in range(len(Rs)):
            R = Rs[i]
            T = Ts[i]
            
            # Convert back to camera-to-world transformation
            R_c2w = R
            T_c2w = -R @ T
            
            c2w = np.eye(4)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = T_c2w
            
            c2ws.append(c2w)
        
        c2ws = np.stack(c2ws)[:, :3, :4]  # Shape: [N, 3, 4]
        
        # Get average pose
        avg_c2w = poses_avg(c2ws)
        
        # Get up vector
        up = normalize(np.mean([c2w[:3, 1] for c2w in c2ws], axis=0))
        
        # Calculate average focal length
        focal = np.mean(focal_lengths)
        
        # Calculate scene bounds
        camera_centers = np.stack([c2w[:3, 3] for c2w in c2ws])
        
        # Get radii for spiral path (use percentiles to be more robust to outliers)
        rads = np.percentile(np.abs(camera_centers - np.mean(camera_centers, axis=0)), 90, 0)
        # Just use the 3 spatial dimensions (x, y, z) - the fourth dimension will be added in render_path_spiral
        rads = rads[:3] * 0.2
        # # To make it more circular, you can make x and y the same radius
        # # By taking the average or maximum of the two
        # avg_rad_xy = (rads[0] + rads[1]) / 2
        # rads[0] = avg_rad_xy
        # rads[1] = avg_rad_xy
        # Generate spiral path
        n_views = 40  # Number of views to render
        render_poses = render_path_spiral(
            avg_c2w,
            up,
            rads,
            focal,
            zdelta=0,  # Small shift in z
            zrate=0.1,   # Rate of change in z direction
            rots=1,      # Number of rotations
            N=n_views    # Number of views
        )
        
        # Convert to proper format for our dataset
        render_Rs = []
        render_Ts = []
        
        for pose in render_poses:
            # Extract camera parameters
            R = pose[:3, :3]
            T = pose[:3, 3]
            
            # Convert to world->camera transformation
            R_w2c = R.T
            T_w2c = -R_w2c @ T
            
            render_Rs.append(R_w2c)
            render_Ts.append(T_w2c)
        
        # Create camera parameters
        num_frames = len(render_Rs)
        # Use mean focal length for all rendered views
        mean_focal = np.mean(focal_lengths)
        render_focal_tensor = torch.full((num_frames, 2), mean_focal, dtype=torch.float32)
        
        # Prepare camera data
        render_cameras_data = {
            'focal_length': render_focal_tensor,
            'principal_point': torch.zeros((num_frames, 2), dtype=torch.float32),  # Center of the image
            'R': torch.tensor(np.stack(render_Rs), dtype=torch.float32),
            'T': torch.tensor(np.stack(render_Ts), dtype=torch.float32)
        }
        
        # Create camera objects
        render_cameras = [
            PerspectiveCameras(
                **{k: v[cami][None].to(device) for k, v in render_cameras_data.items()}
            )
            for cami in range(num_frames)
        ]
        
        # Create dataset entries with empty images (since these are render views)
        render_entries = []
        for i in range(num_frames):
            # Create empty image tensor for rendered views
            empty_image = torch.zeros((H_new, W_new, 3), dtype=torch.float32)
            
            entry = {
                "image": empty_image,  # Empty image for rendered views
                "camera": render_cameras[i],
                "camera_idx": int(i),
                "is_render": True  # Flag to indicate this is a render view
            }
            
            if load_depth:
                # Empty depth tensor
                empty_depth = torch.zeros((H_new, W_new, 1), dtype=torch.float32)
                entry["depth"] = empty_depth
                
            render_entries.append(entry)
        
        print(f"Created {num_frames} interpolated camera poses for validation/test")
        
        return render_entries
    
    # Create datasets based on generate_val flag
    train_dataset = ListDataset(train_entries)
    
    if generate_val:
        # Generate interpolated views for validation/test
        print("Generating interpolated validation and test sets...")
        val_entries = generate_render_poses(Rs, Ts, focal_lengths)
        val_dataset = ListDataset(val_entries)
        test_dataset = ListDataset(val_entries)  # Test is same as val
        print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)} (interpolated), test={len(test_dataset)} (interpolated)")
    else:
        # Use training set for validation and test
        print("Using training set for validation and test sets (original pipeline)...")
        val_dataset = train_dataset
        test_dataset = train_dataset
        print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)} (same as train), test={len(test_dataset)} (same as train)")
    
    # Check if depth is included in the dataset entries
    has_depth = False
    if len(train_dataset) > 0:
        has_depth = "depth" in train_dataset[0]
    print(f"Dataset includes depth information: {has_depth}")
    
    return train_dataset, val_dataset, test_dataset