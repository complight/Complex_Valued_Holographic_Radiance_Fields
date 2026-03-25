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

def load_tandt_pointcloud(points3D_path: str) -> dict:
    """
    Load point cloud data from COLMAP points3D.txt file.
    
    Args:
        points3D_path: Path to the points3D.txt file.
        
    Returns:
        Dictionary containing point positions and colors.
    """
    positions = []
    colors = []
    
    with open(points3D_path, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith('#'):
                continue
            # Skip empty lines
            if line.strip() == '':
                continue
                
            # Parse the line
            # Format: POINT3D_ID X Y Z R G B ERROR TRACK_LENGTH LIST_OF_VIEWS
            parts = line.split()
            if len(parts) < 8:  # Need at least ID, XYZ, RGB, and ERROR
                continue
                
            # Extract position (XYZ)
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                
                positions.append([x, y, z])
                colors.append([r/255.0, g/255.0, b/255.0])  # Normalize RGB to 0-1 range
            except (ValueError, IndexError) as e:
                print(f"Error parsing point cloud data line: {line.strip()} - {e}")
                continue
    
    return {
        'positions': torch.tensor(positions, dtype=torch.float32),
        'colors': torch.tensor(colors, dtype=torch.float32)
    }

def get_tandt_datasets(
    dataset_name: str,
    image_size: Tuple[int, int],
    data_root: str = "./data/tandt",
    device: str = "cuda",
    train_split: float = 0.9,
    val_split: float = 0.1,
    seed: int = 100,
) -> Tuple[Dataset, Dataset, Dataset, Optional[dict]]:
    """
    Loads a dataset from the TandT dataset (train or truck).
    
    Args:
        dataset_name: The name of the dataset ('train' or 'truck').
        image_size: A tuple (height, width) denoting the target image size.
        data_root: The root folder containing TandT data (default: "./data/tandt").
        device: Device to place the camera tensors on.
        train_split: Fraction of data to use for training (default: 0.8).
        val_split: Fraction of data to use for validation (default: 0.1).
        seed: Random seed for train/val/test split (default: 42).
        
    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
        pointcloud_data: Dictionary containing point cloud positions and colors.
    """
    # Validate dataset name
    if dataset_name not in ['train', 'truck']:
        raise ValueError(f"Dataset name must be 'train' or 'truck', got {dataset_name}")
    
    # Build paths
    dataset_dir = os.path.join(data_root, dataset_name)
    transforms_json_path = os.path.join(dataset_dir, f"transforms_{dataset_name}.json")
    images_dir = os.path.join(dataset_dir, "images")
    
    # Add path for COLMAP sparse reconstruction data
    colmap_sparse_dir = os.path.join(dataset_dir, "sparse", "colmap_text")
    points3D_path = os.path.join(colmap_sparse_dir, "points3D.txt")
    
    print(f"Loading TandT dataset '{dataset_name}', image size={str(image_size)} ...")
    print(f"Using transforms file: {transforms_json_path}")
    print(f"Using images directory: {images_dir}")
    print(f"Using point cloud data: {points3D_path}")
    
    # Check if directories exist
    if not os.path.exists(transforms_json_path):
        raise FileNotFoundError(f"Transforms file not found: {transforms_json_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(points3D_path):
        print(f"Warning: Point cloud data not found: {points3D_path}. Will proceed without point cloud.")
        pointcloud_data = None
    else:
        # Load point cloud data
        pointcloud_data = load_tandt_pointcloud(points3D_path)
        print(f"Loaded {len(pointcloud_data['positions'])} points from point cloud data")
    
    # Load all transforms from the JSON file
    with open(transforms_json_path, 'r') as f:
        data = json.load(f)
    
    # Get all frames
    all_frames = data['frames']
    print(f"Found {len(all_frames)} frames in {transforms_json_path}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Randomly shuffle frames for train/val/test split
    frame_indices = np.arange(len(all_frames))
    np.random.shuffle(frame_indices)
    
    # Calculate split sizes
    n_train = int(len(all_frames) * train_split)
    n_val = int(len(all_frames) * val_split)
    n_test = len(all_frames) - n_train - n_val
    
    # Split indices
    train_indices = frame_indices[:n_train]
    val_indices = frame_indices[n_train:n_train+n_val]
    test_indices = frame_indices[n_train+n_val:]
    
    print(f"Split dataset into {n_train} training, {n_val} validation, and {n_test} test frames")
    
    # Function to process frames for a specific split
    def process_split(indices):
        # Create split-specific data structures
        Rs, Ts = [], []
        focal_lengths = []
        image_paths = []
        
        W_new, H_new = image_size
        
        # For each frame in the split
        for idx in tqdm(indices, desc='Processing frames'):
            frame = all_frames[idx]
            
            # Get image path
            file_path = frame['file_path']
            if not file_path.startswith('/'):
                # If relative path, make it absolute
                file_path = os.path.join(images_dir, os.path.basename(file_path))
            
            # Check if image file exists
            if not os.path.exists(file_path):
                # Try finding just the filename in the images directory
                base_name = os.path.basename(file_path)
                alternative_path = os.path.join(images_dir, base_name)
                if os.path.exists(alternative_path):
                    file_path = alternative_path
                else:
                    print(f"Warning: Image file not found: {file_path}")
                    continue
            
            image_paths.append(file_path)
            
            # Extract camera parameters
            c2w = np.array(frame['transform_matrix'])
            c2w[:3, [0, 2]] *= -1  # Flip x and z axes to match expected format
            
            R = c2w[:3, :3]
            T = c2w[:3, 3]
            
            # Convert to world->camera transformation
            R_w2c = R.T
            T_w2c = -R_w2c @ T
            
            Rs.append(R_w2c)
            Ts.append(T_w2c)
            
            # Get focal length
            if 'fl_x' in frame and 'fl_y' in frame:
                focal_x = frame['fl_x']
                focal_y = frame['fl_y']
                width = frame.get('w', data.get('w', 1024.0))  # Default to 1024 if not provided
                focal_length = (focal_x / width) * 2.0  # Normalize by image width and multiply by 2
            elif 'camera_angle_x' in frame:
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
            raise ValueError("No valid frames found for this split")
        
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
        
        # Process RGB images
        for img_path in tqdm(image_paths, desc='Loading images'):
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
        
        # Stack all images
        images = torch.stack(images)
        
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
            entries.append(entry)
        
        print(f"Created dataset split with {len(entries)} entries")
        
        return ListDataset(entries)
    
    # Process each split
    train_dataset = process_split(train_indices)
    val_dataset = process_split(val_indices)
    test_dataset = process_split(test_indices)
    
    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, pointcloud_data