import os
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from pytorch3d.renderer import PerspectiveCameras
from utils import get_nerf_datasets 

def get_camera_params(camera: PerspectiveCameras) -> Dict[str, torch.Tensor]:
    """
    Extract relevant camera parameters for comparison.
    
    Args:
        camera: A PerspectiveCameras object
    
    Returns:
        Dict containing camera parameters (R, T, focal_length, principal_point)
    """
    params = {}
    for k in ['R', 'T', 'focal_length', 'principal_point']:
        if hasattr(camera, k):
            params[k] = getattr(camera, k).detach().cpu()
    return params

def camera_distance(cam1_params: Dict, cam2_params: Dict) -> float:
    """
    Compute a distance metric between two cameras.
    
    Args:
        cam1_params: Dictionary of parameters for first camera
        cam2_params: Dictionary of parameters for second camera
    
    Returns:
        A scalar distance value
    """
    # Compute distance based on rotation and translation
    r_dist = torch.norm(cam1_params['R'] - cam2_params['R']).item()
    t_dist = torch.norm(cam1_params['T'] - cam2_params['T']).item()
    
    # If focal length exists, also consider it
    f_dist = 0.0
    if 'focal_length' in cam1_params and 'focal_length' in cam2_params:
        f_dist = torch.norm(cam1_params['focal_length'] - cam2_params['focal_length']).item()
    
    # Weighted sum of distances
    return r_dist + t_dist + 0.1 * f_dist

def find_duplicate_cameras(datasets: List, threshold: float = 1e-5) -> List[Tuple]:
    """
    Find duplicate cameras across datasets.
    
    Args:
        datasets: List of datasets (train, val, test)
        threshold: Threshold for considering cameras as duplicates
    
    Returns:
        List of tuples (dataset_idx_1, camera_idx_1, dataset_idx_2, camera_idx_2)
        indicating duplicate pairs
    """
    all_cameras = []
    dataset_labels = ['train', 'val', 'test']
    
    # Collect all cameras with their metadata
    for dataset_idx, dataset in enumerate(datasets):
        for sample_idx, sample in enumerate(dataset):
            camera = sample['camera']
            camera_params = get_camera_params(camera)
            all_cameras.append({
                'dataset_idx': dataset_idx,
                'dataset_name': dataset_labels[dataset_idx],
                'sample_idx': sample_idx,
                'camera_idx': sample['camera_idx'],
                'params': camera_params
            })
    
    # Compare all pairs of cameras
    duplicates = []
    for i in range(len(all_cameras)):
        for j in range(i + 1, len(all_cameras)):
            dist = camera_distance(all_cameras[i]['params'], all_cameras[j]['params'])
            if dist < threshold:
                duplicates.append((
                    all_cameras[i]['dataset_name'],
                    all_cameras[i]['camera_idx'],
                    all_cameras[j]['dataset_name'],
                    all_cameras[j]['camera_idx'],
                    dist
                ))
    
    return duplicates

def visualize_duplicates(datasets: List, duplicates: List[Tuple]) -> None:
    """
    Visualize duplicate camera views side by side.
    
    Args:
        datasets: List of datasets (train, val, test)
        duplicates: List of duplicate camera pairs
    """
    dataset_map = {'train': datasets[0], 'val': datasets[1], 'test': datasets[2]}
    
    for dup in duplicates:
        dataset1_name, cam1_idx, dataset2_name, cam2_idx, dist = dup
        
        # Find the sample with the matching camera_idx in each dataset
        sample1 = next((s for s in dataset_map[dataset1_name] if s['camera_idx'] == cam1_idx), None)
        sample2 = next((s for s in dataset_map[dataset2_name] if s['camera_idx'] == cam2_idx), None)
        
        if sample1 is None or sample2 is None:
            print(f"Could not find samples for camera indices {cam1_idx} and {cam2_idx}")
            continue
        
        # Display the images side by side
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(sample1['image'].cpu().numpy())
        plt.title(f"{dataset1_name} - Camera {cam1_idx}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(sample2['image'].cpu().numpy())
        plt.title(f"{dataset2_name} - Camera {cam2_idx}")
        plt.axis('off')
        
        plt.suptitle(f"Distance: {dist:.8f}")
        plt.tight_layout()
        plt.show()

def main():
    # Set parameters
    dataset_name = "materials"  # Change to your dataset name
    image_size = (400, 400)  # Adjust to your image size
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = get_nerf_datasets(
        dataset_name=dataset_name,
        image_size=image_size,
    )
    
    datasets = [train_dataset, val_dataset, test_dataset]
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Find duplicates
    print("Checking for duplicate camera views...")
    duplicates = find_duplicate_cameras(datasets, threshold=1e-5)
    
    if not duplicates:
        print("No duplicate camera views found!")
    else:
        print(f"Found {len(duplicates)} duplicate camera pairs:")
        for dup in duplicates:
            dataset1, cam1, dataset2, cam2, dist = dup
            print(f"  {dataset1} camera {cam1} and {dataset2} camera {cam2} (distance: {dist:.8f})")
        
        # Visualize duplicates
        print("\nVisualizing duplicate views...")
        visualize_duplicates(datasets, duplicates)

if __name__ == "__main__":
    main()