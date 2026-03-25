#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_gaussian_file(file_path):
    """
    Load a .pth file containing Gaussian parameters.
    
    Args:
        file_path: Path to the .pth file
        
    Returns:
        A dictionary containing the Gaussian parameters
    """
    print(f"Loading Gaussian file: {file_path}")
    checkpoint = torch.load(file_path, map_location='cpu')
    
    # Check if we're working with a standard Gaussian format
    required_keys = ["means", "colours"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    
    if missing_keys:
        print(f"Warning: Missing required keys in checkpoint file: {missing_keys}")
        raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")
    
    return checkpoint

def extract_point_cloud(gaussian_data, downsample_factor=1.0):
    """
    Extract point cloud data (positions and colors) from Gaussian data.
    
    Args:
        gaussian_data: Dictionary containing Gaussian parameters
        downsample_factor: Factor to control the density of the resulting point cloud.
                           1.0 means use all points, 0.5 means use half the points, etc.
    
    Returns:
        A dictionary with 'positions' and 'colors' keys containing tensors
    """
    if downsample_factor <= 0 or downsample_factor > 1.0:
        raise ValueError("Downsample factor must be in range (0, 1.0]")
    
    # Get total number of Gaussians
    num_gaussians = len(gaussian_data["means"])
    print(f"Total Gaussians in file: {num_gaussians}")
    
    # Determine how many points to keep
    num_points_to_keep = max(1, int(num_gaussians * downsample_factor))
    
    # If we're keeping all points, just return the original data
    if num_points_to_keep == num_gaussians:
        return {
            "positions": gaussian_data["means"].clone().item(),
            "colors": gaussian_data["colours"].clone().item()
        }
    
    # Otherwise, uniformly sample the desired number of points
    indices = torch.randperm(num_gaussians)[:num_points_to_keep]
    
    # Extract the selected points
    positions = gaussian_data["means"][indices]
    colors = gaussian_data["colours"][indices]
    
    print(f"Downsampled to {num_points_to_keep} points ({downsample_factor:.2%} of original)")
    
    return {
        "positions": positions,
        "colors": colors
    }

def save_point_cloud(point_cloud, output_path):
    """
    Save the point cloud data to a .pth file.
    
    Args:
        point_cloud: Dictionary with 'positions' and 'colors' keys
        output_path: Path to save the point cloud
    """
    torch.save(point_cloud, output_path)
    print(f"Point cloud saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract point cloud from Gaussian weights')
    parser.add_argument('input_file', type=str, help='Path to input .pth file with Gaussian weights')
    parser.add_argument('--output_file', type=str, help='Path to output .pth file for point cloud (default: <input>_point.pth)')
    parser.add_argument('--downsample', type=float, default=1.0, help='Downsample factor (0-1.0) to control point cloud density')
    parser.add_argument('--vis_output', type=str, help='Path to save visualization (if --visualize is set)')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output_file is None:
        input_base = os.path.splitext(args.input_file)[0]
        args.output_file = f"{input_base}_point.pth"
    
    # Load Gaussian weights
    gaussian_data = load_gaussian_file(args.input_file)
    
    # Extract and downsample point cloud
    point_cloud = extract_point_cloud(gaussian_data, args.downsample)
    
    # Save point cloud
    save_point_cloud(point_cloud, args.output_file)

if __name__ == "__main__":
    main()