#!/bin/bash
# This script is used for estimating cam pos using colmap
# This is needed if using original llff dataset, 
# if you use colmap to estimate llff with pinhole camera by yourself, you dont need this script.
# Set input and output paths
dataset="nerf_llff_data"
dataset_name="fern"
IMAGE_DIR="./data/$dataset/$dataset_name/images"
SPARSE_DIR="./data/$dataset/$dataset_name/sparse"

UNDISTORT_DIR="./data/$dataset/$dataset_name/undistorted"
mkdir -p "$UNDISTORT_DIR"
colmap image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$UNDISTORT_DIR" \
    --output_type COLMAP \
    --max_image_size 2000

TXT_DIR="./data/$dataset/$dataset_name/undistorted/colmap_text"
mkdir -p "$TXT_DIR"
colmap model_converter \
    --input_path "$UNDISTORT_DIR/sparse/" \
    --output_path "$TXT_DIR" \
    --output_type TXT