#!/bin/bash

# Set the root directory
ROOT_PATH="/hy-tmp/echoRealm/3DGS_pytorch"

# Set paths to data and COLMAP directories
DATA_PATH="$ROOT_PATH/data/colmap/colmap_lego"

COLMAP_FOLDER="colmap"
DATASET_TYPE="image"
# Define input and output paths
INPUT_DIR="$DATA_PATH/$COLMAP_FOLDER/sparse/0"
OUTPUT_DIR="$DATA_PATH/$COLMAP_FOLDER/sparse/colmap_text"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Converting COLMAP binary files to text format..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run model converter for each binary file
colmap model_converter \
    --input_path "$INPUT_DIR" \
    --output_path "$OUTPUT_DIR" \
    --output_type TXT

echo "Conversion completed. Text files are saved in $OUTPUT_DIR"
