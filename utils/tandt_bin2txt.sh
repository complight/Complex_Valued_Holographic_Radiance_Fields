#!/bin/bash

# Set the root directory
ROOT_PATH="/hy-tmp/echoRealm/3DGS_pytorch"

# Set dataset name and paths
DATASET_NAME="truck"  # Change to "truck" if needed
DATA_PATH="$ROOT_PATH/data/tandt/$DATASET_NAME"

COLMAP_FOLDER="sparse"

# Define input and output paths
INPUT_DIR="$DATA_PATH/$COLMAP_FOLDER/0"
OUTPUT_DIR="$DATA_PATH/$COLMAP_FOLDER/colmap_text"
IMAGES_DIR="$DATA_PATH/images"

# Check if directories exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found at $INPUT_DIR"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi

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

if [ $? -ne 0 ]; then
    echo "Error: COLMAP model conversion failed"
    exit 1
fi

echo "Conversion completed. Text files are saved in $OUTPUT_DIR"
