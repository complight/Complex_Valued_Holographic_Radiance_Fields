# #!/bin/bash

# # Set the root directory
# ROOT_PATH="/hy-tmp/echoRealm/3DGS_pytorch"
# DATASET_NAME="garden" 

# # Set paths to data and COLMAP directories
# DATA_PATH="$ROOT_PATH/data/mip360/$DATASET_NAME"
# SPARSE_INPUT_DIR="$DATA_PATH/sparse/0"
# TEXT_OUTPUT_DIR="$DATA_PATH/sparse/colmap_text"
# IMAGES_DIR="$DATA_PATH/images"

# # Check if input directory exists
# if [ ! -d "$SPARSE_INPUT_DIR" ]; then
#     echo "Error: COLMAP sparse directory not found at $SPARSE_INPUT_DIR"
#     exit 1
# fi

# # Check if images directory exists
# if [ ! -d "$IMAGES_DIR" ]; then
#     echo "Error: Images directory not found at $IMAGES_DIR"
#     exit 1
# fi

# # Create the output directory if it doesn't exist
# mkdir -p "$TEXT_OUTPUT_DIR"

# echo "Converting COLMAP binary files to text format..."

# # Run model converter to convert binary files to text
# colmap model_converter \
#     --input_path "$SPARSE_INPUT_DIR" \
#     --output_path "$TEXT_OUTPUT_DIR" \
#     --output_type TXT

# if [ $? -ne 0 ]; then
#     echo "Error: COLMAP model conversion failed"
#     exit 1
# fi

# echo "COLMAP binary files successfully converted to text format"
# echo "Text files are saved in $TEXT_OUTPUT_DIR"


#!/bin/bash

# Set the root directory
ROOT_PATH="/hy-tmp/echoRealm/3DGS_pytorch"
DATASET_NAME="garden" 

# Set paths to data and COLMAP directories
DATA_PATH="$ROOT_PATH/data/mip360/$DATASET_NAME"
TEXT_OUTPUT_DIR="$DATA_PATH/vggt_sparse/colmap_text"  # Using the existing colmap_txt directory directly
IMAGES_DIR="$DATA_PATH/images_8"

# Check if the text format COLMAP directory exists
if [ ! -d "$TEXT_OUTPUT_DIR" ]; then
    echo "Error: COLMAP text directory not found at $TEXT_OUTPUT_DIR"
    exit 1
fi

# Check if images directory exists
if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi
