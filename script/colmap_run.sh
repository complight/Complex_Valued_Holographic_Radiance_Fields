#!/bin/bash
# This script is used for estimating cam pos using colmap
# Set input and output paths
#  ------------------------- COLMAP ------------------------
# dataset="colmap"
# dataset_name="colmap_mic"
# IMAGE_DIR="./data/$dataset/$dataset_name/image"
# SPARSE_DIR="./data/$dataset/$dataset_name/colmap/sparse"
# TXT_DIR="./data/$dataset/$dataset_name/colmap/sparse/colmap_text"
# # (Optional) Convert to text for easier inspection
# TXT_DIR="./data/$dataset/$dataset_name/sparse/colmap_text"
#  ------------------------- LLFF ------------------------
# dataset="nerf_llff_data"
# dataset_name="trex"
# IMAGE_DIR="./data/$dataset/$dataset_name/images"
# SPARSE_DIR="./data/$dataset/$dataset_name/sparse"
# # (Optional) Convert to text for easier inspection
# TXT_DIR="./data/$dataset/$dataset_name/sparse/colmap_text"
dataset="nerf_llff_data"
dataset_name="room"
IMAGE_DIR="./data/$dataset/$dataset_name/imgs_path_all"
SPARSE_DIR="./data/$dataset/$dataset_name/sparse_path_all"
# (Optional) Convert to text for easier inspection
TXT_DIR="./data/$dataset/$dataset_name/sparse_path_all/colmap_text"
#  ------------------------- mip360 NeRF ------------------------
# dataset="mip360"
# dataset_name="garden"
# IMAGE_DIR="./data/$dataset/$dataset_name/images_spiral"
# SPARSE_DIR="./data/$dataset/$dataset_name/sparse_spiral"
# TXT_DIR="./data/$dataset/$dataset_name/sparse_spiral/colmap_text"

# Check image directory
if [ ! -d "$IMAGE_DIR" ]; then
  echo "ERROR: Image directory does not exist: $IMAGE_DIR"
  exit 1
fi

# Create output directory if it does not exist
mkdir -p "$SPARSE_DIR"

# Run feature extraction
colmap feature_extractor \
    --image_path "$IMAGE_DIR" \
    --database_path "$SPARSE_DIR/database.db" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

# Run exhaustive matcher
colmap exhaustive_matcher \
    --database_path "$SPARSE_DIR/database.db"

# Run sparse reconstruction
colmap mapper \
    --database_path "$SPARSE_DIR/database.db" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_DIR"


mkdir -p "$TXT_DIR"
colmap model_converter \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$TXT_DIR" \
    --output_type TXT


# #!/bin/bash
# # This script is used for estimating cam pos using colmap
# # Set input and output paths
# #  ------------------------- COLMAP ------------------------
# # dataset="colmap"
# # dataset_name="colmap_mic"
# # IMAGE_DIR="./data/$dataset/$dataset_name/image"
# # SPARSE_DIR="./data/$dataset/$dataset_name/colmap/sparse"
# # TXT_DIR="./data/$dataset/$dataset_name/colmap/sparse/colmap_text"
# #  ------------------------- LLFF ------------------------
# # dataset="nerf_llff_data"
# # dataset_name="flower"
# # IMAGE_DIR="./data/$dataset/$dataset_name/images"
# # SPARSE_DIR="./data/$dataset/$dataset_name/sparse"
# # TXT_DIR="./data/$dataset/$dataset_name/sparse/colmap_text"
# dataset="/hy-tmp/TensoRF/log/tensorf_flower_VM/tensorf_flower_VM"
# IMAGE_DIR="$dataset/imgs_path_all"
# SPARSE_DIR="$dataset/sparse_path_all"
# TXT_DIR="$dataset/sparse_path_all/colmap_text"
# #  ------------------------- mip360 NeRF ------------------------
# # dataset="mip360"
# # dataset_name="garden"
# # IMAGE_DIR="./data/$dataset/$dataset_name/images_spiral"
# # SPARSE_DIR="./data/$dataset/$dataset_name/sparse_spiral"
# # TXT_DIR="./data/$dataset/$dataset_name/sparse_spiral/colmap_text"

# mkdir -p "$SPARSE_DIR"

# # Extract more features
# colmap feature_extractor \
#     --image_path "$IMAGE_DIR" \
#     --database_path "$SPARSE_DIR/database.db" \
#     --ImageReader.camera_model PINHOLE \
#     --ImageReader.single_camera 1 \
#     --SiftExtraction.max_num_features 8192 \
#     --SiftExtraction.num_threads -1

# # Better matching with spatial verification
# colmap exhaustive_matcher \
#     --database_path "$SPARSE_DIR/database.db" \
#     --SiftMatching.guided_matching 1 \
#     --SiftMatching.max_ratio 0.85 \
#     --SiftMatching.max_distance 0.7

# # Less conservative triangulation
# colmap mapper \
#     --database_path "$SPARSE_DIR/database.db" \
#     --image_path "$IMAGE_DIR" \
#     --output_path "$SPARSE_DIR" \
#     --Mapper.init_min_tri_angle 4 \
#     --Mapper.filter_max_reproj_error 3 \
#     --Mapper.filter_min_tri_angle 1.5

# # Convert to text
# mkdir -p "$TXT_DIR"
# colmap model_converter \
#     --input_path "$SPARSE_DIR/0" \
#     --output_path "$TXT_DIR" \
#     --output_type TXT