#!/bin/bash
# download.sh - Download and extract datasets for Complex-Valued Holographic Radiance Fields

set -e

mkdir -p data

echo "Downloading colmap dataset..."
gdown 1j290D7jdwF7CdIWMcBR0bqi4zc8kl3g1 -O data/nerf_llff_data.zip

echo "Downloading nerf_llff_data dataset..."
gdown 1Af7_qSSrpEuDNHBlOk7DBUv8wW_6xBPX -O data/colmap.zip

echo "Extracting colmap.zip..."
unzip data/colmap.zip -d data/

echo "Extracting nerf_llff_data.zip..."
unzip  data/nerf_llff_data.zip -d data/

echo "Cleaning up zip files..."
rm data/colmap.zip data/nerf_llff_data.zip

echo "Done. Datasets are ready in ./data/"
