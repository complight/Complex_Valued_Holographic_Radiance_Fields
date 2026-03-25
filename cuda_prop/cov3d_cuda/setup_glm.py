import os
import sys
import subprocess
import shutil
import zipfile
import urllib.request
'''
This file is used in install.py to set up the GLM library.
'''
def setup_glm():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a third_party directory if it doesn't exist
    third_party_dir = os.path.join(current_dir, 'third_party')
    os.makedirs(third_party_dir, exist_ok=True)
    
    # GLM repository URL for the latest stable version
    glm_url = "https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip"
    zip_path = os.path.join(third_party_dir, "glm-0.9.9.8.zip")
    
    # Check if GLM is already downloaded
    glm_dir = os.path.join(third_party_dir, "glm")
    if os.path.exists(glm_dir) and os.path.exists(os.path.join(glm_dir, "glm", "glm.hpp")):
        print("GLM already installed.")
        return glm_dir
    
    # Download GLM if necessary
    if not os.path.exists(zip_path):
        print(f"Downloading GLM from {glm_url}...")
        urllib.request.urlretrieve(glm_url, zip_path)
    
    # Extract GLM
    print("Extracting GLM...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(third_party_dir)
    
    # The extraction creates a glm directory with the version number
    # Rename it to just "glm" for simplicity
    extracted_dirs = [d for d in os.listdir(third_party_dir) if d.startswith('glm-') and os.path.isdir(os.path.join(third_party_dir, d))]
    if extracted_dirs:
        extracted_dir = os.path.join(third_party_dir, extracted_dirs[0])
        if os.path.exists(glm_dir):
            shutil.rmtree(glm_dir)
        shutil.move(extracted_dir, glm_dir)
    
    print(f"GLM installed at: {glm_dir}")
    return glm_dir

if __name__ == "__main__":
    setup_glm()
