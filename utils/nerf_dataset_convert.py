import os
import json
import numpy as np
import torch
from PIL import Image
import math
from tqdm import tqdm

def convert_dataset(dataset, json_root, image_root, output_dir, image_size):
    # Load all frames
    splits = ['train', 'val', 'test']
    all_frames = []
    for split in splits:
        with open(os.path.join(json_root, f'transforms_{split}.json'), 'r') as f:
            data = json.load(f)
            for frame in data['frames']:
                frame['split'] = split
            all_frames.extend(data['frames'])

    # Get original image dimensions
    img_path = os.path.join(image_root, all_frames[0]['file_path'].replace('./', '') + '.png')
    with Image.open(img_path) as img:
        W_orig, H_orig = img.size
    H_new, W_new = image_size
    # Load camera parameters
    with open(os.path.join(json_root, 'transforms_train.json'), 'r') as f:
        train_data = json.load(f)
        camera_angle_x = train_data['camera_angle_x']

    # Calculate normalized focal length (matches original format)
    focal_orig = 0.5 * W_orig / math.tan(0.5 * camera_angle_x)
    scale_factor = image_size[1] / W_orig  # Use width for scaling
    focal_length = (focal_orig / W_orig) * 2.0  # Normalized format

    # Process camera matrices
    Rs, Ts = [], []
    for frame in tqdm(all_frames, desc='Processing cameras'):
        c2w = np.array(frame['transform_matrix'])
        
        c2w[:3, [0, 2]] *= -1  

        R = c2w[:3, :3]
        T = c2w[:3, 3]

        # Convert to world->camera transformation
        R_w2c = R.T
        T_w2c = -R_w2c @ T
        
        Rs.append(R)
        Ts.append(T_w2c)

    cameras = {
        'focal_length': torch.full((len(all_frames), 2), float(focal_length), dtype=torch.float32),
        'principal_point': torch.zeros((len(all_frames), 2), dtype=torch.float32),  # Original uses (0,0)
        'R': torch.tensor(np.stack(Rs), dtype=torch.float32),
        'T': torch.tensor(np.stack(Ts), dtype=torch.float32)
    }

    # Create split indices
    split_indices = []
    current_idx = 0
    for split in splits:
        with open(os.path.join(json_root, f'transforms_{split}.json'), 'r') as f:
            n = len(json.load(f)['frames'])
            split_indices.append(torch.arange(current_idx, current_idx + n, dtype=torch.int64))
            current_idx += n

    # Save data
    data = {'cameras': cameras, 'split': split_indices}
    os.makedirs(output_dir, exist_ok=True)
    torch.save(data, os.path.join(output_dir, f'{dataset}.pth'))

    
    # Process and save images
    image_paths = []
    for frame in all_frames:
        img_path = os.path.join(image_root, frame['file_path'] + '.png')
        image_paths.append(img_path)
    
    # Stack images vertically
    combined = []
    for path in tqdm(image_paths, desc='Processing images'):
        img = Image.open(path).convert('RGB')
        img = img.resize((W_new, H_new), Image.BILINEAR)
        combined.append(np.array(img))
    
    combined_image = np.concatenate(combined, axis=0)
    Image.fromarray(combined_image).save(os.path.join(output_dir, f'{dataset}.png'))
    
    print(f"Conversion complete. Files saved to {output_dir}")

dataset = "ship"
print(dataset)
convert_dataset(
    dataset = dataset,
    json_root=f'../data/nerf_synthetic/{dataset}',
    image_root=f'../data/nerf_synthetic/{dataset}',
    output_dir=f"../data/{dataset}",
    image_size=(800, 800)
)


# file_path = "/hy-tmp/echoRealm/3DGS_pytorch/data/materials/materials.pth"
# data = torch.load(file_path)

# # Print the structure of the data and the first item's value where applicable
# def print_structure(data, indent=0):
#     """Recursively prints the structure of the loaded .pth file with sample values"""
#     indent_str = "  " * indent
#     if isinstance(data, dict):
#         print(f"{indent_str}Dict with {len(data)} keys:")
#         for key, value in data.items():
#             print(f"{indent_str}  Key: {key} -> Type: {type(value)}")
#             if isinstance(value, (list, dict, torch.Tensor)):
#                 print_structure(value, indent + 1)  # Recursively print nested structures
#             else:
#                 print(f"{indent_str}    Sample Value: {value}")  # Print first item value
#     elif isinstance(data, list):
#         print(f"{indent_str}List with {len(data)} elements")
#         if len(data) > 0:
#             print(f"{indent_str}  First Item Type: {type(data[0])}")
#             print_structure(data[0], indent + 1)  # Print only the first element for brevity
#     elif isinstance(data, torch.Tensor):
#         print(f"{indent_str}Tensor with shape: {data.shape}, dtype: {data.dtype}")
#         print(f"{indent_str}  First Few Values: {data.flatten()[:5].tolist()}")  # Show first few values
#     else:
#         print(f"{indent_str}{type(data)}: {data}")

# print_structure(data)

