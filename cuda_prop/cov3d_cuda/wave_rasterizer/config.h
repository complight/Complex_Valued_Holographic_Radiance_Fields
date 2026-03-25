/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS_RGB 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
#define GAUSSIAN_DISTANCE_CUTOFF 1000.0f
#define PLANE_PROBABILITY_CUTOFF 1e-6f
#define DETERMINANT_CUTOFF 1e-6f
#define MAX_ALPHA_VALUE 0.99f
#define MAX_CHANNELS 3
#define MAX_PLANES 8

#endif