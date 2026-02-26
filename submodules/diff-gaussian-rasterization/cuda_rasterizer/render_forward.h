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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(
    const int P, int SHD, int SHM, int SGD, int SGM,
    const float* means3D,
    const float* colors_precomp,
    const float* opacities,
    const glm::vec3* scales,
    const float4* rotations,
    const float* cov3D_precomp,
    const float* shs,
    const float* sg_axis,
    const float* sg_sharpness,
    const float* sg_color,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, const int H,
    const float focal_x, const float focal_y,
    const float tan_fovx, const float tan_fovy,
    const float kernel_size,
    int* radii,
    bool* clamped,
    float2* means2D,
    float* depths,
    float4* ray_planes,
    float3* normals,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered = false);

// Main rasterization method.
void render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float4* ray_planes,
    const float3* normals,
    const float focal_x,
    const float focal_y,
    uint32_t* n_contrib,
    uint32_t* max_contributor,
    const float* bg_color,
    float* out_color,
    float* out_alpha,
    float* out_normal,
    float* out_mdepth,
    float* normal_length,
    bool require_depth);
} // namespace FORWARD

#endif