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

#pragma once
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& sh,
    const torch::Tensor& sg_axis,
    const torch::Tensor& sg_sharpness,
    const torch::Tensor& sg_color,
    const int sh_degree,
    const int sg_degree,
    const float scale_modifier,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool require_depth,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& sh,
    const torch::Tensor& sg_axis,
    const torch::Tensor& sg_sharpness,
    const torch::Tensor& sg_color,
    const int sh_degree,
    const int sg_degree,
    const float scale_modifier,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_mdepth,
    const torch::Tensor& dL_dout_alpha,
    const torch::Tensor& dL_dout_normal,
    const torch::Tensor& alphas,
    const torch::Tensor& normalmap,
    const torch::Tensor& mdepth,
    const torch::Tensor& campos,
    const torch::Tensor& radii,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& tileBuffer,
    const bool require_depth,
    const bool debug);

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix);

std::tuple<int, torch::Tensor, torch::Tensor>
evaluateTransmittancefromSingleView(
    const torch::Tensor& points3D,
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor>
evaluateSDFfromSingleView(
    const torch::Tensor& points3D,
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor>
evaluateColorfromSingleView(
    const torch::Tensor& points3D,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& sh,
    const torch::Tensor& sg_axis,
    const torch::Tensor& sg_sharpness,
    const torch::Tensor& sg_color,
    const int sh_degree,
    const int sg_degree,
    const float scale_modifier,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug);

std::tuple<int, int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleRasterizedDepthCUDA(
    const torch::Tensor& points3D,
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SampleRasterizedDepthBackwardCUDA(
    const torch::Tensor& points3D,
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& inside,
    const torch::Tensor& dL_doutput,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& pointBuffer,
    const torch::Tensor& pointBinningBuffer,
    const torch::Tensor& tileBuffer,
    const torch::Tensor& duplicatedtileBuffer,
    int R, int RN, int TN,
    const bool prefiltered,
    const bool debug);