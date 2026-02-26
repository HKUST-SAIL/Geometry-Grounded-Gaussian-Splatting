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

#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <torch/extension.h>
#include <tuple>

template <bool ZEROS = false>
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        if constexpr (ZEROS) {
            t.zero_();
        }
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
    const bool debug) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts   = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color  = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor out_mdepth = torch::full({1, H, W}, 0.0, float_opts);
    torch::Tensor out_alpha  = torch::full({1, H, W}, 0.0, float_opts);
    torch::Tensor out_normal = torch::full({3, H, W}, 0.0, float_opts);
    torch::Tensor radii      = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer                 = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer              = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer                  = torch::empty({0}, options.device(device));
    torch::Tensor tileBuffer                 = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc    = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc     = resizeFunctional(imgBuffer);
    std::function<char*(size_t)> tileFunc    = resizeFunctional(tileBuffer);

    int rendered = 0;
    if (P != 0) {
        int SHM = 0, SGM = 0;
        if (sh.size(0) != 0) {
            SHM = sh.size(1);
        }
        if (sg_color.size(0) != 0) {
            SGM = sg_color.size(1);
        }

        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            tileFunc,
            P, sh_degree, SHM, sg_degree, SGM,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            sg_axis.contiguous().data_ptr<float>(),
            sg_sharpness.contiguous().data_ptr<float>(),
            sg_color.contiguous().data_ptr<float>(),
            scale_modifier,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            kernel_size,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),
            out_mdepth.contiguous().data_ptr<float>(),
            out_alpha.contiguous().data_ptr<float>(),
            out_normal.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(),
            require_depth,
            debug);
    }
    return std::make_tuple(rendered, out_color, out_alpha, out_normal, out_mdepth, radii, geomBuffer, binningBuffer, imgBuffer, tileBuffer);
}

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
    const bool debug) {
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    int SHM = 0, SGM = 0;
    if (sh.size(0) != 0) {
        SHM = sh.size(1);
    }
    if (sg_color.size(0) != 0) {
        SGM = sg_color.size(1);
    }

    torch::Tensor dL_dmeans3D      = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D      = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors       = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dopacity      = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D        = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dsh           = torch::zeros({P, SHM, 3}, means3D.options());
    torch::Tensor dL_dsg_axis      = torch::zeros({P, SGM, 3}, means3D.options());
    torch::Tensor dL_dsg_sharpness = torch::zeros({P, SGM}, means3D.options());
    torch::Tensor dL_dsg_color     = torch::zeros({P, SGM, 3}, means3D.options());
    torch::Tensor dL_dscales       = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations    = torch::zeros({P, 4}, means3D.options());

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBwdBuffer           = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional<true>(geomBwdBuffer);

    if (P != 0) {
        CudaRasterizer::Rasterizer::backward(
            geomFunc,
            P, sh_degree, SHM, sg_degree, SGM, R,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.data_ptr<float>(),
            rotations.data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            sg_axis.contiguous().data_ptr<float>(),
            sg_sharpness.contiguous().data_ptr<float>(),
            sg_color.contiguous().data_ptr<float>(),
            scale_modifier,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            kernel_size,
            radii.contiguous().data_ptr<int>(),
            alphas.contiguous().data_ptr<float>(),
            normalmap.contiguous().data_ptr<float>(),
            mdepth.contiguous().data_ptr<float>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(tileBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data_ptr<float>(),
            dL_dout_mdepth.contiguous().data_ptr<float>(),
            dL_dout_alpha.contiguous().data_ptr<float>(),
            dL_dout_normal.contiguous().data_ptr<float>(),
            dL_dmeans3D.contiguous().data_ptr<float>(),
            dL_dmeans2D.contiguous().data_ptr<float>(),
            dL_dcolors.contiguous().data_ptr<float>(),
            dL_dopacity.contiguous().data_ptr<float>(),
            dL_dscales.contiguous().data_ptr<float>(),
            dL_drotations.contiguous().data_ptr<float>(),
            dL_dcov3D.contiguous().data_ptr<float>(),
            dL_dsh.contiguous().data_ptr<float>(),
            dL_dsg_axis.contiguous().data_ptr<float>(),
            dL_dsg_sharpness.contiguous().data_ptr<float>(),
            dL_dsg_color.contiguous().data_ptr<float>(),
            require_depth,
            debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dsg_axis, dL_dsg_sharpness, dL_dsg_color, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix) {
    const int P = means3D.size(0);

    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

    if (P != 0) {
        CudaRasterizer::Rasterizer::markVisible(P,
                                                means3D.contiguous().data_ptr<float>(),
                                                viewmatrix.contiguous().data_ptr<float>(),
                                                projmatrix.contiguous().data_ptr<float>(),
                                                present.contiguous().data_ptr<bool>());
    }

    return present;
}

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
    const bool debug) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    if (points3D.ndimension() != 2 || points3D.size(1) != 3) {
        AT_ERROR("points3D must have dimensions (num_points, 3)");
    }

    const int PN = points3D.size(0);
    const int P  = means3D.size(0);
    const int H  = image_height;
    const int W  = image_width;

    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto bool_opts  = means3D.options().dtype(torch::kBool);

    torch::Tensor out_transmittance = torch::full({PN}, 0.0, float_opts);
    torch::Tensor inside            = torch::full({PN}, 0, bool_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer        = torch::empty({0}, options.device(device));
    torch::Tensor pointBuffer          = torch::empty({0}, options.device(device));
    torch::Tensor point_binningBuffer  = torch::empty({0}, options.device(device));
    torch::Tensor tileBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor duplicatedtileBuffer = torch::empty({0}, options.device(device));

    std::function<char*(size_t)> geomFunc           = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc        = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> pointFunc          = resizeFunctional(pointBuffer);
    std::function<char*(size_t)> point_binningFunc  = resizeFunctional(point_binningBuffer);
    std::function<char*(size_t)> tileFunc           = resizeFunctional(tileBuffer);
    std::function<char*(size_t)> duplicatedtileFunc = resizeFunctional(duplicatedtileBuffer);

    int rendered = 0;
    if (P != 0 && PN != 0) {
        rendered = CudaRasterizer::Rasterizer::evaluateTransmittance(
            geomFunc,
            binningFunc,
            pointFunc,
            point_binningFunc,
            tileFunc,
            duplicatedtileFunc,
            PN, P,
            W, H,
            points3D.contiguous().data_ptr<float>(),
            means3D.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            kernel_size,
            prefiltered,
            out_transmittance.contiguous().data_ptr<float>(),
            inside.contiguous().data_ptr<bool>(),
            debug);
    }
    return std::make_tuple(rendered, out_transmittance, inside);
}

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
    const bool debug) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    if (points3D.ndimension() != 2 || points3D.size(1) != 3) {
        AT_ERROR("points3D must have dimensions (num_points, 3)");
    }

    const int PN = points3D.size(0);
    const int P  = means3D.size(0);
    const int H  = image_height;
    const int W  = image_width;

    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto bool_opts  = means3D.options().dtype(torch::kBool);

    torch::Tensor out_depth = torch::full({PN}, 0.0, float_opts);
    torch::Tensor out_sdf   = torch::full({PN}, 0.0, float_opts);
    torch::Tensor inside    = torch::full({PN}, 0, bool_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer        = torch::empty({0}, options.device(device));
    torch::Tensor pointBuffer          = torch::empty({0}, options.device(device));
    torch::Tensor point_binningBuffer  = torch::empty({0}, options.device(device));
    torch::Tensor tileBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor duplicatedtileBuffer = torch::empty({0}, options.device(device));

    std::function<char*(size_t)> geomFunc           = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc        = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> pointFunc          = resizeFunctional(pointBuffer);
    std::function<char*(size_t)> point_binningFunc  = resizeFunctional(point_binningBuffer);
    std::function<char*(size_t)> tileFunc           = resizeFunctional(tileBuffer);
    std::function<char*(size_t)> duplicatedtileFunc = resizeFunctional(duplicatedtileBuffer);

    int rendered = 0;
    if (P != 0 && PN != 0) {
        rendered = CudaRasterizer::Rasterizer::evaluateSDF(
            geomFunc,
            binningFunc,
            pointFunc,
            point_binningFunc,
            tileFunc,
            duplicatedtileFunc,
            PN, P,
            W, H,
            points3D.contiguous().data_ptr<float>(),
            means3D.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            kernel_size,
            prefiltered,
            out_depth.contiguous().data_ptr<float>(),
            out_sdf.contiguous().data_ptr<float>(),
            inside.contiguous().data_ptr<bool>(),
            debug);
    }
    return std::make_tuple(rendered, out_depth, out_sdf, inside);
}

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
    const bool debug) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    if (points3D.ndimension() != 2 || points3D.size(1) != 3) {
        AT_ERROR("points3D must have dimensions (num_points, 3)");
    }

    const int PN = points3D.size(0);
    const int P  = means3D.size(0);
    const int H  = image_height;
    const int W  = image_width;

    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto bool_opts  = means3D.options().dtype(torch::kBool);

    torch::Tensor out_color = torch::full({PN, NUM_CHANNELS}, 0.0, float_opts);
    torch::Tensor inside    = torch::full({PN}, 0, bool_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer        = torch::empty({0}, options.device(device));
    torch::Tensor pointBuffer          = torch::empty({0}, options.device(device));
    torch::Tensor point_binningBuffer  = torch::empty({0}, options.device(device));
    torch::Tensor tileBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor duplicatedtileBuffer = torch::empty({0}, options.device(device));

    std::function<char*(size_t)> geomFunc           = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc        = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> pointFunc          = resizeFunctional(pointBuffer);
    std::function<char*(size_t)> point_binningFunc  = resizeFunctional(point_binningBuffer);
    std::function<char*(size_t)> tileFunc           = resizeFunctional(tileBuffer);
    std::function<char*(size_t)> duplicatedtileFunc = resizeFunctional(duplicatedtileBuffer);

    int rendered = 0;
    if (P != 0 && PN != 0) {
        int SHM = 0, SGM = 0;
        if (sh.size(0) != 0) {
            SHM = sh.size(1);
        }
        if (sg_color.size(0) != 0) {
            SGM = sg_color.size(1);
        }
        rendered = CudaRasterizer::Rasterizer::evaluateColor(
            geomFunc,
            binningFunc,
            pointFunc,
            point_binningFunc,
            tileFunc,
            duplicatedtileFunc,
            PN, P, sh_degree, SHM, sg_degree, SGM,
            background.contiguous().data_ptr<float>(),
            W, H,
            points3D.contiguous().data_ptr<float>(),
            means3D.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            sg_axis.contiguous().data_ptr<float>(),
            sg_sharpness.contiguous().data_ptr<float>(),
            sg_color.contiguous().data_ptr<float>(),
            scale_modifier,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            kernel_size,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),
            inside.contiguous().data_ptr<bool>(),
            debug);
    }
    return std::make_tuple(rendered, out_color, inside);
}

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
    const bool debug) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    if (points3D.ndimension() < 1 || points3D.size(-1) != 3) {
        AT_ERROR("points3D must have shape (..., 3) with last dimension == 3");
    }

    const int PN = points3D.numel() / 3;
    const int P  = means3D.size(0);
    const int H  = image_height;
    const int W  = image_width;

    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto bool_opts  = means3D.options().dtype(torch::kBool);

    torch::Tensor output = torch::zeros_like(points3D);
    std::vector<int64_t> inside_sizes(points3D.sizes().begin(), points3D.sizes().end() - 1);
    torch::Tensor inside = torch::zeros(inside_sizes, bool_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer        = torch::empty({0}, options.device(device));
    torch::Tensor pointBuffer          = torch::empty({0}, options.device(device));
    torch::Tensor point_binningBuffer  = torch::empty({0}, options.device(device));
    torch::Tensor tileBuffer           = torch::empty({0}, options.device(device));
    torch::Tensor duplicatedtileBuffer = torch::empty({0}, options.device(device));

    std::function<char*(size_t)> geomFunc           = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc        = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> pointFunc          = resizeFunctional(pointBuffer);
    std::function<char*(size_t)> point_binningFunc  = resizeFunctional(point_binningBuffer);
    std::function<char*(size_t)> tileFunc           = resizeFunctional(tileBuffer);
    std::function<char*(size_t)> duplicatedtileFunc = resizeFunctional(duplicatedtileBuffer);

    int rendered         = 0;
    int valid_points     = 0;
    int duplicated_tiles = 0;
    if (P != 0 && PN != 0) {
        int3 n = CudaRasterizer::Rasterizer::sampleDepth(
            geomFunc,
            binningFunc,
            pointFunc,
            point_binningFunc,
            tileFunc,
            duplicatedtileFunc,
            PN, P,
            W, H,
            points3D.contiguous().data_ptr<float>(),
            means3D.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            kernel_size,
            prefiltered,
            output.contiguous().data_ptr<float>(),
            inside.contiguous().data_ptr<bool>(),
            debug);
        rendered         = n.x;
        valid_points     = n.y;
        duplicated_tiles = n.z;
    }
    return std::make_tuple(rendered, valid_points, duplicated_tiles, output, inside, geomBuffer, binningBuffer, pointBuffer, point_binningBuffer, tileBuffer, duplicatedtileBuffer);
}

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
    const bool debug) {
    const int PN = points3D.numel() / 3;
    const int P  = means3D.size(0);

    auto float_opts             = means3D.options().dtype(torch::kFloat32);
    torch::Tensor dL_dmeans3D   = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D   = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dopacity   = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D     = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dscales    = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
    // torch::Tensor dL_dpoint3D   = torch::zeros({PN, 3}, means3D.options());
    torch::Tensor dL_dpoint3D = torch::zeros_like(points3D);
    torch::Tensor dL_dpoint2D = torch::zeros({PN, 2}, means3D.options());

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBwdBuffer           = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional<true>(geomBwdBuffer);
    if (P != 0 && PN != 0) {
        CudaRasterizer::Rasterizer::sampleDepthBackward(
            geomFunc,
            PN, P, RN, R, TN,
            image_width, image_height,
            points3D.contiguous().data_ptr<float>(),
            means3D.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            kernel_size,
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(pointBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(pointBinningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(tileBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(duplicatedtileBuffer.contiguous().data_ptr()),
            inside.contiguous().data_ptr<bool>(),
            dL_doutput.contiguous().data_ptr<float>(),
            dL_dmeans2D.contiguous().data_ptr<float>(),
            dL_dpoint2D.contiguous().data_ptr<float>(),
            dL_dopacity.contiguous().data_ptr<float>(),
            dL_dmeans3D.contiguous().data_ptr<float>(),
            dL_dcov3D.contiguous().data_ptr<float>(),
            dL_dscales.contiguous().data_ptr<float>(),
            dL_drotations.contiguous().data_ptr<float>(),
            dL_dpoint3D.contiguous().data_ptr<float>(),
            debug);
    }
    return std::make_tuple(dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dscales, dL_drotations, dL_dpoint3D);
}