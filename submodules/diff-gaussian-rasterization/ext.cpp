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

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("evaluate_transmittance_from_signle_view", &evaluateTransmittancefromSingleView);
  m.def("evaluate_sdf_from_signle_view", &evaluateSDFfromSingleView);
  m.def("evaluate_color_from_signle_view", &evaluateColorfromSingleView);
  m.def("mark_visible", &markVisible);
  m.def("sample_rasterized_depth", &SampleRasterizedDepthCUDA);
  m.def("sample_rasterized_depth_backward", &SampleRasterizedDepthBackwardCUDA);
}