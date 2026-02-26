#ifndef CUDA_RASTERIZER_SAMPLE_H_INCLUDED
#define CUDA_RASTERIZER_SAMPLE_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
// follow code is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields
//  Perform initial steps for each Point prior to integration.
void preprocess_points(
    int PN,
    const float* points3D,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    float2* points2D,
    float* ts,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered);

void evaluateSDF(
    const int num_duplicated_tiles,
    const uint32_t* tile_ids,
    const uint32_t* tile_offsets,
    const uint2* gaussian_ranges,
    const uint2* point_ranges,
    const uint32_t* gaussian_list,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float2* points2D,
    const float* point_ts,
    const float2* gaussians2D,
    const float4* conic_opacity,
    const float4* ray_planes,
    const float3* normals,
    float* median_depth,
    float* output,
    bool* inside_output);

void evaluateTransmittance(
    const int num_duplicated_tiles,
    const uint32_t* tile_ids,
    const uint32_t* tile_offsets,
    const uint2* gaussian_ranges,
    const uint2* point_ranges,
    const uint32_t* gaussian_list,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float2* points2D,
    const float* point_depths,
    const float2* gaussians2D,
    const float4* conic_opacity,
    const float4* ray_planes,
    const float3* normals,
    float* out_transmittance,
    bool* inside);

void evaluateColor(
    const int num_duplicated_tiles,
    const uint32_t* tile_ids,
    const uint32_t* tile_offsets,
    const uint2* gaussian_ranges,
    const uint2* point_ranges,
    const uint32_t* gaussian_list,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float2* points2D,
    const float* point_depths,
    const float2* gaussians2D,
    const float4* conic_opacity,
    const float* colors,
    const float* bg_color,
    float* out_color,
    bool* inside);

void sampleDepth(
    const int num_duplicated_tiles,
    const uint32_t* tile_ids,
    const uint32_t* tile_offsets,
    const uint2* gaussian_ranges,
    const uint2* point_ranges,
    const uint32_t* gaussian_list,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float2* points2D,
    const float2* gaussians2D,
    const float4* conic_opacity,
    const float4* ray_planes,
    const float3* normals,
    uint32_t* n_contrib,
    float* median_depth,
    uint32_t* max_contributor,
    float3* output,
    bool* inside);
} // namespace FORWARD

#endif