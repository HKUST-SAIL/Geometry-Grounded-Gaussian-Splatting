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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rasterizer_impl.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <numeric>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "render_backward.h"
#include "render_forward.h"
#include "sample_backward.h"
#include "sample_forward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n) {
    uint32_t msb  = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
                             const float* orig_points,
                             const float* viewmatrix,
                             const float* projmatrix,
                             bool* present) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 p_org = {orig_points[idx * 3], orig_points[idx * 3 + 1], orig_points[idx * 3 + 2]};
    float3 p_view;
    present[idx] = in_frustum(p_org, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Generate no key/value pair for invisible Gaussians
    if (radii[idx] > 0) {
        // Find this Gaussian's offset in buffer for writing keys/values.
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;

        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // For each tile that the bounding rect overlaps, emit a
        // key/value pair. The key is |  tile ID  |      depth      |,
        // and the value is the ID of the Gaussian. Sorting the values
        // with this key yields Gaussian IDs in a list, such that they
        // are first sorted by tile and then by depth.
        for (int y = rect_min.y; y < rect_max.y; y++) {
            for (int x = rect_min.x; x < rect_max.x; x++) {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= *((uint32_t*)&depths[idx]);
                gaussian_keys_unsorted[off]   = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

__global__ void createWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    const uint32_t* tiles_touched,
    uint64_t* points_keys_unsorted,
    uint32_t* points_values_unsorted,
    dim3 grid) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Generate no key/value pair for invisible Points
    if (tiles_touched[idx] > 0) {
        // Find this Point's offset in buffer for writing keys/values.
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

        // determine the tile that the point is in
        const float2 p = points_xy[idx];
        int x          = min(grid.x - 1, max((int)0, (int)((p.x + 0.5f) / BLOCK_X)));
        int y          = min(grid.y - 1, max((int)0, (int)((p.y + 0.5f) / BLOCK_Y)));

        uint64_t key = y * grid.x + x;
        key <<= 32;
        points_keys_unsorted[off]   = key;
        points_values_unsorted[off] = idx;
    }
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, const uint64_t* point_list_keys, uint2* ranges) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key      = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    if (idx == 0)
        ranges[currtile].x = 0;
    else {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile) {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

__global__ void countPointBatches(int B, const uint2* ranges, uint32_t* tile_rounds) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= B)
        return;
    constexpr int BLOCK_SAMPLES_PRE_ROUND = BLOCK_SIZE * SAMPLE_BATCH_SIZE;

    uint2 range      = ranges[idx];
    const int rounds = (range.y - range.x + BLOCK_SAMPLES_PRE_ROUND - 1) / BLOCK_SAMPLES_PRE_ROUND;
    tile_rounds[idx] = rounds;
}

__global__ void setBlockId(int B, const uint32_t* tile_offsets, const uint32_t* tile_rounds, uint32_t* tile_ids) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= B)
        return;
    uint32_t off     = (idx == 0) ? 0 : tile_offsets[idx - 1];
    const int rounds = tile_rounds[idx];
    for (int i = 0; i < rounds; i++) {
        tile_ids[off++] = idx;
    }
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
    int P,
    float* means3D,
    float* viewmatrix,
    float* projmatrix,
    bool* present) {
    checkFrustum<<<(P + 255) / 256, 256>>>(
        P,
        means3D,
        viewmatrix, projmatrix,
        present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P) {
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.ray_planes, P, 128);
    obtain(chunk, geom.normals, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

CudaRasterizer::GeometryBwdState CudaRasterizer::GeometryBwdState::fromChunk(char*& chunk, size_t P) {
    GeometryBwdState geom;
    obtain(chunk, geom.ray_planes, P, 128);
    obtain(chunk, geom.normals, P, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    return geom;
}

CudaRasterizer::PointState CudaRasterizer::PointState::fromChunk(char*& chunk, size_t P) {
    PointState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.points2D, P, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    obtain(chunk, geom.n_contrib, P, 128);
    obtain(chunk, geom.median_depth, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

CudaRasterizer::TileState<true> CudaRasterizer::TileState<true>::fromChunk(char*& chunk, size_t N) {
    TileState<true> tiles;
    obtain(chunk, tiles.gaussian_ranges, N, 128);
    obtain(chunk, tiles.point_ranges, N, 128);
    obtain(chunk, tiles.tile_rounds, N, 128);
    obtain(chunk, tiles.tile_offsets, N, 128);
    cub::DeviceScan::InclusiveSum(nullptr, tiles.scan_size, tiles.tile_rounds, tiles.tile_rounds, N);
    obtain(chunk, tiles.scanning_space, tiles.scan_size, 128);
    return tiles;
}

CudaRasterizer::TileState<false> CudaRasterizer::TileState<false>::fromChunk(char*& chunk, size_t N) {
    TileState<false> tiles;
    obtain(chunk, tiles.gaussian_ranges, N, 128);
    obtain(chunk, tiles.max_contributor, N, 128);
    return tiles;
}

CudaRasterizer::DuplicatedTileState CudaRasterizer::DuplicatedTileState::fromChunk(char*& chunk, size_t N) {
    DuplicatedTileState duplicated_tiles;
    obtain(chunk, duplicated_tiles.tile_id, N, 128);
    obtain(chunk, duplicated_tiles.max_contributor, N, 128);
    return duplicated_tiles;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N) {
    ImageState img;
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.normal_length, N, 128);
    return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P) {
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> imageBuffer,
    std::function<char*(size_t)> tileBuffer,
    const int P, int SHD, int SHM, int SGD, int SGM,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    const float* rotations,
    const float* cov3D_precomp,
    const float* shs,
    const float* sg_axis,
    const float* sg_sharpness,
    const float* sg_color,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const bool prefiltered,
    float* out_color,
    float* out_mdepth,
    float* out_alpha,
    float* out_normal,
    int* radii,
    bool require_depth,
    bool debug) {
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    const int tiles = tile_grid.x * tile_grid.y;

    size_t chunk_size       = required<GeometryState>(P);
    char* chunkptr          = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    size_t tile_chunk_size     = required<TileState<false>>(tiles);
    char* tile_chunkptr        = tileBuffer(tile_chunk_size);
    TileState<false> tileState = TileState<false>::fromChunk(tile_chunkptr, tiles);

    // Dynamically resize image-based auxiliary buffers during training
    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr    = imageBuffer(img_chunk_size);
    ImageState imgState   = ImageState::fromChunk(img_chunkptr, width * height);

    if (radii == nullptr) {
        radii = geomState.internal_radii;
    }

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }

    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    CHECK_CUDA(FORWARD::preprocess(
                   P, SHD, SHM, SGD, SGM,
                   means3D,
                   colors_precomp,
                   opacities,
                   (glm::vec3*)scales,
                   (float4*)rotations,
                   cov3D_precomp,
                   shs,
                   sg_axis,
                   sg_sharpness,
                   sg_color,
                   scale_modifier,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   kernel_size,
                   radii,
                   geomState.clamped,
                   geomState.means2D,
                   geomState.depths,
                   geomState.ray_planes,
                   geomState.normals,
                   geomState.rgb,
                   geomState.conic_opacity,
                   tile_grid,
                   geomState.tiles_touched,
                   prefiltered),
               debug)
    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr    = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid);
    CHECK_CUDA(, debug);

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   binningState.list_sorting_space,
                   binningState.sorting_size,
                   binningState.point_list_keys_unsorted, binningState.point_list_keys,
                   binningState.point_list_unsorted, binningState.point_list,
                   num_rendered, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.gaussian_ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.gaussian_ranges);
    CHECK_CUDA(, debug);

    // Let each tile blend its range of Gaussians independently in parallel
    const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(
                   tile_grid, block,
                   tileState.gaussian_ranges,
                   binningState.point_list,
                   width, height,
                   geomState.means2D,
                   geomState.conic_opacity,
                   feature_ptr,
                   geomState.ray_planes,
                   geomState.normals,
                   focal_x, focal_y,
                   imgState.n_contrib,
                   tileState.max_contributor,
                   background,
                   out_color,
                   out_alpha,
                   out_normal,
                   out_mdepth,
                   imgState.normal_length,
                   require_depth),
               debug);
    return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
    std::function<char*(size_t)> geometryBuffer,
    const int P, int SHD, int SHM, int SGD, int SGM, int R,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    const float* rotations,
    const float* cov3D_precomp,
    const float* shs,
    const float* sg_axis,
    const float* sg_sharpness,
    const float* sg_color,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const float* campos,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const int* radii,
    const float* alphas,
    const float* normalmap,
    const float* mdepth,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    char* tile_buffer,
    const float* dL_dpix,
    const float* dL_dpix_mdepth,
    const float* dL_dalphas,
    const float* dL_dpixel_normals,
    float* dL_dmean3D,
    float* dL_dmean2D,
    float* dL_dcolor,
    float* dL_dopacity,
    float* dL_dscale,
    float* dL_drot,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dsg_axis,
    float* dL_dsg_sharpness,
    float* dL_dsg_color,
    bool require_depth,
    bool debug) {
    GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    ImageState imgState       = ImageState::fromChunk(img_buffer, width * height);

    size_t chunk_size             = required<GeometryBwdState>(P);
    char* chunkptr                = geometryBuffer(chunk_size);
    GeometryBwdState geomBwdState = GeometryBwdState::fromChunk(chunkptr, P);

    if (radii == nullptr) {
        radii = geomState.internal_radii;
    }

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    TileState<false> tileState = TileState<false>::fromChunk(tile_buffer, tile_grid.x * tile_grid.y);

    // Compute loss gradients w.r.t. 2D mean position, conic matrix,
    // opacity and RGB of Gaussians from per-pixel loss gradients.
    // If we were given precomputed colors and not SHs, use them.
    const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    CHECK_CUDA(BACKWARD::render(
                   tile_grid,
                   block,
                   tileState.gaussian_ranges,
                   binningState.point_list,
                   width, height,
                   background,
                   geomState.means2D,
                   geomState.conic_opacity,
                   color_ptr,
                   geomState.depths,
                   geomState.ray_planes,
                   geomState.normals,
                   alphas,
                   normalmap,
                   mdepth,
                   imgState.normal_length,
                   imgState.n_contrib,
                   tileState.max_contributor,
                   dL_dpix,
                   dL_dpix_mdepth,
                   dL_dalphas,
                   dL_dpixel_normals,
                   focal_x, focal_y,
                   (float3*)dL_dmean2D,
                   geomBwdState.conic_opacity,
                   dL_dcolor,
                   geomBwdState.ray_planes,
                   geomBwdState.normals,
                   require_depth),
               debug)

    // Take care of the rest of preprocessing. Was the precomputed covariance
    // given to us or a scales/rot pair? If precomputed, pass that. If not,
    // use the one we computed ourselves.
    CHECK_CUDA(BACKWARD::preprocess(
                   P, SHD, SHM, SGD, SGM,
                   (float3*)means3D,
                   opacities,
                   (glm::vec3*)scales,
                   (float4*)rotations,
                   cov3D_precomp,
                   shs,
                   sg_axis,
                   sg_sharpness,
                   sg_color,
                   scale_modifier,
                   viewmatrix,
                   projmatrix,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   kernel_size,
                   (glm::vec3*)campos,
                   radii,
                   geomState.clamped,
                   geomBwdState.conic_opacity,
                   geomBwdState.ray_planes,
                   geomBwdState.normals,
                   (float3*)dL_dmean2D,
                   dL_dcolor,
                   (glm::vec3*)dL_dmean3D,
                   dL_dopacity,
                   (glm::vec3*)dL_dscale,
                   (glm::vec4*)dL_drot,
                   dL_dcov3D,
                   dL_dsh,
                   dL_dsg_axis,
                   dL_dsg_sharpness,
                   dL_dsg_color),
               debug)
}

int CudaRasterizer::Rasterizer::evaluateTransmittance(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> pointBuffer,
    std::function<char*(size_t)> point_binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    std::function<char*(size_t)> duplicatedTileBuffer,
    const int PN, const int P,
    const int width, int height,
    const float* points3D,
    const float* means3D,
    const float* opacities,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const bool prefiltered,
    float* out_transmittance,
    bool* inside,
    bool debug) {
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size       = required<GeometryState>(P);
    char* chunkptr          = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const int tiles = tile_grid.x * tile_grid.y;

    size_t tile_chunk_size    = required<TileState<true>>(tiles);
    char* tile_chunkptr       = tileBuffer(tile_chunk_size);
    TileState<true> tileState = TileState<true>::fromChunk(tile_chunkptr, tiles);

    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    CHECK_CUDA(FORWARD::preprocess(
                   P, 0, 0, 0, 0,
                   means3D,
                   geomState.rgb, // give a safe pointer to avoid color computation
                   opacities,
                   (glm::vec3*)scales,
                   (float4*)rotations,
                   cov3D_precomp,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   scale_modifier,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   kernel_size,
                   geomState.internal_radii,
                   geomState.clamped,
                   geomState.means2D,
                   geomState.depths,
                   geomState.ray_planes,
                   geomState.normals,
                   geomState.rgb,
                   geomState.conic_opacity,
                   tile_grid,
                   geomState.tiles_touched,
                   prefiltered),
               debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug);

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr    = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        geomState.internal_radii,
        tile_grid)
        CHECK_CUDA(, debug);

    int bit = getHigherMsb(tiles);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   binningState.list_sorting_space,
                   binningState.sorting_size,
                   binningState.point_list_keys_unsorted, binningState.point_list_keys,
                   binningState.point_list_unsorted, binningState.point_list,
                   num_rendered, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.gaussian_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.gaussian_ranges);
    CHECK_CUDA(, debug)

    // create a list of points similar to the list of gaussians
    size_t point_chunk_size = required<PointState>(PN);
    char* point_chunkptr    = pointBuffer(point_chunk_size);
    PointState pointState   = PointState::fromChunk(point_chunkptr, PN);

    // Run preprocessing per-Point (transformation)
    CHECK_CUDA(FORWARD::preprocess_points(
                   PN,
                   points3D,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   pointState.points2D,
                   pointState.depths,
                   tile_grid,
                   pointState.tiles_touched,
                   prefiltered),
               debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(pointState.scanning_space, pointState.scan_size, pointState.tiles_touched, pointState.point_offsets, PN), debug)

    // Retrieve total number of Point instances to launch and resize aux buffers
    int num_evaluation;
    CHECK_CUDA(cudaMemcpy(&num_evaluation, pointState.point_offsets + PN - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t point_binning_chunk_size = required<BinningState>(num_evaluation);
    char* point_binning_chunkptr    = point_binningBuffer(point_binning_chunk_size);
    BinningState point_binningState = BinningState::fromChunk(point_binning_chunkptr, num_evaluation);

    // For each point to be integrated, produce adequate [ tile | depth ] key
    // and corresponding Point indices to be sorted
    createWithKeys<<<(PN + 255) / 256, 256>>>(
        PN,
        pointState.points2D,
        pointState.depths,
        pointState.point_offsets,
        pointState.tiles_touched,
        point_binningState.point_list_keys_unsorted,
        point_binningState.point_list_unsorted,
        tile_grid);
    CHECK_CUDA(, debug)

    //  Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   point_binningState.list_sorting_space,
                   point_binningState.sorting_size,
                   point_binningState.point_list_keys_unsorted, point_binningState.point_list_keys,
                   point_binningState.point_list_unsorted, point_binningState.point_list,
                   num_evaluation, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.point_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_evaluation > 0)
        identifyTileRanges<<<(num_evaluation + 255) / 256, 256>>>(
            num_evaluation,
            point_binningState.point_list_keys,
            tileState.point_ranges);
    CHECK_CUDA(, debug)

    countPointBatches<<<(tiles + 255) / 256, 256>>>(tiles, tileState.point_ranges, tileState.tile_rounds);
    CHECK_CUDA(, debug)
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(tileState.scanning_space, tileState.scan_size, tileState.tile_rounds, tileState.tile_offsets, tiles), debug);
    int num_duplicated_tiles;
    CHECK_CUDA(cudaMemcpy(&num_duplicated_tiles, tileState.tile_offsets + tiles - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    size_t dup_tile_chunk_size                = required<DuplicatedTileState>(num_duplicated_tiles);
    char* dup_tile_chunkptr                   = duplicatedTileBuffer(dup_tile_chunk_size);
    DuplicatedTileState duplicated_tile_state = DuplicatedTileState::fromChunk(dup_tile_chunkptr, num_duplicated_tiles);
    setBlockId<<<(tiles + 255) / 256, 256>>>(tiles, tileState.tile_offsets, tileState.tile_rounds, duplicated_tile_state.tile_id);
    CHECK_CUDA(, debug)

    // Let each tile blend its range of Gaussians independently in parallel
    // const float* view2gaussian = view2gaussian_precomp;
    CHECK_CUDA(FORWARD::evaluateTransmittance(
                   num_duplicated_tiles,
                   duplicated_tile_state.tile_id,
                   tileState.tile_offsets,
                   tileState.gaussian_ranges,
                   tileState.point_ranges,
                   binningState.point_list,
                   point_binningState.point_list,
                   width, height,
                   focal_x, focal_y,
                   pointState.points2D,
                   pointState.depths,
                   geomState.means2D,
                   geomState.conic_opacity,
                   geomState.ray_planes,
                   geomState.normals,
                   out_transmittance,
                   inside),
               debug)

    return num_rendered;
}

int CudaRasterizer::Rasterizer::evaluateSDF(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> pointBuffer,
    std::function<char*(size_t)> point_binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    std::function<char*(size_t)> duplicatedTileBuffer,
    const int PN, const int P,
    const int width, int height,
    const float* points3D,
    const float* means3D,
    const float* opacities,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const bool prefiltered,
    float* out_depth,
    float* out_sdf,
    bool* inside,
    bool debug) {
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size       = required<GeometryState>(P);
    char* chunkptr          = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const int tiles = tile_grid.x * tile_grid.y;

    // Dynamically resize image-based auxiliary buffers during training
    size_t tile_chunk_size    = required<TileState<true>>(tiles);
    char* tile_chunkptr       = tileBuffer(tile_chunk_size);
    TileState<true> tileState = TileState<true>::fromChunk(tile_chunkptr, tiles);

    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    CHECK_CUDA(FORWARD::preprocess(
                   P, 0, 0, 0, 0,
                   means3D,
                   geomState.rgb, // give a safe pointer to avoid color computation
                   opacities,
                   (glm::vec3*)scales,
                   (float4*)rotations,
                   cov3D_precomp,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   scale_modifier,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   kernel_size,
                   geomState.internal_radii,
                   geomState.clamped,
                   geomState.means2D,
                   geomState.depths,
                   geomState.ray_planes,
                   geomState.normals,
                   geomState.rgb,
                   geomState.conic_opacity,
                   tile_grid,
                   geomState.tiles_touched,
                   prefiltered),
               debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr    = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        geomState.internal_radii,
        tile_grid)
        CHECK_CUDA(, debug);

    int bit = getHigherMsb(tiles);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   binningState.list_sorting_space,
                   binningState.sorting_size,
                   binningState.point_list_keys_unsorted, binningState.point_list_keys,
                   binningState.point_list_unsorted, binningState.point_list,
                   num_rendered, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.gaussian_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.gaussian_ranges);
    CHECK_CUDA(, debug)

    // create a list of points similar to the list of gaussians
    size_t point_chunk_size = required<PointState>(PN);
    char* point_chunkptr    = pointBuffer(point_chunk_size);
    PointState pointState   = PointState::fromChunk(point_chunkptr, PN);

    // Run preprocessing per-Point (transformation)
    CHECK_CUDA(FORWARD::preprocess_points(
                   PN,
                   points3D,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   pointState.points2D,
                   pointState.depths,
                   tile_grid,
                   pointState.tiles_touched,
                   prefiltered),
               debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(pointState.scanning_space, pointState.scan_size, pointState.tiles_touched, pointState.point_offsets, PN), debug)

    // Retrieve total number of Point instances to launch and resize aux buffers
    int num_evaluation;
    CHECK_CUDA(cudaMemcpy(&num_evaluation, pointState.point_offsets + PN - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t point_binning_chunk_size = required<BinningState>(num_evaluation);
    char* point_binning_chunkptr    = point_binningBuffer(point_binning_chunk_size);
    BinningState point_binningState = BinningState::fromChunk(point_binning_chunkptr, num_evaluation);

    // For each point to be integrated, produce adequate [ tile | depth ] key
    // and corresponding Point indices to be sorted
    createWithKeys<<<(PN + 255) / 256, 256>>>(
        PN,
        pointState.points2D,
        pointState.depths,
        pointState.point_offsets,
        pointState.tiles_touched,
        point_binningState.point_list_keys_unsorted,
        point_binningState.point_list_unsorted,
        tile_grid);
    CHECK_CUDA(, debug)

    //  Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   point_binningState.list_sorting_space,
                   point_binningState.sorting_size,
                   point_binningState.point_list_keys_unsorted, point_binningState.point_list_keys,
                   point_binningState.point_list_unsorted, point_binningState.point_list,
                   num_evaluation, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.point_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_evaluation > 0)
        identifyTileRanges<<<(num_evaluation + 255) / 256, 256>>>(
            num_evaluation,
            point_binningState.point_list_keys,
            tileState.point_ranges);
    CHECK_CUDA(, debug)

    countPointBatches<<<(tiles + 255) / 256, 256>>>(tiles, tileState.point_ranges, tileState.tile_rounds);
    CHECK_CUDA(, debug)
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(tileState.scanning_space, tileState.scan_size, tileState.tile_rounds, tileState.tile_offsets, tiles), debug);
    int num_duplicated_tiles;
    CHECK_CUDA(cudaMemcpy(&num_duplicated_tiles, tileState.tile_offsets + tiles - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    size_t dup_tile_chunk_size                = required<DuplicatedTileState>(num_duplicated_tiles);
    char* dup_tile_chunkptr                   = duplicatedTileBuffer(dup_tile_chunk_size);
    DuplicatedTileState duplicated_tile_state = DuplicatedTileState::fromChunk(dup_tile_chunkptr, num_duplicated_tiles);
    setBlockId<<<(tiles + 255) / 256, 256>>>(tiles, tileState.tile_offsets, tileState.tile_rounds, duplicated_tile_state.tile_id);
    CHECK_CUDA(, debug)

    // Let each tile blend its range of Gaussians independently in parallel
    CHECK_CUDA(FORWARD::evaluateSDF(
                   num_duplicated_tiles,
                   duplicated_tile_state.tile_id,
                   tileState.tile_offsets,
                   tileState.gaussian_ranges,
                   tileState.point_ranges,
                   binningState.point_list,
                   point_binningState.point_list,
                   width, height,
                   focal_x, focal_y,
                   pointState.points2D,
                   pointState.depths,
                   geomState.means2D,
                   geomState.conic_opacity,
                   geomState.ray_planes,
                   geomState.normals,
                   out_depth,
                   out_sdf,
                   inside),
               debug)

    return num_rendered;
}

int CudaRasterizer::Rasterizer::evaluateColor(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> pointBuffer,
    std::function<char*(size_t)> point_binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    std::function<char*(size_t)> duplicatedTileBuffer,
    const int PN, const int P,
    const int SHD, const int SHM, const int SGD, const int SGM,
    const float* background,
    const int width, const int height,
    const float* points3D,
    const float* means3D,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    const float* rotations,
    const float* cov3D_precomp,
    const float* shs,
    const float* sg_axis,
    const float* sg_sharpness,
    const float* sg_color,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const bool prefiltered,
    float* out_color,
    bool* inside,
    bool debug) {
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size       = required<GeometryState>(P);
    char* chunkptr          = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const int tiles = tile_grid.x * tile_grid.y;

    // Dynamically resize image-based auxiliary buffers during training
    size_t tile_chunk_size    = required<TileState<true>>(tiles);
    char* tile_chunkptr       = tileBuffer(tile_chunk_size);
    TileState<true> tileState = TileState<true>::fromChunk(tile_chunkptr, tiles);
    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    CHECK_CUDA(FORWARD::preprocess(
                   P, SHD, SHM, SGD, SGM,
                   means3D,
                   colors_precomp,
                   opacities,
                   (glm::vec3*)scales,
                   (float4*)rotations,
                   cov3D_precomp,
                   shs,
                   sg_axis,
                   sg_sharpness,
                   sg_color,
                   scale_modifier,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   kernel_size,
                   geomState.internal_radii,
                   geomState.clamped,
                   geomState.means2D,
                   geomState.depths,
                   geomState.ray_planes,
                   geomState.normals,
                   geomState.rgb,
                   geomState.conic_opacity,
                   tile_grid,
                   geomState.tiles_touched,
                   prefiltered),
               debug)
    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr    = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        geomState.internal_radii,
        tile_grid)
        CHECK_CUDA(, debug);

    int bit = getHigherMsb(tiles);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   binningState.list_sorting_space,
                   binningState.sorting_size,
                   binningState.point_list_keys_unsorted, binningState.point_list_keys,
                   binningState.point_list_unsorted, binningState.point_list,
                   num_rendered, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.gaussian_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.gaussian_ranges);
    CHECK_CUDA(, debug)

    // create a list of points similar to the list of gaussians
    size_t point_chunk_size = required<PointState>(PN);
    char* point_chunkptr    = pointBuffer(point_chunk_size);
    PointState pointState   = PointState::fromChunk(point_chunkptr, PN);

    // Run preprocessing per-Point (transformation)
    CHECK_CUDA(FORWARD::preprocess_points(
                   PN,
                   points3D,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   pointState.points2D,
                   pointState.depths,
                   tile_grid,
                   pointState.tiles_touched,
                   prefiltered),
               debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(pointState.scanning_space, pointState.scan_size, pointState.tiles_touched, pointState.point_offsets, PN), debug)

    // Retrieve total number of Point instances to launch and resize aux buffers
    int num_evaluation;
    CHECK_CUDA(cudaMemcpy(&num_evaluation, pointState.point_offsets + PN - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t point_binning_chunk_size = required<BinningState>(num_evaluation);
    char* point_binning_chunkptr    = point_binningBuffer(point_binning_chunk_size);
    BinningState point_binningState = BinningState::fromChunk(point_binning_chunkptr, num_evaluation);

    // For each point to be integrated, produce adequate [ tile | depth ] key
    // and corresponding Point indices to be sorted
    createWithKeys<<<(PN + 255) / 256, 256>>>(
        PN,
        pointState.points2D,
        pointState.depths,
        pointState.point_offsets,
        pointState.tiles_touched,
        point_binningState.point_list_keys_unsorted,
        point_binningState.point_list_unsorted,
        tile_grid);
    CHECK_CUDA(, debug)

    //  Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   point_binningState.list_sorting_space,
                   point_binningState.sorting_size,
                   point_binningState.point_list_keys_unsorted, point_binningState.point_list_keys,
                   point_binningState.point_list_unsorted, point_binningState.point_list,
                   num_evaluation, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.point_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_evaluation > 0)
        identifyTileRanges<<<(num_evaluation + 255) / 256, 256>>>(
            num_evaluation,
            point_binningState.point_list_keys,
            tileState.point_ranges);
    CHECK_CUDA(, debug)

    countPointBatches<<<(tiles + 255) / 256, 256>>>(tiles, tileState.point_ranges, tileState.tile_rounds);
    CHECK_CUDA(, debug)
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(tileState.scanning_space, tileState.scan_size, tileState.tile_rounds, tileState.tile_offsets, tiles), debug);
    int num_duplicated_tiles;
    CHECK_CUDA(cudaMemcpy(&num_duplicated_tiles, tileState.tile_offsets + tiles - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    size_t dup_tile_chunk_size                = required<DuplicatedTileState>(num_duplicated_tiles);
    char* dup_tile_chunkptr                   = duplicatedTileBuffer(dup_tile_chunk_size);
    DuplicatedTileState duplicated_tile_state = DuplicatedTileState::fromChunk(dup_tile_chunkptr, num_duplicated_tiles);
    setBlockId<<<(tiles + 255) / 256, 256>>>(tiles, tileState.tile_offsets, tileState.tile_rounds, duplicated_tile_state.tile_id);
    CHECK_CUDA(, debug)

    CHECK_CUDA(FORWARD::evaluateColor(
                   num_duplicated_tiles,
                   duplicated_tile_state.tile_id,
                   tileState.tile_offsets,
                   tileState.gaussian_ranges,
                   tileState.point_ranges,
                   binningState.point_list,
                   point_binningState.point_list,
                   width, height,
                   focal_x, focal_y,
                   pointState.points2D,
                   pointState.depths,
                   geomState.means2D,
                   geomState.conic_opacity,
                   geomState.rgb,
                   background,
                   out_color,
                   inside),
               debug)
    return num_rendered;
}

int3 CudaRasterizer::Rasterizer::sampleDepth(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> pointBuffer,
    std::function<char*(size_t)> point_binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    std::function<char*(size_t)> duplicatedTileBuffer,
    const int PN, const int P,
    const int width, int height,
    const float* points3D,
    const float* means3D,
    const float* opacities,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const bool prefiltered,
    float* output,
    bool* inside,
    bool debug) {
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size       = required<GeometryState>(P);
    char* chunkptr          = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

    const int tiles = tile_grid.x * tile_grid.y;
    // Dynamically resize image-based auxiliary buffers during training
    size_t tile_chunk_size    = required<TileState<true>>(tiles);
    char* tile_chunkptr       = tileBuffer(tile_chunk_size);
    TileState<true> tileState = TileState<true>::fromChunk(tile_chunkptr, tiles);

    CHECK_CUDA(FORWARD::preprocess(
                   P, 0, 0, 0, 0,
                   means3D,
                   geomState.rgb, // give a safe pointer to avoid color computation
                   opacities,
                   (glm::vec3*)scales,
                   (float4*)rotations,
                   cov3D_precomp,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   scale_modifier,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   kernel_size,
                   geomState.internal_radii,
                   geomState.clamped,
                   geomState.means2D,
                   geomState.depths,
                   geomState.ray_planes,
                   geomState.normals,
                   geomState.rgb,
                   geomState.conic_opacity,
                   tile_grid,
                   geomState.tiles_touched,
                   prefiltered),
               debug)
    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr    = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        geomState.internal_radii,
        tile_grid);
    CHECK_CUDA(, debug);

    int bit = getHigherMsb(tiles);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   binningState.list_sorting_space,
                   binningState.sorting_size,
                   binningState.point_list_keys_unsorted, binningState.point_list_keys,
                   binningState.point_list_unsorted, binningState.point_list,
                   num_rendered, 0, 32 + bit),
               debug)

    CHECK_CUDA(cudaMemset(tileState.gaussian_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.gaussian_ranges);
    CHECK_CUDA(, debug)

    /**************************************** Integrate ****************************************/
    // create a list of points similar to the list of gaussians
    size_t point_chunk_size = required<PointState>(PN);
    char* point_chunkptr    = pointBuffer(point_chunk_size);
    PointState pointState   = PointState::fromChunk(point_chunkptr, PN);

    // Run preprocessing per-Point (transformation)
    CHECK_CUDA(FORWARD::preprocess_points(
                   PN,
                   points3D,
                   viewmatrix, projmatrix,
                   (glm::vec3*)cam_pos,
                   width, height,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   pointState.points2D,
                   pointState.depths,
                   tile_grid,
                   pointState.tiles_touched,
                   prefiltered),
               debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(pointState.scanning_space, pointState.scan_size, pointState.tiles_touched, pointState.point_offsets, PN), debug)

    // Retrieve total number of Point instances to launch and resize aux buffers
    int num_evaluation;
    CHECK_CUDA(cudaMemcpy(&num_evaluation, pointState.point_offsets + PN - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t point_binning_chunk_size = required<BinningState>(num_evaluation);
    char* point_binning_chunkptr    = point_binningBuffer(point_binning_chunk_size);
    BinningState point_binningState = BinningState::fromChunk(point_binning_chunkptr, num_evaluation);

    // For each point to be integrated, produce adequate [ tile | depth ] key
    // and corresponding Point indices to be sorted
    createWithKeys<<<(PN + 255) / 256, 256>>>(
        PN,
        pointState.points2D,
        pointState.depths,
        pointState.point_offsets,
        pointState.tiles_touched,
        point_binningState.point_list_keys_unsorted,
        point_binningState.point_list_unsorted,
        tile_grid);
    CHECK_CUDA(, debug)

    // TODO: why we use num_evaluation here?
    //  Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                   point_binningState.list_sorting_space,
                   point_binningState.sorting_size,
                   point_binningState.point_list_keys_unsorted, point_binningState.point_list_keys,
                   point_binningState.point_list_unsorted, point_binningState.point_list,
                   num_evaluation, 0, 32 + bit),
               debug);

    CHECK_CUDA(cudaMemset(tileState.point_ranges, 0, tiles * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_evaluation > 0)
        identifyTileRanges<<<(num_evaluation + 255) / 256, 256>>>(
            num_evaluation,
            point_binningState.point_list_keys,
            tileState.point_ranges);
    CHECK_CUDA(, debug)

    countPointBatches<<<(tiles + 255) / 256, 256>>>(tiles, tileState.point_ranges, tileState.tile_rounds);
    CHECK_CUDA(, debug)
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(tileState.scanning_space, tileState.scan_size, tileState.tile_rounds, tileState.tile_offsets, tiles), debug);
    int num_duplicated_tiles;
    CHECK_CUDA(cudaMemcpy(&num_duplicated_tiles, tileState.tile_offsets + tiles - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    size_t dup_tile_chunk_size                = required<DuplicatedTileState>(num_duplicated_tiles);
    char* dup_tile_chunkptr                   = duplicatedTileBuffer(dup_tile_chunk_size);
    DuplicatedTileState duplicated_tile_state = DuplicatedTileState::fromChunk(dup_tile_chunkptr, num_duplicated_tiles);
    setBlockId<<<(tiles + 255) / 256, 256>>>(tiles, tileState.tile_offsets, tileState.tile_rounds, duplicated_tile_state.tile_id);
    CHECK_CUDA(, debug)

    CHECK_CUDA(FORWARD::sampleDepth(
                   num_duplicated_tiles,
                   duplicated_tile_state.tile_id,
                   tileState.tile_offsets,
                   tileState.gaussian_ranges,
                   tileState.point_ranges,
                   binningState.point_list,
                   point_binningState.point_list,
                   width, height,
                   focal_x, focal_y,
                   pointState.points2D,
                   geomState.means2D,
                   geomState.conic_opacity,
                   geomState.ray_planes,
                   geomState.normals,
                   pointState.n_contrib,
                   pointState.median_depth,
                   duplicated_tile_state.max_contributor,
                   (float3*)output,
                   inside),
               debug)

    return {num_rendered, num_evaluation, num_duplicated_tiles};
}

void CudaRasterizer::Rasterizer::sampleDepthBackward(
    std::function<char*(size_t)> geometryBuffer,
    const int PN, const int P,
    const int RN, const int R,
    const int TN,
    const int width, int height,
    const float* points3D,
    const float* means3D,
    const float* opacities,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* campos,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    char* geom_buffer,
    char* binning_buffer,
    char* point_buffer,
    char* point_binning_buffer,
    char* tile_buffer,
    char* duplicatedtileBuffer,
    const bool* inside,
    const float* dL_doutput,
    float* dL_dmean2D,
    float* dL_dpoint2D,
    float* dL_dopacity,
    float* dL_dmean3D,
    float* dL_dcov3D,
    float* dL_dscale,
    float* dL_drot,
    float* dL_dpoint,
    bool debug) {
    GeometryState geomState                   = GeometryState::fromChunk(geom_buffer, P);
    BinningState binningState                 = BinningState::fromChunk(binning_buffer, R);
    PointState pointState                     = PointState::fromChunk(point_buffer, PN);
    BinningState point_binningState           = BinningState::fromChunk(point_binning_buffer, RN);
    DuplicatedTileState duplicated_tile_state = DuplicatedTileState::fromChunk(duplicatedtileBuffer, TN);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    TileState<true> tileState = TileState<true>::fromChunk(tile_buffer, tile_grid.x * tile_grid.y);

    size_t chunk_size             = required<GeometryBwdState>(P);
    char* chunkptr                = geometryBuffer(chunk_size);
    GeometryBwdState geomBwdState = GeometryBwdState::fromChunk(chunkptr, P);

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    // Compute loss gradients w.r.t. 2D mean position, conic matrix,
    // opacity and RGB of Gaussians from per-pixel loss gradients.
    // If we were given precomputed colors and not SHs, use them.
    CHECK_CUDA(BACKWARD::sampleDepth(
                   TN,
                   duplicated_tile_state.max_contributor,
                   duplicated_tile_state.tile_id,
                   tileState.tile_offsets,
                   tileState.gaussian_ranges,
                   tileState.point_ranges,
                   binningState.point_list,
                   point_binningState.point_list,
                   width, height,
                   focal_x, focal_y,
                   pointState.points2D,
                   geomState.means2D,
                   geomState.conic_opacity,
                   geomState.ray_planes,
                   geomState.normals,
                   pointState.n_contrib,
                   pointState.median_depth,
                   inside,
                   (float3*)dL_doutput,
                   (float2*)dL_dpoint2D,
                   (float3*)dL_dmean2D,
                   geomBwdState.conic_opacity,
                   geomBwdState.ray_planes,
                   geomBwdState.normals),
               debug)
    // Take care of the rest of preprocessing. Was the precomputed covariance
    // given to us or a scales/rot pair? If precomputed, pass that. If not,
    // use the one we computed ourselves.
    CHECK_CUDA(BACKWARD::preprocess(
                   P, 0, 0, 0, 0,
                   (float3*)means3D,
                   opacities,
                   (glm::vec3*)scales,
                   (float4*)rotations,
                   cov3D_precomp,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   scale_modifier,
                   viewmatrix,
                   projmatrix,
                   focal_x, focal_y,
                   tan_fovx, tan_fovy,
                   kernel_size,
                   (glm::vec3*)campos,
                   geomState.internal_radii,
                   geomState.clamped,
                   geomBwdState.conic_opacity,
                   geomBwdState.ray_planes,
                   geomBwdState.normals,
                   (float3*)dL_dmean2D,
                   nullptr,
                   (glm::vec3*)dL_dmean3D,
                   dL_dopacity,
                   (glm::vec3*)dL_dscale,
                   (glm::vec4*)dL_drot,
                   dL_dcov3D,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr),
               debug)

    CHECK_CUDA(BACKWARD::preprocess_points(
                   PN,
                   (float3*)points3D,
                   viewmatrix,
                   projmatrix,
                   (glm::vec3*)campos,
                   width, height,
                   tan_fovx, tan_fovy,
                   pointState.tiles_touched,
                   (float2*)dL_dpoint2D,
                   (float3*)dL_dpoint),
               debug)
}