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

#include "auxiliary.h"
#include "config.h"
#include "render_forward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ __forceinline__ glm::vec3 computeColorFromSHSG(int idx, int SHD, int SHM, int SGD, int SGM, const glm::vec3* means, glm::vec3 campos,
                                                          const float* shs, const float* sg_axis, const float* sg_sharpness, const float* sg_color, bool* clamped) {
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir           = dir / glm::length(dir);

    const glm::vec3* sh = reinterpret_cast<const glm::vec3*>(shs) + idx * SHM;
    glm::vec3 result    = SH_C0 * sh[0];

    if (SHD > 0) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result  = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (SHD > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                     SH_C2[0] * xy * sh[4] +
                     SH_C2[1] * yz * sh[5] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                     SH_C2[3] * xz * sh[7] +
                     SH_C2[4] * (xx - yy) * sh[8];

            if (SHD > 2) {
                result = result +
                         SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                         SH_C3[1] * xy * z * sh[10] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                         SH_C3[5] * z * (xx - yy) * sh[14] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    const glm::vec3* sg_axis_p  = reinterpret_cast<const glm::vec3*>(sg_axis) + idx * SGM;
    const float* sg_sharpness_p = sg_sharpness + idx * SGM;
    const glm::vec3* sg_color_p = reinterpret_cast<const glm::vec3*>(sg_color) + idx * SGM;
    for (int g = 0; g < SGD; g++) {
        float gaussian = expf(sg_sharpness_p[g] * (glm::dot(sg_axis_p[g], dir) - 1.0f));
        result += sg_color_p[g] * gaussian;
    }

    result += 0.5f;

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ __forceinline__ bool computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, float kernel_size, const float* cov3D, const float* viewmatrix,
                                             float* cov2D, float3* normals, float4* ray_plane, float& coef, const glm::vec3* scale = nullptr, const float4* rotation = nullptr, const float mod = 1.f) {
    // The following models the steps outlined by equations 29
    // and 31 in "EWA Splatting" (Zwicker et al., 2002).
    // Additionally considers aspect / scaling of viewport.
    // Transposes used to account for row-/column-major conventions.
    float3 t       = transformPoint4x3(mean, viewmatrix);
    const float tc = norm3df(t.x, t.y, t.z);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    float u          = t.x / t.z;
    float v          = t.y / t.z;
    t.x              = fminf(limx, fmaxf(-limx, u)) * t.z;
    t.y              = fminf(limy, fmaxf(-limy, v)) * t.z;
    u                = t.x / t.z;
    v                = t.y / t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.f, -(focal_x * t.x) / (t.z * t.z),
        0.f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.f, 0.f, 0.f);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 cov;
    glm::mat3 cov_cam_inv;

    bool well_conditioned;

    auto find_min_from_triple = [](auto v) -> unsigned int {
        unsigned idx  = 0;
        float min_val = v[0];
        if (v[1] < min_val) {
            min_val = v[1];
            idx     = 1;
        }
        if (v[2] < min_val) {
            idx = 2;
        }
        return idx;
    };

    if (scale) {
        // Create scaling matrix
        glm::mat3 S                = glm::mat3(1.0f);
        glm::mat3 S_inv            = glm::mat3(1.0f);
        const float scale_local[3] = {mod * scale->x, mod * scale->y, mod * scale->z};
        S[0][0]                    = scale_local[0];
        S[1][1]                    = scale_local[1];
        S[2][2]                    = scale_local[2];

        S_inv[0][0] = __frcp_rn(scale_local[0]);
        S_inv[1][1] = __frcp_rn(scale_local[1]);
        S_inv[2][2] = __frcp_rn(scale_local[2]);

        well_conditioned = true;

        // Normalize quaternion to get valid rotation
        float4 rot = *rotation;
        float r    = rot.x;
        float x    = rot.y;
        float y    = rot.z;
        float z    = rot.w;

        // Compute rotation matrix from quaternion
        glm::mat3 R = glm::mat3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
            2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
            2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

        glm::mat3 M = S * R * T;
        // Compute 3D world covariance matrix Sigma
        cov             = glm::transpose(M) * M;
        glm::mat3 M_inv = S_inv * R * W;
        cov_cam_inv     = glm::transpose(M_inv) * M_inv;
    } else {
        glm::mat3 Vrk = glm::mat3(
            cov3D[0], cov3D[1], cov3D[2],
            cov3D[1], cov3D[3], cov3D[4],
            cov3D[2], cov3D[4], cov3D[5]);

        cov = glm::transpose(T) * glm::transpose(Vrk) * T;

        glm::mat3 Vrk_eigen_vector;
        glm::vec3 Vrk_eigen_value;
        int D = glm_modification::findEigenvaluesSymReal(Vrk, Vrk_eigen_value, Vrk_eigen_vector);

        unsigned int min_id = find_min_from_triple(Vrk_eigen_value);

        well_conditioned = Vrk_eigen_value[min_id] > 1E-8;
        glm::vec3 eigenvector_min;
        glm::mat3 Vrk_inv;
        if (well_conditioned) {
            glm::mat3 diag = glm::mat3(1 / Vrk_eigen_value[0], 0, 0,
                                       0, 1 / Vrk_eigen_value[1], 0,
                                       0, 0, 1 / Vrk_eigen_value[2]);
            Vrk_inv        = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
        } else {
            eigenvector_min = Vrk_eigen_vector[min_id];
            Vrk_inv         = glm::outerProduct(eigenvector_min, eigenvector_min);
        }
        cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
    }

    cov2D[0]          = float(cov[0][0] + kernel_size);
    cov2D[1]          = float(cov[0][1]);
    cov2D[2]          = float(cov[1][1] + kernel_size);
    const float det_0 = fmaxf(1e-6f, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
    const float det_1 = fmaxf(1e-6f, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
    coef              = sqrtf(det_0 / det_1);

    // glm::mat3 testm = glm::mat3{
    // 	1,2,3,
    // 	4,5,6,
    // 	7,8,9,
    // };
    // glm::vec3 testv = {1,1,1};
    // glm::vec3 resultm = testm * testv;
    // printf("%f %f %f\n", resultm[0], resultm[1], resultm[2]); 12.000000 15.000000 18.000000

    glm::vec3 uvh   = {u, v, 1};
    glm::vec3 uvh_m = cov_cam_inv * uvh;
    // glm::vec3 uvh_mn = glm::normalize(uvh_m);

    {
        float u2 = u * u;
        float v2 = v * v;
        float uv = u * v;

        const float l = norm3df(t.x, t.y, t.z);

        glm::mat3 nJ_inv = glm::mat3(
            v2 + 1, -uv, 0,
            -uv, u2 + 1, 0,
            -u, -v, 0);

        // float vbn           = glm::dot(uvh_mn, uvh);
        float vb            = glm::dot(uvh_m, uvh);
        float ray_len2      = u2 + v2 + 1;
        float factor_normal = l / ray_len2;
        glm::vec3 plane     = nJ_inv * (uvh_m / vb);
        float rsigmat       = well_conditioned ? sqrtf(vb / ray_len2) : 0.f;

        *ray_plane = {plane[0] * factor_normal / focal_x, plane[1] * factor_normal / focal_y, tc, rsigmat};

        glm::vec3 ray_normal_vector = {-plane[0] * factor_normal, -plane[1] * factor_normal, -1};
        glm::mat3 nJ                = glm::mat3(
            1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
            0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
            t.x / l, t.y / l, t.z / l);
        glm::vec3 cam_normal_vector = nJ * ray_normal_vector;
        glm::vec3 normal_vector     = glm::normalize(cam_normal_vector);

        *normals = {normal_vector.x, normal_vector.y, normal_vector.z};
    }
    return well_conditioned;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ __forceinline__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D) {
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0]     = mod * scale.x;
    S[1][1]     = mod * scale.y;
    S[2][2]     = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot; // / glm::length(rot);
    float r     = q.x;
    float x     = q.y;
    float y     = q.z;
    float z     = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(
    const int P, int SHD, int SHM, int SGD, int SGM,
    const float* orig_points,
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
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    const float kernel_size,
    int* radii,
    bool* clamped,
    float2* points_xy_image,
    float* depths,
    float4* ray_planes,
    float3* normals,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    radii[idx]         = 0;
    tiles_touched[idx] = 0;
    // Perform near culling, quit if outside.
    const float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
    float3 p_view;
    if (!in_frustum(p_orig, viewmatrix, projmatrix, prefiltered, p_view))
        return;
    // Transform point by projecting
    float4 p_hom  = transformPoint4x4(p_orig, projmatrix);
    float p_w     = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

    // If 3D covariance matrix is precomputed, use it, otherwise compute
    // from scaling and rotation parameters.
    const float* cov3D = nullptr;
    if (cov3D_precomp) {
        cov3D = cov3D_precomp + idx * 6;
    }

    // Compute 2D screen-space covariance matrix
    float cov2D[3];
    float ceof;
    computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, kernel_size, cov3D, viewmatrix, cov2D,
                 normals + idx, ray_planes + idx, ceof, scales + idx, rotations + idx, scale_modifier);

    const float3 cov = {cov2D[0], cov2D[1], cov2D[2]};

    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;
    float3 conic  = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute a bounding rectangle
    // of screen-space tiles that this Gaussian overlaps with. Quit if
    // rectangle covers 0 tiles.
    float mid          = 0.5f * (cov.x + cov.z);
    float lambda1      = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    float lambda2      = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
    float my_radius    = ceilf(3.f * sqrtf(fmaxf(lambda1, lambda2)));
    float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // If colors have been precomputed, use them, otherwise convert
    // spherical harmonics coefficients to RGB color.
    if (colors_precomp == nullptr) {
        glm::vec3 result = computeColorFromSHSG(idx, SHD, SHM, SGD, SGM, (glm::vec3*)orig_points, *cam_pos, shs, sg_axis, sg_sharpness, sg_color, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // Store some useful helper data for the next steps.
    depths[idx]          = norm3df(p_view.x, p_view.y, p_view.z);
    radii[idx]           = my_radius;
    points_xy_image[idx] = point_image;
    // Inverse 2D covariance and opacity neatly pack into one float4
    conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx] * ceof};
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS, bool GEOMETRY, uint32_t SPLIT = 8, uint32_t SPLIT_ITERATIONS = 5>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    renderCUDA(
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        const float2* __restrict__ points_xy_image,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ features,
        const float4* __restrict__ ray_planes,
        const float3* __restrict__ normals,
        const float focal_x,
        const float focal_y,
        uint32_t* __restrict__ n_contrib,
        uint32_t* __restrict__ max_contributors,
        const float* __restrict__ bg_color,
        float* __restrict__ out_color,
        float* __restrict__ out_alpha,
        float* __restrict__ out_normal,
        float* __restrict__ out_mdepth,
        float* __restrict__ normal_length) {
    // Identify current tile and associated min/max pixel range.
    auto block                 = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint32_t block_id          = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id            = W * pix.y + pix.x;
    float2 pixf                = {static_cast<float>(pix.x), static_cast<float>(pix.y)};
    const float2 pixnf         = {(pixf.x - static_cast<float>(W - 1) / 2.f) / focal_x, (pixf.y - static_cast<float>(H - 1) / 2.f) / focal_y};
    const float rln            = rnorm3df(pixnf.x, pixnf.y, 1.f);

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range      = ranges[block_id];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo         = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float collected_feature[BLOCK_SIZE * CHANNELS];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    [[maybe_unused]] __shared__ float4 collected_ray_planes[BLOCK_SIZE];
    [[maybe_unused]] __shared__ float3 collected_normals[BLOCK_SIZE];

    // Initialize helper variables
    float T                            = 1.0f;
    uint32_t contributor               = 0;
    uint32_t last_contributor          = 0;
    float C[CHANNELS]                  = {0};
    [[maybe_unused]] float mDepth      = 0;
    [[maybe_unused]] float Normal[3]   = {0};
    [[maybe_unused]] float last_depth  = 0;
    [[maybe_unused]] float last_weight = 0;
    [[maybe_unused]] float mDepthinit  = 0;
#ifdef MOST_VISIBLE_INIT
    [[maybe_unused]] float max_weight = 0;
#endif
    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int block_done = __syncthreads_and(done);
        if (block_done)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id                                  = point_list[range.x + progress];
            collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            for (int ch = 0; ch < CHANNELS; ch++)
                collected_feature[ch * BLOCK_SIZE + block.thread_rank()] = features[coll_id * CHANNELS + ch];
            if constexpr (GEOMETRY) {
                collected_ray_planes[block.thread_rank()] = ray_planes[coll_id];
                collected_normals[block.thread_rank()]    = normals[coll_id];
            }
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // Keep track of current position in range
            contributor++;

            // Resample using conic matrix (cf. "Surface
            // Splatting" by Zwicker et al., 2001)
            float2 xy    = collected_xy[j];
            float2 d     = {xy.x - pixf.x, xy.y - pixf.y};
            float4 con_o = collected_conic_opacity[j];
            float power  = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) {
                continue;
            }

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix).
            float alpha = fminf(0.99f, con_o.w * expf(power));
            if (alpha < 1.0f / 255.0f)
                continue;
            float test_T = T * (1.f - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }

            const float aT = alpha * T;
            // Eq. (3) from 3D Gaussian splatting paper.
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += collected_feature[j + BLOCK_SIZE * ch] * aT;

            if constexpr (GEOMETRY) {
                float4 ray_plane = collected_ray_planes[j];
                float3 normal    = collected_normals[j];
                const float t    = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                Normal[0] += normal.x * aT;
                Normal[1] += normal.y * aT;
                Normal[2] += normal.z * aT;
#if defined(MEDIAN_DEPTH_INIT)
                mDepthinit = T > 0.5f ? t : mDepthinit;
#elif defined(MEAN_DEPTH_INIT)
                mDepthinit += aT * t;
#elif defined(MOST_VISIBLE_INIT)
                const bool more_visible = aT > max_weight;
                mDepthinit              = more_visible ? t : mDepthinit;
                max_weight              = more_visible ? aT : max_weight;
#endif
            }

            T = test_T;

            // Keep track of last range entry to update this
            // pixel.
            last_contributor = contributor;
        }
    }

    using BlockReduce = cub::BlockReduce<uint32_t,
                                         BLOCK_X,
                                         cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                         BLOCK_Y, 1>;

    __shared__ typename BlockReduce::TempStorage temp_storage;
#if CUDA_VERSION_AT_LEAST_12_9
    const uint32_t block_max =
        BlockReduce(temp_storage).Reduce(last_contributor, cuda::maximum<uint32_t>{});
#else
    auto op = [] __device__(uint32_t a, uint32_t b) { return max(a, b); };
    const uint32_t block_max =
        BlockReduce(temp_storage).Reduce(last_contributor, op);
#endif
    if constexpr (GEOMETRY) {
        // scatter the block_max
        __shared__ uint32_t s_block_max;
        if (block.thread_rank() == 0)
            s_block_max = block_max;
        block.sync();
        const uint32_t max_contributor = s_block_max;
        const int rounds               = (max_contributor + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float T_p[SPLIT + 1];
#if defined(MEAN_DEPTH_INIT)
        mDepthinit = mDepthinit / (1.f - T);
#endif
        float depth_min = fmaxf(mDepthinit - SAMPLE_RANGE, 0.f);
        float depth_max = fmaxf(mDepthinit + SAMPLE_RANGE, 0.f);
        bool in_range   = T <= MIN_TRANSMITTANCE;

        auto iter = [&] __device__(auto first_const) {
            constexpr bool FIRST           = decltype(first_const)::value;
            constexpr float ONE_OVER_SPLIT = 1.f / static_cast<float>(SPLIT);
            constexpr int START_ID         = FIRST ? 0 : 1;
            constexpr int END_ID           = FIRST ? SPLIT + 1 : SPLIT;
#pragma unroll
            for (int i = START_ID; i < END_ID; i++) {
                T_p[i] = 1.f;
            }
            float interval = (depth_max - depth_min) * ONE_OVER_SPLIT;
            // bool done      = !in_range || (last_contributor == 0) || !inside;
            bool done   = !in_range;
            int toDo    = max_contributor;
            contributor = 0;
            // Iterate over batches until all done or range is complete
            for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
                // wait for read done
                block.sync();

                // Collectively fetch per-Gaussian data from global to shared
                int progress = i * BLOCK_SIZE + block.thread_rank();
                if (progress < max_contributor) {
                    int coll_id                                  = point_list[range.x + progress];
                    collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
                    collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                    collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
                }
                block.sync();

                // Iterate over current batch
                for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
                    contributor++;
                    done         = contributor >= last_contributor;
                    float2 xy    = collected_xy[j];
                    float2 d     = {xy.x - pixf.x, xy.y - pixf.y};
                    float4 con_o = collected_conic_opacity[j];
                    float power  = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    if (power > 0.0f) {
                        continue;
                    }
                    float alpha = fminf(0.99f, con_o.w * expf(power));
                    if (alpha < 1.0f / 255.0f)
                        continue;

                    float4 ray_plane   = collected_ray_planes[j];
                    const float t_peak = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                    const float rsigma = ray_plane.w;
                    const bool ball    = rsigma > 0;
                    for (int s = START_ID; s < END_ID; s++) {
                        float ts                 = depth_min + interval * s;
                        float delta              = (ts - t_peak) * rsigma;
                        float g                  = ball ? expf(-0.5f * delta * delta) : 0.f;
                        float one_minus_gaussian = 1.f - alpha * g;
                        float rvacancy           = rsqrtf(one_minus_gaussian);
                        T_p[s] *= (ts > t_peak ? (1.f - alpha) : one_minus_gaussian) * rvacancy;
                    }
                }
            }
            if constexpr (FIRST) {
                in_range = (T_p[0] >= 0.5f) && (T_p[SPLIT] <= 0.5f) && in_range;
            }
            int start_id = 0;
#pragma unroll
            for (int p = 1; p < SPLIT; p++) {
                start_id = T_p[p] >= 0.5f ? p : start_id;
            }
            depth_max  = depth_min + (start_id + 1) * interval;
            depth_min  = depth_min + (start_id + 0) * interval;
            T_p[0]     = T_p[start_id];
            T_p[SPLIT] = T_p[start_id + 1];
        };

        iter(std::true_type{});
        for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)
            iter(std::false_type{});

        float w_max = __saturatef((T_p[0] - 0.5f) / (T_p[0] - T_p[SPLIT]));
        float w_min = 1.f - w_max;
        mDepth      = in_range ? w_max * depth_max + w_min * depth_min : 0.f;
    }
    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside) {
        n_contrib[pix_id] = last_contributor;
#pragma unroll
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        out_alpha[pix_id] = 1.f - T;

        if constexpr (GEOMETRY) {
            out_mdepth[pix_id] = mDepth * rln;
#ifdef NORMALIZED_NORMAL
            float len_normal      = norm3df(Normal[0], Normal[1], Normal[2]);
            normal_length[pix_id] = len_normal;
            len_normal            = fmaxf(len_normal, NORMALIZE_EPS);
#else
            float len_normal = 1.f - T;
#endif
#pragma unroll
            for (int ch = 0; ch < 3; ch++)
                out_normal[ch * H * W + pix_id] = last_contributor ? Normal[ch] / len_normal : 0.f;
        }
    }
    if (block.thread_rank() == 0)
        max_contributors[block_id] = block_max;
}

// the Bool inputs can be replaced by an enumeration variable for different functions.
void FORWARD::render(
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
    bool require_depth) {
#define RENDER_CUDA_CALL(template_depth)                                                \
    renderCUDA<NUM_CHANNELS, template_depth, SPLIT, SPLIT_ITERATIONS><<<grid, block>>>( \
        ranges, point_list, W, H, means2D, conic_opacity, colors,                       \
        ray_planes, normals, focal_x, focal_y,                                          \
        n_contrib, max_contributor, bg_color, out_color, out_alpha,                     \
        out_normal, out_mdepth, normal_length)

    if (require_depth)
        RENDER_CUDA_CALL(true);
    else
        RENDER_CUDA_CALL(false);

#undef RENDER_CUDA_CALL
}

void FORWARD::preprocess(
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
    bool prefiltered) {
    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P, SHD, SHM, SGD, SGM,
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3D_precomp,
        shs,
        sg_axis,
        sg_sharpness,
        sg_color,
        scale_modifier,
        viewmatrix,
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        kernel_size,
        radii,
        clamped,
        means2D,
        depths,
        ray_planes,
        normals,
        rgb,
        conic_opacity,
        grid,
        tiles_touched,
        prefiltered);
}
