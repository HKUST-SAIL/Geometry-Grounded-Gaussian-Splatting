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
#include "render_backward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>
namespace cg = cooperative_groups;

// refer to gsplat https://github.com/nerfstudio-project/gsplat/blob/65042cc501d1cdbefaf1d6f61a9a47575eec8c71/gsplat/cuda/include/Utils.cuh#L94
template <uint32_t DIM, class WarpT>
__forceinline__ __device__ void warpSum(float* val, WarpT& warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float& val, WarpT& warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float2& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float3& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float4& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
    val.w = cg::reduce(warp, val.w, cg::plus<float>());
}
// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ __forceinline__ void computeColorFromSHSG(int idx, int SHD, int SHM, int SGD, int SGM, const glm::vec3* means, glm::vec3 campos,
                                                     const float* shs, const glm::vec3* sg_axis, const float* sg_sharpness, const glm::vec3* sg_color, const bool* clamped, const glm::vec3* dL_dcolor,
                                                     glm::vec3* dL_dmeans, glm::vec3* dL_dshs, glm::vec3* dL_dsg_axis, float* dL_dsg_sharpness, glm::vec3* dL_dsg_color) {
    // Compute intermediate values, as it is done during forward
    glm::vec3 pos      = means[idx];
    glm::vec3 dir_orig = pos - campos;
    glm::vec3 dir      = dir_orig / glm::length(dir_orig);

    const glm::vec3* sh = reinterpret_cast<const glm::vec3*>(shs) + idx * SHM;

    // Use PyTorch rule for clamping: if clamping was applied,
    // gradient becomes 0.
    glm::vec3 dL_dRGB = dL_dcolor[idx];
    dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
    dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
    dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

    glm::vec3 dRGBdx(0, 0, 0);
    glm::vec3 dRGBdy(0, 0, 0);
    glm::vec3 dRGBdz(0, 0, 0);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    // Target location for this Gaussian to write SH gradients to
    glm::vec3* dL_dsh = dL_dshs + idx * SHM;

    // No tricks here, just high school-level calculus.
    float dRGBdsh0 = SH_C0;
    dL_dsh[0]      = dRGBdsh0 * dL_dRGB;
    if (SHD > 0) {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1]      = dRGBdsh1 * dL_dRGB;
        dL_dsh[2]      = dRGBdsh2 * dL_dRGB;
        dL_dsh[3]      = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (SHD > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4]      = dRGBdsh4 * dL_dRGB;
            dL_dsh[5]      = dRGBdsh5 * dL_dRGB;
            dL_dsh[6]      = dRGBdsh6 * dL_dRGB;
            dL_dsh[7]      = dRGBdsh7 * dL_dRGB;
            dL_dsh[8]      = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

            if (SHD > 2) {
                float dRGBdsh9  = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9]       = dRGBdsh9 * dL_dRGB;
                dL_dsh[10]      = dRGBdsh10 * dL_dRGB;
                dL_dsh[11]      = dRGBdsh11 * dL_dRGB;
                dL_dsh[12]      = dRGBdsh12 * dL_dRGB;
                dL_dsh[13]      = dRGBdsh13 * dL_dRGB;
                dL_dsh[14]      = dRGBdsh14 * dL_dRGB;
                dL_dsh[15]      = dRGBdsh15 * dL_dRGB;

                dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
                           SH_C3[1] * sh[10] * yz +
                           SH_C3[2] * sh[11] * -2.f * xy +
                           SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                           SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                           SH_C3[5] * sh[14] * 2.f * xz +
                           SH_C3[6] * sh[15] * 3.f * (xx - yy));

                dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
                           SH_C3[1] * sh[10] * xz +
                           SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
                           SH_C3[3] * sh[12] * -3.f * 2.f * yz +
                           SH_C3[4] * sh[13] * -2.f * xy +
                           SH_C3[5] * sh[14] * -2.f * yz +
                           SH_C3[6] * sh[15] * -3.f * 2.f * xy);

                dRGBdz += (SH_C3[1] * sh[10] * xy +
                           SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                           SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                           SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                           SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }

    // The view direction is an input to the computation. View direction
    // is influenced by the Gaussian's mean, so SHs gradients
    // must propagate back into 3D position.
    glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

    const glm::vec3* sg_axis_p  = reinterpret_cast<const glm::vec3*>(sg_axis) + idx * SGM;
    const float* sg_sharpness_p = sg_sharpness + idx * SGM;
    const glm::vec3* sg_color_p = reinterpret_cast<const glm::vec3*>(sg_color) + idx * SGM;
    glm::vec3* dL_dsg_axis_p    = reinterpret_cast<glm::vec3*>(dL_dsg_axis) + idx * SGM;
    float* dL_dsg_sharpness_p   = dL_dsg_sharpness + idx * SGM;
    glm::vec3* dL_dsg_color_p   = reinterpret_cast<glm::vec3*>(dL_dsg_color) + idx * SGM;
    for (int g = 0; g < SGD; g++) {
        const glm::vec3 axis   = sg_axis_p[g];
        const float sharpness  = sg_sharpness_p[g];
        const glm::vec3 gcolor = sg_color_p[g];
        float aux              = glm::dot(axis, dir) - 1.0f;
        float gaussian         = expf(sharpness * aux);
        dL_dsg_color_p[g]      = dL_dRGB * gaussian;
        float dL_dgaussian     = glm::dot(gcolor, dL_dRGB);
        float dL_dexponent     = dL_dgaussian * gaussian;
        dL_dsg_sharpness_p[g]  = dL_dexponent * aux;
        float dL_daux          = dL_dexponent * sharpness;
        dL_dsg_axis_p[g]       = dL_daux * dir;
        dL_ddir += dL_daux * axis;
    }

    // Account for normalization of direction
    float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

    // Gradients of loss w.r.t. Gaussian means, but only the portion
    // that is caused because the mean affects the view-dependent color.
    // Additional mean gradient is accumulated in below methods.
    dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__device__ __forceinline__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::mat3& R, const float4 rot, const float* dL_dcov3D, const glm::vec3 dL_dr, uint32_t min_id, glm::vec3* dL_dscales, glm::vec4* dL_drots) {
    float r = rot.x;
    float x = rot.y;
    float y = rot.z;
    float z = rot.w;

    glm::mat3 S = glm::mat3(1.0f);
    S[0][0]     = scale.x;
    S[1][1]     = scale.y;
    S[2][2]     = scale.z;

    glm::mat3 M = S * R;

    glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
    glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

    // Convert per-element covariance loss gradients to matrix form
    glm::mat3 dL_dSigma = glm::mat3(
        dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
        0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
        0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]);

    // Compute loss gradient w.r.t. matrix M
    // dSigma_dM = 2 * M
    glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

    glm::mat3 Rt     = glm::transpose(R);
    glm::mat3 dL_dMt = glm::transpose(dL_dM);

    // Gradients of loss w.r.t. scale
    glm::vec3* dL_dscale = dL_dscales + idx;
    dL_dscale->x         = glm::dot(Rt[0], dL_dMt[0]);
    dL_dscale->y         = glm::dot(Rt[1], dL_dMt[1]);
    dL_dscale->z         = glm::dot(Rt[2], dL_dMt[2]);

    dL_dMt[0] *= scale.x;
    dL_dMt[1] *= scale.y;
    dL_dMt[2] *= scale.z;

    dL_dMt[min_id] += dL_dr;

    // Gradients of loss w.r.t. normalized quaternion
    glm::vec4 dL_dq;
    dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

    // Gradients of loss w.r.t. unnormalized quaternion
    float4* dL_drot = (float4*)(dL_drots + idx);
    *dL_drot        = float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w}; // dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(
    int P,
    const float3* means,
    const int* radii,
    const float* cov3Ds,
    const glm::vec3* scales,
    const float4* rotations,
    const float* opacities,
    const float mod,
    const float h_x, float h_y,
    const float tan_fovx, const float tan_fovy,
    const float kernel_size,
    const float* view_matrix,
    const float4* dL_dconics,
    const float4* dL_dray_planes,
    const glm::vec3* dL_dnormals,
    glm::vec3* dL_dmeans,
    float* dL_dcov,
    glm::vec3* dL_dscales,
    glm::vec4* dL_drots,
    float* dL_dopacity) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    // Fetch gradients, recompute 2D covariance and relevant
    // intermediate forward results needed in the backward.
    float3 mean                = means[idx];
    float4 dL_dconic           = dL_dconics[idx];
    const glm::vec3 dL_dnormal = dL_dnormals[idx];

    auto load_ray_plane_grad = [h_x, h_y](const float4* dL_dray_planes, auto idx) {
        float4 dL_dray_plane = dL_dray_planes[idx];
        dL_dray_plane.x /= h_x;
        dL_dray_plane.y /= h_y;
        return dL_dray_plane;
    };
    const float4 dL_dray_plane = load_ray_plane_grad(dL_dray_planes, idx);
    float dL_dtc               = dL_dray_plane.z;
    float dL_drsigma           = dL_dray_plane.w;

    float3 t     = transformPoint4x3(mean, view_matrix);
    float rtc    = rnorm3df(t.x, t.y, t.z);
    float3 dL_dt = {t.x * rtc * dL_dtc, t.y * rtc * dL_dtc, t.z * rtc * dL_dtc};

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    float u          = t.x / t.z;
    float v          = t.y / t.z;
    t.x              = fminf(limx, fmaxf(-limx, u)) * t.z;
    t.y              = fminf(limy, fmaxf(-limy, v)) * t.z;

    const float x_grad_mul = u < -limx || u > limx ? 0 : 1;
    const float y_grad_mul = v < -limy || v > limy ? 0 : 1;

    u = t.x / t.z;
    v = t.y / t.z;

    glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
                            0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
                            0, 0, 0);

    glm::mat3 W = glm::mat3(
        view_matrix[0], view_matrix[4], view_matrix[8],
        view_matrix[1], view_matrix[5], view_matrix[9],
        view_matrix[2], view_matrix[6], view_matrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 cov2D;
    glm::mat3 cov_cam_inv;

    glm::mat3 Vrk_inv;
    glm::mat3 Vrk;
    glm::mat3 Vrk_eigen_vector;
    glm::vec3 Vrk_eigen_value;

    glm::mat3 R;
    glm::vec3 scale_local;
    float4 rot;

    bool well_conditioned;
    unsigned int min_id;

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

    if (scales) {
        // Create scaling matrix
        glm::mat3 S            = glm::mat3(1.0f);
        glm::mat3 S_inv        = glm::mat3(1.0f);
        const glm::vec3* scale = scales + idx;
        scale_local            = {mod * scale->x, mod * scale->y, mod * scale->z};
        S[0][0]                = scale_local[0];
        S[1][1]                = scale_local[1];
        S[2][2]                = scale_local[2];

        S_inv[0][0] = __frcp_rn(scale_local[0]);
        S_inv[1][1] = __frcp_rn(scale_local[1]);
        S_inv[2][2] = __frcp_rn(scale_local[2]);

        min_id = find_min_from_triple(scale_local);
        // well_conditioned = scale_local[min_id] > 1E-7;
        well_conditioned = true;

        // Normalize quaternion to get valid rotation
        rot     = rotations[idx];
        float r = rot.x;
        float x = rot.y;
        float y = rot.z;
        float z = rot.w;

        // Compute rotation matrix from quaternion
        R = glm::mat3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
            2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
            2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

        glm::mat3 M = S * R * T;

        // Compute 3D world covariance matrix Sigma
        cov2D = glm::transpose(M) * M;
        M     = S * R;
        Vrk   = glm::transpose(M) * M;

        if (well_conditioned) {
            glm::mat3 M_inv = S_inv * R * W;
            cov_cam_inv     = glm::transpose(M_inv) * M_inv;
            M_inv           = S_inv * R;
            Vrk_inv         = glm::transpose(M_inv) * M_inv;
        } else {
            glm::vec3 M_inv = R[min_id] * W;
            cov_cam_inv     = glm::outerProduct(M_inv, M_inv);
            Vrk_inv         = glm::outerProduct(R[min_id], R[min_id]);
        }
    } else {
        // Reading location of 3D covariance for this Gaussian
        const float* cov3D = cov3Ds + 6 * idx;
        Vrk                = glm::mat3(
            cov3D[0], cov3D[1], cov3D[2],
            cov3D[1], cov3D[3], cov3D[4],
            cov3D[2], cov3D[4], cov3D[5]);

        cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

        glm_modification::findEigenvaluesSymReal(Vrk, Vrk_eigen_value, Vrk_eigen_vector);

        min_id = find_min_from_triple(Vrk_eigen_value);

        well_conditioned = Vrk_eigen_value[min_id] > 1E-8;
        if (well_conditioned) {
            glm::mat3 diag = glm::mat3(1 / Vrk_eigen_value[0], 0, 0,
                                       0, 1 / Vrk_eigen_value[1], 0,
                                       0, 0, 1 / Vrk_eigen_value[2]);
            Vrk_inv        = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
        } else {
            glm::vec3 eigenvector_min = Vrk_eigen_vector[min_id];
            Vrk_inv                   = glm::outerProduct(eigenvector_min, eigenvector_min);
        }
        cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
    }

    const float det_0 = fmaxf(1e-6f, cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1]);
    const float det_1 = fmaxf(1e-6f, (cov2D[0][0] + kernel_size) * (cov2D[1][1] + kernel_size) - cov2D[0][1] * cov2D[0][1]);
    const float coef  = sqrtf(det_0 / det_1);

    glm::vec3 uvh   = {u, v, 1};
    glm::vec3 uvh_m = cov_cam_inv * uvh;
    // glm::vec3 uvh_mn = glm::normalize(uvh_m);

    float u2 = u * u;
    float v2 = v * v;
    float uv = u * v;

    glm::mat3 dL_dVrk;
    glm::vec3 plane;
    float dL_du;
    float dL_dv;
    float dL_dz;
    glm::mat3 dL_dnJ;
    glm::vec3 dL_dr;
    glm::vec3 normal_vector;
    {
        float vb = glm::dot(uvh_m, uvh);
        // float vbn    = glm::dot(uvh_mn, uvh);
        float l      = norm3df(t.x, t.y, t.z);
        glm::mat3 nJ = glm::mat3(
            1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
            0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
            t.x / l, t.y / l, t.z / l);

        glm::mat3 nJ_inv = glm::mat3(
            v2 + 1, -uv, 0,
            -uv, u2 + 1, 0,
            -u, -v, 0);

        float clamp_vb = fmaxf(vb, 1e-7f);
        // float clamp_vbn     = fmaxf(vbn, 1e-7f);
        float ray_len2      = u2 + v2 + 1;
        float ray_len_inv   = rsqrtf(ray_len2);
        float factor_normal = l / ray_len2;
        glm::vec3 uvh_m_vb  = uvh_m / clamp_vb;
        plane               = nJ_inv * uvh_m_vb;
#if 0
        glm::vec2 ray_plane         = {plane.x * factor_normal, plane.y * factor_normal};
#endif
        glm::vec3 ray_normal_vector = {-plane.x * factor_normal, -plane.y * factor_normal, -1};

        glm::vec3 cam_normal_vector     = nJ * ray_normal_vector;
        normal_vector                   = glm::normalize(cam_normal_vector);
        float rlv                       = rnorm3df(normal_vector.x, normal_vector.y, normal_vector.z);
        glm::vec3 dL_dcam_normal_vector = (dL_dnormal - normal_vector * glm::dot(normal_vector, dL_dnormal)) * rlv;
        glm::vec3 dL_dray_normal_vector = glm::transpose(nJ) * dL_dcam_normal_vector;
        dL_dnJ                          = glm::outerProduct(dL_dcam_normal_vector, ray_normal_vector);
        float dL_dfactor_normal         = plane.x * (-dL_dray_normal_vector.x + dL_dray_plane.x) + plane.y * (-dL_dray_normal_vector.y + dL_dray_plane.y);

        glm::vec2 dL_dplane = glm::vec2(
            (-dL_dray_normal_vector.x + dL_dray_plane.x) * factor_normal,
            (-dL_dray_normal_vector.y + dL_dray_plane.y) * factor_normal);
        glm::vec3 dL_dplane_append = glm::vec3(dL_dplane.x, dL_dplane.y, 0);

        const float aux = dL_dplane.x * plane.x + dL_dplane.y * plane.y;

        glm::vec3 W_uvh = W * uvh;

        glm::vec3 dL_duvh_plane = 2 * (-aux) * uvh_m_vb                                                   // Denominator
                                  + (cov_cam_inv / clamp_vb) * glm::transpose(nJ_inv) * dL_dplane_append; // Numerator
        // rsigmat = sqrtf(vb / ray_len2);
        const float rsigmat         = sqrtf(vb / ray_len2);
        float dL_dray_len2_sigma_x2 = -dL_drsigma * rsigmat / ray_len2;
        float dL_du_sigma           = dL_dray_len2_sigma_x2 * u;
        float dL_dv_sigma           = dL_dray_len2_sigma_x2 * v;

        // nJ = glm::mat3(
        //     1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
        //     0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
        //     t.x / l, t.y / l, t.z / l);
        float aux_nJ   = (-dL_dnJ[2][0] * u - dL_dnJ[2][1] * v - dL_dnJ[2][2]) / ray_len2 * ray_len_inv;
        float dL_du_nJ = -dL_dnJ[0][2] / t.z + dL_dnJ[2][0] * ray_len_inv + aux_nJ * u;
        float dL_dv_nJ = -dL_dnJ[1][2] / t.z + dL_dnJ[2][1] * ray_len_inv + aux_nJ * v;
        float dL_dz_nJ = (dL_dnJ[0][0] + dL_dnJ[1][1] - dL_dnJ[0][2] * u - dL_dnJ[1][2] * v) / (-t.z * t.z);

        // nJ_inv = glm::mat3(
        //     v2 + 1, -uv, 0,
        //     -uv, u2 + 1, 0,
        //     -u, -v, 0);
        glm::mat3 dL_dnJ_inv = glm::outerProduct(dL_dplane_append, uvh_m_vb);
        float dL_du_plane    = dL_duvh_plane.x + (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-v) + 2 * dL_dnJ_inv[1][1] * u - dL_dnJ_inv[2][0];
        float dL_dv_plane    = dL_duvh_plane.y + (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-u) + 2 * dL_dnJ_inv[0][0] * v - dL_dnJ_inv[2][1];

        float aux_factor   = dL_dfactor_normal * (-t.z / ray_len2 * ray_len_inv);
        float dL_du_factor = aux_factor * u;
        float dL_dv_factor = aux_factor * v;
        float dL_dz_factor = dL_dfactor_normal * ray_len_inv;

        dL_du = dL_du_nJ + dL_du_plane + dL_du_factor + dL_du_sigma;
        dL_dv = dL_dv_nJ + dL_dv_plane + dL_dv_factor + dL_dv_sigma;
        dL_dz = dL_dz_nJ + dL_dz_factor;

        // float dL_dvb = -aux / clamp_vb + dL_drsigma * 0.5f * rsqrtf(ray_len2 * vb);
        float dL_dvb_xvb = -aux + dL_drsigma * 0.5f * rsigmat;
        if (well_conditioned) {
            // dL_dVrk      = -glm::outerProduct(Vrk_inv * W_uvh, (Vrk_inv / clamp_vb) * (W_uvh * (-aux) + W * glm::transpose(nJ_inv) * dL_dplane_append));
            // dL_dVrk      = -glm::outerProduct(Vrk_inv * W_uvh, (Vrk_inv / clamp_vb) * (W * glm::transpose(nJ_inv) * dL_dplane_append)
            //                                     + Vrk_inv * W_uvh * dL_dvb);
            dL_dVrk = -glm::outerProduct(Vrk_inv * W_uvh,
                                         Vrk_inv * (W * glm::transpose(nJ_inv) * dL_dplane_append + W_uvh * dL_dvb_xvb));
            dL_dVrk = dL_dVrk / vb;
            dL_dr   = glm::vec3(0.f, 0.f, 0.f);
        } else {
            dL_dVrk                        = glm::mat3(0, 0, 0, 0, 0, 0, 0, 0, 0);
            glm::vec3 nJ_inv_dL_dplane_xvb = glm::transpose(nJ_inv) * glm::vec3(dL_dplane.x, dL_dplane.y, 0);
            glm::mat3 dL_dVrk_inv          = glm::outerProduct(W_uvh, W_uvh * dL_dvb_xvb + W * nJ_inv_dL_dplane_xvb) / vb;
            if (scales) {
                glm::vec3 eigenvector_min = R[min_id];
                dL_dr                     = (dL_dVrk_inv + glm::transpose(dL_dVrk_inv)) * eigenvector_min;
            } else {
                glm::vec3 eigenvector_min = Vrk_eigen_vector[min_id];
                glm::vec3 dL_dv           = (dL_dVrk_inv + glm::transpose(dL_dVrk_inv)) * eigenvector_min;
                for (int j = 1; j < 3; j++) {
                    int k       = (j + min_id) % 3;
                    float scale = glm::dot(Vrk_eigen_vector[k], dL_dv) / fminf(Vrk_eigen_value[min_id] - Vrk_eigen_value[k], -1e-7f);
                    dL_dVrk += glm::outerProduct(Vrk_eigen_vector[k] * scale, eigenvector_min);
                }
            }
        }
    }

    const float opacity      = opacities[idx];
    const float dL_dcoef     = dL_dconic.w * opacity;
    const float dL_dsqrtcoef = dL_dcoef * 0.5f / (coef + 1e-6f);
    const float dL_ddet0     = dL_dsqrtcoef / det_1;
    const float dL_ddet1     = -dL_ddet0 * coef;
    // TODO gradient is zero if det_0 or det_1 < 0
    const float dcoef_da = dL_ddet0 * cov2D[1][1] + dL_ddet1 * (cov2D[1][1] + kernel_size);
    const float dcoef_db = (-2.f * cov2D[0][1]) * (dL_ddet0 + dL_ddet1);
    const float dcoef_dc = dL_ddet0 * cov2D[0][0] + dL_ddet1 * (cov2D[0][0] + kernel_size);
    // Use helper variables for 2D covariance entries. More compact.
    float a = cov2D[0][0] + kernel_size;
    float b = cov2D[0][1];
    float c = cov2D[1][1] + kernel_size;

    float denom = a * c - b * b;
    float dL_da = 0, dL_db = 0, dL_dc = 0;
    float denom2inv = 1.0f / ((denom * denom) + 1e-7f);

    float dL_dcov_local[6];
    if (denom2inv != 0) {
        // Gradients of loss w.r.t. entries of 2D covariance matrix,
        // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
        // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        dL_da += dcoef_da;
        dL_dc += dcoef_dc;
        dL_db += dcoef_db;

        // update dL_dopacity
        dL_dopacity[idx] = dL_dconic.w * coef;

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
        // given gradients w.r.t. 2D covariance matrix (diagonal).
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov_local[0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
        dL_dcov_local[3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
        dL_dcov_local[5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
        // given gradients w.r.t. 2D covariance matrix (off-diagonal).
        // Off-diagonal elements appear twice --> double the gradient.
        // cov2D = transpose(T) * transpose(Vrk) * T;
        dL_dcov_local[1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
        dL_dcov_local[2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
        dL_dcov_local[4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
    } else {
        for (int i = 0; i < 6; i++)
            dL_dcov_local[i] = 0;
    }
    dL_dcov_local[0] += dL_dVrk[0][0];
    dL_dcov_local[3] += dL_dVrk[1][1];
    dL_dcov_local[5] += dL_dVrk[2][2];
    dL_dcov_local[1] += dL_dVrk[0][1] + dL_dVrk[1][0];
    dL_dcov_local[2] += dL_dVrk[0][2] + dL_dVrk[2][0];
    dL_dcov_local[4] += dL_dVrk[1][2] + dL_dVrk[2][1];

    if (scales)
        computeCov3D(idx, scale_local, mod, R, rot, dL_dcov_local, dL_dr, min_id, dL_dscales, dL_drots);
    else {
        for (int i = 0; i < 6; i++)
            dL_dcov[6 * idx + i] = dL_dcov_local[i];
    }

    // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
    // cov2D = transpose(T) * transpose(Vrk) * T;
    float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
                    (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
    float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
                    (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
    float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
                    (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
    float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
                    (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
    float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
                    (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
    float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
                    (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

    // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
    // T = W * J
    float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    float tz  = 1.f / t.z;
    float tz2 = tz * tz;
    float tz3 = tz2 * tz;

    // glm::mat3 nJ = glm::mat3(
    // 		1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
    // 		0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
    // 		t.x/l, t.y/l, t.z/l);
    float dL_dtx = x_grad_mul * (-h_x * tz2 * dL_dJ02 + dL_du * tz);
    float dL_dty = y_grad_mul * (-h_y * tz2 * dL_dJ12 + dL_dv * tz);
    float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + ((1 + x_grad_mul) * h_x * t.x) * tz3 * dL_dJ02 + ((1 + y_grad_mul) * h_y * t.y) * tz3 * dL_dJ12 // rendering gradient
                   - (x_grad_mul * dL_du * t.x + y_grad_mul * dL_dv * t.y) * tz2 + dL_dz;                                                                       // depth gradient

    // Account for transformation of mean to t
    // t = transformPoint4x3(mean, view_matrix);
    float3 dL_dmean = transformVec4x3Transpose({dL_dtx + dL_dt.x, dL_dty + dL_dt.y, dL_dtz + dL_dt.z}, view_matrix);

    // Gradients of loss w.r.t. Gaussian means, but only the portion
    // that is caused because the mean affects the covariance matrix.
    // Additional mean gradient is accumulated in BACKWARD::preprocess.
    dL_dmeans[idx] = glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <int C>
__global__ void preprocessCUDA(
    int P, int SHD, int SHM, int SGD, int SGM,
    const float3* means,
    const float* shs,
    const float* sg_axis,
    const float* sg_sharpness,
    const float* sg_color,
    const float scale_modifier,
    const float* view,
    const float* proj,
    const glm::vec3* campos,
    const int* radii,
    const bool* clamped,
    const float3* dL_dmean2D,
    const float* dL_dcolor,
    glm::vec3* dL_dmeans,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dsg_axis,
    float* dL_dsg_sharpness,
    float* dL_dsg_color) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    float3 m = means[idx];

    // Taking care of gradients from the screenspace points
    float4 m_hom = transformPoint4x4(m, proj);
    float m_w    = 1.0f / (m_hom.w + 0.0000001f);

    // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
    // from rendering procedure
    glm::vec3 dL_dmean;
    float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
    float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
    dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

    // That's the third part of the mean gradient.
    dL_dmeans[idx] += dL_dmean;

    // Compute gradient updates due to computing colors from SHs
    if (shs)
        computeColorFromSHSG(idx, SHD, SHM, SGD, SGM, (glm::vec3*)means, *campos,
                             shs, (const glm::vec3*)sg_axis, sg_sharpness, (const glm::vec3*)sg_color,
                             clamped, (const glm::vec3*)dL_dcolor,
                             (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, (glm::vec3*)dL_dsg_axis, dL_dsg_sharpness, (glm::vec3*)dL_dsg_color);
}

// Backward version of the rendering procedure.
template <uint32_t C, bool GEOMETRY>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    renderCUDA(
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        const int W, const int H,
        const float* __restrict__ bg_color,
        const float2* __restrict__ points_xy_image,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ colors,
        const float* __restrict__ depths,
        const float4* __restrict__ ray_planes,
        const float3* __restrict__ normals,
        const float* __restrict__ alphas,
        const float* __restrict__ normalmap,
        const float* __restrict__ mdepth,
        const float* __restrict__ normal_length,
        const uint32_t* __restrict__ n_contrib,
        const uint32_t* __restrict__ max_contributors,
        const float* __restrict__ dL_dpixels,
        const float* __restrict__ dL_dpixel_mdepths,
        const float* __restrict__ dL_dalphas,
        const float* __restrict__ dL_dpixel_normals,
        const float focal_x,
        const float focal_y,
        float3* __restrict__ dL_dmean2D,
        float4* __restrict__ dL_dconic2D,
        float* __restrict__ dL_dcolors,
        float4* __restrict__ dL_dray_planes,
        float3* __restrict__ dL_dnormals) {
    // We rasterize again. Compute necessary block info.
    auto block                       = cg::this_thread_block();
    cg::thread_block_tile<32> warp   = cg::tiled_partition<32>(block);
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    const uint32_t pix_id            = W * pix.y + pix.x;
    const float2 pixf                = {static_cast<float>(pix.x), static_cast<float>(pix.y)};
    const float2 pixnf               = {(pixf.x - static_cast<float>(W - 1) / 2.f) / focal_x, (pixf.y - static_cast<float>(H - 1) / 2.f) / focal_y};

    const bool inside              = pix.x < W && pix.y < H;
    const int block_id             = block.group_index().y * horizontal_blocks + block.group_index().x;
    const uint2 range              = ranges[block_id];
    const uint32_t max_contributor = max_contributors[block_id];

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float collected_colors[C * BLOCK_SIZE];
    [[maybe_unused]] __shared__ float4 collected_ray_planes[BLOCK_SIZE];
    [[maybe_unused]] __shared__ float3 collected_normals[BLOCK_SIZE];

    // In the forward, we stored the final value for T, the
    // product of all (1 - alpha) factors.
    const float w_final = inside ? alphas[pix_id] : 0.f;
    const float T_final = 1.f - w_final;
    [[maybe_unused]] float mDepth;

    float T = T_final;

    // We start from the back. The ID of the last contributing
    // Gaussian is known from each pixel from the forward.
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    float accum_rec[C] = {0};
    float dL_dpixel[C];
    float dL_dfinalT;
    [[maybe_unused]] float accum_t_rec = 0;
    [[maybe_unused]] float dL_dpixel_t;
    [[maybe_unused]] float dL_dpixel_mt;
    [[maybe_unused]] float accum_normal_rec[3] = {0};
    [[maybe_unused]] float dL_dpixel_normal[3];

    if (inside) {
#pragma unroll
        for (int ch = 0; ch < C; ch++)
            dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];

        dL_dfinalT = -dL_dalphas[pix_id];
#pragma unroll
        for (int ch = 0; ch < C; ch++)
            dL_dfinalT += bg_color[ch] * dL_dpixel[ch];

        if constexpr (GEOMETRY) {
            const float inv_w = 1 / w_final;
            const float rln   = rnorm3df(pixnf.x, pixnf.y, 1.f);
            dL_dpixel_mt      = dL_dpixel_mdepths[pix_id] * rln;

            glm::vec3 dL_dpixel_normaln = glm::vec3(dL_dpixel_normals[pix_id],
                                                    dL_dpixel_normals[H * W + pix_id],
                                                    dL_dpixel_normals[2 * H * W + pix_id]);
            glm::vec3 normaln           = glm::vec3(normalmap[pix_id],
                                                    normalmap[H * W + pix_id],
                                                    normalmap[2 * H * W + pix_id]);
#ifdef NORMALIZED_NORMAL
            float normal_len          = normal_length[pix_id];
            const float small         = static_cast<float>(normal_len < NORMALIZE_EPS);
            const float large         = 1.0f - small;
            const float denom         = small * NORMALIZE_EPS + large * normal_len;
            const glm::vec3 proj      = glm::dot(dL_dpixel_normaln, normaln) * normaln * large;
            const glm::vec3 numerator = dL_dpixel_normaln - proj;
#pragma unroll
            for (int ch = 0; ch < 3; ch++)
                dL_dpixel_normal[ch] = numerator[ch] / denom;
#else
#pragma unroll
            for (int ch = 0; ch < 3; ch++)
                dL_dpixel_normal[ch] = dL_dpixel_normaln[ch] * inv_w;
            // dL_dfinalT += glm::dot(dL_dpixel_normaln, normaln) * inv_w;
            dL_dfinalT += dL_dpixel_normal[0] * normaln[0] + dL_dpixel_normal[1] * normaln[1] + dL_dpixel_normal[2] * normaln[2];
#endif
            mDepth = mdepth[pix_id] * norm3df(pixnf.x, pixnf.y, 1.f);
        }
    }

    [[maybe_unused]] float dL_dmt_dT_dtm = 0.f;
    const int rounds     = (max_contributor + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if constexpr (GEOMETRY) {
        float dT_dtm    = 0.f;
        int toDo        = max_contributor;
        int contributor = 0;
        bool done       = (mDepth == 0) || (last_contributor == 0) || !inside;
        // Iterate over batches until all done or range is complete
        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            // End if entire block votes that it is done rasterizing
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
                if (power > 0.0f)
                    continue;

                float alpha = fminf(0.99f, con_o.w * expf(power));
                if (alpha < 1.0f / 255.0f)
                    continue;

                float4 ray_plane    = collected_ray_planes[j];
                const float t_peak  = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                const float rsigma  = ray_plane.w;
                const float t_delta = (mDepth - t_peak) * rsigma;
                const float G_exp   = expf(-0.5f * t_delta * t_delta);
                const float Gt      = alpha * G_exp;
                dT_dtm += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
            }
        }
        dL_dmt_dT_dtm = dL_dpixel_mt / fmaxf(-dT_dtm, 1e-7f);
    }

    bool done            = !inside;
    int toDo             = max_contributor;
    uint32_t contributor = toDo;

    float last_alpha                      = 0;
    float last_color[C]                   = {0};
    [[maybe_unused]] float last_t         = 0;
    [[maybe_unused]] float last_normal[3] = {0};

    // Gradient of pixel coordinate w.r.t. normalized
    // screen-space viewport corrdinates (-1 to 1)
    const float ddelx_dx = 0.5f * W;
    const float ddely_dy = 0.5f * H;

    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // Load auxiliary data into shared memory, start in the BACK
        // and load them in revers order.
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (progress < max_contributor) {
            const int coll_id                            = point_list[range.x + max_contributor - progress - 1];
            collected_id[block.thread_rank()]            = coll_id;
            collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            for (int ch = 0; ch < C; ch++)
                collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];

            if constexpr (GEOMETRY) {
                collected_ray_planes[block.thread_rank()] = ray_planes[coll_id];
                collected_normals[block.thread_rank()]    = normals[coll_id];
            }
        }
        block.sync();

        // Iterate over Gaussians
        for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
            // refer to gsplat that uses warp-wise reduction before atomicAdd
            // Keep track of current Gaussian ID. Skip, if this one
            // is behind the last contributor for this pixel.
            contributor--;

            // Compute blending values, as before.
            const float2 xy    = collected_xy[j];
            const float2 d     = {xy.x - pixf.x, xy.y - pixf.y};
            const float4 con_o = collected_conic_opacity[j];

            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

            const float G     = expf(power);
            const float alpha = fminf(0.99f, con_o.w * G);
            bool valid        = !(done || (contributor >= last_contributor) || (power > 0.0f) || (alpha < 1.0f / 255.0f));

            if (!warp.any(valid))
                continue;

            float dL_dcolors_local[C]                    = {0};
            [[maybe_unused]] float3 dL_dnormals_local    = {0};
            [[maybe_unused]] float2 dL_dray_planes_local = {0};
            [[maybe_unused]] float dL_dtc_local          = 0.f;
            [[maybe_unused]] float dL_drsigma_local      = 0.f;
            float3 dL_dmean2D_local                      = {0};
            float4 dL_dconic2D_local                     = {0};
            if (valid) {
                T                           = T / (1.f - alpha);
                const float blending_weight = alpha * T;

                // Propagate gradients to per-Gaussian colors and keep
                // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
                // pair).
                float dL_dopa = 0.0f;
                for (int ch = 0; ch < C; ch++) {
                    const float c = collected_colors[ch * BLOCK_SIZE + j];
                    // Update last color (to be used in the next iteration)
                    accum_rec[ch]  = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
                    last_color[ch] = c;

                    const float dL_dchannel = dL_dpixel[ch];
                    dL_dopa += (c - accum_rec[ch]) * dL_dchannel;
                    dL_dcolors_local[ch] = blending_weight * dL_dchannel;
                }

                [[maybe_unused]] float dL_dt;
                [[maybe_unused]] float4 ray_plane;
                [[maybe_unused]] float dL_dopa_sigma;
                if constexpr (GEOMETRY) {
                    const float3 normal          = collected_normals[j];
                    const float* normal_ptr      = reinterpret_cast<const float*>(&normal);
                    float* dL_dnormals_local_ptr = reinterpret_cast<float*>(&dL_dnormals_local);
#pragma unroll
                    for (int ch = 0; ch < 3; ch++) {
                        const float n           = normal_ptr[ch];
                        accum_normal_rec[ch]    = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
                        last_normal[ch]         = n;
                        const float dL_dchannel = dL_dpixel_normal[ch];
                        dL_dopa += (n - accum_normal_rec[ch]) * dL_dchannel;
                        dL_dnormals_local_ptr[ch] = blending_weight * dL_dchannel;
                    }
                    ray_plane          = collected_ray_planes[j];
                    const float t_peak = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                    const float rsigma = ray_plane.w;
                    float t_delta      = (mDepth - t_peak) * rsigma;
                    float G_exp        = expf(-0.5f * t_delta * t_delta);
                    float Gt           = alpha * G_exp;

                    float dL_dGt           = dL_dmt_dT_dtm * 0.25f / (1.f - Gt);
                    dL_dGt                 = mDepth > t_peak ? dL_dGt : -dL_dGt;
                    dL_dGt                 = rsigma > 0 ? dL_dGt : 0.f;
                    dL_dopa_sigma          = dL_dGt * G_exp - dL_dmt_dT_dtm * (t_delta > 0 ? 0.5f / (1.f - alpha) : 0.f);
                    float dL_ddelta        = -dL_dGt * Gt * t_delta;
                    dL_drsigma_local       = dL_ddelta * (mDepth - t_peak);
                    const float dL_dt_peak = -dL_ddelta * rsigma;

                    dL_dt = dL_dt_peak;

                    dL_dtc_local           = dL_dt;
                    dL_dray_planes_local.x = dL_dt * d.x;
                    dL_dray_planes_local.y = dL_dt * d.y;
                }

                dL_dopa *= T;
                if constexpr (GEOMETRY) {
                    dL_dopa += dL_dopa_sigma;
                }
                dL_dopa += -T_final / (1.f - alpha) * dL_dfinalT;

                // Update last alpha (to be used in the next iteration)
                last_alpha = alpha;

                // Helpful reusable temporary variables
                const float dL_dG    = con_o.w * dL_dopa;
                const float gdx      = G * d.x;
                const float gdy      = G * d.y;
                const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
                const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

                // Update gradients w.r.t. 2D mean position of the Gaussian
                float dL_ddelx = dL_dG * dG_ddelx;
                float dL_ddely = dL_dG * dG_ddely;

                if constexpr (GEOMETRY) {
                    dL_ddelx += dL_dt * ray_plane.x;
                    dL_ddely += dL_dt * ray_plane.y;
                }
                dL_dmean2D_local.x = dL_ddelx * ddelx_dx;
                dL_dmean2D_local.y = dL_ddely * ddely_dy;
                dL_dmean2D_local.z = fabsf(dL_dmean2D_local.x) + fabsf(dL_dmean2D_local.y);

                dL_dconic2D_local.x = -0.5f * gdx * d.x * dL_dG;
                dL_dconic2D_local.y = -0.5f * gdx * d.y * dL_dG;
                dL_dconic2D_local.z = -0.5f * gdy * d.y * dL_dG;
                dL_dconic2D_local.w = G * dL_dopa;
            }
            warpSum<C>(dL_dcolors_local, warp);
            warpSum(dL_dmean2D_local, warp);
            warpSum(dL_dconic2D_local, warp);
            if constexpr (GEOMETRY) {
                warpSum(dL_dnormals_local, warp);
                warpSum(dL_dray_planes_local, warp);
                warpSum(dL_dtc_local, warp);
                warpSum(dL_drsigma_local, warp);
            }
            if (warp.thread_rank() == 0) {
                const int global_id = collected_id[j];
#pragma unroll
                for (int ch = 0; ch < C; ch++) {
                    atomicAdd(&(dL_dcolors[global_id * C + ch]), dL_dcolors_local[ch]);
                }
                atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_local.x);
                atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_local.y);
                atomicAdd(&dL_dmean2D[global_id].z, dL_dmean2D_local.z);
                atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_local.x);
                atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_local.y);
                atomicAdd(&dL_dconic2D[global_id].z, dL_dconic2D_local.z);
                atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_local.w);
                if constexpr (GEOMETRY) {
                    atomicAdd(&dL_dnormals[global_id].x, dL_dnormals_local.x);
                    atomicAdd(&dL_dnormals[global_id].y, dL_dnormals_local.y);
                    atomicAdd(&dL_dnormals[global_id].z, dL_dnormals_local.z);
                    atomicAdd(&dL_dray_planes[global_id].x, dL_dray_planes_local.x);
                    atomicAdd(&dL_dray_planes[global_id].y, dL_dray_planes_local.y);
                    atomicAdd(&dL_dray_planes[global_id].z, dL_dtc_local);
                    atomicAdd(&dL_dray_planes[global_id].w, dL_drsigma_local);
                }
            }
        }
    }
}

void BACKWARD::preprocess(
    int P, int SHD, int SHM, int SGD, int SGM,
    const float3* means3D,
    const float* opacities,
    const glm::vec3* scales,
    const float4* rotations,
    const float* cov3Ds,
    const float* shs,
    const float* sg_axis,
    const float* sg_sharpness,
    const float* sg_color,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const glm::vec3* campos,
    const int* radii,
    const bool* clamped,
    const float4* dL_dconic,
    const float4* dL_dray_plane,
    const float3* dL_dnormals,
    const float3* dL_dmean2D,
    const float* dL_dcolor,
    glm::vec3* dL_dmean3D,
    float* dL_dopacity,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dsg_axis,
    float* dL_dsg_sharpness,
    float* dL_dsg_color) {
    // Propagate gradients for the path of 2D conic matrix computation.
    // Somewhat long, thus it is its own kernel rather than being part of
    // "preprocess". When done, loss gradient w.r.t. 3D means has been
    // modified and gradient w.r.t. 3D covariance matrix has been computed.
    computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
        P,
        means3D,
        radii,
        cov3Ds,
        scales,
        rotations,
        opacities,
        scale_modifier,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        kernel_size,
        viewmatrix,
        dL_dconic,
        dL_dray_plane,
        (const glm::vec3*)dL_dnormals,
        dL_dmean3D,
        dL_dcov3D,
        dL_dscale,
        dL_drot,
        dL_dopacity);

    // Propagate gradients for remaining steps: finish 3D mean gradients,
    // propagate color gradients to SH (if desireD), propagate 3D covariance
    // matrix gradients to scale and rotation.
    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P, SHD, SHM, SGD, SGM,
        means3D,
        shs,
        sg_axis,
        sg_sharpness,
        sg_color,
        scale_modifier,
        viewmatrix,
        projmatrix,
        campos,
        radii,
        clamped,
        dL_dmean2D,
        dL_dcolor,
        dL_dmean3D,
        dL_dscale,
        dL_drot,
        dL_dcov3D,
        dL_dsh,
        dL_dsg_axis,
        dL_dsg_sharpness,
        dL_dsg_color);
}

// the Bool inputs can be replaced by an enumeration variable for different functions.
void BACKWARD::render(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float* bg_color,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float* depths,
    const float4* ray_planes,
    const float3* normals,
    const float* alphas,
    const float* normalmap,
    const float* mdepth,
    const float* normal_length,
    const uint32_t* n_contrib,
    const uint32_t* max_contributors,
    const float* dL_dpixels,
    const float* dL_dpixel_mdepth,
    const float* dL_dalphas,
    const float* dL_dpixel_normals,
    const float focal_x,
    const float focal_y,
    float3* dL_dmean2D,
    float4* dL_dconic2D,
    float* dL_dcolors,
    float4* dL_dray_planes,
    float3* dL_dnormals,
    bool require_depth) {
#define RENDER_CUDA_CALL(template_depth)                                    \
    renderCUDA<NUM_CHANNELS, template_depth><<<grid, block>>>(              \
        ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors, \
        depths, ray_planes, normals, alphas, normalmap, mdepth,             \
        normal_length, n_contrib, max_contributors, dL_dpixels,             \
        dL_dpixel_mdepth, dL_dalphas, dL_dpixel_normals,                    \
        focal_x, focal_y, dL_dmean2D, dL_dconic2D, dL_dcolors,              \
        dL_dray_planes, dL_dnormals)

    if (require_depth)
        RENDER_CUDA_CALL(true);
    else
        RENDER_CUDA_CALL(false);

#undef RENDER_CUDA_CALL
}