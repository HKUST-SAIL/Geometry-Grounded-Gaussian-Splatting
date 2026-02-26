#include "auxiliary.h"
#include "config.h"
#include "sample_forward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>
namespace cg = cooperative_groups;

// follow code is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields
//  Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessPointsCUDA(
    int P,
    const float* points3D,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    float2* points2D,
    float* ts,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    tiles_touched[idx] = 0;

    // Perform near culling, quit if outside.
    const float3 p_orig = {points3D[3 * idx], points3D[3 * idx + 1], points3D[3 * idx + 2]};
    float3 p_view;
    if (!in_frustum(p_orig, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Transform point by projecting
    float4 p_hom  = transformPoint4x4(p_orig, projmatrix);
    float p_w     = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

    float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};

    // If the point is outside the image, quit.
    if (point_image.x < 0 || point_image.x > W - 1 || point_image.y < 0 || point_image.y > H - 1)
        return;

    // Store some useful helper data for the next steps.
    ts[idx]            = norm3df(p_view.x, p_view.y, p_view.z);
    points2D[idx]      = point_image;
    tiles_touched[idx] = 1;
}

template <int SAMPLES_PRE_ROUND>
__global__ void __launch_bounds__(BLOCK_SIZE)
    evaluateTransmittanceCUDA(
        const uint32_t* __restrict__ tile_ids,
        const uint32_t* __restrict__ tile_offsets,
        const uint2* __restrict__ gaussian_ranges,
        const uint2* __restrict__ point_ranges,
        const uint32_t* __restrict__ gaussian_list,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        const float2* __restrict__ points2D,
        const float* __restrict__ point_ts,
        const float2* __restrict__ gaussians2D,
        const float4* __restrict__ conic_opacity,
        const float4* __restrict__ ray_planes,
        const float3* __restrict__ normals,
        float* __restrict__ out_transmittance,
        bool* __restrict__ inside) {
    auto block                 = cg::this_thread_block();
    const uint32_t block_id    = block.group_index().x;
    const int tile_id          = tile_ids[block_id];
    const uint32_t tile_offset = (tile_id == 0) ? 0 : tile_offsets[tile_id - 1];
    const int p_round          = block_id - tile_offset;
    const uint2 p_range        = point_ranges[tile_id];

    // Load start/end range of IDs to process in bit sorted list.
    const uint2 range = gaussian_ranges[tile_id];
    const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Allocate storage for batches of collectively fetched data.
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float4 collected_ray_planes[BLOCK_SIZE];

    bool done[SAMPLES_PRE_ROUND] = {false};
    float T_point[SAMPLES_PRE_ROUND];
    float T_gaussian[SAMPLES_PRE_ROUND];
    uint32_t point_idx[SAMPLES_PRE_ROUND];
    float2 point_xy[SAMPLES_PRE_ROUND];
    float point_t[SAMPLES_PRE_ROUND];
    int point_done      = 0;
    int point_num_round = 0;
    for (int p = 0; p < SAMPLES_PRE_ROUND; p++) {
        int progress = (p_round * SAMPLES_PRE_ROUND + p) * BLOCK_SIZE + block.thread_rank();
        if (p_range.x + progress < p_range.y) {
            T_point[p]    = 1.f;
            T_gaussian[p] = 1.f;
            int pid       = point_list[p_range.x + progress];
            point_idx[p]  = pid;
            point_xy[p]   = points2D[pid];
            point_t[p]    = point_ts[pid];
            point_num_round++;
        }
    }
    bool all_done = point_num_round == 0;
    int toDo      = range.y - range.x;
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int block_done = __syncthreads_and(all_done);
        if (block_done)
            break;

        block.sync();
        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id                                  = gaussian_list[range.x + progress];
            collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
            const float2 xy        = collected_xy[j];
            const float4 con_o     = collected_conic_opacity[j];
            const float4 ray_plane = collected_ray_planes[j];
            for (int p = 0; p < point_num_round; p++) {
                if (done[p])
                    continue;
                float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                if (power > 0.0f) {
                    continue;
                }

                const float alpha = fminf(0.99f, con_o.w * expf(power));
                if (alpha < 1.0f / 255.0f)
                    continue;
                float test_T = T_gaussian[p] * (1.f - alpha);
                if (test_T < 0.0001f) {
                    done[p] = true;
                    point_done++;
                    continue;
                }
                const float t_peak = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                const float rsigma = ray_plane.w;

                float delta              = (t_peak - point_t[p]) * rsigma;
                float g                  = rsigma > 0 ? expf(-0.5f * delta * delta) : 0.f;
                float one_minus_gaussian = 1.f - alpha * g;
                float rvacancy           = rsqrtf(one_minus_gaussian);
                T_point[p] *= (point_t[p] > t_peak ? (1.f - alpha) : one_minus_gaussian) * rvacancy;
                T_gaussian[p] = test_T;
            }
            all_done = point_done == point_num_round;
        }
    }
    for (int p = 0; p < point_num_round; p++) {
        out_transmittance[point_idx[p]] = T_point[p];
        inside[point_idx[p]]            = true;
    }
}

template <int SAMPLES_PRE_ROUND, uint32_t SPLIT = 8, uint32_t SPLIT_ITERATIONS = 6>
__global__ void __launch_bounds__(BLOCK_SIZE)
    evaluateSDFCUDA(
        const uint32_t* __restrict__ tile_ids,
        const uint32_t* __restrict__ tile_offsets,
        const uint2* __restrict__ gaussian_ranges,
        const uint2* __restrict__ point_ranges,
        const uint32_t* __restrict__ gaussian_list,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        const float2* __restrict__ points2D,
        const float* __restrict__ point_ts,
        const float2* __restrict__ gaussians2D,
        const float4* __restrict__ conic_opacity,
        const float4* __restrict__ ray_planes,
        const float3* __restrict__ normals,
        float* __restrict__ median_depth,
        float* __restrict__ output,
        bool* __restrict__ inside_output) {
    auto block                 = cg::this_thread_block();
    const uint32_t block_id    = block.group_index().x;
    const int tile_id          = tile_ids[block_id];
    const uint32_t tile_offset = (tile_id == 0) ? 0 : tile_offsets[tile_id - 1];
    const int p_round          = block_id - tile_offset;
    const uint2 p_range        = point_ranges[tile_id];

    // Load start/end range of IDs to process in bit sorted list.
    const uint2 range = gaussian_ranges[tile_id];
    const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float4 collected_ray_planes[BLOCK_SIZE];

    bool done[SAMPLES_PRE_ROUND]   = {false};
    float Depth[SAMPLES_PRE_ROUND] = {0.f};
#ifdef MOST_VISIBLE_INIT
    float max_weight[SAMPLES_PRE_ROUND] = {0.f};
#endif
    float T[SAMPLES_PRE_ROUND];
    uint32_t point_idx[SAMPLES_PRE_ROUND];
    float2 point_xy[SAMPLES_PRE_ROUND];
    uint32_t last_contributor[SAMPLES_PRE_ROUND] = {0};
    uint32_t contributor                         = 0;
    int point_done                               = 0;
    int point_num_round                          = 0;
    for (int p = 0; p < SAMPLES_PRE_ROUND; p++) {
        int progress = (p_round * SAMPLES_PRE_ROUND + p) * BLOCK_SIZE + block.thread_rank();
        if (p_range.x + progress < p_range.y) {
            T[p]         = 1.f;
            int pid      = point_list[p_range.x + progress];
            point_idx[p] = pid;
            point_xy[p]  = points2D[pid];
            point_num_round++;
        }
    }
    bool all_done = point_num_round == 0;
    int toDo      = range.y - range.x;
    // first pass to determine last_contributor and initial depth
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int block_done = __syncthreads_and(all_done);
        if (block_done)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id                                  = gaussian_list[range.x + progress];
            collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor++;
            const float2 xy        = collected_xy[j];
            const float4 con_o     = collected_conic_opacity[j];
            const float4 ray_plane = collected_ray_planes[j];
            for (int p = 0; p < point_num_round; p++) {
                if (done[p])
                    continue;
                float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                if (power > 0.0f) {
                    continue;
                }

                float alpha = fminf(0.99f, con_o.w * expf(power));
                if (alpha < 1.0f / 255.0f)
                    continue;
                float test_T = T[p] * (1.f - alpha);
                if (test_T < 0.0001f) {
                    done[p] = true;
                    point_done++;
                    continue;
                }
                float t = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
#if defined(MEDIAN_DEPTH_INIT)
                Depth[p] = T[p] > 0.5f ? t : Depth[p];
#elif defined(MEAN_DEPTH_INIT)
                Depth[p] += alpha * T[p] * t;
#elif defined(MOST_VISIBLE_INIT)
                float aT                = alpha * T[p];
                const bool more_visible = aT > max_weight[p];
                Depth[p]                = more_visible ? t : Depth[p];
                max_weight[p]           = more_visible ? aT : max_weight[p];
#endif
                T[p]                = test_T;
                last_contributor[p] = contributor;
            }
            all_done = point_done == point_num_round;
        }
    }

    using BlockReduce = cub::BlockReduce<uint32_t, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ uint32_t s_block_max;

#if CUDA_VERSION_AT_LEAST_12_9
    const uint32_t block_max =
        BlockReduce(temp_storage).Reduce(last_contributor, cuda::maximum<>{});
#else
    uint32_t thread_max = 0;
    for (int p = 0; p < point_num_round; p++)
        thread_max = max(thread_max, last_contributor[p]);
    auto op = [] __device__(uint32_t a, uint32_t b) { return max(a, b); };
    const uint32_t block_max =
        BlockReduce(temp_storage).Reduce(thread_max, op);
#endif

    if (block.thread_rank() == 0)
        s_block_max = block_max;
    block.sync();

    const uint32_t max_contributor = s_block_max;

    float depth_min[SAMPLES_PRE_ROUND];
    float depth_max[SAMPLES_PRE_ROUND];
    for (int p = 0; p < point_num_round; p++) {
#if defined(MEAN_DEPTH_INIT)
        float mDepthinit = Depth[p] / (1.f - T[p]);
#else
        float mDepthinit = Depth[p];
#endif
        depth_min[p] = fmaxf(mDepthinit - SAMPLE_RANGE_TESTING * 2.f, 0.f);
        depth_max[p] = fmaxf(mDepthinit + SAMPLE_RANGE_TESTING * 2.f, 0.f);
    }
    float T_p[SAMPLES_PRE_ROUND][SPLIT + 1];
    bool inside_range[SAMPLES_PRE_ROUND] = {false};
    for (int p = 0; p < point_num_round; p++) {
        inside_range[p] = T[p] <= MIN_TRANSMITTANCE;
    }

    auto iter = [&] __device__(auto first_const) {
        constexpr bool FIRST = decltype(first_const)::value;
        int toDo             = max_contributor;
        const int rounds     = (max_contributor + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int point_done       = 0;
        uint32_t contributor = 0;
        float interval[SAMPLES_PRE_ROUND];
        constexpr float ONE_OVER_SPLIT = 1.f / static_cast<float>(SPLIT);
        constexpr int START_ID         = FIRST ? 0 : 1;
        constexpr int END_ID           = FIRST ? SPLIT + 1 : SPLIT;
        for (int p = 0; p < point_num_round; p++) {
            done[p] = !(inside_range[p]);
            point_done += static_cast<int>(done[p]);
            interval[p] = (depth_max[p] - depth_min[p]) * ONE_OVER_SPLIT;
#pragma unroll
            for (int s = START_ID; s < END_ID; s++) {
                T_p[p][s] = 1.f;
            }
        }
        all_done = point_done == point_num_round;

        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            block.sync();
            // Collectively fetch per-Gaussian data from global to shared
            int progress = i * BLOCK_SIZE + block.thread_rank();
            if (progress < max_contributor) {
                int coll_id                                  = gaussian_list[range.x + progress];
                collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
                collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
            }
            block.sync();

            // Iterate over current batch
            for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
                contributor++;
                const float2 xy        = collected_xy[j];
                const float4 con_o     = collected_conic_opacity[j];
                const float4 ray_plane = collected_ray_planes[j];
                for (int p = 0; p < point_num_round; p++) {
                    if (done[p])
                        continue;

                    float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    if (power > 0.0f) {
                        continue;
                    }
                    float alpha = fminf(0.99f, con_o.w * expf(power));
                    if (alpha < 1.0f / 255.0f)
                        continue;

                    /********** accumulate transmittance ***********/
                    const float t_peak = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                    const float rsigma = ray_plane.w;
                    const bool ball    = rsigma > 0;
                    for (int s = START_ID; s < END_ID; s++) {
                        float ts                 = depth_min[p] + interval[p] * s;
                        float delta              = (ts - t_peak) * rsigma;
                        float g                  = ball ? expf(-0.5f * delta * delta) : 0.f;
                        float one_minus_gaussian = 1.f - alpha * g;
                        float rvacancy           = rsqrtf(one_minus_gaussian);
                        T_p[p][s] *= (ts > t_peak ? (1.f - alpha) : one_minus_gaussian) * rvacancy;
                    }

                    /***********************************************/
                    done[p] = contributor >= last_contributor[p];
                    point_done += static_cast<int>(contributor == last_contributor[p]);
                }
                all_done = point_done == point_num_round;
            }
        }
        for (int p = 0; p < point_num_round; p++) {
            if constexpr (FIRST) {
                inside_range[p] = (T_p[p][0] >= 0.5f) && (T_p[p][SPLIT] <= 0.5f) && inside_range[p];
            }
            int start_id = 0;
#pragma unroll
            for (int s = 1; s < SPLIT; s++) {
                start_id = T_p[p][s] >= 0.5f ? s : start_id;
            }
            depth_max[p]  = depth_min[p] + (start_id + 1) * interval[p];
            depth_min[p]  = depth_min[p] + (start_id + 0) * interval[p];
            T_p[p][0]     = T_p[p][start_id];
            T_p[p][SPLIT] = T_p[p][start_id + 1];
        }
    };

    iter(std::true_type{});
    for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)
        iter(std::false_type{});

    for (int p = 0; p < point_num_round; p++) {
        float w_max                 = __saturatef((T_p[p][0] - 0.5f) / (T_p[p][0] - T_p[p][SPLIT]));
        float w_min                 = 1.f - w_max;
        float mDepth                = inside_range[p] ? w_max * depth_max[p] + w_min * depth_min[p] : 0.f;
        output[point_idx[p]]        = mDepth - point_ts[point_idx[p]];
        median_depth[point_idx[p]]  = mDepth;
        inside_output[point_idx[p]] = inside_range[p];
    }
}

template <uint32_t CHANNELS, int SAMPLES_PRE_ROUND>
__global__ void __launch_bounds__(BLOCK_SIZE)
    evaluateColorCUDA(
        const uint32_t* __restrict__ tile_ids,
        const uint32_t* __restrict__ tile_offsets,
        const uint2* __restrict__ gaussian_ranges,
        const uint2* __restrict__ point_ranges,
        const uint32_t* __restrict__ gaussian_list,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        const float2* __restrict__ points2D,
        const float* __restrict__ point_ts,
        const float2* __restrict__ gaussians2D,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ colors,
        const float* bg_color,
        float* __restrict__ out_color,
        bool* __restrict__ inside) {
    auto block                 = cg::this_thread_block();
    const uint32_t block_id    = block.group_index().x;
    const int tile_id          = tile_ids[block_id];
    const uint32_t tile_offset = (tile_id == 0) ? 0 : tile_offsets[tile_id - 1];
    const int p_round          = block_id - tile_offset;
    const uint2 p_range        = point_ranges[tile_id];

    // Load start/end range of IDs to process in bit sorted list.
    const uint2 range = gaussian_ranges[tile_id];
    const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Allocate storage for batches of collectively fetched data.
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float collected_color[BLOCK_SIZE * CHANNELS];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    bool done[SAMPLES_PRE_ROUND] = {false};
    float C_point[SAMPLES_PRE_ROUND][CHANNELS];
    float T_gaussian[SAMPLES_PRE_ROUND];
    uint32_t point_idx[SAMPLES_PRE_ROUND];
    float2 point_xy[SAMPLES_PRE_ROUND];
    float point_t[SAMPLES_PRE_ROUND];
    int point_done      = 0;
    int point_num_round = 0;
    for (int p = 0; p < SAMPLES_PRE_ROUND; p++) {
        int progress = (p_round * SAMPLES_PRE_ROUND + p) * BLOCK_SIZE + block.thread_rank();
        if (p_range.x + progress < p_range.y) {
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ch++)
                C_point[p][ch] = 0.f;
            T_gaussian[p] = 1.f;
            int pid       = point_list[p_range.x + progress];
            point_idx[p]  = pid;
            point_xy[p]   = points2D[pid];
            point_t[p]    = point_ts[pid];
            point_num_round++;
        }
    }
    bool all_done = point_num_round == 0;
    int toDo      = range.y - range.x;
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int block_done = __syncthreads_and(all_done);
        if (block_done)
            break;

        block.sync();
        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id                                  = gaussian_list[range.x + progress];
            collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            for (int ch = 0; ch < CHANNELS; ch++)
                collected_color[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * CHANNELS + ch];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
            const float2 xy    = collected_xy[j];
            const float4 con_o = collected_conic_opacity[j];
            for (int p = 0; p < point_num_round; p++) {
                if (done[p])
                    continue;
                float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                if (power > 0.0f) {
                    continue;
                }

                const float alpha = fminf(0.99f, con_o.w * expf(power));
                if (alpha < 1.0f / 255.0f)
                    continue;
                float test_T = T_gaussian[p] * (1.f - alpha);
                if (test_T < 0.0001f) {
                    done[p] = true;
                    point_done++;
                    continue;
                }
                float aT = T_gaussian[p] * alpha;
#pragma unroll
                for (int ch = 0; ch < CHANNELS; ch++)
                    C_point[p][ch] += collected_color[j + BLOCK_SIZE * ch] * aT;
                T_gaussian[p] = test_T;
            }
            all_done = point_done == point_num_round;
        }
    }
    for (int p = 0; p < point_num_round; p++) {
#pragma unroll
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[point_idx[p] * CHANNELS + ch] = C_point[p][ch] + T_gaussian[p] * bg_color[ch];
        inside[point_idx[p]] = true;
    }
}

template <int SAMPLES_PRE_ROUND, uint32_t SPLIT = 8, uint32_t SPLIT_ITERATIONS = 5>
__global__ void __launch_bounds__(BLOCK_SIZE)
    sampleDepthCUDA(
        const uint32_t* __restrict__ tile_ids,
        const uint32_t* __restrict__ tile_offsets,
        const uint2* __restrict__ gaussian_ranges,
        const uint2* __restrict__ point_ranges,
        const uint32_t* __restrict__ gaussian_list,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        const float2* __restrict__ points2D,
        const float2* __restrict__ gaussians2D,
        const float4* __restrict__ conic_opacity,
        const float4* __restrict__ ray_planes,
        const float3* __restrict__ normals,
        uint32_t* __restrict__ n_contrib,
        float* __restrict__ median_depth,
        uint32_t* __restrict__ out_max_contributor,
        float3* __restrict__ output,
        bool* __restrict__ inside_output) {
    auto block                 = cg::this_thread_block();
    const uint32_t block_id    = block.group_index().x;
    const int tile_id          = tile_ids[block_id];
    const uint32_t tile_offset = (tile_id == 0) ? 0 : tile_offsets[tile_id - 1];
    const int p_round          = block_id - tile_offset;
    const uint2 p_range        = point_ranges[tile_id];

    // Load start/end range of IDs to process in bit sorted list.
    const uint2 range = gaussian_ranges[tile_id];
    const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float4 collected_ray_planes[BLOCK_SIZE];

    bool done[SAMPLES_PRE_ROUND]   = {false};
    float Depth[SAMPLES_PRE_ROUND] = {0.f};
#ifdef MOST_VISIBLE_INIT
    float max_weight[SAMPLES_PRE_ROUND] = {0.f};
#endif
    float T[SAMPLES_PRE_ROUND];
    uint32_t point_idx[SAMPLES_PRE_ROUND];
    float2 point_xy[SAMPLES_PRE_ROUND];
    uint32_t last_contributor[SAMPLES_PRE_ROUND] = {0};
    uint32_t contributor                         = 0;
    int point_done                               = 0;
    int point_num_round                          = 0;
    for (int p = 0; p < SAMPLES_PRE_ROUND; p++) {
        int progress = (p_round * SAMPLES_PRE_ROUND + p) * BLOCK_SIZE + block.thread_rank();
        if (p_range.x + progress < p_range.y) {
            T[p]         = 1.f;
            int pid      = point_list[p_range.x + progress];
            point_idx[p] = pid;
            point_xy[p]  = points2D[pid];
            point_num_round++;
        }
    }
    bool all_done = point_num_round == 0;
    int toDo      = range.y - range.x;
    // first pass to determine last_contributor and initial depth
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int block_done = __syncthreads_and(all_done);
        if (block_done)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id                                  = gaussian_list[range.x + progress];
            collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor++;
            const float2 xy        = collected_xy[j];
            const float4 con_o     = collected_conic_opacity[j];
            const float4 ray_plane = collected_ray_planes[j];
            for (int p = 0; p < point_num_round; p++) {
                if (done[p])
                    continue;
                float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
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
                float test_T = T[p] * (1.f - alpha);
                if (test_T < 0.0001f) {
                    done[p] = true;
                    point_done++;
                    continue;
                }
                float t = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
#if defined(MEDIAN_DEPTH_INIT)
                Depth[p] = T[p] > 0.5f ? t : Depth[p];
#elif defined(MEAN_DEPTH_INIT)
                Depth[p] += alpha * T[p] * t;
#elif defined(MOST_VISIBLE_INIT)
                float aT                = alpha * T[p];
                const bool more_visible = aT > max_weight[p];
                Depth[p]                = more_visible ? t : Depth[p];
                max_weight[p]           = more_visible ? aT : max_weight[p];
#endif
                T[p]                = test_T;
                last_contributor[p] = contributor;
            }
            all_done = point_done == point_num_round;
        }
    }

    using BlockReduce = cub::BlockReduce<uint32_t, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ uint32_t s_block_max;

#if CUDA_VERSION_AT_LEAST_12_9
    const uint32_t block_max =
        BlockReduce(temp_storage).Reduce(last_contributor, cuda::maximum<>{});
#else
    uint32_t thread_max = 0;
    for (int p = 0; p < point_num_round; p++)
        thread_max = max(thread_max, last_contributor[p]);
    auto op = [] __device__(uint32_t a, uint32_t b) { return max(a, b); };
    const uint32_t block_max =
        BlockReduce(temp_storage).Reduce(thread_max, op);
#endif

    if (block.thread_rank() == 0)
        s_block_max = block_max;
    block.sync();

    const uint32_t max_contributor = s_block_max;

    float depth_min[SAMPLES_PRE_ROUND];
    float depth_max[SAMPLES_PRE_ROUND];
    for (int p = 0; p < point_num_round; p++) {
#if defined(MEAN_DEPTH_INIT)
        float mDepthinit = Depth[p] / (1.f - T[p]);
#else
        float mDepthinit = Depth[p];
#endif
        depth_min[p] = fmaxf(mDepthinit - SAMPLE_RANGE, 0.f);
        depth_max[p] = fmaxf(mDepthinit + SAMPLE_RANGE, 0.f);
    }
    float T_p[SAMPLES_PRE_ROUND][SPLIT + 1];
    bool inside_range[SAMPLES_PRE_ROUND] = {false};
    for (int p = 0; p < point_num_round; p++) {
        inside_range[p] = T[p] <= MIN_TRANSMITTANCE;
    }

    auto iter = [&] __device__(auto first_const) {
        constexpr bool FIRST = decltype(first_const)::value;
        int toDo             = max_contributor;
        const int rounds     = (max_contributor + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int point_done       = 0;
        uint32_t contributor = 0;
        float interval[SAMPLES_PRE_ROUND];
        constexpr float ONE_OVER_SPLIT = 1.f / static_cast<float>(SPLIT);
        constexpr int START_ID         = FIRST ? 0 : 1;
        constexpr int END_ID           = FIRST ? SPLIT + 1 : SPLIT;
        for (int p = 0; p < point_num_round; p++) {
            done[p] = !(inside_range[p]);
            point_done += static_cast<int>(done[p]);
            interval[p] = (depth_max[p] - depth_min[p]) * ONE_OVER_SPLIT;
#pragma unroll
            for (int s = START_ID; s < END_ID; s++) {
                T_p[p][s] = 1.f;
            }
        }
        all_done = point_done == point_num_round;

        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            // End if entire block votes that it is done rasterizing
            block.sync();
            // Collectively fetch per-Gaussian data from global to shared
            int progress = i * BLOCK_SIZE + block.thread_rank();
            if (progress < max_contributor) {
                int coll_id                                  = gaussian_list[range.x + progress];
                collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
                collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
            }
            block.sync();

            // Iterate over current batch
            for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
                contributor++;
                const float2 xy        = collected_xy[j];
                const float4 con_o     = collected_conic_opacity[j];
                const float4 ray_plane = collected_ray_planes[j];
                for (int p = 0; p < point_num_round; p++) {
                    if (done[p])
                        continue;

                    float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    if (power > 0.0f) {
                        continue;
                    }
                    float alpha = fminf(0.99f, con_o.w * expf(power));
                    if (alpha < 1.0f / 255.0f)
                        continue;

                    /********** accumulate transmittance ***********/
                    const float t_peak = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                    const float rsigma = ray_plane.w;
                    const bool ball    = rsigma > 0;
                    for (int s = START_ID; s < END_ID; s++) {
                        float ts                 = depth_min[p] + interval[p] * s;
                        float delta              = (ts - t_peak) * rsigma;
                        float g                  = ball ? expf(-0.5f * delta * delta) : 0.f;
                        float one_minus_gaussian = 1.f - alpha * g;
                        float rvacancy           = rsqrtf(one_minus_gaussian);
                        T_p[p][s] *= (ts > t_peak ? (1.f - alpha) : one_minus_gaussian) * rvacancy;
                    }

                    /***********************************************/
                    done[p] = contributor >= last_contributor[p];
                    point_done += static_cast<int>(contributor == last_contributor[p]);
                }
                all_done = point_done == point_num_round;
            }
        }
        for (int p = 0; p < point_num_round; p++) {
            if constexpr (FIRST) {
                inside_range[p] = (T_p[p][0] >= 0.5f) && (T_p[p][SPLIT] <= 0.5f) && inside_range[p];
            }
            int start_id = 0;
#pragma unroll
            for (int s = 1; s < SPLIT; s++) {
                start_id = T_p[p][s] >= 0.5f ? s : start_id;
            }
            depth_max[p]  = depth_min[p] + (start_id + 1) * interval[p];
            depth_min[p]  = depth_min[p] + (start_id + 0) * interval[p];
            T_p[p][0]     = T_p[p][start_id];
            T_p[p][SPLIT] = T_p[p][start_id + 1];
        }
    };

    iter(std::true_type{});
    for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)
        iter(std::false_type{});

    for (int p = 0; p < point_num_round; p++) {
        float w_max                 = __saturatef((T_p[p][0] - 0.5f) / (T_p[p][0] - T_p[p][SPLIT]));
        float w_min                 = 1.f - w_max;
        float mDepth                = inside_range[p] ? w_max * depth_max[p] + w_min * depth_min[p] : 0.f;
        float2 pixnf                = {(point_xy[p].x - static_cast<float>(W - 1) / 2.f) / focal_x,
                                       (point_xy[p].y - static_cast<float>(H - 1) / 2.f) / focal_y};
        const float rln             = rnorm3df(pixnf.x, pixnf.y, 1.f);
        float depth                 = mDepth * rln;
        output[point_idx[p]]        = {pixnf.x * depth, pixnf.y * depth, depth};
        median_depth[point_idx[p]]  = mDepth;
        n_contrib[point_idx[p]]     = last_contributor[p];
        inside_output[point_idx[p]] = inside_range[p];
    }
    if (block.thread_rank() == 0)
        out_max_contributor[block_id] = block_max;
}

void FORWARD::preprocess_points(
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
    bool prefiltered) {
    preprocessPointsCUDA<<<(PN + 255) / 256, 256>>>(
        PN,
        points3D,
        viewmatrix,
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        points2D,
        ts,
        grid,
        tiles_touched,
        prefiltered);
}

void FORWARD::evaluateTransmittance(
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
    bool* inside) {
    if (num_duplicated_tiles == 0)
        return;
    evaluateTransmittanceCUDA<SAMPLE_BATCH_SIZE><<<num_duplicated_tiles, BLOCK_SIZE>>>(
        tile_ids,
        tile_offsets,
        gaussian_ranges,
        point_ranges,
        gaussian_list,
        point_list,
        W, H,
        focal_x, focal_y,
        points2D,
        point_depths,
        gaussians2D,
        conic_opacity,
        ray_planes,
        normals,
        out_transmittance,
        inside);
}

void FORWARD::evaluateSDF(
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
    bool* inside_output) {
    if (num_duplicated_tiles == 0)
        return;
    evaluateSDFCUDA<SAMPLE_BATCH_SIZE, SPLIT, SPLIT_ITERATIONS_TESTING><<<num_duplicated_tiles, BLOCK_SIZE>>>(
        tile_ids,
        tile_offsets,
        gaussian_ranges,
        point_ranges,
        gaussian_list,
        point_list,
        W, H,
        focal_x, focal_y,
        points2D,
        point_ts,
        gaussians2D,
        conic_opacity,
        ray_planes,
        normals,
        median_depth,
        output,
        inside_output);
}

void FORWARD::evaluateColor(
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
    bool* inside) {
    if (num_duplicated_tiles == 0)
        return;
    evaluateColorCUDA<NUM_CHANNELS, SAMPLE_BATCH_SIZE><<<num_duplicated_tiles, BLOCK_SIZE>>>(
        tile_ids,
        tile_offsets,
        gaussian_ranges,
        point_ranges,
        gaussian_list,
        point_list,
        W, H,
        focal_x, focal_y,
        points2D,
        point_depths,
        gaussians2D,
        conic_opacity,
        colors,
        bg_color,
        out_color,
        inside);
}

void FORWARD::sampleDepth(
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
    bool* inside) {
    if (num_duplicated_tiles == 0)
        return;
    sampleDepthCUDA<SAMPLE_BATCH_SIZE, SPLIT, SPLIT_ITERATIONS><<<num_duplicated_tiles, BLOCK_SIZE>>>(
        tile_ids,
        tile_offsets,
        gaussian_ranges,
        point_ranges,
        gaussian_list,
        point_list,
        W, H,
        focal_x, focal_y,
        points2D,
        gaussians2D,
        conic_opacity,
        ray_planes,
        normals,
        n_contrib,
        median_depth,
        max_contributor,
        output,
        inside);
}