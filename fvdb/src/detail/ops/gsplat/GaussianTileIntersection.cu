// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianSplatSparse.h"
#include "VectorTypes.cuh"
#include <detail/ops/Ops.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>

namespace {

#define NUM_THREADS 1024

#define CUB_WRAPPER(func, ...)                                                    \
    do {                                                                          \
        size_t temp_storage_bytes = 0;                                            \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                           \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();      \
        auto  temp_storage      = caching_allocator.allocate(temp_storage_bytes); \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);                \
    } while (false)

// Compute the number of 2d image tiles intersected by a set of 2D projected Gaussians.
//
// The input is a set of 2D circles with depths approximating the projection of 3D gaussians onto
// the image plane. Each input circle is encoded as a tuple:
// (mean_u, mean_v, radius, depth)
// where (mean_u, mean_v) are the image-space center of the circle, radius is its radius (in pixels)
// and depth is the (world-space) depth of the Gaussian this circle is approximating.
//
// The output is a set of counts of the number of tiles each Gaussian intersects.
//
template <typename T, typename CountT>
__global__ void
count_tiles_per_gaussian(const uint32_t total_gaussians, const uint32_t num_gaussians_per_camera,
                         const uint32_t tile_size, const uint32_t num_tiles_w,
                         const uint32_t num_tiles_h,
                         const T *__restrict__ means2d,                     // [C, N, 2] or [M, 2]
                         const int32_t *__restrict__ radii,                 // [C, N]    or [M]
                         const bool *__restrict__ tile_mask,                // [C, H, W] or nullptr
                         const int32_t *__restrict__ camera_jidx,           // NULL or [M]
                         CountT *__restrict__ out_num_tiles_per_gaussian) { // [ C * N ] or [ M ]
    // parallelize over total_gaussians
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_gaussians;
         idx += blockDim.x * gridDim.x) {
        // For now we'll upcast float16 and bfloat16 to float32
        using OpT = typename OpType<T>::type;

        const OpT radius = radii[idx];
        if (radius <= 0) {
            out_num_tiles_per_gaussian[idx] = static_cast<CountT>(0);
            return;
        }

        using vec2f = typename Vec2Type<OpT>::type;

        const vec2f mean2d      = *reinterpret_cast<const vec2f *>(means2d + idx * 2);
        const OpT   tile_radius = radius / static_cast<OpT>(tile_size);
        const OpT   tile_mean_u = mean2d.x / static_cast<OpT>(tile_size);
        const OpT   tile_mean_v = mean2d.y / static_cast<OpT>(tile_size);

        // tile_min is inclusive, tile_max is exclusive
        uint2 tile_min, tile_max;
        tile_min.x = min(max(0, (uint32_t)floor(tile_mean_u - tile_radius)), num_tiles_w);
        tile_min.y = min(max(0, (uint32_t)floor(tile_mean_v - tile_radius)), num_tiles_h);
        tile_max.x = min(max(0, (uint32_t)ceil(tile_mean_u + tile_radius)), num_tiles_w);
        tile_max.y = min(max(0, (uint32_t)ceil(tile_mean_v + tile_radius)), num_tiles_h);

        out_num_tiles_per_gaussian[idx] = [&]() {
            if (tile_mask) {
                CountT        num_tiles = 0;
                const int32_t cidx      = (camera_jidx == nullptr)
                                              ? static_cast<int32_t>(idx / num_gaussians_per_camera)
                                              : camera_jidx[idx];
                // loop min / max range and count number of tiles
                for (uint32_t i = tile_min.y; i < tile_max.y; ++i) {
                    for (uint32_t j = tile_min.x; j < tile_max.x; ++j) {
                        if (tile_mask[cidx * num_tiles_h * num_tiles_w + i * num_tiles_w + j]) {
                            num_tiles++;
                        }
                    }
                }
                return num_tiles;
            } else {
                // write out number of tiles per gaussian
                const CountT num_tiles =
                    static_cast<CountT>((tile_max.y - tile_min.y) * (tile_max.x - tile_min.x));
                return num_tiles;
            }
        }();
    }
}

// Encode a float depth into a int64_t
__device__ inline int64_t
encode_depth(float depth) {
    int32_t ret;
    std::memcpy(&ret, &depth, sizeof(depth));
    return static_cast<int64_t>(ret);
}

// Encode a camera index, tile index, and depth into a int64_t
// The layout is (from least to most significant bits):
// [32 bits] depth
// [tile_id_bits] tile_idx
// [camera_id_bits] cidx
__device__ inline int64_t
encode_cam_tile_depth_key(int64_t cidx, int64_t tile_idx, int64_t encoded_depth,
                          int32_t tile_id_bits) {
    int64_t cidx_enc = cidx << tile_id_bits;
    return ((cidx_enc | tile_idx) << 32) | encoded_depth;
}

// Decode a camera index and tile index from a int64_t
// The layout is (from least to most significant bits):
// [32 bits] depth (ignored and dropped)
// [tile_id_bits] tile_idx
// [camera_id_bits] cidx
__device__ inline std::tuple<int32_t, int32_t>
decode_cam_tile_key(int64_t packed_cam_tile_depth_key, int32_t tile_id_bits) {
    int32_t tile_key = packed_cam_tile_depth_key >> 32;
    int32_t cidx_enc = tile_key >> tile_id_bits;
    int32_t tile_idx = tile_key & ((1 << tile_id_bits) - 1);
    return { cidx_enc, tile_idx };
}

// Compute a set of intersections between the 2D projected Gaussians and each tile to be
// rendered on screen.
//
// The input is a set of 2D circles with depths approximating the projection of 3D gaussians
// onto the image plane. Each input circle is encoded as a tuple: (mean_u, mean_v, radius,
// depth) where (mean_u, mean_v) are the image-space center of the circle, radius is its radius
// (in pixels) and depth is the (world-space) depth of the Gaussian this circle is
// approximating.
//
// The output is a set of gaussian/tile intersections where each intersection is parameterized
// as: a tuple key (camera_id, tile_id, depth) gaussian_id value indexing into
// means2d/radii/depths
//
// The key (camera_id, tile_id, depth) is packed into 64 bits and identifies which camera, and
// tile this intersection corresponds to, and the depth of the Gaussian at this intersection.
// (we'll use this to sort the intersections into tiles and by depth).
// The value is the index of the Gaussian in the input arrays.
//
template <typename T>
__global__ void
compute_gaussian_tile_intersections(
    const uint32_t num_cameras, const uint32_t num_gaussians_per_camera,
    const uint32_t total_gaussians, const uint32_t tile_size, const uint32_t num_tiles_w,
    const uint32_t num_tiles_h, const uint32_t tile_id_bits,
    const T *__restrict__ means2d,                      // [C, N, 2] or [M, 2]
    const int32_t *__restrict__ radii,                  // [C, N]    or [M]
    const T *__restrict__ depths,                       // [C, N]    or [M]
    const int32_t *__restrict__ cum_tiles_per_gaussian, // [ C * N ] or [ M ]
    const bool *__restrict__ tile_mask,                 // [C, H, W] or nullptr
    const int32_t *__restrict__ camera_jidx,            // NULL or [M]
    int64_t *__restrict__ intersection_keys,     // [ C * N * num_tiles ] or [ M * num_tiles ]
    int32_t *__restrict__ intersection_values) { // [ C * N * num_tiles ] or [ M * num_tiles ]

    // parallelize over total_gaussians
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_gaussians;
         idx += blockDim.x * gridDim.x) {
        // For now we'll upcast float16 and bfloat16 to float32
        using OpT = typename OpType<T>::type;

        // Get the camera id from the batch indices or use the camera index directly
        const int32_t cidx =
            camera_jidx == nullptr ? idx / num_gaussians_per_camera : camera_jidx[idx];

        const OpT radius = radii[idx];
        if (radius <= 0) {
            return;
        }

        using vec2f = typename Vec2Type<OpT>::type;

        const vec2f mean2d      = *reinterpret_cast<const vec2f *>(means2d + 2 * idx);
        const OpT   tile_radius = radius / static_cast<OpT>(tile_size);
        const OpT   tile_mean_u = mean2d.x / static_cast<OpT>(tile_size);
        const OpT   tile_mean_v = mean2d.y / static_cast<OpT>(tile_size);

        // tile_min is inclusive, tile_max is exclusive
        uint2 tile_min, tile_max;
        tile_min.x = min(max(0, (uint32_t)floor(tile_mean_u - tile_radius)), num_tiles_w);
        tile_min.y = min(max(0, (uint32_t)floor(tile_mean_v - tile_radius)), num_tiles_h);
        tile_max.x = min(max(0, (uint32_t)ceil(tile_mean_u + tile_radius)), num_tiles_w);
        tile_max.y = min(max(0, (uint32_t)ceil(tile_mean_v + tile_radius)), num_tiles_h);

        // If you use float64, we're casting you to float32 so we can
        // pack the depth into the key. In principle this loses precision,
        // in practice it's fine.
        const float depth = depths[idx];

        // Suppose you're using tile_id_bits = 22, then the output for this intersection is
        // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
        // which we pack into an int64_t
        const int64_t depth_enc = encode_depth(depth);

        // For each tile this Gaussian intersects, write out an intersection tuple
        // (camera_id, tile_id, depth, gaussian_id) packed as a uint3
        int64_t cur_isect = (idx == 0) ? 0 : cum_tiles_per_gaussian[idx - 1];
        for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
            for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                // Skip if tile is masked out
                if (tile_mask &&
                    !tile_mask[cidx * num_tiles_h * num_tiles_w + i * num_tiles_w + j]) {
                    continue;
                }
                const int64_t tile_idx = (i * num_tiles_w + j); // Needs to fit in tile_id_bits bits
                const int64_t packed_cam_idx_and_tile_idx =
                    encode_cam_tile_depth_key(cidx, tile_idx, depth_enc, tile_id_bits);
                intersection_keys[cur_isect]   = packed_cam_idx_and_tile_idx;
                intersection_values[cur_isect] = idx;
                cur_isect += 1;
            }
        }
    }
}

__global__ void
compute_tile_offsets_sparse(
    const uint32_t num_intersections, const uint32_t num_tiles, const uint32_t tile_id_bits,
    const int64_t *__restrict__ sorted_intersection_keys, const uint32_t *__restrict__ active_tiles,
    const uint32_t num_active_tiles,
    int32_t *__restrict__ out_offsets) { // [C, n_tiles] or [num_active_tiles]

    // parallelize over active tiles
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx <= num_active_tiles;
         idx += blockDim.x * gridDim.x) {
        if (idx == num_active_tiles) {
            out_offsets[idx] = num_intersections;
            return;
        }

        // get the first intersection for this tile
        const int64_t tile_start_idx_dense = active_tiles[idx];
        const int64_t cam_idx              = tile_start_idx_dense / num_tiles;
        const int64_t tile_idx             = tile_start_idx_dense % num_tiles;

        auto depth_enc = encode_depth(0.0f); // Don't care what depth

        const int64_t packed_cam_idx_and_tile_idx =
            encode_cam_tile_depth_key(cam_idx, tile_idx, depth_enc, tile_id_bits);

        // search in sorted_intersection_keys for the start of the range matching
        // packed_cam_idx_and_tile_idx

        auto compare_keys_ignore_depth = [tile_id_bits](int64_t lhs, int64_t rhs) {
            auto [lhs_cam_idx, lhs_tile_idx] = decode_cam_tile_key(lhs, tile_id_bits);
            auto [rhs_cam_idx, rhs_tile_idx] = decode_cam_tile_key(rhs, tile_id_bits);
            return (lhs_cam_idx < rhs_cam_idx) ||
                   ((lhs_cam_idx == rhs_cam_idx) && (lhs_tile_idx < rhs_tile_idx));
        };

        auto const tile_start =
            thrust::lower_bound(thrust::seq, sorted_intersection_keys,
                                sorted_intersection_keys + num_intersections,
                                packed_cam_idx_and_tile_idx, compare_keys_ignore_depth) -
            sorted_intersection_keys;

        out_offsets[idx] = tile_start;
    }
}

// Given a set of intersections between 2D projected Gaussians and image tiles sorted by
// tile_id, camera_id, and depth, compute the a range of Gaussians that intersect each tile
// encoded as an offset into the sorted intersection array.
// i.e. gaussians[out_offsets[c, i, j]:out_offsets[c, i, j+1]] are the Gaussians that
// intersect tile (i, j) in camera c.
__global__ void
compute_tile_offsets(const uint32_t num_intersections, const uint32_t num_cameras,
                     const uint32_t num_tiles, const uint32_t tile_id_bits,
                     const int64_t *__restrict__ sorted_intersection_keys,
                     int32_t *__restrict__ out_offsets) { // [C, n_tiles]
    // sorted_intersection_keys is [(cidx_0 | tidx_0 | depth_0), ..., (cidx_N | tidx_N |
    // depth_N)] where cidx_i = camera index, tidx_i = tile index, depth_i = depth of the
    // gaussian at the intersection, lexographically sorted.
    //
    // The output is a set of offsets into the sorted_intersection array, such that
    // if offset_cij = offsets[cid][ti][tj], and offset_cij+1 = offsets[cid][ti][tj+1],
    // then sorted_intersections[offset_cij:offset_cij+1] are the intersections for the
    // tile (cid, ti, tj) sorted by depth.

    // Parallelize over intersections
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_intersections;
         idx += blockDim.x * gridDim.x) {
        // Bit-packed key for the camera/tile part of the this intersection
        // i.e. tile_id | camera_id << tile_id_bits
        const int64_t tile_key   = sorted_intersection_keys[idx] >> 32;
        auto [cam_idx, tile_idx] = decode_cam_tile_key(sorted_intersection_keys[idx], tile_id_bits);

        // The first intersection for this camera/tile pair
        const int64_t tile_start_idx = cam_idx * num_tiles + tile_idx;

        if (idx == 0) {
            // The first tile in the first camera writes out 0 as the offset
            // until the first valid tile (inclusive). i.e. tiles before this one
            // have no intersections, so their offset range is [0, 0]
            for (uint32_t i = 0; i < tile_start_idx + 1; ++i) {
                out_offsets[i] = 0;
            }
        }
        if (idx == num_intersections - 1) {
            for (uint32_t i = tile_start_idx + 1; i < num_cameras * num_tiles; ++i) {
                out_offsets[i] = static_cast<int32_t>(num_intersections);
            }
        }

        if (idx > 0) {
            const int64_t prev_tile_key = sorted_intersection_keys[idx - 1] >> 32;

            if (prev_tile_key == tile_key) {
                return;
            }

            auto [prev_cam_idx, prev_tile_idx] =
                decode_cam_tile_key(sorted_intersection_keys[idx - 1], tile_id_bits);
            const int64_t prev_tile_start_idx = prev_cam_idx * num_tiles + prev_tile_idx;

            for (uint32_t i = prev_tile_start_idx + 1; i < tile_start_idx + 1; ++i) {
                out_offsets[i] = static_cast<int32_t>(idx);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor>
gaussianTileIntersectionCUDAImpl(
    const torch::Tensor               &means2d,      // [C, N, 2] or [M, 2]
    const torch::Tensor               &radii,        // [C, N] or [M]
    const torch::Tensor               &depths,       // [C, N] or [M]
    const at::optional<torch::Tensor> &camera_jidx,  // NULL or [M]
    const at::optional<torch::Tensor> &tile_mask,    // NULL or [C, H, W]
    const at::optional<torch::Tensor> &active_tiles, // NULL or [num_active_tiles]
    const uint32_t num_cameras, const uint32_t tile_size, const uint32_t num_tiles_h,
    const uint32_t num_tiles_w) {
    const bool is_packed = camera_jidx.has_value();
    const bool is_sparse = active_tiles.has_value();

    TORCH_CHECK(means2d.is_cuda(), "means2d must be a CUDA tensor");
    TORCH_CHECK(radii.is_cuda(), "radii must be a CUDA tensor");
    TORCH_CHECK(depths.is_cuda(), "depths must be a CUDA tensor");

    if (is_packed) {
        TORCH_CHECK(camera_jidx.value().is_cuda(), "camera_jidx must be a CUDA tensor");
        TORCH_CHECK_VALUE(means2d.dim() == 2, "means2d must have 2 dimensions (M, 2)");
        TORCH_CHECK_VALUE(radii.dim() == 1, "radii must have 1 dimension (M)");
        TORCH_CHECK_VALUE(depths.dim() == 1, "depths must have 1 dimension (M)");
        TORCH_CHECK_VALUE(camera_jidx.value().dim() == 1, "camera_jidx must have 1 dimension (M)");
        TORCH_CHECK_VALUE(radii.size(0) == means2d.size(0),
                          "radii must have the same number of points as means2d");
        TORCH_CHECK_VALUE(depths.size(0) == means2d.size(0),
                          "depths must have the same number of points as means2d");
        TORCH_CHECK_VALUE(camera_jidx.value().size(0) == means2d.size(0),
                          "camera_jidx must have the same number of points as means2d");
    } else {
        TORCH_CHECK_VALUE(means2d.dim() == 3, "means2d must have 3 dimensions (C, N, 2)");
        TORCH_CHECK_VALUE(means2d.size(0) == num_cameras,
                          "means2d must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(means2d.size(2) == 2, "means2d must have 2 points in the last dimension");
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have 2 dimensions (C, N)");
        TORCH_CHECK_VALUE(radii.size(0) == num_cameras,
                          "radii must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(radii.size(1) == means2d.size(1),
                          "radii must have the same number of points as means2d");
        TORCH_CHECK_VALUE(depths.dim() == 2, "depths must have 2 dimensions (C, N)");
        TORCH_CHECK_VALUE(depths.size(0) == num_cameras,
                          "depths must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(depths.size(1) == means2d.size(1),
                          "depths must have the same number of points as means2d");
    }

    if (is_sparse) {
        TORCH_CHECK(tile_mask.value().is_cuda(), "tile_mask must be a CUDA tensor");
        TORCH_CHECK_VALUE(tile_mask.value().dim() == 3,
                          "tile_mask must have 3 dimensions (C, H, W)");
        TORCH_CHECK_VALUE(tile_mask.value().size(0) == num_cameras,
                          "tile_mask must have num_cameras in the first dimension");
        TORCH_CHECK_VALUE(tile_mask.value().size(1) == num_tiles_h,
                          "tile_mask must have num_tiles_h in the second dimension");
        TORCH_CHECK_VALUE(tile_mask.value().size(2) == num_tiles_w,
                          "tile_mask must have num_tiles_w in the third dimension");
        TORCH_CHECK(active_tiles.value().is_cuda(), "active_tiles must be a CUDA tensor");
        TORCH_CHECK_VALUE(active_tiles.value().dim() == 1,
                          "active_tiles must have 1 dimension (num_active_tiles)");
    }

    const uint32_t num_gaussians   = is_packed ? means2d.size(0) : means2d.size(1);
    const uint32_t total_gaussians = is_packed ? means2d.size(0) : num_cameras * num_gaussians;

    // const uint32_t num_cameras      = means2d.size(0);
    const uint32_t total_tiles      = num_tiles_h * num_tiles_w;
    const uint32_t num_tile_id_bits = (uint32_t)floor(log2(total_tiles)) + 1;
    const uint32_t num_cam_id_bits  = (uint32_t)floor(log2(num_cameras)) + 1;
    const auto     camera_jidx_ptr =
        camera_jidx.has_value() ? camera_jidx.value().data_ptr<int32_t>() : nullptr;

    const auto device_guard = at::cuda::OptionalCUDAGuard(at::device_of(means2d));
    const auto stream       = at::cuda::getCurrentCUDAStream(means2d.device().index());

    const uint32_t num_active_tiles =
        is_sparse ? active_tiles.value().size(0) : num_cameras * total_tiles;
    const uint32_t output_num_tiles = num_active_tiles + 1;

    auto output_dims = is_sparse ? at::IntArrayRef({ output_num_tiles })
                                 : at::IntArrayRef({ num_cameras, num_tiles_h, num_tiles_w });

    if (total_gaussians == 0) {
        return std::make_tuple(torch::zeros(output_dims, means2d.options().dtype(torch::kInt32)),
                               torch::empty({ 0 }, means2d.options().dtype(torch::kInt32)));
    }
    using scalar_t = float;

    // Allocate tensor to store the number of tiles each gaussian intersects
    torch::Tensor tiles_per_gaussian_cumsum =
        torch::empty({ total_gaussians }, means2d.options().dtype(torch::kInt32));

    const auto tile_mask_ptr = tile_mask.has_value() ? tile_mask.value().data_ptr<bool>() : nullptr;

    // Count the number of tiles each Gaussian intersects, store in tiles_per_gaussian_cumsum
    const int NUM_BLOCKS = (total_gaussians + NUM_THREADS - 1) / NUM_THREADS;
    count_tiles_per_gaussian<scalar_t, int32_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        total_gaussians, num_gaussians, tile_size, num_tiles_w, num_tiles_h,
        means2d.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(), tile_mask_ptr, camera_jidx_ptr,
        tiles_per_gaussian_cumsum.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // In place cumulative sum to get the total number of intersections
    torch::cumsum_out(tiles_per_gaussian_cumsum, tiles_per_gaussian_cumsum, 0, torch::kInt32);

    // Allocate tensors to store the intersections
    const int64_t total_intersections = tiles_per_gaussian_cumsum[-1].item<int64_t>();
    if (total_intersections == 0) {
        return std::make_tuple(torch::zeros(output_dims, means2d.options().dtype(torch::kInt32)),
                               torch::empty({ 0 }, means2d.options().dtype(torch::kInt32)));
    } else {
        torch::Tensor intersection_keys =
            torch::empty({ total_intersections }, means2d.options().dtype(torch::kInt64));
        torch::Tensor intersection_values =
            torch::empty({ total_intersections }, means2d.options().dtype(torch::kInt32));

        // Compute the set of intersections between each projected Gaussian and each tile,
        // store them in intersection_keys and intersection_values
        // where intersection_keys encodes (camera_id, tile_id, depth) and intersection_values
        // encodes the index of the Gaussian in the input arrays.
        compute_gaussian_tile_intersections<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            num_cameras, num_gaussians, total_gaussians, tile_size, num_tiles_w, num_tiles_h,
            num_tile_id_bits, means2d.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(),
            depths.data_ptr<scalar_t>(), tiles_per_gaussian_cumsum.data_ptr<int32_t>(),
            tile_mask_ptr, camera_jidx_ptr, intersection_keys.data_ptr<int64_t>(),
            intersection_values.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Sort the intersections by their key so intersections within the same tile are grouped
        // together and sorted by their depth (near to far).
        {
            // Allocate tensors to store the sorted intersections
            torch::Tensor keys_sorted = torch::empty_like(intersection_keys);
            torch::Tensor vals_sorted = torch::empty_like(intersection_values);

            // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
            // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(intersection_keys.data_ptr<int64_t>(),
                                              keys_sorted.data_ptr<int64_t>());
            cub::DoubleBuffer<int32_t> d_vals(intersection_values.data_ptr<int32_t>(),
                                              vals_sorted.data_ptr<int32_t>());

            const int32_t num_bits = 32 + num_cam_id_bits + num_tile_id_bits;
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, d_keys, d_vals, total_intersections, 0,
                        num_bits, stream);
            // DoubleBuffer swaps the pointers if the keys were sorted in the input buffer
            // so we need to grab the right buffer.
            if (d_keys.selector == 1) {
                intersection_keys = keys_sorted;
            }
            if (d_vals.selector == 1) {
                intersection_values = vals_sorted;
            }
        }

        if (is_sparse) {
            // Compute a joffsets tensor that stores the offsets into the sorted Gaussian
            // intersections
            torch::Tensor tile_joffsets =
                torch::empty({ output_num_tiles }, means2d.options().dtype(torch::kInt32));
            const int NUM_BLOCKS_2 = (output_num_tiles + NUM_THREADS - 1) / NUM_THREADS;
            compute_tile_offsets_sparse<<<NUM_BLOCKS_2, NUM_THREADS, 0, stream>>>(
                total_intersections, total_tiles, num_tile_id_bits,
                intersection_keys.data_ptr<int64_t>(), active_tiles.value().data_ptr<uint32_t>(),
                num_active_tiles, tile_joffsets.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return std::make_tuple(tile_joffsets, intersection_values);
        } else {
            // Compute a joffsets tensor that stores the offsets into the sorted Gaussian
            // intersections
            torch::Tensor tile_joffsets = torch::empty({ num_cameras, num_tiles_h, num_tiles_w },
                                                       means2d.options().dtype(torch::kInt32));
            const int     NUM_BLOCKS_2  = (total_intersections + NUM_THREADS - 1) / NUM_THREADS;
            compute_tile_offsets<<<NUM_BLOCKS_2, NUM_THREADS, 0, stream>>>(
                total_intersections, num_cameras, total_tiles, num_tile_id_bits,
                intersection_keys.data_ptr<int64_t>(), tile_joffsets.data_ptr<int32_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return std::make_tuple(tile_joffsets, intersection_values);
        }
    }
}

} // namespace

namespace fvdb {
namespace detail {
namespace ops {

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersection<torch::kCUDA>(
    const torch::Tensor               &means2d,     // [C, N, 2] or [M, 2]
    const torch::Tensor               &radii,       // [C, N] or [M]
    const torch::Tensor               &depths,      // [C, N] or [M]
    const at::optional<torch::Tensor> &camera_jidx, // NULL or [M]
    const uint32_t num_cameras, const uint32_t tile_size, const uint32_t num_tiles_h,
    const uint32_t num_tiles_w) {
    return gaussianTileIntersectionCUDAImpl(means2d, radii, depths, camera_jidx, at::nullopt,
                                            at::nullopt, num_cameras, tile_size, num_tiles_h,
                                            num_tiles_w);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersection<torch::kCPU>(
    const torch::Tensor               &means2d,    // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &radii,      // [C, N] or [nnz]
    const torch::Tensor               &depths,     // [C, N] or [nnz]
    const at::optional<torch::Tensor> &camera_ids, // NULL or [M]
    const uint32_t num_cameras, const uint32_t tile_size, const uint32_t num_tiles_h,
    const uint32_t num_tiles_w) {
    TORCH_CHECK(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersectionSparse<torch::kCUDA>(
    const torch::Tensor               &means2d,      // [C, N, 2] or [M, 2]
    const torch::Tensor               &radii,        // [C, N] or [M]
    const torch::Tensor               &depths,       // [C, N] or [M]
    const torch::Tensor               &tile_mask,    // [C, H, W]
    const torch::Tensor               &active_tiles, // [num_active_tiles]
    const at::optional<torch::Tensor> &camera_jidx,  // NULL or [M]
    const uint32_t num_cameras, const uint32_t tile_size, const uint32_t num_tiles_h,
    const uint32_t num_tiles_w) {
    return gaussianTileIntersectionCUDAImpl(means2d, radii, depths, camera_jidx, tile_mask,
                                            active_tiles, num_cameras, tile_size, num_tiles_h,
                                            num_tiles_w);
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersectionSparse<torch::kCPU>(
    const torch::Tensor               &means2d,      // [C, N, 2] or [M, 2]
    const torch::Tensor               &radii,        // [C, N] or [M]
    const torch::Tensor               &depths,       // [C, N] or [M]
    const torch::Tensor               &tile_mask,    // [C, H, W]
    const torch::Tensor               &active_tiles, // [num_active_tiles]
    const at::optional<torch::Tensor> &camera_jidx,  // NULL or [M]
    const uint32_t num_cameras, const uint32_t tile_size, const uint32_t num_tiles_h,
    const uint32_t num_tiles_w) {
    TORCH_CHECK(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
