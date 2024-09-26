// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include "GsplatTypes.cuh"

#include <detail/ops/Ops.h>

#include <ATen/cuda/Atomic.cuh>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace fvdb {
namespace detail {
namespace ops {
namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

template <typename T>
__global__ void
isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C, const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss,           // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,                 // [n_isects]
    int32_t *__restrict__ flatten_ids                // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    uint32_t idx        = cg::this_grid().thread_rank();
    bool     first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);

    OpT tile_radius = radius / static_cast<OpT>(tile_size);
    OpT tile_x      = mean2d.x / static_cast<OpT>(tile_size);
    OpT tile_y      = mean2d.y / static_cast<OpT>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius)), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(tile_y - tile_radius)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] =
            static_cast<int32_t>((tile_max.y - tile_min.y) * (tile_max.x - tile_min.x));
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx      = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_tensor(const torch::Tensor               &means2d,      // [C, N, 2] or [nnz, 2]
                   const torch::Tensor               &radii,        // [C, N] or [nnz]
                   const torch::Tensor               &depths,       // [C, N] or [nnz]
                   const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                   const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                   const uint32_t C, const uint32_t tile_size, const uint32_t tile_width,
                   const uint32_t tile_height, const bool sort, const bool double_buffer) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(depths);
    if (camera_ids.has_value()) {
        GSPLAT_CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        GSPLAT_CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr   = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz         = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(camera_ids.has_value() && gaussian_ids.has_value(),
                    "When packed is set, camera_ids and gaussian_ids must be provided.");
        camera_ids_ptr   = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N           = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t             n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream  = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits  = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    int64_t       n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, means2d.scalar_type(),
            "isect_tiles_total_elems", [&]() {
                isect_tiles<<<(total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                              GSPLAT_N_THREADS, 0, stream>>>(
                    packed, C, N, nnz, camera_ids_ptr, gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(), depths.data_ptr<scalar_t>(), nullptr, tile_size,
                    tile_width, tile_height, tile_n_bits, tiles_per_gauss.data_ptr<int32_t>(),
                    nullptr, nullptr);
            });
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({ -1 }), 0);
        n_isects            = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids   = torch::empty({ n_isects }, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids = torch::empty({ n_isects }, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, means2d.scalar_type(),
            "isect_tiles_n_isects", [&]() {
                isect_tiles<<<(total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                              GSPLAT_N_THREADS, 0, stream>>>(
                    packed, C, N, nnz, camera_ids_ptr, gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(), depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.data_ptr<int64_t>(), tile_size, tile_width, tile_height,
                    tile_n_bits, nullptr, isect_ids.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>());
            });
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted   = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(isect_ids.data_ptr<int64_t>(),
                                              isect_ids_sorted.data_ptr<int64_t>());
            cub::DoubleBuffer<int32_t> d_values(flatten_ids.data_ptr<int32_t>(),
                                                flatten_ids_sorted.data_ptr<int32_t>());
            GSPLAT_CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, d_keys, d_values, n_isects, 0,
                               32 + tile_n_bits + cam_n_bits, stream);
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            GSPLAT_CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, isect_ids.data_ptr<int64_t>(),
                               isect_ids_sorted.data_ptr<int64_t>(),
                               flatten_ids.data_ptr<int32_t>(),
                               flatten_ids_sorted.data_ptr<int32_t>(), n_isects, 0,
                               32 + tile_n_bits + cam_n_bits, stream);
        }
        return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

__global__ void
isect_offset_encode(const uint32_t n_isects, const int64_t *__restrict__ isect_ids,
                    const uint32_t C, const uint32_t n_tiles, const uint32_t tile_n_bits,
                    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr      = isect_id_curr >> tile_n_bits;
    int64_t tid_curr      = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr       = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid,
        // tile_id) pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev  = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

torch::Tensor
isect_offset_encode_tensor(const torch::Tensor &isect_ids, // [n_isects]
                           const uint32_t C, const uint32_t tile_width,
                           const uint32_t tile_height) {
    GSPLAT_DEVICE_GUARD(isect_ids);
    GSPLAT_CHECK_INPUT(isect_ids);

    uint32_t      n_isects = isect_ids.size(0);
    torch::Tensor offsets =
        torch::empty({ C, tile_height, tile_width }, isect_ids.options().dtype(torch::kInt32));
    if (n_isects) {
        uint32_t             n_tiles     = tile_width * tile_height;
        uint32_t             tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
        at::cuda::CUDAStream stream      = at::cuda::getCurrentCUDAStream();
        isect_offset_encode<<<(n_isects + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                              GSPLAT_N_THREADS, 0, stream>>>(
            n_isects, isect_ids.data_ptr<int64_t>(), C, n_tiles, tile_n_bits,
            offsets.data_ptr<int32_t>());
    } else {
        offsets.fill_(0);
    }
    return offsets;
}

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void
rasterize_to_pixels_fwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    const vec2<S> *__restrict__ means2d,      // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,       // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,             // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,          // [C, N] or [nnz]
    const S *__restrict__ backgrounds,        // [C, COLOR_DIM]
    const bool *__restrict__ masks,           // [C, tile_height, tile_width]
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    S *__restrict__ render_colors,            // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas,            // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids            // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto     block     = cg::this_thread_block();
    int32_t  camera_id = block.group_index().x;
    int32_t  tile_id   = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i         = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j         = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * COLOR_DIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    S       px     = (S)j + 0.5f;
    S       py     = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done   = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t        range_start = tile_offsets[tile_id];
    int32_t        range_end   = (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
                                     ? n_isects
                                     : tile_offsets[tile_id + 1];
    const uint32_t block_size  = block.size();
    uint32_t       num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t              *id_batch = (int32_t *)s;                      // [block_size]
    vec3<S>              *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]);         // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    S T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    S pix_out[COLOR_DIM] = { 0.f };
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx         = batch_start + tr;
        if (idx < range_end) {
            int32_t g            = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr]         = g;
            const vec2<S> xy     = means2d[g];
            const S       opac   = opacities[g];
            xy_opacity_batch[tr] = { xy.x, xy.y, opac };
            conic_batch[tr]      = conics[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic   = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S       opac    = xy_opac.z;
            const vec2<S> delta   = { xy_opac.x - px, xy_opac.y - py };
            const S sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            S alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t  g     = id_batch[t];
            const S  vis   = alpha * T;
            const S *c_ptr = colors + g * COLOR_DIM;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor               &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &colors,      // [C, N, channels] or [nnz, channels]
    const torch::Tensor               &opacities,   // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C           = tile_offsets.size(0);         // number of cameras
    uint32_t N           = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels    = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width  = tile_offsets.size(2);
    uint32_t n_isects    = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = { tile_size, tile_size, 1 };
    dim3 blocks  = { C, tile_height, tile_width };

    torch::Tensor renders = torch::empty({ C, image_height, image_width, channels },
                                         means2d.options().dtype(torch::kFloat32));
    torch::Tensor alphas =
        torch::empty({ C, image_height, image_width, 1 }, means2d.options().dtype(torch::kFloat32));
    torch::Tensor last_ids =
        torch::empty({ C, image_height, image_width }, means2d.options().dtype(torch::kInt32));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t       shared_mem =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<CDIM, float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                 " bytes), try lowering tile_size.");
    }
    rasterize_to_pixels_fwd_kernel<CDIM, float><<<blocks, threads, shared_mem, stream>>>(
        C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
        reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()), colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
        masks.has_value() ? masks.value().data_ptr<bool>() : nullptr, image_width, image_height,
        tile_size, tile_width, tile_height, tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(), renders.data_ptr<float>(), alphas.data_ptr<float>(),
        last_ids.data_ptr<int32_t>());

    return std::make_tuple(renders, alphas, last_ids);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor               &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &colors,      // [C, N, channels] or [nnz, channels]
    const torch::Tensor               &opacities,   // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                                         \
    case N:                                                                                    \
        return call_kernel_with_dim<N>(means2d, conics, colors, opacities, backgrounds, masks, \
                                       image_width, image_height, tile_size, tile_offsets,     \
                                       flatten_ids);

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}

/****************************************************************************
 * Rasterization to Pixels Backward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void
rasterize_to_pixels_bwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d,      // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,       // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,             // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,          // [C, N] or [nnz]
    const S *__restrict__ backgrounds,        // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const bool *__restrict__ masks,           // [C, tile_height, tile_width]
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, image_height, image_width,
                                           // COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    vec2<S> *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_colors,            // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities          // [C, N] or [nnz]
) {
    auto     block     = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id   = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i         = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j         = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t        range_start = tile_offsets[tile_id];
    int32_t        range_end   = (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
                                     ? n_isects
                                     : tile_offsets[tile_id + 1];
    const uint32_t block_size  = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t              *id_batch = (int32_t *)s;                      // [block_size]
    vec3<S>              *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]);         // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    S *rgbs_batch = (S *)&conic_batch[block_size];                      // [block_size * COLOR_DIM]

    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alphas[pix_id];
    S T       = T_final;
    // the contribution from gaussians behind the current one
    S buffer[COLOR_DIM] = { 0.f };
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t            tr             = block.thread_rank();
    cg::thread_block_tile<32> warp           = cg::tiled_partition<32>(block);
    const int32_t             warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end  = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx        = batch_end - tr;
        if (idx >= range_start) {
            int32_t g            = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr]         = g;
            const vec2<S> xy     = means2d[g];
            const S       opac   = opacities[g];
            xy_opacity_batch[tr] = { xy.x, xy.y, opac };
            conic_batch[tr]      = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            S       alpha;
            S       opac;
            vec2<S> delta;
            vec3<S> conic;
            S       vis;

            if (valid) {
                conic           = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac            = xy_opac.z;
                delta           = { xy_opac.x - px, xy_opac.y - py };
                S sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                          conic.y * delta.x * delta.y;
                vis   = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S       v_rgb_local[COLOR_DIM] = { 0.f };
            vec3<S> v_conic_local          = { 0.f, 0.f, 0.f };
            vec2<S> v_xy_local             = { 0.f, 0.f };
            vec2<S> v_xy_abs_local         = { 0.f, 0.f };
            S       v_opacity_local        = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const S fac = alpha * T;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                S v_alpha = 0.f;
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_alpha += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    S accum = 0.f;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const S v_sigma = -opac * vis * v_alpha;
                    v_conic_local   = { 0.5f * v_sigma * delta.x * delta.x,
                                        v_sigma * delta.x * delta.y,
                                        0.5f * v_sigma * delta.y * delta.y };
                    v_xy_local      = { v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                        v_sigma * (conic.y * delta.x + conic.z * delta.y) };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = { abs(v_xy_local.x), abs(v_xy_local.y) };
                    }
                    v_opacity_local = vis * v_alpha;
                }

                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }
            }
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<decltype(warp), S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum<decltype(warp), S>(v_xy_abs_local, warp);
            }
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g         = id_batch[t]; // flatten index in [C * N] or [nnz]
                S      *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor               &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &colors,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &opacities,   // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    GSPLAT_CHECK_INPUT(render_alphas);
    GSPLAT_CHECK_INPUT(last_ids);
    GSPLAT_CHECK_INPUT(v_render_colors);
    GSPLAT_CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C           = tile_offsets.size(0);         // number of cameras
    uint32_t N           = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects    = flatten_ids.size(0);
    uint32_t COLOR_DIM   = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width  = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = { tile_size, tile_size, 1 };
    dim3 blocks  = { C, tile_height, tile_width };

    torch::Tensor v_means2d   = torch::zeros_like(means2d);
    torch::Tensor v_conics    = torch::zeros_like(conics);
    torch::Tensor v_colors    = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    if (n_isects) {
        const uint32_t shared_mem = tile_size * tile_size *
                                    (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
                                     sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<CDIM, float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_bwd_kernel<CDIM, float><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed, reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()), colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr, image_width, image_height,
            tile_size, tile_width, tile_height, tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(), render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(), v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
            reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()), v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>());
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor               &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &colors,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &opacities,   // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                                         \
    case N:                                                                                    \
        return call_kernel_with_dim<N>(means2d, conics, colors, opacities, backgrounds, masks, \
                                       image_width, image_height, tile_size, tile_offsets,     \
                                       flatten_ids, render_alphas, last_ids, v_render_colors,  \
                                       v_render_alphas, absgrad);

    switch (COLOR_DIM) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}

} // namespace gsplat

// isect_tiles

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchIsectTiles<torch::kCUDA>(const torch::Tensor &means2d, // [C, N, 2] or [nnz, 2]
                                 const torch::Tensor &radii,   // [C, N] or [nnz]
                                 const torch::Tensor &depths,  // [C, N] or [nnz]
                                 const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                                 const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                                 const uint32_t C, const uint32_t tile_size,
                                 const uint32_t tile_width, const uint32_t tile_height,
                                 const bool sort, const bool double_buffer) {
    return gsplat::isect_tiles_tensor(means2d, radii, depths, camera_ids, gaussian_ids, C,
                                      tile_size, tile_width, tile_height, sort, double_buffer);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchIsectTiles<torch::kCPU>(const torch::Tensor               &means2d, // [C, N, 2] or [nnz, 2]
                                const torch::Tensor               &radii,   // [C, N] or [nnz]
                                const torch::Tensor               &depths,  // [C, N] or [nnz]
                                const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                                const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                                const uint32_t C, const uint32_t tile_size,
                                const uint32_t tile_width, const uint32_t tile_height,
                                const bool sort, const bool double_buffer) {
    TORCH_CHECK(false, "CPU implementation not available");
}

// isect_offset_encode

template <>
torch::Tensor
dispatchIsectOffsetEncode<torch::kCUDA>(const torch::Tensor &isect_ids, // [n_isects]
                                        const uint32_t C, const uint32_t tile_width,
                                        const uint32_t tile_height) {
    return gsplat::isect_offset_encode_tensor(isect_ids, C, tile_width, tile_height);
}

template <>
torch::Tensor
dispatchIsectOffsetEncode<torch::kCPU>(const torch::Tensor &isect_ids, // [n_isects]
                                       const uint32_t C, const uint32_t tile_width,
                                       const uint32_t tile_height) {
    TORCH_CHECK(false, "CPU implementation not available");
}

// rasterize_to_pixels
template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeToPixelsForward<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    return gsplat::rasterize_to_pixels_fwd_tensor(
        means2d, conics, colors, opacities, std::nullopt /*backgrounds*/, std::nullopt /*mask*/,
        image_width, image_height, tile_size, tile_offsets, flatten_ids);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeToPixelsForward<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    TORCH_CHECK(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeToPixelsBackward<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad) {
    return gsplat::rasterize_to_pixels_bwd_tensor(
        means2d, conics, colors, opacities, std::nullopt /*backgrounds*/, std::nullopt /*mask*/,
        image_width, image_height, tile_size, tile_offsets, flatten_ids, render_alphas, last_ids,
        v_render_colors, v_render_alphas, absgrad);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeToPixelsBackward<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad) {
    TORCH_CHECK(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
