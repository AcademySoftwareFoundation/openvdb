// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianVectorTypes.cuh"
#include <detail/ops/Ops.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define PRAGMA_UNROLL _Pragma("unroll")

namespace fvdb {
namespace detail {
namespace ops {

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <typename S, uint32_t COLOR_DIM>
__device__ void
volume_render_tile(const uint32_t tile_start, const uint32_t tile_end, const uint32_t block_size,
                   const uint32_t tile_size, const bool write_pixel, const uint32_t i,
                   const uint32_t j, const typename Vec2Type<S>::type *__restrict__ means2d,
                   const typename Vec3Type<S>::type *__restrict__ conics,
                   const S *__restrict__ colors, const S *__restrict__ opacities,
                   const S *__restrict__ background, const int32_t *__restrict__ tile_gaussian_ids,
                   S *__restrict__ out_tile_colors, S *__restrict__ out_tile_alphas,
                   int32_t *__restrict__ out_tile_last_ids) {
    using coord2t = typename Vec2Type<int32_t>::type;
    using vec2t   = typename Vec2Type<S>::type;
    using vec3t   = typename Vec3Type<S>::type;

    const uint32_t num_batches = (tile_end - tile_start + block_size - 1) / block_size;

    // Ordinal of this thread in the block
    const uint32_t tidx = threadIdx.x * blockDim.y + threadIdx.y;

    // We don't return right away if the pixel is not in the image since we want to use
    // this thread to load gaussians into shared memory
    bool done = !write_pixel;

    extern __shared__ int s[];
    struct gaussian_t {
        int32_t id;                                        // 4 bytes
        vec2t   xy;                                        // 8 bytes
        S       opacity;                                   // 4 bytes
        vec3t   conic;                                     // 12 bytes
    };
    gaussian_t *batch = reinterpret_cast<gaussian_t *>(s); // [block_size]

    const S px = (S)(j) + 0.5f;
    const S py = (S)(i) + 0.5f;

    // NOTE: The accumulated transmittance is used in the backward pass, and since it's a
    //       sum of many small numbers, we should really use double precision. However,
    //       this makes the backward pass 1.5x slower, so we stick with float for now and sort of
    //       just ignore small impact gaussians ¯\_(ツ)_/¯.
    S accum_transmittance = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel

    S pix_out[COLOR_DIM] = { 0.f };
    for (uint32_t b = 0; b < num_batches; ++b) {
        // Sync threads before we start integrating the next batch
        // If all threads are done, we can break early
        if (__syncthreads_count(done) == block_size) {
            break;
        }

        // Each thread fetches one gaussian from front to back (tile_gaussian_ids is depth sorted)
        const uint32_t batch_start = tile_start + block_size * b;
        const uint32_t idx         = batch_start + tidx;
        if (idx < tile_end) {
            const int32_t g     = tile_gaussian_ids[idx]; // which gaussian we're rendering
            const vec2t   xy    = means2d[g];
            const S       opac  = opacities[g];
            const vec3t   conic = conics[g];
            batch[tidx]         = { g, xy, opac, conic };
        }

        // Sync threads so all gaussians for this batch are loaded in shared memory
        __syncthreads();

        // Volume render Gaussians in this batch
        const uint32_t batch_size = min(block_size, tile_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const gaussian_t gaussian = batch[t];

            const vec3t conic = gaussian.conic;
            const vec2t delta = { gaussian.xy.x - px, gaussian.xy.y - py };
            const S     sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            const S alpha = min(0.999f, gaussian.opacity * __expf(-sigma));

            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_transmittance = accum_transmittance * (1.0f - alpha);
            if (next_transmittance <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            const S  vis   = alpha * accum_transmittance;
            const S *c_ptr = colors + gaussian.id * COLOR_DIM;
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }

            cur_idx             = batch_start + t;
            accum_transmittance = next_transmittance;
        }
    }

    if (write_pixel) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        *out_tile_alphas = 1.0f - accum_transmittance;
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            out_tile_colors[k] = background == nullptr
                                     ? pix_out[k]
                                     : (pix_out[k] + accum_transmittance * background[k]);
        }
        // index in bin of last gaussian in this pixel
        *out_tile_last_ids = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t COLOR_DIM, typename S>
__global__ void
rasterize_forward(const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
                  const typename Vec2Type<S>::type *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
                  const typename Vec3Type<S>::type *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
                  const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
                  const S *__restrict__ opacities,   // [C, N] or [nnz]
                  const S *__restrict__ backgrounds, // [C, COLOR_DIM]
                  const bool *__restrict__ masks,    // [C, tile_height, tile_width]
                  const uint32_t image_width, const uint32_t image_height,
                  const uint32_t image_origin_w, const uint32_t image_origin_h,
                  const uint32_t tile_origin_w, const uint32_t tile_origin_h,
                  const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
                  const int32_t *__restrict__ tile_offsets,      // [C, tile_height, tile_width]
                  const int32_t *__restrict__ tile_gaussian_ids, // [n_isects]
                  S *__restrict__ out_render_colors, // [C, image_height, image_width, COLOR_DIM]
                  S *__restrict__ out_render_alphas, // [C, image_height, image_width, 1]
                  int32_t *__restrict__ out_last_ids // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    const int32_t camera_id = blockIdx.x;

    // blockIdx.yz runs from [0, num_tiles_h] x [0, num_tiles_w]
    const int32_t tile_id =
        (blockIdx.y + tile_origin_h) * tile_width + (blockIdx.z + tile_origin_w);

    // Pixel coordinates run from [0, height] x [0, width]
    // i.e. they are in the local coordinates of the crop starting from pixel
    //      [image_origin_h, image_origin_w] with size [image_height, image_width]
    const uint32_t i      = blockIdx.y * tile_size + threadIdx.y;
    const uint32_t j      = blockIdx.z * tile_size + threadIdx.x;
    const int32_t  pix_id = i * image_width + j;

    tile_offsets += camera_id * tile_height * tile_width;
    auto const camera_pix_offset = camera_id * image_height * image_width + pix_id;
    out_render_colors += camera_pix_offset * COLOR_DIM;
    out_render_alphas += camera_pix_offset;
    out_last_ids += camera_pix_offset;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    const bool pixel_in_image = (i < image_height && j < image_width);

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && pixel_in_image && !masks[tile_id]) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            out_render_colors[k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int32_t  range_start = tile_offsets[tile_id];
    const int32_t  range_end   = (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
                                     ? n_isects
                                     : tile_offsets[tile_id + 1];
    const uint32_t block_size  = blockDim.x * blockDim.y;

    // Pixel coordinates in the global image (not just the local crop)
    const uint32_t global_i = i + image_origin_h;
    const uint32_t global_j = j + image_origin_w;
    return volume_render_tile<S, COLOR_DIM>(range_start, range_end, block_size, tile_size,
                                            pixel_in_image, global_i, global_j, means2d, conics,
                                            colors, opacities, backgrounds, tile_gaussian_ids,
                                            out_render_colors, out_render_alphas, out_last_ids);
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
call_fwd_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor               &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &colors,      // [C, N, channels] or [nnz, channels]
    const torch::Tensor               &opacities,   // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t image_origin_w,
    const uint32_t image_origin_h, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tile_gaussian_ids // [n_isects]
) {
    using vec3t = typename Vec3Type<float>::type;
    using vec2t = typename Vec2Type<float>::type;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    TORCH_CHECK_VALUE(means2d.dim() == 3 || means2d.dim() == 2,
                      "means2d must have 3 dimensions (C, N, 2) or 2 dimensions (nnz, 2)");
    TORCH_CHECK_VALUE(conics.dim() == 3 || conics.dim() == 2,
                      "conics must have 3 dimensions (C, N, 3) or 2 dimensions (nnz, 3)");
    TORCH_CHECK_VALUE(
        colors.dim() == 3 || colors.dim() == 2,
        "colors must have 3 dimensions (C, N, channels) or 2 dimensions (nnz, channels)");
    TORCH_CHECK_VALUE(opacities.dim() == 2 || opacities.dim() == 1,
                      "opacities must have 2 dimensions (C, N) or 1 dimension (nnz)");
    if (backgrounds.has_value()) {
        TORCH_CHECK_VALUE(backgrounds.value().dim() == 2,
                          "backgrounds must have 2 dimensions (C, channels)");
    }
    if (masks.has_value()) {
        TORCH_CHECK_VALUE(masks.value().dim() == 3,
                          "masks must have 3 dimensions (C, tile_height, tile_width)");
    }
    TORCH_CHECK_VALUE(tile_offsets.dim() == 3,
                      "tile_offsets must have 3 dimensions (C, tile_height, tile_width)");
    TORCH_CHECK_VALUE(tile_gaussian_ids.dim() == 1,
                      "tile_gaussian_ids must have 1 dimension (n_isects)");

    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(tile_gaussian_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    const bool packed = means2d.dim() == 2;

    const uint32_t C           = tile_offsets.size(0);         // number of cameras
    const uint32_t N           = packed ? 0 : means2d.size(1); // number of gaussians
    const uint32_t channels    = colors.size(-1);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width  = tile_offsets.size(2);
    const uint32_t n_isects    = tile_gaussian_ids.size(0);

    const uint32_t tile_origin_w = image_origin_w / tile_size;
    const uint32_t tile_origin_h = image_origin_h / tile_size;
    const uint32_t tile_extent_w = (image_width + tile_size - 1) / tile_size;
    const uint32_t tile_extent_h = (image_height + tile_size - 1) / tile_size;

    // The rendered images to return (one per camera)
    torch::Tensor out_images = torch::empty({ C, image_height, image_width, channels },
                                            means2d.options().dtype(torch::kFloat32));
    torch::Tensor alphas =
        torch::empty({ C, image_height, image_width, 1 }, means2d.options().dtype(torch::kFloat32));
    torch::Tensor last_ids =
        torch::empty({ C, image_height, image_width }, means2d.options().dtype(torch::kInt32));

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Each pixel in each tile will cache a gaussian consisting of:
    //   - int32_t  gaussian_id; -- 4 bytes
    //   - vec2t    xy;          -- 8 bytes for float32
    //   - scalar_t opacity;     -- 4 bytes for float32
    //   - vec3t    conic;       -- 12 bytes for float32
    const uint32_t shared_mem =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(vec2t) + sizeof(float) + sizeof(vec3t));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(rasterize_forward<CDIM, float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                 " bytes), try lowering tile_size.");
    }

    const dim3 threads = { tile_size, tile_size, 1 };
    const dim3 blocks  = { C, tile_extent_h, tile_extent_w };
    rasterize_forward<CDIM, float><<<blocks, threads, shared_mem, stream>>>(
        C, N, n_isects, packed, reinterpret_cast<vec2t *>(means2d.data_ptr<float>()),
        reinterpret_cast<vec3t *>(conics.data_ptr<float>()), colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
        masks.has_value() ? masks.value().data_ptr<bool>() : nullptr, image_width, image_height,
        image_origin_w, image_origin_h, tile_origin_w, tile_origin_h, tile_size, tile_width,
        tile_height, tile_offsets.data_ptr<int32_t>(), tile_gaussian_ids.data_ptr<int32_t>(),
        out_images.data_ptr<float>(), alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(out_images, alphas, last_ids);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,          // [C, N, 2]
    const torch::Tensor &conics,           // [C, N, 3]
    const torch::Tensor &colors,           // [C, N, D]
    const torch::Tensor &opacities,        // [N]
    const uint32_t image_width, const uint32_t image_height, const uint32_t image_origin_w,
    const uint32_t image_origin_h, const uint32_t tile_size,
    const torch::Tensor &tile_offsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tile_gaussian_ids // [n_isects]
) {
    CHECK_INPUT(colors);
    const uint32_t channels = colors.size(-1);

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

#define __CALL_FWD_(N)                                                                         \
    case N:                                                                                    \
        return call_fwd_kernel_with_dim<N>(                                                    \
            means2d, conics, colors, opacities, backgrounds, masks, image_width, image_height, \
            image_origin_w, image_origin_h, tile_size, tile_offsets, tile_gaussian_ids);
    // Make channels a compile time constant and do everything in register space but at the expense
    // of making this code ugly.
    // NOTE: We do powers of two and powers of two plus one to handle rendering common feature
    // channel dimensions with an optional additional depth channel
    switch (channels) {
        __CALL_FWD_(1)
        __CALL_FWD_(2)
        __CALL_FWD_(3)
        __CALL_FWD_(4)
        __CALL_FWD_(5)
        __CALL_FWD_(8)
        __CALL_FWD_(9)
        __CALL_FWD_(16)
        __CALL_FWD_(17)
        __CALL_FWD_(32)
        __CALL_FWD_(33)
        __CALL_FWD_(64)
        __CALL_FWD_(65)
        __CALL_FWD_(128)
        __CALL_FWD_(129)
        __CALL_FWD_(256)
        __CALL_FWD_(257)
        __CALL_FWD_(512)
        __CALL_FWD_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t image_origin_w,
    const uint32_t image_origin_h, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tile_gaussian_ids // [n_isects]
) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
