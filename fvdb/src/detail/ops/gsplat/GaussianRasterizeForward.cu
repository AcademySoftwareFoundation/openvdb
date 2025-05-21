// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "Gaussian2D.cuh"
#include "GaussianVectorTypes.cuh"
#include <detail/ops/Ops.h>
#include <detail/utils/AccessorHelpers.cuh>

#include <optional>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define PRAGMA_UNROLL _Pragma("unroll")

namespace fvdb::detail::ops {
namespace {

template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED> struct DeviceArgs {
    constexpr static std::size_t NUM_OUTER_DIMS = IS_PACKED ? 1 : 2;
    using vec2t                                 = typename Vec2Type<ScalarType>::type;
    using vec3t                                 = typename Vec3Type<ScalarType>::type;
    using FeatureAccessorType                   = fvdb::TorchRAcc64<ScalarType, NUM_OUTER_DIMS + 1>;

    uint32_t mNumCameras;
    uint32_t mNumGaussiansPerCamera;
    uint32_t mTotalIntersections;
    uint32_t mImageWidth;
    uint32_t mImageHeight;
    uint32_t mImageOriginW;
    uint32_t mImageOriginH;
    uint32_t mTileOriginW;
    uint32_t mTileOriginH;
    uint32_t mTileSize;
    uint32_t mNumTilesW;
    uint32_t mNumTilesH;
    vec2t *__restrict__ mMeans2d;              // [C, N, 2] or [nnz, 2]
    vec3t *__restrict__ mConics;               // [C, N, 3] or [nnz, 3]
    FeatureAccessorType mFeatures;             // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    ScalarType *__restrict__ mOpacities;       // [C, N] or [nnz]
    ScalarType *__restrict__ mBackgrounds;     // [C, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    bool *__restrict__ mMasks;                 // [C, nTilesH, nTilesW]
    int32_t *__restrict__ mTileOffsets;        // [C, nTilesH, nTilesW]
    int32_t *__restrict__ mTileGaussianIds;    // [totalIntersections]
    ScalarType *__restrict__ mOutFeatures;     // [C, imgH, imgW, NUM_CHANNELS]
    ScalarType *__restrict__ mOutAlphas;       // [C, imgH, imgW, 1]
    int32_t *__restrict__ mOutLastGaussianIds; // [C, imgH, imgW]

    DeviceArgs(const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
               const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
               const torch::Tensor &features,  // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
               const torch::Tensor &opacities, // [C, N] or [nnz]
               const at::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
               const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
               const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
               const uint32_t imageOriginH, const uint32_t tileSize,
               const torch::Tensor &tileOffsets,               // [C, numTilesH, numTilesW]
               const torch::Tensor &tileGaussianIds            // [totalIntersections]
               )
        : mFeatures(
              features
                  .packed_accessor64<ScalarType, NUM_OUTER_DIMS + 1, torch::RestrictPtrTraits>()) {
        TORCH_CHECK(features.is_cuda(), "Input must be a CUDA tensor");

        checkInputTensor(means2d, "means2d");
        checkInputTensor(conics, "conics");
        checkInputTensor(opacities, "opacities");
        checkInputTensor(tileOffsets, "tileOffsets");
        checkInputTensor(tileGaussianIds, "tileGaussianIds");
        if (backgrounds.has_value()) {
            checkInputTensor(backgrounds.value(), "backgrounds");
        }
        if (masks.has_value()) {
            checkInputTensor(masks.value(), "masks");
        }

        const int64_t numCameras            = tileOffsets.size(0);
        const int64_t numGaussiansPerCamera = IS_PACKED ? 0 : means2d.size(1);
        const int64_t totalGaussians        = IS_PACKED ? means2d.size(0) : 0;
        const int64_t numTilesW             = tileOffsets.size(2);
        const int64_t numTilesH             = tileOffsets.size(1);

        if constexpr (IS_PACKED) {
            TORCH_CHECK_VALUE(means2d.dim() == 2, "Bad number of dims for means2d");
            TORCH_CHECK_VALUE(totalGaussians == means2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(2 == means2d.size(1), "Bad size for means2d");

            TORCH_CHECK_VALUE(conics.dim() == 2, "Bad number of dims for conics");
            TORCH_CHECK_VALUE(totalGaussians == conics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(3 == conics.size(1), "Bad size for conics");

            TORCH_CHECK_VALUE(features.dim() == 2, "Bad number of dims for features");
            TORCH_CHECK_VALUE(totalGaussians == features.size(0), "Bad size for features");
            TORCH_CHECK_VALUE(NUM_CHANNELS == features.size(1), "Bad size for features");

            TORCH_CHECK_VALUE(opacities.dim() == 1, "Bad number of dims for opacities");
            TORCH_CHECK_VALUE(totalGaussians == opacities.size(0), "Bad size for opacities");
        } else {
            TORCH_CHECK_VALUE(means2d.dim() == 3, "Bad number of dims for means2d");
            TORCH_CHECK_VALUE(numCameras == means2d.size(0), "Bad size for means2d");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == means2d.size(1), "Bad size for means2d");
            TORCH_CHECK_VALUE(2 == means2d.size(2), "Bad size for means2d");

            TORCH_CHECK_VALUE(conics.dim() == 3, "Bad number of dims for conics");
            TORCH_CHECK_VALUE(numCameras == conics.size(0), "Bad size for conics");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == conics.size(1), "Bad size for conics");
            TORCH_CHECK_VALUE(3 == conics.size(2), "Bad size for conics");

            TORCH_CHECK_VALUE(features.dim() == 3, "Bad number of dims for features");
            TORCH_CHECK_VALUE(numCameras == features.size(0), "Bad size for features");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == features.size(1), "Bad size for features");
            TORCH_CHECK_VALUE(NUM_CHANNELS == features.size(2), "Bad size for features");

            TORCH_CHECK_VALUE(opacities.dim() == 2, "Bad number of dims for opacities");
            TORCH_CHECK_VALUE(numCameras == opacities.size(0), "Bad size for opacities");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == opacities.size(1), "Bad size for opacities");
        }

        if (backgrounds.has_value()) {
            TORCH_CHECK_VALUE(backgrounds.value().dim() == 2, "Bad number of dims for backgrounds");
            TORCH_CHECK_VALUE(numCameras == backgrounds.value().size(0),
                              "Bad size for backgrounds");
            TORCH_CHECK_VALUE(NUM_CHANNELS == backgrounds.value().size(1),
                              "Bad size for backgrounds");
        }
        if (masks.has_value()) {
            TORCH_CHECK_VALUE(masks.value().dim() == 3, "Bad number of dims for masks");
            TORCH_CHECK_VALUE(numCameras == masks.value().size(0), "Bad size for masks");
            TORCH_CHECK_VALUE(numTilesH == masks.value().size(1), "Bad size for masks");
            TORCH_CHECK_VALUE(numTilesW == masks.value().size(2), "Bad size for masks");
        }

        TORCH_CHECK_VALUE(tileOffsets.dim() == 3, "Bad number of dims for tileOffsets");
        TORCH_CHECK_VALUE(numCameras == tileOffsets.size(0), "Bad size for tileOffsets");
        TORCH_CHECK_VALUE(numTilesH == tileOffsets.size(1), "Bad size for tileOffsets");
        TORCH_CHECK_VALUE(numTilesW == tileOffsets.size(2), "Bad size for tileOffsets");

        mNumCameras            = tileOffsets.size(0);
        mNumGaussiansPerCamera = IS_PACKED ? 0 : means2d.size(1);
        mTotalIntersections    = tileGaussianIds.size(0);
        mImageWidth            = imageWidth;
        mImageHeight           = imageHeight;
        mImageOriginW          = imageOriginW;
        mImageOriginH          = imageOriginH;
        mTileOriginW           = imageOriginW / tileSize;
        mTileOriginH           = imageOriginH / tileSize;
        mTileSize              = tileSize;
        mNumTilesW             = tileOffsets.size(2);
        mNumTilesH             = tileOffsets.size(1);

        static_assert(NUM_OUTER_DIMS == 1 || NUM_OUTER_DIMS == 2, "NUM_OUTER_DIMS must be 1 or 2");

        mMeans2d   = reinterpret_cast<vec2t *>(means2d.data_ptr<ScalarType>());
        mConics    = reinterpret_cast<vec3t *>(conics.data_ptr<ScalarType>());
        mOpacities = opacities.data_ptr<ScalarType>();
        mBackgrounds =
            backgrounds.has_value() ? backgrounds.value().data_ptr<ScalarType>() : nullptr;
        mMasks           = masks.has_value() ? masks.value().data_ptr<bool>() : nullptr;
        mTileOffsets     = tileOffsets.data_ptr<int32_t>();
        mTileGaussianIds = tileGaussianIds.data_ptr<int32_t>();
    }

    void
    setOutputArguments(const torch::Tensor &outFeatures, const torch::Tensor &outAlphas,
                       const torch::Tensor &outLastGaussianIds) {
        mOutFeatures        = outFeatures.data_ptr<ScalarType>();
        mOutAlphas          = outAlphas.data_ptr<ScalarType>();
        mOutLastGaussianIds = outLastGaussianIds.data_ptr<int32_t>();
    }

    inline __device__ void
    advancePointersToCameraPixel(const uint32_t cameraId, const uint32_t i, const uint32_t j) {
        const int32_t pixId = i * mImageWidth + j;

        // Move all the pointers forward to the current camera and pixel
        const std::ptrdiff_t offsetForPixels = cameraId * mImageHeight * mImageWidth + pixId;
        const std::ptrdiff_t offsetForTiles  = cameraId * mNumTilesH * mNumTilesW;

        mTileOffsets += offsetForTiles;
        mOutFeatures += offsetForPixels * NUM_CHANNELS;
        mOutAlphas += offsetForPixels;
        mOutLastGaussianIds += offsetForPixels;
        if (mBackgrounds != nullptr) {
            mBackgrounds += cameraId * NUM_CHANNELS;
        }
        if (mMasks != nullptr) {
            mMasks += offsetForTiles;
        }
    }

    inline void
    checkInputTensor(const torch::Tensor &x, const std::string &name) {
        TORCH_CHECK(x.is_cuda(), "Input ", name, " must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "Input ", name, " must be contiguous");
    }

    __device__ void
    volumeRenderTileForward(const uint32_t tileStart, const uint32_t tileEnd,
                            const uint32_t blockSize, const uint32_t tileSize,
                            const bool writePixel, const uint32_t i, const uint32_t j) {
        using coord2t = typename Vec2Type<int32_t>::type;

        const uint32_t numBatches = (tileEnd - tileStart + blockSize - 1) / blockSize;

        // Ordinal of this thread in the block
        const uint32_t tidx = threadIdx.x * blockDim.y + threadIdx.y;

        // We don't return right away if the pixel is not in the image since we want to use
        // this thread to load gaussians into shared memory
        bool done = !writePixel;

        extern __shared__ int   s[];
        Gaussian2D<ScalarType> *sharedGaussians =
            reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        const ScalarType px = ScalarType(j) + 0.5f;
        const ScalarType py = ScalarType(i) + 0.5f;

        // NOTE: The accumulated transmittance is used in the backward pass, and since it's a
        //       sum of many small numbers, we should really use double precision. However,
        //       this makes the backward pass 1.5x slower, so we stick with float for now and sort
        //       of just ignore small impact gaussians ¯\_(ツ)_/¯.
        ScalarType accumTransmittance = 1.0f;
        // index of most recent gaussian to write to this thread's pixel
        int32_t curIdx = -1;

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        ScalarType pixOut[NUM_CHANNELS] = { 0.f };
        for (uint32_t b = 0; b < numBatches; ++b) {
            // Sync threads before we start integrating the next batch
            // If all threads are done, we can break early
            if (__syncthreads_count(done) == blockSize) {
                break;
            }

            // Each thread fetches one gaussian from front to back (tile_gaussian_ids is depth
            // sorted)
            const uint32_t batchStart = tileStart + blockSize * b;
            const uint32_t idx        = batchStart + tidx;
            if (idx < tileEnd) {
                const int32_t g       = mTileGaussianIds[idx]; // which gaussian we're rendering
                sharedGaussians[tidx] = { g, mMeans2d[g], mOpacities[g], mConics[g] };
            }

            // Sync threads so all gaussians for this batch are loaded in shared memory
            __syncthreads();

            // Volume render Gaussians in this batch
            const uint32_t batchSize = min(blockSize, tileEnd - batchStart);
            for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
                const Gaussian2D<ScalarType> gaussian = sharedGaussians[t];

                const vec2t      delta = gaussian.delta(px, py);
                const ScalarType sigma = gaussian.sigma(delta);
                const ScalarType alpha = min(0.999f, gaussian.opacity * __expf(-sigma));

                // TODO: are we quantizing the alpha too early? They could add up to significant
                // opacity.
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    continue;
                }

                const ScalarType nextTransmittance = accumTransmittance * (ScalarType(1.0) - alpha);
                if (nextTransmittance <= ScalarType(1e-4)) { // this pixel is done: exclusive
                    done = true;
                    break;
                }

                const ScalarType vis = alpha * accumTransmittance;
                if constexpr (IS_PACKED) {
                    const auto featureAccessor = mFeatures[gaussian.id];
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                        pixOut[k] += featureAccessor[k] * vis;
                    }
                } else {
                    const int32_t cid             = gaussian.id / mNumGaussiansPerCamera;
                    const int32_t gid             = gaussian.id % mNumGaussiansPerCamera;
                    const auto    featureAccessor = mFeatures[cid][gid];
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                        pixOut[k] += featureAccessor[k] * vis;
                    }
                }

                curIdx             = batchStart + t;
                accumTransmittance = nextTransmittance;
            }
        }

        if (writePixel) {
            // Here T is the transmittance AFTER the last gaussian in this pixel.
            // We (should) store double precision as T would be used in backward
            // pass and it can be very small and causing large diff in gradients
            // with float32. However, double precision makes the backward pass 1.5x
            // slower so we stick with float for now.
            *mOutAlphas = 1.0f - accumTransmittance;
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                mOutFeatures[k] = mBackgrounds == nullptr
                                      ? pixOut[k]
                                      : (pixOut[k] + accumTransmittance * mBackgrounds[k]);
            }
            // index in bin of last gaussian in this pixel
            *mOutLastGaussianIds = curIdx;
        }
    }
};

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansForward(DeviceArgs<ScalarType, NUM_CHANNELS, IS_PACKED> args) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    const int32_t cameraId = blockIdx.x;

    // blockIdx.yz runs from [0, numTilesH] x [0, numTilesW]
    const int32_t tileId =
        (blockIdx.y + args.mTileOriginH) * args.mNumTilesW + (blockIdx.z + args.mTileOriginW);

    // Pixel coordinates run from [0, height] x [0, width]
    // i.e. they are in the local coordinates of the crop starting from pixel
    //      [image_origin_h, image_origin_w] with size [image_height, image_width]
    const uint32_t i = blockIdx.y * args.mTileSize + threadIdx.y;
    const uint32_t j = blockIdx.z * args.mTileSize + threadIdx.x;

    args.advancePointersToCameraPixel(cameraId, i, j);

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    const bool pixelInImage = (i < args.mImageHeight && j < args.mImageWidth);
    // when the mask is provided, render the background feature/color and return
    // if this tile is labeled as False
    if (args.mMasks != nullptr && pixelInImage && !args.mMasks[tileId]) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            args.mOutFeatures[k] = args.mBackgrounds == nullptr ? 0.0f : args.mBackgrounds[k];
        }
        return;
    }

    // Figure out the first and (one past the) last Gaussian ID in this block/tile
    const int32_t firstGaussianIdInBlock = args.mTileOffsets[tileId];
    const int32_t lastGaussianIdInBlock =
        (cameraId == args.mNumCameras - 1) && (tileId == args.mNumTilesW * args.mNumTilesH - 1)
            ? args.mTotalIntersections
            : args.mTileOffsets[tileId + 1];
    const uint32_t blockSize = blockDim.x * blockDim.y;

    // Pixel coordinates in the global image (not just the local crop)
    const uint32_t globalI = i + args.mImageOriginH;
    const uint32_t globalJ = j + args.mImageOriginW;
    args.volumeRenderTileForward(firstGaussianIdInBlock, lastGaussianIdInBlock, blockSize,
                                 args.mTileSize, pixelInImage, globalI, globalJ);
}

template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launchRasterizeForwardKernel(
    // Gaussian parameters
    const torch::Tensor               &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &features,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor               &opacities,   // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
    const uint32_t imageOriginH, const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,    // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds // [n_isects]
) {
    using vec3t = typename Vec3Type<ScalarType>::type;
    using vec2t = typename Vec2Type<ScalarType>::type;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    TORCH_CHECK_VALUE(means2d.dim() == 3 || means2d.dim() == 2,
                      "means2d must have 3 dimensions (C, N, 2) or 2 dimensions (nnz, 2)");
    TORCH_CHECK_VALUE(conics.dim() == 3 || conics.dim() == 2,
                      "conics must have 3 dimensions (C, N, 3) or 2 dimensions (nnz, 3)");
    TORCH_CHECK_VALUE(
        features.dim() == 3 || features.dim() == 2,
        "features must have 3 dimensions (C, N, channels) or 2 dimensions (nnz, channels)");
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
    TORCH_CHECK_VALUE(tileOffsets.dim() == 3,
                      "tile_offsets must have 3 dimensions (C, tile_height, tile_width)");
    TORCH_CHECK_VALUE(tileGaussianIds.dim() == 1,
                      "tile_gaussian_ids must have 1 dimension (n_isects)");

    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(features);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tileOffsets);
    CHECK_INPUT(tileGaussianIds);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    const bool packed = means2d.dim() == 2;

    const uint32_t C          = tileOffsets.size(0);          // number of cameras
    const uint32_t N          = packed ? 0 : means2d.size(1); // number of gaussians
    const uint32_t channels   = features.size(-1);
    const uint32_t tileHeight = tileOffsets.size(1);
    const uint32_t tileWidth  = tileOffsets.size(2);
    const uint32_t nIsects    = tileGaussianIds.size(0);

    const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
    const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;

    // The rendered images to return (one per camera)
    torch::Tensor outFeatures = torch::empty({ C, imageHeight, imageWidth, channels },
                                             means2d.options().dtype(torch::kFloat32));
    torch::Tensor alphas =
        torch::empty({ C, imageHeight, imageWidth, 1 }, means2d.options().dtype(torch::kFloat32));
    torch::Tensor lastIds =
        torch::empty({ C, imageHeight, imageWidth }, means2d.options().dtype(torch::kInt32));

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Each pixel in each tile will cache a gaussian consisting of:
    //   - int32_t  gaussian_id; -- 4 bytes
    //   - vec2t    xy;          -- 8 bytes for float32
    //   - scalar_t opacity;     -- 4 bytes for float32
    //   - vec3t    conic;       -- 12 bytes for float32
    const uint32_t sharedMem =
        tileSize * tileSize * (sizeof(int32_t) + sizeof(vec2t) + sizeof(float) + sizeof(vec3t));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(rasterizeGaussiansForward<ScalarType, NUM_CHANNELS, IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", sharedMem,
                 " bytes), try lowering tile_size.");
    }

    const dim3 blockDim = { tileSize, tileSize, 1 };
    const dim3 gridDim  = { C, tileExtentH, tileExtentW };

    auto args = DeviceArgs<ScalarType, NUM_CHANNELS, IS_PACKED>(
        means2d, conics, features, opacities, backgrounds, masks, imageWidth, imageHeight,
        imageOriginW, imageOriginH, tileSize, tileOffsets, tileGaussianIds);

    args.setOutputArguments(outFeatures, alphas, lastIds);

    rasterizeGaussiansForward<<<gridDim, blockDim, sharedMem, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(outFeatures, alphas, lastIds);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,        // [C, N, 2]
    const torch::Tensor &conics,         // [C, N, 3]
    const torch::Tensor &features,       // [C, N, D]
    const torch::Tensor &opacities,      // [N]
    const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
    const uint32_t imageOriginH, const uint32_t tileSize,
    const torch::Tensor &tileOffsets,    // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds // [n_isects]
) {
    CHECK_INPUT(features);
    const uint32_t channels = features.size(-1);
    const bool     isPacked = means2d.dim() == 2;

    const std::optional<torch::Tensor> backgrounds = std::nullopt;
    const std::optional<torch::Tensor> masks       = std::nullopt;

#define __CALL_FWD_(N)                                                                             \
    case N: {                                                                                      \
        if (isPacked) {                                                                            \
            return launchRasterizeForwardKernel<float, N, true>(                                   \
                means2d, conics, features, opacities, backgrounds, masks, imageWidth, imageHeight, \
                imageOriginW, imageOriginH, tileSize, tileOffsets, tileGaussianIds);               \
        } else {                                                                                   \
            return launchRasterizeForwardKernel<float, N, false>(                                  \
                means2d, conics, features, opacities, backgrounds, masks, imageWidth, imageHeight, \
                imageOriginW, imageOriginH, tileSize, tileOffsets, tileGaussianIds);               \
        }                                                                                          \
    }
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
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
    const uint32_t imageOriginH, const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,    // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds // [n_isects]
) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
