// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "VectorTypes.cuh"
#include <detail/ops/Ops.h>
#include <detail/utils/cuda/Utils.cuh>

#include <ATen/cuda/Atomic.cuh>

#include <cooperative_groups.h>

namespace fvdb::detail::ops {

template <typename ScalarType, uint32_t COLOR_DIM, bool IS_PACKED> struct DeviceArgs {
    constexpr static uint32_t N_OUTER_DIMS = IS_PACKED ? 1 : 2;
    using vec2t                            = typename Vec2Type<ScalarType>::type;
    using vec3t                            = typename Vec3Type<ScalarType>::type;
    using ColorAccessorType                = fvdb::TorchRAcc64<ScalarType, N_OUTER_DIMS + 1>;

    uint32_t numCameras;
    uint32_t numGaussiansPerCamera;
    uint32_t totalIntersections;
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageOriginW;
    uint32_t imageOriginH;
    uint32_t tileOriginW;
    uint32_t tileOriginH;
    uint32_t tileSize;
    uint32_t numTilesW;
    uint32_t numTilesH;
    vec2t *__restrict__ means2d;                   // [C, N, 2] or [nnz, 2]
    vec3t *__restrict__ conics;                    // [C, N, 3] or [nnz, 3]
    ColorAccessorType colors;                      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    ScalarType *__restrict__ opacities;            // [C, N] or [nnz]
    ScalarType *__restrict__ backgrounds;          // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    bool *__restrict__ masks;                      // [C, nTilesH, nTilesW]
    int32_t *__restrict__ tileOffsets;             // [C, nTilesH, nTilesW]
    int32_t *__restrict__ tileGaussianIds;         // [totalIntersections]
    ScalarType *__restrict__ renderedAlphas;       // [C, imgH, imgW, 1]
    int32_t *__restrict__ lastGaussianIdsPerPixel; // [C, imgH, imgW]
    ScalarType *__restrict__ dLossDRenderedColors; // [C, imgH, imgW, COLOR_DIM]
    ScalarType *__restrict__ dLossDRenderedAlphas; // [C, imgH, imgW, 1]
    vec2t *__restrict__ outDLossDMeans2dAbs;       // [C, N, 2] or [nnz, 2]
    vec2t *__restrict__ outDLossDMeans2d;          // [C, N, 2] or [nnz, 2]
    vec3t *__restrict__ outDLossDConics;           // [C, N, 3] or [nnz, 3]
    ScalarType *__restrict__ outDLossDColors;      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    ScalarType *__restrict__ outDLossDOpacities;   // [C, N] or [nnz]

    struct alignas(32) Gaussian {                  // 28 bytes
        int32_t    id;                             // 4 bytes
        vec2t      xy;                             // 8 bytes
        ScalarType opacity;                        // 4 bytes
        vec3t      conic;                          // 12 bytes

        inline __device__ vec2t
        delta(const ScalarType px, const ScalarType py) const {
            return { xy.x - px, xy.y - py };
        }

        inline __device__ ScalarType
        sigma(const vec2t delta) const {
            return ScalarType{ 0.5 } * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                   conic.y * delta.x * delta.y;
        }
    };

    struct alignas(16) Color {                             // 12 bytes for 3 channels (align to 16)
        ScalarType rgb[COLOR_DIM];                         // 4 * COLOR_DIM bytes
    };

    DeviceArgs(const torch::Tensor               &means2d, // [C, N, 2] or [nnz, 2]
               const torch::Tensor               &conics,  // [C, N, 3] or [nnz, 3]
               const torch::Tensor               &colors,  // [C, N, 3] or [nnz, 3]
               const torch::Tensor               &opacities,   // [C, N] or [nnz]
               const at::optional<torch::Tensor> &backgrounds, // [C, 3]
               const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
               const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
               const uint32_t imageOriginH, const uint32_t tileSize,
               const torch::Tensor &tileOffsets,               // [C, numTilesH, numTilesW]
               const torch::Tensor &tileGaussianIds,           // [totalIntersections]
               const torch::Tensor &renderedAlphas,            // [C, imageHeight, imageWidth, 1]
               const torch::Tensor &lastGaussianIdsPerPixel,   // [C, imageHeight, imageWidth]
               const torch::Tensor &dLossDRenderedColors,      // [C, imageHeight, imageWidth, 3]
               const torch::Tensor &dLossDRenderedAlphas       // [C, imageHeight, imageWidth, 1]
               )
        : colors(
              colors.packed_accessor64<ScalarType, N_OUTER_DIMS + 1, torch::RestrictPtrTraits>()) {
        TORCH_CHECK(colors.is_cuda(), "Input must be a CUDA tensor");

        checkInputTensor(means2d, "means2d");
        checkInputTensor(conics, "conics");
        checkInputTensor(opacities, "opacities");
        checkInputTensor(tileOffsets, "tileOffsets");
        checkInputTensor(tileGaussianIds, "tileGaussianIds");
        checkInputTensor(renderedAlphas, "renderedAlphas");
        checkInputTensor(lastGaussianIdsPerPixel, "lastGaussianIdsPerPixel");
        checkInputTensor(dLossDRenderedColors, "dLossDRenderedColors");
        checkInputTensor(dLossDRenderedAlphas, "dLossDRenderedAlphas");
        if (backgrounds.has_value()) {
            checkInputTensor(backgrounds.value(), "backgrounds");
        }
        if (masks.has_value()) {
            checkInputTensor(masks.value(), "masks");
        }

        this->numCameras            = tileOffsets.size(0);
        this->numGaussiansPerCamera = IS_PACKED ? 0 : means2d.size(1);
        this->totalIntersections    = tileGaussianIds.size(0);
        this->imageWidth            = imageWidth;
        this->imageHeight           = imageHeight;
        this->imageOriginW          = imageOriginW;
        this->imageOriginH          = imageOriginH;
        this->tileOriginW           = imageOriginW / tileSize;
        this->tileOriginH           = imageOriginH / tileSize;
        this->tileSize              = tileSize;
        this->numTilesW             = tileOffsets.size(2);
        this->numTilesH             = tileOffsets.size(1);

        static_assert(N_OUTER_DIMS == 1 || N_OUTER_DIMS == 2, "N_OUTER_DIMS must be 1 or 2");

        this->means2d   = reinterpret_cast<vec2t *>(means2d.data_ptr<ScalarType>());
        this->conics    = reinterpret_cast<vec3t *>(conics.data_ptr<ScalarType>());
        this->opacities = opacities.data_ptr<ScalarType>();
        this->backgrounds =
            backgrounds.has_value() ? backgrounds.value().data_ptr<ScalarType>() : nullptr;
        this->masks           = masks.has_value() ? masks.value().data_ptr<bool>() : nullptr;
        this->tileOffsets     = tileOffsets.data_ptr<int32_t>();
        this->tileGaussianIds = tileGaussianIds.data_ptr<int32_t>();
        this->renderedAlphas  = renderedAlphas.data_ptr<ScalarType>();
        this->lastGaussianIdsPerPixel = lastGaussianIdsPerPixel.data_ptr<int32_t>();
        this->dLossDRenderedColors    = dLossDRenderedColors.data_ptr<ScalarType>();
        this->dLossDRenderedAlphas    = dLossDRenderedAlphas.data_ptr<ScalarType>();
    }

    void
    setOutputArguments(const torch::Tensor &outDLossDMeans2d, const torch::Tensor &outDLossDConics,
                       const torch::Tensor &outDLossDColors,
                       const torch::Tensor &outDLossDOpacities,
                       const torch::Tensor &outDLossDMeans2dAbs, const bool absgrad) {
        this->outDLossDMeans2dAbs =
            absgrad ? reinterpret_cast<vec2t *>(outDLossDMeans2dAbs.data_ptr<ScalarType>())
                    : nullptr;
        this->outDLossDMeans2d = reinterpret_cast<vec2t *>(outDLossDMeans2d.data_ptr<ScalarType>());
        this->outDLossDConics  = reinterpret_cast<vec3t *>(outDLossDConics.data_ptr<ScalarType>());
        this->outDLossDColors  = outDLossDColors.data_ptr<ScalarType>();
        this->outDLossDOpacities = outDLossDOpacities.data_ptr<ScalarType>();
    }

    inline __device__ void
    advancePointersToCamera(const uint32_t cameraId) {
        // Move all the pointers forward to the current camera
        const std::ptrdiff_t offsetForPixels = cameraId * imageHeight * imageWidth;
        const std::ptrdiff_t offsetForTiles  = cameraId * numTilesH * numTilesW;

        tileOffsets += offsetForTiles;
        renderedAlphas += offsetForPixels;
        lastGaussianIdsPerPixel += offsetForPixels;
        dLossDRenderedColors += offsetForPixels * COLOR_DIM;
        dLossDRenderedAlphas += offsetForPixels;
        if (backgrounds != nullptr) {
            backgrounds += cameraId * COLOR_DIM;
        }
        if (masks != nullptr) {
            masks += offsetForTiles;
        }
    }

    inline __device__ void
    fetchGaussianIntoSharedMemory(const int32_t idx, Gaussian *sharedGaussians,
                                  Color *sharedGaussianColors, const int32_t threadOrdinal) const {
        const int32_t    g             = tileGaussianIds[idx]; // Gaussian index in [C * N] or [nnz]
        const vec2t      xy            = means2d[g];
        const ScalarType opac          = opacities[g];
        const vec3t      conic         = conics[g];
        sharedGaussians[threadOrdinal] = { g, xy, opac, conic };

        if constexpr (IS_PACKED) {
            const auto colorAccessor = colors[g];
#pragma unroll COLOR_DIM
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                sharedGaussianColors[threadOrdinal].rgb[k] = colorAccessor[k];
            }
        } else {
            // colors: [C, N, COLOR_DIM]
            // colors[c, n, k] = [c * N * COLOR_DIM + n * COLOR_DIM + k]
            // g = c * N + n
            const int32_t cid           = g / numGaussiansPerCamera;
            const int32_t gid           = g % numGaussiansPerCamera;
            const auto    colorAccessor = colors[cid][gid];
#pragma unroll COLOR_DIM
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                const ScalarType colorValueK               = colorAccessor[k];
                sharedGaussianColors[threadOrdinal].rgb[k] = colorValueK;
            }
        }
    }

    template <uint32_t WARP_TILE_SIZE>
    inline __device__ void
    volumeRenderTileBackward(const cooperative_groups::thread_block_tile<WARP_TILE_SIZE> &warp,
                             const uint32_t i, const uint32_t j,
                             const int32_t firstGaussianIdInBlock,
                             const int32_t lastGaussianIdInBlock, const uint32_t blockSize) {
        using vec3t = typename Vec3Type<ScalarType>::type;
        using vec2t = typename Vec2Type<ScalarType>::type;

        // (i, j) coordinates are relative to the specified image origin which may be a crop
        // so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = (ScalarType)(j + imageOriginW) + ScalarType{ 0.5 };
        const ScalarType py = (ScalarType)(i + imageOriginH) + ScalarType{ 0.5 };

        // The ordinal of this pixel in its output image (does not account for batch dimension)
        const int32_t pixelOrdinalInImage = min(i * imageWidth + j, imageWidth * imageHeight - 1);

        // Whether this pixel is inside the image bounds.
        // NOTE: We keep threads which correspond to pixels outside the image bounds around
        //       to load gaussians from global memory, but they do not contribute to the output.
        const bool pixelInImage = (i < imageHeight && j < imageWidth);

        extern __shared__ int s[];

        Gaussian *sharedGaussians = reinterpret_cast<Gaussian *>(s); // [blockSize]
        Color    *sharedGaussianColors =
            reinterpret_cast<Color *>(&sharedGaussians[blockSize]);  // [blockSize]

        // this is the T AFTER the last gaussian in this pixel
        const ScalarType finalTransmittance = ScalarType{ 1 } - renderedAlphas[pixelOrdinalInImage];
        ScalarType       transmittance      = finalTransmittance;

        // the contribution from gaussians behind the current one
        ScalarType buffer[COLOR_DIM] = {};

        // Gradient of the loss with respect to the color output of the forward pass at this
        // pixel
        ScalarType dLossDRenderedPixelColor[COLOR_DIM];
#pragma unroll COLOR_DIM
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            dLossDRenderedPixelColor[k] = dLossDRenderedColors[pixelOrdinalInImage * COLOR_DIM + k];
        }
        // Gradient of the loss with respect to the alpha output of the forward pass at this
        // pixel
        const ScalarType dLossDRenderPixelAlpha = dLossDRenderedAlphas[pixelOrdinalInImage];

        // ID of the last Gaussian to contribute to this pixel and the last gaussian id to
        // contribute to any pixel in this block
        const int32_t lastGaussianIdInPixel =
            pixelInImage ? lastGaussianIdsPerPixel[pixelOrdinalInImage] : 0;
        const int32_t lastGaussianIdInWarp = warpMax(lastGaussianIdInPixel, warp);

        // Process Gaussians in batches of size blockSize (i.e. one Gaussian per thread in the
        // block), and batchEnd is the index of the last gaussian.
        const uint32_t threadOrdinal = threadIdx.x * blockDim.y + threadIdx.y;
        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;
        for (uint32_t b = 0; b < numBatches; ++b) {
            // resync all threads before writing next batch of shared mem
            __syncthreads();

            // Each thread fetches one gaussian into shared memory.
            // Gaussians are stored in shared memory locations in order of decreasing distance from
            // the camera. Gaussians are processed in batches of size blockSize (i.e. one Gaussian
            // per thread in the block), and batchEnd is the index of the last gaussian. NOTE: These
            // values can be negative so must be int32 instead of uint32
            const int32_t batchEnd = lastGaussianIdInBlock - 1 - blockSize * b;
            const int32_t idx      = batchEnd - threadOrdinal;
            if (idx >= firstGaussianIdInBlock) {
                fetchGaussianIntoSharedMemory(idx, sharedGaussians, sharedGaussianColors,
                                              threadOrdinal);
            }

            // Sync threads so all gaussians for this batch are loaded in shared memory
            __syncthreads();

            // process gaussians in the current batch for this pixel
            // 0 index is the furthest back gaussian in the batch
            // For each Gaussian which contributes to this pixel, compute this pixel's gradient
            // contribution to that Gaussian
            const int32_t batchSize = min(blockSize, batchEnd + 1 - firstGaussianIdInBlock);
            for (uint32_t t = max(0, batchEnd - lastGaussianIdInWarp); t < batchSize; ++t) {
                constexpr ScalarType ALPHA_THRESHOLD = ScalarType{ 0.999 };

                bool valid = pixelInImage;
                if (batchEnd - t > lastGaussianIdInPixel) {
                    valid = false;
                }

                const vec3t      conic = sharedGaussians[t].conic;
                const vec2t      xy    = sharedGaussians[t].xy;
                const ScalarType opac  = sharedGaussians[t].opacity;
                const vec2t      delta = sharedGaussians[t].delta(px, py);
                const ScalarType sigma = sharedGaussians[t].sigma(delta);
                const ScalarType vis   = __expf(-sigma);
                const ScalarType alpha = min(ALPHA_THRESHOLD, opac * vis);

                valid = valid &&
                        !(sigma < ScalarType{ 0 } || alpha < static_cast<ScalarType>(1.f / 255.f));
                // if there are no active thread in this warp, skip this loop
                if (!warp.any(valid)) {
                    continue;
                }

                // How much each pixel contributes to the gradient of the parameters for this
                // gaussian Initialize to 0 and only set if this pixel is valid
                ScalarType pixelRGBGradientContribution[COLOR_DIM] = { ScalarType{ 0 } };
                vec3t      pixelConicGradientContribution   = { ScalarType{ 0 }, ScalarType{ 0 },
                                                                ScalarType{ 0 } };
                vec2t      pixelMean2dGradientContribution  = { ScalarType{ 0 }, ScalarType{ 0 } };
                vec2t pixelMean2dAbsGradientContribution    = { ScalarType{ 0 }, ScalarType{ 0 } };
                ScalarType pixelOpacityGradientContribution = ScalarType{ 0 };
                if (valid) {
                    // Compute the transmittance for the current gaussian
                    const ScalarType ra = 1.0f / (1.0f - alpha);
                    transmittance *= ra;

                    // Update the contribution of this pixel to the color gradient of the
                    // Gaussian
                    const ScalarType fac = alpha * transmittance;
#pragma unroll COLOR_DIM
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        pixelRGBGradientContribution[k] = fac * dLossDRenderedPixelColor[k];
                    }

                    // Contribution from this pixel to the alpha value for this Gaussian
                    ScalarType pixelAlphaGradientContribution = ScalarType{ 0 };
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        pixelAlphaGradientContribution +=
                            (sharedGaussianColors[t].rgb[k] * transmittance - buffer[k] * ra) *
                            dLossDRenderedPixelColor[k];
                    }
                    pixelAlphaGradientContribution +=
                        finalTransmittance * ra * dLossDRenderPixelAlpha;

                    // Factor in the contribution from the background to this pixel
                    if (backgrounds != nullptr) {
                        ScalarType accum = ScalarType{ 0 };
#pragma unroll COLOR_DIM
                        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                            accum += backgrounds[k] * dLossDRenderedPixelColor[k];
                        }
                        pixelAlphaGradientContribution += -finalTransmittance * ra * accum;
                    }

                    if (opac * vis <= ALPHA_THRESHOLD) {
                        // Contribution from this pixel to sigma for this Gaussian
                        const ScalarType pixelSigmaGradientContribution =
                            -opac * vis * pixelAlphaGradientContribution;
                        pixelConicGradientContribution = {
                            ScalarType{ 0.5 } * pixelSigmaGradientContribution * delta.x * delta.x,
                            pixelSigmaGradientContribution * delta.x * delta.y,
                            ScalarType{ 0.5 } * pixelSigmaGradientContribution * delta.y * delta.y
                        };
                        pixelMean2dGradientContribution = {
                            pixelSigmaGradientContribution *
                                (conic.x * delta.x + conic.y * delta.y),
                            pixelSigmaGradientContribution * (conic.y * delta.x + conic.z * delta.y)
                        };
                        if (outDLossDMeans2dAbs != nullptr) {
                            pixelMean2dAbsGradientContribution = {
                                abs(pixelMean2dGradientContribution.x),
                                abs(pixelMean2dGradientContribution.y)
                            };
                        }
                        pixelOpacityGradientContribution = vis * pixelAlphaGradientContribution;
                    }

#pragma unroll COLOR_DIM
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        buffer[k] += sharedGaussianColors[t].rgb[k] * fac;
                    }
                }

                // Accumulate the gradient contribution to this Gaussian from every pixel in the
                // block
                warpSumMut<COLOR_DIM, decltype(warp), ScalarType>(pixelRGBGradientContribution,
                                                                  warp);
                warpSumMut<decltype(warp), ScalarType>(pixelConicGradientContribution, warp);
                warpSumMut<decltype(warp), ScalarType>(pixelMean2dGradientContribution, warp);
                if (outDLossDMeans2dAbs != nullptr) {
                    warpSumMut<decltype(warp), ScalarType>(pixelMean2dAbsGradientContribution,
                                                           warp);
                }
                warpSumMut<decltype(warp), ScalarType>(pixelOpacityGradientContribution, warp);

                // The first thread in the block accumulates the gradient contribution from the
                // whole block into the global gradient of this Gaussian
                if (warp.thread_rank() == 0) {
                    // Id of this gaussian in the block in [0, C * N] or [0, nnz]
                    const int32_t g = sharedGaussians[t].id;

                    // Accumulate the gradient contribution from this pixel to the global
                    // gradient for the color of this Gaussian
                    ScalarType *dlLossDColorsGaussianPtr = &outDLossDColors[COLOR_DIM * g];
#pragma unroll COLOR_DIM
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        gpuAtomicAdd(dlLossDColorsGaussianPtr + k, pixelRGBGradientContribution[k]);
                    }

                    // Accumulate the gradient contribution from this pixel to the global
                    // gradient for the 3d conic of this Gaussian
                    vec3t *dLossDConicsGaussianPtr = &outDLossDConics[g];
                    gpuAtomicAdd(&dLossDConicsGaussianPtr->x, pixelConicGradientContribution.x);
                    gpuAtomicAdd(&dLossDConicsGaussianPtr->y, pixelConicGradientContribution.y);
                    gpuAtomicAdd(&dLossDConicsGaussianPtr->z, pixelConicGradientContribution.z);

                    // Accumulate the gradient contribution from this pixel to the global
                    // gradient for the 2d mean of this Gaussian
                    vec2t *dLossDMeans2DGaussianPtr = &outDLossDMeans2d[g];
                    gpuAtomicAdd(&dLossDMeans2DGaussianPtr->x, pixelMean2dGradientContribution.x);
                    gpuAtomicAdd(&dLossDMeans2DGaussianPtr->y, pixelMean2dGradientContribution.y);

                    // Accumulate the gradient contribution from this pixel to the global
                    // gradient for the absolute value of the 2d mean of this Gaussian
                    if (outDLossDMeans2dAbs != nullptr) {
                        vec2t *dLossDMeans2dAbsGaussianPtr = &outDLossDMeans2dAbs[g];
                        gpuAtomicAdd(&dLossDMeans2dAbsGaussianPtr->x,
                                     pixelMean2dAbsGradientContribution.x);
                        gpuAtomicAdd(&dLossDMeans2dAbsGaussianPtr->y,
                                     pixelMean2dAbsGradientContribution.y);
                    }

                    // Accumulate the gradient contribution from this pixel to the global
                    // gradient for the opacity of this Gaussian
                    gpuAtomicAdd(&outDLossDOpacities[g], pixelOpacityGradientContribution);
                }
            }
        }
    }

    inline void
    checkInputTensor(const torch::Tensor &x, const std::string &name) {
        TORCH_CHECK(x.is_cuda(), "Input ", name, " must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "Input ", name, " must be contiguous");
    }

    std::size_t
    getSharedMemSize() const {
        return tileSize * tileSize * (sizeof(Gaussian) + sizeof(Color));
    }

    const dim3
    getBlockDim() const {
        return { tileSize, tileSize, 1 };
    }

    const dim3
    getGridDim() const {
        const uint32_t tileExtentW = (imageWidth + tileSize - 1) / tileSize;
        const uint32_t tileExtentH = (imageHeight + tileSize - 1) / tileSize;
        return { numCameras, tileExtentH, tileExtentW };
    }
};

template <typename ScalarType, uint32_t COLOR_DIM, bool IS_PACKED>
__global__ void
rasterizeGaussiansBackward(DeviceArgs<ScalarType, COLOR_DIM, IS_PACKED> args) {
    namespace cg = cooperative_groups;

    auto block = cg::this_thread_block();

    // Advance argument pointers so they start at the image for the current camera
    // This is a bit ugly but it makes it so we can just use tile/pixel coordinates
    // during volume rendering which is cleaner
    const uint32_t cameraId = block.group_index().x; // Which camera are we processing
    args.advancePointersToCamera(cameraId);

    // Ordinal of this tile in the current of image/camera (in row major order).
    // tileOrdinal is in [0, numTilesW * numTilesH]
    const int32_t tileOrdinal = (block.group_index().y + args.tileOriginH) * args.numTilesW +
                                block.group_index().z + args.tileOriginW;

    // If the caller provides a per-tile mask and this tile is masked, do nothing and return
    if (args.masks != nullptr && !args.masks[tileOrdinal]) {
        return;
    }

    // Pixel coordinates of the top left of the current tile.
    // Pixel coordinates run from [0, height] x [0, width]
    const uint32_t i = block.group_index().y * args.tileSize + block.thread_index().y;
    const uint32_t j = block.group_index().z * args.tileSize + block.thread_index().x;

    // Figure out the first and (one past the) last Gaussian ID in this block/tile
    const int32_t firstGaussianIdInBlock = args.tileOffsets[tileOrdinal];
    const int32_t lastGaussianIdInBlock =
        (cameraId == args.numCameras - 1) && (tileOrdinal == args.numTilesW * args.numTilesH - 1)
            ? args.totalIntersections
            : args.tileOffsets[tileOrdinal + 1];

    // Compute the backward pass for the current tile starting at pixel (i, j)
    // and containing Gaussians with ids in [firstGaussianIdInBlock, lastGaussianIdInBlock)
    constexpr uint32_t WARP_TILE_SIZE                = 32; // TODO (fwilliams): Tune this value
    const cg::thread_block_tile<WARP_TILE_SIZE> warp = cg::tiled_partition<WARP_TILE_SIZE>(block);
    args.volumeRenderTileBackward<WARP_TILE_SIZE>(warp, i, j, firstGaussianIdInBlock,
                                                  lastGaussianIdInBlock, block.size());
}

template <typename ScalarType, uint32_t COLOR_DIM, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeGaussiansWithTemplatedColorDim(
    const torch::Tensor               &means2d,     // [C, N, 2] or [nnz, 2]
    const torch::Tensor               &conics,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &colors,      // [C, N, 3] or [nnz, 3]
    const torch::Tensor               &opacities,   // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
    const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
    const uint32_t imageOriginH, const uint32_t tileSize,
    const torch::Tensor &tileOffsets,               // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,           // [totalIntersections]
    const torch::Tensor &renderedAlphas,            // [C, imageHeight, imageWidth, 1]
    const torch::Tensor &lastGaussianIdsPerPixel,   // [C, imageHeight, imageWidth]
    const torch::Tensor &dLossDRenderedColors,      // [C, imageHeight, imageWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,      // [C, imageHeight, imageWidth, 1]
    bool                 absgrad) {
    TORCH_CHECK(tileSize > 0, "Tile size must be greater than 0");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    DeviceArgs<ScalarType, COLOR_DIM, IS_PACKED> deviceArgs(
        means2d, conics, colors, opacities, backgrounds, masks, imageWidth, imageHeight,
        imageOriginW, imageOriginH, tileSize, tileOffsets, tileGaussianIds, renderedAlphas,
        lastGaussianIdsPerPixel, dLossDRenderedColors, dLossDRenderedAlphas);

    torch::Tensor outDLossDMeans2d   = torch::zeros_like(means2d);
    torch::Tensor outDLossDConics    = torch::zeros_like(conics);
    torch::Tensor outDLossDColors    = torch::zeros_like(colors);
    torch::Tensor outDLossDOpacities = torch::zeros_like(opacities);
    torch::Tensor outDLossDMeans2dAbs;
    if (absgrad) {
        outDLossDMeans2dAbs = torch::zeros_like(means2d);
    }

    // Just return empty tensors if there are no gaussians, cameras, or intersections
    if (means2d.numel() == 0 || tileGaussianIds.numel() == 0) {
        return std::make_tuple(outDLossDMeans2dAbs, outDLossDMeans2d, outDLossDConics,
                               outDLossDColors, outDLossDOpacities);
    }

    deviceArgs.setOutputArguments(outDLossDMeans2d, outDLossDConics, outDLossDColors,
                                  outDLossDOpacities, outDLossDMeans2dAbs, absgrad);

    const std::size_t sharedMemSize = deviceArgs.getSharedMemSize();
    const dim3        blockDim      = deviceArgs.getBlockDim();
    const dim3        gridDim       = deviceArgs.getGridDim();

    if (cudaFuncSetAttribute(rasterizeGaussiansBackward<ScalarType, COLOR_DIM, IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMemSize) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", sharedMemSize,
                 " bytes), try lowering tileSize.");
    }
    rasterizeGaussiansBackward<ScalarType, COLOR_DIM, IS_PACKED>
        <<<gridDim, blockDim, sharedMemSize, stream>>>(deviceArgs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(outDLossDMeans2dAbs, outDLossDMeans2d, outDLossDConics, outDLossDColors,
                           outDLossDOpacities);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeBackward<torch::kCUDA>(
    const torch::Tensor &means2d,                 // [C, N, 2]
    const torch::Tensor &conics,                  // [C, N, 3]
    const torch::Tensor &colors,                  // [C, N, 3]
    const torch::Tensor &opacities,               // [N]
    const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
    const uint32_t imageOriginH, const uint32_t tileSize,
    const torch::Tensor &tileOffsets,             // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,         // [totalIntersections]
    const torch::Tensor &renderedAlphas,          // [C, imageHeight, imageWidth, 1]
    const torch::Tensor &lastGaussianIdsPerPixel, // [C, imageHeight, imageWidth]
    const torch::Tensor &dLossDRenderedColors,    // [C, imageHeight, imageWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,    // [C, imageHeight, imageWidth, 1]
    bool                 absgrad) {
    TORCH_CHECK(colors.is_cuda(), "Input colors must be a CUDA tensor");
    TORCH_CHECK(means2d.is_cuda(), "Input means2d must be a CUDA tensor");
    uint32_t   colorDim = colors.size(-1);
    const bool isPacked = means2d.dim() == 2;

#define __GS__CALL_BWD_(N)                                                                       \
    case N: {                                                                                    \
        if (isPacked) {                                                                          \
            return rasterizeGaussiansWithTemplatedColorDim<float, N, true>(                      \
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,                 \
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,      \
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel, \
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad);                            \
        } else {                                                                                 \
            return rasterizeGaussiansWithTemplatedColorDim<float, N, false>(                     \
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,                 \
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,      \
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel, \
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad);                            \
        }                                                                                        \
    }

    switch (colorDim) {
        __GS__CALL_BWD_(1)
        __GS__CALL_BWD_(2)
        __GS__CALL_BWD_(3)
        __GS__CALL_BWD_(4)
        __GS__CALL_BWD_(5)
        __GS__CALL_BWD_(8)
        __GS__CALL_BWD_(9)
        __GS__CALL_BWD_(16)
        __GS__CALL_BWD_(17)
        __GS__CALL_BWD_(32)
        __GS__CALL_BWD_(33)
        __GS__CALL_BWD_(64)
        __GS__CALL_BWD_(65)
        __GS__CALL_BWD_(128)
        __GS__CALL_BWD_(129)
        __GS__CALL_BWD_(256)
        __GS__CALL_BWD_(257)
        __GS__CALL_BWD_(512)
        __GS__CALL_BWD_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", colorDim);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeBackward<torch::kCPU>(
    const torch::Tensor &means2d,                 // [C, N, 2]
    const torch::Tensor &conics,                  // [C, N, 3]
    const torch::Tensor &colors,                  // [C, N, 3]
    const torch::Tensor &opacities,               // [N]
    const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
    const uint32_t imageOriginH, const uint32_t tileSize,
    const torch::Tensor &tileOffsets,             // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,         // [totalIntersections]
    const torch::Tensor &renderedAlphas,          // [C, imageHeight, imageWidth, 1]
    const torch::Tensor &lastGaussianIdsPerPixel, // [C, imageHeight, imageWidth]
    const torch::Tensor &dLossDRenderedColors,    // [C, imageHeight, imageWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,    // [C, imageHeight, imageWidth, 1]
    bool                 absgrad) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
