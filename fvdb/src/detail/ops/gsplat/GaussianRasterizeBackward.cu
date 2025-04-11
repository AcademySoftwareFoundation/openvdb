// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianVectorTypes.cuh"
#include <detail/ops/Ops.h>
#include <detail/utils/cuda/Utils.cuh>

#include <ATen/cuda/Atomic.cuh>

#include <cooperative_groups.h>

namespace fvdb::detail::ops {

template <typename ScalarType> struct alignas(32) Gaussian { // 28 bytes
    using vec2t = typename Vec2Type<ScalarType>::type;
    using vec3t = typename Vec3Type<ScalarType>::type;

    int32_t    id;      // 4 bytes
    vec2t      xy;      // 8 bytes
    ScalarType opacity; // 4 bytes
    vec3t      conic;   // 12 bytes

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

template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
struct DeviceArgs {
    constexpr static size_t NUM_OUTER_DIMS = IS_PACKED ? 1 : 2;
    using vec2t                            = typename Vec2Type<ScalarType>::type;
    using vec3t                            = typename Vec3Type<ScalarType>::type;
    using ColorAccessorType                = fvdb::TorchRAcc64<ScalarType, NUM_OUTER_DIMS + 1>;

    constexpr static bool IS_CHUNKED = (NUM_CHANNELS != NUM_SHARED_CHANNELS);

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
    vec2t *__restrict__ mMeans2d;                     // [C, N, 2] or [nnz, 2]
    vec3t *__restrict__ mConics;                      // [C, N, 3] or [nnz, 3]
    ColorAccessorType mColors;                        // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    ScalarType *__restrict__ mOpacities;              // [C, N] or [nnz]
    ScalarType *__restrict__ mBackgrounds;            // [C, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    bool *__restrict__ mMasks;                        // [C, nTilesH, nTilesW]
    int32_t *__restrict__ mTileOffsets;               // [C, nTilesH, nTilesW]
    int32_t *__restrict__ mTileGaussianIds;           // [totalIntersections]
    ScalarType *__restrict__ mRenderedAlphas;         // [C, imgH, imgW, 1]
    int32_t *__restrict__ mLastGaussianIdsPerPixel;   // [C, imgH, imgW]
    ScalarType *__restrict__ mDLossDRenderedColors;   // [C, imgH, imgW, NUM_CHANNELS]
    ScalarType *__restrict__ mDLossDRenderedAlphas;   // [C, imgH, imgW, 1]
    vec2t *__restrict__ mOutDLossDMeans2dAbs;         // [C, N, 2] or [nnz, 2]
    vec2t *__restrict__ mOutDLossDMeans2d;            // [C, N, 2] or [nnz, 2]
    vec3t *__restrict__ mOutDLossDConics;             // [C, N, 3] or [nnz, 3]
    ScalarType *__restrict__ mOutDLossDColors;        // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    ScalarType *__restrict__ mOutDLossDOpacities;     // [C, N] or [nnz]

    DeviceArgs(
        const torch::Tensor               &means2d,   // [C, N, 2] or [nnz, 2]
        const torch::Tensor               &conics,    // [C, N, 3] or [nnz, 3]
        const torch::Tensor               &colors,    // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const torch::Tensor               &opacities, // [C, N] or [nnz]
        const at::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth, const uint32_t imageHeight, const uint32_t imageOriginW,
        const uint32_t imageOriginH, const uint32_t tileSize,
        const torch::Tensor &tileOffsets,               // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds,           // [totalIntersections]
        const torch::Tensor &renderedAlphas,            // [C, imageHeight, imageWidth, 1]
        const torch::Tensor &lastGaussianIdsPerPixel,   // [C, imageHeight, imageWidth]
        const torch::Tensor &dLossDRenderedColors, // [C, imageHeight, imageWidth, NUM_CHANNELS]
        const torch::Tensor &dLossDRenderedAlphas  // [C, imageHeight, imageWidth, 1]
        )
        : mColors(
              colors
                  .packed_accessor64<ScalarType, NUM_OUTER_DIMS + 1, torch::RestrictPtrTraits>()) {
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

            TORCH_CHECK_VALUE(colors.dim() == 2, "Bad number of dims for colors");
            TORCH_CHECK_VALUE(totalGaussians == colors.size(0), "Bad size for colors");
            TORCH_CHECK_VALUE(NUM_CHANNELS == colors.size(1), "Bad size for colors");

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

            TORCH_CHECK_VALUE(colors.dim() == 3, "Bad number of dims for colors");
            TORCH_CHECK_VALUE(numCameras == colors.size(0), "Bad size for colors");
            TORCH_CHECK_VALUE(numGaussiansPerCamera == colors.size(1), "Bad size for colors");
            TORCH_CHECK_VALUE(NUM_CHANNELS == colors.size(2), "Bad size for colors");

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

        TORCH_CHECK_VALUE(renderedAlphas.dim() == 4, "Bad number of dims for renderedAlphas");
        TORCH_CHECK_VALUE(numCameras == renderedAlphas.size(0), "Bad size for renderedAlphas");
        TORCH_CHECK_VALUE(imageHeight == renderedAlphas.size(1), "Bad size for renderedAlphas");
        TORCH_CHECK_VALUE(imageWidth == renderedAlphas.size(2), "Bad size for renderedAlphas");
        TORCH_CHECK_VALUE(1 == renderedAlphas.size(3), "Bad size for renderedAlphas");

        TORCH_CHECK_VALUE(numCameras == lastGaussianIdsPerPixel.size(0),
                          "Bad size for lastGaussianIdsPerPixel");
        TORCH_CHECK_VALUE(imageHeight == lastGaussianIdsPerPixel.size(1),
                          "Bad size for lastGaussianIdsPerPixel");
        TORCH_CHECK_VALUE(imageWidth == lastGaussianIdsPerPixel.size(2),
                          "Bad size for lastGaussianIdsPerPixel");

        TORCH_CHECK_VALUE(dLossDRenderedColors.dim() == 4,
                          "Bad number of dims for dLossDRenderedColors");
        TORCH_CHECK_VALUE(numCameras == dLossDRenderedColors.size(0),
                          "Bad size for dLossDRenderedColors");
        TORCH_CHECK_VALUE(imageHeight == dLossDRenderedColors.size(1),
                          "Bad size for dLossDRenderedColors");
        TORCH_CHECK_VALUE(imageWidth == dLossDRenderedColors.size(2),
                          "Bad size for dLossDRenderedColors");
        TORCH_CHECK_VALUE(NUM_CHANNELS == dLossDRenderedColors.size(3),
                          "Bad size for dLossDRenderedColors");

        TORCH_CHECK_VALUE(dLossDRenderedAlphas.dim() == 4,
                          "Bad number of dims for dLossDRenderedAlphas");
        TORCH_CHECK_VALUE(numCameras == dLossDRenderedAlphas.size(0),
                          "Bad size for dLossDRenderedAlphas");
        TORCH_CHECK_VALUE(imageHeight == dLossDRenderedAlphas.size(1),
                          "Bad size for dLossDRenderedAlphas");
        TORCH_CHECK_VALUE(imageWidth == dLossDRenderedAlphas.size(2),
                          "Bad size for dLossDRenderedAlphas");
        TORCH_CHECK_VALUE(1 == dLossDRenderedAlphas.size(3), "Bad size for dLossDRenderedAlphas");

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
        mMasks                   = masks.has_value() ? masks.value().data_ptr<bool>() : nullptr;
        mTileOffsets             = tileOffsets.data_ptr<int32_t>();
        mTileGaussianIds         = tileGaussianIds.data_ptr<int32_t>();
        mRenderedAlphas          = renderedAlphas.data_ptr<ScalarType>();
        mLastGaussianIdsPerPixel = lastGaussianIdsPerPixel.data_ptr<int32_t>();
        mDLossDRenderedColors    = dLossDRenderedColors.data_ptr<ScalarType>();
        mDLossDRenderedAlphas    = dLossDRenderedAlphas.data_ptr<ScalarType>();
    }

    void
    setOutputArguments(const torch::Tensor &outDLossDMeans2d, const torch::Tensor &outDLossDConics,
                       const torch::Tensor &outDLossDColors,
                       const torch::Tensor &outDLossDOpacities,
                       const torch::Tensor &outDLossDMeans2dAbs, const bool absgrad) {
        mOutDLossDMeans2dAbs =
            absgrad ? reinterpret_cast<vec2t *>(outDLossDMeans2dAbs.data_ptr<ScalarType>())
                    : nullptr;
        mOutDLossDMeans2d   = reinterpret_cast<vec2t *>(outDLossDMeans2d.data_ptr<ScalarType>());
        mOutDLossDConics    = reinterpret_cast<vec3t *>(outDLossDConics.data_ptr<ScalarType>());
        mOutDLossDColors    = outDLossDColors.data_ptr<ScalarType>();
        mOutDLossDOpacities = outDLossDOpacities.data_ptr<ScalarType>();
    }

    inline __device__ void
    advancePointersToCamera(const uint32_t cameraId) {
        // Move all the pointers forward to the current camera
        const std::ptrdiff_t offsetForPixels = cameraId * mImageHeight * mImageWidth;
        const std::ptrdiff_t offsetForTiles  = cameraId * mNumTilesH * mNumTilesW;

        mTileOffsets += offsetForTiles;
        mRenderedAlphas += offsetForPixels;
        mLastGaussianIdsPerPixel += offsetForPixels;
        mDLossDRenderedColors += offsetForPixels * NUM_CHANNELS;
        mDLossDRenderedAlphas += offsetForPixels;
        if (mBackgrounds != nullptr) {
            mBackgrounds += cameraId * NUM_CHANNELS;
        }
        if (mMasks != nullptr) {
            mMasks += offsetForTiles;
        }
    }

    inline __device__ void
    fetchGaussianIntoSharedMemory(const int32_t g, Gaussian<ScalarType> *outGaussian) const {
        const vec2t      xy    = mMeans2d[g];
        const ScalarType opac  = mOpacities[g];
        const vec3t      conic = mConics[g];
        *outGaussian           = { g, xy, opac, conic };
    }

    inline __device__ void
    fetchGaussianColorIntoSharedMemory(const int32_t g, const size_t channelStart,
                                       const size_t numChannels, ScalarType *outColor) {
        if constexpr (IS_PACKED) {
            const auto colorAccessor = mColors[g];
            for (uint32_t k = 0; k < numChannels; ++k) {
                outColor[k] = colorAccessor[k + channelStart];
            }
        } else {
            // colors: [C, N, NUM_CHANNELS]
            // colors[c, n, k] = [c * N * NUM_CHANNELS + n * NUM_CHANNELS + k]
            // g = c * N + n
            const int32_t cid           = g / mNumGaussiansPerCamera;
            const int32_t gid           = g % mNumGaussiansPerCamera;
            const auto    colorAccessor = mColors[cid][gid];
            if constexpr (IS_CHUNKED) {
                for (auto k = 0; k < numChannels; ++k) {
                    outColor[k] = colorAccessor[k + channelStart];
                }
            } else {
#pragma unroll NUM_CHANNELS
                for (auto k = 0; k < NUM_CHANNELS; ++k) {
                    outColor[k] = colorAccessor[k];
                }
            }
        }
    }

    inline __device__ void
    atomicAddRGBGradientContributions(const int32_t     g,
                                      const ScalarType *pixelRGBGradientContribution,
                                      const size_t channelStart, const size_t numChannels) {
        // Accumulate the gradient contribution from this pixel to the global
        // gradient for the color of this Gaussian
        ScalarType *dlLossDColorsGaussianPtr = &mOutDLossDColors[NUM_CHANNELS * g];
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                gpuAtomicAdd(dlLossDColorsGaussianPtr + channelStart + k,
                             pixelRGBGradientContribution[k]);
            }
        } else {
#pragma unroll NUM_CHANNELS
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                gpuAtomicAdd(dlLossDColorsGaussianPtr + k, pixelRGBGradientContribution[k]);
            }
        }
    }

    inline __device__ void
    atomicAddMeans2dConicsAndOpacitiesGradientContributions(
        const int32_t g, const vec3t &pixelConicGradientContribution,
        const vec2t     &pixelMean2dGradientContribution,
        const vec2t     &pixelMean2dAbsGradientContribution,
        const ScalarType pixelOpacityGradientContribution) const {
        // Accumulate the gradient contribution from this pixel to the global
        // gradient for the 3d conic of this Gaussian
        vec3t *dLossDConicsGaussianPtr = &mOutDLossDConics[g];
        gpuAtomicAdd(&dLossDConicsGaussianPtr->x, pixelConicGradientContribution.x);
        gpuAtomicAdd(&dLossDConicsGaussianPtr->y, pixelConicGradientContribution.y);
        gpuAtomicAdd(&dLossDConicsGaussianPtr->z, pixelConicGradientContribution.z);

        // Accumulate the gradient contribution from this pixel to the global
        // gradient for the 2d mean of this Gaussian
        vec2t *dLossDMeans2DGaussianPtr = &mOutDLossDMeans2d[g];
        gpuAtomicAdd(&dLossDMeans2DGaussianPtr->x, pixelMean2dGradientContribution.x);
        gpuAtomicAdd(&dLossDMeans2DGaussianPtr->y, pixelMean2dGradientContribution.y);

        // Accumulate the gradient contribution from this pixel to the global
        // gradient for the absolute value of the 2d mean of this Gaussian
        if (mOutDLossDMeans2dAbs != nullptr) {
            vec2t *dLossDMeans2dAbsGaussianPtr = &mOutDLossDMeans2dAbs[g];
            gpuAtomicAdd(&dLossDMeans2dAbsGaussianPtr->x, pixelMean2dAbsGradientContribution.x);
            gpuAtomicAdd(&dLossDMeans2dAbsGaussianPtr->y, pixelMean2dAbsGradientContribution.y);
        }

        // Accumulate the gradient contribution from this pixel to the global
        // gradient for the opacity of this Gaussian
        gpuAtomicAdd(&mOutDLossDOpacities[g], pixelOpacityGradientContribution);
    }

    inline __device__ void
    accumulateColorStep(const ScalarType *gaussianColor, const ScalarType fac,
                        const size_t numChannels, ScalarType *outAccumColor) const {
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                outAccumColor[k] += gaussianColor[k] * fac;
            }
        } else {
#pragma unroll NUM_CHANNELS
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                outAccumColor[k] += gaussianColor[k] * fac;
            }
        }
    }

    inline __device__ void
    calculatePixelRGBGradientContribution(const ScalarType  fac,
                                          const ScalarType *dLossDRenderedPixelColor,
                                          ScalarType       *outPixelRGBGradientContribution) const {
#pragma unroll NUM_SHARED_CHANNELS
        for (uint32_t k = 0; k < NUM_SHARED_CHANNELS; ++k) {
            outPixelRGBGradientContribution[k] = fac * dLossDRenderedPixelColor[k];
        }
    }

    inline __device__ ScalarType
    calculatePixelAlphaGradientContribution(
        const ScalarType finalTransmittance, const ScalarType oneOverOneMinusAlpha,
        const ScalarType accumTransmittance, const ScalarType *accumColor,
        const ScalarType *gaussianColor, const ScalarType *dLossDRenderedPixelColor,
        const ScalarType dLossDRenderPixelAlpha, const size_t numChannels,
        const bool includeLastTerm) const {
        ScalarType pixelAlphaGradientContribution = ScalarType{ 0 };
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                pixelAlphaGradientContribution +=
                    (gaussianColor[k] * accumTransmittance - accumColor[k] * oneOverOneMinusAlpha) *
                    dLossDRenderedPixelColor[k];
            }
        } else {
#pragma unroll NUM_CHANNELS
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                pixelAlphaGradientContribution +=
                    (gaussianColor[k] * accumTransmittance - accumColor[k] * oneOverOneMinusAlpha) *
                    dLossDRenderedPixelColor[k];
            }
        }

        if (includeLastTerm) {
            pixelAlphaGradientContribution +=
                finalTransmittance * oneOverOneMinusAlpha * dLossDRenderPixelAlpha;
        }

        // Factor in the contribution from the background to this pixel
        if (mBackgrounds != nullptr) {
            ScalarType accum = ScalarType{ 0 };
            if constexpr (IS_CHUNKED) {
                for (uint32_t k = 0; k < numChannels; ++k) {
                    accum += mBackgrounds[k] * dLossDRenderedPixelColor[k];
                }
            } else {
#pragma unroll NUM_CHANNELS
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    accum += mBackgrounds[k] * dLossDRenderedPixelColor[k];
                }
            }
            if (includeLastTerm) {
                pixelAlphaGradientContribution +=
                    -finalTransmittance * oneOverOneMinusAlpha * accum;
            }
        }

        return pixelAlphaGradientContribution;
    }

    inline __device__ void
    calculateMeansConicsAndOpacitiesPixelGradientContribution(
        ScalarType opac, ScalarType vis, ScalarType pixelAlphaGradientContribution,
        const vec3t &conic, const vec2t &delta, vec3t &outPixelConicGradientContribution,
        vec2t &outPixelMean2dGradientContribution, vec2t &outPixelMean2dAbsGradientContribution,
        ScalarType &outPixelOpacityGradientContribution) const {
        // Contribution from this pixel to sigma for this Gaussian
        const ScalarType pixelSigmaGradientContribution =
            -opac * vis * pixelAlphaGradientContribution;
        outPixelConicGradientContribution = {
            ScalarType{ 0.5 } * pixelSigmaGradientContribution * delta.x * delta.x,
            pixelSigmaGradientContribution * delta.x * delta.y,
            ScalarType{ 0.5 } * pixelSigmaGradientContribution * delta.y * delta.y
        };
        outPixelMean2dGradientContribution = {
            pixelSigmaGradientContribution * (conic.x * delta.x + conic.y * delta.y),
            pixelSigmaGradientContribution * (conic.y * delta.x + conic.z * delta.y)
        };
        if (mOutDLossDMeans2dAbs != nullptr) {
            outPixelMean2dAbsGradientContribution = { abs(outPixelMean2dGradientContribution.x),
                                                      abs(outPixelMean2dGradientContribution.y) };
        }
        outPixelOpacityGradientContribution = vis * pixelAlphaGradientContribution;
    }

    inline __device__ bool
    calculateGradientContributions(
        const Gaussian<ScalarType> &gaussian, const ScalarType *gaussianColor,
        const ScalarType *dLossDRenderedPixelColor, const ScalarType dLossDRenderPixelAlpha,
        const ScalarType px, const ScalarType py, const ScalarType finalTransmittance,
        const size_t numChannels, const bool calculateMeansConicsAndOpacitiesGradient,
        ScalarType &accumTransmittance, ScalarType *accumColor,
        ScalarType *outPixelRGBGradientContribution, vec3t &outPixelConicGradientContribution,
        vec2t &outPixelMean2dGradientContribution, vec2t &outPixelMean2dAbsGradientContribution,
        ScalarType &outPixelOpacityGradientContribution) const {
        constexpr ScalarType ALPHA_THRESHOLD = ScalarType{ 0.999 };

        const vec3t      conic = gaussian.conic;
        const ScalarType opac  = gaussian.opacity;
        const vec2t      delta = gaussian.delta(px, py);
        const ScalarType sigma = gaussian.sigma(delta);
        const ScalarType vis   = __expf(-sigma);
        const ScalarType alpha = min(ALPHA_THRESHOLD, opac * vis);

        const bool gaussianIsValid =
            !(sigma < ScalarType{ 0 } || alpha < static_cast<ScalarType>(1.f / 255.f));
        // if there are no active thread in this warp, skip this loop
        if (!gaussianIsValid) {
            return false;
        }

        // Compute the transmittance for the current gaussian
        const ScalarType oneOverOneMinusAlpha = ScalarType{ 1 } / (ScalarType{ 1 } - alpha);
        accumTransmittance *= oneOverOneMinusAlpha;

        // Update the contribution of this pixel to the color gradient of the
        // Gaussian
        const ScalarType fac = alpha * accumTransmittance;
        calculatePixelRGBGradientContribution(fac, dLossDRenderedPixelColor,
                                              outPixelRGBGradientContribution);

        // Contribution from this pixel to the alpha value for this Gaussian
        const ScalarType pixelAlphaGradientContribution = calculatePixelAlphaGradientContribution(
            finalTransmittance, oneOverOneMinusAlpha, accumTransmittance, accumColor, gaussianColor,
            dLossDRenderedPixelColor, dLossDRenderPixelAlpha, numChannels,
            calculateMeansConicsAndOpacitiesGradient);

        if (opac * vis <= ALPHA_THRESHOLD) {
            calculateMeansConicsAndOpacitiesPixelGradientContribution(
                opac, vis, pixelAlphaGradientContribution, conic, delta,
                outPixelConicGradientContribution, outPixelMean2dGradientContribution,
                outPixelMean2dAbsGradientContribution, outPixelOpacityGradientContribution);
        }

        accumulateColorStep(gaussianColor, fac, numChannels, accumColor);

        return true;
    }

    template <size_t WARP_TILE_SIZE>
    inline __device__ void
    volumeRenderTileBackward(const cooperative_groups::thread_block_tile<WARP_TILE_SIZE> &warp,
                             const uint32_t i, const uint32_t j,
                             const int32_t firstGaussianIdInBlock,
                             const int32_t lastGaussianIdInBlock, const uint32_t blockSize) {
        // (i, j) coordinates are relative to the specified image origin which may be a crop
        // so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = (ScalarType)(j + mImageOriginW) + ScalarType{ 0.5 };
        const ScalarType py = (ScalarType)(i + mImageOriginH) + ScalarType{ 0.5 };

        // The ordinal of this pixel in its output image (does not account for batch dimension)
        const int32_t pixelOrdinalInImage =
            min(i * mImageWidth + j, mImageWidth * mImageHeight - 1);

        // Whether this pixel is inside the image bounds.
        // NOTE: We keep threads which correspond to pixels outside the image bounds around
        //       to load gaussians from global memory, but they do not contribute to the output.
        const bool pixelInImage = (i < mImageHeight && j < mImageWidth);

        extern __shared__ int s[];

        Gaussian<ScalarType> *sharedGaussians =
            reinterpret_cast<Gaussian<ScalarType> *>(s);                 // [blockSize]
        ScalarType *sharedGaussianColors =
            reinterpret_cast<ScalarType *>(&sharedGaussians[blockSize]); // [blockSize]

        // this is the T AFTER the last gaussian in this pixel
        const ScalarType finalTransmittance =
            ScalarType{ 1 } - mRenderedAlphas[pixelOrdinalInImage];

        // Gradient of the loss with respect to the alpha output of the forward pass at this
        // pixel
        const ScalarType dLossDRenderPixelAlpha = mDLossDRenderedAlphas[pixelOrdinalInImage];

        // ID of the last Gaussian to contribute to this pixel and the last gaussian id to
        // contribute to any pixel in this block
        const int32_t lastGaussianIdInPixel =
            pixelInImage ? mLastGaussianIdsPerPixel[pixelOrdinalInImage] : 0;
        const int32_t lastGaussianIdInWarp = warpMax(lastGaussianIdInPixel, warp);

        // Process Gaussians in batches of size blockSize (i.e. one Gaussian per thread in the
        // block), and batchEnd is the index of the last gaussian.
        const uint32_t threadOrdinal = threadIdx.x * blockDim.y + threadIdx.y;
        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        constexpr size_t NUM_CHUNKS =
            (NUM_CHANNELS + NUM_SHARED_CHANNELS - 1) / NUM_SHARED_CHANNELS;
        for (size_t chunk = 0; chunk < NUM_CHUNKS; chunk += 1) {
            const size_t channelStart = chunk * NUM_SHARED_CHANNELS;
            const size_t numChannels  = min(NUM_CHANNELS - channelStart, NUM_SHARED_CHANNELS);
            const bool   isLastChunk  = chunk == (NUM_CHUNKS - 1);

            ScalarType accumTransmittance = finalTransmittance;

            // the contribution from gaussians behind the current one
            ScalarType accumColor[NUM_SHARED_CHANNELS] = { ScalarType(0) };

            // Gradient of the loss with respect to the color output of the forward pass at this
            // pixel
            ScalarType dLossDRenderedPixelColor[NUM_SHARED_CHANNELS];
            for (auto k = 0; k < numChannels; ++k) {
                dLossDRenderedPixelColor[k] =
                    mDLossDRenderedColors[pixelOrdinalInImage * NUM_CHANNELS + channelStart + k];
            }
            for (auto k = numChannels; k < NUM_SHARED_CHANNELS; ++k) {
                dLossDRenderedPixelColor[k] = ScalarType{ 0 };
            }

            for (uint32_t b = 0; b < numBatches; ++b) {
                // resync all threads before writing next batch of shared mem
                __syncthreads();

                // Each thread fetches one gaussian into shared memory.
                // Gaussians are stored in shared memory locations in order of decreasing
                // distance from the camera. Gaussians are processed in batches of size
                // blockSize (i.e. one Gaussian per thread in the block), and batchEnd is the
                // index of the last gaussian. NOTE: These values can be negative so must be
                // int32 instead of uint32
                const int32_t batchEnd = lastGaussianIdInBlock - 1 - blockSize * b;
                const int32_t idx      = batchEnd - threadOrdinal;
                if (idx >= firstGaussianIdInBlock) {
                    const int32_t g = mTileGaussianIds[idx]; // Gaussian index in [C * N] or [nnz]
                    Gaussian<ScalarType> *gaussian = &sharedGaussians[threadOrdinal];
                    fetchGaussianIntoSharedMemory(g, gaussian);
                    ScalarType *color = &sharedGaussianColors[threadOrdinal * NUM_SHARED_CHANNELS];
                    fetchGaussianColorIntoSharedMemory(g, channelStart, numChannels, color);
                }

                // Sync threads so all gaussians for this batch are loaded in shared memory
                __syncthreads();

                // process gaussians in the current batch for this pixel
                // 0 index is the furthest back gaussian in the batch
                // For each Gaussian which contributes to this pixel, compute this pixel's
                // gradient contribution to that Gaussian
                const int32_t batchSize = min(blockSize, batchEnd + 1 - firstGaussianIdInBlock);
                for (uint32_t t = max(0, batchEnd - lastGaussianIdInWarp); t < batchSize; ++t) {
                    bool valid = pixelInImage;
                    if (batchEnd - t > lastGaussianIdInPixel) {
                        valid = false;
                    }
                    // How much each pixel contributes to the gradient of the parameters for
                    // this gaussian Initialize to 0 and only set if this pixel is valid
                    ScalarType pixelRGBGradientContribution[NUM_SHARED_CHANNELS] = { ScalarType{
                        0 } };
                    vec3t      pixelConicGradientContribution = { ScalarType{ 0 }, ScalarType{ 0 },
                                                                  ScalarType{ 0 } };
                    vec2t pixelMean2dGradientContribution    = { ScalarType{ 0 }, ScalarType{ 0 } };
                    vec2t pixelMean2dAbsGradientContribution = { ScalarType{ 0 }, ScalarType{ 0 } };
                    ScalarType pixelOpacityGradientContribution = ScalarType{ 0 };

                    valid =
                        valid &&
                        calculateGradientContributions(
                            sharedGaussians[t], &sharedGaussianColors[t * NUM_SHARED_CHANNELS],
                            dLossDRenderedPixelColor, dLossDRenderPixelAlpha, px, py,
                            finalTransmittance, numChannels, isLastChunk, accumTransmittance,
                            accumColor, pixelRGBGradientContribution,
                            pixelConicGradientContribution, pixelMean2dGradientContribution,
                            pixelMean2dAbsGradientContribution, pixelOpacityGradientContribution);

                    // if there are no active thread in this warp, skip this loop
                    if (!warp.any(valid)) {
                        continue;
                    }

                    // Accumulate the gradient contribution to this Gaussian from every
                    // pixel in the block
                    if constexpr (IS_CHUNKED) {
                        warpSumMut<decltype(warp), ScalarType>(pixelRGBGradientContribution,
                                                               numChannels, warp);
                    } else {
                        warpSumMut<NUM_SHARED_CHANNELS, decltype(warp), ScalarType>(
                            pixelRGBGradientContribution, warp);
                    }

                    warpSumMut<decltype(warp), ScalarType>(pixelConicGradientContribution, warp);
                    warpSumMut<decltype(warp), ScalarType>(pixelMean2dGradientContribution, warp);
                    if (mOutDLossDMeans2dAbs != nullptr) {
                        warpSumMut<decltype(warp), ScalarType>(pixelMean2dAbsGradientContribution,
                                                               warp);
                    }
                    warpSumMut<decltype(warp), ScalarType>(pixelOpacityGradientContribution, warp);

                    // The first thread in the block accumulates the gradient
                    // contribution from the whole block into the global gradient of
                    // this Gaussian
                    if (warp.thread_rank() == 0) {
                        atomicAddRGBGradientContributions(sharedGaussians[t].id,
                                                          pixelRGBGradientContribution,
                                                          channelStart, numChannels);
                        atomicAddMeans2dConicsAndOpacitiesGradientContributions(
                            sharedGaussians[t].id, pixelConicGradientContribution,
                            pixelMean2dGradientContribution, pixelMean2dAbsGradientContribution,
                            pixelOpacityGradientContribution);
                    }
                }
            }
        }
    }

    inline void
    checkInputTensor(const torch::Tensor &x, const std::string &name) {
        TORCH_CHECK(x.is_cuda(), "Input ", name, " must be a CUDA tensor");
        TORCH_CHECK(x.is_contiguous(), "Input ", name, " must be contiguous");
    }

    const dim3
    getBlockDim() const {
        return { mTileSize, mTileSize, 1 };
    }

    const dim3
    getGridDim() const {
        const uint32_t tileExtentW = (mImageWidth + mTileSize - 1) / mTileSize;
        const uint32_t tileExtentH = (mImageHeight + mTileSize - 1) / mTileSize;
        return { mNumCameras, tileExtentH, tileExtentW };
    }
};

template <typename ScalarType>
size_t
getSharedMemRequirements(const size_t numColorChannels, const size_t tileSize) {
    // typedef Gaussian<ScalarType> GaussianT;
    return tileSize * tileSize * (32 + numColorChannels * sizeof(ScalarType));
}

template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansBackward(
    DeviceArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED> args) {
    namespace cg = cooperative_groups;

    auto block = cg::this_thread_block();

    // Advance argument pointers so they start at the image for the current camera
    // This is a bit ugly but it makes it so we can just use tile/pixel coordinates
    // during volume rendering which is cleaner
    const uint32_t cameraId = block.group_index().x; // Which camera are we processing
    args.advancePointersToCamera(cameraId);

    // Ordinal of this tile in the current of image/camera (in row major order).
    // tileOrdinal is in [0, numTilesW * numTilesH]
    const int32_t tileOrdinal = (block.group_index().y + args.mTileOriginH) * args.mNumTilesW +
                                block.group_index().z + args.mTileOriginW;

    // If the caller provides a per-tile mask and this tile is masked, do nothing and return
    if (args.mMasks != nullptr && !args.mMasks[tileOrdinal]) {
        return;
    }

    // Pixel coordinates of the top left of the current tile.
    // Pixel coordinates run from [0, height] x [0, width]
    const uint32_t i = block.group_index().y * args.mTileSize + block.thread_index().y;
    const uint32_t j = block.group_index().z * args.mTileSize + block.thread_index().x;

    // Figure out the first and (one past the) last Gaussian ID in this block/tile
    const int32_t firstGaussianIdInBlock = args.mTileOffsets[tileOrdinal];
    const int32_t lastGaussianIdInBlock =
        (cameraId == args.mNumCameras - 1) && (tileOrdinal == args.mNumTilesW * args.mNumTilesH - 1)
            ? args.mTotalIntersections
            : args.mTileOffsets[tileOrdinal + 1];

    // Compute the backward pass for the current tile starting at pixel (i, j)
    // and containing Gaussians with ids in [firstGaussianIdInBlock, lastGaussianIdInBlock)
    constexpr uint32_t WARP_TILE_SIZE                = 32; // TODO (fwilliams): Tune this value
    const cg::thread_block_tile<WARP_TILE_SIZE> warp = cg::tiled_partition<WARP_TILE_SIZE>(block);

    args.volumeRenderTileBackward<WARP_TILE_SIZE>(warp, i, j, firstGaussianIdInBlock,
                                                  lastGaussianIdInBlock, block.size());
}

template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
callRasterizeBackwardsWithTemplatedSharedChannels(
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
    bool absgrad, at::cuda::CUDAStream stream) {
    TORCH_CHECK(tileSize > 0, "Tile size must be greater than 0");

    DeviceArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED> deviceArgs(
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

    const size_t numChannels =
        (NUM_SHARED_CHANNELS == NUM_CHANNELS) ? NUM_CHANNELS : NUM_SHARED_CHANNELS + 1;
    const size_t sharedMemSize = getSharedMemRequirements<ScalarType>(numChannels, tileSize);
    const dim3   blockDim      = deviceArgs.getBlockDim();
    const dim3   gridDim       = deviceArgs.getGridDim();

    if (cudaFuncSetAttribute(
            rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", sharedMemSize,
                 " bytes), try lowering tileSize.");
    }
    rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED>
        <<<gridDim, blockDim, sharedMemSize, stream>>>(deviceArgs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(outDLossDMeans2dAbs, outDLossDMeans2d, outDLossDConics, outDLossDColors,
                           outDLossDOpacities);
}

template <typename ScalarType, size_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
callRasterizeBackwardsWithCorrectSharedChannels(
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
    const bool absgrad, const int64_t numSharedSharedChannelsOverride) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    auto callWithSharedChannels = [&](size_t numSharedChannels) {
        if (numSharedChannels == NUM_CHANNELS) {
            return callRasterizeBackwardsWithTemplatedSharedChannels<ScalarType, NUM_CHANNELS,
                                                                     NUM_CHANNELS, IS_PACKED>(
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad, stream);
        } else if (numSharedChannels == 64) {
            return callRasterizeBackwardsWithTemplatedSharedChannels<ScalarType, NUM_CHANNELS, 64,
                                                                     IS_PACKED>(
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad, stream);
        } else if (numSharedChannels == 32) {
            return callRasterizeBackwardsWithTemplatedSharedChannels<ScalarType, NUM_CHANNELS, 32,
                                                                     IS_PACKED>(
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad, stream);
        } else if (numSharedChannels == 16) {
            return callRasterizeBackwardsWithTemplatedSharedChannels<ScalarType, NUM_CHANNELS, 16,
                                                                     IS_PACKED>(
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel,
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad, stream);
        } else {
            if (numSharedSharedChannelsOverride > 0) {
                AT_ERROR("Invalid numSharedChannelsOverride. Must be 64, 32, or 16.");
            } else {
                AT_ERROR("Failed to set maximum shared memory size");
            }
        }
    };

    if (numSharedSharedChannelsOverride > 0) {
        return callWithSharedChannels(numSharedSharedChannelsOverride);
    } else {
        cudaDeviceProp deviceProperties;
        if (cudaGetDeviceProperties(&deviceProperties, stream.device_index()) != cudaSuccess) {
            AT_ERROR("Failed to query device properties");
        }
        const size_t maxSharedMemory = deviceProperties.sharedMemPerBlockOptin;

        const size_t sharedMemChannelOptions[4] = { NUM_CHANNELS, 64, 32, 16 };
        for (size_t i = 0; i < 4; ++i) {
            const size_t numSharedChannels = sharedMemChannelOptions[i];
            if (getSharedMemRequirements<ScalarType>(numSharedChannels, tileSize) <=
                maxSharedMemory) {
                return callWithSharedChannels(numSharedChannels);
            }
        }
        AT_ERROR("Failed to set maximum shared memory size");
    }
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
    const bool absgrad, const int64_t numSharedSharedChannelsOverride) {
    TORCH_CHECK(colors.is_cuda(), "Input colors must be a CUDA tensor");
    TORCH_CHECK(means2d.is_cuda(), "Input means2d must be a CUDA tensor");
    uint32_t   colorDim = colors.size(-1);
    const bool isPacked = means2d.dim() == 2;

#define __GS__CALL_BWD_(N)                                                                       \
    case N: {                                                                                    \
        if (isPacked) {                                                                          \
            return callRasterizeBackwardsWithCorrectSharedChannels<float, N, true>(              \
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,                 \
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,      \
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel, \
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad,                             \
                numSharedSharedChannelsOverride);                                                \
        } else {                                                                                 \
            return callRasterizeBackwardsWithCorrectSharedChannels<float, N, false>(             \
                means2d, conics, colors, opacities, at::nullopt /*backgrounds*/,                 \
                at::nullopt /*masks*/, imageWidth, imageHeight, imageOriginW, imageOriginH,      \
                tileSize, tileOffsets, tileGaussianIds, renderedAlphas, lastGaussianIdsPerPixel, \
                dLossDRenderedColors, dLossDRenderedAlphas, absgrad,                             \
                numSharedSharedChannelsOverride);                                                \
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
        __GS__CALL_BWD_(47)
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
    const bool absgrad, const int64_t numSharedSharedChannelsOverride) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace fvdb::detail::ops
