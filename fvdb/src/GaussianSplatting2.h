// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_GAUSSIANSPLATTING2_H
#define FVDB_GAUSSIANSPLATTING2_H

#include <detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/all.h>

namespace fvdb {

class GaussianSplat3d {
  public:
    using ProjectionType = fvdb::detail::ops::ProjectionType;

    struct RenderState {
        torch::Tensor perGaussian2dMean;         // [C, N, 2]
        torch::Tensor perGaussianConic;          // [C, N, 3]
        torch::Tensor perGaussianRenderQuantity; // [C, N, 3]
        torch::Tensor perGaussianDepth;          // [C, N, 1]
        torch::Tensor perGaussianOpacity;        // [N]
        torch::Tensor perGaussianRadius;         // [C, N]
        torch::Tensor tileOffsets;               // [C, num_tiles_h, num_tiles_w, 2]
        torch::Tensor tileGaussianIds; // [C, num_tiles_h, num_tiles_w, max_gaussians_per_tile]

        fvdb::detail::ops::RenderSettings mRenderSettings;

        ssize_t
        imageHeight() const {
            return mRenderSettings.imageHeight;
        }

        ssize_t
        imageWidth() const {
            return mRenderSettings.imageWidth;
        }

        float
        nearPlane() const {
            return mRenderSettings.nearPlane;
        }

        float
        farPlane() const {
            return mRenderSettings.farPlane;
        }

        ProjectionType
        projectionType() const {
            return mRenderSettings.projectionType;
        }

        int64_t
        shDegreeToUse() const {
            return mRenderSettings.shDegreeToUse;
        }

        float
        minRadius2d() const {
            return mRenderSettings.radiusClip;
        }

        float
        eps2d() const {
            return mRenderSettings.eps2d;
        }

        bool
        antialias() const {
            return mRenderSettings.antialias;
        }

        torch::Tensor
        means2d() const {
            return perGaussian2dMean;
        }

        torch::Tensor
        conics() const {
            return perGaussianConic;
        }

        torch::Tensor
        renderQuantities() const {
            return perGaussianRenderQuantity;
        }

        torch::Tensor
        depths() const {
            return perGaussianDepth;
        }

        torch::Tensor
        opacities() const {
            return perGaussianOpacity;
        }

        torch::Tensor
        radii() const {
            return perGaussianRadius;
        }

        torch::Tensor
        offsets() const {
            return tileOffsets;
        }

        torch::Tensor
        gaussianIds() const {
            return tileGaussianIds;
        }
    };

  private:
    torch::Tensor mMeans;     // [N, 3]
    torch::Tensor mQuats;     // [N, 4]
    torch::Tensor mScales;    // [N, 3]
    torch::Tensor mOpacities; // [N]
    torch::Tensor mShCoeffs;  // [N, K, 3]

    // Used for subdivision during optimization
    torch::Tensor mNormalizedMean2dGradients;
    // If you require grad, we store the radii for the backward pass
    torch::Tensor mPerGaussianRadiiForGrad;
    bool          mRequiresGrad = false;

    RenderState precomputeImpl(const torch::Tensor                     &worldToCameraMatrices,
                               const torch::Tensor                     &projectionMatrices,
                               const fvdb::detail::ops::RenderSettings &settings);

    std::tuple<torch::Tensor, torch::Tensor>
    renderCropFromStateImpl(const RenderState &state, const size_t tileSize,
                            const ssize_t cropWidth, const ssize_t cropHeight,
                            const ssize_t cropOriginW, const ssize_t cropOriginH);

    std::tuple<torch::Tensor, torch::Tensor>
    renderImpl(const torch::Tensor &worldToCameraMatrices, const torch::Tensor &projectionMatrices,
               const fvdb::detail::ops::RenderSettings &settings);

  public:
    torch::Tensor
    means() const {
        return mMeans;
    }
    torch::Tensor
    quats() const {
        return mQuats;
    }
    torch::Tensor
    scales() const {
        return mScales;
    }
    torch::Tensor
    opacities() const {
        return mOpacities;
    }
    torch::Tensor
    shCoeffs() const {
        return mShCoeffs;
    }

    torch::Tensor
    expScales() const {
        return torch::exp(mScales);
    }

    torch::Tensor
    sigmoidOpacities() const {
        return torch::sigmoid(mOpacities);
    }

    RenderState precomputeRenderStateForImages(const torch::Tensor &worldToCameraMatrices,
                                               const torch::Tensor &projectionMatrices,
                                               size_t imageWidth, size_t imageHeight,
                                               const float near, const float far,
                                               const ProjectionType projectionType,
                                               const int64_t shDegreeToUse, const float minRadius2d,
                                               const float eps2d, const bool antialias);

    RenderState precomputeRenderStateForDepths(const torch::Tensor &worldToCameraMatrices,
                                               const torch::Tensor &projectionMatrices,
                                               size_t imageWidth, size_t imageHeight,
                                               const float near, const float far,
                                               const ProjectionType projectionType,
                                               const float minRadius2d, const float eps2d,
                                               const bool antialias);

    RenderState precomputeRenderStateForImagesAndDepths(
        const torch::Tensor &worldToCameraMatrices, const torch::Tensor &projectionMatrices,
        size_t imageWidth, size_t imageHeight, const float near, const float far,
        const ProjectionType projectionType, const int64_t shDegreeToUse, const float minRadius2d,
        const float eps2d, const bool antialias);

    void savePly(const std::string &filename) const;

    std::tuple<torch::Tensor, torch::Tensor>
    renderFromState(const GaussianSplat3d::RenderState &state, const ssize_t cropWidth = -1,
                    const ssize_t cropHeight = -1, const ssize_t cropOriginW = -1,
                    const ssize_t cropOriginH = -1, const size_t tileSize = 16);

    std::tuple<torch::Tensor, torch::Tensor> renderImages(
        const torch::Tensor &worldToCameraMatrices, const torch::Tensor &projectionMatrices,
        const size_t imageWidth, const size_t imageHeight, const float near, const float far,
        const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
        const int64_t shDegreeToUse = -1, const size_t tileSize = 16, const float minRadius2d = 0.0,
        const float eps2d = 0.3, const bool antialias = false);

    std::tuple<torch::Tensor, torch::Tensor>
    renderDepths(const torch::Tensor &worldToCameraMatrices,
                 const torch::Tensor &projectionMatrices, const size_t imageWidth,
                 const size_t imageHeight, const float near, const float far,
                 const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                 const size_t tileSize = 16, const float minRadius2d = 0.0, const float eps2d = 0.3,
                 const bool antialias = false);

    std::tuple<torch::Tensor, torch::Tensor> renderImagesAndDepths(
        const torch::Tensor &worldToCameraMatrices, const torch::Tensor &projectionMatrices,
        const size_t imageWidth, const size_t imageHeight, const float near, const float far,
        const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
        const int64_t shDegreeToUse = -1, const size_t tileSize = 16, const float minRadius2d = 0.0,
        const float eps2d = 0.3, const bool antialias = false);

    GaussianSplat3d(const torch::Tensor &means, const torch::Tensor &quats,
                    const torch::Tensor &scales, const torch::Tensor &opacities,
                    const torch::Tensor &shCoeffs, const bool requiresGrad = false);
};

} // namespace fvdb

#endif // FVDB_GAUSSIANSPLATTING2_H
