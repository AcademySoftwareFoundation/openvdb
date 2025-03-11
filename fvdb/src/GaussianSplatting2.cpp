// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianSplatting2.h"

#include <detail/autograd/Autograd.h>

namespace fvdb {

using RenderMode     = fvdb::detail::ops::RenderSettings::RenderMode;
using RenderSettings = fvdb::detail::ops::RenderSettings;

torch::Tensor
evaluateSphericalHarmonics(const torch::Tensor &shCoeffs, const torch::Tensor &radii,
                           const torch::optional<torch::Tensor> directions,
                           const int                            shDegreeToUse = -1) {
    const int K              = shCoeffs.size(0); // number of SH bases
    const int actualShDegree = shDegreeToUse < 0 ? (std::sqrt(K) - 1) : shDegreeToUse;
    TORCH_CHECK(K >= (actualShDegree + 1) * (actualShDegree + 1),
                "K must be at least (shDegreeToUse + 1)^2");
    auto shResults = detail::autograd::EvaluateSphericalHarmonics::apply(actualShDegree, directions,
                                                                         shCoeffs, radii);
    return shResults[0];
}

GaussianSplat3d::GaussianSplat3d(const torch::Tensor &means, const torch::Tensor &quats,
                                 const torch::Tensor &scales, const torch::Tensor &opacities,
                                 const torch::Tensor &shCoeffs, const bool requiresGrad)
    : mMeans(means.contiguous()), mQuats(quats.contiguous()), mScales(scales.contiguous()),
      mOpacities(opacities.contiguous()), mShCoeffs(shCoeffs.contiguous()),
      mRequiresGrad(requiresGrad) {
    const int64_t N = means.size(0); // number of gaussians

    TORCH_CHECK(mMeans.sizes() == torch::IntArrayRef({ N, 3 }), "means must have shape (N, 3)");
    TORCH_CHECK(mQuats.sizes() == torch::IntArrayRef({ N, 4 }), "quats must have shape (N, 4)");
    TORCH_CHECK(mScales.sizes() == torch::IntArrayRef({ N, 3 }), "scales must have shape (N, 3)");
    TORCH_CHECK(mOpacities.sizes() == torch::IntArrayRef({ N }), "opacities must have shape (N)");
    TORCH_CHECK(mShCoeffs.size(1) == N, "sh_coeffs must have shape (K, N, D)");
    TORCH_CHECK(mShCoeffs.dim() == 3, "sh_coeffs must have shape (K, N, D)");
    TORCH_CHECK(mMeans.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(mQuats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(mScales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(mOpacities.is_contiguous(), "opacities must be contiguous");
}

GaussianSplat3d::RenderState
GaussianSplat3d::precomputeImpl(const torch::Tensor  &worldToCameraMatrices,
                                const torch::Tensor  &projectionMatrices,
                                const RenderSettings &settings) {
    const bool ortho = settings.projectionType == fvdb::detail::ops::ProjectionType::ORTHOGRAPHIC;
    const int  C     = worldToCameraMatrices.size(0); // number of cameras
    const int  N     = mMeans.size(0);                // number of gaussians
    const int  K     = mShCoeffs.size(0);             // number of SH bases
    const int  D     = mShCoeffs.size(-1);            // Dimension of output

    TORCH_CHECK(worldToCameraMatrices.sizes() == torch::IntArrayRef({ C, 4, 4 }),
                "worldToCameraMatrices must have shape (C, 4, 4)");
    TORCH_CHECK(projectionMatrices.sizes() == torch::IntArrayRef({ C, 3, 3 }),
                "projectionMatrices must have shape (C, 3, 3)");
    TORCH_CHECK(worldToCameraMatrices.is_contiguous(), "worldToCameraMatrices must be contiguous");
    TORCH_CHECK(projectionMatrices.is_contiguous(), "projectionMatrices must be contiguous");

    RenderState ret;
    ret.mRenderSettings = settings;

    // Project to image plane [differentiable]
    const auto projectionResults = detail::autograd::ProjectGaussians::apply(
        mMeans, mQuats, expScales(), worldToCameraMatrices, projectionMatrices, settings.imageWidth,
        settings.imageHeight, settings.eps2d, settings.nearPlane, settings.farPlane,
        settings.radiusClip, settings.antialias, ortho);
    ret.perGaussianRadius  = projectionResults[0];
    ret.perGaussian2dMean  = projectionResults[1];
    ret.perGaussianDepth   = projectionResults[2];
    ret.perGaussianConic   = projectionResults[3];
    ret.perGaussianOpacity = sigmoidOpacities().repeat({ C, 1 });
    if (settings.antialias) {
        ret.perGaussianOpacity *= projectionResults[4];
        // FIXME (Francis): The contiguity requirement is dumb and should be
        // removed by using accessors in the kernel
        ret.perGaussianOpacity = ret.perGaussianOpacity.contiguous();
    }

    ret.perGaussianRenderQuantity = [&]() {
        torch::Tensor renderQuantity;
        if (settings.renderMode == RenderMode::DEPTH) {
            renderQuantity = ret.perGaussianDepth.unsqueeze(-1); // [C, N, 1]
        } else if (settings.renderMode == RenderMode::RGB ||
                   settings.renderMode == RenderMode::RGBD) {
            if (K == 1 || settings.shDegreeToUse == 0) {
                // Handle the case where we only have degree zero spherical harmonics, which just
                // represent diffuse colors. This means that each Gaussian receives the same color
                // regardless of which camera sees it, and we can just expand the colors to the
                // correct shape (without reallocating memory). i.e. the color tensor has shape [C,
                // N, D] but only allocates NxD floats in memory. This is useful for rendering e.g.
                // high dimensional diffuse features.
                renderQuantity =
                    evaluateSphericalHarmonics(mShCoeffs.unsqueeze(1), ret.perGaussianRadius,
                                               torch::nullopt, settings.shDegreeToUse);
                renderQuantity = renderQuantity.expand({ C, -1, -1 });

            } else {
                // FIXME (Francis): Do this in the kernel instead of materializing a large
                //                  tensor here. It's a bit annoying because we'll have to update
                //                  the current backward pass
                const torch::Tensor camToWorldMatrices = torch::inverse(worldToCameraMatrices);
                // Equivalent to dirs = means[None, :, :] - camToWorldMatrices[:, None, :3, 3]
                // NOTE: dirs are not normalized here, they get normalized in the spherical
                //       harmonics evaluation kernel
                const torch::Tensor dirs =
                    mMeans.index({ torch::indexing::None, torch::indexing::Slice(),
                                   torch::indexing::Slice() }) -
                    camToWorldMatrices.index({ torch::indexing::Slice(), torch::indexing::None,
                                               torch::indexing::Slice(0, 3),
                                               3 }); // [1, N, 3] - [C, 1, 3]
                renderQuantity =
                    evaluateSphericalHarmonics(mShCoeffs.unsqueeze(1).expand({ -1, C, -1, -1 }),
                                               ret.perGaussianRadius, dirs, settings.shDegreeToUse);
            }

            if (settings.renderMode == RenderMode::RGBD) {
                renderQuantity = torch::cat({ renderQuantity, ret.perGaussianDepth.unsqueeze(-1) },
                                            -1); // [C, N, D + 1]
            }
        } else {
            TORCH_CHECK_VALUE(false, "Invalid render mode");
        }
        return renderQuantity;
    }();

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int numTilesW = std::ceil(settings.imageWidth / static_cast<float>(settings.tileSize));
    const int numTilesH = std::ceil(settings.imageHeight / static_cast<float>(settings.tileSize));
    const auto [tileOffsets, tileGaussianIds] = FVDB_DISPATCH_KERNEL_DEVICE(mMeans.device(), [&]() {
        return detail::ops::dispatchGaussianTileIntersection<DeviceTag>(
            ret.perGaussian2dMean, ret.perGaussianRadius, ret.perGaussianDepth, at::nullopt, C,
            settings.tileSize, numTilesH, numTilesW);
    });
    ret.tileOffsets                           = tileOffsets;     // [C, TH, TW]
    ret.tileGaussianIds                       = tileGaussianIds; // [TOT_INTERSECTIONS]

    return ret;
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderCropFromStateImpl(const RenderState &state, const size_t tileSize,
                                         const ssize_t cropWidth, const ssize_t cropHeight,
                                         const ssize_t cropOriginW, const ssize_t cropOriginH) {
    // Negative values mean use the whole image, but all values must be negative
    if (cropWidth <= 0 || cropHeight <= 0 || cropOriginW < 0 || cropOriginH < 0) {
        TORCH_CHECK_VALUE(cropWidth <= 0 && cropHeight <= 0 && cropOriginW <= 0 && cropOriginH <= 0,
                          "Invalid crop dimensions");
    } else {
        TORCH_CHECK_VALUE(cropWidth > 0 && cropHeight > 0 && cropOriginW >= 0 && cropOriginH >= 0,
                          "Invalid crop dimensions");
    }

    const size_t cropWidth_   = cropWidth <= 0 ? state.imageWidth() : cropWidth;
    const size_t cropHeight_  = cropHeight <= 0 ? state.imageHeight() : cropHeight;
    const size_t cropOriginW_ = cropOriginW < 0 ? 0 : cropOriginW;
    const size_t cropOriginH_ = cropOriginH < 0 ? 0 : cropOriginH;

    // Rasterize projected Gaussians to pixels (differentiable)
    // NOTE:  precomputeRenderState* performs input checking, we need to apply some further checking
    // before GaussianRasterizeToPixels
    auto outputs = detail::autograd::RasterizeGaussiansToPixels::apply(
        state.perGaussian2dMean, state.perGaussianConic, state.perGaussianRenderQuantity,
        state.perGaussianOpacity, cropWidth_, cropHeight_, cropOriginW_, cropOriginH_, tileSize,
        state.tileOffsets, state.tileGaussianIds, false);
    torch::Tensor renderedImage  = outputs[0];
    torch::Tensor renderedAlphas = outputs[1];

    return { renderedImage, renderedAlphas };
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderImpl(const torch::Tensor                     &worldToCameraMatrices,
                            const torch::Tensor                     &projectionMatrices,
                            const fvdb::detail::ops::RenderSettings &settings) {
    const RenderState state = precomputeImpl(worldToCameraMatrices, projectionMatrices, settings);
    return renderCropFromStateImpl(state, settings.tileSize, settings.imageWidth,
                                   settings.imageHeight, 0, 0);
}

GaussianSplat3d::RenderState
GaussianSplat3d::precomputeRenderStateForImages(
    const torch::Tensor &worldToCameraMatrices, const torch::Tensor &projectionMatrices,
    size_t imageWidth, size_t imageHeight, const float near, const float far,
    const ProjectionType projectionType, const int64_t shDegreeToUse, const float minRadius2d,
    const float eps2d, const bool antialias) {
    RenderSettings settings;
    settings.imageWidth     = imageWidth;
    settings.imageHeight    = imageHeight;
    settings.nearPlane      = near;
    settings.farPlane       = far;
    settings.projectionType = projectionType;
    settings.shDegreeToUse  = shDegreeToUse;
    settings.radiusClip     = minRadius2d;
    settings.eps2d          = eps2d;
    settings.antialias      = antialias;
    settings.shDegreeToUse  = shDegreeToUse;

    settings.renderMode = RenderMode::RGB;

    return precomputeImpl(worldToCameraMatrices, projectionMatrices, settings);
}

GaussianSplat3d::RenderState
GaussianSplat3d::precomputeRenderStateForDepths(const torch::Tensor &worldToCameraMatrices,
                                                const torch::Tensor &projectionMatrices,
                                                size_t imageWidth, size_t imageHeight,
                                                const float near, const float far,
                                                const ProjectionType projectionType,
                                                const float minRadius2d, const float eps2d,
                                                const bool antialias) {
    RenderSettings settings;
    settings.imageWidth     = imageWidth;
    settings.imageHeight    = imageHeight;
    settings.nearPlane      = near;
    settings.farPlane       = far;
    settings.projectionType = projectionType;
    settings.shDegreeToUse  = -1;
    settings.radiusClip     = minRadius2d;
    settings.eps2d          = eps2d;
    settings.antialias      = antialias;
    settings.renderMode     = RenderMode::DEPTH;

    return precomputeImpl(worldToCameraMatrices, projectionMatrices, settings);
}

GaussianSplat3d::RenderState
GaussianSplat3d::precomputeRenderStateForImagesAndDepths(
    const torch::Tensor &worldToCameraMatrices, const torch::Tensor &projectionMatrices,
    size_t imageWidth, size_t imageHeight, const float near, const float far,
    const GaussianSplat3d::ProjectionType projectionType, const int64_t shDegreeToUse,
    const float minRadius2d, const float eps2d, const bool antialias) {
    RenderSettings settings;
    settings.imageWidth     = imageWidth;
    settings.imageHeight    = imageHeight;
    settings.nearPlane      = near;
    settings.farPlane       = far;
    settings.projectionType = projectionType;
    settings.shDegreeToUse  = shDegreeToUse;
    settings.radiusClip     = minRadius2d;
    settings.eps2d          = eps2d;
    settings.antialias      = antialias;

    settings.renderMode = RenderMode::RGBD;

    return precomputeImpl(worldToCameraMatrices, projectionMatrices, settings);
}

void
GaussianSplat3d::savePly(const std::string &filename) const {
    // TODO: Implement
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderFromState(const GaussianSplat3d::RenderState &state, const ssize_t cropWidth,
                                 const ssize_t cropHeight, const ssize_t cropOriginW,
                                 const ssize_t cropOriginH, const size_t tileSize) {
    return renderCropFromStateImpl(state, tileSize, cropWidth, cropHeight, cropOriginW,
                                   cropOriginH);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderImages(const torch::Tensor &worldToCameraMatrices,
                              const torch::Tensor &projectionMatrices, const size_t imageWidth,
                              const size_t imageHeight, const float near, const float far,
                              const ProjectionType projectionType, const int64_t shDegreeToUse,
                              const size_t tileSize, const float minRadius2d, const float eps2d,
                              const bool antialias) {
    RenderSettings settings;
    settings.imageWidth     = imageWidth;
    settings.imageHeight    = imageHeight;
    settings.nearPlane      = near;
    settings.farPlane       = far;
    settings.projectionType = projectionType;
    settings.shDegreeToUse  = shDegreeToUse;
    settings.radiusClip     = minRadius2d;
    settings.eps2d          = eps2d;
    settings.antialias      = antialias;
    settings.tileSize       = tileSize;
    settings.renderMode     = RenderSettings::RenderMode::RGB;

    return renderImpl(worldToCameraMatrices, projectionMatrices, settings);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderDepths(const torch::Tensor &worldToCameraMatrices,
                              const torch::Tensor &projectionMatrices, const size_t imageWidth,
                              const size_t imageHeight, const float near, const float far,
                              const ProjectionType projectionType, const size_t tileSize,
                              const float minRadius2d, const float eps2d, const bool antialias) {
    RenderSettings settings;
    settings.imageWidth     = imageWidth;
    settings.imageHeight    = imageHeight;
    settings.nearPlane      = near;
    settings.farPlane       = far;
    settings.projectionType = projectionType;
    settings.shDegreeToUse  = -1;
    settings.radiusClip     = minRadius2d;
    settings.eps2d          = eps2d;
    settings.tileSize       = tileSize;
    settings.renderMode     = RenderSettings::RenderMode::DEPTH;

    return renderImpl(worldToCameraMatrices, projectionMatrices, settings);
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianSplat3d::renderImagesAndDepths(
    const torch::Tensor &worldToCameraMatrices, const torch::Tensor &projectionMatrices,
    const size_t imageWidth, const size_t imageHeight, const float near, const float far,
    const ProjectionType projectionType, const int64_t shDegreeToUse, const size_t tileSize,
    const float minRadius2d, const float eps2d, const bool antialias) {
    RenderSettings settings;
    settings.imageWidth     = imageWidth;
    settings.imageHeight    = imageHeight;
    settings.nearPlane      = near;
    settings.farPlane       = far;
    settings.projectionType = projectionType;
    settings.shDegreeToUse  = shDegreeToUse;
    settings.radiusClip     = minRadius2d;
    settings.eps2d          = eps2d;
    settings.antialias      = antialias;
    settings.tileSize       = tileSize;
    settings.renderMode     = RenderSettings::RenderMode::RGBD;

    return renderImpl(worldToCameraMatrices, projectionMatrices, settings);
}
}; // namespace fvdb
