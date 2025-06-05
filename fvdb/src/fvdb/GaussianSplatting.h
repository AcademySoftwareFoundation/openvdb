// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_GAUSSIANSPLATTING_H
#define FVDB_GAUSSIANSPLATTING_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/GaussianRenderSettings.h>

#include <torch/all.h>

namespace fvdb {

/// @brief A class representing a Gaussian splat scene in 3D space.
/// This class is used to store the parameters of the Gaussians in the scene and provides
/// methods to project the Gaussians onto a 2D image plane, render images and depths,
/// and save the scene to a PLY file.
/// The Gaussians are represented by their means, quaternions (for rotation), log scales,
/// logit opacities, and SH coefficients. We use log_scales and logit_opacities since we can
/// optimize these quantities without clipping them to a specific range.
class GaussianSplat3d {
  public:
    GaussianSplat3d(const torch::Tensor &means,
                    const torch::Tensor &quats,
                    const torch::Tensor &logScales,
                    const torch::Tensor &logitOpacities,
                    const torch::Tensor &sh0,
                    const torch::Tensor &shN,
                    const bool requiresGrad = false);

    /// @brief Create a GaussianSplat3d object from a state_dict (similar to Pytorch's nn.Module).
    /// @param stateDict A dictionary containing the state of the GaussianSplat3d object.
    /// @return A GaussianSplat3d object created from the state_dict.
    GaussianSplat3d(const std::unordered_map<std::string, torch::Tensor> &stateDict) {
        loadStateDict(stateDict);
    }

    using ProjectionType = fvdb::detail::ops::ProjectionType;

    /// @brief A set of projected Gaussians that can be used to render images.
    struct ProjectedGaussianSplats {
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

  public:
    /// @brief Return the means of the Gaussians in this scene.
    /// @return An [N, 3]-shaped tensor representing the means of the Gaussians in this scenes.
    torch::Tensor
    means() const {
        return mMeans;
    }

    /// @brief Return the quaternions of the Gaussians in this scene which define the rotation
    ///        component of the covariance of each Gaussian (in the form [x, y, z, w]).
    /// @return An [N, 4]-shaped tensor representing the quaternions of the Gaussians in this scene.
    torch::Tensor
    quats() const {
        return mQuats;
    }

    /// @brief Return the log of the scales of the Gaussians in this scene.
    /// @return An [N]-shaped tensor representing the log of the scales of the
    ///         Gaussians in this scene.
    torch::Tensor
    logScales() const {
        return mLogScales;
    }

    /// @brief Return the logit (inverse of Sigmoid) of the opacities of the Gaussians in this
    ///        scene.
    /// @return An [N]-shaped tensor representing the logit of the opacities of the
    ///         Gaussians in this scene.
    torch::Tensor
    logitOpacities() const {
        return mLogitOpacities;
    }

    /// @brief Return the diffuse SH coefficients of the Gaussians in this scene
    /// @return An [1, N, D]-shaped tensor representing the diffuse SH coefficients of the
    /// Gaussians in this scene.
    torch::Tensor
    sh0() const {
        return mSh0;
    }

    /// @brief Return the directionally-dependent SH coefficients of the Gaussians in this scene
    /// @return A [K-1, N, D]-shaped tensor representing the directionally-dependent SH
    ///         coefficients of the Gaussians in this scene.
    torch::Tensor
    shN() const {
        return mShN;
    }

    /// @brief Return the scales of the Gaussians in this scene.
    /// @return An [N, 3]-shaped tensor representing the scales of the Gaussians in this scene.
    ///         (i.e. exp(logScales)).
    torch::Tensor
    scales() const {
        return torch::exp(mLogScales);
    }

    /// @brief Return the opacities of the Gaussians in this scene.
    /// @return An [N]-shaped tensor representing the opacities of the Gaussians in this scene.
    ///         (i.e. sigmoid(logitOpacities)).
    torch::Tensor
    opacities() const {
        return torch::sigmoid(mLogitOpacities);
    }

    /// @brief Whether to track the maximum 2D radii across backward passes for the projected
    ///        Gaussians in this scene. This is used during optimization to decide whether to
    ///        split/delete/duplicate Gaussians.
    /// @return Whether to track the maximum 2D radii across backward passes for the projected
    ///         Gaussians in this scene.
    bool
    trackMax2dRadiiForGrad() const {
        return mTrackMax2dRadiiForGrad;
    }

    /// @brief Set whether to track the maximum 2D radii across backward passes for the projected
    ///        Gaussians in this scene. This is used during optimization to decide whether to
    ///        split/delete/duplicate Gaussians.
    /// @param track Whether to track the maximum 2D radii across backward passes for the projected
    ///              Gaussians in this scene.
    void
    setTrackMax2dRadiiForGrad(bool track) {
        mTrackMax2dRadiiForGrad = track;
    }

    /// @brief Set the log of the opacities of the Gaussians in this scene.
    /// @param logitOpacities An [N]-shaped tensor representing the log of the opacities of the
    ///                     Gaussians in this scene.
    void
    setLogitOpacities(const torch::Tensor &logitOpacities) {
        TORCH_CHECK_VALUE(logitOpacities.sizes() == mLogitOpacities.sizes(),
                          "logit_opacities must have the same shape as the current opacities");
        TORCH_CHECK_VALUE(
            logitOpacities.device() == mLogitOpacities.device(),
            "logit_opacities must be on the same device as the current logit_opacities");
        mLogitOpacities = logitOpacities;
    }

    /// @brief Set the log of the scales of the Gaussians in this scene.
    /// @param logScales An [N, 3]-shaped tensor representing the log of the scales of the
    void
    setLogScales(const torch::Tensor &logScales) {
        TORCH_CHECK_VALUE(logScales.sizes() == mLogScales.sizes(),
                          "log_scales must have the same shape as the current scales");
        TORCH_CHECK_VALUE(logScales.device() == mLogScales.device(),
                          "log_scales must be on the same device as the current log_scales");
        mLogScales = logScales;
    }

    /// @brief Set the quaternions of the Gaussians in this scene which define the rotation
    ///        component of the covariance of each Gaussian (in the form [x, y, z, w]).
    /// @param quats An [N, 4]-shaped tensor representing the quaternions of the Gaussians in this
    ///              scene.
    void
    setQuats(const torch::Tensor &quats) {
        TORCH_CHECK_VALUE(quats.sizes() == mQuats.sizes(),
                          "quats must have the same shape as the current quats");
        TORCH_CHECK_VALUE(quats.device() == mQuats.device(),
                          "quats must be on the same device as the current quats");
        mQuats = quats;
    }

    /// @brief Set the means of the Gaussians in this scene.
    /// @param means An [N, 3]-shaped tensor representing the means of the Gaussians in this scene.
    void
    setMeans(const torch::Tensor &means) {
        TORCH_CHECK_VALUE(means.sizes() == mMeans.sizes(),
                          "means must have the same shape as the current means");
        TORCH_CHECK_VALUE(means.device() == mMeans.device(),
                          "means must be on the same device as the current means");
        mMeans = means;
    }

    /// @brief Set the diffuse SH coefficients of the Gaussians in this scene.
    /// @param sh0 An [1, N, D]-shaped tensor representing the diffuse SH coefficients of the
    ///            Gaussians in this scene.
    void
    setSh0(const torch::Tensor &sh0) {
        TORCH_CHECK_VALUE(sh0.sizes() == mSh0.sizes(),
                          "sh0 must have the same shape as the current sh0");
        TORCH_CHECK_VALUE(sh0.device() == mSh0.device(),
                          "sh0 must be on the same device as the current sh0");
        mSh0 = sh0;
    }

    /// @brief Set the directionally-dependent SH coefficients of the Gaussians in this scene.
    /// @param shN A [K-1, N, D]-shaped tensor representing the directionally-dependent SH
    ///            coefficients of the Gaussians in this scene.
    void
    setShN(const torch::Tensor &shN) {
        TORCH_CHECK_VALUE(shN.sizes() == mShN.sizes(),
                          "shN must have the same shape as the current shN");
        TORCH_CHECK_VALUE(shN.device() == mShN.device(),
                          "shN must be on the same device as the current shN");
        mShN = shN;
    }

    /// @brief Set whether we should track gradients and accumulated projection statisitics used
    ///        for splitting/deleting/duplicating Gaussians during optimization
    ///        (see @ref accumulated2dMeansGradientNormsForGrad, @ref gradientStepCountsForGrad,
    ///        @ref accumulatedMax2dRadiiForGrad) for the projected Gaussians in this scene.
    /// @param requiresGrad Whether to track gradients and accumulated projection statisitics used
    ///                     for splitting/deleting/duplicating Gaussians during optimization.
    void
    setRequiresGrad(bool requiresGrad) {
        if (requiresGrad == mRequiresGrad) {
            return;
        }
        if (requiresGrad == false) {
            mAccumulatedNormalized2dMeansGradientNormsForGrad = torch::Tensor();
            mAccumulated2dRadiiForGrad                        = torch::Tensor();
            mGradientStepCountForGrad                         = torch::Tensor();
        }
        mRequiresGrad = requiresGrad;
    }

    /// @brief Set the data of the GaussianSplat3d object from the given tensors.
    /// @param means An [N, 3]-shaped tensor representing the means of the Gaussians in this scene.
    /// @param quats An [N, 4]-shaped tensor representing the quaternions of the Gaussians in this
    ///              scene.
    /// @param logScales An [N, 3]-shaped tensor representing the log of the scales of the
    ///                  Gaussians in this scene.
    /// @param logitOpacities An [N]-shaped tensor representing the logit of the opacities of the
    ///                     Gaussians in this scene.
    /// @param sh0 An [1, N, D]-shaped tensor representing the diffuse SH coefficients of the
    ///            Gaussians in this scene.
    /// @param shN A [K-1, N, D]-shaped tensor representing the directionally-dependent SH
    ///            coefficients of the Gaussians in this scene.
    /// @param requiresGrad Whether to track gradients and accumulated projection statisitics used
    ///                     for splitting/deleting/duplicating Gaussians during optimization.
    void setState(const torch::Tensor &means,
                  const torch::Tensor &quats,
                  const torch::Tensor &logScales,
                  const torch::Tensor &logitOpacities,
                  const torch::Tensor &sh0,
                  const torch::Tensor &shN,
                  const bool requiresGrad = false);

    /// @brief Return the number of Gaussians in the scene.
    /// @return The number of Gaussians in the scene.
    int64_t
    numGaussians() const {
        return mMeans.size(0);
    }

    /// @brief Return the number of SH basis coeffients used in the scene.
    /// @return The number of SH bases used in the scene.
    int64_t
    numShBases() const {
        return mShN.size(0) + 1;
    }

    /// @brief Return the number of channels used in the scene (e.g. 3 for RGB colors).
    /// @return The number of channels used in the scene.
    int64_t
    numChannels() const {
        return mShN.size(2);
    }

    /// @brief Return whether gradients are being tracked for the Gaussians in this scene.
    /// @return Whether gradients are being tracked for the Gaussians in this scene.
    bool
    requiresGrad() const {
        return mRequiresGrad;
    }

    /// @brief Return the accumulated gradient norms of projected Gaussians in this
    ///        scene across backward passes.
    ///        This is used during optimization to decide whether to split/delete/duplicate
    ///        Gaussians.
    /// @return An [N]-shaped tensor representing the accumulated gradient norms of projected
    ///         Gaussians in this scene across backward passes or an empty tensor if
    ///         requiresGrad is false.
    torch::Tensor
    accumulated2dMeansGradientNormsForGrad() const {
        return mAccumulatedNormalized2dMeansGradientNormsForGrad;
    }

    /// @brief Return the accumulated maximum 2D radii of projected Gaussians in this
    ///        scene across backward passes.
    ///        This is used during optimization to decide whether to split/delete/duplicate
    ///        Gaussians.
    /// @return An [N]-shaped tensor representing the accumulated maximum 2D radii of projected
    ///         Gaussians in this scene across backward passes or an empty tensor if
    ///         requiresGrad and trackMax2dRadiiForGrad track are false.
    torch::Tensor
    accumulatedMax2dRadiiForGrad() const {
        return mAccumulated2dRadiiForGrad;
    }

    /// @brief Return the backward passes used to accumulate each Gaussian during optimization.
    ///        This is used during optimization to decide whether to split/delete/duplicate
    ///        Gaussians.
    /// @return An [N]-shaped tensor representing the backward passes used to accumulate each
    ///         Gaussian during optimization or an empty tensor if requiresGrad is false.
    torch::Tensor
    gradientStepCountsForGrad() const {
        return mGradientStepCountForGrad;
    }

    /// @brief Reset the gradient statistics of the Gaussians in this scene.
    ///        See @ref accumulated2dMeansGradientNormsForGrad, @ref gradientStepCountsForGrad,
    ///        @ref accumulatedMax2dRadiiForGrad.
    /// @note This function is only valid if requiresGrad is true.
    void
    resetGradState() {
        TORCH_CHECK(mRequiresGrad, "Cannot reset gradient state when requires_grad is false");
        mAccumulatedNormalized2dMeansGradientNormsForGrad = torch::Tensor();
        mAccumulated2dRadiiForGrad                        = torch::Tensor();
        mGradientStepCountForGrad                         = torch::Tensor();
    }

    /// @brief Return the state of the GaussianSplat3d object as a dictionary (similar to Pytorch's
    /// nn.Module).
    /// @return A dictionary containing the state of the GaussianSplat3d object.
    std::unordered_map<std::string, torch::Tensor> stateDict() const;

    /// @brief Load the state of the GaussianSplat3d object from a state_dict (similar to Pytorch's
    /// nn.Module).
    /// @param stateDict A dictionary containing the state of the GaussianSplat3d object.
    void loadStateDict(const std::unordered_map<std::string, torch::Tensor> &stateDict);

    /// @brief Precompute the projected Gaussians to be re-used for rendering images (e.g. if you
    /// want to render multiple images with the same camera settings or image patches).
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param shDegreeToUse Degree of SH to use for rendering (use -1 to use all SH bases)
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return ProjectedGaussianSplats object that can be used to render images with @ref
    /// renderFromProjectedGaussians
    ProjectedGaussianSplats projectGaussiansForImages(const torch::Tensor &worldToCameraMatrices,
                                                      const torch::Tensor &projectionMatrices,
                                                      size_t imageWidth,
                                                      size_t imageHeight,
                                                      const float near,
                                                      const float far,
                                                      const ProjectionType projectionType,
                                                      const int64_t shDegreeToUse,
                                                      const float minRadius2d,
                                                      const float eps2d,
                                                      const bool antialias);

    /// @brief Precompute the projected Gaussians to be re-used for rendering depths (e.g. if
    /// you want to render multiple depth maps with the same camera settings or image patches).
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return ProjectedGaussianSplats object that can be used to render depths with @ref
    /// renderFromProjectedGaussians
    ProjectedGaussianSplats projectGaussiansForDepths(const torch::Tensor &worldToCameraMatrices,
                                                      const torch::Tensor &projectionMatrices,
                                                      size_t imageWidth,
                                                      size_t imageHeight,
                                                      const float near,
                                                      const float far,
                                                      const ProjectionType projectionType,
                                                      const float minRadius2d,
                                                      const float eps2d,
                                                      const bool antialias);

    /// @brief Precompute the projected Gaussians to be re-used for rendering images and depths
    /// (e.g. if you want to render multiple images and depth maps with the same camera settings
    /// or image patches).
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param shDegreeToUse Degree of SH to use for rendering (use -1 to use all SH bases)
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return ProjectedGaussianSplats object that can be used to render images and depths with
    /// @ref renderFromProjectedGaussians
    ProjectedGaussianSplats
    projectGaussiansForImagesAndDepths(const torch::Tensor &worldToCameraMatrices,
                                       const torch::Tensor &projectionMatrices,
                                       size_t imageWidth,
                                       size_t imageHeight,
                                       const float near,
                                       const float far,
                                       const ProjectionType projectionType,
                                       const int64_t shDegreeToUse,
                                       const float minRadius2d,
                                       const float eps2d,
                                       const bool antialias);

    /// Save this scene to a PLY file with the given filename
    void savePly(const std::string &filename) const;

    /// @brief Render using precomputed projected Gaussians (see
    /// @ref projectGaussiansForImages, @ref projectGaussiansForDepths,
    /// @ref projectGaussiansForImagesAndDepths).
    /// Optionally lets you render a cropped image by specifying the crop width, height, and origin.
    /// @param projectedGaussians ProjectedGaussianSplats object obtained from @ref
    /// projectGaussiansForImages, @ref projectGaussiansForDepths, or @ref
    /// projectGaussiansForImagesAndDepths
    /// @param cropWidth Width of the cropped image (use -1 for no cropping)
    /// @param cropHeight Height of the cropped image (use -1 for no cropping)
    /// @param cropOriginW Origin of the cropped image in the width dimension (use -1 for no
    /// cropping)
    /// @param cropOriginH Origin of the cropped image in the height dimension (use -1 for no
    /// cropping)
    /// @param tileSize Size of the tiles used for rendering
    /// @return Tuple of two tensors:
    ///     images: A [C, W, H, D|1|D+1] tensor containing the the rendered image
    ///             (or depth or image and depth) for each camera
    ///     alphas: A [C, W, H, 1] tensor containing the alpha values of the rendered images
    std::tuple<torch::Tensor, torch::Tensor>
    renderFromProjectedGaussians(const GaussianSplat3d::ProjectedGaussianSplats &projectedGaussians,
                                 const ssize_t cropWidth   = -1,
                                 const ssize_t cropHeight  = -1,
                                 const ssize_t cropOriginW = -1,
                                 const ssize_t cropOriginH = -1,
                                 const size_t tileSize     = 16);

    /// @brief Render images of this Gaussian splat scene from the given camera matrices and
    /// projection matrices.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param shDegreeToUse Degree of SH to use for rendering (use -1 to use all SH bases)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return Tuple of two tensors:
    ///     images: A [C, W, H, D] tensor containing the the rendered image for each camera
    ///     alphas: A [C, W, H, 1] tensor containing the alpha values of the rendered images
    std::tuple<torch::Tensor, torch::Tensor>
    renderImages(const torch::Tensor &worldToCameraMatrices,
                 const torch::Tensor &projectionMatrices,
                 const size_t imageWidth,
                 const size_t imageHeight,
                 const float near,
                 const float far,
                 const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                 const int64_t shDegreeToUse         = -1,
                 const size_t tileSize               = 16,
                 const float minRadius2d             = 0.0,
                 const float eps2d                   = 0.3,
                 const bool antialias                = false);

    /// @brief Render depths of this Gaussian splat scene from the given camera matrices and
    /// projection matrices.
    /// @param worldToCameraMatrices [C, 4, 4] Camera-to-world matrices
    /// @param projectionMatrices [C, 4, 4] Projection matrices
    /// @param imageWidth Width of the image
    /// @param imageHeight Height of the image
    /// @param near Near plane
    /// @param far Far plane
    /// @param projectionType Type of projection (PERSPECTIVE or ORTHOGRAPHIC)
    /// @param tileSize Size of the tiles used for rendering
    /// @param minRadius2d Minimum radius in pixels below which projected Gaussians are ignored
    /// @param eps2d Blur factor for antialiasing (only used if antialias is true)
    /// @param antialias Whether to antialias the image
    /// @return Tuple of two tensors:
    ///     images: A [C, W, H, 1] tensor containing the the rendered depths for each camera
    ///     alphas: A [C, W, H, 1] tensor containing the alpha values of the rendered depths
    std::tuple<torch::Tensor, torch::Tensor>
    renderDepths(const torch::Tensor &worldToCameraMatrices,
                 const torch::Tensor &projectionMatrices,
                 const size_t imageWidth,
                 const size_t imageHeight,
                 const float near,
                 const float far,
                 const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                 const size_t tileSize               = 16,
                 const float minRadius2d             = 0.0,
                 const float eps2d                   = 0.3,
                 const bool antialias                = false);

    std::tuple<torch::Tensor, torch::Tensor>
    renderImagesAndDepths(const torch::Tensor &worldToCameraMatrices,
                          const torch::Tensor &projectionMatrices,
                          const size_t imageWidth,
                          const size_t imageHeight,
                          const float near,
                          const float far,
                          const ProjectionType projectionType = ProjectionType::PERSPECTIVE,
                          const int64_t shDegreeToUse         = -1,
                          const size_t tileSize               = 16,
                          const float minRadius2d             = 0.0,
                          const float eps2d                   = 0.3,
                          const bool antialias                = false);

  private:
    torch::Tensor mMeans;          // [N, 3]
    torch::Tensor mQuats;          // [N, 4]
    torch::Tensor mLogScales;      // [N, 3]
    torch::Tensor mLogitOpacities; // [N]
    torch::Tensor mSh0;            // [1, N, D]
    torch::Tensor mShN;            // [K-1, N, D]

    // Used for subdivision during optimization
    torch::Tensor mAccumulatedNormalized2dMeansGradientNormsForGrad; // [N]
    torch::Tensor mAccumulated2dRadiiForGrad;                        // [N]
    torch::Tensor mGradientStepCountForGrad;                         // [N]
    bool mRequiresGrad           = false;
    bool mTrackMax2dRadiiForGrad = false;

    static void checkState(const torch::Tensor &means,
                           const torch::Tensor &quats,
                           const torch::Tensor &logScales,
                           const torch::Tensor &logitOpacities,
                           const torch::Tensor &sh0,
                           const torch::Tensor &shN);

    ProjectedGaussianSplats projectGaussiansImpl(const torch::Tensor &worldToCameraMatrices,
                                                 const torch::Tensor &projectionMatrices,
                                                 const fvdb::detail::ops::RenderSettings &settings);

    std::tuple<torch::Tensor, torch::Tensor>
    renderCropFromProjectedGaussiansImpl(const ProjectedGaussianSplats &state,
                                         const size_t tileSize,
                                         const ssize_t cropWidth,
                                         const ssize_t cropHeight,
                                         const ssize_t cropOriginW,
                                         const ssize_t cropOriginH);

    std::tuple<torch::Tensor, torch::Tensor>
    renderImpl(const torch::Tensor &worldToCameraMatrices,
               const torch::Tensor &projectionMatrices,
               const fvdb::detail::ops::RenderSettings &settings);

    torch::Tensor evalSphericalHarmonicsImpl(const int64_t shDegreeToUse,
                                             const torch::Tensor &worldToCameraMatrices,
                                             const torch::Tensor &perGaussianProjectedRadii) const;
};

std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRenderJagged(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
                     const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
                     const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
                     const JaggedTensor &opacities, // [N1 + N2 + ...]
                     const JaggedTensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
                     const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
                     const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
                     const uint32_t image_width,
                     const uint32_t image_height,
                     const float near_plane          = 0.01,
                     const float far_plane           = 1e10,
                     const int sh_degree_to_use      = -1,
                     const int tile_size             = 16,
                     const float radius_clip         = 0.0,
                     const float eps2d               = 0.3,
                     const bool antialias            = false,
                     const bool render_depth_channel = false,
                     const bool return_debug_info    = false,
                     const bool render_depth_only    = false,
                     const bool ortho                = false);

} // namespace fvdb

#endif // FVDB_GAUSSIANSPLATTING_H
