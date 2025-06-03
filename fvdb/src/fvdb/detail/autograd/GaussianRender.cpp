// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/autograd/GaussianRender.h>
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/utils/Utils.h>

namespace fvdb {
namespace detail {
namespace autograd {

ProjectGaussians::VariableList
ProjectGaussians::forward(ProjectGaussians::AutogradContext *ctx,
                          const ProjectGaussians::Variable &means,
                          const ProjectGaussians::Variable &quats,
                          const ProjectGaussians::Variable &scales,
                          const ProjectGaussians::Variable &worldToCamMatrices,
                          const ProjectGaussians::Variable &projectionMatrices,
                          const uint32_t imageWidth,
                          const uint32_t imageHeight,
                          const float eps2d,
                          const float nearPlane,
                          const float farPlane,
                          const float minRadius2D,
                          const bool calcCompensations,
                          const bool ortho,
                          std::optional<Variable> outNormalizeddLossdMeans2dNormAccum,
                          std::optional<Variable> outNormalizedMaxRadiiAccum,
                          std::optional<Variable> outGradientStepCount) {
    TORCH_CHECK(means.dim() == 2, "means must have shape (N, 3)");
    TORCH_CHECK(worldToCamMatrices.dim() == 3, "worldToCamMatrices must have shape (C, 4, 4)");
    TORCH_CHECK(projectionMatrices.dim() == 3, "projectionMatrices must have shape (C, 3, 3)");

    auto variables   = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionForward<DeviceTag>(means,
                                                                 quats,
                                                                 scales,
                                                                 worldToCamMatrices,
                                                                 projectionMatrices,
                                                                 imageWidth,
                                                                 imageHeight,
                                                                 eps2d,
                                                                 nearPlane,
                                                                 farPlane,
                                                                 minRadius2D,
                                                                 calcCompensations,
                                                                 ortho);
    });
    Variable radii   = std::get<0>(variables);
    Variable means2d = std::get<1>(variables);
    Variable depths  = std::get<2>(variables);
    Variable conics  = std::get<3>(variables);

    ctx->saved_data["imageWidth"]        = static_cast<int64_t>(imageWidth);
    ctx->saved_data["imageHeight"]       = static_cast<int64_t>(imageHeight);
    ctx->saved_data["eps2d"]             = static_cast<double>(eps2d);
    ctx->saved_data["calcCompensations"] = static_cast<bool>(calcCompensations);
    ctx->saved_data["ortho"]             = static_cast<bool>(ortho);

    const bool saveAccumState         = outNormalizeddLossdMeans2dNormAccum.has_value();
    const bool trackMaxRadii          = outNormalizedMaxRadiiAccum.has_value();
    ctx->saved_data["saveAccumState"] = saveAccumState;
    ctx->saved_data["trackMaxRadii"]  = trackMaxRadii;
    if (saveAccumState) {
        ctx->saved_data["outNormalizeddLossdMeans2dNormAccum"] =
            outNormalizeddLossdMeans2dNormAccum.value();
        ctx->saved_data["outGradientStepCount"] = outGradientStepCount.value();
    }
    if (trackMaxRadii) {
        ctx->saved_data["outNormalizedMaxRadiiAccum"] = outNormalizedMaxRadiiAccum.value();
    }

    if (calcCompensations) {
        Variable compensations = std::get<4>(variables);
        ctx->save_for_backward({means,
                                quats,
                                scales,
                                worldToCamMatrices,
                                projectionMatrices,
                                radii,
                                conics,
                                compensations});
        return {radii, means2d, depths, conics, compensations};
    } else {
        ctx->save_for_backward(
            {means, quats, scales, worldToCamMatrices, projectionMatrices, radii, conics});
        return {radii, means2d, depths, conics};
    }
}

ProjectGaussians::VariableList
ProjectGaussians::backward(ProjectGaussians::AutogradContext *ctx,
                           ProjectGaussians::VariableList gradOutput) {
    Variable dLossDRadii   = gradOutput.at(0);
    Variable dLossDMeans2d = gradOutput.at(1);
    Variable dLossDDepths  = gradOutput.at(2);
    Variable dLossDConics  = gradOutput.at(3);

    // ensure the gradients are contiguous if they are not None
    if (dLossDRadii.defined()) {
        dLossDRadii = dLossDRadii.contiguous();
    }
    if (dLossDMeans2d.defined()) {
        dLossDMeans2d = dLossDMeans2d.contiguous();
    }
    if (dLossDDepths.defined()) {
        dLossDDepths = dLossDDepths.contiguous();
    }
    if (dLossDConics.defined()) {
        dLossDConics = dLossDConics.contiguous();
    }

    VariableList saved          = ctx->get_saved_variables();
    Variable means              = saved.at(0);
    Variable quats              = saved.at(1);
    Variable scales             = saved.at(2);
    Variable worldToCamMatrices = saved.at(3);
    Variable projectionMatrices = saved.at(4);
    Variable radii              = saved.at(5);
    Variable conics             = saved.at(6);

    const bool calcCompensations = ctx->saved_data["calcCompensations"].toBool();

    at::optional<Variable> compensations, dLossDCompensations;
    if (calcCompensations) {
        Variable vcomp = gradOutput.at(4);
        if (vcomp.defined()) {
            vcomp = vcomp.contiguous();
        }
        dLossDCompensations = vcomp;
        compensations       = saved.at(7);
    }

    const int imageWidth      = static_cast<int>(ctx->saved_data["imageWidth"].toInt());
    const int imageHeight     = static_cast<int>(ctx->saved_data["imageHeight"].toInt());
    const float eps2d         = static_cast<float>(ctx->saved_data["eps2d"].toDouble());
    const bool ortho          = ctx->saved_data["ortho"].toBool();
    const bool saveAccumState = ctx->saved_data["saveAccumState"].toBool();
    const bool trackMaxRadii  = ctx->saved_data["trackMaxRadii"].toBool();

    auto [normalizeddLossdMeans2dNormAccum, normalizedMaxRadiiAccum, gradientStepCount] = [&]() {
        return std::make_tuple(
            saveAccumState ? std::optional<at::Tensor>(
                                 ctx->saved_data["outNormalizeddLossdMeans2dNormAccum"].toTensor())
                           : std::nullopt,
            trackMaxRadii ? std::optional<at::Tensor>(
                                ctx->saved_data["outNormalizedMaxRadiiAccum"].toTensor())
                          : std::nullopt,
            saveAccumState
                ? std::optional<at::Tensor>(ctx->saved_data["outGradientStepCount"].toTensor())
                : std::nullopt);
    }();
    auto variables = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionBackward<DeviceTag>(means,
                                                                  quats,
                                                                  scales,
                                                                  worldToCamMatrices,
                                                                  projectionMatrices,
                                                                  compensations,
                                                                  imageWidth,
                                                                  imageHeight,
                                                                  eps2d,
                                                                  radii,
                                                                  conics,
                                                                  dLossDMeans2d,
                                                                  dLossDDepths,
                                                                  dLossDConics,
                                                                  dLossDCompensations,
                                                                  ctx->needs_input_grad(4),
                                                                  ortho,
                                                                  normalizeddLossdMeans2dNormAccum,
                                                                  normalizedMaxRadiiAccum,
                                                                  gradientStepCount);
    });

    Variable dLossDMeans = std::get<0>(variables);
    // Variable dLossDCovars = std::get<1>(variables);
    Variable dLossDQuats       = std::get<2>(variables);
    Variable dLossDScales      = std::get<3>(variables);
    Variable dLossDWorldToCams = std::get<4>(variables);

    return {dLossDMeans,
            dLossDQuats,
            dLossDScales,
            dLossDWorldToCams,
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable()};
}

RasterizeGaussiansToPixels::VariableList
RasterizeGaussiansToPixels::forward(
    RasterizeGaussiansToPixels::AutogradContext *ctx,
    const RasterizeGaussiansToPixels::Variable &means2d,   // [C, N, 2]
    const RasterizeGaussiansToPixels::Variable &conics,    // [C, N, 3]
    const RasterizeGaussiansToPixels::Variable &colors,    // [C, N, 3]
    const RasterizeGaussiansToPixels::Variable &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const RasterizeGaussiansToPixels::Variable &tileOffsets,     // [C, tile_height, tile_width]
    const RasterizeGaussiansToPixels::Variable &tileGaussianIds, // [n_isects]
    const bool absgrad) {
    // const int C = means2d.size(0);
    // const int N = means2d.size(1);

    auto variables          = FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeForward<DeviceTag>(means2d,
                                                                conics,
                                                                colors,
                                                                opacities,
                                                                imageWidth,
                                                                imageHeight,
                                                                imageOriginW,
                                                                imageOriginH,
                                                                tileSize,
                                                                tileOffsets,
                                                                tileGaussianIds);
    });
    Variable renderedColors = std::get<0>(variables);
    Variable renderedAlphas = std::get<1>(variables);
    Variable lastIds        = std::get<2>(variables);

    ctx->save_for_backward({means2d,
                            conics,
                            colors,
                            opacities,
                            tileOffsets,
                            tileGaussianIds,
                            renderedAlphas,
                            lastIds});
    ctx->saved_data["imageWidth"]   = (int64_t)imageWidth;
    ctx->saved_data["imageHeight"]  = (int64_t)imageHeight;
    ctx->saved_data["tileSize"]     = (int64_t)tileSize;
    ctx->saved_data["imageOriginW"] = (int64_t)imageOriginW;
    ctx->saved_data["imageOriginH"] = (int64_t)imageOriginH;
    ctx->saved_data["absgrad"]      = absgrad;

    return {renderedColors, renderedAlphas};
}

RasterizeGaussiansToPixels::VariableList
RasterizeGaussiansToPixels::backward(RasterizeGaussiansToPixels::AutogradContext *ctx,
                                     RasterizeGaussiansToPixels::VariableList gradOutput) {
    Variable dLossDRenderedColors = gradOutput.at(0);
    Variable dLossDRenderedAlphas = gradOutput.at(1);

    // ensure the gradients are contiguous if they are not None
    if (dLossDRenderedColors.defined()) {
        dLossDRenderedColors = dLossDRenderedColors.contiguous();
    }
    if (dLossDRenderedAlphas.defined()) {
        dLossDRenderedAlphas = dLossDRenderedAlphas.contiguous();
    }

    VariableList saved       = ctx->get_saved_variables();
    Variable means2d         = saved.at(0);
    Variable conics          = saved.at(1);
    Variable colors          = saved.at(2);
    Variable opacities       = saved.at(3);
    Variable tileOffsets     = saved.at(4);
    Variable tileGaussianIds = saved.at(5);
    Variable renderedAlphas  = saved.at(6);
    Variable lastIds         = saved.at(7);

    const int imageWidth   = (int)ctx->saved_data["imageWidth"].toInt();
    const int imageHeight  = (int)ctx->saved_data["imageHeight"].toInt();
    const int tileSize     = (int)ctx->saved_data["tileSize"].toInt();
    const int imageOriginW = (int)ctx->saved_data["imageOriginW"].toInt();
    const int imageOriginH = (int)ctx->saved_data["imageOriginH"].toInt();
    const bool absgrad     = ctx->saved_data["absgrad"].toBool();

    auto variables = FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeBackward<DeviceTag>(means2d,
                                                                 conics,
                                                                 colors,
                                                                 opacities,
                                                                 imageWidth,
                                                                 imageHeight,
                                                                 imageOriginW,
                                                                 imageOriginH,
                                                                 tileSize,
                                                                 tileOffsets,
                                                                 tileGaussianIds,
                                                                 renderedAlphas,
                                                                 lastIds,
                                                                 dLossDRenderedColors,
                                                                 dLossDRenderedAlphas,
                                                                 absgrad);
    });
    Variable dLossDMean2dAbs;
    if (absgrad) {
        dLossDMean2dAbs = std::get<0>(variables);
        // means2d.absgrad = dLossDMean2dAbs;
    } else {
        dLossDMean2dAbs = Variable();
    }
    Variable dLossDMeans2d   = std::get<1>(variables);
    Variable dLossDConics    = std::get<2>(variables);
    Variable dLossDColors    = std::get<3>(variables);
    Variable dLossDOpacities = std::get<4>(variables);

    return {
        dLossDMeans2d,
        dLossDConics,
        dLossDColors,
        dLossDOpacities,
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
        Variable(),
    };
}

ProjectGaussiansJagged::VariableList
ProjectGaussiansJagged::forward(
    ProjectGaussiansJagged::AutogradContext *ctx,
    const ProjectGaussiansJagged::Variable &gSizes,             // [B] gaussian sizes
    const ProjectGaussiansJagged::Variable &means,              // [ggz, 3]
    const ProjectGaussiansJagged::Variable &quats,              // [ggz, 4] optional
    const ProjectGaussiansJagged::Variable &scales,             // [ggz, 3] optional
    const ProjectGaussiansJagged::Variable &cSizes,             // [B] camera sizes
    const ProjectGaussiansJagged::Variable &worldToCamMatrices, // [ccz, 4, 4]
    const ProjectGaussiansJagged::Variable &projectionMatrices, // [ccz, 3, 3]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    const float nearPlane,
    const float farPlane,
    const float minRadius2D,
    const bool ortho) {
    auto variables   = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionJaggedForward<DeviceTag>(gSizes,
                                                                       means,
                                                                       quats,
                                                                       scales,
                                                                       cSizes,
                                                                       worldToCamMatrices,
                                                                       projectionMatrices,
                                                                       imageWidth,
                                                                       imageHeight,
                                                                       eps2d,
                                                                       nearPlane,
                                                                       farPlane,
                                                                       minRadius2D,
                                                                       ortho);
    });
    Variable radii   = std::get<0>(variables);
    Variable means2d = std::get<1>(variables);
    Variable depths  = std::get<2>(variables);
    Variable conics  = std::get<3>(variables);

    ctx->save_for_backward({gSizes,
                            means,
                            quats,
                            scales,
                            cSizes,
                            worldToCamMatrices,
                            projectionMatrices,
                            radii,
                            conics});
    ctx->saved_data["imageWidth"]  = (int64_t)imageWidth;
    ctx->saved_data["imageHeight"] = (int64_t)imageHeight;
    ctx->saved_data["eps2d"]       = (double)eps2d;
    ctx->saved_data["ortho"]       = (bool)ortho;

    return {radii, means2d, depths, conics};
}

ProjectGaussiansJagged::VariableList
ProjectGaussiansJagged::backward(ProjectGaussiansJagged::AutogradContext *ctx,
                                 ProjectGaussiansJagged::VariableList gradOutput) {
    Variable dLossDRadii   = gradOutput.at(0);
    Variable dLossDMeans2d = gradOutput.at(1);
    Variable dLossDDepths  = gradOutput.at(2);
    Variable dLossDConics  = gradOutput.at(3);

    // ensure the gradients are contiguous if they are not None
    if (dLossDRadii.defined())
        dLossDRadii = dLossDRadii.contiguous();
    if (dLossDMeans2d.defined())
        dLossDMeans2d = dLossDMeans2d.contiguous();
    if (dLossDDepths.defined())
        dLossDDepths = dLossDDepths.contiguous();
    if (dLossDConics.defined())
        dLossDConics = dLossDConics.contiguous();

    VariableList saved          = ctx->get_saved_variables();
    Variable gSizes             = saved.at(0);
    Variable means              = saved.at(1);
    Variable quats              = saved.at(2);
    Variable scales             = saved.at(3);
    Variable cSizes             = saved.at(4);
    Variable worldToCamMatrices = saved.at(5);
    Variable projectionMatrices = saved.at(6);
    Variable radii              = saved.at(7);
    Variable conics             = saved.at(8);

    const int imageWidth  = (int)ctx->saved_data["imageWidth"].toInt();
    const int imageHeight = (int)ctx->saved_data["imageHeight"].toInt();
    const float eps2d     = (float)ctx->saved_data["eps2d"].toDouble();
    const bool ortho      = (bool)ctx->saved_data["ortho"].toBool();

    auto variables       = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionJaggedBackward<DeviceTag>(gSizes,
                                                                        means,
                                                                        quats,
                                                                        scales,
                                                                        cSizes,
                                                                        worldToCamMatrices,
                                                                        projectionMatrices,
                                                                        imageWidth,
                                                                        imageHeight,
                                                                        eps2d,
                                                                        radii,
                                                                        conics,
                                                                        dLossDMeans2d,
                                                                        dLossDDepths,
                                                                        dLossDConics,
                                                                        ctx->needs_input_grad(6),
                                                                        ortho);
    });
    Variable dLossDMeans = std::get<0>(variables);
    // Variable dLossDCovars = std::get<1>(variables);
    Variable dLossDQuats       = std::get<2>(variables);
    Variable dLossDScales      = std::get<3>(variables);
    Variable dLossDWorldToCams = std::get<4>(variables);

    return {Variable(),
            dLossDMeans,
            dLossDQuats,
            dLossDScales,
            Variable(),
            dLossDWorldToCams,
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable(),
            Variable()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
