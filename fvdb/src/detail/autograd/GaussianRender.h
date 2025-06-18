// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_GAUSSIANRENDER_H
#define FVDB_DETAIL_AUTOGRAD_GAUSSIANRENDER_H

#include <torch/all.h>
#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct ProjectGaussians : public torch::autograd::Function<ProjectGaussians> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList
    forward(AutogradContext *ctx,
            const Variable  &means,              // [N, 3]
            const Variable  &quats,              // [N, 4]
            const Variable  &scales,             // [N, 3]
            const Variable  &camToWorldMatrices, // [C, 4, 4]
            const Variable  &projectionMatrices, // [C, 3, 3]
            const uint32_t imageWidth, const uint32_t imageHeight, const float eps2d,
            const float nearPlane, const float farPlane, const float minRadius2D,
            const bool calcCompensions, const bool ortho,
            std::optional<Variable> outNormalizeddLossdMeans2dNormAccum = std::nullopt,
            std::optional<Variable> outNormalizedMaxRadiiAccum          = std::nullopt,
            std::optional<Variable> outGradientStepCount                = std::nullopt);

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

struct RasterizeGaussiansToPixels : public torch::autograd::Function<RasterizeGaussiansToPixels> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList forward(AutogradContext *ctx,
                                const Variable  &means2d,   // [C, N, 2]
                                const Variable  &conics,    // [C, N, 3]
                                const Variable  &colors,    // [C, N, 3]
                                const Variable  &opacities, // [N]
                                const uint32_t imageWidth, const uint32_t imageHeight,
                                const uint32_t imageOriginW, const uint32_t imageOriginH,
                                const uint32_t  tileSize,
                                const Variable &tileOffsets,     // [C, tile_height, tile_width]
                                const Variable &tileGaussianIds, // [n_isects]
                                const bool      absgrad);

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

struct ProjectGaussiansJagged : public torch::autograd::Function<ProjectGaussiansJagged> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList forward(AutogradContext *ctx,
                                const Variable  &gSizes,             // [B] gaussian sizes
                                const Variable  &means,              // [ggz, 3]
                                const Variable  &quats,              // [ggz, 4] optional
                                const Variable  &scales,             // [ggz, 3] optional
                                const Variable  &cSizes,             // [B] camera sizes
                                const Variable  &camToWorldMatrices, // [ccz, 4, 4]
                                const Variable  &projectionMatrices, // [ccz, 3, 3]
                                const uint32_t imageWidth, const uint32_t imageHeight,
                                const float eps2d, const float nearPlane, const float farPlane,
                                const float minRadius2D, const bool ortho);

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_GAUSSIANRENDER_H
