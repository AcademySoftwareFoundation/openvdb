// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_EVALUATESPHERICALHARMONICS_H
#define FVDB_DETAIL_AUTOGRAD_EVALUATESPHERICALHARMONICS_H

#include <torch/all.h>
#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct EvaluateSphericalHarmonics : public torch::autograd::Function<EvaluateSphericalHarmonics> {
    using VariableList    = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static VariableList
    forward(AutogradContext *ctx, const ssize_t shDegreeToUse, const size_t numCameras,
            const std::optional<Variable>  viewDirections, // [N, 3] or empty for deg 0
            const Variable                &sh0Coeffs,      // [1, N, 3]
            const std::optional<Variable> &shNCoeffs,      // [K-1, N, 3]
            const Variable                &radii           // [N,]
    );

    static VariableList backward(AutogradContext *ctx, VariableList gradOutput);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_EVALUATESPHERICALHARMONICS_H
