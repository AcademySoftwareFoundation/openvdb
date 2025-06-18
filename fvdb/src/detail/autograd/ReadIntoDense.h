// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_READINTODENSE_H
#define FVDB_DETAIL_AUTOGRAD_READINTODENSE_H

#include <Types.h>
#include <detail/GridBatchImpl.h>

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct ReadIntoDense : public torch::autograd::Function<ReadIntoDense> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx, c10::intrusive_ptr<GridBatchImpl> grid,
                                 Variable                         sparseData,
                                 const std::optional<Vec3iBatch> &maybeMinCoord,
                                 const std::optional<Vec3i>      &maybeGridSize);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_READINTODENSE_H