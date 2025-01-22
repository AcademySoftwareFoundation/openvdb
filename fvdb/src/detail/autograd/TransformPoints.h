// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_TRANSFORMPOINTS_H
#define FVDB_DETAIL_AUTOGRAD_TRANSFORMPOINTS_H

#include <detail/GridBatchImpl.h>

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct TransformPoints : public torch::autograd::Function<TransformPoints> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;
    using JaggedVariable  = JaggedTensor;

    static variable_list forward(AutogradContext *ctx, c10::intrusive_ptr<GridBatchImpl> grid,
                                 JaggedVariable points, Variable pointsData, bool isInverse,
                                 bool isDual);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_TRANSFORMPOINTS_H