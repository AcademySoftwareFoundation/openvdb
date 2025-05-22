// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_VOLUMERENDER_H
#define FVDB_DETAIL_AUTOGRAD_VOLUMERENDER_H

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct VolumeRender : public torch::autograd::Function<VolumeRender> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const Variable &sigmas,
                                 const Variable &rgbs,
                                 const Variable &deltaTs,
                                 const Variable &ts,
                                 const Variable &raysAcc,
                                 double tsmtThreshold);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_VOLUMERENDER_H