// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_AVGPOOLGRID_H
#define FVDB_DETAIL_AUTOGRAD_AVGPOOLGRID_H

#include <fvdb/detail/GridBatchImpl.h>

#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct AvgPoolGrid : public torch::autograd::Function<AvgPoolGrid> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> fineGrid,
                                 c10::intrusive_ptr<GridBatchImpl> coarseGrid,
                                 nanovdb::Coord poolingFactor,
                                 nanovdb::Coord stride,
                                 Variable fineData);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_AVGPOOLGRID_H
