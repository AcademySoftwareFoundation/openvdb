#pragma once
#include <torch/autograd.h>

#include "detail/GridBatchImpl.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct UpsampleGrid : public torch::autograd::Function<UpsampleGrid> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> coarseGrid,
                                 c10::intrusive_ptr<GridBatchImpl> fineGrid,
                                 nanovdb::Coord upsamplingFactor,
                                 Variable coarseData);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb