#pragma once

#include <torch/autograd.h>

#include "detail/GridBatchImpl.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct MaxPoolGrid : public torch::autograd::Function<MaxPoolGrid> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> fineGrid,
                                 c10::intrusive_ptr<GridBatchImpl> coarseGrid,
                                 nanovdb::Coord poolingFactor,
                                 nanovdb::Coord stride,
                                 Variable fineData);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb