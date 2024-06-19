#pragma once

#include <torch/autograd.h>

#include "detail/GridBatchImpl.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct SplatIntoGridTrilinear : public torch::autograd::Function<SplatIntoGridTrilinear> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;
    using JaggedVariable = JaggedTensor;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> grid,
                                 JaggedTensor points,
                                 Variable pointData);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};


struct SplatIntoGridBezier : public torch::autograd::Function<SplatIntoGridBezier> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;
    using JaggedVariable = JaggedTensor;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> grid,
                                 JaggedVariable points,
                                 Variable pointData);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb