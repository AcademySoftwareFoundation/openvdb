#pragma once

#include <torch/autograd.h>

#include "detail/GridBatchImpl.h"

#include "Types.h"

namespace fvdb {
namespace detail {
namespace autograd {

struct ReadIntoDense : public torch::autograd::Function<ReadIntoDense> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> grid,
                                 Variable sparseData,
                                 const torch::optional<Vec3iBatch>& maybeMinCoord,
                                 const torch::optional<Vec3i>& maybeGridSize);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb