#pragma once

#include <torch/autograd.h>

#include "detail/ops/Ops.h"

#include "SparseConvPackInfo.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct SparseConvolutionHalo : public torch::autograd::Function<SparseConvolutionHalo> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> grid,
                                 Variable inFeatures,
                                 Variable kernels,
                                 int variant);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        TORCH_CHECK(false, "SparseConvolutionHalo::backward not implemented");
    }
};

} // namespace autograd
} // namespace detail
} // namespace fvdb
