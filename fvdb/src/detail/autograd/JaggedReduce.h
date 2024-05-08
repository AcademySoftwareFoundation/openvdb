#pragma once

#include <torch/autograd.h>

#include "detail/GridBatchImpl.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct JaggedSum : public torch::autograd::Function<JaggedSum> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 Variable jdata, Variable jidx, int64_t dim_size);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

struct JaggedMin : public torch::autograd::Function<JaggedMin> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 Variable jdata, Variable jidx, int64_t dim_size);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

struct JaggedMax : public torch::autograd::Function<JaggedMax> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 Variable jdata, Variable jidx, int64_t dim_size);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb