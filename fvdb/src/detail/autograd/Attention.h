#pragma once

#include <torch/autograd.h>


namespace fvdb {
namespace detail {
namespace autograd {

struct Attention : public torch::autograd::Function<Attention>
{
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const Variable& query,
                                 const Variable& key,
                                 const Variable& value,
                                 const Variable& qLengths,
                                 const Variable& kvLengths,
                                 float scale);

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb