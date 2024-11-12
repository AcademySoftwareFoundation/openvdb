// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "SparseConvolutionHalo.h"

#include <detail/ops/convolution/backend/ConvOps.h>
#include <detail/utils/Utils.h>

namespace fvdb {
namespace detail {
namespace autograd {

SparseConvolutionHalo::variable_list
SparseConvolutionHalo::forward(SparseConvolutionHalo::AutogradContext *ctx,
                               c10::intrusive_ptr<GridBatchImpl>       grid,
                               SparseConvolutionHalo::Variable         inFeatures,
                               SparseConvolutionHalo::Variable kernels, int variant) {
    // Check kernels
    TORCH_CHECK_TYPE(kernels.is_floating_point(), "kernels must have a floating point type");
    TORCH_CHECK_VALUE(
        kernels.dim() == 5,
        std::string(
            "Expected kernels to have 5 dimensions (shape (out_ch, in_ch, d, h, w)) but got ") +
            std::to_string(kernels.dim()) + " dimensions");
    TORCH_CHECK_NOT_IMPLEMENTED(kernels.size(2) == kernels.size(3) &&
                                    kernels.size(3) == kernels.size(4) && kernels.size(2) == 3,
                                "sparse_conv_halo only supports kernels of size 3x3x3");

    // Check features
    TORCH_CHECK_VALUE(inFeatures.is_contiguous(), "features must be contiguous");
    TORCH_CHECK_TYPE(inFeatures.is_floating_point(), "features must have a floating point type");
    TORCH_CHECK_VALUE(
        inFeatures.dim() == 2,
        std::string("Expected features to have 2 dimensions (shape (n, nF)) but got ") +
            std::to_string(inFeatures.dim()) + " dimensions");
    TORCH_CHECK_VALUE(
        kernels.size(1) == inFeatures.size(1),
        "Expected input channels of kernels (" + std::to_string(kernels.size(1)) +
            ") to equal input channels of features: " + std::to_string(inFeatures.size(1)));

    // [O, I, 3, 3, 3] to [3, 3, 3, I, O]
    kernels = kernels.permute({ 4, 3, 2, 1, 0 }).contiguous();

    torch::Tensor outFeatures = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
        return ops::dispatchSparseConvolutionHalo<DeviceTag>(*grid, inFeatures, kernels, variant);
    });

    // Save data for backward in context
    ctx->save_for_backward({ inFeatures, kernels });
    ctx->saved_data["grid"]    = grid;
    ctx->saved_data["variant"] = variant;

    return variable_list({ outFeatures });
}

SparseConvolutionHalo::variable_list
SparseConvolutionHalo::backward(AutogradContext *ctx, variable_list grad_output) {
    variable_list saved = ctx->get_saved_variables();
    TORCH_CHECK(
        saved.size() > 0,
        "No backward context computed during forward. Please pass in training=True when calling kmap.build_implicit_gemm()");
    auto grid    = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    int  variant = ctx->saved_data["variant"].toInt();

    Variable inFeatures = saved.at(0);
    Variable kernels    = saved.at(1);                                           // [3, 3, 3, I, O]
    Variable gradOut    = grad_output.at(0);

    kernels = kernels.permute({ 0, 1, 2, 4, 3 }).flip({ 0, 1, 2 }).contiguous(); // [3, 3, 3, O, I]
    torch::Tensor gradInput  = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
        return ops::dispatchSparseConvolutionHalo<DeviceTag>(*grid, gradOut, kernels, variant);
    });
    torch::Tensor gradKernel = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
        return ops::dispatchSparseConvolutionHaloGrad<DeviceTag>(*grid, inFeatures, gradOut);
    });

    // [3, 3, 3, I, O] to [O, I, 3, 3, 3]
    gradKernel = gradKernel.permute({ 4, 3, 2, 1, 0 }).contiguous();

    return { torch::Tensor(), gradInput, gradKernel, torch::Tensor() };
}

} // namespace autograd
} // namespace detail
} // namespace fvdb