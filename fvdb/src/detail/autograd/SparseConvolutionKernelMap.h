#pragma once

#include <torch/autograd.h>

#include "detail/ops/convolution/backend/ConvOps.h"

#include "SparseConvPackInfo.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct SparseConvolutionKernelMap : public torch::autograd::Function<SparseConvolutionKernelMap> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 Variable inFeatures, Variable kernels,
                                 const SparseConvPackInfo& packInfo,
                                 bool transposed) {

        TORCH_CHECK(packInfo.neighborMap().has_value() && packInfo.neighborSizes().has_value(),
                    "Neighbor map must be built for sparse convolution");

        torch::Tensor nbmaps = packInfo.neighborMap().value();
        torch::Tensor nbsizes = packInfo.neighborSizes().value();
        const std::vector<int> sizes = { (int) packInfo.sourceGrid().total_voxels(), (int) packInfo.targetGrid().total_voxels() };
        const bool middleAcceleration = !(packInfo.sourceGrid().is_mutable() && packInfo.targetGrid().is_mutable()) && \
                                    packInfo.stride().value() == Vec3iOrScalar(1).value();

        TORCH_CHECK(packInfo.sourceGrid().is_mutable() == packInfo.targetGrid().is_mutable(), "Source and target grids must both be mutable or immutable");

        // Check features
        TORCH_CHECK_VALUE(inFeatures.is_contiguous(), "features must be contiguous");
        TORCH_CHECK_TYPE(inFeatures.is_floating_point(), "features must have a floating point type");
        TORCH_CHECK_VALUE(inFeatures.dim() == 2, std::string("Expected features to have 2 dimensions (shape (n, nF)) but got ") +
                                                 std::to_string(inFeatures.dim()) + " dimensions");

        // Check kernels
        TORCH_CHECK_TYPE(kernels.is_floating_point(), "kernels must have a floating point type");
        for (int i = 0; i < kernels.dim(); i += 1) {
            TORCH_CHECK_VALUE(kernels.size(i) != 0, "kernels tensor has zero dimension (dim = " + std::to_string(i) + ")");
        }
        // Check pack info
        TORCH_CHECK(nbmaps.is_contiguous() && nbmaps.scalar_type() == torch::kInt32, "nbmaps must be contiguous");
        TORCH_CHECK(nbsizes.is_contiguous() && nbsizes.scalar_type() == torch::kInt32, "nbsizes must be contiguous");

        auto opt = torch::TensorOptions().dtype(torch::kInt32).device(inFeatures.device());
        torch::Tensor kWidth = torch::empty({3,}, opt);
        if (!transposed) {
            TORCH_CHECK_VALUE(inFeatures.size(0) == sizes[0], "The number of input features must match the number of voxels");
            TORCH_CHECK_VALUE(kernels.dim() == 5, std::string("Expected kernels to have 5 dimensions (shape (out_ch, in_ch, d, h, w)) but got ") +
                                        std::to_string(kernels.dim()) + " dimensions");
            TORCH_CHECK_VALUE(kernels.size(1) == inFeatures.size(1),
                            "Expected input channels of kernels (" + std::to_string(kernels.size(1)) +
                            ") to equal input channels of features: " + std::to_string(inFeatures.size(1)));
            const int outC = kernels.size(0), inC = kernels.size(1);
            kWidth[0] = kernels.size(2);
            kWidth[1] = kernels.size(3);
            kWidth[2] = kernels.size(4);
            kernels = kernels.permute({4, 3, 2, 1, 0}).reshape({-1, inC, outC}).contiguous();
        } else {
            TORCH_CHECK_VALUE(inFeatures.size(0) == sizes[1], "The number of input features must match the number of voxels");
            TORCH_CHECK_VALUE(kernels.dim() == 5, std::string("Expected kernels to have 5 dimensions (shape (in_ch, out_ch, d, h, w)) but got ") +
                                        std::to_string(kernels.dim()) + " dimensions");
            TORCH_CHECK_VALUE(kernels.size(0) == inFeatures.size(1),
                            "Expected input channels of kernels (" + std::to_string(kernels.size(0)) +
                            ") to equal input channels of features: " + std::to_string(inFeatures.size(1)));
            const int inC = kernels.size(0), outC = kernels.size(1);
            kWidth[0] = kernels.size(2);
            kWidth[1] = kernels.size(3);
            kWidth[2] = kernels.size(4);
            kernels = kernels.permute({4, 3, 2, 0, 1}).reshape({-1, inC, outC}).contiguous();
        }

        // Save for backward
        ctx->save_for_backward({inFeatures, kernels, nbmaps, nbsizes});
        ctx->saved_data["transposed"] = transposed;
        ctx->saved_data["kernel_width"] = kWidth;
        ctx->saved_data["use_me"] = packInfo.useME();

        torch::Tensor output;
        if (packInfo.targetGrid().total_voxels() > 0) {
            auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
            if (!transposed) {
                output = torch::zeros({sizes[1], kernels.size(-1)}, opt);
            } else {
                output = torch::zeros({sizes[0], kernels.size(-1)}, opt);
            }
            // NOTE: Francis: We need .cpu().contiguous() here because we copied the convolution
            //       implementation from torch_sparse which runs std::max_element on a pointer
            //       to this tensor D: which is fucking awful...
            // TODO: Francis: Fix torch_sparse conv to be robust
            FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
                ops::dispatchSparseConvolutionKernelMap<DeviceTag>(
                        inFeatures, output, kernels, nbmaps,
                        nbsizes.cpu().contiguous(), transposed, middleAcceleration);
            });
        } else {
            auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
            output = torch::empty({0, kernels.size(-1)}, opt);
        }

        return {output};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        // Use data saved in forward
        variable_list saved = ctx->get_saved_variables();
        Variable inFeatures = saved.at(0);
        Variable kernels = saved.at(1);
        Variable nbmaps = saved.at(2);
        Variable nbsizes = saved.at(3);
        bool transposed = ctx->saved_data["transposed"].toBool();
        torch::Tensor kWidth = ctx->saved_data["kernel_width"].toTensor();
        bool use_me = ctx->saved_data["use_me"].toBool();

        torch::Tensor gradInput = torch::zeros_like(inFeatures);
        torch::Tensor gradWeight = torch::zeros_like(kernels);

        Variable gradOut = grad_output.at(0);

        if (gradOut.size(0) != 0) {
            if (use_me && gradOut.is_cuda()) {
                ops::dispatchMESparseConvolutionKernelMapGrad(
                    inFeatures, gradInput, gradOut.contiguous(), kernels, gradWeight, nbmaps,
                    nbsizes.cpu().contiguous(), transposed);
            } else {
                FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
                    ops::dispatchSparseConvolutionKernelMapGrad<DeviceTag>(
                        inFeatures, gradInput, gradOut.contiguous(), kernels, gradWeight, nbmaps,
                        nbsizes.cpu().contiguous(), transposed);
                });
            }
        }

        const int outC = gradWeight.size(-1), inC = gradWeight.size(-2);
        if (!transposed) {
            gradWeight = gradWeight.reshape({kWidth[2].item<int32_t>(), kWidth[1].item<int32_t>(), kWidth[0].item<int32_t>(), inC, outC}).permute({4, 3, 2, 1, 0});
        } else {
            gradWeight = gradWeight.reshape({kWidth[2].item<int32_t>(), kWidth[1].item<int32_t>(), kWidth[0].item<int32_t>(), inC, outC}).permute({3, 4, 2, 1, 0});
        }
        return {gradInput, gradWeight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

} // namespace autograd
} // namespace detail
} // namespace fvdb
