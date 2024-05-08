#pragma once

#include <torch/autograd.h>

#include "detail/ops/Ops.h"

#include "SparseConvPackInfo.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct SparseConvolutionImplicitGEMM : public torch::autograd::Function<SparseConvolutionImplicitGEMM> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 Variable inFeatures, Variable kernels,
                                 SparseConvPackInfo packInfo,
                                 bool transposed) {
        if (transposed) {
            packInfo = packInfo.transposed();
        }

        TORCH_CHECK(packInfo.outInMap().has_value(),
                    "Out In Map must be built for IGEMM sparse convolution");
        torch::Tensor outInMap = packInfo.outInMap().value();
        bool useTF32 = packInfo.useTF32();

        const std::vector<int> sizes = { (int) packInfo.sourceGrid().total_voxels(), (int) packInfo.targetGrid().total_voxels() };
        TORCH_CHECK(packInfo.sourceGrid().is_mutable() == packInfo.targetGrid().is_mutable(), "Source and target grids must both be mutable or immutable");

        // Check features and kernels
        TORCH_CHECK_VALUE(inFeatures.is_contiguous(), "features must be contiguous");
        TORCH_CHECK_TYPE(inFeatures.is_floating_point(), "features must have a floating point type");
        TORCH_CHECK_VALUE(inFeatures.dim() == 2, std::string("Expected features to have 2 dimensions (shape (n, nF)) but got ") +
                                                 std::to_string(inFeatures.dim()) + " dimensions");
        TORCH_CHECK_TYPE(kernels.is_floating_point(), "kernels must have a floating point type");
        for (int i = 0; i < kernels.dim(); i += 1) {
            TORCH_CHECK_VALUE(kernels.size(i) != 0, "kernels tensor has zero dimension (dim = " + std::to_string(i) + ")");
        }

        auto opt = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
        torch::Tensor kWidth = torch::empty({3,}, opt);
        int inC, outC;
        if (!transposed) {
            TORCH_CHECK_VALUE(inFeatures.size(0) == sizes[0], "The number of input features must match the number of voxels");
            TORCH_CHECK_VALUE(kernels.dim() == 5, std::string("Expected kernels to have 5 dimensions (shape (out_ch, in_ch, d, h, w)) but got ") +
                                        std::to_string(kernels.dim()) + " dimensions");
            TORCH_CHECK_VALUE(kernels.size(1) == inFeatures.size(1),
                            "Expected input channels of kernels (" + std::to_string(kernels.size(1)) +
                            ") to equal input channels of features: " + std::to_string(inFeatures.size(1)));
            outC = kernels.size(0); inC = kernels.size(1);
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
            int inC = kernels.size(0); outC = kernels.size(1);
            kWidth[0] = kernels.size(2);
            kWidth[1] = kernels.size(3);
            kWidth[2] = kernels.size(4);
            kernels = kernels.permute({4, 3, 2, 0, 1}).reshape({-1, inC, outC}).contiguous();
        }

        torch::Tensor output;
        if (packInfo.targetGrid().total_voxels() > 0) {
            auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
            int outFeats = transposed ? sizes[0] : sizes[1];
            bool canSort = !transposed || (packInfo.kernelSize().value() < fvdb::Vec3iOrScalar(4).value());
            if (packInfo.reoderOutInMap().has_value() && canSort) {
                output = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
                    return ops::dispatchSparseConvolutionImplicitGEMMSorted<DeviceTag>(
                            inFeatures, kernels,
                            packInfo.reoderOutInMap().value(),
                            packInfo.reducedSortedMask().value(),
                            packInfo.reorderLoc().value(),
                            outFeats, outC, useTF32, true);
                });
            } else {
                output = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
                    return ops::dispatchSparseConvolutionImplicitGEMM<DeviceTag>(
                            inFeatures, kernels, outInMap, outFeats, outC, useTF32, true);
                });
            }
        } else {
            auto opt = torch::TensorOptions().dtype(inFeatures.dtype()).device(inFeatures.device());
            output = torch::empty({0, kernels.size(-1)}, opt);
        }

        // Save for backward (for training mode)
        if (packInfo.outInMapBwd().has_value()) {
            ctx->save_for_backward({inFeatures, kernels,
                                    packInfo.outInMapBwd().value(),
                                    packInfo.reorderOutInMapBwd().value(),
                                    packInfo.sortedMaskBwdW().value(),
                                    packInfo.sortedMaskBwdD().value(),
                                    packInfo.reorderLocBwd().value()});
        }
        ctx->saved_data["use_tf32"] = useTF32;
        ctx->saved_data["kernel_width"] = kWidth;
        ctx->saved_data["transposed"] = transposed;

        return {output};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
        variable_list saved = ctx->get_saved_variables();
        TORCH_CHECK(saved.size() > 0, "No backward context computed during forward. Please pass in training=True when calling kmap.build_implicit_gemm()");

        Variable inFeatures = saved.at(0);
        Variable kernels = saved.at(1);
        Variable outInMapBwd = saved.at(2);
        Variable reorderOutInMapBwd = saved.at(3);
        Variable sortedMaskBwdW = saved.at(4);
        Variable sortedMaskBwdD = saved.at(5);
        Variable reorderLocBwd = saved.at(6);
        bool useTF32 = ctx->saved_data["use_tf32"].toBool();
        torch::Tensor kWidth = ctx->saved_data["kernel_width"].toTensor();
        bool transposed = ctx->saved_data["transposed"].toBool();

        Variable gradOut = grad_output.at(0);
        torch::Tensor gradInput, gradWeight;

        // Dispatching following torchsparse++
        int kernelVolume = kernels.size(0);
        int inC = kernels.size(1);
        int outC = kernels.size(2);

        if (kernelVolume < 32) {
            gradInput = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
                return ops::dispatchSparseConvolutionImplicitGEMMSorted<DeviceTag>(
                        gradOut, kernels.transpose(2, 1).contiguous(),
                        reorderOutInMapBwd,
                        sortedMaskBwdD,
                        reorderLocBwd,
                        inFeatures.size(0), inC, useTF32, true);
            });
            gradWeight = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
                return ops::dispatchSparseConvolutionImplicitGEMMGradSorted<DeviceTag>(
                        gradOut, inFeatures, reorderOutInMapBwd,
                        sortedMaskBwdW, reorderLocBwd, 32,
                        useTF32, true);
            });
        } else {
            gradInput = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
                return ops::dispatchSparseConvolutionImplicitGEMM<DeviceTag>(
                        gradOut, kernels.transpose(2, 1).contiguous(),
                        outInMapBwd,
                        inFeatures.size(0), inC, useTF32, true);
            });
            gradWeight = FVDB_DISPATCH_KERNEL_DEVICE(inFeatures.device(), [&]() {
                return ops::dispatchSparseConvolutionImplicitGEMMGrad<DeviceTag>(
                        gradOut, inFeatures, outInMapBwd, 32,
                        useTF32, true);
            });
        }

        if (!transposed) {
            gradWeight = gradWeight.reshape(
                {kWidth[2].item<int32_t>(), kWidth[1].item<int32_t>(), kWidth[0].item<int32_t>(), outC, inC}).permute({3, 4, 2, 1, 0});
        } else {
            gradWeight = gradWeight.reshape(
                {kWidth[2].item<int32_t>(), kWidth[1].item<int32_t>(), kWidth[0].item<int32_t>(), outC, inC}).permute({4, 3, 2, 1, 0});
        }

        return {gradInput, gradWeight, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

} // namespace autograd
} // namespace detail
} // namespace fvdb
