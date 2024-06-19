#pragma once

#include <vector>

#include <nanovdb/NanoVDB.h>

#include <torch/autograd.h>

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"

#include "detail/GridBatchImpl.h"
#include "Types.h"


namespace fvdb {
namespace detail {
namespace autograd {

struct ReadFromDense : public torch::autograd::Function<ReadFromDense> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> grid,
                                 Variable denseData,
                                 const Vec3iBatch& denseOrigins) {
        TORCH_CHECK_VALUE(denseData.dim() > 4, "dense data must have shape [B, W, H, D, *]");
        TORCH_CHECK_VALUE(denseData.size(0) == grid->batchSize(), "dense data must have shape [B, W, H, D, *]");
        TORCH_CHECK_VALUE(denseData.is_contiguous(), "sparse_data must be contiguous");
        grid->checkDevice(denseData);

        // Non empty
        grid->checkNonEmptyGrid();

        // [B, W, H, D, -1]
        torch::Tensor denseDataReshape = featureCoalescedView(denseData, 4);

        // [N, -1]
        torch::Tensor ret = torch::zeros({grid->totalVoxels(), denseDataReshape.size(4)}, denseData.options());

        // nanovdb::Coord denseOriginNvdb = tensorToCoord(denseOrigins);
        // NanoVDB coordinates are int32
        torch::Tensor denseOriginsI32 = denseOrigins.tensorValue(grid->batchSize(), false /*onlyPositive*/, "dense_origins").to(denseData.device());

        FVDB_DISPATCH_KERNEL_DEVICE(grid->device(), [&]() {
            ops::dispatchReadFromDense<DeviceTag>(
                *grid, denseDataReshape, denseOriginsI32, ret, false);
        });

        // Reshape [B, N, -1] to [B, N, *] given [B, W, H, D, *]
        torch::Tensor retReshape = ret.view(
            spliceShape({grid->totalVoxels()}, denseData, 4));

        // Save shape information for backward
        ctx->saved_data["dense_origin"] = denseOriginsI32;
        ctx->saved_data["grid_size"] = coordToTensor(nanovdb::Coord(denseData.size(1), denseData.size(2), denseData.size(3)));
        ctx->saved_data["grid"] = grid;
        ctx->saved_data["dummy_tensor"] = torch::empty({0}, denseData.options());
        torch::Tensor retShape = torch::empty({(int64_t) denseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
        auto acc = retShape.accessor<int64_t, 1>();
        for (int i = 0; i < denseData.dim(); i++) {
            acc[i] = denseData.size(i);
        }
        ctx->saved_data["final_shape"] = retShape;

        return variable_list({retReshape});  // [N, *]
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output) {

        // Use data saved in forward
        torch::Tensor denseOrigins = ctx->saved_data["dense_origin"].toTensor();  // [B, 3]
        nanovdb::Coord gridSize = tensorToCoord(ctx->saved_data["grid_size"].toTensor());
        auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
        torch::TensorOptions denseDataOpts = ctx->saved_data["dummy_tensor"].toTensor().options();
        std::vector<int64_t> finalShapeTensor = intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());

        Variable gradOut = grad_output.at(0); // [N, *]
        torch::Tensor gradOutReshape = featureCoalescedView(gradOut); // [N, -1]
        torch::Tensor ret = torch::zeros({grid->batchSize(), gridSize[0], gridSize[1], gridSize[2], gradOutReshape.size(1)}, denseDataOpts);  // [B, W, H, D, -1]

        FVDB_DISPATCH_KERNEL_DEVICE(grid->device(), [&]() {
            ops::dispatchReadIntoDense<DeviceTag>(*grid, gradOutReshape, denseOrigins, ret, false);
        });

        torch::Tensor retReshape = ret.view(finalShapeTensor); // [B, W, H, D, *]

        return {torch::Tensor(), retReshape, torch::Tensor()};
    }
};

} // namespace autograd
} // namespace detail
} // namespace fvdb