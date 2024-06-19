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

struct FillToGrid : public torch::autograd::Function<FillToGrid> {
    using variable_list = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<GridBatchImpl> fromGrid,
                                 c10::intrusive_ptr<GridBatchImpl> toGrid,
                                 Variable fromFeatures,
                                 const int default_value=0.0) {
        TORCH_CHECK_VALUE(fromFeatures.size(0) == fromGrid->totalVoxels(), "fromFeatures must conform to fromGrid");
        TORCH_CHECK_VALUE(fromGrid->batchSize() == toGrid->batchSize(), "fromGrid and toGrid must have the same batch size");

        torch::Tensor fromFeaturesReshape = featureCoalescedView(fromFeatures);
        torch::Tensor ret = torch::full({toGrid->totalVoxels(), fromFeaturesReshape.size(1)},
                                        default_value, fromFeaturesReshape.options());
        auto outShape = spliceShape({toGrid->totalVoxels()}, fromFeatures, 1);  // [B*M, *]

        // Dispatch to kernel.
        FVDB_DISPATCH_KERNEL_DEVICE(fromGrid->device(), [&]() {
            ops::dispatchFillToGrid<DeviceTag>(
                *fromGrid, *toGrid, fromFeaturesReshape, ret);
        });

        ctx->saved_data["from_grid"] = fromGrid;
        ctx->saved_data["to_grid"] = toGrid;

        return variable_list({ret.reshape(outShape)});
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_output) {
        torch::Tensor gradFeatures = grad_output[0];
        torch::Tensor gradFeaturesReshape = featureCoalescedView(gradFeatures);

        auto fromGrid = ctx->saved_data["from_grid"].toCustomClass<GridBatchImpl>();
        auto toGrid = ctx->saved_data["to_grid"].toCustomClass<GridBatchImpl>();
        auto outShape = spliceShape({fromGrid->totalVoxels()}, gradFeatures, 1);  // [B*M, *]

        // The default grad_input is always 0.0, since gradient will only propagate for overlapped voxels.
        torch::Tensor gradInput = torch::zeros({fromGrid->totalVoxels(), gradFeaturesReshape.size(1)},
                                               gradFeaturesReshape.options());

        // Dispatch same kernel but with to and from switched.
        FVDB_DISPATCH_KERNEL_DEVICE(fromGrid->device(), [&]() {
            ops::dispatchFillToGrid<DeviceTag>(
                *toGrid, *fromGrid, gradFeaturesReshape, gradInput);
        });

        return variable_list({torch::Tensor(), torch::Tensor(), gradInput.reshape(outShape), torch::Tensor()});
    }
};

} // namespace autograd
} // namespace detail
} // namespace fvdb
