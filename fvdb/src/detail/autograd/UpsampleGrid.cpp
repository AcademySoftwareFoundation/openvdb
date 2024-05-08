#include "UpsampleGrid.h"

#include <vector>

#include <nanovdb/NanoVDB.h>

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"

namespace fvdb {
namespace detail {
namespace autograd {

UpsampleGrid::variable_list UpsampleGrid::forward(UpsampleGrid::AutogradContext *ctx,
                                                  c10::intrusive_ptr<GridBatchImpl> coarseGrid,
                                                  c10::intrusive_ptr<GridBatchImpl> fineGrid,
                                                  nanovdb::Coord upsamplingFactor,
                                                  UpsampleGrid::Variable coarseData) {
    // Save data for backward in context
    ctx->save_for_backward({coarseData});

    ctx->saved_data["coarse_grid"] = coarseGrid;
    ctx->saved_data["fine_grid"] = fineGrid;
    ctx->saved_data["upsampling_factor_x"] = (int64_t) upsamplingFactor[0];
    ctx->saved_data["upsampling_factor_y"] = (int64_t) upsamplingFactor[1];
    ctx->saved_data["upsampling_factor_z"] = (int64_t) upsamplingFactor[2];

    if (fineGrid->totalVoxels() == 0) {
        return variable_list({torch::empty({0, coarseData.size(1)}, coarseData.options())});
    }

    torch::Tensor ret = FVDB_DISPATCH_KERNEL_DEVICE(coarseData.device(), [&]() {
        return ops::dispatchUpsampleGridNearest<DeviceTag>(
            *coarseGrid, *fineGrid, coarseData, upsamplingFactor);
    });
    return variable_list({ret});
}

UpsampleGrid::variable_list UpsampleGrid::backward(UpsampleGrid::AutogradContext *ctx,
                                                   UpsampleGrid::variable_list grad_output) {

    // // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable coarseData = saved.at(0);

    auto fineGrid = ctx->saved_data["fine_grid"].toCustomClass<GridBatchImpl>();
    auto coarseGrid = ctx->saved_data["coarse_grid"].toCustomClass<GridBatchImpl>();
    const int64_t upsamplingFactorX = ctx->saved_data["upsampling_factor_x"].toInt();
    const int64_t upsamplingFactorY = ctx->saved_data["upsampling_factor_y"].toInt();
    const int64_t upsamplingFactorZ = ctx->saved_data["upsampling_factor_z"].toInt();
    const nanovdb::Coord upsamplingFactor(upsamplingFactorX, upsamplingFactorY, upsamplingFactorZ);

    Variable gradOut = grad_output.at(0);  // [#fine_voxels, *]
    if (fineGrid->totalVoxels() == 0) {
        auto ret = torch::zeros_like(coarseData);
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), ret};
    }

    torch::Tensor outGradIn = FVDB_DISPATCH_KERNEL_DEVICE(coarseData.device(), [&]() {
        return ops::dispatchUpsampleGridNearestBackward<DeviceTag>(
            *fineGrid, *coarseGrid,
            gradOut,
            coarseData,
            upsamplingFactor
        );
    });

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), outGradIn};
}

}  // namespace autograd
}  // namespace detail
}  // namespace fvdb
