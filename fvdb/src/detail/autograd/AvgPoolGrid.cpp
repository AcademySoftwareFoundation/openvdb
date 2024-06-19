#include "AvgPoolGrid.h"

#include <nanovdb/NanoVDB.h>

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"


namespace fvdb {
namespace detail {
namespace autograd {

AvgPoolGrid::variable_list AvgPoolGrid::forward(AvgPoolGrid::AutogradContext *ctx,
                                                c10::intrusive_ptr<GridBatchImpl> fineGrid,
                                                c10::intrusive_ptr<GridBatchImpl> coarseGrid,
                                                nanovdb::Coord poolingFactor,
                                                nanovdb::Coord stride,
                                                AvgPoolGrid::Variable fineData) {

    torch::Tensor outCoarseData = FVDB_DISPATCH_KERNEL_DEVICE(fineData.device(), [&]() {
        return ops::dispatchDownsampleGridAvgPool<DeviceTag>(
            *fineGrid, *coarseGrid, fineData, poolingFactor, stride);
    });

    ctx->save_for_backward({fineData});
    ctx->saved_data["fine_grid"] = fineGrid;
    ctx->saved_data["coarse_grid"] = coarseGrid;
    ctx->saved_data["pooling_factor_x"] = (int64_t) poolingFactor[0];
    ctx->saved_data["pooling_factor_y"] = (int64_t) poolingFactor[1];
    ctx->saved_data["pooling_factor_z"] = (int64_t) poolingFactor[2];
    ctx->saved_data["stride_x"] = (int64_t) stride[0];
    ctx->saved_data["stride_y"] = (int64_t) stride[1];
    ctx->saved_data["stride_z"] = (int64_t) stride[2];

    return variable_list({outCoarseData});
}

AvgPoolGrid::variable_list AvgPoolGrid::backward(AvgPoolGrid::AutogradContext *ctx,
                                                 AvgPoolGrid::variable_list grad_output) {

    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable fineData = saved.at(0);
    auto fineGrid = ctx->saved_data["fine_grid"].toCustomClass<GridBatchImpl>();
    auto coarseGrid = ctx->saved_data["coarse_grid"].toCustomClass<GridBatchImpl>();
    const int64_t poolingFactorX = ctx->saved_data["pooling_factor_x"].toInt();
    const int64_t poolingFactorY = ctx->saved_data["pooling_factor_y"].toInt();
    const int64_t poolingFactorZ = ctx->saved_data["pooling_factor_z"].toInt();
    const int64_t strideX = ctx->saved_data["stride_x"].toInt();
    const int64_t strideY = ctx->saved_data["stride_y"].toInt();
    const int64_t strideZ = ctx->saved_data["stride_z"].toInt();
    const nanovdb::Coord poolingFactor(poolingFactorX, poolingFactorY, poolingFactorZ);
    const nanovdb::Coord stride(strideX, strideY, strideZ);
    Variable gradOut = grad_output.at(0).contiguous();  // [#coarse_voxels | #coarse_corners, *]

    Variable outGradIn = FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
        return ops::dispatchDownsampleGridAvgPoolBackward<DeviceTag>(
            *coarseGrid, *fineGrid,
            fineData,
            gradOut,
            poolingFactor,
            stride
        );
    });

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), outGradIn};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb