#include "TransformPoints.h"

#include <vector>

#include <nanovdb/NanoVDB.h>

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"



namespace fvdb {
namespace detail {
namespace autograd {


TransformPoints::variable_list TransformPoints::forward(TransformPoints::AutogradContext *ctx,
                                                        c10::intrusive_ptr<GridBatchImpl> grid,
                                                        TransformPoints::JaggedVariable points,
                                                        Variable pointsData,
                                                        bool isInverse,
                                                        bool isDual) {

    grid->checkDevice(points);
    TORCH_CHECK_VALUE(points.dim() == 2, "points must have shape [B*N, 3]");
    TORCH_CHECK_VALUE(points.size(-1) == 3, "points must have shape [B*N, 3]");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    points.check_valid();

    // FIXME: (@fwilliams) This is a hack because we need to pass tensors to the autograd engine :/
    JaggedTensor pointsWrap = points.jagged_like(pointsData);

    torch::Tensor outTxPoints;
    if (isInverse) {
        outTxPoints = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
            return ops::dispatchInvTransformPointsToGrid<DeviceTag>(
                *grid, pointsWrap, !isDual);
        });
    } else {
        outTxPoints = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
            return ops::dispatchTransformPointsToGrid<DeviceTag>(
                *grid, pointsWrap, !isDual);
        });
    }

    ctx->save_for_backward({points.joffsets()});

    ctx->saved_data["grid"] = grid;
    ctx->saved_data["isDual"] = isDual;
    ctx->saved_data["isInverse"] = isInverse;

    return {outTxPoints};  // [B*N, 3]
}


TransformPoints::variable_list TransformPoints::backward(TransformPoints::AutogradContext *ctx,
                                                         TransformPoints::variable_list grad_output) {

    variable_list saved = ctx->get_saved_variables();

    Variable pointsJOffsets = saved.at(0);
    Variable gradOut = grad_output.at(0);  // [B*N, 3]

    // Use data saved in forward
    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    const bool isDual = ctx->saved_data["isDual"].toBool();
    const bool isInverse = ctx->saved_data["isInverse"].toBool();

    Variable outGradIn; // = torch::empty_like(gradOut);  // [B*N, 3]
    if (isInverse) {
        outGradIn = FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
            return ops::dispatchInvTransformPointsToGridBackward<DeviceTag>(
                *grid, JaggedTensor::from_data_and_offsets(gradOut, pointsJOffsets), !isDual);
        });
    } else {
        outGradIn = FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
            return ops::dispatchTransformPointsToGridBackward<DeviceTag>(
                *grid, JaggedTensor::from_data_and_offsets(gradOut, pointsJOffsets), !isDual);
        });
    }

    // Variable outGradIn = outGradInReshape.reshape(getShapeButReplaceFirstDim(fineData.size(0), gradOut));
    return {torch::Tensor(), torch::Tensor(), outGradIn, torch::Tensor(), torch::Tensor()};
}


} // namespace autograd
} // namespace detail
} // namespace fvdb