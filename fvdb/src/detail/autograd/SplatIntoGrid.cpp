#include "SplatIntoGrid.h"

#include "detail/ops/Ops.h"

#include "detail/utils/Utils.h"


void checkForwardInputs(c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
                        fvdb::detail::autograd::SplatIntoGridTrilinear::JaggedVariable points,
                        fvdb::detail::autograd::SplatIntoGridTrilinear::Variable data) {
    grid->checkNonEmptyGrid();
    TORCH_CHECK_VALUE(points.device() == data.device(), "points and data must be on the same device");
    grid->checkDevice(points);
    grid->checkDevice(data);
    points.check_valid();

    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == data.dtype(), "all tensors must have the same type");
    TORCH_CHECK_VALUE(points.dim() == 2, "Expected points to have shape [B*M, 3] (wrong number of dimensions)");
    TORCH_CHECK(points.numel() > 0, "Empty tensor (points)");
    TORCH_CHECK(points.size(1) == 3, "points must have shape [B*M, 3] (points must be 3D)");

    TORCH_CHECK_TYPE(data.is_floating_point(), "point_data must have a floating point type");
    TORCH_CHECK_VALUE(data.dim() >= 2, "Expected data to have shape [B*M, *] (at least 3 dimensions)");
    TORCH_CHECK(data.numel() > 0, "Empty tensor (data)");
    TORCH_CHECK(data.size(0) == points.size(0), "point_data must have one value per point (shape [B*M, *]) (incorrect first dimension must match number of points)");
}

namespace fvdb {
namespace detail {
namespace autograd {


SplatIntoGridTrilinear::variable_list SplatIntoGridTrilinear::forward(SplatIntoGridTrilinear::AutogradContext *ctx,
                                                                      c10::intrusive_ptr<GridBatchImpl> grid,
                                                                      SplatIntoGridTrilinear::JaggedVariable points,
                                                                      SplatIntoGridTrilinear::Variable pointData) {

    checkForwardInputs(grid, points, pointData);

    torch::Tensor outGridData = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
        return ops::dispatchSplatIntoGridTrilinear<DeviceTag>(*grid, points, pointData);
    });

    // Save data for backward in context
    ctx->save_for_backward({pointData, points.jdata(), points.joffsets()});
    ctx->saved_data["grid"] = grid;
    // int64_t numOutputValues = grid->totalVoxels();

    return variable_list({outGridData});
}

SplatIntoGridTrilinear::variable_list SplatIntoGridTrilinear::backward(SplatIntoGridTrilinear::AutogradContext *ctx,
                                                                       SplatIntoGridTrilinear::variable_list grad_output) {

    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable pointData = saved.at(0);  // [B*M, *]

    Variable pointCoords = saved.at(1);  // [B*M, 3]
    Variable pointJOffsets = saved.at(2);  // [B*M,]
    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    Variable gradOut = grad_output.at(0);  // [N, *]

    auto ret = FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
        return ops::dispatchSampleGridTrilinear<DeviceTag>(
                *grid, JaggedTensor::from_data_and_offsets(pointCoords, pointJOffsets), gradOut);
    });

    return {torch::Tensor(), torch::Tensor(), ret[0]};
}




SplatIntoGridBezier::variable_list SplatIntoGridBezier::forward(SplatIntoGridBezier::AutogradContext *ctx,
                                                                c10::intrusive_ptr<GridBatchImpl> grid,
                                                                SplatIntoGridBezier::JaggedVariable points,
                                                                SplatIntoGridBezier::Variable pointData) {

    checkForwardInputs(grid, points, pointData);

    torch::Tensor outGridData = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
        return ops::dispatchSplatIntoGridBezier<DeviceTag>(*grid, points, pointData);
    });

    // Save data for backward in context
    ctx->save_for_backward({pointData, points.jdata(), points.joffsets()});
    ctx->saved_data["grid"] = grid;

    return variable_list({outGridData});
}

SplatIntoGridBezier::variable_list SplatIntoGridBezier::backward(SplatIntoGridBezier::AutogradContext *ctx,
                                                                 SplatIntoGridBezier::variable_list grad_output) {

    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable pointData = saved.at(0);  // [B*M, *]

    Variable pointCoords = saved.at(1);  // [B*M, 3]
    Variable pointJOffsets = saved.at(2);  // [B*M,]

    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    Variable gradOut = grad_output.at(0);  // [N, *]

    torch::Tensor outGradIn = FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
        return ops::dispatchSampleGridBezier<DeviceTag>(
            *grid, JaggedTensor::from_data_and_offsets(pointCoords, pointJOffsets), gradOut)[0];
    });

    return {torch::Tensor(), torch::Tensor(), outGradIn};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb