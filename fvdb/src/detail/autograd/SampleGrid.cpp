#include "SampleGrid.h"

#include "detail/ops/Ops.h"

#include "detail/utils/Utils.h"




void checkForwardInputs(c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
                        fvdb::detail::autograd::SampleGridTrilinear::JaggedVariable points,
                        fvdb::detail::autograd::SampleGridTrilinear::Variable data,
                        bool returnGrad) {
    grid->checkNonEmptyGrid();
    TORCH_CHECK_VALUE(points.device() == data.device(), "points and data must be on the same device");
    grid->checkDevice(points);
    grid->checkDevice(data);
    points.check_valid();

    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK_TYPE(points.dtype() == data.dtype(), "all tensors must have the same type");
    TORCH_CHECK_VALUE(points.dim() == 2, "Expected points to have shape [B*M, 3] (wrong number of dimensions)");
    TORCH_CHECK(points.numel() > 0, "Empty tensor (points)");
    TORCH_CHECK(points.size(1) == 3, "points must have shape [B, M, 3] (points must be 3D)");

    TORCH_CHECK_TYPE(data.is_floating_point(), "data must have a floating point type");
    TORCH_CHECK_VALUE(data.dim() >= 2, "Expected data to have shape [N, *] (at least 2 dimensions)");
    TORCH_CHECK(data.numel() > 0, "Empty tensor (data)");
    TORCH_CHECK(data.size(0) == grid->totalVoxels(), "grid_data must have one value per voxel (shape [N, *]) (wrong first dimension)");
}


namespace fvdb {
namespace detail {
namespace autograd {

SampleGridTrilinear::variable_list SampleGridTrilinear::forward(SampleGridTrilinear::AutogradContext *ctx,
                                                                c10::intrusive_ptr<GridBatchImpl> grid,
                                                                SampleGridTrilinear::JaggedVariable points,
                                                                SampleGridTrilinear::Variable data,
                                                                bool returnGrad) {
    checkForwardInputs(grid, points, data, returnGrad);

    auto ret = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
        if (returnGrad) {
            return ops::dispatchSampleGridTrilinearWithGrad<DeviceTag>(*grid, points, data);
        } else {
            return ops::dispatchSampleGridTrilinear<DeviceTag>(*grid, points, data);
        }
    });

    // Save data for backward in context
    ctx->save_for_backward({data, points.jdata(), points.joffsets()});
    ctx->saved_data["grid"] = grid;
    ctx->saved_data["return_grad"] = returnGrad;
    return ret;
}




SampleGridTrilinear::variable_list SampleGridTrilinear::backward(SampleGridTrilinear::AutogradContext *ctx,
                                                                 SampleGridTrilinear::variable_list grad_output) {

    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable data = saved.at(0);

    Variable pointCoords = saved.at(1);
    Variable pointJOffsets = saved.at(2);
    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    bool returnGrad = ctx->saved_data["return_grad"].toBool();
    Variable gradOut = grad_output.at(0);  // [B*M, *]

    torch::Tensor outGrad = FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
        if (returnGrad) {
            Variable gradPtsGrad = grad_output.at(1);  // [B*M, -1, 3]
            return ops::dispatchSampleGridTrilinearWithGradBackward<DeviceTag>(
                *grid, JaggedTensor::from_data_and_offsets(pointCoords, pointJOffsets), data, gradOut, gradPtsGrad);
        } else {
            return ops::dispatchSplatIntoGridTrilinear<DeviceTag>(
                    *grid, JaggedTensor::from_data_and_offsets(pointCoords, pointJOffsets), gradOut);
        }
    });

    return {torch::Tensor(), torch::Tensor(), outGrad, torch::Tensor()};
}








SampleGridBezier::variable_list SampleGridBezier::forward(SampleGridBezier::AutogradContext *ctx,
                                                          c10::intrusive_ptr<GridBatchImpl> grid,
                                                          SampleGridBezier::JaggedVariable points,
                                                          SampleGridBezier::Variable data,
                                                          bool returnGrad) {
    checkForwardInputs(grid, points, data, returnGrad);


    std::vector<torch::Tensor> ret = FVDB_DISPATCH_KERNEL_DEVICE(points.device(), [&]() {
        if (returnGrad) {
            return ops::dispatchSampleGridBezierWithGrad<DeviceTag>(*grid, points, data);
        } else {
            return ops::dispatchSampleGridBezier<DeviceTag>(*grid, points, data);
        }
    });

    // Save data for backward in context
    ctx->save_for_backward({data, points.jdata(), points.joffsets()});
    ctx->saved_data["grid"] = grid;
    ctx->saved_data["return_grad"] = returnGrad;

    return ret;
}


SampleGridBezier::variable_list SampleGridBezier::backward(SampleGridBezier::AutogradContext *ctx,
                                                           SampleGridBezier::variable_list grad_output) {

    // Use data saved in forward
    variable_list saved = ctx->get_saved_variables();
    Variable data = saved.at(0);

    Variable pointCoords = saved.at(1);
    Variable pointJOffsets = saved.at(2);

    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    bool returnGrad = ctx->saved_data["return_grad"].toBool();
    Variable gradOut = grad_output.at(0);  // [B*M, *]

    Variable outGrad = FVDB_DISPATCH_KERNEL_DEVICE(gradOut.device(), [&]() {
        if (returnGrad) {
            Variable gradPtsGrad = grad_output.at(1);  // [B*M, -1, 3]
            return ops::dispatchSampleGridBezierWithGradBackward<DeviceTag>(
                *grid, JaggedTensor::from_data_and_offsets(pointCoords, pointJOffsets), gradOut, gradPtsGrad, data);
        } else {
            return ops::dispatchSplatIntoGridBezier<DeviceTag>(
                *grid, JaggedTensor::from_data_and_offsets(pointCoords, pointJOffsets), gradOut);
        }
    });

    return {torch::Tensor(), torch::Tensor(), outGrad, torch::Tensor()};
}



}  // namespace autograd
}  // namespace detail
}  // namespace fvdb
