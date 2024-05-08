#include "ReadIntoDense.h"

#include <nanovdb/NanoVDB.h>

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"


namespace fvdb {
namespace detail {
namespace autograd {

ReadIntoDense::variable_list ReadIntoDense::forward(ReadIntoDense::AutogradContext *ctx,
                                                    c10::intrusive_ptr<GridBatchImpl> grid,
                                                    ReadIntoDense::Variable sparseData,
                                                    const torch::optional<Vec3iBatch>& maybeMinCoord,
                                                    const torch::optional<Vec3i>& maybeGridSize) {
    TORCH_CHECK_VALUE(sparseData.dim() > 1, "sparse_data must have shape [num_voxels, *]");
    TORCH_CHECK_VALUE(sparseData.size(0) == grid->totalVoxels(), "sparseData must have shape (num_voxels, *) where num_voxels = " + std::to_string(grid->totalVoxels()));
    TORCH_CHECK_VALUE(sparseData.is_contiguous(), "sparse_data must be contiguous");
    grid->checkDevice(sparseData);

    // Non empty
    grid->checkNonEmptyGrid();

    nanovdb::CoordBBox gridbb = grid->totalBBox(); // FIXME: Batched should use maximum bounding box which we need to compute

    // Min coord is an integer tensor of shape [3,] or [B, 3] representing the minimum coordinate of the dense tensor
    torch::Tensor denseOrigins;
    if (maybeMinCoord.has_value()) {
        denseOrigins = maybeMinCoord.value().tensorValue(grid->batchSize(), false /*onlyPositive*/, "min_coord").to(sparseData.device());
    } else {
        denseOrigins = coordToTensor(gridbb.min()).to(torch::kInt32).unsqueeze(0).repeat({grid->batchSize(), 1}).to(sparseData.device());
    }
    TORCH_CHECK_VALUE(denseOrigins.dim() == 2, "min_coord must have shape [3,] or [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(0) == grid->batchSize(), "min_coord must have shape [3,] or [B, 3]");
    TORCH_CHECK_VALUE(denseOrigins.size(1) == 3, "min_coord must have shape [3,] or [B, 3]");

    nanovdb::Coord gridSize = gridbb.dim();
    if (maybeGridSize.has_value()) {
        gridSize = maybeGridSize.value().value();
    }
    TORCH_CHECK_VALUE(gridSize[0] >= 0 && gridSize[1] >= 0 && gridSize[2] >= 0, "grid_size must be non-negative");

    torch::Tensor sparseDataReshape = featureCoalescedView(sparseData);  // [N, -1]
    TORCH_CHECK_VALUE(sparseDataReshape.is_contiguous(), "sparse_data must be contiguous");
    torch::Tensor ret = torch::zeros({grid->batchSize(), gridSize[0], gridSize[1], gridSize[2], sparseDataReshape.size(1)}, sparseData.options()); // [B, W, H, D, -1]
    FVDB_DISPATCH_KERNEL_DEVICE(grid->device(), [&]() {
        ops::dispatchReadIntoDense<DeviceTag>(*grid, sparseDataReshape, denseOrigins, ret, false);
    });
    torch::Tensor retReshape = ret.view(spliceShape({grid->batchSize(), gridSize[0], gridSize[1], gridSize[2]}, sparseData));
    TORCH_CHECK(retReshape.is_contiguous(), "retReshape must be contiguous");

    // Save shape information for backward
    ctx->saved_data["dense_origins"] = denseOrigins;
    ctx->saved_data["grid_size"] = coordToTensor(gridSize);
    torch::Tensor retShape = torch::empty({(int64_t) sparseData.dim()}, torch::TensorOptions().dtype(torch::kLong));
    auto acc = retShape.accessor<int64_t, 1>();
    for (int i = 0; i < sparseData.dim(); i++) {
        acc[i] = sparseData.size(i);
    }
    ctx->saved_data["final_shape"] = retShape;
    ctx->saved_data["first_dim"] = sparseDataReshape.size(0);
    ctx->saved_data["last_dim"] = sparseDataReshape.size(1);
    ctx->saved_data["dummy_tensor"] = torch::empty({0}, sparseData.options());
    ctx->saved_data["grid"] = grid;

    return variable_list({retReshape});
}

ReadIntoDense::variable_list ReadIntoDense::backward(ReadIntoDense::AutogradContext *ctx,
                                                     ReadIntoDense::variable_list grad_output) {

    // Use data saved in forward
    torch::Tensor denseOrigins = ctx->saved_data["dense_origins"].toTensor();  // [B, 3]
    int64_t firstDim = ctx->saved_data["first_dim"].toInt();
    int64_t lastDim = ctx->saved_data["last_dim"].toInt();
    std::vector<int64_t> finalShapeTensor = intTensor1DToStdVector(ctx->saved_data["final_shape"].toTensor());
    torch::TensorOptions sparseDataOpts = ctx->saved_data["dummy_tensor"].toTensor().options();
    auto grid = ctx->saved_data["grid"].toCustomClass<GridBatchImpl>();
    Variable gradOut = grad_output.at(0);  // [B, W, H, D, *]

    torch::Tensor gradOutReshape = featureCoalescedView(gradOut, 4);  // [B, W, H, D, -1]

    torch::Tensor ret = torch::zeros({firstDim, lastDim}, sparseDataOpts);  // [N, -1]

    FVDB_DISPATCH_KERNEL_DEVICE(grid->device(), [&]() {
        ops::dispatchReadFromDense<DeviceTag>(*grid, gradOutReshape, denseOrigins, ret, false);
    });

    torch::Tensor retReshape = ret.view(finalShapeTensor);  // [N, *]

    return {torch::Tensor(), retReshape, torch::Tensor(), torch::Tensor()};
}

} // namespace autograd
} // namespace detail
} // namespace fvdb