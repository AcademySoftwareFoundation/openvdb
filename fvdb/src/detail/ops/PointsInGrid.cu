#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void pointsInGridCallback(int32_t bidx, int32_t eidx,
                                             JaggedAccessor<ScalarType, 2> points,
                                             TorchAccessor<bool, 1> outMask,
                                             BatchGridAccessor<GridType> batchAccessor,
                                             bool ignoreMasked) {
    const auto* gpuGrid = batchAccessor.grid(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    const VoxelCoordTransform& transform = batchAccessor.primalTransform(bidx);

    const auto pointPos = points.data()[eidx];
    const nanovdb::Coord vox = transform.apply((ScalarType) pointPos[0],
                                               (ScalarType) pointPos[1],
                                               (ScalarType) pointPos[2]).round();

    const bool isActive = ignoreMasked ? primalAcc.isActive(vox) : primalAcc.template get<ActiveOrUnmasked<GridType>>(vox);
    outMask[eidx] = isActive;
}


template <c10::DeviceType DeviceTag>
JaggedTensor PointsInGrid(const GridBatchImpl& batchHdl, const JaggedTensor& points, bool ignoreDisabledVoxels) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(points);
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK(points.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") + std::to_string(points.dim()) + " dimensions");
    TORCH_CHECK(points.size(0) > 0, "Empty tensor (points)");
    TORCH_CHECK(points.size(1) == 3, "Expected 3 dimensional points but got points.shape[1] = " + std::to_string(points.size(1)));

    auto opts = torch::TensorOptions().dtype(torch::kBool).device(points.device());
    torch::Tensor outMask = torch::empty({points.size(0)}, opts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "PointsInGrid", [&]() {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outMaskAccessor = tensorAccessor<DeviceTag, bool, 1>(outMask);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    pointsInGridCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, ptsA, outMaskAccessor, batchAcc, ignoreDisabledVoxels);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(1024, 1, points, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    pointsInGridCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, ptsA, outMaskAccessor, batchAcc, ignoreDisabledVoxels);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, points, cb);
            }
        });
    });

    return points.jagged_like(outMask);
}



template <>
JaggedTensor dispatchPointsInGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                              const JaggedTensor& points,
                                              bool ignoreMasked) {
    return PointsInGrid<torch::kCUDA>(batchHdl, points, ignoreMasked);
}

template <>
JaggedTensor dispatchPointsInGrid<torch::kCPU>(const GridBatchImpl& batchHdl,
                                             const JaggedTensor& points,
                                             bool ignoreMasked) {
    return PointsInGrid<torch::kCPU>(batchHdl, points, ignoreMasked);
}


} // namespace ops
} // namespace detail
} // namespace fvdb
