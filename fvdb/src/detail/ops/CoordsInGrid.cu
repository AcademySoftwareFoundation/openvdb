#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void coordsInGridCallback(int32_t bidx, int32_t eidx,
                                             JaggedAccessor<ScalarType, 2> ijk,
                                             TensorAccessor<bool, 1> outMask,
                                             BatchGridAccessor<GridType> batchAccessor,
                                             bool ignoreMasked) {
    const auto* gpuGrid = batchAccessor.grid(bidx);
    auto primalAcc = gpuGrid->getAccessor();

    const auto& ijkCoord = ijk.data()[eidx];
    const nanovdb::Coord vox(ijkCoord[0], ijkCoord[1], ijkCoord[2]);
    const bool isActive = ignoreMasked ? primalAcc.isActive(vox) : primalAcc.template get<ActiveOrUnmasked<GridType>>(vox);
    outMask[eidx] = isActive;
}


template <c10::DeviceType DeviceTag>
JaggedTensor CoordsInGrid(const GridBatchImpl& batchHdl, const JaggedTensor& ijk, bool ignoreMasked) {

    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(ijk);
    TORCH_CHECK_TYPE(!ijk.is_floating_point(), "ijk must have an integeral type");
    TORCH_CHECK(ijk.dim() == 2, std::string("Expected ijk to have 2 dimensions (shape (n, 3)) but got ") + std::to_string(ijk.dim()) + " dimensions");
    TORCH_CHECK(ijk.size(0) > 0, "Empty tensor (ijk)");
    TORCH_CHECK(ijk.size(1) == 3, "Expected 3 dimensional ijk but got ijk.shape[1] = " + std::to_string(ijk.size(1)));

    auto opts = torch::TensorOptions().dtype(torch::kBool).device(ijk.device());
    torch::Tensor outMask = torch::empty({ijk.size(0)}, opts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_INTEGRAL_TYPES(ijk.scalar_type(), "CoordsInGrid", [&]() {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outMaskAccessor = tensorAccessor<DeviceTag, bool, 1>(outMask);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ijkAcc) {
                    coordsInGridCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, ijkAcc, outMaskAccessor, batchAcc, ignoreMasked);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(1024, 1, ijk, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ijkAcc) {
                    coordsInGridCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, ijkAcc, outMaskAccessor, batchAcc, ignoreMasked);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, ijk, cb);
            }
        });
    });

    return ijk.jagged_like(outMask);
}


template <>
JaggedTensor dispatchCoordsInGrid<torch::kCUDA>(const GridBatchImpl& batchHdl, const JaggedTensor& coords, bool ignoreMasked) {
    return CoordsInGrid<torch::kCUDA>(batchHdl, coords, ignoreMasked);
}

template <>
JaggedTensor dispatchCoordsInGrid<torch::kCPU>(const GridBatchImpl& batchHdl, const JaggedTensor& coords, bool ignoreMasked) {
    return CoordsInGrid<torch::kCPU>(batchHdl, coords, ignoreMasked);
}


} // namespace ops
} // namespace detail
} // namespace fvdb
