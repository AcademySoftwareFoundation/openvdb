#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"



namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void voxelNeighborhoodCallback(int32_t bidx, int32_t eidx,
                                                  JaggedAccessor<ScalarType, 2> coords,
                                                  TensorAccessor<int64_t, 4> outIndex,
                                                  BatchGridAccessor<GridType> batchAccessor,
                                                  nanovdb::Coord extentMin,
                                                  nanovdb::Coord extentMax,
                                                  int32_t shift) {
    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    auto acc = gpuGrid->getAccessor();

    auto coord = coords.data()[eidx];
    nanovdb::Coord ijk0 = nanovdb::Coord(coord[0], coord[1], coord[2]) << shift;

    for (int32_t i = extentMin[0]; i <= extentMax[0]; i += 1) {
        for (int32_t j = extentMin[1]; j <= extentMax[1]; j += 1) {
            for (int32_t k = extentMin[2]; k <= extentMax[2]; k += 1) {
                const nanovdb::Coord ijk = nanovdb::Coord(i, j, k) + ijk0;
                const int64_t index = acc.template get<ActiveOrUnmasked<GridType>>(ijk) ? ((int64_t) acc.getValue(ijk) - 1) : -1;
                outIndex[eidx][i - extentMin[0]][j - extentMin[1]][k - extentMin[2]] = index;
            }
        }
    }
}


template <c10::DeviceType DeviceTag>
JaggedTensor VoxelNeighborhood(const GridBatchImpl& batchHdl,
                                     const JaggedTensor& ijk,
                                     nanovdb::Coord extentMin,
                                     nanovdb::Coord extentMax,
                                     int32_t shift) {
    batchHdl.checkDevice(ijk);
    TORCH_CHECK_TYPE(at::isIntegralType(ijk.scalar_type(), false), "ijk must have an integer type");
    TORCH_CHECK(ijk.dim() == 2, std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                                   std::to_string(ijk.dim()) + " dimensions");
    TORCH_CHECK(ijk.size(0) > 0, "Empty tensor (coords)");
    TORCH_CHECK(ijk.size(1) == 3,
                "Expected 3 dimensional coords but got points.shape[1] = " +
                std::to_string(ijk.size(1)));

    for (int i = 0; i < 3; i++) {
        TORCH_CHECK(extentMin[i] <= extentMax[i], "Extent min must be less than or equal to extent max");
    }
    TORCH_CHECK(shift >= 0, "Bitshift must be non-negative");
    const nanovdb::Coord extentPerAxis = (extentMax - extentMin) + nanovdb::Coord(1);
    const uint32_t numVals = extentPerAxis[0] * extentPerAxis[1] * extentPerAxis[2];

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(ijk.device());
    torch::Tensor outIndex = torch::empty({ijk.size(0), extentPerAxis[0], extentPerAxis[1], extentPerAxis[2]}, opts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_INTEGRAL_TYPES(ijk.scalar_type(), "VoxelNeighborhood", [&]() {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outIndexAcc = tensorAccessor<DeviceTag, int64_t, 4>(outIndex);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    voxelNeighborhoodCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, ptsA, outIndexAcc, batchAcc, extentMin, extentMax, shift);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(256, 1, ijk, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    voxelNeighborhoodCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, ptsA, outIndexAcc, batchAcc, extentMin, extentMax, shift);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, ijk, cb);
            }
        });
    });

    return ijk.jagged_like(outIndex);
}



template <>
JaggedTensor dispatchVoxelNeighborhood<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                   const JaggedTensor& coords,
                                                   nanovdb::Coord extentMin,
                                                   nanovdb::Coord extentMax,
                                                   int32_t shift) {
    return VoxelNeighborhood<torch::kCUDA>(batchHdl, coords, extentMin, extentMax, shift);
}

template <>
JaggedTensor dispatchVoxelNeighborhood<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                  const JaggedTensor& coords,
                                                   nanovdb::Coord extentMin,
                                                   nanovdb::Coord extentMax,
                                                   int32_t shift) {
    return VoxelNeighborhood<torch::kCPU>(batchHdl, coords, extentMin, extentMax, shift);
}


} // namespace ops
} // namespace detail
} // namespace fvdb