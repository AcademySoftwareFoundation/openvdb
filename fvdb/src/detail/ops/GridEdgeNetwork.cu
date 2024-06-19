#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

/// @brief Per-voxel callback which computes the active grid coordinates for a batch of grids
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void gridEdgeNetworkCallback(int16_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
                                                GridBatchImpl::Accessor<GridType> gridAccessor,
                                                TorchAccessor<float, 2> outV,
                                                TorchAccessor<int16_t, 1> outVBidx,
                                                TorchAccessor<int64_t, 2> outE,
                                                TorchAccessor<int16_t, 1> outEBidx,
                                                bool returnVoxelCoordinates) {

    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);

    const VoxelCoordTransform tx = gridAccessor.primalTransform(batchIdx);

    const nanovdb::Coord voxIjk = leaf.offsetToGlobalCoord(voxelIdx);
    if (leaf.isActive(voxelIdx)) {
        int64_t vIdx = (int64_t) leaf.getValue(voxelIdx) - 1;
        const int64_t globalIdx = baseOffset + vIdx;
        const int64_t countV = globalIdx * 8;
        const int64_t countE = globalIdx * 12;

        for (int idx = 0; idx < 8; idx += 1) {
            const int32_t iz((idx & 1));
            const int32_t iy((idx & 2) >> 1);
            const int32_t ix((idx & 4) >> 2);
            nanovdb::Vec3f xyz = (voxIjk + nanovdb::Coord(ix, iy, iz)).asVec3s() - nanovdb::Vec3f(0.5);
            if (!returnVoxelCoordinates) {
                xyz = tx.applyInv(xyz);
            }

            for (int i = 0; i < 3; i += 1) {
                outV[countV + idx][i] = xyz[i];
            }
            outVBidx[countV + idx] = batchIdx;
        }

        const int64_t eBase = countV - baseOffset * 8;
        outE[countE + 0][0] = 0 + eBase; outE[countE + 0][1] = 1 + eBase;
        outE[countE + 1][0] = 0 + eBase; outE[countE + 1][1] = 2 + eBase;
        outE[countE + 2][0] = 0 + eBase; outE[countE + 2][1] = 4 + eBase;
        outE[countE + 3][0] = 2 + eBase; outE[countE + 3][1] = 3 + eBase;
        outE[countE + 4][0] = 2 + eBase; outE[countE + 4][1] = 6 + eBase;
        outE[countE + 5][0] = 3 + eBase; outE[countE + 5][1] = 7 + eBase;
        outE[countE + 6][0] = 3 + eBase; outE[countE + 6][1] = 1 + eBase;
        outE[countE + 7][0] = 7 + eBase; outE[countE + 7][1] = 6 + eBase;
        outE[countE + 8][0] = 6 + eBase; outE[countE + 8][1] = 4 + eBase;
        outE[countE + 9][0] = 7 + eBase; outE[countE + 9][1] = 5 + eBase;
        outE[countE + 10][0] = 5 + eBase; outE[countE + 10][1] = 4 + eBase;
        outE[countE + 11][0] = 1 + eBase; outE[countE + 11][1] = 5 + eBase;
        #pragma unroll
        for (int i = 0; i < 12; i += 1) {
            outEBidx[countE + i] = batchIdx;
        }
    }
}

template <c10::DeviceType DeviceTag>
std::vector<JaggedTensor> GridEdgeNetwork(const GridBatchImpl& batchHdl, bool returnVoxelCoordinates) {
    batchHdl.checkNonEmptyGrid();

    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> std::vector<JaggedTensor> {
        const int64_t numUnmasked = batchHdl.totalEnabledVoxels(false /*ignoreDisabledVoxels*/);

        auto optsV = torch::TensorOptions().dtype(torch::kFloat32).device(batchHdl.device());
        torch::Tensor outV = torch::empty({8 * numUnmasked, 3}, optsV);

        auto optsE = torch::TensorOptions().dtype(torch::kInt64).device(batchHdl.device());
        torch::Tensor outE = torch::empty({12 * numUnmasked, 2}, optsE);

        auto optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(batchHdl.device());
        torch::Tensor outVBidx = torch::empty({8 * numUnmasked}, optsBIdx);
        torch::Tensor outEBidx = torch::empty({12 * numUnmasked}, optsBIdx);

        auto outVAcc = tensorAccessor<DeviceTag, float, 2>(outV);
        auto outVBidxAcc = tensorAccessor<DeviceTag, int16_t, 1>(outVBidx);
        auto outEAcc = tensorAccessor<DeviceTag, int64_t, 2>(outE);
        auto outEBidxAcc = tensorAccessor<DeviceTag, int16_t, 1>(outEBidx);

        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
                gridEdgeNetworkCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, gridAccessor, outVAcc, outVBidxAcc, outEAcc, outEBidxAcc, returnVoxelCoordinates);
            };
            forEachVoxelCUDA<GridType>(1024, 1, batchHdl, cb);
        } else {
            auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
                gridEdgeNetworkCallback<GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, gridAccessor, outVAcc, outVBidxAcc, outEAcc, outEBidxAcc, returnVoxelCoordinates);
            };
            forEachVoxelCPU<GridType>(1, batchHdl, cb);
        }

        // FIXME: (@fwilliams) Be smarter about this
        const torch::Tensor outVBidx2 = batchHdl.batchSize() == 1 ? torch::empty({0}, optsBIdx) : outVBidx;
        const torch::Tensor outEBidx2 = batchHdl.batchSize() == 1 ? torch::empty({0}, optsBIdx) : outEBidx;
        return {
            JaggedTensor::from_data_and_jidx(outV, outVBidx2, batchHdl.batchSize()),
            JaggedTensor::from_data_and_jidx(outE, outEBidx2, batchHdl.batchSize())
        };
    });
}


template <>
std::vector<JaggedTensor> dispatchGridEdgeNetwork<torch::kCUDA>(const GridBatchImpl& gridHdl, bool returnVoxelCoordinates) {
    return GridEdgeNetwork<torch::kCUDA>(gridHdl, returnVoxelCoordinates);
}


template <>
std::vector<JaggedTensor> dispatchGridEdgeNetwork<torch::kCPU>(const GridBatchImpl& gridHdl, bool returnVoxelCoordinates) {
    return GridEdgeNetwork<torch::kCPU>(gridHdl, returnVoxelCoordinates);
}


} // namespace ops
} // namespace detail
} // namespace fvdb