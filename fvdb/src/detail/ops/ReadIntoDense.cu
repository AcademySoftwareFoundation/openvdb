#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"

namespace fvdb {
namespace detail {
namespace ops {

template <typename GridType, typename ScalarType>
__hostdev__ inline void readIntoDenseVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                   GridBatchImpl::Accessor<GridType> batchHandle,
                                                   torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> denseOrigins,        // [B, 3]
                                                   torch::PackedTensorAccessor32<ScalarType, 2, torch::RestrictPtrTraits> inSparseTensor,   // [B*N, C]
                                                   torch::PackedTensorAccessor32<ScalarType, 5, torch::RestrictPtrTraits> outDenseTensor,   // [B, W, H, D, C]
                                                   bool ignoreMasked) {

    using LeafNodeT = typename nanovdb::NanoGrid<GridType>::LeafNodeType;

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchHandle.grid(batchIdx);
    const nanovdb::Coord denseDim(outDenseTensor.size(1), outDenseTensor.size(2), outDenseTensor.size(3));
    const nanovdb::Coord denseOrigin(denseOrigins[batchIdx][0], denseOrigins[batchIdx][1], denseOrigins[batchIdx][2]);
    const nanovdb::CoordBBox bbox(denseOrigin, denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
    const int64_t baseOffset = batchHandle.voxelOffset(batchIdx);

    const LeafNodeT& leaf = gpuGrid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord voxIjk = leaf.offsetToGlobalCoord(voxelIdx);

    const bool isActive = ignoreMasked ? leaf.isActive(voxelIdx) : leaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx);

    const nanovdb::Coord ijk = voxIjk - denseOrigin;
    const int64_t offset = baseOffset + leaf.getValue(voxelIdx) - 1;

    if (isActive && bbox.isInside(voxIjk)) {
        outDenseTensor[batchIdx][ijk[0]][ijk[1]][ijk[2]][channelIdx] = inSparseTensor[offset][channelIdx];
    }
}


template <typename GridType, typename ScalarType>
void readIntoDenseCPU(const GridBatchImpl::Accessor<GridType>& gridHandle,
                      const torch::TensorAccessor<ScalarType, 2> inGridData,
                      const torch::TensorAccessor<int32_t, 2> denseOrigins,
                      torch::TensorAccessor<ScalarType, 5> outDenseTensor,
                      bool ignoreMasked,
                      bool isContiguous) {

    for (size_t bi = 0; bi < gridHandle.batchSize(); bi += 1) {
        const nanovdb::NanoGrid<GridType>* grid = gridHandle.grid(bi);

        const nanovdb::Coord bbmin(denseOrigins[bi][0], denseOrigins[bi][1], denseOrigins[bi][2]);
        const nanovdb::Coord bbsize(outDenseTensor.size(1), outDenseTensor.size(2), outDenseTensor.size(3));
        const nanovdb::CoordBBox bbox(bbmin, bbmin + bbsize - nanovdb::Coord(1, 1, 1));
        const int64_t baseOffset = gridHandle.voxelOffset(bi);

        auto outBatch = outDenseTensor[bi];

        for (auto it = ActiveVoxelIterator<GridType, -1>(grid->tree(), ignoreMasked, baseOffset); it.isValid(); it++) {
            const nanovdb::Coord voxIjk = it->first;
            if (bbox.isInside(voxIjk)) {
                const nanovdb::Coord ijk = voxIjk - bbox.min();

                if (isContiguous) {
                    memcpy(outBatch[ijk[0]][ijk[1]][ijk[2]].data(),
                           inGridData[it->second].data(),
                           inGridData.size(1) * sizeof(ScalarType));
                } else {
                    for (int c = 0; c < inGridData.size(1); ++c) {
                        outBatch[ijk[0]][ijk[1]][ijk[2]][c] = inGridData[it->second][c];
                    }
                }
            }
        }
    }
}


template <>
void dispatchReadIntoDense<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                       const torch::Tensor& inGridData,
                                       const torch::Tensor& denseOrigins,
                                       torch::Tensor& outDenseTensor,
                                       bool ignoreMasked) {

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(outDenseTensor.scalar_type(), "readIntoDense", [&]() {
            auto outDenseAcc = outDenseTensor.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc = denseOrigins.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();
            auto inGridDataAcc = inGridData.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> batchAcc) {
                readIntoDenseVoxelCallback<GridType, scalar_t>(bidx, lidx, vidx, cidx, batchAcc, denseOriginsAcc, inGridDataAcc, outDenseAcc, ignoreMasked);
            };
            forEachVoxelCUDA<GridType>(1024, inGridData.size(1), batchHdl, callback);

        });
    });

}


template <>
void dispatchReadIntoDense<torch::kCPU>(const GridBatchImpl& gridHdl,
                                      const torch::Tensor& inGridData,
                                      const torch::Tensor& denseOrigins,
                                      torch::Tensor& outDenseTensor,
                                      bool ignoreMasked) {
    bool isContiguous = inGridData.is_contiguous() && outDenseTensor.is_contiguous();

    FVDB_DISPATCH_GRID_TYPES(gridHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(outDenseTensor.scalar_type(), "readIntoDense", [&]() {
            readIntoDenseCPU(gridHdl.hostAccessor<GridType>(),
                             inGridData.accessor<scalar_t, 2>(),
                             denseOrigins.accessor<int32_t, 2>(),
                             outDenseTensor.accessor<scalar_t, 5>(),
                             ignoreMasked, isContiguous);
        });
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb