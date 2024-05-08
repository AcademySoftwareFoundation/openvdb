#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename GridType, typename ScalarType>
__hostdev__ inline void readFromDenseVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                   GridBatchImpl::Accessor<GridType> batchHandle,
                                                   torch::PackedTensorAccessor32<ScalarType, 5, torch::RestrictPtrTraits> inDenseTensor,   // [B, W, H, D, C]
                                                   torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> denseOrigins,       // [B, 3]
                                                   torch::PackedTensorAccessor32<ScalarType, 2, torch::RestrictPtrTraits> outSparseTensor, // [B*N, C]
                                                   bool ignoreMasked) {

    using LeafNodeT = typename nanovdb::NanoGrid<GridType>::LeafNodeType;

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchHandle.grid(batchIdx);
    const nanovdb::Coord denseDim(inDenseTensor.size(1), inDenseTensor.size(2), inDenseTensor.size(3));
    const nanovdb::Coord denseOrigin(denseOrigins[batchIdx][0], denseOrigins[batchIdx][1], denseOrigins[batchIdx][2]);
    const nanovdb::CoordBBox bbox(denseOrigin, denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
    const int64_t baseOffset = batchHandle.voxelOffset(batchIdx);

    const LeafNodeT& leaf = gpuGrid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord voxIjk = leaf.offsetToGlobalCoord(voxelIdx);

    const bool isActive = ignoreMasked ? leaf.isActive(voxelIdx) : leaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx);

    const nanovdb::Coord ijk = voxIjk - denseOrigin;
    const int64_t offset = baseOffset + leaf.getValue(voxelIdx) - 1;

    if (isActive && bbox.isInside(voxIjk)) {
         outSparseTensor[offset][channelIdx] = inDenseTensor[batchIdx][ijk[0]][ijk[1]][ijk[2]][channelIdx];
    }
}


template <typename GridType, typename ScalarType>
void readFromDenseCPU(const GridBatchImpl::Accessor<GridType>& gridHandle,
                      const torch::TensorAccessor<ScalarType, 5> inDenseTensor,
                      const  torch::TensorAccessor<int32_t, 2> denseOrigins,
                      torch::TensorAccessor<ScalarType, 2> outSparseTensor,
                      bool ignoreMasked,
                      bool isContiguous) {


    for (size_t bi = 0; bi < gridHandle.batchSize(); bi += 1) {
        const nanovdb::NanoGrid<GridType>* grid = gridHandle.grid(bi);
        const nanovdb::Coord denseOrigin(denseOrigins[bi][0], denseOrigins[bi][1], denseOrigins[bi][2]);
        const nanovdb::Coord denseDim(inDenseTensor.size(1), inDenseTensor.size(2), inDenseTensor.size(3));
        const nanovdb::CoordBBox bbox(denseOrigin, denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
        const int64_t baseOffset = gridHandle.voxelOffset(bi);
        auto inBatch = inDenseTensor[bi];

        for (auto it = ActiveVoxelIterator<GridType, -1>(grid->tree(), ignoreMasked, baseOffset); it.isValid(); it++) {
            const nanovdb::Coord voxIjk = it->first;
            if (bbox.isInside(voxIjk)) {
                const nanovdb::Coord ijk = voxIjk - denseOrigin;
                if (isContiguous) {
                    memcpy(outSparseTensor[it->second].data(),
                        inBatch[ijk[0]][ijk[1]][ijk[2]].data(),
                        outSparseTensor.size(1) * sizeof(ScalarType));
                } else {
                    for (int c = 0; c < outSparseTensor.size(1); ++c) {
                        outSparseTensor[it->second][c] = inBatch[ijk[0]][ijk[1]][ijk[2]][c];
                    }
                }
            }
        }
    }
}


template <>
void dispatchReadFromDense<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                       const torch::Tensor& inDenseTensor,
                                       const torch::Tensor& denseOrigins,
                                       torch::Tensor& outSparseTensor,
                                       bool ignoreMasked) {

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(inDenseTensor.scalar_type(), "readFromDense", [&]() {
            auto inDenseAcc = inDenseTensor.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc = denseOrigins.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();
            auto outSparseAcc = outSparseTensor.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> batchAcc) {
                readFromDenseVoxelCallback<GridType, scalar_t>(bidx, lidx, vidx, cidx, batchAcc, inDenseAcc, denseOriginsAcc, outSparseAcc, ignoreMasked);
            };
            forEachVoxelCUDA<GridType>(1024, outSparseTensor.size(1), batchHdl, callback);

        });
    });

}


template <>
void dispatchReadFromDense<torch::kCPU>(const GridBatchImpl& gridHdl,
                                      const torch::Tensor& inDenseTensor,
                                      const torch::Tensor& denseOrigins,
                                      torch::Tensor& outSparseTensor,
                                      bool ignoreMasked) {

    bool isContiguous = inDenseTensor.is_contiguous() && outSparseTensor.is_contiguous();

    FVDB_DISPATCH_GRID_TYPES(gridHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(inDenseTensor.scalar_type(), "readFromDense", [&]() {
            readFromDenseCPU(gridHdl.hostAccessor<GridType>(),
                             inDenseTensor.accessor<scalar_t, 5>(),
                             denseOrigins.accessor<int32_t, 2>(),
                             outSparseTensor.accessor<scalar_t, 2>(),
                             ignoreMasked, isContiguous);
        });
    });
}


} // namespace ops
} // namespace detail
} // namespace fvdb