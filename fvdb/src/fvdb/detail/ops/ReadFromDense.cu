// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType>
__hostdev__ inline void
readFromDenseVoxelCallback(
    int32_t batchIdx,
    int32_t leafIdx,
    int32_t voxelIdx,
    int32_t channelIdx,
    GridBatchImpl::Accessor<nanovdb::ValueOnIndex> batchHandle,
    torch::PackedTensorAccessor64<ScalarType, 5, torch::RestrictPtrTraits>
        inDenseTensor, // [B, W, H, D, C]
    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> denseOrigins, // [B, 3]
    torch::PackedTensorAccessor64<ScalarType, 2, torch::RestrictPtrTraits>
        outSparseTensor                                                               // [B*N, C]
) {
    using LeafNodeT = typename nanovdb::OnIndexGrid::LeafNodeType;

    const nanovdb::OnIndexGrid *gpuGrid = batchHandle.grid(batchIdx);
    const nanovdb::Coord denseDim(
        inDenseTensor.size(1), inDenseTensor.size(2), inDenseTensor.size(3));
    const nanovdb::Coord denseOrigin(
        denseOrigins[batchIdx][0], denseOrigins[batchIdx][1], denseOrigins[batchIdx][2]);
    const nanovdb::CoordBBox bbox(denseOrigin, denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
    const int64_t baseOffset = batchHandle.voxelOffset(batchIdx);

    const LeafNodeT &leaf       = gpuGrid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord voxIjk = leaf.offsetToGlobalCoord(voxelIdx);

    const bool isActive = leaf.isActive(voxelIdx);

    const nanovdb::Coord ijk = voxIjk - denseOrigin;
    const int64_t offset     = baseOffset + leaf.getValue(voxelIdx) - 1;

    if (isActive && bbox.isInside(voxIjk)) {
        outSparseTensor[offset][channelIdx] =
            inDenseTensor[batchIdx][ijk[0]][ijk[1]][ijk[2]][channelIdx];
    }
}

template <typename ScalarType>
void
readFromDenseCPU(const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> &gridHandle,
                 const torch::TensorAccessor<ScalarType, 5> inDenseTensor,
                 const torch::TensorAccessor<int32_t, 2> denseOrigins,
                 torch::TensorAccessor<ScalarType, 2> outSparseTensor,
                 bool isContiguous) {
    for (int64_t bi = 0; bi < gridHandle.batchSize(); bi += 1) {
        const nanovdb::OnIndexGrid *grid = gridHandle.grid(bi);
        const nanovdb::Coord denseOrigin(
            denseOrigins[bi][0], denseOrigins[bi][1], denseOrigins[bi][2]);
        const nanovdb::Coord denseDim(
            inDenseTensor.size(1), inDenseTensor.size(2), inDenseTensor.size(3));
        const nanovdb::CoordBBox bbox(denseOrigin,
                                      denseOrigin + denseDim - nanovdb::Coord(1, 1, 1));
        const int64_t baseOffset = gridHandle.voxelOffset(bi);
        auto inBatch             = inDenseTensor[bi];

        for (auto it = ActiveVoxelIterator<-1>(grid->tree(), baseOffset); it.isValid(); it++) {
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
void
dispatchReadFromDense<torch::kCUDA>(const GridBatchImpl &batchHdl,
                                    const torch::Tensor &inDenseTensor,
                                    const torch::Tensor &denseOrigins,
                                    torch::Tensor &outSparseTensor) {
    AT_DISPATCH_V2(
        inDenseTensor.scalar_type(),
        "readFromDense",
        AT_WRAP([&]() {
            auto inDenseAcc =
                inDenseTensor.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>();
            auto denseOriginsAcc =
                denseOrigins.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto outSparseAcc =
                outSparseTensor.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto callback = [=] __device__(int32_t bidx,
                                           int32_t lidx,
                                           int32_t vidx,
                                           int32_t cidx,
                                           GridBatchImpl::Accessor<nanovdb::ValueOnIndex>
                                               batchAcc) {
                readFromDenseVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, batchAcc, inDenseAcc, denseOriginsAcc, outSparseAcc);
            };
            forEachVoxelCUDA<nanovdb::ValueOnIndex>(
                1024, outSparseTensor.size(1), batchHdl, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf,
        c10::kBFloat16);
}

template <>
void
dispatchReadFromDense<torch::kCPU>(const GridBatchImpl &gridHdl,
                                   const torch::Tensor &inDenseTensor,
                                   const torch::Tensor &denseOrigins,
                                   torch::Tensor &outSparseTensor) {
    bool isContiguous = inDenseTensor.is_contiguous() && outSparseTensor.is_contiguous();

    AT_DISPATCH_V2(inDenseTensor.scalar_type(),
                   "readFromDense",
                   AT_WRAP([&]() {
                       readFromDenseCPU(gridHdl.hostAccessor<nanovdb::ValueOnIndex>(),
                                        inDenseTensor.accessor<scalar_t, 5>(),
                                        denseOrigins.accessor<int32_t, 2>(),
                                        outSparseTensor.accessor<scalar_t, 2>(),
                                        isContiguous);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf,
                   c10::kBFloat16);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
