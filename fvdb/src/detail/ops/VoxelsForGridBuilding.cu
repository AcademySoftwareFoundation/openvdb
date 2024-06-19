#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

__device__ inline void copyCoords(const int16_t bidx,
                                  const int64_t base,
                                  const nanovdb::Coord& ijk0,
                                  const nanovdb::Coord& bmin,
                                  const nanovdb::Coord& bmax,
                                  TorchRAcc64<int32_t, 2> outIJK,
                                  TorchRAcc64<int16_t, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    int32_t count = 0;
    for (int di = bmin[0]; di <= bmax[0]; di += 1) {
        for (int dj = bmin[1]; dj <= bmax[1]; dj += 1) {
            for (int dk = bmin[2]; dk <= bmax[2]; dk += 1) {
                ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                outIJK[base + count][0] = ijk[0];
                outIJK[base + count][1] = ijk[1];
                outIJK[base + count][2] = ijk[2];
                outIJKBIdx[base + count] = bidx;
                count += 1;
            }
        }
    }
}

template <typename GridType>
__device__ inline void copyCoordsWithoutBorder(
                                  const typename nanovdb::DefaultReadAccessor<GridType> gridAccessor,
                                  const int16_t bidx,
                                  const int64_t base,
                                  const nanovdb::Coord& ijk0,
                                  const nanovdb::Coord& bmin,
                                  const nanovdb::Coord& bmax,
                                  const TorchRAcc64<int64_t, 1> packInfoBase,
                                  TorchRAcc64<int32_t, 2> outIJK,
                                  TorchRAcc64<int16_t, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    bool active = true;
    for (int di = bmin[0]; di <= bmax[0]; di += 1) {
        for (int dj = bmin[1]; dj <= bmax[1]; dj += 1) {
            for (int dk = bmin[2]; dk <= bmax[2]; dk += 1) {
                ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                active = active && gridAccessor.isActive(ijk);
            }
        }
    }
    if (active) {
        int64_t outBase = packInfoBase[base];
        outIJK[outBase][0] = ijk0[0];
        outIJK[outBase][1] = ijk0[1];
        outIJK[outBase][2] = ijk0[2];
        outIJKBIdx[outBase] = bidx;
    }
}

template <typename GridType>
__device__ inline void countCoordsWithoutBorder(
                                  const typename nanovdb::DefaultReadAccessor<GridType> gridAccessor,
                                  const int16_t bidx,
                                  const int64_t base,
                                  const nanovdb::Coord& ijk0,
                                  const nanovdb::Coord& bmin,
                                  const nanovdb::Coord& bmax,
                                  TorchRAcc64<int64_t, 1> outCounter) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    bool active = true;
    for (int di = bmin[0]; di <= bmax[0]; di += 1) {
        for (int dj = bmin[1]; dj <= bmax[1]; dj += 1) {
            for (int dk = bmin[2]; dk <= bmax[2]; dk += 1) {
                ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                active = active && gridAccessor.isActive(ijk);
            }
        }
    }

    outCounter[base] = active ? 1 : 0;
}

__device__ inline void copyCoords(const int16_t bidx,
                                  const int64_t base,
                                  const nanovdb::Coord size,
                                  const nanovdb::Coord& ijk0,
                                  TorchRAcc64<int32_t, 2> outIJK,
                                  TorchRAcc64<int16_t, 1> outIJKBIdx) {
    return copyCoords(bidx, base, ijk0, nanovdb::Coord(0), size - nanovdb::Coord(1), outIJK, outIJKBIdx);
}

template <typename GridType>
__device__ void convIjkForGridCallback(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                                       const GridBatchImpl::Accessor<GridType> batchAcc,
                                       const nanovdb::Coord& kernelSize, const nanovdb::Coord& stride,
                                       int kernelVolume,
                                       TorchRAcc32<int32_t, 2> outIJKData,
                                       TorchRAcc32<int16_t, 1> outIJKBIdx,
                                       TorchRAcc32<bool, 1> outMask) {

    const nanovdb::NanoGrid<GridType>* gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = gridPtr->tree().template getFirstNode<0>()[lidx];
    if (!leaf.isActive(vidx)) return;

    const nanovdb::Coord& srcIjk = leaf.offsetToGlobalCoord(vidx);
    const int64_t index = ((int64_t) leaf.getValue(vidx)) - 1;
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    int lower[3], upper[3];
    for(int i = 0; i < 3; ++i) {
        if (kernelSize[i] % 2 == 0) {
            lower[i] = 0; upper[i] = kernelSize[i] - 1;
        } else {
            lower[i] = -(kernelSize[i] - 1) / 2;
            upper[i] = (kernelSize[i] - 1) / 2;
        }
    }

    int64_t count = 0;
    for (int di = lower[0]; di <= upper[0]; di += 1) {
        for (int dj = lower[1]; dj <= upper[1]; dj += 1) {
            for (int dk = lower[2]; dk <= upper[2]; dk += 1, count += 1) {
                const nanovdb::Coord dstIjk = srcIjk + nanovdb::Coord(dk, dj, di);
                if (dstIjk[0] % stride[2] != 0 || dstIjk[1] % stride[1] != 0 || dstIjk[2] % stride[0] != 0) continue;
                //  The original torchsparse implementation has a weird bug that checks the coordsMin.
                //  if (dstIjk[0] < coordsMin[0] || dstIjk[1] < coordsMin[1] || dstIjk[2] < coordsMin[2])
                //      continue;
                // if (dstIjk[0] > coordsMax[0] || dstIjk[1] > coordsMax[1] || dstIjk[2] > coordsMax[2])
                //     continue;

                const int64_t base = (baseOffset + index) * kernelVolume + count;
                outIJKData[base][0] = dstIjk[0] / stride[2];
                outIJKData[base][1] = dstIjk[1] / stride[1];
                outIJKData[base][2] = dstIjk[2] / stride[0];
                outIJKBIdx[base] = bidx;
                outMask[base] = true;
            }
        }
    }

}


template <typename ScalarT>
__device__ void paddedIJKForPointsCallback(int32_t bidx, int32_t eidx,
                                           const JaggedRAcc32<ScalarT, 2> points,
                                           const VoxelCoordTransform* transforms,
                                           const int32_t totalPadAmount,
                                           const nanovdb::Coord bmin,
                                           const nanovdb::Coord bmax,
                                           TorchRAcc64<int32_t, 2> outIJKData,
                                           TorchRAcc64<int16_t, 1> outIJKBIdx) {
    using MathT = typename at::opmath_type<ScalarT>;
    const auto& point = points.data()[eidx];
    const VoxelCoordTransform& transform = transforms[bidx];
    const nanovdb::Coord ijk0 = transform.apply(static_cast<MathT>(point[0]),
                                                static_cast<MathT>(point[1]),
                                                static_cast<MathT>(point[2])).round();
    const int64_t base = eidx * totalPadAmount;
    copyCoords(bidx, base, ijk0, bmin, bmax, outIJKData, outIJKBIdx);
}

template <typename ScalarT>
__device__ void paddedIJKForCoordsCallback(int32_t bidx, int32_t eidx,
                                           const JaggedRAcc32<ScalarT, 2> coords,
                                           const int32_t totalPadAmount,
                                           const nanovdb::Coord bmin,
                                           const nanovdb::Coord bmax,
                                           TorchRAcc64<int32_t, 2> outIJKData,
                                           TorchRAcc64<int16_t, 1> outIJKBIdx) {
    const auto coord = coords.data()[eidx];
    const nanovdb::Coord ijk0(coord[0], coord[1], coord[2]);
    const int32_t base = eidx * totalPadAmount;
    copyCoords(bidx, base, ijk0, bmin, bmax, outIJKData, outIJKBIdx);
}


template <typename ScalarT>
__device__ void nearestNeighborIJKForPointCallback(int16_t bidx, int32_t eidx,
                                                   const JaggedRAcc32<ScalarT, 2> points,
                                                   const VoxelCoordTransform* transforms,
                                                   TorchRAcc64<int32_t, 2> outIJKData,
                                                   TorchRAcc64<int16_t, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));

    using MathT = typename at::opmath_type<ScalarT>;

    const auto pt = points.data()[eidx];
    const VoxelCoordTransform& transform = transforms[bidx];
    const nanovdb::Coord ijk0 = transform.apply(static_cast<MathT>(pt[0]),
                                                static_cast<MathT>(pt[1]),
                                                static_cast<MathT>(pt[2])).floor();
    const int32_t base = eidx * 8;
    #pragma unroll
    for (int di = 0; di <= 1; di += 1) {
        #pragma unroll
        for (int dj = 0; dj <= 1; dj += 1) {
            #pragma unroll
            for (int dk = 0; dk <= 1; dk += 1) {
                const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                const int32_t count = di * 4 + dj * 2 + dk;
                outIJKData[base + count][0] = ijk[0];
                outIJKData[base + count][1] = ijk[1];
                outIJKData[base + count][2] = ijk[2];
                outIJKBIdx[base + count] = bidx;
            }
        }
    }
}




template <typename GridType>
__device__ void ijkForGridVoxelCallback(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                                        const GridBatchImpl::Accessor<GridType> batchAcc,
                                        const nanovdb::Coord bmin, const nanovdb::Coord bmax,
                                        TorchRAcc64<int32_t, 2> outIJKData,
                                        TorchRAcc64<int16_t, 1> outIJKBIdx) {

    const nanovdb::Coord dims = bmax - bmin + nanovdb::Coord(1);
    const int32_t totalPadAmount = dims[0] * dims[1] * dims[2];

    const nanovdb::NanoGrid<GridType>* gridPtr = batchAcc.grid(bidx);
    const int64_t totalVoxels = gridPtr->activeVoxelCount();
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value = ((int64_t) leaf.getValue(vidx)) - 1;
        const int64_t base = (baseOffset + value) * totalPadAmount;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        copyCoords(bidx, base, ijk0, bmin, bmax, outIJKData, outIJKBIdx);
    }
}


template <typename GridType>
__device__ void ijkForGridVoxelCallbackWithoutBorder(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                                                     const GridBatchImpl::Accessor<GridType> batchAcc,
                                                     const nanovdb::Coord bmin, const nanovdb::Coord bmax,
                                                     const TorchRAcc64<int64_t, 1> packInfoBase,
                                                     TorchRAcc64<int32_t, 2> outIJKData,
                                                     TorchRAcc64<int16_t, 1> outIJKBIdx) {

    const nanovdb::Coord dims = bmax - bmin + nanovdb::Coord(1);

    const nanovdb::NanoGrid<GridType>* gridPtr = batchAcc.grid(bidx);
    const auto gridAccessor = gridPtr->getAccessor();
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value = ((int64_t) leaf.getValue(vidx)) - 1;
        const int64_t base = baseOffset + value;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        copyCoordsWithoutBorder<GridType>(gridAccessor, bidx, base, ijk0, bmin, bmax, packInfoBase, outIJKData, outIJKBIdx);
    }
}

template <typename GridType>
__device__ void ijkForGridVoxelCallbackWithoutBorderCount(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                                                          const GridBatchImpl::Accessor<GridType> batchAcc,
                                                          const nanovdb::Coord bmin, const nanovdb::Coord bmax,
                                                          TorchRAcc64<int64_t, 1> outCounter) {

    const nanovdb::Coord dims = bmax - bmin + nanovdb::Coord(1);

    const nanovdb::NanoGrid<GridType>* gridPtr = batchAcc.grid(bidx);
    const auto gridAccessor = gridPtr->getAccessor();
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value = ((int64_t) leaf.getValue(vidx)) - 1;
        const int64_t base = baseOffset + value;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        countCoordsWithoutBorder<GridType>(gridAccessor, bidx, base, ijk0, bmin, bmax, outCounter);
    }
}


template <typename GridType>
__device__ void coarseIjkForFineGridVoxelCallback(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                                                  const GridBatchImpl::Accessor<GridType> batchAcc,
                                                  nanovdb::Coord coarseningFactor,
                                                  TorchRAcc64<int32_t, 2> outIJKData,
                                                  TorchRAcc64<int16_t, 1> outIJKBIdx) {

    const nanovdb::NanoGrid<GridType>* gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value = ((int64_t) leaf.getValue(vidx)) - 1;
        const int64_t index = (baseOffset + value);
        const nanovdb::Coord fineIjk = leaf.offsetToGlobalCoord(vidx);
        const nanovdb::Coord coarseIjk = (fineIjk.asVec3d() / coarseningFactor.asVec3d()).floor();
        outIJKData[index][0] = coarseIjk[0];
        outIJKData[index][1] = coarseIjk[1];
        outIJKData[index][2] = coarseIjk[2];
        outIJKBIdx[index] = bidx;
    }
}


template <typename GridType>
__device__ void fineIjkForCoarseGridVoxelCallback(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                                                  const GridBatchImpl::Accessor<GridType> batchAcc,
                                                  nanovdb::Coord upsamplingFactor,
                                                  TorchRAcc64<int32_t, 2> outIJKData,
                                                  TorchRAcc64<int16_t, 1> outIJKBIdx) {

    const nanovdb::NanoGrid<GridType>* gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);
    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];
    if (leaf.isActive(vidx)) {
        const int64_t value = ((int64_t) leaf.getValue(vidx)) - 1;
        const int64_t index = (baseOffset + value) * totalPadAmount;
        const nanovdb::Coord coarseIjk = leaf.offsetToGlobalCoord(vidx);
        const nanovdb::Coord fineIjk(coarseIjk[0] * upsamplingFactor[0], coarseIjk[1] * upsamplingFactor[1], coarseIjk[2] * upsamplingFactor[2]);
        copyCoords(bidx, index, upsamplingFactor, fineIjk, outIJKData, outIJKBIdx);
    }
}



template <>
JaggedTensor dispatchFineIJKForCoarseGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                        nanovdb::Coord upsamplingFactor,
                                                        const torch::optional<JaggedTensor>& maybeMask) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(batchHdl.device());
    torch::Tensor outIJK = torch::empty({batchHdl.totalVoxels() * totalPadAmount, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * totalPadAmount}, optsBIdx); // TODO: Don't populate for single batch

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&] {
        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> bacc) {
            fineIjkForCoarseGridVoxelCallback<GridType>(bidx, lidx, vidx, cidx, bacc, upsamplingFactor, outIJKAcc, outIJKBIdxAcc);
        };

        forEachVoxelCUDA<GridType>(1024, 1, batchHdl, cb);
    });

    // FIXME: (Francis) this uses a bunch of extra memory. Maybe we can avoid it in the future by allowing for
    // invalid values in the nanovdb GPU grid building process
    if (maybeMask.has_value()) {
        std::vector<torch::Tensor> stack;
        stack.reserve(totalPadAmount);
        for (int i = 0; i < totalPadAmount; ++i) {
            stack.push_back(maybeMask.value().jdata());
        }
        torch::Tensor mask = torch::stack(stack, 1).view({-1});
        outIJK = outIJK.index({mask}).contiguous();
        outIJKBIdx = outIJKBIdx.index({mask}).contiguous();
        return JaggedTensor::from_data_and_jidx(outIJK, outIJKBIdx, batchHdl.batchSize());
    }

    return JaggedTensor::from_data_and_offsets(outIJK, batchHdl.voxelOffsets(true) * totalPadAmount);
}


template <>
JaggedTensor dispatchCoarseIJKForFineGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                        nanovdb::Coord coarseningFactor) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(batchHdl.device());
    torch::Tensor outIJK = torch::empty({batchHdl.totalVoxels(), 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels()}, optsBIdx);  // TODO: Don't populate for single batch

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&] {
        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> bacc) {
            coarseIjkForFineGridVoxelCallback<GridType>(bidx, lidx, vidx, cidx, bacc, coarseningFactor, outIJKAcc, outIJKBIdxAcc);
        };

        forEachVoxelCUDA<GridType>(1024, 1, batchHdl, cb);
    });

    return JaggedTensor::from_data_and_offsets(outIJK, batchHdl.voxelOffsets(true));
}


template <>
JaggedTensor dispatchPaddedIJKForGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                    const nanovdb::Coord& bmin,
                                                    const nanovdb::Coord& bmax) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const nanovdb::Coord dims = bmax - bmin + nanovdb::Coord(1);
    const int32_t totalPadAmount = dims[0] * dims[1] * dims[2];

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(batchHdl.device());
    torch::Tensor outIJK = torch::empty({batchHdl.totalVoxels() * totalPadAmount, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * totalPadAmount}, optsBIdx);  // TODO: Don't populate for single batch

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&] {
        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> bacc) {
            ijkForGridVoxelCallback<GridType>(bidx, lidx, vidx, cidx, bacc, bmin, bmax, outIJKAcc, outIJKBIdxAcc);
        };
        forEachVoxelCUDA<GridType>(1024, 1, batchHdl, cb);
    });

    return JaggedTensor::from_data_and_offsets(outIJK, batchHdl.voxelOffsets(true) * totalPadAmount);
}


template <>
JaggedTensor dispatchPaddedIJKForGridWithoutBorder<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                 const nanovdb::Coord& bmin,
                                                                 const nanovdb::Coord& bmax) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const nanovdb::Coord dims = bmax - bmin + nanovdb::Coord(1);

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(batchHdl.device());
    const torch::TensorOptions optsCounter = torch::TensorOptions().dtype(torch::kInt64).device(batchHdl.device());

    torch::Tensor outCounter = torch::empty({batchHdl.totalVoxels()}, optsCounter);
    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&] {
        auto outCounterAcc = outCounter.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
        auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> bacc) {
            ijkForGridVoxelCallbackWithoutBorderCount<GridType>(bidx, lidx, vidx, cidx, bacc, bmin, bmax, outCounterAcc);
        };
        forEachVoxelCUDA<GridType>(512, 1, batchHdl, cb);
    });
    torch::Tensor cumCounts = torch::cumsum(outCounter, 0);
    int64_t numVoxels = cumCounts[-1].item<int64_t>();
    torch::Tensor packInfoBase = cumCounts - outCounter;

    torch::Tensor outIJK = torch::empty({numVoxels, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({numVoxels}, optsBIdx);  // TODO: Don't populate for single batch
    if (numVoxels == 0) {
        // TODO(ruilong): Shall we raise error? Do we support empty grid?
        return JaggedTensor::from_data_and_jidx(outIJK, outIJKBIdx, batchHdl.batchSize());
    }

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&] {
        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();
        auto packInfoBaseAcc = packInfoBase.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> bacc) {
            ijkForGridVoxelCallbackWithoutBorder<GridType>(bidx, lidx, vidx, cidx, bacc, bmin, bmax, packInfoBaseAcc, outIJKAcc, outIJKBIdxAcc);
        };
        forEachVoxelCUDA<GridType>(512, 1, batchHdl, cb);
    });

    return JaggedTensor::from_data_and_jidx(outIJK, outIJKBIdx, batchHdl.batchSize());
}


template <>
JaggedTensor dispatchPaddedIJKForPoints<torch::kCUDA>(const JaggedTensor& jaggedPoints,
                                                      const nanovdb::Coord& bmin,
                                                      const nanovdb::Coord& bmax,
                                                      const std::vector<VoxelCoordTransform>& transforms) {
    TORCH_CHECK(jaggedPoints.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(jaggedPoints.device().has_index(), "GridBatchImpl must have a valid index");

    const nanovdb::Coord dims = bmax - bmin + nanovdb::Coord(1);
    const int32_t totalPadAmount = dims[0] * dims[1] * dims[2];

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(jaggedPoints.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(jaggedPoints.device());
    torch::Tensor outIJK = torch::empty({jaggedPoints.jdata().size(0) * totalPadAmount, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({jaggedPoints.jdata().size(0) * totalPadAmount}, optsBIdx);  // TODO: Don't populate for single batch

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(jaggedPoints.scalar_type(), "paddedIJKForPoints", [&] {
        RAIIRawDeviceBuffer<VoxelCoordTransform> transformsDVec(transforms.size(), jaggedPoints.device());
        transformsDVec.setData((VoxelCoordTransform*) transforms.data(), true /* blocking */);
        const VoxelCoordTransform* transformDevPtr = transformsDVec.devicePtr;

        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pacc) {
            paddedIJKForPointsCallback(bidx, eidx, pacc, transformDevPtr, totalPadAmount, bmin, bmax, outIJKAcc, outIJKBIdxAcc);
        };
        forEachJaggedElementChannelCUDA<scalar_t, 2>(1024, 1, jaggedPoints, cb);
    });
    return JaggedTensor::from_data_and_offsets(outIJK, jaggedPoints.joffsets() * totalPadAmount);
}


template <>
JaggedTensor dispatchPaddedIJKForCoords<torch::kCUDA>(const JaggedTensor& jaggedCoords,
                                                      const nanovdb::Coord& bmin,
                                                      const nanovdb::Coord& bmax) {
    TORCH_CHECK(jaggedCoords.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(jaggedCoords.device().has_index(), "GridBatchImpl must have a valid index");

    const nanovdb::Coord dims = bmax - bmin + nanovdb::Coord(1);
    const int32_t totalPadAmount = dims[0] * dims[1] * dims[2];

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(jaggedCoords.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(jaggedCoords.device());

    torch::Tensor outIJK = torch::empty({jaggedCoords.jdata().size(0) * totalPadAmount, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({jaggedCoords.jdata().size(0) * totalPadAmount}, optsBIdx);  // TODO: Don't populate for single batch

    AT_DISPATCH_INTEGRAL_TYPES(jaggedCoords.scalar_type(), "paddedIJKForCoords", [&] {
        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> cacc) {
            paddedIJKForCoordsCallback(bidx, eidx, cacc, totalPadAmount, bmin, bmax, outIJKAcc, outIJKBIdxAcc);
        };

        forEachJaggedElementChannelCUDA<scalar_t, 2>(256, 1, jaggedCoords, cb);
    });

    return JaggedTensor::from_data_and_offsets(outIJK, jaggedCoords.joffsets() * totalPadAmount);
}


template <>
JaggedTensor dispatchNearestNeighborIJKForPoints<torch::kCUDA>(const JaggedTensor& jaggedPoints,
                                                               const std::vector<VoxelCoordTransform>& transforms) {
    TORCH_CHECK(jaggedPoints.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(jaggedPoints.device().has_index(), "GridBatchImpl must have a valid index");

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(jaggedPoints.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(jaggedPoints.device());
    torch::Tensor outIJK = torch::empty({jaggedPoints.jdata().size(0) * 8, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({jaggedPoints.jdata().size(0) * 8}, optsBIdx);  // TODO: Don't populate for single batch

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(jaggedPoints.scalar_type(), "nearestNeighborIJKForPoints", [&] {
        RAIIRawDeviceBuffer<VoxelCoordTransform> transformsDVec(transforms.size(), jaggedPoints.device());
        transformsDVec.setData((VoxelCoordTransform*) transforms.data(), true /* blocking */);
        const VoxelCoordTransform* transformDevPtr = transformsDVec.devicePtr;

        auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor64<int16_t, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pacc) {
            nearestNeighborIJKForPointCallback(bidx, eidx, pacc, transformDevPtr, outIJKAcc, outIJKBIdxAcc);
        };

        forEachJaggedElementChannelCUDA<scalar_t, 2>(256, 1, jaggedPoints, cb);
    });

    return JaggedTensor::from_data_and_offsets(outIJK, jaggedPoints.joffsets() * 8);
}

template <>
JaggedTensor dispatchConvIJKForGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                  const nanovdb::Coord& kernelSize,
                                                  const nanovdb::Coord& stride) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    if (stride == nanovdb::Coord(1) || stride == kernelSize) {
        return dispatchCoarseIJKForFineGrid<torch::kCUDA>(batchHdl, nanovdb::Coord(stride));
    }

    const int32_t kernelVolume = kernelSize.x() * kernelSize.y() * kernelSize.z();

    const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx = torch::TensorOptions().dtype(torch::kInt16).device(batchHdl.device());
    const torch::TensorOptions optsMask = torch::TensorOptions().dtype(torch::kBool).device(batchHdl.device());
    torch::Tensor outIJK = torch::empty({batchHdl.totalVoxels() * kernelVolume, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * kernelVolume}, optsBIdx);
    torch::Tensor outMask = torch::zeros({batchHdl.totalVoxels() * kernelVolume}, optsMask);

    // For each voxel in source grid, compute possible voxels in target grid that affect them
    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&] {
        auto outIJKAcc = outIJK.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();
        auto outIJKBIdxAcc = outIJKBIdx.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>();
        auto outMaskAcc = outMask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>();

        auto cb = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
                GridBatchImpl::Accessor<GridType> bacc) {
            convIjkForGridCallback<GridType>(
                    bidx, lidx, vidx, cidx, bacc, kernelSize, stride, kernelVolume,
                    outIJKAcc, outIJKBIdxAcc, outMaskAcc);
        };
        forEachVoxelCUDA<GridType>(256, 1, batchHdl, cb);
    });

    outIJK = outIJK.index({outMask});
    outIJKBIdx = outIJKBIdx.index({outMask});

    return JaggedTensor::from_data_and_jidx(outIJK, outIJKBIdx, batchHdl.batchSize());
}

} // namespace ops
} // namesapce detail
} // namespace fvdb
