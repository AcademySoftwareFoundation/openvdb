// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "nanovdb/NanoVDB.h"

#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/ForEachCPU.h>
#include <detail/utils/cuda/ForEachCUDA.cuh>
#include <detail/utils/cuda/RAIIRawDeviceBuffer.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb::detail::ops {

__device__ void
coarseIjkForFineGridVoxelCallback(int32_t bidx,
                                  int32_t lidx,
                                  int32_t vidx,
                                  int32_t cidx,
                                  const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> batchAcc,
                                  nanovdb::Coord coarseningFactor,
                                  TorchRAcc64<int32_t, 2> outIJKData,
                                  TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value            = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t index            = (baseOffset + value);
        const nanovdb::Coord fineIjk   = leaf.offsetToGlobalCoord(vidx);
        const nanovdb::Coord coarseIjk = (fineIjk.asVec3d() / coarseningFactor.asVec3d()).floor();
        outIJKData[index][0]           = coarseIjk[0];
        outIJKData[index][1]           = coarseIjk[1];
        outIJKData[index][2]           = coarseIjk[2];
        outIJKBIdx[index]              = bidx;
    }
}

template <>
JaggedTensor
dispatchCoarseIJKForFineGrid<torch::kCUDA>(const GridBatchImpl &batchHdl,
                                           nanovdb::Coord coarseningFactor) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK = torch::empty({batchHdl.totalVoxels(), 3}, optsData);
    torch::Tensor outIJKBIdx =
        torch::empty({batchHdl.totalVoxels()}, optsBIdx); // TODO: Don't populate for single batch

    auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

    auto cb = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchImpl::Accessor<nanovdb::ValueOnIndex> bacc) {
        coarseIjkForFineGridVoxelCallback(
            bidx, lidx, vidx, cidx, bacc, coarseningFactor, outIJKAcc, outIJKBIdxAcc);
    };

    forEachVoxelCUDA<nanovdb::ValueOnIndex>(1024, 1, batchHdl, cb);

    return JaggedTensor::from_data_offsets_and_list_ids(
        outIJK, batchHdl.voxelOffsets(), batchHdl.jlidx());
}

} // namespace fvdb::detail::ops
