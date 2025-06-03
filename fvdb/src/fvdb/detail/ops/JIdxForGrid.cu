// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/ops/Ops.h>

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <>
torch::Tensor
dispatchJIdxForGrid<torch::kCPU>(const GridBatchImpl &gridBatch) {
    return ops::dispatchJIdxForJOffsets<torch::kCPU>(gridBatch.voxelOffsets(),
                                                     gridBatch.totalVoxels());
}

template <>
torch::Tensor
dispatchJIdxForGrid<torch::kCUDA>(const GridBatchImpl &gridBatch) {
    return ops::dispatchJIdxForJOffsets<torch::kCUDA>(gridBatch.voxelOffsets(),
                                                      gridBatch.totalVoxels());
}

} // namespace ops
} // namespace detail
} // namespace fvdb
