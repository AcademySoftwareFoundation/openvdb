// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_PACKINFOOPS_H
#define FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_PACKINFOOPS_H

#include <Types.h>
#include <detail/GridBatchImpl.h>

#include <torch/extension.h>

namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchBrickHaloBuffer(const GridBatchImpl &batchHdl, bool benchmark);

template <c10::DeviceType>
void dispatchConvolutionKernelMap(const GridBatchImpl &source, const GridBatchImpl &target,
                                  torch::Tensor &kernelMap, const Vec3iOrScalar &kernelSize,
                                  const Vec3iOrScalar &stride);

template <c10::DeviceType>
torch::Tensor dispatchBitmaskFromOutInMap(const torch::Tensor &outInMap, const int splitMaskNum,
                                          int validN);

template <c10::DeviceType>
torch::Tensor dispatchReorderOutInMap(const torch::Tensor &outInMap,
                                      const torch::Tensor &reorderLoc);

template <c10::DeviceType>
torch::Tensor dispatchReduceMask(const torch::Tensor &bitmask, const int reduceTile);

template <c10::DeviceType>
void dispatchTransposeOutInMap(const torch::Tensor &outInMap, const torch::Tensor &outInMapT);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_CONVOLUTION_PACK_INFO_PACKINFOOPS_H