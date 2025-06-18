// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_JAGGED_JAGGEDOPS_H
#define FVDB_DETAIL_OPS_JAGGED_JAGGEDOPS_H

#include <fvdb/JaggedTensor.h>
#include <fvdb/Types.h>
#include <fvdb/detail/utils/Utils.h>

#include <torch/extension.h>

namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType> torch::Tensor dispatchJaggedArgsort(const JaggedTensor &jt);

template <c10::DeviceType>
torch::Tensor dispatchJaggedSum(const torch::Tensor &jdata,
                                const torch::Tensor &jidx,
                                const torch::Tensor &joffsets,
                                int64_t dimSize);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchJaggedMin(const torch::Tensor &jdata,
                                             const torch::Tensor &jidx,
                                             const torch::Tensor &joffsets,
                                             int64_t dimSize);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchJaggedMax(const torch::Tensor &jdata,
                                             const torch::Tensor &jidx,
                                             const torch::Tensor &joffsets,
                                             int64_t dimSize);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_JAGGED_JAGGEDOPS_H
