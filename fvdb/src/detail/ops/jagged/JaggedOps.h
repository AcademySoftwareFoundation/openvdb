#pragma once

#include <torch/extension.h>

#include "detail/utils/Utils.h"
#include "Types.h"


namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType>
torch::Tensor dispatchJaggedArgsort(const JaggedTensor& jt);

template <c10::DeviceType>
torch::Tensor dispatchJaggedSum(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchJaggedMin(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize);

template <c10::DeviceType>
std::vector<torch::Tensor> dispatchJaggedMax(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize);

} // namespace ops
} // namespace detail
} // namespace fvdb
