// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include "detail/ops/Ops.h"
#include "detail/utils/cuda/Utils.cuh"

#include <cub/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>


namespace fvdb {
namespace detail {
namespace ops {


template <typename T>
__global__ void setZero(T* thingToSet) {
    *thingToSet = 0;
}

template <>
torch::Tensor dispatchJOffsetsForJIdx<torch::kCUDA>(torch::Tensor jidx, torch::Tensor jdata, int64_t numTensors) {
    TORCH_CHECK_VALUE(jdata.device().is_cuda(), "Invalid device for jdata");

    if (jidx.size(0) == 0 && numTensors == 1) {
        torch::Tensor ret = torch::empty({2}, JOffsetsScalarType);
        auto acc = ret.accessor<JOffsetsType, 1>();
        acc[0] = 0;
        acc[1] = jdata.size(0);
        return ret.to(jdata.device());
    }

    TORCH_CHECK_VALUE(jidx.device().is_cuda(), "Invalid device for jidx");
    TORCH_CHECK_VALUE(jidx.scalar_type() == JIdxScalarType, "Invalid scalar type for jidx. Got ", jidx.scalar_type(), " but expected ", JIdxScalarType);
    TORCH_CHECK_VALUE(jidx.is_contiguous(), "jidx must be contiguous");
    TORCH_CHECK_VALUE(jidx.size(0) == jdata.size(0), "jidx and jdata must have the same number of elments");

    const size_t numItems = jidx.size(0);

    // FIXME: Francis -- write a dummy output iterator so we don't actually allocate here.
    torch::Tensor dummyOut = torch::empty({numTensors}, torch::TensorOptions().dtype(JIdxScalarType).device(jdata.device()));

    torch::Tensor joffsetsOut = torch::empty({numTensors + 1}, torch::TensorOptions().dtype(JIdxScalarType).device(jdata.device()));
    torch::Tensor numRunsOut = torch::empty({1}, torch::TensorOptions().dtype(JIdxScalarType).device(jdata.device()));

    // Get current cuda stream for device
    at::cuda::CUDAStream currentStream = at::cuda::getCurrentCUDAStream(jdata.device().index());

    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        jidx.data_ptr<JIdxType>(), // keys in
        dummyOut.data_ptr<JIdxType>(), // unique out (dummy)
        joffsetsOut.data_ptr<JIdxType>() + 1, // counts out
        numRunsOut.data_ptr<JIdxType>(), // num runs out
        numItems, currentStream.stream());

    // Allocate temporary storage
    d_temp_storage = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(temp_storage_bytes, currentStream.stream());

    // Do the actual reduce by key operation
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        jidx.data_ptr<JIdxType>(), // keys in
        dummyOut.data_ptr<JIdxType>(), // values out (dummy)
        joffsetsOut.data_ptr<JIdxType>() + 1, // unique out
        numRunsOut.data_ptr<JIdxType>(), // num runs out
        numItems, currentStream.stream());


    // Free up scratch memory
    c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);

    // Zero out the first element
    setZero<<<1, 1>>>(joffsetsOut.data_ptr<JIdxType>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return joffsetsOut.cumsum(0, JOffsetsScalarType);
}

template <>
torch::Tensor dispatchJOffsetsForJIdx<torch::kCPU>(torch::Tensor jidx, torch::Tensor jdata, int64_t numTensors) {

    TORCH_CHECK_VALUE(jidx.dim() == 1, "jidx must be a 1D tensor");
    TORCH_CHECK_VALUE(jdata.device().is_cpu(), "Invalid device for jdata");

    if (jidx.size(0) == 0 && numTensors == 1) {
        torch::Tensor ret = torch::empty({2}, JOffsetsScalarType);
        auto acc = ret.accessor<JOffsetsType, 1>();
        acc[0] = 0;
        acc[1] = jdata.size(0);
        return ret.to(jdata.device());
    }

    TORCH_CHECK_VALUE(jidx.device().is_cpu(), "Invalid device for jidx");

    // Get the number of unique batch indices assuming jidx is always sorted
    // It should be of the form [0, ..., 0, 1, ..., 1, 3, ..., 3, ...]
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> uniqueRes = torch::unique_dim(jidx, 0, false, false, true);
    torch::Tensor uniqueBatchValues = std::get<0>(uniqueRes);  // [0, 1, 3, ...]
    torch::Tensor uniqueBatchCounts = std::get<2>(uniqueRes);  // [n0, n1, n3, ...]

    torch::Tensor fullBatchCounts = torch::full(
            {numTensors + 1}, 0, torch::TensorOptions().dtype(JOffsetsScalarType).device(jdata.device()));
    fullBatchCounts.index({torch::indexing::Slice(1, torch::indexing::None, 1)}).index_put_({uniqueBatchValues.to(torch::kLong)}, uniqueBatchCounts);

    torch::Tensor cumOffsets = torch::cumsum(fullBatchCounts, 0, JOffsetsScalarType);
    return cumOffsets;
}

} // namespace ops
} // namespace detail
} // namespace fvdb

