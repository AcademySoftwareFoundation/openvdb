// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/cuda/Utils.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

__global__ void
__launch_bounds__(1024)
    jIdxForJOffsetsChild(fvdb::JOffsetsType start, fvdb::JOffsetsType end, fvdb::JIdxType i,
                         TorchRAcc32<fvdb::JIdxType, 1> outJIdx) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (start + idx >= end) {
        return;
    }

    outJIdx[start + idx] = i;
}

__global__ void
__launch_bounds__(256) jIdxForJOffsetsParent(const TorchRAcc32<fvdb::JOffsetsType, 1> offsets,
                                             TorchRAcc32<fvdb::JIdxType, 1>           outJIdx) {
    const int32_t batchIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx >= offsets.size(0) - 1) {
        return;
    }

    auto count = offsets[batchIdx + 1] - offsets[batchIdx];
    if (!count) {
        return;
    }

    constexpr int NUM_CHILD_THREADS = 1024;
    const int     numChildBlocks    = GET_BLOCKS(count, NUM_CHILD_THREADS);
    jIdxForJOffsetsChild<<<numChildBlocks, NUM_CHILD_THREADS>>>(
        offsets[batchIdx], offsets[batchIdx + 1], static_cast<fvdb::JIdxType>(batchIdx), outJIdx);
}

template <>
torch::Tensor
dispatchJIdxForJOffsets<torch::kCUDA>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");

    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    torch::Tensor retJIdx = torch::empty({ numElements }, options);
    if (!numElements) {
        return retJIdx;
    }

    constexpr int NUM_PARENT_THREADS = 256;
    const int     numParentBlocks    = GET_BLOCKS(joffsets.size(0) - 1, NUM_PARENT_THREADS);
    jIdxForJOffsetsParent<<<numParentBlocks, NUM_PARENT_THREADS>>>(
        joffsets.packed_accessor32<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>(),
        retJIdx.packed_accessor32<fvdb::JIdxType, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return retJIdx;
}

template <>
torch::Tensor
dispatchJIdxForJOffsets<torch::kCPU>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");

    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    if (!numElements) {
        return torch::empty({ numElements }, options);
    }

    std::vector<torch::Tensor> batchIdxs;
    batchIdxs.reserve(joffsets.size(0));
    for (int i = 0; i < joffsets.size(0) - 1; i += 1) {
        batchIdxs.push_back(torch::full(
            { joffsets[i + 1].item<fvdb::JOffsetsType>() - joffsets[i].item<fvdb::JOffsetsType>() },
            i, options));
    }
    return torch::cat(batchIdxs, 0);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
