// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/cuda/Utils.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

__global__ void
jIdxForJOffsets(TorchRAcc32<fvdb::JOffsetsType, 1> offsets,
                TorchRAcc32<fvdb::JIdxType, 1>     outJIdx) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= outJIdx.size(0)) {
        return;
    }

    fvdb::JIdxType left = 0, right = offsets.size(0) - 1;

    while (left <= right) {
        fvdb::JIdxType mid = left + (right - left) / 2;

        // Check if key is present at mid
        if (idx >= offsets[mid] && idx < offsets[mid + 1]) {
            outJIdx[idx] = mid;
            return;
        }

        if (offsets[mid] < idx) {
            // If key greater, ignore left half
            left = mid + 1;
        } else {
            // If key is smaller, ignore right half
            right = mid - 1;
        }
    }

    outJIdx[idx] = -1;
}

template <>
torch::Tensor
dispatchJIdxForJOffsets<torch::kCUDA>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");

    if (numElements == 0) {
        return torch::zeros(
            { 0 }, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device()));
    }
    torch::Tensor retJIdx =
        torch::empty({ numElements },
                     torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device()));

    const int NUM_THREADS = 1024;
    const int NUM_BLOCKS  = GET_BLOCKS(numElements, NUM_THREADS);
    jIdxForJOffsets<<<NUM_BLOCKS, NUM_THREADS>>>(
        joffsets.packed_accessor32<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>(),
        retJIdx.packed_accessor32<fvdb::JIdxType, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return retJIdx;
}

template <>
torch::Tensor
dispatchJIdxForJOffsets<torch::kCPU>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negaive number of elements");
    if (numElements == 0) {
        return torch::zeros(
            { 0 }, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device()));
    }
    std::vector<torch::Tensor> batchIdxs;
    batchIdxs.reserve(joffsets.size(0));
    for (int i = 0; i < joffsets.size(0) - 1; i += 1) {
        batchIdxs.push_back(torch::full(
            { joffsets[i + 1].item<fvdb::JOffsetsType>() - joffsets[i].item<fvdb::JOffsetsType>() },
            i, torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device())));
    }
    return torch::cat(batchIdxs, 0);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
