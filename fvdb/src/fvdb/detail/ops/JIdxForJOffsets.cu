// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <int blockSize>
__global__ __launch_bounds__(blockSize) void
jIdxForJOffsets(TorchRAcc32<fvdb::JOffsetsType, 1> offsets,
                TorchRAcc32<fvdb::JIdxType, 1> outJIdx) {
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
    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    if (!numElements) {
        return torch::zeros({0}, options);
    }
    torch::Tensor retJIdx = torch::empty({numElements}, options);

    const int NUM_BLOCKS = GET_BLOCKS(numElements, DEFAULT_BLOCK_DIM);
    jIdxForJOffsets<DEFAULT_BLOCK_DIM><<<NUM_BLOCKS, DEFAULT_BLOCK_DIM>>>(
        joffsets.packed_accessor32<fvdb::JOffsetsType, 1, torch::RestrictPtrTraits>(),
        retJIdx.packed_accessor32<fvdb::JIdxType, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return retJIdx;
}

template <int blockSize>
__global__ __launch_bounds__(blockSize) void
jIdxFill(fvdb::JOffsetsType start,
         fvdb::JOffsetsType end,
         fvdb::JIdxType i,
         TorchRAcc32<fvdb::JIdxType, 1> outJIdx) {
    for (int64_t idx = start + blockIdx.x * blockDim.x + threadIdx.x; idx < end;
         idx += blockDim.x * gridDim.x) {
        outJIdx[idx] = i;
    }
}

template <>
torch::Tensor
dispatchJIdxForJOffsets<torch::kPrivateUse1>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");
    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    if (!numElements) {
        return torch::zeros({0}, options);
    }
    torch::Tensor retJIdx = torch::empty({numElements}, options);

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        auto deviceJOffsetsCount =
            ((joffsets.size(0) - 1) + c10::cuda::device_count() - 1) / c10::cuda::device_count();
        auto deviceJOffsetsStart = deviceId * deviceJOffsetsCount;
        auto deviceJOffsetsEnd   = (deviceId + 1) * deviceJOffsetsCount;
        deviceJOffsetsEnd        = std::min(deviceJOffsetsEnd, joffsets.size(0) - 1);
        for (auto i = deviceJOffsetsStart; i < deviceJOffsetsEnd; ++i) {
            auto start = joffsets[i].item<fvdb::JOffsetsType>();
            auto end   = joffsets[i + 1].item<fvdb::JOffsetsType>();
            if (start < end) {
                const int numBlocks = GET_BLOCKS(end - start, DEFAULT_BLOCK_DIM);
                jIdxFill<DEFAULT_BLOCK_DIM><<<numBlocks, DEFAULT_BLOCK_DIM, 0, stream>>>(
                    start,
                    end,
                    static_cast<fvdb::JIdxType>(i),
                    retJIdx.packed_accessor32<fvdb::JIdxType, 1, torch::RestrictPtrTraits>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }

    return retJIdx;
}

template <>
torch::Tensor
dispatchJIdxForJOffsets<torch::kCPU>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0,
                "Cannot call dispatchJIDxForOffsets with negative number of elements");
    auto options = torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(joffsets.device());
    if (!numElements) {
        return torch::zeros({0}, options);
    }
    std::vector<torch::Tensor> batchIdxs;
    batchIdxs.reserve(joffsets.size(0));
    for (int i = 0; i < joffsets.size(0) - 1; i += 1) {
        auto count =
            joffsets[i + 1].item<fvdb::JOffsetsType>() - joffsets[i].item<fvdb::JOffsetsType>();
        batchIdxs.push_back(torch::full({count}, i, options));
    }
    return torch::cat(batchIdxs, 0);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
