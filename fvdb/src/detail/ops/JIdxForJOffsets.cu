#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"

namespace fvdb {
namespace detail {
namespace ops {

__global__ void jIdxForJOffsets(TorchRAcc32<int64_t, 2> offsets,
                                TorchRAcc32<int16_t, 1> outJIdx) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= outJIdx.size(0)) {
        return;
    }

    int16_t jidx = 0;
    for (int i = 0; i < offsets.size(0); i += 1) {
        bool inRange = idx >= offsets[i][0] && idx < offsets[i][1];
        jidx = inRange ? i : jidx;
    }

    outJIdx[idx] = jidx;
}


template <>
torch::Tensor dispatchJIdxForJOffsets<torch::kCUDA>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0, "Cannot call dispatchJIDxForOffsets with negaive number of elements");

    if (numElements == 0) {
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt16).device(joffsets.device()));
    }
    torch::Tensor retJIdx = torch::empty({numElements}, torch::TensorOptions().dtype(torch::kInt16).device(joffsets.device()));

    const int blockSize = 1024;
    const int gridSize = (numElements + blockSize - 1) / blockSize;
    jIdxForJOffsets<<<gridSize, blockSize>>>(joffsets.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                                                retJIdx.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return retJIdx;
}

template <>
torch::Tensor dispatchJIdxForJOffsets<torch::kCPU>(torch::Tensor joffsets, int64_t numElements) {
    TORCH_CHECK(numElements >= 0, "Cannot call dispatchJIDxForOffsets with negaive number of elements");
    if (numElements == 0) {
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt16).device(joffsets.device()));
    }
    std::vector<torch::Tensor> batchIdxs;
    batchIdxs.reserve(joffsets.size(0));
    for (int i = 0; i < joffsets.size(0); i += 1) {
        batchIdxs.push_back(torch::full({joffsets[i][1].item<int64_t>() - joffsets[i][0].item<int64_t>()}, i, torch::TensorOptions().dtype(torch::kInt16).device(joffsets.device())));
    }
    return torch::cat(batchIdxs, 0);
}

}  // namespace ops
}  // namespace detail
}  // namespace fvdb