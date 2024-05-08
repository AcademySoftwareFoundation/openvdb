#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

#include "PackInfoOps.h"
#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

__global__ void bitmaskFromOutInMapCallback(TorchRAcc32<int, 2> outInMap, TorchRAcc32<int, 2> bitmask,
                                            int validN, int kernelVolume, int splitMaskNum) {

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tidx / splitMaskNum;
    if (idx >= validN) return;

    int splitMaskIter = tidx % splitMaskNum;
    int splitMaskLen = (kernelVolume + splitMaskNum - 1) / splitMaskNum;
    if (splitMaskIter == (splitMaskNum - 1))
        splitMaskLen = kernelVolume - splitMaskIter * splitMaskLen;

    int curBitmask = 0;
    // Compress subrows into bitmask
    for (int i = 0; i < splitMaskLen; i++) {
        curBitmask += (int)(outInMap[idx][splitMaskIter * splitMaskLen + i] >= 0) * (int)(1u << i);
    }
    bitmask[splitMaskIter][idx] = curBitmask;

}

__global__ void __launch_bounds__(128) reorderOutInMapCallback(TorchRAcc32<int, 2> outInMap,
                                                               TorchRAcc32<int, 2> reorderLoc,
                                                               TorchRAcc32<int, 2> reorderOutInMap,
                                                               int kernelVolume, int splitMaskLen) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tidx / kernelVolume;
    int splitMaskIter = tidx % kernelVolume;
    if (idx >= outInMap.size(0)) return;

    int inputRowIdx = reorderLoc[splitMaskIter / splitMaskLen][idx];
    reorderOutInMap[idx][splitMaskIter] = outInMap[inputRowIdx][splitMaskIter];
}

__global__ void __launch_bounds__(128) reduceMaskCallback(TorchRAcc32<int, 2> bitmask,
                                                          int reduceTile,
                                                          TorchRAcc32<int, 2> reducedBitmask) {
    int splitMaskIter = blockIdx.y;
    int threadSize = reduceTile / 4;
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;

    int bitmaskLocal = 0;
    __shared__ int bitmaskShared[128];
    int* finalReducePtr = bitmaskShared + (laneIdx << 2);
    int blockOffset = blockIdx.x * 128 * threadSize;
    int threadOffset = blockOffset + (threadIdx.x * threadSize);

    int loadLen = min(threadSize, bitmask.size(1) - threadOffset);

    #pragma unroll
    for (int i = 0; i < loadLen; i++) {
        int loadOffset = i + threadOffset;
        bitmaskLocal = bitmaskLocal | bitmask[splitMaskIter][loadOffset];
    }
    bitmaskShared[threadIdx.x] = bitmaskLocal;
    __syncthreads();

    // Reduce for 1st warp
    if (warpIdx == 0) {
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            finalReducePtr[0] = finalReducePtr[0] | finalReducePtr[i];
        }
        int outputOffset = (blockIdx.x << 5) + laneIdx;
        if (outputOffset < reducedBitmask.size(1)) {
            reducedBitmask[splitMaskIter][outputOffset] = finalReducePtr[0];
        }
    }

}

__global__ void transposeOutInMapCallback(TorchRAcc32<int, 2> outInMap, TorchRAcc32<int, 2> outInMapT) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int outIdx = tidx / outInMap.size(1);
    int kernelIdx = tidx % outInMap.size(1);

    if (outIdx >= outInMap.size(0)) return;
    int inIdx = outInMap[outIdx][kernelIdx];
    if (inIdx < 0) return;

    outInMapT[inIdx][kernelIdx] = outIdx;
}

template <>
torch::Tensor dispatchBitmaskFromOutInMap<torch::kCUDA>(const torch::Tensor& outInMap, const int splitMaskNum, int validN) {
    torch::Tensor bitmask = torch::full({splitMaskNum, outInMap.size(0)}, -1,
        torch::device(outInMap.device()).dtype(torch::ScalarType::Int));
    if (splitMaskNum > 0 && validN > 0) {
        bitmaskFromOutInMapCallback<<<(splitMaskNum * outInMap.size(0) + 255) / 256, 256>>>(
            outInMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bitmask.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            validN, outInMap.size(1), splitMaskNum);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return bitmask;
}

template <>
torch::Tensor dispatchReorderOutInMap<torch::kCUDA>(const torch::Tensor& outInMap, const torch::Tensor& reorderLoc) {
    int splitMaskNum = reorderLoc.size(0);
    int splitMaskLen = (outInMap.size(1) + splitMaskNum - 1) / splitMaskNum;
    torch::Tensor reorderOutInMap = torch::empty({outInMap.size(0), outInMap.size(1)},
        torch::device(outInMap.device()).dtype(torch::ScalarType::Int));

    if (outInMap.size(0) > 0 && outInMap.size(1) > 0) {
        reorderOutInMapCallback<<<(outInMap.size(0) * outInMap.size(1) + 127) / 128, 128>>>(
            outInMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            reorderLoc.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            reorderOutInMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            outInMap.size(1), splitMaskLen);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return reorderOutInMap;
}

template <>
torch::Tensor dispatchReduceMask<torch::kCUDA>(const torch::Tensor& bitmask, const int reduceTile) {
    TORCH_CHECK(reduceTile % 4 == 0, "Reduce tile should be multiple of 4")
    int reducedRowNum = (bitmask.size(1) + reduceTile - 1) / reduceTile;

    torch::Tensor reducedBitmask = torch::zeros({bitmask.size(0), reducedRowNum},
        torch::device(bitmask.device()).dtype(torch::ScalarType::Int));

    if (bitmask.size(0) > 0 && bitmask.size(1) > 0) {
        reduceMaskCallback<<<dim3((reducedRowNum + 31) / 32, reducedBitmask.size(0)), 128>>>(
            bitmask.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            reduceTile,
            reducedBitmask.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return reducedBitmask;
}

template <>
void dispatchTransposeOutInMap<torch::kCUDA>(const torch::Tensor& outInMap, const torch::Tensor& outInMapT) {
    if (outInMap.size(0) > 0 && outInMap.size(1) > 0) {
        transposeOutInMapCallback<<<(outInMap.size(0) * outInMap.size(1) + 255) / 256, 256>>>(
            outInMap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            outInMapT.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
