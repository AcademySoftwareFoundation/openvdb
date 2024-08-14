// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <ATen/cuda/Atomic.cuh>

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"
#include "detail/utils/cuda/Utils.cuh"

namespace fvdb {
namespace detail {
namespace ops {

// __global__ void makeJOffsetsForListJt(const TorchRAcc32<JOffsetsType, 1> inJoffsets, int64_t idxVal,
//                                       TorchRAcc32<JOffsetsType, 1> outJoffsets) {
//     JOffsetsType startIdx = inJoffsets[idxVal];
//     JOffsetsType endIdx = inJoffsets[idxVal + 1];
//     outJoffsets[0] = 0;
//     outJoffsets[1] = endIdx - startIdx;
// }


__global__ void getJOffsetsMask(const int64_t idxVal,
                                const TorchRAcc32<JLIdxType, 2> jlidx,
                                const TorchRAcc32<JOffsetsType, 1> inJoffsets,
                                TorchRAcc32<JOffsetsType, 1> offsetsAndRange) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= jlidx.size(0)) {
        return;
    }

    JLIdxType lid = jlidx[idx][0];
    JLIdxType prevLid = -1;
    if (idx - 1 >= 0) {
        prevLid = jlidx[idx - 1][0];
    }
    const bool lidMatches = lid == idxVal;
    const bool prevLidMatches = prevLid == idxVal;
    const bool isLastIdx = idx == (jlidx.size(0) - 1);

    if (lidMatches && !prevLidMatches) {
        offsetsAndRange[0] = inJoffsets[idx];
        offsetsAndRange[2] = idx;
    }

    if (!lidMatches && prevLidMatches) {
        offsetsAndRange[1] = inJoffsets[idx];
        offsetsAndRange[3] = idx;
    } else if (lidMatches && isLastIdx) {
        offsetsAndRange[1] = inJoffsets[idx + 1];
        offsetsAndRange[3] = idx + 1;
    }
}


// __global__ void computeJLidx(const int64_t startIdx, const int64_t idxVal,
//                              const TorchRAcc32<JLIdxType, 2> inJLIdx,
//                              TorchRAcc32<JLIdxType, 2> outJLidx) {
//     int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

//     if (idx >= outJLidx.size(0)) {
//         return;
//     }
//     outJLidx[idx][0] = inJLIdx[idx + startIdx][0] - idxVal;
//     outJLidx[idx][1] = inJLIdx[idx + startIdx][1];
// }


JaggedTensor jaggedTensorIndexMultiListCuda(const JaggedTensor& jt, int64_t idxVal) {
    if (idxVal < 0) {
        idxVal += jt.num_outer_lists();
    }
    TORCH_CHECK_INDEX(idxVal >= 0 && idxVal < jt.num_outer_lists(),
                      "Index ", idxVal, " is out of bounds for JaggedTensor with ",
                      jt.num_outer_lists(), " elements");

    torch::Tensor joffsets = jt.joffsets();
    torch::Tensor jdata = jt.jdata();
    torch::Tensor jlidx = jt.jlidx();

    TORCH_CHECK_VALUE(jlidx.dim() == 2, "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(jlidx.numel() == 0 || jlidx.size(0) == (joffsets.size(0) - 1), "Corrupt list indices. This should never happen");

    torch::Tensor offsetsAndRange = torch::empty({4}, torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU).pinned_memory(true));
    offsetsAndRange = offsetsAndRange.to(jt.device());
    auto inJLidxAcc = jlidx.packed_accessor32<JLIdxType, 2, torch::RestrictPtrTraits>();
    auto inJOffsetsAcc = joffsets.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>();
    auto offsetsAndRangeAcc = offsetsAndRange.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>();
    const int numBlocks = GET_BLOCKS(joffsets.size(0), 1024);
    getJOffsetsMask<<<numBlocks, 1024>>>(idxVal, inJLidxAcc, inJOffsetsAcc, offsetsAndRangeAcc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    offsetsAndRange = offsetsAndRange.cpu();
    const JOffsetsType elementStartOffset = offsetsAndRange[0].item<JOffsetsType>();
    const JOffsetsType elementEndOffset = offsetsAndRange[1].item<JOffsetsType>();
    const JOffsetsType startIdx = offsetsAndRange[2].item<JOffsetsType>();
    const JOffsetsType endIdx = offsetsAndRange[3].item<JOffsetsType>();
    torch::Tensor retOffsets = joffsets.index({torch::indexing::Slice(startIdx, endIdx+1)}) - elementStartOffset;
    const torch::Tensor retData = jdata.index({torch::indexing::Slice(elementStartOffset, elementEndOffset)});

    torch::Tensor retListIdx;
    int64_t retNumOuterLists;
    if (jlidx.size(1) > 1 && jlidx.size(1) > 2) {
        TORCH_CHECK(false, "We don't support ldim > 2.");
        // const auto lidxOpts = torch::TensorOptions().dtype(JLIdxScalarType).device(jdata.device());
        // retListIdx = torch::empty({retOffsets.size(0)-1, 2}, lidxOpts);
        // auto outJLidxAcc = retListIdx.packed_accessor32<JLIdxType, 2, torch::RestrictPtrTraits>();
        // const int numBlocksJLidx = GET_BLOCKS(retListIdx.size(0), 1024);
        // computeJLidx<<<numBlocksJLidx, 1024>>>(startIdx, idxVal, inJLidxAcc, outJLidxAcc);
        // C10_CUDA_KERNEL_LAUNCH_CHECK();
        // retNumOuterLists = std::get<0>(torch::unique_dim(retListIdx, 0)).size(0);
    } else {
        retListIdx = torch::empty({0, 1}, torch::TensorOptions().dtype(JLIdxScalarType).device(jdata.device()));
        retNumOuterLists = retOffsets.size(0) - 1;
    }

    const torch::Tensor retJidx = JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0));
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJidx, retListIdx, retNumOuterLists);
}





JaggedTensor jaggedTensorIndexMultiListCpu(const JaggedTensor& jt, int64_t idxVal) {
    if (idxVal < 0) {
        idxVal += jt.num_outer_lists();
    }
    TORCH_CHECK_INDEX(idxVal >= 0 && idxVal < jt.num_outer_lists(),
                      "Index ", idxVal, " is out of bounds for JaggedTensor with ",
                      jt.num_outer_lists(), " elements");

    torch::Tensor joffsets = jt.joffsets();
    torch::Tensor jdata = jt.jdata();
    torch::Tensor jlidx = jt.jlidx();

    TORCH_CHECK_VALUE(jlidx.dim() == 2, "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(jlidx.numel() == 0 || jlidx.size(0) == (joffsets.size(0) - 1), "Corrupt list indices. This should never happen");
    const torch::Tensor joffsetCat = torch::stack({
        joffsets.index({torch::indexing::Slice(0, jt.num_tensors())}),
        joffsets.index({torch::indexing::Slice(1, jt.num_tensors()+1)})
    }, 1);
    const torch::Tensor mask = jlidx.index({torch::indexing::Slice(), 0}).eq(idxVal);
    const torch::Tensor selectedOffsets = joffsetCat.index({mask});

    const JOffsetsType startIdx = selectedOffsets[0][0].item<JOffsetsType>();
    const JOffsetsType endIdx = selectedOffsets[-1][1].item<JOffsetsType>();

    const torch::Tensor retData = jdata.index({torch::indexing::Slice(startIdx, endIdx)});

    const torch::Tensor retOffsets = torch::cat({
        selectedOffsets.index({torch::indexing::Slice(), 0}),
        selectedOffsets.index({-1, 1}).unsqueeze(0)
    }) - startIdx;
    torch::Tensor retListIdx;
    int64_t retNumOuterLists;
    if (jlidx.size(1) > 1 && jlidx.size(1) > 2) {
        TORCH_CHECK(false, "We don't support ldim > 2.");
        // retListIdx = jlidx.index({mask, torch::indexing::Slice(1, jlidx.size(1))});
        // if (retListIdx.dim() == 0) {
        //     retListIdx = retListIdx.unsqueeze(1);
        // }
        // retNumOuterLists = std::get<0>(torch::unique_dim(retListIdx, 0)).size(0);
    } else {
        retListIdx = torch::empty({0, 1}, torch::TensorOptions().dtype(JLIdxScalarType).device(jdata.device()));
        retNumOuterLists = retOffsets.size(0) - 1;
    }

    const torch::Tensor retJidx = JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0));
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJidx, retListIdx, retNumOuterLists);
}


JaggedTensor jaggedTensorIndexOneList(const JaggedTensor& jt, int64_t idxVal) {
    if (idxVal < 0) {
        idxVal += jt.num_outer_lists();
    }
    TORCH_CHECK_INDEX(idxVal >= 0 && idxVal < jt.num_outer_lists(),
                      "Index ", idxVal, " is out of bounds for JaggedTensor with ",
                      jt.num_outer_lists(), " elements");

    torch::Tensor joffsets = jt.joffsets();
    torch::Tensor jdata = jt.jdata();
    torch::Tensor jlidx = jt.jlidx();

    TORCH_CHECK(jt.ldim() == 1, "bad list indexes. this should never happen");
    const JOffsetsType startIdx = joffsets[idxVal].item<JOffsetsType>();
    const JOffsetsType endIdx = joffsets[idxVal+1].item<JOffsetsType>();
    const torch::Tensor retJoffsets = torch::tensor({JOffsetsType(0), endIdx - startIdx}, torch::TensorOptions().dtype(JOffsetsScalarType).device(jdata.device()));
    const torch::Tensor retData = jdata.index({torch::indexing::Slice(startIdx, endIdx)});
    const torch::Tensor retJidx = torch::empty({0}, torch::TensorOptions().dtype(JIdxScalarType));
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        retData, retJoffsets, retJidx, jlidx, retJoffsets.size(0) - 1);
}



template <>
JaggedTensor dispatchJaggedTensorIndex<torch::kCPU>(const JaggedTensor& jt, int64_t idxVal) {
    if (jt.jlidx().size(0) == 0) {
        return jaggedTensorIndexOneList(jt, idxVal);
    } else {
        return jaggedTensorIndexMultiListCpu(jt, idxVal);
    }
}


template <>
JaggedTensor dispatchJaggedTensorIndex<torch::kCUDA>(const JaggedTensor& jt, int64_t idxVal) {
    if (jt.jlidx().size(0) == 0) {
        return jaggedTensorIndexOneList(jt, idxVal);
    } else {
        return jaggedTensorIndexMultiListCuda(jt, idxVal);
    }
}



} // namespace ops
} // namespace detail
} // namespace fvdb