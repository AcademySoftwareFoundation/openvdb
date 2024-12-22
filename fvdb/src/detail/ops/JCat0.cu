// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "Ops.h"
#include <detail/utils/Utils.h>
#include <detail/utils/cuda/Utils.cuh>

#include <ATen/cuda/Atomic.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace fvdb {
namespace detail {
namespace ops {

__global__ void
computeTensorSizes(const JOffsetsType *__restrict__ const *__restrict__ offsets,
                   const size_t numOffsets, TorchRAcc32<JOffsetsType, 1> outTensorSizes) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > outTensorSizes.size(0) - 1) {
        return;
    }

    // Calculate the size of the concatenated output tensor
    JOffsetsType tensorSize = 0;
    for (size_t i = 0; i < numOffsets; ++i) {
        tensorSize += offsets[i][idx + 1] - offsets[i][idx];
    }
    outTensorSizes[idx + 1] = tensorSize;

    // One thread will write out the zero in the begining
    if (idx == 0) {
        outTensorSizes[0] = 0;
    }
}

template <typename IdxT>
__global__ void
computeIndexPutArg(
    const size_t jti, const JOffsetsType *__restrict__ const *__restrict__ offsets,
    const size_t numOffsets,
    const TorchRAcc32<JIdxType, 1>     inJIdxI,     // Jidx of the i^th input tensor
    const TorchRAcc32<JOffsetsType, 1> inJoffsetsI, // JOffsets of the i^th input tensor
    const TorchRAcc32<JOffsetsType, 1> outJOffsets, // Output JOffsets (already computed earlier)
    TorchRAcc32<IdxT, 1>               outSelIdx,   // Output selection indices
    TorchRAcc32<JIdxType, 1>           outJIdx) {             // Output Jidx
    int32_t       idx         = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t numElements = inJIdxI.size(0);

    if (idx >= numElements) {
        return;
    }

    const JIdxType jidx = inJIdxI[idx]; // Which tensor this element belongs to

    // Where in the output tensor we're going to write to
    JOffsetsType tensorWriteOffset = 0;
    for (size_t i = 0; i < jti; ++i) {
        tensorWriteOffset += offsets[i][jidx + 1] - offsets[i][jidx];
    }

    const JOffsetsType writeBaseOffset =
        outJOffsets[jidx]; // Start of the concatenated tensor in the output
    const JOffsetsType tensorOffsetOut =
        writeBaseOffset +
        tensorWriteOffset; // Start of where we're copying the input tensor to in the output
    const JOffsetsType tensorOffsetIn = inJoffsetsI[jidx]; // Start of the tensor in the input
    const JOffsetsType elementOffsetInTensor =
        idx - tensorOffsetIn; // Which element in the tensor we are copying

    outSelIdx[idx] =
        elementOffsetInTensor +
        tensorOffsetOut; // Which element in the output the current input element will go to
    outJIdx[elementOffsetInTensor] =
        jidx;            // Which tensor the current element belongs to in the output
}

template <>
JaggedTensor
dispatchJCat0<torch::kCUDA>(const std::vector<JaggedTensor> &vec) {
    c10::cuda::CUDAGuard deviceGuard(vec[0].device());

    int64_t                             totalElements = 0;
    int64_t                             maxElements   = 0;
    thrust::host_vector<JOffsetsType *> offsets;
    offsets.reserve(vec.size());

    auto size_0  = vec[0].jdata().sizes();
    auto dtype_0 = vec[0].jdata().dtype();
    for (auto &jt: vec) {
        totalElements += jt.jdata().size(0);
        maxElements = std::max(maxElements, jt.jdata().size(0));
        offsets.push_back(jt.joffsets().data_ptr<JOffsetsType>());
        TORCH_CHECK_VALUE(jt.joffsets().size(0) == vec[0].joffsets().size(0),
                          "All jagged tensors must have the same number of tensors");
        TORCH_CHECK_VALUE(jt.joffsets().is_contiguous(),
                          "All jagged tensors must have contiguous offsets");
        TORCH_CHECK_VALUE(jt.device().is_cuda(), "All jagged tensors must be on the same device");
        TORCH_CHECK(jt.jdata().dtype() == dtype_0, "All jagged tensors must have the same dtype");
        auto sizes_i = jt.jdata().sizes();
        for (size_t i = 1; i < sizes_i.size(); ++i) {
            TORCH_CHECK_VALUE(sizes_i[i] == size_0[i],
                              "All jagged tensors must have the same eshape");
        }
    }

    thrust::device_vector<JOffsetsType *> offsets_d = offsets;
    torch::Tensor                         outJOffsets =
        torch::empty({ vec[0].joffsets().size(0) },
                     torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCUDA));
    const int64_t numThreadsCalcTensorSizes = 1024;
    const int64_t numBlocksCalcTensorSizes =
        GET_BLOCKS(outJOffsets.size(0), numThreadsCalcTensorSizes);
    computeTensorSizes<<<numBlocksCalcTensorSizes, numThreadsCalcTensorSizes>>>(
        thrust::raw_pointer_cast(offsets_d.data()), offsets_d.size(),
        outJOffsets.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    torch::cumsum_out(outJOffsets, outJOffsets, 0);

    auto          outShape = spliceShape({ totalElements }, vec[0].jdata());
    torch::Tensor outJData = torch::empty(
        outShape, torch::TensorOptions().dtype(vec[0].scalar_type()).device(torch::kCUDA));
    torch::Tensor outJIdx = torch::empty(
        { totalElements }, torch::TensorOptions().dtype(JIdxScalarType).device(torch::kCUDA));

    c10::ScalarType idxType = torch::kInt32;
    if (maxElements < std::numeric_limits<int32_t>::max()) {
        idxType = torch::kInt32;
    } else if (maxElements < std::numeric_limits<int64_t>::max()) {
        idxType = torch::kInt64;
    } else {
        TORCH_CHECK(false, "Cannot handle more than ", std::numeric_limits<int64_t>::max(),
                    " elements");
    }

    torch::Tensor selectIndices =
        torch::zeros({ maxElements }, torch::TensorOptions().dtype(idxType).device(torch::kCUDA));
    for (size_t jti = 0; jti < vec.size(); ++jti) {
        const JaggedTensor &jt = vec[jti];
        AT_DISPATCH_INTEGRAL_TYPES(selectIndices.scalar_type(), "computeIndexPutArg", [&] {
            const int64_t numElements                  = jt.jdata().size(0);
            const int64_t numThreadsComputeIndexPutArg = 1024;
            const int64_t numBlocksComputeIndexPutArg =
                GET_BLOCKS(numElements, numThreadsComputeIndexPutArg);
            computeIndexPutArg<<<numBlocksComputeIndexPutArg, numThreadsComputeIndexPutArg>>>(
                jti, thrust::raw_pointer_cast(offsets_d.data()), offsets_d.size(),
                jt.jidx().packed_accessor32<JIdxType, 1, torch::RestrictPtrTraits>(),
                jt.joffsets().packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>(),
                outJOffsets.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>(),
                selectIndices.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                outJIdx.packed_accessor32<JIdxType, 1, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            torch::Tensor selIdxI = selectIndices.index({ torch::indexing::Slice(0, numElements) });

            outJData.index_put_({ selIdxI, torch::indexing::Ellipsis }, jt.jdata());
        });
    }

    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
        outJData, outJOffsets, outJIdx, vec[0].jlidx(), vec[0].num_outer_lists());
}

template <>
JaggedTensor
dispatchJCat0<torch::kCPU>(const std::vector<JaggedTensor> &vec) {
    const auto device = vec[0].device();
    const auto dtype  = vec[0].scalar_type();

    int64_t totalElements = 0;
    for (const auto &jvec: vec) {
        TORCH_CHECK_VALUE(jvec.joffsets().size(0) == vec[0].joffsets().size(0),
                          "All tensors must have the same number of lists");
        TORCH_CHECK_VALUE(jvec.jdata().dim() == vec[0].jdata().dim(),
                          "All tensors must have the same number of dimensions");
        TORCH_CHECK_VALUE(jvec.device() == device, "All tensors must be on the same device");
        TORCH_CHECK_VALUE(jvec.scalar_type() == dtype,
                          "All tensors must have the same scalar type");
        totalElements += jvec.jdata().size(0);
    }
    const auto         shape      = fvdb::detail::spliceShape({ totalElements }, vec[0].jdata());
    const JOffsetsType numOffsets = vec[0].joffsets().size(0);

    torch::Tensor outJdata =
        torch::empty(shape, torch::TensorOptions().device(device).dtype(dtype));
    torch::Tensor outJoffsets = torch::empty(
        { numOffsets }, torch::TensorOptions().device(device).dtype(JOffsetsScalarType));
    torch::Tensor outJidx = torch::empty(
        { totalElements }, torch::TensorOptions().device(device).dtype(JIdxScalarType));
    torch::Tensor outJLidx         = vec[0].jlidx();
    const int64_t outNumOuterLists = vec[0].num_outer_lists();

    JOffsetsType startOffset = 0;
    for (JOffsetsType i = 0; i < numOffsets - 1; ++i) {
        JOffsetsType numElements = 0;

        std::vector<torch::Tensor> tensorsToCat;
        tensorsToCat.reserve(vec.size());
        for (const auto &jvec: vec) {
            const JOffsetsType startIdx = jvec.joffsets()[i].item<JOffsetsType>();
            const JOffsetsType endIdx   = jvec.joffsets()[i + 1].item<JOffsetsType>();
            torch::Tensor      jdataSlice =
                jvec.jdata().index({ torch::indexing::Slice(startIdx, endIdx) });
            tensorsToCat.push_back(jdataSlice);
            numElements += (endIdx - startIdx);
        }

        outJdata.index({ torch::indexing::Slice(startOffset, startOffset + numElements) })
            .copy_(torch::cat(tensorsToCat, 0));
        outJidx.index({ torch::indexing::Slice(startOffset, startOffset + numElements) })
            .copy_(torch::full({ numElements }, i,
                               torch::TensorOptions().dtype(JIdxScalarType).device(device)));
        outJoffsets[i]     = startOffset;
        outJoffsets[i + 1] = startOffset + numElements;
        startOffset += numElements;
    }
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(outJdata, outJoffsets, outJidx,
                                                                  outJLidx, outNumOuterLists);
}

} // namespace ops
} // namespace detail
} // namespace fvdb