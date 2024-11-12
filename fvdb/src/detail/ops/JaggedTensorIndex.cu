// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/Utils.h>
#include <detail/utils/cuda/Utils.cuh>

#include <ATen/cuda/Atomic.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace fvdb {
namespace detail {
namespace ops {

// This kernel computes the offsets for an integer indexing operation
__global__ void
getJOffsetsIndexMask(const int64_t idxVal, const TorchRAcc32<JLIdxType, 2> jlidx,
                     const TorchRAcc32<JOffsetsType, 1> inJoffsets,
                     TorchRAcc32<JOffsetsType, 1>       offsetsAndRange) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= jlidx.size(0)) {
        return;
    }

    JLIdxType lid     = jlidx[idx][0];
    JLIdxType prevLid = -1;
    if (idx - 1 >= 0) {
        prevLid = jlidx[idx - 1][0];
    }
    const bool lidMatches     = lid == idxVal;
    const bool prevLidMatches = prevLid == idxVal;
    const bool isLastIdx      = idx == (jlidx.size(0) - 1);

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

// Computes a mask for the data tensor for a slice operation
__global__ void
makeDataSliceMask(const int64_t start, const int64_t end, const int64_t step,
                  const TorchRAcc32<JIdxType, 1> inJIdx, const TorchRAcc32<JLIdxType, 2> inJLidx,
                  TorchRAcc32<bool, 1> outDataMask, bool isLdim1, bool oneTensor) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= outDataMask.size(0)) {
        return;
    }

    if (isLdim1) {
        const JIdxType jidx        = oneTensor ? 0 : inJIdx[idx]; // Which tensor we're in
        const bool elementIncluded = (jidx >= start && jidx < end && (jidx - start) % step == 0);
        outDataMask[idx]           = elementIncluded;
    } else {
        const JIdxType  jidx = oneTensor ? 0 : inJIdx[idx]; // Which tensor this element belongs to
        const JLIdxType lidx = inJLidx[jidx][0];            // Which list this tensor belongs to
        const bool      isIncluded = (lidx >= start && lidx < end &&
                                 (lidx - start) % step == 0); // Is the list included in the slice?
        outDataMask[idx] =
            isIncluded; // The element belongs to a tensor in a list that is included in the slice
    }
}

// Computes a the new joffsets and jlidx tensor for a slice operation. Note that the output joffsets
// and jlidx have redundant values that need to be masked out. We allocate them to be the size of
// the input so we don't need to track the size of the output tensors.
// This kernel also computes the appropriate masks
__global__ void
makeOffsetsSliceMask(const int64_t start, const int64_t end, const int64_t step,
                     const TorchRAcc32<JOffsetsType, 1> inJoffsets,
                     const TorchRAcc32<JLIdxType, 2> inJLidx, TorchRAcc32<bool, 1> outOffsetsMask,
                     TorchRAcc32<JOffsetsType, 1> outTensorSizes,
                     TorchRAcc32<JLIdxType, 2> outJLIdx, bool isLdim1) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0) {
        outTensorSizes[0] = 0;
        outOffsetsMask[0] = true;
    }

    if (idx >= (inJoffsets.size(0) - 1)) {
        return;
    }

    const JOffsetsType inOrdinal =
        isLdim1 ? idx : inJLidx[idx][0];       // either jidx when ldim1 or lidx when ldim2
    const JOffsetsType outOrdinal =
        (inOrdinal - start + step - 1) / step; // which tensor or list this offset belongs to

    const bool offsetIncluded =
        (inOrdinal >= start && inOrdinal < end && (inOrdinal - start) % step == 0);
    outOffsetsMask[idx + 1] = offsetIncluded;

    if (offsetIncluded) {
        outTensorSizes[idx + 1] = inJoffsets[idx + 1] - inJoffsets[idx];

        if (!isLdim1) {
            outJLIdx[idx][0] = outOrdinal;
            outJLIdx[idx][1] = inJLidx[idx][1];
        }
    }
}

// When we're indexing with a jagged tensor, each indexing tensor i_AB has integers in the range
// [0, t_AB.size(dim))] where t_AB is the tensor being indexed. We need to convert these to global
// indices into the jdata tensor by adding the appropriate joffset to each index
template <typename IndexType>
__global__ void
calculateIndexShiftForEachElement(const TorchRAcc64<JOffsetsType, 1> inJOffsets,
                                  const TorchRAcc64<JIdxType, 1>     inJIdx,
                                  TorchRAcc64<IndexType, 1>          outAdd) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= outAdd.size(0)) {
        return;
    }

    JIdxType jidx = inJIdx.size(0) > 0 ? inJIdx[idx] : 0;
    outAdd[idx]   = inJOffsets[jidx];
}

// This corresponds to indexing with a JaggedTensor. i.e. using each tensor in an indexing
// JaggedTensor to index the corresponding tensor in the JaggedTensor
// i.e. jt = JaggedTensor([[t_11, t_12], [t_21, t_22, t_23], ...])
//      indices = JaggedTensor([[i_11, i_12], [i_21, i_22, i_23], ...])
//      jt[indices] -> JaggedTensor([[t_11[i_11], t_12[i_12]], [t_21[i_21], t_22[i_22], t_23[i_23]],
//      ...])
// Here indices can be integers or a boolean mask
JaggedTensor
jaggedTensorIndexJaggedTensorImpl(const JaggedTensor &jt, const JaggedTensor &jtIndices) {
    TORCH_CHECK_VALUE(jtIndices.device() == jt.device(),
                      "indices must be on the same device as the JaggedTensor");

    TORCH_CHECK_INDEX(jtIndices.jlidx().dim() == jt.jlidx().dim(),
                      "Indices must have the same list structure as JaggedTensor being indexed");
    for (int i = 0; i < jt.jlidx().dim(); ++i) {
        TORCH_CHECK_INDEX(
            jtIndices.jlidx().size(i) == jt.jlidx().size(i),
            "Indices must have the same list structure as JaggedTensor being indexed");
    }
    if (Config::global().pendanticErrorCheckingEnabled()) {
        // This is a slow check that we cap optionally do for correctness.
        TORCH_CHECK_INDEX(
            torch::all(jtIndices.jlidx() == jt.jlidx()).item<bool>(),
            "Indices must have the same list structure as JaggedTensor being indexed. ",
            "This error was raised because config.pendatic_error_checking was enabled");
    }

    c10::ScalarType idxDtype = jtIndices.scalar_type();
    const bool      isIndexType =
        (idxDtype == c10::ScalarType::Long || idxDtype == c10::ScalarType::Int ||
         idxDtype == c10::ScalarType::Byte || idxDtype == c10::ScalarType::Bool);
    TORCH_CHECK_INDEX(
        isIndexType,
        "JaggedTensors used as indices must be long, int, byte or bool JaggedTensors but got ",
        idxDtype);

    torch::Tensor selidx;
    if (jtIndices.scalar_type() == torch::kBool) {
        selidx = jtIndices.jdata();
    } else {
        if (jt.device().is_cpu()) {
            // FIXME (Francis): We're not checking out of range here and it's sketchy! Fix in a
            // unified CUDA kernel
            selidx = jtIndices.jdata().clone();
            for (int i = 0; i < jtIndices.joffsets().size(0) - 1; ++i) {
                const JOffsetsType start = jtIndices.joffsets()[i].item<JOffsetsType>();
                const JOffsetsType end   = jtIndices.joffsets()[i + 1].item<JOffsetsType>();
                const JOffsetsType add   = jt.joffsets()[i].item<JOffsetsType>();
                selidx.index({ torch::indexing::Slice(start, end) }).add_(add);
            }
        } else {
            torch::Tensor selidxAdd =
                torch::empty({ jtIndices.jdata().size(0) }, jtIndices.jdata().options());

            AT_DISPATCH_INTEGRAL_TYPES(
                jtIndices.scalar_type(), "calculateIndexShiftForEachElement", [&] {
                    const int64_t MAX_BLOCKS = 4194302; // floor((2^32 - 1) / 1024)
                    const int64_t numBlocks  = GET_BLOCKS(jtIndices.jdata().size(0), 1024);
                    TORCH_INTERNAL_ASSERT(numBlocks < MAX_BLOCKS, "Too many blocks");
                    calculateIndexShiftForEachElement<scalar_t><<<numBlocks, 1024>>>(
                        jt.joffsets()
                            .packed_accessor64<JOffsetsType, 1, torch::RestrictPtrTraits>(),
                        jtIndices.jidx().packed_accessor64<JIdxType, 1, torch::RestrictPtrTraits>(),
                        selidxAdd.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>());
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            for (int i = 1; i < jtIndices.jdata().dim(); i += 1) {
                selidxAdd = selidxAdd.unsqueeze(-1);
            }
            selidx = selidxAdd + jtIndices.jdata();
        }
    }

    const torch::Tensor retJdata = jt.jdata().index({ selidx });
    torch::Tensor       retJidx  = jt.jidx().size(0) > 0 ? jt.jidx().index({ selidx }) : jt.jidx();
    if (retJidx.dim() > 1) {
        std::vector<at::indexing::TensorIndex> idx;
        idx.reserve(retJidx.dim());
        idx.push_back(at::indexing::Slice());
        for (int i = 1; i < retJidx.dim(); ++i) {
            idx.push_back(0);
        }
        retJidx = retJidx.index(idx);
    }
    retJidx = retJidx.contiguous();
    const torch::Tensor retJOffsets =
        JaggedTensor::joffsets_from_jidx_and_jdata(retJidx, retJdata, jt.num_tensors());
    const torch::Tensor retListIdx = jt.jlidx();

    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retJdata, retJOffsets, retJidx,
                                                                  retListIdx, jt.num_outer_lists());
}

// This corresponds to indexing with a slice
// i.e. jt = JaggedTensor([...])
//      jt[2:11:4] -> JaggedTensor([...]) where every fourth entry from the third to the tenth list
//      (inclusive) is selected
JaggedTensor
jaggedTensorIndexSliceCuda(const JaggedTensor &jt, int64_t start, int64_t end, int64_t step) {
    // Convert indexes to positive values in the range [0, num_outer_lists]
    if (start < 0) {
        start += jt.num_outer_lists();
    }
    if (end < 0) {
        end += jt.num_outer_lists();
    }
    if (start >= end) {
        start = end;
    }
    start = std::max(start, (int64_t)0);
    end   = std::min(end, jt.num_outer_lists());

    // Single list case with step size 1 (ldim = 1)
    if (jt.ldim() == 1 && step == 1) {
        TORCH_CHECK(jt.ldim() == 1, "bad list indexes. this should never happen");
        const JOffsetsType  startIdx = jt.joffsets()[start].item<JOffsetsType>();
        const JOffsetsType  endIdx   = jt.joffsets()[end].item<JOffsetsType>();
        const torch::Tensor retLidx =
            jt.jlidx().numel() == 0 ? jt.jlidx()
                                    : jt.jlidx().index({ torch::indexing::Slice(start, end) });
        return JaggedTensor::from_data_offsets_and_list_ids(
            jt.jdata().index({ torch::indexing::Slice(startIdx, endIdx) }),
            jt.joffsets().index({ torch::indexing::Slice(start, end + 1) }) - startIdx, retLidx);
    }

    // Compute a boolean mask for the data tensor and offsets as well as the tensor sizes (which we
    // cumsum) and list ids The offsets mask is used so we can just write the tensor sizes/lidx to
    // the output tensor and then select only the active values. Otherwise, we'd need something like
    // a binsearch
    const torch::TensorOptions maskOpts =
        torch::TensorOptions().device(jt.device()).dtype(torch::kBool);
    torch::Tensor dataMask    = torch::empty({ jt.jdata().size(0) }, maskOpts);
    torch::Tensor offsetsMask = torch::empty({ jt.joffsets().size(0) }, maskOpts);
    torch::Tensor outJLIdx    = torch::empty_like(jt.jlidx());
    torch::Tensor outJOffsets = torch::empty_like(jt.joffsets());

    auto joffsetsAcc = jt.joffsets().packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>();
    auto jidxAcc     = jt.jidx().packed_accessor32<JIdxType, 1, torch::RestrictPtrTraits>();
    auto jlidxAcc    = jt.jlidx().packed_accessor32<JLIdxType, 2, torch::RestrictPtrTraits>();
    auto dataMaskAcc = dataMask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>();
    auto offsetsMaskAcc = offsetsMask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>();
    auto outJOffsetsAcc =
        outJOffsets.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>();
    auto outJLIdxAcc = outJLIdx.packed_accessor32<JLIdxType, 2, torch::RestrictPtrTraits>();

    auto callKernel = [=]() {
        const int64_t MAX_BLOCKS    = 4194302; // floor((2^32 - 1) / 1024)
        const int64_t numBlocksData = GET_BLOCKS(jt.jdata().size(0), 1024);
        TORCH_INTERNAL_ASSERT(numBlocksData < MAX_BLOCKS, "Too many blocks");
        makeDataSliceMask<<<numBlocksData, 1024>>>(start, end, step, jidxAcc, jlidxAcc, dataMaskAcc,
                                                   jt.ldim() == 1, jt.num_tensors() == 1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int numBlocksOffsets = GET_BLOCKS(jt.joffsets().size(0) - 1, 1024);
        TORCH_INTERNAL_ASSERT(numBlocksOffsets < MAX_BLOCKS, "Too many blocks");
        makeOffsetsSliceMask<<<numBlocksOffsets, 1024>>>(start, end, step, joffsetsAcc, jlidxAcc,
                                                         offsetsMaskAcc, outJOffsetsAcc,
                                                         outJLIdxAcc, jt.ldim() == 1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    callKernel();

    const torch::Tensor outData = jt.jdata().index({ dataMask });

    outJOffsets = outJOffsets.index({ offsetsMask });
    torch::cumsum_out(outJOffsets, outJOffsets, 0);

    torch::Tensor outJIdx =
        outJOffsets.size(0) > 2
            ? fvdb::JaggedTensor::jidx_from_joffsets(outJOffsets, outData.size(0))
            : torch::empty(
                  { 0 }, torch::TensorOptions().dtype(JIdxScalarType).device(jt.jdata().device()));

    outJLIdx =
        jt.ldim() > 1
            ? outJLIdx.index(
                  { offsetsMask.index({ torch::indexing::Slice(1, offsetsMask.size(0), 1) }) })
            : torch::empty(
                  { 0, 1 },
                  torch::TensorOptions().dtype(JLIdxScalarType).device(jt.jdata().device()));

    const JOffsetsType totalItems = (end - start + step - 1) / step;
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(outData, outJOffsets, outJIdx,
                                                                  outJLIdx, totalItems);
}

// This corresponds to indexing with a slice
// i.e. jt = JaggedTensor([...])
//      jt[2:11:4] -> JaggedTensor([...]) where every fourth entry from the third to the tenth list
//      (inclusive) is selected
JaggedTensor
jaggedTensorIndexSliceCpu(const JaggedTensor &jt, int64_t start, int64_t end, int64_t step) {
    // Convert indexes to positive values
    if (start < 0) {
        start += jt.num_outer_lists();
    }
    if (end < 0) {
        end += jt.num_outer_lists();
    }
    if (start >= end) {
        start = end;
    }

    start = std::max(start, static_cast<int64_t>(0));
    end   = std::min(end, jt.num_outer_lists());

    if (jt.ldim() == 1 && step == 1) {
        TORCH_CHECK(jt.ldim() == 1, "bad list indexes. this should never happen");
        const JOffsetsType  startIdx = jt.joffsets()[start].item<JOffsetsType>();
        const JOffsetsType  endIdx   = jt.joffsets()[end].item<JOffsetsType>();
        const torch::Tensor retLidx =
            jt.jlidx().numel() == 0 ? jt.jlidx()
                                    : jt.jlidx().index({ torch::indexing::Slice(start, end) });
        return JaggedTensor::from_data_offsets_and_list_ids(
            jt.jdata().index({ torch::indexing::Slice(startIdx, endIdx) }),
            jt.joffsets().index({ torch::indexing::Slice(start, end + 1) }) - startIdx, retLidx);
    } else if (jt.ldim() > 1 && step == 1) {
        // Find all tensors that belong to the slice
        const torch::Tensor outerLidx  = jt.jlidx().index({ torch::indexing::Slice(), 0 });
        const torch::Tensor lidxMask   = outerLidx.ge(start).logical_and(outerLidx.lt(end));
        const torch::Tensor joffsetCat = torch::stack(
            { jt.joffsets().index({ torch::indexing::Slice(0, jt.num_tensors()) }),
              jt.joffsets().index({ torch::indexing::Slice(1, jt.num_tensors() + 1) }) },
            1);

        // Start and end element index of each tensor in the slice
        const torch::Tensor selectedOffsets = joffsetCat.index({ lidxMask });

        // Get the start and end offsets into the data tensor for the slice
        JOffsetsType startIdx =
            selectedOffsets.size(0) > 0 ? selectedOffsets[0][0].item<JOffsetsType>() : 0;
        JOffsetsType endIdx =
            selectedOffsets.size(0) > 0 ? selectedOffsets[-1][1].item<JOffsetsType>() : 0;

        // Slice the data tensor
        const torch::Tensor retData =
            jt.jdata().index({ torch::indexing::Slice(startIdx, endIdx) });

        // Subtract the start offset from the selected offsets to get the new offsets
        // NOTE: This assumes offsets are always contiguous
        const torch::Tensor retOffsets =
            selectedOffsets.numel() > 0
                ? torch::cat({ selectedOffsets.index({ torch::indexing::Slice(), 0 }),
                               selectedOffsets.index({ -1, 1 }).unsqueeze(0) }) -
                      startIdx
                : torch::zeros(
                      { 1 },
                      torch::TensorOptions().dtype(JOffsetsScalarType).device(jt.jdata().device()));

        // Slice the list indices and subtract the start index
        TORCH_CHECK(jt.jlidx().size(1) > 1, "bad list indexes. this should never happen");
        torch::Tensor retListIdx = jt.jlidx().index({ lidxMask });
        retListIdx.index({ torch::indexing::Slice(), 0 }) -= start;
        if (retListIdx.dim() == 0) {
            retListIdx = retListIdx.unsqueeze(1);
        }
        const int64_t       retNumOuterLists = end - start;
        const torch::Tensor retJidx =
            retOffsets.size(0) > 2
                ? JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0))
                : torch::empty(
                      { 0 },
                      torch::TensorOptions().dtype(JIdxScalarType).device(jt.jdata().device()));
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJidx,
                                                                      retListIdx, retNumOuterLists);
    } else if (jt.ldim() == 1 && step > 1) {
        const JOffsetsType         totalItems = (end - start + step - 1) / step;
        const torch::TensorOptions offsetsOpts =
            torch::TensorOptions().dtype(JOffsetsScalarType).device(jt.jdata().device());
        torch::Tensor dataMask = torch::zeros(
            { jt.jdata().size(0) }, torch::TensorOptions().dtype(torch::kBool).device(jt.device()));
        torch::Tensor retOffsets = torch::empty({ totalItems + 1 }, offsetsOpts);

        auto    retOffsetsAcc = retOffsets.accessor<JOffsetsType, 1>();
        auto    joffsetsAcc   = jt.joffsets().accessor<JOffsetsType, 1>();
        int64_t count         = 0;
        retOffsetsAcc[0]      = 0;
        for (int64_t i = start; i < end; i += step) {
            JOffsetsType startIdx = joffsetsAcc[i];
            JOffsetsType endIdx   = joffsetsAcc[i + 1];
            dataMask.index({ torch::indexing::Slice(startIdx, endIdx) }).fill_(true);
            retOffsetsAcc[count + 1] = endIdx - startIdx;
            count += 1;
        }
        torch::cumsum_out(retOffsets, retOffsets, 0);
        const torch::Tensor retData = jt.jdata().index({ dataMask });
        const torch::Tensor retJIdx =
            retOffsets.size(0) > 2
                ? JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0))
                : torch::empty(
                      { 0 },
                      torch::TensorOptions().dtype(JIdxScalarType).device(jt.jdata().device()));
        const torch::Tensor retJLidx = torch::zeros(
            { 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(jt.jdata().device()));
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJIdx,
                                                                      retJLidx, totalItems);
    } else {
        // Find all tensors that belong to the slice
        const torch::Tensor outerLidx = jt.jlidx().index({ torch::indexing::Slice(), 0 });
        const torch::Tensor lidxMask  = outerLidx.ge(start)
                                           .logical_and(outerLidx.lt(end))
                                           .logical_and((outerLidx - start) % step == 0);
        const torch::Tensor selectedOffsets =
            torch::stack(
                { jt.joffsets().index({ torch::indexing::Slice(0, jt.num_tensors()) }),
                  jt.joffsets().index({ torch::indexing::Slice(1, jt.num_tensors() + 1) }) },
                1)
                .index({ lidxMask });

        const torch::Tensor selectedLidx = jt.jlidx().index({ lidxMask });

        const torch::TensorOptions offsetsOpts =
            torch::TensorOptions().dtype(JOffsetsScalarType).device(jt.jdata().device());
        const torch::TensorOptions lidxOpts =
            torch::TensorOptions().dtype(JLIdxScalarType).device(jt.jdata().device());

        torch::Tensor dataMask = torch::zeros(
            { jt.jdata().size(0) }, torch::TensorOptions().dtype(torch::kBool).device(jt.device()));
        torch::Tensor retOffsets = torch::empty({ selectedOffsets.size(0) + 1 }, offsetsOpts);
        torch::Tensor retJLidx   = torch::empty({ selectedOffsets.size(0), jt.ldim() }, lidxOpts);

        auto retOffsetsAcc = retOffsets.accessor<JOffsetsType, 1>();
        auto retJLidxAcc   = retJLidx.accessor<JLIdxType, 2>();
        auto selOffsetsAcc = selectedOffsets.accessor<JOffsetsType, 2>();
        auto selLidxAcc    = selectedLidx.accessor<JLIdxType, 2>();
        retOffsetsAcc[0]   = 0;
        JLIdxType count    = -1;
        for (int i = 0; i < retOffsets.size(0) - 1; i += 1) {
            if (i == 0 || selLidxAcc[i][0] != selLidxAcc[i - 1][0]) {
                count += 1;
            }

            JOffsetsType startIdx = selOffsetsAcc[i][0];
            JOffsetsType endIdx   = selOffsetsAcc[i][1];

            dataMask.index({ torch::indexing::Slice(startIdx, endIdx) }).fill_(true);
            retOffsetsAcc[i + 1] = endIdx - startIdx;
            retJLidxAcc[i][0]    = count;
            retJLidxAcc[i][1]    = selLidxAcc[i][1];
        }
        count += 1;
        torch::cumsum_out(retOffsets, retOffsets, 0);
        const torch::Tensor retData = jt.jdata().index({ dataMask });
        const torch::Tensor retJIdx =
            retOffsets.size(0) > 2
                ? JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0))
                : torch::empty(
                      { 0 },
                      torch::TensorOptions().dtype(JIdxScalarType).device(jt.jdata().device()));
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJIdx,
                                                                      retJLidx, count);
    }
}

// Special case of integer indexing where the JaggedTensor is just a list of tensors and not a list
// of lists of tensors. We call this from the CPU and GPU implementations which is why it's factored
// out i.e. jt = JaggedTensor([t_0, t_1, t_2, ..., t_n])
//      jt[2] -> JaggedTensor([t_2]) where the 3rd list is selected
JaggedTensor
jaggedTensorIndexIntOneList(const JaggedTensor &jt, int64_t idxVal) {
    torch::Tensor joffsets = jt.joffsets();
    torch::Tensor jdata    = jt.jdata();
    torch::Tensor jlidx    = jt.jlidx();

    TORCH_CHECK(jt.ldim() == 1, "bad list indexes. this should never happen");
    const JOffsetsType  startIdx = joffsets[idxVal].item<JOffsetsType>();
    const JOffsetsType  endIdx   = joffsets[idxVal + 1].item<JOffsetsType>();
    const torch::Tensor retJoffsets =
        torch::tensor({ JOffsetsType(0), endIdx - startIdx },
                      torch::TensorOptions().dtype(JOffsetsScalarType).device(jdata.device()));
    const torch::Tensor retData = jdata.index({ torch::indexing::Slice(startIdx, endIdx) });
    const torch::Tensor retJidx = torch::empty({ 0 }, torch::TensorOptions().dtype(JIdxScalarType));
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retJoffsets, retJidx,
                                                                  jlidx, retJoffsets.size(0) - 1);
}

// This corresponds to indexing with an integer
// i.e. jt = JaggedTensor([...])
//      jt[2] -> JaggedTensor([...]) where the 3rd list is selected
JaggedTensor
jaggedTensorIndexIntCuda(const JaggedTensor &jt, int64_t idxVal) {
    if (idxVal < 0) {
        idxVal += jt.num_outer_lists();
    }
    TORCH_CHECK_INDEX(idxVal >= 0 && idxVal < jt.num_outer_lists(), "Index ", idxVal,
                      " is out of bounds for JaggedTensor with ", jt.num_outer_lists(),
                      " elements");

    if (jt.jlidx().size(0) == 0) {
        return jaggedTensorIndexIntOneList(jt, idxVal);
    }

    torch::Tensor joffsets = jt.joffsets();
    torch::Tensor jdata    = jt.jdata();
    torch::Tensor jlidx    = jt.jlidx();

    TORCH_CHECK_VALUE(jlidx.dim() == 2, "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(jlidx.numel() == 0 || jlidx.size(0) == (joffsets.size(0) - 1),
                      "Corrupt list indices. This should never happen");

    torch::Tensor offsetsAndRange = torch::empty(
        { 4 },
        torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU).pinned_memory(true));
    offsetsAndRange    = offsetsAndRange.to(jt.device());
    auto inJLidxAcc    = jlidx.packed_accessor32<JLIdxType, 2, torch::RestrictPtrTraits>();
    auto inJOffsetsAcc = joffsets.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>();
    auto offsetsAndRangeAcc =
        offsetsAndRange.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>();

    const int64_t MAX_BLOCKS = 4194302; // floor((2^32 - 1) / 1024)
    const int64_t numBlocks  = GET_BLOCKS(joffsets.size(0), 1024);
    TORCH_INTERNAL_ASSERT(numBlocks < MAX_BLOCKS, "Too many blocks");
    getJOffsetsIndexMask<<<numBlocks, 1024>>>(idxVal, inJLidxAcc, inJOffsetsAcc,
                                              offsetsAndRangeAcc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    offsetsAndRange                       = offsetsAndRange.cpu();
    const JOffsetsType elementStartOffset = offsetsAndRange[0].item<JOffsetsType>();
    const JOffsetsType elementEndOffset   = offsetsAndRange[1].item<JOffsetsType>();
    const JOffsetsType startIdx           = offsetsAndRange[2].item<JOffsetsType>();
    const JOffsetsType endIdx             = offsetsAndRange[3].item<JOffsetsType>();
    torch::Tensor      retOffsets =
        joffsets.index({ torch::indexing::Slice(startIdx, endIdx + 1) }) - elementStartOffset;
    const torch::Tensor retData =
        jdata.index({ torch::indexing::Slice(elementStartOffset, elementEndOffset) });

    torch::Tensor retListIdx;
    int64_t       retNumOuterLists;
    if (jlidx.size(1) > 1 && jlidx.size(1) > 2) {
        TORCH_CHECK(false, "We don't support ldim > 2.");
        // const auto lidxOpts =
        // torch::TensorOptions().dtype(JLIdxScalarType).device(jdata.device()); retListIdx =
        // torch::empty({retOffsets.size(0)-1, 2}, lidxOpts); auto outJLidxAcc =
        // retListIdx.packed_accessor32<JLIdxType, 2, torch::RestrictPtrTraits>(); const int
        // numBlocksJLidx = GET_BLOCKS(retListIdx.size(0), 1024); computeJLidx<<<numBlocksJLidx,
        // 1024>>>(startIdx, idxVal, inJLidxAcc, outJLidxAcc); C10_CUDA_KERNEL_LAUNCH_CHECK();
        // retNumOuterLists = std::get<0>(torch::unique_dim(retListIdx, 0)).size(0);
    } else {
        retListIdx = torch::empty(
            { 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(jdata.device()));
        retNumOuterLists = retOffsets.size(0) - 1;
    }

    const torch::Tensor retJidx = JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0));
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJidx,
                                                                  retListIdx, retNumOuterLists);
}

// This corresponds to indexing with an integer
// i.e. jt = JaggedTensor([...])
//      jt[2] -> JaggedTensor([...]) where the 3rd list is selected
JaggedTensor
jaggedTensorIndexIntCpu(const JaggedTensor &jt, int64_t idxVal) {
    if (idxVal < 0) {
        idxVal += jt.num_outer_lists();
    }
    TORCH_CHECK_INDEX(idxVal >= 0 && idxVal < jt.num_outer_lists(), "Index ", idxVal,
                      " is out of bounds for JaggedTensor with ", jt.num_outer_lists(),
                      " elements");

    if (jt.jlidx().size(0) == 0) {
        return jaggedTensorIndexIntOneList(jt, idxVal);
    }

    torch::Tensor joffsets = jt.joffsets();
    torch::Tensor jdata    = jt.jdata();
    torch::Tensor jlidx    = jt.jlidx();

    TORCH_CHECK_VALUE(jlidx.dim() == 2, "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(jlidx.numel() == 0 || jlidx.size(0) == (joffsets.size(0) - 1),
                      "Corrupt list indices. This should never happen");
    const torch::Tensor joffsetCat =
        torch::stack({ joffsets.index({ torch::indexing::Slice(0, jt.num_tensors()) }),
                       joffsets.index({ torch::indexing::Slice(1, jt.num_tensors() + 1) }) },
                     1);
    const torch::Tensor mask            = jlidx.index({ torch::indexing::Slice(), 0 }).eq(idxVal);
    const torch::Tensor selectedOffsets = joffsetCat.index({ mask });

    const JOffsetsType startIdx = selectedOffsets[0][0].item<JOffsetsType>();
    const JOffsetsType endIdx   = selectedOffsets[-1][1].item<JOffsetsType>();

    const torch::Tensor retData = jdata.index({ torch::indexing::Slice(startIdx, endIdx) });

    const torch::Tensor retOffsets =
        torch::cat({ selectedOffsets.index({ torch::indexing::Slice(), 0 }),
                     selectedOffsets.index({ -1, 1 }).unsqueeze(0) }) -
        startIdx;
    torch::Tensor retListIdx;
    int64_t       retNumOuterLists;
    if (jlidx.size(1) > 1 && jlidx.size(1) > 2) {
        TORCH_CHECK(false, "We don't support ldim > 2.");
        // retListIdx = jlidx.index({mask, torch::indexing::Slice(1, jlidx.size(1))});
        // if (retListIdx.dim() == 0) {
        //     retListIdx = retListIdx.unsqueeze(1);
        // }
        // retNumOuterLists = std::get<0>(torch::unique_dim(retListIdx, 0)).size(0);
    } else {
        retListIdx = torch::empty(
            { 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(jdata.device()));
        retNumOuterLists = retOffsets.size(0) - 1;
    }

    const torch::Tensor retJidx = JaggedTensor::jidx_from_joffsets(retOffsets, retData.size(0));
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJidx,
                                                                  retListIdx, retNumOuterLists);
}

// This corresponds to indexing with an integer
// i.e. jt = JaggedTensor([...])
//      jt[2] -> JaggedTensor([...]) where the 3rd list is selected
template <>
JaggedTensor
dispatchJaggedTensorIndexInt<torch::kCPU>(const JaggedTensor &jt, int64_t idxVal) {
    return jaggedTensorIndexIntCpu(jt, idxVal);
}
template <>
JaggedTensor
dispatchJaggedTensorIndexInt<torch::kCUDA>(const JaggedTensor &jt, int64_t idxVal) {
    c10::cuda::CUDAGuard deviceGuard(jt.device());
    return jaggedTensorIndexIntCuda(jt, idxVal);
}

// This corresponds to indexing with a slice
// i.e. jt = JaggedTensor([...])
//      jt[2:11:4] -> JaggedTensor([...]) where every fourth entry from the third to the tenth list
//      (inclusive) is selected
template <>
JaggedTensor
dispatchJaggedTensorIndexSlice<torch::kCPU>(const JaggedTensor &jt, int64_t start, int64_t end,
                                            int64_t step) {
    return jaggedTensorIndexSliceCpu(jt, start, end, step);
}
template <>
JaggedTensor
dispatchJaggedTensorIndexSlice<torch::kCUDA>(const JaggedTensor &jt, int64_t start, int64_t end,
                                             int64_t step) {
    c10::cuda::CUDAGuard deviceGuard(jt.device());
    return jaggedTensorIndexSliceCuda(jt, start, end, step);
}

// This corresponds to indexing with a JaggedTensor. i.e. using each tensor in an indexing
// JaggedTensor to index the corresponding tensor in the JaggedTensor
// i.e. jt = JaggedTensor([[t_11, t_12], [t_21, t_22, t_23], ...])
//      indices = JaggedTensor([[i_11, i_12], [i_21, i_22, i_23], ...])
//      jt[indices] -> JaggedTensor([[t_11[i_11], t_12[i_12]], [t_21[i_21], t_22[i_22], t_23[i_23]],
//      ...])
// Here indices can be integers or a boolean mask
template <>
JaggedTensor
dispatchJaggedTensorIndexJaggedTensor<torch::kCPU>(const JaggedTensor &jt,
                                                   const JaggedTensor &idx) {
    return jaggedTensorIndexJaggedTensorImpl(jt, idx);
}
template <>
JaggedTensor
dispatchJaggedTensorIndexJaggedTensor<torch::kCUDA>(const JaggedTensor &jt,
                                                    const JaggedTensor &idx) {
    c10::cuda::CUDAGuard deviceGuard(jt.device());
    return jaggedTensorIndexJaggedTensorImpl(jt, idx);
}

} // namespace ops
} // namespace detail
} // namespace fvdb