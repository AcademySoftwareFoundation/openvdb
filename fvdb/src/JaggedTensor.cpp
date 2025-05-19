// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "JaggedTensor.h"

#include "Config.h"
#include <detail/autograd/JaggedReduce.h>
#include <detail/ops/Ops.h>
#include <detail/ops/jagged/JaggedOps.h>
#include <detail/utils/Utils.h>

#include <optional>

namespace fvdb {

void
JaggedTensor::binary_op_check(const JaggedTensor &other) const {
    TORCH_CHECK(this->device() == other.device(),
                "device should match between this tensor and other tensor");
    TORCH_CHECK(mData.sizes().equals(other.jdata().sizes()),
                "data shape should match between this tensor and other tensor");
    TORCH_CHECK(mBatchIdx.sizes().equals(other.jidx().sizes()),
                "batch indices' shape should match between this tensor and other tensor");
    TORCH_CHECK(mOffsets.sizes().equals(other.joffsets().sizes()),
                "offsets shape should match between this tensor and other tensor");
    if (Config::global().pendanticErrorCheckingEnabled()) {
        // This is a slow check that we cap optionally do for correctness.
        TORCH_CHECK_VALUE(torch::equal(mOffsets, other.joffsets()),
                          "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK_VALUE(
            torch::equal(other.mListIdx, mListIdx), "JaggedTensors must have the same lshape. ",
            "This error was raised because config.pendatic_error_checking was enabled");
    }
}

torch::Tensor
JaggedTensor::joffsets_from_jidx_and_jdata(torch::Tensor jidx, torch::Tensor jdata,
                                           int64_t num_tensors) {
    return FVDB_DISPATCH_KERNEL_DEVICE(jdata.device(), [&]() {
        return detail::ops::dispatchJOffsetsForJIdx<DeviceTag>(jidx, jdata, num_tensors);
    });
}

torch::Tensor
JaggedTensor::jidx_from_joffsets(torch::Tensor joffsets, int64_t num_elements) {
    return FVDB_DISPATCH_KERNEL_DEVICE(joffsets.device(), [&]() {
        return detail::ops::dispatchJIdxForJOffsets<DeviceTag>(joffsets, num_elements);
    });
}

JaggedTensor::JaggedTensor(torch::Tensor data)
    : mData(data), mBatchIdx(torch::empty(
                       { 0 }, torch::TensorOptions().dtype(JIdxScalarType).device(data.device()))) {
    mListIdx =
        torch::empty({ 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(data.device()));
    mOffsets       = joffsets_from_jidx_and_jdata(mBatchIdx, mData, 1);
    mNumOuterLists = 1;
}

JaggedTensor::JaggedTensor(const std::vector<torch::Tensor> &tensors) {
    // TODO: (Francis): rewrite as a cuda kernel
    TORCH_CHECK(tensors.size() > 0, "empty tensor list");

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (tensors.size() == 1) {
        // If you have a single element tensor with 0 dimensions, we unsqueeze it to make it 1D
        mData = tensors[0];
        if (tensors[0].dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        mBatchIdx = torch::empty(
            { 0 }, torch::TensorOptions().dtype(JIdxScalarType).device(mData.device()));
        mOffsets =
            torch::tensor({ JOffsetsType(0), mData.size(0) },
                          torch::TensorOptions().dtype(JOffsetsScalarType).device(mData.device()));
        mListIdx = torch::empty(
            { 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device()));
        mNumOuterLists = 1;
        return;
    }

    torch::Device device = tensors[0].device();

    std::vector<torch::Tensor> jIdxs;
    mOffsets              = torch::empty({ (JOffsetsType)tensors.size() + 1 },
                                         torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    auto elementCountsAcc = mOffsets.accessor<JOffsetsType, 1>();
    elementCountsAcc[0]   = 0;

    jIdxs.reserve(tensors.size());
    std::vector<torch::Tensor> tensorsReshaped; // Reshape 0D tensors to 1D
    tensorsReshaped.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK_VALUE(tensors[i].device() == device, "All tensors must be on the same device");
        if (tensors[i].dim() == 0 && tensors[i].numel() == 1) {
            tensorsReshaped.push_back(tensors[i].view({ 1 }));
        } else {
            tensorsReshaped.push_back(tensors[i]);
        }
        jIdxs.push_back(torch::full(
            { tensorsReshaped[i].size(0) }, (int)i,
            torch::TensorOptions().dtype(JIdxScalarType).device(tensorsReshaped[i].device())));
        elementCountsAcc[i + 1] = tensorsReshaped[i].size(0);
    }
    mOffsets = mOffsets.to(tensors[0].device());
    torch::cumsum_out(mOffsets, mOffsets, 0);
    mBatchIdx = torch::cat(jIdxs, 0);
    mData     = torch::cat(tensorsReshaped, 0);
    mListIdx = torch::empty({ 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(device));
    mNumOuterLists = tensors.size();
}

JaggedTensor::JaggedTensor(const std::vector<std::vector<torch::Tensor>> &tensors) {
    // TODO: (Francis): rewrite as a cuda kernel
    torch::Device device         = torch::kCPU;
    bool          deviceIsNotSet = true;
    JOffsetsType  totalTensors   = 0;

    TORCH_CHECK(tensors.size() > 0, "empty tensor list");
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = 0; j < tensors[i].size(); j += 1) {
            if (deviceIsNotSet) {
                device         = tensors[i][j].device();
                deviceIsNotSet = false;
            }
            TORCH_CHECK_VALUE(tensors[i][j].device() == device,
                              "All tensors must be on the same device");
            totalTensors += 1;
        }
    }

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (totalTensors == 1) {
        TORCH_CHECK(tensors.size() == 1,
                    "Single tensor must be a 1D tensor. This should never happen.");
        TORCH_CHECK(tensors[0].size() == 1,
                    "Single tensor must be a 1D tensor. This should never happen.");
        mData = tensors[0][0];
        if (mData.dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        mBatchIdx = torch::empty(
            { 0 }, torch::TensorOptions().dtype(JIdxScalarType).device(mData.device()));
        mOffsets =
            torch::tensor({ JOffsetsType(0), mData.size(0) },
                          torch::TensorOptions().dtype(JOffsetsScalarType).device(mData.device()));
        mListIdx = torch::zeros(
            { 1, 2 }, torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device()));
        mNumOuterLists = 1;
        return;
    }

    // Number of elements per tensor
    std::vector<torch::Tensor> batchIdxs;
    batchIdxs.reserve(totalTensors);

    mOffsets              = torch::empty({ totalTensors + 1 },
                                         torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    auto elementCountsAcc = mOffsets.accessor<JOffsetsType, 1>();
    elementCountsAcc[0]   = 0;

    torch::Tensor listIndexes =
        torch::empty({ totalTensors, (JLIdxType)2 },
                     torch::TensorOptions().dtype(JLIdxScalarType).device(torch::kCPU));
    auto listIndexesAcc = listIndexes.accessor<JLIdxType, 2>();

    std::vector<torch::Tensor> tensorsReshaped; // Reshape 0D tensors to 1D
    tensorsReshaped.reserve(totalTensors);

    int64_t tensorCount = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = 0; j < tensors[i].size(); j += 1) {
            listIndexesAcc[tensorCount][0] = i;
            listIndexesAcc[tensorCount][1] = j;

            torch::Tensor tij = tensors[i][j];
            if (tij.dim() == 0 && tij.numel() == 1) {
                tensorsReshaped.push_back(tij.view({ 1 }));
            } else {
                tensorsReshaped.push_back(tij);
            }
            batchIdxs.push_back(
                torch::full({ tensorsReshaped[tensorCount].size(0) }, tensorCount,
                            torch::TensorOptions().dtype(JIdxScalarType).device(device)));
            elementCountsAcc[tensorCount + 1] = tensorsReshaped[tensorCount].size(0);
            tensorCount += 1;
        }
    }

    mOffsets = mOffsets.to(device);
    torch::cumsum_out(mOffsets, mOffsets, 0);
    mBatchIdx      = torch::cat(batchIdxs, 0);
    mData          = torch::cat(tensorsReshaped, 0);
    mListIdx       = listIndexes.to(device);
    mNumOuterLists = tensors.size();
}

JaggedTensor::JaggedTensor(const std::vector<int64_t> &lsizes, const torch::Tensor data) {
    // TODO: (Francis): rewrite as a cuda kernel
    TORCH_CHECK_VALUE(lsizes.size() > 0, "empty list sizes");

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (lsizes.size() == 1) {
        TORCH_CHECK_VALUE(lsizes[0] == data.size(0),
                          "Sum of list sizes must equal the number of elements in data");
        mOffsets =
            torch::tensor({ JOffsetsType(0), data.size(0) },
                          torch::TensorOptions().dtype(JOffsetsScalarType).device(data.device()));
        mListIdx = torch::empty(
            { 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(data.device()));
        mNumOuterLists = 1;
        mBatchIdx =
            torch::empty({ 0 }, torch::TensorOptions().dtype(JIdxScalarType).device(data.device()));
        mData = data;
        if (mData.dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        return;
    }

    torch::Tensor offsetsCPU =
        torch::empty({ (JOffsetsType)lsizes.size() + 1 },
                     torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    auto offsetsCPUAcc = offsetsCPU.accessor<JOffsetsType, 1>();

    mListIdx =
        torch::empty({ 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType).device(data.device()));
    mNumOuterLists = lsizes.size();

    JOffsetsType cumulativeElements = 0;
    for (size_t i = 0; i < lsizes.size(); ++i) {
        offsetsCPUAcc[i] = cumulativeElements;
        cumulativeElements += lsizes[i];
    }
    offsetsCPUAcc[lsizes.size()] = cumulativeElements;
    TORCH_CHECK_VALUE(cumulativeElements == data.size(0),
                      "Sum of list sizes must equal the number of elements in data");

    mOffsets  = offsetsCPU.to(data.device());
    mData     = data;
    mBatchIdx = jidx_from_joffsets(mOffsets, data.size(0));
}

JaggedTensor::JaggedTensor(const std::vector<std::vector<int64_t>> &lsizes,
                           const int64_t totalTensors, const torch::Tensor data) {
    // TODO (Francis) : Rewrite as a cuda kernel
    TORCH_CHECK_VALUE(lsizes.size() > 0, "empty lshape");

    // This is an implementation detail where we don't store jidx for
    // a single list since everything is just zero by default.
    if (totalTensors == 1) {
        TORCH_CHECK(lsizes.size() == 1,
                    "Single tensor must be a 1D tensor. This should never happen.");
        TORCH_CHECK(lsizes[0].size() == 1,
                    "Single tensor must be a 1D tensor. This should never happen.");
        TORCH_CHECK_VALUE(lsizes[0][0] == data.size(0), "Invalid size for data tensor.");
        mData = data;
        if (mData.dim() == 0) {
            mData = mData.unsqueeze(0);
        }
        TORCH_CHECK(mData.dim() > 0,
                    "assigned data must have shape [N, ...], but got data.dim() = 0");
        mBatchIdx = torch::empty(
            { 0 }, torch::TensorOptions().dtype(JIdxScalarType).device(mData.device()));
        mOffsets =
            torch::tensor({ JOffsetsType(0), mData.size(0) },
                          torch::TensorOptions().dtype(JOffsetsScalarType).device(mData.device()));
        mListIdx = torch::zeros(
            { 1, 2 }, torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device()));
        mNumOuterLists = 1;
        return;
    }

    torch::Tensor offsetsCPU =
        torch::empty({ (JOffsetsType)totalTensors + 1 },
                     torch::TensorOptions().dtype(JOffsetsScalarType).device(torch::kCPU));
    torch::Tensor listIdsCPU =
        torch::empty({ (JLIdxType)totalTensors, 2 },
                     torch::TensorOptions().dtype(JLIdxScalarType).device(torch::kCPU));
    auto offsetsCPUAcc = offsetsCPU.accessor<JOffsetsType, 1>();
    auto listIdsCPUAcc = listIdsCPU.accessor<JLIdxType, 2>();

    JOffsetsType cumulativeElements = 0;
    int64_t      tensorCount        = 0;
    for (size_t i = 0; i < lsizes.size(); ++i) {
        TORCH_CHECK_VALUE(lsizes[i].size() > 0, "empty lshape");
        for (size_t j = 0; j < lsizes[i].size(); j += 1) {
            offsetsCPUAcc[tensorCount]    = cumulativeElements;
            listIdsCPUAcc[tensorCount][0] = i;
            listIdsCPUAcc[tensorCount][1] = j;
            cumulativeElements += lsizes[i][j];
            tensorCount += 1;
        }
    }
    offsetsCPUAcc[totalTensors] = cumulativeElements;
    TORCH_CHECK_VALUE(cumulativeElements == data.size(0),
                      "Sum of list sizes must equal the number of elements in data");

    mOffsets       = offsetsCPU.to(data.device());
    mListIdx       = listIdsCPU.to(data.device());
    mBatchIdx      = jidx_from_joffsets(mOffsets, data.size(0));
    mData          = data;
    mNumOuterLists = lsizes.size();
}

void
JaggedTensor::recompute_lsizes_if_dirty() {
    if (!mLShapeCache.mDirty) {
        return;
    }
    mLShapeCache.clear();
    if (ldim() == 1) {
        const torch::Tensor offsetsCpu = mOffsets.cpu();
        const auto          acc        = offsetsCpu.accessor<JOffsetsType, 1>();
        for (int i = 0; i < num_tensors(); ++i) {
            const JOffsetsType startIdx = acc[i];
            const JOffsetsType endIdx   = acc[i + 1];
            mLShapeCache.mLShape1.push_back(endIdx - startIdx);
        }
        mLShapeCache.mDirty = false;
        return;
    } else if (ldim() == 2) {
        const torch::Tensor offsetsCpu = mOffsets.cpu();
        const torch::Tensor listIdxCpu = mListIdx.cpu();
        const auto          offAcc     = offsetsCpu.accessor<JOffsetsType, 1>();
        const auto          lixAcc     = listIdxCpu.accessor<JLIdxType, 2>();

        ssize_t currentList = -1;
        for (int i = 0; i < num_tensors(); ++i) {
            const JLIdxType outerIdx = lixAcc[i][0];

            if (outerIdx != currentList) {
                currentList += 1;
                mLShapeCache.mLShape2.push_back(std::vector<int64_t>());
            }
            const JOffsetsType startIdx = offAcc[i];
            const JOffsetsType endIdx   = offAcc[i + 1];
            mLShapeCache.mLShape2.back().push_back(endIdx - startIdx);
        }
        mLShapeCache.mDirty = false;
        return;
    } else {
        TORCH_CHECK(false,
                    "Unsupported list dimension. Currently JaggedTensor only supports up to 2.");
    }
}

std::vector<torch::Tensor>
JaggedTensor::unbind1() const {
    std::vector<torch::Tensor> ret(num_tensors());

    int64_t ldim = mListIdx.size(1);
    if (ldim != 1) {
        TORCH_WARN(
            "Calling unbind on a multidimensional list of jagged tensors will return a flattened list");
    }

    torch::Tensor offsetsCpu = mOffsets.cpu();
    auto          acc        = offsetsCpu.accessor<JOffsetsType, 1>();
    for (int i = 0; i < num_tensors(); ++i) {
        const JOffsetsType startIdx = acc[i];
        const JOffsetsType endIdx   = acc[i + 1];

        ret[i] = mData.index({ torch::indexing::Slice(startIdx, endIdx) });
    }

    return ret;
}

std::vector<std::vector<torch::Tensor>>
JaggedTensor::unbind2() const {
    std::vector<std::vector<torch::Tensor>> ret;

    int64_t ldim = mListIdx.size(1);

    if (ldim != 2) {
        TORCH_CHECK_VALUE(false, "Called unbind2() on a list with list dimension != 2");
    }

    torch::Tensor listIdxCpu  = mListIdx.cpu();
    torch::Tensor offsetsCpu  = mOffsets.cpu();
    ssize_t       currentList = -1;

    auto offAcc = offsetsCpu.accessor<JOffsetsType, 1>();
    auto lixAcc = listIdxCpu.accessor<JLIdxType, 2>();

    for (int i = 0; i < num_tensors(); ++i) {
        const JLIdxType outerIdx = lixAcc[i][0];

        if (outerIdx != currentList) {
            currentList += 1;
            ret.push_back(std::vector<torch::Tensor>());
        }
        const JOffsetsType startIdx = offAcc[i];
        const JOffsetsType endIdx   = offAcc[i + 1];

        ret.back().push_back(mData.index({ torch::indexing::Slice(startIdx, endIdx) }));
    }

    return ret;
}

std::vector<int64_t>
JaggedTensor::lsizes1() const {
    TORCH_CHECK(ldim() == 1, "Nesting dimension must be 1");
    const_cast<JaggedTensor *>(this)->recompute_lsizes_if_dirty();
    return mLShapeCache.mLShape1;
}

std::vector<std::vector<int64_t>>
JaggedTensor::lsizes2() const {
    TORCH_CHECK(ldim() == 2, "Nesting dimension must be 2");
    const_cast<JaggedTensor *>(this)->recompute_lsizes_if_dirty();
    return mLShapeCache.mLShape2;
}

int64_t
JaggedTensor::ldim() const {
    TORCH_CHECK_VALUE(mListIdx.dim() == 2, "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(mListIdx.numel() == 0 || mListIdx.size(0) == (mOffsets.size(0) - 1),
                      "Corrupt list indices. This should never happen");
    return mListIdx.size(1);
}

std::vector<int64_t>
JaggedTensor::esizes() const {
    std::vector<int64_t> sizes;
    for (size_t i = 1; i < mData.sizes().size(); i++) {
        sizes.push_back(mData.size(i));
    }
    return sizes;
}

int64_t
JaggedTensor::edim() const {
    return mData.dim() > 0 ? mData.dim() - 1 : 0;
}

JaggedTensor
JaggedTensor::jagged_like(torch::Tensor data) const {
    TORCH_CHECK_VALUE(data.dim() > 0,
                      "assigned data must have shape [N, ...], but got data.dim() = 0");
    TORCH_CHECK_VALUE(mListIdx.dim() == 2, "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(mListIdx.numel() == 0 || mListIdx.size(0) == (mOffsets.size(0) - 1),
                      "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(data.size(0) == mData.size(0),
                      "Assigned data must have the same number of elements as the JaggedTensor");

    JaggedTensor ret;
    ret.mBatchIdx      = jidx();
    ret.mOffsets       = joffsets();
    ret.mListIdx       = jlidx();
    ret.mNumOuterLists = mNumOuterLists;
    ret.mData          = data.to(device());
    ret.mLShapeCache   = mLShapeCache;
    return ret;
}

JaggedTensor
JaggedTensor::from_data_indices_and_list_ids(torch::Tensor data, torch::Tensor indices,
                                             torch::Tensor list_ids, int64_t num_tensors) {
    JaggedTensor ret;
    ret.mData          = data;
    ret.mBatchIdx      = indices;
    ret.mListIdx       = list_ids;
    ret.mOffsets       = joffsets_from_jidx_and_jdata(indices, data, num_tensors);
    ret.mNumOuterLists = ret.joffsets().size(0) - 1;
    ret.mLShapeCache.markDirty();
    return ret;
}

JaggedTensor
JaggedTensor::from_data_offsets_and_list_ids(torch::Tensor data, torch::Tensor offsets,
                                             torch::Tensor list_ids) {
    TORCH_CHECK_VALUE(
        list_ids.dim() == 2,
        "Invalid list indices when constructing JaggedTensor from data, offsets, and list indices");
    TORCH_CHECK_VALUE(
        list_ids.numel() == 0 || list_ids.size(0) == (offsets.size(0) - 1),
        "Invalid list indices when constructing JaggedTensor from data, offsets, and list indices");
    TORCH_CHECK_VALUE(
        offsets.dim() == 1,
        "Invalid offsets when constructing JaggedTensor from data, offsets, and list indices");

    JaggedTensor ret;
    ret.mData          = data;
    ret.mOffsets       = offsets;
    ret.mListIdx       = list_ids;
    ret.mNumOuterLists = offsets.size(0) - 1;
    ret.mBatchIdx      = jidx_from_joffsets(offsets, data.size(0));
    ret.mLShapeCache.markDirty();
    return ret;
}

JaggedTensor
JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(torch::Tensor jdata, torch::Tensor joffsets,
                                                       torch::Tensor jidx, torch::Tensor lidx,
                                                       int64_t numOuterLists) {
    TORCH_CHECK_VALUE(
        lidx.dim() == 2,
        "Invalid list indices when constructing JaggedTensor from data, offsets, indices, list indices");
    TORCH_CHECK_VALUE(
        lidx.numel() == 0 || lidx.size(0) == (joffsets.size(0) - 1),
        "Invalid list indices when constructing JaggedTensor from data, offsets, indices, list indices");
    TORCH_CHECK_VALUE(
        joffsets.dim() == 1,
        "Invalid offsets when constructing JaggedTensor from data, offsets, indices, list indices");
    JaggedTensor ret;
    ret.mData          = jdata;
    ret.mOffsets       = joffsets;
    ret.mListIdx       = lidx;
    ret.mNumOuterLists = numOuterLists;
    ret.mBatchIdx      = jidx;
    ret.mLShapeCache.markDirty();
    ret.recompute_lsizes_if_dirty();
    return ret;
}

void
JaggedTensor::set_data(const torch::Tensor &data) {
    TORCH_CHECK_VALUE(data.dim() > 0,
                      "assigned data must have shape [N, ...], but got data.dim() = 0");
    TORCH_CHECK_VALUE((data.device() == mBatchIdx.device()) ||
                          (mBatchIdx.numel() == 0 && num_tensors() == 1),
                      "Incorrect device for data");
    TORCH_CHECK_VALUE(data.device() == mOffsets.device(), "Incorrect device for data");
    TORCH_CHECK_VALUE(mListIdx.dim() == 2, "Corrupt list indices. This should never happen");
    TORCH_CHECK_VALUE(mListIdx.numel() == 0 || mListIdx.size(0) == (mOffsets.size(0) - 1),
                      "Corrupt list indices. This should never happen");

    if (mBatchIdx.size(0) == 0) {
        TORCH_CHECK(mOffsets.dim() == 1, "bad offsets. this should never happen");
        TORCH_CHECK(mOffsets.size(0) == (num_outer_lists() + 1),
                    "bad offsets. this should never happen");
        TORCH_CHECK_VALUE(data.size(0) == mData.size(0), "assigned data must have shape [N, ...]");
    } else {
        TORCH_CHECK_VALUE(data.size(0) == mBatchIdx.size(0),
                          "assigned data must have shape [N, ...]");
    }
    mData = data;
}

JaggedTensor
JaggedTensor::rmask(const torch::Tensor &mask) const {
    TORCH_CHECK(mask.device() == mBatchIdx.device(),
                "mask must be on the same device as the JaggedTensor");
    TORCH_CHECK(mask.dim() == 1, "mask must be 1-dimensional");
    TORCH_CHECK(mask.size(0) == mData.size(0),
                "mask must have the same size as the first dimension of the JaggedTensor");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be of type bool");

    TORCH_CHECK((mask.size(0) == mBatchIdx.size(0)) ||
                    (mBatchIdx.size(0) == 0 && mOffsets.size(0) == 2),
                "Bad jidx. This should never happen. mask.size(0) = ", mask.size(0),
                " mBatchIdx.size(0) = ", mBatchIdx.size(0));
    const torch::Tensor retData     = mData.index({ mask, "..." });
    const torch::Tensor retBatchIds = mBatchIdx.size(0) > 0 ? mBatchIdx.index({ mask }) : mBatchIdx;
    const torch::Tensor retOffsets =
        joffsets_from_jidx_and_jdata(retBatchIds, retData, num_tensors());
    return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retBatchIds,
                                                                  mListIdx, mNumOuterLists);
}

JaggedTensor
JaggedTensor::index(JaggedTensorIndex idx) const {
    if (idx.is_integer()) {
        return FVDB_DISPATCH_KERNEL_DEVICE(mData.device(), [&]() {
            return detail::ops::dispatchJaggedTensorIndexInt<DeviceTag>(*this, idx.integer());
        });
    } else if (idx.is_slice()) {
        int64_t start = idx.slice().start().as_int_unchecked();
        int64_t end   = idx.slice().stop().as_int_unchecked();
        int64_t step  = idx.slice().step().as_int_unchecked();
        TORCH_CHECK_VALUE(step >= 1, "step in slice must be >= 1");

        // Deal with symbolic int
        if (start >= at::indexing::INDEX_MAX) {
            start = mNumOuterLists;
        }
        if (end <= at::indexing::INDEX_MIN) {
            end = 0;
        }

        return FVDB_DISPATCH_KERNEL_DEVICE(mData.device(), [&]() {
            return detail::ops::dispatchJaggedTensorIndexSlice<DeviceTag>(*this, start, end, step);
        });
    } else if (idx.is_ellipsis()) {
        return *this;
    } else if (idx.is_jagged_tensor()) {
        const JaggedTensor &jtIndices = idx.jagged_tensor();
        return FVDB_DISPATCH_KERNEL_DEVICE(mData.device(), [&]() {
            return detail::ops::dispatchJaggedTensorIndexJaggedTensor<DeviceTag>(*this, jtIndices);
        });
    } else {
        TORCH_CHECK_VALUE(false, "Unsupported indexing operation");
    }
}

JaggedTensor
JaggedTensor::jreshape(const std::vector<int64_t> &lsizes) const {
    return JaggedTensor(lsizes, mData);
}

JaggedTensor
JaggedTensor::jreshape(const std::vector<std::vector<int64_t>> &lsizes) const {
    return JaggedTensor(lsizes, num_tensors(), mData);
}

JaggedTensor
JaggedTensor::jreshape_as(const JaggedTensor &other) const {
    return other.jagged_like(mData);
}

JaggedTensor
JaggedTensor::jflatten(const int64_t dim) const {
    int64_t jdim = dim;
    if (dim < 0) {
        jdim += ldim();
    }
    TORCH_CHECK_INDEX(jdim >= 0 && jdim < ldim(), "Invalid dimension to flatten");

    if (ldim() == 2) {
        if (jdim == 1) {
            torch::Tensor newJIdx = mListIdx.index({ torch::indexing::Slice(), 0 })
                                        .index({ mBatchIdx.to(torch::kInt) })
                                        .to(JIdxScalarType);
            torch::Tensor newOffsets =
                joffsets_from_jidx_and_jdata(newJIdx, mData, num_outer_lists());
            return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                mData, newOffsets, newJIdx,
                torch::empty({ 0, 1 },
                             torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device())),
                newOffsets.size(0) - 1);
        } else {
            return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                mData, mOffsets, mBatchIdx,
                torch::empty({ 0, 1 },
                             torch::TensorOptions().dtype(JLIdxScalarType).device(mData.device())),
                mOffsets.size(0) - 1);
        }
    } else if (ldim() == 1) {
        return JaggedTensor(mData);
    } else {
        TORCH_CHECK(false,
                    "Unsupported list dimension. Currently JaggedTensor only supports up to 2.");
    }
}
// JaggedTensor JaggedTensor::jagged_argsort() {
//     jidx_from_joffsets(); // why??
//     torch::Tensor argsortIdx = FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
//         return detail::ops::dispatchJaggedArgsort<DeviceTag>(*this);
//     });
//
//     return jagged_like(argsortIdx);
// }

JaggedTensor
JaggedTensor::jsum(int64_t dim, bool keepdim) const {
    const int64_t jdim = mData.dim();
    TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim < jdim, "dim must be between ", -(jdim - 1),
                      " and ", jdim - 1, " inclusive");
    if (dim < 0) {
        dim += jdim;
    }

    if (dim == 0) {
        torch::Tensor retData;
        if (mBatchIdx.size(0) == 0) {
            retData = mData.sum(0).unsqueeze(0);
        } else {
            retData =
                detail::autograd::JaggedSum::apply(jdata(), jidx(), joffsets(), num_tensors())[0];
        }
        const torch::Tensor retOffsets = torch::arange(
            0, retData.size(0) + 1,
            torch::TensorOptions().dtype(JOffsetsScalarType).device(retData.device()));
        const torch::Tensor retJidx = jidx_from_joffsets(retOffsets, retData.size(0));

        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retData, retOffsets, retJidx,
                                                                      mListIdx, mNumOuterLists);
    } else {
        return jagged_like(mData.sum(dim, keepdim));
    }
}

std::vector<JaggedTensor>
JaggedTensor::jmin(int64_t dim, bool keepdim) const {
    const int64_t jdim = mData.dim();
    TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim <= jdim, "dim must be between ", -(jdim - 1),
                      " and ", jdim - 1, " inclusive");
    if (dim < 0) {
        dim += jdim;
    }

    if (dim == 0) {
        torch::Tensor minVals, minIndices;
        if (mBatchIdx.size(0) == 0) {
            auto minTuple = mData.min(0);
            minVals       = std::get<0>(minTuple).unsqueeze(0);
            minIndices    = std::get<1>(minTuple).unsqueeze(0);
        } else {
            auto minTuple =
                detail::autograd::JaggedMin::apply(jdata(), jidx(), joffsets(), num_tensors());
            minVals    = minTuple[0];
            minIndices = minTuple[1];
        }

        const torch::Tensor retOffsets = torch::arange(
            0, minVals.size(0) + 1,
            torch::TensorOptions().dtype(JOffsetsScalarType).device(minVals.device()));
        const torch::Tensor retJidx = jidx_from_joffsets(retOffsets, minVals.size(0));

        JaggedTensor retVals = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            minVals, retOffsets, retJidx, mListIdx, mNumOuterLists);
        JaggedTensor retIdxs = retVals.jagged_like(minIndices);
        return { retVals, retIdxs };
    } else {
        auto          minTuple   = mData.min(dim, keepdim);
        torch::Tensor minVals    = std::get<0>(minTuple);
        torch::Tensor minIndices = std::get<1>(minTuple);
        return { jagged_like(minVals), jagged_like(minIndices) };
    }
}

JaggedTensor
JaggedTensor::jsqueeze(std::optional<int64_t> dim) const {
    torch::Tensor jdataSqueezed = dim.has_value() ? mData.squeeze(dim.value()) : mData.squeeze();
    if (jdataSqueezed.dim() == 0) {
        jdataSqueezed = jdataSqueezed.unsqueeze(0);
    }
    return jagged_like(jdataSqueezed);
}

std::vector<JaggedTensor>
JaggedTensor::jmax(int64_t dim, bool keepdim) const {
    const int64_t jdim = mData.dim();
    TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim <= jdim, "dim must be between ", -(jdim - 1),
                      " and ", jdim - 1, " inclusive");
    if (dim < 0) {
        dim += jdim;
    }

    if (dim == 0) {
        torch::Tensor maxVals, maxIndices;
        if (mBatchIdx.size(0) == 0) {
            auto maxTuple = mData.max(0);
            maxVals       = std::get<0>(maxTuple).unsqueeze(0);
            maxIndices    = std::get<1>(maxTuple).unsqueeze(0);
        } else {
            auto maxTuple =
                detail::autograd::JaggedMax::apply(jdata(), jidx(), joffsets(), num_tensors());
            maxVals    = maxTuple[0];
            maxIndices = maxTuple[1];
        }

        const torch::Tensor retOffsets = torch::arange(
            0, maxVals.size(0) + 1,
            torch::TensorOptions().dtype(JOffsetsScalarType).device(maxVals.device()));
        const torch::Tensor retJidx = jidx_from_joffsets(retOffsets, maxVals.size(0));
        JaggedTensor        retVals = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            maxVals, retOffsets, retJidx, mListIdx, mNumOuterLists);
        JaggedTensor retIdxs = retVals.jagged_like(maxIndices);
        return { retVals, retIdxs };
    } else {
        auto          maxTuple   = mData.max(dim, keepdim);
        torch::Tensor maxVals    = std::get<0>(maxTuple);
        torch::Tensor maxIndices = std::get<1>(maxTuple);
        return { jagged_like(maxVals), jagged_like(maxIndices) };
    }
}

JaggedTensor
JaggedTensor::jcat(const std::vector<JaggedTensor> &vec, std::optional<int64_t> dimension) {
    // Null dimension is just list concatenation
    if (!dimension.has_value()) {
        TORCH_CHECK_VALUE(vec.size() > 0, "Empty jagged tensor list");

        // Concat along the batch dimension
        std::vector<torch::Tensor> data;
        std::vector<torch::Tensor> offsets;
        std::vector<torch::Tensor> lidx;
        JOffsetsType               curOffset     = 0;
        int64_t                    totalLists    = 0;
        torch::Tensor              curListOffset = torch::zeros(
            { 1, vec[0].mListIdx.size(1) },
            torch::TensorOptions().dtype(JLIdxScalarType).device(vec[0].mData.device()));
        for (size_t i = 0; i < vec.size(); ++i) {
            const auto &jvec = vec[i];
            TORCH_CHECK_VALUE(jvec.mData.device() == vec[0].mData.device(),
                              "All JaggedTensors must be on the same device");
            TORCH_CHECK_VALUE(jvec.mListIdx.size(1) == vec[0].mListIdx.size(1),
                              "All JaggedTensors must have the same list dimension");
            TORCH_CHECK_VALUE(jvec.scalar_type() == vec[0].scalar_type(),
                              "All JaggedTensors must have the same scalar type");

            data.push_back(jvec.mData);
            if (i < vec.size() - 1) {
                offsets.push_back(jvec.mOffsets.index({ torch::indexing::Slice(0, -1) }) +
                                  curOffset);
            } else {
                offsets.push_back(jvec.mOffsets + curOffset);
            }
            lidx.push_back(jvec.mListIdx + curListOffset);
            curOffset += jvec.mData.size(0);
            curListOffset[0][0] += jvec.mNumOuterLists;
            totalLists += jvec.mNumOuterLists;
        }
        const torch::Tensor retJData    = torch::cat(data, 0);
        const torch::Tensor retJOffsets = torch::cat(offsets, 0);
        const torch::Tensor retJidx     = jidx_from_joffsets(retJOffsets, retJData.size(0));
        const torch::Tensor retLidx     = torch::cat(lidx, 0);
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(retJData, retJOffsets,
                                                                      retJidx, retLidx, totalLists);
    } else {
        int64_t dim = dimension.value();
        TORCH_CHECK_VALUE(vec.size() > 0, "empty tensor list");
        const int64_t jdim = vec[0].mData.dim();
        TORCH_CHECK_INDEX(dim >= -(jdim - 1) && dim <= jdim, "dim must be between ", -(jdim - 1),
                          " and ", jdim - 1, " inclusive");
        if (dim < 0) {
            dim += jdim;
        }

        if (dim == 0) {
            return FVDB_DISPATCH_KERNEL_DEVICE(
                vec[0].device(), [&]() { return detail::ops::dispatchJCat0<DeviceTag>(vec); });
        } else {
            std::vector<torch::Tensor> data;
            for (const auto &jvec: vec) {
                data.push_back(jvec.mData);
            }
            return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                torch::cat(data, dim), vec[0].mOffsets, vec[0].mBatchIdx, vec[0].mListIdx,
                vec[0].mNumOuterLists);
        }
    }
}

JaggedTensor
JaggedTensor::to(at::TensorOptions options, bool non_blocking, bool copy,
                 std::optional<at::MemoryFormat> memory_format) const {
    JaggedTensor ret = *this;
    ret.mData        = ret.mData.to(options, non_blocking, copy, memory_format);
    ret.mBatchIdx    = ret.mBatchIdx.to(ret.mData.device(), non_blocking, copy, memory_format);
    ret.mOffsets     = ret.mOffsets.to(ret.mData.device(), non_blocking, copy, memory_format);
    ret.mListIdx     = ret.mListIdx.to(ret.mData.device(), non_blocking, copy, memory_format);
    return ret;
}

JaggedTensor
JaggedTensor::to(std::optional<torch::ScalarType> dtype, std::optional<at::Layout> layout,
                 std::optional<at::Device> device, std::optional<bool> pin_memory,
                 bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format) {
    JaggedTensor ret = *this;
    ret.mData = ret.mData.to(dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
    ret.mBatchIdx = ret.mBatchIdx.to(JIdxScalarType, layout, device, pin_memory, non_blocking, copy,
                                     memory_format);
    ret.mOffsets  = ret.mOffsets.to(JOffsetsScalarType, layout, device, pin_memory, non_blocking,
                                    copy, memory_format);
    ret.mListIdx  = ret.mListIdx.to(JLIdxScalarType, layout, device, pin_memory, non_blocking, copy,
                                    memory_format);
    return ret;
}
JaggedTensor
JaggedTensor::to(torch::Device device, torch::ScalarType dtype, bool non_blocking, bool copy,
                 std::optional<at::MemoryFormat> memory_format) {
    JaggedTensor ret = *this;
    ret.mData        = ret.mData.to(device, dtype, non_blocking, copy, memory_format);
    ret.mBatchIdx    = ret.mBatchIdx.to(device, non_blocking, copy, memory_format);
    ret.mOffsets     = ret.mOffsets.to(device, non_blocking, copy, memory_format);
    ret.mListIdx     = ret.mListIdx.to(device, non_blocking, copy, memory_format);
    return ret;
}
JaggedTensor
JaggedTensor::to(torch::ScalarType dtype, bool non_blocking, bool copy,
                 std::optional<at::MemoryFormat> memory_format) {
    JaggedTensor ret = *this;
    ret.mData        = ret.mData.to(dtype, non_blocking, copy, memory_format);
    ret.mBatchIdx    = ret.mBatchIdx.to(JIdxScalarType, non_blocking, copy, memory_format);
    ret.mOffsets     = ret.mOffsets.to(JOffsetsScalarType, non_blocking, copy, memory_format);
    ret.mListIdx     = ret.mListIdx.to(JLIdxScalarType, non_blocking, copy, memory_format);
    return ret;
}

JaggedTensor
JaggedTensor::sqrt() const {
    return jagged_like(torch::sqrt(mData));
}
JaggedTensor
JaggedTensor::abs() const {
    return jagged_like(torch::abs(mData));
}

JaggedTensor
JaggedTensor::floor() const {
    return jagged_like(torch::floor(mData));
}

JaggedTensor
JaggedTensor::ceil() const {
    return jagged_like(torch::ceil(mData));
}

JaggedTensor
JaggedTensor::round(int decimals) const {
    return jagged_like(torch::round(mData, decimals));
}

JaggedTensor &
JaggedTensor::sqrt_() {
    mData.sqrt_();
    return *this;
}
JaggedTensor &
JaggedTensor::abs_() {
    mData.abs_();
    return *this;
}

JaggedTensor &
JaggedTensor::floor_() {
    mData.floor_();
    return *this;
}

JaggedTensor &
JaggedTensor::ceil_() {
    mData.ceil_();
    return *this;
}

JaggedTensor &
JaggedTensor::round_(int decimals) {
    mData.round_(decimals);
    return *this;
}

const JaggedTensor &
JaggedTensor::set_requires_grad(bool requires_grad) const {
    mData.set_requires_grad(requires_grad);
    return *this;
}

bool
JaggedTensor::requires_grad() const {
    return mData.requires_grad();
}

JaggedTensor
JaggedTensor::detach() const {
    return jagged_like(mData.detach());
}

JaggedTensor
JaggedTensor::clone() const {
    return jagged_like(mData.clone());
}

JaggedTensor
JaggedTensor::operator+(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData + other.mData);
}
JaggedTensor
JaggedTensor::operator+(const int other) const {
    return jagged_like(mData + other);
}
JaggedTensor
JaggedTensor::operator+(const float other) const {
    return jagged_like(mData + other);
}
JaggedTensor
JaggedTensor::operator+(const torch::Tensor &other) const {
    return jagged_like(mData + other);
}

JaggedTensor &
JaggedTensor::operator+=(const JaggedTensor &other) {
    binary_op_check(other);
    mData += other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator+=(const int other) {
    mData += other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator+=(const float other) {
    mData += other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator+=(const torch::Tensor &other) {
    mData += other;
    return *this;
}

JaggedTensor
JaggedTensor::operator-(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData - other.mData);
}
JaggedTensor
JaggedTensor::operator-(const int other) const {
    return jagged_like(mData - other);
}
JaggedTensor
JaggedTensor::operator-(const float other) const {
    return jagged_like(mData - other);
}
JaggedTensor
JaggedTensor::operator-(const torch::Tensor &other) const {
    return jagged_like(mData - other);
}

JaggedTensor
JaggedTensor::operator-() const {
    return jagged_like(-mData);
}

JaggedTensor &
JaggedTensor::operator-=(const JaggedTensor &other) {
    binary_op_check(other);
    mData -= other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator-=(const int other) {
    mData -= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator-=(const float other) {
    mData -= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator-=(const torch::Tensor &other) {
    mData -= other;
    return *this;
}

JaggedTensor
JaggedTensor::operator*(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData * other.mData);
}
JaggedTensor
JaggedTensor::operator*(const int other) const {
    return jagged_like(mData * other);
}
JaggedTensor
JaggedTensor::operator*(const float other) const {
    return jagged_like(mData * other);
}
JaggedTensor
JaggedTensor::operator*(const torch::Tensor &other) const {
    return jagged_like(mData * other);
}

JaggedTensor &
JaggedTensor::operator*=(const JaggedTensor &other) {
    binary_op_check(other);
    mData *= other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator*=(const int other) {
    mData *= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator*=(const float other) {
    mData *= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator*=(const torch::Tensor &other) {
    mData *= other;
    return *this;
}

JaggedTensor
JaggedTensor::operator/(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData / other.mData);
}
JaggedTensor
JaggedTensor::operator/(const int other) const {
    return jagged_like(mData / other);
}
JaggedTensor
JaggedTensor::operator/(const float other) const {
    return jagged_like(mData / other);
}
JaggedTensor
JaggedTensor::operator/(const torch::Tensor &other) const {
    return jagged_like(mData / other);
}

JaggedTensor &
JaggedTensor::operator/=(const JaggedTensor &other) {
    binary_op_check(other);
    mData /= other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator/=(const int other) {
    mData /= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator/=(const float other) {
    mData /= other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator/=(const torch::Tensor &other) {
    mData /= other;
    return *this;
}

JaggedTensor
JaggedTensor::floordiv(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(torch::floor_divide(mData, other.mData));
}
JaggedTensor
JaggedTensor::floordiv(const int other) const {
    return jagged_like(torch::floor_divide(mData, other));
}
JaggedTensor
JaggedTensor::floordiv(const float other) const {
    return jagged_like(torch::floor_divide(mData, other));
}
JaggedTensor
JaggedTensor::floordiv(const torch::Tensor &other) const {
    return jagged_like(torch::floor_divide(mData, other));
}

JaggedTensor &
JaggedTensor::floordiveq(const JaggedTensor &other) {
    binary_op_check(other);
    mData.floor_divide_(other.mData);
    return *this;
}
JaggedTensor &
JaggedTensor::floordiveq(const int other) {
    mData = torch::floor_divide(mData, other);
    return *this;
}
JaggedTensor &
JaggedTensor::floordiveq(const float other) {
    mData = torch::floor_divide(mData, other);
    return *this;
}
JaggedTensor &
JaggedTensor::floordiveq(const torch::Tensor &other) {
    mData.floor_divide_(other);
    return *this;
}

JaggedTensor
JaggedTensor::operator%(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData % other.mData);
}
JaggedTensor
JaggedTensor::operator%(const int other) const {
    return jagged_like(mData % other);
}
JaggedTensor
JaggedTensor::operator%(const float other) const {
    return jagged_like(mData % other);
}
JaggedTensor
JaggedTensor::operator%(const torch::Tensor &other) const {
    return jagged_like(mData % other);
}

JaggedTensor &
JaggedTensor::operator%=(const JaggedTensor &other) {
    binary_op_check(other);
    mData = mData % other.mData;
    return *this;
}
JaggedTensor &
JaggedTensor::operator%=(const int other) {
    mData = mData % other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator%=(const float other) {
    mData = mData % other;
    return *this;
}
JaggedTensor &
JaggedTensor::operator%=(const torch::Tensor &other) {
    mData = mData % other;
    return *this;
}

JaggedTensor
JaggedTensor::pow(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData.pow(other.mData));
}
JaggedTensor
JaggedTensor::pow(const int other) const {
    return jagged_like(mData.pow(other));
}
JaggedTensor
JaggedTensor::pow(const float other) const {
    return jagged_like(mData.pow(other));
}
JaggedTensor
JaggedTensor::pow(const torch::Tensor &other) const {
    return jagged_like(mData.pow(other));
}

JaggedTensor &
JaggedTensor::poweq(const JaggedTensor &other) {
    binary_op_check(other);
    mData.pow_(other.mData);
    return *this;
}
JaggedTensor &
JaggedTensor::poweq(const int other) {
    mData = mData.pow(other);
    return *this;
}
JaggedTensor &
JaggedTensor::poweq(const float other) {
    mData = mData.pow(other);
    return *this;
}
JaggedTensor &
JaggedTensor::poweq(const torch::Tensor &other) {
    mData.pow_(other);
    return *this;
}

JaggedTensor
JaggedTensor::operator>(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData > other.mData);
}
JaggedTensor
JaggedTensor::operator>(const int other) const {
    return jagged_like(mData > other);
}
JaggedTensor
JaggedTensor::operator>(const float other) const {
    return jagged_like(mData > other);
}
JaggedTensor
JaggedTensor::operator>(const torch::Tensor &other) const {
    return jagged_like(mData > other);
}

JaggedTensor
JaggedTensor::operator>=(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData >= other.mData);
}
JaggedTensor
JaggedTensor::operator>=(const int other) const {
    return jagged_like(mData >= other);
}
JaggedTensor
JaggedTensor::operator>=(const float other) const {
    return jagged_like(mData >= other);
}
JaggedTensor
JaggedTensor::operator>=(const torch::Tensor &other) const {
    return jagged_like(mData >= other);
}

JaggedTensor
JaggedTensor::operator<(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData < other.mData);
}
JaggedTensor
JaggedTensor::operator<(const int other) const {
    return jagged_like(mData < other);
}
JaggedTensor
JaggedTensor::operator<(const float other) const {
    return jagged_like(mData < other);
}
JaggedTensor
JaggedTensor::operator<(const torch::Tensor &other) const {
    return jagged_like(mData < other);
}

JaggedTensor
JaggedTensor::operator<=(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData <= other.mData);
}
JaggedTensor
JaggedTensor::operator<=(const int other) const {
    return jagged_like(mData <= other);
}
JaggedTensor
JaggedTensor::operator<=(const float other) const {
    return jagged_like(mData <= other);
}
JaggedTensor
JaggedTensor::operator<=(const torch::Tensor &other) const {
    return jagged_like(mData <= other);
}

JaggedTensor
JaggedTensor::operator==(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData == other.mData);
}
JaggedTensor
JaggedTensor::operator==(const int other) const {
    return jagged_like(mData == other);
}
JaggedTensor
JaggedTensor::operator==(const float other) const {
    return jagged_like(mData == other);
}
JaggedTensor
JaggedTensor::operator==(const torch::Tensor &other) const {
    return jagged_like(mData == other);
}

JaggedTensor
JaggedTensor::operator!=(const JaggedTensor &other) const {
    binary_op_check(other);
    return jagged_like(mData != other.mData);
}
JaggedTensor
JaggedTensor::operator!=(const int other) const {
    return jagged_like(mData != other);
}
JaggedTensor
JaggedTensor::operator!=(const float other) const {
    return jagged_like(mData != other);
}
JaggedTensor
JaggedTensor::operator!=(const torch::Tensor &other) const {
    return jagged_like(mData != other);
}

} // namespace fvdb
