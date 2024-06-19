#include "JaggedTensor.h"

#include "detail/ops/Ops.h"
#include "detail/autograd/JaggedReduce.h"
#include "detail/ops/jagged/JaggedOps.h"

namespace fvdb {

void JaggedTensor::computeJOffsetsFromJIdx(const int64_t batchSize) {
    if (mBatchIdx.size(0) == 0 && batchSize == 1) {
        torch::Tensor ret = torch::empty({1, 2}, torch::kInt64);
        auto acc = ret.accessor<int64_t, 2>();
        acc[0][0] = 0;
        acc[0][1] = mData.size(0);
        mOffsets = ret.to(mData.device());
        return;
    }

    // Get the number of unique batch indices assuming jidx is always sorted
    // It should be of the form [0, ..., 0, 1, ..., 1, 3, ..., 3, ...]
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> uniqueRes = torch::unique_dim(mBatchIdx, 0, false, false, true);
    torch::Tensor uniqueBatchValues = std::get<0>(uniqueRes);  // [0, 1, 3, ...]
    torch::Tensor uniqueBatchCounts = std::get<2>(uniqueRes);  // [n0, n1, n3, ...]

    torch::Tensor fullBatchCounts = torch::full(
            {batchSize}, 0, torch::TensorOptions().dtype(torch::kInt64).device(mData.device()));
    fullBatchCounts.index_put_({uniqueBatchValues.to(torch::kLong)}, uniqueBatchCounts);

    torch::Tensor cumOffsets = torch::cumsum(fullBatchCounts, 0, torch::kInt64);
    mOffsets = torch::stack({cumOffsets - fullBatchCounts, cumOffsets}, 1);
}

void JaggedTensor::computeJidxFromJOffsets() {
    FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        mBatchIdx = detail::ops::dispatchJIdxForJOffsets<DeviceTag>(mOffsets, mData.size(0));
    });
}

JaggedTensor::JaggedTensor(torch::Tensor data)
    : mData(data), mBatchIdx(torch::empty({0}, torch::TensorOptions().dtype(torch::kInt16).device(data.device()))) {
    computeJOffsetsFromJIdx(1);
}

JaggedTensor::JaggedTensor(const std::vector<torch::Tensor>& tensors) {
    TORCH_CHECK(tensors.size() > 0, "empty tensor list");

    if (tensors.size() == 1) {
        // If you have a single element tensor with 0 dimensions, we unsqueeze it to make it 1D
        if (tensors[0].numel() == 1 && tensors[0].dim() == 0) {
            mData = tensors[0].view({1}).clone();
        } else {
            mData = tensors[0].clone();
        }
        mBatchIdx = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt16).device(mData.device()));
        mOffsets = torch::tensor({int64_t(0), mData.size(0)}, torch::TensorOptions().dtype(torch::kInt64).device(mData.device())).unsqueeze(0);
        return;
    }

    torch::Device device = tensors[0].device();

    std::vector<torch::Tensor> batchIdxs;
    torch::Tensor elementCounts = torch::empty({(int64_t) tensors.size()}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto elementCountsAcc = elementCounts.accessor<int64_t, 1>();

    batchIdxs.reserve(tensors.size());
    std::vector<torch::Tensor> tensorsReshaped;  // Reshape 0D tensors to 1D
    tensorsReshaped.reserve(tensors.size());
    for (int i = 0; i < (int) tensors.size(); i += 1) {
        TORCH_CHECK_VALUE(tensors[i].device() == device, "All tensors must be on the same device");
        if (tensors[i].dim() == 0 && tensors[i].numel() == 1) {
            tensorsReshaped.push_back(tensors[i].view({1}));
        } else {
            tensorsReshaped.push_back(tensors[i]);
        }
        batchIdxs.push_back(torch::full({tensorsReshaped[i].size(0)}, i, torch::TensorOptions().dtype(torch::kInt16).device(tensorsReshaped[i].device())));
        elementCountsAcc[i] = tensorsReshaped[i].size(0);
    }
    torch::Tensor cumCounts = torch::cumsum(elementCounts, 0);
    mOffsets = torch::stack({cumCounts - elementCounts, cumCounts}, 1).to(tensors[0].device());
    mBatchIdx = torch::cat(batchIdxs, 0);
    mData = torch::cat(tensorsReshaped, 0);

}


JaggedTensor JaggedTensor::jagged_like(torch::Tensor data) const {
    JaggedTensor ret;
    ret.mBatchIdx = jidx();
    ret.mOffsets = joffsets();
    ret.set_data(data.to(device()));
    return ret;
}

JaggedTensor JaggedTensor::from_data_and_jidx(torch::Tensor data, torch::Tensor jidx, int64_t batch_size) {
    JaggedTensor ret;
    ret.mData = data;
    ret.mBatchIdx = jidx;
    ret.computeJOffsetsFromJIdx(batch_size);
    return ret;
}

JaggedTensor JaggedTensor::from_data_and_offsets(torch::Tensor data, torch::Tensor offsets) {
    JaggedTensor ret;
    ret.mData = data;
    ret.mOffsets = offsets;
    ret.computeJidxFromJOffsets();
    return ret;
}

JaggedTensor JaggedTensor::from_data_offsets_and_jidx_unsafe(torch::Tensor data, torch::Tensor offsets, torch::Tensor jidx) {
    JaggedTensor ret;
    ret.mData = data;
    ret.mOffsets = offsets;
    ret.mBatchIdx = jidx;
    return ret;
}

void JaggedTensor::set_data(const torch::Tensor& data) {
    TORCH_CHECK_VALUE(data.dim() > 0, "assigned data must have shape [N, ...], but got data.dim() = 0");
    TORCH_CHECK_VALUE(data.device() == mBatchIdx.device(), "Incorrect device for data");

    if (mBatchIdx.size(0) == 0) {
        TORCH_CHECK(mOffsets.dim() == 2, "bad offsets. this should never happen");
        TORCH_CHECK(mOffsets.size(0) == batch_size(), "bad offsets. this should never happen");
        TORCH_CHECK(mOffsets.size(1) == 2, "bad offsets, this should never happen.");
        TORCH_CHECK_VALUE(mOffsets[0][1].item<int64_t>() == data.size(0),  "assigned data must have shape [N, ...]");
    } else {
        TORCH_CHECK_VALUE(data.size(0) == mBatchIdx.size(0), "assigned data must have shape [N, ...]");
    }
    mData = data;
}

JaggedTensor JaggedTensor::r_masked_select(const torch::Tensor& mask) const {
    TORCH_CHECK(mask.device() == mBatchIdx.device(), "mask must be on the same device as the JaggedTensor");
    TORCH_CHECK(mask.dim() == 1, "mask must be 1-dimensional");
    TORCH_CHECK(mask.size(0) == mData.size(0), "mask must have the same size as the first dimension of the JaggedTensor");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be of type bool");

    // TODO: This requires an extra index select and is currently slow
    if (mBatchIdx.size(0) > 0) {
        TORCH_CHECK(mask.size(0) == mBatchIdx.size(0), "Bad jidx. This should never happen");
        return JaggedTensor::from_data_and_jidx(mData.index({mask, "..."}), mBatchIdx.index({mask}), batch_size());
    } else {
        return JaggedTensor::from_data_and_jidx(mData.index({mask, "..."}), mBatchIdx, batch_size());
    }
}

// TODO: Jagged sizes
int64_t JaggedTensor::size(int64_t dim) const {
    return mData.size(dim);
}

JaggedTensor JaggedTensor::index(const at::indexing::TensorIndex& idx) const {
    if (idx.is_integer()) {
        auto idxVal = idx.integer();
        if (idxVal < 0) {
            idxVal += mData.size(0);
        }
        TORCH_CHECK_INDEX(idxVal >= 0 && idxVal < mOffsets.size(0),
                          "Index ", idx.integer(), " is out of bounds for JaggedTensor with ",
                          mOffsets.size(0), " elements");

        int64_t startIdx = mOffsets[idxVal][0].item<int64_t>();
        int64_t endIdx = mOffsets[idxVal][1].item<int64_t>();
        return JaggedTensor::from_data_offsets_and_jidx_unsafe(
            mData.index({torch::indexing::Slice(startIdx, endIdx)}),
            torch::tensor({int64_t(0), endIdx - startIdx}, torch::TensorOptions().dtype(torch::kInt64).device(mData.device())).unsqueeze(0),
            torch::zeros({endIdx - startIdx}, torch::TensorOptions().dtype(torch::kInt16).device(mData.device())));

    } else if (idx.is_slice()) {
        int64_t startIdx = idx.slice().start().as_int_unchecked();
        int64_t endIdx = idx.slice().stop().as_int_unchecked();
        int64_t step = idx.slice().step().as_int_unchecked();

        // Deal with symbolic int
        if (startIdx >= at::indexing::INDEX_MAX) {
            startIdx = mOffsets.size(0);
        }
        if (endIdx <= at::indexing::INDEX_MIN) {
            endIdx = 0;
        }

        if (startIdx < 0) {
            startIdx += mOffsets.size(0);
        }
        if (endIdx < 0) {
            endIdx += mOffsets.size(0);
        }

        if (startIdx > endIdx) {
            startIdx = endIdx;
        }

        startIdx = std::max(startIdx, (int64_t) 0);
        endIdx = std::min(endIdx, mOffsets.size(0));

        TORCH_CHECK_INDEX(step == 1, "step must be 1 for JaggedTensor. Only contiguous slicing is supported.");

        int64_t startOffset = mOffsets[startIdx][0].item<int64_t>();
        int64_t endOffset = mOffsets[endIdx - 1][1].item<int64_t>();
        return JaggedTensor::from_data_and_offsets(
            mData.index({torch::indexing::Slice(startOffset, endOffset)}),
            mOffsets.index({torch::indexing::Slice(startIdx, endIdx)}) - startIdx);
    } else if (idx.is_ellipsis()) {
        return *this;
    } else {
        TORCH_CHECK_INDEX(false, "Unsupported indexing operation");
    }
}

torch::Tensor JaggedTensor::jagged_argsort() {
    computeJidxFromJOffsets();
    return FVDB_DISPATCH_KERNEL_DEVICE(device(), [&]() {
        return detail::ops::dispatchJaggedArgsort<DeviceTag>(*this);
    });
}

torch::Tensor JaggedTensor::jagged_sum() const {
    if (mBatchIdx.size(0) == 0) {
        return mData.sum(0).unsqueeze(0);
    }
    return detail::autograd::JaggedSum::apply(jdata(), jidx(), batch_size())[0];
}

std::vector<torch::Tensor> JaggedTensor::jagged_min() const {
    if (mBatchIdx.size(0) == 0) {
        auto minTuple = mData.min(0);
        return {std::get<0>(minTuple).unsqueeze(0), std::get<1>(minTuple).unsqueeze(0)};
    }
    return detail::autograd::JaggedMin::apply(jdata(), jidx(), batch_size());
}

std::vector<torch::Tensor> JaggedTensor::jagged_max() const {
    if (mBatchIdx.size(0) == 0) {
        auto maxTuple = mData.max(0);
        return {std::get<0>(maxTuple).unsqueeze(0), std::get<1>(maxTuple).unsqueeze(0)};
    }
    return detail::autograd::JaggedMax::apply(jdata(), jidx(), batch_size());
}

JaggedTensor JaggedTensor::concatenate(const std::vector<JaggedTensor>& vec, int dim) {
    TORCH_CHECK(vec.size() > 0, "empty tensor list");

    // TODO: Extend this when multiple jagged dimension is implemented.
    int64_t v0BatchSize = vec[0].batch_size();
    int64_t v0JDim = vec[0].mData.dim();
    TORCH_CHECK(dim >= -(v0JDim + 1) && dim <= v0JDim, "dim must be between ", -(v0JDim + 1), " and ", v0JDim);

    if (dim < 0) {
        dim += v0JDim + 1;
    }

    if (dim == 0) {
        // Concat along the batch dimension
        std::vector<torch::Tensor> data;
        std::vector<torch::Tensor> offsets;
        int64_t curOffset = 0;

        for (const auto& jvec : vec) {
            data.push_back(jvec.mData);
            offsets.push_back(jvec.mOffsets + curOffset);
            curOffset += jvec.mOffsets[-1][1].item<int64_t>();
        }
        return JaggedTensor::from_data_and_offsets(torch::cat(data, 0), torch::cat(offsets, 0));

    } else if (dim == 1) {
        // Concat along the jagged dimension
        for (const auto& jvec : vec) {
            TORCH_CHECK_VALUE(jvec.batch_size() == v0BatchSize, "All tensors must have the same batch size");
        }

        std::vector<torch::Tensor> tensors(v0BatchSize);
        for (int b = 0; b < v0BatchSize; b += 1) {
            std::vector<torch::Tensor> batchElement;
            for (const auto& jvec : vec) {
                batchElement.push_back(jvec.index({b}).mData);
            }
            tensors[b] = torch::cat(batchElement);
        }
        return JaggedTensor(tensors);
    } else {
        // Concat along the data dimension
        std::vector<torch::Tensor> data;
        for (const auto& jvec : vec) {
            data.push_back(jvec.mData);
        }
        return JaggedTensor::from_data_offsets_and_jidx_unsafe(torch::cat(data, dim - 1), vec[0].mOffsets, vec[0].mBatchIdx);
    }

}


} // namespace fvdb
