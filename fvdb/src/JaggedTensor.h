#pragma once

#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/all.h>

#include "detail/utils/Utils.h"

namespace fvdb {


template <typename ScalarT, size_t NDims>
class JaggedAccessor {
    torch::TensorAccessor<int16_t, 1> mBatchIdx;
    torch::TensorAccessor<int64_t, 2> mOffsets;
    torch::TensorAccessor<ScalarT, NDims> mData;

    friend class JaggedTensor;

    JaggedAccessor(torch::TensorAccessor<int16_t, 1> batchIdx,
                   torch::TensorAccessor<int64_t, 2> offsets,
                   torch::TensorAccessor<ScalarT, NDims> data)
        : mBatchIdx(batchIdx), mOffsets(offsets), mData(data) {}
public:

    template <typename T, size_t N>
    using TensorAccessorType = torch::TensorAccessor<T, N>;

    inline __hostdev__ int64_t elementCount() const {
        return mData.size(0);
    }

    inline __hostdev__ int16_t batchIdx(int32_t idx) const {
        return mBatchIdx.size(0) > 0 ? mBatchIdx[idx] : 0;
    }

    inline __hostdev__ int64_t offsetStart(int32_t idx) const {
        return mOffsets[idx][0];
    }

    inline __hostdev__ int64_t offsetEnd(int32_t idx) const {
        return mOffsets[idx][1];
    }

    inline __hostdev__ const torch::TensorAccessor<ScalarT, NDims>& data() const {
        return mData;
    }
};


template <typename ScalarT, size_t NDims, template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
class PackedJaggedAccessor32 {
    torch::PackedTensorAccessor32<int16_t, 1, PtrTraits> mBatchIdx;
    torch::PackedTensorAccessor32<int64_t, 2, PtrTraits> mOffsets;
    torch::PackedTensorAccessor32<ScalarT, NDims, PtrTraits> mData;

    friend class JaggedTensor;

    PackedJaggedAccessor32(torch::PackedTensorAccessor32<int16_t, 1, PtrTraits> batchIdx,
                           torch::PackedTensorAccessor32<int64_t, 2, PtrTraits> offsets,
                           torch::PackedTensorAccessor32<ScalarT, NDims, PtrTraits> data)
        : mBatchIdx(batchIdx), mOffsets(offsets), mData(data) {}

public:

    template <typename T, size_t N>
    using TensorAccessorType = torch::TensorAccessor<T, N, PtrTraits>;

    inline __hostdev__ int64_t elementCount() const {
        return mData.size(0);
    }

    inline __hostdev__ int16_t batchIdx(int32_t idx) const {
        return mBatchIdx.size(0) > 0 ? mBatchIdx[idx] : 0;
    }

    inline __hostdev__ int64_t offsetStart(int32_t idx) const {
        return mOffsets[idx][0];
    }

    inline __hostdev__ int64_t offsetEnd(int32_t idx) const {
        return mOffsets[idx][1];
    }

    inline __hostdev__ const torch::PackedTensorAccessor32<ScalarT, NDims, PtrTraits>& data() const {
        return mData;
    }
};


class JaggedTensor : public torch::CustomClassHolder {
    torch::Tensor mData;
    torch::Tensor mBatchIdx;
    torch::Tensor mOffsets;

    void computeJOffsetsFromJIdx(const int64_t batchSize);
    void computeJidxFromJOffsets();

public:

    static JaggedTensor from_data_and_jidx(torch::Tensor data, torch::Tensor jidx, int64_t batch_size);
    static JaggedTensor from_data_and_offsets(torch::Tensor data, torch::Tensor offsets);
    static JaggedTensor from_data_offsets_and_jidx_unsafe(torch::Tensor data, torch::Tensor offsets, torch::Tensor jidx);

    JaggedTensor jagged_like(torch::Tensor data) const;

    JaggedTensor() {
      mData = torch::Tensor();
      mBatchIdx = torch::Tensor();
    }

    JaggedTensor(torch::Tensor data);

    JaggedTensor(const std::vector<torch::Tensor>& tensors);

    void set_data(const torch::Tensor& data);

    const torch::Tensor& jdata() const { return mData; }
    const torch::Tensor& jidx() const { return mBatchIdx; }
    const torch::Tensor& joffsets() const { return mOffsets; }

    template <typename Scalar, size_t NDims>
    JaggedAccessor<Scalar, NDims> accessor() const {
        return JaggedAccessor<Scalar, NDims>(
            mBatchIdx.accessor<int16_t, 1>(),
            mOffsets.accessor<int64_t, 2>(),
            mData.accessor<Scalar, NDims>());
    }

    template <typename Scalar, size_t NDims, template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
    PackedJaggedAccessor32<Scalar, NDims, PtrTraits> packed_accessor32() const {
        return PackedJaggedAccessor32<Scalar, NDims, PtrTraits>(
            mBatchIdx.packed_accessor32<int16_t, 1, PtrTraits>(),
            mOffsets.packed_accessor32<int64_t, 2, PtrTraits>(),
            mData.packed_accessor32<Scalar, NDims, PtrTraits>());
    }

    JaggedTensor index(const at::indexing::TensorIndex& idx) const;

    // TODO: Implement jagged sizes
    int64_t size(int64_t dim) const;

    int64_t batch_size() const { return mOffsets.size(0); }

    int64_t jagged_dim() const {
        if (mBatchIdx.size(0) == 0) {
            return 0;
        } else {
            return 1;
        }
    }

    // TODO: This will eventually always be true but for now keep it around for
    //       sanity checking
    inline void check_valid() const {
        TORCH_CHECK((jidx().size(0) == 0 && joffsets().size(0) == 1) || (jidx().size(0) == jdata().size(0)), "tensor must be a valid jagged tensor");
        TORCH_CHECK(jidx().device() == jdata().device(), "batch index and data must be on the same device");
        TORCH_CHECK(jidx().dtype() == torch::kInt16, "batch index must be int16");
        TORCH_CHECK(joffsets().device() == jdata().device(), "offsets and data must be on the same device");
    }

    inline int64_t element_count() const {
        return jdata().size(0);
    }

    inline torch::Device device() const {
        return mData.device();
    }

    caffe2::TypeMeta dtype() const {
        return mData.dtype();
    }

    torch::Layout layout() const {
        return mData.layout();
    }

    inline torch::ScalarType scalar_type() const {
        return mData.scalar_type();
    }

    inline bool is_cuda() const {
        return mData.is_cuda();
    }

    inline bool is_cpu() const {
        return mData.is_cpu();
    }

    inline int64_t dim() const {
        return mData.dim();
    }

    int64_t get_device() const {
        return mData.get_device();
    }

    bool is_complex() const {
        return at::isComplexType(this->scalar_type());
    }

    bool is_floating_point() const {
        return at::isFloatingType(this->scalar_type());
    }

    bool is_signed() const {
        return at::isSignedType(this->scalar_type());
    }

    int64_t numel() const {
        return mData.numel();
    }

    JaggedTensor r_masked_select(const torch::Tensor& mask) const;

    inline bool is_contiguous() const {
        return mData.is_contiguous();
    }

    inline JaggedTensor contiguous() const {
        return JaggedTensor::from_data_offsets_and_jidx_unsafe(mData.contiguous(), mOffsets.contiguous(), mBatchIdx.contiguous());
    }

    inline JaggedTensor to(at::TensorOptions options={}, bool non_blocking=false, bool copy=false, c10::optional<at::MemoryFormat> memory_format=c10::nullopt) const {
        JaggedTensor ret = *this;
        ret.mData = ret.mData.to(options, non_blocking, copy, memory_format);
        ret.mBatchIdx = ret.mBatchIdx.to(ret.mData.device(), non_blocking, copy, memory_format);
        ret.mOffsets = ret.mOffsets.to(ret.mData.device(), non_blocking, copy, memory_format);
        return ret;
    }

    inline JaggedTensor to(c10::optional<torch::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
        JaggedTensor ret = *this;
        ret.mData = ret.mData.to(dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
        ret.mBatchIdx = ret.mBatchIdx.to(torch::kInt16, layout, device, pin_memory, non_blocking, copy, memory_format);
        ret.mOffsets = ret.mOffsets.to(torch::kInt64, layout, device, pin_memory, non_blocking, copy, memory_format);
        return ret;
    }

    inline JaggedTensor to(torch::Device device, torch::ScalarType dtype, bool non_blocking=false, bool copy=false, c10::optional<at::MemoryFormat> memory_format=c10::nullopt) {
        JaggedTensor ret = *this;
        ret.mData = ret.mData.to(device, dtype, non_blocking, copy, memory_format);
        ret.mBatchIdx = ret.mBatchIdx.to(device, non_blocking, copy, memory_format);
        ret.mOffsets = ret.mOffsets.to(device, non_blocking, copy, memory_format);
        return ret;
    }

    inline JaggedTensor to(torch::ScalarType dtype, bool non_blocking=false, bool copy=false, c10::optional<at::MemoryFormat> memory_format=c10::nullopt) {
        JaggedTensor ret = *this;
        ret.mData = ret.mData.to(dtype, non_blocking, copy, memory_format);
        ret.mBatchIdx = ret.mBatchIdx.to(torch::kInt16, non_blocking, copy, memory_format);
        ret.mOffsets = ret.mOffsets.to(torch::kInt64, non_blocking, copy, memory_format);
        return ret;
    }

    torch::TensorOptions options() const {
        return torch::TensorOptions().dtype(dtype()).device(device()).layout(layout());
    }

    JaggedTensor cuda() const {
        return to(this->options().device(torch::kCUDA), /*non_blocking*/ false, /*copy*/ false);
    }

    JaggedTensor cpu() const {
        return to(this->options().device(torch::kCPU), /*non_blocking*/ false, /*copy*/ false);
    }

    JaggedTensor operator+(const JaggedTensor& other) const {
        TORCH_CHECK(this->device() == other.device(), "device should match between this tensor and other tensor");
        TORCH_CHECK(mData.sizes() == other.jdata().sizes(), "data shape should match between this tensor and other tensor");
        TORCH_CHECK(mBatchIdx.sizes() == other.jidx().sizes(), "batch indices' shape should match between this tensor and other tensor");
        TORCH_CHECK(mOffsets.sizes() == other.joffsets().sizes(), "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK(torch::equal(mOffsets, other.joffsets()), "offsets shape should match between this tensor and other tensor");

        return jagged_like(mData + other.mData);
    }

    JaggedTensor operator+(const int other) const {
        return jagged_like(mData + other);
    }

    JaggedTensor operator+(const float other) const {
        return jagged_like(mData + other);
    }

    JaggedTensor operator+(const torch::Tensor& other) const {
        return jagged_like(mData + other);
    }

    JaggedTensor operator-(const JaggedTensor& other) const {
        TORCH_CHECK(this->device() == other.device(), "device should match between this tensor and other tensor");
        TORCH_CHECK(mData.sizes() == other.jdata().sizes(), "data shape should match between this tensor and other tensor");
        TORCH_CHECK(mBatchIdx.sizes() == other.jidx().sizes(), "batch indices' shape should match between this tensor and other tensor");
        TORCH_CHECK(mOffsets.sizes() == other.joffsets().sizes(), "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK(torch::equal(mOffsets, other.joffsets()), "offsets shape should match between this tensor and other tensor");

        return jagged_like(mData - other.mData);
    }

    JaggedTensor operator-(const int other) const {
        return jagged_like(mData - other);
    }

    JaggedTensor operator-(const float other) const {
        return jagged_like(mData - other);
    }

    JaggedTensor operator-(const torch::Tensor& other) const {
        return jagged_like(mData - other);
    }

    JaggedTensor operator-() const {
        return jagged_like(-mData);
    }

    JaggedTensor operator*(const JaggedTensor& other) const {
        TORCH_CHECK(this->device() == other.device(), "device should match between this tensor and other tensor");
        TORCH_CHECK(mData.sizes() == other.jdata().sizes(), "data shape should match between this tensor and other tensor");
        TORCH_CHECK(mBatchIdx.sizes() == other.jidx().sizes(), "batch indices' shape should match between this tensor and other tensor");
        TORCH_CHECK(mOffsets.sizes() == other.joffsets().sizes(), "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK(torch::equal(mOffsets, other.joffsets()), "offsets shape should match between this tensor and other tensor");

        return jagged_like(mData * other.mData);
    }

    JaggedTensor operator*(const int other) const {
        return jagged_like(mData * other);
    }

    JaggedTensor operator*(const float other) const {
        return jagged_like(mData * other);
    }

    JaggedTensor operator*(const torch::Tensor& other) const {
        return jagged_like(mData * other);
    }

    JaggedTensor operator/(const JaggedTensor& other) const {
        TORCH_CHECK(this->device() == other.device(), "device should match between this tensor and other tensor");
        TORCH_CHECK(mData.sizes() == other.jdata().sizes(), "data shape should match between this tensor and other tensor");
        TORCH_CHECK(mBatchIdx.sizes() == other.jidx().sizes(), "batch indices' shape should match between this tensor and other tensor");
        TORCH_CHECK(mOffsets.sizes() == other.joffsets().sizes(), "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK(torch::equal(mOffsets, other.joffsets()), "offsets shape should match between this tensor and other tensor");

        return jagged_like(mData / other.mData);
    }

    JaggedTensor operator/(const int other) const {
        return jagged_like(mData / other);
    }

    JaggedTensor operator/(const float other) const {
        return jagged_like(mData / other);
    }

    JaggedTensor operator/(const torch::Tensor& other) const {
        return jagged_like(mData / other);
    }

    JaggedTensor floordiv(const JaggedTensor& other) const {
        TORCH_CHECK(this->device() == other.device(), "device should match between this tensor and other tensor");
        TORCH_CHECK(mData.sizes() == other.jdata().sizes(), "data shape should match between this tensor and other tensor");
        TORCH_CHECK(mBatchIdx.sizes() == other.jidx().sizes(), "batch indices' shape should match between this tensor and other tensor");
        TORCH_CHECK(mOffsets.sizes() == other.joffsets().sizes(), "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK(torch::equal(mOffsets, other.joffsets()), "offsets shape should match between this tensor and other tensor");

        return jagged_like(torch::floor_divide(mData, other.mData));
    }

    JaggedTensor floordiv(const int other) const {
        return jagged_like(torch::floor_divide(mData, other));
    }

    JaggedTensor floordiv(const float other) const {
        return jagged_like(torch::floor_divide(mData, other));
    }

    JaggedTensor floordiv(const torch::Tensor& other) const {
        return jagged_like(torch::floor_divide(mData, other));
    }

    JaggedTensor operator%(const JaggedTensor& other) const {
        TORCH_CHECK(this->device() == other.device(), "device should match between this tensor and other tensor");
        TORCH_CHECK(mData.sizes() == other.jdata().sizes(), "data shape should match between this tensor and other tensor");
        TORCH_CHECK(mBatchIdx.sizes() == other.jidx().sizes(), "batch indices' shape should match between this tensor and other tensor");
        TORCH_CHECK(mOffsets.sizes() == other.joffsets().sizes(), "offsets shape should match between this tensor and other tensor");
        TORCH_CHECK(torch::equal(mOffsets, other.joffsets()), "offsets shape should match between this tensor and other tensor");

        return jagged_like(mData % other.mData);
    }

    JaggedTensor operator%(const int other) const {
        return jagged_like(mData % other);
    }

    JaggedTensor operator%(const float other) const {
        return jagged_like(mData % other);
    }

    JaggedTensor operator%(const torch::Tensor& other) const {
        return jagged_like(mData % other);
    }

    JaggedTensor round(int decimals = 0) const {
        return jagged_like(torch::round(mData, decimals));
    }

    const JaggedTensor& set_requires_grad(bool requires_grad) const {
        mData.set_requires_grad(requires_grad);
        return *this;
    }

    bool requires_grad() const {
        return mData.requires_grad();
    }

    JaggedTensor detach() const {
        return jagged_like(mData.detach());
    }

    JaggedTensor clone() const {
        return jagged_like(mData.clone());
    }

    static JaggedTensor concatenate(const std::vector<JaggedTensor>& vec, int dim = 0);

    /// @brief Sorts each batch element in ascending order, note that jdata has to be 1-dimensional
    /// @return An indexing tensor with the same size as jdata, that permutes the elements of data to be in sorted order
    torch::Tensor jagged_argsort();

    /// @brief Compute the summation of each batch element
    /// @return A tensor of size (batch_size, *) containing the sum of each batch element, feature dimensions are preserved
    torch::Tensor jagged_sum() const;

    /// @brief Compute the minimum of each batch element
    /// @return Minimum value of size (batch_size, *) and argmin of size (batch_size, *)
    std::vector<torch::Tensor> jagged_min() const;

    /// @brief Compute the maximum of each batch element
    /// @return Maximum value of size (batch_size, *) and argmax of size (batch_size, *)
    std::vector<torch::Tensor> jagged_max() const;

};



} // namespace fvdb