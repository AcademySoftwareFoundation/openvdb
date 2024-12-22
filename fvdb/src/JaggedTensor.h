// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_JAGGEDTENSOR_H
#define FVDB_JAGGEDTENSOR_H

#include <detail/utils/Utils.h>

#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/extension.h>

namespace fvdb {

struct JaggedTensorIndex;

using JIdxType     = int32_t;
using JOffsetsType = int64_t;
using JLIdxType    = int32_t;

constexpr c10::ScalarType JIdxScalarType     = c10::CppTypeToScalarType<JIdxType>::value;
constexpr c10::ScalarType JOffsetsScalarType = c10::CppTypeToScalarType<JOffsetsType>::value;
constexpr c10::ScalarType JLIdxScalarType    = c10::CppTypeToScalarType<JLIdxType>::value;

template <typename ScalarT, size_t NDims> class JaggedAccessor {
    torch::TensorAccessor<JIdxType, 1>     mBatchIdx;
    torch::TensorAccessor<JOffsetsType, 1> mOffsets;
    torch::TensorAccessor<JLIdxType, 2>    mListIndexes;
    torch::TensorAccessor<ScalarT, NDims>  mData;

    friend class JaggedTensor;

    JaggedAccessor(torch::TensorAccessor<JIdxType, 1>     batchIdx,
                   torch::TensorAccessor<JOffsetsType, 1> offsets,
                   torch::TensorAccessor<JLIdxType, 2>    listIndexes,
                   torch::TensorAccessor<ScalarT, NDims>  data)
        : mBatchIdx(batchIdx), mOffsets(offsets), mListIndexes(listIndexes), mData(data) {}

  public:
    template <typename T, size_t N> using TensorAccessorType = torch::TensorAccessor<T, N>;

    inline __hostdev__ int64_t
    elementCount() const {
        return mData.size(0);
    }

    inline __hostdev__ JIdxType
    batchIdx(int64_t idx) const {
        return mBatchIdx.size(0) > 0 ? mBatchIdx[idx] : 0;
    }

    inline __hostdev__ JOffsetsType
    offsetStart(int64_t idx) const {
        return mOffsets[idx];
    }

    inline __hostdev__ JOffsetsType
    offsetEnd(int64_t idx) const {
        return mOffsets[idx + 1];
    }

    inline __hostdev__ const torch::TensorAccessor<ScalarT, NDims>                          &
    data() const {
        return mData;
    }
};

template <typename ScalarT, size_t NDims,
          template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
class PackedJaggedAccessor32 {
    torch::PackedTensorAccessor32<JIdxType, 1, PtrTraits>     mBatchIdx;
    torch::PackedTensorAccessor32<JOffsetsType, 1, PtrTraits> mOffsets;
    torch::PackedTensorAccessor32<JLIdxType, 2, PtrTraits>    mListIndexes;
    torch::PackedTensorAccessor32<ScalarT, NDims, PtrTraits>  mData;

    friend class JaggedTensor;

    PackedJaggedAccessor32(torch::PackedTensorAccessor32<JIdxType, 1, PtrTraits>     batchIdx,
                           torch::PackedTensorAccessor32<JOffsetsType, 1, PtrTraits> offsets,
                           torch::PackedTensorAccessor32<JLIdxType, 2, PtrTraits>    listIndexes,
                           torch::PackedTensorAccessor32<ScalarT, NDims, PtrTraits>  data)
        : mBatchIdx(batchIdx), mOffsets(offsets), mListIndexes(listIndexes), mData(data) {}

  public:
    template <typename T, size_t N>
    using TensorAccessorType = torch::TensorAccessor<T, N, PtrTraits>;

    inline __hostdev__ int64_t
    elementCount() const {
        return mData.size(0);
    }

    inline __hostdev__ JIdxType
    batchIdx(int64_t idx) const {
        return mBatchIdx.size(0) > 0 ? mBatchIdx[idx] : 0;
    }

    inline __hostdev__ JOffsetsType
    offsetStart(int64_t idx) const {
        return mOffsets[idx];
    }

    inline __hostdev__ JOffsetsType
    offsetEnd(int64_t idx) const {
        return mOffsets[idx + 1];
    }

    inline __hostdev__ const torch::PackedTensorAccessor32<ScalarT, NDims, PtrTraits>                          &
    data() const {
        return mData;
    }
};

class JaggedTensor : public torch::CustomClassHolder {
    torch::Tensor mData;          // Actual data indexed by a jagged tensor
    torch::Tensor mBatchIdx;      // Which (linear) batch is each datum in
    torch::Tensor mOffsets;       // Offset of each tensor in the list of lists
    torch::Tensor mListIdx;       // LoL indexing of tensor with shape [num_tensors, ldim]
    int64_t       mNumOuterLists; // Number of outer lists in this JaggedTensor

    // Store the number of elements in each tensor in the jagged tensor
    // Computing this requires a GPU -> CPU copy so we cache it
    struct {
        std::vector<int64_t>                           mLShape1;
        std::vector<std::vector<int64_t>>              mLShape2;
        std::vector<std::vector<std::vector<int64_t>>> mLShape3;
        bool                                           mDirty = true;
        void
        markDirty() {
            mDirty = true;
        }
        void
        clear() {
            mLShape1.clear();
            mLShape2.clear();
            mLShape3.clear();
            mDirty = true;
        }
    } mLShapeCache;

    void recompute_lsizes_if_dirty();

    void binary_op_check(const JaggedTensor &other) const;

  public:
    static torch::Tensor joffsets_from_jidx_and_jdata(torch::Tensor jidx, torch::Tensor jdata,
                                                      int64_t num_tensors);
    static torch::Tensor jidx_from_joffsets(torch::Tensor joffsets, int64_t num_elements);
    static JaggedTensor  from_jdata_joffsets_jidx_and_lidx_unsafe(torch::Tensor jdata,
                                                                  torch::Tensor joffsets,
                                                                  torch::Tensor jidx,
                                                                  torch::Tensor jlidx,
                                                                  int64_t       numOuterLists);

    static JaggedTensor from_data_indices_and_list_ids(torch::Tensor data, torch::Tensor indices,
                                                       torch::Tensor list_ids, int64_t num_tensors);
    static JaggedTensor from_data_offsets_and_list_ids(torch::Tensor data, torch::Tensor offsets,
                                                       torch::Tensor list_ids);

    /// @brief Concatenate the list of JaggedTensors along a given dimension.
    ///        There are two modes for this function.
    ///        1. If dim is an integer:
    ///            e.g. if [jt_a, jt_b] are two JaggedTensors of the form
    ///            jt_a = [[a_11, a_12], [a_21], [a_31, a_32]] and jt_b = [[b_11, b_12], [b_21],
    ///            [b_31, b_32]], then JaggedTensor::jcat({jt_a, jt_b}) will return a JaggedTensor
    ///            of the form
    ///            [[torch.cat([a_11, b_11], dim=dim), torch.cat([a_12, b_12], dim=dim)],
    ///             [torch.cat([a_21, b_21], dim=dim)],
    ///             [torch.cat([a_31, b_31], dim=dim), torch.cat([a_32, b_32], dim=dim)]]
    ///        2. If dim is c10::nullopt:
    ///            e.g. if [jt_a, jt_b] are two JaggedTensors of the form
    ///            jt_a = [[a_11, a_12], [a_21], [a_31, a_32]] and jt_b = [[b_11], [b_21, b_22]],
    ///            then JaggedTensor::jcat({jt_a, jt_b}) will return a JaggedTensor of the form
    ///            [[a_11, a_12], [a_21], [a_31, a_32], [b_11], [b_21, b_22]]
    /// @param vec A vector of JaggedTensors to concatenate
    /// @param dim The dimension along which to concatenate each JaggedTensor or c10::nullopt to
    /// concatenate
    ///            the JaggedTensors as lists
    /// @return A JaggedTensor containing the concatenated data
    static JaggedTensor jcat(const std::vector<JaggedTensor> &vec, c10::optional<int64_t> dim);

    /// @brief Create an empty JaggedTensor
    JaggedTensor() {
        mData          = torch::Tensor();
        mBatchIdx      = torch::empty({ 0 }, torch::TensorOptions().dtype(JIdxScalarType));
        mOffsets       = torch::zeros({ 1 }, torch::TensorOptions().dtype(JOffsetsScalarType));
        mListIdx       = torch::empty({ 0, 1 }, torch::TensorOptions().dtype(JLIdxScalarType));
        mNumOuterLists = 0;
    }

    /// @brief Create a JaggedTensor representing a list with a single tensor. Note this function
    /// does not copy the
    ///        data tensor, it only creates a view of it.
    /// @param data The data tensor
    JaggedTensor(torch::Tensor data);

    /// @brief Create a JaggedTensor representing a list of tensors.
    /// @param tensors A list of tensors
    JaggedTensor(const std::vector<torch::Tensor> &tensors);

    /// @brief Create a JaggedTensor representing a list of lists of tensors.
    /// @param tensors A list of lists of tensors
    JaggedTensor(const std::vector<std::vector<torch::Tensor>> &tensors);

    /// @brief Create a JaggedTensor representing a list of tensors where the number of elements in
    /// each tensor is given
    ///        by the lsizes vector. i.e. if lsizes = [2, 1, 2], then the first tensor will have 2
    ///        elements, the second tensor will have 1 element, and the third tensor will have 2
    ///        elements. The raw data tensor must then have a number of elements equal to the sum of
    ///        the elements in lsizes (i.e. shape [sum(lsizes), ...])
    /// @param lsizes A vector of integers indicating the number of elements in each tensor
    /// @param data The raw data tensor
    JaggedTensor(const std::vector<int64_t> &lsizes, const torch::Tensor data);

    /// @brief Create a JaggedTensor representing a list of lists of tensors where the number of
    /// elements in each tensor
    ///       is given by the lsizes vector. i.e. if lsizes = [[2, 1], [5, 6, 7]], then the first
    ///       list will have 2 tensors with 1 and 2 elements respectively and the second list will
    ///       have 3 tensors with 5, 6, and 7 elements respectively. The raw data tensor must then
    ///       have a number of elements equal to the sum of the elements in lsizes (i.e. shape
    ///       [sum(lsizes), ...])
    /// @param lsizes A vector of vectors of integers indicating the number of elements in each
    /// tensor
    /// @param total_tensors The total number of tensors in the list of lists
    /// @param data The raw data tensor
    JaggedTensor(const std::vector<std::vector<int64_t>> &lsizes, const int64_t total_tensors,
                 const torch::Tensor data);

    /// @brief Create a JaggedTensor with the same list structure as this one but with the given raw
    /// data.
    ///        The returned JaggedTensor will share the same memory for indices/list ids/offsets as
    ///        this one those are modified.
    /// @param data A tensor with the same number of elements as the original data
    /// @return A JaggedTensor with the same list structure as this one but with the given data
    JaggedTensor jagged_like(torch::Tensor data) const;

    /// @brief Set the raw data of this JaggedTensor to the given tensor
    /// @param data A data tensor with the same number of elements as the original data
    void set_data(const torch::Tensor &data);

    /// @brief  Get the raw data indexed by this JaggedTensor
    /// @return The raw data tensor
    const torch::Tensor &
    jdata() const {
        return mData;
    }

    /// @brief Get the indices of this jagged tensor. i.e. a tensor of size (num_elements,)
    /// indicating which
    ///        tensor each element belongs to
    /// @return The indices of this JaggedTensor
    const torch::Tensor &
    jidx() const {
        return mBatchIdx;
    }

    /// @brief Get the offsets of each tensor indexed by this JaggedTensor. i.e. a tensor of size
    /// (num_tensors + 1)
    ///        where joffsets[i] is the start offset in jdata and joffsets[i+1] is the end offset in
    ///        jdata
    /// @return The offsets of each tensor indexed by this JaggedTensor
    const torch::Tensor &
    joffsets() const {
        return mOffsets;
    }

    /// @brief Get the list indices of each tensor indexed by this JaggedTensor. i.e. a tensor of
    /// size (num_tensors, ldim)
    ///        where e.g. jlidx[i][j] is the index of the j-th list in the i-th tensor (for a list
    ///        of lists JaggedTensor)
    /// @return The list indices of each tensor indexed by this JaggedTensor
    const torch::Tensor &
    jlidx() const {
        return mListIdx;
    }

    /// @brief Get the number of outer lists in this JaggedTensor
    int64_t
    num_outer_lists() const {
        return mNumOuterLists;
    }

    /// @brief Get the number of tensors in this JaggedTensor
    int64_t
    num_tensors() const {
        return mOffsets.size(0) - 1;
    }

    /// @brief Get the number of elements in each tensor indexed by this JaggedTensor. Assumes the
    /// JaggedTensor has ldim() == 1
    ///        i.e. it represents a list of tensors
    /// @return The number of elements in each tensor indexed by this JaggedTensor
    std::vector<int64_t> lsizes1() const;

    /// @brief Get the number of elements in each tensor indexed by this JaggedTensor. Assumes
    /// JaggedTensor has ldim() == 2
    ///        i.e. it represents a list of lists of tensors
    /// @return The number of elements in each tensor indexed by this JaggedTensor such that
    /// lsizes2()[i][j] is the number of elements
    ///         in the j-th tensor in i-th list
    std::vector<std::vector<int64_t>> lsizes2() const;

    /// @brief Get the number of nested lists encoded by this JaggedTensor. An ldim of one means
    /// this JaggedTensor encodes a list
    //         of tensors, an ldim of 2 means this JaggedTensor encodes a list of lists of tensors,
    //         etc.
    /// @return The number of nested lists encoded by this JaggedTensor
    int64_t ldim() const;

    /// @brief Get the size of each element indexed by this JaggedTensor. i.e. if the JaggedTensor
    /// represents a list of tensors
    ///        where each tensor has shape [N, A, B, C], then esizes() will return [A, B, C]
    /// @return The size of each element indexed by this JaggedTensor
    std::vector<int64_t> esizes() const;

    /// @brief Get the number of dimensions of each element indexed by this JaggedTensor. i.e. if
    /// the JaggedTensor represents a list of tensors
    ///        where each tensor has shape [N, A, B, C], then edim() will return 3
    /// @return The number of dimensions of each element indexed by this JaggedTensor
    int64_t edim() const;

    /// @brief Convert the JaggedTensor to a list of tensors assuming this JaggedTensor represents a
    /// list of tensors.
    ///        Note this function doesn't work for nested lists of tensors (instead use unbind2())
    /// @return A list of tensors where each tensor is indexed by this JaggedTensor.
    std::vector<torch::Tensor> unbind1() const;

    /// @brief Convert the JaggedTensor to a list of lists of tensors assuming this JaggedTensor
    /// represents a list of lists of tensors.
    ///        Note this function doesn't work for a flat list of tensors (instead use unbind1())
    /// @return A list of lists of tensors where each tensor is indexed by this JaggedTensor.
    std::vector<std::vector<torch::Tensor>> unbind2() const;

    /// @brief Index this JaggedTensor along the outer list dimension. There are several ways to
    /// index a JaggedTensor jt:
    ///       1. Indexing with an integer jt[i] will return the i^th list in this tensor (or a list
    ///       containing the i^th
    ///          tensor if jt represents a list of tensors)
    ///       2. Indexing with a slice jt[2:5] will return a JaggedTensor containing the 2nd, 3rd,
    ///       and 4th lists in this tensor
    ///          Note: We currently only support contiguous slices (i.e. stride = 1)
    ///       3. Indexing with another JaggedTensor of boolean mask values jt[mask]
    ///          will return a JaggedTensor containing tensors masked by the boolean mask
    ///          i.e. jt[mask][i][j].jdata = jt[i][j].jdata[mask[i][j].jdata]
    ///       4. Indexing with a tensor of integer indices jt[indices] will return a JaggedTensor
    ///       containing tensors
    ///          indexed by the integer indices. i.e. jt[indices][i][j].jdata =
    ///          jt[i][j].jdata[indices[i][j]]
    ///       5. Indexing with ellipses jt[...] is a no-op
    /// @param idx The index to use to index this JaggedTensor
    /// @return A JaggedTensor containing the indexed data
    JaggedTensor index(JaggedTensorIndex idx) const;

    /// @brief Reshape this JaggedTensor to have a new list structure. The provided lshape should be
    /// compatible with
    ///        this tensor. i.e. the sum of the elements in lshape should be equal to the number of
    ///        elements in this JaggedTensor.
    ///        Note this function creates a view over the original JaggedTensor so modifying the
    ///        returned JaggedTensor will modify the original tensor.
    /// @param lsizes The new list structure
    /// @return A JaggedTensor with the new list structure
    JaggedTensor jreshape(const std::vector<int64_t> &lsizes) const;
    JaggedTensor jreshape(const std::vector<std::vector<int64_t>> &lsizes) const;

    /// @brief Reshape this JaggedTensor to have the same list structure as another JaggedTensor.
    ///       Note this function creates a view over the original JaggedTensor so modifying the
    ///       returned JaggedTensor will modify the original tensor.
    /// @param other The JaggedTensor to reshape this JaggedTensor to have the same list structure
    /// as
    /// @return A JaggedTensor with the same list structure as the other JaggedTensor
    JaggedTensor jreshape_as(const JaggedTensor &other) const;

    /// Flatten one of the list dimensions of this JaggedTensor. i.e. if this JaggedTensor
    /// represents a list of lists of tensors then jflatten(0) will flatten the outer list dimension
    /// and jflatten(1) will flatten the inner list dimension. e.g. if this JaggedTensor represents
    /// a list of lists of tensors [[A, B], [C], [D, E]] then
    ///     - jflatten(0) will return a JaggedTensor [A, B, C, D, E]
    ///     - jflatten(1) will return a JaggedTensor [[torch.cat(A, B, dim=0)], [C], [torch.cat(D,
    ///     E, dim=0)]]
    /// e.g. if this JaggedTensor represents a list of tensors with shapes [A, B, C] then
    ///    - jflatten(0) will return a JaggedTensor with shape [torch.cat(A, B, C, dim=0)]
    ///    - jflatten(1) will raise an exception as there is no inner list dimension
    /// Note this function creates a view over the original JaggedTensor so modifying the returned
    /// JaggedTensor will modify the original tensor.
    /// @param dim The dimension to flatten
    /// @return A JaggedTensor with the flattened list dimension
    JaggedTensor jflatten(const int64_t dim = 0) const;

    /// @brief Sorts each batch element in ascending order, note that jdata has to be 1-dimensional
    /// @return An indexing tensor with the same size as jdata, that permutes the elements of data
    /// to be in sorted order
    // JaggedTensor jagged_argsort();

    /// @brief Compute the summation of each batch element
    /// @param dim The dimension to sum over
    /// @param keepdim Whether to keep the summed dimension
    /// @return A tensor of size (batch_size, *) containing the sum of each batch element, feature
    /// dimensions are preserved
    JaggedTensor jsum(int64_t dim = 0, bool keepdim = false) const;

    /// @brief Compute the minimum of each batch element
    /// @param dim The dimension to sum over
    /// @param keepdim Whether to keep the min dimension
    /// @return Minimum value of size (batch_size, *) and argmin of size (batch_size, *)
    std::vector<JaggedTensor> jmin(int64_t dim = 0, bool keepdim = false) const;

    /// @brief Compute the maximum of each batch element
    /// @param dim The dimension to sum over
    /// @param keepdim Whether to keep the max dimension
    /// @return Maximum value of size (batch_size, *) and argmax of size (batch_size, *)
    std::vector<JaggedTensor> jmax(int64_t dim = 0, bool keepdim = false) const;

    // Operators on raw data
    inline int64_t
    rsize(int64_t dim) const {
        return mData.size(dim);
    }
    inline int64_t
    rdim() const {
        return mData.dim();
    }
    inline std::vector<int64_t>
    rsizes() const {
        return mData.sizes().vec();
    }
    JaggedTensor rmask(const torch::Tensor &mask) const;

    /// @brief Get an accessor for the JaggedTensor. Useful for reading/writing values in the
    /// JaggedTensor
    /// @tparam Scalar The type of the data in the JaggedTensor
    /// @tparam NDims The number of dimensions of the data in the JaggedTensor (i.e. edim() + 1)
    /// @return An accessor for the JaggedTensor
    template <typename Scalar, size_t NDims>
    JaggedAccessor<Scalar, NDims>
    accessor() const {
        return JaggedAccessor<Scalar, NDims>(
            mBatchIdx.accessor<JIdxType, 1>(), mOffsets.accessor<JOffsetsType, 1>(),
            mListIdx.accessor<JLIdxType, 2>(), mData.accessor<Scalar, NDims>());
    }

    /// @brief Get a packed accessor for the JaggedTensor. Useful for reading/writing values in the
    /// JaggedTensor in Cuda
    /// @tparam Scalar The type of the data in the JaggedTensor
    /// @tparam NDims The number of dimensions of the data in the JaggedTensor (i.e. edim() + 1)
    /// @tparam PtrTraits The type of the pointer traits for the packed accessor
    /// @return A packed accessor for the JaggedTensor
    template <typename Scalar, size_t NDims,
              template <typename U> typename PtrTraits = torch::DefaultPtrTraits>
    PackedJaggedAccessor32<Scalar, NDims, PtrTraits>
    packed_accessor32() const {
        return PackedJaggedAccessor32<Scalar, NDims, PtrTraits>(
            mBatchIdx.packed_accessor32<JIdxType, 1, PtrTraits>(),
            mOffsets.packed_accessor32<JOffsetsType, 1, PtrTraits>(),
            mListIdx.packed_accessor32<JLIdxType, 2, PtrTraits>(),
            mData.packed_accessor32<Scalar, NDims, PtrTraits>());
    }

    /// @brief Raise an exception if the JaggedTensor is in an invalid state
    inline void
    check_valid() const {
        TORCH_CHECK((jidx().size(0) == 0 && joffsets().size(0) == 2) ||
                        (jidx().size(0) == jdata().size(0)),
                    "tensor must be a valid JaggedTensor");
        TORCH_CHECK(jidx().device() == jdata().device(),
                    "batch index and data must be on the same device");
        TORCH_CHECK(jidx().dtype() == JIdxScalarType, "batch index must be int");
        TORCH_CHECK(joffsets().device() == jdata().device(),
                    "offsets and data must be on the same device");
        TORCH_CHECK_VALUE(jlidx().numel() == 0 || jlidx().size(0) == (joffsets().size(0) - 1),
                          "Corrupt list indices. This should never happen");
    }

    inline int64_t
    element_count() const {
        return jdata().size(0);
    }

    inline torch::Device
    device() const {
        return mData.device();
    }

    caffe2::TypeMeta
    dtype() const {
        return mData.dtype();
    }

    torch::Layout
    layout() const {
        return mData.layout();
    }

    inline torch::ScalarType
    scalar_type() const {
        return mData.scalar_type();
    }

    inline bool
    is_cuda() const {
        return mData.is_cuda();
    }

    inline bool
    is_cpu() const {
        return mData.is_cpu();
    }

    int64_t
    get_device() const {
        return mData.get_device();
    }

    bool
    is_complex() const {
        return at::isComplexType(this->scalar_type());
    }

    bool
    is_floating_point() const {
        return at::isFloatingType(this->scalar_type());
    }

    bool
    is_signed() const {
        return at::isSignedType(this->scalar_type());
    }

    int64_t
    numel() const {
        return mData.numel();
    }

    inline bool
    is_contiguous() const {
        return mData.is_contiguous();
    }

    inline JaggedTensor
    contiguous() const {
        return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
            mData.contiguous(), mOffsets.contiguous(), mBatchIdx.contiguous(),
            mListIdx.contiguous(), mNumOuterLists);
    }

    inline JaggedTensor
    to(at::TensorOptions options = {}, bool non_blocking = false, bool copy = false,
       c10::optional<at::MemoryFormat> memory_format = c10::nullopt) const {
        JaggedTensor ret = *this;
        ret.mData        = ret.mData.to(options, non_blocking, copy, memory_format);
        ret.mBatchIdx    = ret.mBatchIdx.to(ret.mData.device(), non_blocking, copy, memory_format);
        ret.mOffsets     = ret.mOffsets.to(ret.mData.device(), non_blocking, copy, memory_format);
        ret.mListIdx     = ret.mListIdx.to(ret.mData.device(), non_blocking, copy, memory_format);
        return ret;
    }

    inline JaggedTensor
    to(c10::optional<torch::ScalarType> dtype, c10::optional<at::Layout> layout,
       c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking,
       bool copy, c10::optional<at::MemoryFormat> memory_format) {
        JaggedTensor ret = *this;
        ret.mData =
            ret.mData.to(dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
        ret.mBatchIdx = ret.mBatchIdx.to(JIdxScalarType, layout, device, pin_memory, non_blocking,
                                         copy, memory_format);
        ret.mOffsets = ret.mOffsets.to(JOffsetsScalarType, layout, device, pin_memory, non_blocking,
                                       copy, memory_format);
        ret.mListIdx = ret.mListIdx.to(JLIdxScalarType, layout, device, pin_memory, non_blocking,
                                       copy, memory_format);
        return ret;
    }
    inline JaggedTensor
    to(torch::Device device, torch::ScalarType dtype, bool non_blocking = false, bool copy = false,
       c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
        JaggedTensor ret = *this;
        ret.mData        = ret.mData.to(device, dtype, non_blocking, copy, memory_format);
        ret.mBatchIdx    = ret.mBatchIdx.to(device, non_blocking, copy, memory_format);
        ret.mOffsets     = ret.mOffsets.to(device, non_blocking, copy, memory_format);
        ret.mListIdx     = ret.mListIdx.to(device, non_blocking, copy, memory_format);
        return ret;
    }
    inline JaggedTensor
    to(torch::ScalarType dtype, bool non_blocking = false, bool copy = false,
       c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
        JaggedTensor ret = *this;
        ret.mData        = ret.mData.to(dtype, non_blocking, copy, memory_format);
        ret.mBatchIdx    = ret.mBatchIdx.to(JIdxScalarType, non_blocking, copy, memory_format);
        ret.mOffsets     = ret.mOffsets.to(JOffsetsScalarType, non_blocking, copy, memory_format);
        ret.mListIdx     = ret.mListIdx.to(JLIdxScalarType, non_blocking, copy, memory_format);
        return ret;
    }

    torch::TensorOptions
    options() const {
        return torch::TensorOptions().dtype(dtype()).device(device()).layout(layout());
    }

    JaggedTensor
    cuda() const {
        return to(this->options().device(torch::kCUDA), /*non_blocking*/ false, /*copy*/ false);
    }

    JaggedTensor
    cpu() const {
        return to(this->options().device(torch::kCPU), /*non_blocking*/ false, /*copy*/ false);
    }

    JaggedTensor operator+(const JaggedTensor &other) const;
    JaggedTensor operator+(const int other) const;
    JaggedTensor operator+(const float other) const;
    JaggedTensor operator+(const torch::Tensor &other) const;

    JaggedTensor &operator+=(const JaggedTensor &other);
    JaggedTensor &operator+=(const int other);
    JaggedTensor &operator+=(const float other);
    JaggedTensor &operator+=(const torch::Tensor &other);

    JaggedTensor operator-(const JaggedTensor &other) const;
    JaggedTensor operator-(const int other) const;
    JaggedTensor operator-(const float other) const;
    JaggedTensor operator-(const torch::Tensor &other) const;

    JaggedTensor operator-() const;

    JaggedTensor &operator-=(const JaggedTensor &other);
    JaggedTensor &operator-=(const int other);
    JaggedTensor &operator-=(const float other);
    JaggedTensor &operator-=(const torch::Tensor &other);

    JaggedTensor operator*(const JaggedTensor &other) const;
    JaggedTensor operator*(const int other) const;
    JaggedTensor operator*(const float other) const;
    JaggedTensor operator*(const torch::Tensor &other) const;

    JaggedTensor &operator*=(const JaggedTensor &other);
    JaggedTensor &operator*=(const int other);
    JaggedTensor &operator*=(const float other);
    JaggedTensor &operator*=(const torch::Tensor &other);

    JaggedTensor operator/(const JaggedTensor &other) const;
    JaggedTensor operator/(const int other) const;
    JaggedTensor operator/(const float other) const;
    JaggedTensor operator/(const torch::Tensor &other) const;

    JaggedTensor &operator/=(const JaggedTensor &other);
    JaggedTensor &operator/=(const int other);
    JaggedTensor &operator/=(const float other);
    JaggedTensor &operator/=(const torch::Tensor &other);

    JaggedTensor floordiv(const JaggedTensor &other) const;
    JaggedTensor floordiv(const int other) const;
    JaggedTensor floordiv(const float other) const;
    JaggedTensor floordiv(const torch::Tensor &other) const;

    JaggedTensor &floordiveq(const JaggedTensor &other);
    JaggedTensor &floordiveq(const int other);
    JaggedTensor &floordiveq(const float other);
    JaggedTensor &floordiveq(const torch::Tensor &other);

    JaggedTensor operator%(const JaggedTensor &other) const;
    JaggedTensor operator%(const int other) const;
    JaggedTensor operator%(const float other) const;
    JaggedTensor operator%(const torch::Tensor &other) const;

    JaggedTensor &operator%=(const JaggedTensor &other);
    JaggedTensor &operator%=(const int other);
    JaggedTensor &operator%=(const float other);
    JaggedTensor &operator%=(const torch::Tensor &other);

    JaggedTensor pow(const JaggedTensor &other) const;
    JaggedTensor pow(const int other) const;
    JaggedTensor pow(const float other) const;
    JaggedTensor pow(const torch::Tensor &other) const;

    JaggedTensor &poweq(const JaggedTensor &other);
    JaggedTensor &poweq(const int other);
    JaggedTensor &poweq(const float other);
    JaggedTensor &poweq(const torch::Tensor &other);

    JaggedTensor operator>(const JaggedTensor &other) const;
    JaggedTensor operator>(const int other) const;
    JaggedTensor operator>(const float other) const;
    JaggedTensor operator>(const torch::Tensor &other) const;

    JaggedTensor operator>=(const JaggedTensor &other) const;
    JaggedTensor operator>=(const int other) const;
    JaggedTensor operator>=(const float other) const;
    JaggedTensor operator>=(const torch::Tensor &other) const;

    JaggedTensor operator<(const JaggedTensor &other) const;
    JaggedTensor operator<(const int other) const;
    JaggedTensor operator<(const float other) const;
    JaggedTensor operator<(const torch::Tensor &other) const;

    JaggedTensor operator<=(const JaggedTensor &other) const;
    JaggedTensor operator<=(const int other) const;
    JaggedTensor operator<=(const float other) const;
    JaggedTensor operator<=(const torch::Tensor &other) const;

    JaggedTensor operator==(const JaggedTensor &other) const;
    JaggedTensor operator==(const int other) const;
    JaggedTensor operator==(const float other) const;
    JaggedTensor operator==(const torch::Tensor &other) const;

    JaggedTensor operator!=(const JaggedTensor &other) const;
    JaggedTensor operator!=(const int other) const;
    JaggedTensor operator!=(const float other) const;
    JaggedTensor operator!=(const torch::Tensor &other) const;

    JaggedTensor sqrt() const;
    JaggedTensor abs() const;
    JaggedTensor round(int decimals = 0) const;
    JaggedTensor floor() const;
    JaggedTensor ceil() const;

    JaggedTensor &sqrt_();
    JaggedTensor &abs_();
    JaggedTensor &round_(int decimals = 0);
    JaggedTensor &floor_();
    JaggedTensor &ceil_();

    const JaggedTensor &set_requires_grad(bool requires_grad) const;
    bool                requires_grad() const;
    JaggedTensor        detach() const;
    JaggedTensor        clone() const;
};

struct JaggedTensorIndex {
    JaggedTensorIndex(c10::nullopt_t) : mType(JaggedTensorIndexType::None) {}
    JaggedTensorIndex(int64_t integer) : mType(JaggedTensorIndexType::Integer), mInteger(integer) {}
    JaggedTensorIndex(torch::indexing::EllipsisIndexType)
        : mType(JaggedTensorIndexType::Ellipsis) {}
    JaggedTensorIndex(at::Tensor tensor) : mType(JaggedTensorIndexType::Tensor), mTensor(tensor) {}
    JaggedTensorIndex(torch::indexing::Slice slice)
        : mType(JaggedTensorIndexType::Slice), mSlice(slice) {}
    JaggedTensorIndex(fvdb::JaggedTensor jaggedTensor)
        : mType(JaggedTensorIndexType::JaggedTensor), mJaggedTensor(jaggedTensor) {}

    template <class T, class = typename std::enable_if<std::is_same<bool, T>::value>::type>
    JaggedTensorIndex(T boolean) : mType(JaggedTensorIndexType::Boolean), mBoolean(boolean) {}

    inline bool
    is_none() const {
        return mType == JaggedTensorIndexType::None;
    }

    inline bool
    is_ellipsis() const {
        return mType == JaggedTensorIndexType::Ellipsis;
    }

    inline bool
    is_integer() const {
        return mType == JaggedTensorIndexType::Integer;
    }

    inline bool
    is_boolean() const {
        return mType == JaggedTensorIndexType::Boolean;
    }

    inline bool
    is_slice() const {
        return mType == JaggedTensorIndexType::Slice;
    }

    inline bool
    is_tensor() const {
        return mType == JaggedTensorIndexType::Tensor;
    }

    inline bool
    is_jagged_tensor() const {
        return mType == JaggedTensorIndexType::JaggedTensor;
    }

    inline int64_t
    integer() const {
        return mInteger;
    }

    inline bool
    boolean() const {
        return mBoolean;
    }

    inline const torch::indexing::Slice &
    slice() const {
        return mSlice;
    }

    inline const torch::Tensor &
    tensor() const {
        return mTensor;
    }

    inline const fvdb::JaggedTensor &
    jagged_tensor() const {
        return mJaggedTensor;
    }

  private:
    enum class JaggedTensorIndexType {
        None,
        Ellipsis,
        Integer,
        Slice,
        Tensor,
        Boolean,
        JaggedTensor
    };
    JaggedTensorIndexType mType;

    torch::Tensor          mTensor;
    int64_t                mInteger;
    torch::indexing::Slice mSlice;
    bool                   mBoolean;
    fvdb::JaggedTensor     mJaggedTensor;
};

} // namespace fvdb

#endif // FVDB_JAGGEDTENSOR_H