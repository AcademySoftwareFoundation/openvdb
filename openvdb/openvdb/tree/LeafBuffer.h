// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/io/Compression.h> // for io::readCompressedValues(), etc
#include <openvdb/util/NodeMasks.h>
#include <openvdb/util/Assert.h>
#if OPENVDB_ABI_VERSION_NUMBER < 14
#include <tbb/spin_mutex.h>
#include <atomic>
#endif
#include <algorithm> // for std::swap
#include <cstddef> // for offsetof()
#include <iostream>
#include <memory>
#include <type_traits>


class TestLeaf;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {


/// @brief Array of fixed size 2<SUP>3<I>Log2Dim</I></SUP> that stores
/// the voxel values of a LeafNode
template<typename T, Index Log2Dim>
class LeafBuffer
{
public:
    using ValueType = T;
    using StorageType = ValueType;
    using NodeMaskType = util::NodeMask<Log2Dim>;
    static const Index SIZE = 1 << 3 * Log2Dim;

    /// Default constructor
    inline LeafBuffer(): mData(new ValueType[SIZE]) {}
    /// Construct a buffer populated with the specified value.
    explicit inline LeafBuffer(const ValueType&);
    /// Copy constructor
    inline LeafBuffer(const LeafBuffer&);
    /// Construct a buffer but don't allocate memory for the full array of values.
    LeafBuffer(PartialCreate, const ValueType&): mData(nullptr) {}
    /// Destructor
    inline ~LeafBuffer();

    OPENVDB_DEPRECATED_MESSAGE("Always returns false. This method is deprecated and will be removed. Delayed loading is no longer supported.")
    bool isOutOfCore() const { return false; }
    /// Return @c true if memory for this buffer has not yet been allocated.
    bool empty() const { return !mData; }
    /// Allocate memory for this buffer if it has not already been allocated.
    bool allocate() { if (mData == nullptr) mData = new ValueType[SIZE]; return true; }

    /// Populate this buffer with a constant value.
    inline void fill(const ValueType&);

    /// Return a const reference to the i'th element of this buffer.
    const ValueType& getValue(Index i) const { return this->at(i); }
    /// Return a const reference to the i'th element of this buffer.
    const ValueType& operator[](Index i) const { return this->at(i); }
    /// Set the i'th value of this buffer to the specified value.
    inline void setValue(Index i, const ValueType&);

    /// Copy the other buffer's values into this buffer.
    inline LeafBuffer& operator=(const LeafBuffer&);

    /// @brief Return @c true if the contents of the other buffer
    /// exactly equal the contents of this buffer.
    inline bool operator==(const LeafBuffer&) const;
    /// @brief Return @c true if the contents of the other buffer
    /// are not exactly equal to the contents of this buffer.
    inline bool operator!=(const LeafBuffer& other) const { return !(other == *this); }

    /// Exchange this buffer's values with the other buffer's values.
    inline void swap(LeafBuffer&);

    /// Return the memory footprint of this buffer in bytes.
    inline Index memUsage() const;
    OPENVDB_DEPRECATED_MESSAGE("Use memUsage() instead. This method is deprecated and will be removed. Delayed loading is no longer supported.")
    inline Index memUsageIfLoaded() const { return memUsage(); }
    /// Return the number of values contained in this buffer.
    static Index size() { return SIZE; }

    /// @brief Return a const pointer to the array of voxel values.
    /// @details This method guarantees that the buffer is allocated and loaded.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    const ValueType* data() const;
    /// @brief Return a pointer to the array of voxel values.
    /// @details This method guarantees that the buffer is allocated and loaded.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    ValueType* data();

private:
    /// If this buffer is empty, return zero, otherwise return the value at index @ i.
    inline const ValueType& at(Index i) const;

    /// @brief Return a non-const reference to the value at index @a i.
    /// @details This method is private since it makes assumptions about the
    /// buffer's memory layout.  LeafBuffers associated with custom leaf node types
    /// (e.g., a bool buffer implemented as a bitmask) might not be able to
    /// return non-const references to their values.
    ValueType& operator[](Index i) { return const_cast<ValueType&>(this->at(i)); }

    bool deallocate();

    ValueType* mData = nullptr;
#if OPENVDB_ABI_VERSION_NUMBER < 14
    // Deprecated members kept for ABI compatibility
    std::atomic<Index32> mDeprecatedAtomic{0};
    tbb::spin_mutex mDeprecatedSpinMutex;
#endif

    friend class ::TestLeaf;
    // Allow the parent LeafNode to access this buffer's data pointer.
    template<typename, Index> friend class LeafNode;
}; // class LeafBuffer


////////////////////////////////////////


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::LeafBuffer(const ValueType& val)
    : mData(new ValueType[SIZE])
{
    this->fill(val);
}


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::~LeafBuffer()
{
    this->deallocate();
}


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::LeafBuffer(const LeafBuffer& other)
    : mData(nullptr)
{
    if (other.mData != nullptr) {
        this->allocate();
        ValueType* target = mData;
        const ValueType* source = other.mData;
        Index n = SIZE;
        while (n--) *target++ = *source++;
    }
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::setValue(Index i, const ValueType& val)
{
    OPENVDB_ASSERT(i < SIZE);
    if (mData) mData[i] = val;
}


template<typename T, Index Log2Dim>
inline LeafBuffer<T, Log2Dim>&
LeafBuffer<T, Log2Dim>::operator=(const LeafBuffer& other)
{
    if (&other != this) {
        if (other.mData != nullptr) {
            this->allocate();
            ValueType* target = mData;
            const ValueType* source = other.mData;
            Index n = SIZE;
            while (n--) *target++ = *source++;
        }
    }
    return *this;
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::fill(const ValueType& val)
{
    if (mData != nullptr) {
        ValueType* target = mData;
        Index n = SIZE;
        while (n--) *target++ = val;
    }
}


template<typename T, Index Log2Dim>
inline bool
LeafBuffer<T, Log2Dim>::operator==(const LeafBuffer& other) const
{
    const ValueType *target = mData, *source = other.mData;
    if (!target && !source) return true;
    if (!target || !source) return false;
    Index n = SIZE;
    while (n && math::isExactlyEqual(*target++, *source++)) --n;
    return n == 0;
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::swap(LeafBuffer& other)
{
    std::swap(mData, other.mData);
}


template<typename T, Index Log2Dim>
inline Index
LeafBuffer<T, Log2Dim>::memUsage() const
{
    size_t n = sizeof(*this);
    if (mData) n += SIZE * sizeof(ValueType);
    return static_cast<Index>(n);
}


template<typename T, Index Log2Dim>
inline const typename LeafBuffer<T, Log2Dim>::ValueType*
LeafBuffer<T, Log2Dim>::data() const
{
    if (mData == nullptr) {
        LeafBuffer* self = const_cast<LeafBuffer*>(this);
        if (mData == nullptr) self->mData = new ValueType[SIZE];
    }
    return mData;
}

template<typename T, Index Log2Dim>
inline typename LeafBuffer<T, Log2Dim>::ValueType*
LeafBuffer<T, Log2Dim>::data()
{
    if (mData == nullptr) {
        if (mData == nullptr) mData = new ValueType[SIZE];
    }
    return mData;
}


template<typename T, Index Log2Dim>
inline const typename LeafBuffer<T, Log2Dim>::ValueType&
LeafBuffer<T, Log2Dim>::at(Index i) const
{
    static const ValueType sZero = zeroVal<T>();
    OPENVDB_ASSERT(i < SIZE);
    // We can't use the ternary operator here, otherwise Visual C++ returns
    // a reference to a temporary.
    if (mData) return mData[i]; else return sZero;
}


template<typename T, Index Log2Dim>
inline bool
LeafBuffer<T, Log2Dim>::deallocate()
{
    if (mData != nullptr) {
        delete[] mData;
        mData = nullptr;
        return true;
    }
    return false;
}


////////////////////////////////////////


// Partial specialization for bool ValueType
template<Index Log2Dim>
class LeafBuffer<bool, Log2Dim>
{
public:
    using NodeMaskType = util::NodeMask<Log2Dim>;
    using WordType = typename NodeMaskType::Word;
    using ValueType = bool;
    using StorageType = WordType;

    static const Index WORD_COUNT = NodeMaskType::WORD_COUNT;
    static const Index SIZE = 1 << 3 * Log2Dim;

    static inline const bool sOn = true;
    static inline const bool sOff = false;

    LeafBuffer() {}
    LeafBuffer(bool on): mData(on) {}
    LeafBuffer(const NodeMaskType& other): mData(other) {}
    LeafBuffer(const LeafBuffer& other): mData(other.mData) {}
    ~LeafBuffer() {}
    void fill(bool val) { mData.set(val); }
    LeafBuffer& operator=(const LeafBuffer& b) { if (&b != this) { mData=b.mData; } return *this; }

    const bool& getValue(Index i) const
    {
        OPENVDB_ASSERT(i < SIZE);
        // We can't use the ternary operator here, otherwise Visual C++ returns
        // a reference to a temporary.
        if (mData.isOn(i)) return sOn; else return sOff;
    }
    const bool& operator[](Index i) const { return this->getValue(i); }

    bool operator==(const LeafBuffer& other) const { return mData == other.mData; }
    bool operator!=(const LeafBuffer& other) const { return mData != other.mData; }

    void setValue(Index i, bool val) { OPENVDB_ASSERT(i < SIZE); mData.set(i, val); }

    void swap(LeafBuffer& other) { if (&other != this) std::swap(mData, other.mData); }

    Index memUsage() const { return sizeof(*this); }
    OPENVDB_DEPRECATED_MESSAGE("Use memUsage() instead. This method is deprecated and will be removed. Delayed loading is no longer supported.")
    Index memUsageIfLoaded() const { return memUsage(); }
    static Index size() { return SIZE; }

    /// @brief Return a pointer to the C-style array of words encoding the bits.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    WordType* data() { return &(mData.template getWord<WordType>(0)); }
    /// @brief Return a const pointer to the C-style array of words encoding the bits.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    const WordType* data() const { return const_cast<LeafBuffer*>(this)->data(); }

    /// @brief Return raw LeafBuffer data
    const NodeMaskType& storage() const { return mData; }

private:
    // Allow the parent LeafNode to access this buffer's data.
    template<typename, Index> friend class LeafNode;

    NodeMaskType mData;
}; // class LeafBuffer

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED
