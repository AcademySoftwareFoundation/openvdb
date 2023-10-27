// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/io/Compression.h> // for io::readCompressedValues(), etc
#include <openvdb/util/NodeMasks.h>
#include <tbb/spin_mutex.h>
#include <algorithm> // for std::swap
#include <atomic>
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

#ifdef OPENVDB_USE_DELAYED_LOADING
    struct FileInfo
    {
        FileInfo(): bufpos(0) , maskpos(0) {}
        std::streamoff bufpos;
        std::streamoff maskpos;
        io::MappedFile::Ptr mapping;
        SharedPtr<io::StreamMetadata> meta;
    };
#endif

    /// Default constructor
    inline LeafBuffer(): mData(new ValueType[SIZE])
    {
#ifdef OPENVDB_USE_DELAYED_LOADING
        mOutOfCore = 0;
#endif
    }
    /// Construct a buffer populated with the specified value.
    explicit inline LeafBuffer(const ValueType&);
    /// Copy constructor
    inline LeafBuffer(const LeafBuffer&);
    /// Construct a buffer but don't allocate memory for the full array of values.
    LeafBuffer(PartialCreate, const ValueType&): mData(nullptr)
    {
#ifdef OPENVDB_USE_DELAYED_LOADING
        mOutOfCore = 0;
#endif
    }
    /// Destructor
    inline ~LeafBuffer();

    /// Return @c true if this buffer's values have not yet been read from disk.
    bool isOutOfCore() const
    {
#ifdef OPENVDB_USE_DELAYED_LOADING
        return bool(mOutOfCore);
#else
        return false;
#endif
    }
    /// Return @c true if memory for this buffer has not yet been allocated.
    bool empty() const { return !mData || this->isOutOfCore(); }
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
    inline Index memUsageIfLoaded() const;
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

    inline void setOutOfCore(bool b)
    {
        (void) b;
#ifdef OPENVDB_USE_DELAYED_LOADING
        mOutOfCore = b;
#endif
    }
    // To facilitate inlining in the common case in which the buffer is in-core,
    // the loading logic is split into a separate function, doLoad().
    inline void loadValues() const
    {
#ifdef OPENVDB_USE_DELAYED_LOADING
        if (this->isOutOfCore()) this->doLoad();
#endif
    }
    inline void doLoad() const;
    inline bool detachFromFile();

    using FlagsType = std::atomic<Index32>;

#ifdef OPENVDB_USE_DELAYED_LOADING
    union {
        ValueType* mData;
        FileInfo*  mFileInfo;
    };
#else
    ValueType* mData;
#endif
    FlagsType mOutOfCore; // interpreted as bool; extra bits reserved for future use
    tbb::spin_mutex mMutex; // 1 byte
    //int8_t mReserved[3]; // padding for alignment

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
#ifdef OPENVDB_USE_DELAYED_LOADING
    mOutOfCore = 0;
#endif
    this->fill(val);
}


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::~LeafBuffer()
{
#ifdef OPENVDB_USE_DELAYED_LOADING
    if (this->isOutOfCore()) {
        this->detachFromFile();
    } else {
        this->deallocate();
    }
#else
    this->deallocate();
#endif
}


template<typename T, Index Log2Dim>
inline
LeafBuffer<T, Log2Dim>::LeafBuffer(const LeafBuffer& other)
    : mData(nullptr)
#ifdef OPENVDB_USE_DELAYED_LOADING
    , mOutOfCore(other.mOutOfCore.load())
#endif
{
#ifdef OPENVDB_USE_DELAYED_LOADING
    if (other.isOutOfCore()) {
        mFileInfo = new FileInfo(*other.mFileInfo);
    } else {
#endif
        if (other.mData != nullptr) {
            this->allocate();
            ValueType* target = mData;
            const ValueType* source = other.mData;
            Index n = SIZE;
            while (n--) *target++ = *source++;
        }
#ifdef OPENVDB_USE_DELAYED_LOADING
    }
#endif
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::setValue(Index i, const ValueType& val)
{
    assert(i < SIZE);
    this->loadValues();
    if (mData) mData[i] = val;
}


template<typename T, Index Log2Dim>
inline LeafBuffer<T, Log2Dim>&
LeafBuffer<T, Log2Dim>::operator=(const LeafBuffer& other)
{
    if (&other != this) {
#ifdef OPENVDB_USE_DELAYED_LOADING
        if (this->isOutOfCore()) {
            this->detachFromFile();
        } else {
            if (other.isOutOfCore()) this->deallocate();
        }
        if (other.isOutOfCore()) {
            mOutOfCore.store(other.mOutOfCore.load(std::memory_order_acquire),
                             std::memory_order_release);
            mFileInfo = new FileInfo(*other.mFileInfo);
        } else {
#endif
            if (other.mData != nullptr) {
                this->allocate();
                ValueType* target = mData;
                const ValueType* source = other.mData;
                Index n = SIZE;
                while (n--) *target++ = *source++;
            }
#ifdef OPENVDB_USE_DELAYED_LOADING
        }
#endif
    }
    return *this;
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::fill(const ValueType& val)
{
    this->detachFromFile();
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
    this->loadValues();
    other.loadValues();
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
#ifdef OPENVDB_USE_DELAYED_LOADING
    // Two atomics can't be swapped because it would require hardware support:
    // https://en.wikipedia.org/wiki/Double_compare-and-swap
    // Note that there's a window in which other.mOutOfCore could be written
    // between our load from it and our store to it.
    auto tmp = other.mOutOfCore.load(std::memory_order_acquire);
    tmp = mOutOfCore.exchange(std::move(tmp));
    other.mOutOfCore.store(std::move(tmp), std::memory_order_release);
#endif
}


template<typename T, Index Log2Dim>
inline Index
LeafBuffer<T, Log2Dim>::memUsage() const
{
    size_t n = sizeof(*this);
#ifdef OPENVDB_USE_DELAYED_LOADING
    if (this->isOutOfCore()) n += sizeof(FileInfo);
    else {
#endif
        if (mData) n += SIZE * sizeof(ValueType);
#ifdef OPENVDB_USE_DELAYED_LOADING
    }
#endif
    return static_cast<Index>(n);
}


template<typename T, Index Log2Dim>
inline Index
LeafBuffer<T, Log2Dim>::memUsageIfLoaded() const
{
    size_t n = sizeof(*this);
    n += SIZE * sizeof(ValueType);
    return static_cast<Index>(n);
}


template<typename T, Index Log2Dim>
inline const typename LeafBuffer<T, Log2Dim>::ValueType*
LeafBuffer<T, Log2Dim>::data() const
{
    this->loadValues();
    if (mData == nullptr) {
        LeafBuffer* self = const_cast<LeafBuffer*>(this);
#ifdef OPENVDB_USE_DELAYED_LOADING
        // This lock will be contended at most once.
        tbb::spin_mutex::scoped_lock lock(self->mMutex);
#endif
        if (mData == nullptr) self->mData = new ValueType[SIZE];
    }
    return mData;
}

template<typename T, Index Log2Dim>
inline typename LeafBuffer<T, Log2Dim>::ValueType*
LeafBuffer<T, Log2Dim>::data()
{
    this->loadValues();
    if (mData == nullptr) {
#ifdef OPENVDB_USE_DELAYED_LOADING
        // This lock will be contended at most once.
        tbb::spin_mutex::scoped_lock lock(mMutex);
#endif
        if (mData == nullptr) mData = new ValueType[SIZE];
    }
    return mData;
}


template<typename T, Index Log2Dim>
inline const typename LeafBuffer<T, Log2Dim>::ValueType&
LeafBuffer<T, Log2Dim>::at(Index i) const
{
    static const ValueType sZero = zeroVal<T>();
    assert(i < SIZE);
    this->loadValues();
    // We can't use the ternary operator here, otherwise Visual C++ returns
    // a reference to a temporary.
    if (mData) return mData[i]; else return sZero;
}


template<typename T, Index Log2Dim>
inline bool
LeafBuffer<T, Log2Dim>::deallocate()
{

    if (mData != nullptr) {
#ifdef OPENVDB_USE_DELAYED_LOADING
        if (this->isOutOfCore())    return false;
#endif
        delete[] mData;
        mData = nullptr;
        return true;
    }
    return false;
}


template<typename T, Index Log2Dim>
inline void
LeafBuffer<T, Log2Dim>::doLoad() const
{
#ifdef OPENVDB_USE_DELAYED_LOADING
    if (!this->isOutOfCore()) return;

    LeafBuffer<T, Log2Dim>* self = const_cast<LeafBuffer<T, Log2Dim>*>(this);

    // This lock will be contended at most once, after which this buffer
    // will no longer be out-of-core.
    tbb::spin_mutex::scoped_lock lock(self->mMutex);
    if (!this->isOutOfCore()) return;

    std::unique_ptr<FileInfo> info(self->mFileInfo);
    assert(info.get() != nullptr);
    assert(info->mapping.get() != nullptr);
    assert(info->meta.get() != nullptr);

    /// @todo For now, we have to clear the mData pointer in order for allocate() to take effect.
    self->mData = nullptr;
    self->allocate();

    SharedPtr<std::streambuf> buf = info->mapping->createBuffer();
    std::istream is(buf.get());

    io::setStreamMetadataPtr(is, info->meta, /*transfer=*/true);

    NodeMaskType mask;
    is.seekg(info->maskpos);
    mask.load(is);

    is.seekg(info->bufpos);
    io::readCompressedValues(is, self->mData, SIZE, mask, io::getHalfFloat(is));

    self->setOutOfCore(false);
#endif
}


template<typename T, Index Log2Dim>
inline bool
LeafBuffer<T, Log2Dim>::detachFromFile()
{
#ifdef OPENVDB_USE_DELAYED_LOADING
    if (this->isOutOfCore()) {
        delete mFileInfo;
        mFileInfo = nullptr;
        this->setOutOfCore(false);
        return true;
    }
#endif
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
        assert(i < SIZE);
        // We can't use the ternary operator here, otherwise Visual C++ returns
        // a reference to a temporary.
        if (mData.isOn(i)) return sOn; else return sOff;
    }
    const bool& operator[](Index i) const { return this->getValue(i); }

    bool operator==(const LeafBuffer& other) const { return mData == other.mData; }
    bool operator!=(const LeafBuffer& other) const { return mData != other.mData; }

    void setValue(Index i, bool val) { assert(i < SIZE); mData.set(i, val); }

    void swap(LeafBuffer& other) { if (&other != this) std::swap(mData, other.mData); }

    Index memUsage() const { return sizeof(*this); }
    Index memUsageIfLoaded() const { return sizeof(*this); }
    static Index size() { return SIZE; }

    /// @brief Return a pointer to the C-style array of words encoding the bits.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    WordType* data() { return &(mData.template getWord<WordType>(0)); }
    /// @brief Return a const pointer to the C-style array of words encoding the bits.
    /// @warning This method should only be used by experts seeking low-level optimizations.
    const WordType* data() const { return const_cast<LeafBuffer*>(this)->data(); }

private:
    // Allow the parent LeafNode to access this buffer's data.
    template<typename, Index> friend class LeafNode;

    NodeMaskType mData;
}; // class LeafBuffer

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_LEAFBUFFER_HAS_BEEN_INCLUDED
