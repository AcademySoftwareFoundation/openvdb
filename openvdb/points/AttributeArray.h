///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

/// @file points/AttributeArray.h
///
/// @authors Dan Bailey, Mihai Alden, Nick Avramoussis, James Bird, Khang Ngo
///
/// @brief  Attribute Array storage templated on type and compression codec.

#ifndef OPENVDB_POINTS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/math/QuantizedUnitVec.h>
#include <openvdb/util/Name.h>
#include <openvdb/util/logging.h>
#include <openvdb/io/io.h> // MappedFile
#include <openvdb/io/Compression.h> // COMPRESS_BLOSC

#include "IndexIterator.h"
#include "StreamCompression.h"

#include <tbb/spin_mutex.h>
#include <tbb/atomic.h>

#include <memory>
#include <string>
#include <type_traits>


class TestAttributeArray;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {


using NamePair = std::pair<Name, Name>;

namespace points {


////////////////////////////////////////

// Utility methods

template <typename IntegerT, typename FloatT>
inline IntegerT
floatingPointToFixedPoint(const FloatT s)
{
    static_assert(std::is_unsigned<IntegerT>::value, "IntegerT must be unsigned");
    if (FloatT(0.0) > s) return std::numeric_limits<IntegerT>::min();
    else if (FloatT(1.0) <= s) return std::numeric_limits<IntegerT>::max();
    return IntegerT(std::floor(s * FloatT(std::numeric_limits<IntegerT>::max())));
}


template <typename FloatT, typename IntegerT>
inline FloatT
fixedPointToFloatingPoint(const IntegerT s)
{
    static_assert(std::is_unsigned<IntegerT>::value, "IntegerT must be unsigned");
    return FloatT(s) / FloatT((std::numeric_limits<IntegerT>::max()));
}

template <typename IntegerVectorT, typename FloatT>
inline IntegerVectorT
floatingPointToFixedPoint(const math::Vec3<FloatT>& v)
{
    return IntegerVectorT(
        floatingPointToFixedPoint<typename IntegerVectorT::ValueType>(v.x()),
        floatingPointToFixedPoint<typename IntegerVectorT::ValueType>(v.y()),
        floatingPointToFixedPoint<typename IntegerVectorT::ValueType>(v.z()));
}

template <typename FloatVectorT, typename IntegerT>
inline FloatVectorT
fixedPointToFloatingPoint(const math::Vec3<IntegerT>& v)
{
    return FloatVectorT(
        fixedPointToFloatingPoint<typename FloatVectorT::ValueType>(v.x()),
        fixedPointToFloatingPoint<typename FloatVectorT::ValueType>(v.y()),
        fixedPointToFloatingPoint<typename FloatVectorT::ValueType>(v.z()));
}


////////////////////////////////////////


/// Base class for storing attribute data
class OPENVDB_API AttributeArray
{
protected:
    struct AccessorBase;
    template <typename T> struct Accessor;

    using AccessorBasePtr = std::shared_ptr<AccessorBase>;

public:
    enum Flag {
        TRANSIENT = 0x1,            /// by default not written to disk
        HIDDEN = 0x2,               /// hidden from UIs or iterators
        OUTOFCORE = 0x4,            /// data not yet loaded from disk (deprecated flag as of ABI=5)
        CONSTANTSTRIDE = 0x8,       /// stride size does not vary in the array
        STREAMING = 0x10            /// streaming mode collapses attributes when first accessed
    };

    enum SerializationFlag {
        WRITESTRIDED = 0x1,         /// data is marked as strided when written
        WRITEUNIFORM = 0x2,         /// data is marked as uniform when written
        WRITEMEMCOMPRESS = 0x4,     /// data is marked as compressed in-memory when written
        WRITEPAGED = 0x8            /// data is written out in pages
    };

    using Ptr           = std::shared_ptr<AttributeArray>;
    using ConstPtr      = std::shared_ptr<const AttributeArray>;

    using FactoryMethod = Ptr (*)(Index, Index, bool);

    template <typename ValueType, typename CodecType> friend class AttributeHandle;

    AttributeArray() = default;
    AttributeArray(const AttributeArray&) = default;
    AttributeArray& operator=(const AttributeArray&) = default;
    virtual ~AttributeArray() = default;

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const = 0;

    /// Return an uncompressed copy of this attribute (will return a copy if not compressed).
    virtual AttributeArray::Ptr copyUncompressed() const = 0;

    /// Return the number of elements in this array.
    /// @note This does not count each data element in a strided array
    virtual Index size() const = 0;

    /// Return the stride of this array.
    /// @note a return value of zero means a non-constant stride
    virtual Index stride() const = 0;

    /// Return the total number of data elements in this array.
    /// @note This counts each data element in a strided array
    virtual Index dataSize() const = 0;

    /// Return the number of bytes of memory used by this attribute.
    virtual size_t memUsage() const = 0;

    /// Create a new attribute array of the given (registered) type, length and stride.
    static Ptr create(const NamePair& type, Index length, Index stride = 1, bool constantStride = true);
    /// Return @c true if the given attribute type name is registered.
    static bool isRegistered(const NamePair& type);
    /// Clear the attribute type registry.
    static void clearRegistry();

    /// Return the name of this attribute's type.
    virtual const NamePair& type() const = 0;
    /// Return @c true if this attribute is of the same type as the template parameter.
    template<typename AttributeArrayType>
    bool isType() const { return this->type() == AttributeArrayType::attributeType(); }

    /// Return @c true if this attribute has a value type the same as the template parameter
    template<typename ValueType>
    bool hasValueType() const { return this->type().first == typeNameAsString<ValueType>(); }

    /// Set value at given index @a n from @a sourceIndex of another @a sourceArray
    virtual void set(const Index n, const AttributeArray& sourceArray, const Index sourceIndex) = 0;

    /// Return @c true if this array is stored as a single uniform value.
    virtual bool isUniform() const = 0;
    /// @brief  If this array is uniform, replace it with an array of length size().
    /// @param  fill if true, assign the uniform value to each element of the array.
    virtual void expand(bool fill = true) = 0;
    /// Replace the existing array with a uniform zero value.
    virtual void collapse() = 0;
    /// Compact the existing array to become uniform if all values are identical
    virtual bool compact() = 0;

    /// Return @c true if this array is compressed.
    bool isCompressed() const { return mCompressedBytes != 0; }
    /// Compress the attribute array.
    virtual bool compress() = 0;
    /// Uncompress the attribute array.
    virtual bool decompress() = 0;

    /// @brief   Specify whether this attribute should be hidden (e.g., from UI or iterators).
    /// @details This is useful if the attribute is used for blind data or as scratch space
    ///          for a calculation.
    /// @note    Attributes are not hidden by default.
    void setHidden(bool state);
    /// Return @c true if this attribute is hidden (e.g., from UI or iterators).
    bool isHidden() const { return bool(mFlags & HIDDEN); }

    /// @brief Specify whether this attribute should only exist in memory
    ///        and not be serialized during stream output.
    /// @note  Attributes are not transient by default.
    void setTransient(bool state);
    /// Return @c true if this attribute is not serialized during stream output.
    bool isTransient() const { return bool(mFlags & TRANSIENT); }

    /// @brief Specify whether this attribute is to be streamed off disk, in which
    ///        case, the attributes are collapsed after being first loaded leaving them
    ///        in a destroyed state.
    ///  @note This operation is not thread-safe.
    void setStreaming(bool state);
    /// Return @c true if this attribute is in streaming mode.
    bool isStreaming() const { return bool(mFlags & STREAMING); }

    /// Return @c true if this attribute has a constant stride
    bool hasConstantStride() const { return bool(mFlags & CONSTANTSTRIDE); }

    /// @brief Retrieve the attribute array flags
    uint8_t flags() const { return mFlags; }

    /// Read attribute metadata and buffers from a stream.
    virtual void read(std::istream&) = 0;
    /// Write attribute metadata and buffers to a stream.
    /// @param outputTransient if true, write out transient attributes
    virtual void write(std::ostream&, bool outputTransient) const = 0;
    /// Write attribute metadata and buffers to a stream, don't write transient attributes.
    virtual void write(std::ostream&) const = 0;

    /// Read attribute metadata from a stream.
    virtual void readMetadata(std::istream&) = 0;
    /// Write attribute metadata to a stream.
    /// @param outputTransient if true, write out transient attributes
    /// @param paged           if true, data is written out in pages
    virtual void writeMetadata(std::ostream&, bool outputTransient, bool paged) const = 0;

    /// Read attribute buffers from a stream.
    virtual void readBuffers(std::istream&) = 0;
    /// Write attribute buffers to a stream.
    /// @param outputTransient if true, write out transient attributes
    virtual void writeBuffers(std::ostream&, bool outputTransient) const = 0;

    /// Read attribute buffers from a paged stream.
    virtual void readPagedBuffers(compression::PagedInputStream&) = 0;
    /// Write attribute buffers to a paged stream.
    /// @param outputTransient if true, write out transient attributes
    virtual void writePagedBuffers(compression::PagedOutputStream&, bool outputTransient) const = 0;

    /// Ensures all data is in-core
    virtual void loadData() const = 0;

    /// Check the compressed bytes and flags. If they are equal, perform a deeper
    /// comparison check necessary on the inherited types (TypedAttributeArray)
    /// Requires non operator implementation due to inheritance
    bool operator==(const AttributeArray& other) const;
    bool operator!=(const AttributeArray& other) const { return !this->operator==(other); }

private:
    friend class ::TestAttributeArray;

    /// Virtual function used by the comparison operator to perform
    /// comparisons on inherited types
    virtual bool isEqual(const AttributeArray& other) const = 0;

protected:
    /// @brief Specify whether this attribute has a constant stride or not.
    void setConstantStride(bool state);

    /// Obtain an Accessor that stores getter and setter functors.
    virtual AccessorBasePtr getAccessor() const = 0;

    /// Register a attribute type along with a factory function.
    static void registerType(const NamePair& type, FactoryMethod);
    /// Remove a attribute type from the registry.
    static void unregisterType(const NamePair& type);

    size_t mCompressedBytes = 0;
    uint8_t mFlags = 0;
    uint8_t mSerializationFlags = 0;

#if OPENVDB_ABI_VERSION_NUMBER >= 5
    tbb::atomic<Index32> mOutOfCore = 0; // interpreted as bool
#endif

    /// used for out-of-core, paged reading
    compression::PageHandle::Ptr mPageHandle;
}; // class AttributeArray


////////////////////////////////////////


/// Accessor base class for AttributeArray storage where type is not available
struct AttributeArray::AccessorBase { virtual ~AccessorBase() = default; };

/// Templated Accessor stores typed function pointers used in binding
/// AttributeHandles
template <typename T>
struct AttributeArray::Accessor : public AttributeArray::AccessorBase
{
    using GetterPtr = T (*)(const AttributeArray* array, const Index n);
    using SetterPtr = void (*)(AttributeArray* array, const Index n, const T& value);
    using ValuePtr  = void (*)(AttributeArray* array, const T& value);

    Accessor(GetterPtr getter, SetterPtr setter, ValuePtr collapser, ValuePtr filler) :
        mGetter(getter), mSetter(setter), mCollapser(collapser), mFiller(filler) { }

    GetterPtr mGetter;
    SetterPtr mSetter;
    ValuePtr  mCollapser;
    ValuePtr  mFiller;
}; // struct AttributeArray::Accessor


////////////////////////////////////////


namespace attribute_traits
{
    template <typename T> struct TruncateTrait { };
    template <> struct TruncateTrait<float> { using Type = half; };
    template <> struct TruncateTrait<int> { using Type = short; };

    template <typename T> struct TruncateTrait<math::Vec3<T>> {
        using Type = math::Vec3<typename TruncateTrait<T>::Type>;
    };

    template <bool OneByte, typename T> struct UIntTypeTrait { };
    template<typename T> struct UIntTypeTrait</*OneByte=*/true, T> { using Type = uint8_t; };
    template<typename T> struct UIntTypeTrait</*OneByte=*/false, T> { using Type = uint16_t; };
    template<typename T> struct UIntTypeTrait</*OneByte=*/true, math::Vec3<T>> {
        using Type = math::Vec3<uint8_t>;
    };
    template<typename T> struct UIntTypeTrait</*OneByte=*/false, math::Vec3<T>> {
        using Type = math::Vec3<uint16_t>;
    };
}


////////////////////////////////////////


// Attribute codec schemes

struct UnknownCodec { };


struct NullCodec
{
    template <typename T>
    struct Storage { using Type = T; };

    template<typename ValueType> static void decode(const ValueType&, ValueType&);
    template<typename ValueType> static void encode(const ValueType&, ValueType&);
    static const char* name() { return "null"; }
};


struct TruncateCodec
{
    template <typename T>
    struct Storage { using Type = typename attribute_traits::TruncateTrait<T>::Type; };

    template<typename StorageType, typename ValueType> static void decode(const StorageType&, ValueType&);
    template<typename StorageType, typename ValueType> static void encode(const ValueType&, StorageType&);
    static const char* name() { return "trnc"; }
};


// Fixed-point codec range for voxel-space positions [-0.5,0.5]
struct PositionRange
{
    static const char* name() { return "fxpt"; }
    template <typename ValueType> static ValueType encode(const ValueType& value) { return value + ValueType(0.5); }
    template <typename ValueType> static ValueType decode(const ValueType& value) { return value - ValueType(0.5); }
};


// Fixed-point codec range for unsigned values in the unit range [0.0,1.0]
struct UnitRange
{
    static const char* name() { return "ufxpt"; }
    template <typename ValueType> static ValueType encode(const ValueType& value) { return value; }
    template <typename ValueType> static ValueType decode(const ValueType& value) { return value; }
};


template <bool OneByte, typename Range=PositionRange>
struct FixedPointCodec
{
    template <typename T>
    struct Storage { using Type = typename attribute_traits::UIntTypeTrait<OneByte, T>::Type; };

    template<typename StorageType, typename ValueType> static void decode(const StorageType&, ValueType&);
    template<typename StorageType, typename ValueType> static void encode(const ValueType&, StorageType&);

    static const char* name() {
        static const std::string Name = std::string(Range::name()) + (OneByte ? "8" : "16");
        return Name.c_str();
    }
};


struct UnitVecCodec
{
    using StorageType = uint16_t;

    template <typename T>
    struct Storage { using Type = StorageType; };

    template<typename T> static void decode(const StorageType&, math::Vec3<T>&);
    template<typename T> static void encode(const math::Vec3<T>&, StorageType&);
    static const char* name() { return "uvec"; }
};


////////////////////////////////////////


/// Typed class for storing attribute data
template<typename ValueType_, typename Codec_ = NullCodec>
class TypedAttributeArray: public AttributeArray
{
public:
    using Ptr           = std::shared_ptr<TypedAttributeArray>;
    using ConstPtr      = std::shared_ptr<const TypedAttributeArray>;

    using ValueType     = ValueType_;
    using Codec         = Codec_;
    using StorageType   = typename Codec::template Storage<ValueType>::Type;

    //////////

    /// Default constructor, always constructs a uniform attribute.
    explicit TypedAttributeArray(Index n = 1, Index strideOrTotalSize = 1, bool constantStride = true,
        const ValueType& uniformValue = zeroVal<ValueType>());
    /// Deep copy constructor (optionally decompress during copy).
    TypedAttributeArray(const TypedAttributeArray&, bool uncompress = false);
    /// Deep copy assignment operator.
    TypedAttributeArray& operator=(const TypedAttributeArray&);
    /// Move constructor disabled.
    TypedAttributeArray(TypedAttributeArray&&) = delete;
    /// Move assignment operator disabled.
    TypedAttributeArray& operator=(TypedAttributeArray&&) = delete;

    virtual ~TypedAttributeArray() { this->deallocate(); }

    /// Return a copy of this attribute.
    AttributeArray::Ptr copy() const override;

    /// Return an uncompressed copy of this attribute (will just return a copy if not compressed).
    AttributeArray::Ptr copyUncompressed() const override;

    /// Return a new attribute array of the given length @a n and @a stride with uniform value zero.
    static Ptr create(Index n, Index strideOrTotalSize = 1, bool constantStride = true);

    /// Cast an AttributeArray to TypedAttributeArray<T>
    static TypedAttributeArray& cast(AttributeArray& attributeArray);

    /// Cast an AttributeArray to TypedAttributeArray<T>
    static const TypedAttributeArray& cast(const AttributeArray& attributeArray);

    /// Return the name of this attribute's type (includes codec)
    static const NamePair& attributeType();
    /// Return the name of this attribute's type.
    const NamePair& type() const override { return attributeType(); }

    /// Return @c true if this attribute type is registered.
    static bool isRegistered();
    /// Register this attribute type along with a factory function.
    static void registerType();
    /// Remove this attribute type from the registry.
    static void unregisterType();

    /// Return the number of elements in this array.
    Index size() const override { return mSize; }

    /// Return the stride of this array.
    /// @note A return value of zero means a variable stride
    Index stride() const override { return hasConstantStride() ? mStrideOrTotalSize : 0; }

    /// Return the size of the data in this array.
    Index dataSize() const override {
        return hasConstantStride() ? mSize * mStrideOrTotalSize : mStrideOrTotalSize;
    }

    /// Return the number of bytes of memory used by this attribute.
    size_t memUsage() const override;

    /// Return the value at index @a n (assumes uncompressed and in-core)
    ValueType getUnsafe(Index n) const;
    /// Return the value at index @a n
    ValueType get(Index n) const;
    /// Return the @a value at index @a n (assumes uncompressed and in-core)
    template<typename T> void getUnsafe(Index n, T& value) const;
    /// Return the @a value at index @a n
    template<typename T> void get(Index n, T& value) const;

    /// Non-member equivalent to getUnsafe() that static_casts array to this TypedAttributeArray
    /// (assumes uncompressed and in-core)
    static ValueType getUnsafe(const AttributeArray* array, const Index n);

    /// Set @a value at the given index @a n (assumes uncompressed and in-core)
    void setUnsafe(Index n, const ValueType& value);
    /// Set @a value at the given index @a n
    void set(Index n, const ValueType& value);
    /// Set @a value at the given index @a n (assumes uncompressed and in-core)
    template<typename T> void setUnsafe(Index n, const T& value);
    /// Set @a value at the given index @a n
    template<typename T> void set(Index n, const T& value);

    /// Non-member equivalent to setUnsafe() that static_casts array to this TypedAttributeArray
    /// (assumes uncompressed and in-core)
    static void setUnsafe(AttributeArray* array, const Index n, const ValueType& value);

    /// Set value at given index @a n from @a sourceIndex of another @a sourceArray
    void set(const Index n, const AttributeArray& sourceArray, const Index sourceIndex) override;

    /// Return @c true if this array is stored as a single uniform value.
    bool isUniform() const override { return mIsUniform; }
    /// @brief  Replace the single value storage with an array of length size().
    /// @note   Non-uniform attributes are unchanged.
    /// @param  fill toggle to initialize the array elements with the pre-expanded value.
    void expand(bool fill = true) override;
    /// Replace the existing array with a uniform zero value.
    void collapse() override;
    /// Compact the existing array to become uniform if all values are identical
    bool compact() override;

    /// Replace the existing array with the given uniform value.
    void collapse(const ValueType& uniformValue);
    /// @brief Fill the existing array with the given value.
    /// @note Identical to collapse() except a non-uniform array will not become uniform.
    void fill(const ValueType& value);

    /// Non-member equivalent to collapse() that static_casts array to this TypedAttributeArray
    static void collapse(AttributeArray* array, const ValueType& value);
    /// Non-member equivalent to fill() that static_casts array to this TypedAttributeArray
    static void fill(AttributeArray* array, const ValueType& value);

    /// Compress the attribute array.
    bool compress() override;
    /// Uncompress the attribute array.
    bool decompress() override;

    /// Read attribute data from a stream.
    void read(std::istream&) override;
    /// Write attribute data to a stream.
    /// @param os              the output stream
    /// @param outputTransient if true, write out transient attributes
    void write(std::ostream& os, bool outputTransient) const override;
    /// Write attribute data to a stream, don't write transient attributes.
    void write(std::ostream&) const override;

    /// Read attribute metadata from a stream.
    void readMetadata(std::istream&) override;
    /// Write attribute metadata to a stream.
    /// @param os              the output stream
    /// @param outputTransient if true, write out transient attributes
    /// @param paged           if true, data is written out in pages
    void writeMetadata(std::ostream& os, bool outputTransient, bool paged) const override;

    /// Read attribute buffers from a stream.
    void readBuffers(std::istream&) override;
    /// Write attribute buffers to a stream.
    /// @param os              the output stream
    /// @param outputTransient if true, write out transient attributes
    void writeBuffers(std::ostream& os, bool outputTransient) const override;

    /// Read attribute buffers from a paged stream.
    void readPagedBuffers(compression::PagedInputStream&) override;
    /// Write attribute buffers to a paged stream.
    /// @param os              the output stream
    /// @param outputTransient if true, write out transient attributes
    void writePagedBuffers(compression::PagedOutputStream& os, bool outputTransient) const override;

    /// Return @c true if this buffer's values have not yet been read from disk.
    inline bool isOutOfCore() const;

    /// Ensures all data is in-core
    void loadData() const override;

protected:
    AccessorBasePtr getAccessor() const override;

private:
    /// Load data from memory-mapped file.
    inline void doLoad() const;
    /// Load data from memory-mapped file (unsafe as this function is not protected by a mutex).
    /// @param compression if true, loading previously compressed data will re-compressed it
    inline void doLoadUnsafe(const bool compression = true) const;
    /// Compress in-core data assuming mutex is locked
    inline bool compressUnsafe();

    /// Toggle out-of-core state
    inline void setOutOfCore(const bool);

    /// Compare the this data to another attribute array. Used by the base class comparison operator
    bool isEqual(const AttributeArray& other) const override;

    size_t arrayMemUsage() const;
    void allocate();
    void deallocate();

    /// Helper function for use with registerType()
    static AttributeArray::Ptr factory(Index n, Index strideOrTotalSize, bool constantStride) {
        return TypedAttributeArray::create(n, strideOrTotalSize, constantStride);
    }

    static tbb::atomic<const NamePair*> sTypeName;
    std::unique_ptr<StorageType[]>      mData;
    Index                               mSize;
    Index                               mStrideOrTotalSize;
    bool                                mIsUniform = false;
    tbb::spin_mutex                     mMutex;
}; // class TypedAttributeArray


////////////////////////////////////////


/// AttributeHandles provide access to specific TypedAttributeArray methods without needing
/// to know the compression codec, however these methods also incur the cost of a function pointer
template <typename ValueType, typename CodecType = UnknownCodec>
class AttributeHandle
{
public:
    using Handle    = AttributeHandle<ValueType, CodecType>;
    using Ptr       = std::shared_ptr<Handle>;
    using UniquePtr = std::unique_ptr<Handle>;

protected:
    using GetterPtr = ValueType (*)(const AttributeArray* array, const Index n);
    using SetterPtr = void (*)(AttributeArray* array, const Index n, const ValueType& value);
    using ValuePtr  = void (*)(AttributeArray* array, const ValueType& value);

public:
    static Ptr create(const AttributeArray& array, const bool preserveCompression = true);

    AttributeHandle(const AttributeArray& array, const bool preserveCompression = true);

    AttributeHandle(const AttributeHandle&) = default;
    AttributeHandle& operator=(const AttributeHandle&) = default;

    virtual ~AttributeHandle();

    Index stride() const { return mStrideOrTotalSize; }
    Index size() const { return mSize; }

    bool isUniform() const;
    bool hasConstantStride() const;

    ValueType get(Index n, Index m = 0) const;

protected:
    Index index(Index n, Index m) const;

    const AttributeArray* mArray;

    GetterPtr mGetter;
    SetterPtr mSetter;
    ValuePtr  mCollapser;
    ValuePtr  mFiller;

private:
    friend class ::TestAttributeArray;

    template <bool IsUnknownCodec>
    typename std::enable_if<IsUnknownCodec, bool>::type compatibleType() const;

    template <bool IsUnknownCodec>
    typename std::enable_if<!IsUnknownCodec, bool>::type compatibleType() const;

    template <bool IsUnknownCodec>
    typename std::enable_if<IsUnknownCodec, ValueType>::type get(Index index) const;

    template <bool IsUnknownCodec>
    typename std::enable_if<!IsUnknownCodec, ValueType>::type get(Index index) const;

    // local copy of AttributeArray (to preserve compression)
    AttributeArray::Ptr mLocalArray;

    Index mStrideOrTotalSize;
    Index mSize;
    bool mCollapseOnDestruction;
}; // class AttributeHandle


////////////////////////////////////////


/// Write-able version of AttributeHandle
template <typename ValueType, typename CodecType = UnknownCodec>
class AttributeWriteHandle : public AttributeHandle<ValueType, CodecType>
{
public:
    using Handle    = AttributeWriteHandle<ValueType, CodecType>;
    using Ptr       = std::shared_ptr<Handle>;
    using ScopedPtr = std::unique_ptr<Handle>;

    static Ptr create(AttributeArray& array, const bool expand = true);

    AttributeWriteHandle(AttributeArray& array, const bool expand = true);

    virtual ~AttributeWriteHandle() = default;

    /// @brief  If this array is uniform, replace it with an array of length size().
    /// @param  fill if true, assign the uniform value to each element of the array.
    void expand(bool fill = true);

    /// Replace the existing array with a uniform value (zero if none provided).
    void collapse();
    void collapse(const ValueType& uniformValue);

    /// Compact the existing array to become uniform if all values are identical
    bool compact();

    /// @brief Fill the existing array with the given value.
    /// @note Identical to collapse() except a non-uniform array will not become uniform.
    void fill(const ValueType& value);

    void set(Index n, const ValueType& value);
    void set(Index n, Index m, const ValueType& value);

private:
    friend class ::TestAttributeArray;

    template <bool IsUnknownCodec>
    typename std::enable_if<IsUnknownCodec, void>::type set(Index index, const ValueType& value) const;

    template <bool IsUnknownCodec>
    typename std::enable_if<!IsUnknownCodec, void>::type set(Index index, const ValueType& value) const;
}; // class AttributeWriteHandle


////////////////////////////////////////


// Attribute codec implementation


template<typename ValueType>
inline void
NullCodec::decode(const ValueType& data, ValueType& val)
{
    val = data;
}


template<typename ValueType>
inline void
NullCodec::encode(const ValueType& val, ValueType& data)
{
    data = val;
}


template<typename StorageType, typename ValueType>
inline void
TruncateCodec::decode(const StorageType& data, ValueType& val)
{
    val = static_cast<ValueType>(data);
}


template<typename StorageType, typename ValueType>
inline void
TruncateCodec::encode(const ValueType& val, StorageType& data)
{
    data = static_cast<StorageType>(val);
}


template <bool OneByte, typename Range>
template<typename StorageType, typename ValueType>
inline void
FixedPointCodec<OneByte, Range>::decode(const StorageType& data, ValueType& val)
{
    val = fixedPointToFloatingPoint<ValueType>(data);

    // shift value range to be -0.5 => 0.5 (as this is most commonly used for position)

    val = Range::template decode<ValueType>(val);
}


template <bool OneByte, typename Range>
template<typename StorageType, typename ValueType>
inline void
FixedPointCodec<OneByte, Range>::encode(const ValueType& val, StorageType& data)
{
    // shift value range to be -0.5 => 0.5 (as this is most commonly used for position)

    const ValueType newVal = Range::template encode<ValueType>(val);

    data = floatingPointToFixedPoint<StorageType>(newVal);
}


template<typename T>
inline void
UnitVecCodec::decode(const StorageType& data, math::Vec3<T>& val)
{
    val = math::QuantizedUnitVec::unpack(data);
}


template<typename T>
inline void
UnitVecCodec::encode(const math::Vec3<T>& val, StorageType& data)
{
    data = math::QuantizedUnitVec::pack(val);
}


////////////////////////////////////////

// TypedAttributeArray implementation

template<typename ValueType_, typename Codec_>
tbb::atomic<const NamePair*> TypedAttributeArray<ValueType_, Codec_>::sTypeName;


template<typename ValueType_, typename Codec_>
TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray(
    Index n, Index strideOrTotalSize, bool constantStride, const ValueType& uniformValue)
    : mData(new StorageType[1])
    , mSize(n)
    , mStrideOrTotalSize(strideOrTotalSize)
    , mIsUniform(true)
{
    if (constantStride) {
        this->setConstantStride(true);
        if (strideOrTotalSize == 0) {
            OPENVDB_THROW(ValueError, "Creating a TypedAttributeArray with a constant stride requires that " \
                                        "stride to be at least one.")
        }
    }
    else {
        this->setConstantStride(false);
        if (mStrideOrTotalSize < n) {
            OPENVDB_THROW(ValueError, "Creating a TypedAttributeArray with a non-constant stride must have " \
                                        "a total size of at least the number of elements in the array.")
        }
    }
    mSize = std::max(Index(1), mSize);
    mStrideOrTotalSize = std::max(Index(1), mStrideOrTotalSize);
    Codec::encode(uniformValue, mData.get()[0]);
}


template<typename ValueType_, typename Codec_>
TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray(const TypedAttributeArray& rhs, bool uncompress)
    : AttributeArray(rhs)
    , mSize(rhs.mSize)
    , mStrideOrTotalSize(rhs.mStrideOrTotalSize)
    , mIsUniform(rhs.mIsUniform)
{
    // disable uncompress if data is not compressed

    if (!this->isCompressed())  uncompress = false;

    if (this->isOutOfCore()) {
        // do nothing
    } else if (mIsUniform) {
        this->allocate();
        mData.get()[0] = rhs.mData.get()[0];
    } else if (this->isCompressed()) {
        std::unique_ptr<char[]> buffer;
        if (uncompress) {
            const char* charBuffer = reinterpret_cast<const char*>(rhs.mData.get());
            size_t uncompressedBytes = compression::bloscUncompressedSize(charBuffer);
            buffer = compression::bloscDecompress(charBuffer, uncompressedBytes);
        }
        if (buffer) {
            mCompressedBytes = 0;
        } else {
            // decompression wasn't requested or failed so deep copy instead
            buffer.reset(new char[mCompressedBytes]);
            std::memcpy(buffer.get(), rhs.mData.get(), mCompressedBytes);
        }
        assert(buffer);
        mData.reset(reinterpret_cast<StorageType*>(buffer.release()));
    } else {
        this->allocate();
        std::memcpy(mData.get(), rhs.mData.get(), this->arrayMemUsage());
    }
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray&
TypedAttributeArray<ValueType_, Codec_>::operator=(const TypedAttributeArray& rhs)
{
    if (&rhs != this) {
        tbb::spin_mutex::scoped_lock lock(mMutex);

        this->deallocate();

        mFlags = rhs.mFlags;
        mSerializationFlags = rhs.mSerializationFlags;
        mCompressedBytes = rhs.mCompressedBytes;
        mSize = rhs.mSize;
        mStrideOrTotalSize = rhs.mStrideOrTotalSize;
        mIsUniform = rhs.mIsUniform;

        if (rhs.isOutOfCore()) {
            mPageHandle = rhs.mPageHandle;
        } else if (mIsUniform) {
            this->allocate();
            mData.get()[0] = rhs.mData.get()[0];
        } else if (this->isCompressed()) {
            std::unique_ptr<char[]> buffer(new char[mCompressedBytes]);
            std::memcpy(buffer.get(), rhs.mData.get(), mCompressedBytes);
            mData.reset(reinterpret_cast<StorageType*>(buffer.release()));
        } else {
            this->allocate();
            std::memcpy(mData.get(), rhs.mData.get(), arrayMemUsage());
        }
    }
}


template<typename ValueType_, typename Codec_>
inline const NamePair&
TypedAttributeArray<ValueType_, Codec_>::attributeType()
{
    if (sTypeName == nullptr) {
        NamePair* s = new NamePair(typeNameAsString<ValueType>(), Codec::name());
        if (sTypeName.compare_and_swap(s, nullptr) != nullptr) delete s;
    }
    return *sTypeName;
}


template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::isRegistered()
{
    return AttributeArray::isRegistered(TypedAttributeArray::attributeType());
}


template<typename ValueType_, typename Codec_>
inline void
TypedAttributeArray<ValueType_, Codec_>::registerType()
{
    AttributeArray::registerType(TypedAttributeArray::attributeType(), TypedAttributeArray::factory);
}


template<typename ValueType_, typename Codec_>
inline void
TypedAttributeArray<ValueType_, Codec_>::unregisterType()
{
    AttributeArray::unregisterType(TypedAttributeArray::attributeType());
}


template<typename ValueType_, typename Codec_>
inline typename TypedAttributeArray<ValueType_, Codec_>::Ptr
TypedAttributeArray<ValueType_, Codec_>::create(Index n, Index stride, bool constantStride)
{
    return Ptr(new TypedAttributeArray(n, stride, constantStride));
}

template<typename ValueType_, typename Codec_>
inline TypedAttributeArray<ValueType_, Codec_>&
TypedAttributeArray<ValueType_, Codec_>::cast(AttributeArray& attributeArray)
{
    if (!attributeArray.isType<TypedAttributeArray>()) {
        OPENVDB_THROW(TypeError, "Invalid Attribute Type");
    }
    return static_cast<TypedAttributeArray&>(attributeArray);
}

template<typename ValueType_, typename Codec_>
inline const TypedAttributeArray<ValueType_, Codec_>&
TypedAttributeArray<ValueType_, Codec_>::cast(const AttributeArray& attributeArray)
{
    if (!attributeArray.isType<TypedAttributeArray>()) {
        OPENVDB_THROW(TypeError, "Invalid Attribute Type");
    }
    return static_cast<const TypedAttributeArray&>(attributeArray);
}

template<typename ValueType_, typename Codec_>
AttributeArray::Ptr
TypedAttributeArray<ValueType_, Codec_>::copy() const
{
    return AttributeArray::Ptr(new TypedAttributeArray<ValueType, Codec>(*this));
}


template<typename ValueType_, typename Codec_>
AttributeArray::Ptr
TypedAttributeArray<ValueType_, Codec_>::copyUncompressed() const
{
    return AttributeArray::Ptr(new TypedAttributeArray<ValueType, Codec>(*this, /*decompress = */true));
}


template<typename ValueType_, typename Codec_>
size_t
TypedAttributeArray<ValueType_, Codec_>::arrayMemUsage() const
{
    if (this->isOutOfCore())        return 0;
    if (this->isCompressed())       return mCompressedBytes;

    return (mIsUniform ? 1 : this->dataSize()) * sizeof(StorageType);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::allocate()
{
    assert(!mData);
    if (mIsUniform) {
        mData.reset(new StorageType[1]);
    }
    else {
        const size_t size(this->dataSize());
        assert(size > 0);
        mData.reset(new StorageType[size]);
    }
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::deallocate()
{
    // detach from file if delay-loaded
    if (this->isOutOfCore()) {
        this->setOutOfCore(false);
        this->mPageHandle.reset();
    }
    if (mData)      mData.reset();
}


template<typename ValueType_, typename Codec_>
size_t
TypedAttributeArray<ValueType_, Codec_>::memUsage() const
{
    return sizeof(*this) + (bool(mData) ? this->arrayMemUsage() : 0);
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::ValueType
TypedAttributeArray<ValueType_, Codec_>::getUnsafe(Index n) const
{
    assert(n < this->dataSize());
    assert(!this->isOutOfCore());
    assert(!this->isCompressed());

    ValueType val;
    Codec::decode(/*in=*/mData.get()[mIsUniform ? 0 : n], /*out=*/val);
    return val;
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::ValueType
TypedAttributeArray<ValueType_, Codec_>::get(Index n) const
{
    if (n >= this->dataSize())           OPENVDB_THROW(IndexError, "Out-of-range access.");
    if (this->isOutOfCore())            this->doLoad();
    if (this->isCompressed())           const_cast<TypedAttributeArray*>(this)->decompress();

    return this->getUnsafe(n);
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::getUnsafe(Index n, T& val) const
{
    val = static_cast<T>(this->getUnsafe(n));
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::get(Index n, T& val) const
{
    val = static_cast<T>(this->get(n));
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::ValueType
TypedAttributeArray<ValueType_, Codec_>::getUnsafe(const AttributeArray* array, const Index n)
{
    return static_cast<const TypedAttributeArray<ValueType, Codec>*>(array)->getUnsafe(n);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::setUnsafe(Index n, const ValueType& val)
{
    assert(n < this->dataSize());
    assert(!this->isOutOfCore());
    assert(!this->isCompressed());
    assert(!this->isUniform());

    // this unsafe method assumes the data is not uniform, however if it is, this redirects the index
    // to zero, which is marginally less efficient but ensures not writing to an illegal address

    Codec::encode(/*in=*/val, /*out=*/mData.get()[mIsUniform ? 0 : n]);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::set(Index n, const ValueType& val)
{
    if (n >= this->dataSize())           OPENVDB_THROW(IndexError, "Out-of-range access.");
    if (this->isOutOfCore())            this->doLoad();
    if (this->isCompressed())           this->decompress();
    if (this->isUniform())              this->expand();

    this->setUnsafe(n, val);
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::setUnsafe(Index n, const T& val)
{
    this->setUnsafe(n, static_cast<ValueType>(val));
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::set(Index n, const T& val)
{
    this->set(n, static_cast<ValueType>(val));
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::setUnsafe(AttributeArray* array, const Index n, const ValueType& value)
{
    static_cast<TypedAttributeArray<ValueType, Codec>*>(array)->setUnsafe(n, value);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::set(Index n, const AttributeArray& sourceArray, const Index sourceIndex)
{
    const TypedAttributeArray& sourceTypedArray = static_cast<const TypedAttributeArray&>(sourceArray);

    ValueType sourceValue;
    sourceTypedArray.get(sourceIndex, sourceValue);

    this->set(n, sourceValue);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::expand(bool fill)
{
    if (!mIsUniform)    return;

    const StorageType val = mData.get()[0];

    {
        tbb::spin_mutex::scoped_lock lock(mMutex);
        this->deallocate();
        mIsUniform = false;
        this->allocate();
    }

    mCompressedBytes = 0;

    if (fill) {
        for (Index i = 0; i < this->dataSize(); ++i)  mData.get()[i] = val;
    }
}


template<typename ValueType_, typename Codec_>
bool
TypedAttributeArray<ValueType_, Codec_>::compact()
{
    if (mIsUniform)     return true;

    // compaction is not possible if any values are different
    const ValueType_ val = this->get(0);
    for (Index i = 1; i < this->dataSize(); i++) {
        if (!math::isExactlyEqual(this->get(i), val)) return false;
    }

    this->collapse(this->get(0));
    return true;
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::collapse()
{
    this->collapse(zeroVal<ValueType>());
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::collapse(const ValueType& uniformValue)
{
    if (!mIsUniform) {
        tbb::spin_mutex::scoped_lock lock(mMutex);
        this->deallocate();
        mIsUniform = true;
        this->allocate();
    }
    Codec::encode(uniformValue, mData.get()[0]);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::collapse(AttributeArray* array, const ValueType& value)
{
    static_cast<TypedAttributeArray<ValueType, Codec>*>(array)->collapse(value);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::fill(const ValueType& value)
{
    if (this->isOutOfCore()) {
        tbb::spin_mutex::scoped_lock lock(mMutex);
        this->deallocate();
        this->allocate();
    }

    const Index size = mIsUniform ? 1 : this->dataSize();
    for (Index i = 0; i < size; ++i)  {
        Codec::encode(value, mData.get()[i]);
    }
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::fill(AttributeArray* array, const ValueType& value)
{
    static_cast<TypedAttributeArray<ValueType, Codec>*>(array)->fill(value);
}


template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::compress()
{
    if (!compression::bloscCanCompress())     return false;

    if (!mIsUniform && !this->isCompressed()) {

        tbb::spin_mutex::scoped_lock lock(mMutex);

        this->doLoadUnsafe(/*compression=*/false);

        if (this->isCompressed())   return true;

        return this->compressUnsafe();
    }

    return false;
}


template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::compressUnsafe()
{
    if (!compression::bloscCanCompress())     return false;
    if (mIsUniform)                           return false;

    // assumes mutex is locked and data is not out-of-core

    const bool writeCompress = (mSerializationFlags & WRITEMEMCOMPRESS);
    const size_t inBytes = writeCompress ? mCompressedBytes : this->arrayMemUsage();

    if (inBytes > 0) {
        size_t outBytes;
        const char* charBuffer = reinterpret_cast<const char*>(mData.get());
        std::unique_ptr<char[]> buffer = compression::bloscCompress(charBuffer, inBytes, outBytes);
        if (buffer) {
            mData.reset(reinterpret_cast<StorageType*>(buffer.release()));
            mCompressedBytes = outBytes;
            return true;
        }
    }

    return false;
}


template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::decompress()
{
    tbb::spin_mutex::scoped_lock lock(mMutex);

    const bool writeCompress = (mSerializationFlags & WRITEMEMCOMPRESS);

    if (writeCompress) {
        this->doLoadUnsafe(/*compression=*/false);
        return true;
    }

    if (this->isCompressed()) {
        this->doLoadUnsafe();
        const char* charBuffer = reinterpret_cast<const char*>(this->mData.get());
        size_t uncompressedBytes = compression::bloscUncompressedSize(charBuffer);
        std::unique_ptr<char[]> buffer = compression::bloscDecompress(charBuffer, uncompressedBytes);
        if (buffer) {
            mData.reset(reinterpret_cast<StorageType*>(buffer.release()));
            mCompressedBytes = 0;
            return true;
        }
    }

    return false;
}


template<typename ValueType_, typename Codec_>
bool
TypedAttributeArray<ValueType_, Codec_>::isOutOfCore() const
{
#if OPENVDB_ABI_VERSION_NUMBER >= 5
    return mOutOfCore;
#else
    return (mFlags & OUTOFCORE);
#endif
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::setOutOfCore(const bool b)
{
#if OPENVDB_ABI_VERSION_NUMBER >= 5
    mOutOfCore = b;
#else
    if (b) mFlags = static_cast<uint8_t>(mFlags | OUTOFCORE);
    else   mFlags = static_cast<uint8_t>(mFlags & ~OUTOFCORE);
#endif
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::doLoad() const
{
    if (!(this->isOutOfCore()))     return;

    TypedAttributeArray<ValueType_, Codec_>* self =
        const_cast<TypedAttributeArray<ValueType_, Codec_>*>(this);

    // This lock will be contended at most once, after which this buffer
    // will no longer be out-of-core.
    tbb::spin_mutex::scoped_lock lock(self->mMutex);
    this->doLoadUnsafe();
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::loadData() const
{
    this->doLoad();
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::read(std::istream& is)
{
    this->readMetadata(is);
    this->readBuffers(is);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::readMetadata(std::istream& is)
{
    // read data

    Index64 bytes = Index64(0);
    is.read(reinterpret_cast<char*>(&bytes), sizeof(Index64));
    bytes = bytes - /*flags*/sizeof(Int16) - /*size*/sizeof(Index);

    uint8_t flags = uint8_t(0);
    is.read(reinterpret_cast<char*>(&flags), sizeof(uint8_t));
    mFlags = flags;

    uint8_t serializationFlags = uint8_t(0);
    is.read(reinterpret_cast<char*>(&serializationFlags), sizeof(uint8_t));
    mSerializationFlags = serializationFlags;

    Index size = Index(0);
    is.read(reinterpret_cast<char*>(&size), sizeof(Index));
    mSize = size;

    // warn if an unknown flag has been set
    if (mFlags >= 0x20) {
        OPENVDB_LOG_WARN("Unknown attribute flags for VDB file format.");
    }
    // error if an unknown serialization flag has been set,
    // as this will adjust the layout of the data and corrupt the ability to read
    if (mSerializationFlags >= 0x10) {
        OPENVDB_THROW(IoError, "Unknown attribute serialization flags for VDB file format.");
    }

    // read uniform and compressed state

    mIsUniform = mSerializationFlags & WRITEUNIFORM;
    mCompressedBytes = bytes;

    // read strided value (set to 1 if array is not strided)

    if (mSerializationFlags & WRITESTRIDED) {
        Index stride = Index(0);
        is.read(reinterpret_cast<char*>(&stride), sizeof(Index));
        mStrideOrTotalSize = stride;
    }
    else {
        mStrideOrTotalSize = 1;
    }
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::readBuffers(std::istream& is)
{
    if ((mSerializationFlags & WRITEPAGED)) {
        // use readBuffers(PagedInputStream&) for paged buffers
        OPENVDB_THROW(IoError, "Cannot read paged AttributeArray buffers.");
    }

    tbb::spin_mutex::scoped_lock lock(mMutex);

    this->deallocate();

    uint8_t bloscCompressed(0);
    if (!mIsUniform)    is.read(reinterpret_cast<char*>(&bloscCompressed), sizeof(uint8_t));

    std::unique_ptr<char[]> buffer(new char[mCompressedBytes]);
    is.read(buffer.get(), mCompressedBytes);

    if (mIsUniform) {
        // zero compressed bytes as uniform values are not compressed in memory
        mCompressedBytes = Index64(0);
    }
    else if (!(mSerializationFlags & WRITEMEMCOMPRESS)) {
        // zero compressed bytes if not compressed in memory
        mCompressedBytes = Index64(0);
    }

    // compressed on-disk

    if (bloscCompressed == uint8_t(1)) {

        // decompress buffer

        const size_t inBytes = this->dataSize() * sizeof(StorageType);
        std::unique_ptr<char[]> newBuffer = compression::bloscDecompress(buffer.get(), inBytes);
        if (newBuffer)  buffer.reset(newBuffer.release());
    }

    // set data to buffer

    mData.reset(reinterpret_cast<StorageType*>(buffer.release()));

    // clear all write flags

    if (mIsUniform)     mSerializationFlags &= uint8_t(~WRITEUNIFORM & ~WRITEMEMCOMPRESS & ~WRITEPAGED);
    else                mSerializationFlags &= uint8_t(~WRITEUNIFORM & ~WRITEPAGED);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::readPagedBuffers(compression::PagedInputStream& is)
{
    if (!(mSerializationFlags & WRITEPAGED)) {
        if (!is.sizeOnly()) this->readBuffers(is.getInputStream());
        return;
    }

    // If this array is being read from a memory-mapped file, delay loading of its data
    // until the data is actually accessed.
    io::MappedFile::Ptr mappedFile = io::getMappedFilePtr(is.getInputStream());
    const bool delayLoad = (mappedFile.get() != nullptr);

    if (is.sizeOnly())
    {
        mPageHandle = is.createHandle(mCompressedBytes);
        return;
    }

    assert(mPageHandle);

    tbb::spin_mutex::scoped_lock lock(mMutex);

    this->deallocate();

    this->setOutOfCore(delayLoad);
    is.read(mPageHandle, mCompressedBytes, delayLoad);

    if (!delayLoad) {
        std::unique_ptr<char[]> buffer = mPageHandle->read();
        mData.reset(reinterpret_cast<StorageType*>(buffer.release()));
    }

    // zero compressed bytes as not compressed in memory

    if (mIsUniform) {
        // zero compressed bytes as uniform values are not compressed in memory
        mCompressedBytes = Index64(0);
    }
    else if (!(mSerializationFlags & WRITEMEMCOMPRESS)) {
        mCompressedBytes = Index64(0);
    }

    // clear all write flags

    if (mIsUniform)     mSerializationFlags &= uint8_t(~WRITEUNIFORM & ~WRITEMEMCOMPRESS & ~WRITEPAGED);
    else                mSerializationFlags &= uint8_t(~WRITEUNIFORM & ~WRITEPAGED);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::write(std::ostream& os) const
{
    this->write(os, /*outputTransient=*/false);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::write(std::ostream& os, bool outputTransient) const
{
    this->writeMetadata(os, outputTransient, /*paged=*/false);
    this->writeBuffers(os, outputTransient);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::writeMetadata(std::ostream& os, bool outputTransient, bool paged) const
{
    if (!outputTransient && this->isTransient())    return;

#if OPENVDB_ABI_VERSION_NUMBER >= 5
    uint8_t flags(mFlags);
#else
    uint8_t flags(mFlags & uint8_t(~OUTOFCORE));
#endif
    uint8_t serializationFlags(0);
    Index size(mSize);
    Index stride(mStrideOrTotalSize);
    bool strideOfOne(this->stride() == 1);

    bool bloscCompression = io::getDataCompression(os) & io::COMPRESS_BLOSC;

    // any compressed data needs to be loaded if out-of-core
    if (bloscCompression || this->isCompressed())    this->doLoad();

    size_t compressedBytes = 0;

    if (!strideOfOne)
    {
        serializationFlags |= WRITESTRIDED;
    }

    if (mIsUniform)
    {
        serializationFlags |= WRITEUNIFORM;
        if (bloscCompression && paged)      serializationFlags |= WRITEPAGED;
    }
    else if (bloscCompression && paged)
    {
        serializationFlags |= WRITEPAGED;
        if (this->isCompressed()) {
            serializationFlags |= WRITEMEMCOMPRESS;
            const char* charBuffer = reinterpret_cast<const char*>(mData.get());
            compressedBytes = compression::bloscUncompressedSize(charBuffer);
        }
    }
    else if (this->isCompressed())
    {
        serializationFlags |= WRITEMEMCOMPRESS;
        compressedBytes = mCompressedBytes;
    }
    else if (bloscCompression)
    {
        const char* charBuffer = reinterpret_cast<const char*>(mData.get());
        const size_t inBytes = this->arrayMemUsage();
        compressedBytes = compression::bloscCompressedSize(charBuffer, inBytes);
    }

    Index64 bytes = /*flags*/ sizeof(Int16) + /*size*/ sizeof(Index);

    bytes += (compressedBytes > 0) ? compressedBytes : this->arrayMemUsage();

    // write data

    os.write(reinterpret_cast<const char*>(&bytes), sizeof(Index64));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(uint8_t));
    os.write(reinterpret_cast<const char*>(&serializationFlags), sizeof(uint8_t));
    os.write(reinterpret_cast<const char*>(&size), sizeof(Index));

    // write strided
    if (!strideOfOne)       os.write(reinterpret_cast<const char*>(&stride), sizeof(Index));
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::writeBuffers(std::ostream& os, bool outputTransient) const
{
    if (!outputTransient && this->isTransient())    return;

    this->doLoad();

    if (this->isUniform()) {
        os.write(reinterpret_cast<const char*>(mData.get()), sizeof(StorageType));
    }
    else if (this->isCompressed())
    {
        uint8_t bloscCompressed(0);
        os.write(reinterpret_cast<const char*>(&bloscCompressed), sizeof(uint8_t));
        os.write(reinterpret_cast<const char*>(mData.get()), mCompressedBytes);
    }
    else if (io::getDataCompression(os) & io::COMPRESS_BLOSC)
    {
        std::unique_ptr<char[]> compressedBuffer;
        size_t compressedBytes = 0;
        const char* charBuffer = reinterpret_cast<const char*>(mData.get());
        const size_t inBytes = this->arrayMemUsage();
        compressedBuffer = compression::bloscCompress(charBuffer, inBytes, compressedBytes);
        if (compressedBuffer) {
            uint8_t bloscCompressed(1);
            os.write(reinterpret_cast<const char*>(&bloscCompressed), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(compressedBuffer.get()), compressedBytes);
        }
        else {
            uint8_t bloscCompressed(0);
            os.write(reinterpret_cast<const char*>(&bloscCompressed), sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(mData.get()), inBytes);
        }
    }
    else
    {
        uint8_t bloscCompressed(0);
        os.write(reinterpret_cast<const char*>(&bloscCompressed), sizeof(uint8_t));
        os.write(reinterpret_cast<const char*>(mData.get()), this->arrayMemUsage());
    }
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::writePagedBuffers(compression::PagedOutputStream& os, bool outputTransient) const
{
    if (!outputTransient && this->isTransient())    return;

    // paged compression only available when Blosc is enabled
    bool bloscCompression = io::getDataCompression(os.getOutputStream()) & io::COMPRESS_BLOSC;
    if (!bloscCompression) {
        if (!os.sizeOnly())   this->writeBuffers(os.getOutputStream(), outputTransient);
        return;
    }

    this->doLoad();

    const char* buffer;
    size_t bytes;

    std::unique_ptr<char[]> uncompressedBuffer;
    if (this->isCompressed()) {
        // paged streams require uncompressed buffers, so locally decompress

        const char* charBuffer = reinterpret_cast<const char*>(this->mData.get());
        bytes = compression::bloscUncompressedSize(charBuffer);
        uncompressedBuffer = compression::bloscDecompress(charBuffer, bytes);
        buffer = reinterpret_cast<const char*>(uncompressedBuffer.get());
    }
    else {
        buffer = reinterpret_cast<const char*>(mData.get());
        bytes = this->arrayMemUsage();
    }

    os.write(buffer, bytes);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::doLoadUnsafe(const bool compression) const
{
    if (!(this->isOutOfCore()))     return;

    // this function expects the mutex to already be locked

    TypedAttributeArray<ValueType_, Codec_>* self = const_cast<TypedAttributeArray<ValueType_, Codec_>*>(this);

    assert(self->mPageHandle);

    std::unique_ptr<char[]> buffer = self->mPageHandle->read();

    self->mData.reset(reinterpret_cast<StorageType*>(buffer.release()));

    self->mPageHandle.reset();

    // if data was compressed prior to being written to disk, re-compress

    if (self->mSerializationFlags & WRITEMEMCOMPRESS) {
        if (compression)    self->compressUnsafe();
        else                self->mCompressedBytes = 0;
    }

    // clear all write and out-of-core flags

#if OPENVDB_ABI_VERSION_NUMBER >= 5
    self->mOutOfCore = false;
#else
    self->mFlags &= uint8_t(~OUTOFCORE);
#endif
    self->mSerializationFlags &= uint8_t(~WRITEUNIFORM & ~WRITEMEMCOMPRESS & ~WRITEPAGED);
}


template<typename ValueType_, typename Codec_>
AttributeArray::AccessorBasePtr
TypedAttributeArray<ValueType_, Codec_>::getAccessor() const
{
    // use the faster 'unsafe' get and set methods as attribute handles
    // ensure data is uncompressed and in-core when constructed

    return AccessorBasePtr(new AttributeArray::Accessor<ValueType_>(
        &TypedAttributeArray<ValueType_, Codec_>::getUnsafe,
        &TypedAttributeArray<ValueType_, Codec_>::setUnsafe,
        &TypedAttributeArray<ValueType_, Codec_>::collapse,
        &TypedAttributeArray<ValueType_, Codec_>::fill));
}


template<typename ValueType_, typename Codec_>
bool
TypedAttributeArray<ValueType_, Codec_>::isEqual(const AttributeArray& other) const
{
    const TypedAttributeArray<ValueType_, Codec_>* const otherT = dynamic_cast<const TypedAttributeArray<ValueType_, Codec_>* >(&other);
    if(!otherT) return false;
    if(this->mSize != otherT->mSize ||
       this->mStrideOrTotalSize != otherT->mStrideOrTotalSize ||
       this->mIsUniform != otherT->mIsUniform ||
       *this->sTypeName != *otherT->sTypeName) return false;

    this->doLoad();
    otherT->doLoad();

    const StorageType *target = this->mData.get(), *source = otherT->mData.get();
    if (!target && !source) return true;
    if (!target || !source) return false;
    Index n = this->mIsUniform ? 1 : mSize;
    while (n && math::isExactlyEqual(*target++, *source++)) --n;
    return n == 0;
}

////////////////////////////////////////


/// Accessor to call unsafe get and set methods based on templated Codec and Value
template <typename CodecType, typename ValueType>
struct AccessorEval
{
    using GetterPtr = ValueType (*)(const AttributeArray* array, const Index n);
    using SetterPtr = void (*)(AttributeArray* array, const Index n, const ValueType& value);

    /// Getter that calls to TypedAttributeArray::getUnsafe()
    /// @note Functor argument is provided but not required for the generic case
    static ValueType get(GetterPtr /*functor*/, const AttributeArray* array, const Index n) {
        return TypedAttributeArray<ValueType, CodecType>::getUnsafe(array, n);
    }

    /// Getter that calls to TypedAttributeArray::setUnsafe()
    /// @note Functor argument is provided but not required for the generic case
    static void set(SetterPtr /*functor*/, AttributeArray* array, const Index n, const ValueType& value) {
        TypedAttributeArray<ValueType, CodecType>::setUnsafe(array, n, value);
    }
};


/// Partial specialization when Codec is not known at compile-time to use the supplied functor instead
template <typename ValueType>
struct AccessorEval<UnknownCodec, ValueType>
{
    using GetterPtr = ValueType (*)(const AttributeArray* array, const Index n);
    using SetterPtr = void (*)(AttributeArray* array, const Index n, const ValueType& value);

    /// Getter that calls the supplied functor
    static ValueType get(GetterPtr functor, const AttributeArray* array, const Index n) {
        return (*functor)(array, n);
    }

    /// Setter that calls the supplied functor
    static void set(SetterPtr functor, AttributeArray* array, const Index n, const ValueType& value) {
        (*functor)(array, n, value);
    }
};


////////////////////////////////////////

// AttributeHandle implementation

template <typename ValueType, typename CodecType>
typename AttributeHandle<ValueType, CodecType>::Ptr
AttributeHandle<ValueType, CodecType>::create(const AttributeArray& array, const bool preserveCompression)
{
    return  typename AttributeHandle<ValueType, CodecType>::Ptr(
            new AttributeHandle<ValueType, CodecType>(array, preserveCompression));
}

template <typename ValueType, typename CodecType>
AttributeHandle<ValueType, CodecType>::AttributeHandle(const AttributeArray& array, const bool preserveCompression)
    : mArray(&array)
    , mStrideOrTotalSize(array.hasConstantStride() ? array.stride() : 1)
    , mSize(array.hasConstantStride() ? array.size() : array.dataSize())
    , mCollapseOnDestruction(preserveCompression && array.isStreaming())
{
    if (!this->compatibleType<std::is_same<CodecType, UnknownCodec>::value>()) {
        OPENVDB_THROW(TypeError, "Cannot bind handle due to incompatible type of AttributeArray.");
    }

    // load data if delay-loaded

    mArray->loadData();

    // if array is compressed and preserve compression is true, copy and decompress
    // into a local copy that is destroyed with handle to maintain thread-safety

    if (array.isCompressed())
    {
        if (preserveCompression && !array.isStreaming()) {
            mLocalArray = array.copyUncompressed();
            mLocalArray->decompress();
            mArray = mLocalArray.get();
        }
        else {
            const_cast<AttributeArray*>(mArray)->decompress();
        }
    }

    // bind getter and setter methods

    AttributeArray::AccessorBasePtr accessor = mArray->getAccessor();
    assert(accessor);

    AttributeArray::Accessor<ValueType>* typedAccessor = static_cast<AttributeArray::Accessor<ValueType>*>(accessor.get());

    mGetter = typedAccessor->mGetter;
    mSetter = typedAccessor->mSetter;
    mCollapser = typedAccessor->mCollapser;
    mFiller = typedAccessor->mFiller;
}

template <typename ValueType, typename CodecType>
AttributeHandle<ValueType, CodecType>::~AttributeHandle()
{
    // if enabled, attribute is collapsed on destruction of the handle to save memory
    if (mCollapseOnDestruction)  const_cast<AttributeArray*>(this->mArray)->collapse();
}

template <typename ValueType, typename CodecType>
template <bool IsUnknownCodec>
typename std::enable_if<IsUnknownCodec, bool>::type
AttributeHandle<ValueType, CodecType>::compatibleType() const
{
    // if codec is unknown, just check the value type

    return mArray->hasValueType<ValueType>();
}

template <typename ValueType, typename CodecType>
template <bool IsUnknownCodec>
typename std::enable_if<!IsUnknownCodec, bool>::type
AttributeHandle<ValueType, CodecType>::compatibleType() const
{
    // if the codec is known, check the value type and codec

    return mArray->isType<TypedAttributeArray<ValueType, CodecType>>();
}

template <typename ValueType, typename CodecType>
Index AttributeHandle<ValueType, CodecType>::index(Index n, Index m) const
{
    Index index = n * mStrideOrTotalSize + m;
    assert(index < (mSize * mStrideOrTotalSize));
    return index;
}

template <typename ValueType, typename CodecType>
ValueType AttributeHandle<ValueType, CodecType>::get(Index n, Index m) const
{
    return this->get<std::is_same<CodecType, UnknownCodec>::value>(this->index(n, m));
}

template <typename ValueType, typename CodecType>
template <bool IsUnknownCodec>
typename std::enable_if<IsUnknownCodec, ValueType>::type
AttributeHandle<ValueType, CodecType>::get(Index index) const
{
    // if the codec is unknown, use the getter functor

    return (*mGetter)(mArray, index);
}

template <typename ValueType, typename CodecType>
template <bool IsUnknownCodec>
typename std::enable_if<!IsUnknownCodec, ValueType>::type
AttributeHandle<ValueType, CodecType>::get(Index index) const
{
    // if the codec is known, call the method on the attribute array directly

    return TypedAttributeArray<ValueType, CodecType>::getUnsafe(mArray, index);
}

template <typename ValueType, typename CodecType>
bool AttributeHandle<ValueType, CodecType>::isUniform() const
{
    return mArray->isUniform();
}

template <typename ValueType, typename CodecType>
bool AttributeHandle<ValueType, CodecType>::hasConstantStride() const
{
    return mArray->hasConstantStride();
}

////////////////////////////////////////

// AttributeWriteHandle implementation

template <typename ValueType, typename CodecType>
typename AttributeWriteHandle<ValueType, CodecType>::Ptr
AttributeWriteHandle<ValueType, CodecType>::create(AttributeArray& array, const bool expand)
{
    return  typename AttributeWriteHandle<ValueType, CodecType>::Ptr(
            new AttributeWriteHandle<ValueType, CodecType>(array, expand));
}

template <typename ValueType, typename CodecType>
AttributeWriteHandle<ValueType, CodecType>::AttributeWriteHandle(AttributeArray& array, const bool expand)
    : AttributeHandle<ValueType, CodecType>(array, /*preserveCompression = */ false)
{
    if (expand)     array.expand();
}

template <typename ValueType, typename CodecType>
void AttributeWriteHandle<ValueType, CodecType>::set(Index n, const ValueType& value)
{
    this->set<std::is_same<CodecType, UnknownCodec>::value>(this->index(n, 0), value);
}

template <typename ValueType, typename CodecType>
void AttributeWriteHandle<ValueType, CodecType>::set(Index n, Index m, const ValueType& value)
{
    this->set<std::is_same<CodecType, UnknownCodec>::value>(this->index(n, m), value);
}

template <typename ValueType, typename CodecType>
void AttributeWriteHandle<ValueType, CodecType>::expand(const bool fill)
{
    const_cast<AttributeArray*>(this->mArray)->expand(fill);
}

template <typename ValueType, typename CodecType>
void AttributeWriteHandle<ValueType, CodecType>::collapse()
{
    const_cast<AttributeArray*>(this->mArray)->collapse();
}

template <typename ValueType, typename CodecType>
bool AttributeWriteHandle<ValueType, CodecType>::compact()
{
    return const_cast<AttributeArray*>(this->mArray)->compact();
}

template <typename ValueType, typename CodecType>
void AttributeWriteHandle<ValueType, CodecType>::collapse(const ValueType& uniformValue)
{
    this->mCollapser(const_cast<AttributeArray*>(this->mArray), uniformValue);
}

template <typename ValueType, typename CodecType>
void AttributeWriteHandle<ValueType, CodecType>::fill(const ValueType& value)
{
    this->mFiller(const_cast<AttributeArray*>(this->mArray), value);
}

template <typename ValueType, typename CodecType>
template <bool IsUnknownCodec>
typename std::enable_if<IsUnknownCodec, void>::type
AttributeWriteHandle<ValueType, CodecType>::set(Index index, const ValueType& value) const
{
    // if the codec is unknown, use the setter functor

    (*this->mSetter)(const_cast<AttributeArray*>(this->mArray), index, value);
}

template <typename ValueType, typename CodecType>
template <bool IsUnknownCodec>
typename std::enable_if<!IsUnknownCodec, void>::type
AttributeWriteHandle<ValueType, CodecType>::set(Index index, const ValueType& value) const
{
    // if the codec is known, call the method on the attribute array directly

    TypedAttributeArray<ValueType, CodecType>::setUnsafe(const_cast<AttributeArray*>(this->mArray), index, value);
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
