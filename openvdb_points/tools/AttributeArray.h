///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
//
/// @file AttributeArray.h
///
/// @authors Dan Bailey, Mihai Alden, Peter Cucka
///
/// @brief  Attribute Array storage templated on type and compression codec.
///


#ifndef OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED

#include <openvdb_points/Types.h>
#include <openvdb/math/QuantizedUnitVec.h>
#include <openvdb/util/Name.h>
#include <openvdb/util/logging.h>
#include <openvdb/io/io.h> // MappedFile
#include <openvdb/io/Compression.h> // COMPRESS_BLOSC

#include <openvdb_points/tools/IndexIterator.h>

#include <tbb/spin_mutex.h>
#include <tbb/atomic.h>

#include <boost/scoped_array.hpp>

#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {


// Add new typedef for a Name pair
typedef std::pair<Name, Name> NamePair;

namespace tools {


////////////////////////////////////////

// Attribute Compression methods


namespace attribute_compression {

/// @brief Returns true if compression is available
bool canCompress();

/// @brief Retrieves the uncompressed size of buffer when uncompressed
///
/// @param buffer the compressed buffer
int uncompressedSize(const char* buffer);

/// @brief Retrieves the compressed size of buffer when compressed
///
/// @param buffer the uncompressed buffer
/// @param typeSize the size of the data type
/// @param uncompressedBytes number of uncompressed bytes
int compressedSize(const char* buffer, const size_t typeSize, const int uncompressedBytes);

/// @brief Compress and return the compressed buffer.
///
/// @param buffer the buffer to compress
/// @param typeSize the size of the data type
/// @param uncompressedBytes number of uncompressed bytes
/// @param compressedBytes number of compressed bytes (written to this variable)
/// @param cleanup if true, the supplied buffer will be deleted prior to allocating new memory
char* compress( char* buffer, const size_t typeSize,
                const int uncompressedBytes, int& compressedBytes,
                const bool cleanup = false);

/// @brief Compress and return the compressed buffer.
///
/// @param buffer the buffer to compress
/// @param typeSize the size of the data type
/// @param uncompressedBytes number of uncompressed bytes
/// @param compressedBytes number of compressed bytes (written to this variable)
///
/// @note Unlike the non-const buffer version, the buffer will never be deleted.
char* compress( const char* buffer, const size_t typeSize,
                const int uncompressedBytes, int& compressedBytes);

/// @brief Decompress and return the uncompressed buffer.
///
/// @param buffer the buffer to decompress
/// @param expectedBytes the number of bytes expected once the buffer is decompressed
/// @param cleanup if true, the supplied buffer will be deleted prior to allocating new memory
char* decompress(char* buffer, const int expectedBytes, const bool cleanup = false);

/// @brief Decompress and return the uncompressed buffer.
///
/// @param buffer the buffer to decompress
/// @param expectedBytes the number of bytes expected once the buffer is decompressed
///
/// @note Unlike the non-const buffer version, the buffer will never be deleted.
char* decompress(const char* buffer, const int expectedBytes);

} // namespace attribute_compression


////////////////////////////////////////

// Utility methods

template <typename IntegerT, typename FloatT>
inline IntegerT
floatingPointToFixedPoint(const FloatT s)
{
    BOOST_STATIC_ASSERT(boost::is_unsigned<IntegerT>::value);
    if (FloatT(0.0) > s) return std::numeric_limits<IntegerT>::min();
    else if (FloatT(1.0) <= s) return std::numeric_limits<IntegerT>::max();
    return IntegerT(std::floor(s * FloatT(std::numeric_limits<IntegerT>::max())));
}


template <typename FloatT, typename IntegerT>
inline FloatT
fixedPointToFloatingPoint(const IntegerT s)
{
    BOOST_STATIC_ASSERT(boost::is_unsigned<IntegerT>::value);
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

// Attribute codec schemes

template<typename StorageType_>
struct NullAttributeCodec
{
    typedef StorageType_ StorageType;
    template<typename ValueType> static void decode(const StorageType&, ValueType&);
    template<typename ValueType> static void encode(const StorageType&, ValueType&);
    static const char* name() { return "null"; }
};


template<typename IntType>
struct FixedPointAttributeCodec
{
    typedef IntType StorageType;
    template<typename ValueType> static void decode(const StorageType&, ValueType&);
    template<typename ValueType> static void encode(const ValueType&, StorageType&);
    static const char* name() { return "fxpt"; }
};


struct UnitVecAttributeCodec
{
    typedef uint16_t StorageType;
    template<typename T> static void decode(const StorageType&, math::Vec3<T>&);
    template<typename T> static void encode(const math::Vec3<T>&, StorageType&);
    static const char* name() { return "uvec"; }
};


////////////////////////////////////////


/// Base class for storing attribute data
class AttributeArray
{
protected:
    struct AccessorBase;
    template <typename T> struct Accessor;

    typedef boost::shared_ptr<AccessorBase>             AccessorBasePtr;

public:
    enum Flag { TRANSIENT = 0x1, HIDDEN = 0x2, GROUP=0x4, WRITEUNIFORM=0x8,
                WRITEMEMCOMPRESS=0x10, WRITEDISKCOMPRESS=0x20, OUTOFCORE=0x40 };

#ifndef OPENVDB_2_ABI_COMPATIBLE
    struct FileInfo
    {
        FileInfo(): bufpos(0), bytes(0) {}
        std::streamoff bufpos;
        Index64 bytes;
        io::MappedFile::Ptr mapping;
        boost::shared_ptr<io::StreamMetadata> meta;
    };
#endif

    typedef boost::shared_ptr<AttributeArray>           Ptr;
    typedef boost::shared_ptr<const AttributeArray>     ConstPtr;

    template <typename> friend class AttributeHandle;

    typedef Ptr (*FactoryMethod)(size_t);

    AttributeArray() : mCompressedBytes(0), mFlags(0) {}
    virtual ~AttributeArray() {}

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const = 0;

    /// Return an uncompressed copy of this attribute (will return a copy if not compressed).
    virtual AttributeArray::Ptr copyUncompressed() const = 0;

    /// Return the length of this array.
    virtual size_t size() const = 0;

    /// Return the number of bytes of memory used by this attribute.
    virtual size_t memUsage() const = 0;

    /// Create a new attribute array of the given (registered) type and length.
    static Ptr create(const NamePair& type, size_t length);
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
    bool hasValueType() const { return this->type().first == typeNameAsString<ValueType>();}

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

    /// @brief Retrieve the attribute array flags
    uint16_t flags() const { return mFlags; }

    IndexIter beginIndex() const;

    /// Read attribute metadata and buffers from a stream.
    virtual void read(std::istream&) = 0;
    /// Write attribute metadata and buffers to a stream.
    virtual void write(std::ostream&) const = 0;

    /// Ensures all data is in-core
    virtual void loadData() const = 0;

    /// Check the compressed bytes and flags. If they are equal, perform a deeper
    /// comparison check necessary on the inherited types (TypedAttributeArray)
    /// Requires non operator implementation due to inheritance
    bool operator==(const AttributeArray& other) const;
    bool operator!=(const AttributeArray& other) const { return !this->operator==(other); }

private:
    /// Virtual function used by the comparison operator to perform
    /// comparisons on inherited types
    virtual bool isEqual(const AttributeArray& other) const = 0;

protected:
    /// Obtain an Accessor that stores getter and setter functors.
    virtual AccessorBasePtr getAccessor() const = 0;

    /// Register a attribute type along with a factory function.
    static void registerType(const NamePair& type, FactoryMethod);
    /// Remove a attribute type from the registry.
    static void unregisterType(const NamePair& type);

    size_t mCompressedBytes;
    uint16_t mFlags;

    /// Out-of-core data
#ifndef OPENVDB_2_ABI_COMPATIBLE
    boost::shared_ptr<FileInfo> mFileInfo;
#endif
}; // class AttributeArray


////////////////////////////////////////


/// Accessor base class for AttributeArray storage where type is not available
struct AttributeArray::AccessorBase { };

/// Templated Accessor stores typed function pointers used in binding
/// AttributeHandles
template <typename T>
struct AttributeArray::Accessor : public AttributeArray::AccessorBase
{
    typedef T (*GetterPtr)(const AttributeArray* array, const Index n);
    typedef void (*SetterPtr)(AttributeArray* array, const Index n, const T& value);
    typedef void (*ValuePtr)(AttributeArray* array, const T& value);

    Accessor(GetterPtr getter, SetterPtr setter, ValuePtr collapser, ValuePtr filler) :
        mGetter(getter), mSetter(setter), mCollapser(collapser), mFiller(filler) { }

    GetterPtr mGetter;
    SetterPtr mSetter;
    ValuePtr  mCollapser;
    ValuePtr  mFiller;
}; // struct AttributeArray::Accessor


////////////////////////////////////////


/// Typed class for storing attribute data
template<typename ValueType_, typename Codec_ = NullAttributeCodec<ValueType_> >
class TypedAttributeArray: public AttributeArray
{
public:
    typedef boost::shared_ptr<TypedAttributeArray>          Ptr;
    typedef boost::shared_ptr<const TypedAttributeArray>    ConstPtr;

    typedef ValueType_                  ValueType;
    typedef Codec_                      Codec;
    typedef typename Codec::StorageType StorageType;

    //////////

    /// Default constructor, always constructs a uniform attribute.
    explicit TypedAttributeArray(size_t n = 1,
        const ValueType& uniformValue = zeroVal<ValueType>());
    /// Deep copy constructor (optionally decompress during copy).
    TypedAttributeArray(const TypedAttributeArray&, bool uncompress = false);
    /// Deep copy assignment operator.
    TypedAttributeArray& operator=(const TypedAttributeArray&);

    virtual ~TypedAttributeArray() { this->deallocate(); }

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const;

    /// Return an uncompressed copy of this attribute (will just return a copy if not compressed).
    virtual AttributeArray::Ptr copyUncompressed() const;

    /// Return a new attribute array of the given length @a n with uniform value zero.
    static Ptr create(size_t n);

    /// Cast an AttributeArray to TypedAttributeArray<T>
    static TypedAttributeArray& cast(AttributeArray& attributeArray);

    /// Cast an AttributeArray to TypedAttributeArray<T>
    static const TypedAttributeArray& cast(const AttributeArray& attributeArray);

    /// Return the name of this attribute's type (includes codec)
    static const NamePair& attributeType();
    /// Return the name of this attribute's type.
    virtual const NamePair& type() const { return attributeType(); }

    /// Return @c true if this attribute type is registered.
    static bool isRegistered();
    /// Register this attribute type along with a factory function.
    static void registerType();
    /// Remove this attribute type from the registry.
    static void unregisterType();

    /// Return the length of this array.
    virtual size_t size() const { return mSize; };

    /// Return the number of bytes of memory used by this attribute.
    virtual size_t memUsage() const;

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
    virtual void set(const Index n, const AttributeArray& sourceArray, const Index sourceIndex);

    /// Return @c true if this array is stored as a single uniform value.
    virtual bool isUniform() const { return mIsUniform; }
    /// @brief  Replace the single value storage with an array of length size().
    /// @note   Non-uniform attributes are unchanged.
    /// @param  fill toggle to initialize the array elements with the pre-expanded value.
    virtual void expand(bool fill = true);
    /// Replace the existing array with a uniform zero value.
    virtual void collapse();
    /// Compact the existing array to become uniform if all values are identical
    virtual bool compact();

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
    virtual bool compress();
    /// Uncompress the attribute array.
    virtual bool decompress();

    /// Read attribute data from a stream.
    virtual void read(std::istream& is);
    /// Write attribute data to a stream.
    virtual void write(std::ostream& os) const;

    /// Return @c true if this buffer's values have not yet been read from disk.
    inline bool isOutOfCore() const;

    /// Ensures all data is in-core
    virtual void loadData() const;

protected:
    virtual AccessorBasePtr getAccessor() const;

private:
    /// Load data from memory-mapped file.
    inline void doLoad() const;
    /// Load data from memory-mapped file (unsafe as this function is not protected by a mutex).
    inline void doLoadUnsafe() const;

    /// Toggle out-of-core state
    inline void setOutOfCore(const bool);

    /// Compare the this data to another attribute array. Used by the base class comparison operator
    virtual bool isEqual(const AttributeArray& other) const;

    size_t arrayMemUsage() const;
    void allocate(const size_t size);
    void deallocate();

    /// Helper function for use with registerType()
    static AttributeArray::Ptr factory(size_t n) { return TypedAttributeArray::create(n); }

    static tbb::atomic<const NamePair*> sTypeName;
    StorageType*    mData;
    size_t          mSize;
    bool            mIsUniform;
    tbb::spin_mutex mMutex;
}; // class TypedAttributeArray


////////////////////////////////////////


/// AttributeHandles provide access to specific TypedAttributeArray methods without needing
/// to know the compression codec, however these methods also incur the cost of a function pointer
template <typename T>
class AttributeHandle
{
public:
    typedef boost::shared_ptr<AttributeHandle<T> > Ptr;

protected:
    typedef T (*GetterPtr)(const AttributeArray* array, const Index n);
    typedef void (*SetterPtr)(AttributeArray* array, const Index n, const T& value);
    typedef void (*ValuePtr)(AttributeArray* array, const T& value);

public:
    static Ptr create(const AttributeArray& array, const bool preserveCompression = true);

    AttributeHandle(const AttributeArray& array, const bool preserveCompression = true);

    bool isUniform() const;

    T get(Index n) const;

protected:
    const AttributeArray* mArray;

    GetterPtr mGetter;
    SetterPtr mSetter;
    ValuePtr  mCollapser;
    ValuePtr  mFiller;

private:
    // local copy of AttributeArray (to preserve compression)
    AttributeArray::Ptr mLocalArray;
}; // class AttributeHandle


/// Write-able version of AttributeHandle
template <typename T>
class AttributeWriteHandle : public AttributeHandle<T>
{
public:
    typedef boost::shared_ptr<AttributeWriteHandle<T> > Ptr;

    static Ptr create(AttributeArray& array);

    AttributeWriteHandle(AttributeArray& array);

    /// @brief  If this array is uniform, replace it with an array of length size().
    /// @param  fill if true, assign the uniform value to each element of the array.
    void expand(bool fill = true);

    /// Replace the existing array with a uniform value (zero if none provided).
    void collapse();
    void collapse(const T& uniformValue);

    /// Compact the existing array to become uniform if all values are identical
    bool compact();

    /// @brief Fill the existing array with the given value.
    /// @note Identical to collapse() except a non-uniform array will not become uniform.
    void fill(const T& value);

    void set(Index n, const T& value);
}; // class AttributeWriteHandle


typedef AttributeHandle<float> AttributeHandleROF;
typedef AttributeWriteHandle<float> AttributeHandleRWF;

typedef AttributeHandle<Vec3f> AttributeHandleROVec3f;
typedef AttributeWriteHandle<Vec3f> AttributeHandleRWVec3f;


////////////////////////////////////////


// Attribute codec implementation


template<typename StorageType_>
template<typename ValueType>
inline void
NullAttributeCodec<StorageType_>::decode(const StorageType& data, ValueType& val)
{
    val = static_cast<ValueType>(data);
}


template<typename StorageType_>
template<typename ValueType>
inline void
NullAttributeCodec<StorageType_>::encode(const StorageType& val, ValueType& data)
{
    data = static_cast<StorageType>(val);
}


template<typename IntType>
template<typename ValueType>
inline void
FixedPointAttributeCodec<IntType>::decode(const StorageType& data, ValueType& val)
{
    val = fixedPointToFloatingPoint<ValueType>(data);

    // shift value range to be -0.5 => 0.5 (as this is most commonly used for position)

    val -= ValueType(0.5);
}


template<typename IntType>
template<typename ValueType>
inline void
FixedPointAttributeCodec<IntType>::encode(const ValueType& val, StorageType& data)
{
    // shift value range to be -0.5 => 0.5 (as this is most commonly used for position)

    const ValueType newVal = val + ValueType(0.5);

    data = floatingPointToFixedPoint<StorageType>(newVal);
}


template<typename T>
inline void
UnitVecAttributeCodec::decode(const StorageType& data, math::Vec3<T>& val)
{
    val = math::QuantizedUnitVec::unpack(data);
}


template<typename T>
inline void
UnitVecAttributeCodec::encode(const math::Vec3<T>& val, StorageType& data)
{
    data = math::QuantizedUnitVec::pack(val);
}


////////////////////////////////////////

// TypedAttributeArray implementation

template<typename ValueType_, typename Codec_>
tbb::atomic<const NamePair*> TypedAttributeArray<ValueType_, Codec_>::sTypeName;


template<typename ValueType_, typename Codec_>
TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray(
    size_t n, const ValueType& uniformValue)
    : AttributeArray()
    , mData(new StorageType[1])
    , mSize(n)
    , mIsUniform(true)
    , mMutex()
{
    mSize = std::max(size_t(1), mSize);
    Codec::encode(uniformValue, mData[0]);
}


template<typename ValueType_, typename Codec_>
TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray(const TypedAttributeArray& rhs, bool uncompress)
    : AttributeArray(rhs)
    , mData(NULL)
    , mSize(rhs.mSize)
    , mIsUniform(rhs.mIsUniform)
    , mMutex()
{
    using attribute_compression::decompress;
    using attribute_compression::uncompressedSize;

    // disable uncompress if data is not compressed

    if (!this->isCompressed())  uncompress = false;

    if (mIsUniform) {
        this->allocate(1);
        mData[0] = rhs.mData[0];
    } else if (this->isOutOfCore()) {
        // do nothing
    } else if (this->isCompressed()) {
        char* buffer = 0;
        if (uncompress) {
            rhs.doLoad();
            const char* charBuffer = reinterpret_cast<char*>(rhs.mData);
            buffer = decompress(charBuffer, uncompressedSize(charBuffer));
        }
        if (buffer)         mCompressedBytes = 0;
        else {
            // decompression wasn't requested or failed so deep copy instead
            buffer = new char[mCompressedBytes];
            memcpy(buffer, rhs.mData, mCompressedBytes);
        }
        assert(buffer);
        mData = reinterpret_cast<StorageType*>(buffer);
    } else {
        this->allocate(mSize);
        memcpy(mData, rhs.mData, mSize * sizeof(StorageType));
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
        mCompressedBytes = rhs.mCompressedBytes;
        mSize = rhs.mSize;
        mIsUniform = rhs.mIsUniform;

        if (mIsUniform) {
            this->allocate(1);
            mData[0] = rhs.mData[0];
#ifndef OPENVDB_2_ABI_COMPATIBLE
        } else if (rhs.isOutOfCore()) {
            mFileInfo = rhs.mFileInfo;
#endif
        } else if (this->isCompressed()) {
            char* buffer = new char[mCompressedBytes];
            memcpy(buffer, rhs.mData, mCompressedBytes);
            mData = reinterpret_cast<StorageType*>(buffer);
        } else {
            this->allocate(mSize);
            memcpy(mData, rhs.mData, mSize * sizeof(StorageType));
        }
    }
}


template<typename ValueType_, typename Codec_>
inline const NamePair&
TypedAttributeArray<ValueType_, Codec_>::attributeType()
{
    if (sTypeName == NULL) {
        std::ostringstream ostr1, ostr2;
        ostr1 << typeNameAsString<ValueType>();
        ostr2 << Codec::name() << "_" << typeNameAsString<StorageType>();
        NamePair* s = new NamePair(ostr1.str(), ostr2.str());
        if (sTypeName.compare_and_swap(s, NULL) != NULL) delete s;
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
TypedAttributeArray<ValueType_, Codec_>::create(size_t n)
{
    return Ptr(new TypedAttributeArray(n));
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
    if (mIsUniform)                 return sizeof(StorageType);
    if (this->isOutOfCore())        return 0;
    if (this->isCompressed())       return mCompressedBytes;

    return mSize * sizeof(StorageType);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::allocate(const size_t size)
{
    assert(!mData);
    mData = new StorageType[size];
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::deallocate()
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    // detach from file if delay-loaded
    if (this->isOutOfCore()) {
        this->setOutOfCore(false);
        this->mFileInfo.reset();
    }
#endif
    if (mData) {
        delete[] mData;
        mData = NULL;
    }
}


template<typename ValueType_, typename Codec_>
size_t
TypedAttributeArray<ValueType_, Codec_>::memUsage() const
{
    return sizeof(*this) + (mData != NULL ? this->arrayMemUsage() : 0);
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::ValueType
TypedAttributeArray<ValueType_, Codec_>::getUnsafe(Index n) const
{
    assert(!this->isCompressed());
    assert(!this->isOutOfCore());

    ValueType val;
    Codec::decode(/*in=*/mData[mIsUniform ? 0 : n], /*out=*/val);
    return val;
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::ValueType
TypedAttributeArray<ValueType_, Codec_>::get(Index n) const
{
    if (this->isCompressed())           const_cast<TypedAttributeArray*>(this)->decompress();
    else if (this->isOutOfCore())       this->doLoad();

    return this->getUnsafe(n);
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::getUnsafe(Index n, T& val) const
{
    assert(!this->isCompressed());
    assert(!this->isOutOfCore());

    ValueType tmp;
    Codec::decode(/*in=*/mData[mIsUniform ? 0 : n], /*out=*/tmp);
    val = static_cast<T>(tmp);
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::get(Index n, T& val) const
{
    if (this->isCompressed())           const_cast<TypedAttributeArray*>(this)->decompress();
    else if (this->isOutOfCore())       this->doLoad();

    this->getUnsafe(n, val);
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
    assert(!this->isCompressed());
    assert(!this->isOutOfCore());

    if (mIsUniform)     this->expand();

    Codec::encode(/*in=*/val, /*out=*/mData[n]);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::set(Index n, const ValueType& val)
{
    if (this->isCompressed())           this->decompress();
    else if (this->isOutOfCore())       this->doLoad();

    this->setUnsafe(n, val);
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::setUnsafe(Index n, const T& val)
{
    assert(!this->isCompressed());
    assert(!this->isOutOfCore());

    if (mIsUniform)     this->expand();

    const ValueType tmp = static_cast<ValueType>(val);
    Codec::encode(/*in=*/tmp, /*out=*/mData[n]);
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::set(Index n, const T& val)
{
    if (this->isCompressed())           this->decompress();
    else if (this->isOutOfCore())       this->doLoad();

    this->setUnsafe(n, val);
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

    const StorageType val = mData[0];

    {
        tbb::spin_mutex::scoped_lock lock(mMutex);
        this->deallocate();
        this->allocate(mSize);
    }

    mCompressedBytes = 0;
    mIsUniform = false;

    if (fill) {
        for (size_t i = 0; i < mSize; ++i)  mData[i] = val;
    }
}


template<typename ValueType_, typename Codec_>
bool
TypedAttributeArray<ValueType_, Codec_>::compact()
{
    if (mIsUniform)     return true;

    // compaction is not possible if any values are different
    const ValueType_ val = this->get(0);
    for (size_t i = 1; i < size(); i++) {
        if (this->get(i) != val)    return false;
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
        this->allocate(1);
        mIsUniform = true;
    }
    Codec::encode(uniformValue, mData[0]);
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
        this->allocate(mSize);
    }

    const size_t size = mIsUniform ? 1 : mSize;
    for (size_t i = 0; i < size; ++i)  {
        Codec::encode(value, mData[i]);
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
    using attribute_compression::canCompress;
    using attribute_compression::compress;

    if (!canCompress())     return false;

    if (!mIsUniform && !this->isCompressed()) {

        tbb::spin_mutex::scoped_lock lock(mMutex);

        this->doLoadUnsafe();

        const size_t typeSize = sizeof(typename Codec_::StorageType);
        const int inBytes = int(mSize * sizeof(StorageType));
        int outBytes;
        char* charBuffer = reinterpret_cast<char*>(mData);
        char* buffer = compress(charBuffer, typeSize, inBytes, outBytes, /*cleanup=*/true);

        if (buffer) {
            mData = reinterpret_cast<StorageType*>(buffer);
            mCompressedBytes = size_t(outBytes);
            return true;
        }
    }

    return false;
}


template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::decompress()
{
    using attribute_compression::decompress;
    using attribute_compression::uncompressedSize;

    tbb::spin_mutex::scoped_lock lock(mMutex);

    if (this->isCompressed()) {
        this->doLoadUnsafe();
        char* charBuffer = reinterpret_cast<char*>(this->mData);
        char* buffer = decompress(charBuffer, uncompressedSize(charBuffer));
        if (buffer) {
            mData = reinterpret_cast<StorageType*>(buffer);
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
#ifndef OPENVDB_2_ABI_COMPATIBLE
    return (mFlags & OUTOFCORE);
#else
    return false;
#endif
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::setOutOfCore(const bool b)
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (b)  mFlags |= OUTOFCORE;
    else    mFlags &= ~OUTOFCORE;
#else
    (void) b;
#endif
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::doLoad() const
{
#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!(this->isOutOfCore()))     return;

    TypedAttributeArray<ValueType_, Codec_>* self = const_cast<TypedAttributeArray<ValueType_, Codec_>*>(this);

    // This lock will be contended at most once, after which this buffer
    // will no longer be out-of-core.
    tbb::spin_mutex::scoped_lock lock(self->mMutex);
    this->doLoadUnsafe();
#endif
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
    using attribute_compression::decompress;

    // read data

    Index64 bytes = Index64(0);
    is.read(reinterpret_cast<char*>(&bytes), sizeof(Index64));
    bytes = bytes - /*flags*/sizeof(Int16) - /*size*/sizeof(Index64);

    Int16 flags = Int16(0);
    is.read(reinterpret_cast<char*>(&flags), sizeof(Int16));
    mFlags = flags;

    Index64 size = Index64(0);
    is.read(reinterpret_cast<char*>(&size), sizeof(Index64));
    mSize = size;

    char* buffer = new char[bytes];

    // read uniform and compressed state

    mIsUniform = mFlags & WRITEUNIFORM;
    mCompressedBytes = mFlags & WRITEMEMCOMPRESS ? bytes : Index64(0);

    // clear uniform and compress flags

    mFlags &= Int16(~WRITEUNIFORM & ~WRITEMEMCOMPRESS);

    tbb::spin_mutex::scoped_lock lock(mMutex);

    this->deallocate();

#ifndef OPENVDB_2_ABI_COMPATIBLE
    // If this array is being read from a memory-mapped file, delay loading of its data
    // until the data is actually accessed.
    io::MappedFile::Ptr mappedFile = io::getMappedFilePtr(is);
    const bool delayLoad = (mappedFile.get() != NULL);

    if (delayLoad) {
        this->setOutOfCore(true);
        mFileInfo.reset(new FileInfo);
        mFileInfo->bufpos = is.tellg();
        mFileInfo->mapping = mappedFile;
        mFileInfo->bytes = bytes;
        mFileInfo->meta = io::getStreamMetadataPtr(is);

        // read and discard buffer
        is.read(buffer, bytes);
        delete[] buffer;
        return;
    }
#endif

    is.read(buffer, bytes);

    // compressed on-disk

    if (mFlags & WRITEDISKCOMPRESS) {

        // decompress buffer

        const int inBytes = int(mSize * sizeof(StorageType));
        char* newBuffer = decompress(buffer, inBytes, /*cleanup=*/true);
        if (newBuffer)  buffer = newBuffer;
    }

    // set data to buffer

    mData = reinterpret_cast<StorageType*>(buffer);

    // clear all write flags

    mFlags &= Int16(~WRITEDISKCOMPRESS);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::write(std::ostream& os) const
{
    using attribute_compression::compress;

    if (this->isTransient())    return;

    Int16 flags(mFlags);
    Index64 size(mSize);

    boost::scoped_array<char> compressedBuffer;
    int compressedBytes = 0;

    this->doLoad();

    if (mIsUniform)
    {
        flags |= WRITEUNIFORM;
    }
    else if (this->isCompressed())
    {
        flags |= WRITEMEMCOMPRESS;
    }
    else if (io::getDataCompression(os) & io::COMPRESS_BLOSC)
    {
        const char* charBuffer = reinterpret_cast<const char*>(mData);
        const size_t typeSize = sizeof(typename Codec_::StorageType);
        const int inBytes = int(mSize * sizeof(StorageType));
        compressedBuffer.reset(compress(charBuffer, typeSize, inBytes, compressedBytes));
        if (compressedBuffer)   flags |= WRITEDISKCOMPRESS;
    }

    Index64 bytes = /*flags*/ sizeof(Int16) + /*size*/ sizeof(Index64);

    bytes += compressedBuffer ? compressedBytes : this->arrayMemUsage();

    // write data

    os.write(reinterpret_cast<const char*>(&bytes), sizeof(Index64));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(Int16));
    os.write(reinterpret_cast<const char*>(&size), sizeof(Index64));

    if (compressedBuffer)   os.write(reinterpret_cast<const char*>(compressedBuffer.get()), compressedBytes);
    else                    os.write(reinterpret_cast<const char*>(mData), this->arrayMemUsage());
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::doLoadUnsafe() const
{
    using attribute_compression::decompress;

#ifndef OPENVDB_2_ABI_COMPATIBLE
    if (!(this->isOutOfCore()))     return;

    // this function expects the mutex to already be locked

    TypedAttributeArray<ValueType_, Codec_>* self = const_cast<TypedAttributeArray<ValueType_, Codec_>*>(this);

    assert(self->mFileInfo);
    assert(self->mFileInfo->mapping.get() != NULL);

    FileInfo& info = *(self->mFileInfo);

    boost::shared_ptr<std::streambuf> buf = info.mapping->createBuffer();
    std::istream is(buf.get());

    const Index64 bytes = info.bytes;

    is.seekg(info.bufpos);

    char* buffer = new char[bytes];
    is.read(buffer, bytes);

    // compressed on-disk

    if (mFlags & WRITEDISKCOMPRESS) {

        // decompress buffer

        const int inBytes = int(mSize * sizeof(StorageType));
        char* newBuffer = decompress(buffer, inBytes, /*cleanup=*/true);
        if (newBuffer)  buffer = newBuffer;
    }

    // set data to buffer

    self->mData = reinterpret_cast<StorageType*>(buffer);

    // clear write and out-of-core flags

    self->mFlags &= Int16(~WRITEDISKCOMPRESS & ~OUTOFCORE);
#endif
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
       this->mIsUniform != otherT->mIsUniform ||
       *this->sTypeName != *otherT->sTypeName) return false;

    this->doLoad();

    const StorageType *target = this->mData, *source = otherT->mData;
    if (!target && !source) return true;
    if (!target || !source) return false;
    Index n = this->mIsUniform ? 1 : mSize;
    while (n && math::isExactlyEqual(*target++, *source++)) --n;
    return n == 0;
}

////////////////////////////////////////

// AttributeHandle implementation

template <typename T>
typename AttributeHandle<T>::Ptr
AttributeHandle<T>::create(const AttributeArray& array, const bool preserveCompression)
{
    return typename AttributeHandle<T>::Ptr(new AttributeHandle<T>(array, preserveCompression));
}

template <typename T>
AttributeHandle<T>::AttributeHandle(const AttributeArray& array, const bool preserveCompression)
    : mArray(&array)
{
    // load data if delay-loaded

    mArray->loadData();

    // if array is compressed and preserve compression is true, copy and decompress
    // into a local copy that is destroyed with handle to maintain thread-safety

    if (array.isCompressed())
    {
        if (preserveCompression) {
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

    AttributeArray::Accessor<T>* typedAccessor = static_cast<AttributeArray::Accessor<T>*>(accessor.get());

    if (!typedAccessor) {
        OPENVDB_THROW(RuntimeError, "Cannot bind AttributeHandle due to mis-matching types.");
    }

    mGetter = typedAccessor->mGetter;
    mSetter = typedAccessor->mSetter;
    mCollapser = typedAccessor->mCollapser;
    mFiller = typedAccessor->mFiller;
}


template <typename T>
T AttributeHandle<T>::get(Index n) const
{
    return mGetter(mArray, n);
}

template <typename T>
bool AttributeHandle<T>::isUniform() const
{
    return mArray->isUniform();
}

////////////////////////////////////////

// AttributeWriteHandle implementation

template <typename T>
typename AttributeWriteHandle<T>::Ptr
AttributeWriteHandle<T>::create(AttributeArray& array)
{
    return typename AttributeWriteHandle<T>::Ptr(new AttributeWriteHandle<T>(array));
}

template <typename T>
AttributeWriteHandle<T>::AttributeWriteHandle(AttributeArray& array)
    : AttributeHandle<T>(array, /*preserveCompression = */ false) { }

template <typename T>
void AttributeWriteHandle<T>::set(Index n, const T& value)
{
    this->mSetter(const_cast<AttributeArray*>(this->mArray), n, value);
}

template <typename T>
void AttributeWriteHandle<T>::expand(const bool fill)
{
    const_cast<AttributeArray*>(this->mArray)->expand(fill);
}

template <typename T>
void AttributeWriteHandle<T>::collapse()
{
    const_cast<AttributeArray*>(this->mArray)->collapse();
}

template <typename T>
bool AttributeWriteHandle<T>::compact()
{
    return const_cast<AttributeArray*>(this->mArray)->compact();
}

template <typename T>
void AttributeWriteHandle<T>::collapse(const T& uniformValue)
{
    this->mCollapser(const_cast<AttributeArray*>(this->mArray), uniformValue);
}

template <typename T>
void AttributeWriteHandle<T>::fill(const T& value)
{
    this->mFiller(const_cast<AttributeArray*>(this->mArray), value);
}

} // namespace tools

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
