///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
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

#include <openvdb/openvdb.h>
#include <openvdb_points/Types.h>
#include <openvdb/math/QuantizedUnitVec.h>
#include <openvdb/util/Name.h>
#include <openvdb/util/logging.h>

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

#include <tbb/spin_mutex.h>
#include <tbb/atomic.h>

#include <boost/scoped_array.hpp>
#include <boost/integer_traits.hpp> // const_max

#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {


// Add new typedef for a Name pair
typedef std::pair<Name, Name> NamePair;


namespace tools {


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

    /// Set value at given index @a n from @a sourceIndex of another @a sourceArray
    virtual void set(const Index n, const AttributeArray& sourceArray, const Index sourceIndex) = 0;

    /// Return @c true if this array is stored as a single uniform value.
    virtual bool isUniform() const = 0;
    /// @brief  If this array is uniform, replace it with an array of length size().
    /// @param  fill if true, assign the uniform value to each element of the array.
    virtual void expand(bool fill = true) = 0;
    /// Replace the existing array with a uniform zero value.
    virtual void collapse() = 0;

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

    /// Read attribute data from a stream.
    virtual void read(std::istream&) = 0;
    /// Write attribute data to a stream.
    virtual void write(std::ostream&) const = 0;

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

    enum { TRANSIENT = 0x1, HIDDEN = 0x2 };
    size_t mCompressedBytes;
    uint16_t mFlags;
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

    Accessor(GetterPtr getter, SetterPtr setter) :
        mGetter(getter), mSetter(setter) { }

    GetterPtr mGetter;
    SetterPtr mSetter;
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
    TypedAttributeArray(const TypedAttributeArray&, const bool decompress = false);
    /// Deep copy assignment operator.
    TypedAttributeArray& operator=(const TypedAttributeArray&);

    virtual ~TypedAttributeArray() { this->deallocate(); }

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const;

    /// Return an uncompressed copy of this attribute (will just return a copy if not compressed).
    virtual AttributeArray::Ptr copyUncompressed() const;

    /// Return a new attribute array of the given length @a n with uniform value zero.
    static Ptr create(size_t n);

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

    /// Return the value at index @a n
    ValueType get(Index n) const;
    /// Return the @a value at index @a n
    template<typename T> void get(Index n, T& value) const;

    /// Non-member equivalent to get() that static_casts array to this TypedAttributeArray
    static ValueType get(const AttributeArray* array, const Index n);

    /// Non-member equivalent to set() that static_casts array to this TypedAttributeArray
    static void set(AttributeArray* array, const Index n, const ValueType& value);

    /// Set @a value at the given index @a n
    void set(Index n, const ValueType& value);
    /// Set @a value at the given index @a n
    template<typename T> void set(Index n, const T& value);

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
    /// Replace the existing array with the given uniform value.
    void collapse(const ValueType& uniformValue);

    /// Compress the attribute array.
    virtual bool compress();
    /// Uncompress the attribute array.
    virtual bool decompress();
    /// Uncompress the compressed buffer supplied into the attribute array.
    bool decompress(const StorageType* compressedData);

    /// Read attribute data from a stream.
    virtual void read(std::istream& is);
    /// Write attribute data to a stream.
    virtual void write(std::ostream& os) const;


protected:
    virtual AccessorBasePtr getAccessor() const;

private:
    /// Compare the this data to another attribute array. Used by the base class comparison operator
    virtual bool isEqual(const AttributeArray& other) const;

    size_t arrayMemUsage() const;
    void allocate(bool fill = true);
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

public:
    static Ptr create(const AttributeArray& array, const bool preserveCompression = true);

    AttributeHandle(const AttributeArray& array, const bool preserveCompression = true);

    T get(Index n) const;

protected:
    const AttributeArray* mArray;

    GetterPtr mGetter;
    SetterPtr mSetter;

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
TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray(const TypedAttributeArray& rhs, const bool decompress)
    : AttributeArray(rhs)
    , mData(NULL)
    , mSize(rhs.mSize)
    , mIsUniform(rhs.mIsUniform)
    , mMutex()
{
    if (mIsUniform) {
        mData = new StorageType[1];
        mData[0] = rhs.mData[0];
    } else if (mCompressedBytes != 0 && decompress) {
        this->decompress(rhs.mData);
    } else if (mCompressedBytes != 0) {
        char* buffer = new char[mCompressedBytes];
        memcpy(buffer, rhs.mData, mCompressedBytes);
        mData = reinterpret_cast<StorageType*>(buffer);
    } else {
        mData = new StorageType[mSize];
        memcpy(mData, rhs.mData, mSize * sizeof(StorageType));
    }
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray&
TypedAttributeArray<ValueType_, Codec_>::operator=(const TypedAttributeArray& rhs)
{
    if (&rhs != this) {
        this->deallocate();

        mFlags= rhs.mFlags;
        mCompressedBytes = rhs.mCompressedBytes;
        mSize = rhs.mSize;
        mIsUniform = rhs.mIsUniform;

        if (mIsUniform) {
            mData = new StorageType[1];
            mData[0] = rhs.mData[0];
        } else if (mCompressedBytes != 0) {
            char* buffer = new char[mCompressedBytes];
            memcpy(buffer, rhs.mData, mCompressedBytes);
            mData = reinterpret_cast<StorageType*>(buffer);
        } else {
            mData = new StorageType[mSize];
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
    return mCompressedBytes != 0 ? mCompressedBytes :
        (mIsUniform ? sizeof(StorageType) : (mSize * sizeof(StorageType)));
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::allocate(bool fill)
{
    tbb::spin_mutex::scoped_lock lock(mMutex);

    StorageType val = mIsUniform ? mData[0] : zeroVal<StorageType>();

    if (mData) {
        delete[] mData;
        mData = NULL;
    }

    mCompressedBytes = 0;
    mIsUniform = false;

    mData = new StorageType[mSize];
    if (fill) {
        for (size_t i = 0; i < mSize; ++i) mData[i] = val;
    }
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::deallocate()
{
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
TypedAttributeArray<ValueType_, Codec_>::get(Index n) const
{
    if (mCompressedBytes != 0) const_cast<TypedAttributeArray*>(this)->decompress();
    if (mIsUniform) n = 0;

    ValueType val;
    Codec::decode(/*in=*/mData[n], /*out=*/val);
    return val;
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::get(Index n, T& val) const
{
    if (mCompressedBytes != 0) const_cast<TypedAttributeArray*>(this)->decompress();
    if (mIsUniform) n = 0;

    ValueType tmp;
    Codec::decode(/*in=*/mData[n], /*out=*/tmp);
    val = static_cast<T>(tmp);
}


template<typename ValueType_, typename Codec_>
typename TypedAttributeArray<ValueType_, Codec_>::ValueType
TypedAttributeArray<ValueType_, Codec_>::get(const AttributeArray* array, const Index n)
{
    return static_cast<const TypedAttributeArray<ValueType, Codec>*>(array)->get(n);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::set(Index n, const ValueType& val)
{
    if (mCompressedBytes != 0) this->decompress();
    if (mIsUniform) this->allocate();

    Codec::encode(/*in=*/val, /*out=*/mData[n]);
}


template<typename ValueType_, typename Codec_>
template<typename T>
void
TypedAttributeArray<ValueType_, Codec_>::set(Index n, const T& val)
{
    const ValueType tmp = static_cast<ValueType>(val);

    if (mCompressedBytes != 0) this->decompress();
    if (mIsUniform) this->allocate();

    Codec::encode(/*in=*/tmp, /*out=*/mData[n]);
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
TypedAttributeArray<ValueType_, Codec_>::set(AttributeArray* array, const Index n, const ValueType& value)
{
    static_cast<TypedAttributeArray<ValueType, Codec>*>(array)->set(n, value);
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
        this->deallocate();
        mData = new StorageType[1];
        mIsUniform = true;
    }
    Codec::encode(uniformValue, mData[0]);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::expand(bool fill)
{
    this->allocate(fill);
}


template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::compress()
{
#ifdef OPENVDB_USE_BLOSC

    tbb::spin_mutex::scoped_lock lock(mMutex);

    if (!mIsUniform && mCompressedBytes == 0) {

        const int inBytes = int(mSize * sizeof(StorageType));
        int bufBytes = inBytes + BLOSC_MAX_OVERHEAD;
        boost::scoped_array<char> outBuf(new char[bufBytes]);

        bufBytes = blosc_compress_ctx(
            /*clevel=*/9, // 0 (no compression) to 9 (maximum compression)
            /*doshuffle=*/true,
            /*typesize=*/1,
            /*srcsize=*/inBytes,
            /*src=*/mData,
            /*dest=*/outBuf.get(),
            /*destsize=*/bufBytes,
            BLOSC_LZ4_COMPNAME,
            /*blocksize=*/256,
            /*numthreads=*/1);

        if (bufBytes <= 0) {
            std::ostringstream ostr;
            ostr << "Blosc failed to compress " << inBytes << " byte" << (inBytes == 1 ? "" : "s");
            if (bufBytes < 0) ostr << " (internal error " << bufBytes << ")";
            OPENVDB_LOG_DEBUG(ostr.str());
            return false;
        }

        this->deallocate();

        char* outData = new char[bufBytes];
        std::memcpy(outData, outBuf.get(), size_t(bufBytes));
        mData = reinterpret_cast<StorageType*>(outData);

        mCompressedBytes = size_t(bufBytes);
        return true;
    }

#else
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
#endif

    return false;
}


template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::decompress(const StorageType* compressedData)
{
#ifdef OPENVDB_USE_BLOSC

    size_t inBytes, compressedBytes, blockSize;
    blosc_cbuffer_sizes(compressedData, &inBytes, &compressedBytes, &blockSize);

    int bufBytes = inBytes + BLOSC_MAX_OVERHEAD;
    boost::scoped_array<char> outBuf(new char[bufBytes]);

    const int outBytes = blosc_decompress_ctx(
         /*src=*/compressedData, /*dest=*/outBuf.get(), bufBytes, /*numthreads=*/1);

    if (bufBytes < 1) {
        OPENVDB_LOG_DEBUG("blosc_decompress() returned error code " << bufBytes);
        return false;
    }

    if (mData) delete[] mData;

    char* outData = new char[outBytes];
    std::memcpy(outData, outBuf.get(), size_t(outBytes));
    mData = reinterpret_cast<StorageType*>(outData);

    mCompressedBytes = 0;
    return true;

#else

    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");

#endif
}

template<typename ValueType_, typename Codec_>
inline bool
TypedAttributeArray<ValueType_, Codec_>::decompress()
{
    tbb::spin_mutex::scoped_lock lock(mMutex);

    if (mCompressedBytes != 0) {
        return this->decompress(this->mData);
    }

    return false;
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::read(std::istream& is)
{
    // AttributeArray data
    Int16 flags = Int16(0);
    is.read(reinterpret_cast<char*>(&flags), sizeof(Int16));
    mFlags = flags;

    Index64 compressedBytes = Index64(0);
    is.read(reinterpret_cast<char*>(&compressedBytes), sizeof(Index64));
    mCompressedBytes = size_t(compressedBytes);

    // TypedAttributeArray data
    Int16 isUniform = Int16(0);
    is.read(reinterpret_cast<char*>(&isUniform), sizeof(Int16));
    mIsUniform = bool(isUniform);

    Index64 arrayLength = Index64(0);
    is.read(reinterpret_cast<char*>(&arrayLength), sizeof(Index64));
    mSize = size_t(arrayLength);

    this->deallocate();
    const size_t bufferSize = this->arrayMemUsage();

    char* buffer = new char[bufferSize];
    is.read(buffer, bufferSize);
    mData = reinterpret_cast<StorageType*>(buffer);
}


template<typename ValueType_, typename Codec_>
void
TypedAttributeArray<ValueType_, Codec_>::write(std::ostream& os) const
{
    if (!this->isTransient()) {

        // AttributeArray data
        os.write(reinterpret_cast<const char*>(&mFlags), sizeof(Int16));

        Index64 compressedBytes = Index64(mCompressedBytes);
        os.write(reinterpret_cast<const char*>(&compressedBytes), sizeof(Index64));

        // TypedAttributeArray data
        Int16 isUniform = Int16(mIsUniform);
        os.write(reinterpret_cast<char*>(&isUniform), sizeof(Int16));

        Index64 arrayLength = Index64(mSize);
        os.write(reinterpret_cast<const char*>(&arrayLength), sizeof(Index64));

        os.write(reinterpret_cast<const char*>(mData), this->arrayMemUsage());

    }
}


template<typename ValueType_, typename Codec_>
AttributeArray::AccessorBasePtr
TypedAttributeArray<ValueType_, Codec_>::getAccessor() const
{
    return AccessorBasePtr(new AttributeArray::Accessor<ValueType_>(
        &TypedAttributeArray<ValueType_, Codec_>::get, &TypedAttributeArray<ValueType_, Codec_>::set));
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
    // if array is compressed and preserve compression is true, copy and decompress
    // into a local copy that is destroyed with handle to maintain thread-safety

    if (array.isCompressed() && preserveCompression) {
        mLocalArray = array.copyUncompressed();
        mLocalArray->decompress();
        mArray = mLocalArray.get();
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
}


template <typename T>
T AttributeHandle<T>::get(Index n) const
{
    return mGetter(mArray, n);
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


} // namespace tools

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
