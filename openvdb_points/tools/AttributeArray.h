///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
//
/// @file AttributeArray.h
///
/// @authors Mihai Alden, Peter Cucka


#ifndef OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>
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

#include <set>
#include <map>
#include <vector>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////

// Utility methods

template <typename IntegerT, typename FloatT>
inline IntegerT
floatingPointToFixedPoint(const FloatT s)
{
    if (FloatT(0.0) > s) return std::numeric_limits<IntegerT>::min();
    else if (FloatT(1.0) <= s) return std::numeric_limits<IntegerT>::max();
    return IntegerT(std::floor(s * FloatT(std::numeric_limits<IntegerT>::max() + 1)));
}


template <typename FloatT, typename IntegerT>
inline FloatT
fixedPointToFloatingPoint(const IntegerT s)
{
    return FloatT(s) / FloatT((std::numeric_limits<IntegerT>::max() + 1));
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


template<typename IntType>
struct FixedPositionAttributeCodec
{
    typedef IntType StorageType;
    template<typename ValueType> static void decode(const StorageType& s, ValueType& v){
        FixedPointAttributeCodec<IntType>::decode(s, v);
        v -= ValueType(0.5);
    }
    template<typename ValueType> static void encode(const ValueType& v, StorageType& s){
        FixedPointAttributeCodec<IntType>::encode(v + ValueType(0.5), s);
    }
    static const char* name() { return "fxps"; }
};


struct UnitVecAttributeCodec
{
    typedef uint16_t StorageType;
    template<typename T> static void decode(const StorageType&, math::Vec3<T>&);
    template<typename T> static void encode(const math::Vec3<T>&, StorageType&);
    static const char* name() { return "uvec"; }
};


template <typename FloatType, typename IntType>
struct VecAttributeCodec
{
    struct StorageType { FloatType magnitude; IntType direction;
        StorageType(): magnitude(0), direction(0) { }
    };
    template<typename T> static void decode(const StorageType&, math::Vec3<T>&);
    template<typename T> static void encode(const math::Vec3<T>&, StorageType&);
    static const char* name() { return "vec"; }
};


////////////////////////////////////////


/// Base class for storing point attribute information in a LeafNode
class AttributeArray
{
public:
    typedef boost::shared_ptr<AttributeArray>           Ptr;
    typedef boost::shared_ptr<const AttributeArray>     ConstPtr;

    typedef Ptr (*FactoryMethod)(size_t);

    AttributeArray() : mCompressedBytes(0), mFlags(0) {}
    virtual ~AttributeArray() {}

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const = 0;

    /// Return the length of this array.
    virtual size_t size() const = 0;

    /// Return the number of bytes of memory used by this attribute.
    virtual size_t memUsage() const = 0;

    /// Create a new attribute array of the given (registered) type and length.
    static Ptr create(const Name& type, size_t length);
    /// Return @c true if the given attribute type name is registered.
    static bool isRegistered(const Name &type);
    /// Clear the attribute type registry.
    static void clearRegistry();

    /// Return the name of this attribute's type.
    virtual const Name& type() const = 0;
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
    /// Register a attribute type along with a factory function.
    static void registerType(const Name& type, FactoryMethod);
    /// Remove a attribute type from the registry.
    static void unregisterType(const Name& type);

    enum { TRANSIENT = 0x1, HIDDEN = 0x2 };
    size_t mCompressedBytes;
    uint16_t mFlags;
}; // class AttributeArray


////////////////////////////////////////


/// Templated attribute class to hold specific types
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
    /// Deep copy constructor.
    TypedAttributeArray(const TypedAttributeArray&);
    /// Deep copy assignment operator.
    TypedAttributeArray& operator=(const TypedAttributeArray&);

    virtual ~TypedAttributeArray() { this->deallocate(); }

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const;

    /// Return a new attribute array of the given length @a n with uniform value zero.
    static Ptr create(size_t n);

    /// Return the name of this attribute's type.
    static const Name& attributeType();
    /// Return the name of this attribute's type.
    virtual const Name& type() const { return attributeType(); }

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

    /// Read attribute data from a stream.
    virtual void read(std::istream& is);
    /// Write attribute data to a stream.
    virtual void write(std::ostream& os) const;

private:
    /// Compare the this data to another attribute array. Used by the base class comparison operator
    virtual bool isEqual(const AttributeArray& other) const;

    size_t arrayMemUsage() const;
    void allocate(bool fill = true);
    void deallocate();

    /// Helper function for use with registerType()
    static AttributeArray::Ptr factory(size_t n) { return TypedAttributeArray::create(n); }

    static tbb::atomic<const Name*> sTypeName;
    StorageType*    mData;
    size_t          mSize;
    bool            mIsUniform;
    tbb::spin_mutex mMutex;
}; // class TypedAttributeArray

////////////////////////////////////////


/// Ordered collection of uniquely-named attribute arrays
class AttributeSet
{
public:
    enum { INVALID_POS = boost::integer_traits<size_t>::const_max };

    typedef boost::shared_ptr<AttributeSet> Ptr;
    typedef boost::shared_ptr<const AttributeSet> ConstPtr;

    class Descriptor;

    typedef boost::shared_ptr<Descriptor> DescriptorPtr;
    typedef boost::shared_ptr<const Descriptor> DescriptorConstPtr;

    //////////

    struct Util
    {
        /// Attribute and type name pair.
        struct NameAndType {
            NameAndType(const std::string& n = "", const std::string& t = ""): name(n), type(t) {}
            std::string name, type;
        };

        typedef std::vector<NameAndType> NameAndTypeVec;
    };

    //////////

    AttributeSet();

    /// Construct from the given descriptor
    explicit AttributeSet(const DescriptorPtr&, size_t arrayLength = 1);

    /// Shallow copy constructor, the descriptor and attribute arrays will be shared.
    AttributeSet(const AttributeSet&);

    //@{
    /// @brief  Return a reference to this attribute set's descriptor, which might
    ///         be shared with other sets.
    Descriptor& descriptor() { return *mDescr; }
    const Descriptor& descriptor() const { return *mDescr; }
    //@}

    /// @brief Return a pointer to this attribute set's descriptor, which might be
    /// shared with other sets
    DescriptorPtr descriptorPtr() const { return mDescr; }

    /// Return the number of attributes in this set.
    size_t size() const { return mAttrs.size(); }

    /// Return the number of bytes of memory used by this attribute set.
    size_t memUsage() const;

    /// @brief  Return the position of the attribute array whose name is @a name,
    ///         or @c INVALID_POS if no match is found.
    size_t find(const std::string& name) const;

    /// @brief  Replace the attribute array whose name is @a name.
    /// @return The position of the updated attribute array or @c INVALID_POS
    ///         if the given name does not exist or if the replacement failed because
    ///         the new array type does not comply with the descriptor.
    size_t replace(const std::string& name, const AttributeArray::Ptr&);

    /// @brief  Replace the attribute array stored at position @a pos in this container.
    /// @return The position of the updated attribute array or @c INVALID_POS
    ///         if replacement failed because the new array type does not comply with
    ///         the descriptor.
    size_t replace(size_t pos, const AttributeArray::Ptr&);

    //@{
    /// @brief  Return a pointer to the attribute array whose name is @a name or
    ///         a null pointer if no match is found.
    const AttributeArray* getConst(const std::string& name) const;
    const AttributeArray* get(const std::string& name) const;
    AttributeArray*       get(const std::string& name);
    //@}


    //@{
    /// @brief  Return a pointer to the attribute array stored at position @a pos
    ///         in this set.
    const AttributeArray* getConst(size_t pos) const;
    const AttributeArray* get(size_t pos) const;
    AttributeArray*       get(size_t pos);
    //@}


    /// Return true if the attribute array stored at position @a pos is shared.
    bool isShared(size_t pos) const;
    /// @brief  If the attribute array stored at position @a pos is shared,
    ///         replace the array with a deep copy of itself that is not
    ///         shared with anyone else.
    void makeUnique(size_t pos);

    /// Copy attribute values for a target index @a n from another @a attributeSet for index @a source
    void copyAttributeValues(const Index n, const AttributeSet& attributeSet, const Index source);

    /// Append attribute @a attribute (simple method)
    AttributeArray::Ptr appendAttribute(const Util::NameAndType& attribute);

    /// Append attribute @a attribute (descriptor-sharing)
    /// Requires current descriptor to match @a expected
    /// On append, current descriptor is replaced with @a replacement
    AttributeArray::Ptr appendAttribute(const Util::NameAndType& attribute,
                                        const Descriptor& expected, DescriptorPtr& replacement);

    /// Drop attributes with @a pos indices (simple method)
    /// Creates a new descriptor for this attribute set
    void dropAttributes(const std::vector<size_t>& pos);

    /// Drop attributes with @a pos indices (descriptor-sharing method)
    /// Requires current descriptor to match @a expected
    /// On drop, current descriptor is replaced with @a replacement
    void dropAttributes(const std::vector<size_t>& pos,
                        const Descriptor& expected, DescriptorPtr& replacement);

    //
    /// @todo implement a I/O registry to handle shared descriptor objects.
    //

    /// Read the entire set from a stream.
    void read(std::istream&);
    /// Write the entire set to a stream.
    void write(std::ostream&) const;

    /// This will read the attribute descriptor from a stream, but no attribute data.
    void readMetadata(std::istream&);
    /// This will write the attribute descriptor to a stream, but no attribute data.
    void writeMetadata(std::ostream&) const;

    /// Read attribute data from a stream.
    void readAttributes(std::istream&);
    /// Write attribute data to a stream.
    void writeAttributes(std::ostream&) const;

    /// Compare the descriptors and attribute arrays on the attribute sets
    /// Exit early if the descriptors do not match
    bool operator==(const AttributeSet& other) const;
    bool operator!=(const AttributeSet& other) const { return !this->operator==(other); }

private:
    /// Disallow assignment, since it wouldn't be obvious whether the copy is deep or shallow.
    AttributeSet& operator=(const AttributeSet&);

    typedef std::vector<AttributeArray::Ptr> AttrArrayVec;

    DescriptorPtr mDescr;
    AttrArrayVec  mAttrs;
}; // class AttributeSet

////////////////////////////////////////


/// @brief  An immutable object that stores name, type and AttributeSet position
///         for a constant collection of attribute arrays.
/// @note   The attribute name is actually mutable, but the attribute type
///         and position can not be changed after creation.
class AttributeSet::Descriptor
{
public:
    typedef boost::shared_ptr<Descriptor> Ptr;
    typedef std::map<std::string, size_t> NameToPosMap;
    typedef NameToPosMap::const_iterator  ConstIterator;

    typedef Util::NameAndType             NameAndType;
    typedef Util::NameAndTypeVec          NameAndTypeVec;

    /// Utility method to construct a NameAndType sequence.
    struct Inserter {
        NameAndTypeVec vec;
        Inserter& add(const NameAndType& nameAndType) {
            vec.push_back(nameAndType); return *this;
        }
        Inserter& add(const std::string& name, const std::string& type) {
            vec.push_back(NameAndType(name, type)); return *this;
        }
        Inserter& add(const NameAndTypeVec& other) {
            for (NameAndTypeVec::const_iterator it = other.begin(), itEnd = other.end(); it != itEnd; ++it) {
                vec.push_back(NameAndType(it->name, it->type));
            }
            return *this;
        }
    };

    //////////

    Descriptor();

    /// Create a new descriptor from the given attribute and type name pairs.
    static Ptr create(const NameAndTypeVec&);

    Ptr duplicateAppend(const NameAndType& attribute) const;
    Ptr duplicateAppend(const NameAndTypeVec& vec) const;
    Ptr duplicateDrop(const std::vector<size_t>& pos) const;

    /// Return the number of attributes in this descriptor.
    size_t size() const { return mTypes.size(); }

    /// Return the number of bytes of memory used by this attribute set.
    size_t memUsage() const;

    /// @brief  Return the position of the attribute array whose name is @a name,
    ///         or @c INVALID_POS if no match is found.
    size_t find(const std::string& name) const;

    /// Rename an attribute array
    size_t rename(const std::string& fromName, const std::string& toName);

    /// Return the name of the attribute array's type.
    const std::string& type(size_t pos) const { assert(pos != AttributeSet::INVALID_POS); return mTypes[pos]; }

    /// Return true if this descriptor is equal to the given one.
    bool operator==(const Descriptor&) const;
    /// Return true if this descriptor is not equal to the given one.
    bool operator!=(const Descriptor& rhs) const { return !this->operator==(rhs); }

    /// Return a reference to the name-to-position map.
    const NameToPosMap& map() const { return mNameMap; }

    /// Append to a vector of names and types from this Descriptor in position order
    void appendTo(NameAndTypeVec& attrs) const;

    /// Serialize this descriptor to the given stream.
    void write(std::ostream&) const;
    /// Unserialize this transform from the given stream.
    void read(std::istream&);

private:
    size_t insert(const std::string& name, const std::string& typeName);
    NameToPosMap                mNameMap;
    std::vector<std::string>    mTypes;
}; // class Descriptor


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
}


template<typename IntType>
template<typename ValueType>
inline void
FixedPointAttributeCodec<IntType>::encode(const ValueType& val, StorageType& data)
{
    data = floatingPointToFixedPoint<StorageType>(val);
}


template<typename T>
inline void
UnitVecAttributeCodec::decode(const StorageType& data, math::Vec3<T>& val)
{
    val = math::QuantizedUnitVec::unpack(data);
}


template<typename T>
inline void UnitVecAttributeCodec::encode(const math::Vec3<T>& val, StorageType& data)
{
    data = math::QuantizedUnitVec::pack(val);
}


template<typename FloatType, typename IntType>
template<typename T>
inline void
VecAttributeCodec<FloatType, IntType>::decode(const StorageType& data, math::Vec3<T>& val)
{
    UnitVecAttributeCodec<IntType>::template decode<T>(data.direction, val);
    val *= T(data.magnitude);
}


template<typename FloatType, typename IntType>
template<typename T>
inline void
VecAttributeCodec<FloatType, IntType>::encode(const math::Vec3<T>& val, StorageType& data)
{
    const double d = val.length();
    data.magnitude = static_cast<FloatType>(d);

    math::Vec3d dir = val;
    if (!math::isApproxEqual(d, 0.0, math::Tolerance<double>::value())) {
        dir *= 1.0 / d;
    }

    UnitVecAttributeCodec<IntType>::template encode<T>(dir, data.direction);
}


////////////////////////////////////////

// TypedAttributeArray implementation


template<typename ValueType_, typename Codec_>
tbb::atomic<const Name*> TypedAttributeArray<ValueType_, Codec_>::sTypeName;


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
TypedAttributeArray<ValueType_, Codec_>::TypedAttributeArray(const TypedAttributeArray& rhs)
    : AttributeArray(rhs)
    , mData(NULL)
    , mSize(rhs.mSize)
    , mIsUniform(rhs.mIsUniform)
    , mMutex()
{
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
inline const Name&
TypedAttributeArray<ValueType_, Codec_>::attributeType()
{
    if (sTypeName == NULL) {
        std::ostringstream ostr;
        ostr << typeNameAsString<ValueType>() << "_" << Codec::name()
             << "_" << typeNameAsString<StorageType>();
        Name* s = new Name(ostr.str());
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
TypedAttributeArray<ValueType_, Codec_>::decompress()
{
#ifdef OPENVDB_USE_BLOSC

    tbb::spin_mutex::scoped_lock lock(mMutex);

    if (mCompressedBytes != 0) {

        size_t inBytes = 0;
        blosc_cbuffer_sizes(mData, &inBytes, NULL, NULL);

        int bufBytes = inBytes + BLOSC_MAX_OVERHEAD;
        boost::scoped_array<char> outBuf(new char[bufBytes]);

        const int outBytes = blosc_decompress_ctx(
             /*src=*/mData, /*dest=*/outBuf.get(), bufBytes, /*numthreads=*/1);

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
    }

#else

    if (mCompressedBytes != 0) { // throw if the array is compressed.
        OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
    }

#endif

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


} // namespace tools

namespace attrstorage {
typedef tools::VecAttributeCodec<float, uint16_t>::StorageType FloatUInt16;
} // namespace attrstorage

namespace math {

template<> inline bool
isExactlyEqual< attrstorage::FloatUInt16, attrstorage::FloatUInt16>(
                    const attrstorage::FloatUInt16& a, const attrstorage::FloatUInt16& b)
{
    return a.magnitude == b.magnitude && a.direction == b.direction;
}
} // namespace math

template<> inline attrstorage::FloatUInt16
zeroVal<attrstorage::FloatUInt16>() { return attrstorage::FloatUInt16(); }

template<> inline const char* typeNameAsString<attrstorage::FloatUInt16>() {
    return "floatuint16";
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
