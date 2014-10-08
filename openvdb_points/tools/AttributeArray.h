// DreamWorks Animation LLC Confidential Information.
// TM and (c) 2014 DreamWorks Animation LLC.  All Rights Reserved.
// Reproduction in whole or in part without prior written permission of a
// duly authorized representative is prohibited.
//
/// @file AttributeArray.h
///
/// @note For evaluation purposes, do not distribute.
///
/// @authors Mihai Alden, Peter Cucka


#ifndef OPENVDB_POINTS_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <openvdb/math/QuantizedUnitVec.h>
#include <openvdb/util/Name.h>

#include <tbb/spin_mutex.h>
#include <tbb/atomic.h>

#include <boost/integer_traits.hpp> // const_max
#include <boost/algorithm/string.hpp> // split and is_any_of

#include <set>
#include <map>
#include <vector>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////

/// @todo move the following functions to math.h

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


template <typename IntegerT, typename FloatT>
inline openvdb::math::Vec3<IntegerT>
floatingPointToFixedPoint(const openvdb::math::Vec3<FloatT>& v)
{
    return openvdb::math::Vec3<IntegerT>(
        floatingPointToFixedPoint<IntegerT>(v.x()),
        floatingPointToFixedPoint<IntegerT>(v.y()),
        floatingPointToFixedPoint<IntegerT>(v.z()));
}


template <typename FloatT, typename IntegerT>
inline openvdb::math::Vec3<FloatT>
fixedPointToFloatingPoint(const openvdb::math::Vec3<IntegerT>& v)
{
    return openvdb::math::Vec3<FloatT>(
        fixedPointToFloatingPoint<FloatT>(v.x()),
        fixedPointToFloatingPoint<FloatT>(v.y()),
        fixedPointToFloatingPoint<FloatT>(v.z()));
}


////////////////////////////////////////

/// Attribute compression schemes

template<typename StorageType_>
struct NullAttributeCodec
{
    typedef StorageType_ StorageType;

    template<typename ValueType>
    static void decode(const StorageType& data, ValueType& val)
    {
        val = static_cast<ValueType>(data);
    }

    template<typename ValueType>
    static void encode(const StorageType& val, ValueType& data)
    {
        data = static_cast<StorageType>(val);
    }

    static const char* name() { return "null"; }
};


template<typename IntType>
struct FixedPointAttributeCodec
{
    typedef IntType StorageType;

    template<typename ValueType>
    static void decode(const StorageType& data, ValueType& val)
    {
        val = fixedPointToFloatingPoint<ValueType>(data);
    }

    template<typename ValueType>
    static void encode(const ValueType& val, StorageType& data)
    {
        data = floatingPointToFixedPoint<StorageType>(val);
    }

    static const char* name() { return "fxpt"; }
};


struct UnitVecAttributeCodec
{
    typedef uint16_t StorageType;

    template<typename T>
    static void decode(const StorageType& data, openvdb::math::Vec3<T>& val)
    {
        val = openvdb::math::QuantizedUnitVec::unpack(data);
    }

    template<typename T>
    static void encode(const openvdb::math::Vec3<T>& val, StorageType& data)
    {
        data = openvdb::math::QuantizedUnitVec::pack(val);
    }

    static const char* name() { return "uvec"; }
};


struct VelocityAttributeCodec
{
    struct QuantizedVelocity {
        float magnitude;
        uint16_t direction;
    };

    typedef QuantizedVelocity StorageType;

    template<typename T>
    static void decode(const StorageType& data, openvdb::math::Vec3<T>& val)
    {
        val = openvdb::math::QuantizedUnitVec::unpack(data.direction);
        val *= T(data.magnitude);
    }

    template<typename T>
    static void encode(const openvdb::math::Vec3<T>& val, StorageType& data)
    {
        const double d = val.length();
        data.magnitude = static_cast<float>(d);

        openvdb::math::Vec3d dir = val;
        if (!openvdb::math::isApproxEqual(d, 0.0, openvdb::math::Tolerance<double>::value())) {
            dir *= 1.0 / d;
        }

        data.direction = openvdb::math::QuantizedUnitVec::pack(dir);
    }

    static const char* name() { return "qvel"; }
};


////////////////////////////////////////


/// Base class for storing point attribute information in a LeafNode
class AttributeArray
{
public:
    typedef boost::shared_ptr<AttributeArray>           Ptr;
    typedef boost::shared_ptr<const AttributeArray>     ConstPtr;

    typedef Ptr (*FactoryMethod)(size_t);

    AttributeArray() : mFlags(0) {}
    virtual ~AttributeArray() {}

    /// Return the length of this array.
    virtual size_t size() const = 0;

    /// Return the number of bytes of memory used by this attribute.
    virtual size_t memUsage() const = 0;

    /// Create a new attribute array of the given (registered) type and length.
    static Ptr create(const openvdb::Name& type, size_t length);
    /// Return @c true if the given attribute type name is registered.
    static bool isRegistered(const openvdb::Name &type);
    /// Clear the attribute type registry.
    static void clearRegistry();

    /// Return the name of this attribute's type.
    virtual const openvdb::Name& type() const = 0;

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const = 0;

    /// Return @c true if this attribute is of the same type as the template parameter.
    template<typename AttributeArrayType>
    bool isType() const { return this->type() == AttributeArrayType::attributeType(); }

    /// Return @c true if this array is stored as a single uniform value.
    virtual bool isUniform() const = 0;
    /// @brief  If this array is uniform, replace it with an array of length size().
    /// @param  fill if true, assign the uniform value to each element of the array.
    virtual void expand(bool fill = true) = 0;
    /// Replace the existing array with a uniform zero value.
    virtual void collapse() = 0;

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

protected:

    /// Register a attribute type along with a factory function.
    static void registerType(const openvdb::Name& type, FactoryMethod);
    /// Remove a attribute type from the registry.
    static void unregisterType(const openvdb::Name& type);

    enum { TRANSIENT = 0x1, HIDDEN = 0x2 };
    uint16_t mFlags;
}; // class AttributeArray


////////////////////////////////////////


/// Templated attribute class to hold specific types
template<typename ValueType_, typename CompressionPolicy_ = NullAttributeCodec<ValueType_> >
class TypedAttributeArray: public AttributeArray
{
public:
    typedef boost::shared_ptr<TypedAttributeArray>          Ptr;
    typedef boost::shared_ptr<const TypedAttributeArray>    ConstPtr;

    typedef ValueType_                                      ValueType;
    typedef CompressionPolicy_                              CompressionPolicy;
    typedef typename CompressionPolicy::StorageType         StorageType;

    //////////

    /// Default constructor, always constructs a uniform attribute.
    explicit TypedAttributeArray(size_t n = 1,
        const ValueType& uniformValue = openvdb::zeroVal<ValueType>());
    /// Deep copy constructor
    TypedAttributeArray(const TypedAttributeArray&);
    //TypedAttributeArray& operator=(const TypedAttributeArray&); /// @todo

    virtual ~TypedAttributeArray() { this->deallocate(); }

    /// Return a copy of this attribute.
    virtual AttributeArray::Ptr copy() const;

    /// Return a new attribute array of the given length @a n with uniform value zero.
    static Ptr create(size_t n);

    /// Return the name of this attribute's type.
    static const openvdb::Name& attributeType();
    /// Return the name of this attribute's type.
    virtual const openvdb::Name& type() const { return attributeType(); }

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
    ValueType get(openvdb::Index n) const;
    /// Return the @a value at index @a n
    template<typename T> void get(openvdb::Index n, T& value) const;

    /// Set @a value at the given index @a n
    void set(openvdb::Index n, const ValueType& value);
    /// Set @a value at the given index @a n
    template<typename T> void set(openvdb::Index n, const T& value);

    /// Return @c true if this array is stored as a single uniform value.
    virtual bool isUniform() const { return mIsUniform; }

    /// @brief  Replace the the single value storage with a an array of length size().
    /// @note   Non-uniform attributes are unchanged.
    /// @param  fill toggle to initialize the array elements with the pre-expanded value.
    virtual void expand(bool fill = true);
    /// Replace the existing array with a uniform zero value.
    virtual void collapse();
    /// Replace the existing array with the given uniform value.
    void collapse(const ValueType& uniformValue);

    /// Read attribute data from a stream.
    virtual void read(std::istream& is);
    /// Write attribute data to a stream.
    virtual void write(std::ostream& os) const;

private:
    void allocate(bool fill = true);
    void deallocate();

    /// Helper function for use with registerType()
    static AttributeArray::Ptr factory(size_t n) { return TypedAttributeArray::create(n); }

    static tbb::atomic<const openvdb::Name*> sTypeName;
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

    class Iterator;
    class Descriptor;
    typedef boost::shared_ptr<Descriptor> DescriptorPtr;
    typedef boost::shared_ptr<const Descriptor> DescriptorConstPtr;

    //////////

    AttributeSet();
    AttributeSet(const AttributeSet&);
    explicit AttributeSet(const DescriptorPtr&);

    /// Update this attribute set to match the given descriptor.
    void update(const DescriptorPtr&);

    Descriptor& descriptor() { return *mDescr; }
    const Descriptor& descriptor() const { return *mDescr; }
    DescriptorPtr descriptorPtr() const { return mDescr; }

    /// Return the number of attributes in this set.
    size_t size() const { return mAttrs.size(); }
    /// Return the number of bytes of memory used by this attribute set.
    size_t memUsage() const;

    size_t find(const std::string& name) const;// { return mDescr->find(name); }

    /// @todo Enforce type constraints
    size_t replace(const std::string& name, const AttributeArray::Ptr&);
    size_t replace(size_t pos, const AttributeArray::Ptr&);

    const AttributeArray* getConst(const std::string& name) const;
    const AttributeArray* get(const std::string& name) const;
    AttributeArray*       get(const std::string& name);

    const AttributeArray* getConst(size_t pos) const;
    const AttributeArray* get(size_t pos) const;
    AttributeArray*       get(size_t pos);

    AttributeArray::Ptr   getSharedPtr(const std::string& name);
    AttributeArray::Ptr   getSharedPtr(size_t pos);

    bool isUnique(size_t pos) const;
    void makeUnique(size_t pos);

    void read(std::istream&);
    void write(std::ostream&) const;

private:
    typedef std::vector<AttributeArray::Ptr> AttrArrayVec;

    DescriptorPtr   mDescr;
    AttrArrayVec    mAttrs;
}; // class AttributeSet


////////////////////////////////////////


class AttributeSet::Descriptor
{
public:
    typedef boost::shared_ptr<Descriptor> Ptr;

    // The descriptor can optionally enforce value type and compression type
    // in the associated attribute sets.
    enum TypeConstraint {
        CSTR_NONE = 0,
        CSTR_VALUE_TYPE,    // Value type only
        CSTR_FULL_TYPE      // Value type and compression type
    };

    /// @todo Construct with type constraints
    //static Ptr create(const std::map<std::string, std::string>&, const TypePolicy& t = VALUE_TYPE_REQUIREMENT);


    static Ptr create(const std::set<std::string>&);
    // Construct from list of comma or space separated attribute names
    static Ptr create(const std::string& names);

    Descriptor(const TypeConstraint& cstr = CSTR_NONE)
        : mTypeConstraint(cstr), mId(sNextId++), mNextPos(0)
    {
    }

    size_t size() const { return mDictionary.size(); }
    size_t memUsage() const;

    size_t find(const std::string& name) const;

    size_t rename(const std::string& fromName, const std::string& toName);


    openvdb::Index64 id() const { return mId; }
    void write(std::ostream&) const { } /// @todo
    void read(std::istream&) { } /// @todo

    const TypeConstraint& typeConstraint() const { return mTypeConstraint; }

private:

    size_t insert(const std::string& name, const std::string& typeName = "");

    struct AttributeInfo { std::string typeName; size_t pos; };

    typedef std::map<std::string, AttributeInfo> AttrDictionary;

    const TypeConstraint mTypeConstraint;
    const openvdb::Index64        mId;
    size_t               mNextPos;
    AttrDictionary       mDictionary;

    static tbb::atomic<openvdb::Index64> sNextId;
}; // class Descriptor


/// @todo
tbb::atomic<openvdb::Index64> AttributeSet::Descriptor::sNextId;


////////////////////////////////////////


class AttributeSet::Iterator
{
public:
    Iterator() {}

private:
};


////////////////////////////////////////

// TypedAttributeArray implementation


template<typename ValueType_, typename CompressionPolicy_>
tbb::atomic<const openvdb::Name*> TypedAttributeArray<ValueType_, CompressionPolicy_>::sTypeName;


template<typename ValueType_, typename CompressionPolicy_>
TypedAttributeArray<ValueType_, CompressionPolicy_>::TypedAttributeArray(
    size_t n, const ValueType& uniformValue)
    : AttributeArray()
    , mData(new StorageType[1])
    , mSize(n)
    , mIsUniform(true)
    , mMutex()
{
    mSize = std::max(size_t(1), mSize);
    CompressionPolicy::encode(uniformValue, mData[0]);
}


template<typename ValueType_, typename CompressionPolicy_>
TypedAttributeArray<ValueType_, CompressionPolicy_>::TypedAttributeArray(const TypedAttributeArray& rhs)
    : AttributeArray(rhs)
    , mData(NULL)
    , mSize(rhs.mSize)
    , mIsUniform(rhs.mIsUniform)
    , mMutex()
{
    if (mIsUniform) {
        mData = new StorageType[1];
        mData[0] = rhs.mData[0];
    } else {
        mData = new StorageType[mSize];
        memcpy(mData, rhs.mData, mSize * sizeof(StorageType));
    }
}


template<typename ValueType_, typename CompressionPolicy_>
inline const openvdb::Name&
TypedAttributeArray<ValueType_, CompressionPolicy_>::attributeType()
{
    if (sTypeName == NULL) {
        std::ostringstream ostr;
        ostr << openvdb::typeNameAsString<ValueType>() << "_" << CompressionPolicy::name()
             << "_" << openvdb::typeNameAsString<StorageType>();
        openvdb::Name* s = new openvdb::Name(ostr.str());
        if (sTypeName.compare_and_swap(s, NULL) != NULL) delete s;
    }
    return *sTypeName;
}

template<typename ValueType_, typename CompressionPolicy_>
inline bool
TypedAttributeArray<ValueType_, CompressionPolicy_>::isRegistered()
{
    return AttributeArray::isRegistered(TypedAttributeArray::attributeType());
}


template<typename ValueType_, typename CompressionPolicy_>
inline void
TypedAttributeArray<ValueType_, CompressionPolicy_>::registerType()
{
    AttributeArray::registerType(TypedAttributeArray::attributeType(), TypedAttributeArray::factory);
}


template<typename ValueType_, typename CompressionPolicy_>
inline void
TypedAttributeArray<ValueType_, CompressionPolicy_>::unregisterType()
{
    AttributeArray::unregisterType(TypedAttributeArray::attributeType());
}


template<typename ValueType_, typename CompressionPolicy_>
inline typename TypedAttributeArray<ValueType_, CompressionPolicy_>::Ptr
TypedAttributeArray<ValueType_, CompressionPolicy_>::create(size_t n)
{
    return Ptr(new TypedAttributeArray(n));
}


template<typename ValueType_, typename CompressionPolicy_>
AttributeArray::Ptr
TypedAttributeArray<ValueType_, CompressionPolicy_>::copy() const
{
    return AttributeArray::Ptr(new TypedAttributeArray<ValueType, CompressionPolicy>(*this));
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::allocate(bool fill)
{
    tbb::spin_mutex::scoped_lock lock(mMutex);

    StorageType val = mIsUniform ? mData[0] : openvdb::zeroVal<StorageType>();

    if (mData) {
        delete mData;
        mData = NULL;
    }

    mIsUniform = false;
    mData = new StorageType[mSize];
    if (fill) {
        for (size_t i = 0; i < mSize; ++i) mData[i] = val;
    }
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::deallocate()
{
    if (mData) {
        delete mData;
        mData = NULL;
    }
}


template<typename ValueType_, typename CompressionPolicy_>
size_t
TypedAttributeArray<ValueType_, CompressionPolicy_>::memUsage() const
{
    return sizeof(*this) + (mIsUniform ? sizeof(StorageType) : (mSize * sizeof(StorageType)));
}


template<typename ValueType_, typename CompressionPolicy_>
typename TypedAttributeArray<ValueType_, CompressionPolicy_>::ValueType
TypedAttributeArray<ValueType_, CompressionPolicy_>::get(openvdb::Index n) const
{
    if (mIsUniform) n = 0;
    ValueType val;
    CompressionPolicy::decode(/*in=*/mData[n], /*out=*/val);
    return val;
}


template<typename ValueType_, typename CompressionPolicy_>
template<typename T>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::get(openvdb::Index n, T& val) const
{
    if (mIsUniform) n = 0;
    ValueType tmp;
    CompressionPolicy::decode(/*in=*/mData[n], /*out=*/tmp);
    val = static_cast<T>(tmp);
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::set(openvdb::Index n, const ValueType& val)
{
    if (mIsUniform) this->allocate();
    CompressionPolicy::encode(/*in=*/val, /*out=*/mData[n]);
}


template<typename ValueType_, typename CompressionPolicy_>
template<typename T>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::set(openvdb::Index n, const T& val)
{
    const ValueType tmp = static_cast<ValueType>(val);
    if (mIsUniform) this->allocate();
    CompressionPolicy::encode(/*in=*/tmp, /*out=*/mData[n]);
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::collapse()
{
    this->collapse(openvdb::zeroVal<ValueType>());
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::collapse(const ValueType& uniformValue)
{
    if (!mIsUniform) {
        this->deallocate();
        mData = new StorageType[1];
        mIsUniform = true;
    }
    CompressionPolicy::encode(uniformValue, mData[0]);
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::expand(bool fill)
{
    this->allocate(fill);
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::read(std::istream& is)
{
    openvdb::Int16 flags = openvdb::Int16(0);
    is.read(reinterpret_cast<char*>(&flags), sizeof(openvdb::Int16));
    mFlags = flags;

    openvdb::Int16 isUniform = openvdb::Int16(0);
    is.read(reinterpret_cast<char*>(&isUniform), sizeof(openvdb::Int16));
    mIsUniform = bool(isUniform);

    openvdb::Index64 arrayLength = openvdb::Index64(0);
    is.read(reinterpret_cast<char*>(&arrayLength), sizeof(openvdb::Index64));
    mSize = size_t(arrayLength);

    this->deallocate();

    size_t count = mIsUniform ? 1 : mSize;
    mData = new StorageType[count];
    is.read(reinterpret_cast<char*>(mData), count * sizeof(StorageType));
}


template<typename ValueType_, typename CompressionPolicy_>
void
TypedAttributeArray<ValueType_, CompressionPolicy_>::write(std::ostream& os) const
{
    if (!this->isTransient()) {
        os.write(reinterpret_cast<const char*>(&mFlags), sizeof(openvdb::Int16));

        openvdb::Int16 isUniform = openvdb::Int16(mIsUniform);
        os.write(reinterpret_cast<char*>(&isUniform), sizeof(openvdb::Int16));

        openvdb::Index64 arraylength = openvdb::Index64(mSize);
        os.write(reinterpret_cast<const char*>(&arraylength), sizeof(openvdb::Index64));

        size_t count = mIsUniform ? 1 : mSize;
        os.write(reinterpret_cast<const char*>(mData), count * sizeof(StorageType));
    }
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_POINTS_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED


// TM and (c) 2014 DreamWorks Animation LLC.  All Rights Reserved.
// Reproduction in whole or in part without prior written permission of a
// duly authorized representative is prohibited.
