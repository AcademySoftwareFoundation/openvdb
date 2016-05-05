///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#ifndef OPENVDB_METADATA_METADATA_HAS_BEEN_INCLUDED
#define OPENVDB_METADATA_METADATA_HAS_BEEN_INCLUDED

#include <iostream>
#include <string>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h> // for math::isZero()
#include <openvdb/util/Name.h>
#include <openvdb/Exceptions.h>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief Base class for storing metadata information in a grid.
class OPENVDB_API Metadata
{
public:
    typedef boost::shared_ptr<Metadata> Ptr;
    typedef boost::shared_ptr<const Metadata> ConstPtr;

    Metadata() {}
    virtual ~Metadata() {}

    /// Return the type name of the metadata.
    virtual Name typeName() const = 0;

    /// Return a copy of the metadata.
    virtual Metadata::Ptr copy() const = 0;

    /// Copy the given metadata into this metadata.
    virtual void copy(const Metadata& other) = 0;

    /// Return a textual representation of this metadata.
    virtual std::string str() const = 0;

    /// Return the boolean representation of this metadata (empty strings
    /// and zeroVals evaluate to false; most other values evaluate to true).
    virtual bool asBool() const = 0;

    /// Return @c true if the given metadata is equivalent to this metadata.
    bool operator==(const Metadata& other) const;
    /// Return @c true if the given metadata is different from this metadata.
    bool operator!=(const Metadata& other) const { return !(*this == other); }

    /// Return the size of this metadata in bytes.
    virtual Index32 size() const = 0;

    /// Unserialize this metadata from a stream.
    void read(std::istream&);
    /// Serialize this metadata to a stream.
    void write(std::ostream&) const;

    /// Create new metadata of the given type.
    static Metadata::Ptr createMetadata(const Name& typeName);

    /// Return @c true if the given type is known by the metadata type registry.
    static bool isRegisteredType(const Name& typeName);

    /// Clear out the metadata registry.
    static void clearRegistry();

    /// Register the given metadata type along with a factory function.
    static void registerType(const Name& typeName, Metadata::Ptr (*createMetadata)());
    static void unregisterType(const Name& typeName);

protected:
    /// Read the size of the metadata from a stream.
    static Index32 readSize(std::istream&);
    /// Write the size of the metadata to a stream.
    void writeSize(std::ostream&) const;

    /// Read the metadata from a stream.
    virtual void readValue(std::istream&, Index32 numBytes) = 0;
    /// Write the metadata to a stream.
    virtual void writeValue(std::ostream&) const = 0;

private:
    // Disallow copying of instances of this class.
    Metadata(const Metadata&);
    Metadata& operator=(const Metadata&);
};


/// @brief Subclass to read (and ignore) data of an unregistered type
class OPENVDB_API UnknownMetadata: public Metadata
{
public:
    UnknownMetadata() {}
    virtual ~UnknownMetadata() {}
    virtual Name typeName() const { return "<unknown>"; }
    virtual Metadata::Ptr copy() const { OPENVDB_THROW(TypeError, "Metadata has unknown type"); }
    virtual void copy(const Metadata&) { OPENVDB_THROW(TypeError, "Destination has unknown type"); }
    virtual std::string str() const { return "<unknown>"; }
    virtual bool asBool() const { return false; }
    virtual Index32 size() const { return 0; }

protected:
    virtual void readValue(std::istream&s, Index32 numBytes);
    virtual void writeValue(std::ostream&) const;
};


/// @brief Templated metadata class to hold specific types.
template<typename T>
class TypedMetadata: public Metadata
{
public:
    typedef boost::shared_ptr<TypedMetadata<T> > Ptr;
    typedef boost::shared_ptr<const TypedMetadata<T> > ConstPtr;

    TypedMetadata();
    TypedMetadata(const T& value);
    TypedMetadata(const TypedMetadata<T>& other);
    virtual ~TypedMetadata();

    virtual Name typeName() const;
    virtual Metadata::Ptr copy() const;
    virtual void copy(const Metadata& other);
    virtual std::string str() const;
    virtual bool asBool() const;
    virtual Index32 size() const { return static_cast<Index32>(sizeof(T)); }

    /// Set this metadata's value.
    void setValue(const T&);
    /// Return this metadata's value.
    T& value();
    const T& value() const;

    // Static specialized function for the type name. This function must be
    // template specialized for each type T.
    static Name staticTypeName() { return typeNameAsString<T>(); }

    /// Create new metadata of this type.
    static Metadata::Ptr createMetadata();

    static void registerType();
    static void unregisterType();
    static bool isRegisteredType();

protected:
    virtual void readValue(std::istream&, Index32 numBytes);
    virtual void writeValue(std::ostream&) const;

private:
    T mValue;
};

/// Write a Metadata to an output stream
std::ostream& operator<<(std::ostream& ostr, const Metadata& metadata);


////////////////////////////////////////


inline void
Metadata::writeSize(std::ostream& os) const
{
    const Index32 n = this->size();
    os.write(reinterpret_cast<const char*>(&n), sizeof(Index32));
}


inline Index32
Metadata::readSize(std::istream& is)
{
    Index32 n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(Index32));
    return n;
}


inline void
Metadata::read(std::istream& is)
{
    const Index32 numBytes = this->readSize(is);
    this->readValue(is, numBytes);
}


inline void
Metadata::write(std::ostream& os) const
{
    this->writeSize(os);
    this->writeValue(os);
}


////////////////////////////////////////


template <typename T>
inline
TypedMetadata<T>::TypedMetadata() : mValue(T())
{
}

template <typename T>
inline
TypedMetadata<T>::TypedMetadata(const T &value) : mValue(value)
{
}

template <typename T>
inline
TypedMetadata<T>::TypedMetadata(const TypedMetadata<T> &other) :
    Metadata(),
    mValue(other.mValue)
{
}

template <typename T>
inline
TypedMetadata<T>::~TypedMetadata()
{
}

template <typename T>
inline Name
TypedMetadata<T>::typeName() const
{
    return TypedMetadata<T>::staticTypeName();
}

template <typename T>
inline void
TypedMetadata<T>::setValue(const T& val)
{
    mValue = val;
}

template <typename T>
inline T&
TypedMetadata<T>::value()
{
    return mValue;
}

template <typename T>
inline const T&
TypedMetadata<T>::value() const
{
    return mValue;
}

template <typename T>
inline Metadata::Ptr
TypedMetadata<T>::copy() const
{
    Metadata::Ptr metadata(new TypedMetadata<T>());
    metadata->copy(*this);
    return metadata;
}

template <typename T>
inline void
TypedMetadata<T>::copy(const Metadata &other)
{
    const TypedMetadata<T>* t = dynamic_cast<const TypedMetadata<T>*>(&other);
    if (t == NULL) OPENVDB_THROW(TypeError, "Incompatible type during copy");
    mValue = t->mValue;
}


template<typename T>
inline void
TypedMetadata<T>::readValue(std::istream& is, Index32 /*numBytes*/)
{
    //assert(this->size() == numBytes);
    is.read(reinterpret_cast<char*>(&mValue), this->size());
}

template<typename T>
inline void
TypedMetadata<T>::writeValue(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&mValue), this->size());
}

template <typename T>
inline std::string
TypedMetadata<T>::str() const
{
    std::ostringstream ostr;
    ostr << mValue;
    return ostr.str();
}

template<typename T>
inline bool
TypedMetadata<T>::asBool() const
{
    return !math::isZero(mValue);
}

template <typename T>
inline Metadata::Ptr
TypedMetadata<T>::createMetadata()
{
    Metadata::Ptr ret(new TypedMetadata<T>());
    return ret;
}

template <typename T>
inline void
TypedMetadata<T>::registerType()
{
    Metadata::registerType(TypedMetadata<T>::staticTypeName(),
                           TypedMetadata<T>::createMetadata);
}

template <typename T>
inline void
TypedMetadata<T>::unregisterType()
{
    Metadata::unregisterType(TypedMetadata<T>::staticTypeName());
}

template <typename T>
inline bool
TypedMetadata<T>::isRegisteredType()
{
    return Metadata::isRegisteredType(TypedMetadata<T>::staticTypeName());
}


template<>
inline std::string
TypedMetadata<bool>::str() const
{
    return (mValue ? "true" : "false");
}


inline std::ostream&
operator<<(std::ostream& ostr, const Metadata& metadata)
{
    ostr << metadata.str();
    return ostr;
}


typedef TypedMetadata<bool>            BoolMetadata;
typedef TypedMetadata<double>          DoubleMetadata;
typedef TypedMetadata<float>           FloatMetadata;
typedef TypedMetadata<boost::int32_t>  Int32Metadata;
typedef TypedMetadata<boost::int64_t>  Int64Metadata;
typedef TypedMetadata<Vec2d>           Vec2DMetadata;
typedef TypedMetadata<Vec2i>           Vec2IMetadata;
typedef TypedMetadata<Vec2s>           Vec2SMetadata;
typedef TypedMetadata<Vec3d>           Vec3DMetadata;
typedef TypedMetadata<Vec3i>           Vec3IMetadata;
typedef TypedMetadata<Vec3s>           Vec3SMetadata;
typedef TypedMetadata<Mat4s>           Mat4SMetadata;
typedef TypedMetadata<Mat4d>           Mat4DMetadata;

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_METADATA_METADATA_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
