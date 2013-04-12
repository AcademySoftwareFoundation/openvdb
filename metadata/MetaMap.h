///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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

#ifndef OPENVDB_METADATA_METAMAP_HAS_BEEN_INCLUDED
#define OPENVDB_METADATA_METAMAP_HAS_BEEN_INCLUDED

#include <iosfwd>
#include <map>
#include <openvdb/metadata/Metadata.h>
#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief Provides functionality storing type agnostic metadata information.
/// Grids and other structures can inherit from this to attain metadata
/// functionality.
class OPENVDB_API MetaMap
{
public:
    typedef boost::shared_ptr<MetaMap> Ptr;
    typedef boost::shared_ptr<const MetaMap> ConstPtr;

    typedef std::map<Name, Metadata::Ptr> MetadataMap;
    typedef MetadataMap::iterator MetaIterator;
    typedef MetadataMap::const_iterator ConstMetaIterator;
        ///< @todo this should really iterate over a map of Metadata::ConstPtrs

    /// Constructor
    MetaMap() {}
    MetaMap(const MetaMap& other);

    /// Destructor
    virtual ~MetaMap() {}

    /// Return a copy of this map whose fields are shared with this map.
    MetaMap::Ptr copyMeta() const;
    /// Return a deep copy of this map that shares no data with this map.
    MetaMap::Ptr deepCopyMeta() const;

    /// Assign to this map a deep copy of another map.
    MetaMap& operator=(const MetaMap&);

    /// Read in all the Meta information the given stream.
    void readMeta(std::istream&);

    /// Write out all the Meta information to the given stream.
    void writeMeta(std::ostream&) const;

    /// Insert a new metadata or overwrite existing. If Metadata with given name
    /// doesn't exist, a new Metadata field is added. If it does exist and given
    /// metadata is of the same type, then overwrite existing with new value. If
    /// it does exist and not of the same type, then throw an exception.
    ///
    /// @param name the name of the metadata.
    /// @param metadata the actual metadata to store.
    void insertMeta(const Name& name, const Metadata& metadata);

    /// Removes an existing metadata field from the grid. If the metadata with
    /// the given name doesn't exist, do nothing.
    ///
    /// @param name the name of the metadata field to remove.
    void removeMeta(const Name &name);

    //@{
    /// @return a pointer to the metadata with the given name, NULL if no such
    /// field exists.
    Metadata::Ptr operator[](const Name&);
    Metadata::ConstPtr operator[](const Name&) const;
    //@}

    //@{
    /// @return pointer to TypedMetadata, NULL if type and name mismatch.
    template<typename T> typename T::Ptr getMetadata(const Name &name);
    template<typename T> typename T::ConstPtr getMetadata(const Name &name) const;
    //@}

    /// @return direct access to the underlying value stored by the given
    /// metadata name. Here T is the type of the value stored. If there is a
    /// mismatch, then throws an exception.
    template<typename T> T& metaValue(const Name &name);
    template<typename T> const T& metaValue(const Name &name) const;

    /// Functions for iterating over the Metadata.
    MetaIterator beginMeta() { return mMeta.begin(); }
    MetaIterator endMeta() { return mMeta.end(); }
    ConstMetaIterator beginMeta() const { return mMeta.begin(); }
    ConstMetaIterator endMeta() const { return mMeta.end(); }

    void clearMetadata() { mMeta.clear(); }

    size_t metaCount() const { return mMeta.size(); }

    bool empty() const { return mMeta.empty(); }

    /// @return string representation of MetaMap
    std::string str() const;

private:
    /// @return a pointer to TypedMetadata with the given template parameter.
    /// @throw LookupError if no field with the given name is found.
    /// @throw TypeError if the given field is not of type T.
    template<typename T>
    typename TypedMetadata<T>::Ptr getValidTypedMetadata(const Name&) const;

    MetadataMap mMeta;
};

/// Write a MetaMap to an output stream
std::ostream& operator<<(std::ostream&, const MetaMap&);


////////////////////////////////////////


inline Metadata::Ptr
MetaMap::operator[](const Name& name)
{
    MetaIterator iter = mMeta.find(name);
    return (iter == mMeta.end() ? Metadata::Ptr() : iter->second);
}

inline Metadata::ConstPtr
MetaMap::operator[](const Name &name) const
{
    ConstMetaIterator iter = mMeta.find(name);
    return (iter == mMeta.end() ? Metadata::Ptr() : iter->second);
}


////////////////////////////////////////


template <typename T>
inline typename T::Ptr
MetaMap::getMetadata(const Name &name)
{
    ConstMetaIterator iter = mMeta.find(name);
    if(iter == mMeta.end()) {
        return typename T::Ptr();
    }

    // To ensure that we get valid conversion if the metadata pointers cross dso
    // boundaries, we have to check the qualified typename and then do a static
    // cast. This is slower than doing a dynamic_pointer_cast, but is safer when
    // pointers cross dso boundaries.
    if (iter->second->typeName() == T::staticTypeName()) {
        return boost::static_pointer_cast<T, Metadata>(iter->second);
    } // else
    return typename T::Ptr();
}

template <typename T>
inline typename T::ConstPtr
MetaMap::getMetadata(const Name &name) const
{
    ConstMetaIterator iter = mMeta.find(name);
    if(iter == mMeta.end()) {
        return typename T::ConstPtr();
    }
    // To ensure that we get valid conversion if the metadata pointers cross dso
    // boundaries, we have to check the qualified typename and then do a static
    // cast. This is slower than doing a dynamic_pointer_cast, but is safer when
    // pointers cross dso boundaries.
    if (iter->second->typeName() == T::staticTypeName()) {
        return boost::static_pointer_cast<const T, const Metadata>(iter->second);
    } // else
    return typename T::ConstPtr();
}


////////////////////////////////////////


template <typename T>
inline typename TypedMetadata<T>::Ptr
MetaMap::getValidTypedMetadata(const Name &name) const
{
    ConstMetaIterator iter = mMeta.find(name);
    if (iter == mMeta.end()) OPENVDB_THROW(LookupError, "Cannot find metadata " << name);

    // To ensure that we get valid conversion if the metadata pointers cross dso
    // boundaries, we have to check the qualified typename and then do a static
    // cast. This is slower than doing a dynamic_pointer_cast, but is safer when
    // pointers cross dso boundaries.
    typename TypedMetadata<T>::Ptr m;
    if (iter->second->typeName() == TypedMetadata<T>::staticTypeName()) {
        m = boost::static_pointer_cast<TypedMetadata<T>, Metadata>(iter->second);
    }
    if (!m) OPENVDB_THROW(TypeError, "Invalid type for metadata " << name);
    return m;
}


////////////////////////////////////////


template <typename T>
inline T&
MetaMap::metaValue(const Name &name)
{
    typename TypedMetadata<T>::Ptr m = getValidTypedMetadata<T>(name);
    return m->value();
}


template <typename T>
inline const T&
MetaMap::metaValue(const Name &name) const
{
    typename TypedMetadata<T>::Ptr m = getValidTypedMetadata<T>(name);
    return m->value();
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_METADATA_METAMAP_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
