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
/// @file AttributeSet.h
///
/// @authors Dan Bailey, Mihai Alden, Peter Cucka
///
/// @brief  Set of Attribute Arrays which tracks metadata about each array.
///


#ifndef OPENVDB_TOOLS_ATTRIBUTE_SET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_ATTRIBUTE_SET_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/metadata/MetaMap.h>

#include <boost/integer_traits.hpp> // integer_traits
#include <boost/shared_ptr.hpp> // shared_ptr

#include <vector>
#include <cctype> // isalnum

#include <openvdb_points/tools/AttributeArray.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


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
            NameAndType(const std::string& n, const NamePair& t)
                : name(n), type(t) {}
            Name name;
            NamePair type;
        };

        typedef std::vector<NameAndType> NameAndTypeVec;
        typedef std::map<std::string, size_t> NameToPosMap;
        typedef std::pair<size_t, uint8_t> GroupIndex;
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

    //@{
    /// @brief Return the group offset from the name or index of the group
    /// A group attribute array is a single byte (8-bit), each bit of which
    /// can denote a group. The group offset is the position of the bit that
    /// denotes the requested group if all group attribute arrays in the set
    /// (and only attribute arrays marked as group) were to be laid out linearly
    /// according to their order in the set.
    size_t groupOffset(const Name& groupName) const;
    size_t groupOffset(const Util::GroupIndex& index) const;
    //@}

    /// Return the group index from the name of the group
    Util::GroupIndex groupIndex(const Name& groupName) const;
    /// Return the group index from the offset of the group
    /// @note see offset description for groupOffset()
    Util::GroupIndex groupIndex(const size_t offset) const;

    /// Create an iterator for iterating through point indices
    IndexIter beginIndex() const;

    /// Return true if the attribute array stored at position @a pos is shared.
    bool isShared(size_t pos) const;
    /// @brief  If the attribute array stored at position @a pos is shared,
    ///         replace the array with a deep copy of itself that is not
    ///         shared with anyone else.
    void makeUnique(size_t pos);

    /// Append attribute @a attribute (simple method)
    template <typename AttributeType>
    AttributeArray::Ptr appendAttribute(const Name& name,
                                        Metadata::Ptr defaultValue = Metadata::Ptr());

    /// Append attribute @a attribute (descriptor-sharing)
    /// Requires current descriptor to match @a expected
    /// On append, current descriptor is replaced with @a replacement
    template <typename AttributeType>
    AttributeArray::Ptr appendAttribute(const Descriptor& expected, DescriptorPtr& replacement);

    /// Drop attributes with @a pos indices (simple method)
    /// Creates a new descriptor for this attribute set
    void dropAttributes(const std::vector<size_t>& pos);

    /// Drop attributes with @a pos indices (descriptor-sharing method)
    /// Requires current descriptor to match @a expected
    /// On drop, current descriptor is replaced with @a replacement
    void dropAttributes(const std::vector<size_t>& pos,
                        const Descriptor& expected, DescriptorPtr& replacement);

    /// Re-name attributes in set to match a provided descriptor
    /// Replaces own descriptor with @a replacement
    void renameAttributes(const Descriptor& expected, const DescriptorPtr& replacement);

    /// Re order attribute set to match a provided descriptor
    /// Replaces own descriptor with @a replacement
    void reorderAttributes(const DescriptorPtr& replacement);

    /// Swap current descriptor with a @a replacement
    /// Note the provided Descriptor must be identical to the replacement
    void resetDescriptor(const DescriptorPtr& replacement);

    /// Read the entire set from a stream.
    void read(std::istream&);
    /// Write the entire set to a stream.
    /// @param outputTransient if true, write out transient attributes
    void write(std::ostream&, bool outputTransient = false) const;

    /// This will read the attribute descriptor from a stream, but no attribute data.
    void readMetadata(std::istream&);
    /// This will write the attribute descriptor to a stream, but no attribute data.
    /// @param outputTransient if true, write out transient attributes
    void writeMetadata(std::ostream&, bool outputTransient = false) const;

    /// Read attribute data from a stream.
    void readAttributes(std::istream&);
    /// Write attribute data to a stream.
    /// @param outputTransient if true, write out transient attributes
    void writeAttributes(std::ostream&, bool outputTransient = false) const;

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

    typedef Util::NameAndType             NameAndType;
    typedef Util::NameAndTypeVec          NameAndTypeVec;
    typedef Util::GroupIndex              GroupIndex;
    typedef Util::NameToPosMap            NameToPosMap;
    typedef NameToPosMap::const_iterator  ConstIterator;

    /// Utility method to construct a NameAndType sequence.
    struct Inserter {
        NameAndTypeVec vec;
        Inserter& add(const NameAndType& nameAndType) {
            vec.push_back(nameAndType); return *this;
        }
        Inserter& add(const Name& name, const NamePair& type) {
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

    /// Copy constructor
    Descriptor(const Descriptor&);

    /// Create a new descriptor from the given attribute and type name pairs.
    static Ptr create(const NameAndTypeVec&);

    /// Create a new descriptor from the given attribute and type name pairs
    /// and copy the group maps and metamap.
    static Ptr create(const NameAndTypeVec&, const NameToPosMap&, const MetaMap&);

    /// Create a new descriptor from a position attribute type and assumes "P" (for convenience).
    static Ptr create(const NamePair&);

    /// Create a new descriptor as a duplicate with a new attribute appended
    Ptr duplicateAppend(const NameAndType& attribute) const;

    /// Create a new descriptor as a duplicate with existing attributes dropped
    Ptr duplicateDrop(const std::vector<size_t>& pos) const;

    /// Return the number of attributes in this descriptor.
    size_t size() const { return mTypes.size(); }

    /// Return the number of attributes with this attribute type
    template <typename AttributeArrayType>
    size_t count() const;

    /// Return the number of bytes of memory used by this attribute set.
    size_t memUsage() const;

    /// @brief  Return the position of the attribute array whose name is @a name,
    ///         or @c INVALID_POS if no match is found.
    size_t find(const std::string& name) const;

    /// Rename an attribute array
    size_t rename(const std::string& fromName, const std::string& toName);

    /// Return the name of the attribute array's type.
    const Name& valueType(size_t pos) const;
    /// Return the name of the attribute array's type.
    const NamePair& type(size_t pos) const;

    /// Retrieve metadata map
    MetaMap& getMetadata();
    const MetaMap& getMetadata() const;

    /// Return true if the attribute has a default value
    bool hasDefaultValue(const Name& name) const;
    /// Get a default value for an existing attribute
    template <typename ValueType>
    ValueType getDefaultValue(const Name& name) const;
    /// Set a default value for an existing attribute
    void setDefaultValue(const Name& name, const Metadata& defaultValue);
    // Remove the default value if it exists
    void removeDefaultValue(const Name& name);
    // Prune any default values for which the key is no longer present
    void pruneUnusedDefaultValues();

    /// Return true if this descriptor is equal to the given one.
    bool operator==(const Descriptor&) const;
    /// Return true if this descriptor is not equal to the given one.
    bool operator!=(const Descriptor& rhs) const { return !this->operator==(rhs); }
    /// Return true if this descriptor contains the same attributes
    /// as the given descriptor, ignoring attribute order
    bool hasSameAttributes(const Descriptor& rhs) const;

    /// Return a reference to the name-to-position map.
    const NameToPosMap& map() const { return mNameMap; }
    /// Return a reference to the name-to-position group map.
    const NameToPosMap& groupMap() const { return mGroupMap; }

    /// Append to a vector of names and types from this Descriptor in position order
    void appendTo(NameAndTypeVec& attrs) const;

    /// Return @c true if group exists
    bool hasGroup(const Name& group) const;
    /// Define a group name to offset mapping
    void setGroup(const Name& group, const size_t offset);
    /// Drop any mapping keyed by group name
    void dropGroup(const Name& group);
    /// Clear all groups
    void clearGroups();

    /// Return a unique name for an attribute array based on given name
    const Name uniqueName(const Name& name) const;

    /// Return true if the name is valid
    static bool validName(const Name& name);

    /// Extract each name from nameStr into includeNames, or into excludeNames if name prefixed with caret
    static void parseNames( std::vector<std::string>& includeNames,
                            std::vector<std::string>& excludeNames,
                            const std::string& nameStr);

    /// Serialize this descriptor to the given stream.
    void write(std::ostream&) const;
    /// Unserialize this transform from the given stream.
    void read(std::istream&);

private:
    size_t insert(const std::string& name, const NamePair& typeName);

    NameToPosMap                mNameMap;
    std::vector<NamePair>       mTypes;
    NameToPosMap                mGroupMap;
    MetaMap                     mMetadata;
}; // class Descriptor


template <typename AttributeArrayType>
size_t AttributeSet::Descriptor::count() const
{
    size_t count = 0;
    for (std::vector<NamePair>::const_iterator  it = mTypes.begin(),
                                                itEnd = mTypes.end(); it != itEnd; ++it) {
        const NamePair& type = *it;
        if (type == AttributeArrayType::attributeType())    count++;
    }
    return count;
}


template <typename AttributeType>
AttributeArray::Ptr
AttributeSet::appendAttribute(  const Name& name,
                                Metadata::Ptr defaultValue)
{
    AttributeSet::Util::NameAndType nameAndType(name, AttributeType::attributeType());

    Descriptor::Ptr descriptor = mDescr->duplicateAppend(nameAndType);

    // store the attribute default value in the descriptor metadata
    if (defaultValue)   descriptor->setDefaultValue(name, *defaultValue);

    return this->appendAttribute<AttributeType>(*mDescr, descriptor);
}


template <typename AttributeType>
AttributeArray::Ptr
AttributeSet::appendAttribute(const Descriptor& expected, DescriptorPtr& replacement)
{
    // ensure the descriptor is as expected
    if (*mDescr != expected) {
        OPENVDB_THROW(LookupError, "Cannot append attributes as descriptors do not match.")
    }

    const size_t offset = mDescr->size();

    mDescr = replacement;

    assert(mDescr->size() >= offset);

    // extract the array length from the first attribute array if it exists

    const size_t arrayLength = offset > 0 ? this->get(0)->size() : 1;

    // append the new array

    AttributeArray::Ptr array = AttributeType::create(arrayLength);

    mAttrs.push_back(array);

    return array;
}


template <typename ValueType>
ValueType
AttributeSet::Descriptor::getDefaultValue(const Name& name) const
{
    typedef typename TypedMetadata<ValueType>::ConstPtr MetadataPtr;

    const size_t pos = find(name);
    if (pos == INVALID_POS) {
        OPENVDB_THROW(LookupError, "Cannot find attribute name to set default value.")
    }

    std::stringstream ss;
    ss << "default:" << name;

    MetadataPtr metadata = mMetadata.getMetadata<TypedMetadata<ValueType> >(ss.str());

    if (metadata)   return metadata->value();

    return zeroVal<ValueType>();
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )

