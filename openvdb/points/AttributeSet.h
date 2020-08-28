// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/AttributeSet.h
///
/// @authors Dan Bailey, Mihai Alden
///
/// @brief  Set of Attribute Arrays which tracks metadata about each array.

#ifndef OPENVDB_POINTS_ATTRIBUTE_SET_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_ATTRIBUTE_SET_HAS_BEEN_INCLUDED

#include "AttributeArray.h"
#include <openvdb/version.h>
#include <openvdb/MetaMap.h>

#include <limits>
#include <memory>
#include <vector>


class TestAttributeSet;


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


using GroupType = uint8_t;


////////////////////////////////////////


/// Ordered collection of uniquely-named attribute arrays
class OPENVDB_API AttributeSet
{
public:
    enum { INVALID_POS = std::numeric_limits<size_t>::max() };

    using Ptr                   = std::shared_ptr<AttributeSet>;
    using ConstPtr              = std::shared_ptr<const AttributeSet>;
    using UniquePtr             = std::unique_ptr<AttributeSet>;

    class Descriptor;

    using DescriptorPtr         = std::shared_ptr<Descriptor>;
    using DescriptorConstPtr    = std::shared_ptr<const Descriptor>;

    //////////

    struct Util
    {
        /// Attribute and type name pair.
        struct NameAndType {
            NameAndType(const std::string& n, const NamePair& t, const Index s = 1)
                : name(n), type(t), stride(s) {}
            Name name;
            NamePair type;
            Index stride;
        };

        using NameAndTypeVec    = std::vector<NameAndType>;
        using NameToPosMap      = std::map<std::string, size_t>;
        using GroupIndex        = std::pair<size_t, uint8_t>;
    };

    //////////

    AttributeSet();

    /// Construct a new AttributeSet from the given AttributeSet.
    /// @param attributeSet the old attribute set
    /// @param arrayLength the desired length of the arrays in the new AttributeSet
    /// @param lock an optional scoped registry lock to avoid contention
    /// @note This constructor is typically used to resize an existing AttributeSet as
    ///       it transfers attribute metadata such as hidden and transient flags
    AttributeSet(const AttributeSet& attributeSet, Index arrayLength,
        const AttributeArray::ScopedRegistryLock* lock = nullptr);

    /// Construct a new AttributeSet from the given Descriptor.
    /// @param descriptor stored in the new AttributeSet and used in construction
    /// @param arrayLength the desired length of the arrays in the new AttributeSet
    /// @param lock an optional scoped registry lock to avoid contention
    /// @note Descriptors do not store attribute metadata such as hidden and transient flags
    ///       which live on the AttributeArrays, so for constructing from an existing AttributeSet
    ///       use the AttributeSet(const AttributeSet&, Index) constructor instead
    AttributeSet(const DescriptorPtr& descriptor, Index arrayLength = 1,
        const AttributeArray::ScopedRegistryLock* lock = nullptr);

    /// Shallow copy constructor, the descriptor and attribute arrays will be shared.
    AttributeSet(const AttributeSet&);

    /// Disallow copy assignment, since it wouldn't be obvious whether the copy is deep or shallow.
    AttributeSet& operator=(const AttributeSet&) = delete;

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

    /// Return the indices of the attribute arrays which are group attribute arrays
    std::vector<size_t> groupAttributeIndices() const;

    /// Return true if the attribute array stored at position @a pos is shared.
    bool isShared(size_t pos) const;
    /// @brief  If the attribute array stored at position @a pos is shared,
    ///         replace the array with a deep copy of itself that is not
    ///         shared with anyone else.
    void makeUnique(size_t pos);

    /// Append attribute @a attribute (simple method)
    AttributeArray::Ptr appendAttribute(const Name& name,
                                        const NamePair& type,
                                        const Index strideOrTotalSize = 1,
                                        const bool constantStride = true,
                                        const Metadata* defaultValue = nullptr);

    /// Append attribute @a attribute (descriptor-sharing)
    /// Requires current descriptor to match @a expected
    /// On append, current descriptor is replaced with @a replacement
    /// Provide a @a lock object to avoid contention from appending in parallel
    AttributeArray::Ptr appendAttribute(const Descriptor& expected, DescriptorPtr& replacement,
                                        const size_t pos, const Index strideOrTotalSize = 1,
                                        const bool constantStride = true,
                                        const Metadata* defaultValue = nullptr,
                                        const AttributeArray::ScopedRegistryLock* lock = nullptr);

    /// @brief Remove and return an attribute array by name
    /// @param name the name of the attribute array to release
    /// @details Detaches the attribute array from this attribute set and returns it, if
    /// @a name is invalid, returns an empty shared pointer. This also updates the descriptor
    /// to remove the reference to the attribute array.
    /// @note AttributeArrays are stored as shared pointers, so they are not guaranteed
    /// to be unique. Check the reference count before blindly re-using in a new AttributeSet.
    AttributeArray::Ptr removeAttribute(const Name& name);

    /// @brief Remove and return an attribute array by index
    /// @param pos the position index of the attribute to release
    /// @details Detaches the attribute array from this attribute set and returns it, if
    /// @a pos is invalid, returns an empty shared pointer. This also updates the descriptor
    /// to remove the reference to the attribute array.
    /// @note AttributeArrays are stored as shared pointers, so they are not guaranteed
    /// to be unique. Check the reference count before blindly re-using in a new AttributeSet.
    AttributeArray::Ptr removeAttribute(const size_t pos);

    /// @brief Remove and return an attribute array by index (unsafe method)
    /// @param pos the position index of the attribute to release
    /// @details Detaches the attribute array from this attribute set and returns it, if
    /// @a pos is invalid, returns an empty shared pointer.
    /// In cases where the AttributeSet is due to be destroyed, a small performance
    /// advantage can be gained by leaving the attribute array as a nullptr and not
    /// updating the descriptor. However, this leaves the AttributeSet in an invalid
    /// state making it unsafe to call any methods that implicitly derefence the attribute array.
    /// @note AttributeArrays are stored as shared pointers, so they are not guaranteed
    /// to be unique. Check the reference count before blindly re-using in a new AttributeSet.
    /// @warning Only use this method if you're an expert and know the risks of not
    /// updating the array of attributes or the descriptor.
    AttributeArray::Ptr removeAttributeUnsafe(const size_t pos);

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

    /// Replace the current descriptor with a @a replacement
    /// Note the provided Descriptor must be identical to the replacement
    /// unless @a allowMismatchingDescriptors is true (default is false)
    void resetDescriptor(const DescriptorPtr& replacement, const bool allowMismatchingDescriptors = false);

    /// Read the entire set from a stream.
    void read(std::istream&);
    /// Write the entire set to a stream.
    /// @param outputTransient if true, write out transient attributes
    void write(std::ostream&, bool outputTransient = false) const;

    /// This will read the attribute descriptor from a stream.
    void readDescriptor(std::istream&);
    /// This will write the attribute descriptor to a stream.
    /// @param outputTransient if true, write out transient attributes
    void writeDescriptor(std::ostream&, bool outputTransient = false) const;

    /// This will read the attribute metadata from a stream.
    void readMetadata(std::istream&);
    /// This will write the attribute metadata to a stream.
    /// @param outputTransient if true, write out transient attributes
    /// @param paged           if true, data is written out in pages
    void writeMetadata(std::ostream&, bool outputTransient = false, bool paged = false) const;

    /// This will read the attribute data from a stream.
    void readAttributes(std::istream&);
    /// This will write the attribute data to a stream.
    /// @param outputTransient if true, write out transient attributes
    void writeAttributes(std::ostream&, bool outputTransient = false) const;

    /// Compare the descriptors and attribute arrays on the attribute sets
    /// Exit early if the descriptors do not match
    bool operator==(const AttributeSet& other) const;
    bool operator!=(const AttributeSet& other) const { return !this->operator==(other); }

private:
    using AttrArrayVec = std::vector<AttributeArray::Ptr>;

    DescriptorPtr mDescr;
    AttrArrayVec  mAttrs;
}; // class AttributeSet

////////////////////////////////////////


/// A container for ABI=5 to help ease introduction of upcoming features
namespace future {
    class Container
    {
        class Element { };
        std::vector<std::shared_ptr<Element>> mElements;
    };
}


////////////////////////////////////////


/// @brief  An immutable object that stores name, type and AttributeSet position
///         for a constant collection of attribute arrays.
/// @note   The attribute name is actually mutable, but the attribute type
///         and position can not be changed after creation.
class OPENVDB_API AttributeSet::Descriptor
{
public:
    using Ptr               = std::shared_ptr<Descriptor>;

    using NameAndType       = Util::NameAndType;
    using NameAndTypeVec    = Util::NameAndTypeVec;
    using GroupIndex        = Util::GroupIndex;
    using NameToPosMap      = Util::NameToPosMap;
    using ConstIterator     = NameToPosMap::const_iterator;

    /// Utility method to construct a NameAndType sequence.
    struct Inserter {
        NameAndTypeVec vec;
        Inserter& add(const NameAndType& nameAndType) {
            vec.push_back(nameAndType); return *this;
        }
        Inserter& add(const Name& name, const NamePair& type) {
            vec.emplace_back(name, type); return *this;
        }
        Inserter& add(const NameAndTypeVec& other) {
            for (NameAndTypeVec::const_iterator it = other.begin(), itEnd = other.end(); it != itEnd; ++it) {
                vec.emplace_back(it->name, it->type);
            }
            return *this;
        }
    };

    //////////

    Descriptor();

    /// Copy constructor
    Descriptor(const Descriptor&);

    /// Create a new descriptor from a position attribute type and assumes "P" (for convenience).
    static Ptr create(const NamePair&);

    /// Create a new descriptor as a duplicate with a new attribute appended
    Ptr duplicateAppend(const Name& name, const NamePair& type) const;

    /// Create a new descriptor as a duplicate with existing attributes dropped
    Ptr duplicateDrop(const std::vector<size_t>& pos) const;

    /// Return the number of attributes in this descriptor.
    size_t size() const { return mTypes.size(); }

    /// Return the number of attributes with this attribute type
    size_t count(const NamePair& type) const;

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
    template<typename ValueType>
    ValueType getDefaultValue(const Name& name) const
    {
        const size_t pos = find(name);
        if (pos == INVALID_POS) {
            OPENVDB_THROW(LookupError, "Cannot find attribute name to set default value.")
        }

        std::stringstream ss;
        ss << "default:" << name;

        auto metadata = mMetadata.getMetadata<TypedMetadata<ValueType>>(ss.str());

        if (metadata)   return metadata->value();

        return zeroVal<ValueType>();
    }
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

    /// Return @c true if group exists
    bool hasGroup(const Name& group) const;
    /// @brief Define a group name to offset mapping
    /// @param group group name
    /// @param offset group offset
    /// @param checkValidOffset throws if offset out-of-range or in-use
    void setGroup(const Name& group, const size_t offset,
        const bool checkValidOffset = false);
    /// Drop any mapping keyed by group name
    void dropGroup(const Name& group);
    /// Clear all groups
    void clearGroups();
    /// Rename a group
    size_t renameGroup(const std::string& fromName, const std::string& toName);
    /// Return a unique name for a group based on given name
    const Name uniqueGroupName(const Name& name) const;

    //@{
    /// @brief Return the group offset from the name or index of the group
    /// A group attribute array is a single byte (8-bit), each bit of which
    /// can denote a group. The group offset is the position of the bit that
    /// denotes the requested group if all group attribute arrays in the set
    /// (and only attribute arrays marked as group) were to be laid out linearly
    /// according to their order in the set.
    size_t groupOffset(const Name& groupName) const;
    size_t groupOffset(const GroupIndex& index) const;
    //@}

    /// Return the group index from the name of the group
    GroupIndex groupIndex(const Name& groupName) const;
    /// Return the group index from the offset of the group
    /// @note see offset description for groupOffset()
    GroupIndex groupIndex(const size_t offset) const;

    /// Return number of bits occupied by a group attribute array
    static size_t groupBits() { return sizeof(GroupType) * CHAR_BIT; }

    /// Return the total number of available groups
    /// (group bits * number of group attributes)
    size_t availableGroups() const;

    /// Return the number of empty group slots which correlates to the number of groups
    /// that can be stored without increasing the number of group attribute arrays
    size_t unusedGroups() const;

    /// Return @c true if there are sufficient empty slots to allow compacting
    bool canCompactGroups() const;

    /// @brief Return a group offset that is not in use
    /// @param hint if provided, request a specific offset as a hint
    /// @return index of an offset or size_t max if no available group offsets
    size_t unusedGroupOffset(size_t hint = std::numeric_limits<size_t>::max()) const;

    /// @brief Determine if a move is required to efficiently compact the data and store the
    /// source name, offset and the target offset in the input parameters
    /// @param sourceName source name
    /// @param sourceOffset source offset
    /// @param targetOffset target offset
    /// @return @c true if move is required to compact the data
    bool requiresGroupMove(Name& sourceName, size_t& sourceOffset, size_t& targetOffset) const;

    /// @brief Test if there are any group names shared by both descriptors which
    /// have a different index
    /// @param rhs the descriptor to compare with
    /// @return @c true if an index collision exists
    bool groupIndexCollision(const Descriptor& rhs) const;

    /// Return a unique name for an attribute array based on given name
    const Name uniqueName(const Name& name) const;

    /// Return true if the name is valid
    static bool validName(const Name& name);

    /// @brief Extract each name from @a nameStr into @a includeNames, or into @a excludeNames
    /// if the name is prefixed with a caret.
    /// @param nameStr       the input string of names
    /// @param includeNames  on exit, the list of names that are not prefixed with a caret
    /// @param excludeNames  on exit, the list of names that are prefixed with a caret
    /// @param includeAll    on exit, @c true if a "*" wildcard is present in the @a includeNames
    static void parseNames( std::vector<std::string>& includeNames,
                            std::vector<std::string>& excludeNames,
                            bool& includeAll,
                            const std::string& nameStr);

    /// @brief Extract each name from @a nameStr into @a includeNames, or into @a excludeNames
    /// if the name is prefixed with a caret.
    static void parseNames( std::vector<std::string>& includeNames,
                            std::vector<std::string>& excludeNames,
                            const std::string& nameStr);

    /// Serialize this descriptor to the given stream.
    void write(std::ostream&) const;
    /// Unserialize this transform from the given stream.
    void read(std::istream&);

protected:
    /// Append to a vector of names and types from this Descriptor in position order
    void appendTo(NameAndTypeVec& attrs) const;

    /// Create a new descriptor from the given attribute and type name pairs
    /// and copy the group maps and metamap.
    static Ptr create(const NameAndTypeVec&, const NameToPosMap&, const MetaMap&);

    size_t insert(const std::string& name, const NamePair& typeName);

private:
    friend class ::TestAttributeSet;

    NameToPosMap                mNameMap;
    std::vector<NamePair>       mTypes;
    NameToPosMap                mGroupMap;
    MetaMap                     mMetadata;
    // as this change is part of an ABI change, there's no good reason to reduce the reserved
    // space aside from keeping the memory size of an AttributeSet the same for convenience
    // (note that this assumes a typical three-pointer implementation for std::vector)
    future::Container           mFutureContainer;   // occupies 3 reserved slots
    int64_t                     mReserved[5];       // for future use
}; // class Descriptor

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_ATTRIBUTE_ARRAY_HAS_BEEN_INCLUDED
