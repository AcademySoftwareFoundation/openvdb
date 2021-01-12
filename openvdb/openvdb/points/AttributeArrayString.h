// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/AttributeArrayString.h
///
/// @author Dan Bailey
///
/// @brief  Attribute array storage for string data using Descriptor Metadata.

#ifndef OPENVDB_POINTS_ATTRIBUTE_ARRAY_STRING_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_ATTRIBUTE_ARRAY_STRING_HAS_BEEN_INCLUDED

#include "AttributeArray.h"
#include <memory>
#include <deque>
#include <unordered_map>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


////////////////////////////////////////


namespace attribute_traits
{
    template <bool Truncate> struct StringTypeTrait { using Type = Index; };
    template<> struct StringTypeTrait</*Truncate=*/true> { using Type = uint16_t; };
}


template <bool Truncate>
struct StringCodec
{
    using ValueType = Index;

    template <typename T>
    struct Storage { using Type = typename attribute_traits::StringTypeTrait<Truncate>::Type; };

    template<typename StorageType> static void decode(const StorageType&, ValueType&);
    template<typename StorageType> static void encode(const ValueType&, StorageType&);
    static const char* name() { return Truncate ? "str_trnc" : "str"; }
};


using StringAttributeArray = TypedAttributeArray<Index, StringCodec<false>>;


////////////////////////////////////////


/// Class to compute a string->index map from all string:N metadata
class OPENVDB_API StringMetaCache
{
public:
    using UniquePtr = std::unique_ptr<StringMetaCache>;
    using ValueMap = std::unordered_map<Name, Index>;

    StringMetaCache() = default;
    explicit StringMetaCache(const MetaMap& metadata);

    /// Return @c true if no string elements in metadata
    bool empty() const { return mCache.empty(); }
    /// Returns the number of string elements in metadata
    size_t size() const { return mCache.size(); }

    /// Clears and re-populates the cache
    void reset(const MetaMap& metadata);

    /// Insert a new element in the cache
    void insert(const Name& key, Index index);

    /// Retrieve the value map (string -> index)
    const ValueMap& map() const { return mCache; }

private:
    ValueMap mCache;
}; // StringMetaCache


////////////////////////////////////////


/// Class to help with insertion of keyed string values into metadata
class OPENVDB_API StringMetaInserter
{
public:
    using UniquePtr = std::unique_ptr<StringMetaInserter>;

    explicit StringMetaInserter(MetaMap& metadata);

    /// Returns @c true if key exists
    bool hasKey(const Name& key) const;
    /// Returns @c true if index exists
    bool hasIndex(Index index) const;

    /// @brief Insert the string into the metadata using the hint if non-zero
    /// @param name the string to insert
    /// @param hint requested index to use if non-zero and not already in use
    /// @note the hint can be used to insert non-sequentially so as to avoid an
    /// expensive re-indexing of string keys
    /// @return the chosen index which will match hint if the hint was used
    Index insert(const Name& name, Index hint = Index(0));

    /// Reset the cache from the metadata
    void resetCache();

private:
    using IndexPairArray = std::deque<std::pair<Index, Index>>;

    MetaMap& mMetadata;
    IndexPairArray mIdBlocks;
    StringMetaCache mCache;
}; // StringMetaInserter


////////////////////////////////////////


template <bool Truncate>
template<typename StorageType>
inline void
StringCodec<Truncate>::decode(const StorageType& data, ValueType& val)
{
    val = static_cast<ValueType>(data);
}


template <bool Truncate>
template<typename StorageType>
inline void
StringCodec<Truncate>::encode(const ValueType& val, StorageType& data)
{
    data = static_cast<ValueType>(val);
}


////////////////////////////////////////


inline bool isString(const AttributeArray& array)
{
    return array.isType<StringAttributeArray>();
}


////////////////////////////////////////


class OPENVDB_API StringAttributeHandle
{
public:
    using Ptr = std::shared_ptr<StringAttributeHandle>;//SharedPtr<StringAttributeHandle>;
    using UniquePtr = std::unique_ptr<StringAttributeHandle>;

    static Ptr create(const AttributeArray& array, const MetaMap& metadata, const bool preserveCompression = true);

    StringAttributeHandle(  const AttributeArray& array,
                            const MetaMap& metadata,
                            const bool preserveCompression = true);

    Index stride() const { return mHandle.stride(); }
    Index size() const { return mHandle.size(); }

    bool isUniform() const { return mHandle.isUniform(); }
    bool hasConstantStride() const { return mHandle.hasConstantStride(); }

    Name get(Index n, Index m = 0) const;
    void get(Name& name, Index n, Index m = 0) const;

    /// @brief Returns a reference to the array held in the Handle.
    const AttributeArray& array() const;

protected:
    AttributeHandle<Index, StringCodec<false>>    mHandle;
    const MetaMap&                                          mMetadata;
}; // class StringAttributeHandle


////////////////////////////////////////


class OPENVDB_API StringAttributeWriteHandle : public StringAttributeHandle
{
public:
    using Ptr = std::shared_ptr<StringAttributeWriteHandle>;//SharedPtr<StringAttributeWriteHandle>;
    using UniquePtr = std::unique_ptr<StringAttributeWriteHandle>;

    static Ptr create(AttributeArray& array, const MetaMap& metadata, const bool expand = true);

    StringAttributeWriteHandle( AttributeArray& array,
                                const MetaMap& metadata,
                                const bool expand = true);

    /// @brief  If this array is uniform, replace it with an array of length size().
    /// @param  fill if true, assign the uniform value to each element of the array.
    void expand(bool fill = true);

    /// @brief Set membership for the whole array and attempt to collapse
    void collapse();
    /// @brief Set membership for the whole array and attempt to collapse
    /// @param name Name of the String
    void collapse(const Name& name);

    /// Compact the existing array to become uniform if all values are identical
    bool compact();

    /// @brief Fill the existing array with the given value.
    /// @note Identical to collapse() except a non-uniform array will not become uniform.
    void fill(const Name& name);

    /// Set the value of the index to @a name
    void set(Index n, const Name& name);
    void set(Index n, Index m, const Name& name);

    /// Reset the value cache from the metadata
    void resetCache();

    /// @brief Returns a reference to the array held in the Write Handle.
    AttributeArray& array();

    /// @brief  Returns whether or not the metadata cache contains a given value.
    /// @param  name Name of the String.
    bool contains(const Name& name) const;

private:
    /// Retrieve the index of this string value from the cache
    /// @note throws if name does not exist in cache
    Index getIndex(const Name& name) const;

    StringMetaCache                                     mCache;
    AttributeWriteHandle<Index, StringCodec<false>>     mWriteHandle;
}; // class StringAttributeWriteHandle


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_ATTRIBUTE_ARRAY_STRING_HAS_BEEN_INCLUDED

