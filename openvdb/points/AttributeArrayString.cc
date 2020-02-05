// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/AttributeArrayString.cc

#include "AttributeArrayString.h"

#include <openvdb/Metadata.h>
#include <openvdb/MetaMap.h>

#include <tbb/parallel_sort.h>

#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


namespace {

    Name getStringKey(const Index index)
    {
        return "string:" + std::to_string(index - 1);
    }

} // namespace


////////////////////////////////////////


// StringMetaCache implementation


StringMetaCache::StringMetaCache(const MetaMap& metadata)
{
    this->reset(metadata);
}


void StringMetaCache::insert(const Name& key, Index index)
{
    mCache[key] = index;
}


void StringMetaCache::reset(const MetaMap& metadata)
{
    mCache.clear();

    // populate the cache

    for (auto it = metadata.beginMeta(), itEnd = metadata.endMeta(); it != itEnd; ++it) {
        const Name& key = it->first;
        const Metadata::Ptr& meta = it->second;

        // attempt to cast metadata to StringMetadata
        const StringMetadata* stringMeta = dynamic_cast<StringMetadata*>(meta.get());
        if (!stringMeta)                            continue;

        // string attribute metadata must have a key that starts "string:"
        if (key.compare(0, 7, "string:") != 0)      continue;

        // remove "string:" and cast to Index
        Index index = 1 + static_cast<Index>(
            std::stoul(key.substr(7, key.size() - 7)));

        // add to the cache
        this->insert(stringMeta->value(), index);
    }
}


////////////////////////////////////////

// StringMetaInserter implementation


StringMetaInserter::StringMetaInserter(MetaMap& metadata)
    : mMetadata(metadata)
    , mIdBlocks()
    , mCache()
{
    // populate the cache
    resetCache();
}


bool StringMetaInserter::hasKey(const Name& key) const
{
    return mCache.map().find(key) != mCache.map().end();
}


bool StringMetaInserter::hasIndex(Index index) const
{
    return bool(mMetadata[getStringKey(index)]);
}


Index StringMetaInserter::insert(const Name& name, Index hint)
{
    using IterT = IndexPairArray::iterator;

    // if name already exists, return the index

    const auto& cacheMap = mCache.map();
    auto it = cacheMap.find(name);
    if (it != cacheMap.end()) {
        return it->second;
    }

    Index index = 1;

    Name hintKey;
    bool canUseHint = false;

    // hint must be non-zero to have been requested

    if (hint > Index(0)) {
        hintKey = getStringKey(hint);
        // check if hint is already in use
        if (!bool(mMetadata[hintKey])) {
            canUseHint = true;
            index = hint;
        }
    }

    // look through the id blocks for hint or index

    IterT iter = mIdBlocks.begin();
    for (; iter != mIdBlocks.end(); ++iter) {
        const Index start = iter->first;
        const Index end = start + iter->second;

        if (index < start || index >= end) break;
        if (!canUseHint)    index = end;
    }

    // index now holds the next valid index. if it's 1 (the beginning
    // iterator) no initial block exists - add it

    IterT prevIter;
    if (iter == mIdBlocks.begin()) {
        prevIter = mIdBlocks.emplace(iter, 1, 1);
        iter = std::next(prevIter);
    }
    else {
        // accumulate the id block size where the next index is going
        prevIter = std::prev(iter);
        prevIter->second++;
    }

    // see if this block and the next block can be compacted

    if (iter != mIdBlocks.end() &&
        prevIter->second + 1 == iter->first) {
        prevIter->second += iter->second;
        mIdBlocks.erase(iter);
    }

    // insert into metadata

    const Name key = getStringKey(index);
    mMetadata.insertMeta(key, StringMetadata(name));

    // update the cache

    mCache.insert(name, index);

    return index;
}


void StringMetaInserter::resetCache()
{
    mCache.reset(mMetadata);
    mIdBlocks.clear();

    std::vector<Index> stringIndices;
    stringIndices.reserve(mCache.size());

    if (mCache.empty()) return;

    const auto& cacheMap = mCache.map();

    for (auto it = cacheMap.cbegin(); it != cacheMap.cend(); ++it) {
        const Index index = it->second;

        stringIndices.emplace_back(index);
    }

    tbb::parallel_sort(stringIndices.begin(), stringIndices.end());

    // bucket string indices

    Index key = stringIndices.front();
    Index size = 0;

    // For each id, see if it's adjacent id is sequentially increasing and continue to
    // track how many are until we find a value that isn't. Store the start and length
    // of each of these blocks. For example, the following container could be created
    // consisting of 3 elements:
    //   key  ->  size
    //   -------------
    //   7    ->  1000  (values 7->1007)
    //   1020 ->  5     (values 1020->1025)
    //   2013 ->  30    (values 2013->2043)
    // Note that the end value is exclusive (values 1007, 1025 and 2043 do not exist
    // given the above example)

    for (const Index id : stringIndices) {
        if (key + size != id) {
            assert(size > 0);
            mIdBlocks.emplace_back(key, size);
            size = 0;
            key = id;
        }
        ++size;
    }

    // add the last block
    mIdBlocks.emplace_back(key, size);
}


////////////////////////////////////////

// StringAttributeHandle implementation


StringAttributeHandle::Ptr
StringAttributeHandle::create(const AttributeArray& array, const MetaMap& metadata, const bool preserveCompression)
{
    return std::make_shared<StringAttributeHandle>(array, metadata, preserveCompression);
}


StringAttributeHandle::StringAttributeHandle(const AttributeArray& array,
                                             const MetaMap& metadata,
                                             const bool preserveCompression)
        : mHandle(array, preserveCompression)
        , mMetadata(metadata)
{
    if (!isString(array)) {
        OPENVDB_THROW(TypeError, "Cannot create a StringAttributeHandle for an attribute array that is not a string.");
    }
}


Name StringAttributeHandle::get(Index n, Index m) const
{
    Name name;
    this->get(name, n, m);
    return name;
}


void StringAttributeHandle::get(Name& name, Index n, Index m) const
{
    Index index = mHandle.get(n, m);

    // index zero is reserved for an empty string

    if (index == 0) {
        name = "";
        return;
    }

    const Name key = getStringKey(index);

    // key is assumed to exist in metadata

    openvdb::StringMetadata::ConstPtr meta = mMetadata.getMetadata<StringMetadata>(key);

    if (!meta) {
        OPENVDB_THROW(LookupError, "String attribute cannot be found with index - \"" << index << "\".");
    }

    name = meta->value();
}

const AttributeArray& StringAttributeHandle::array() const
{
    return mHandle.array();
}


////////////////////////////////////////

// StringAttributeWriteHandle implementation

StringAttributeWriteHandle::Ptr
StringAttributeWriteHandle::create(AttributeArray& array, const MetaMap& metadata, const bool expand)
{
    return std::make_shared<StringAttributeWriteHandle>(array, metadata, expand);
}


StringAttributeWriteHandle::StringAttributeWriteHandle(AttributeArray& array,
                                                       const MetaMap& metadata,
                                                       const bool expand)
    : StringAttributeHandle(array, metadata, /*preserveCompression=*/ false)
    , mWriteHandle(array, expand)
{
    // populate the cache
    resetCache();
}


void StringAttributeWriteHandle::expand(bool fill)
{
    mWriteHandle.expand(fill);
}


void StringAttributeWriteHandle::collapse()
{
    // zero is used for an empty string
    mWriteHandle.collapse(0);
}


void StringAttributeWriteHandle::collapse(const Name& name)
{
    Index index = getIndex(name);
    mWriteHandle.collapse(index);
}


bool StringAttributeWriteHandle::compact()
{
    return mWriteHandle.compact();
}


void StringAttributeWriteHandle::fill(const Name& name)
{
    Index index = getIndex(name);
    mWriteHandle.fill(index);
}


void StringAttributeWriteHandle::set(Index n, const Name& name)
{
    Index index = getIndex(name);
    mWriteHandle.set(n, /*stride*/0, index);
}


void StringAttributeWriteHandle::set(Index n, Index m, const Name& name)
{
    Index index = getIndex(name);
    mWriteHandle.set(n, m, index);
}


void StringAttributeWriteHandle::resetCache()
{
    mCache.reset(mMetadata);
}


AttributeArray& StringAttributeWriteHandle::array()
{
    return mWriteHandle.array();
}


bool StringAttributeWriteHandle::contains(const Name& name) const
{
    // empty strings always have an index at index zero
    if (name.empty())   return true;
    const auto& cacheMap = mCache.map();
    return cacheMap.find(name) != cacheMap.end();
}


Index StringAttributeWriteHandle::getIndex(const Name& name) const
{
    // zero used for an empty string
    if (name.empty())   return Index(0);

    const auto& cacheMap = mCache.map();
    auto it = cacheMap.find(name);

    if (it == cacheMap.end()) {
        OPENVDB_THROW(LookupError, "String does not exist in Metadata, insert it and reset the cache - \"" << name << "\".");
    }

    return it->second;
}


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
