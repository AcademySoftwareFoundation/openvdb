// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/AttributeSet.cc

#include "AttributeSet.h"
#include "AttributeGroup.h"

#include <algorithm> // std::equal
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


namespace {
    // remove the items from the vector corresponding to the indices
    template <typename T>
    void eraseIndices(std::vector<T>& vec,
                      const std::vector<size_t>& indices)
    {
        // early-exit if no indices to erase

        if (indices.empty())    return;

        // build the sorted, unique indices to remove

        std::vector<size_t> toRemove(indices);
        std::sort(toRemove.rbegin(), toRemove.rend());
        toRemove.erase(unique(toRemove.begin(), toRemove.end()), toRemove.end());

        // throw if the largest index is out of range

        if (*toRemove.begin() >= vec.size()) {
            OPENVDB_THROW(LookupError, "Cannot erase indices as index is out of range.")
        }

        // erase elements from the back

        for (auto it = toRemove.cbegin(); it != toRemove.cend(); ++it) {
            vec.erase(vec.begin() + (*it));
        }
    }

    // return true if a string begins with a particular substring
    bool startsWith(const std::string& str, const std::string& prefix)
    {
        return str.compare(0, prefix.length(), prefix) == 0;
    }
}

////////////////////////////////////////


// AttributeSet implementation


AttributeSet::AttributeSet()
    : mDescr(new Descriptor())
{
}


AttributeSet::AttributeSet(const AttributeSet& attrSet, Index arrayLength,
    const AttributeArray::ScopedRegistryLock* lock)
    : mDescr(attrSet.descriptorPtr())
    , mAttrs(attrSet.descriptor().size(), AttributeArray::Ptr())
{
    std::unique_ptr<AttributeArray::ScopedRegistryLock> localLock;
    if (!lock) {
        localLock.reset(new AttributeArray::ScopedRegistryLock);
        lock = localLock.get();
    }

    const MetaMap& meta = mDescr->getMetadata();
    bool hasMetadata = meta.metaCount();

    for (const auto& namePos : mDescr->map()) {
        const size_t& pos = namePos.second;
        Metadata::ConstPtr metadata;
        if (hasMetadata)    metadata = meta["default:" + namePos.first];
        const AttributeArray* existingArray = attrSet.getConst(pos);
        const bool constantStride = existingArray->hasConstantStride();
        const Index stride = constantStride ? existingArray->stride() : existingArray->dataSize();

        AttributeArray::Ptr array = AttributeArray::create(mDescr->type(pos), arrayLength,
            stride, constantStride, metadata.get(), lock);

        // transfer hidden and transient flags
        if (existingArray->isHidden())      array->setHidden(true);
        if (existingArray->isTransient())   array->setTransient(true);

        mAttrs[pos] = array;
    }
}


AttributeSet::AttributeSet(const DescriptorPtr& descr, Index arrayLength,
    const AttributeArray::ScopedRegistryLock* lock)
    : mDescr(descr)
    , mAttrs(descr->size(), AttributeArray::Ptr())
{
    std::unique_ptr<AttributeArray::ScopedRegistryLock> localLock;
    if (!lock) {
        localLock.reset(new AttributeArray::ScopedRegistryLock);
        lock = localLock.get();
    }

    const MetaMap& meta = mDescr->getMetadata();
    bool hasMetadata = meta.metaCount();

    for (const auto& namePos : mDescr->map()) {
        const size_t& pos = namePos.second;
        Metadata::ConstPtr metadata;
        if (hasMetadata)    metadata = meta["default:" + namePos.first];
        mAttrs[pos] = AttributeArray::create(mDescr->type(pos), arrayLength,
            /*stride=*/1, /*constantStride=*/true, metadata.get(), lock);
    }
}


AttributeSet::AttributeSet(const AttributeSet& rhs)
    : mDescr(rhs.mDescr)
    , mAttrs(rhs.mAttrs)
{
}


size_t
AttributeSet::memUsage() const
{
    size_t bytes = sizeof(*this) + mDescr->memUsage();
    for (const auto& attr : mAttrs) {
        bytes += attr->memUsage();
    }
    return bytes;
}


size_t
AttributeSet::find(const std::string& name) const
{
    return mDescr->find(name);
}


size_t
AttributeSet::replace(const std::string& name, const AttributeArray::Ptr& attr)
{
    const size_t pos = this->find(name);
    return pos != INVALID_POS ? this->replace(pos, attr) : INVALID_POS;
}


size_t
AttributeSet::replace(size_t pos, const AttributeArray::Ptr& attr)
{
    assert(pos != INVALID_POS);
    assert(pos < mAttrs.size());

    if (attr->type() != mDescr->type(pos)) {
        return INVALID_POS;
    }

    mAttrs[pos] = attr;
    return pos;
}


const AttributeArray*
AttributeSet::getConst(const std::string& name) const
{
    const size_t pos = this->find(name);
    if (pos < mAttrs.size()) return this->getConst(pos);
    return nullptr;
}


const AttributeArray*
AttributeSet::get(const std::string& name) const
{
    return this->getConst(name);
}


AttributeArray*
AttributeSet::get(const std::string& name)
{
    const size_t pos = this->find(name);
    if (pos < mAttrs.size()) return this->get(pos);
    return nullptr;
}


const AttributeArray*
AttributeSet::getConst(size_t pos) const
{
    assert(pos != INVALID_POS);
    assert(pos < mAttrs.size());
    return mAttrs[pos].get();
}


const AttributeArray*
AttributeSet::get(size_t pos) const
{
    assert(pos != INVALID_POS);
    assert(pos < mAttrs.size());
    return this->getConst(pos);
}


AttributeArray*
AttributeSet::get(size_t pos)
{
    makeUnique(pos);
    return mAttrs[pos].get();
}


size_t
AttributeSet::groupOffset(const Name& group) const
{
    return mDescr->groupOffset(group);
}


size_t
AttributeSet::groupOffset(const Util::GroupIndex& index) const
{
    return mDescr->groupOffset(index);
}


AttributeSet::Descriptor::GroupIndex
AttributeSet::groupIndex(const Name& group) const
{
    return mDescr->groupIndex(group);
}


AttributeSet::Descriptor::GroupIndex
AttributeSet::groupIndex(const size_t offset) const
{
    return mDescr->groupIndex(offset);
}

std::vector<size_t>
AttributeSet::groupAttributeIndices() const
{
    std::vector<size_t> indices;

    for (const auto& namePos : mDescr->map()) {
        const AttributeArray* array = this->getConst(namePos.first);
        if (isGroup(*array)) {
            indices.push_back(namePos.second);
        }
    }

    return indices;
}

bool
AttributeSet::isShared(size_t pos) const
{
    assert(pos != INVALID_POS);
    assert(pos < mAttrs.size());
    return !mAttrs[pos].unique();
}


void
AttributeSet::makeUnique(size_t pos)
{
    assert(pos != INVALID_POS);
    assert(pos < mAttrs.size());
    if (!mAttrs[pos].unique()) {
        mAttrs[pos] = mAttrs[pos]->copy();
    }
}


AttributeArray::Ptr
AttributeSet::appendAttribute(  const Name& name,
                                const NamePair& type,
                                const Index strideOrTotalSize,
                                const bool constantStride,
                                const Metadata* defaultValue)
{
    Descriptor::Ptr descriptor = mDescr->duplicateAppend(name, type);

    // store the attribute default value in the descriptor metadata
    if (defaultValue)   descriptor->setDefaultValue(name, *defaultValue);

    // extract the index from the descriptor
    const size_t pos = descriptor->find(name);

    return this->appendAttribute(*mDescr, descriptor, pos, strideOrTotalSize, constantStride, defaultValue);
}


AttributeArray::Ptr
AttributeSet::appendAttribute(  const Descriptor& expected, DescriptorPtr& replacement,
                                const size_t pos, const Index strideOrTotalSize, const bool constantStride,
                                const Metadata* defaultValue,
                                const AttributeArray::ScopedRegistryLock* lock)
{
    // ensure the descriptor is as expected
    if (*mDescr != expected) {
        OPENVDB_THROW(LookupError, "Cannot append attributes as descriptors do not match.")
    }

    assert(replacement->size() >= mDescr->size());

    const size_t offset = mDescr->size();

    // extract the array length from the first attribute array if it exists

    const Index arrayLength = offset > 0 ? this->get(0)->size() : 1;

    // extract the type from the descriptor

    const NamePair& type = replacement->type(pos);

    // append the new array

    AttributeArray::Ptr array = AttributeArray::create(
        type, arrayLength, strideOrTotalSize, constantStride,
        defaultValue, lock);

    // if successful, update Descriptor and append the created array

    mDescr = replacement;

    mAttrs.push_back(array);

    return array;
}


AttributeArray::Ptr
AttributeSet::removeAttribute(const Name& name)
{
    const size_t pos = this->find(name);
    if (pos == INVALID_POS) return AttributeArray::Ptr();
    return this->removeAttribute(pos);
}


AttributeArray::Ptr
AttributeSet::removeAttribute(const size_t pos)
{
    if (pos >= mAttrs.size())     return AttributeArray::Ptr();

    assert(mAttrs[pos]);
    AttributeArray::Ptr array;
    std::swap(array, mAttrs[pos]);
    assert(array);

    // safely drop the attribute and update the descriptor
    std::vector<size_t> toDrop{pos};
    this->dropAttributes(toDrop);

    return array;
}


AttributeArray::Ptr
AttributeSet::removeAttributeUnsafe(const size_t pos)
{
    if (pos >= mAttrs.size())     return AttributeArray::Ptr();

    assert(mAttrs[pos]);
    AttributeArray::Ptr array;
    std::swap(array, mAttrs[pos]);

    return array;
}


void
AttributeSet::dropAttributes(const std::vector<size_t>& pos)
{
    if (pos.empty())    return;

    Descriptor::Ptr descriptor = mDescr->duplicateDrop(pos);

    this->dropAttributes(pos, *mDescr, descriptor);
}


void
AttributeSet::dropAttributes(   const std::vector<size_t>& pos,
                                const Descriptor& expected, DescriptorPtr& replacement)
{
    if (pos.empty())    return;

    // ensure the descriptor is as expected
    if (*mDescr != expected) {
        OPENVDB_THROW(LookupError, "Cannot drop attributes as descriptors do not match.")
    }

    mDescr = replacement;

    eraseIndices(mAttrs, pos);

    // remove any unused default values

    mDescr->pruneUnusedDefaultValues();
}


void
AttributeSet::renameAttributes(const Descriptor& expected, const DescriptorPtr& replacement)
{
    // ensure the descriptor is as expected
    if (*mDescr != expected) {
        OPENVDB_THROW(LookupError, "Cannot rename attribute as descriptors do not match.")
    }

    mDescr = replacement;
}


void
AttributeSet::reorderAttributes(const DescriptorPtr& replacement)
{
    if (*mDescr == *replacement) {
        this->resetDescriptor(replacement);
        return;
    }

    if (!mDescr->hasSameAttributes(*replacement)) {
        OPENVDB_THROW(LookupError, "Cannot reorder attributes as descriptors do not contain the same attributes.")
    }

    AttrArrayVec attrs(replacement->size());

    // compute target indices for attributes from the given decriptor
    for (const auto& namePos : mDescr->map()) {
        const size_t index = replacement->find(namePos.first);
        attrs[index] = AttributeArray::Ptr(mAttrs[namePos.second]);
    }

    // copy the ordering to the member attributes vector and update descriptor to be target
    std::copy(attrs.begin(), attrs.end(), mAttrs.begin());
    mDescr = replacement;
}


void
AttributeSet::resetDescriptor(const DescriptorPtr& replacement, const bool allowMismatchingDescriptors)
{
    // ensure the descriptors match
    if (!allowMismatchingDescriptors && *mDescr != *replacement) {
        OPENVDB_THROW(LookupError, "Cannot swap descriptor as replacement does not match.")
    }

    mDescr = replacement;
}


void
AttributeSet::read(std::istream& is)
{
    this->readDescriptor(is);
    this->readMetadata(is);
    this->readAttributes(is);
}


void
AttributeSet::write(std::ostream& os, bool outputTransient) const
{
    this->writeDescriptor(os, outputTransient);
    this->writeMetadata(os, outputTransient);
    this->writeAttributes(os, outputTransient);
}


void
AttributeSet::readDescriptor(std::istream& is)
{
    mDescr->read(is);
}


void
AttributeSet::writeDescriptor(std::ostream& os, bool outputTransient) const
{
    // build a vector of all attribute arrays that have a transient flag
    // unless also writing transient attributes

    std::vector<size_t> transientArrays;

    if (!outputTransient) {
        for (size_t i = 0; i < size(); i++) {
            const AttributeArray* array = this->getConst(i);
            if (array->isTransient()) {
                transientArrays.push_back(i);
            }
        }
    }

    // write out a descriptor without transient attributes

    if (transientArrays.empty()) {
        mDescr->write(os);
    }
    else {
        Descriptor::Ptr descr = mDescr->duplicateDrop(transientArrays);
        descr->write(os);
    }
}


void
AttributeSet::readMetadata(std::istream& is)
{
    AttrArrayVec(mDescr->size()).swap(mAttrs); // allocate vector

    for (size_t n = 0, N = mAttrs.size(); n < N; ++n) {
        mAttrs[n] = AttributeArray::create(mDescr->type(n), 1, 1);
        mAttrs[n]->readMetadata(is);
    }
}


void
AttributeSet::writeMetadata(std::ostream& os, bool outputTransient, bool paged) const
{
    // write attribute metadata

    for (size_t i = 0; i < size(); i++) {
        const AttributeArray* array = this->getConst(i);
        array->writeMetadata(os, outputTransient, paged);
    }
}


void
AttributeSet::readAttributes(std::istream& is)
{
    for (size_t i = 0; i < mAttrs.size(); i++) {
        mAttrs[i]->readBuffers(is);
    }
}


void
AttributeSet::writeAttributes(std::ostream& os, bool outputTransient) const
{
    for (auto attr : mAttrs) {
        attr->writeBuffers(os, outputTransient);
    }
}


bool
AttributeSet::operator==(const AttributeSet& other) const {
    if(*this->mDescr != *other.mDescr) return false;
    if(this->mAttrs.size() != other.mAttrs.size()) return false;

    for (size_t n = 0; n < this->mAttrs.size(); ++n) {
        if (*this->mAttrs[n] != *other.mAttrs[n]) return false;
    }
    return true;
}

////////////////////////////////////////

// AttributeSet::Descriptor implementation


AttributeSet::Descriptor::Descriptor()
{
}


AttributeSet::Descriptor::Descriptor(const Descriptor& rhs)
    : mNameMap(rhs.mNameMap)
    , mTypes(rhs.mTypes)
    , mGroupMap(rhs.mGroupMap)
    , mMetadata(rhs.mMetadata)
{
}


bool
AttributeSet::Descriptor::operator==(const Descriptor& rhs) const
{
    if (this == &rhs) return true;

    if (mTypes.size()   != rhs.mTypes.size() ||
        mNameMap.size() != rhs.mNameMap.size() ||
        mGroupMap.size() != rhs.mGroupMap.size()) {
        return false;
    }

    for (size_t n = 0, N = mTypes.size(); n < N; ++n) {
        if (mTypes[n] != rhs.mTypes[n]) return false;
    }

    if (this->mMetadata != rhs.mMetadata)  return false;

    return  std::equal(mGroupMap.begin(), mGroupMap.end(), rhs.mGroupMap.begin()) &&
            std::equal(mNameMap.begin(), mNameMap.end(), rhs.mNameMap.begin());
}


bool
AttributeSet::Descriptor::hasSameAttributes(const Descriptor& rhs) const
{
    if (this == &rhs) return true;

    if (mTypes.size()   != rhs.mTypes.size() ||
        mNameMap.size() != rhs.mNameMap.size() ||
        mGroupMap.size() != rhs.mGroupMap.size()) {
        return false;
    }

    for (const auto& namePos : mNameMap) {
        const size_t index = rhs.find(namePos.first);

        if (index == INVALID_POS) return false;

        if (mTypes[namePos.second] != rhs.mTypes[index]) return false;
    }

    return std::equal(mGroupMap.begin(), mGroupMap.end(), rhs.mGroupMap.begin());
}


size_t
AttributeSet::Descriptor::count(const NamePair& matchType) const
{
    return std::count(mTypes.begin(), mTypes.end(), matchType);
}


size_t
AttributeSet::Descriptor::memUsage() const
{
    size_t bytes = sizeof(NameToPosMap::mapped_type) * this->size();
    for (const auto& namePos : mNameMap) {
        bytes += namePos.first.capacity();
    }

    for (const NamePair& type : mTypes) {
         bytes += type.first.capacity();
         bytes += type.second.capacity();
    }

    return sizeof(*this) + bytes;
}


size_t
AttributeSet::Descriptor::find(const std::string& name) const
{
    auto it = mNameMap.find(name);
    if (it != mNameMap.end()) {
        return it->second;
    }
    return INVALID_POS;
}


size_t
AttributeSet::Descriptor::rename(const std::string& fromName, const std::string& toName)
{
    if (!validName(toName))  throw RuntimeError("Attribute name contains invalid characters - " + toName);

    size_t pos = INVALID_POS;

    // check if the new name is already used.
    auto it = mNameMap.find(toName);
    if (it != mNameMap.end()) return pos;

    it = mNameMap.find(fromName);
    if (it != mNameMap.end()) {
        pos = it->second;
        mNameMap.erase(it);
        mNameMap[toName] = pos;

        // rename default value if it exists

        std::stringstream ss;
        ss << "default:" << fromName;

        Metadata::Ptr defaultValue = mMetadata[ss.str()];

        if (defaultValue) {
            mMetadata.removeMeta(ss.str());
            ss.str("");
            ss << "default:" << toName;
            mMetadata.insertMeta(ss.str(), *defaultValue);
        }
    }
    return pos;
}

size_t
AttributeSet::Descriptor::renameGroup(const std::string& fromName, const std::string& toName)
{
    if (!validName(toName))  throw RuntimeError("Group name contains invalid characters - " + toName);

    size_t pos = INVALID_POS;

    // check if the new name is already used.
    auto it = mGroupMap.find(toName);
    if (it != mGroupMap.end()) return pos;

    it = mGroupMap.find(fromName);
    if (it != mGroupMap.end()) {
        pos = it->second;
        mGroupMap.erase(it);
        mGroupMap[toName] = pos;
    }

    return pos;
}


const Name&
AttributeSet::Descriptor::valueType(size_t pos) const
{
    // pos is assumed to exist
    return this->type(pos).first;
}


const NamePair&
AttributeSet::Descriptor::type(size_t pos) const
{
    // assert that pos is valid and in-range

    assert(pos != AttributeSet::INVALID_POS);
    assert(pos < mTypes.size());

    return mTypes[pos];
}


MetaMap&
AttributeSet::Descriptor::getMetadata()
{
    return mMetadata;
}


const MetaMap&
AttributeSet::Descriptor::getMetadata() const
{
    return mMetadata;
}


bool
AttributeSet::Descriptor::hasDefaultValue(const Name& name) const
{
    std::stringstream ss;
    ss << "default:" << name;

    return bool(mMetadata[ss.str()]);
}


void
AttributeSet::Descriptor::setDefaultValue(const Name& name, const Metadata& defaultValue)
{
    const size_t pos = find(name);
    if (pos == INVALID_POS) {
        OPENVDB_THROW(LookupError, "Cannot find attribute name to set default value.")
    }

    // check type of metadata matches attribute type

    const Name& valueType = this->valueType(pos);
    if (valueType != defaultValue.typeName()) {
        OPENVDB_THROW(TypeError, "Mis-matching Default Value Type");
    }

    std::stringstream ss;
    ss << "default:" << name;

    mMetadata.insertMeta(ss.str(), defaultValue);
}


void
AttributeSet::Descriptor::removeDefaultValue(const Name& name)
{
    std::stringstream ss;
    ss << "default:" << name;

    mMetadata.removeMeta(ss.str());
}


void
AttributeSet::Descriptor::pruneUnusedDefaultValues()
{
    // store any default metadata keys for which the attribute name is no longer present

    std::vector<Name> metaToErase;

    for (auto it = mMetadata.beginMeta(), itEnd = mMetadata.endMeta(); it != itEnd; ++it) {
        const Name name = it->first;

        // ignore non-default metadata
        if (!startsWith(name, "default:"))   continue;

        const Name defaultName = name.substr(8, it->first.size() - 8);

        if (mNameMap.find(defaultName) == mNameMap.end()) {
            metaToErase.push_back(name);
        }
    }

    // remove this metadata

    for (const Name& name : metaToErase) {
        mMetadata.removeMeta(name);
    }
}


size_t
AttributeSet::Descriptor::insert(const std::string& name, const NamePair& typeName)
{
    if (!validName(name))  throw RuntimeError("Attribute name contains invalid characters - " + name);

    size_t pos = INVALID_POS;
    auto it = mNameMap.find(name);
    if (it != mNameMap.end()) {
        assert(it->second < mTypes.size());
        if (mTypes[it->second] != typeName) {
            OPENVDB_THROW(KeyError,
                "Cannot insert into a Descriptor with a duplicate name, but different type.")
        }
        pos = it->second;
    } else {

        if (!AttributeArray::isRegistered(typeName)) {
            OPENVDB_THROW(KeyError, "Failed to insert '" << name
                << "' with unregistered attribute type '" << typeName.first << "_" << typeName.second);
        }

        pos = mTypes.size();
        mTypes.push_back(typeName);
        mNameMap.insert(it, NameToPosMap::value_type(name, pos));
    }
    return pos;
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::create(const NameAndTypeVec& attrs,
                                 const NameToPosMap& groupMap,
                                 const MetaMap& metadata)
{
    auto newDescriptor = std::make_shared<Descriptor>();

    for (const NameAndType& attr : attrs) {
        newDescriptor->insert(attr.name, attr.type);
    }

    newDescriptor->mGroupMap = groupMap;
    newDescriptor->mMetadata = metadata;

    return newDescriptor;
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::create(const NamePair& positionType)
{
    auto descr = std::make_shared<Descriptor>();
    descr->insert("P", positionType);
    return descr;
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::duplicateAppend(const Name& name, const NamePair& type) const
{
    Inserter attributes;

    this->appendTo(attributes.vec);
    attributes.add(NameAndType(name, type));

    return Descriptor::create(attributes.vec, mGroupMap, mMetadata);
}


AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::duplicateDrop(const std::vector<size_t>& pos) const
{
    NameAndTypeVec vec;
    this->appendTo(vec);

    Descriptor::Ptr descriptor;

    // If groups exist, check to see if those arrays are being dropped

    if (!mGroupMap.empty()) {

        // extract all attribute array group indices and specific groups
        // to drop

        std::vector<size_t> groups, groupsToDrop;
        for (size_t i = 0; i < vec.size(); i++) {
            if (vec[i].type == GroupAttributeArray::attributeType()) {
                groups.push_back(i);
                if (std::find(pos.begin(), pos.end(), i) != pos.end()) {
                    groupsToDrop.push_back(i);
                }
            }
        }

        // drop the indices in indices from vec

        eraseIndices(vec, pos);

        if (!groupsToDrop.empty()) {

            // configure group mapping if group arrays have been dropped

            NameToPosMap droppedGroupMap = mGroupMap;

            const size_t GROUP_BITS = sizeof(GroupType) * CHAR_BIT;
            for (auto iter = droppedGroupMap.begin(); iter != droppedGroupMap.end();) {
                const size_t groupArrayPos = iter->second / GROUP_BITS;
                const size_t arrayPos = groups[groupArrayPos];
                if (std::find(pos.begin(), pos.end(), arrayPos) != pos.end()) {
                    iter = droppedGroupMap.erase(iter);
                }
                else {
                    size_t offset(0);
                    for (const size_t& idx : groupsToDrop) {
                        if (idx >= arrayPos) break;
                        ++offset;
                    }
                    iter->second -= (offset * GROUP_BITS);
                    ++iter;
                }
            }

            descriptor = Descriptor::create(vec, droppedGroupMap, mMetadata);

            // remove any unused default values

            descriptor->pruneUnusedDefaultValues();

            return descriptor;
        }
    }
    else {

        // drop the indices in pos from vec

        eraseIndices(vec, pos);
    }

    descriptor = Descriptor::create(vec, mGroupMap, mMetadata);

    // remove any unused default values

    descriptor->pruneUnusedDefaultValues();

    return descriptor;
}

void
AttributeSet::Descriptor::appendTo(NameAndTypeVec& attrs) const
{
    // build a std::map<pos, name> (ie key and value swapped)

    using PosToNameMap = std::map<size_t, std::string>;

    PosToNameMap posToNameMap;

    for (const auto& namePos : mNameMap) {
        posToNameMap[namePos.second] = namePos.first;
    }

    // std::map is sorted by key, so attributes can now be inserted in position order

    for (const auto& posName : posToNameMap) {
        attrs.emplace_back(posName.second, this->type(posName.first));
    }
}

bool
AttributeSet::Descriptor::hasGroup(const Name& group) const
{
    return mGroupMap.find(group) != mGroupMap.end();
}

void
AttributeSet::Descriptor::setGroup(const Name& group, const size_t offset,
    const bool checkValidOffset)
{
    if (!validName(group)) {
        throw RuntimeError("Group name contains invalid characters - " + group);
    }
    if (checkValidOffset) {
        // check offset is not out-of-range
        if (offset >= this->availableGroups()) {
            throw RuntimeError("Group offset is out-of-range - " + group);
        }
        // check offset is not already in use
        for (const auto& namePos : mGroupMap) {
            if (namePos.second == offset) {
                throw RuntimeError("Group offset is already in use - " + group);
            }
        }
    }

    mGroupMap[group] = offset;
}

void
AttributeSet::Descriptor::dropGroup(const Name& group)
{
    mGroupMap.erase(group);
}

void
AttributeSet::Descriptor::clearGroups()
{
    mGroupMap.clear();
}

const Name
AttributeSet::Descriptor::uniqueName(const Name& name) const
{
    auto it = mNameMap.find(name);
    if (it == mNameMap.end()) return name;

    std::ostringstream ss;
    size_t i(0);
    while (it != mNameMap.end()) {
        ss.str("");
        ss << name << i++;
        it = mNameMap.find(ss.str());
    }
    return ss.str();
}

const Name
AttributeSet::Descriptor::uniqueGroupName(const Name& name) const
{
    auto it = mGroupMap.find(name);
    if (it == mGroupMap.end()) return name;

    std::ostringstream ss;
    size_t i(0);
    while (it != mGroupMap.end()) {
        ss.str("");
        ss << name << i++;
        it = mGroupMap.find(ss.str());
    }
    return ss.str();
}

size_t
AttributeSet::Descriptor::groupOffset(const Name& group) const
{
    const auto it = mGroupMap.find(group);
    if (it == mGroupMap.end()) {
        return INVALID_POS;
    }
    return it->second;
}

size_t
AttributeSet::Descriptor::groupOffset(const Util::GroupIndex& index) const
{
    if (index.first >= mNameMap.size()) {
        OPENVDB_THROW(LookupError, "Out of range group index.")
    }

    if (mTypes[index.first] != GroupAttributeArray::attributeType()) {
        OPENVDB_THROW(LookupError, "Group index invalid.")
    }

    // find the relative index in the group attribute arrays

    size_t relativeIndex = 0;
    for (const auto& namePos : mNameMap) {
        if (namePos.second < index.first &&
            mTypes[namePos.second] == GroupAttributeArray::attributeType()) {
            relativeIndex++;
        }
    }

    const size_t GROUP_BITS = sizeof(GroupType) * CHAR_BIT;

    return (relativeIndex * GROUP_BITS) + index.second;
}

AttributeSet::Descriptor::GroupIndex
AttributeSet::Descriptor::groupIndex(const Name& group) const
{
    const size_t offset = this->groupOffset(group);
    if (offset == INVALID_POS) {
        OPENVDB_THROW(LookupError, "Group not found - " << group << ".")
    }
    return this->groupIndex(offset);
}

AttributeSet::Descriptor::GroupIndex
AttributeSet::Descriptor::groupIndex(const size_t offset) const
{
    // extract all attribute array group indices

    std::vector<size_t> groups;
    for (const auto& namePos : mNameMap) {
        if (mTypes[namePos.second] == GroupAttributeArray::attributeType()) {
            groups.push_back(namePos.second);
        }
    }

    if (offset >= groups.size() * this->groupBits()) {
        OPENVDB_THROW(LookupError, "Out of range group offset - " << offset << ".")
    }

    // adjust relative offset to find offset into the array vector

    std::sort(groups.begin(), groups.end());
    return Util::GroupIndex(groups[offset / this->groupBits()],
                static_cast<uint8_t>(offset % this->groupBits()));
}

size_t
AttributeSet::Descriptor::availableGroups() const
{
    // the number of group attributes * number of bits per group

    const size_t groupAttributes =
        this->count(GroupAttributeArray::attributeType());

    return groupAttributes * this->groupBits();
}

size_t
AttributeSet::Descriptor::unusedGroups() const
{
    // compute total slots (one slot per bit of the group attributes)

    const size_t availableGroups = this->availableGroups();

    if (availableGroups == 0)   return 0;

    // compute slots in use

    const size_t usedGroups = mGroupMap.size();

    return availableGroups - usedGroups;
}

bool
AttributeSet::Descriptor::canCompactGroups() const
{
    // can compact if more unused groups than in one group attribute array

    return this->unusedGroups() >= this->groupBits();
}

size_t
AttributeSet::Descriptor::unusedGroupOffset(size_t hint) const
{
    // all group offsets are in use

    if (unusedGroups() == size_t(0)) {
        return std::numeric_limits<size_t>::max();
    }

    // build a list of group indices

    std::vector<size_t> indices;
    indices.reserve(mGroupMap.size());
    for (const auto& namePos : mGroupMap) {
        indices.push_back(namePos.second);
    }

    std::sort(indices.begin(), indices.end());

    // return hint if not already in use

    if (hint != std::numeric_limits<Index>::max() &&
        hint < availableGroups() &&
        std::find(indices.begin(), indices.end(), hint) == indices.end()) {
        return hint;
    }

    // otherwise return first index not present

    size_t offset = 0;
    for (const size_t& index : indices) {
        if (index != offset)     break;
        offset++;
    }

    return offset;
}

bool
AttributeSet::Descriptor::requiresGroupMove(Name& sourceName,
    size_t& sourceOffset, size_t& targetOffset) const
{
    targetOffset = this->unusedGroupOffset();

    for (const auto& namePos : mGroupMap) {

        // move only required if source comes after the target

        if (namePos.second >= targetOffset) {
            sourceName = namePos.first;
            sourceOffset = namePos.second;
            return true;
        }
    }

    return false;
}

bool
AttributeSet::Descriptor::groupIndexCollision(const Descriptor& rhs) const
{
    const auto& groupMap = this->groupMap();
    const auto& otherGroupMap = rhs.groupMap();

    // iterate both group maps at the same time and find any keys that occur
    // in both maps and test their values for equality

    auto groupsIt1 = groupMap.cbegin();
    auto groupsIt2 = otherGroupMap.cbegin();

    while (groupsIt1 != groupMap.cend() && groupsIt2 != otherGroupMap.cend()) {
        if (groupsIt1->first < groupsIt2->first) {
            ++groupsIt1;
        } else if (groupsIt1->first > groupsIt2->first) {
            ++groupsIt2;
        } else {
            if (groupsIt1->second != groupsIt2->second) {
                return true;
            } else {
                ++groupsIt1;
                ++groupsIt2;
            }
        }
    }

    return false;
}

bool
AttributeSet::Descriptor::validName(const Name& name)
{
    if (name.empty())   return false;
    return std::find_if(name.begin(), name.end(),
            [&](int c) { return !(isalnum(c) || (c == '_') || (c == '|') || (c == ':')); } ) == name.end();
}

void
AttributeSet::Descriptor::parseNames(   std::vector<std::string>& includeNames,
                                        std::vector<std::string>& excludeNames,
                                        bool& includeAll,
                                        const std::string& nameStr)
{
    includeAll = false;

    std::stringstream tokenStream(nameStr);

    Name token;
    while (tokenStream >> token) {

        bool negate = startsWith(token, "^") || startsWith(token, "!");

        if (negate) {
            if (token.length() < 2) throw RuntimeError("Negate character (^) must prefix a name.");
            token = token.substr(1, token.length()-1);
            if (!validName(token))  throw RuntimeError("Name contains invalid characters - " + token);
            excludeNames.push_back(token);
        }
        else if (!includeAll) {
            if (token == "*") {
                includeAll = true;
                includeNames.clear();
                continue;
            }
            if (!validName(token))  throw RuntimeError("Name contains invalid characters - " + token);
            includeNames.push_back(token);
        }
    }
}

void
AttributeSet::Descriptor::parseNames(   std::vector<std::string>& includeNames,
                                        std::vector<std::string>& excludeNames,
                                        const std::string& nameStr)
{
    bool includeAll = false;
    Descriptor::parseNames(includeNames, excludeNames, includeAll, nameStr);
}

void
AttributeSet::Descriptor::write(std::ostream& os) const
{
    const Index64 arraylength = Index64(mTypes.size());
    os.write(reinterpret_cast<const char*>(&arraylength), sizeof(Index64));

    for (const NamePair& np : mTypes) {
        writeString(os, np.first);
        writeString(os, np.second);
    }

    for (auto it = mNameMap.begin(), endIt = mNameMap.end(); it != endIt; ++it) {
        writeString(os, it->first);
        os.write(reinterpret_cast<const char*>(&it->second), sizeof(Index64));
    }

    const Index64 grouplength = Index64(mGroupMap.size());
    os.write(reinterpret_cast<const char*>(&grouplength), sizeof(Index64));

    for (auto groupIt = mGroupMap.cbegin(), endGroupIt = mGroupMap.cend(); groupIt != endGroupIt; ++groupIt) {
        writeString(os, groupIt->first);
        os.write(reinterpret_cast<const char*>(&groupIt->second), sizeof(Index64));
    }

    mMetadata.writeMeta(os);
}


void
AttributeSet::Descriptor::read(std::istream& is)
{
    Index64 arraylength = 0;
    is.read(reinterpret_cast<char*>(&arraylength), sizeof(Index64));

    std::vector<NamePair>(size_t(arraylength)).swap(mTypes);

    for (NamePair& np : mTypes) {
        np.first = readString(is);
        np.second = readString(is);
    }

    mNameMap.clear();
    std::pair<std::string, size_t> nameAndOffset;

    for (Index64 n = 0; n < arraylength; ++n) {
        nameAndOffset.first = readString(is);
        if (!validName(nameAndOffset.first))  throw IoError("Attribute name contains invalid characters - " + nameAndOffset.first);
        is.read(reinterpret_cast<char*>(&nameAndOffset.second), sizeof(Index64));
        mNameMap.insert(nameAndOffset);
    }

    Index64 grouplength = 0;
    is.read(reinterpret_cast<char*>(&grouplength), sizeof(Index64));

    for (Index64 n = 0; n < grouplength; ++n) {
        nameAndOffset.first = readString(is);
        if (!validName(nameAndOffset.first))  throw IoError("Group name contains invalid characters - " + nameAndOffset.first);
        is.read(reinterpret_cast<char*>(&nameAndOffset.second), sizeof(Index64));
        mGroupMap.insert(nameAndOffset);
    }

    mMetadata.readMeta(is);
}



////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
