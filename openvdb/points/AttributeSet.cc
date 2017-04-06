///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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


AttributeSet::AttributeSet(const AttributeSet& attrSet, Index arrayLength)
    : mDescr(attrSet.descriptorPtr())
    , mAttrs(attrSet.descriptor().size(), AttributeArray::Ptr())
{
    for (const auto& namePos : mDescr->map()) {
        const size_t& pos = namePos.second;
        AttributeArray::Ptr array = AttributeArray::create(mDescr->type(pos), arrayLength, 1);

        // transfer hidden and transient flags
        if (attrSet.getConst(pos)->isHidden())      array->setHidden(true);
        if (attrSet.getConst(pos)->isTransient())   array->setTransient(true);

        mAttrs[pos] = array;
    }
}


AttributeSet::AttributeSet(const DescriptorPtr& descr, Index arrayLength)
    : mDescr(descr)
    , mAttrs(descr->size(), AttributeArray::Ptr())
{
    for (const auto& namePos : mDescr->map()) {
        const size_t& pos = namePos.second;
        mAttrs[pos] = AttributeArray::create(mDescr->type(pos), arrayLength, 1);
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
    const Descriptor::NameToPosMap& map = this->descriptor().groupMap();
    const Descriptor::ConstIterator it = map.find(group);
    if (it == map.end()) {
        return INVALID_POS;
    }
    return it->second;
}


size_t
AttributeSet::groupOffset(const Util::GroupIndex& index) const
{
    if (index.first >= mAttrs.size()) {
        OPENVDB_THROW(LookupError, "Out of range group index.")
    }

    if (!isGroup(*mAttrs[index.first])) {
        OPENVDB_THROW(LookupError, "Group index invalid.")
    }

    // find the relative index in the group attribute arrays

    size_t relativeIndex = 0;
    for (size_t i = 0; i < mAttrs.size(); i++) {
        if (i < index.first && isGroup(*mAttrs[i]))    relativeIndex++;
    }

    const size_t GROUP_BITS = sizeof(GroupType) * CHAR_BIT;

    return (relativeIndex * GROUP_BITS) + index.second;
}


AttributeSet::Descriptor::GroupIndex
AttributeSet::groupIndex(const Name& group) const
{
    const size_t offset = this->groupOffset(group);
    if (offset == INVALID_POS) {
        OPENVDB_THROW(LookupError, "Group not found - " << group << ".")
    }
    return this->groupIndex(offset);
}


AttributeSet::Descriptor::GroupIndex
AttributeSet::groupIndex(const size_t offset) const
{
    // extract all attribute array group indices

    const size_t GROUP_BITS = sizeof(GroupType) * CHAR_BIT;

    std::vector<unsigned> groups;
    for (size_t i = 0; i < mAttrs.size(); i++) {
        if (isGroup(*mAttrs[i])) {
            groups.push_back(static_cast<unsigned>(i));
        }
    }

    if (offset >= groups.size() * GROUP_BITS) {
        OPENVDB_THROW(LookupError, "Out of range group offset - " << offset << ".")
    }

    // adjust relative offset to find offset into the array vector

    return Util::GroupIndex(groups[offset / GROUP_BITS],
			    static_cast<uint8_t>(offset % GROUP_BITS));
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
                                Metadata::Ptr defaultValue)
{
    Descriptor::Ptr descriptor = mDescr->duplicateAppend(name, type);

    // store the attribute default value in the descriptor metadata
    if (defaultValue)   descriptor->setDefaultValue(name, *defaultValue);

    // extract the index from the descriptor
    const size_t pos = descriptor->find(name);

    return this->appendAttribute(*mDescr, descriptor, pos, strideOrTotalSize, constantStride);
}


AttributeArray::Ptr
AttributeSet::appendAttribute(  const Descriptor& expected, DescriptorPtr& replacement,
                                const size_t pos, const Index strideOrTotalSize, const bool constantStride)
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

    AttributeArray::Ptr array = AttributeArray::create(type, arrayLength, strideOrTotalSize, constantStride);

    // if successful, update Descriptor and append the created array

    mDescr = replacement;

    mAttrs.push_back(array);

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

    // drop the indices in pos from vec

    eraseIndices(vec, pos);

    Descriptor::Ptr descriptor = Descriptor::create(vec, mGroupMap, mMetadata);

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
AttributeSet::Descriptor::setGroup(const Name& group, const size_t offset)
{
    if (!validName(group))  throw RuntimeError("Group name contains invalid characters - " + group);

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
    std::ostringstream ss;
    for (size_t i = 0; i < this->size() + 1; i++) {
        ss.str("");
        ss << name << i;
        if (this->find(ss.str()) == INVALID_POS)    break;
    }
    return ss.str();
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
                                        const std::string& nameStr)
{
    bool includeAll = false;

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

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
