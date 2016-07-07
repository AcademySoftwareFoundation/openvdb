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
/// @file AttributeSet.cc
///
/// @authors Dan Bailey, Mihai Alden, Peter Cucka

#include <boost/algorithm/string/predicate.hpp> // for boost::starts_with()

#include <openvdb_points/tools/AttributeGroup.h>
#include <openvdb_points/tools/AttributeSet.h>

#include <algorithm> // std::equal
#include <string>

#include <boost/algorithm/string/predicate.hpp> // boost::starts_with

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


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

        for (std::vector<size_t>::const_iterator    it = toRemove.begin();
                                                    it != toRemove.end(); ++it) {
            vec.erase(vec.begin() + (*it));
        }
    }
}

////////////////////////////////////////


// AttributeSet implementation


AttributeSet::AttributeSet()
    : mDescr(new Descriptor())
    , mAttrs()
{
}


AttributeSet::AttributeSet(const DescriptorPtr& descr, size_t arrayLength)
    : mDescr(descr)
    , mAttrs(descr->size(), AttributeArray::Ptr())
{
    for (Descriptor::ConstIterator it = mDescr->map().begin(),
        end = mDescr->map().end(); it != end; ++it) {
        const size_t pos = it->second;
        mAttrs[pos] = AttributeArray::create(mDescr->type(pos), arrayLength);
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
    for (size_t n = 0, N = mAttrs.size(); n < N; ++n) {
        bytes += mAttrs[n]->memUsage();
    }
    return bytes;
}


size_t
AttributeSet::size(const uint16_t flag) const
{
    size_t count = 0;
    for (AttrArrayVec::const_iterator   it = mAttrs.begin(),
                                        itEnd = mAttrs.end(); it != itEnd; ++it) {
        if ((*it)->flags() & flag)  count++;
    }
    return count;
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
    return NULL;
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
    return NULL;
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

    if (!GroupAttributeArray::isGroup(*mAttrs[index.first])) {
        OPENVDB_THROW(LookupError, "Group index invalid.")
    }

    // find the relative index in the group attribute arrays

    size_t relativeIndex = 0;
    for (unsigned i = 0; i < mAttrs.size(); i++) {
        if (i < index.first && GroupAttributeArray::isGroup(*mAttrs[i]))    relativeIndex++;
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
    for (unsigned i = 0; i < mAttrs.size(); i++) {
        if (GroupAttributeArray::isGroup(*mAttrs[i]))      groups.push_back(i);
    }

    if (offset >= groups.size() * GROUP_BITS) {
        OPENVDB_THROW(LookupError, "Out of range group offset - " << offset << ".")
    }

    // adjust relative offset to find offset into the array vector

    return Util::GroupIndex(groups[offset / GROUP_BITS], offset % GROUP_BITS);
}


IndexIter
AttributeSet::beginIndex() const
{
    const Index32 size = this->size() == 0 ? 0 : this->get(0)->size();
    return IndexIter(0, size);
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
AttributeSet::appendAttribute(  const Descriptor::NameAndType& attribute,
                                Metadata::Ptr defaultValue)
{
    Descriptor::NameAndTypeVec vec;
    vec.push_back(attribute);

    Descriptor::Ptr descriptor = mDescr->duplicateAppend(vec);

    // store the attribute default value in the descriptor metadata
    if (defaultValue)   descriptor->setDefaultValue(attribute.name, *defaultValue);

    return this->appendAttribute(attribute, *mDescr, descriptor);
}


AttributeArray::Ptr
AttributeSet::appendAttribute(const Descriptor::NameAndType& attribute,
                              const Descriptor& expected, DescriptorPtr& replacement)
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

    AttributeArray::Ptr array = AttributeArray::create(attribute.type, arrayLength);

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
    for (Descriptor::ConstIterator it = mDescr->map().begin(),
        end = mDescr->map().end(); it != end; ++it) {
        const size_t index = replacement->find(it->first);
        attrs[index] = AttributeArray::Ptr(mAttrs[it->second]);
    }

    // copy the ordering to the member attributes vector and update descriptor to be target
    std::copy(attrs.begin(), attrs.end(), mAttrs.begin());
    mDescr = replacement;
}


void
AttributeSet::resetDescriptor(const DescriptorPtr& replacement)
{
    // ensure the descriptors match
    if (*mDescr != *replacement) {
        OPENVDB_THROW(LookupError, "Cannot swap descriptor as replacement does not match.")
    }

    mDescr = replacement;
}


void
AttributeSet::read(std::istream& is)
{
    this->readMetadata(is);
    this->readAttributes(is);
}


void
AttributeSet::write(std::ostream& os) const
{
    this->writeMetadata(os);
    this->writeAttributes(os);
}


void
AttributeSet::readMetadata(std::istream& is)
{
    mDescr->read(is);
}


void
AttributeSet::writeMetadata(std::ostream& os) const
{
    // build a vector of all attribute arrays that have a transient flag

    std::vector<size_t> transient;

    for (size_t i = 0; i < size(); i++) {
        const AttributeArray* array = this->getConst(i);
        if (array->isTransient()) {
            transient.push_back(i);
        }
    }

    // write out a descriptor without transient attributes

    if (transient.empty()) {
        mDescr->write(os);
    }
    else {
        Descriptor::Ptr descr = mDescr->duplicateDrop(transient);
        descr->write(os);
    }
}


void
AttributeSet::readAttributes(std::istream& is)
{
    if (!mDescr) {
        OPENVDB_THROW(IllegalValueException, "Attribute set descriptor not defined.");
    }

    AttrArrayVec(mDescr->size()).swap(mAttrs); // allocate vector

    for (size_t n = 0, N = mAttrs.size(); n < N; ++n) {
        mAttrs[n] = AttributeArray::create(mDescr->type(n), 1);
        mAttrs[n]->read(is);
    }
}


void
AttributeSet::writeAttributes(std::ostream& os) const
{
    for (size_t n = 0, N = mAttrs.size(); n < N; ++n) {
        mAttrs[n]->write(os);
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
    : mNameMap()
    , mTypes()
    , mGroupMap()
    , mMetadata()
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

    for (NameToPosMap::const_iterator it = mNameMap.begin(),
        end = mNameMap.end(); it != end; ++it) {
        const size_t index = rhs.find(it->first);

        if (index == INVALID_POS) return false;

        if (mTypes[it->second] != rhs.mTypes[index]) return false;
    }

    return std::equal(mGroupMap.begin(), mGroupMap.end(), rhs.mGroupMap.begin());
}


size_t
AttributeSet::Descriptor::memUsage() const
{
    size_t bytes = sizeof(NameToPosMap::mapped_type) * this->size();
    for (NameToPosMap::const_iterator it = mNameMap.begin(),
        end = mNameMap.end(); it != end; ++it) {
        bytes += it->first.capacity();
    }

    for (size_t n = 0, N = mTypes.size(); n < N; ++n) {
         bytes += mTypes[n].first.capacity();
         bytes += mTypes[n].second.capacity();
    }

    return sizeof(*this) + bytes;
}


size_t
AttributeSet::Descriptor::find(const std::string& name) const
{
    NameToPosMap::const_iterator it = mNameMap.find(name);
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
    NameToPosMap::iterator it = mNameMap.find(toName);
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

    for (MetaMap::ConstMetaIterator it = mMetadata.beginMeta(),
                                    itEnd = mMetadata.endMeta(); it != itEnd; ++it) {
        const Name name = it->first;

        // ignore non-default metadata
        if (!boost::starts_with(name, "default:"))   continue;

        const Name defaultName = name.substr(8, it->first.size() - 8);

        if (mNameMap.find(defaultName) == mNameMap.end()) {
            metaToErase.push_back(name);
        }
    }

    // remove this metadata

    for (std::vector<Name>::const_iterator  it = metaToErase.begin(),
                                            endIt = metaToErase.end(); it != endIt; ++it) {
        mMetadata.removeMeta(*it);
    }
}


size_t
AttributeSet::Descriptor::insert(const std::string& name, const NamePair& typeName)
{
    if (!validName(name))  throw RuntimeError("Attribute name contains invalid characters - " + name);

    size_t pos = INVALID_POS;
    NameToPosMap::iterator it = mNameMap.find(name);
    if (it != mNameMap.end()) {
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
AttributeSet::Descriptor::create(const NameAndTypeVec& attrs)
{
    Ptr descr(new Descriptor());
    for (size_t n = 0, N = attrs.size(); n < N; ++n) {
        const std::string& name = attrs[n].name;
        descr->insert(name, attrs[n].type);
    }
    return descr;
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::create(const NameAndTypeVec& attrs,
                                 const NameToPosMap& groupMap, const MetaMap& metadata)
{
    Ptr newDescriptor(create(attrs));

    newDescriptor->mGroupMap = groupMap;
    newDescriptor->mMetadata = metadata;

    return newDescriptor;
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::create(const NamePair& positionType)
{
    Ptr descr(new Descriptor());
    descr->insert("P", positionType);
    return descr;
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::duplicateAppend(const NameAndType& attribute) const
{
    Inserter attributes;

    this->appendTo(attributes.vec);
    attributes.add(attribute);

    return Descriptor::create(attributes.vec, mGroupMap, mMetadata);
}


AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::duplicateAppend(const NameAndTypeVec& vec) const
{
    Inserter attributes;

    this->appendTo(attributes.vec);
    attributes.add(vec);

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

    typedef std::map<size_t, std::string> PosToNameMap;

    PosToNameMap posToNameMap;

    for (NameToPosMap::const_iterator   it = mNameMap.begin(),
                                        endIt = mNameMap.end(); it != endIt; ++it) {

        posToNameMap[it->second] = it->first;
    }

    // std::map is sorted by key, so attributes can now be inserted in position order

    for (PosToNameMap::const_iterator   it = posToNameMap.begin(),
                                        endIt = posToNameMap.end(); it != endIt; ++it) {

        attrs.push_back(NameAndType(it->second, this->type(it->first)));
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
    struct Internal {
        static bool isNotValidChar(int c) { return !(isalnum(c) || (c == '_') || (c == '|') || (c == ':')); }
    };
    if (name.empty())   return false;
    return std::find_if(name.begin(), name.end(), Internal::isNotValidChar) == name.end();
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

        bool negate = boost::starts_with(token, "^") || boost::starts_with(token, "!");

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

    for(Index64 n = 0; n < arraylength; ++n) {
        writeString(os, mTypes[n].first);
        writeString(os, mTypes[n].second);
    }

    NameToPosMap::const_iterator it = mNameMap.begin(), endIt = mNameMap.end();
    for (; it != endIt; ++it) {
        writeString(os, it->first);
        os.write(reinterpret_cast<const char*>(&it->second), sizeof(Index64));
    }

    const Index64 grouplength = Index64(mGroupMap.size());
    os.write(reinterpret_cast<const char*>(&grouplength), sizeof(Index64));

    NameToPosMap::const_iterator groupIt = mGroupMap.begin(), endGroupIt = mGroupMap.end();
    for (; groupIt != endGroupIt; ++groupIt) {
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

    for(Index64 n = 0; n < arraylength; ++n) {
        const Name type1 = readString(is);
        const Name type2 = readString(is);
        mTypes[n] = NamePair(type1, type2);
    }

    mNameMap.clear();
    std::pair<std::string, size_t> nameAndOffset;

    for(Index64 n = 0; n < arraylength; ++n) {
        nameAndOffset.first = readString(is);
        if (!validName(nameAndOffset.first))  throw IoError("Attribute name contains invalid characters - " + nameAndOffset.first);
        is.read(reinterpret_cast<char*>(&nameAndOffset.second), sizeof(Index64));
        mNameMap.insert(nameAndOffset);
    }

    Index64 grouplength = 0;
    is.read(reinterpret_cast<char*>(&grouplength), sizeof(Index64));

    for(Index64 n = 0; n < grouplength; ++n) {
        nameAndOffset.first = readString(is);
        if (!validName(nameAndOffset.first))  throw IoError("Group name contains invalid characters - " + nameAndOffset.first);
        is.read(reinterpret_cast<char*>(&nameAndOffset.second), sizeof(Index64));
        mGroupMap.insert(nameAndOffset);
    }

    mMetadata.readMeta(is);
}



////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
