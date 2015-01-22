///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
//
/// @file AttributeArray.cc
///
/// @note For evaluation purposes, do not distribute.
///
/// @authors Mihai Alden, Peter Cucka

#include "AttributeArray.h"

#include <algorithm> // std::equal


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


namespace {

typedef std::map<std::string, AttributeArray::FactoryMethod> AttributeFactoryMap;
typedef AttributeFactoryMap::const_iterator AttributeFactoryMapCIter;

struct LockedAttributeRegistry
{
    tbb::spin_mutex     mMutex;
    AttributeFactoryMap mMap;
};


// Declare this at file scope to ensure thread-safe initialization.
tbb::spin_mutex sInitAttributeRegistryMutex;


// Global function for accessing the registry
LockedAttributeRegistry*
getAttributeRegistry()
{
    tbb::spin_mutex::scoped_lock lock(sInitAttributeRegistryMutex);

    static LockedAttributeRegistry* registry = NULL;

    if (registry == NULL) {

#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.
__pragma(warning(disable:1711))
#endif
        // This assignment is mutex-protected and therefore thread-safe.
        registry = new LockedAttributeRegistry();

#ifdef __ICC
__pragma(warning(default:1711))
#endif

    }

    return registry;
}

} // unnamed namespace


////////////////////////////////////////

// AttributeArray implementation


AttributeArray::Ptr
AttributeArray::create(const Name& type, size_t length)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    AttributeFactoryMapCIter iter = registry->mMap.find(type);

    if (iter == registry->mMap.end()) {
        OPENVDB_THROW(LookupError, "Cannot create attribute of unregistered type " << type);
    }

    return (iter->second)(length);
}


bool
AttributeArray::isRegistered(const Name &type)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);
    return (registry->mMap.find(type) != registry->mMap.end());
}


void
AttributeArray::clearRegistry()
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);
    registry->mMap.clear();
}


void
AttributeArray::registerType(const Name& type, FactoryMethod factory)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    AttributeFactoryMapCIter iter = registry->mMap.find(type);

    if (iter == registry->mMap.end()) {
        registry->mMap[type] = factory;

    } else if (iter->second != factory) {
        OPENVDB_THROW(KeyError, "Attribute type " << type
            << " is already registered with different factory method.");
    }
}


void
AttributeArray::unregisterType(const Name& type)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    registry->mMap.erase(type);
}


void
AttributeArray::setTransient(bool state)
{
    if (state) mFlags |= Int16(TRANSIENT);
    else mFlags &= ~Int16(TRANSIENT);
}


void
AttributeArray::setHidden(bool state)
{
    if (state) mFlags |= Int16(HIDDEN);
    else mFlags &= ~Int16(HIDDEN);
}


bool
AttributeArray::operator==(const AttributeArray& other) const {
    if(this->mCompressedBytes != other.mCompressedBytes ||
       this->mFlags != other.mFlags) return false;
    return this->isEqual(other);
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


void
AttributeSet::copyAttributeValues(const Index n, const AttributeSet& attributeSet, const Index source)
{
    const Descriptor& sourceDescriptor = attributeSet.descriptor();

    if (sourceDescriptor != *mDescr) {
        OPENVDB_THROW(LookupError, "Attribute set descriptors are not equal.");
    }

    // copy the attribute values for each attribute array

    for (size_t i = 0; i < this->size(); i++) {
        AttributeArray* array = this->get(i);
        array->set(n, *attributeSet.getConst(i), source);
    }
}


void
AttributeSet::appendAttributes(const Descriptor::NameAndTypeVec& vec)
{
    Descriptor::Ptr descriptor = mDescr->duplicateAppend(vec);

    this->appendAttributes(vec, *mDescr, descriptor);
}


void
AttributeSet::appendAttributes(const Descriptor::NameAndTypeVec& vec,
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

    // append the new arrays

    for (Descriptor::NameAndTypeVec::const_iterator it = vec.begin(), itEnd = vec.end(); it != itEnd; ++it) {
        mAttrs.push_back(AttributeArray::create(it->type, arrayLength));
    }
}


void
AttributeSet::dropAttributes(const std::vector<size_t>& pos)
{
    Descriptor::Ptr descriptor = mDescr->duplicateDrop(pos);

    this->dropAttributes(pos, *mDescr, descriptor);
}


void
AttributeSet::dropAttributes(   const std::vector<size_t>& pos,
                                const Descriptor& expected, DescriptorPtr& replacement)
{
    // ensure the descriptor is as expected
    if (*mDescr != expected) {
        OPENVDB_THROW(LookupError, "Cannot drop attributes as descriptors do not match.")
    }

    mDescr = replacement;

    // sort the positions to remove

    std::vector<size_t> orderedPos(pos);
    std::sort(orderedPos.begin(), orderedPos.end());

    // erase elements in reverse order

    for (std::vector<size_t>::const_reverse_iterator    it = pos.rbegin();
                                                        it != pos.rend(); ++it) {
        mAttrs.erase(mAttrs.begin() + (*it));
    }
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
    mDescr->write(os);
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
{
}


bool
AttributeSet::Descriptor::operator==(const Descriptor& rhs) const
{
    if (this == &rhs) return true;

    if (mTypes.size()   != rhs.mTypes.size() ||
        mNameMap.size() != rhs.mNameMap.size()) {
        return false;
    }

    for (size_t n = 0, N = mTypes.size(); n < N; ++n) {
        if (mTypes[n] != rhs.mTypes[n]) return false;
    }

    return std::equal(mNameMap.begin(), mNameMap.end(), rhs.mNameMap.begin());
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
         bytes += mTypes[n].capacity();
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
    size_t pos = INVALID_POS;

    // check if the new name is already used.
    NameToPosMap::iterator it = mNameMap.find(toName);
    if (it != mNameMap.end()) return pos;

    it = mNameMap.find(fromName);
    if (it != mNameMap.end()) {
        pos = it->second;
        mNameMap.erase(it);
        mNameMap[toName] = pos;
    }
    return pos;
}


size_t
AttributeSet::Descriptor::insert(const std::string& name, const std::string& typeName)
{
    size_t pos = INVALID_POS;
    NameToPosMap::iterator it = mNameMap.find(name);
    if (it != mNameMap.end()) {
        pos = it->second;
    } else {

        if (!AttributeArray::isRegistered(typeName)) {
            OPENVDB_THROW(KeyError, "Failed to insert '" << name
                << "' with unregistered attribute type '" << typeName);
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
        if (name.length() > 0) {
            descr->insert(name, attrs[n].type);
        }
    }
    return descr;
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::duplicateAppend(const NameAndTypeVec& vec) const
{
    Inserter attributes;

    this->appendTo(attributes.vec);
    attributes.add(vec);

    return Descriptor::create(attributes.vec);
}

AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::duplicateDrop(const std::vector<size_t>& pos) const
{
    NameAndTypeVec vec;
    this->appendTo(vec);

    // sort the positions to remove

    std::vector<size_t> orderedPos(pos);
    std::sort(orderedPos.begin(), orderedPos.end());

    // erase elements in reverse order

    for (std::vector<size_t>::const_reverse_iterator    it = pos.rbegin();
                                                        it != pos.rend(); ++it) {
        vec.erase(vec.begin() + (*it));
    }

    return Descriptor::create(vec);
}

void
AttributeSet::Descriptor::appendTo(NameAndTypeVec& attrs) const
{
    const size_t size = mNameMap.size();

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


void
AttributeSet::Descriptor::write(std::ostream& os) const
{
    const Index64 arraylength = Index64(mTypes.size());
    os.write(reinterpret_cast<const char*>(&arraylength), sizeof(Index64));

    for(Index64 n = 0; n < arraylength; ++n) {
        writeString(os, mTypes[n]);
    }

    NameToPosMap::const_iterator it = mNameMap.begin(), endIt = mNameMap.end();
    for (; it != endIt; ++it) {
        writeString(os, it->first);
        os.write(reinterpret_cast<const char*>(&it->second), sizeof(Index64));
    }
}


void
AttributeSet::Descriptor::read(std::istream& is)
{
    Index64 arraylength = 0;
    is.read(reinterpret_cast<char*>(&arraylength), sizeof(Index64));

    std::vector<std::string>(size_t(arraylength)).swap(mTypes);

    for(Index64 n = 0; n < arraylength; ++n) {
         mTypes[n] = readString(is);
    }

    mNameMap.clear();
    std::pair<std::string, size_t> nameAndOffset;

    for(Index64 n = 0; n < arraylength; ++n) {
        nameAndOffset.first = readString(is);
        is.read(reinterpret_cast<char*>(&nameAndOffset.second), sizeof(Index64));
        mNameMap.insert(nameAndOffset);
    }
}



////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
