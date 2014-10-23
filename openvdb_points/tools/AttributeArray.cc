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


////////////////////////////////////////

// AttributeSet implementation

/*
bool validTypes(const std::string& t1, const std::string& t2, const Descriptor::TypeConstraint c)
{
    if (c == Descriptor::CONSTRAIN_TYPE) {
        const size_t p = t1.find_first_of('_');
        if (p == std::string::npos || t1.compare(0, p, t2) != 0) {
            return false;
        }
    } else if (c == Descriptor::CONSTRAIN_TYPE_AND_CODEC && t1 != t2) {
        return false;
    }
    return true;
}
*/
AttributeSet::AttributeSet()
    : mDescr(new Descriptor())
    , mAttrs()
{
}


AttributeSet::AttributeSet(const AttributeSet& rhs)
    : mDescr(rhs.mDescr)
    , mAttrs(rhs.mAttrs)
{
}


AttributeSet::AttributeSet(const DescriptorPtr& descr)
    : mDescr(descr)
    , mAttrs(descr->size(), AttributeArray::Ptr())
{
}


void
AttributeSet::update(const DescriptorPtr& descr)
{
    AttrArrayVec attrs(descr->size(), AttributeArray::Ptr());

    /// @todo preserve similarly named attributes

    mAttrs.swap(attrs);
    mDescr = descr;
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
    return pos != INVALID_POS ? this->replace(pos, attr) : pos;
}


size_t
AttributeSet::replace(size_t pos, const AttributeArray::Ptr& attr)
{
    assert(pos != INVALID_POS);
    assert(pos < mAttrs.size());

    /*if (!validTypes(attr->type(), mDescr->type(pos), mDescr->typeConstraint())) {
        return INVALID_POS;
    }*/

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
AttributeSet::isUnique(size_t pos) const
{
    assert(pos != INVALID_POS);
    assert(pos < mAttrs.size());
    return mAttrs[pos].unique();
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
AttributeSet::read(std::istream&)
{
    /// @todo
}

void
AttributeSet::write(std::ostream&) const
{
    /// @todo
}


////////////////////////////////////////

// AttributeSet::Descriptor implementation


tbb::atomic<Index64> AttributeSet::Descriptor::sNextId;


AttributeSet::Descriptor::Descriptor()
    : mNameMap()
    , mTypes()
    , mId(sNextId++)
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
    NameToPosMap::iterator it = mNameMap.find(fromName);
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
AttributeSet::Descriptor::create(const std::vector<NameAndType>& attrs)
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


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
