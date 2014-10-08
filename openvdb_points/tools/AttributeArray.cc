// DreamWorks Animation LLC Confidential Information.
// TM and (c) 2014 DreamWorks Animation LLC.  All Rights Reserved.
// Reproduction in whole or in part without prior written permission of a
// duly authorized representative is prohibited.
//
/// @file AttributeArray.cc
///
/// @note For evaluation purposes, do not distribute.
///
/// @authors Mihai Alden, Peter Cucka

#include "AttributeArray.h"

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
AttributeArray::create(const openvdb::Name& type, size_t length)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    AttributeFactoryMapCIter iter = registry->mMap.find(type);

    if (iter == registry->mMap.end()) {
        OPENVDB_THROW(openvdb::LookupError, "Cannot create attribute of unregistered type " << type);
    }

    return (iter->second)(length);
}


bool
AttributeArray::isRegistered(const openvdb::Name &type)
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
AttributeArray::registerType(const openvdb::Name& type, FactoryMethod factory)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    if (registry->mMap.find(type) != registry->mMap.end()) {
        OPENVDB_THROW(openvdb::KeyError, "Attribute type " << type << " is already registered");
    }

    registry->mMap[type] = factory;
}


void
AttributeArray::unregisterType(const openvdb::Name& type)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    registry->mMap.erase(type);
}


void
AttributeArray::setTransient(bool state)
{
    if (state) mFlags |= TRANSIENT;
    else mFlags &= ~TRANSIENT;
}


void
AttributeArray::setHidden(bool state)
{
    if (state) mFlags |= HIDDEN;
    else mFlags &= ~HIDDEN;
}


////////////////////////////////////////

// AttributeSet implementation


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


size_t
AttributeSet::Descriptor::memUsage() const
{
    size_t strBytes = 0;
    for (AttrDictionary::const_iterator it = mDictionary.begin(),
        end = mDictionary.end(); it != end; ++it) {
        strBytes += it->first.size();
    }
    return sizeof(*this) + sizeof(size_t) * mDictionary.size() + strBytes;
}


size_t
AttributeSet::Descriptor::find(const std::string& name) const
{
    AttrDictionary::const_iterator it = mDictionary.find(name);
    if (it != mDictionary.end()) {
        return it->second.pos;
    }
    return INVALID_POS;
}


size_t
AttributeSet::Descriptor::insert(const std::string& name, const std::string& typeName)
{
    size_t pos = INVALID_POS;
    AttrDictionary::iterator it = mDictionary.find(name);
    if (it != mDictionary.end()) {
        pos = it->second.pos;
    } else {
        pos = mNextPos++;
        AttributeInfo info;
        info.pos = pos;
        info.typeName = typeName;
        mDictionary.insert(it, AttrDictionary::value_type(name, info));
    }
    return pos;
}


AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::create(const std::set<std::string>& names)
{
    Ptr descr(new Descriptor());

    for (std::set<std::string>::const_iterator it = names.begin(),
        end = names.end(); it != end; ++it) {
        descr->insert(*it);
    }
    return descr;
}


AttributeSet::Descriptor::Ptr
AttributeSet::Descriptor::create(const std::string& names)
{
    std::vector<std::string> splitvec;
    boost::algorithm::split(splitvec, names, boost::is_any_of(", "));

    std::set<std::string> nameset;
    for (size_t n = 0, N = splitvec.size(); n < N; ++n) {
        if (0 < splitvec[n].length()) nameset.insert(splitvec[n]);
    }

    return create(nameset);
}

////////////////////////////////////////











} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// TM and (c) 2014 DreamWorks Animation LLC.  All Rights Reserved.
// Reproduction in whole or in part without prior written permission of a
// duly authorized representative is prohibited.
