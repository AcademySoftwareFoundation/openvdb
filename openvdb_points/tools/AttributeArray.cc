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
/// @file AttributeArray.cc
///
/// @authors Dan Bailey, Mihai Alden, Peter Cucka

#include <map>

#include <openvdb_points/tools/AttributeArray.h>

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


namespace {

using AttributeFactoryMap = std::map<NamePair, AttributeArray::FactoryMethod>;

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

    static LockedAttributeRegistry* registry = nullptr;

    if (registry == nullptr) {

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
AttributeArray::create(const NamePair& type, size_t length, Index stride)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    auto iter = registry->mMap.find(type);

    if (iter == registry->mMap.end()) {
        OPENVDB_THROW(LookupError, "Cannot create attribute of unregistered type " << type.first << "_" << type.second);
    }

    return (iter->second)(length, stride);
}


bool
AttributeArray::isRegistered(const NamePair& type)
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
AttributeArray::registerType(const NamePair& type, FactoryMethod factory)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    auto iter = registry->mMap.find(type);

    if (iter == registry->mMap.end()) {
        registry->mMap[type] = factory;

    } else if (iter->second != factory) {
        OPENVDB_THROW(KeyError, "Attribute type " << type.first << "_" << type.second
            << " is already registered with different factory method.");
    }
}


void
AttributeArray::unregisterType(const NamePair& type)
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
    this->loadData();
    other.loadData();
    if(this->mCompressedBytes != other.mCompressedBytes ||
       this->mFlags != other.mFlags) return false;
    return this->isEqual(other);
}



////////////////////////////////////////



} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
