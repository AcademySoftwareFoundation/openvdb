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

#include "Metadata.h"

#include <map>
#include <sstream>
#include <vector>
#include <tbb/mutex.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

typedef tbb::mutex Mutex;
typedef Mutex::scoped_lock Lock;

typedef Metadata::Ptr (*createMetadata)();
typedef std::map<Name, createMetadata> MetadataFactoryMap;
typedef MetadataFactoryMap::const_iterator MetadataFactoryMapCIter;

struct LockedMetadataTypeRegistry {
    LockedMetadataTypeRegistry() {}
    ~LockedMetadataTypeRegistry() {}
    Mutex mMutex;
    MetadataFactoryMap mMap;
};

// Declare this at file scope to ensure thread-safe initialization
static Mutex theInitMetadataTypeRegistryMutex;

// Global function for accessing the regsitry
static LockedMetadataTypeRegistry*
getMetadataTypeRegistry()
{
    Lock lock(theInitMetadataTypeRegistryMutex);

    static LockedMetadataTypeRegistry *registry = NULL;

    if(registry == NULL) {
#if defined(__ICC)
__pragma(warning(disable:1711)) // disable ICC "assignment to static variable" warnings
#endif
        registry = new LockedMetadataTypeRegistry();
#if defined(__ICC)
__pragma(warning(default:1711))
#endif
    }

    return registry;
}

bool
Metadata::isRegisteredType(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    return (registry->mMap.find(typeName) != registry->mMap.end());
}

void
Metadata::registerType(const Name &typeName, Metadata::Ptr (*createMetadata)())
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    if (registry->mMap.find(typeName) != registry->mMap.end()) {
        OPENVDB_THROW(KeyError,
            "Cannot register " << typeName << ". Type is already registered");
    }

    registry->mMap[typeName] = createMetadata;
}

void
Metadata::unregisterType(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    registry->mMap.erase(typeName);
}

Metadata::Ptr
Metadata::createMetadata(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    MetadataFactoryMapCIter iter = registry->mMap.find(typeName);

    if (iter == registry->mMap.end()) {
        OPENVDB_THROW(LookupError,
            "Cannot create metadata for unregistered type " << typeName);
    }

    return (iter->second)();
}

void
Metadata::clearRegistry()
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    Lock lock(registry->mMutex);

    registry->mMap.clear();
}


////////////////////////////////////////


void
UnknownMetadata::readValue(std::istream& is, Index32 numBytes)
{
    // Read and discard the metadata (without seeking, because
    // the stream might not be seekable).
    const size_t BUFFER_SIZE = 1024;
    std::vector<char> buffer(BUFFER_SIZE);
    for (Index32 bytesRemaining = numBytes; bytesRemaining > 0; ) {
        const Index32 bytesToSkip = std::min<Index32>(bytesRemaining, BUFFER_SIZE);
        is.read(&buffer[0], bytesToSkip);
        bytesRemaining -= bytesToSkip;
    }
}


void
UnknownMetadata::writeValue(std::ostream&) const
{
    OPENVDB_THROW(TypeError, "Metadata has unknown type");
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
