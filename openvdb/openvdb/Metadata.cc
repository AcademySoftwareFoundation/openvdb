// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "Metadata.h"

#include <algorithm> // for std::min()
#include <map>
#include <mutex>
#include <sstream>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

using createMetadata = Metadata::Ptr (*)();
using MetadataFactoryMap = std::map<Name, createMetadata>;
using MetadataFactoryMapCIter = MetadataFactoryMap::const_iterator;

struct LockedMetadataTypeRegistry {
    LockedMetadataTypeRegistry() {}
    ~LockedMetadataTypeRegistry() {}
    std::mutex mMutex;
    MetadataFactoryMap mMap;
};

// Global function for accessing the regsitry
static LockedMetadataTypeRegistry*
getMetadataTypeRegistry()
{
    static LockedMetadataTypeRegistry registry;
    return &registry;
}

bool
Metadata::isRegisteredType(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    std::lock_guard<std::mutex> lock(registry->mMutex);

    return (registry->mMap.find(typeName) != registry->mMap.end());
}

void
Metadata::registerType(const Name &typeName, Metadata::Ptr (*createMetadata)())
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    std::lock_guard<std::mutex> lock(registry->mMutex);

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
    std::lock_guard<std::mutex> lock(registry->mMutex);

    registry->mMap.erase(typeName);
}

Metadata::Ptr
Metadata::createMetadata(const Name &typeName)
{
    LockedMetadataTypeRegistry *registry = getMetadataTypeRegistry();
    std::lock_guard<std::mutex> lock(registry->mMutex);

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
    std::lock_guard<std::mutex> lock(registry->mMutex);

    registry->mMap.clear();
}


////////////////////////////////////////


bool
Metadata::operator==(const Metadata& other) const
{
    if (other.size() != this->size()) return false;
    if (other.typeName() != this->typeName()) return false;

    std::ostringstream
        bytes(std::ios_base::binary),
        otherBytes(std::ios_base::binary);
    try {
        this->writeValue(bytes);
        other.writeValue(otherBytes);
        return (bytes.str() == otherBytes.str());
    } catch (Exception&) {}
    return false;
}


////////////////////////////////////////


Metadata::Ptr
UnknownMetadata::copy() const
{
    Metadata::Ptr metadata{new UnknownMetadata{mTypeName}};
    static_cast<UnknownMetadata*>(metadata.get())->setValue(mBytes);
    return metadata;
}


void
UnknownMetadata::copy(const Metadata& other)
{
    std::ostringstream ostr(std::ios_base::binary);
    other.write(ostr);
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    const auto numBytes = readSize(istr);
    readValue(istr, numBytes);
}


void
UnknownMetadata::readValue(std::istream& is, Index32 numBytes)
{
    mBytes.clear();
    if (numBytes > 0) {
        ByteVec buffer(numBytes);
        is.read(reinterpret_cast<char*>(&buffer[0]), numBytes);
        mBytes.swap(buffer);
    }
}


void
UnknownMetadata::writeValue(std::ostream& os) const
{
    if (!mBytes.empty()) {
        os.write(reinterpret_cast<const char*>(&mBytes[0]), mBytes.size());
    }
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
