// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "Codec.h"

#include <openvdb/codecs/ScalarCodec.h>
#include <openvdb/codecs/BoolCodec.h>
#include <openvdb/codecs/PointIndexCodec.h>
#include <openvdb/codecs/ValueMaskCodec.h>

#include <openvdb/openvdb.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

namespace {

using CodecFactoryMap = std::map<std::string, Codec::Ptr>;
using CodecFactoryMapCIter = CodecFactoryMap::const_iterator;

struct LockedCodecRegistry {
    LockedCodecRegistry() {}
    ~LockedCodecRegistry() {}
    std::mutex mMutex;
    CodecFactoryMap mMap;
};

// Global function for accessing the registry
static LockedCodecRegistry*
getCodecRegistry()
{
    static LockedCodecRegistry registry;
    return &registry;
}

} // unnamed namespace


namespace internal {

template <typename GridT>
struct RegisterCodec { inline void operator()() { CodecRegistry::registerCodec<codecs::ScalarCodec<GridT>>(); } };

template <typename GridT>
struct RegisterConvertCodec {
    inline void operator()()
    {
        CodecRegistry::registerCodec<codecs::ScalarCodec<MaskGrid, GridT, CodecMode::ReadOnly>>();
        CodecRegistry::registerCodec<codecs::ScalarCodec<BoolGrid, GridT, CodecMode::ReadOnly>>();
    }
}; // struct RegisterConvertCodec

void initialize()
{
    NumericGridTypes::foreach<RegisterCodec>();
    Vec3GridTypes::foreach<RegisterCodec>();

    CodecRegistry::registerCodec<codecs::BoolCodec<BoolGrid>>();
    CodecRegistry::registerCodec<codecs::ValueMaskCodec<MaskGrid>>();
    CodecRegistry::registerCodec<codecs::PointIndexCodec<tools::PointIndexGrid>>();

    // register the plugin that converts from scalar to mask/bool
    NumericGridTypes::foreach<RegisterConvertCodec>();

    // register the plugin that converts from float to half
    CodecRegistry::registerCodec<codecs::ScalarCodec<HalfGrid, FloatGrid, CodecMode::ReadOnly>>();
}

void uninitialize()
{
    CodecRegistry::clear();
}

} // namespace internal


////////////////////////////////////////


bool
CodecRegistry::isRegistered(const std::string& name)
{
    LockedCodecRegistry* registry = getCodecRegistry();
    std::lock_guard<std::mutex> lock(registry->mMutex);

    return (registry->mMap.find(name) != registry->mMap.end());
}


void
CodecRegistry::registerCodecByName(const std::string& name, Codec::Ptr&& codec)
{
    LockedCodecRegistry* registry = getCodecRegistry();
    std::lock_guard<std::mutex> lock(registry->mMutex);

    if (registry->mMap.find(name) != registry->mMap.end()) {
        OPENVDB_THROW(KeyError,
            "Cannot register codec " << name << ". Codec is already registered");
    }

    registry->mMap[name] = std::move(codec);
}


Codec*
CodecRegistry::get(const std::string& name)
{
    LockedCodecRegistry* registry = getCodecRegistry();
    std::lock_guard<std::mutex> lock(registry->mMutex);

    CodecFactoryMapCIter iter = registry->mMap.find(name);

    return (iter != registry->mMap.end()) ? iter->second.get() : nullptr;
}


void
CodecRegistry::clear()
{
    LockedCodecRegistry* registry = getCodecRegistry();
    std::lock_guard<std::mutex> lock(registry->mMutex);

    registry->mMap.clear();
}


} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

