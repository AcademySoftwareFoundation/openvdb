// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "openvdb.h"
#include "io/DelayedLoadMetadata.h"
#include "points/PointDataGrid.h"
#include "tools/PointIndexGrid.h"
#include "util/logging.h"

#include <atomic>
#include <mutex>

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

#if OPENVDB_ABI_VERSION_NUMBER <= 8
    #error ABI <= 8 is no longer supported
#endif

// If using an OPENVDB_ABI_VERSION_NUMBER that has been deprecated, issue an
// error directive. This can be optionally suppressed by defining:
//   OPENVDB_USE_DEPRECATED_ABI_<VERSION>=ON.
#ifndef OPENVDB_USE_DEPRECATED_ABI_9
    #if OPENVDB_ABI_VERSION_NUMBER == 9
        #error ABI = 9 is deprecated, CMake option OPENVDB_USE_DEPRECATED_ABI_9 suppresses this error
    #endif
#endif
#ifndef OPENVDB_USE_DEPRECATED_ABI_10
    #if OPENVDB_ABI_VERSION_NUMBER == 10
        #error ABI = 10 is deprecated, CMake option OPENVDB_USE_DEPRECATED_ABI_10 suppresses this error
    #endif
#endif

// If using a future OPENVDB_ABI_VERSION_NUMBER, issue an error directive.
// This can be optionally suppressed by defining:
//   OPENVDB_USE_FUTURE_ABI_<VERSION>=ON.
#ifndef OPENVDB_USE_FUTURE_ABI_12
    #if OPENVDB_ABI_VERSION_NUMBER == 12
        #error ABI = 12 is still in active development and has not been finalized, \
CMake option OPENVDB_USE_FUTURE_ABI_12 suppresses this error
    #endif
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace {
inline std::mutex& GetInitMutex()
{
    static std::mutex sInitMutex;
    return sInitMutex;
}

std::atomic<bool> sIsInitialized{false};
}

/// @todo  Change registerX() methods to simply be register()...
template <typename GridT> struct RegisterGrid { inline void operator()() { GridT::registerGrid(); } };
template <typename MetaT> struct RegisterMeta { inline void operator()() { MetaT::registerType(); } };
template <typename MapT>  struct RegisterMap  { inline void operator()() { MapT::registerMap(); } };

void
initialize()
{
    if (sIsInitialized.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(GetInitMutex());
    if (sIsInitialized.load(std::memory_order_acquire)) return; // Double-checked lock

    logging::initialize();

    // Register metadata.
    Metadata::clearRegistry();
    MetaTypes::foreach<RegisterMeta>();

    // Register maps
    math::MapRegistry::clear();
    MapTypes::foreach<RegisterMap>();

    // Register common grid types.
    GridBase::clearRegistry();
    GridTypes::foreach<RegisterGrid>();

    // Register types associated with point index grids.
    Metadata::registerType(typeNameAsString<PointIndex32>(), Int32Metadata::createMetadata);
    Metadata::registerType(typeNameAsString<PointIndex64>(), Int64Metadata::createMetadata);

    // Register types associated with point data grids.
    points::internal::initialize();

#ifdef OPENVDB_USE_BLOSC
    blosc_init();
    if (blosc_set_compressor("lz4") < 0) {
        OPENVDB_LOG_WARN("Blosc LZ4 compressor is unavailable");
    }
    /// @todo blosc_set_nthreads(int nthreads);
#endif

#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.
// This assignment is mutex-protected and therefore thread-safe.
__pragma(warning(disable:1711))
#endif

    sIsInitialized.store(true, std::memory_order_release);

#ifdef __ICC
__pragma(warning(default:1711))
#endif
}


void
uninitialize()
{
    std::lock_guard<std::mutex> lock(GetInitMutex());
#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.
// This assignment is mutex-protected and therefore thread-safe.
__pragma(warning(disable:1711))
#endif

    sIsInitialized.store(false, std::memory_order_seq_cst); // Do we need full memory order?

#ifdef __ICC
__pragma(warning(default:1711))
#endif

    Metadata::clearRegistry();
    GridBase::clearRegistry();
    math::MapRegistry::clear();
    points::internal::uninitialize();

#ifdef OPENVDB_USE_BLOSC
    // We don't want to destroy Blosc, because it might have been
    // initialized by some other library.
    //blosc_destroy();
#endif
}

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
