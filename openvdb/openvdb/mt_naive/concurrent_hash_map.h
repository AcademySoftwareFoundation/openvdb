
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file concurrent_hash_map.h

#ifndef OPENVDB_CONCURRENT_HASH_MAP_HAS_BEEN_INCLUDED
#define OPENVDB_CONCURRENT_HASH_MAP_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/concurrent_hash_map.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

using namespace ::tbb;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_CONCURRENT_HASH_MAP_HAS_BEEN_INCLUDED
