
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file tick_count.h

#ifndef OPENVDB_TICK_COUNT_HAS_BEEN_INCLUDED
#define OPENVDB_TICK_COUNT_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/tick_count.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

using tick_count = ::tbb::tick_count;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_TICK_COUNT_HAS_BEEN_INCLUDED
