
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file parallel_for.h

#ifndef OPENVDB_PARALLEL_FOR_HAS_BEEN_INCLUDED
#define OPENVDB_PARALLEL_FOR_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/parallel_for.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

using namespace ::tbb;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_PARALLEL_FOR_HAS_BEEN_INCLUDED
