
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file parallel_sort.h

#ifndef OPENVDB_PARALLEL_SORT_HAS_BEEN_INCLUDED
#define OPENVDB_PARALLEL_SORT_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/parallel_sort.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace mt = ::tbb;

} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_PARALLEL_SORT_HAS_BEEN_INCLUDED
