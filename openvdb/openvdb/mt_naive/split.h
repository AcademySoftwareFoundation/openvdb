
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file blocked_range.h

#ifndef OPENVDB_SPLIT_HAS_BEEN_INCLUDED
#define OPENVDB_SPLIT_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/tbb_stddef.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

using split = ::tbb::split; // TODO LT: Switch over to the below implementation

#if 0
//! Dummy type that distinguishes splitting constructor from copy constructor.
/**
 * See description of parallel_for and parallel_reduce for example usages.
 */
class split {
};
#endif

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_SPLIT_HAS_BEEN_INCLUDED
