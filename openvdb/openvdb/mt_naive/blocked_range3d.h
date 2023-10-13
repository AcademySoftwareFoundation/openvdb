
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file blocked_range3d.h

#ifndef OPENVDB_BLOCKED_RANGE3D_HAS_BEEN_INCLUDED
#define OPENVDB_BLOCKED_RANGE3D_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/blocked_range3d.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

template<typename PageValue, typename RowValue=PageValue, typename ColValue=RowValue>
using blocked_range3d = ::tbb::blocked_range3d<PageValue, RowValue, ColValue>;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_BLOCKED_RANGE3D_HAS_BEEN_INCLUDED
