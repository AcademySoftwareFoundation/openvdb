
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file task.h

#ifndef OPENVDB_TASK_HAS_BEEN_INCLUDED
#define OPENVDB_TASK_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/task.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

using task = ::tbb::task;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_TASK_HAS_BEEN_INCLUDED
