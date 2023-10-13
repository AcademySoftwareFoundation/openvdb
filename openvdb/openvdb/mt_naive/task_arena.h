
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file task_arena.h

#ifndef OPENVDB_TASK_ARENA_HAS_BEEN_INCLUDED
#define OPENVDB_TASK_ARENA_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/task_arena.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

namespace this_task_arena = ::tbb::this_task_arena;
using task_arena = ::tbb::task_arena;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_TASK_ARENA_HAS_BEEN_INCLUDED
