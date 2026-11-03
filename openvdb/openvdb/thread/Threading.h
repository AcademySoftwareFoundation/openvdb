// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file Threading.h

#ifndef OPENVDB_THREAD_THREADING_HAS_BEEN_INCLUDED
#define OPENVDB_THREAD_THREADING_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/version.h>
#include <tbb/task_group.h>
#include <tbb/blocked_range.h>
#include <tbb/task.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace thread {

inline OPENVDB_UBSAN_SUPPRESS("vptr") bool cancelGroupExecution()
{
    auto ctx = tbb::task::current_context();
    return ctx ? ctx->cancel_group_execution() : false;
}

inline OPENVDB_UBSAN_SUPPRESS("vptr") bool isGroupExecutionCancelled()
{
    auto ctx = tbb::task::current_context();
    return ctx ? ctx->is_group_execution_cancelled() : false;
}

} // namespace thread
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_THREAD_THREADING_HAS_BEEN_INCLUDED
