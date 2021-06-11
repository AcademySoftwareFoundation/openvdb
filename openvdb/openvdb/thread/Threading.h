// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file Threading.h

#ifndef OPENVDB_THREAD_THREADING_HAS_BEEN_INCLUDED
#define OPENVDB_THREAD_THREADING_HAS_BEEN_INCLUDED

#include "openvdb/version.h"

/// @note tbb/blocked_range.h is the ONLY include that persists from TBB 2020
///   to TBB 2021 that itself includes the TBB specific version header files.
///   In TBB 2020, the version header was called tbb/stddef.h. In 2021, it's
///   called tbb/version.h. We include tbb/blocked_range.h here to indirectly
///   access the version defines in a consistent way so that downstream
///   software doesn't need to provide compile time defines.
#include <tbb/blocked_range.h>
#include <tbb/task.h>
#include <tbb/task_group.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace thread {

inline bool cancelGroupExecution()
{
    // @note 12000 was the 2021.1-beta05 release. The 2021.1-beta08 release
    //   introduced current_context().
#if TBB_INTERFACE_VERSION >= 12002
    auto ctx = tbb::task::current_context();
    return ctx ? ctx->cancel_group_execution() : false;
#else
    return tbb::task::self().cancel_group_execution();
#endif
}

inline bool isGroupExecutionCancelled()
{
    // @note 12000 was the 2021.1-beta05 release. The 2021.1-beta08 release
    //   introduced current_context().
#if TBB_INTERFACE_VERSION >= 12002
    auto ctx = tbb::task::current_context();
    return ctx ? ctx->is_group_execution_cancelled() : false;
#else
    return tbb::task::self().is_cancelled();
#endif
}

} // namespace thread
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_THREAD_THREADING_HAS_BEEN_INCLUDED
