
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file enumerable_thread_specific.h

#ifndef OPENVDB_ENUMERABLE_THREAD_SPECIFIC_HAS_BEEN_INCLUDED
#define OPENVDB_ENUMERABLE_THREAD_SPECIFIC_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/enumerable_thread_specific.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

template <typename T,
              typename Allocator=::tbb::cache_aligned_allocator<T>,
              ::tbb::ets_key_usage_type ETS_key_type=::tbb::ets_no_key >
using enumerable_thread_specific = ::tbb::enumerable_thread_specific<T, Allocator, ETS_key_type>;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_ENUMERABLE_THREAD_SPECIFIC_HAS_BEEN_INCLUDED
