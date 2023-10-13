
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file concurrent_vector.h

#ifndef OPENVDB_CONCURRENT_VECTOR_HAS_BEEN_INCLUDED
#define OPENVDB_CONCURRENT_VECTOR_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/concurrent_vector.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

template<typename T, class A = ::tbb::cache_aligned_allocator<T> >
using concurrent_vector = ::tbb::concurrent_vector<T, A>;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_CONCURRENT_VECTOR_HAS_BEEN_INCLUDED