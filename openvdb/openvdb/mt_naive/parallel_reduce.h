
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file parallel_reduce.h

#ifndef OPENVDB_PARALLEL_REDUCE_HAS_BEEN_INCLUDED
#define OPENVDB_PARALLEL_REDUCE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <openvdb/mt/partitioner.h>

#include <tbb/parallel_reduce.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

//! Parallel iteration with reduction and default partitioner.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( Range && range, Body && body ) {
    ::tbb::parallel_reduce( std::forward<Range>(range), std::forward<Body>(body) );
}

template<typename Range, typename Value, typename RealBody>
Value parallel_reduce( Range && range, Value && identity, RealBody && real_body) {
    return ::tbb::parallel_reduce(
        std::forward<Range>(range),
        std::forward<Value>(identity),
        std::forward<RealBody>(real_body)
    );
}

template<typename Range, typename Value, typename RealBody, typename Reduction>
Value parallel_reduce( Range && range, Value && identity, RealBody && real_body, Reduction && reduction) {
    return ::tbb::parallel_reduce(
        std::forward<Range>(range),
        std::forward<Value>(identity),
        std::forward<RealBody>(real_body),
        std::forward<Reduction>(reduction)
    );
}

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_PARALLEL_REDUCE_HAS_BEEN_INCLUDED
