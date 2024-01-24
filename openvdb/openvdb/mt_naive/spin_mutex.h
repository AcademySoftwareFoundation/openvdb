
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file spin_mutex.h

#ifndef OPENVDB_SPIN_MUTEX_HAS_BEEN_INCLUDED
#define OPENVDB_SPIN_MUTEX_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <tbb/spin_mutex.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

using spin_mutex = ::tbb::spin_mutex;

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_SPIN_MUTEX_HAS_BEEN_INCLUDED
