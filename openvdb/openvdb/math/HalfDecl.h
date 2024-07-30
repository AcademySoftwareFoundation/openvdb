// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_HALFDECL_HAS_BEEN_INCLUDED
#define OPENVDB_HALFDECL_HAS_BEEN_INCLUDED

#ifdef OPENVDB_USE_IMATH_HALF
#ifdef OPENVDB_IMATH_VERSION
#include <Imath/half.h>
#else
#include <OpenEXR/half.h>
#endif
namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {
using half = half;
}}}
#else
#include <openvdb/math/Half.h>
namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {
using half = internal::half;
}}}
#endif


#endif // OPENVDB_HALFDECL_HAS_BEEN_INCLUDED
