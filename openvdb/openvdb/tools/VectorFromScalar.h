// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file VectorMerge.h
///
/// @brief Functions to produce a vector-valued grid
/// from separate scalar grids
///
/// @author Tim Straubigner

#ifndef OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Threaded method to convert three scalar-valued grids into a single vector-valued grid.
///        The transforms and resolutions of the three input grids must be equal. The new vector
///        grid topology is the union of all scalar grid topologies.
///
/// @return A shared pointer to a new grid with the same tree configuration, transform, and
///         resolution as the input grids. Each voxel of the new grid is a vector whose
///         component values are taken from the input grids at the corresponding voxel
///         location, using background values when one of the source grids has no voxels
///         defined there.
///
/// @param x                    Grid to use as the first vector component.
/// @param y                    Grid to use as the second vector component.
/// @param z                    Grid to use as the third vector component.
template<typename ScalarGridT>
typename ScalarGridT::template ValueConverter<math::Vec3<typename ScalarGridT::ValueType>>::Type::Ptr
vectorFromScalar(const ScalarGridT& x, const ScalarGridT& y, const ScalarGridT& z)
{
    // TODO
    throw std::runtime_error("not implemented");
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb



#endif // OPENVDB_TOOLS_VECTOR_FROM_SCALAR_HAS_BEEN_INCLUDED
