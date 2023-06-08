// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/PointCount.h
///
/// @author Dan Bailey
///
/// @brief  Methods for counting points in VDB Point grids.

#ifndef OPENVDB_POINTS_POINT_COUNT_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_COUNT_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include "PointDataGrid.h"
#include "PointMask.h"
#include "IndexFilter.h"

#include <tbb/parallel_reduce.h>

#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Count the total number of points in a PointDataTree
/// @param tree         the PointDataTree in which to count the points
/// @param filter       an optional index filter
/// @param inCoreOnly   if true, points in out-of-core leaf nodes are not counted
/// @param threaded     enable or disable threading  (threading is enabled by default)
template <typename PointDataTreeT, typename FilterT = NullFilter>
inline Index64 pointCount(  const PointDataTreeT& tree,
                            const FilterT& filter = NullFilter(),
                            const bool inCoreOnly = false,
                            const bool threaded = true);

/// @brief Populate an array of cumulative point offsets per leaf node.
/// @param pointOffsets     array of offsets to be populated
/// @param tree             the PointDataTree from which to populate the offsets
/// @param filter           an optional index filter
/// @param inCoreOnly       if true, points in out-of-core leaf nodes are ignored
/// @param threaded         enable or disable threading  (threading is enabled by default)
/// @return The final cumulative point offset.
template <typename PointDataTreeT, typename FilterT = NullFilter>
inline Index64 pointOffsets(std::vector<Index64>& pointOffsets,
                            const PointDataTreeT& tree,
                            const FilterT& filter = NullFilter(),
                            const bool inCoreOnly = false,
                            const bool threaded = true);

/// @brief Generate a new grid with voxel values to store the number of points per voxel
/// @param grid             the PointDataGrid to use to compute the count grid
/// @param filter           an optional index filter
/// @note The return type of the grid must be an integer or floating-point scalar grid.
template <typename PointDataGridT,
    typename GridT = typename PointDataGridT::template ValueConverter<Int32>::Type,
    typename FilterT = NullFilter>
inline typename GridT::Ptr
pointCountGrid( const PointDataGridT& grid,
                const FilterT& filter = NullFilter());

/// @brief Generate a new grid that uses the supplied transform with voxel values to store the
///        number of points per voxel.
/// @param grid             the PointDataGrid to use to compute the count grid
/// @param transform        the transform to use to compute the count grid
/// @param filter           an optional index filter
/// @note The return type of the grid must be an integer or floating-point scalar grid.
template <typename PointDataGridT,
    typename GridT = typename PointDataGridT::template ValueConverter<Int32>::Type,
    typename FilterT = NullFilter>
inline typename GridT::Ptr
pointCountGrid( const PointDataGridT& grid,
                const openvdb::math::Transform& transform,
                const FilterT& filter = NullFilter());

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointCountImpl.h"

#endif // OPENVDB_POINTS_POINT_COUNT_HAS_BEEN_INCLUDED
