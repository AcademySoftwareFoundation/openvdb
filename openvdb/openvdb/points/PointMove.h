// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey
///
/// @file PointMove.h
///
/// @brief Ability to move VDB Points using a custom deformer.
///
/// Deformers used when moving points are in world space by default and must adhere
/// to the interface described in the example below:
/// @code
/// struct MyDeformer
/// {
///     // A reset is performed on each leaf in turn before the points in that leaf are
///     // deformed. A leaf and leaf index (standard leaf traversal order) are supplied as
///     // the arguments, which matches the functor interface for LeafManager::foreach().
///     template <typename LeafNoteType>
///     void reset(LeafNoteType& leaf, size_t idx);
///
///     // Evaluate the deformer and modify the given position to generate the deformed
///     // position. An index iterator is supplied as the argument to allow querying the
///     // point offset or containing voxel coordinate.
///     template <typename IndexIterT>
///     void apply(Vec3d& position, const IndexIterT& iter) const;
/// };
/// @endcode
///
/// @note The DeformerTraits struct (defined in PointMask.h) can be used to configure
/// a deformer to evaluate in index space.

#ifndef OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/util/Assert.h>

#include "PointDataGrid.h"
#include "PointMask.h"

#include <tbb/concurrent_vector.h>

#include <algorithm>
#include <iterator> // for std::begin(), std::end()
#include <map>
#include <numeric> // for std::iota()
#include <tuple>
#include <unordered_map>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

// dummy object for future use
namespace future { struct Advect { }; }

/// @brief Move points in a PointDataGrid using a custom deformer
/// @param points           the PointDataGrid containing the points to be moved.
/// @param deformer         a custom deformer that defines how to move the points.
/// @param filter           an optional index filter
/// @param objectNotInUse   for future use, this object is currently ignored
/// @param threaded         enable or disable threading  (threading is enabled by default)
template <typename PointDataGridT, typename DeformerT, typename FilterT = NullFilter>
inline void movePoints(PointDataGridT& points,
                       DeformerT& deformer,
                       const FilterT& filter = NullFilter(),
                       future::Advect* objectNotInUse = nullptr,
                       bool threaded = true);


/// @brief Move points in a PointDataGrid using a custom deformer and a new transform
/// @param points           the PointDataGrid containing the points to be moved.
/// @param transform        target transform to use for the resulting points.
/// @param deformer         a custom deformer that defines how to move the points.
/// @param filter           an optional index filter
/// @param objectNotInUse   for future use, this object is currently ignored
/// @param threaded         enable or disable threading  (threading is enabled by default)
template <typename PointDataGridT, typename DeformerT, typename FilterT = NullFilter>
inline void movePoints(PointDataGridT& points,
                       const math::Transform& transform,
                       DeformerT& deformer,
                       const FilterT& filter = NullFilter(),
                       future::Advect* objectNotInUse = nullptr,
                       bool threaded = true);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointMoveImpl.h"

#endif // OPENVDB_POINTS_POINT_MOVE_HAS_BEEN_INCLUDED
