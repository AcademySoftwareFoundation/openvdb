// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Nick Avramoussis
///
/// @file points/PointScatter.h
///
/// @brief Various point scattering methods for generating VDB Points.
///
///  All random number calls are made to the same generator to produce
///  temporarily consistent results in relation to the provided seed. This
///  comes with some multi-threaded performance trade-offs.

#ifndef OPENVDB_POINTS_POINT_SCATTER_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_SCATTER_HAS_BEEN_INCLUDED

#include <type_traits>
#include <algorithm>
#include <thread>
#include <random>

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/util/NullInterrupter.h>

#include "AttributeArray.h"
#include "PointCount.h"
#include "PointDataGrid.h"

#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief The free functions depend on the following class:
///
/// The @c InterrupterT template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = nullptr) // called when computations begin
///   void end()                             // called when computations end
///   bool wasInterrupted(int percent=-1)    // return true to break computation
///};
/// @endcode
///
/// @note If no template argument is provided for this InterrupterT
/// the util::NullInterrupter is used which implies that all
/// interrupter calls are no-ops (i.e. incurs no computational overhead).


/// @brief Uniformly scatter a total amount of points in active regions
///
/// @param grid         A source grid. The resulting PointDataGrid will copy this grids
///                     transform and scatter in its active voxelized topology.
/// @param count        The total number of points to scatter
/// @param seed         A seed for the RandGenT
/// @param spread       The spread of points as a scale from each voxels center. A value of
///                     1.0f indicates points can be placed anywhere within the voxel, where
///                     as a value of 0.0f will force all points to be created exactly at the
///                     centers of each voxel.
/// @param interrupter  An optional interrupter
/// @note returns the scattered PointDataGrid
template<
    typename GridT,
    typename RandGenT = std::mt19937,
    typename PositionArrayT = TypedAttributeArray<Vec3f, NullCodec>,
    typename PointDataGridT = Grid<
        typename points::TreeConverter<typename GridT::TreeType>::Type>,
    typename InterrupterT = util::NullInterrupter>
inline typename PointDataGridT::Ptr
uniformPointScatter(const GridT& grid,
                    const Index64 count,
                    const unsigned int seed = 0,
                    const float spread = 1.0f,
                    InterrupterT* interrupter = nullptr);

/// @brief Uniformly scatter a fixed number of points per active voxel. If the pointsPerVoxel
///        value provided is a fractional value, each voxel calculates a delta value of
///        how likely it is to contain an extra point.
///
/// @param grid            A source grid. The resulting PointDataGrid will copy this grids
///                        transform and scatter in its active voxelized topology.
/// @param pointsPerVoxel  The number of points to scatter per voxel
/// @param seed            A seed for the RandGenT
/// @param spread          The spread of points as a scale from each voxels center. A value of
///                        1.0f indicates points can be placed anywhere within the voxel, where
///                        as a value of 0.0f will force all points to be created exactly at the
///                        centers of each voxel.
/// @param interrupter     An optional interrupter
/// @note returns the scattered PointDataGrid

template<
    typename GridT,
    typename RandGenT = std::mt19937,
    typename PositionArrayT = TypedAttributeArray<Vec3f, NullCodec>,
    typename PointDataGridT = Grid<
        typename points::TreeConverter<typename GridT::TreeType>::Type>,
    typename InterrupterT = util::NullInterrupter>
inline typename PointDataGridT::Ptr
denseUniformPointScatter(const GridT& grid,
                         const float pointsPerVoxel,
                         const unsigned int seed = 0,
                         const float spread = 1.0f,
                         InterrupterT* interrupter = nullptr);

/// @brief Non uniformly scatter points per active voxel. The pointsPerVoxel value is used
///        to weight each grids cell value to compute a fixed number of points for every
///        active voxel. If the computed result is a fractional value, each voxel calculates
///        a delta value of how likely it is to contain an extra point.
///
/// @param grid            A source grid. The resulting PointDataGrid will copy this grids
///                        transform, voxelized topology and use its values to compute a
///                        target points per voxel. The grids ValueType must be convertible
///                        to a scalar value. Only active and larger than zero values will
///                        contain points.
/// @param pointsPerVoxel  The number of points to scatter per voxel
/// @param seed            A seed for the RandGenT
/// @param spread          The spread of points as a scale from each voxels center. A value of
///                        1.0f indicates points can be placed anywhere within the voxel, where
///                        as a value of 0.0f will force all points to be created exactly at the
///                        centers of each voxel.
/// @param interrupter     An optional interrupter
/// @note returns the scattered PointDataGrid
template<
    typename GridT,
    typename RandGenT = std::mt19937,
    typename PositionArrayT = TypedAttributeArray<Vec3f, NullCodec>,
    typename PointDataGridT = Grid<
        typename points::TreeConverter<typename GridT::TreeType>::Type>,
    typename InterrupterT = util::NullInterrupter>
inline typename PointDataGridT::Ptr
nonUniformPointScatter(const GridT& grid,
                       const float pointsPerVoxel,
                       const unsigned int seed = 0,
                       const float spread = 1.0f,
                       InterrupterT* interrupter = nullptr);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointScatterImpl.h"

#endif // OPENVDB_POINTS_POINT_SCATTER_HAS_BEEN_INCLUDED
