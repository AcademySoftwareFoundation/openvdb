// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Nick Avramoussis
///
/// @file PointRasterizeTrilinear.h
///
/// @brief Transfer schemes for rasterizing point data
///

#ifndef OPENVDB_POINTS_RASTERIZE_TRILINEAR_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_RASTERIZE_TRILINEAR_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tree/ValueAccessor.h>

#include "PointDataGrid.h"
#include "PointMask.h"
#include "PointTransfer.h"

#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

///
template <typename ValueT, bool Staggered = true>
struct TrilinearTraits
{
    using ResultT = typename std::conditional<
        VecTraits<ValueT>::IsVec, ValueT, math::Vec3<ValueT>>::type;
    template <typename PointDataTreeT>
    using TreeT = typename PointDataTreeT::template ValueConverter<ResultT>::Type;
};

///
template <typename ValueT>
struct TrilinearTraits<ValueT, false>
{
    using ResultT = ValueT;
    template <typename PointDataTreeT>
    using TreeT = typename PointDataTreeT::template ValueConverter<ResultT>::Type;
};

/// @brief Perform weighted trilinear rasterization of all points within a
///   voxel. This method takes and returns a tree i.e. ignores grid
///   transformations.
/// @details Accumulates values and weights according to a simple 0-1-0 weighted
///   hat function. This algorithm is an exact inverse of a trilinear
///   interpolation and thus a key method used in PIC/FLIP style simulations.
///   Returns a tree of the same precision as the input source attribute, but
///   may be of a different math type depending on the value of the Staggered
///   template attribute. If Staggered is true, this method produces values at
///   each voxels negative faces, causing scalar attributes to produce
///   math::Vec3<ValueT> tree types. The result Tree type is equal to:
///     TrilinearTraits<ValueT, Staggered>::template TreeT<PointDataTreeT>
/// @tparam Staggered whether to perform a staggered or collocated rasterization
/// @tparam ValueT    the value type of the point attribute to rasterize
/// @param points     the point tree to be rasterized
/// @param attribute  the name of the attribute to rasterize. Must be a scalar
///   or Vec3 attribute.
/// @param filter     an optional point filter to use
template <bool Staggered,
    typename ValueT,
    typename FilterT = NullFilter,
    typename PointDataTreeT = PointDataTree>
inline auto
rasterizeTrilinear(const PointDataTreeT& points,
           const std::string& attribute,
           const FilterT& filter = NullFilter());

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointRasterizeTrilinearImpl.h"

#endif //OPENVDB_POINTS_RASTERIZE_TRILINEAR_HAS_BEEN_INCLUDED
