// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @author Nick Avramoussis
///
/// @file PointRasterizeTrilinear.h
///
/// @brief Weighted trilinear rasterization of point data
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
#include <openvdb/util/Assert.h>
#include <openvdb/simd/Simd.h>

#include "PointDataGrid.h"
#include "PointMask.h"
#include "PointTransfer.h"

#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief  Traits for staggered (MAC) trilinear rasterization. This method
///   only works on scalar or Vec3 attribute types and only ever returns a Vec3
///   tree. It is typically used for velocity rasterization in fluid
///   simulations to avoid the pressure-velocity decoupling problem that occurs
///   on collocated (cell-centered) grids. The resulting Vec3 tree effectively
///   represents the velocity values on the cell faces rather than the cell
///   centers.
template <typename ValueT, bool Staggered = true>
struct TrilinearTraits
{
private:
    using FltT = typename types_internal::flt_t<sizeof(typename ValueTraits<ValueT>::ElementType)*CHAR_BIT>::type;
    static_assert(ValueTraits<ValueT>::IsScalar ||
        (ValueTraits<ValueT>::IsVec && ValueTraits<ValueT>::Size == 3),
        "Source attribute type must be scalar or Vec3 for staggered rasterization.");
public:
    /// @brief  Resulting rasterized Tree ValueType (always Vec3)
    using ResultT = math::Vec3<FltT>;
    /// @brief  Resulting rasterized TreeType
    template <typename PointDataTreeT>
    using TreeT = typename PointDataTreeT::template ValueConverter<ResultT>::Type;
};

/// @brief  Traits for collocated, cell-centered rasterization. Work for any
///   scalar, vector or matrix attribute. Note that integer values are
///   interpolated at their respective floating point precision and returned
///   as trees at that precision.
template <typename ValueT>
struct TrilinearTraits<ValueT, false>
{
private:
    using FltT = typename types_internal::flt_t<sizeof(typename ValueTraits<ValueT>::ElementType)*CHAR_BIT>::type;
public:
    /// @brief  Resulting rasterized Tree ValueType
    using ResultT = typename ConvertElementType<ValueT, FltT>::Type;
    /// @brief  Resulting rasterized TreeType
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
/// @param attribute  the name of the attribute to rasterize. Must be a Vec3
///   attribute for Staggered rasterization. Otherwise, can be any scalar,
///   Vector or Matrix type. Integer values are interpolated at their float
///   bitwidth precision and returned as a tree at that precision e.g:
///     int32 -> float, FloatTree.
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
