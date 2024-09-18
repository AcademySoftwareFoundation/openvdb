// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/PointMask.h
///
/// @author Dan Bailey
///
/// @brief  Methods for extracting masks from VDB Point grids.

#ifndef OPENVDB_POINTS_POINT_MASK_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_MASK_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/ValueTransformer.h> // valxform::SumOp
#include <openvdb/util/Assert.h>

#include "PointDataGrid.h"
#include "IndexFilter.h"

#include <tbb/combinable.h>

#include <type_traits>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Extract a Mask Tree from a Point Data Tree
/// @param tree         the PointDataTree to extract the mask from.
/// @param filter       an optional index filter
/// @param threaded     enable or disable threading  (threading is enabled by default)
template <typename PointDataTreeT,
          typename MaskTreeT = typename PointDataTreeT::template ValueConverter<bool>::Type,
          typename FilterT = NullFilter>
inline typename std::enable_if<std::is_base_of<TreeBase, PointDataTreeT>::value &&
    std::is_same<typename MaskTreeT::ValueType, bool>::value, typename MaskTreeT::Ptr>::type
convertPointsToMask(const PointDataTreeT& tree,
                    const FilterT& filter = NullFilter(),
                    bool threaded = true);

/// @brief Extract a Mask Grid from a Point Data Grid
/// @param grid         the PointDataGrid to extract the mask from.
/// @param filter       an optional index filter
/// @param threaded     enable or disable threading  (threading is enabled by default)
/// @note this method is only available for Bool Grids and Mask Grids
template <typename PointDataGridT,
          typename MaskGridT = typename PointDataGridT::template ValueConverter<bool>::Type,
          typename FilterT = NullFilter>
inline typename std::enable_if<std::is_base_of<GridBase, PointDataGridT>::value &&
    std::is_same<typename MaskGridT::ValueType, bool>::value, typename MaskGridT::Ptr>::type
convertPointsToMask(const PointDataGridT& grid,
                    const FilterT& filter = NullFilter(),
                    bool threaded = true);

/// @brief Extract a Mask Grid from a Point Data Grid using a new transform
/// @param grid         the PointDataGrid to extract the mask from.
/// @param transform    target transform for the mask.
/// @param filter       an optional index filter
/// @param threaded     enable or disable threading  (threading is enabled by default)
/// @note this method is only available for Bool Grids and Mask Grids
template <typename PointDataGridT,
          typename MaskT = typename PointDataGridT::template ValueConverter<bool>::Type,
          typename FilterT = NullFilter>
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(const PointDataGridT& grid,
                    const openvdb::math::Transform& transform,
                    const FilterT& filter = NullFilter(),
                    bool threaded = true);

/// @brief No-op deformer (adheres to the deformer interface documented in PointMove.h)
struct NullDeformer
{
    template <typename LeafT>
    void reset(LeafT&, size_t /*idx*/ = 0) { }

    template <typename IterT>
    void apply(Vec3d&, IterT&) const { }
};

/// @brief Deformer Traits for optionally configuring deformers to be applied
/// in index-space. The default is world-space.
template <typename DeformerT>
struct DeformerTraits
{
    static const bool IndexSpace = false;
};

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointMaskImpl.h"

#endif // OPENVDB_POINTS_POINT_MASK_HAS_BEEN_INCLUDED
