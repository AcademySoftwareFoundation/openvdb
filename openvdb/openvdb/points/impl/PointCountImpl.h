// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey
///
/// @file PointCountImpl.h
///

#ifndef OPENVDB_POINTS_POINT_COUNT_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_COUNT_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

template <typename PointDataTreeT, typename FilterT>
Index64 pointCount(const PointDataTreeT& tree,
                   const FilterT& filter,
                   const bool inCoreOnly,
                   const bool threaded)
{
    using LeafManagerT = tree::LeafManager<const PointDataTreeT>;
    using LeafRangeT = typename LeafManagerT::LeafRange;

    auto countLambda =
        [&filter, &inCoreOnly] (const LeafRangeT& range, Index64 sum) -> Index64 {
            for (const auto& leaf : range) {
                if (inCoreOnly && leaf.buffer().isOutOfCore())  continue;
                auto state = filter.state(leaf);
                if (state == index::ALL) {
                    sum += leaf.pointCount();
                } else if (state != index::NONE) {
                    sum += iterCount(leaf.beginIndexAll(filter));
                }
            }
            return sum;
        };

    LeafManagerT leafManager(tree);
    if (threaded) {
        return tbb::parallel_reduce(leafManager.leafRange(), Index64(0), countLambda,
            [] (Index64 n, Index64 m) -> Index64 { return n + m; });
    }
    else {
        return countLambda(leafManager.leafRange(), Index64(0));
    }
}


template <typename PointDataTreeT, typename FilterT>
Index64 pointOffsets(   std::vector<Index64>& pointOffsets,
                        const PointDataTreeT& tree,
                        const FilterT& filter,
                        const bool inCoreOnly,
                        const bool threaded)
{
    using LeafT = typename PointDataTreeT::LeafNodeType;
    using LeafManagerT = typename tree::LeafManager<const PointDataTreeT>;

    // allocate and zero values in point offsets array

    pointOffsets.assign(tree.leafCount(), Index64(0));
    if (pointOffsets.empty()) return 0;

    // compute total points per-leaf

    LeafManagerT leafManager(tree);
    leafManager.foreach(
        [&pointOffsets, &filter, &inCoreOnly](const LeafT& leaf, size_t pos) {
            if (inCoreOnly && leaf.buffer().isOutOfCore())  return;
            auto state = filter.state(leaf);
            if (state == index::ALL) {
                pointOffsets[pos] = leaf.pointCount();
            } else if (state != index::NONE) {
                pointOffsets[pos] = iterCount(leaf.beginIndexAll(filter));
            }
        },
    threaded);

    // turn per-leaf totals into cumulative leaf totals

    Index64 pointOffset(pointOffsets[0]);
    for (size_t n = 1; n < pointOffsets.size(); n++) {
        pointOffset += pointOffsets[n];
        pointOffsets[n] = pointOffset;
    }

    return pointOffset;
}


template <typename PointDataGridT, typename GridT, typename FilterT>
typename GridT::Ptr
pointCountGrid( const PointDataGridT& points,
                const FilterT& filter)
{
    static_assert(std::is_integral<typename GridT::ValueType>::value ||
                  std::is_floating_point<typename GridT::ValueType>::value,
        "openvdb::points::pointCountGrid must return an integer or floating-point scalar grid");

    using PointDataTreeT = typename PointDataGridT::TreeType;
    using TreeT = typename GridT::TreeType;

    typename TreeT::Ptr tree =
        point_mask_internal::convertPointsToScalar<TreeT, PointDataTreeT, FilterT>
            (points.tree(), filter);

    typename GridT::Ptr grid(new GridT(tree));
    grid->setTransform(points.transform().copy());
    return grid;
}


template <typename PointDataGridT, typename GridT, typename FilterT>
typename GridT::Ptr
pointCountGrid( const PointDataGridT& points,
                const openvdb::math::Transform& transform,
                const FilterT& filter)
{
    static_assert(  std::is_integral<typename GridT::ValueType>::value ||
                    std::is_floating_point<typename GridT::ValueType>::value,
        "openvdb::points::pointCountGrid must return an integer or floating-point scalar grid");

    // This is safe because the PointDataGrid can only be modified by the deformer
    using AdapterT = TreeAdapter<typename PointDataGridT::TreeType>;
    auto& nonConstPoints = const_cast<typename AdapterT::NonConstGridType&>(points);

    NullDeformer deformer;
    return point_mask_internal::convertPointsToScalar<GridT>(
        nonConstPoints, transform, filter, deformer);
}


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_COUNT_IMPL_HAS_BEEN_INCLUDED
