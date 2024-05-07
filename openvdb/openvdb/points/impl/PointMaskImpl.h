// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file PointMaskImpl.h
///
/// @author Dan Bailey
///

#ifndef OPENVDB_POINTS_POINT_MASK_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_MASK_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace point_mask_internal {

template <typename LeafT>
void voxelSum(LeafT& leaf, const Index offset, const typename LeafT::ValueType& value)
{
    leaf.modifyValue(offset, tools::valxform::SumOp<typename LeafT::ValueType>(value));
}

// overload PointDataLeaf access to use setOffsetOn(), as modifyValue()
// is intentionally disabled to avoid accidental usage

template <typename T, Index Log2Dim>
void voxelSum(PointDataLeafNode<T, Log2Dim>& leaf, const Index offset,
    const typename PointDataLeafNode<T, Log2Dim>::ValueType& value)
{
    leaf.setOffsetOn(offset, leaf.getValue(offset) + value);
}


/// @brief Combines multiple grids into one by stealing leaf nodes and summing voxel values
/// This class is designed to work with thread local storage containers such as tbb::combinable
template<typename GridT>
struct GridCombinerOp
{
    using CombinableT = typename tbb::combinable<GridT>;

    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueType = typename TreeT::ValueType;
    using SumOp = tools::valxform::SumOp<typename TreeT::ValueType>;

    GridCombinerOp(GridT& grid)
        : mTree(grid.tree()) {}

    void operator()(const GridT& grid)
    {
        for (auto leaf = grid.tree().beginLeaf(); leaf; ++leaf) {
            auto* newLeaf = mTree.probeLeaf(leaf->origin());
            if (!newLeaf) {
                // if the leaf doesn't yet exist in the new tree, steal it
                auto& tree = const_cast<GridT&>(grid).tree();
                mTree.addLeaf(tree.template stealNode<LeafT>(leaf->origin(),
                    zeroVal<ValueType>(), false));
            }
            else {
                // otherwise increment existing values
                for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                    voxelSum(*newLeaf, iter.offset(), ValueType(*iter));
                }
            }
        }
    }

private:
    TreeT& mTree;
}; // struct GridCombinerOp


/// @brief Compute scalar grid from PointDataGrid while evaluating the point filter
template <typename TreeT, typename PointDataTreeT, typename FilterT>
struct PointsToScalarOp
{
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename LeafT::ValueType;
    // This method is also used by PointCount so ValueT may not be bool
    static constexpr bool IsBool =
        std::is_same<ValueT, bool>::value;

    PointsToScalarOp(const PointDataTreeT& tree,
                     const FilterT& filter)
        : mPointDataAccessor(tree)
        , mFilter(filter) {}

    void operator()(LeafT& leaf, size_t /*idx*/) const
    {
        // assumes matching topology
        const auto* const pointLeaf =
            mPointDataAccessor.probeConstLeaf(leaf.origin());
        OPENVDB_ASSERT(pointLeaf);

        for (auto value = leaf.beginValueOn(); value; ++value) {
            const auto iter = pointLeaf->beginIndexVoxel(value.getCoord(), mFilter);
            if (IsBool) {
                if (!iter) value.setValueOn(false);
            }
            else {
                const Index64 count = points::iterCount(iter);
                if (count > Index64(0)) value.setValue(ValueT(count));
                else                    value.setValueOn(false);
            }
        }
    }

private:
    const tree::ValueAccessor<const PointDataTreeT> mPointDataAccessor;
    const FilterT& mFilter;
}; // struct PointsToScalarOp


/// @brief Compute scalar grid from PointDataGrid using a different transform
///        and while evaluating the point filter
template <typename GridT, typename PointDataGridT, typename FilterT, typename DeformerT>
struct PointsToTransformedScalarOp
{
    using PointDataLeafT = typename PointDataGridT::TreeType::LeafNodeType;
    using ValueT = typename GridT::TreeType::ValueType;
    using HandleT = AttributeHandle<Vec3f>;
    using CombinableT = typename GridCombinerOp<GridT>::CombinableT;

    PointsToTransformedScalarOp(const math::Transform& targetTransform,
                                const math::Transform& sourceTransform,
                                const FilterT& filter,
                                const DeformerT& deformer,
                                CombinableT& combinable)
        : mTargetTransform(targetTransform)
        , mSourceTransform(sourceTransform)
        , mFilter(filter)
        , mDeformer(deformer)
        , mCombinable(combinable) { }

    void operator()(const PointDataLeafT& leaf, size_t idx) const
    {
        DeformerT deformer(mDeformer);

        auto& grid = mCombinable.local();
        auto& countTree = grid.tree();
        tree::ValueAccessor<typename GridT::TreeType> accessor(countTree);

        deformer.reset(leaf, idx);

        auto handle = HandleT::create(leaf.constAttributeArray("P"));

        for (auto iter = leaf.beginIndexOn(mFilter); iter; iter++) {

            // extract index-space position

            Vec3d position = handle->get(*iter) + iter.getCoord().asVec3d();

            // if deformer is designed to be used in index-space, perform deformation prior
            // to transforming position to world-space, otherwise perform deformation afterwards

            if (DeformerTraits<DeformerT>::IndexSpace) {
                deformer.template apply<decltype(iter)>(position, iter);
                position = mSourceTransform.indexToWorld(position);
            }
            else {
                position = mSourceTransform.indexToWorld(position);
                deformer.template apply<decltype(iter)>(position, iter);
            }

            // determine coord of target grid

            const Coord ijk = mTargetTransform.worldToIndexCellCentered(position);

            // increment count in target voxel

            auto* newLeaf = accessor.touchLeaf(ijk);
            OPENVDB_ASSERT(newLeaf);
            voxelSum(*newLeaf, newLeaf->coordToOffset(ijk), ValueT(1));
        }
    }

private:
    const openvdb::math::Transform& mTargetTransform;
    const openvdb::math::Transform& mSourceTransform;
    const FilterT& mFilter;
    const DeformerT& mDeformer;
    CombinableT& mCombinable;
}; // struct PointsToTransformedScalarOp


template<typename TreeT, typename PointDataTreeT, typename FilterT>
inline typename TreeT::Ptr convertPointsToScalar(
    const PointDataTreeT& points,
    const FilterT& filter,
    bool threaded = true)
{
    using point_mask_internal::PointsToScalarOp;

    using ValueT = typename TreeT::ValueType;

    // copy the topology from the points tree

    typename TreeT::Ptr tree(new TreeT(/*background=*/false));
    tree->topologyUnion(points);

    // early exit if no leaves

    if (points.leafCount() == 0) return tree;

    // early exit if mask and no group logic

    if (std::is_same<ValueT, bool>::value && filter.state() == index::ALL) return tree;

    // evaluate point group filters to produce a subset of the generated mask

    tree::LeafManager<TreeT> leafManager(*tree);

    if (filter.state() == index::ALL) {
        NullFilter nullFilter;
        PointsToScalarOp<TreeT, PointDataTreeT, NullFilter> pointsToScalarOp(
            points, nullFilter);
        leafManager.foreach(pointsToScalarOp, threaded);
    } else {
        // build mask from points in parallel only where filter evaluates to true
        PointsToScalarOp<TreeT, PointDataTreeT, FilterT> pointsToScalarOp(
            points, filter);
        leafManager.foreach(pointsToScalarOp, threaded);
    }

    return tree;
}


template<typename GridT, typename PointDataGridT, typename FilterT, typename DeformerT>
inline typename GridT::Ptr convertPointsToScalar(
    PointDataGridT& points,
    const math::Transform& transform,
    const FilterT& filter,
    const DeformerT& deformer,
    bool threaded = true)
{
    using point_mask_internal::PointsToTransformedScalarOp;
    using point_mask_internal::GridCombinerOp;

    using CombinerOpT = GridCombinerOp<GridT>;
    using CombinableT = typename GridCombinerOp<GridT>::CombinableT;

    typename GridT::Ptr grid = GridT::create();
    grid->setTransform(transform.copy());

    // use the simpler method if the requested transform matches the existing one

    const math::Transform& pointsTransform = points.constTransform();

    if (transform == pointsTransform && std::is_same<NullDeformer, DeformerT>()) {
        using TreeT = typename GridT::TreeType;
        typename TreeT::Ptr tree =
            convertPointsToScalar<TreeT>(points.tree(), filter, threaded);
        grid->setTree(tree);
        return grid;
    }

    // early exit if no leaves

    if (points.constTree().leafCount() == 0)  return grid;

    // compute mask grids in parallel using new transform

    CombinableT combiner;

    tree::LeafManager<typename PointDataGridT::TreeType> leafManager(points.tree());

    if (filter.state() == index::ALL) {
        NullFilter nullFilter;
        PointsToTransformedScalarOp<GridT, PointDataGridT, NullFilter, DeformerT> pointsToScalarOp(
            transform, pointsTransform, nullFilter, deformer, combiner);
        leafManager.foreach(pointsToScalarOp, threaded);
    } else {
        PointsToTransformedScalarOp<GridT, PointDataGridT, FilterT, DeformerT> pointsToScalarOp(
            transform, pointsTransform, filter, deformer, combiner);
        leafManager.foreach(pointsToScalarOp, threaded);
    }

    // combine the mask grids into one

    CombinerOpT combineOp(*grid);
    combiner.combine_each(combineOp);

    return grid;
}


} // namespace point_mask_internal

/// @endcond

////////////////////////////////////////


template <typename PointDataTreeT, typename MaskTreeT, typename FilterT>
inline typename std::enable_if<std::is_base_of<TreeBase, PointDataTreeT>::value &&
    std::is_same<typename MaskTreeT::ValueType, bool>::value, typename MaskTreeT::Ptr>::type
convertPointsToMask(const PointDataTreeT& tree,
    const FilterT& filter,
    bool threaded)
{
    return point_mask_internal::convertPointsToScalar<MaskTreeT>(
        tree, filter, threaded);
}


template<typename PointDataGridT, typename MaskGridT, typename FilterT>
inline typename std::enable_if<std::is_base_of<GridBase, PointDataGridT>::value &&
    std::is_same<typename MaskGridT::ValueType, bool>::value, typename MaskGridT::Ptr>::type
convertPointsToMask(
    const PointDataGridT& points,
    const FilterT& filter,
    bool threaded)
{
    using PointDataTreeT = typename PointDataGridT::TreeType;
    using MaskTreeT = typename MaskGridT::TreeType;

    typename MaskTreeT::Ptr tree =
        convertPointsToMask<PointDataTreeT, MaskTreeT, FilterT>
            (points.tree(), filter, threaded);

    typename MaskGridT::Ptr grid(new MaskGridT(tree));
    grid->setTransform(points.transform().copy());
    return grid;
}


template<typename PointDataGridT, typename MaskT, typename FilterT>
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(
    const PointDataGridT& points,
    const openvdb::math::Transform& transform,
    const FilterT& filter,
    bool threaded)
{
    // This is safe because the PointDataGrid can only be modified by the deformer
    using AdapterT = TreeAdapter<typename PointDataGridT::TreeType>;
    auto& nonConstPoints = const_cast<typename AdapterT::NonConstGridType&>(points);

    NullDeformer deformer;
    return point_mask_internal::convertPointsToScalar<MaskT>(
        nonConstPoints, transform, filter, deformer, threaded);
}


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_MASK_IMPL_HAS_BEEN_INCLUDED
