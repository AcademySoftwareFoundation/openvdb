///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

/// @file points/PointMask.h
///
/// @author Dan Bailey
///
/// @brief  Methods for extracting masks from VDB Point grids.

#ifndef OPENVDB_POINTS_POINT_MASK_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_MASK_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/ValueTransformer.h> // valxform::SumOp

#include "PointDataGrid.h"
#include "IndexFilter.h"

#include <tbb/combinable.h>

#include <type_traits>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


/// @brief Extract a Mask Grid from a Point Data Grid
/// @param grid         the PointDataGrid to extract the mask from.
/// @param filter       an optional index filter
/// @note this method is only available for Bool Grids and Mask Grids
template <typename PointDataGridT,
          typename MaskT = typename PointDataGridT::template ValueConverter<bool>::Type,
          typename FilterT = NullFilter>
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(const PointDataGridT& grid,
                    const FilterT& filter = NullFilter());


/// @brief Extract a Mask Grid from a Point Data Grid using a new transform
/// @param grid         the PointDataGrid to extract the mask from.
/// @param transform    target transform for the mask.
/// @param filter       an optional index filter
/// @note this method is only available for Bool Grids and Mask Grids
template <typename PointDataGridT,
          typename MaskT = typename PointDataGridT::template ValueConverter<bool>::Type,
          typename FilterT = NullFilter>
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(const PointDataGridT& grid,
                    const openvdb::math::Transform& transform,
                    const FilterT& filter = NullFilter());


////////////////////////////////////////


namespace point_mask_internal {

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
                    newLeaf->modifyValue(iter.getCoord(), SumOp(*iter));
                }
            }
        }
    }

private:
    TreeT& mTree;
}; // struct GridCombinerOp


/// @brief Compute scalar grid from PointDataGrid while evaluating the point filter
template <typename GridT, typename PointDataGridT, typename FilterT>
struct PointsToScalarOp
{
    using TreeT = typename GridT::TreeType;
    using LeafManagerT = typename tree::LeafManager<TreeT>;
    using ValueT = typename TreeT::LeafNodeType::ValueType;

    PointsToScalarOp(   const PointDataGridT& grid,
                        const FilterT& filter)
        : mPointDataAccessor(grid.getConstAccessor())
        , mFilter(filter) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {

            const auto* const pointLeaf =
                mPointDataAccessor.probeConstLeaf(leaf->origin());

            // assumes matching topology
            assert(pointLeaf);

            for (auto value = leaf->beginValueOn(); value; ++value) {
                const Index64 count = points::iterCount(
                    pointLeaf->beginIndexVoxel(value.getCoord(), mFilter));
                if (count > Index64(0)) {
                    value.setValue(static_cast<ValueT>(count));}
                else {
                    // disable any empty voxels
                    value.setValueOn(false);
                }
            }
        }
    }

private:
    const typename PointDataGridT::ConstAccessor mPointDataAccessor;
    const FilterT& mFilter;
}; // struct PointsToScalarOp


/// @brief Compute scalar grid from PointDataGrid using a different transform
///        and while evaluating the point filter
template <typename GridT, typename PointDataGridT, typename FilterT>
struct PointsToTransformedScalarOp
{
    using LeafManagerT = tree::LeafManager<const typename PointDataGridT::TreeType>;
    using ValueType = typename GridT::TreeType::LeafNodeType::ValueType;
    using HandleT = AttributeHandle<Vec3f>;
    using CombinableT = typename GridCombinerOp<GridT>::CombinableT;

    PointsToTransformedScalarOp(const math::Transform& newTransform,
                         const math::Transform& transform,
                         const FilterT& filter,
                         CombinableT& combinable)
        : mNewTransform(newTransform)
        , mTransform(transform)
        , mFilter(filter)
        , mCombinable(combinable) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const
    {
        auto& grid = mCombinable.local();
        auto& countTree = grid.tree();
        tree::ValueAccessor<typename GridT::TreeType> accessor(countTree);

        for (auto leaf = range.begin(); leaf; ++leaf) {

            auto handle = HandleT::create(leaf->constAttributeArray("P"));

            for (auto iter = leaf->beginIndexOn(mFilter); iter; iter++) {
                const Vec3d position = mTransform.indexToWorld(handle->get(*iter) +
                    iter.getCoord().asVec3d());
                const Coord ijk = mNewTransform.worldToIndexCellCentered(position);

                accessor.modifyValue(ijk, tools::valxform::SumOp<ValueType>(1));
            }
        }
    }

private:
    const openvdb::math::Transform& mNewTransform;
    const openvdb::math::Transform& mTransform;
    const FilterT& mFilter;
    CombinableT& mCombinable;
}; // struct PointsToTransformedScalarOp


template<typename PointDataGridT, typename GridT, typename FilterT>
inline typename GridT::Ptr convertPointsToScalar(
    const PointDataGridT& points,
    const FilterT& filter)
{
    using point_mask_internal::PointsToScalarOp;

    using GridTreeT = typename GridT::TreeType;
    using ValueT = typename GridTreeT::ValueType;

    // copy the topology from the points grid

    typename GridTreeT::Ptr tree(new GridTreeT(points.constTree(),
        false, openvdb::TopologyCopy()));
    typename GridT::Ptr grid = GridT::create(tree);
    grid->setTransform(points.transform().copy());

    // early exit if no leaves

    if (points.constTree().leafCount() == 0)            return grid;

    FilterT newFilter(filter);

    // early exit if mask and no group logic

    if (std::is_same<ValueT, bool>::value && filter.state() == index::ALL) return grid;

    // evaluate point group filters to produce a subset of the generated mask

    tree::LeafManager<GridTreeT> leafManager(*tree);

    if (filter.state() == index::ALL) {
        NullFilter nullFilter;
        PointsToScalarOp<GridT, PointDataGridT, NullFilter> pointsToScalarOp(
            points, nullFilter);
        tbb::parallel_for(leafManager.leafRange(), pointsToScalarOp);
    } else {
        // build mask from points in parallel only where filter evaluates to true
        FilterT newFilter(filter);
        PointsToScalarOp<GridT, PointDataGridT, FilterT> pointsToScalarOp(
            points, newFilter);
        tbb::parallel_for(leafManager.leafRange(), pointsToScalarOp);
    }

    return grid;
}


template<typename PointDataGridT, typename GridT, typename FilterT>
inline typename GridT::Ptr convertPointsToScalar(
    const PointDataGridT& points,
    const openvdb::math::Transform& transform,
    const FilterT& filter)
{
    using point_mask_internal::PointsToTransformedScalarOp;
    using point_mask_internal::GridCombinerOp;

    using CombinerOpT = GridCombinerOp<GridT>;
    using CombinableT = typename GridCombinerOp<GridT>::CombinableT;

    // use the simpler method if the requested transform matches the existing one

    const openvdb::math::Transform& pointsTransform = points.constTransform();

    if (transform == pointsTransform) {
        return convertPointsToScalar<PointDataGridT, GridT>(
            points, filter);
    }

    typename GridT::Ptr grid = GridT::create();
    grid->setTransform(transform.copy());

    // early exit if no leaves

    if (points.constTree().leafCount() == 0)  return grid;

    // compute mask grids in parallel using new transform

    CombinableT combiner;

    tree::LeafManager<const typename PointDataGridT::TreeType> leafManager(points.tree());

    if (filter.state() == index::ALL) {
        NullFilter nullFilter;
        PointsToTransformedScalarOp<GridT, PointDataGridT, NullFilter> pointsToScalarOp(
            transform, pointsTransform, nullFilter, combiner);
        tbb::parallel_for(leafManager.leafRange(), pointsToScalarOp);
    } else {
        FilterT newFilter(filter);
        PointsToTransformedScalarOp<GridT, PointDataGridT, FilterT> pointsToScalarOp(
            transform, pointsTransform, newFilter, combiner);
        tbb::parallel_for(leafManager.leafRange(), pointsToScalarOp);
    }

    // combine the mask grids into one

    CombinerOpT combineOp(*grid);
    combiner.combine_each(combineOp);

    return grid;
}


} // namespace point_mask_internal


////////////////////////////////////////


template<typename PointDataGridT, typename MaskT, typename FilterT>
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(
    const PointDataGridT& points,
    const FilterT& filter)
{
    return point_mask_internal::convertPointsToScalar<PointDataGridT, MaskT>(
        points, filter);
}


template<typename PointDataGridT, typename MaskT, typename FilterT>
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(
    const PointDataGridT& points,
    const openvdb::math::Transform& transform,
    const FilterT& filter)
{
    return point_mask_internal::convertPointsToScalar<PointDataGridT, MaskT>(
        points, transform, filter);
}


////////////////////////////////////////


// deprecated functions


template <typename PointDataGridT,
          typename MaskT = typename PointDataGridT::template ValueConverter<bool>::Type>
OPENVDB_DEPRECATED
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(const PointDataGridT& grid,
                    const std::vector<Name>& includeGroups,
                    const std::vector<Name>& excludeGroups)
{
    auto leaf = grid.tree().cbeginLeaf();
    if (!leaf)  return MaskT::create();
    MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
    return convertPointsToMask(grid, filter);
}


template <typename PointDataGridT,
          typename MaskT = typename PointDataGridT::template ValueConverter<bool>::Type>
OPENVDB_DEPRECATED
inline typename std::enable_if<std::is_same<typename MaskT::ValueType, bool>::value,
    typename MaskT::Ptr>::type
convertPointsToMask(const PointDataGridT& grid,
                    const openvdb::math::Transform& transform,
                    const std::vector<Name>& includeGroups,
                    const std::vector<Name>& excludeGroups)
{
    auto leaf = grid.tree().cbeginLeaf();
    if (!leaf)  return MaskT::create();
    MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());
    return convertPointsToMask(grid, transform, filter);
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_MASK_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
