///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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

/// @file Clip.h
///
/// @brief Functions to clip a grid against a bounding box or against
/// another grid's active voxel topology

#ifndef OPENVDB_TOOLS_CLIP_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_CLIP_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/math/Math.h> // for math::isNegative()
#include <openvdb/tree/LeafManager.h>
#include "GridTransformer.h" // for tools::resampleToMatch()
#include "Prune.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <type_traits> // for std::enable_if, std::is_same


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Clip the given grid against a world-space bounding box
/// and return a new grid containing the result.
/// @param grid          the grid to be clipped
/// @param bbox          a world-space bounding box
/// @param keepInterior  if true, discard voxels that lie outside the bounding box;
///     if false, discard voxels that lie inside the bounding box
/// @warning Clipping a level set will likely produce a grid that is
/// no longer a valid level set.
template<typename GridType> OPENVDB_STATIC_SPECIALIZATION
inline typename GridType::Ptr
clip(const GridType& grid, const BBoxd& bbox, bool keepInterior = true);

/// @brief Clip a grid against the active voxels of another grid
/// and return a new grid containing the result.
/// @param grid          the grid to be clipped
/// @param mask          a grid whose active voxels form a boolean clipping mask
/// @param keepInterior  if true, discard voxels that do not intersect the mask;
///     if false, discard voxels that intersect the mask
/// @details The mask grid need not have the same transform as the source grid.
/// Also, if the mask grid is a level set, consider using tools::sdfInteriorMask
/// to construct a new mask comprising the interior (rather than the narrow band)
/// of the level set.
/// @warning Clipping a level set will likely produce a grid that is
/// no longer a valid level set.
template<typename GridType, typename MaskTreeType> OPENVDB_STATIC_SPECIALIZATION
inline typename GridType::Ptr
clip(const GridType& grid, const Grid<MaskTreeType>& mask, bool keepInterior = true);


////////////////////////////////////////


namespace clip_internal {

// Use either MaskGrids or BoolGrids internally.
// (MaskGrids have a somewhat lower memory footprint.)
using MaskValueType = ValueMask;
//using MaskValueType = bool;


template<typename TreeT>
class MaskInteriorVoxels
{
public:
    using ValueT = typename TreeT::ValueType;
    using LeafNodeT = typename TreeT::LeafNodeType;

    MaskInteriorVoxels(const TreeT& tree): mAcc(tree) {}

    template<typename LeafNodeType>
    void operator()(LeafNodeType& leaf, size_t /*leafIndex*/) const
    {
        const auto* refLeaf = mAcc.probeConstLeaf(leaf.origin());
        if (refLeaf) {
            for (auto iter = leaf.beginValueOff(); iter; ++iter) {
                const auto pos = iter.pos();
                leaf.setActiveState(pos, math::isNegative(refLeaf->getValue(pos)));
            }
        }
    }

private:
     tree::ValueAccessor<const TreeT> mAcc;
};


////////////////////////////////////////


template<typename TreeT>
class CopyLeafNodes
{
public:
    using MaskTreeT = typename TreeT::template ValueConverter<MaskValueType>::Type;
    using MaskLeafManagerT = tree::LeafManager<const MaskTreeT>;

    CopyLeafNodes(const TreeT&, const MaskLeafManagerT&);

    void run(bool threaded = true);

    typename TreeT::Ptr tree() const { return mNewTree; }

    CopyLeafNodes(CopyLeafNodes&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(const CopyLeafNodes& rhs) { mNewTree->merge(*rhs.mNewTree); }

private:
    const MaskTreeT* mClipMask;
    const TreeT* mTree;
    const MaskLeafManagerT* mLeafNodes;
    typename TreeT::Ptr mNewTree;
};


template<typename TreeT>
CopyLeafNodes<TreeT>::CopyLeafNodes(const TreeT& tree, const MaskLeafManagerT& leafNodes)
    : mTree(&tree)
    , mLeafNodes(&leafNodes)
    , mNewTree(new TreeT(mTree->background()))
{
}


template<typename TreeT>
CopyLeafNodes<TreeT>::CopyLeafNodes(CopyLeafNodes& rhs, tbb::split)
    : mTree(rhs.mTree)
    , mLeafNodes(rhs.mLeafNodes)
    , mNewTree(new TreeT(mTree->background()))
{
}


template<typename TreeT>
void
CopyLeafNodes<TreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mLeafNodes->getRange(), *this);
    else (*this)(mLeafNodes->getRange());
}


template<typename TreeT>
void
CopyLeafNodes<TreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    tree::ValueAccessor<TreeT> acc(*mNewTree);
    tree::ValueAccessor<const TreeT> refAcc(*mTree);

    for (auto n = range.begin(); n != range.end(); ++n) {
        const auto& maskLeaf = mLeafNodes->leaf(n);
        const auto& ijk = maskLeaf.origin();
        const auto* refLeaf = refAcc.probeConstLeaf(ijk);

        auto* newLeaf = acc.touchLeaf(ijk);

        if (refLeaf) {
            for (auto it = maskLeaf.cbeginValueOn(); it; ++it) {
                const auto pos = it.pos();
                newLeaf->setValueOnly(pos, refLeaf->getValue(pos));
                newLeaf->setActiveState(pos, refLeaf->isValueOn(pos));
            }
        } else {
            typename TreeT::ValueType value;
            bool isActive = refAcc.probeValue(ijk, value);

            for (auto it = maskLeaf.cbeginValueOn(); it; ++it) {
                const auto pos = it.pos();
                newLeaf->setValueOnly(pos, value);
                newLeaf->setActiveState(pos, isActive);
            }
        }
    }
}


////////////////////////////////////////


struct BoolSampler
{
    static const char* name() { return "bin"; }
    static int radius() { return 2; }
    static bool mipmap() { return false; }
    static bool consistent() { return true; }

    template<class TreeT>
    static bool sample(const TreeT& inTree,
        const Vec3R& inCoord, typename TreeT::ValueType& result)
    {
        Coord ijk;
        ijk[0] = int(std::floor(inCoord[0]));
        ijk[1] = int(std::floor(inCoord[1]));
        ijk[2] = int(std::floor(inCoord[2]));
        return inTree.probeValue(ijk, result);
    }
};


////////////////////////////////////////


// Convert a grid of one type to a grid of another type
template<typename FromGridT, typename ToGridT>
struct ConvertGrid
{
    using FromGridCPtrT = typename FromGridT::ConstPtr;
    using ToGridPtrT = typename ToGridT::Ptr;
    ToGridPtrT operator()(const FromGridCPtrT& grid) { return ToGridPtrT(new ToGridT(*grid)); }
};

// Partial specialization that avoids copying when
// the input and output grid types are the same
template<typename GridT>
struct ConvertGrid<GridT, GridT>
{
    using GridCPtrT = typename GridT::ConstPtr;
    GridCPtrT operator()(const GridCPtrT& grid) { return grid; }
};


////////////////////////////////////////


// Convert a grid of arbitrary type to a mask grid with the same tree configuration
// and return a pointer to the new grid.
/// @private
template<typename GridT>
inline typename std::enable_if<!std::is_same<MaskValueType, typename GridT::BuildType>::value,
    typename GridT::template ValueConverter<MaskValueType>::Type::Ptr>::type
convertToMaskGrid(const GridT& grid)
{
    using MaskGridT = typename GridT::template ValueConverter<MaskValueType>::Type;
    auto mask = MaskGridT::create(/*background=*/false);
    mask->topologyUnion(grid);
    mask->setTransform(grid.constTransform().copy());
    return mask;
}

// Overload that avoids any processing if the input grid is already a mask grid
/// @private
template<typename GridT>
inline typename std::enable_if<std::is_same<MaskValueType, typename GridT::BuildType>::value,
    typename GridT::ConstPtr>::type
convertToMaskGrid(const GridT& grid)
{
    return grid.copy(); // shallow copy
}


////////////////////////////////////////


/// @private
template<typename GridType>
inline typename GridType::Ptr
doClip(
    const GridType& grid,
    const typename GridType::template ValueConverter<MaskValueType>::Type& clipMask,
    bool keepInterior)
{
    using TreeT = typename GridType::TreeType;
    using MaskTreeT = typename GridType::TreeType::template ValueConverter<MaskValueType>::Type;

    const auto gridClass = grid.getGridClass();
    const auto& tree = grid.tree();

    MaskTreeT gridMask(false);
    gridMask.topologyUnion(tree);

    if (gridClass == GRID_LEVEL_SET) {
        tree::LeafManager<MaskTreeT> leafNodes(gridMask);
        leafNodes.foreach(MaskInteriorVoxels<TreeT>(tree));

        tree::ValueAccessor<const TreeT> acc(tree);

        typename MaskTreeT::ValueAllIter iter(gridMask);
        iter.setMaxDepth(MaskTreeT::ValueAllIter::LEAF_DEPTH - 1);

        for ( ; iter; ++iter) {
            iter.setActiveState(math::isNegative(acc.getValue(iter.getCoord())));
        }
    }

    if (keepInterior) {
        gridMask.topologyIntersection(clipMask.constTree());
    } else {
        gridMask.topologyDifference(clipMask.constTree());
    }

    typename GridType::Ptr outGrid;
    {
        // Copy voxel values and states.
        tree::LeafManager<const MaskTreeT> leafNodes(gridMask);
        CopyLeafNodes<TreeT> maskOp(tree, leafNodes);
        maskOp.run();
        outGrid = GridType::create(maskOp.tree());
    }
    {
        // Copy tile values and states.
        tree::ValueAccessor<const TreeT> refAcc(tree);
        tree::ValueAccessor<const MaskTreeT> maskAcc(gridMask);

        typename TreeT::ValueAllIter it(outGrid->tree());
        it.setMaxDepth(TreeT::ValueAllIter::LEAF_DEPTH - 1);
        for ( ; it; ++it) {
            Coord ijk = it.getCoord();

            if (maskAcc.isValueOn(ijk)) {
                typename TreeT::ValueType value;
                bool isActive = refAcc.probeValue(ijk, value);

                it.setValue(value);
                if (!isActive) it.setValueOff();
            }
        }
    }

    outGrid->setTransform(grid.transform().copy());
    if (gridClass != GRID_LEVEL_SET) outGrid->setGridClass(gridClass);

    return outGrid;
}

} // namespace clip_internal


////////////////////////////////////////


/// @private
template<typename GridType>
OPENVDB_STATIC_SPECIALIZATION
inline typename GridType::Ptr
clip(const GridType& grid, const BBoxd& bbox, bool keepInterior)
{
    using MaskValueT = clip_internal::MaskValueType;
    using MaskGridT = typename GridType::template ValueConverter<MaskValueT>::Type;

    // Transform the world-space bounding box into the source grid's index space.
    Vec3d idxMin, idxMax;
    math::calculateBounds(grid.constTransform(), bbox.min(), bbox.max(), idxMin, idxMax);
    CoordBBox region(Coord::floor(idxMin), Coord::floor(idxMax));
    // Construct a boolean mask grid that is true inside the index-space bounding box
    // and false everywhere else.
    MaskGridT clipMask(/*background=*/false);
    clipMask.fill(region, /*value=*/true, /*active=*/true);

    return clip_internal::doClip(grid, clipMask, keepInterior);
}


/// @private
template<typename SrcGridType, typename ClipTreeType>
OPENVDB_STATIC_SPECIALIZATION
inline typename SrcGridType::Ptr
clip(const SrcGridType& srcGrid, const Grid<ClipTreeType>& clipGrid, bool keepInterior)
{
    using MaskValueT = clip_internal::MaskValueType;
    using ClipGridType = Grid<ClipTreeType>;
    using SrcMaskGridType = typename SrcGridType::template ValueConverter<MaskValueT>::Type;
    using ClipMaskGridType = typename ClipGridType::template ValueConverter<MaskValueT>::Type;

    // Convert the clipping grid to a boolean-valued mask grid with the same tree configuration.
    auto maskGrid = clip_internal::convertToMaskGrid(clipGrid);

    // Resample the mask grid into the source grid's index space.
    if (srcGrid.constTransform() != maskGrid->constTransform()) {
        auto resampledMask = ClipMaskGridType::create(/*background=*/false);
        resampledMask->setTransform(srcGrid.constTransform().copy());
        tools::resampleToMatch<clip_internal::BoolSampler>(*maskGrid, *resampledMask);
        tools::prune(resampledMask->tree());
        maskGrid = resampledMask;
    }

    // Convert the mask grid to a mask grid with the same tree configuration as the source grid.
    auto clipMask = clip_internal::ConvertGrid<
        /*from=*/ClipMaskGridType, /*to=*/SrcMaskGridType>()(maskGrid);

    // Clip the source grid against the mask grid.
    return clip_internal::doClip(srcGrid, *clipMask, keepInterior);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_CLIP_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
