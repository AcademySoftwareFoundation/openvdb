///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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
//
/// @file Clip.h
///
/// @brief Functions to clip a grid against a bounding box or against
/// another grid's active voxel topology

#ifndef OPENVDB_TOOLS_CLIP_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_CLIP_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>// for ValueMask
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>// for isNegative
#include <openvdb/tree/LeafManager.h>
#include "GridTransformer.h" // for resampleToMatch()
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/utility/enable_if.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include "Prune.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Clip the given grid against a world-space bounding box
/// and return a new grid containing the result.
/// @warning Clipping a level set will likely produce a grid that is
/// no longer a valid level set.
template<typename GridType> OPENVDB_STATIC_SPECIALIZATION
inline typename GridType::Ptr clip(const GridType& grid, const BBoxd&);

/// @brief Clip a grid against the active voxels of another grid
/// and return a new grid containing the result.
/// @param grid  the grid to be clipped
/// @param mask  a grid whose active voxels form a boolean clipping mask
/// @details The mask grid need not have the same transform as the source grid.
/// Also, if the mask grid is a level set, consider using tools::sdfInteriorMask
/// to construct a new mask comprising the interior (rather than the narrow band)
/// of the level set.
/// @warning Clipping a level set will likely produce a grid that is
/// no longer a valid level set.
template<typename GridType, typename MaskTreeType> OPENVDB_STATIC_SPECIALIZATION
inline typename GridType::Ptr clip(const GridType& grid, const Grid<MaskTreeType>& mask);


////////////////////////////////////////


namespace clip_internal {


////////////////////////////////////////


template<typename TreeT>
class MaskInteriorVoxels
{
public:
    typedef typename TreeT::ValueType ValueT;
    typedef typename TreeT::LeafNodeType LeafNodeT;

    MaskInteriorVoxels(const TreeT& tree): mAcc(tree) {}

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t /*leafIndex*/) const
    {
        const LeafNodeT *refLeaf = mAcc.probeConstLeaf(leaf.origin());
        if (refLeaf) {
            typename LeafNodeType::ValueOffIter iter = leaf.beginValueOff();
            for ( ; iter; ++iter) {
                const Index pos = iter.pos();
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
    typedef typename TreeT::template ValueConverter<ValueMask>::Type MaskTreeT;
    typedef tree::LeafManager<const MaskTreeT> MaskLeafManagerT;

    CopyLeafNodes(const TreeT& tree, const MaskLeafManagerT& leafNodes);

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
    typedef typename TreeT::LeafNodeType LeafT;
    typedef typename MaskTree::LeafNodeType MaskLeafT;
    typename MaskLeafT::ValueOnCIter it;

    tree::ValueAccessor<TreeT> acc(*mNewTree);
    tree::ValueAccessor<const TreeT> refAcc(*mTree);

    for (size_t n = range.begin(); n != range.end(); ++n) {
        const MaskLeafT& maskLeaf = mLeafNodes->leaf(n);
        const Coord& ijk = maskLeaf.origin();
        const LeafT* refLeaf = refAcc.probeConstLeaf(ijk);

        LeafT* newLeaf = acc.touchLeaf(ijk);

        if (refLeaf) {
            for (it = maskLeaf.cbeginValueOn(); it; ++it) {
                const Index pos = it.pos();
                newLeaf->setValueOnly(pos, refLeaf->getValue(pos));
                newLeaf->setActiveState(pos, refLeaf->isValueOn(pos));
            }
        } else {
            typename TreeT::ValueType value;
            bool isActive = refAcc.probeValue(ijk, value);

            for (it = maskLeaf.cbeginValueOn(); it; ++it) {
                const Index pos = it.pos();
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
    typedef typename FromGridT::Ptr FromGridPtrT;
    typedef typename ToGridT::Ptr ToGridPtrT;
    ToGridPtrT operator()(const FromGridPtrT& grid) { return ToGridPtrT(new ToGridT(*grid)); }
};

// Partial specialization that avoids copying when
// the input and output grid types are the same
template<typename GridT>
struct ConvertGrid<GridT, GridT>
{
    typedef typename GridT::Ptr GridPtrT;
    GridPtrT operator()(const GridPtrT& grid) { return grid; }
};


////////////////////////////////////////


// Convert a grid of arbitrary type to a mask grid and return a pointer to the new grid.
template<typename GridT>
inline typename boost::disable_if<boost::is_same<ValueMask, typename GridT::BuildType>,
    typename GridT::template ValueConverter<ValueMask>::Type::Ptr>::type
convertToMaskGrid(const GridT& grid)
{
    typedef typename GridT::template ValueConverter<ValueMask>::Type MaskGridT;
    typedef typename MaskGridT::Ptr MaskGridPtrT;

    // Convert the input grid to a boolean mask grid (with the same tree configuration).
    MaskGridPtrT mask = MaskGridT::create(/*background=*/false);
    mask->topologyUnion(grid);
    mask->setTransform(grid.constTransform().copy());
    return mask;
}

// Overload that avoids any processing if the input grid is already a mask grid
template<typename GridT>
inline typename boost::enable_if<boost::is_same<ValueMask, typename GridT::BuildType>,
                                 typename GridT::Ptr>::type
convertToMaskGrid(const GridT& grid)
{
    return grid.copy(); // shallow copy
}


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
doClip(const GridType& grid, const typename GridType::template ValueConverter<ValueMask>::Type& aMask)
{
    typedef typename GridType::TreeType TreeT;
    typedef typename GridType::TreeType::template ValueConverter<ValueMask>::Type MaskTreeT;

    const GridClass gridClass = grid.getGridClass();
    const TreeT& tree = grid.tree();

    MaskTreeT mask(false);
    mask.topologyUnion(tree);

    if (gridClass == GRID_LEVEL_SET) {
        tree::LeafManager<MaskTreeT> leafNodes(mask);
        leafNodes.foreach(MaskInteriorVoxels<TreeT>(tree));

        tree::ValueAccessor<const TreeT> acc(tree);

        typename MaskTreeT::ValueAllIter iter(mask);
        iter.setMaxDepth(MaskTreeT::ValueAllIter::LEAF_DEPTH - 1);

        for ( ; iter; ++iter) {
            iter.setActiveState(math::isNegative(acc.getValue(iter.getCoord())));
        }
    }

    mask.topologyIntersection(aMask.constTree());

    typename GridType::Ptr outGrid;
    {
        // Copy voxel values and states.
        tree::LeafManager<const MaskTreeT> leafNodes(mask);
        CopyLeafNodes<TreeT> maskOp(tree, leafNodes);
        maskOp.run();
        outGrid = GridType::create(maskOp.tree());
    }
    {
        // Copy tile values and states.
        tree::ValueAccessor<const TreeT> refAcc(tree);
        tree::ValueAccessor<const MaskTreeT> maskAcc(mask);

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


template<typename GridType>
OPENVDB_STATIC_SPECIALIZATION
inline typename GridType::Ptr
clip(const GridType& grid, const BBoxd& bbox)
{
    typedef typename GridType::template ValueConverter<ValueMask>::Type MaskGridT;

    // Transform the world-space bounding box into the source grid's index space.
    Vec3d idxMin, idxMax;
    math::calculateBounds(grid.constTransform(), bbox.min(), bbox.max(), idxMin, idxMax);
    CoordBBox region(Coord::floor(idxMin), Coord::floor(idxMax));
    // Construct a boolean mask grid that is true inside the index-space bounding box
    // and false everywhere else.
    MaskGridT clipMask(/*background=*/false);
    clipMask.fill(region, /*value=*/true, /*active=*/true);

    return clip_internal::doClip(grid, clipMask);
}


template<typename GridType1, typename TreeType2>
OPENVDB_STATIC_SPECIALIZATION
inline typename GridType1::Ptr
clip(const GridType1& grid1, const Grid<TreeType2>& grid2)
{
    typedef typename GridType1::template ValueConverter<ValueMask>::Type MaskGridT1;
    typedef typename MaskGridT1::Ptr MaskGridPtrT1;

    typedef Grid<TreeType2> GridType2;
    typedef typename GridType2::template ValueConverter<ValueMask>::Type MaskGridT2;
    typedef typename MaskGridT2::Ptr MaskGridPtrT2;

    // Convert the mask grid to a boolean grid with the same tree configuration.
    MaskGridPtrT2 maskGrid = clip_internal::convertToMaskGrid( grid2 );

    // Resample the boolean mask grid into the source grid's index space.
    if (grid1.constTransform() != maskGrid->constTransform()) {
        MaskGridPtrT2 resampledMask = MaskGridT2::create(/*background=*/false);
        resampledMask->setTransform(grid1.constTransform().copy());
        tools::resampleToMatch<clip_internal::BoolSampler>(*maskGrid, *resampledMask);
        tools::prune(resampledMask->tree());
        maskGrid = resampledMask;
    }

    // Convert the bool mask grid to a bool grid of the same configuration as the source grid.
    MaskGridPtrT1 clipMask =
        clip_internal::ConvertGrid</*from=*/MaskGridT2, /*to=*/MaskGridT1>()( maskGrid );

    // Clip the source grid against the boolean mask grid.
    return clip_internal::doClip(grid1, *clipMask);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_CLIP_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
