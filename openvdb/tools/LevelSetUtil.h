///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file tools/LevelSetUtil.h
///
/// @brief Miscellaneous utilities that operate primarily or exclusively
/// on level set grids

#ifndef OPENVDB_TOOLS_LEVELSETUTIL_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETUTIL_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/Prune.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <limits>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

// MS Visual C++ requires this extra level of indirection in order to compile
// THIS MUST EXIST IN AN UNNAMED NAMESPACE IN ORDER TO COMPILE ON WINDOWS
namespace {

template<typename GridType>
inline typename GridType::ValueType lsutilGridMax()
{
    return std::numeric_limits<typename GridType::ValueType>::max();
}

template<typename GridType>
inline typename GridType::ValueType lsutilGridZero()
{
    return zeroVal<typename GridType::ValueType>();
}

} // unnamed namespace


////////////////////////////////////////


/// @brief Threaded method to convert a sparse level set/SDF into a sparse fog volume
///
/// @details For a level set, the active and negative-valued interior half of the
/// narrow band becomes a linear ramp from 0 to 1; the inactive interior becomes
/// active with a constant value of 1; and the exterior, including the background
/// and the active exterior half of the narrow band, becomes inactive with a constant
/// value of 0.  The interior, though active, remains sparse.
/// @details For a generic SDF, a specified cutoff distance determines the width
/// of the ramp, but otherwise the result is the same as for a level set.
///
/// @param grid            level set/SDF grid to transform
/// @param cutoffDistance  optional world space cutoff distance for the ramp
///                        (automatically clamped if greater than the interior
///                        narrow band width)
template<class GridType>
inline void
sdfToFogVolume(
    GridType& grid,
    typename GridType::ValueType cutoffDistance = lsutilGridMax<GridType>());


////////////////////////////////////////


/// @brief Threaded method to extract an interior region mask from a level set/SDF grid
///
/// @return a shared pointer to a new boolean grid with the same tree configuration and
///         transform as the incoming @c grid and whose active voxels correspond to
///         the interior of the input SDF
///
/// @param grid  a level set/SDF grid
/// @param iso   threshold below which values are considered to be part of the interior region
///
template<class GridType>
inline typename Grid<typename GridType::TreeType::template ValueConverter<bool>::Type>::Ptr
sdfInteriorMask(
    const GridType& grid,
    typename GridType::ValueType iso = lsutilGridZero<GridType>());


////////////////////////////////////////


/// @brief Threaded operator that finds the minimum and maximum values
/// among the active leaf-level voxels of a grid
/// @details This is useful primarily for level set grids, which have
/// no active tiles (all of their active voxels are leaf-level).
template<class TreeType>
class MinMaxVoxel
{
public:
    typedef tree::LeafManager<TreeType> LeafArray;
    typedef typename TreeType::ValueType ValueType;

    // LeafArray = openvdb::tree::LeafManager<TreeType> leafs(myTree)
    MinMaxVoxel(LeafArray&);

    void runParallel();
    void runSerial();

    const ValueType& minVoxel() const { return mMin; }
    const ValueType& maxVoxel() const { return mMax; }

    inline MinMaxVoxel(const MinMaxVoxel<TreeType>&, tbb::split);
    inline void operator()(const tbb::blocked_range<size_t>&);
    inline void join(const MinMaxVoxel<TreeType>&);

private:
    LeafArray& mLeafArray;
    ValueType mMin, mMax;
};


////////////////////////////////////////


// Internal utility objects and implementation details
namespace internal {

template<typename ValueType>
class FogVolumeOp
{
public:
    FogVolumeOp(ValueType cutoffDistance)
        : mWeight(ValueType(1.0) / cutoffDistance)
    {
    }

    // cutoff has to be < 0.0
    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t/*leafIndex*/) const
    {
        const ValueType zero = zeroVal<ValueType>();

        for (typename LeafNodeType::ValueAllIter iter = leaf.beginValueAll(); iter; ++iter) {
            ValueType& value = const_cast<ValueType&>(iter.getValue());
            if (value > zero) {
                value = zero;
                iter.setValueOff();
            } else {
                value = std::min(ValueType(1.0), value * mWeight);
                iter.setValueOn(value > zero);
            }
        }
    }

private:
    ValueType mWeight;
}; // class FogVolumeOp


template<typename TreeType>
class InteriorMaskOp
{
public:
    InteriorMaskOp(const TreeType& tree, typename TreeType::ValueType iso)
        : mTree(tree)
        , mIso(iso)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t/*leafIndex*/) const
    {
        const Coord origin = leaf.origin();
        const typename TreeType::LeafNodeType* refLeafPt = mTree.probeConstLeaf(origin);

        if (refLeafPt != NULL) {

            const typename TreeType::LeafNodeType& refLeaf = *refLeafPt;
            typename LeafNodeType::ValueAllIter iter = leaf.beginValueAll();

            for (; iter; ++iter) {
                if (refLeaf.getValue(iter.pos()) < mIso) {
                    iter.setValueOn();
                } else {
                    iter.setValueOff();
                }
            }
        }
    }

private:
    const TreeType& mTree;
    typename TreeType::ValueType mIso;
}; // class InteriorMaskOp

} // namespace internal


////////////////////////////////////////


template <class TreeType>
MinMaxVoxel<TreeType>::MinMaxVoxel(LeafArray& leafs)
    : mLeafArray(leafs)
    , mMin(std::numeric_limits<ValueType>::max())
    , mMax(-mMin)
{
}


template <class TreeType>
inline
MinMaxVoxel<TreeType>::MinMaxVoxel(const MinMaxVoxel<TreeType>& rhs, tbb::split)
    : mLeafArray(rhs.mLeafArray)
    , mMin(rhs.mMin)
    , mMax(rhs.mMax)
{
}


template <class TreeType>
void
MinMaxVoxel<TreeType>::runParallel()
{
    tbb::parallel_reduce(mLeafArray.getRange(), *this);
}


template <class TreeType>
void
MinMaxVoxel<TreeType>::runSerial()
{
    (*this)(mLeafArray.getRange());
}


template <class TreeType>
inline void
MinMaxVoxel<TreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    typename TreeType::LeafNodeType::ValueOnCIter iter;

    for (size_t n = range.begin(); n < range.end(); ++n) {
        iter = mLeafArray.leaf(n).cbeginValueOn();
        for (; iter; ++iter) {
            const ValueType value = iter.getValue();
            mMin = std::min(mMin, value);
            mMax = std::max(mMax, value);
        }
    }
}


template <class TreeType>
inline void
MinMaxVoxel<TreeType>::join(const MinMaxVoxel<TreeType>& rhs)
{
    mMin = std::min(mMin, rhs.mMin);
    mMax = std::max(mMax, rhs.mMax);
}



////////////////////////////////////////


template <class GridType>
inline void
sdfToFogVolume(GridType& grid, typename GridType::ValueType cutoffDistance)
{
    typedef typename GridType::TreeType TreeType;
    typedef typename GridType::ValueType ValueType;

    cutoffDistance = -std::abs(cutoffDistance);

    TreeType& tree = const_cast<TreeType&>(grid.tree());

    { // Transform all voxels (parallel, over leaf nodes)
        tree::LeafManager<TreeType> leafs(tree);

        MinMaxVoxel<TreeType> minmax(leafs);
        minmax.runParallel();

        // Clamp to the interior band width.
        if (minmax.minVoxel() > cutoffDistance) {
            cutoffDistance = minmax.minVoxel();
        }

        leafs.foreach(internal::FogVolumeOp<ValueType>(cutoffDistance));
    }

    // Transform all tile values (serial, but the iteration
    // is constrained from descending into leaf nodes)
    const ValueType zero = zeroVal<ValueType>();
    typename TreeType::ValueAllIter iter(tree);
    iter.setMaxDepth(TreeType::ValueAllIter::LEAF_DEPTH - 1);

    for ( ; iter; ++iter) {
        ValueType& value = const_cast<ValueType&>(iter.getValue());

        if (value > zero) {
            value = zero;
            iter.setValueOff();
        } else {
            value = ValueType(1.0);
            iter.setActiveState(true);
        }
    }

    // Update the tree background value.

    typename TreeType::Ptr newTree(new TreeType(/*background=*/zero));
    newTree->merge(tree);
    // This is faster than calling Tree::setBackground, since we only need
    // to update the value that is returned for coordinates that don't fall
    // inside an allocated node. All inactive tiles and voxels have already
    // been updated in the previous step so the Tree::setBackground method
    // will in this case do a redundant traversal of the tree to update the
    // inactive values once more.

    //newTree->pruneInactive();
    grid.setTree(newTree);

    grid.setGridClass(GRID_FOG_VOLUME);
}


////////////////////////////////////////


template <class GridType>
inline typename Grid<typename GridType::TreeType::template ValueConverter<bool>::Type>::Ptr
sdfInteriorMask(const GridType& grid, typename GridType::ValueType iso)
{
    typedef typename GridType::TreeType::template ValueConverter<bool>::Type BoolTreeType;
    typedef Grid<BoolTreeType> BoolGridType;

    typename BoolGridType::Ptr maskGrid(BoolGridType::create(false));
    maskGrid->setTransform(grid.transform().copy());
    BoolTreeType& maskTree = maskGrid->tree();

    maskTree.topologyUnion(grid.tree());

    { // Evaluate voxels (parallel, over leaf nodes)

        tree::LeafManager<BoolTreeType> leafs(maskTree);

        leafs.foreach(internal::InteriorMaskOp<typename GridType::TreeType>(grid.tree(), iso));
    }

    // Evaluate tile values (serial, but the iteration
    // is constrained from descending into leaf nodes)

    tree::ValueAccessor<const typename GridType::TreeType> acc(grid.tree());
    typename BoolTreeType::ValueAllIter iter(maskTree);
    iter.setMaxDepth(BoolTreeType::ValueAllIter::LEAF_DEPTH - 1);

    for ( ; iter; ++iter) {
        iter.setActiveState(acc.getValue(iter.getCoord()) < iso);
    }

    tools::pruneInactive(maskTree);

    return maskGrid;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETUTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
