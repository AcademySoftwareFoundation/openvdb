///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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
/// @brief  Miscellaneous utilities that operate primarily or exclusively
///         on level set grids
///
/// @author Mihai Alden


#ifndef OPENVDB_TOOLS_LEVEL_SET_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_UTIL_HAS_BEEN_INCLUDED

#include "MeshToVolume.h" // for traceExteriorBoundaries
#include "SignedFloodFill.h" // for signedFloodFill

#include <openvdb/Types.h>
#include <openvdb/Grid.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
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


/// @brief Threaded method to construct a boolean mask that represents interior regions
///        in a signed distance field.
///
/// @return A shared pointer to either a boolean grid or tree with the same tree
///         configuration and potentially transform as the input @c volume and whose active
///         and @c true values correspond to the interior of the input signed distance field.
///
/// @param volume               Signed distance field / level set volume.
/// @param isovalue             Threshold below which values are considered part of the
///                             interior region.
template<class GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
sdfInteriorMask(
    const GridOrTreeType& volume,
    typename GridOrTreeType::ValueType isovalue = lsutilGridZero<GridOrTreeType>());


/// @brief  Extracts the interior regions of a signed distance field and topologically enclosed
///         (watertight) regions of value greater than the @a isovalue (cavities) that can arise
///         as the result of CSG union operations between different shapes where at least one of
///         the shapes has a concavity that is capped.
///
///         For example the enclosed region of a capped bottle would include the walls and
///         the interior cavity.
///
/// @return A shared pointer to either a boolean grid or tree with the same tree configuration
///         and potentially transform as the input @c volume and whose active and @c true values
///         correspond to the interior and enclosed regions in the input signed distance field.
///
/// @param volume       Signed distance field / level set volume.
/// @param isovalue     Threshold below which values are considered part of the interior region.
/// @param fillMask     Optional boolean tree, when provided enclosed cavity regions that are not
///                     completely filled by this mask are ignored.
///
///                     For instance if the fill mask does not completely fill the bottle in the
///                     previous example only the walls and cap are returned and the interior
///                     cavity will be ignored.
template<typename GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
extractEnclosedRegion(const GridOrTreeType& volume,
    typename GridOrTreeType::ValueType isovalue = lsutilGridZero<GridOrTreeType>(),
    const typename TreeAdapter<GridOrTreeType>::TreeType::template ValueConverter<bool>::Type* fillMask = NULL);




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Internal utility objects and implementation details


namespace level_set_util_internal {


template<typename LeafNodeType>
struct MaskInteriorVoxels {

    typedef typename LeafNodeType::ValueType                ValueType;
    typedef tree::LeafNode<bool, LeafNodeType::LOG2DIM>     BoolLeafNodeType;

    MaskInteriorVoxels(
        ValueType isovalue, const LeafNodeType ** nodes, BoolLeafNodeType ** maskNodes)
        : mNodes(nodes), mMaskNodes(maskNodes), mIsovalue(isovalue)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        BoolLeafNodeType * maskNodePt = NULL;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            mMaskNodes[n] = NULL;
            const LeafNodeType& node = *mNodes[n];

            if (!maskNodePt) {
                maskNodePt = new BoolLeafNodeType(node.origin(), false);
            } else {
                maskNodePt->setOrigin(node.origin());
            }

            const ValueType* values = &node.getValue(0);
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                if (values[i] < mIsovalue) maskNodePt->setValueOn(i, true);
            }

            if (maskNodePt->onVoxelCount() > 0) {
                mMaskNodes[n] = maskNodePt;
                maskNodePt = NULL;
            }
        }

        if (maskNodePt) delete maskNodePt;
    }

    LeafNodeType        const * const * const mNodes;
    BoolLeafNodeType                 ** const mMaskNodes;
    ValueType                           const mIsovalue;
}; // MaskInteriorVoxels


template<typename TreeType, typename InternalNodeType>
struct MaskInteriorTiles {

    typedef typename TreeType::ValueType    ValueType;

    MaskInteriorTiles(ValueType isovalue, const TreeType& tree, InternalNodeType ** maskNodes)
        : mTree(&tree), mMaskNodes(maskNodes), mIsovalue(isovalue) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        tree::ValueAccessor<const TreeType> acc(*mTree);
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            typename InternalNodeType::ValueAllIter it = mMaskNodes[n]->beginValueAll();
            for (; it; ++it) {
                if (acc.getValue(it.getCoord()) < mIsovalue) {
                    it.setValue(true);
                    it.setValueOn(true);
                }
            }
        }
    }

    TreeType            const * const mTree;
    InternalNodeType         ** const mMaskNodes;
    ValueType                   const mIsovalue;
}; // MaskInteriorTiles


template<typename TreeType>
struct PopulateTree {

    typedef typename TreeType::ValueType    ValueType;
    typedef typename TreeType::LeafNodeType LeafNodeType;

    PopulateTree(TreeType& tree, LeafNodeType** leafnodes,
        const size_t * nodexIndexMap, ValueType background)
        : mNewTree(background)
        , mTreePt(&tree)
        , mNodes(leafnodes)
        , mNodeIndexMap(nodexIndexMap)
    {
    }

    PopulateTree(PopulateTree& rhs, tbb::split)
        : mNewTree(rhs.mNewTree.background())
        , mTreePt(&mNewTree)
        , mNodes(rhs.mNodes)
        , mNodeIndexMap(rhs.mNodeIndexMap)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {

        tree::ValueAccessor<TreeType> acc(*mTreePt);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            for (size_t i = mNodeIndexMap[n], I = mNodeIndexMap[n + 1]; i < I; ++i) {
                if (mNodes[i] != NULL) acc.addLeaf(mNodes[i]);
            }
        }
    }

    void join(PopulateTree& rhs) { mTreePt->merge(*rhs.mTreePt); }

private:
    TreeType                      mNewTree;
    TreeType              * const mTreePt;
    LeafNodeType         ** const mNodes;
    size_t          const * const mNodeIndexMap;
}; // PopulateTree


/// @brief Negative active values are set @c 0, everything else is set to @c 1.
template<typename LeafNodeType>
struct LabelBoundaryVoxels {

    typedef typename LeafNodeType::ValueType                ValueType;
    typedef tree::LeafNode<char, LeafNodeType::LOG2DIM>     CharLeafNodeType;

    LabelBoundaryVoxels(
        ValueType isovalue, const LeafNodeType ** nodes, CharLeafNodeType ** maskNodes)
        : mNodes(nodes), mMaskNodes(maskNodes), mIsovalue(isovalue)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        CharLeafNodeType * maskNodePt = NULL;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            mMaskNodes[n] = NULL;
            const LeafNodeType& node = *mNodes[n];

            if (!maskNodePt) {
                maskNodePt = new CharLeafNodeType(node.origin(), 1);
            } else {
                maskNodePt->setOrigin(node.origin());
            }

            typename LeafNodeType::ValueOnCIter it;
            for (it = node.cbeginValueOn(); it; ++it) {
                maskNodePt->setValueOn(it.pos(), ((*it - mIsovalue) < 0.0) ? 0 : 1);
            }

            if (maskNodePt->onVoxelCount() > 0) {
                mMaskNodes[n] = maskNodePt;
                maskNodePt = NULL;
            }
        }

        if (maskNodePt) delete maskNodePt;
    }

    LeafNodeType        const * const * const mNodes;
    CharLeafNodeType                 ** const mMaskNodes;
    ValueType                           const mIsovalue;
}; // LabelBoundaryVoxels


template<typename LeafNodeType>
struct FlipRegionSign {
    typedef typename LeafNodeType::ValueType ValueType;

    FlipRegionSign(LeafNodeType ** nodes) : mNodes(nodes) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            ValueType* values = const_cast<ValueType*>(&mNodes[n]->getValue(0));
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                values[i] = values[i] < 0 ? 1 : -1;
            }
        }
    }

    LeafNodeType ** const mNodes;
}; // FlipRegionSign


template<typename LeafNodeType>
struct FindMinVoxelValue {

    typedef typename LeafNodeType::ValueType    ValueType;

    FindMinVoxelValue(LeafNodeType const * const * const leafnodes)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(leafnodes)
    {
    }

    FindMinVoxelValue(FindMinVoxelValue& rhs, tbb::split)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(rhs.mNodes)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            const ValueType* data = mNodes[n]->buffer().data();
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                minValue = std::min(minValue, data[i]);
            }
        }
    }

    void join(FindMinVoxelValue& rhs) { minValue = std::min(minValue, rhs.minValue); }

    ValueType minValue;

    LeafNodeType const * const * const mNodes;
}; // FindMinVoxelValue


template<typename InternalNodeType>
struct FindMinTileValue {

    typedef typename InternalNodeType::ValueType    ValueType;

    FindMinTileValue(InternalNodeType const * const * const nodes)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(nodes)
    {
    }

    FindMinTileValue(FindMinTileValue& rhs, tbb::split)
        : minValue(std::numeric_limits<ValueType>::max())
        , mNodes(rhs.mNodes)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            typename InternalNodeType::ValueAllCIter it = mNodes[n]->beginValueAll();
            for (; it; ++it) {
                minValue = std::min(minValue, *it);
            }
        }
    }

    void join(FindMinTileValue& rhs) { minValue = std::min(minValue, rhs.minValue); }

    ValueType minValue;

    InternalNodeType const * const * const mNodes;
}; // FindMinTileValue


template<typename LeafNodeType>
struct SDFVoxelsToFogVolume {

    typedef typename LeafNodeType::ValueType ValueType;

    SDFVoxelsToFogVolume(LeafNodeType ** nodes, ValueType cutoffDistance)
        : mNodes(nodes), mWeight(ValueType(1.0) / cutoffDistance)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& node = *mNodes[n];
            node.setValuesOff();

            ValueType* values = node.buffer().data();
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                values[i] = values[i] > ValueType(0.0) ? ValueType(0.0) : values[i] * mWeight;
                if (values[i] > ValueType(0.0)) node.setValueOn(i);
            }

            if (node.onVoxelCount() == 0) {
                delete mNodes[n];
                mNodes[n] = NULL;
            }
        }
    }

    LeafNodeType    ** const mNodes;
    ValueType          const mWeight;
}; // SDFVoxelsToFogVolume


template<typename TreeType, typename InternalNodeType>
struct SDFTilesToFogVolume {

    SDFTilesToFogVolume(const TreeType& tree, InternalNodeType ** nodes)
        : mTree(&tree), mNodes(nodes) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        typedef typename TreeType::ValueType ValueType;
        tree::ValueAccessor<const TreeType> acc(*mTree);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            typename InternalNodeType::ValueAllIter it = mNodes[n]->beginValueAll();
            for (; it; ++it) {
                if (acc.getValue(it.getCoord()) < ValueType(0.0)) {
                    it.setValue(ValueType(1.0));
                    it.setValueOn(true);
                }
            }
        }
    }

    TreeType            const * const mTree;
    InternalNodeType         ** const mNodes;
}; // SDFTilesToFogVolume


template<typename TreeType>
struct FillMaskBoundary {

    typedef typename TreeType::ValueType                            ValueType;
    typedef typename TreeType::LeafNodeType                         LeafNodeType;
    typedef typename TreeType::template ValueConverter<bool>::Type  BoolTreeType;
    typedef typename BoolTreeType::LeafNodeType                     BoolLeafNodeType;

    FillMaskBoundary(const TreeType& tree, ValueType isovalue, const BoolTreeType& fillMask,
        const BoolLeafNodeType ** fillNodes, BoolLeafNodeType ** newNodes)
        : mTree(&tree), mFillMask(&fillMask), mFillNodes(fillNodes), mNewNodes(newNodes), mIsovalue(isovalue)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        tree::ValueAccessor<const BoolTreeType> maskAcc(*mFillMask);
        tree::ValueAccessor<const TreeType> distAcc(*mTree);

        boost::scoped_array<char> valueMask(new char[BoolLeafNodeType::SIZE]);

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            mNewNodes[n] = NULL;
            const BoolLeafNodeType& node = *mFillNodes[n];
            const Coord& origin = node.origin();

            const bool denseNode = node.isDense();

            // possible early out if the fill mask is dense
            if (denseNode) {

                int denseNeighbors = 0;

                const BoolLeafNodeType* neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(-1, 0, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(BoolLeafNodeType::DIM, 0, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, -1, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, BoolLeafNodeType::DIM, 0));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, 0, -1));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                neighborNode = maskAcc.probeConstLeaf(origin.offsetBy(0, 0, BoolLeafNodeType::DIM));
                if (neighborNode && neighborNode->isDense()) ++denseNeighbors;

                if (denseNeighbors == 6) continue;
            }

            // rest value mask
            memset(valueMask.get(), 0, sizeof(char) * BoolLeafNodeType::SIZE);

            const typename TreeType::LeafNodeType* distNode = distAcc.probeConstLeaf(origin);

            // check internal voxel neighbors

            bool earlyTermination = false;

            if (!denseNode) {
                if (distNode) {
                    evalInternalNeighborsP(valueMask.get(), node, *distNode);
                    evalInternalNeighborsN(valueMask.get(), node, *distNode);
                } else if (distAcc.getValue(origin) > mIsovalue) {
                    earlyTermination = evalInternalNeighborsP(valueMask.get(), node);
                    if (!earlyTermination) earlyTermination = evalInternalNeighborsN(valueMask.get(), node);
                }
            }

            // check external voxel neighbors

            if (!earlyTermination) {
                evalExternalNeighborsX<true>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsX<false>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsY<true>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsY<false>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsZ<true>(valueMask.get(), node, maskAcc, distAcc);
                evalExternalNeighborsZ<false>(valueMask.get(), node, maskAcc, distAcc);
            }

            // Export marked boundary voxels.

            int numBoundaryValues = 0;
            for (Index i = 0, I = BoolLeafNodeType::SIZE; i < I; ++i) {
                numBoundaryValues += valueMask[i] == 1;
            }

            if (numBoundaryValues > 0) {
                mNewNodes[n] = new BoolLeafNodeType(origin, false);
                for (Index i = 0, I = BoolLeafNodeType::SIZE; i < I; ++i) {
                    if (valueMask[i] == 1) mNewNodes[n]->setValueOn(i);
                }
            }
        }
    }

private:

    // Check internal voxel neighbors in positive {x, y, z} directions.

    void evalInternalNeighborsP(char* valueMask, const BoolLeafNodeType& node, const LeafNodeType& distNode) const {

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM - 1; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos + 1) && distNode.getValue(pos + 1)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM - 1; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos + BoolLeafNodeType::DIM) &&
                        distNode.getValue(pos + BoolLeafNodeType::DIM)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM - 1; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos + BoolLeafNodeType::DIM * BoolLeafNodeType::DIM) &&
                        distNode.getValue(pos + BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    bool evalInternalNeighborsP(char* valueMask, const BoolLeafNodeType& node) const {

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM - 1; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos + 1)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM - 1; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos + BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM - 1; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) &&
                        !node.isValueOn(pos + BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // Check internal voxel neighbors in negative {x, y, z} directions.

    void evalInternalNeighborsN(char* valueMask, const BoolLeafNodeType& node, const LeafNodeType& distNode) const {

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 1; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos - 1) && distNode.getValue(pos - 1)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 1; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos - BoolLeafNodeType::DIM) &&
                        distNode.getValue(pos - BoolLeafNodeType::DIM)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }

        for (Index x = 1; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (valueMask[pos] != 0 || !node.isValueOn(pos)) continue;

                    if (!node.isValueOn(pos - BoolLeafNodeType::DIM * BoolLeafNodeType::DIM) &&
                        distNode.getValue(pos - BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)  > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }


    bool evalInternalNeighborsN(char* valueMask, const BoolLeafNodeType& node) const {

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 1; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos - 1)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 0; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 1; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) && !node.isValueOn(pos - BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        for (Index x = 1; x < BoolLeafNodeType::DIM; ++x) {
            const Index xPos = x << (2 * BoolLeafNodeType::LOG2DIM);
            for (Index y = 0; y < BoolLeafNodeType::DIM; ++y) {
                const Index yPos = xPos + (y << BoolLeafNodeType::LOG2DIM);
                for (Index z = 0; z < BoolLeafNodeType::DIM; ++z) {
                    const Index pos = yPos + z;

                    if (node.isValueOn(pos) &&
                        !node.isValueOn(pos - BoolLeafNodeType::DIM * BoolLeafNodeType::DIM)) {
                        valueMask[pos] = 1;
                        return true;
                    }
                }
            }
        }

        return false;
    }


    // Check external voxel neighbors

    // If UpWind is true check the X+ oriented node face, else the X- oriented face.
    template<bool UpWind>
    void evalExternalNeighborsX(char* valueMask, const BoolLeafNodeType& node,
        const tree::ValueAccessor<const BoolTreeType>& maskAcc,
        const tree::ValueAccessor<const TreeType>& distAcc) const {

        const Coord& origin = node.origin();
        Coord ijk(0, 0, 0), nijk;
        int step = -1;

        if (UpWind) {
            step = 1;
            ijk[0] = int(BoolLeafNodeType::DIM) - 1;
        }

        const Index xPos = ijk[0] << (2 * int(BoolLeafNodeType::LOG2DIM));

        for (ijk[1] = 0; ijk[1] < int(BoolLeafNodeType::DIM); ++ijk[1]) {
            const Index yPos = xPos + (ijk[1] << int(BoolLeafNodeType::LOG2DIM));

            for (ijk[2] = 0; ijk[2] < int(BoolLeafNodeType::DIM); ++ijk[2]) {
                const Index pos = yPos + ijk[2];

                if (valueMask[pos] == 0 && node.isValueOn(pos)) {

                    nijk = origin + ijk.offsetBy(step, 0, 0);

                    if (!maskAcc.isValueOn(nijk) && distAcc.getValue(nijk) > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    // If UpWind is true check the Y+ oriented node face, else the Y- oriented face.
    template<bool UpWind>
    void evalExternalNeighborsY(char* valueMask, const BoolLeafNodeType& node,
        const tree::ValueAccessor<const BoolTreeType>& maskAcc,
        const tree::ValueAccessor<const TreeType>& distAcc) const {

        const Coord& origin = node.origin();
        Coord ijk(0, 0, 0), nijk;
        int step = -1;

        if (UpWind) {
            step = 1;
            ijk[1] = int(BoolLeafNodeType::DIM) - 1;
        }

        const Index yPos = ijk[1] << int(BoolLeafNodeType::LOG2DIM);

        for (ijk[0] = 0;  ijk[0] < int(BoolLeafNodeType::DIM); ++ijk[0]) {
            const Index xPos = yPos + (ijk[0] << (2 * int(BoolLeafNodeType::LOG2DIM)));

            for (ijk[2] = 0; ijk[2] < int(BoolLeafNodeType::DIM); ++ijk[2]) {
                const Index pos = xPos + ijk[2];

                if (valueMask[pos] == 0 && node.isValueOn(pos)) {

                    nijk = origin + ijk.offsetBy(0, step, 0);
                    if (!maskAcc.isValueOn(nijk) && distAcc.getValue(nijk) > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    // If UpWind is true check the Z+ oriented node face, else the Z- oriented face.
    template<bool UpWind>
    void evalExternalNeighborsZ(char* valueMask, const BoolLeafNodeType& node,
        const tree::ValueAccessor<const BoolTreeType>& maskAcc,
        const tree::ValueAccessor<const TreeType>& distAcc) const {

        const Coord& origin = node.origin();
        Coord ijk(0, 0, 0), nijk;
        int step = -1;

        if (UpWind) {
            step = 1;
            ijk[2] = int(BoolLeafNodeType::DIM) - 1;
        }

        for (ijk[0] = 0;  ijk[0] < int(BoolLeafNodeType::DIM); ++ijk[0]) {
            const Index xPos = ijk[0] << (2 * int(BoolLeafNodeType::LOG2DIM));

            for (ijk[1] = 0; ijk[1] < int(BoolLeafNodeType::DIM); ++ijk[1]) {
                const Index pos = ijk[2] + xPos + (ijk[1] << int(BoolLeafNodeType::LOG2DIM));

                if (valueMask[pos] == 0 && node.isValueOn(pos)) {

                    nijk = origin + ijk.offsetBy(0, 0, step);
                    if (!maskAcc.isValueOn(nijk) && distAcc.getValue(nijk) > mIsovalue) {
                        valueMask[pos] = 1;
                    }
                }
            }
        }
    }

    //////////

    TreeType                    const * const mTree;
    BoolTreeType                const * const mFillMask;
    BoolLeafNodeType    const * const * const mFillNodes;
    BoolLeafNodeType                 ** const mNewNodes;
    ValueType                           const mIsovalue;
}; // FillMaskBoundary


/// @brief Constructs a memory light char tree that represents the exterior region with @c +1
///        and the interior regions with @c -1.
template <class TreeType>
inline typename TreeType::template ValueConverter<char>::Type::Ptr
computeEnclosedRegionMask(const TreeType& tree, typename TreeType::ValueType isovalue,
    const typename TreeType::template ValueConverter<bool>::Type* fillMask)
{
    typedef typename TreeType::LeafNodeType                                         LeafNodeType;
    typedef typename TreeType::RootNodeType                                         RootNodeType;
    typedef typename RootNodeType::NodeChainType                                    NodeChainType;
    typedef typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type      InternalNodeType;

    typedef typename TreeType::template ValueConverter<char>::Type                  CharTreeType;
    typedef typename CharTreeType::LeafNodeType                                     CharLeafNodeType;
    typedef typename CharTreeType::RootNodeType                                     CharRootNodeType;
    typedef typename CharRootNodeType::NodeChainType                                CharNodeChainType;

    typedef typename TreeType::template ValueConverter<bool>::Type                  BoolTreeType;
    typedef typename BoolTreeType::LeafNodeType                                     BoolLeafNodeType;

    /////

    const TreeType* treePt = &tree;

    size_t numLeafNodes = 0, numInternalNodes = 0;

    std::vector<const LeafNodeType*> nodes;
    std::vector<size_t> leafnodeCount;

    {
        // compute the prefix sum of the leafnode count in each internal node.
        std::vector<const InternalNodeType*> internalNodes;
        treePt->getNodes(internalNodes);

        numInternalNodes = internalNodes.size();

        leafnodeCount.push_back(0);
        for (size_t n = 0; n < numInternalNodes; ++n) {
            leafnodeCount.push_back(leafnodeCount.back() + internalNodes[n]->leafCount());
        }

        numLeafNodes = leafnodeCount.back();

        // extract all leafnodes
        nodes.reserve(numLeafNodes);

        for (size_t n = 0; n < numInternalNodes; ++n) {
            internalNodes[n]->getNodes(nodes);
        }
    }

    // create mask leafnodes
    boost::scoped_array<CharLeafNodeType*> maskNodes(new CharLeafNodeType*[numLeafNodes]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeafNodes),
        LabelBoundaryVoxels<LeafNodeType>(isovalue, &nodes[0], maskNodes.get()));

    // create mask grid
    typename CharTreeType::Ptr maskTree(new CharTreeType(1));

    PopulateTree<CharTreeType> populate(*maskTree, maskNodes.get(), &leafnodeCount[0], 1);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, numInternalNodes), populate);

    // optionally evaluate the fill mask

    std::vector<CharLeafNodeType*> extraMaskNodes;

    if (fillMask) {

        std::vector<const BoolLeafNodeType*> fillMaskNodes;
        fillMask->getNodes(fillMaskNodes);

        boost::scoped_array<BoolLeafNodeType*> boundaryMaskNodes(new BoolLeafNodeType*[fillMaskNodes.size()]);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, fillMaskNodes.size()),
            FillMaskBoundary<TreeType>(tree, isovalue, *fillMask, &fillMaskNodes[0], boundaryMaskNodes.get()));

        tree::ValueAccessor<CharTreeType> maskAcc(*maskTree);

        for (size_t n = 0, N = fillMaskNodes.size(); n < N; ++n) {

            if (boundaryMaskNodes[n] == NULL) continue;

            const BoolLeafNodeType& boundaryNode = *boundaryMaskNodes[n];
            const Coord& origin = boundaryNode.origin();

            CharLeafNodeType* maskNodePt = maskAcc.probeLeaf(origin);

            if (!maskNodePt) {
                maskNodePt = maskAcc.touchLeaf(origin);
                extraMaskNodes.push_back(maskNodePt);
            }

            char* data = maskNodePt->buffer().data();

            typename BoolLeafNodeType::ValueOnCIter it = boundaryNode.cbeginValueOn();
            for (; it; ++it) {
                if (data[it.pos()] != 0) data[it.pos()] = -1;
            }

            delete boundaryMaskNodes[n];
        }
    }

    // eliminate enclosed regions
    tools::traceExteriorBoundaries(*maskTree);

    // flip voxel sign to negative inside and positive outside.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeafNodes),
        FlipRegionSign<CharLeafNodeType>(maskNodes.get()));

    if (!extraMaskNodes.empty()) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, extraMaskNodes.size()),
            FlipRegionSign<CharLeafNodeType>(&extraMaskNodes[0]));
    }

    // propagate sign information into tile region
    tools::signedFloodFill(*maskTree);

    return maskTree;
} // computeEnclosedRegionMask()


template <class TreeType>
inline typename TreeType::template ValueConverter<bool>::Type::Ptr
computeInteriorMask(const TreeType& tree, typename TreeType::ValueType iso)
{
    typedef typename TreeType::LeafNodeType                                         LeafNodeType;
    typedef typename TreeType::RootNodeType                                         RootNodeType;
    typedef typename RootNodeType::NodeChainType                                    NodeChainType;
    typedef typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type      InternalNodeType;

    typedef typename TreeType::template ValueConverter<bool>::Type                  BoolTreeType;
    typedef typename BoolTreeType::LeafNodeType                                     BoolLeafNodeType;
    typedef typename BoolTreeType::RootNodeType                                     BoolRootNodeType;
    typedef typename BoolRootNodeType::NodeChainType                                BoolNodeChainType;
    typedef typename boost::mpl::at<BoolNodeChainType, boost::mpl::int_<1> >::type  BoolInternalNodeType;

    /////
    size_t numLeafNodes = 0, numInternalNodes = 0;

    std::vector<const LeafNodeType*> nodes;
    std::vector<size_t> leafnodeCount;

    {
        // compute the prefix sum of the leafnode count in each internal node.
        std::vector<const InternalNodeType*> internalNodes;
        tree.getNodes(internalNodes);

        numInternalNodes = internalNodes.size();

        leafnodeCount.push_back(0);
        for (size_t n = 0; n < numInternalNodes; ++n) {
            leafnodeCount.push_back(leafnodeCount.back() + internalNodes[n]->leafCount());
        }

        numLeafNodes = leafnodeCount.back();

        // extract all leafnodes
        nodes.reserve(numLeafNodes);

        for (size_t n = 0; n < numInternalNodes; ++n) {
            internalNodes[n]->getNodes(nodes);
        }
    }

    // create mask leafnodes
    boost::scoped_array<BoolLeafNodeType*> maskNodes(new BoolLeafNodeType*[numLeafNodes]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numLeafNodes),
        MaskInteriorVoxels<LeafNodeType>(iso, &nodes[0], maskNodes.get()));


    // create mask grid
    typename BoolTreeType::Ptr maskTree(new BoolTreeType(false));

    PopulateTree<BoolTreeType> populate(*maskTree, maskNodes.get(), &leafnodeCount[0], false);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, numInternalNodes), populate);


    // evaluate tile values
    std::vector<BoolInternalNodeType*> internalMaskNodes;
    maskTree->getNodes(internalMaskNodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, internalMaskNodes.size()),
        MaskInteriorTiles<TreeType, BoolInternalNodeType>(iso, tree, &internalMaskNodes[0]));

    tree::ValueAccessor<const TreeType> acc(tree);

    typename BoolTreeType::ValueAllIter it(*maskTree);
    it.setMaxDepth(BoolTreeType::ValueAllIter::LEAF_DEPTH - 2);

    for ( ; it; ++it) {
        if (acc.getValue(it.getCoord()) < iso) {
            it.setValue(true);
            it.setActiveState(true);
        }
    }

    return maskTree;
} // computeInteriorMask()


template<typename TreeType>
struct GridOrTreeConstructor
{
    typedef typename TreeType::template ValueConverter<bool>::Type::Ptr BoolTreePtrType;
    static BoolTreePtrType construct(const TreeType&, BoolTreePtrType& maskTree) { return maskTree; }
};


template<typename TreeType>
struct GridOrTreeConstructor<Grid<TreeType> >
{
    typedef Grid<TreeType>                                          GridType;

    typedef typename TreeType::template ValueConverter<bool>::Type  BoolTreeType;
    typedef typename BoolTreeType::Ptr                              BoolTreePtrType;
    typedef Grid<BoolTreeType>                                      BoolGridType;
    typedef typename BoolGridType::Ptr                              BoolGridPtrType;

    static BoolGridPtrType construct(const GridType& grid, BoolTreePtrType& maskTree) {
        BoolGridPtrType maskGrid(BoolGridType::create(maskTree));
        maskGrid->setTransform(grid.transform().copy());
        return maskGrid;
    }
};


} // namespace level_set_util_internal


////////////////////////////////////////


template <class GridType>
inline void
sdfToFogVolume(GridType& grid, typename GridType::ValueType cutoffDistance)
{
    typedef typename GridType::ValueType                                            ValueType;
    typedef typename GridType::TreeType                                             TreeType;
    typedef typename TreeType::LeafNodeType                                         LeafNodeType;
    typedef typename TreeType::RootNodeType                                         RootNodeType;
    typedef typename RootNodeType::NodeChainType                                    NodeChainType;
    typedef typename boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type      InternalNodeType;

    //////////

    TreeType& tree = grid.tree();

    size_t numLeafNodes = 0, numInternalNodes = 0;

    std::vector<LeafNodeType*> nodes;
    std::vector<size_t> leafnodeCount;

    {
        // Compute the prefix sum of the leafnode count in each internal node.
        std::vector<InternalNodeType*> internalNodes;
        tree.getNodes(internalNodes);

        numInternalNodes = internalNodes.size();

        leafnodeCount.push_back(0);
        for (size_t n = 0; n < numInternalNodes; ++n) {
            leafnodeCount.push_back(leafnodeCount.back() + internalNodes[n]->leafCount());
        }

        numLeafNodes = leafnodeCount.back();

        // Steal all leafnodes (Removes them from the tree and transfers ownership.)
        nodes.reserve(numLeafNodes);

        for (size_t n = 0; n < numInternalNodes; ++n) {
            internalNodes[n]->stealNodes(nodes, tree.background(), false);
        }

        // Clamp cutoffDistance to min sdf value
        ValueType minSDFValue = std::numeric_limits<ValueType>::max();

        {
            level_set_util_internal::FindMinTileValue<InternalNodeType> minOp(&internalNodes[0]);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, internalNodes.size()), minOp);
            minSDFValue = std::min(minSDFValue, minOp.minValue);
        }

        if (minSDFValue > ValueType(0.0)) {
            level_set_util_internal::FindMinVoxelValue<LeafNodeType> minOp(&nodes[0]);
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), minOp);
            minSDFValue = std::min(minSDFValue, minOp.minValue);
        }

        cutoffDistance = -std::abs(cutoffDistance);
        cutoffDistance = minSDFValue > cutoffDistance ? minSDFValue : cutoffDistance;
    }

    // Transform voxel values and delete leafnodes that are uniformly zero after the transformation.
    // (Positive values are set to zero with inactive state and negative values are remapped
    // from zero to one with active state.)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
        level_set_util_internal::SDFVoxelsToFogVolume<LeafNodeType>(&nodes[0], cutoffDistance));

    // Populate a new tree with the remaining leafnodes
    typename TreeType::Ptr newTree(new TreeType(ValueType(0.0)));

    level_set_util_internal::PopulateTree<TreeType> populate(*newTree, &nodes[0], &leafnodeCount[0], 0);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, numInternalNodes), populate);

    // Transform tile values (Negative valued tiles are set to 1.0 with active state.)
    std::vector<InternalNodeType*> internalNodes;
    newTree->getNodes(internalNodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, internalNodes.size()),
        level_set_util_internal::SDFTilesToFogVolume<TreeType, InternalNodeType>(tree, &internalNodes[0]));

    {
        tree::ValueAccessor<const TreeType> acc(tree);

        typename TreeType::ValueAllIter it(*newTree);
        it.setMaxDepth(TreeType::ValueAllIter::LEAF_DEPTH - 2);

        for ( ; it; ++it) {
            if (acc.getValue(it.getCoord()) < ValueType(0.0)) {
                it.setValue(ValueType(1.0));
                it.setActiveState(true);
            }
        }
    }

    // Insert missing root level tiles. (The new tree is constructed from the remaining leafnodes
    // and will therefore not contain any root level tiles that may exist in the original tree.)
    {
        typename TreeType::ValueAllIter it(tree);
        it.setMaxDepth(TreeType::ValueAllIter::ROOT_DEPTH);
        for ( ; it; ++it) {
            if (it.getValue() <  ValueType(0.0)) {
                newTree->addTile(TreeType::ValueAllIter::ROOT_LEVEL, it.getCoord(), ValueType(1.0), true);
            }
        }
    }

    grid.setTree(newTree);
    grid.setGridClass(GRID_FOG_VOLUME);
}


////////////////////////////////////////


template <class GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
sdfInteriorMask(const GridOrTreeType& volume, typename GridOrTreeType::ValueType isovalue)
{
    typedef typename TreeAdapter<GridOrTreeType>::TreeType TreeType;
    const TreeType& tree = TreeAdapter<GridOrTreeType>::tree(volume);

    typedef typename TreeType::template ValueConverter<bool>::Type::Ptr BoolTreePtrType;
    BoolTreePtrType mask = level_set_util_internal::computeInteriorMask(tree, isovalue);

    return level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::construct(volume, mask);
}


template<typename GridOrTreeType>
inline typename GridOrTreeType::template ValueConverter<bool>::Type::Ptr
extractEnclosedRegion(const GridOrTreeType& volume,
    typename GridOrTreeType::ValueType isovalue,
    const typename TreeAdapter<GridOrTreeType>::TreeType::template ValueConverter<bool>::Type* fillMask)
{
    typedef typename TreeAdapter<GridOrTreeType>::TreeType TreeType;
    const TreeType& tree = TreeAdapter<GridOrTreeType>::tree(volume);

    typedef typename TreeType::template ValueConverter<char>::Type::Ptr CharTreePtrType;
    CharTreePtrType regionMask = level_set_util_internal::computeEnclosedRegionMask(tree, isovalue, fillMask);

    typedef typename TreeType::template ValueConverter<bool>::Type::Ptr BoolTreePtrType;
    BoolTreePtrType mask = level_set_util_internal::computeInteriorMask(*regionMask, 0);

    return level_set_util_internal::GridOrTreeConstructor<GridOrTreeType>::construct(volume, mask);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )

