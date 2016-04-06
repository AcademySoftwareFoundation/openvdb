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
/// @file    TopologyToLevelSet.h
///
/// @brief   This tool converts active grid topology to a into a signed
///          distance field encoded as a narrow band level set.
///
/// @details The boundary between active and inactive voxels is treated
///          as the zero crossing for the level set.
///
/// @par Example:
/// Combine with @c tools::createPointMaskGrid for fast point cloud to level set conversion.
///
/// @author  D.J. Hill

#ifndef OPENVDB_TOOLS_TOPOLOGY_TO_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_TOPOLOGY_TO_LEVELSET_HAS_BEEN_INCLUDED

#include "LevelSetFilter.h"
#include "Morphology.h" // for {dilate|erode}Voxels
#include "Prune.h"// for pruneInactive
#include "SignedFloodFill.h" // for signedFloodFill

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/FiniteDifference.h> // for math::BiasedGradientScheme
#include <openvdb/util/NullInterrupter.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief  Compute the narrow-band signed distance to the boundary
///         between active and inactive voxels in the input grid.
///
/// @return A shared pointer to a new signed distance field of type @c float
///
/// @param grid           Input grid of arbitrary type whose active voxels are used
///                       in constructing the level set.
/// @param halfBandWidth  Half the width of the narrow band, in voxel units
/// @param closingWidth   Number of iterations used to first expand and then shrink
///                       the filled voxel region.
///                       This causes holes and valleys to be filled.
/// @param dilation       Number of iterations used to expand the filled voxel region.
/// @param smoothingSteps Number of smoothing interations
template<typename GridType>
inline typename GridType::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridType& grid, int halfBandWidth = 3, int closingWidth = 1,
    int dilation = 0, int smoothingSteps = 0);


/// @brief  Compute the narrow-band signed distance to the boundary
///         between active and inactive voxels in the input grid.
///
/// @return A shared pointer to a new signed distance field of type @c float
///
/// @param grid           Input grid of arbitrary type whose active voxels are used
///                       in constructing the level set.
/// @param halfBandWidth  Half the width of the narrow band, in voxel units
/// @param closingWidth   Number of iterations used to first expand and then shrink
///                       the filled voxel region.
///                       This causes holes and valleys to be filled.
/// @param dilation       Number of iterations used to expand the filled voxel region.
/// @param smoothingSteps Number of smoothing interations
/// @param interrupt      Optional object adhering to the util::NullInterrupter interface.
template<typename GridType, typename InterrupterType>
inline typename GridType::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridType& grid, int halfBandWidth = 3, int closingWidth = 1,
    int dilation = 0, int smoothingSteps = 0, InterrupterType* interrupt = NULL);


////////////////////////////////////////
////////////////////////////////////////

// Implementation details


namespace ttls_internal {


template<typename TreeType>
struct OffsetAndMinComp
{
    typedef typename TreeType::LeafNodeType     LeafNodeType;
    typedef typename TreeType::ValueType        ValueType;

    OffsetAndMinComp(std::vector<LeafNodeType*>& lhsNodes, const TreeType& rhsTree, ValueType offset)
        : mLhsNodes(lhsNodes.empty() ? NULL : &lhsNodes[0]), mRhsTree(&rhsTree), mOffset(offset)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        typedef typename LeafNodeType::ValueOnIter Iterator;

        tree::ValueAccessor<const TreeType> rhsAcc(*mRhsTree);
        const ValueType offset = mOffset;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& lhsNode = *mLhsNodes[n];
            const LeafNodeType * rhsNodePt = rhsAcc.probeConstLeaf(lhsNode.origin());
            if (!rhsNodePt) continue;

            for (Iterator it = lhsNode.beginValueOn(); it; ++it) {
                ValueType& val = const_cast<ValueType&>(it.getValue());
                val = std::min(val, offset + rhsNodePt->getValue(it.pos()));
            }
        }
    }

private:
    LeafNodeType    *       * const mLhsNodes;
    TreeType          const * const mRhsTree;
    ValueType                 const mOffset;
}; // struct OffsetAndMinComp


template<typename GridType, typename InterrupterType>
inline void
normalizeLevelSet(GridType& grid, const int halfWidthInVoxels, InterrupterType* interrupt = NULL)
{
    LevelSetFilter<GridType, GridType, InterrupterType> filter(grid, interrupt);
    filter.setSpatialScheme(math::FIRST_BIAS);
    filter.setNormCount(halfWidthInVoxels);
    filter.normalize();
    filter.prune();
}


template<typename GridType, typename InterrupterType>
inline void
smoothLevelSet(GridType& grid, int iterations, int halfBandWidthInVoxels, InterrupterType* interrupt = NULL)
{
    typedef typename GridType::ValueType        ValueType;
    typedef typename GridType::TreeType         TreeType;
    typedef typename TreeType::LeafNodeType     LeafNodeType;

    GridType filterGrid(grid);

    LevelSetFilter<GridType, GridType, InterrupterType> filter(filterGrid, interrupt);
    filter.setSpatialScheme(math::FIRST_BIAS);

    for (int n = 0; n < iterations; ++n) {
        if (interrupt && interrupt->wasInterrupted()) break;
        filter.mean(1);
    }

    std::vector<LeafNodeType*> nodes;
    grid.tree().getNodes(nodes);

    const ValueType offset = ValueType(double(0.5) * grid.transform().voxelSize()[0]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, nodes.size()),
        OffsetAndMinComp<TreeType>(nodes, filterGrid.tree(), -offset));

    // Clean up any damanage that was done by the min operation
    normalizeLevelSet(grid, halfBandWidthInVoxels, interrupt);
}


} // namespace ttls_internal


////////////////////////////////////////


template<typename GridType, typename InterrupterType>
inline typename GridType::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridType& grid, int halfBandWidth, int closingWidth,
    int dilation, int smoothingSteps, InterrupterType* interrupt)
{
    typedef typename GridType::template ValueConverter<float>::Type             FloatGridType;
    typedef typename FloatGridType::TreeType                                    FloatTreeType;
    typedef typename FloatTreeType::template ValueConverter<ValueMask>::Type    MaskTreeType;

    halfBandWidth = std::max(halfBandWidth, 1);
    closingWidth = std::max(closingWidth, 0);

    MaskTreeType regionMask(grid.tree(), false, TopologyCopy());

    // closing operation and padding
    openvdb::tools::dilateVoxels(regionMask, closingWidth + dilation);
    openvdb::tools::erodeVoxels(regionMask, closingWidth);


    // Construct inside band mask

    MaskTreeType coreMask(regionMask);
    openvdb::tools::erodeVoxels(coreMask, halfBandWidth);

    regionMask.topologyDifference(coreMask);
    tools::pruneInactive(regionMask,  /*threading=*/true);

    // Generate a volume with an implicit zero crossing at the boundary
    // between active and inactive values in the input grid.

    const float width = float(grid.transform().voxelSize()[0] * double(halfBandWidth));

    typename FloatTreeType::Ptr resultTree(
        new FloatTreeType(regionMask, /*inactive=*/width, /*active=*/-width, openvdb::TopologyCopy()));

    // Construct outside band mask
    openvdb::tools::dilateVoxels(regionMask, halfBandWidth);
    regionMask.topologyDifference(coreMask);
    tools::pruneInactive(regionMask,  /*threading=*/true);

    // Activate outside band
    resultTree->topologyUnion(regionMask);

    // Update interior sign
    tools::signedFloodFill(*resultTree);

    // Embed the tree in a grid to define a transform and voxel size.
    typename FloatGridType::Ptr resultGrid = FloatGridType::create(resultTree);
    resultGrid->setTransform(grid.transform().copy());
    resultGrid->setGridClass(GRID_LEVEL_SET);

    // Use a PDE based scheme to propagate distance values from the
    // implicit zero crossing.
    ttls_internal::normalizeLevelSet(*resultGrid, 3*halfBandWidth, interrupt);

    // Optinal smooting operation
    if (smoothingSteps > 0) {
        ttls_internal::smoothLevelSet(*resultGrid, smoothingSteps, halfBandWidth, interrupt);
    }

    return resultGrid;
}


template<typename GridType>
inline typename GridType::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridType& grid, int halfBandWidth, int closingWidth, int dilation, int smoothingSteps)
{
    util::NullInterrupter interrupt;
    return topologyToLevelSet(grid, halfBandWidth, closingWidth, dilation, smoothingSteps, &interrupt);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif //OPENVDB_TOOLS_DENSESPARSETOOLS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
