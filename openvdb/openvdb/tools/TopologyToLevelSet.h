// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file    TopologyToLevelSet.h
///
/// @brief   This tool generates a narrow-band signed distance field / level set
///          from the interface between active and inactive voxels in a vdb grid.
///
/// @par Example:
/// Combine with @c tools::PointsToVolume for fast point cloud to level set conversion.

#ifndef OPENVDB_TOOLS_TOPOLOGY_TO_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_TOPOLOGY_TO_LEVELSET_HAS_BEEN_INCLUDED

#include "LevelSetFilter.h"
#include "Morphology.h" // for erodeActiveValues and dilateActiveValues

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/FiniteDifference.h> // for math::BiasedGradientScheme
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>

#include <tbb/task_group.h>

#include <algorithm> // for std::min(), std::max()
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief   Compute the narrow-band signed distance to the interface between
///          active and inactive voxels in the input grid.
///
/// @return  A shared pointer to a new sdf / level set grid of type @c float
///
/// @param grid            Input grid of arbitrary type whose active voxels are used
///                        in constructing the level set.
/// @param halfWidth       Half the width of the narrow band in voxel units.
/// @param closingSteps    Number of morphological closing steps used to fill gaps
///                        in the active voxel region.
/// @param dilation        Number of voxels to expand the active voxel region.
/// @param smoothingSteps  Number of smoothing interations.
template<typename GridT>
typename GridT::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridT& grid,
    int halfWidth = 3,
    int closingSteps = 1,
    int dilation = 0,
    int smoothingSteps = 0);


/// @brief   Compute the narrow-band signed distance to the interface between
///          active and inactive voxels in the input grid.
///
/// @return  A shared pointer to a new sdf / level set grid of type @c float
///
/// @param grid            Input grid of arbitrary type whose active voxels are used
///                        in constructing the level set.
/// @param halfWidth       Half the width of the narrow band in voxel units.
/// @param closingSteps    Number of morphological closing steps used to fill gaps
///                        in the active voxel region.
/// @param dilation        Number of voxels to expand the active voxel region.
/// @param smoothingSteps  Number of smoothing interations.
/// @param interrupt       Optional object adhering to the util::NullInterrupter interface.
template<typename GridT, typename InterrupterT>
typename GridT::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridT& grid,
    int halfWidth = 3,
    int closingSteps = 1,
    int dilation = 0,
    int smoothingSteps = 0,
    InterrupterT* interrupt = nullptr);


////////////////////////////////////////

/// @cond OPENVDB_DOCS_INTERNAL

namespace ttls_internal {


template<typename TreeType>
struct OffsetAndMinComp
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using ValueType = typename TreeType::ValueType;

    OffsetAndMinComp(std::vector<LeafNodeType*>& lhsNodes,
        const TreeType& rhsTree,
        ValueType offset)
        : mLhsNodes(lhsNodes)
        , mRhsTree(rhsTree)
        , mOffset(offset) {}

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        using Iterator = typename LeafNodeType::ValueOnIter;

        tree::ValueAccessor<const TreeType> rhsAcc(mRhsTree);
        const ValueType offset = mOffset;

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            LeafNodeType& lhsNode = *mLhsNodes[n];
            const LeafNodeType* rhsNodePt = rhsAcc.probeConstLeaf(lhsNode.origin());
            if (!rhsNodePt) continue;

            auto* const data = lhsNode.buffer().data();
            for (Iterator it = lhsNode.beginValueOn(); it; ++it) {
                ValueType& val = data[it.pos()];
                val = std::min(val, offset + rhsNodePt->getValue(it.pos()));
            }
        }
    }

private:
    std::vector<LeafNodeType*>&  mLhsNodes;
    const TreeType&  mRhsTree;
    const ValueType  mOffset;
}; // struct OffsetAndMinComp


template<typename GridType, typename InterrupterType>
inline void
normalizeLevelSet(GridType& grid, const int halfWidthInVoxels, InterrupterType* interrupt = nullptr)
{
    LevelSetFilter<GridType, GridType, InterrupterType> filter(grid, interrupt);
    filter.setSpatialScheme(math::FIRST_BIAS);
    filter.setNormCount(halfWidthInVoxels);
    filter.normalize();
    filter.prune();
}


template<typename GridType, typename InterrupterType>
inline void
smoothLevelSet(GridType& grid, int iterations, int halfBandWidthInVoxels,
    InterrupterType* interrupt = nullptr)
{
    using ValueType = typename GridType::ValueType;
    using TreeType = typename GridType::TreeType;
    using LeafNodeType = typename TreeType::LeafNodeType;

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

/// @endcond

template<typename GridT, typename InterrupterT>
typename GridT::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridT& grid, int halfWidth, int closingSteps, int dilation,
    int smoothingSteps, InterrupterT* interrupt)
{
    using MaskTreeT = typename GridT::TreeType::template ValueConverter<ValueMask>::Type;
    using FloatTreeT = typename GridT::TreeType::template ValueConverter<float>::Type;
    using FloatGridT = Grid<FloatTreeT>;

    // Check inputs

    halfWidth = std::max(halfWidth, 1);
    closingSteps = std::max(closingSteps, 0);
    dilation = std::max(dilation, 0);

    if (!grid.hasUniformVoxels()) {
        OPENVDB_THROW(ValueError, "Non-uniform voxels are not supported!");
    }

    // Copy the topology into a MaskGrid.
    MaskTreeT maskTree(grid.tree(), false/*background*/, openvdb::TopologyCopy());

    // Morphological closing operation.
    tools::dilateActiveValues(maskTree, closingSteps + dilation, tools::NN_FACE, tools::PRESERVE_TILES);
    tools::erodeActiveValues(maskTree, /*iterations=*/closingSteps, tools::NN_FACE, tools::PRESERVE_TILES);
    tools::pruneInactive(maskTree);

    // Generate a volume with an implicit zero crossing at the boundary
    // between active and inactive values in the input grid.
    const float background = float(grid.voxelSize()[0]) * float(halfWidth);
    typename FloatTreeT::Ptr lsTree(
        new FloatTreeT(maskTree, /*out=*/background, /*in=*/-background, openvdb::TopologyCopy()));

    tbb::task_group pool;
    pool.run([&]() {
        tools::erodeActiveValues(maskTree, /*iterations=*/halfWidth, tools::NN_FACE, tools::PRESERVE_TILES);
        tools::pruneInactive(maskTree);
    });
    pool.run([&]() {
        tools::dilateActiveValues(*lsTree, /*iterations=*/halfWidth, tools::NN_FACE, tools::PRESERVE_TILES);
    });
    pool.wait();// wait for both tasks to complete

    lsTree->topologyDifference(maskTree);
    maskTree.clear();

    lsTree->voxelizeActiveTiles(); // has to come before ::pruneLevelSet
    tools::pruneLevelSet(*lsTree,  /*threading=*/true);

    // Create a level set grid from the tree
    typename FloatGridT::Ptr lsGrid = FloatGridT::create(lsTree);
    lsGrid->setTransform(grid.transform().copy());
    lsGrid->setGridClass(openvdb::GRID_LEVEL_SET);

    // Use a PDE based scheme to propagate distance values from the
    // implicit zero crossing.
    ttls_internal::normalizeLevelSet(*lsGrid, 3*halfWidth, interrupt);

    // Additional filtering
    if (smoothingSteps > 0) {
        ttls_internal::smoothLevelSet(*lsGrid, smoothingSteps, halfWidth, interrupt);
    }

    return lsGrid;
}


template<typename GridT>
typename GridT::template ValueConverter<float>::Type::Ptr
topologyToLevelSet(const GridT& grid, int halfWidth, int closingSteps, int dilation, int smoothingSteps)
{
    util::NullInterrupter interrupt;
    return topologyToLevelSet(grid, halfWidth, closingSteps, dilation, smoothingSteps, &interrupt);
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_TOPOLOGYTOLEVELSET
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    Grid<TreeT>::ValueConverter<float>::Type::Ptr topologyToLevelSet(const Grid<TreeT>&, int, int, int, int, \
        util::NullInterrupter*)
OPENVDB_ALL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_TOPOLOGY_TO_LEVELSET_HAS_BEEN_INCLUDED

