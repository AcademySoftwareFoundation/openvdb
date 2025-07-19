// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/VoxelBlockManager.cuh

    \author Eftychios Sifakis

    \date January 27, 2025

    \brief Implements device functions to build and query the VoxelBlockManager.
           The VoxelBlockManager is an acceleration structure for sequential access
           and stencil operations over solely the active voxels of an OnIndexGrid
           which enables SIMT parallelism independent of occupancy.
*/

#include <cuda/atomic>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/cuda/Util.h>

#ifndef NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED

namespace nanovdb {

namespace tools::cuda {

/// @brief This class allows for sequential access to contiguous spans (i.e.
/// blocks) of active voxels in an IndexGrid on the device, and efficient
/// computation of linear offsets of stencils (e.g. 3-wide box stencil).
/// @tparam BlockWidth Number of active voxels in a contiguous span
template <int BlockWidth>
struct VoxelBlockManager
{
    // The efficiency of the functions in this class are contingent on
    // threadblock-level coordination, which manifests either as using shared
    // memory for synchronization, or warp-level shift operations.

    // The jumpMap is packed in uint64_t's, one per 64 consecutive bits. Hence
    // JumpMapLength is the number of uint64_t that are needed to straddle the
    // total BlockWidth bits of the jumpMap.
    static constexpr int JumpMapLength = BlockWidth/64;

    static constexpr uint32_t UnusedLeafIndex = 0xffffffff;
    static constexpr uint16_t UnusedVoxelOffset = 0xffff;

    /// @brief Given a grid and the associated jumpMap in global memory, compute
    /// the leaf indices and voxel offsets in shared memory. This function
    /// writes to shared memory and synchronizes threads and thus should not be
    /// called from divergent threads within a thread block.
    /// @tparam BuildT Build type of the grid
    /// @param grid
    /// @param firstLeafID
    /// @param jumpMap
    /// @param blockFirstOffset
    /// @param smem_leafIndex Leaf indices stored in shared memory
    /// @param smem_voxelOffset Voxel offsets stored in shared memory
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    decodeInverseMaps(
        NanoGrid<BuildT> *grid,
        const uint32_t firstLeafID,
        const uint64_t *jumpMap,
        const uint64_t blockFirstOffset,
        uint32_t *smem_leafIndex,
        uint16_t *smem_voxelOffset)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        int tID = threadIdx.x;

        // Count how many additional leaves (following the one indicated by firstLeafID)
        // overlap with this voxel block
        int nExtraLeaves = 0;
        for (int i = 0; i < JumpMapLength; i++)
            nExtraLeaves += util::countOn(jumpMap[i]);

        // Initialize leafIndex & voxelOffset to sentinel values
        // for blocks that extend beyond the last active voxel in the grid
        if (tID < BlockWidth)
#pragma cuda unroll
            for (int i = 0; i < BlockWidth; i += blockDim.x) {
                smem_leafIndex[i+tID] = UnusedLeafIndex;
                smem_voxelOffset[i+tID] = UnusedVoxelOffset;
            }
        __syncthreads();

        NANOVDB_ASSERT(blockDim.x <= 512);
        const auto& tree = grid->tree();
        // Loop through all leafNodes overlapping the voxel block
        // with all threads in threadblock working collaboratively within each leafNode
        for (int leafID = firstLeafID; leafID <= firstLeafID + nExtraLeaves; leafID++) {
            const auto& leaf = tree.template getFirstNode<0>()[leafID];
            const Coord origin = leaf.origin();
            for (int threadOffset = 0; threadOffset < 512; threadOffset += blockDim.x) {
                int localOffset = threadOffset + tID;
                int index = leaf.data()->getValue(localOffset);
                if ((index >= blockFirstOffset) && (index < blockFirstOffset + BlockWidth)) {
                    int blockOffset = index - blockFirstOffset;
                    // Write inverse map to shared memory; no collisions
                    smem_leafIndex[blockOffset] = leafID;
                    smem_voxelOffset[blockOffset] = localOffset;
                }
            }
        }
        __syncthreads();
    }

    /// @brief Given a grid and its decoded voxel map, compute the stencil.
    /// This function accesses shared memory but does not synchronize threads
    /// so it may be called from divergent threads within a thread block.
    /// offsets for a 3x3x3 box stencil.
    /// @tparam BuildT Build type of the grid
    /// @param grid
    /// @param smem_leafIndex Leaf indices stored in shared memory
    /// @param smem_voxelOffset Voxel offsets stored in shared memory
    /// @param stencilIndices Pointer to output stencil indices. Must have
    /// length of at least 27 (corresponding to the 3x3x3 stencil)
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeBoxStencil(
        const NanoGrid<BuildT> *grid,
        const uint32_t *smem_leafIndex,
        const uint16_t *smem_voxelOffset,
        uint64_t *stencilIndices)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        int tID = threadIdx.x;
        const auto& tree = grid->tree();
        if (smem_leafIndex[tID] != UnusedLeafIndex) {
            // This presumes that leaf nodes are fixed-size and sequentially accessible in memory
            const auto& leaf = tree.template getFirstNode<0>()[ smem_leafIndex[tID] ];
            const Coord coord = leaf.offsetToGlobalCoord( smem_voxelOffset[tID] );
            const auto index = leaf.getValue( smem_voxelOffset[tID] );
            for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++) {
                int spokeID = ( di + 1 ) * 9 + ( dj + 1 ) * 3 + dk + 1;
                const auto neighbor = coord.offsetBy( di, dj, dk );
                stencilIndices[spokeID] = tree.getValue( neighbor );
            }
        }
    }
};

/// @brief This functor calculates the firstLeafID and jumpMap for the
/// VoxelBlockManager over the subset of the Tree nodes specified by
/// firstOffset, lastOffset, and nBlocks.
template<int BlockWidthLog2, int NumThreads>
struct BuildVoxelBlockManagerFunctor
{
    static constexpr int BlockWidth = 1 << BlockWidthLog2;
    static constexpr int JumpMapLength = BlockWidth/64;
    static constexpr int SlicesPerLowerNode = 8;
    static constexpr int LeafNodesPerSlice = 4096/SlicesPerLowerNode;

    static constexpr int MaxThreadsPerBlock = NumThreads;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    void __device__
    operator()(
        uint64_t firstOffset,
        uint64_t lastOffset,
        int nBlocks,
        const NanoGrid<ValueOnIndex> *grid,
        uint32_t *firstLeafID,
        uint64_t *jumpMap)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        using JumpMapType = uint64_t (&)[][JumpMapLength];

        int tID = threadIdx.x;
        int blockID = blockIdx.x;
        int sliceID = blockIdx.y;

        const auto& tree = grid->tree();

        const auto& lower = tree.getFirstNode<1>()[blockID];
        for ( std::size_t jj = sliceID*LeafNodesPerSlice; jj < (sliceID+1)*LeafNodesPerSlice; jj += MaxThreadsPerBlock )
            if ( lower.childMask().isOn(jj+tID) )
            {
                auto& leaf = *lower.getChild(jj+tID);
                const auto leafFirstOffset = leaf.data()->firstOffset();
                const auto leafValueCount = leaf.data()->valueCount();
                const auto leafLastOffset = leafFirstOffset + leafValueCount - 1;

                auto leafIndex = util::PtrDiff(&leaf,tree.getFirstNode<0>()) / sizeof(NanoLeaf<ValueOnIndex>);

                if ( ( leafFirstOffset > lastOffset ) || (leafLastOffset < firstOffset) ) continue;

                int lastBlock = (leafLastOffset - firstOffset) >> BlockWidthLog2;
                lastBlock = min(lastBlock, nBlocks-1);
                uint64_t firstBlock = (leafFirstOffset < firstOffset) ? 0 :
                    (leafFirstOffset - firstOffset) >> BlockWidthLog2;

                // For all but the first block touched, mark the firstLeaf as being this one
                for ( uint64_t b = lastBlock; b > firstBlock; --b )
                    firstLeafID[b] = leafIndex;
                if (leafFirstOffset < firstOffset) { firstLeafID[0] = leafIndex; continue; }

                const auto offsetInBlock = (leafFirstOffset - 1) & (BlockWidth - 1);
                if ( !offsetInBlock ) {
                    // If the first leaf starts exactly at the beginning of a
                    // block, register it in mFirstLeaf too
                    firstLeafID[firstBlock] = leafIndex;
                } else {
                    // Otherwise, mark it in the jumpMap
                    // The specific uint64_t in the jumpMap to be marked is at element offset (offsetInBlock>>6), i.e. offsetBlock/64
                    // and bit offset (offsetInBlock & 0x3f), i.e. offsetInBlock%64
                    ::cuda::atomic_ref<uint64_t> ref(jumpMap[firstBlock * JumpMapLength + (offsetInBlock>>6)]);
                    ref.fetch_or(0x1ul << (offsetInBlock & 0x3f));
                }
            }

        return;
    }

};

/// @brief Helper function to build a VoxelBlockManager on the device
template<int BlockWidthLog2, int NumThreads = 128>
void buildVoxelBlockManager(
    uint64_t firstOffset,
    uint64_t lastOffset,
    int nBlocks,
    int lowerCount,
    const NanoGrid<ValueOnIndex> *grid,
    uint32_t *firstLeafID,
    uint64_t *jumpMap,
    cudaStream_t stream = 0)
{
    using Op = BuildVoxelBlockManagerFunctor<BlockWidthLog2, NumThreads>;
    util::cuda::operatorKernel<Op>
        <<<dim3(lowerCount,Op::SlicesPerLowerNode,1), NumThreads, 0, stream>>>(
            firstOffset, lastOffset, nBlocks, grid, firstLeafID, jumpMap);
}

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED
