// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/GridStats.cuh

    \author Ken Museth

    \date October 9, 2023

    \brief Re-computes min/max/avg/var/bbox information for each node in a
           pre-existing NanoVDB grid on the device.
*/

#ifndef NANOVDB_TOOLS_CUDA_GRIDSTATS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_GRIDSTATS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridStats.h>

namespace nanovdb {

namespace tools::cuda {

/// @brief Update, i.e. re-compute, grid statistics like min/max, stats and bbox
///        information for an existing NanoVDB Grid.
/// @param grid   Grid whose stats to update
/// @param mode   Mode of computation for the statistics.
/// @param stream Optional cuda stream (defaults to zero)
template<typename BuildT>
void updateGridStats(NanoGrid<BuildT> *d_grid, StatsMode mode = StatsMode::Default, cudaStream_t stream = 0);

//================================================================================================

/// @brief Allows for the construction of NanoVDB grids without any dependency
template<typename BuildT, typename StatsT = Stats<typename NanoGrid<BuildT>::ValueType>>
class GridStats
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = typename GridT::TreeType;
    using ValueT = typename TreeT::ValueType;
    using Node0  = typename TreeT::Node0; // leaf
    using Node1  = typename TreeT::Node1; // lower
    using Node2  = typename TreeT::Node2; // upper
    using RootT  = typename TreeT::Node3; // root
    static_assert(util::is_same<ValueT, typename StatsT::ValueType>::value, "Mismatching type");

    ValueT mDelta; // skip rendering of node if: node.max < -mDelta || node.min > mDelta

public:
    GridStats(ValueT delta = ValueT(0)) : mDelta(delta) {}

    void update(GridT *d_grid, cudaStream_t stream = 0);

}; // cuda::GridStats

//================================================================================================

namespace {// define cuda kernels in an unnamed namespace

// One warp per leaf: lanes stride the 512 voxel slots (mask-gated) and merge
// their partial statistics through shared memory; lane 0 handles the bbox
// update and the final store. Launch with 128 threads (4 warps) per block.
template<typename BuildT, typename StatsT>
__global__ void processLeaf(NodeManager<BuildT> *d_nodeMgr, StatsT *d_stats)
{
    constexpr uint32_t WarpsPerBlock = 4;
    __shared__ StatsT sStats[WarpsPerBlock * 32];

    const uint32_t warpID = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t tid = blockIdx.x * WarpsPerBlock + warpID;// leaf index
    if (tid >= d_nodeMgr->leafCount()) return;
    auto &d_leaf = d_nodeMgr->leaf(tid);

    bool nonEmpty = false;
    if (lane == 0) nonEmpty = d_leaf.updateBBox();// updates active bounding box (also updates data->mFlags)
    nonEmpty = __shfl_sync(0xffffffffu, nonEmpty, 0);

    if (nonEmpty) {
        if constexpr(StatsT::hasStats()) {
            StatsT stats;
            const auto &mask = d_leaf.valueMask();
            for (uint32_t i = lane; i < 512; i += 32)
                if (mask.isOn(i)) stats.add(d_leaf.getValue(i));
            StatsT *sWarp = sStats + (warpID << 5);
            sWarp[lane] = stats;
            __syncwarp();
            for (uint32_t d = 16; d; d >>= 1) {
                if (lane < d) sWarp[lane].add(sWarp[lane + d]);
                __syncwarp();
            }
            if (lane == 0) {
                if constexpr(StatsT::hasAverage()) {
                    d_stats[tid] = sWarp[0];
                    *reinterpret_cast<uint32_t*>(&d_leaf.mMinimum) = tid;
                } else {
                    sWarp[0].setStats(d_leaf);
                }
            }
        }
    }
    if (lane == 0) d_leaf.mFlags &= ~uint8_t(1u);// enable rendering
}// processLeaf

// One block per internal node: threads stride the node's child table (4096 or
// 32768 entries) and merge partial bboxes/statistics through shared memory -
// the former one-thread-per-node kernel serialized up to 32768 child visits
// per thread and left the device nearly empty at typical node counts.
template<typename BuildT, typename StatsT, int LEVEL>
__global__ void processInternal(NodeManager<BuildT> *d_nodeMgr, StatsT *d_stats)
{
    using ChildT = typename NanoNode<BuildT,LEVEL-1>::type;
    using NodeT = typename NanoNode<BuildT,LEVEL>::type;
    constexpr uint32_t Threads = 128;
    __shared__ StatsT sStats[Threads];
    __shared__ CoordBBox sBBox[Threads];
    __shared__ int sSlot;// any one child's d_stats slot, claimed for this node

    const uint32_t nodeID = blockIdx.x;
    const uint32_t tID = threadIdx.x;
    if (nodeID >= d_nodeMgr->nodeCount(LEVEL)) return;
    auto &d_node = d_nodeMgr->template node<LEVEL>(nodeID);
    if (tID == 0) sSlot = -1;
    __syncthreads();

    CoordBBox bbox;// empty
    StatsT stats;
    int mySlot = -1;
    for (uint32_t i = tID; i < NodeT::SIZE; i += Threads) {
        if (d_node.childMask().isOn(i)) {
            auto &child = *d_node.getChild(i);
            bbox.expand( child.bbox() );
            if constexpr(StatsT::hasAverage()) {
                const int slot = *reinterpret_cast<const uint32_t*>(&child.mMinimum);
                StatsT &s = d_stats[slot];
                s.setStats(child);
                stats.add(s);
                mySlot = slot;
            } else if constexpr(StatsT::hasMinMax()) {
                stats.add(child.minimum());
                stats.add(child.maximum());
            }
        } else if (d_node.valueMask().isOn(i)) {
            const Coord ijk = d_node.offsetToGlobalCoord(i);
            bbox[0].minComponent(ijk);
            bbox[1].maxComponent(ijk + Coord(ChildT::DIM - 1));
            if constexpr(StatsT::hasStats()) stats.add(d_node.data()->getValue(i), ChildT::NUM_VALUES);
        }
    }
    if constexpr(StatsT::hasAverage())
        if (mySlot >= 0) atomicMax(&sSlot, mySlot);
    sStats[tID] = stats;
    sBBox[tID] = bbox;
    __syncthreads();
    for (uint32_t d = Threads >> 1; d; d >>= 1) {
        if (tID < d) {
            sStats[tID].add(sStats[tID + d]);
            sBBox[tID].expand(sBBox[tID + d]);
        }
        __syncthreads();
    }
    if (tID == 0) {
        d_node.mBBox = sBBox[0];
        if constexpr(StatsT::hasAverage()) {
            d_stats[sSlot] = sStats[0];
            *reinterpret_cast<uint32_t*>(&d_node.mMinimum) = sSlot;
        } else if constexpr(StatsT::hasMinMax()) {
            sStats[0].setStats(d_node);
        }
        d_node.mFlags &= ~uint64_t(1u);// enable rendering
    }
}// processInternal

template<typename BuildT, typename StatsT>
__global__ void processRootAndGrid(NodeManager<BuildT> *d_nodeMgr, StatsT *d_stats)
{
    using ChildT = NanoUpper<BuildT>;
    using ValueT = typename ChildT::ValueType;

    // process root
    auto &root = d_nodeMgr->root();
    root.mBBox = CoordBBox();
    if (root.isEmpty()) {
        root.mMinimum = root.mMaximum = root.mBackground;
        root.mAverage = root.mStdDevi = 0;
    } else {
        ValueT v;
        StatsT s;
        for (auto it = root.beginDense(); it; ++it) {
            if (auto *child = it.probeChild(v)) {
                root.mBBox.expand( child->bbox() );
                if constexpr(StatsT::hasAverage()) {
                    StatsT &stats = d_stats[*reinterpret_cast<uint32_t*>(&child->mMinimum)];
                    stats.setStats(*child);
                    s.add(stats);
                } else if constexpr(StatsT::hasMinMax()){
                    s.add(child->minimum());
                    s.add(child->maximum());
                }
            } else if (it.isValueOn()) {
                const Coord ijk = it.getCoord();
                root.mBBox[0].minComponent(ijk);
                root.mBBox[1].maxComponent(ijk + Coord(ChildT::DIM - 1));
                if constexpr(StatsT::hasStats()) s.add(v, ChildT::NUM_VALUES);
            }
        }
        s.setStats(root);
    }

    // process Grid
    auto& grid = d_nodeMgr->grid();
    const auto& indexBBox = root.bbox();
    if (indexBBox.empty()) {
        grid.mWorldBBox = Vec3dBBox();
        grid.setBBoxOn(false);
    } else {
        // Note that below max is offset by one since CoordBBox.max is inclusive
        // while bbox<Vec3d>.max is exclusive. However, min is inclusive in both
        // CoordBBox and Vec3dBBox. This also guarantees that a grid with a single
        // active voxel, does not have an empty world bbox! E.g. if a grid with a
        // unit index-to-world transformation only contains the active voxel (0,0,0)
        // then indeBBox = (0,0,0) -> (0,0,0) and then worldBBox = (0.0, 0.0, 0.0)
        // -> (1.0, 1.0, 1.0). This is a consequence of the different definitions
        // of index and world bounding boxes inherited from OpenVDB!
        grid.mWorldBBox = CoordBBox(indexBBox[0], indexBBox[1].offsetBy(1)).transform(grid.map());
        grid.setBBoxOn(true);
    }

    // set bit flags
    grid.setMinMaxOn(StatsT::hasMinMax());
    grid.setAverageOn(StatsT::hasAverage());
    grid.setStdDeviationOn(StatsT::hasStdDeviation());
}// processRootAndGrid

}// cuda kernels are defined in an unnamed namespace

//================================================================================================

template<typename BuildT, typename StatsT>
void GridStats<BuildT, StatsT>::update(NanoGrid<BuildT> *d_grid, cudaStream_t stream)
{
    static const uint32_t threadsPerBlock = 128;
    auto blocksPerGrid = [&](uint32_t count)->uint32_t{return (count + (threadsPerBlock - 1)) / threadsPerBlock;};

    auto nodeMgrHandle = nanovdb::cuda::createNodeManager(d_grid, CudaDeviceBuffer(), stream);
    auto *d_nodeMgr = nodeMgrHandle.template deviceMgr<BuildT>();

    uint32_t nodeCount[3];// {leaf, lower, upper}
    cudaCheck(cudaMemcpyAsync(nodeCount, (char*)d_grid + sizeof(GridData) + 4*sizeof(uint64_t), 3*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    //cudaStreamSynchronize(stream);// finish all device tasks in stream

    StatsT *d_stats = nullptr;

    if constexpr(StatsT::hasAverage()) cudaCheck(util::cuda::mallocAsync((void**)&d_stats, nodeCount[0]*sizeof(StatsT), stream));

    // warp per leaf (4 warps per 128-thread block); block per internal node
    processLeaf<BuildT><<<blocksPerGrid(nodeCount[0]*32), threadsPerBlock, 0, stream>>>(d_nodeMgr, d_stats);

    if (nodeCount[1]) processInternal<BuildT, StatsT, 1><<<nodeCount[1], threadsPerBlock, 0, stream>>>(d_nodeMgr, d_stats);

    if (nodeCount[2]) processInternal<BuildT, StatsT, 2><<<nodeCount[2], threadsPerBlock, 0, stream>>>(d_nodeMgr, d_stats);

    processRootAndGrid<BuildT><<<1, 1, 0, stream>>>(d_nodeMgr, d_stats);

    if constexpr(StatsT::hasAverage()) cudaCheck(util::cuda::freeAsync(d_stats, stream));

} // cuda::GridStats::update( Grid )

//================================================================================================

template<typename BuildT>
void updateGridStats(NanoGrid<BuildT> *d_grid, StatsMode mode, cudaStream_t stream)
{
    if (d_grid == nullptr && mode == StatsMode::Disable) {
        return;
    } else if (mode == StatsMode::BBox || util::is_same<bool, BuildT>::value) {
        GridStats<BuildT, NoopStats<BuildT> > stats;
        stats.update(d_grid, stream);
    } else if (mode == StatsMode::MinMax) {
        GridStats<BuildT, Extrema<BuildT> > stats;
        stats.update(d_grid, stream);
    } else if (mode == StatsMode::All) {
        GridStats<BuildT, Stats<BuildT> > stats;
        stats.update(d_grid, stream);
    } else {
        throw std::runtime_error("GridStats: Unsupported statistics mode.");
    }
}// cuda::updateGridStats

}// namespace tools::cuda

template<typename BuildT>
[[deprecated("Use nanovdb::cuda::updateGridStats instead")]]
void cudaGridStats(NanoGrid<BuildT> *d_grid, tools::StatsMode mode = tools::StatsMode::Default, cudaStream_t stream = 0)
{
    tools::cuda::updateGridStats<BuildT>(d_grid, mode, stream);
}

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_GRIDSTATS_CUH_HAS_BEEN_INCLUDED
