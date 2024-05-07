// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaGridStats.cuh

    \author Ken Museth

    \date October 9, 2023

    \brief Re-computes min/max/avg/var/bbox information for each node in a
           pre-existing NanoVDB grid on the device.
*/

#ifndef NANOVDB_CUDAGRIDSTATS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDAGRIDSTATS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridStats.h>

namespace nanovdb {

/// @brief Re-computes the min/max, stats and bbox information for an existing NanoVDB Grid
///
/// @param grid   Grid whose stats to update
/// @param mode   Mode of computation for the statistics.
/// @param stream Optional cuda stream (defaults to zero)
template<typename BuildT>
void cudaGridStats(NanoGrid<BuildT> *d_grid, StatsMode mode = StatsMode::Default, cudaStream_t stream = 0);

//================================================================================================

/// @brief Allows for the construction of NanoVDB grids without any dependecy
template<typename BuildT, typename StatsT = Stats<typename NanoGrid<BuildT>::ValueType>>
class CudaGridStats
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = typename GridT::TreeType;
    using ValueT = typename TreeT::ValueType;
    using Node0  = typename TreeT::Node0; // leaf
    using Node1  = typename TreeT::Node1; // lower
    using Node2  = typename TreeT::Node2; // upper
    using RootT  = typename TreeT::Node3; // root
    static_assert(is_same<ValueT, typename StatsT::ValueType>::value, "Mismatching type");

    ValueT mDelta; // skip rendering of node if: node.max < -mDelta || node.min > mDelta

public:
    CudaGridStats(ValueT delta = ValueT(0)) : mDelta(delta) {}

    void operator()(GridT *d_grid, cudaStream_t stream = 0);

}; // CudaGridStats

//================================================================================================

namespace {// define cuda kernels in an unnamed namespace

template<typename BuildT, typename StatsT>
__global__ void processLeaf(NodeManager<BuildT> *d_nodeMgr, StatsT *d_stats)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_nodeMgr->leafCount()) return;
    auto &d_leaf = d_nodeMgr->leaf(tid);

    if (d_leaf.updateBBox()) {// updates active bounding box (also updates data->mFlags) and return true if non-empty
        if constexpr(StatsT::hasStats()) {
            StatsT stats;
            for (auto it = d_leaf.cbeginValueOn(); it; ++it) stats.add(*it);
            if constexpr(StatsT::hasAverage()) {
                d_stats[tid] = stats;
                *reinterpret_cast<uint32_t*>(&d_leaf.mMinimum) = tid;
            } else {
                stats.setStats(d_leaf);
            }
        }
    }
    d_leaf.mFlags &= ~uint8_t(1u);// enable rendering
}// processLeaf

template<typename BuildT, typename StatsT, int LEVEL>
__global__ void processInternal(NodeManager<BuildT> *d_nodeMgr, StatsT *d_stats)
{
    using ChildT = typename NanoNode<BuildT,LEVEL-1>::type;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_nodeMgr->nodeCount(LEVEL)) return;
    auto &d_node = d_nodeMgr->template node<LEVEL>(tid);
    auto &bbox   = d_node.mBBox;
    bbox         = CoordBBox();// empty bbox
    StatsT stats;
    uint32_t childID = 0u;

    for (auto it = d_node.beginChild(); it; ++it) {
        auto &child = *it;
        bbox.expand( child.bbox() );
        if constexpr(StatsT::hasAverage()) {
            childID = *reinterpret_cast<uint32_t*>(&child.mMinimum);
            StatsT &s = d_stats[childID];
            s.setStats(child);
            stats.add(s);
        } else if constexpr(StatsT::hasMinMax()) {
            stats.add(child.minimum());
            stats.add(child.maximum());
        }
    }
    for (auto it = d_node.cbeginValueOn(); it; ++it) {
        const Coord ijk = it.getCoord();
        bbox[0].minComponent(ijk);
        bbox[1].maxComponent(ijk + Coord(ChildT::DIM - 1));
        if constexpr(StatsT::hasStats()) stats.add(*it, ChildT::NUM_VALUES);
    }
    if constexpr(StatsT::hasAverage()) {
        d_stats[childID] = stats;
        *reinterpret_cast<uint32_t*>(&d_node.mMinimum) = childID;
    } else if constexpr(StatsT::hasMinMax()) {
        stats.setStats(d_node);
    }
    d_node.mFlags &= ~uint64_t(1u);// enable rendering
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
        grid.mWorldBBox = BBox<Vec3d>();
        grid.setBBoxOn(false);
    } else {
        // Note that below max is offset by one since CoordBBox.max is inclusive
        // while bbox<Vec3d>.max is exclusive. However, min is inclusive in both
        // CoordBBox and BBox<Vec3d>. This also guarantees that a grid with a single
        // active voxel, does not have an empty world bbox! E.g. if a grid with a
        // unit index-to-world transformation only contains the active voxel (0,0,0)
        // then indeBBox = (0,0,0) -> (0,0,0) and then worldBBox = (0.0, 0.0, 0.0)
        // -> (1.0, 1.0, 1.0). This is a consequence of the different definitions
        // of index and world bounding boxes inherited from OpenVDB!
        const Coord min = indexBBox[0];
        const Coord max = indexBBox[1] + Coord(1);

        auto& wBBox = grid.mWorldBBox;
        const auto& map = grid.map();
        wBBox[0] = wBBox[1] = map.applyMap(Vec3d(min[0], min[1], min[2]));
        wBBox.expand(map.applyMap(Vec3d(min[0], min[1], max[2])));
        wBBox.expand(map.applyMap(Vec3d(min[0], max[1], min[2])));
        wBBox.expand(map.applyMap(Vec3d(max[0], min[1], min[2])));
        wBBox.expand(map.applyMap(Vec3d(max[0], max[1], min[2])));
        wBBox.expand(map.applyMap(Vec3d(max[0], min[1], max[2])));
        wBBox.expand(map.applyMap(Vec3d(min[0], max[1], max[2])));
        wBBox.expand(map.applyMap(Vec3d(max[0], max[1], max[2])));
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
void CudaGridStats<BuildT, StatsT>::operator()(NanoGrid<BuildT> *d_grid, cudaStream_t stream)
{
    static const uint32_t threadsPerBlock = 128;
    auto blocksPerGrid = [&](uint32_t count)->uint32_t{return (count + (threadsPerBlock - 1)) / threadsPerBlock;};

    auto nodeMgrHandle = cudaCreateNodeManager(d_grid, CudaDeviceBuffer(), stream);
    auto *d_nodeMgr = nodeMgrHandle.template deviceMgr<BuildT>();

    uint32_t nodeCount[3];// {leaf, lower, upper}
    cudaCheck(cudaMemcpyAsync(nodeCount, (char*)d_grid + sizeof(GridData) + 4*sizeof(uint64_t), 3*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);// finish all device tasks in stream

    StatsT *d_stats = nullptr;

    if constexpr(StatsT::hasAverage()) cudaCheck(cudaMallocAsync((void**)&d_stats, nodeCount[0]*sizeof(StatsT), stream));

    processLeaf<BuildT><<<blocksPerGrid(nodeCount[0]), threadsPerBlock, 0, stream>>>(d_nodeMgr, d_stats);

    processInternal<BuildT, StatsT, 1><<<blocksPerGrid(nodeCount[1]), threadsPerBlock, 0, stream>>>(d_nodeMgr, d_stats);

    processInternal<BuildT, StatsT, 2><<<blocksPerGrid(nodeCount[2]), threadsPerBlock, 0, stream>>>(d_nodeMgr, d_stats);

    processRootAndGrid<BuildT><<<1, 1, 0, stream>>>(d_nodeMgr, d_stats);

    if constexpr(StatsT::hasAverage()) cudaCheck(cudaFreeAsync(d_stats, stream));

} // CudaGridStats::operator()( Grid )

//================================================================================================

template<typename BuildT>
void cudaGridStats(NanoGrid<BuildT> *d_grid, StatsMode mode, cudaStream_t stream)
{
    if (d_grid == nullptr && mode == StatsMode::Disable) {
        return;
    } else if (mode == StatsMode::BBox || is_same<bool, BuildT>::value) {
        CudaGridStats<BuildT, NoopStats<BuildT> > stats;
        stats(d_grid, stream);
    } else if (mode == StatsMode::MinMax) {
        CudaGridStats<BuildT, Extrema<BuildT> > stats;
        stats(d_grid, stream);
    } else if (mode == StatsMode::All) {
        CudaGridStats<BuildT, Stats<BuildT> > stats;
        stats(d_grid, stream);
    } else {
        throw std::runtime_error("cudaGridStats: Unsupported statistics mode.");
    }
}// cudaGridStats

} // namespace nanovdb

#endif // NANOVDB_CUDAGRIDSTATS_CUH_HAS_BEEN_INCLUDED
