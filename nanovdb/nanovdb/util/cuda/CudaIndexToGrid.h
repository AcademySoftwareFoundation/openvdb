// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaIndexToGrid.h

    \author Ken Museth

    \date April 17, 2023

    \brief Combines an IndexGrid and values into a regular Grid on the device
*/

#ifndef NVIDIA_CUDA_INDEX_TO_GRID_H_HAS_BEEN_INCLUDED
#define NVIDIA_CUDA_INDEX_TO_GRID_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include "CudaDeviceBuffer.h"
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/cuda/GpuTimer.h>
#include <nanovdb/util/cuda/CudaUtils.h>

namespace nanovdb {

// cudeIndexGridToGrid

/// @brief Freestanding function that combines an IndexGrid and values into a regular Grid
/// @tparam DstBuildT Build time of the destination/output Grid
/// @tparam SrcBuildT  Build type of the source/input IndexGrid
/// @tparam BufferT Type of the buffer used for allocation of the destination Grid
/// @param d_srcGrid Device pointer to source/input IndexGrid, i.e. SrcBuildT={ValueIndex,ValueOnIndex,ValueIndexMask,ValueOnIndexMask}
/// @param d_srcValues Device pointer to an array of values
/// @param pool Memory pool used to create a buffer for the destination/output Grid
/// @note If d_srcGrid has stats (min,max,avg,std-div), the d_srcValues is also assumed
///       to have the same information, all of which are then copied to the destination/output grid.
///       An exception to this rule is if the type of d_srcValues is different from the stats type
///       NanoRoot<DstBuildT>::FloatType, e.g. if DstBuildT=Vec3f then NanoRoot<DstBuildT>::FloatType=float,
///       in which case average and standard-deviation is undefined in the output grid.
/// @return
template<typename DstBuildT, typename SrcBuildT, typename BufferT = CudaDeviceBuffer>
typename enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
cudaIndexToGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool = BufferT());


template<typename DstBuildT, typename SrcBuildT, typename BufferT = CudaDeviceBuffer>
typename enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
cudaCreateNanoGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool = BufferT())
{
    return cudaIndexToGrid<DstBuildT, SrcBuildT, BufferT>(d_srcGrid, d_srcValues, pool);
}

namespace {// anonymous namespace

template<typename SrcBuildT>
class CudaIndexToGrid
{
    using SrcGridT = NanoGrid<SrcBuildT>;
public:
    struct NodeAccessor;

    /// @brief Constructor from a source IndeGrid
    /// @param srcGrid Device pointer to IndexGrid used as the source
    CudaIndexToGrid(const SrcGridT *d_srcGrid);

    ~CudaIndexToGrid() {cudaCheck(cudaFree(mDevNodeAcc));}

    /// @brief Toggle on and off verbose mode
    /// @param on if true verbose is turned on
    void setVerbose(bool on = true) {mVerbose = on; }

    /// @brief Set the name of the destination/output grid
    /// @param name Name used for the destination grid
    void setGridName(const std::string &name) {mGridName = name;}

    template<typename DstBuildT, typename BufferT = CudaDeviceBuffer>
    GridHandle<BufferT> getHandle(const typename BuildToValueMap<DstBuildT>::type *srcValues, const BufferT &buffer = BufferT());

private:
    GpuTimer mTimer;
    std::string mGridName;
    bool mVerbose{true};
    NodeAccessor mNodeAcc, *mDevNodeAcc;

    template<typename DstBuildT, typename BufferT>
    BufferT getBuffer(const BufferT &pool);
};// CudaIndexToGrid

//================================================================================================

template<typename SrcBuildT>
struct CudaIndexToGrid<SrcBuildT>::NodeAccessor
{
    uint64_t grid, tree, root, node[3], meta, blind, size;// byte offsets, node: 0=leaf,1=lower, 2=upper
    const SrcGridT *d_srcGrid;// device point to source IndexGrid
    void *d_dstPtr;// device pointer to buffer with destination Grid
    char *d_gridName;
    uint32_t nodeCount[4];// 0=leaf, 1=lower, 2=upper, 3=root tiles

    __device__ const NanoGrid<SrcBuildT>& srcGrid() const {return *d_srcGrid;}
    __device__ const NanoTree<SrcBuildT>& srcTree() const {return d_srcGrid->tree();}
    __device__ const NanoRoot<SrcBuildT>& srcRoot() const {return d_srcGrid->tree().root();}
    template <int LEVEL>
    __device__ const typename NanoNode<SrcBuildT, LEVEL>::type& srcNode(int i) const {
        return *(this->srcTree().template getFirstNode<LEVEL>() + i);
    }

    template <typename DstBuildT>
    __device__ NanoGrid<DstBuildT>& dstGrid() const {return *PtrAdd<NanoGrid<DstBuildT>>(d_dstPtr, grid);}
    template <typename DstBuildT>
    __device__ NanoTree<DstBuildT>& dstTree() const {return *PtrAdd<NanoTree<DstBuildT>>(d_dstPtr, tree);}
    template <typename DstBuildT>
    __device__ NanoRoot<DstBuildT>& dstRoot() const {return *PtrAdd<NanoRoot<DstBuildT>>(d_dstPtr, root);}
    template <typename DstBuildT, int LEVEL>
    __device__ typename NanoNode<DstBuildT, LEVEL>::type& dstNode(int i) const {
        return *(PtrAdd<typename NanoNode<DstBuildT,LEVEL>::type>(d_dstPtr, node[LEVEL])+i);
    }
};// CudaIndexToGrid<SrcBuildT>::NodeAccessor

//================================================================================================

template<typename SrcBuildT, typename DstBuildT>
__global__ void cudaProcessGridTreeRoot(typename CudaIndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
                                        const typename BuildToValueMap<DstBuildT>::type *srcValues)
{
    using SrcValueT = typename BuildToValueMap<DstBuildT>::type;
    using DstStatsT = typename NanoRoot<DstBuildT>::FloatType;

    auto &srcGrid = nodeAcc->srcGrid();
    auto &dstGrid = nodeAcc->template dstGrid<DstBuildT>();
    auto &srcTree = srcGrid.tree();
    auto &dstTree = nodeAcc->template dstTree<DstBuildT>();
    auto &srcRoot = srcTree.root();
    auto &dstRoot = nodeAcc->template dstRoot<DstBuildT>();

    // process Grid
    *dstGrid.data() = *srcGrid.data();
    dstGrid.mGridType = mapToGridType<DstBuildT>();
    dstGrid.mData1 = 0u;

    // process Tree
    *dstTree.data() = *srcTree.data();
    dstTree.setRoot(&dstRoot);
    dstTree.setFirstNode(&nodeAcc->template dstNode<DstBuildT,2>(0));
    dstTree.setFirstNode(&nodeAcc->template dstNode<DstBuildT,1>(0));
    dstTree.setFirstNode(&nodeAcc->template dstNode<DstBuildT,0>(0));

    // process Root
    dstRoot.mBBox = srcRoot.mBBox;
    dstRoot.mTableSize = srcRoot.mTableSize;
    dstRoot.mBackground = srcValues[srcRoot.mBackground];
    if (srcGrid.hasMinMax()) {
        dstRoot.mMinimum = srcValues[srcRoot.mMinimum];
        dstRoot.mMaximum = srcValues[srcRoot.mMaximum];
    }
    if constexpr(is_same<SrcValueT, DstStatsT>::value) {// e.g. {float,float} or {Vec3f,float}
        if (srcGrid.hasAverage())      dstRoot.mAverage = srcValues[srcRoot.mAverage];
        if (srcGrid.hasStdDeviation()) dstRoot.mStdDevi = srcValues[srcRoot.mStdDevi];
    }
}// cudaProcessGridTreeRoot

//================================================================================================

template<typename SrcBuildT, typename DstBuildT>
__global__ void cudaProcessRootTiles(typename CudaIndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
                                     const typename BuildToValueMap<DstBuildT>::type *srcValues)
{
    const auto tid = blockIdx.x;

    // Process children and tiles
    const auto &srcTile = *nodeAcc->srcRoot().tile(tid);
    auto &dstTile = *nodeAcc->template dstRoot<DstBuildT>().tile(tid);
    dstTile.key   = srcTile.key;
    if (srcTile.child) {
        dstTile.child = sizeof(NanoRoot<DstBuildT>) + sizeof(NanoRoot<DstBuildT>::Tile)*((srcTile.child - sizeof(NanoRoot<SrcBuildT>))/sizeof(NanoRoot<SrcBuildT>::Tile));
        dstTile.value = srcValues[0];// set to background
        dstTile.state = false;
    } else {
        dstTile.child = 0;// i.e. no child node
        dstTile.value = srcValues[srcTile.value];
        dstTile.state = srcTile.state;
    }
}// cudaProcessRootTiles

//================================================================================================

template<typename SrcBuildT, typename DstBuildT, int LEVEL>
__global__ void cudaProcessInternalNodes(typename CudaIndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
                                         const typename BuildToValueMap<DstBuildT>::type *srcValues)
{
    using SrcNodeT  = typename NanoNode<SrcBuildT, LEVEL>::type;
    using DstNodeT  = typename NanoNode<DstBuildT, LEVEL>::type;
    using SrcChildT = typename SrcNodeT::ChildNodeType;
    using DstChildT = typename DstNodeT::ChildNodeType;
    using SrcValueT = typename BuildToValueMap<DstBuildT>::type;
    using DstStatsT = typename NanoRoot<DstBuildT>::FloatType;

    auto &srcNode = nodeAcc->template srcNode<LEVEL>(blockIdx.x);
    auto &dstNode = nodeAcc->template dstNode<DstBuildT, LEVEL>(blockIdx.x);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        dstNode.mBBox = srcNode.mBBox;
        dstNode.mFlags = srcNode.mFlags;
        dstNode.mValueMask = srcNode.mValueMask;
        dstNode.mChildMask = srcNode.mChildMask;
        auto &srcGrid = nodeAcc->srcGrid();
        if (srcGrid.hasMinMax()) {
            dstNode.mMinimum = srcValues[srcNode.mMinimum];
            dstNode.mMaximum = srcValues[srcNode.mMaximum];
        }
        if constexpr(is_same<SrcValueT, DstStatsT>::value) {// e.g. {float,float} or {Vec3f,float}
            if (srcGrid.hasAverage())      dstNode.mAverage = srcValues[srcNode.mAverage];
            if (srcGrid.hasStdDeviation()) dstNode.mStdDevi = srcValues[srcNode.mStdDevi];
        }
    }
    const uint64_t nodeSkip = nodeAcc->nodeCount[LEVEL] - blockIdx.x, srcOff = sizeof(SrcNodeT)*nodeSkip, dstOff = sizeof(DstNodeT)*nodeSkip;// offset to first node of child type
    const int off = blockDim.x*blockDim.y*threadIdx.x + blockDim.x*threadIdx.y;
    for (int threadIdx_z=0; threadIdx_z<blockDim.x; ++threadIdx_z) {
        const int i = off + threadIdx_z;
        if (srcNode.mChildMask.isOn(i)) {
            if constexpr(sizeof(SrcNodeT)==sizeof(DstNodeT) && sizeof(SrcChildT)==sizeof(DstChildT)) {
                dstNode.mTable[i].child = srcNode.mTable[i].child;
            } else {
                const uint64_t childID = (srcNode.mTable[i].child - srcOff)/sizeof(SrcChildT);
                dstNode.mTable[i].child = dstOff + childID*sizeof(DstChildT);
            }
        } else {
            dstNode.mTable[i].value = srcValues[srcNode.mTable[i].value];
        }
    }
}// cudaProcessInternalNodes

//================================================================================================

template<typename SrcBuildT, typename DstBuildT>
__global__ void cudaProcessLeafNodes(typename CudaIndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
                                     const typename BuildToValueMap<DstBuildT>::type *srcValues)
{
    using SrcValueT = typename BuildToValueMap<DstBuildT>::type;
    using DstStatsT = typename NanoRoot<DstBuildT>::FloatType;
    static_assert(!BuildTraits<DstBuildT>::is_special, "Invalid destination type!");
    auto &srcLeaf = nodeAcc->template srcNode<0>(blockIdx.x);
    auto &dstLeaf = nodeAcc->template dstNode<DstBuildT,0>(blockIdx.x);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        dstLeaf.mBBoxMin = srcLeaf.mBBoxMin;
        for (int i=0; i<3; ++i) dstLeaf.mBBoxDif[i] = srcLeaf.mBBoxDif[i];
        dstLeaf.mFlags = srcLeaf.mFlags;
        dstLeaf.mValueMask = srcLeaf.mValueMask;
        ///
        auto &srcGrid = nodeAcc->srcGrid();
        if (srcGrid.hasMinMax()) {
            dstLeaf.mMinimum = srcValues[srcLeaf.getMin()];
            dstLeaf.mMaximum = srcValues[srcLeaf.getMax()];
        }
        if constexpr(is_same<SrcValueT, DstStatsT>::value) {// e.g. {float,float} or {Vec3f,float}
            if (srcGrid.hasAverage())      dstLeaf.mAverage = srcValues[srcLeaf.getAvg()];
            if (srcGrid.hasStdDeviation()) dstLeaf.mStdDevi = srcValues[srcLeaf.getDev()];
        }
    }
    const int off = blockDim.x*blockDim.y*threadIdx.x + blockDim.x*threadIdx.y;
    auto *dst = dstLeaf.mValues + off;
    for (int threadIdx_z=0; threadIdx_z<blockDim.x; ++threadIdx_z) {
        const int i = off + threadIdx_z;
        *dst++ = srcValues[srcLeaf.getValue(i)];
    }
}// cudaProcessLeafNodes

//================================================================================================

template <typename SrcBuildT>
__global__ void cudaCpyNodeCount(const NanoGrid<SrcBuildT> *srcGrid,
                                 typename CudaIndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc)
{
    assert(srcGrid->isSequential());
    nodeAcc->d_srcGrid = srcGrid;
    for (int i=0; i<3; ++i) nodeAcc->nodeCount[i] = srcGrid->tree().nodeCount(i);
    nodeAcc->nodeCount[3] = srcGrid->tree().root().tileCount();
}

}// anonymous namespace

//================================================================================================

template <typename SrcBuildT>
CudaIndexToGrid<SrcBuildT>::CudaIndexToGrid(const SrcGridT *d_srcGrid)
{
    NANOVDB_ASSERT(d_srcGrid);
    cudaCheck(cudaMalloc((void**)&mDevNodeAcc, sizeof(NodeAccessor)));
    cudaCpyNodeCount<SrcBuildT><<<1,1>>>(d_srcGrid, mDevNodeAcc);
    cudaCheckError();
    cudaCheck(cudaMemcpy(&mNodeAcc, mDevNodeAcc, sizeof(NodeAccessor), cudaMemcpyDeviceToHost));// mNodeAcc = *mDevNodeAcc
}

//================================================================================================

template <typename SrcBuildT>
template <typename DstBuildT, typename BufferT>
GridHandle<BufferT> CudaIndexToGrid<SrcBuildT>::getHandle(const typename BuildToValueMap<DstBuildT>::type *srcValues,
                                                              const BufferT &pool)
{
    if (mVerbose) mTimer.start("Initiate buffer");
    auto buffer = this->template getBuffer<DstBuildT, BufferT>(pool);

    if (mVerbose) mTimer.restart("Process grid,tree,root");
    cudaProcessGridTreeRoot<SrcBuildT,DstBuildT><<<1, 1>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process root children and tiles");
    cudaProcessRootTiles<SrcBuildT,DstBuildT><<<mNodeAcc.nodeCount[3], 1>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    cudaCheck(cudaFree(mNodeAcc.d_gridName));

    if (mVerbose) mTimer.restart("Process upper internal nodes");
    cudaProcessInternalNodes<SrcBuildT,DstBuildT,2><<<mNodeAcc.nodeCount[2], dim3(32,32)>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process lower internal nodes");
    cudaProcessInternalNodes<SrcBuildT,DstBuildT,1><<<mNodeAcc.nodeCount[1], dim3(16,16)>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process leaf nodes");
    cudaProcessLeafNodes<SrcBuildT,DstBuildT><<<mNodeAcc.nodeCount[0], dim3(8,8)>>>(mDevNodeAcc, srcValues);
    if (mVerbose) mTimer.stop();
    cudaCheckError();

    return GridHandle<BufferT>(std::move(buffer));
}// CudaIndexToGrid::getHandle

//================================================================================================

template <typename SrcBuildT>
template <typename DstBuildT, typename BufferT>
inline BufferT CudaIndexToGrid<SrcBuildT>::getBuffer(const BufferT &pool)
{
    mNodeAcc.grid  = 0;// grid is always stored at the start of the buffer!
    mNodeAcc.tree  = NanoGrid<DstBuildT>::memUsage(); // grid ends and tree begins
    mNodeAcc.root  = mNodeAcc.tree  + NanoTree<DstBuildT>::memUsage(); // tree ends and root node begins
    mNodeAcc.node[2] = mNodeAcc.root  + NanoRoot<DstBuildT>::memUsage(mNodeAcc.nodeCount[3]); // root node ends and upper internal nodes begin
    mNodeAcc.node[1] = mNodeAcc.node[2] + NanoUpper<DstBuildT>::memUsage()*mNodeAcc.nodeCount[2]; // upper internal nodes ends and lower internal nodes begin
    mNodeAcc.node[0] = mNodeAcc.node[1] + NanoLower<DstBuildT>::memUsage()*mNodeAcc.nodeCount[1]; // lower internal nodes ends and leaf nodes begin
    mNodeAcc.meta  = mNodeAcc.node[0]  + NanoLeaf<DstBuildT>::DataType::memUsage()*mNodeAcc.nodeCount[0];// leaf nodes end and blind meta data begins
    mNodeAcc.blind = mNodeAcc.meta  + 0*sizeof(GridBlindMetaData); // meta data ends and blind data begins
    mNodeAcc.size  = mNodeAcc.blind;// end of buffer
    auto buffer = BufferT::create(mNodeAcc.size, &pool, false);
    mNodeAcc.d_dstPtr = buffer.deviceData();
    if (mNodeAcc.d_dstPtr == nullptr) throw std::runtime_error("Failed memory allocation on the device");

    if (size_t size = mGridName.size()) {
        cudaCheck(cudaMalloc((void**)&mNodeAcc.d_gridName, size));
        cudaCheck(cudaMemcpy(mNodeAcc.d_gridName, mGridName.data(), size, cudaMemcpyHostToDevice));
    } else {
        mNodeAcc.d_gridName = nullptr;
    }
    cudaCheck(cudaMemcpy(mDevNodeAcc, &mNodeAcc, sizeof(NodeAccessor), cudaMemcpyHostToDevice));// copy NodeAccessor CPU -> GPU
    return buffer;
}

//================================================================================================

template<typename DstBuildT, typename SrcBuildT, typename BufferT>
typename enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
cudaIndexToGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool)
{
    CudaIndexToGrid<SrcBuildT> converter(d_srcGrid);
    return converter.template getHandle<DstBuildT, BufferT>(d_srcValues, pool);
}

}// nanovdb namespace

#endif // NVIDIA_CUDA_INDEX_TO_GRID_H_HAS_BEEN_INCLUDED
