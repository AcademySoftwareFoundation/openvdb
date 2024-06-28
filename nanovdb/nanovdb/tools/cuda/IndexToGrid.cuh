// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/tools/cuda/IndexToGrid.cuh

    \author Ken Museth

    \date April 17, 2023

    \brief Combines an IndexGrid and values into a regular Grid on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_INDEXTOGRID_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_INDEXTOGRID_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

namespace nanovdb {// ================================================================

namespace tools::cuda {// ============================================================

/// @brief Freestanding function that combines an IndexGrid and values into a regular Grid
/// @tparam DstBuildT Build time of the destination/output Grid
/// @tparam SrcBuildT  Build type of the source/input IndexGrid
/// @tparam BufferT Type of the buffer used for allocation of the destination Grid
/// @param d_srcGrid Device pointer to source/input IndexGrid, i.e. SrcBuildT={ValueIndex,ValueOnIndex,ValueIndexMask,ValueOnIndexMask}
/// @param d_srcValues Device pointer to an array of values
/// @param pool Memory pool used to create a buffer for the destination/output Grid
/// @param stream optional CUDA stream (defaults to CUDA stream 0
/// @note If d_srcGrid has stats (min,max,avg,std-div), the d_srcValues is also assumed
///       to have the same information, all of which are then copied to the destination/output grid.
///       An exception to this rule is if the type of d_srcValues is different from the stats type
///       NanoRoot<DstBuildT>::FloatType, e.g. if DstBuildT=Vec3f then NanoRoot<DstBuildT>::FloatType=float,
///       in which case average and standard-deviation is undefined in the output grid.
/// @return returns handle to grid that combined IndexGrid and values
template<typename DstBuildT, typename SrcBuildT, typename BufferT = nanovdb::cuda::DeviceBuffer>
typename util::enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
indexToGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool = BufferT(), cudaStream_t stream = 0);


template<typename DstBuildT, typename SrcBuildT, typename BufferT = nanovdb::cuda::DeviceBuffer>
typename util::enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
createNanoGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool = BufferT(), cudaStream_t stream = 0)
{
    return indexToGrid<DstBuildT, SrcBuildT, BufferT>(d_srcGrid, d_srcValues, pool, stream);
}

namespace {// anonymous namespace

template<typename SrcBuildT>
class IndexToGrid
{
    using SrcGridT = NanoGrid<SrcBuildT>;
public:
    struct NodeAccessor;

    /// @brief Constructor from a source IndeGrid
    /// @param srcGrid Device pointer to IndexGrid used as the source
    IndexToGrid(const SrcGridT *d_srcGrid, cudaStream_t stream = 0);

    ~IndexToGrid() {cudaCheck(util::cuda::freeAsync(mDevNodeAcc, mStream));}

    /// @brief Toggle on and off verbose mode
    /// @param on if true verbose is turned on
    void setVerbose(bool on = true) {mVerbose = on; }

    /// @brief Set the name of the destination/output grid
    /// @param name Name used for the destination grid
    void setGridName(const std::string &name) {mGridName = name;}

    /// @brief Combines the IndexGrid with values to produce a regular Grid
    /// @tparam DstBuildT Template parameter of the destination grid and value type
    /// @tparam BufferT Template parameter of the memory allocator
    /// @param srcValues pointer to values that will be inserted into the output grid
    /// @param buffer optional buffer used for memory allocation
    /// @return A new GridHandle with the grid of type @c DstBuildT
    template<typename DstBuildT, typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT> getHandle(const typename BuildToValueMap<DstBuildT>::type *srcValues, const BufferT &buffer = BufferT());

private:
    cudaStream_t      mStream{0};
    util::cuda::Timer mTimer;
    std::string       mGridName;
    bool              mVerbose{false};
    NodeAccessor      mNodeAcc, *mDevNodeAcc;

    template<typename DstBuildT, typename BufferT>
    BufferT getBuffer(const BufferT &pool);
};// IndexToGrid

//================================================================================================

template<typename SrcBuildT>
struct IndexToGrid<SrcBuildT>::NodeAccessor
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
    __device__ NanoGrid<DstBuildT>& dstGrid() const {return *util::PtrAdd<NanoGrid<DstBuildT>>(d_dstPtr, grid);}
    template <typename DstBuildT>
    __device__ NanoTree<DstBuildT>& dstTree() const {return *util::PtrAdd<NanoTree<DstBuildT>>(d_dstPtr, tree);}
    template <typename DstBuildT>
    __device__ NanoRoot<DstBuildT>& dstRoot() const {return *util::PtrAdd<NanoRoot<DstBuildT>>(d_dstPtr, root);}
    template <typename DstBuildT, int LEVEL>
    __device__ typename NanoNode<DstBuildT, LEVEL>::type& dstNode(int i) const {
        return *(util::PtrAdd<typename NanoNode<DstBuildT,LEVEL>::type>(d_dstPtr, node[LEVEL])+i);
    }
};// IndexToGrid<SrcBuildT>::NodeAccessor

//================================================================================================

template<typename SrcBuildT, typename DstBuildT>
__global__ void processGridTreeRootKernel(typename IndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
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
    dstGrid.mGridType = toGridType<DstBuildT>();
    dstGrid.mData1 = 0u;
    // we will recompute GridData::mChecksum later

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
    if constexpr(util::is_same<SrcValueT, DstStatsT>::value) {// e.g. {float,float} or {Vec3f,float}
        if (srcGrid.hasAverage())      dstRoot.mAverage = srcValues[srcRoot.mAverage];
        if (srcGrid.hasStdDeviation()) dstRoot.mStdDevi = srcValues[srcRoot.mStdDevi];
    }
}// processGridTreeRootKernel

//================================================================================================

template<typename SrcBuildT, typename DstBuildT>
__global__ void processRootTilesKernel(typename IndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
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
}// processRootTilesKernel

//================================================================================================

template<typename SrcBuildT, typename DstBuildT, int LEVEL>
__global__ void processNodesKernel(typename IndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
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
        if constexpr(util::is_same<SrcValueT, DstStatsT>::value) {// e.g. {float,float} or {Vec3f,float}
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
}// processNodesKernel

//================================================================================================

template<typename SrcBuildT, typename DstBuildT>
__global__ void processLeafsKernel(typename IndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc,
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
        if constexpr(util::is_same<SrcValueT, DstStatsT>::value) {// e.g. {float,float} or {Vec3f,float}
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
}// processLeafsKernel

//================================================================================================

template <typename SrcBuildT>
__global__ void cpyNodeCountKernel(const NanoGrid<SrcBuildT> *srcGrid,
                                   typename IndexToGrid<SrcBuildT>::NodeAccessor *nodeAcc)
{
    assert(srcGrid->isSequential());
    nodeAcc->d_srcGrid = srcGrid;
    for (int i=0; i<3; ++i) nodeAcc->nodeCount[i] = srcGrid->tree().nodeCount(i);
    nodeAcc->nodeCount[3] = srcGrid->tree().root().tileCount();
}

}// anonymous namespace

//================================================================================================

template <typename SrcBuildT>
IndexToGrid<SrcBuildT>::IndexToGrid(const SrcGridT *d_srcGrid, cudaStream_t stream)
    : mStream(stream), mTimer(stream)
{
    NANOVDB_ASSERT(d_srcGrid);
    cudaCheck(util::cuda::mallocAsync((void**)&mDevNodeAcc, sizeof(NodeAccessor), mStream));
    cpyNodeCountKernel<SrcBuildT><<<1, 1, 0, mStream>>>(d_srcGrid, mDevNodeAcc);
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&mNodeAcc, mDevNodeAcc, sizeof(NodeAccessor), cudaMemcpyDeviceToHost, mStream));// mNodeAcc = *mDevNodeAcc
}

//================================================================================================

template <typename SrcBuildT>
template <typename DstBuildT, typename BufferT>
GridHandle<BufferT> IndexToGrid<SrcBuildT>::getHandle(const typename BuildToValueMap<DstBuildT>::type *srcValues,
                                                          const BufferT &pool)
{
    if (mVerbose) mTimer.start("Initiate buffer");
    auto buffer = this->template getBuffer<DstBuildT, BufferT>(pool);

    if (mVerbose) mTimer.restart("Process grid,tree,root");
    processGridTreeRootKernel<SrcBuildT,DstBuildT><<<1, 1, 0, mStream>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process root children and tiles");
    processRootTilesKernel<SrcBuildT,DstBuildT><<<mNodeAcc.nodeCount[3], 1, 0, mStream>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    cudaCheck(util::cuda::freeAsync(mNodeAcc.d_gridName, mStream));

    if (mVerbose) mTimer.restart("Process upper internal nodes");
    processNodesKernel<SrcBuildT,DstBuildT,2><<<mNodeAcc.nodeCount[2], dim3(32,32), 0, mStream>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process lower internal nodes");
    processNodesKernel<SrcBuildT,DstBuildT,1><<<mNodeAcc.nodeCount[1], dim3(16,16), 0, mStream>>>(mDevNodeAcc, srcValues);
    cudaCheckError();

    if (mVerbose) mTimer.restart("Process leaf nodes");
    processLeafsKernel<SrcBuildT,DstBuildT><<<mNodeAcc.nodeCount[0], dim3(8,8), 0, mStream>>>(mDevNodeAcc, srcValues);
    if (mVerbose) mTimer.stop();
    cudaCheckError();

    if (mVerbose) mTimer.restart("Compute checksums");
    updateChecksum((GridData*)mNodeAcc.d_dstPtr, mStream);
    if (mVerbose) mTimer.stop();

    //cudaStreamSynchronize(mStream);// finish all device tasks in mStream
    return GridHandle<BufferT>(std::move(buffer));
}// IndexToGrid::getHandle

//================================================================================================

template <typename SrcBuildT>
template <typename DstBuildT, typename BufferT>
inline BufferT IndexToGrid<SrcBuildT>::getBuffer(const BufferT &pool)
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
    auto buffer = BufferT::create(mNodeAcc.size, &pool, false, mStream);
    mNodeAcc.d_dstPtr = buffer.deviceData();
    if (mNodeAcc.d_dstPtr == nullptr) throw std::runtime_error("Failed memory allocation on the device");

    if (size_t size = mGridName.size()) {
        cudaCheck(util::cuda::mallocAsync((void**)&mNodeAcc.d_gridName, size, mStream));
        cudaCheck(cudaMemcpyAsync(mNodeAcc.d_gridName, mGridName.data(), size, cudaMemcpyHostToDevice, mStream));
    } else {
        mNodeAcc.d_gridName = nullptr;
    }
    cudaCheck(cudaMemcpyAsync(mDevNodeAcc, &mNodeAcc, sizeof(NodeAccessor), cudaMemcpyHostToDevice, mStream));// copy NodeAccessor CPU -> GPU
    return buffer;
}

//================================================================================================

template<typename DstBuildT, typename SrcBuildT, typename BufferT>
typename util::enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
indexToGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool, cudaStream_t stream)
{
    IndexToGrid<SrcBuildT> converter(d_srcGrid, stream);
    return converter.template getHandle<DstBuildT>(d_srcValues, pool);
}

}// namespace tools::cuda  =============================================================

template<typename DstBuildT, typename SrcBuildT, typename BufferT = cuda::DeviceBuffer>
[[deprecated("Use nanovdb::cuda::indexToGrid instead")]]
typename util::enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
cudaIndexToGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool = BufferT(), cudaStream_t stream = 0)
{
    return tools::cuda::indexToGrid<DstBuildT, SrcBuildT, BufferT>(d_srcGrid, d_srcValues, pool, stream);
}


template<typename DstBuildT, typename SrcBuildT, typename BufferT = cuda::DeviceBuffer>
[[deprecated("Use nanovdb::cuda::indexToGrid instead")]]
typename util::enable_if<BuildTraits<SrcBuildT>::is_index, GridHandle<BufferT>>::type
cudaCreateNanoGrid(const NanoGrid<SrcBuildT> *d_srcGrid, const typename BuildToValueMap<DstBuildT>::type *d_srcValues, const BufferT &pool = BufferT(), cudaStream_t stream = 0)
{
    return tools::cuda::indexToGrid<DstBuildT, SrcBuildT, BufferT>(d_srcGrid, d_srcValues, pool, stream);
}

}// nanovdb namespace ===================================================================

#endif // NVIDIA_TOOLS_CUDA_INDEXTOGRID_CUH_HAS_BEEN_INCLUDED
