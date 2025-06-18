// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MergeGrids.cuh

    \authors Efty Sifakis

    \brief Morphological union of NanoVDB indexGrids on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_MERGEGRIDS_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_MERGEGRIDS_CUH_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/TempDevicePool.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/util/cuda/Morphology.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>


namespace nanovdb {

namespace tools::cuda {

template <typename BuildT>
class MergeGrids
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_index);// For now, only indexGrids supported

    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;
    using UpperMaskT = Mask<5>*;
    using LowerMaskT = Mask<4>(*)[Mask<5>::SIZE];

public:

    struct Data {
        void     *d_bufferPtr;
        uint64_t grid, tree, root, upper, lower, leaf, size;// byte offsets to nodes in buffer
        uint32_t nodeCount[3];// 0=leaf,1=lower, 2=upper
        uint32_t *d_upperOffsets;
        __hostdev__ GridT&  getGrid() const {return *util::PtrAdd<GridT>(d_bufferPtr, grid);}
        __hostdev__ TreeT&  getTree() const {return *util::PtrAdd<TreeT>(d_bufferPtr, tree);}
        __hostdev__ RootT&  getRoot() const {return *util::PtrAdd<RootT>(d_bufferPtr, root);}
        __hostdev__ UpperT& getUpper(int i) const {return *(util::PtrAdd<UpperT>(d_bufferPtr, upper)+i);}
        __hostdev__ LowerT& getLower(int i) const {return *(util::PtrAdd<LowerT>(d_bufferPtr, lower)+i);}
        __hostdev__ LeafT&  getLeaf(int i) const {return *(util::PtrAdd<LeafT>(d_bufferPtr, leaf)+i);}
    };// Data

    /// @brief Constructor
    /// @param d_srcGrid1 first source device grid to be merged
    /// @param d_srcGrid2 second source device grid to be merged
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    MergeGrids(GridT* d_srcGrid1, GridT* d_srcGrid2, cudaStream_t stream = 0)
        : mStream(stream), mTimer(stream)
    {
        mData = nanovdb::cuda::DeviceBuffer::create(sizeof(Data));
        mDeviceSrcGrid1 = d_srcGrid1;
        mDeviceSrcGrid2 = d_srcGrid2;
        // TODO: Should this be moved in one of the process functions?
        cudaCheck(cudaMemcpy(&mSrcTreeData1, util::PtrAdd(mDeviceSrcGrid1, GridT::memUsage()),
            TreeT::memUsage(), cudaMemcpyDeviceToHost));// copy TreeData from GPU -> CPU
        cudaCheck(cudaMemcpy(&mSrcTreeData2, util::PtrAdd(mDeviceSrcGrid2, GridT::memUsage()),
            TreeT::memUsage(), cudaMemcpyDeviceToHost));// copy TreeData from GPU -> CPU
    }

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mChecksum = mode;}

    /// @brief Creates a handle to the merged grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

    void mergeRoot();

    void allocateInternalMaskBuffers();

    void mergeInternalNodes();

    void countNodes();

    void processGridTreeRoot();

    void processUpperNodes();

    void processLowerNodes();

    void mergeLeafNodes();

    void processBBox();

    void postProcessGridTree();

private:
    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    GridT                        *mDeviceSrcGrid1;
    GridT                        *mDeviceSrcGrid2;
    GridT                        *mDeviceSrcGrid;
    TreeData                     mSrcTreeData1;
    TreeData                     mSrcTreeData2;
    nanovdb::cuda::DeviceBuffer  mMergedRoot;
    nanovdb::cuda::DeviceBuffer  mUpperMasks;
    nanovdb::cuda::DeviceBuffer  mLowerMasks;
    nanovdb::cuda::DeviceBuffer  mUpperOffsets;
    nanovdb::cuda::DeviceBuffer  mLowerOffsets;
    nanovdb::cuda::DeviceBuffer  mLeafOffsets;
    nanovdb::cuda::DeviceBuffer  mVoxelOffsets;
    nanovdb::cuda::DeviceBuffer  mLowerParents;
    nanovdb::cuda::DeviceBuffer  mLeafParents;
    nanovdb::cuda::DeviceBuffer  mData;
    CheckMode                    mChecksum{CheckMode::Disable};

public:
    GridT* deviceSrcGrid1()   { return mDeviceSrcGrid1; }
    GridT* deviceSrcGrid2()   { return mDeviceSrcGrid2; }
    auto deviceMergedRoot() { return static_cast<RootT*>(mMergedRoot.deviceData()); }
    auto hostMergedRoot()   { return static_cast<RootT*>(mMergedRoot.data()); }
    void* deviceUpperMasks() { return mUpperMasks.deviceData(); }
    void* deviceLowerMasks() { return mLowerMasks.deviceData(); }
    Data* data()             { return static_cast<Data*>(mData.data()); }
    Data* deviceData()       { return static_cast<Data*>(mData.deviceData()); }
private:
    nanovdb::cuda::TempDevicePool mTempDevicePool;

public:
    // TODO: make private
    template<typename BufferT>
    BufferT getBuffer(const BufferT &buffer);
};// tools::cuda::MergeGrids<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else// ndef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
MergeGrids<BuildT>::getHandle(const BufferT &pool)
{
    // Merge root nodes
    if (mVerbose==1) mTimer.start("\nMerging root nodes");
    mergeRoot();

    // Allocate memory for merged upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    allocateInternalMaskBuffers();

    // Merge masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Merge internal nodes");
    mergeInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count merged tree nodes");
    countNodes();

    cudaStreamSynchronize(mStream);

    // Allocate new device grid buffer for merged result
    if (mVerbose==1) mTimer.restart("Allocating merged grid buffer");
    auto buffer = getBuffer(pool);

    // Process GridData/TreeData/RootData of merged result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of merged result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    processUpperNodes();

    // Process lower nodes of merged result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    processLowerNodes();

    // Merge leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Merging leaf nodes");
    mergeLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    processBBox();

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    postProcessGridTree();
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}// MergeGrids<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeRoot()
{
    // Creates a new merged tree root with the merged tiles of the two input root topologies

    int device = 0;
    cudaGetDevice(&device);

    // Make a host copy of the source root topology RootNode for both inputs

    auto deviceSrcRoot1 = static_cast<RootT*>(util::PtrAdd(mDeviceSrcGrid1, GridT::memUsage() + mSrcTreeData1.mNodeOffset[3]));
    auto deviceSrcRoot2 = static_cast<RootT*>(util::PtrAdd(mDeviceSrcGrid2, GridT::memUsage() + mSrcTreeData2.mNodeOffset[3]));
    uint64_t rootSize1 = mSrcTreeData1.mNodeOffset[2] - mSrcTreeData1.mNodeOffset[3];
    uint64_t rootSize2 = mSrcTreeData2.mNodeOffset[2] - mSrcTreeData2.mNodeOffset[3];
    auto srcRootBuffer1 = nanovdb::HostBuffer::create(rootSize1);
    auto srcRootBuffer2 = nanovdb::HostBuffer::create(rootSize2);
    cudaCheck(cudaMemcpy(srcRootBuffer1.data(), deviceSrcRoot1, rootSize1, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(srcRootBuffer2.data(), deviceSrcRoot2, rootSize2, cudaMemcpyDeviceToHost));
    auto srcRoot1 = static_cast<RootT*>(srcRootBuffer1.data());
    auto srcRoot2 = static_cast<RootT*>(srcRootBuffer2.data());

    // Merge tiles of two sources in a sorted container

    std::set<nanovdb::Coord> mergedTileOrigins;
    for (uint32_t t = 0; t < srcRoot1->tileCount(); t++) {
        auto origin = srcRoot1->tile(t)->origin();
        if (!mergedTileOrigins.count(origin)) mergedTileOrigins.insert(origin); }
    for (uint32_t t = 0; t < srcRoot2->tileCount(); t++) {
        auto origin = srcRoot2->tile(t)->origin();
        if (!mergedTileOrigins.count(origin)) mergedTileOrigins.insert(origin); }
    std::vector<typename RootT::Tile> mergedTiles;
    mergedTiles.reserve(mergedTileOrigins.size());
    for (const auto& origin: mergedTileOrigins) {
        typename RootT::Tile tile{RootT::CoordToKey(origin)};
        mergedTiles.push_back(tile); }

    // Package the new root topology into a RootNode plus Tile list; upload to the GPU
    uint64_t rootSize = RootT::memUsage(mergedTiles.size());
    mMergedRoot = nanovdb::cuda::DeviceBuffer::create(rootSize);
    auto mergedRootPtr = static_cast<RootT*>(mMergedRoot.data());
    mergedRootPtr->mTableSize = mergedTiles.size();
    for (std::size_t t = 0; t < mergedTiles.size(); t++)
        *mergedRootPtr->tile(t) = mergedTiles[t];
    mMergedRoot.deviceUpload(device, mStream, false);
}// MergeGrids<BuildT>::mergeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::allocateInternalMaskBuffers()
{
    // Allocate (and zero-fill) buffers large enough to hold:
    // (a) The serialized masks of all upper nodes, for all tiles in the merged root node, and
    // (b) The serialized masks of all densified lower nodes, as if every upper node had a full set of 32^3 lower children
    int device = 0;
    cudaGetDevice(&device);
    uint64_t upperSize = hostMergedRoot()->tileCount() * sizeof(Mask<5>);
    uint64_t lowerSize = hostMergedRoot()->tileCount() * Mask<5>::SIZE * sizeof(Mask<4>);
    mUpperMasks = nanovdb::cuda::DeviceBuffer::create(upperSize, nullptr, device, mStream);
    if (mUpperMasks.deviceData() == nullptr) throw std::runtime_error("Failed to allocate upper mask buffer on device");
    cudaCheck(cudaMemsetAsync(mUpperMasks.deviceData(), 0, upperSize, mStream));
    mLowerMasks = nanovdb::cuda::DeviceBuffer::create( lowerSize, nullptr, device, mStream );
    if (mLowerMasks.deviceData() == nullptr) throw std::runtime_error("Failed to allocate lower mask buffer on device");
    cudaCheck(cudaMemsetAsync(mLowerMasks.deviceData(), 0, lowerSize, mStream));
}// MergeGrids<BuildT>::allocateInternalMaskBuffers

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeInternalNodes()
{
    // Merges the masks of upper and lower nodes from both input topologies into the
    // densified, pre-allocated mask arrays of the merged result
    using Op = morphology::cuda::MergeInternalNodesFunctor<BuildT>;
    util::cuda::operatorKernel<Op>
        <<<mSrcTreeData1.mNodeCount[1], Op::MaxThreadsPerBlock, 0, mStream>>>
        (deviceSrcGrid1(), deviceMergedRoot(), deviceUpperMasks(), deviceLowerMasks());
    util::cuda::operatorKernel<Op>
        <<<mSrcTreeData2.mNodeCount[1], Op::MaxThreadsPerBlock, 0, mStream>>>
        (deviceSrcGrid2(), deviceMergedRoot(), deviceUpperMasks(), deviceLowerMasks());
}// MergeGrids<BuildT>::mergeInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::countNodes()
{
    // Computes prefix sums of (a) non-empty lower nodes, (b) counts of their leaf children,
    // and (c) count of the merged root tiles. These are used to reconstruct child offsets
    // for the internal nodes of the merged tree, as well as the tile table at the root.
    auto mergedTileCount = hostMergedRoot()->tileCount();
    std::size_t size = mergedTileCount*Mask<5>::SIZE;

    int device = 0;
    cudaGetDevice(&device);
    nanovdb::cuda::DeviceBuffer upperCountsBuffer = nanovdb::cuda::DeviceBuffer::create(mergedTileCount*sizeof(uint32_t), nullptr, device, mStream);
    nanovdb::cuda::DeviceBuffer lowerCountsBuffer = nanovdb::cuda::DeviceBuffer::create(size*sizeof(uint32_t), nullptr, device, mStream);
    nanovdb::cuda::DeviceBuffer leafCountsBuffer = nanovdb::cuda::DeviceBuffer::create(size*sizeof(uint32_t), nullptr, device, mStream);

    using CountType = uint32_t (*)[Mask<5>::SIZE];
    auto lowerCounts = reinterpret_cast<CountType>( lowerCountsBuffer.deviceData() );
    auto leafCounts = reinterpret_cast<CountType>( leafCountsBuffer.deviceData() );

    using Op = morphology::cuda::EnumerateNodesFunctor;
    util::cuda::operatorKernel<Op>
        <<<dim3(mergedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, mStream>>>
        (deviceUpperMasks(), deviceLowerMasks(), lowerCounts, leafCounts);

    mUpperOffsets = nanovdb::cuda::DeviceBuffer::create((mergedTileCount+1)*sizeof(uint32_t), nullptr, device, mStream);
    mLowerOffsets = nanovdb::cuda::DeviceBuffer::create((size+1)*sizeof(uint32_t), nullptr, device, mStream);
    mLeafOffsets = nanovdb::cuda::DeviceBuffer::create((size+1)*sizeof(uint32_t), nullptr, device, mStream);

    cudaCheck(cudaMemsetAsync(mLowerOffsets.deviceData(), 0, sizeof(uint32_t), mStream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint32_t*>(lowerCountsBuffer.deviceData()),
        static_cast<uint32_t*>(mLowerOffsets.deviceData())+1,
        size);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[1], static_cast<uint32_t*>(mLowerOffsets.deviceData())+size, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));

    cudaCheck(cudaMemsetAsync(mLeafOffsets.deviceData(), 0, sizeof(uint32_t), mStream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint32_t*>(leafCountsBuffer.deviceData()),
        static_cast<uint32_t*>(mLeafOffsets.deviceData())+1,
        size);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[0], static_cast<uint32_t*>(mLeafOffsets.deviceData())+size, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));

    util::cuda::lambdaKernel<<<numBlocks(mergedTileCount), mNumThreads, 0, mStream>>>(
        mergedTileCount,
        [] __device__(size_t tileID, CountType lowerOffsets, uint32_t* upperCounts)
            { upperCounts[tileID] = (lowerOffsets[tileID+1][0] > lowerOffsets[tileID][0]) ? 1 : 0; },
        static_cast<CountType>(mLowerOffsets.deviceData()),
        static_cast<uint32_t*>(upperCountsBuffer.deviceData()));

    cudaCheck(cudaMemsetAsync( mUpperOffsets.deviceData(), 0, sizeof(uint32_t), mStream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint32_t*>(upperCountsBuffer.deviceData()),
        static_cast<uint32_t*>(mUpperOffsets.deviceData())+1,
        mergedTileCount);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[2], static_cast<uint32_t*>(mUpperOffsets.deviceData())+mergedTileCount, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
}// MergeGrids<BuildT>::countNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
template <typename BufferT>
BufferT MergeGrids<BuildT>::getBuffer(const BufferT &pool)
{
    // Allocates a device buffer for the destination (merged) grid, once the topology/size of the tree is known
    data()->grid  = 0;// grid is always stored at the start of the buffer!
    data()->tree  = NanoGrid<BuildT>::memUsage();// grid ends and tree begins
    data()->root  = data()->tree  + NanoTree<BuildT>::memUsage(); // tree ends and root node begins
    data()->upper = data()->root  + NanoRoot<BuildT>::memUsage(data()->nodeCount[2]);// root node ends and upper internal nodes begin
    data()->lower = data()->upper + NanoUpper<BuildT>::memUsage()*data()->nodeCount[2];// upper internal nodes ends and lower internal nodes begin
    data()->leaf  = data()->lower + NanoLower<BuildT>::memUsage()*data()->nodeCount[1];// lower internal nodes ends and leaf nodes begin
    data()->size  = data()->leaf  + NanoLeaf<BuildT>::DataType::memUsage()*data()->nodeCount[0];// leaf nodes end and blind meta data begins

    int device = 0;
    cudaGetDevice(&device);
    auto buffer = BufferT::create(data()->size, &pool, device, mStream);// only allocate buffer on the device
    cudaCheck(cudaMemsetAsync(buffer.deviceData(), 0, data()->size, mStream));

    data()->d_bufferPtr = buffer.deviceData();
    if (data()->d_bufferPtr == nullptr) throw std::runtime_error("Failed to allocate grid buffer on the device");
    data()->d_upperOffsets = static_cast<uint32_t*>(mUpperOffsets.deviceData());
    mData.deviceUpload(device, mStream, false);

    return buffer;
}// MergeGrids<BuildT>::getBuffer

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
struct BuildGridTreeRootFunctor
{
    __device__
    void operator()(size_t, typename MergeGrids<BuildT>::Data *d_data) {

        // process Root
        auto &root = d_data->getRoot();
        root.mTableSize = d_data->nodeCount[2];
        root.mBackground = NanoRoot<BuildT>::ValueType(0);// background_value
        root.mMinimum = root.mMaximum = NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = NanoRoot<BuildT>::FloatType(0);
        root.mBBox = CoordBBox(); // To be further updated after the leaf-level voxel merging

        // process Tree
        auto &tree = d_data->getTree();
        tree.setRoot(&root);
        tree.setFirstNode(&d_data->getUpper(0));
        tree.setFirstNode(&d_data->getLower(0));
        tree.setFirstNode(&d_data->getLeaf(0));
        tree.mNodeCount[2] = d_data->nodeCount[2];
        tree.mNodeCount[1] = d_data->nodeCount[1];
        tree.mNodeCount[0] = d_data->nodeCount[0];
        tree.mVoxelCount = 0; // Actual voxel count will only be known once leaf masks have been merged
        // TODO: Does this need to be updated later?
        tree.mTileCount[2] = tree.mTileCount[1] =  tree.mTileCount[0] = 0;

        // process Grid
        // TODO: Do we need to update anything else (beyond what is here and in post-process?)
        auto &grid = d_data->getGrid();
        grid.mChecksum = ~uint64_t(0);// set all bits on which means it's disabled
    }
};

template <typename BuildT>
void MergeGrids<BuildT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // TODO: Check for instances where extra processing is needed
    // TODO: check that the second grid input has consistent GridData, too
    cudaCheck(cudaMemcpyAsync(&data()->getGrid(), deviceSrcGrid1()->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, BuildGridTreeRootFunctor<BuildT>(), deviceData());
    cudaCheckError();
}// MergeGrids<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
struct BuildUpperNodesFunctor
{
    __device__
    void operator()(size_t mergedTileID, typename MergeGrids<BuildT>::Data *d_data, NanoRoot<BuildT> *d_mergedRoot) {
        uint32_t tileID = d_data->d_upperOffsets[mergedTileID];
        if (tileID != d_data->d_upperOffsets[mergedTileID+1]) // TODO: This should always succeed, if the inputs were properly pruned
        {
            auto &root  = d_data->getRoot();
            auto &dstUpper = d_data->getUpper(tileID);
            auto &mergedTile = *d_mergedRoot->tile(mergedTileID);
            root.tile(tileID)->setChild( mergedTile.origin(), &dstUpper, &root );
            dstUpper.mBBox = CoordBBox(); // To be further updated after the leaf-level voxel merging
            // TODO: Is this accurate? Any other flags that should be set?
            dstUpper.mFlags = (uint64_t)GridFlags::HasBBox;
        }
    }
};

template <typename BuildT>
inline void MergeGrids<BuildT>::processUpperNodes()
{
    // Connect all newly allocated upper nodes to their respective tiles
    // Also fill in any necessary part of the preamble (in InternalData) of upper nodes
    auto mergedTileCount = hostMergedRoot()->tileCount();

    util::cuda::lambdaKernel<<<numBlocks(mergedTileCount), mNumThreads, 0, mStream>>>(
        mergedTileCount, BuildUpperNodesFunctor<BuildT>(), deviceData(), deviceMergedRoot());
    cudaCheckError();
}// MergeGrids<BuildT>::processUpperNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
inline void MergeGrids<BuildT>::processLowerNodes()
{
    // Fill out the contents of all newly allocated lower nodes (using the densified upper/lower mask arrays)
    // Also fill in the preamble (most of LeafData) for their leaf children
    auto mergedTileCount = hostMergedRoot()->tileCount();
    using CountType = uint32_t (*)[Mask<5>::SIZE];
 
    int device = 0;
    cudaGetDevice(&device);
    std::size_t lowerCount = data()->nodeCount[1];
    mLowerParents = nanovdb::cuda::DeviceBuffer::create(lowerCount*sizeof(uint32_t), nullptr, device, mStream);
    std::size_t leafCount = data()->nodeCount[0];
    mLeafParents = nanovdb::cuda::DeviceBuffer::create(leafCount*sizeof(uint32_t), nullptr, device, mStream);

    using Op = morphology::cuda::ProcessLowerNodesFunctor<BuildT>;
    util::cuda::operatorKernel<Op>
        <<<dim3(mergedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, mStream>>>(
            deviceUpperMasks(),
            deviceLowerMasks(),
            static_cast<uint32_t*>(mUpperOffsets.deviceData()),
            static_cast<CountType>(mLowerOffsets.deviceData()),
            static_cast<CountType>(mLeafOffsets.deviceData()),
            static_cast<GridT*>(data()->d_bufferPtr),
            static_cast<uint32_t*>(mLowerParents.deviceData()),
            static_cast<uint32_t*>(mLeafParents.deviceData())
        );
    cudaCheckError();
    mMergedRoot.clear(mStream);
    mUpperMasks.clear(mStream);
    mLowerMasks.clear(mStream);
    mLowerOffsets.clear(mStream);
    mLeafOffsets.clear(mStream);
}// MergeGrids<BuildT>::processLowerNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
struct UpdateLeafVoxelCountsAndPrefixSumFunctor
{
    __device__
    void operator()(size_t leafID, typename MergeGrids<BuildT>::Data *d_data, uint64_t *d_voxelCounts) {
        auto &leaf = d_data->getGrid().tree().template getFirstNode<0>()[leafID];
        const uint64_t *w = leaf.mValueMask.words();
        uint64_t prefixSum = 0, sum = util::countOn(*w++);
        prefixSum = sum;
        for (int n = 9; n < 55; n += 9) {// n=i*9 where i=1,2,..6
            sum += util::countOn(*w++);
            prefixSum |= sum << n; }// each pre-fixed sum is encoded in 9 bits
        sum += util::countOn(*w);
        d_voxelCounts[leafID] = sum;
        leaf.mPrefixSum = prefixSum; }
};

template <typename BuildT>
struct UpdateLeafVoxelOffsetsFunctor
{
    __device__
    void operator()(size_t leafID, typename MergeGrids<BuildT>::Data *d_data, uint64_t *d_voxelOffsets) {
        auto &leaf = d_data->getGrid().tree().template getFirstNode<0>()[leafID];
        leaf.mOffset = d_voxelOffsets[leafID]+1; }
};

template<typename BuildT>
void MergeGrids<BuildT>::mergeLeafNodes()
{
    using Op = morphology::cuda::MergeLeafNodesFunctor<BuildT>;
    nanovdb::util::cuda::operatorKernel<Op>
        <<<dim3(mSrcTreeData1.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
        (deviceSrcGrid1(), static_cast<GridT*>(data()->d_bufferPtr));
    nanovdb::util::cuda::operatorKernel<Op>
        <<<dim3(mSrcTreeData2.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
        (deviceSrcGrid2(), static_cast<GridT*>(data()->d_bufferPtr));

    int device = 0;
    cudaGetDevice(&device);
    std::size_t leafCount = data()->nodeCount[0];
    mVoxelOffsets = nanovdb::cuda::DeviceBuffer::create((leafCount+1)*sizeof(uint64_t), nullptr, device, mStream);
    cudaCheck(cudaMemsetAsync(mVoxelOffsets.deviceData(), 0, sizeof(uint64_t), mStream));
    util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, mStream>>>(
        leafCount, UpdateLeafVoxelCountsAndPrefixSumFunctor<BuildT>(), deviceData(), static_cast<uint64_t*>(mVoxelOffsets.deviceData())+1);
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint64_t*>(mVoxelOffsets.deviceData())+1,
        static_cast<uint64_t*>(mVoxelOffsets.deviceData())+1,
        leafCount);
    util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, mStream>>>(
        leafCount, UpdateLeafVoxelOffsetsFunctor<BuildT>(), deviceData(), static_cast<uint64_t*>(mVoxelOffsets.deviceData()));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
struct UpdateAndPropagateLeafBBoxFunctor
{
    __device__
    void operator()(size_t tid, typename MergeGrids<BuildT>::Data *d_data, const uint32_t* leafParents) {
        auto &lower = d_data->getLower(leafParents[tid]);
        auto &leaf = d_data->getLeaf(tid);
        leaf.updateBBox();
        lower.mBBox.expandAtomic(leaf.bbox());
    }
};

template <typename BuildT>
struct PropagateLowerBBoxFunctor
{
    __device__
    void operator()(size_t tid, typename MergeGrids<BuildT>::Data *d_data, const uint32_t* lowerParents) {
        auto &upper = d_data->getUpper(lowerParents[tid]);
        auto &lower = d_data->getLower(tid);
        upper.mBBox.expandAtomic(lower.bbox()); }
};

template <typename BuildT>
struct PropagateUpperBBoxFunctor
{
    __device__
    void operator()(size_t tid, typename MergeGrids<BuildT>::Data *d_data) {
        d_data->getRoot().mBBox.expandAtomic(d_data->getUpper(tid).bbox());
    }
};

template <typename BuildT>
struct UpdateRootWorldBBoxFunctor
{
    __device__
    void operator()(size_t tid, typename MergeGrids<BuildT>::Data *d_data) {
        // TODO: check that the correct semantics are followed in this transformation
        auto BBox = d_data->getRoot().mBBox;
        BBox.max() += 1;
        d_data->getGrid().mWorldBBox = BBox.transform(d_data->getGrid().data()->mMap);
    }
};

template <typename BuildT>
inline void MergeGrids<BuildT>::processBBox()
{
    // TODO: Do we need a special case when flags indicates no bounding box?

    // update and propagate bbox from leaf -> lower/parent nodes
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[0]), mNumThreads, 0, mStream>>>(
        data()->nodeCount[0], UpdateAndPropagateLeafBBoxFunctor<BuildT>(), deviceData(), static_cast<uint32_t*>(mLeafParents.deviceData()));
    mLeafParents.clear(mStream);
    cudaCheckError();

    // propagate bbox from lower -> upper/parent node
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[1]), mNumThreads, 0, mStream>>>(
        data()->nodeCount[1], PropagateLowerBBoxFunctor<BuildT>(), deviceData(), static_cast<uint32_t*>(mLowerParents.deviceData()));
    mLowerParents.clear(mStream);
    cudaCheckError();

    // propagate bbox from upper -> root/parent node
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[2]), mNumThreads, 0, mStream>>>(data()->nodeCount[2], PropagateUpperBBoxFunctor<BuildT>(), deviceData());
    cudaCheckError();

    // update the world-bbox in the root node
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, UpdateRootWorldBBoxFunctor<BuildT>(), deviceData());
    cudaCheckError();
}// MergeGrids<BuildT>::processBBox

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
struct PostProcessGridTreeFunctor
{
    __device__
    void operator()(size_t tid, typename MergeGrids<BuildT>::Data *d_data, uint64_t* d_voxelOffsets) {
        auto& grid = d_data->getGrid();
        auto& tree = grid.tree();
        auto leafCount = tree.mNodeCount[0];
        tree.mVoxelCount = d_voxelOffsets[leafCount];
        grid.mGridSize = d_data->size;
        grid.mGridCount = 1;
        grid.mBlindMetadataOffset = d_data->size;
        grid.mData1 = tree.mVoxelCount+1;
    }
};

template <typename BuildT>
inline void MergeGrids<BuildT>::postProcessGridTree()
{
    // Finish updates to GridData/TreeData and (optionally) update checksum
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, PostProcessGridTreeFunctor<BuildT>(), deviceData(), static_cast<uint64_t*>(mVoxelOffsets.deviceData()));
    cudaCheckError();
    mVoxelOffsets.clear(mStream);

    tools::cuda::updateChecksum((GridData*)data()->d_bufferPtr, mChecksum, mStream);
}// MergeGrids<BuildT>::postProcessGridTree

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MERGEGRIDS_CUH_HAS_BEEN_INCLUDED
