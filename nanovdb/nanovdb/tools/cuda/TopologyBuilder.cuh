// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/TopologyBuilder.cuh

    \authors Efty Sifakis

    \brief Shared functionality of (mostly morphology) operators that alter the voxel content of grids

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_TOPOLOGYBUILDER_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_TOPOLOGYBUILDER_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/util/cuda/Morphology.cuh>

namespace nanovdb {

namespace tools::cuda {

/// @brief Shared grid/tree offsets and node counts handed to the device
///        functors. Independent of the resource the builder allocates from,
///        so it lives outside TopologyBuilder and stays one type across every
///        ResourceT instantiation.
template <typename BuildT>
struct TopologyBuilderData {
    void     *d_bufferPtr;
    uint64_t grid, tree, root, upper, lower, leaf, size;// byte offsets to nodes in buffer
    uint32_t nodeCount[3];// 0=leaf,1=lower, 2=upper
    uint32_t *d_upperOffsets;
    __hostdev__ NanoGrid<BuildT>&  getGrid() const {return *util::PtrAdd<NanoGrid<BuildT>>(d_bufferPtr, grid);}
    __hostdev__ NanoTree<BuildT>&  getTree() const {return *util::PtrAdd<NanoTree<BuildT>>(d_bufferPtr, tree);}
    __hostdev__ NanoRoot<BuildT>&  getRoot() const {return *util::PtrAdd<NanoRoot<BuildT>>(d_bufferPtr, root);}
    __hostdev__ NanoUpper<BuildT>& getUpper(int i) const {return *(util::PtrAdd<NanoUpper<BuildT>>(d_bufferPtr, upper)+i);}
    __hostdev__ NanoLower<BuildT>& getLower(int i) const {return *(util::PtrAdd<NanoLower<BuildT>>(d_bufferPtr, lower)+i);}
    __hostdev__ NanoLeaf<BuildT>&  getLeaf(int i) const {return *(util::PtrAdd<NanoLeaf<BuildT>>(d_bufferPtr, leaf)+i);}
};// TopologyBuilderData

template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class TopologyBuilder
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex);// For now, only OnIndexGrids supported

    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;

    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "TopologyBuilder allocates stream-ordered scratch and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "TopologyBuilder reinterprets byte scratch as word-sized types and requires word-aligned allocations");

    /// @brief Device-only scratch storage, borrowing the injected resource
    ///        through a ResourceRef so all traffic reaches the caller's
    ///        instance (which may be stateful) rather than a copy. These
    ///        buffers are never read on the host, so they use the single-space
    ///        Buffer rather than the dual DeviceBuffer, whose host pointer and
    ///        per-device array they would leave unused.
    using ScratchT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>;

public:

    /// @param stream cuda stream the scratch allocations are ordered on
    /// @param resource resource instance all device scratch is allocated from;
    ///        must outlive this builder
    TopologyBuilder(cudaStream_t stream, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mUpperMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mUpperOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mVoxelOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerParents(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafParents(stream, resource, 0, nanovdb::cuda::noInit)
        , mResource(&resource)
        , mTempDevicePool(resource)
    {
        mData = nanovdb::cuda::DeviceBuffer::create(sizeof(Data));
    }

    using Data = TopologyBuilderData<BuildT>;

    void allocateInternalMaskBuffers(cudaStream_t stream);

    void countNodes(cudaStream_t stream);

    template<typename BufferT>
    BufferT getBuffer(const BufferT &buffer, cudaStream_t stream);

    void processUpperNodes(cudaStream_t stream);

    void processLowerNodes(cudaStream_t stream);

    void processLeafOffsets(cudaStream_t stream);

    void processBBox(cudaStream_t stream);

    void postProcessGridTree(cudaStream_t stream);

    nanovdb::cuda::DeviceBuffer  mProcessedRoot;
    ScratchT                     mUpperMasks;
    ScratchT                     mLowerMasks;
    ScratchT                     mUpperOffsets;
    ScratchT                     mLowerOffsets;
    ScratchT                     mLeafOffsets;
    ScratchT                     mVoxelOffsets;
    ScratchT                     mLowerParents;
    ScratchT                     mLeafParents;
    nanovdb::cuda::DeviceBuffer  mData;
    CheckMode                    mChecksum{CheckMode::Disable};

    auto deviceProcessedRoot() { return static_cast<RootT*>(mProcessedRoot.deviceData()); }
    auto hostProcessedRoot()   { return static_cast<RootT*>(mProcessedRoot.data()); }
    void* deviceUpperMasks() { return mUpperMasks.data(); }
    void* deviceLowerMasks() { return mLowerMasks.data(); }
    Data* data()             { return static_cast<Data*>(mData.data()); }
    Data* deviceData()       { return static_cast<Data*>(mData.deviceData()); }

private:
    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    ResourceT*                        mResource;// non-owning; all device scratch routes through this instance
    nanovdb::cuda::TempPool<ResourceT> mTempDevicePool;
};// tools::cuda::TopologyBuilder<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, stream)); \
    mTempDevicePool.reallocate(stream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, stream));
#else// ndef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, stream)); \
    mTempDevicePool.reallocate(stream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, stream));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void TopologyBuilder<BuildT, ResourceT>::allocateInternalMaskBuffers(cudaStream_t stream)
{
    if (hostProcessedRoot()->tileCount() == 0) return; // Processing empty grid(s); nothing to allocate

    // Allocate (and zero-fill) buffers large enough to hold:
    // (a) The serialized masks of all upper nodes, for all tiles in the updated root node, and
    // (b) The serialized masks of all densified lower nodes, as if every upper node had a full set of 32^3 lower children
    uint64_t upperSize = hostProcessedRoot()->tileCount() * sizeof(Mask<5>);
    uint64_t lowerSize = hostProcessedRoot()->tileCount() * Mask<5>::SIZE * sizeof(Mask<4>);
    mUpperMasks = ScratchT(stream, *mResource, upperSize, nanovdb::cuda::noInit);
    if (mUpperMasks.data() == nullptr) throw std::runtime_error("Failed to allocate upper mask buffer on device");
    cudaCheck(cudaMemsetAsync(mUpperMasks.data(), 0, upperSize, stream));
    mLowerMasks = ScratchT(stream, *mResource, lowerSize, nanovdb::cuda::noInit);
    if (mLowerMasks.data() == nullptr) throw std::runtime_error("Failed to allocate lower mask buffer on device");
    cudaCheck(cudaMemsetAsync(mLowerMasks.data(), 0, lowerSize, stream));
}// TopologyBuilder<BuildT, ResourceT>::allocateInternalMaskBuffers

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void TopologyBuilder<BuildT, ResourceT>::countNodes(cudaStream_t stream)
{
    auto processedTileCount = hostProcessedRoot()->tileCount();
    if (processedTileCount == 0) { // Processing empty grid(s); zero nodes at all levels
        data()->nodeCount[0] = data()->nodeCount[1] = data()->nodeCount[2] = 0;
        return;
    }

    // Computes prefix sums of (a) non-empty lower nodes, (b) counts of their leaf children,
    // and (c) count of the speculatively updated root tiles that have actually been used.
    // These are used to reconstruct child offsets for the internal nodes of the updated tree,
    // as well as the tile table at the root.
    std::size_t size = processedTileCount*Mask<5>::SIZE;

    ScratchT upperCountsBuffer = ScratchT(stream, *mResource, processedTileCount*sizeof(uint32_t), nanovdb::cuda::noInit);
    ScratchT lowerCountsBuffer = ScratchT(stream, *mResource, size*sizeof(uint32_t), nanovdb::cuda::noInit);
    ScratchT leafCountsBuffer = ScratchT(stream, *mResource, size*sizeof(uint32_t), nanovdb::cuda::noInit);

    using CountType = uint32_t (*)[Mask<5>::SIZE];
    auto lowerCounts = reinterpret_cast<CountType>(lowerCountsBuffer.data());
    auto leafCounts = reinterpret_cast<CountType>(leafCountsBuffer.data());

    using Op = util::morphology::cuda::EnumerateNodesFunctor;
    util::cuda::operatorKernel<Op>
        <<<dim3(processedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, stream>>>
        (deviceUpperMasks(), deviceLowerMasks(), lowerCounts, leafCounts);

    mUpperOffsets = ScratchT(stream, *mResource, (processedTileCount+1)*sizeof(uint32_t), nanovdb::cuda::noInit);
    mLowerOffsets = ScratchT(stream, *mResource, (size+1)*sizeof(uint32_t), nanovdb::cuda::noInit);
    mLeafOffsets = ScratchT(stream, *mResource, (size+1)*sizeof(uint32_t), nanovdb::cuda::noInit);

    cudaCheck(cudaMemsetAsync(mLowerOffsets.data(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        reinterpret_cast<uint32_t*>(lowerCountsBuffer.data()),
        reinterpret_cast<uint32_t*>(mLowerOffsets.data())+1,
        size);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[1], reinterpret_cast<uint32_t*>(mLowerOffsets.data())+size, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

    cudaCheck(cudaMemsetAsync(mLeafOffsets.data(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        reinterpret_cast<uint32_t*>(leafCountsBuffer.data()),
        reinterpret_cast<uint32_t*>(mLeafOffsets.data())+1,
        size);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[0], reinterpret_cast<uint32_t*>(mLeafOffsets.data())+size, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

    util::cuda::lambdaKernel<<<numBlocks(processedTileCount), mNumThreads, 0, stream>>>(
        processedTileCount,
        [] __device__(size_t tileID, CountType lowerOffsets, uint32_t* upperCounts)
            { upperCounts[tileID] = (lowerOffsets[tileID+1][0] > lowerOffsets[tileID][0]) ? 1 : 0; },
        reinterpret_cast<CountType>(mLowerOffsets.data()),
        reinterpret_cast<uint32_t*>(upperCountsBuffer.data()));

    cudaCheck(cudaMemsetAsync( mUpperOffsets.data(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        reinterpret_cast<uint32_t*>(upperCountsBuffer.data()),
        reinterpret_cast<uint32_t*>(mUpperOffsets.data())+1,
        processedTileCount);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[2], reinterpret_cast<uint32_t*>(mUpperOffsets.data())+processedTileCount, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
}// TopologyBuilder<BuildT, ResourceT>::countNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT, typename ResourceT>
template <typename BufferT>
BufferT TopologyBuilder<BuildT, ResourceT>::getBuffer(const BufferT &pool, cudaStream_t stream)
{
    // Allocates a device buffer for the destination grid, once the topology/size of the tree is known
    data()->grid  = 0;// grid is always stored at the start of the buffer!
    data()->tree  = GridT::memUsage();// grid ends and tree begins
    data()->root  = data()->tree  + TreeT::memUsage(); // tree ends and root node begins
    data()->upper = data()->root  + RootT::memUsage(data()->nodeCount[2]);// root node ends and upper internal nodes begin
    data()->lower = data()->upper + UpperT::memUsage()*data()->nodeCount[2];// upper internal nodes ends and lower internal nodes begin
    data()->leaf  = data()->lower + LowerT::memUsage()*data()->nodeCount[1];// lower internal nodes ends and leaf nodes begin
    data()->size  = data()->leaf  + LeafT::DataType::memUsage()*data()->nodeCount[0];// leaf nodes end and blind meta data begins

    int device = 0;
    cudaGetDevice(&device);
    auto buffer = BufferT::create(data()->size, &pool, device, stream);// only allocate buffer on the device
    cudaCheck(cudaMemsetAsync(buffer.deviceData(), 0, data()->size, stream));

    data()->d_bufferPtr = buffer.deviceData();
    if (data()->d_bufferPtr == nullptr) throw std::runtime_error("Failed to allocate grid buffer on the device");
    if (data()->nodeCount[2] != 0) // Unless the result is an empty grid
        data()->d_upperOffsets = reinterpret_cast<uint32_t*>(mUpperOffsets.data());
    mData.deviceUpload(device, stream, false);

    return buffer;
}// TopologyBuilder<BuildT, ResourceT>::getBuffer

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct BuildGridTreeRootFunctor
{
    __device__
    void operator()(size_t, TopologyBuilderData<BuildT> *d_data) {

        // process Root
        auto &root = d_data->getRoot();
        root.mTableSize = d_data->nodeCount[2];
        root.mBackground = NanoRoot<BuildT>::ValueType(0);// background_value
        root.mMinimum = root.mMaximum = NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = NanoRoot<BuildT>::FloatType(0);
        root.mBBox = CoordBBox(); // To be further updated after the leaf-level voxel update

        // process Tree
        auto &tree = d_data->getTree();
        tree.setRoot(&root);
        if (d_data->nodeCount[2]) {
            tree.setFirstNode(&d_data->getUpper(0));
            tree.setFirstNode(&d_data->getLower(0));
            tree.setFirstNode(&d_data->getLeaf(0));
        }
        else {
            tree.template setFirstNode<NanoUpper<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLower<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLeaf<BuildT>>(nullptr);
        }
        tree.mNodeCount[2] = d_data->nodeCount[2];
        tree.mNodeCount[1] = d_data->nodeCount[1];
        tree.mNodeCount[0] = d_data->nodeCount[0];
        tree.mVoxelCount = 0; // Actual voxel count (for non-empty grids) will only be known
                              // once leaf masks have been processed
        tree.mTileCount[2] = tree.mTileCount[1] =  tree.mTileCount[0] = 0;

        // process Grid
        // The GridData header has already been copied from the input;
        // reset what is necessary, and assert that others are at the expected values
        auto &grid = d_data->getGrid();

#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        NANOVDB_ASSERT(grid.mMagic == NANOVDB_MAGIC_GRID);
#else
        NANOVDB_ASSERT(grid.mMagic == NANOVDB_MAGIC_NUMB);
#endif
        grid.mChecksum.disable(); // all 64 bits ON means checksum is disabled
        NANOVDB_ASSERT(grid.mVersion == Version());
        NANOVDB_ASSERT(grid.mFlags.isMaskOn(GridFlags::IsBreadthFirst));
        grid.mFlags.initMask({GridFlags::IsBreadthFirst}); // expected flags (HasBBox will be set later if grid is non-empty)
        grid.mGridIndex = 0u; // Possibly overwriting input; returned grid has batch size 1
        grid.mGridCount = 1u; // Possibly overwriting input; returned grid has batch size 1
        grid.mGridSize = d_data->size;
        // grid.mGridName expected to have been copied verbatim from input
        // grid.mMap expected to have been copied verbatim from input
        grid.mWorldBBox = Vec3dBBox();// invalid bbox
        grid.mVoxelSize = grid.mMap.getVoxelSize();
        NANOVDB_ASSERT(grid.mGridClass == GridClass::IndexGrid);
        NANOVDB_ASSERT(grid.mGridType == toGridType<BuildT>());
        grid.mBlindMetadataOffset = d_data->size; // i.e. no blind data, even if the input grid had any
        grid.mBlindMetadataCount = 0u; // i.e. no blind data
        NANOVDB_ASSERT(grid.mData0 == 0u); // zero padding
        grid.mData1 = 1u; // This will be updated (unless this is an empty grid) after voxels have been processed
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        NANOVDB_ASSERT(grid.mData2 == 0u);
#else
        NANOVDB_ASSERT(grid.mData2 == NANOVDB_MAGIC_GRID);
#endif
    }
};

}// namespace topology::detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

/// @brief Variant of BuildGridTreeRootFunctor for grids constructed from scratch
///        (i.e. with no source grid to copy metadata from). Sets all GridData fields
///        explicitly rather than asserting they were pre-initialized.
template <typename BuildT>
struct InitGridTreeRootFunctor
{
    Map map; // transform to embed in the output grid

    __device__
    void operator()(size_t, TopologyBuilderData<BuildT> *d_data) {

        // process Root (identical to BuildGridTreeRootFunctor)
        auto &root = d_data->getRoot();
        root.mTableSize = d_data->nodeCount[2];
        root.mBackground = NanoRoot<BuildT>::ValueType(0);
        root.mMinimum = root.mMaximum = NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = NanoRoot<BuildT>::FloatType(0);
        root.mBBox = CoordBBox();

        // process Tree (identical to BuildGridTreeRootFunctor)
        auto &tree = d_data->getTree();
        tree.setRoot(&root);
        if (d_data->nodeCount[2]) {
            tree.setFirstNode(&d_data->getUpper(0));
            tree.setFirstNode(&d_data->getLower(0));
            tree.setFirstNode(&d_data->getLeaf(0));
        } else {
            tree.template setFirstNode<NanoUpper<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLower<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLeaf<BuildT>>(nullptr);
        }
        tree.mNodeCount[2] = d_data->nodeCount[2];
        tree.mNodeCount[1] = d_data->nodeCount[1];
        tree.mNodeCount[0] = d_data->nodeCount[0];
        tree.mVoxelCount = 0;
        tree.mTileCount[2] = tree.mTileCount[1] = tree.mTileCount[0] = 0;

        // process Grid — set all fields explicitly (no source grid to copy from)
        auto &grid = d_data->getGrid();
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        grid.mMagic = NANOVDB_MAGIC_GRID;
#else
        grid.mMagic = NANOVDB_MAGIC_NUMB;
#endif
        grid.mChecksum.disable();
        grid.mVersion = Version();
        grid.mFlags.initMask({GridFlags::IsBreadthFirst});
        grid.mGridIndex = 0u;
        grid.mGridCount = 1u;
        grid.mGridSize = d_data->size;
        // grid.mGridName is left zeroed; caller copies name via cudaMemcpyAsync
        grid.mMap = map;
        grid.mWorldBBox = Vec3dBBox();
        grid.mVoxelSize = map.getVoxelSize();
        grid.mGridClass = GridClass::IndexGrid;
        grid.mGridType = toGridType<BuildT>();
        grid.mBlindMetadataOffset = d_data->size;
        grid.mBlindMetadataCount = 0u;
        grid.mData0 = 0u;
        grid.mData1 = 1u;
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        grid.mData2 = 0u;
#else
        grid.mData2 = NANOVDB_MAGIC_GRID;
#endif
    }
};

}// namespace topology::detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct BuildUpperNodesFunctor
{
    __device__
    void operator()(size_t processedTileID, TopologyBuilderData<BuildT> *d_data, NanoRoot<BuildT> *d_processedRoot) {
        uint32_t tileID = d_data->d_upperOffsets[processedTileID];
        if (tileID != d_data->d_upperOffsets[processedTileID+1]) // if the offsets are the same, this was a speculatively introduced tile which was not necessary
        {
            auto &root  = d_data->getRoot();
            auto &dstUpper = d_data->getUpper(tileID);
            auto &processedTile = *d_processedRoot->tile(processedTileID);
            root.tile(tileID)->setChild( processedTile.origin(), &dstUpper, &root );
            dstUpper.mBBox = CoordBBox(); // To be further updated after the operation has been applied at leaf-level
            // TODO: Is this accurate? Any other flags that should be set?
            dstUpper.mFlags = (uint64_t)GridFlags::HasBBox;
        }
    }
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processUpperNodes(cudaStream_t stream)
{
    // Connect all newly allocated upper nodes to their respective tiles
    // Also fill in any necessary part of the preamble (in InternalData) of upper nodes
    auto processedTileCount = hostProcessedRoot()->tileCount();

    if (processedTileCount) { // Unless output grid is empty
        util::cuda::lambdaKernel<<<numBlocks(processedTileCount), mNumThreads, 0, stream>>>(
            processedTileCount, topology::detail::BuildUpperNodesFunctor<BuildT>(), deviceData(), deviceProcessedRoot());
        cudaCheckError();
    }
}// TopologyBuilder<BuildT, ResourceT>::processUpperNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processLowerNodes(cudaStream_t stream)
{
    // Fill out the contents of all newly allocated lower nodes (using the densified upper/lower mask arrays)
    // Also fill in the preamble (most of LeafData) for their leaf children
    auto processedTileCount = hostProcessedRoot()->tileCount();
    using CountType = uint32_t (*)[Mask<5>::SIZE];
 
    if (processedTileCount) { // Unless output grid is empty
        std::size_t lowerCount = data()->nodeCount[1];
        mLowerParents = ScratchT(stream, *mResource, lowerCount*sizeof(uint32_t), nanovdb::cuda::noInit);
        std::size_t leafCount = data()->nodeCount[0];
        mLeafParents = ScratchT(stream, *mResource, leafCount*sizeof(uint32_t), nanovdb::cuda::noInit);

        using Op = util::morphology::cuda::ProcessLowerNodesFunctor<BuildT>;
        util::cuda::operatorKernel<Op>
            <<<dim3(processedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, stream>>>(
                deviceUpperMasks(),
                deviceLowerMasks(),
                reinterpret_cast<uint32_t*>(mUpperOffsets.data()),
                reinterpret_cast<CountType>(mLowerOffsets.data()),
                reinterpret_cast<CountType>(mLeafOffsets.data()),
                static_cast<GridT*>(data()->d_bufferPtr),
                reinterpret_cast<uint32_t*>(mLowerParents.data()),
                reinterpret_cast<uint32_t*>(mLeafParents.data())
            );
        cudaCheckError();
    }

    mProcessedRoot.clear(stream);
    mUpperMasks.destroy(stream);
    mLowerMasks.destroy(stream);
    mLowerOffsets.destroy(stream);
    mLeafOffsets.destroy(stream);
}// TopologyBuilder<BuildT, ResourceT>::processLowerNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct UpdateLeafVoxelCountsAndPrefixSumFunctor
{
    __device__
    void operator()(size_t leafID, TopologyBuilderData<BuildT> *d_data, uint64_t *d_voxelCounts) {
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
    void operator()(size_t leafID, TopologyBuilderData<BuildT> *d_data, uint64_t *d_voxelOffsets) {
        auto &leaf = d_data->getGrid().tree().template getFirstNode<0>()[leafID];
        leaf.mOffset = d_voxelOffsets[leafID]+1; }
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processLeafOffsets(cudaStream_t stream)
{
    std::size_t leafCount = data()->nodeCount[0];
    if (leafCount) { // Unless output grid is empty
        mVoxelOffsets = ScratchT(stream, *mResource, (leafCount+1)*sizeof(uint64_t), nanovdb::cuda::noInit);
        cudaCheck(cudaMemsetAsync(mVoxelOffsets.data(), 0, sizeof(uint64_t), stream));
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, stream>>>(
            leafCount, topology::detail::UpdateLeafVoxelCountsAndPrefixSumFunctor<BuildT>(), deviceData(), reinterpret_cast<uint64_t*>(mVoxelOffsets.data())+1);
        CALL_CUBS(DeviceScan::InclusiveSum,
            reinterpret_cast<uint64_t*>(mVoxelOffsets.data())+1,
            reinterpret_cast<uint64_t*>(mVoxelOffsets.data())+1,
            leafCount);
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, stream>>>(
            leafCount, topology::detail::UpdateLeafVoxelOffsetsFunctor<BuildT>(), deviceData(), reinterpret_cast<uint64_t*>(mVoxelOffsets.data()));
    }
}// TopologyBuilder<BuildT, ResourceT>::processLeafOffsets

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Undefine utility macro for cub functions
#ifdef CALL_CUBS
#undef CALL_CUBS
#endif

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct UpdateAndPropagateLeafBBoxFunctor
{
    __device__
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data, const uint32_t* leafParents) {
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
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data, const uint32_t* lowerParents) {
        auto &upper = d_data->getUpper(lowerParents[tid]);
        auto &lower = d_data->getLower(tid);
        upper.mBBox.expandAtomic(lower.bbox()); }
};

template <typename BuildT>
struct PropagateUpperBBoxFunctor
{
    __device__
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data) {
        d_data->getRoot().mBBox.expandAtomic(d_data->getUpper(tid).bbox());
    }
};

template <typename BuildT>
struct UpdateRootWorldBBoxFunctor
{
    __device__
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data) {
        // TODO: check that the correct semantics are followed in this transformation
        auto BBox = d_data->getRoot().mBBox;
        BBox.max() += 1;
        d_data->getGrid().mFlags.setMaskOn(GridFlags::HasBBox);
        d_data->getGrid().mWorldBBox = BBox.transform(d_data->getGrid().data()->mMap);
    }
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processBBox(cudaStream_t stream)
{
    if (data()->nodeCount[0] == 0) return; // Output grid is empty; retain empty bounding box

    // TODO: Do we need a special case when flags indicates no bounding box?

    // update and propagate bbox from leaf -> lower/parent nodes
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[0]), mNumThreads, 0, stream>>>(
        data()->nodeCount[0], topology::detail::UpdateAndPropagateLeafBBoxFunctor<BuildT>(), deviceData(), reinterpret_cast<uint32_t*>(mLeafParents.data()));
    mLeafParents.destroy(stream);
    cudaCheckError();

    // propagate bbox from lower -> upper/parent node
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[1]), mNumThreads, 0, stream>>>(
        data()->nodeCount[1], topology::detail::PropagateLowerBBoxFunctor<BuildT>(), deviceData(), reinterpret_cast<uint32_t*>(mLowerParents.data()));
    mLowerParents.destroy(stream);
    cudaCheckError();

    // propagate bbox from upper -> root/parent node
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[2]), mNumThreads, 0, stream>>>(data()->nodeCount[2], topology::detail::PropagateUpperBBoxFunctor<BuildT>(), deviceData());
    cudaCheckError();

    // update the world-bbox in the root node
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, topology::detail::UpdateRootWorldBBoxFunctor<BuildT>(), deviceData());
    cudaCheckError();
}// TopologyBuilder<BuildT, ResourceT>::processBBox

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct PostProcessGridTreeFunctor
{
    __device__
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data, uint64_t* d_voxelOffsets) {
        auto& grid = d_data->getGrid();
        auto& tree = grid.tree();
        auto leafCount = tree.mNodeCount[0];
        tree.mVoxelCount = d_voxelOffsets[leafCount];
        grid.mData1 = tree.mVoxelCount+1;
    }
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::postProcessGridTree(cudaStream_t stream)
{
    // Finish updates to GridData/TreeData and (optionally) update checksum
    if (data()->nodeCount[0]) // if grid is empty, the default values are correct
        util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, topology::detail::PostProcessGridTreeFunctor<BuildT>(), deviceData(), reinterpret_cast<uint64_t*>(mVoxelOffsets.data()));
    cudaCheckError();
    mVoxelOffsets.destroy(stream);

    tools::cuda::updateChecksum((GridData*)data()->d_bufferPtr, mChecksum, stream);
}// TopologyBuilder<BuildT, ResourceT>::postProcessGridTree

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_TOPOLOGYBUILDER_CUH_HAS_BEEN_INCLUDED
