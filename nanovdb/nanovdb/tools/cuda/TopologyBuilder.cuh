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
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/util/cuda/Morphology.cuh>

namespace nanovdb {

namespace tools::cuda {

template <typename BuildT>
class TopologyBuilder
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex);// For now, only OnIndexGrids supported

    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;

public:

    TopologyBuilder(cudaStream_t stream)
    {
        mData = nanovdb::cuda::DeviceBuffer::create(sizeof(Data));
    }

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

    auto deviceProcessedRoot() { return static_cast<RootT*>(mProcessedRoot.deviceData()); }
    auto hostProcessedRoot()   { return static_cast<RootT*>(mProcessedRoot.data()); }
    void* deviceUpperMasks() { return mUpperMasks.deviceData(); }
    void* deviceLowerMasks() { return mLowerMasks.deviceData(); }
    Data* data()             { return static_cast<Data*>(mData.data()); }
    Data* deviceData()       { return static_cast<Data*>(mData.deviceData()); }

private:
    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    nanovdb::cuda::TempDevicePool mTempDevicePool;
};// tools::cuda::TopologyBuilder<BuildT>

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

template<typename BuildT>
void TopologyBuilder<BuildT>::allocateInternalMaskBuffers(cudaStream_t stream)
{
    if (hostProcessedRoot()->tileCount() == 0) return; // Processing empty grid(s); nothing to allocate

    // Allocate (and zero-fill) buffers large enough to hold:
    // (a) The serialized masks of all upper nodes, for all tiles in the updated root node, and
    // (b) The serialized masks of all densified lower nodes, as if every upper node had a full set of 32^3 lower children
    int device = 0;
    cudaGetDevice(&device);
    uint64_t upperSize = hostProcessedRoot()->tileCount() * sizeof(Mask<5>);
    uint64_t lowerSize = hostProcessedRoot()->tileCount() * Mask<5>::SIZE * sizeof(Mask<4>);
    mUpperMasks = nanovdb::cuda::DeviceBuffer::create(upperSize, nullptr, device, stream);
    if (mUpperMasks.deviceData() == nullptr) throw std::runtime_error("Failed to allocate upper mask buffer on device");
    cudaCheck(cudaMemsetAsync(mUpperMasks.deviceData(), 0, upperSize, stream));
    mLowerMasks = nanovdb::cuda::DeviceBuffer::create( lowerSize, nullptr, device, stream );
    if (mLowerMasks.deviceData() == nullptr) throw std::runtime_error("Failed to allocate lower mask buffer on device");
    cudaCheck(cudaMemsetAsync(mLowerMasks.deviceData(), 0, lowerSize, stream));
}// TopologyBuilder<BuildT>::allocateInternalMaskBuffers

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void TopologyBuilder<BuildT>::countNodes(cudaStream_t stream)
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

    int device = 0;
    cudaGetDevice(&device);
    nanovdb::cuda::DeviceBuffer upperCountsBuffer = nanovdb::cuda::DeviceBuffer::create(processedTileCount*sizeof(uint32_t), nullptr, device, stream);
    nanovdb::cuda::DeviceBuffer lowerCountsBuffer = nanovdb::cuda::DeviceBuffer::create(size*sizeof(uint32_t), nullptr, device, stream);
    nanovdb::cuda::DeviceBuffer leafCountsBuffer = nanovdb::cuda::DeviceBuffer::create(size*sizeof(uint32_t), nullptr, device, stream);

    using CountType = uint32_t (*)[Mask<5>::SIZE];
    auto lowerCounts = reinterpret_cast<CountType>( lowerCountsBuffer.deviceData() );
    auto leafCounts = reinterpret_cast<CountType>( leafCountsBuffer.deviceData() );

    using Op = util::morphology::cuda::EnumerateNodesFunctor;
    util::cuda::operatorKernel<Op>
        <<<dim3(processedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, stream>>>
        (deviceUpperMasks(), deviceLowerMasks(), lowerCounts, leafCounts);

    mUpperOffsets = nanovdb::cuda::DeviceBuffer::create((processedTileCount+1)*sizeof(uint32_t), nullptr, device, stream);
    mLowerOffsets = nanovdb::cuda::DeviceBuffer::create((size+1)*sizeof(uint32_t), nullptr, device, stream);
    mLeafOffsets = nanovdb::cuda::DeviceBuffer::create((size+1)*sizeof(uint32_t), nullptr, device, stream);

    cudaCheck(cudaMemsetAsync(mLowerOffsets.deviceData(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint32_t*>(lowerCountsBuffer.deviceData()),
        static_cast<uint32_t*>(mLowerOffsets.deviceData())+1,
        size);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[1], static_cast<uint32_t*>(mLowerOffsets.deviceData())+size, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

    cudaCheck(cudaMemsetAsync(mLeafOffsets.deviceData(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint32_t*>(leafCountsBuffer.deviceData()),
        static_cast<uint32_t*>(mLeafOffsets.deviceData())+1,
        size);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[0], static_cast<uint32_t*>(mLeafOffsets.deviceData())+size, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

    util::cuda::lambdaKernel<<<numBlocks(processedTileCount), mNumThreads, 0, stream>>>(
        processedTileCount,
        [] __device__(size_t tileID, CountType lowerOffsets, uint32_t* upperCounts)
            { upperCounts[tileID] = (lowerOffsets[tileID+1][0] > lowerOffsets[tileID][0]) ? 1 : 0; },
        static_cast<CountType>(mLowerOffsets.deviceData()),
        static_cast<uint32_t*>(upperCountsBuffer.deviceData()));

    cudaCheck(cudaMemsetAsync( mUpperOffsets.deviceData(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint32_t*>(upperCountsBuffer.deviceData()),
        static_cast<uint32_t*>(mUpperOffsets.deviceData())+1,
        processedTileCount);
    cudaCheck(cudaMemcpyAsync(&data()->nodeCount[2], static_cast<uint32_t*>(mUpperOffsets.deviceData())+processedTileCount, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
}// TopologyBuilder<BuildT>::countNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
template <typename BufferT>
BufferT TopologyBuilder<BuildT>::getBuffer(const BufferT &pool, cudaStream_t stream)
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
        data()->d_upperOffsets = static_cast<uint32_t*>(mUpperOffsets.deviceData());
    mData.deviceUpload(device, stream, false);

    return buffer;
}// TopologyBuilder<BuildT>::getBuffer

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct BuildGridTreeRootFunctor
{
    __device__
    void operator()(size_t, typename TopologyBuilder<BuildT>::Data *d_data) {

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

template <typename BuildT>
struct BuildUpperNodesFunctor
{
    __device__
    void operator()(size_t processedTileID, typename TopologyBuilder<BuildT>::Data *d_data, NanoRoot<BuildT> *d_processedRoot) {
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

template <typename BuildT>
inline void TopologyBuilder<BuildT>::processUpperNodes(cudaStream_t stream)
{
    // Connect all newly allocated upper nodes to their respective tiles
    // Also fill in any necessary part of the preamble (in InternalData) of upper nodes
    auto processedTileCount = hostProcessedRoot()->tileCount();

    if (processedTileCount) { // Unless output grid is empty
        util::cuda::lambdaKernel<<<numBlocks(processedTileCount), mNumThreads, 0, stream>>>(
            processedTileCount, topology::detail::BuildUpperNodesFunctor<BuildT>(), deviceData(), deviceProcessedRoot());
        cudaCheckError();
    }
}// TopologyBuilder<BuildT>::processUpperNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
inline void TopologyBuilder<BuildT>::processLowerNodes(cudaStream_t stream)
{
    // Fill out the contents of all newly allocated lower nodes (using the densified upper/lower mask arrays)
    // Also fill in the preamble (most of LeafData) for their leaf children
    auto processedTileCount = hostProcessedRoot()->tileCount();
    using CountType = uint32_t (*)[Mask<5>::SIZE];
 
    if (processedTileCount) { // Unless output grid is empty
        int device = 0;
        cudaGetDevice(&device);
        std::size_t lowerCount = data()->nodeCount[1];
        mLowerParents = nanovdb::cuda::DeviceBuffer::create(lowerCount*sizeof(uint32_t), nullptr, device, stream);
        std::size_t leafCount = data()->nodeCount[0];
        mLeafParents = nanovdb::cuda::DeviceBuffer::create(leafCount*sizeof(uint32_t), nullptr, device, stream);

        using Op = util::morphology::cuda::ProcessLowerNodesFunctor<BuildT>;
        util::cuda::operatorKernel<Op>
            <<<dim3(processedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, stream>>>(
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
    }

    mProcessedRoot.clear(stream);
    mUpperMasks.clear(stream);
    mLowerMasks.clear(stream);
    mLowerOffsets.clear(stream);
    mLeafOffsets.clear(stream);
}// TopologyBuilder<BuildT>::processLowerNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct UpdateLeafVoxelCountsAndPrefixSumFunctor
{
    __device__
    void operator()(size_t leafID, typename TopologyBuilder<BuildT>::Data *d_data, uint64_t *d_voxelCounts) {
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
    void operator()(size_t leafID, typename TopologyBuilder<BuildT>::Data *d_data, uint64_t *d_voxelOffsets) {
        auto &leaf = d_data->getGrid().tree().template getFirstNode<0>()[leafID];
        leaf.mOffset = d_voxelOffsets[leafID]+1; }
};

}// namespace topology::detail

template<typename BuildT>
inline void TopologyBuilder<BuildT>::processLeafOffsets(cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    std::size_t leafCount = data()->nodeCount[0];
    if (leafCount) { // Unless output grid is empty
        mVoxelOffsets = nanovdb::cuda::DeviceBuffer::create((leafCount+1)*sizeof(uint64_t), nullptr, device, stream);
        cudaCheck(cudaMemsetAsync(mVoxelOffsets.deviceData(), 0, sizeof(uint64_t), stream));
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, stream>>>(
            leafCount, topology::detail::UpdateLeafVoxelCountsAndPrefixSumFunctor<BuildT>(), deviceData(), static_cast<uint64_t*>(mVoxelOffsets.deviceData())+1);
        CALL_CUBS(DeviceScan::InclusiveSum,
            static_cast<uint64_t*>(mVoxelOffsets.deviceData())+1,
            static_cast<uint64_t*>(mVoxelOffsets.deviceData())+1,
            leafCount);
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, stream>>>(
            leafCount, topology::detail::UpdateLeafVoxelOffsetsFunctor<BuildT>(), deviceData(), static_cast<uint64_t*>(mVoxelOffsets.deviceData()));
    }
}// TopologyBuilder<BuildT>::processLeafOffsets

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
    void operator()(size_t tid, typename TopologyBuilder<BuildT>::Data *d_data, const uint32_t* leafParents) {
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
    void operator()(size_t tid, typename TopologyBuilder<BuildT>::Data *d_data, const uint32_t* lowerParents) {
        auto &upper = d_data->getUpper(lowerParents[tid]);
        auto &lower = d_data->getLower(tid);
        upper.mBBox.expandAtomic(lower.bbox()); }
};

template <typename BuildT>
struct PropagateUpperBBoxFunctor
{
    __device__
    void operator()(size_t tid, typename TopologyBuilder<BuildT>::Data *d_data) {
        d_data->getRoot().mBBox.expandAtomic(d_data->getUpper(tid).bbox());
    }
};

template <typename BuildT>
struct UpdateRootWorldBBoxFunctor
{
    __device__
    void operator()(size_t tid, typename TopologyBuilder<BuildT>::Data *d_data) {
        // TODO: check that the correct semantics are followed in this transformation
        auto BBox = d_data->getRoot().mBBox;
        BBox.max() += 1;
        d_data->getGrid().mFlags.setMaskOn(GridFlags::HasBBox);
        d_data->getGrid().mWorldBBox = BBox.transform(d_data->getGrid().data()->mMap);
    }
};

}// namespace topology::detail

template <typename BuildT>
inline void TopologyBuilder<BuildT>::processBBox(cudaStream_t stream)
{
    if (data()->nodeCount[0] == 0) return; // Output grid is empty; retain empty bounding box

    // TODO: Do we need a special case when flags indicates no bounding box?

    // update and propagate bbox from leaf -> lower/parent nodes
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[0]), mNumThreads, 0, stream>>>(
        data()->nodeCount[0], topology::detail::UpdateAndPropagateLeafBBoxFunctor<BuildT>(), deviceData(), static_cast<uint32_t*>(mLeafParents.deviceData()));
    mLeafParents.clear(stream);
    cudaCheckError();

    // propagate bbox from lower -> upper/parent node
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[1]), mNumThreads, 0, stream>>>(
        data()->nodeCount[1], topology::detail::PropagateLowerBBoxFunctor<BuildT>(), deviceData(), static_cast<uint32_t*>(mLowerParents.deviceData()));
    mLowerParents.clear(stream);
    cudaCheckError();

    // propagate bbox from upper -> root/parent node
    util::cuda::lambdaKernel<<<numBlocks(data()->nodeCount[2]), mNumThreads, 0, stream>>>(data()->nodeCount[2], topology::detail::PropagateUpperBBoxFunctor<BuildT>(), deviceData());
    cudaCheckError();

    // update the world-bbox in the root node
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, topology::detail::UpdateRootWorldBBoxFunctor<BuildT>(), deviceData());
    cudaCheckError();
}// TopologyBuilder<BuildT>::processBBox

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct PostProcessGridTreeFunctor
{
    __device__
    void operator()(size_t tid, typename TopologyBuilder<BuildT>::Data *d_data, uint64_t* d_voxelOffsets) {
        auto& grid = d_data->getGrid();
        auto& tree = grid.tree();
        auto leafCount = tree.mNodeCount[0];
        tree.mVoxelCount = d_voxelOffsets[leafCount];
        grid.mData1 = tree.mVoxelCount+1;
    }
};

}// namespace topology::detail

template <typename BuildT>
inline void TopologyBuilder<BuildT>::postProcessGridTree(cudaStream_t stream)
{
    // Finish updates to GridData/TreeData and (optionally) update checksum
    if (data()->nodeCount[0]) // if grid is empty, the default values are correct
        util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, topology::detail::PostProcessGridTreeFunctor<BuildT>(), deviceData(), static_cast<uint64_t*>(mVoxelOffsets.deviceData()));
    cudaCheckError();
    mVoxelOffsets.clear(stream);

    tools::cuda::updateChecksum((GridData*)data()->d_bufferPtr, mChecksum, stream);
}// TopologyBuilder<BuildT>::postProcessGridTree

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_TOPOLOGYBUILDER_CUH_HAS_BEEN_INCLUDED
