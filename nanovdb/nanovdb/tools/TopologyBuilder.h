// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/TopologyBuilder.h

    \authors Efty Sifakis

    \brief Shared functionality of (mostly morphology) operators that alter the voxel content of grids

    \warning In this initial stage of the host-side port, this header still contains cuda device
             code; include it only from .cu files (or other .cuh files) until CUDA dependencies
             are progressively removed.
*/

#ifndef NANOVDB_TOOLS_TOPOLOGYBUILDER_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_TOPOLOGYBUILDER_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/util/Morphology.h>
#include <nanovdb/util/PrefixSum.h>
#include <nanovdb/util/cuda/Morphology.cuh>

namespace nanovdb {

namespace tools {

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

    // Storage policy for device-only scratch (mask/offset/parent arrays + the
    // local count buffers in countNodes()). UnifiedBuffer keeps kernels happy
    // while making the scratch host-visible without explicit deviceDownload,
    // which is the foundation needed before later phases replace kernel
    // launches with host loops. End state of the port is HostBuffer.
    // The dual-mode mProcessedRoot and mData remain DeviceBuffer for now
    // and are migrated separately (see TopologyCpuPortPlan.md §4.4).
    using ScratchBufferT = nanovdb::cuda::UnifiedBuffer;

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
    ScratchBufferT               mUpperMasks;
    ScratchBufferT               mLowerMasks;
    ScratchBufferT               mUpperOffsets;
    ScratchBufferT               mLowerOffsets;
    ScratchBufferT               mLeafOffsets;
    ScratchBufferT               mVoxelOffsets;
    ScratchBufferT               mLowerParents;
    ScratchBufferT               mLeafParents;
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
};// tools::TopologyBuilder<BuildT>

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
    mUpperMasks = ScratchBufferT::create(upperSize, nullptr, device, stream);
    if (mUpperMasks.deviceData() == nullptr) throw std::runtime_error("Failed to allocate upper mask buffer on device");
    cudaCheck(cudaMemsetAsync(mUpperMasks.deviceData(), 0, upperSize, stream));
    mLowerMasks = ScratchBufferT::create( lowerSize, nullptr, device, stream );
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
    //
    // Allocation strategy: only the offsets buffers (size N+1) are allocated. Each is laid out
    // so that offsets[0] is the seeded 0 and offsets[1..N] is the inclusive-scan output. We
    // therefore have the enumeration kernel and the upper-counts forEach write their per-element
    // count values *directly* into offsets+1, then run an in-place inclusive scan over the same
    // N-element region. Reading offsets from [0..N-1] gives the exclusive sum; reading from
    // [1..N] gives the inclusive sum; offsets[N] is the total.
    std::size_t size = processedTileCount*Mask<5>::SIZE;

    int device = 0;
    cudaGetDevice(&device);
    mUpperOffsets = ScratchBufferT::create((processedTileCount+1)*sizeof(uint32_t), nullptr, device, stream);
    mLowerOffsets = ScratchBufferT::create((size+1)*sizeof(uint32_t), nullptr, device, stream);
    mLeafOffsets  = ScratchBufferT::create((size+1)*sizeof(uint32_t), nullptr, device, stream);

    using CountType = uint32_t (*)[Mask<5>::SIZE];
    auto upperOffsets          = static_cast<uint32_t*>(mUpperOffsets.data());
    auto lowerOffsets          = static_cast<CountType>(mLowerOffsets.data());
    auto leafOffsets           = static_cast<uint32_t*>(mLeafOffsets.data());
    auto lowerOffsetsFlattened = static_cast<uint32_t*>(mLowerOffsets.data());

    // Seed the leading zero of each offsets array (host writes; visible to the kernel via
    // the implicit barrier at kernel launch). The kernel never touches index [0].
    upperOffsets[0]          = 0;
    lowerOffsetsFlattened[0] = 0;
    leafOffsets[0]           = 0;

    // The counts that the enumeration kernel and the upper-counts forEach produce land directly
    // in the offsets+1 region; the subsequent in-place inclusive scan turns them into offsets.
    auto upperCounts = upperOffsets + 1;
    auto lowerCounts = reinterpret_cast<CountType>(lowerOffsetsFlattened + 1);
    auto leafCounts  = reinterpret_cast<CountType>(leafOffsets + 1);

    // Drain upstream CUDA writes to mUpperMasks/mLowerMasks (cudaMemsetAsync zero-fill
    // in allocateInternalMaskBuffers + the operator-specific *InternalNodes kernel)
    // before host code reads them.
    cudaCheck(cudaStreamSynchronize(stream));

    util::morphology::EnumerateNodes(
        deviceUpperMasks(), deviceLowerMasks(),
        lowerCounts, leafCounts,
        processedTileCount);

    // In-place inclusive prefix sums (TBB-backed when NANOVDB_USE_TBB is defined).
    util::inclusiveScan(lowerOffsetsFlattened + 1, size, 0u, /*threaded=*/true, std::plus<uint32_t>{});
    util::inclusiveScan(leafOffsets + 1,           size, 0u, /*threaded=*/true, std::plus<uint32_t>{});
    data()->nodeCount[1] = lowerOffsetsFlattened[size];
    data()->nodeCount[0] = leafOffsets[size];

    // Host-side derivation of upperCounts from the (just-scanned) lower offsets.
    util::forEach(0, processedTileCount, 1, [=](const util::Range1D &r) {
        for (auto tileID = r.begin(); tileID != r.end(); ++tileID)
            upperCounts[tileID] = (lowerOffsets[tileID+1][0] > lowerOffsets[tileID][0]) ? 1 : 0;
    });

    util::inclusiveScan(upperOffsets + 1, processedTileCount, 0u, /*threaded=*/true, std::plus<uint32_t>{});
    data()->nodeCount[2] = upperOffsets[processedTileCount];
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

template <typename BuildT>
inline void TopologyBuilder<BuildT>::processUpperNodes(cudaStream_t stream)
{
    // Connect all newly allocated upper nodes to their respective tiles
    // Also fill in any necessary part of the preamble (in InternalData) of upper nodes
    auto processedTileCount = hostProcessedRoot()->tileCount();
    if (processedTileCount == 0) return; // output grid is empty

    // Drain upstream CUDA work (cudaMemsetAsync zero-fill of the output grid buffer +
    // deviceUpload of mData in getBuffer) so the host code below can read/write through
    // these regions safely.
    cudaCheck(cudaStreamSynchronize(stream));

    auto data          = this->data();
    auto processedRoot = hostProcessedRoot();
    auto &root         = data->getRoot();

    util::forEach(0, processedTileCount, 1, [=, &root](const util::Range1D &r) {
        for (auto processedTileID = r.begin(); processedTileID != r.end(); ++processedTileID) {
            const uint32_t tileID = data->d_upperOffsets[processedTileID];
            // If the offsets are the same, this was a speculatively introduced tile which was
            // not necessary.
            if (tileID != data->d_upperOffsets[processedTileID+1]) {
                auto &dstUpper      = data->getUpper(tileID);
                auto &processedTile = *processedRoot->tile(processedTileID);
                root.tile(tileID)->setChild(processedTile.origin(), &dstUpper, &root);
                // mBBox will be further updated after the operation has been applied at leaf-level.
                dstUpper.mBBox = CoordBBox();
                // TODO: Is this accurate? Any other flags that should be set?
                dstUpper.mFlags = (uint64_t)GridFlags::HasBBox;
            }
        }
    });
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
        mLowerParents = ScratchBufferT::create(lowerCount*sizeof(uint32_t), nullptr, device, stream);
        std::size_t leafCount = data()->nodeCount[0];
        mLeafParents = ScratchBufferT::create(leafCount*sizeof(uint32_t), nullptr, device, stream);

        util::morphology::ProcessLowerNodes<BuildT>(
            mUpperMasks.data(),
            mLowerMasks.data(),
            static_cast<uint32_t*>(mUpperOffsets.data()),
            static_cast<CountType>(mLowerOffsets.data()),
            static_cast<CountType>(mLeafOffsets.data()),
            static_cast<GridT*>(data()->d_bufferPtr),
            static_cast<uint32_t*>(mLowerParents.data()),
            static_cast<uint32_t*>(mLeafParents.data()),
            processedTileCount
        );
    }

    mProcessedRoot.clear(stream);
    mUpperMasks.clear();
    mLowerMasks.clear();
    mLowerOffsets.clear();
    mLeafOffsets.clear();
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
        mVoxelOffsets = ScratchBufferT::create((leafCount+1)*sizeof(uint64_t), nullptr, device, stream);
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
    __hostdev__
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
    __hostdev__
    void operator()(size_t tid, typename TopologyBuilder<BuildT>::Data *d_data, const uint32_t* lowerParents) {
        auto &upper = d_data->getUpper(lowerParents[tid]);
        auto &lower = d_data->getLower(tid);
        upper.mBBox.expandAtomic(lower.bbox()); }
};

template <typename BuildT>
struct PropagateUpperBBoxFunctor
{
    __hostdev__
    void operator()(size_t tid, typename TopologyBuilder<BuildT>::Data *d_data) {
        d_data->getRoot().mBBox.expandAtomic(d_data->getUpper(tid).bbox());
    }
};

template <typename BuildT>
struct UpdateRootWorldBBoxFunctor
{
    __hostdev__
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

    // Drain upstream CUDA writes (the operator-specific *LeafNodes kernel populates the leaf
    // value masks in the managed output buffer) before host code reads them below.
    cudaCheck(cudaStreamSynchronize(stream));

    auto d_data       = data();
    auto leafParents  = static_cast<uint32_t*>(mLeafParents.data());
    auto lowerParents = static_cast<uint32_t*>(mLowerParents.data());

    // update and propagate bbox from leaf -> lower/parent nodes. Concurrent children of a
    // shared parent race into parent.mBBox via expandAtomic (now host-callable). The forEach
    // barrier between levels ensures each level's bboxes are final before the next reads them.
    util::forEach(0, d_data->nodeCount[0], 1, [=](const util::Range1D &r) {
        topology::detail::UpdateAndPropagateLeafBBoxFunctor<BuildT> op;
        for (auto tid = r.begin(); tid != r.end(); ++tid) op(tid, d_data, leafParents);
    });
    mLeafParents.clear();

    // propagate bbox from lower -> upper/parent node
    util::forEach(0, d_data->nodeCount[1], 1, [=](const util::Range1D &r) {
        topology::detail::PropagateLowerBBoxFunctor<BuildT> op;
        for (auto tid = r.begin(); tid != r.end(); ++tid) op(tid, d_data, lowerParents);
    });
    mLowerParents.clear();

    // propagate bbox from upper -> root/parent node
    util::forEach(0, d_data->nodeCount[2], 1, [=](const util::Range1D &r) {
        topology::detail::PropagateUpperBBoxFunctor<BuildT> op;
        for (auto tid = r.begin(); tid != r.end(); ++tid) op(tid, d_data);
    });

    // update the world-bbox in the root node (single element)
    topology::detail::UpdateRootWorldBBoxFunctor<BuildT>()(0, d_data);
}// TopologyBuilder<BuildT>::processBBox

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct PostProcessGridTreeFunctor
{
    __hostdev__
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
    if (data()->nodeCount[0]) { // if grid is empty, the default values are correct
        // Drain upstream device writes to mVoxelOffsets (processLeafOffsets) before host read.
        cudaCheck(cudaStreamSynchronize(stream));
        topology::detail::PostProcessGridTreeFunctor<BuildT>()(0, data(), static_cast<uint64_t*>(mVoxelOffsets.data()));
    }
    mVoxelOffsets.clear();

    tools::cuda::updateChecksum((GridData*)data()->d_bufferPtr, mChecksum, stream);
}// TopologyBuilder<BuildT>::postProcessGridTree

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools

}// namespace nanovdb

#endif // NANOVDB_TOOLS_TOPOLOGYBUILDER_H_HAS_BEEN_INCLUDED
