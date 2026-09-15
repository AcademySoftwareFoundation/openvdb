// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/TopologyBuilder.cuh

    \authors Efty Sifakis

    \brief Shared functionality of (mostly morphology) operators that alter the voxel content of grids

    \details The builder constructs one grid, or a batch of B independent grids laid out back to
             back in one buffer (grid g carries mGridIndex = g and mGridCount = B). Every scratch
             array is batch-wide and indexed by processed root tile or by node; each grid records
             where its tiles and nodes start (TopologyBuilderData::tileBase, upperBase, lowerBase,
             leafBase) so the build kernels can rebase into that grid's own node arrays. With one
             grid every base is zero and the builder behaves exactly as the single-grid consumers
             expect.

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
#include <nanovdb/cuda/HandleStorage.h>
#include <nanovdb/cuda/PinnedResource.h> // for the pinned host staging of the processed roots and builder parameters

#include <algorithm> // for std::fill_n
#include <cstddef> // for std::byte, std::size_t
#include <cstring> // for std::memset
#include <map> // for the processed tile maps
#include <stdexcept> // for std::runtime_error
#include <vector>

namespace nanovdb {

namespace tools::cuda {

/// @brief Per-grid offsets, node counts and batch position handed to the device functors.
///        Independent of the resource the builder allocates from, so it lives outside
///        TopologyBuilder and stays one type across every ResourceT instantiation.
template <typename BuildT>
struct TopologyBuilderData {
    void     *d_bufferPtr;// start of this grid in the output buffer
    uint64_t grid, tree, root, upper, lower, leaf, size;// byte offsets to nodes, relative to d_bufferPtr
    uint32_t nodeCount[3];// 0=leaf,1=lower, 2=upper
    uint32_t *d_upperOffsets;// batch-wide, indexed by processed tile; shared by every grid
    uint32_t gridIndex, gridCount;// position of this grid in the batch (0 of 1 for a single grid)
    uint32_t tileCount, tileBase;// this grid's processed tiles, and its first index into the batch-wide tile arrays
    uint32_t upperBase, lowerBase, leafBase;// this grid's first index into the batch-wide node arrays
    uint64_t processedRootOffset;// byte offset of this grid's processed root in the staging buffer
    __hostdev__ NanoGrid<BuildT>&  getGrid() const {return *util::PtrAdd<NanoGrid<BuildT>>(d_bufferPtr, grid);}
    __hostdev__ NanoTree<BuildT>&  getTree() const {return *util::PtrAdd<NanoTree<BuildT>>(d_bufferPtr, tree);}
    __hostdev__ NanoRoot<BuildT>&  getRoot() const {return *util::PtrAdd<NanoRoot<BuildT>>(d_bufferPtr, root);}
    __hostdev__ NanoUpper<BuildT>& getUpper(int i) const {return *(util::PtrAdd<NanoUpper<BuildT>>(d_bufferPtr, upper)+i);}
    __hostdev__ NanoLower<BuildT>& getLower(int i) const {return *(util::PtrAdd<NanoLower<BuildT>>(d_bufferPtr, lower)+i);}
    __hostdev__ NanoLeaf<BuildT>&  getLeaf(int i) const {return *(util::PtrAdd<NanoLeaf<BuildT>>(d_bufferPtr, leaf)+i);}
};// TopologyBuilderData

namespace topology::detail {

/// @brief Finds the grid that owns batch-wide node index @a index, given the per-grid
///        base offsets at member @a base. Returns the last grid whose base does not exceed
///        the index; empty grids share their successor's base and are skipped that way.
template <typename BuildT>
__device__ inline uint32_t gridOfIndex(const TopologyBuilderData<BuildT> *d_data,
                                       uint32_t gridCount,
                                       uint32_t index,
                                       uint32_t TopologyBuilderData<BuildT>::*base)
{
    uint32_t lo = 0, hi = gridCount;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (d_data[mid].*base <= index) lo = mid + 1; else hi = mid;
    }
    return lo - 1;
}

/// @brief Sort key of a root tile: the offset-shifted encoding PointsToGrid orders tiles by,
///        so tiles sort by coordinate with negative coordinates first. It differs from the key
///        RootData::CoordToKey stores in the tile; consumers must use this one for ordering and
///        the stored one for the tile itself.
__hostdev__ inline uint64_t tileSortKey(const Coord &ijk)
{
    // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
    static constexpr int64_t kOffset = int64_t(1) << 31;
    return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
           (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
           (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);  // x is the upper 21 bits
}

/// @brief Speculative root tiles in canonical order, keyed by tileSortKey
template <typename RootT>
using ProcessedTileMap = std::map<uint64_t, typename RootT::DataType::Tile>;

/// @brief Adds the root tile containing @a ijk. Only the key is set; the child offset and
///        value are filled by the builder once the tile is known to be non-empty.
template <typename RootT>
inline void insertProcessedTile(ProcessedTileMap<RootT> &tiles, const Coord &ijk)
{
    tiles.emplace(tileSortKey(ijk), typename RootT::DataType::Tile{RootT::CoordToKey(ijk)});
}

/// @brief Adds every root tile that overlaps @a bbox (an empty bbox adds nothing).
template <typename RootT>
inline void insertProcessedTiles(ProcessedTileMap<RootT> &tiles, const CoordBBox &bbox)
{
    if (bbox.empty()) return;
    static constexpr int32_t kLog2Dim = RootT::ChildNodeType::TOTAL;// 12: root tiles are 4096^3
    const Coord lo = bbox.min() >> kLog2Dim, hi = bbox.max() >> kLog2Dim;// arithmetic shift floors negatives
    for (int32_t i = lo[0]; i <= hi[0]; ++i)
    for (int32_t j = lo[1]; j <= hi[1]; ++j)
    for (int32_t k = lo[2]; k <= hi[2]; ++k)
        insertProcessedTile<RootT>(tiles, Coord(i, j, k) << kLog2Dim);
}

/// @brief Writes the tile table of a processed root allocated with RootT::memUsage(tiles.size())
template <typename RootT>
inline void packProcessedRoot(const ProcessedTileMap<RootT> &tiles, RootT *root)
{
    root->mTableSize = static_cast<uint32_t>(tiles.size());
    uint32_t t = 0;
    for (const auto &[key, tile] : tiles) *root->tile(t++) = tile;
}

}// namespace topology::detail

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
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= NANOVDB_DATA_ALIGNMENT,
                  "TopologyBuilder stages the processed root as bytes and reinterprets it as a root node, which requires NANOVDB_DATA_ALIGNMENT-aligned allocations");

    /// @brief Device-only scratch storage, borrowing the injected resource
    ///        through a ResourceRef so all traffic reaches the caller's
    ///        instance (which may be stateful) rather than a copy. These
    ///        buffers are never read on the host, so they use the single-space
    ///        Buffer rather than the dual DeviceBuffer, whose host pointer and
    ///        per-device array they would leave unused.
    using ScratchT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>;
    template<typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;
    using UpperMaskBufT = BufT<Mask<5>>;
    using LowerMaskBufT = BufT<Mask<4>>;
    using HostStagingT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>;

public:

    using Data = TopologyBuilderData<BuildT>;

private:
    using HostDataT = nanovdb::cuda::Buffer<Data, nanovdb::cuda::PinnedResource>;

public:

    /// @param stream cuda stream the scratch allocations are ordered on
    /// @param resource resource instance all device scratch is allocated from;
    ///        must outlive this builder
    TopologyBuilder(cudaStream_t stream, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mDeviceRoot(stream, resource, 0, nanovdb::cuda::noInit)
        , mUpperMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mUpperOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mVoxelOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerParents(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafParents(stream, resource, 0, nanovdb::cuda::noInit)
        , mTileToGrid(stream, resource, 0, nanovdb::cuda::noInit)
        , mHostData(1, nanovdb::cuda::noInit)
        , mDeviceData(stream, resource, 0, nanovdb::cuda::noInit)
        , mResource(&resource)
        , mTempDevicePool(resource)
    {
        this->resetHostData(1);
    }

    void allocateInternalMaskBuffers(cudaStream_t stream);

    void countNodes(cudaStream_t stream);

    template<typename BufferT>
    BufferT getBuffer(const BufferT &buffer, cudaStream_t stream);

    /// @brief Initializes the grid, tree and root headers of every grid in the batch. The
    ///        caller copies each source GridData header into place first (name and map are
    ///        kept); single-grid consumers may keep launching BuildGridTreeRootFunctor themselves.
    void processGridTreeRoot(cudaStream_t stream);

    void processUpperNodes(cudaStream_t stream);

    void processLowerNodes(cudaStream_t stream);

    void processLeafOffsets(cudaStream_t stream);

    void processBBox(cudaStream_t stream);

    void postProcessGridTree(cudaStream_t stream);

    HostStagingT                 mHostRoot; // host staging for the processed roots, back to back (pinned, so the upload is asynchronous)
    ScratchT                     mDeviceRoot; // device copy, made by uploadProcessedRoot
    UpperMaskBufT                mUpperMasks;
    LowerMaskBufT                mLowerMasks;
    BufT<uint32_t>               mUpperOffsets;
    BufT<uint32_t>               mLowerOffsets;
    BufT<uint32_t>               mLeafOffsets;
    BufT<uint64_t>               mVoxelOffsets;
    BufT<uint32_t>               mLowerParents;
    BufT<uint32_t>               mLeafParents;
    BufT<uint32_t>               mTileToGrid; // owning grid of every processed tile; left empty for a single grid
    HostDataT                    mHostData; // host side of the builder parameters, one per grid
    BufT<Data>                   mDeviceData; // device copy, made by uploadData
    CheckMode                    mChecksum{CheckMode::Disable};

    /// @brief Number of grids in the batch (1 unless allocateProcessedRoots was given more)
    uint32_t gridCount() const { return static_cast<uint32_t>(mHostData.size()); }

    RootT* deviceProcessedRoot(uint32_t g = 0) { return mDeviceRoot.empty() ? nullptr : util::PtrAdd<RootT>(mDeviceRoot.data(), this->data(g)->processedRootOffset); }
    RootT* hostProcessedRoot(uint32_t g = 0)   { return mHostRoot.empty()   ? nullptr : util::PtrAdd<RootT>(mHostRoot.data(), this->data(g)->processedRootOffset); }

    /// @brief Allocates (pinned) host staging for one processed root of @a bytes and returns it
    ///        for the caller to fill (including mTableSize); any previous roots are dropped and
    ///        the builder is reset to a single grid.
    RootT* allocateProcessedRoot(uint64_t bytes)
    {
        this->resetHostData(1);
        mHostRoot = HostStagingT(bytes, nanovdb::cuda::noInit);
        return this->hostProcessedRoot(0);
    }

    /// @brief Allocates (pinned) host staging for the processed roots of a batch, one root
    ///        of @a tileCounts[g] tiles per grid, and resets the builder to that batch size.
    ///        Each root's mTableSize is set; the caller fills the tiles, in tileSortKey
    ///        order, through hostProcessedRoot(g). Returns the root of grid 0.
    RootT* allocateProcessedRoots(const std::vector<uint32_t> &tileCounts)
    {
        const uint32_t count = static_cast<uint32_t>(tileCounts.size());
        if (count == 0) throw std::runtime_error("TopologyBuilder::allocateProcessedRoots requires at least one grid");
        this->resetHostData(count);
        uint64_t bytes = 0;
        for (uint32_t g = 0; g < count; ++g) {
            this->data(g)->processedRootOffset = bytes;
            bytes += RootT::memUsage(tileCounts[g]);// a multiple of NANOVDB_DATA_ALIGNMENT, so every root stays aligned
        }
        mHostRoot = HostStagingT(bytes, nanovdb::cuda::noInit);
        for (uint32_t g = 0; g < count; ++g) this->hostProcessedRoot(g)->mTableSize = tileCounts[g];
        return this->hostProcessedRoot(0);
    }

    /// @brief Copies the host-staged processed roots to the device, allocating
    ///        through the builder's resource when the device copy is missing
    ///        or too small, and records which grid owns each processed tile.
    void uploadProcessedRoot(cudaStream_t stream)
    {
        const uint32_t tileTotal = this->totalTileCount();
        if (mDeviceRoot.size() < mHostRoot.size())
            mDeviceRoot = ScratchT(stream, nanovdb::cuda::ResourceRef<ResourceT>(*mResource), mHostRoot.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpyAsync(mDeviceRoot.data(), mHostRoot.data(), mHostRoot.size(), cudaMemcpyHostToDevice, stream));
        if (this->gridCount() > 1 && tileTotal) {
            std::vector<uint32_t> tileToGrid(tileTotal);
            for (uint32_t g = 0; g < this->gridCount(); ++g)
                std::fill_n(tileToGrid.begin() + this->data(g)->tileBase, this->data(g)->tileCount, g);
            mTileToGrid = BufT<uint32_t>(stream, *mResource, tileTotal, nanovdb::cuda::noInit);
            // the source is pageable, so this copy completes before it returns and the vector may go out of scope
            cudaCheck(cudaMemcpyAsync(mTileToGrid.data(), tileToGrid.data(), tileTotal * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        }
    }

    /// @brief Copies the builder parameters of every grid to the device, allocating through
    ///        the builder's resource on first use or when the batch size changed.
    void uploadData(cudaStream_t stream)
    {
        if (mDeviceData.size() != this->gridCount())
            mDeviceData = BufT<Data>(stream, nanovdb::cuda::ResourceRef<ResourceT>(*mResource), this->gridCount(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpyAsync(mDeviceData.data(), mHostData.data(), mHostData.size_bytes(), cudaMemcpyHostToDevice, stream));
    }
    Mask<5>* deviceUpperMasks() { return mUpperMasks.data(); }
    /// @brief The densified lower masks: one row of Mask<5>::SIZE Mask<4> per upper node,
    ///        indexed [upper node][lower node offset]. The row shape is fixed here, beside the
    ///        allocation that defines it, so consumers never re-derive the stride.
    Mask<4> (*deviceLowerMasks())[Mask<5>::SIZE] { return reinterpret_cast<Mask<4>(*)[Mask<5>::SIZE]>(mLowerMasks.data()); }
    //@{
    /// @brief The lower and leaf node offsets viewed one row of Mask<5>::SIZE per upper node,
    ///        indexed [upper node][lower node offset]; the row shape is fixed here, beside the
    ///        allocation that defines it, so consumers never re-derive the stride
    uint32_t (*lowerOffsetRows())[Mask<5>::SIZE] { return reinterpret_cast<uint32_t(*)[Mask<5>::SIZE]>(mLowerOffsets.data()); }
    uint32_t (*leafOffsetRows())[Mask<5>::SIZE] { return reinterpret_cast<uint32_t(*)[Mask<5>::SIZE]>(mLeafOffsets.data()); }
    //@}
    /// @brief A borrowing reference to the builder's resource, for consumers
    ///        allocating sibling scratch from the same instance.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    /// @brief Builder parameters of grid @a g (grid 0 by default, the whole grid for single-grid builds)
    Data* data(uint32_t g = 0)  { return mHostData.data() + g; }
    /// @brief Device copy of the parameters of every grid, gridCount() entries
    Data* deviceData()          { return mDeviceData.data(); }

    /// @brief Processed tiles over the whole batch; also refreshes every grid's tileCount and tileBase
    uint32_t totalTileCount()
    {
        if (mHostRoot.empty()) return mTileTotal;// roots already released; keep the last layout
        uint32_t base = 0;
        for (uint32_t g = 0; g < this->gridCount(); ++g) {
            Data *d = this->data(g);
            d->tileCount = this->hostProcessedRoot(g)->tileCount();
            d->tileBase  = base;
            base += d->tileCount;
        }
        return mTileTotal = base;
    }

    /// @brief Nodes at @a level (0=leaf, 1=lower, 2=upper) over the whole batch
    uint32_t totalNodeCount(int level) const
    {
        uint32_t total = 0;
        for (uint32_t g = 0; g < this->gridCount(); ++g) total += mHostData.data()[g].nodeCount[level];
        return total;
    }

private:
    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    /// @brief Zeroes the host parameters of @a count grids and numbers them
    void resetHostData(uint32_t count)
    {
        if (mHostData.size() != count) mHostData = HostDataT(count, nanovdb::cuda::noInit);
        std::memset(mHostData.data(), 0, mHostData.size_bytes());
        for (uint32_t g = 0; g < count; ++g) {
            this->data(g)->gridIndex = g;
            this->data(g)->gridCount = count;
        }
        mTileTotal = 0;
    }

    const uint32_t* tileToGrid() const { return mTileToGrid.empty() ? nullptr : mTileToGrid.data(); }

    uint32_t                          mTileTotal{0};
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
    const uint32_t tileTotal = this->totalTileCount();
    if (tileTotal == 0) return; // Processing empty grid(s); nothing to allocate

    // Allocate (and zero-fill) the mask arrays:
    // (a) one Mask<5> per tile of the processed roots, and
    // (b) Mask<5>::SIZE Mask<4> per tile, as if every upper node had a full set of 32^3 lower children
    const uint64_t upperMaskCount = tileTotal;
    const uint64_t lowerMaskCount = upperMaskCount * Mask<5>::SIZE;
    mUpperMasks = UpperMaskBufT(stream, *mResource, upperMaskCount, nanovdb::cuda::noInit);
    if (mUpperMasks.data() == nullptr) throw std::runtime_error("Failed to allocate upper mask buffer on device");
    cudaCheck(cudaMemsetAsync(mUpperMasks.data(), 0, mUpperMasks.size_bytes(), stream));
    mLowerMasks = LowerMaskBufT(stream, *mResource, lowerMaskCount, nanovdb::cuda::noInit);
    if (mLowerMasks.data() == nullptr) throw std::runtime_error("Failed to allocate lower mask buffer on device");
    cudaCheck(cudaMemsetAsync(mLowerMasks.data(), 0, mLowerMasks.size_bytes(), stream));
}// TopologyBuilder<BuildT, ResourceT>::allocateInternalMaskBuffers

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

/// @brief Reads each grid's node counts and node bases off the batch-wide offset scans,
///        at the boundaries of that grid's processed tiles
template <typename BuildT>
struct GatherNodeCountsFunctor
{
    __device__
    void operator()(size_t g, TopologyBuilderData<BuildT> *d_data,
                    const uint32_t *upperOffsets, const uint32_t *lowerOffsets, const uint32_t *leafOffsets) {
        auto &d = d_data[g];
        const size_t t0 = d.tileBase, t1 = t0 + d.tileCount;
        constexpr size_t S = Mask<5>::SIZE;
        d.upperBase = upperOffsets[t0];
        d.lowerBase = lowerOffsets[t0 * S];
        d.leafBase  = leafOffsets[t0 * S];
        d.nodeCount[2] = upperOffsets[t1] - d.upperBase;
        d.nodeCount[1] = lowerOffsets[t1 * S] - d.lowerBase;
        d.nodeCount[0] = leafOffsets[t1 * S] - d.leafBase;
    }
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
void TopologyBuilder<BuildT, ResourceT>::countNodes(cudaStream_t stream)
{
    const uint32_t processedTileCount = this->totalTileCount();
    if (processedTileCount == 0) { // Processing empty grid(s); zero nodes at all levels
        for (uint32_t g = 0; g < this->gridCount(); ++g) {
            Data *d = this->data(g);
            d->nodeCount[0] = d->nodeCount[1] = d->nodeCount[2] = 0;
            d->upperBase = d->lowerBase = d->leafBase = 0;
        }
        return;
    }

    // Computes prefix sums of (a) non-empty lower nodes, (b) counts of their leaf children,
    // and (c) count of the speculatively updated root tiles that have actually been used.
    // These are used to reconstruct child offsets for the internal nodes of the updated tree,
    // as well as the tile table at the root. The scans run over the whole batch; each grid's
    // counts are the differences at its tile boundaries.
    std::size_t size = std::size_t(processedTileCount)*Mask<5>::SIZE;

    BufT<uint32_t> upperCountsBuffer = BufT<uint32_t>(stream, *mResource, processedTileCount, nanovdb::cuda::noInit);
    BufT<uint32_t> lowerCountsBuffer = BufT<uint32_t>(stream, *mResource, size, nanovdb::cuda::noInit);
    BufT<uint32_t> leafCountsBuffer = BufT<uint32_t>(stream, *mResource, size, nanovdb::cuda::noInit);

    using CountType = uint32_t (*)[Mask<5>::SIZE];
    auto lowerCounts = reinterpret_cast<CountType>(lowerCountsBuffer.data());
    auto leafCounts = reinterpret_cast<CountType>(leafCountsBuffer.data());

    using Op = util::morphology::cuda::EnumerateNodesFunctor;
    util::cuda::operatorKernel<Op>
        <<<dim3(processedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, stream>>>
        (deviceUpperMasks(), deviceLowerMasks(), lowerCounts, leafCounts);

    mUpperOffsets = BufT<uint32_t>(stream, *mResource, processedTileCount+1, nanovdb::cuda::noInit);
    mLowerOffsets = BufT<uint32_t>(stream, *mResource, size+1, nanovdb::cuda::noInit);
    mLeafOffsets = BufT<uint32_t>(stream, *mResource, size+1, nanovdb::cuda::noInit);

    cudaCheck(cudaMemsetAsync(mLowerOffsets.data(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        lowerCountsBuffer.data(),
        mLowerOffsets.data()+1,
        size);

    cudaCheck(cudaMemsetAsync(mLeafOffsets.data(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        leafCountsBuffer.data(),
        mLeafOffsets.data()+1,
        size);

    util::cuda::lambdaKernel<<<numBlocks(processedTileCount), mNumThreads, 0, stream>>>(
        processedTileCount,
        [] __device__(size_t tileID, CountType lowerOffsets, uint32_t* upperCounts)
            { upperCounts[tileID] = (lowerOffsets[tileID+1][0] > lowerOffsets[tileID][0]) ? 1 : 0; },
        lowerOffsetRows(),
        upperCountsBuffer.data());

    cudaCheck(cudaMemsetAsync( mUpperOffsets.data(), 0, sizeof(uint32_t), stream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        upperCountsBuffer.data(),
        mUpperOffsets.data()+1,
        processedTileCount);

    // One gather over the grids and one copy back replace per-level scalar readbacks;
    // the caller synchronizes before reading data()->nodeCount, as before.
    this->uploadData(stream);
    util::cuda::lambdaKernel<<<numBlocks(this->gridCount()), mNumThreads, 0, stream>>>(
        this->gridCount(), topology::detail::GatherNodeCountsFunctor<BuildT>(), deviceData(),
        mUpperOffsets.data(), mLowerOffsets.data(), mLeafOffsets.data());
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(mHostData.data(), mDeviceData.data(), mHostData.size_bytes(), cudaMemcpyDeviceToHost, stream));
}// TopologyBuilder<BuildT, ResourceT>::countNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT, typename ResourceT>
template <typename BufferT>
BufferT TopologyBuilder<BuildT, ResourceT>::getBuffer(const BufferT &pool, cudaStream_t stream)
{
    // Allocates one device buffer for the destination grids, once the topology/size of every tree is known.
    // Grids are laid out back to back; each grid's offsets below are relative to its own start.
    uint64_t totalSize = 0;
    std::vector<uint64_t> gridOffsets(this->gridCount());
    for (uint32_t g = 0; g < this->gridCount(); ++g) {
        Data *d = this->data(g);
        d->grid  = 0;// grid is always stored at the start of its section
        d->tree  = GridT::memUsage();// grid ends and tree begins
        d->root  = d->tree  + TreeT::memUsage(); // tree ends and root node begins
        d->upper = d->root  + RootT::memUsage(d->nodeCount[2]);// root node ends and upper internal nodes begin
        d->lower = d->upper + UpperT::memUsage()*d->nodeCount[2];// upper internal nodes ends and lower internal nodes begin
        d->leaf  = d->lower + LowerT::memUsage()*d->nodeCount[1];// lower internal nodes ends and leaf nodes begin
        d->size  = d->leaf  + LeafT::DataType::memUsage()*d->nodeCount[0];// leaf nodes end and blind meta data begins
        gridOffsets[g] = totalSize;
        totalSize += d->size;// every node type has NANOVDB_DATA_ALIGNMENT-sized memUsage, so the next grid stays aligned
    }

    int device = 0;
    cudaGetDevice(&device);
    auto buffer = nanovdb::cuda::detail::createDeviceStorage<BufferT>(totalSize, &pool, device, stream); // only allocate buffer on the device
    void *base = nanovdb::cuda::detail::deviceStorageData(buffer);
    if (base == nullptr) throw std::runtime_error("Failed to allocate grid buffer on the device");
    cudaCheck(cudaMemsetAsync(base, 0, totalSize, stream));

    for (uint32_t g = 0; g < this->gridCount(); ++g) {
        Data *d = this->data(g);
        d->d_bufferPtr = util::PtrAdd(base, gridOffsets[g]);
        d->d_upperOffsets = mUpperOffsets.data();// batch-wide; never read for a grid without tiles
    }
    this->uploadData(stream);

    return buffer;
}// TopologyBuilder<BuildT, ResourceT>::getBuffer

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct BuildGridTreeRootFunctor
{
    __device__
    void operator()(size_t g, TopologyBuilderData<BuildT> *d_data) {
        const auto &d = d_data[g];

        // process Root
        auto &root = d.getRoot();
        root.mTableSize = d.nodeCount[2];
        root.mBackground = NanoRoot<BuildT>::ValueType(0);// background_value
        root.mMinimum = root.mMaximum = NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = NanoRoot<BuildT>::FloatType(0);
        root.mBBox = CoordBBox(); // To be further updated after the leaf-level voxel update

        // process Tree
        auto &tree = d.getTree();
        tree.setRoot(&root);
        if (d.nodeCount[2]) {
            tree.setFirstNode(&d.getUpper(0));
            tree.setFirstNode(&d.getLower(0));
            tree.setFirstNode(&d.getLeaf(0));
        }
        else {
            tree.template setFirstNode<NanoUpper<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLower<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLeaf<BuildT>>(nullptr);
        }
        tree.mNodeCount[2] = d.nodeCount[2];
        tree.mNodeCount[1] = d.nodeCount[1];
        tree.mNodeCount[0] = d.nodeCount[0];
        tree.mVoxelCount = 0; // Actual voxel count (for non-empty grids) will only be known
                              // once leaf masks have been processed
        tree.mTileCount[2] = tree.mTileCount[1] =  tree.mTileCount[0] = 0;

        // process Grid
        // The GridData header has already been copied from the input;
        // reset what is necessary, and assert that others are at the expected values
        auto &grid = d.getGrid();

#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        NANOVDB_ASSERT(grid.mMagic == NANOVDB_MAGIC_GRID);
#else
        NANOVDB_ASSERT(grid.mMagic == NANOVDB_MAGIC_NUMB);
#endif
        grid.mChecksum.disable(); // all 64 bits ON means checksum is disabled
        NANOVDB_ASSERT(grid.mVersion == Version());
        NANOVDB_ASSERT(grid.mFlags.isMaskOn(GridFlags::IsBreadthFirst));
        grid.mFlags.initMask({GridFlags::IsBreadthFirst}); // expected flags (HasBBox will be set later if grid is non-empty)
        grid.mGridIndex = d.gridIndex; // Possibly overwriting input; the returned grid takes its place in the batch
        grid.mGridCount = d.gridCount;
        grid.mGridSize = d.size;
        // grid.mGridName expected to have been copied verbatim from input
        // grid.mMap expected to have been copied verbatim from input
        grid.mWorldBBox = Vec3dBBox();// invalid bbox
        grid.mVoxelSize = grid.mMap.getVoxelSize();
        NANOVDB_ASSERT(grid.mGridClass == GridClass::IndexGrid);
        NANOVDB_ASSERT(grid.mGridType == toGridType<BuildT>());
        grid.mBlindMetadataOffset = d.size; // i.e. no blind data, even if the input grid had any
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
    void operator()(size_t g, TopologyBuilderData<BuildT> *d_data) {
        const auto &d = d_data[g];

        // process Root (identical to BuildGridTreeRootFunctor)
        auto &root = d.getRoot();
        root.mTableSize = d.nodeCount[2];
        root.mBackground = NanoRoot<BuildT>::ValueType(0);
        root.mMinimum = root.mMaximum = NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = NanoRoot<BuildT>::FloatType(0);
        root.mBBox = CoordBBox();

        // process Tree (identical to BuildGridTreeRootFunctor)
        auto &tree = d.getTree();
        tree.setRoot(&root);
        if (d.nodeCount[2]) {
            tree.setFirstNode(&d.getUpper(0));
            tree.setFirstNode(&d.getLower(0));
            tree.setFirstNode(&d.getLeaf(0));
        } else {
            tree.template setFirstNode<NanoUpper<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLower<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLeaf<BuildT>>(nullptr);
        }
        tree.mNodeCount[2] = d.nodeCount[2];
        tree.mNodeCount[1] = d.nodeCount[1];
        tree.mNodeCount[0] = d.nodeCount[0];
        tree.mVoxelCount = 0;
        tree.mTileCount[2] = tree.mTileCount[1] = tree.mTileCount[0] = 0;

        // process Grid — set all fields explicitly (no source grid to copy from)
        auto &grid = d.getGrid();
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        grid.mMagic = NANOVDB_MAGIC_GRID;
#else
        grid.mMagic = NANOVDB_MAGIC_NUMB;
#endif
        grid.mChecksum.disable();
        grid.mVersion = Version();
        grid.mFlags.initMask({GridFlags::IsBreadthFirst});
        grid.mGridIndex = d.gridIndex;
        grid.mGridCount = d.gridCount;
        grid.mGridSize = d.size;
        // grid.mGridName is left zeroed; caller copies name via cudaMemcpyAsync
        grid.mMap = map;
        grid.mWorldBBox = Vec3dBBox();
        grid.mVoxelSize = map.getVoxelSize();
        grid.mGridClass = GridClass::IndexGrid;
        grid.mGridType = toGridType<BuildT>();
        grid.mBlindMetadataOffset = d.size;
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

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processGridTreeRoot(cudaStream_t stream)
{
    util::cuda::lambdaKernel<<<numBlocks(this->gridCount()), mNumThreads, 0, stream>>>(
        this->gridCount(), topology::detail::BuildGridTreeRootFunctor<BuildT>(), deviceData());
    cudaCheckError();
}// TopologyBuilder<BuildT, ResourceT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct BuildUpperNodesFunctor
{
    __device__
    void operator()(size_t processedTileID, TopologyBuilderData<BuildT> *d_data,
                    void *d_processedRoots, const uint32_t *tileToGrid) {
        const auto &d = d_data[tileToGrid ? tileToGrid[processedTileID] : 0u];
        uint32_t tileID = d.d_upperOffsets[processedTileID];
        if (tileID != d.d_upperOffsets[processedTileID+1]) // if the offsets are the same, this was a speculatively introduced tile which was not necessary
        {
            tileID -= d.upperBase;// this grid's own upper node index
            auto &root  = d.getRoot();
            auto &dstUpper = d.getUpper(tileID);
            auto *processedRoot = util::PtrAdd<NanoRoot<BuildT>>(d_processedRoots, d.processedRootOffset);
            auto &processedTile = *processedRoot->tile(processedTileID - d.tileBase);
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
    const uint32_t processedTileCount = this->totalTileCount();

    if (processedTileCount) { // Unless output grid is empty
        util::cuda::lambdaKernel<<<numBlocks(processedTileCount), mNumThreads, 0, stream>>>(
            processedTileCount, topology::detail::BuildUpperNodesFunctor<BuildT>(), deviceData(),
            static_cast<void*>(mDeviceRoot.data()), this->tileToGrid());
        cudaCheckError();
    }
}// TopologyBuilder<BuildT, ResourceT>::processUpperNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processLowerNodes(cudaStream_t stream)
{
    // Fill out the contents of all newly allocated lower nodes (using the densified upper/lower mask arrays)
    // Also fill in the preamble (most of LeafData) for their leaf children
    const uint32_t processedTileCount = this->totalTileCount();

    if (processedTileCount) { // Unless output grid is empty
        std::size_t lowerCount = this->totalNodeCount(1);
        mLowerParents = BufT<uint32_t>(stream, *mResource, lowerCount, nanovdb::cuda::noInit);
        std::size_t leafCount = this->totalNodeCount(0);
        mLeafParents = BufT<uint32_t>(stream, *mResource, leafCount, nanovdb::cuda::noInit);

        using Op = util::morphology::cuda::ProcessLowerNodesFunctor<BuildT>;
        util::cuda::operatorKernel<Op>
            <<<dim3(processedTileCount, Op::SlicesPerUpperNode, 1), Op::MaxThreadsPerBlock, 0, stream>>>(
                deviceUpperMasks(),
                deviceLowerMasks(),
                mUpperOffsets.data(),
                lowerOffsetRows(),
                leafOffsetRows(),
                deviceData(),
                this->tileToGrid(),
                mLowerParents.data(),
                mLeafParents.data()
            );
        cudaCheckError();
    }

    mHostRoot.destroy();
    mDeviceRoot.destroy(stream);
    mUpperMasks.destroy(stream);
    mLowerMasks.destroy(stream);
    mLowerOffsets.destroy(stream);
    mLeafOffsets.destroy(stream);
    mTileToGrid.destroy(stream);
}// TopologyBuilder<BuildT, ResourceT>::processLowerNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct UpdateLeafVoxelCountsAndPrefixSumFunctor
{
    __device__
    void operator()(size_t leafID, TopologyBuilderData<BuildT> *d_data, uint32_t gridCount, uint64_t *d_voxelCounts) {
        const auto &d = d_data[gridOfIndex(d_data, gridCount, uint32_t(leafID), &TopologyBuilderData<BuildT>::leafBase)];
        auto &leaf = d.getLeaf(uint32_t(leafID) - d.leafBase);
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
    void operator()(size_t leafID, TopologyBuilderData<BuildT> *d_data, uint32_t gridCount, uint64_t *d_voxelOffsets) {
        const auto &d = d_data[gridOfIndex(d_data, gridCount, uint32_t(leafID), &TopologyBuilderData<BuildT>::leafBase)];
        auto &leaf = d.getLeaf(uint32_t(leafID) - d.leafBase);
        leaf.mOffset = d_voxelOffsets[leafID] - d_voxelOffsets[d.leafBase] + 1; }// offsets restart at 1 in every grid
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processLeafOffsets(cudaStream_t stream)
{
    std::size_t leafCount = this->totalNodeCount(0);
    if (leafCount) { // Unless output grid is empty
        mVoxelOffsets = BufT<uint64_t>(stream, *mResource, leafCount+1, nanovdb::cuda::noInit);
        cudaCheck(cudaMemsetAsync(mVoxelOffsets.data(), 0, sizeof(uint64_t), stream));
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, stream>>>(
            leafCount, topology::detail::UpdateLeafVoxelCountsAndPrefixSumFunctor<BuildT>(), deviceData(), this->gridCount(), mVoxelOffsets.data()+1);
        CALL_CUBS(DeviceScan::InclusiveSum,
            mVoxelOffsets.data()+1,
            mVoxelOffsets.data()+1,
            leafCount);
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, stream>>>(
            leafCount, topology::detail::UpdateLeafVoxelOffsetsFunctor<BuildT>(), deviceData(), this->gridCount(), mVoxelOffsets.data());
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
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data, uint32_t gridCount, const uint32_t* leafParents) {
        const auto &d = d_data[gridOfIndex(d_data, gridCount, uint32_t(tid), &TopologyBuilderData<BuildT>::leafBase)];
        auto &lower = d.getLower(leafParents[tid]);// parents are stored as the grid's own node indices
        auto &leaf = d.getLeaf(uint32_t(tid) - d.leafBase);
        leaf.updateBBox();
        lower.mBBox.expandAtomic(leaf.bbox());
    }
};

template <typename BuildT>
struct PropagateLowerBBoxFunctor
{
    __device__
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data, uint32_t gridCount, const uint32_t* lowerParents) {
        const auto &d = d_data[gridOfIndex(d_data, gridCount, uint32_t(tid), &TopologyBuilderData<BuildT>::lowerBase)];
        auto &upper = d.getUpper(lowerParents[tid]);
        auto &lower = d.getLower(uint32_t(tid) - d.lowerBase);
        upper.mBBox.expandAtomic(lower.bbox()); }
};

template <typename BuildT>
struct PropagateUpperBBoxFunctor
{
    __device__
    void operator()(size_t tid, TopologyBuilderData<BuildT> *d_data, uint32_t gridCount) {
        const auto &d = d_data[gridOfIndex(d_data, gridCount, uint32_t(tid), &TopologyBuilderData<BuildT>::upperBase)];
        d.getRoot().mBBox.expandAtomic(d.getUpper(uint32_t(tid) - d.upperBase).bbox());
    }
};

template <typename BuildT>
struct UpdateRootWorldBBoxFunctor
{
    __device__
    void operator()(size_t g, TopologyBuilderData<BuildT> *d_data) {
        const auto &d = d_data[g];
        if (d.nodeCount[0] == 0) return; // empty grid; retain empty bounding box
        // TODO: check that the correct semantics are followed in this transformation
        auto BBox = d.getRoot().mBBox;
        BBox.max() += 1;
        d.getGrid().mFlags.setMaskOn(GridFlags::HasBBox);
        d.getGrid().mWorldBBox = BBox.transform(d.getGrid().data()->mMap);
    }
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::processBBox(cudaStream_t stream)
{
    const uint32_t leafCount = this->totalNodeCount(0);
    if (leafCount == 0) return; // Output grid(s) empty; retain empty bounding boxes

    // TODO: Do we need a special case when flags indicates no bounding box?

    // update and propagate bbox from leaf -> lower/parent nodes
    util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, stream>>>(
        leafCount, topology::detail::UpdateAndPropagateLeafBBoxFunctor<BuildT>(), deviceData(), this->gridCount(), mLeafParents.data());
    mLeafParents.destroy(stream);
    cudaCheckError();

    // propagate bbox from lower -> upper/parent node
    const uint32_t lowerCount = this->totalNodeCount(1);
    util::cuda::lambdaKernel<<<numBlocks(lowerCount), mNumThreads, 0, stream>>>(
        lowerCount, topology::detail::PropagateLowerBBoxFunctor<BuildT>(), deviceData(), this->gridCount(), mLowerParents.data());
    mLowerParents.destroy(stream);
    cudaCheckError();

    // propagate bbox from upper -> root/parent node
    const uint32_t upperCount = this->totalNodeCount(2);
    util::cuda::lambdaKernel<<<numBlocks(upperCount), mNumThreads, 0, stream>>>(
        upperCount, topology::detail::PropagateUpperBBoxFunctor<BuildT>(), deviceData(), this->gridCount());
    cudaCheckError();

    // update the world-bbox in the root node of every grid
    util::cuda::lambdaKernel<<<numBlocks(this->gridCount()), mNumThreads, 0, stream>>>(
        this->gridCount(), topology::detail::UpdateRootWorldBBoxFunctor<BuildT>(), deviceData());
    cudaCheckError();
}// TopologyBuilder<BuildT, ResourceT>::processBBox

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct PostProcessGridTreeFunctor
{
    __device__
    void operator()(size_t g, TopologyBuilderData<BuildT> *d_data, uint64_t* d_voxelOffsets) {
        const auto &d = d_data[g];
        if (d.nodeCount[0] == 0) return; // empty grid; the default values are correct
        auto& grid = d.getGrid();
        auto& tree = grid.tree();
        tree.mVoxelCount = d_voxelOffsets[d.leafBase + d.nodeCount[0]] - d_voxelOffsets[d.leafBase];
        grid.mData1 = tree.mVoxelCount+1;
    }
};

}// namespace topology::detail

template<typename BuildT, typename ResourceT>
inline void TopologyBuilder<BuildT, ResourceT>::postProcessGridTree(cudaStream_t stream)
{
    // Finish updates to GridData/TreeData and (optionally) update checksums
    if (this->totalNodeCount(0)) // if every grid is empty, the default values are correct
        util::cuda::lambdaKernel<<<numBlocks(this->gridCount()), mNumThreads, 0, stream>>>(
            this->gridCount(), topology::detail::PostProcessGridTreeFunctor<BuildT>(), deviceData(), mVoxelOffsets.data());
    cudaCheckError();
    mVoxelOffsets.destroy(stream);

    for (uint32_t g = 0; g < this->gridCount(); ++g)
        tools::cuda::updateChecksum((GridData*)this->data(g)->d_bufferPtr, mChecksum, stream);
}// TopologyBuilder<BuildT, ResourceT>::postProcessGridTree

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_TOPOLOGYBUILDER_CUH_HAS_BEEN_INCLUDED
