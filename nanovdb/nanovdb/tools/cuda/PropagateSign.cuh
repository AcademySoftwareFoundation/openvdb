// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/PropagateSign.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Propagates active-voxel signs into the inactive regions of a NanoVDB index grid.

    \warning This header contains CUDA device code and must be included from a .cu or .cuh file.
*/

#ifndef NVIDIA_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

#include <cstddef>   // std::size_t
#include <cstdint>   // int8_t, uint8_t, uint32_t, uint64_t
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <tuple>     // std::tuple
#include <utility>   // std::move
#include <vector>    // std::vector

namespace nanovdb {

namespace tools::cuda {

/// @brief Propagate the signs of active voxels through every inactive level of an index grid.
/// todo: Naming choice to discuss : interior masks vs invert masks
/// @details This is the index-grid counterpart of tools::cuda::signedFloodFill (SignedFloodFill.cuh), which
///          only works on float grids: it writes +/-background in place into the inactive values themselves.
///          An index grid (e.g. ValueOnIndex) has no inactive values to write to; its values live in
///          separate arrays indexed by the active voxels, and every inactive voxel shares the background
///          index. So instead of modifying the grid, PropagateSign takes the signs of the active voxels from
///          @a deviceSigns and returns, for every tree level, a mask of the inactive regions that are inside.
///
///          The result has one entry per tree level, each covering what that level owns:
///          - leaf masks: one Mask<3> per leaf; bit n is on if voxel n is inside (an active voxel with
///            sign -1, or an inactive voxel reached from one).
///          - lower and upper masks: one Mask per node; bit n is on if slot n has no child and is inside.
///          - root-tile origins: the inside root-tile positions that have no upper child.
/// @tparam BuildT NanoVDB index-grid build type.
/// @tparam ResourceT Stream-ordered resource used for all output and scratch allocations.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class PropagateSign
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PropagateSign allocates stream-ordered storage and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "PropagateSign stores uint64_t masks and requires word-aligned allocations");

    using GridT = NanoGrid<BuildT>;

    template <typename T>
    using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:
    /// @param deviceGrid Device index grid whose active values index @a deviceSigns.
    /// @param deviceSigns Per-active-value signs: -1 inside and +1 outside.
    /// @param resource Allocation resource; must outlive the returned buffers.
    /// @note @a deviceGrid and @a deviceSigns must be ready in @a stream order (if they were written on
    ///       another stream, make @a stream wait for it, e.g. with cudaStreamWaitEvent), and must remain
    ///       valid until propagate() returns.
    PropagateSign(const GridT* deviceGrid, const int8_t* deviceSigns,
                  cudaStream_t stream = 0,
                  ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream)
        , mTimer(stream)
        , mDeviceGrid(deviceGrid)
        , mDeviceSigns(deviceSigns)
        , mResource(&resource)
        , mLeafInteriorMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerInteriorMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mUpperInteriorMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mRootInteriorTileOrigins(stream, resource, 0, nanovdb::cuda::noInit)
        , mRootMask(stream, resource, 0, nanovdb::cuda::noInit)
    {}

    /// @brief Toggle timing output.
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Propagate interior (-1) sign and return the leaf, lower, upper, and root-tile outputs.
    /// @note The stream is synchronized before this method returns.
    std::tuple<BufferT<nanovdb::Mask<3>>,
               BufferT<nanovdb::Mask<4>>,
               BufferT<nanovdb::Mask<5>>,
               BufferT<nanovdb::Coord>> propagate()
    {
        if (mVerbose==1) mTimer.start("Allocating interior masks");
        initializeMasks();
        if (mVerbose==1) mTimer.stop();
        // Root propagation is needed when the active bounds contain positions without an Upper child.
        const bool needsRootPropagation = mRootMask.size() > mUpperInteriorMasks.size();

        if (mVerbose==1) mTimer.start("Leaf interior fill");
        fillLeafInteriorMasks();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Seeding coarser levels from leaf faces");
        emitSeedsToCoarserLevels<0>();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Lower interior fill");
        fillLowerInteriorMasks();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Seeding coarser levels from lower faces");
        emitSeedsToCoarserLevels<1>();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Upper interior fill");
        fillUpperInteriorMasks();
        if (mVerbose==1) mTimer.stop();

        if (needsRootPropagation) {
            if (mVerbose==1) mTimer.start("Seeding root tiles from upper faces");
            emitSeedsToCoarserLevels<2>();
            if (mVerbose==1) mTimer.stop();

            if (mVerbose==1) mTimer.start("Root interior fill");
            fillRootInteriorMasks();
            if (mVerbose==1) mTimer.stop();
        }

        cudaCheck(cudaStreamSynchronize(mStream));
        return {std::move(mLeafInteriorMasks), std::move(mLowerInteriorMasks),
                std::move(mUpperInteriorMasks), std::move(mRootInteriorTileOrigins)};
    }

private:

    // --- Pipeline stages (run in order by propagate). ---
    // Stages 6-7 run only when the active root bounds contain positions without Upper children.

    // Setup: allocate per-node masks and a dense root mask.
    void initializeMasks();

    // Stage 1: propagate active inside signs through inactive voxels within each leaf.
    void fillLeafInteriorMasks();

    // Stages 2, 4, 6: seed childless slots at coarser levels from the inside faces of
    // filled leaf (0), lower (1), or upper (2) nodes.
    // Here, we do not need to emit seed to same level.
    template <int Level>
    void emitSeedsToCoarserLevels();

    // Stage 3: fill childless slots within each lower node from its interior seeds.
    void fillLowerInteriorMasks();

    // Stage 5: fill childless slots within each upper node from its interior seeds.
    void fillUpperInteriorMasks();

    // Stage 7: fill root positions with Upper children as walls, then collect interior tile origins.
    void fillRootInteriorMasks();

    /// @brief Borrowed handle to the injected resource, for the scratch allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t       mStream{0};
    util::cuda::Timer  mTimer;
    int                mVerbose{0};
    const GridT*       mDeviceGrid{nullptr};
    const int8_t*      mDeviceSigns{nullptr};
    ResourceT*         mResource{nullptr};

    // output masks for each levels and coord list for root tile.
    BufferT<nanovdb::Mask<3>> mLeafInteriorMasks;
    BufferT<nanovdb::Mask<4>> mLowerInteriorMasks;
    BufferT<nanovdb::Mask<5>> mUpperInteriorMasks;
    BufferT<nanovdb::Coord>   mRootInteriorTileOrigins; // Output: origins of interior root tiles.

    // Root-level scratch. Absent root tiles are not stored in the grid, so the root flood runs on a dense
    // 3D array of root tiles instead: every root-tile position inside the active index bbox, in units of
    // root tiles (one root tile spans NanoUpper<BuildT>::DIM voxels per axis).
    // Tile (i, j, k) of that array, counted from mRootTileMin, is stored at
    // (i * mRootTileDims[1] + j) * mRootTileDims[2] + k, i.e. z varies fastest as in NanoVDB nodes.
    BufferT<uint8_t>           mRootMask;       // One byte per root-tile position: 1 if interior, else 0.
    nanovdb::Coord             mRootTileMin{0};  // Root-tile coordinate (voxel coordinate >> TOTAL) of entry 0.
    nanovdb::Coord             mRootTileDims{0}; // Number of root-tile positions along x, y and z.
}; // tools::cuda::PropagateSign<BuildT, ResourceT>

//----------------------------------------------------------------------------------------------------------------------
// Setup: allocate the per-level masks.

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::initializeMasks()
{
    // Inputs are expected to be ready in mStream order, but the caller's work that writes them may still be
    // queued on mStream. DeviceGridTraits reads the grid with a blocking cudaMemcpy on the default stream,
    // which does not wait for mStream in general (e.g. for a non-blocking stream), so wait for it here.
    cudaCheck(cudaStreamSynchronize(mStream));
    const auto treeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceGrid);
    const auto& [leafCount, lowerCount, upperCount] = treeData.mNodeCount;

    //Initialize Leaf and internal node masks
    mLeafInteriorMasks = BufferT<nanovdb::Mask<3>>(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
    mLowerInteriorMasks = BufferT<nanovdb::Mask<4>>(mStream, this->ref(), lowerCount, nanovdb::cuda::noInit);
    mUpperInteriorMasks = BufferT<nanovdb::Mask<5>>(mStream, this->ref(), upperCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mLowerInteriorMasks.data(), 0, mLowerInteriorMasks.size_bytes(), mStream));
    cudaCheck(cudaMemsetAsync(mUpperInteriorMasks.data(), 0, mUpperInteriorMasks.size_bytes(), mStream));

    // Unlike node masks, root propagation must also cover absent root tiles, which the grid does not store.
    // An absent tile can only be interior if active voxels enclose it, so it lies inside the bbox of the
    // active voxels: allocate one byte per root-tile position in that bbox (see mRootMask).
    const auto activeIndexBBox = util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(mDeviceGrid, treeData);

    // Voxel bbox -> bbox of the root tiles containing it. The arithmetic shift rounds down, so negative
    // coordinates land in the right tile (e.g. voxel x = -100 is in root tile x = -1).
    constexpr int RootTileShift = NanoUpper<BuildT>::TOTAL; // log2 of the root-tile width in voxels = 12
    const nanovdb::CoordBBox rootTileBounds(activeIndexBBox.min() >> RootTileShift,
                                           activeIndexBBox.max() >> RootTileShift);
    mRootTileMin = rootTileBounds.min();
    mRootTileDims = rootTileBounds.dim(); // (0,0,0) if the grid has no active voxels
    const std::size_t rootCellCount = rootTileBounds.volume();
    // RootPropagationFunctor indexes root tiles with int and runs up to rootCellCount + 2 sweeps.
    constexpr std::size_t MaxRootCellCount = std::size_t(std::numeric_limits<int>::max()) - 2;
    if (rootCellCount > MaxRootCellCount)
        throw std::runtime_error("nanovdb::tools::cuda::PropagateSign: active bounds span too many root tiles");

    // All zero: no root tile is interior until the faces of upper nodes seed it.
    mRootMask = BufferT<uint8_t>(mStream, this->ref(), rootCellCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mRootMask.data(), 0, mRootMask.size_bytes(), mStream));
}// PropagateSign<BuildT, ResourceT>::initializeMasks

//----------------------------------------------------------------------------------------------------------------------
// Stage 1: fill each leaf from its active voxels.
//
// Active voxels keep their own sign and act as walls; the inside ones seed a flood through the leaf's
// inactive voxels. The flood is Jacobi style: in each sweep every unreached inactive voxel turns on if a
// face neighbor is already on, so a sweep is one parallel step with no ordering between threads. A
// scanline fill, which runs to the end of an open row in one sweep, needs fewer sweeps only when the
// inside region winds through long corridors; next to a surface the flood crosses a node in a few
// sweeps, so the simpler Jacobi sweep is no slower there.

namespace signing::detail {

/// @brief Fills one leaf's interior mask from the signs of its active voxels, one block per leaf and one
///        thread per voxel. FaceSeedFunctor<BuildT, 0> then seeds coarser levels from the filled leaf.
template <typename BuildT>
struct LeafPropagationFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    // True if any of the six face neighbors of local within this leaf is interior.
    __device__ static bool hasInteriorNeighbor(const uint8_t* interior, const nanovdb::Coord& local)
    {
        const auto& inside = reinterpret_cast<const uint8_t(&)[8][8][8]>(*interior);
        const int x = local[0], y = local[1], z = local[2];
        return (x > 0 && inside[x-1][y][z]) || (x < 7 && inside[x+1][y][z]) ||
               (y > 0 && inside[x][y-1][z]) || (y < 7 && inside[x][y+1][z]) ||
               (z > 0 && inside[x][y][z-1]) || (z < 7 && inside[x][y][z+1]);
    }

    // TODO: Rename signs to signSidecar?
    __device__ void operator()(const NanoGrid<BuildT>* grid, const int8_t* signs,
                               nanovdb::Mask<3>* masks)
    {
        const int leafID = blockIdx.x;
        const int tID = threadIdx.x;
        const auto& leaf = grid->tree().template getFirstNode<0>()[leafID];

        __shared__ uint8_t interior[NanoLeaf<BuildT>::SIZE]; // 1 = interior; 0 = active exterior or unreached inactive voxel.
        __shared__ int changed; // Set when a sweep reaches a new interior voxel.

        const bool isActive = leaf.isActive(uint32_t(tID));

        // Active inside voxels seed the fill; active outside voxels remain walls.
        interior[tID] = isActive && (signs[leaf.getValue(uint32_t(tID))] == int8_t(-1));
        __syncthreads();

        const nanovdb::Coord local = NanoLeaf<BuildT>::OffsetToLocalCoord(tID);

        // Synchronous flood fill: each sweep reads the previous mask before any writes.
        // Every sweep but the last turns on at least one of the at most SIZE - 1 unreached voxels, so the
        // bound is a backstop, not the mechanism that ends the loop.
        constexpr int MaxPropagationSweeps = NanoLeaf<BuildT>::SIZE; // 512

        int iteration = 0;
        for (; iteration < MaxPropagationSweeps; ++iteration) {
            if (tID == 0) changed = 0;
            __syncthreads();

            bool turnOn = false;
            // Active signs are fixed; only unreached inactive voxels can become interior.
            if (!isActive && !interior[tID]) turnOn = hasInteriorNeighbor(interior, local);

            __syncthreads(); // finish this sweep's reads before updating the shared mask
            if (turnOn) {
                interior[tID] = 1;
                changed = 1; // benign race: every writer stores 1
            }
            __syncthreads();
            const bool done = changed == 0;
            __syncthreads(); // all threads read changed before the next reset
            if (done) break;
        }
        NANOVDB_ASSERT(iteration < MaxPropagationSweeps); // exhausted the backstop: the fill did not converge

        // Thread tID owns bit tID of the leaf mask, so each warp's ballot is exactly the mask's
        // 32-bit word (tID >> 5); warps write disjoint words, so no atomics are needed.
        const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, interior[tID]);
        if ((tID & 31) == 0) reinterpret_cast<uint32_t*>(masks[leafID].words())[tID >> 5] = ballot;
    }
}; // LeafPropagationFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillLeafInteriorMasks()
{
    if (mLeafInteriorMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = signing::detail::LeafPropagationFunctor<BuildT>;
    util::cuda::operatorKernel<Op><<<unsigned(mLeafInteriorMasks.size()), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mDeviceSigns, mLeafInteriorMasks.data());
    cudaCheckError();
}// PropagateSign<BuildT, ResourceT>::fillLeafInteriorMasks

//----------------------------------------------------------------------------------------------------------------------
// Stages 2, 4 and 6: seed coarser levels across node faces.
//
// Each face of a filled node that has an inside slot seeds the region across that face, if that region is
// a childless slot of a coarser level: a lower or upper slot, or a root-tile position. Pushing seeds from
// the filled nodes (scatter), rather than having every childless slot look at its neighbors (gather),
// keeps the work proportional to the filled nodes: most childless slots touch no node, and one childless
// slot can border many finer nodes. One probe per face suffices: nodes are aligned to their size, so the
// block of the same size across a face lies inside a single slot of every coarser level.

namespace signing::detail {

enum Face : int { minusX, plusX, minusY, plusY, minusZ, plusZ, faceCount };

template <int Log2Dim>
__device__ inline uint32_t faceSlotIndex(Face face, int u, int v)
{
    constexpr int LastSlot = (1 << Log2Dim) - 1;
    switch (face) {
    case minusX: return (0        << (2 * Log2Dim)) | (u << Log2Dim) | v;
    case plusX:  return (LastSlot << (2 * Log2Dim)) | (u << Log2Dim) | v;
    case minusY: return (u        << (2 * Log2Dim)) | (0 << Log2Dim) | v;
    case plusY:  return (u        << (2 * Log2Dim)) | (LastSlot << Log2Dim) | v;
    case minusZ: return (u        << (2 * Log2Dim)) | (v << Log2Dim);
    case plusZ:
    default:     return (u        << (2 * Log2Dim)) | (v << Log2Dim) | LastSlot;
    }
}

__device__ inline nanovdb::Coord faceNeighbor(nanovdb::Coord origin, Face face, int distance)
{
    switch (face) {
    case minusX: return origin.offsetBy(-distance, 0, 0);
    case plusX:  return origin.offsetBy( distance, 0, 0);
    case minusY: return origin.offsetBy(0, -distance, 0);
    case plusY:  return origin.offsetBy(0,  distance, 0);
    case minusZ: return origin.offsetBy(0, 0, -distance);
    case plusZ:
    default:     return origin.offsetBy(0, 0,  distance);
    }
}

/// @return 0 for a root value tile or leaf, 1 for lower, 2 for upper, and 3 for absent root space.
template <typename BuildT>
__hostdev__ inline int probeChildlessSlot(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk,
                                          uint64_t& nodeID, uint32_t& slot) //todo: is there any better name for this?
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;

    const auto& tree = grid.tree();
    const auto* tile = tree.root().probeTile(ijk);
    if (!tile) return 3;
    if (!tile->isChild()) return 0;

    const UpperT* upper = tree.root().getChild(tile);
    const LowerT* lower = upper->probeChild(ijk);
    if (!lower) {
        nodeID = util::PtrDiff(upper, tree.template getFirstNode<2>()) / sizeof(UpperT);
        slot = UpperT::CoordToOffset(ijk);
        return 2;
    }

    slot = LowerT::CoordToOffset(ijk);
    if (!lower->childMask().isOn(slot)) {
        nodeID = util::PtrDiff(lower, tree.template getFirstNode<1>()) / sizeof(LowerT);
        return 1;
    }
    return 0;
}

/// @brief Position (i, j, k), counted from the first root tile, of entry @a n of the dense root-tile
///        array described at PropagateSign::mRootMask. rootTileIndex is its inverse.
__hostdev__ inline nanovdb::Coord rootTilePosition(uint64_t n, const nanovdb::Coord& tileDims)
{
    const uint64_t sliceSize = uint64_t(tileDims[1]) * uint64_t(tileDims[2]);
    return nanovdb::Coord(int(n / sliceSize), int(n / tileDims[2] % tileDims[1]), int(n % tileDims[2]));
}

/// @brief Entry of the dense root-tile array at position @a p, counted from the first root tile.
__hostdev__ inline uint64_t rootTileIndex(const nanovdb::Coord& p, const nanovdb::Coord& tileDims)
{
    return (uint64_t(p[0]) * tileDims[1] + p[1]) * tileDims[2] + p[2];
}

template <typename BuildT, int Level>
__device__ inline void seedAcrossFace(const NanoGrid<BuildT>& grid, const nanovdb::Coord& origin, Face face,
                                      nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks,
                                      uint8_t* rootMask, nanovdb::Coord tileMin, nanovdb::Coord tileDims)
{
    const nanovdb::Coord ijk = faceNeighbor(origin, face, NanoNode<BuildT, Level>::type::DIM);
    uint64_t targetNodeID;
    uint32_t slot;
    const int level = probeChildlessSlot(grid, ijk, targetNodeID, slot);
    if (level <= Level) return; // only a coarser slot receives a seed

    if (level == 1) {
        lowerMasks[targetNodeID].setOnAtomic(slot);
    } else if (level == 2) {
        upperMasks[targetNodeID].setOnAtomic(slot);
    } else {
        const nanovdb::Coord p = (ijk >> NanoUpper<BuildT>::TOTAL) - tileMin;
        if (p[0] >= 0 && p[0] < tileDims[0] &&
            p[1] >= 0 && p[1] < tileDims[1] &&
            p[2] >= 0 && p[2] < tileDims[2]) {
            rootMask[rootTileIndex(p, tileDims)] = 1;
        }
    }
}

/// @brief For one filled leaf (Level 0), lower (1) or upper (2) node, one block per node, seeds the coarser
///        childless slot across each face that has an inside slot: a bit of @a lowerMasks or @a upperMasks,
///        which InternalPropagationFunctor then floods, or an entry of @a rootMask.
template <typename BuildT, int Level>
struct FaceSeedFunctor
{
    static_assert(Level >= 0 && Level <= 2, "FaceSeedFunctor requires a leaf, lower or upper node");

    using NodeT = typename NanoNode<BuildT, Level>::type;
    using MaskT = nanovdb::Mask<NodeT::LOG2DIM>;
    static constexpr int Log2Dim  = NodeT::LOG2DIM;
    static constexpr int FaceSize = 1 << (2 * Log2Dim);
    // A leaf has one thread per face voxel; in a lower or upper node each thread covers several face slots.
    static constexpr int MaxThreadsPerBlock         = Level == 0 ? faceCount * FaceSize : 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* grid, const MaskT* masks,
                               nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks, uint8_t* rootMask,
                               nanovdb::Coord tileMin, nanovdb::Coord tileDims)
    {
        const int nodeID = blockIdx.x;
        const int tID = threadIdx.x;
        const auto& node = grid->tree().template getFirstNode<Level>()[nodeID];

        __shared__ int faceHasInside[faceCount];
        if (tID < faceCount) faceHasInside[tID] = 0;
        __syncthreads();

        for (int i = tID; i < faceCount * FaceSize; i += blockDim.x) {
            const Face face = static_cast<Face>(i >> (2 * Log2Dim));
            const int u = (i >> Log2Dim) & ((1 << Log2Dim) - 1);
            const int v = i & ((1 << Log2Dim) - 1);
            const uint32_t n = faceSlotIndex<Log2Dim>(face, u, v);
            if constexpr (Level > 0) {
                if (node.childMask().isOn(n)) continue;
            }
            if (masks[nodeID].isOn(n)) faceHasInside[face] = 1; // all writers store 1
        }
        __syncthreads();

        if (tID < faceCount && faceHasInside[tID])
            seedAcrossFace<BuildT, Level>(*grid, node.origin(), static_cast<Face>(tID),
                                          lowerMasks, upperMasks, rootMask, tileMin, tileDims);
    }
}; // FaceSeedFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
template <int Level>
void PropagateSign<BuildT, ResourceT>::emitSeedsToCoarserLevels()
{
    auto launch = [&](const auto& interiorMasks) {
        if (interiorMasks.size() == 0) return; // a kernel launch with zero blocks is an error
        using Op = signing::detail::FaceSeedFunctor<BuildT, Level>;
        util::cuda::operatorKernel<Op><<<unsigned(interiorMasks.size()), Op::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceGrid, interiorMasks.data(), mLowerInteriorMasks.data(), mUpperInteriorMasks.data(),
            mRootMask.data(), mRootTileMin, mRootTileDims);
        cudaCheckError();
    };
    if constexpr (Level == 0)
        launch(mLeafInteriorMasks);
    else if constexpr (Level == 1)
        launch(mLowerInteriorMasks);
    else
        launch(mUpperInteriorMasks);
}// PropagateSign<BuildT, ResourceT>::emitSeedsToCoarserLevels

//----------------------------------------------------------------------------------------------------------------------
// Stages 3 and 5: fill each lower or upper node from its seeds.
//
// Slots with a child are walls; the seeds set by FaceSeedFunctor spread through the childless slots with
// sweeps like those of the leaf fill.

namespace signing::detail {

/// @brief Fills one lower (Level 1) or upper (Level 2) node's interior mask in place, one block per node,
///        from the seeds FaceSeedFunctor set in it.
template <typename BuildT, int Level>
struct InternalPropagationFunctor
{
    static_assert(Level == 1 || Level == 2, "InternalPropagationFunctor requires a lower or upper node");

    static constexpr int Log2Dim = Level == 1 ? NanoLower<BuildT>::LOG2DIM : NanoUpper<BuildT>::LOG2DIM;
    static constexpr int Dim     = 1 << Log2Dim;
    static constexpr int SlotCount = 1 << (3 * Log2Dim);
    static constexpr int WordCount = SlotCount >> 6;
    using MaskT = nanovdb::Mask<Log2Dim>;

    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* grid, MaskT* masks)
    {
        const int nodeID = blockIdx.x;
        const int tID = threadIdx.x;
        const auto& node = grid->tree().template getFirstNode<Level>()[nodeID];

        // A __shared__ variable cannot run Mask's zero-filling constructor, so the masks live in
        // raw shared words that are viewed as MaskT.
        __shared__ uint64_t wallWords[WordCount];
        __shared__ uint64_t interiorWords[WordCount];
        __shared__ int changed;
        const auto& walls = reinterpret_cast<const MaskT&>(wallWords);
        auto& interior = reinterpret_cast<MaskT&>(interiorWords);

        for (int word = tID; word < WordCount; word += blockDim.x) {
            wallWords[word] = node.childMask().words()[word];
            interiorWords[word] = masks[nodeID].words()[word] & ~wallWords[word];
        }
        __syncthreads();

        // Once a bit turns on it is final, so same-sweep reads may only accelerate the flood.
        // Every sweep but the last turns on at least one of the SlotCount slots, so the bound is a backstop.
        constexpr int MaxPropagationSweeps = SlotCount + 1;

        int iteration = 0;
        for (; iteration < MaxPropagationSweeps; ++iteration) {
            if (tID == 0) changed = 0;
            __syncthreads();

            bool any = false;
            for (int n = tID; n < SlotCount; n += blockDim.x) {
                if (walls.isOn(n) || interior.isOn(n)) continue;
                const nanovdb::Coord local = node.OffsetToLocalCoord(uint32_t(n));
                constexpr int dx = 1 << (2 * Log2Dim);
                constexpr int dy = 1 << Log2Dim;
                if ((local[0] > 0       && interior.isOn(n - dx)) ||
                    (local[0] < Dim - 1 && interior.isOn(n + dx)) ||
                    (local[1] > 0       && interior.isOn(n - dy)) ||
                    (local[1] < Dim - 1 && interior.isOn(n + dy)) ||
                    (local[2] > 0       && interior.isOn(n - 1))  ||
                    (local[2] < Dim - 1 && interior.isOn(n + 1))) {
                    interior.setOnAtomic(n);
                    any = true;
                }
            }
            if (any) changed = 1;
            __syncthreads();
            const bool done = changed == 0;
            __syncthreads(); // all threads read changed before the next reset
            if (done) break;
        }
        NANOVDB_ASSERT(iteration < MaxPropagationSweeps); // exhausted the backstop: the fill did not converge

        for (int word = tID; word < WordCount; word += blockDim.x)
            masks[nodeID].words()[word] = interiorWords[word];
    }
}; // InternalPropagationFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillLowerInteriorMasks()
{
    if (mLowerInteriorMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = signing::detail::InternalPropagationFunctor<BuildT, 1>;
    util::cuda::operatorKernel<Op><<<unsigned(mLowerInteriorMasks.size()), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mLowerInteriorMasks.data());
    cudaCheckError();
}// PropagateSign<BuildT, ResourceT>::fillLowerInteriorMasks

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillUpperInteriorMasks()
{
    if (mUpperInteriorMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = signing::detail::InternalPropagationFunctor<BuildT, 2>;
    util::cuda::operatorKernel<Op><<<unsigned(mUpperInteriorMasks.size()), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mUpperInteriorMasks.data());
    cudaCheckError();
}// PropagateSign<BuildT, ResourceT>::fillUpperInteriorMasks

//----------------------------------------------------------------------------------------------------------------------
// Stage 7: fill the root-tile positions.
//
// Positions with an upper child are walls; the seeds from upper-node faces spread through the other
// positions of the dense root-tile array (see PropagateSign::mRootMask). A single block floods the whole
// array, so this stage is meant for active bounds that span few root tiles.

namespace signing::detail {

/// @brief Marks entry @a n of the dense root-tile array as a wall if an upper node exists at that position.
template <typename BuildT>
struct RootWallFunctor
{
    __device__ void operator()(size_t n, const NanoGrid<BuildT>* grid, uint8_t* walls,
                               nanovdb::Coord tileMin, nanovdb::Coord tileDims) const
    {
        const nanovdb::Coord ijk = (tileMin + rootTilePosition(n, tileDims)) << NanoUpper<BuildT>::TOTAL;
        const auto* tile = grid->tree().root().probeTile(ijk);
        walls[n] = tile && tile->isChild();
    }
}; // RootWallFunctor

/// @brief Fills the dense root-tile array in place, in a single block, from the seeds FaceSeedFunctor set
///        in it, around the walls from RootWallFunctor.
struct RootPropagationFunctor
{
    static constexpr int MaxThreadsPerBlock         = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const uint8_t* walls, uint8_t* mask,
                               nanovdb::Coord tileDims)
    {
        const int tID = threadIdx.x;
        const int width = tileDims[0];
        const int height = tileDims[1];
        const int depth = tileDims[2];
        const int total = width * height * depth;
        __shared__ int changed;

        for (int n = tID; n < total; n += blockDim.x)
            mask[n] = !walls[n] && mask[n];
        __syncthreads();

        // Every simple path contains fewer than total cells; the bound is a convergence backstop.
        for (int iteration = 0; iteration < total + 2; ++iteration) {
            if (tID == 0) changed = 0;
            __syncthreads();

            bool any = false;
            for (int n = tID; n < total; n += blockDim.x) {
                if (walls[n] || mask[n]) continue;
                const nanovdb::Coord p = rootTilePosition(n, tileDims);
                const bool turnOn =
                    (p[0] > 0          && mask[n - height * depth]) ||
                    (p[0] < width - 1  && mask[n + height * depth]) ||
                    (p[1] > 0          && mask[n - depth]) ||
                    (p[1] < height - 1 && mask[n + depth]) ||
                    (p[2] > 0          && mask[n - 1]) ||
                    (p[2] < depth - 1  && mask[n + 1]);
                if (turnOn) {
                    mask[n] = 1;
                    any = true;
                }
            }
            if (any) changed = 1; // benign race: every writer stores 1
            __syncthreads();
            const bool done = changed == 0;
            __syncthreads(); // all threads read changed before the next reset
            if (done) return;
        }
        NANOVDB_ASSERT(false);
    }
}; // RootPropagationFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillRootInteriorMasks()
{
    const std::size_t rootCellCount = mRootMask.size();
    BufferT<uint8_t> walls(mStream, this->ref(), rootCellCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(walls.data(), 0, walls.size_bytes(), mStream));

    static constexpr unsigned int threadsPerBlock = 128;
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(rootCellCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        rootCellCount, signing::detail::RootWallFunctor<BuildT>{},
        mDeviceGrid, walls.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();

    using RootPropagationOp = signing::detail::RootPropagationFunctor;
    util::cuda::operatorKernel<RootPropagationOp>
        <<<1, RootPropagationOp::MaxThreadsPerBlock, 0, mStream>>>(
        walls.data(), mRootMask.data(), mRootTileDims);
    cudaCheckError();

    std::vector<uint8_t> interior(rootCellCount);
    cudaCheck(cudaMemcpyAsync(interior.data(), mRootMask.data(), rootCellCount, cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    // The dense root-tile array is small, so its interior positions are collected on the host.
    std::vector<nanovdb::Coord> origins;
    for (std::size_t n = 0; n < rootCellCount; ++n) {
        if (!interior[n]) continue;
        origins.push_back((mRootTileMin + signing::detail::rootTilePosition(n, mRootTileDims))
                          << NanoUpper<BuildT>::TOTAL);
    }
    mRootInteriorTileOrigins = BufferT<nanovdb::Coord>(mStream, this->ref(), origins.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(mRootInteriorTileOrigins.data(), origins.data(), mRootInteriorTileOrigins.size_bytes(),
                              cudaMemcpyHostToDevice, mStream));
}// PropagateSign<BuildT, ResourceT>::fillRootInteriorMasks

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
