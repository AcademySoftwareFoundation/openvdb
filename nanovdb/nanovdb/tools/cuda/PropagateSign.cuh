// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/PropagateSign.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Propagates active-voxel signs into the inactive regions of a NanoVDB index grid.

    \warning This header contains CUDA device code and must be included from a .cu or .cuh file.
*/

#ifndef NANOVDB_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/util/MorphologyHelpers.h>
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
/// todo: Choose the name of the output masks: "interior masks" or "invert masks".
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
///
///          Preconditions, not checked (PruneGrid and MeshToGrid output satisfies 1 and 4):
///          1. No empty nodes: every leaf has an active voxel, and every lower and upper node has a child.
///          2. The active voxels shield both sides: no voxel with sign -1 is face-adjacent to an inactive
///             outside voxel, and no other active voxel to an inactive inside one.
///          3. The inside is bounded: it lies within the bounding box of the active voxels.
///          4. Active values are leaf voxels only (no active tiles), and the root holds only child tiles.
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
    /// @param deviceSigns Per-active-value signs: -1 inside, any other value (including 0) outside.
    /// @param resource Allocation resource; must outlive the returned buffers.
    /// @note @a deviceGrid and @a deviceSigns must be ready in @a stream order (if they were written on
    ///       another stream, make @a stream wait for it, e.g. with cudaStreamWaitEvent), and must remain
    ///       valid until propagate() returns.
    /// todo: Handle inputs that break the preconditions, e.g. root value tiles, which an OpenVDB level set converted
    ///       to NanoVDB can have.
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
        // Root propagation is needed when the active bounds contain root-tile positions without an upper child.
        const bool needsRootPropagation = mRootMask.size() > mUpperInteriorMasks.size();

        if (mVerbose==1) mTimer.start("Leaf interior fill and seeding coarser levels");
        fillLeafAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Lower interior fill and seeding coarser levels");
        fillLowerAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Upper interior fill and seeding root tiles");
        fillUpperAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (needsRootPropagation) {
            if (mVerbose==1) mTimer.start("Root interior fill");
            fillRootInteriorMasks();
            if (mVerbose==1) mTimer.stop();
        }

        // The dense root-tile array is scratch: release it now, ordered on mStream behind its last reader.
        mRootMask.destroy();

        cudaCheck(cudaStreamSynchronize(mStream));
        return {std::move(mLeafInteriorMasks), std::move(mLowerInteriorMasks),
                std::move(mUpperInteriorMasks), std::move(mRootInteriorTileOrigins)};
    }

private:

    // --- Pipeline stages (run in order by propagate). ---
    // Stage 4 runs only when the active root bounds contain root-tile positions without upper children.

    // Setup: allocate per-node masks and a dense root mask.
    void initializeMasks();

    // Stage 1: fill leaf interiors and seed coarser childless slots from their faces.
    void fillLeafAndSeedCoarserLevels();

    // Stage 2: fill lower interiors and seed coarser childless slots from their faces.
    void fillLowerAndSeedCoarserLevels();

    // Stage 3: fill upper interiors and seed coarser childless slots from their faces.
    void fillUpperAndSeedCoarserLevels();

    // Stage 4: fill the root-tile positions, with those that have upper children as walls, then collect the
    // origins of the interior ones.
    void fillRootInteriorMasks();

    /// @brief Borrowed handle to the injected resource, for the scratch allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t       mStream{0};
    util::cuda::Timer  mTimer;
    int                mVerbose{0};
    const GridT*       mDeviceGrid{nullptr};
    const int8_t*      mDeviceSigns{nullptr};
    ResourceT*         mResource{nullptr};

    // Outputs: one mask per node of each level, and the origins of the interior root-tile positions.
    BufferT<nanovdb::Mask<3>> mLeafInteriorMasks;
    BufferT<nanovdb::Mask<4>> mLowerInteriorMasks;
    BufferT<nanovdb::Mask<5>> mUpperInteriorMasks;
    BufferT<nanovdb::Coord>   mRootInteriorTileOrigins;

    // Root-level scratch. A root-tile position is the place of one root tile, NanoUpper<BuildT>::DIM voxels
    // per axis; it is absent if the root has no tile there, and then the grid does not store it. So the root
    // flood runs on a dense 3D array with one entry per root-tile position inside the active index bbox.
    // Position (i, j, k) of that array, counted from mRootTileMin, is stored at entry
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
    // Inputs are ready in mStream order, so read the grid's tree header on mStream (the read blocks).
    const auto treeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceGrid, mStream);
    const auto& [leafCount, lowerCount, upperCount] = treeData.mNodeCount;

    //Initialize Leaf and internal node masks
    mLeafInteriorMasks = BufferT<nanovdb::Mask<3>>(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
    mLowerInteriorMasks = BufferT<nanovdb::Mask<4>>(mStream, this->ref(), lowerCount, nanovdb::cuda::noInit);
    mUpperInteriorMasks = BufferT<nanovdb::Mask<5>>(mStream, this->ref(), upperCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mLowerInteriorMasks.data(), 0, mLowerInteriorMasks.size_bytes(), mStream));
    cudaCheck(cudaMemsetAsync(mUpperInteriorMasks.data(), 0, mUpperInteriorMasks.size_bytes(), mStream));

    // Unlike node masks, root propagation must also cover absent root-tile positions, which the grid does not
    // store. An absent position can only be interior if active voxels enclose it, so it lies inside the bbox
    // of the active voxels: allocate one byte per root-tile position in that bbox (see mRootMask).
    const auto activeIndexBBox = util::cuda::DeviceGridTraits<BuildT>::getIndexBBox(mDeviceGrid, treeData, mStream);

    // Voxel bbox -> bbox of the root tiles containing it. The arithmetic shift rounds down, so negative
    // coordinates land in the right tile (e.g. voxel x = -100 is in root tile x = -1).
    // NanoUpper<BuildT>::TOTAL is log2 of the root-tile width in voxels.
    const nanovdb::CoordBBox rootTileBounds(activeIndexBBox.min() >> NanoUpper<BuildT>::TOTAL,
                                           activeIndexBBox.max() >> NanoUpper<BuildT>::TOTAL);
    mRootTileMin = rootTileBounds.min();
    mRootTileDims = rootTileBounds.dim(); // (0,0,0) if the grid has no active voxels
    const std::size_t rootPositionCount = rootTileBounds.volume();
    // RootPropagationFunctor indexes root-tile positions with int and runs up to rootPositionCount + 1 sweeps.
    constexpr std::size_t MaxRootPositionCount = std::size_t(std::numeric_limits<int>::max()) - 1;
    if (rootPositionCount > MaxRootPositionCount)
        throw std::runtime_error("nanovdb::tools::cuda::PropagateSign: active bounds span too many root tiles");

    // All zero: no root-tile position is interior until a face of a leaf, lower or upper node seeds it.
    mRootMask = BufferT<uint8_t>(mStream, this->ref(), rootPositionCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mRootMask.data(), 0, mRootMask.size_bytes(), mStream));
}// PropagateSign<BuildT, ResourceT>::initializeMasks

//----------------------------------------------------------------------------------------------------------------------

namespace signing::detail {

// Stages 1, 2 and 3: fill each leaf, lower or upper node, then seed coarser levels across its faces.
//
// In a leaf, active voxels keep their own sign and act as walls; the inside ones seed a flood through the
// leaf's inactive voxels. In a lower or upper node, slots with a child are walls, and the seeds are the
// slots that the faces of finer nodes set in the previous stages. The flood is Jacobi style: in each sweep
// every unreached open slot turns on if a face neighbor is already on, so a sweep is one parallel step with
// no ordering between threads. Under the preconditions no wall makes the flood detour inside a node, so the
// number of sweeps is bounded by the node's size (see BitFloodFunctor).
//
// One kernel per level, one warp per node: lane x holds x-slice x of the node's mask in registers as a
// util::morphology::Mask2D, and lanes x >= Dim hold empty slices. The warp fills the mask with the warp
// MaskShift of util/cuda/MorphologyHelpers.cuh, stores it, and then seeds coarser levels across the node's
// faces (seedAcrossFace).

/// @brief The calling lane's slice of the seeds of @a leaf, its active voxels whose sign is -1: lane x gets
///        x-slice x, and lanes x >= 8 get empty slices. All 32 lanes of the warp call it together.
template <typename BuildT>
__device__ util::morphology::Mask2D<3> leafSeedSlice(const NanoLeaf<BuildT>& leaf, const int8_t* signs)
{
    // Each ballot tests 32 consecutive voxels, half of the one-word slice n / 64, and the lane holding that
    // slice keeps the answers. So all 32 lanes test voxels, 16 each, not only the 8 lanes holding slices.
    const int lane = threadIdx.x & 31;
    util::morphology::Mask2D<3> s{};
    for (uint32_t first = 0; first < NanoLeaf<BuildT>::SIZE; first += 32) {
        const uint32_t n      = first + uint32_t(lane);
        const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, leaf.isActive(n) && signs[leaf.getValue(n)] == int8_t(-1));
        if (lane == int(first / 64)) s.mWords[0] |= uint64_t(ballot) << (first % 64);
    }
    return s;
}

/// @brief A face of a node. In a set of faces, bit f stands for face f.
enum Face : int { minusX, plusX, minusY, plusY, minusZ, plusZ, faceCount };

/// @brief The set of the node's faces on which x-slice @a x, held as @a s, has an on slot: minusX and plusX if
///        it is the first or last slice and not empty, and minusY, plusY, minusZ and plusZ if it has an on slot
///        with y = 0, y = Dim - 1, z = 0 or z = Dim - 1. The z edges are the slots a unit z shift zero-fills:
///        z = 0 has no source under dk = -1, and z = Dim - 1 none under dk = +1.
template <int LOG2DIM>
__hostdev__ uint32_t facesWithOnSlots(const util::morphology::Mask2D<LOG2DIM>& s, int x)
{
    constexpr int      Dim       = util::morphology::Mask2D<LOG2DIM>::DIM;
    constexpr int      WordCount = util::morphology::Mask2D<LOG2DIM>::WORD_COUNT;
    constexpr uint64_t FirstY    = util::morphology::detail::firstWordOfY0<LOG2DIM>;   // y = 0: the low Dim bits of the first word
    constexpr uint64_t LastY     = FirstY << (64 - Dim);                               // y = Dim - 1: the high Dim bits of the last word
    constexpr uint64_t FirstZ    = ~util::morphology::detail::zShiftBits<LOG2DIM, -1>; // z = 0, in every y of a word
    constexpr uint64_t LastZ     = ~util::morphology::detail::zShiftBits<LOG2DIM, +1>; // z = Dim - 1, in every y of a word
    uint64_t firstZ = 0, lastZ = 0;
    for (int j = 0; j < WordCount; ++j) { firstZ |= s.mWords[j] & FirstZ; lastZ |= s.mWords[j] & LastZ; }
    const bool nonEmpty = !s.isOff();
    return uint32_t(x == 0 && nonEmpty)                      << minusX |
           uint32_t(x == Dim - 1 && nonEmpty)                << plusX  |
           uint32_t((s.mWords[0] & FirstY) != 0)             << minusY |
           uint32_t((s.mWords[WordCount - 1] & LastY) != 0) << plusY  |
           uint32_t(firstZ != 0)                             << minusZ |
           uint32_t(lastZ != 0)                              << plusZ;
}

// Seeding coarser levels across node faces.
//
// Each face of a filled node that has an inside slot seeds the region across that face, if that region is
// a childless slot of a coarser level: a lower or upper slot, or a root-tile position. Pushing seeds from
// the filled nodes (scatter), rather than having every childless slot look at its neighbors (gather),
// keeps the work proportional to the filled nodes: most childless slots touch no node, and one childless
// slot can border many finer nodes. One probe per face suffices: nodes are aligned to their size, so the
// block of the same size across a face lies inside a single slot of every coarser level.

/// @brief The coordinate @a distance away from @a origin across @a face.
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

/// @brief The slot that holds a coordinate's value, i.e. where the tree's descent terminates: a leaf voxel, a
///        childless slot of a lower or upper node, or an absent root-tile position (not a slot of any node).
struct TerminalSlot
{
    uint32_t level;  // LEVEL of the node holding the slot (NanoRoot's for an absent root-tile position)
    uint64_t nodeID; // which lower or upper node; 0 otherwise
    uint32_t slot;   // which of that node's slots; 0 otherwise
};

/// @brief The terminal slot of @a ijk. The root must hold only child tiles.
template <typename BuildT>
__hostdev__ inline TerminalSlot probeTerminalSlot(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;

    const auto& tree = grid.tree();
    const auto* tile = tree.root().probeTile(ijk);
    if (!tile) return {NanoRoot<BuildT>::LEVEL, 0, 0};
    if (!tile->isChild()) {
        NANOVDB_ASSERT(false);                  // a root value tile
        return {NanoLeaf<BuildT>::LEVEL, 0, 0}; // nothing to seed
    }

    const UpperT* upper = tree.root().getChild(tile);
    const LowerT* lower = upper->probeChild(ijk);
    if (!lower)
        return {UpperT::LEVEL, util::PtrDiff(upper, tree.template getFirstNode<2>()) / sizeof(UpperT),
                UpperT::CoordToOffset(ijk)};

    const uint32_t slot = LowerT::CoordToOffset(ijk);
    if (!lower->childMask().isOn(slot))
        return {LowerT::LEVEL, util::PtrDiff(lower, tree.template getFirstNode<1>()) / sizeof(LowerT), slot};
    return {NanoLeaf<BuildT>::LEVEL, 0, 0};
}

/// @brief Position (i, j, k), counted from the first root-tile position, of entry @a n of the dense root-tile
///        array described at PropagateSign::mRootMask. rootTileIndex is its inverse.
__hostdev__ inline nanovdb::Coord rootTilePosition(uint64_t n, const nanovdb::Coord& tileDims)
{
    const uint64_t sliceSize = uint64_t(tileDims[1]) * uint64_t(tileDims[2]);
    return nanovdb::Coord(int(n / sliceSize), int(n / tileDims[2] % tileDims[1]), int(n % tileDims[2]));
}

/// @brief Entry of the dense root-tile array at position @a p, counted from the first root-tile position.
__hostdev__ inline uint64_t rootTileIndex(const nanovdb::Coord& p, const nanovdb::Coord& tileDims)
{
    return (uint64_t(p[0]) * tileDims[1] + p[1]) * tileDims[2] + p[2];
}

/// @brief Seeds the region across @a face of the Level node at @a origin, if it is a childless slot of a
///        coarser level: its bit in @a lowerMasks or @a upperMasks, set atomically since neighboring nodes
///        may seed the same slot, or its entry of @a rootMask if it is an absent root-tile position inside
///        the dense root-tile array.
template <typename BuildT, int Level>
__device__ inline void seedAcrossFace(const NanoGrid<BuildT>& grid, const nanovdb::Coord& origin, Face face,
                                      nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks,
                                      uint8_t* rootMask, nanovdb::Coord tileMin, nanovdb::Coord tileDims)
{
    const nanovdb::Coord ijk = faceNeighbor(origin, face, NanoNode<BuildT, Level>::type::DIM);
    const auto [level, nodeID, slot] = probeTerminalSlot(grid, ijk);
    if (level <= uint32_t(Level)) return; // only a coarser slot receives a seed

    if (level == NanoLower<BuildT>::LEVEL) {
        lowerMasks[nodeID].setOnAtomic(slot);
    } else if (level == NanoUpper<BuildT>::LEVEL) {
        upperMasks[nodeID].setOnAtomic(slot);
    } else { // an absent root-tile position
        const nanovdb::Coord p = (ijk >> NanoUpper<BuildT>::TOTAL) - tileMin;
        if (p[0] >= 0 && p[0] < tileDims[0] &&
            p[1] >= 0 && p[1] < tileDims[1] &&
            p[2] >= 0 && p[2] < tileDims[2]) {
            rootMask[rootTileIndex(p, tileDims)] = 1;
        }
    }
}

/// @brief Fills one leaf (Level 0), lower (1) or upper (2) node's interior mask in @a masks, one warp per
///        node, then seeds the childless slots of coarser levels across the node's inside faces.
template <typename BuildT, int Level>
struct BitFloodFunctor
{
    using NodeT  = typename NanoNode<BuildT, Level>::type;
    using MaskT  = nanovdb::Mask<NodeT::LOG2DIM>;
    using SliceT = util::morphology::Mask2D<NodeT::LOG2DIM>;

    static constexpr int      Dim                        = SliceT::DIM;
    static constexpr int      SlotCount                  = Dim * Dim * Dim;
    static constexpr int      MaxSweeps                  = Level == 0 ? 3 * (Dim - 1) + 1 : 3 * (Dim - 1);
    static constexpr int      NodesPerBlock              = 4;           // best in practice
    static constexpr int      MaxThreadsPerBlock         = 32 * NodesPerBlock;
    static constexpr int      MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* grid, const int8_t* signs, MaskT* masks, uint32_t nodeCount,
                               nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks, uint8_t* rootMask,
                               nanovdb::Coord tileMin, nanovdb::Coord tileDims)
    {
        // One warp per node; the block's last warps may have no node, and leave together.
        const int      x      = threadIdx.x & 31; // the lane, which holds x-slice x
        const int      warpID = threadIdx.x >> 5; // the block's warp, which fills one node
        const uint32_t nodeID = blockIdx.x * NodesPerBlock + warpID;
        if (nodeID >= nodeCount) return;
        const auto& node     = grid->tree().template getFirstNode<Level>()[nodeID];
        const bool  hasSlice = x < Dim;

        // Interior starts as the seeds; open slots are those the interior may spread into.
        SliceT interior{}, open{};
        if constexpr (Level == 0) {
            // Leaf: seeds are active voxels whose sign is -1; inactive voxels are open.
            interior = leafSeedSlice(node, signs);
            if (hasSlice) {
                open = SliceT::load(node.valueMask(), x);
                open.toggle();
            }
        } else if (hasSlice) {
            // Lower/upper: seeds were written into masks by the finer level; childless slots are open.
            open = SliceT::load(node.childMask(), x);
            open.toggle();
            interior = SliceT::load(masks[nodeID], x) & open;
        }

        // Each sweep adds the open face neighbors of the previous interior. A sweep that changes nothing ends
        // the fill; any other sweep adds at least one slot, so SlotCount + 1 sweeps suffice for any input.
        // Under preconditions 1 and 2 (see PropagateSign), a straight walk from an inside open slot toward the
        // nearest wall crosses only inside open slots and ends on a seed: the wall itself in a leaf (a voxel
        // with sign -1), the slot just before it in a lower or upper node (seeded across the child's face).
        // So at most 3 * (Dim - 1) sweeps turn slots on in a leaf, one fewer in a lower or upper node, and
        // one more sweep changes nothing: MaxSweeps, i.e. 22, 45 and 93.
        int sweep = 0;
        for (; sweep <= SlotCount; ++sweep) {
            SliceT neighbors{};
            util::morphology::cuda::MaskShift<-1,  0,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 1,  0,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0, -1,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  1,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  0, -1>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  0,  1>(interior, neighbors);
            const SliceT next = interior | (open & neighbors);
            if (!__any_sync(0xFFFFFFFFu, next != interior)) break;
            interior = next;
        }
        NANOVDB_ASSERT(sweep < MaxSweeps); // the input breaks preconditions 1 or 2
        if (hasSlice) interior.store(masks[nodeID], x);

        // Bit f is set if face f holds an inside slot. Each lane first finds the faces on which its own slice
        // has an inside slot; the xor-shuffle butterfly then ORs these sets over all 32 lanes, so every lane
        // ends up with the set for the whole node.
        uint32_t faceHasInside = facesWithOnSlots(interior, x);
        for (int offset = 16; offset > 0; offset >>= 1) faceHasInside |= __shfl_xor_sync(0xFFFFFFFFu, faceHasInside, offset);

        // Lane f seeds across face f if that face has an inside slot.
        if (x < faceCount && ((faceHasInside >> x) & 1u))
            seedAcrossFace<BuildT, Level>(*grid, node.origin(), Face(x),
                                          lowerMasks, upperMasks, rootMask, tileMin, tileDims);
    }
}; // BitFloodFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillLeafAndSeedCoarserLevels()
{
    if (mLeafInteriorMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = signing::detail::BitFloodFunctor<BuildT, 0>;
    const uint32_t count = uint32_t(mLeafInteriorMasks.size());
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mDeviceSigns, mLeafInteriorMasks.data(), count,
        mLowerInteriorMasks.data(), mUpperInteriorMasks.data(), mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}// PropagateSign<BuildT, ResourceT>::fillLeafAndSeedCoarserLevels

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillLowerAndSeedCoarserLevels()
{
    if (mLowerInteriorMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = signing::detail::BitFloodFunctor<BuildT, 1>;
    const uint32_t count = uint32_t(mLowerInteriorMasks.size());
    const int8_t*  noSigns = nullptr;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, noSigns, mLowerInteriorMasks.data(), count,
        mLowerInteriorMasks.data(), mUpperInteriorMasks.data(), mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}// PropagateSign<BuildT, ResourceT>::fillLowerAndSeedCoarserLevels

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillUpperAndSeedCoarserLevels()
{
    if (mUpperInteriorMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = signing::detail::BitFloodFunctor<BuildT, 2>;
    const uint32_t count = uint32_t(mUpperInteriorMasks.size());
    const int8_t*  noSigns = nullptr;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, noSigns, mUpperInteriorMasks.data(), count,
        mLowerInteriorMasks.data(), mUpperInteriorMasks.data(), mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}// PropagateSign<BuildT, ResourceT>::fillUpperAndSeedCoarserLevels

//----------------------------------------------------------------------------------------------------------------------

namespace signing::detail {

// Stage 4: fill the root-tile positions.
//
// Positions with an upper child are walls; the seeds that leaf, lower and upper faces set in stages 1-3
// spread through the other positions of the dense root-tile array (see PropagateSign::mRootMask). A single
// block floods the whole array, so this stage is meant for active bounds that span few root tiles.

/// @brief Marks entry @a n of the dense root-tile array as a wall if an upper node exists at that position.
template <typename BuildT>
struct RootWallFunctor
{
    __device__ void operator()(size_t n, const NanoGrid<BuildT>* grid, uint8_t* walls,
                               nanovdb::Coord tileMin, nanovdb::Coord tileDims) const
    {
        const nanovdb::Coord ijk = (tileMin + rootTilePosition(n, tileDims)) << NanoUpper<BuildT>::TOTAL;
        const auto* tile = grid->tree().root().probeTile(ijk);
        NANOVDB_ASSERT(!tile || tile->isChild()); // a root value tile
        walls[n] = tile && tile->isChild();
    }
}; // RootWallFunctor

/// @brief Fills the dense root-tile array @a mask, in a single block, from the seeds that BitFloodFunctor set
///        in it at every level, around the walls from RootWallFunctor. @a next is scratch of the same size.
struct RootPropagationFunctor
{
    static constexpr int MaxThreadsPerBlock         = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const uint8_t* walls, uint8_t* mask, uint8_t* next,
                               nanovdb::Coord tileDims)
    {
        const int tID = threadIdx.x;
        const int width = tileDims[0];
        const int height = tileDims[1];
        const int depth = tileDims[2];
        const int total = width * height * depth;

        for (int n = tID; n < total; n += blockDim.x)
            mask[n] = !walls[n] && mask[n];
        __syncthreads();

        // Jacobi sweeps, as in the node fills: each sweep reads cur and writes next, so no position is read and
        // written in the same sweep. A sweep that changes nothing ends the fill, and leaves the result in both
        // arrays; any other sweep turns at least one position on, so total + 1 sweeps suffice for any input.
        // Under preconditions 1 and 2 (see PropagateSign), a straight walk from an inside position toward the
        // nearest upper node crosses only inside positions and ends on a seed, the position just before that
        // node (seeded across its face). So at most (width - 1) + (height - 1) + (depth - 1) - 1 sweeps turn
        // positions on, and one more sweep changes nothing.
        uint8_t* cur = mask;
        int sweep = 0;
        for (; sweep <= total; ++sweep) {
            bool changed = false;
            for (int n = tID; n < total; n += blockDim.x) {
                const nanovdb::Coord p = rootTilePosition(n, tileDims);
                const bool on = cur[n] || (!walls[n] &&
                    ((p[0] > 0          && cur[n - height * depth]) ||
                     (p[0] < width - 1  && cur[n + height * depth]) ||
                     (p[1] > 0          && cur[n - depth]) ||
                     (p[1] < height - 1 && cur[n + depth]) ||
                     (p[2] > 0          && cur[n - 1]) ||
                     (p[2] < depth - 1  && cur[n + 1])));
                next[n] = on;
                changed |= on && !cur[n];
            }
            if (!__syncthreads_or(changed)) break; // also makes this sweep's writes visible to the next sweep
            uint8_t* swap = cur;
            cur  = next;
            next = swap;
        }
        NANOVDB_ASSERT(sweep < (width - 1) + (height - 1) + (depth - 1)); // the input breaks preconditions 1 or 2
    }
}; // RootPropagationFunctor

} // namespace signing::detail

template <typename BuildT, typename ResourceT>
void PropagateSign<BuildT, ResourceT>::fillRootInteriorMasks()
{
    const std::size_t rootPositionCount = mRootMask.size();
    BufferT<uint8_t> walls(mStream, this->ref(), rootPositionCount, nanovdb::cuda::noInit);
    BufferT<uint8_t> next(mStream, this->ref(), rootPositionCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(walls.data(), 0, walls.size_bytes(), mStream));

    static constexpr unsigned int threadsPerBlock = 128;
    util::cuda::lambdaKernel
        <<<util::cuda::blocksPerGrid(rootPositionCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        rootPositionCount, signing::detail::RootWallFunctor<BuildT>{},
        mDeviceGrid, walls.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();

    using RootPropagationOp = signing::detail::RootPropagationFunctor;
    util::cuda::operatorKernel<RootPropagationOp>
        <<<1, RootPropagationOp::MaxThreadsPerBlock, 0, mStream>>>(
        walls.data(), mRootMask.data(), next.data(), mRootTileDims);
    cudaCheckError();

    std::vector<uint8_t> interior(rootPositionCount);
    cudaCheck(cudaMemcpyAsync(interior.data(), mRootMask.data(), rootPositionCount, cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    // The dense root-tile array is small, so its interior positions are collected on the host.
    std::vector<nanovdb::Coord> origins;
    for (std::size_t n = 0; n < rootPositionCount; ++n) {
        if (!interior[n]) continue;
        origins.push_back((mRootTileMin + signing::detail::rootTilePosition(n, mRootTileDims))
                          << NanoUpper<BuildT>::TOTAL);
    }
    mRootInteriorTileOrigins = BufferT<nanovdb::Coord>(mStream, this->ref(), origins.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(mRootInteriorTileOrigins.data(), origins.data(), mRootInteriorTileOrigins.size_bytes(),
                              cudaMemcpyHostToDevice, mStream));
}// PropagateSign<BuildT, ResourceT>::fillRootInteriorMasks

//----------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
