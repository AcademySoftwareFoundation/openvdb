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
#include <cstdint>   // uint8_t, uint32_t, uint64_t
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move
#include <vector>    // std::vector

namespace nanovdb {

namespace tools::cuda {

/// @brief Propagate the signs of the active voxels of a ValueOnIndex level set through every inactive level.
/// @details This is the index-grid counterpart of tools::cuda::signedFloodFill (SignedFloodFill.cuh), which
///          only works on float grids: it writes +/-background in place into the inactive values themselves.
///          A ValueOnIndex grid has no inactive values to write to: its SDF values live in a separate sidecar
///          indexed by the active voxels, and every inactive voxel shares the background, slot 0. So instead
///          of modifying the grid, PropagateSign reads the SDF sidecar @a deviceSdf and returns invert masks:
///          a bit is on for an inactive element (voxel, childless slot, or absent root tile) whose value is
///          the background with its sign inverted. The value of the level set at an inactive element is thus
///          -sdf[0] if its bit is on and sdf[0] otherwise; active voxels take their own values, and their bits
///          are off. The masks are topology: negating every value of the sidecar, slot 0 included, gives the
///          complement of the level set without touching them.
///
///          Inside means a negative SDF value (for an inactive element, that of the region it lies in). The
///          masks mark the inactive inside regions, i.e. those whose sign differs from the background's
///          (precondition 3). These are not the bounded regions in general: in a hollow ball the cavity is
///          bounded but has the background's sign, so its bits are off.
///
///          The result has one entry per tree level, each covering what that level owns:
///          - leaf: one Mask<3> per leaf; bit n is on if voxel n is inactive and inside.
///          - lower and upper: one Mask per node; bit n is on if slot n has no child and is inside.
///          - root: the origins of the absent root tiles that are inside.
///
///          Preconditions, not checked (PruneGrid and MeshToGrid output satisfies 1 and 4):
///          1. No empty nodes: every leaf has an active voxel, and every lower and upper node has a child.
///          2. The active voxels shield both sides: no voxel with a negative SDF value is face-adjacent to an
///             inactive outside voxel, and no other active voxel to an inactive inside one.
///          3. Only the background's sign reaches infinity: the background sdf[0] is positive, and the inside
///             lies within the bounding box of the active voxels.
///          4. Active values are leaf voxels only (no active tiles), and the root holds only child tiles.
/// @tparam BuildT The grid's build type, which must be ValueOnIndex.
/// @tparam ValueT Floating-point type of the SDF values.
/// @tparam ResourceT Stream-ordered resource used for all output and scratch allocations.
template <typename BuildT, typename ValueT = float, typename ResourceT = nanovdb::cuda::DeviceResource>
class PropagateSign
{
    static_assert(util::is_same<BuildT, ValueOnIndex>::value,
                  "PropagateSign requires a ValueOnIndex grid: inactive voxels share the background index 0");
    static_assert(util::is_floating_point<ValueT>::value,
                  "PropagateSign reads SDF values: ValueT must be a floating-point type");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PropagateSign allocates stream-ordered storage and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "PropagateSign stores uint64_t masks and requires word-aligned allocations");

    using GridT = NanoGrid<BuildT>;

    template <typename T>
    using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:
    /// @brief The result of propagate(): the invert masks of every tree level (see the class description).
    struct InvertMasks
    {
        BufferT<nanovdb::Mask<3>> leaf;            // one per leaf
        BufferT<nanovdb::Mask<4>> lower;           // one per lower node
        BufferT<nanovdb::Mask<5>> upper;           // one per upper node
        BufferT<nanovdb::Coord>   rootTileOrigins; // the absent root tiles that are inside
    };

    /// @param deviceGrid Device ValueOnIndex grid whose active voxels index @a deviceSdf.
    /// @param deviceSdf SDF sidecar: the SDF value of each active voxel at its value index (leaf.getValue(n));
    ///        slot 0 is the background. A voxel is inside if its value is negative; zero counts as outside.
    ///        Only the values of active voxels are read.
    /// @param resource Allocation resource; must outlive the returned buffers.
    /// @note @a deviceGrid and @a deviceSdf must be ready in @a stream order (if they were written on
    ///       another stream, make @a stream wait for it, e.g. with cudaStreamWaitEvent), and must remain
    ///       valid until propagate() returns.
    /// todo: Handle inputs that break the preconditions, e.g. root value tiles, which an OpenVDB level set converted
    ///       to NanoVDB can have.
    PropagateSign(const GridT* deviceGrid, const ValueT* deviceSdf,
                  cudaStream_t stream = 0,
                  ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream)
        , mTimer(stream)
        , mDeviceGrid(deviceGrid)
        , mDeviceSdf(deviceSdf)
        , mResource(&resource)
        , mLeafInvertMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mLowerInvertMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mUpperInvertMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mRootInvertTileOrigins(stream, resource, 0, nanovdb::cuda::noInit)
        , mRootMask(stream, resource, 0, nanovdb::cuda::noInit)
    {}

    /// @brief Toggle timing output.
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Compute the invert masks of every tree level.
    /// @note The stream is synchronized before this method returns.
    InvertMasks propagate()
    {
        if (mVerbose==1) mTimer.start("Allocating invert masks");
        initializeMasks();
        if (mVerbose==1) mTimer.stop();
        // Root propagation is needed when the active bounds contain root-tile positions without an upper child.
        const bool needsRootPropagation = mRootMask.size() > mUpperInvertMasks.size();

        if (mVerbose==1) mTimer.start("Leaf fill and seeding coarser levels");
        fillLeafAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Lower fill and seeding coarser levels");
        fillLowerAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Upper fill and seeding root tiles");
        fillUpperAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (needsRootPropagation) {
            if (mVerbose==1) mTimer.start("Root fill");
            fillRootInvertTiles();
            if (mVerbose==1) mTimer.stop();
        }

        // The dense root-tile array is scratch: release it now, ordered on mStream behind its last reader.
        mRootMask.destroy();

        cudaCheck(cudaStreamSynchronize(mStream));
        return {std::move(mLeafInvertMasks), std::move(mLowerInvertMasks),
                std::move(mUpperInvertMasks), std::move(mRootInvertTileOrigins)};
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
    void fillRootInvertTiles();

    /// @brief Borrowed handle to the injected resource, for the scratch allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t       mStream{0};
    util::cuda::Timer  mTimer;
    int                mVerbose{0};
    const GridT*       mDeviceGrid{nullptr};
    const ValueT*      mDeviceSdf{nullptr};
    ResourceT*         mResource{nullptr};

    // Outputs: one invert mask per node of each level, and the origins of the inside absent root tiles.
    BufferT<nanovdb::Mask<3>> mLeafInvertMasks;
    BufferT<nanovdb::Mask<4>> mLowerInvertMasks;
    BufferT<nanovdb::Mask<5>> mUpperInvertMasks;
    BufferT<nanovdb::Coord>   mRootInvertTileOrigins;

    // Root-level scratch. A root-tile position is the place of one root tile, NanoUpper<BuildT>::DIM voxels
    // per axis; it is absent if the root has no tile there, and then the grid does not store it. So the root
    // flood runs on a dense 3D array with one entry per root-tile position inside the active index bbox.
    // Position (i, j, k) of that array, counted from mRootTileMin, is stored at entry
    // (i * mRootTileDims[1] + j) * mRootTileDims[2] + k, i.e. z varies fastest as in NanoVDB nodes.
    BufferT<uint8_t>           mRootMask;       // One byte per root-tile position: 1 if interior, else 0.
    nanovdb::Coord             mRootTileMin{0};  // Root-tile coordinate (voxel coordinate >> TOTAL) of entry 0.
    nanovdb::Coord             mRootTileDims{0}; // Number of root-tile positions along x, y and z.
}; // tools::cuda::PropagateSign<BuildT, ValueT, ResourceT>

//----------------------------------------------------------------------------------------------------------------------
// Setup: allocate the per-level masks.

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::initializeMasks()
{
    // Inputs are ready in mStream order, so read the grid's tree header on mStream (the read blocks).
    const auto treeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceGrid, mStream);
    const auto& [leafCount, lowerCount, upperCount] = treeData.mNodeCount;

    //Initialize Leaf and internal node masks
    mLeafInvertMasks = BufferT<nanovdb::Mask<3>>(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
    mLowerInvertMasks = BufferT<nanovdb::Mask<4>>(mStream, this->ref(), lowerCount, nanovdb::cuda::noInit);
    mUpperInvertMasks = BufferT<nanovdb::Mask<5>>(mStream, this->ref(), upperCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mLowerInvertMasks.data(), 0, mLowerInvertMasks.size_bytes(), mStream));
    cudaCheck(cudaMemsetAsync(mUpperInvertMasks.data(), 0, mUpperInvertMasks.size_bytes(), mStream));

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
}// PropagateSign<BuildT, ValueT, ResourceT>::initializeMasks

//----------------------------------------------------------------------------------------------------------------------

namespace floodfill::detail {

// Stages 1, 2 and 3: fill each leaf, lower or upper node, then seed coarser levels across its faces.
//
// The labels are those of a flood fill over all of space at voxel resolution: it starts from the active
// voxels with a negative SDF value, spreads across faces through inactive voxels, and stops at the other
// active voxels. So face-adjacent inactive voxels always share a label. Under preconditions 1-4 (see
// PropagateSign) this fill can be done node by node, finest level first, because of two facts:
// - A leaf's fill is exact on its own. A region of inside inactive voxels in a leaf borders, within the
//   leaf, only voxels with a negative SDF value (precondition 2). If it bordered none, it would be the whole
//   leaf, which has an active voxel (1). So it touches a voxel with a negative SDF value in its own leaf,
//   and the leaf's fill reaches it.
// - A node face-adjacent to an inside childless slot has that whole face inside: its inactive voxels share
//   the slot's label, and its active ones have a negative SDF value (2). The finest nodes holding the face
//   mark it, and seed the slot across it (seedAcrossFace); this is why every level seeds every coarser
//   level directly. A region of inside childless slots always borders a child (1), or at the root an upper
//   node (3), so it is seeded, and the coarser level's fill spreads over it.
// Seeds only go to strictly coarser levels, so each level is final before a coarser fill reads it.
//
// In a leaf, active voxels keep their own sign and act as walls; the inside ones seed a flood fill through
// the leaf's inactive voxels. In a lower or upper node, slots with a child are walls, and the seeds are the
// slots that the faces of finer nodes set in the previous stages. The flood is Jacobi style: in each sweep
// every unreached open slot turns on if a face neighbor is already on, so a sweep is one parallel step with
// no ordering between threads. Under preconditions 1 and 2, every wall facing an inside region is a seed or
// has a seed next to it, so the flood never has to go around a wall, and the number of sweeps is bounded by
// the node's size (see BitFloodFunctor).
//
// One kernel per level, one warp per node: lane x holds x-slice x of the node's mask in registers as a
// util::morphology::Mask2D, and lanes x >= Dim hold empty slices. The warp fills the mask with the warp
// MaskShift of util/cuda/MorphologyHelpers.cuh, stores its inactive part as the node's invert mask, and then
// seeds coarser levels across the node's faces (seedAcrossFace).

/// @brief The calling lane's x-slice of the seeds of @a leaf, i.e. the interior at sweep 0 of its fill: its
///        active voxels with a negative SDF value. Lane x gets x-slice x, and lanes x >= 8 get empty slices.
///        All 32 lanes of the warp call it together.
/// @param sdf SDF sidecar of the grid, indexed by value index; read only at the leaf's active voxels.
template <typename BuildT, typename ValueT>
__device__ util::morphology::Mask2D<3> leafSeedSlice(const NanoLeaf<BuildT>& leaf, const ValueT* sdf)
{
    // Each ballot tests 32 consecutive voxels, half of the one-word slice n / 64, and the lane holding that
    // slice keeps the answers. So all 32 lanes test voxels, 16 each, not only the 8 lanes holding slices.
    // In a ValueOnIndex leaf an inactive voxel has value index 0, the background: it is never a seed, and
    // its index is not read.
    const int lane = threadIdx.x & 31;
    util::morphology::Mask2D<3> s{};
    for (uint32_t first = 0; first < NanoLeaf<BuildT>::SIZE; first += 32) {
        const uint64_t index  = leaf.getValue(first + uint32_t(lane));
        const bool     seed   = index != 0 && sdf[index] < ValueT(0); // zero is outside
        const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, seed);
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

/// @brief Fills one leaf (Level 0), lower (1) or upper (2) node, one warp per node, stores its invert mask in
///        @a masks, then seeds the childless slots of coarser levels across the node's inside faces.
template <typename BuildT, typename ValueT, int Level>
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

    __device__ void operator()(const NanoGrid<BuildT>* grid, const ValueT* sdf, MaskT* masks, uint32_t nodeCount,
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
            // Leaf: seeds are active voxels with a negative SDF value; inactive voxels are open.
            interior = leafSeedSlice(node, sdf);
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
        // with a negative SDF value), the slot just before it in a lower or upper node (seeded across the
        // child's face, see the stage comment above).
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

        // The invert mask holds only the open inside slots: a leaf's inside active voxels take their own
        // values, so their bits are off (in lower and upper nodes the interior is already within open). The
        // face seeding below still uses the whole interior, since a face of inside active voxels seeds too.
        if (hasSlice) (interior & open).store(masks[nodeID], x);

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

} // namespace floodfill::detail

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillLeafAndSeedCoarserLevels()
{
    if (mLeafInvertMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = floodfill::detail::BitFloodFunctor<BuildT, ValueT, 0>;
    const uint32_t count = uint32_t(mLeafInvertMasks.size());
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mDeviceSdf, mLeafInvertMasks.data(), count,
        mLowerInvertMasks.data(), mUpperInvertMasks.data(), mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}// PropagateSign<BuildT, ValueT, ResourceT>::fillLeafAndSeedCoarserLevels

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillLowerAndSeedCoarserLevels()
{
    if (mLowerInvertMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = floodfill::detail::BitFloodFunctor<BuildT, ValueT, 1>;
    const uint32_t count = uint32_t(mLowerInvertMasks.size());
    const ValueT*  noSdf = nullptr;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, noSdf, mLowerInvertMasks.data(), count,
        mLowerInvertMasks.data(), mUpperInvertMasks.data(), mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}// PropagateSign<BuildT, ValueT, ResourceT>::fillLowerAndSeedCoarserLevels

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillUpperAndSeedCoarserLevels()
{
    if (mUpperInvertMasks.size() == 0) return; // a kernel launch with zero blocks is an error
    using Op = floodfill::detail::BitFloodFunctor<BuildT, ValueT, 2>;
    const uint32_t count = uint32_t(mUpperInvertMasks.size());
    const ValueT*  noSdf = nullptr;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, noSdf, mUpperInvertMasks.data(), count,
        mLowerInvertMasks.data(), mUpperInvertMasks.data(), mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}// PropagateSign<BuildT, ValueT, ResourceT>::fillUpperAndSeedCoarserLevels

//----------------------------------------------------------------------------------------------------------------------

namespace floodfill::detail {

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

} // namespace floodfill::detail

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillRootInvertTiles()
{
    const std::size_t rootPositionCount = mRootMask.size();
    BufferT<uint8_t> walls(mStream, this->ref(), rootPositionCount, nanovdb::cuda::noInit);
    BufferT<uint8_t> next(mStream, this->ref(), rootPositionCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(walls.data(), 0, walls.size_bytes(), mStream));

    static constexpr unsigned int threadsPerBlock = 128;
    util::cuda::lambdaKernel
        <<<util::cuda::blocksPerGrid(rootPositionCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        rootPositionCount, floodfill::detail::RootWallFunctor<BuildT>{},
        mDeviceGrid, walls.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();

    using RootPropagationOp = floodfill::detail::RootPropagationFunctor;
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
        origins.push_back((mRootTileMin + floodfill::detail::rootTilePosition(n, mRootTileDims))
                          << NanoUpper<BuildT>::TOTAL);
    }
    mRootInvertTileOrigins = BufferT<nanovdb::Coord>(mStream, this->ref(), origins.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(mRootInvertTileOrigins.data(), origins.data(), mRootInvertTileOrigins.size_bytes(),
                              cudaMemcpyHostToDevice, mStream));
}// PropagateSign<BuildT, ValueT, ResourceT>::fillRootInvertTiles

//----------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
