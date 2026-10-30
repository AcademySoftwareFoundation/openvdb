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
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh> // topology::detail::tileSortKey
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <algorithm> // std::none_of
#include <cstddef>   // std::size_t
#include <cstdint>   // uint8_t, uint32_t, uint64_t
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move, std::pair
#include <vector>    // std::vector

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif
#endif // ifndef CALL_CUBS

namespace nanovdb {

namespace tools::cuda {

/// @brief Propagate the signs of the active voxels of a ValueOnIndex level set through every inactive level.
/// @details This is the index-grid counterpart of tools::cuda::signedFloodFill (SignedFloodFill.cuh), which
///          only works on float grids: it writes +/-background in place into the inactive values themselves.
///          A ValueOnIndex grid has no inactive values to write to: its SDF values live in a separate sidecar
///          indexed by the active voxels, and every inactive voxel shares the background, slot 0. So instead
///          of modifying the grid, PropagateSign reads the SDF sidecar @a deviceSdf and computes invert masks:
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
///          The result has one entry per tree level, each covering what that level owns. The caller allocates
///          the three mask arrays, which propagate() fills, and propagate() returns the root-tile origins:
///          - leaf: one Mask<3> per leaf; bit n is on if voxel n is inactive and inside.
///          - lower and upper: one Mask per node; bit n is on if slot n has no child and is inside.
///          - root: the origins of the absent root tiles that are inside, in increasing (x, y, z) order.
///
///          Preconditions (PruneGrid and MeshToGrid output satisfies 1 and 4). Only 4 is checked, by propagate(),
///          which throws if it fails; the others are the caller's to ensure:
///          1. No empty nodes: every leaf has an active voxel, and every lower and upper node has a child.
///          2. The active voxels shield both sides: no voxel with a negative SDF value is face-adjacent to an
///             inactive outside voxel, and no other active voxel to an inactive inside one.
///          3. Only the background's sign reaches infinity: the background sdf[0] is positive, and the inside
///             lies within the bounding box of the active voxels.
///          4. Active values are leaf voxels only (no active tiles), and the root holds only child tiles.
/// @tparam BuildT The grid's build type, which must be ValueOnIndex.
/// @tparam ValueT Floating-point type of the SDF values.
/// @tparam ResourceT Stream-ordered resource used for the root-tile output and all scratch allocations.
template <typename BuildT, typename ValueT = float, typename ResourceT = nanovdb::cuda::DeviceResource>
class PropagateSign
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex,
                  "PropagateSign requires a ValueOnIndex grid: inactive voxels share the background index 0");
    static_assert(util::is_floating_point<ValueT>::value,
                  "PropagateSign reads SDF values: ValueT must be a floating-point type");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PropagateSign allocates stream-ordered storage and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "PropagateSign stores scratch up to uint64_t width and requires word-aligned allocations");

    using GridT = NanoGrid<BuildT>;

    template <typename T>
    using BufferT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:
    /// @param deviceGrid Device ValueOnIndex grid whose active voxels index @a deviceSdf.
    /// @param deviceSdf SDF sidecar: the SDF value of each active voxel at its value index (leaf.getValue(n));
    ///        slot 0 is the background. A voxel is inside if its value is negative; zero counts as outside.
    ///        Only the values of active voxels are read.
    /// @param resource Allocation resource; must outlive the returned buffer.
    /// @note @a deviceGrid and @a deviceSdf must be ready in @a stream order (if they were written on
    ///       another stream, make @a stream wait for it, e.g. with cudaStreamWaitEvent), and must remain
    ///       valid until propagate() returns.
    /// todo: Handle inputs that break the preconditions, e.g. root value tiles, which an OpenVDB level set converted
    ///       to NanoVDB can have (propagate() now rejects them).
    PropagateSign(const GridT* deviceGrid, const ValueT* deviceSdf,
                  cudaStream_t stream = 0,
                  ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream)
        , mTimer(stream)
        , mDeviceGrid(deviceGrid)
        , mDeviceSdf(deviceSdf)
        , mResource(&resource)
        , mTempDevicePool(resource)
        , mRootInvertTileOrigins(stream, resource, 0, nanovdb::cuda::noInit)
        , mRootFaceSeeds(stream, resource, 0, nanovdb::cuda::noInit)
    {}

    /// @brief Toggle timing output.
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Compute the invert masks of every tree level: fill the caller's leaf, lower and upper masks, and
    ///        return the origins of the inside absent root tiles.
    /// @param leafMasks Device array of one Mask<3> per leaf, indexed like the grid's leaves; need not be
    ///        initialized.
    /// @param lowerMasks Device array of one Mask<4> per lower node, indexed like the grid's lower nodes; need not
    ///        be initialized.
    /// @param upperMasks Device array of one Mask<5> per upper node, indexed like the grid's upper nodes; need not
    ///        be initialized.
    /// @return { origins, hasInvertRootTiles }: a device buffer with the origins of the inside absent root tiles,
    ///         which the root does not store, and whether there is any.
    /// @throw std::runtime_error if the grid breaks precondition 4: an active tile, or a root value tile.
    /// @note The masks must be ready to be written in @a stream order. The stream is synchronized before this
    ///       method returns.
    std::pair<BufferT<nanovdb::Coord>, bool> propagate(nanovdb::Mask<3>* leafMasks, nanovdb::Mask<4>* lowerMasks,
                                                       nanovdb::Mask<5>* upperMasks)
    {
        mLeafInvertMasks = leafMasks;
        mLowerInvertMasks = lowerMasks;
        mUpperInvertMasks = upperMasks;

        if (mVerbose==1) mTimer.start("Initializing invert masks");
        initializeMasks();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Leaf fill and seeding coarser levels");
        fillLeafAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Lower fill and seeding coarser levels");
        fillLowerAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Upper fill and seeding root tiles");
        fillUpperAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Root fill");
        fillRootInvertTiles();
        if (mVerbose==1) mTimer.stop();

        // The root face seeds are scratch: release them now, ordered on mStream behind their last reader.
        mRootFaceSeeds.destroy();

        cudaCheck(cudaStreamSynchronize(mStream));
        const bool hasInvertRootTiles = mRootInvertTileOrigins.size() > 0;
        return {std::move(mRootInvertTileOrigins), hasInvertRootTiles};
    }

private:

    // --- Pipeline stages (run in order by propagate). ---
    // Stage 4 expands only when stages 1-3 seeded an absent root-tile position.

    // Setup: clear the caller's lower and upper masks, and allocate the root face seeds.
    void initializeMasks();

    // Stage 1: fill leaf interiors and seed coarser childless slots from their faces.
    void fillLeafAndSeedCoarserLevels();

    // Stage 2: fill lower interiors and seed coarser childless slots from their faces.
    void fillLowerAndSeedCoarserLevels();

    // Stage 3: fill upper interiors and seed coarser childless slots from their faces.
    void fillUpperAndSeedCoarserLevels();

    // Stage 4: if stages 1-3 seeded an absent root-tile position, expand the seeds through the absent root-tile
    // positions, and return the origins of those reached.
    void fillRootInvertTiles();

    /// @brief Borrowed handle to the injected resource, for the scratch allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t       mStream{0};
    util::cuda::Timer  mTimer;
    int                mVerbose{0};
    const GridT*       mDeviceGrid{nullptr};
    const ValueT*      mDeviceSdf{nullptr};
    ResourceT*         mResource{nullptr};
    nanovdb::cuda::TempPool<ResourceT> mTempDevicePool;
    typename NanoTree<BuildT>::DataType mTreeData{}; // host copy of the grid's tree header, read by initializeMasks()

    // Outputs: the caller's invert masks, one per node of each level, and the origins of the inside absent root tiles.
    nanovdb::Mask<3>*         mLeafInvertMasks{nullptr};
    nanovdb::Mask<4>*         mLowerInvertMasks{nullptr};
    nanovdb::Mask<5>*         mUpperInvertMasks{nullptr};
    BufferT<nanovdb::Coord>   mRootInvertTileOrigins;

    // Root-level scratch. A root-tile position is the place of one root tile, NanoUpper<BuildT>::DIM voxels
    // per axis; it is absent if the root has no tile there, and then the grid does not store it. So the seeds
    // at the root are kept per root-table entry instead: word e is a set of faces (see floodfill::detail::Face),
    // whose bit f is set if the absent root-tile position across face f of entry e is interior.
    BufferT<uint32_t>         mRootFaceSeeds;
}; // tools::cuda::PropagateSign<BuildT, ValueT, ResourceT>

//----------------------------------------------------------------------------------------------------------------------
// Setup: clear the caller's lower and upper masks, and allocate the root face seeds.

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::initializeMasks()
{
    // Inputs are ready in mStream order, so read the grid's tree header on mStream (the read blocks).
    mTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceGrid, mStream);

    // Precondition 4, checked here once so that the kernels can assume it: no active tiles at any level, and a
    // root that holds only child tiles. Every upper node is the child of one root-table entry, so the root holds
    // no value tile, active or not, if and only if its table has as many entries as there are upper nodes.
    const uint32_t entryCount = util::cuda::DeviceGridTraits<BuildT>::getRootTableSize(mDeviceGrid, mTreeData, mStream);
    if (mTreeData.mTileCount[0] || mTreeData.mTileCount[1] || mTreeData.mTileCount[2])
        throw std::runtime_error("PropagateSign: the grid has active tiles (precondition 4)");
    if (entryCount != mTreeData.mNodeCount[2])
        throw std::runtime_error("PropagateSign: the root holds value tiles (precondition 4)");

    // Stage 1 writes every leaf mask whole. Lower and upper masks receive seeds (atomic ORs) before their own
    // fill, so they start at zero.
    cudaCheck(cudaMemsetAsync(mLowerInvertMasks, 0, mTreeData.mNodeCount[1] * sizeof(nanovdb::Mask<4>), mStream));
    cudaCheck(cudaMemsetAsync(mUpperInvertMasks, 0, mTreeData.mNodeCount[2] * sizeof(nanovdb::Mask<5>), mStream));

    // One word per root-table entry (see mRootFaceSeeds), sized by the table rather than by the extent of the
    // active voxels. All zero: no absent root-tile position is interior until a face of a leaf, lower or upper
    // node seeds it.
    mRootFaceSeeds = BufferT<uint32_t>(mStream, this->ref(), entryCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mRootFaceSeeds.data(), 0, mRootFaceSeeds.size_bytes(), mStream));
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

/// @brief The seeds of @a leaf's flood fill: its *active* voxels with a negative SDF value.
/// @details Warp-collective: one warp per leaf, and all 32 lanes call it together and share the work over the
///          leaf's 512 voxels. The seeds are conceptually a Mask<3>, but are returned as one Mask2D<3> x-slice per
///          lane: lane x < 8 gets x-slice x (word x of the Mask<3>: voxels 64 x to 64 x + 63), and lanes x >= 8
///          get empty slices.
/// @param sdf SDF sidecar of the grid, indexed by value index; read only at the leaf's active voxels.
template <typename BuildT, typename ValueT>
__device__ util::morphology::Mask2D<3> leafSeedSlice(const NanoLeaf<BuildT>& leaf, const ValueT* sdf)
{
    // Each iteration tests 32 consecutive voxels, one per lane, and the ballot gathers the answers into one
    // word (bit l: voxel first + l). That word is half of x-slice first / 64, and the lane holding that slice
    // ORs it in. So all 32 lanes test voxels, 16 each, though only lanes 0-7 keep results.
    // In a ValueOnIndex leaf an inactive voxel has value index 0, the background: it is never a seed, and
    // its SDF value is not read.
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

/// @brief A face of a node. A set of faces is a uint32_t, in which bit f stands for face f.
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
    return uint32_t(x == 0 && nonEmpty)                     << minusX |
           uint32_t(x == Dim - 1 && nonEmpty)               << plusX  |
           uint32_t((s.mWords[0] & FirstY) != 0)            << minusY |
           uint32_t((s.mWords[WordCount - 1] & LastY) != 0) << plusY  |
           uint32_t(firstZ != 0)                            << minusZ |
           uint32_t(lastZ != 0)                             << plusZ;
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

/// @brief Seeds the region across @a face of the Level node at @a origin if it is a childless slot of a coarser
///        level. The tree is descended toward the block across the face, and the seed goes where the descent ends:
///        - at the root, which has no entry there: an absent root-tile position, recorded as bit @a face of the
///          @a rootFaceSeeds word of the root-table entry that holds the node. The node lies in that entry's
///          upper node, so a face of the node that borders another root tile lies on the same face of the upper
///          node, and the position across it is the entry's neighbor across that face.
///        - at an upper or lower node without a child there: that slot's bit in @a upperMasks or @a lowerMasks.
///        A descent that reaches a node of the seeding node's level seeds nothing. Many nodes may set the same
///        bit, and the nodes of one root-table entry set bits of the same word, so every bit is set atomically.
template <typename BuildT, int Level>
__device__ inline void seedAcrossFace(const NanoGrid<BuildT>& grid, const nanovdb::Coord& origin, Face face,
                                      nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks,
                                      uint32_t* rootFaceSeeds)
{
    const auto& tree = grid.tree();
    const auto& root = tree.root();
    const nanovdb::Coord ijk = faceNeighbor(origin, face, NanoNode<BuildT, Level>::type::DIM);

    const auto* tile = root.probeTile(ijk);
    if (!tile) { // ends at the root: an absent root-tile position
        const auto* ownTile = root.probeTile(origin); // the seeding node's own entry, which always exists
        NANOVDB_ASSERT(ownTile);
        atomicOr(&rootFaceSeeds[ownTile - root.tile(0)], 1u << face);
        return;
    }
    NANOVDB_ASSERT(tile->isChild()); // precondition 4, checked by initializeMasks(): no root value tiles
    if (uint32_t(Level) == NanoUpper<BuildT>::LEVEL) return; // an upper node across: our own level, which fills itself

    const auto* upper = root.getChild(tile);
    const auto* lower = upper->probeChild(ijk);
    if (!lower) { // ends at an upper node: a childless upper slot
        upperMasks[upper - tree.template getFirstNode<2>()].setOnAtomic(NanoUpper<BuildT>::CoordToOffset(ijk));
        return;
    }
    if (uint32_t(Level) == NanoLower<BuildT>::LEVEL) return; // a lower node across: our own level, which fills itself

    const uint32_t slot = NanoLower<BuildT>::CoordToOffset(ijk);
    if (!lower->childMask().isOn(slot)) // ends at a lower node: a childless lower slot
        lowerMasks[lower - tree.template getFirstNode<1>()].setOnAtomic(slot);
}

/// @brief Fills one leaf (Level 0), lower (1) or upper (2) node, one warp per node, stores its invert mask in
///        @a masks, then seeds the childless slots of coarser levels across the node's inside faces.
///        Warp-collective: all 32 lanes of a warp call it together, and lane x holds x-slice x of the node.
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
                               nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks, uint32_t* rootFaceSeeds)
    {
        // One warp per node; the block's last warps may have no node, and leave together.
        const int      x      = threadIdx.x & 31; // the lane, which holds x-slice x
        const int      warpID = threadIdx.x >> 5; // the block's warp, which fills one node
        const uint32_t nodeID = blockIdx.x * NodesPerBlock + warpID;
        if (nodeID >= nodeCount) return;
        const auto& node     = grid->tree().template getFirstNode<Level>()[nodeID];
        // Lane x holds x-slice x if x < Dim: lanes [0, 8) in a leaf, [0, 16) in a lower node, and all 32 in an
        // upper node. The other lanes hold empty slices, and take part only in the warp's collective operations.
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
        // Under preconditions 1 and 2 (see PropagateSign), a Manhattan-shortest walk from an inside open slot to
        // the nearest wall (in the Manhattan metric) crosses only inside open slots, since any wall on it would be
        // nearer, and ends on a seed: the wall itself in a leaf (a voxel with a negative SDF value), the slot just
        // before it in a lower or upper node (seeded across the child's face, see the stage comment above).
        // So no slot is farther from a seed than the node's Manhattan diameter, 3 * (Dim - 1): at most that
        // many sweeps turn slots on in a leaf, one fewer in a lower or upper node, and one more sweep changes
        // nothing: MaxSweeps, i.e. 22, 45 and 93.
        int sweep = 0;
        for (; sweep <= SlotCount; ++sweep) {
            SliceT neighbors{};
            util::morphology::cuda::MaskShift<-1,  0,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 1,  0,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0, -1,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  1,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  0, -1>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  0,  1>(interior, neighbors);
            // Only open slots are entered, so active voxels keep their start: seeds, or walls if nonnegative. A
            // leaf's seeds may spread into any open neighbor by precondition 2: no voxel with a negative SDF value
            // is face-adjacent to an inactive outside voxel. Beyond that, face-adjacent open slots share a label.
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
            seedAcrossFace<BuildT, Level>(*grid, node.origin(), Face(x), lowerMasks, upperMasks, rootFaceSeeds);
    }
}; // BitFloodFunctor

} // namespace floodfill::detail

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillLeafAndSeedCoarserLevels()
{
    const uint32_t count = mTreeData.mNodeCount[0];
    if (count == 0) return; // a kernel launch with zero blocks is an error
    using Op = floodfill::detail::BitFloodFunctor<BuildT, ValueT, 0>;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, mDeviceSdf, mLeafInvertMasks, count,
        mLowerInvertMasks, mUpperInvertMasks, mRootFaceSeeds.data());
    cudaCheckError();
}// PropagateSign<BuildT, ValueT, ResourceT>::fillLeafAndSeedCoarserLevels

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillLowerAndSeedCoarserLevels()
{
    const uint32_t count = mTreeData.mNodeCount[1];
    if (count == 0) return; // a kernel launch with zero blocks is an error
    using Op = floodfill::detail::BitFloodFunctor<BuildT, ValueT, 1>;
    const ValueT*  noSdf = nullptr;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, noSdf, mLowerInvertMasks, count,
        mLowerInvertMasks, mUpperInvertMasks, mRootFaceSeeds.data());
    cudaCheckError();
}// PropagateSign<BuildT, ValueT, ResourceT>::fillLowerAndSeedCoarserLevels

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillUpperAndSeedCoarserLevels()
{
    const uint32_t count = mTreeData.mNodeCount[2];
    if (count == 0) return; // a kernel launch with zero blocks is an error
    using Op = floodfill::detail::BitFloodFunctor<BuildT, ValueT, 2>;
    const ValueT*  noSdf = nullptr;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(size_t(count) * 32, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, noSdf, mUpperInvertMasks, count,
        mLowerInvertMasks, mUpperInvertMasks, mRootFaceSeeds.data());
    cudaCheckError();
}// PropagateSign<BuildT, ValueT, ResourceT>::fillUpperAndSeedCoarserLevels

//----------------------------------------------------------------------------------------------------------------------

namespace floodfill::detail {

// Stage 4: the interior absent root-tile positions.
//
// Stages 1-3 record each seed at the root as a face bit of a root-table entry (see seedAcrossFace). If no bit is
// set, no absent root-tile position is interior: every connected set of interior ones borders an upper node
// whose face toward it is inside, and that face seeds it. Otherwise the known positions, i.e. the entries and the
// interior positions found so far, grow in rounds. Round 1 adds the seeded positions; each later round adds the
// face neighbors of the positions the previous round added that are not known yet. These are absent and interior
// too: absent positions hold no active voxel, so face-adjacent ones share a label. The rounds stop when one adds
// nothing. A round sorts and deduplicates its candidates and looks them up in the sorted known positions, so its
// work follows the positions it adds, not the extent of the active voxels.
//
// Under the preconditions the interior lies within the bounding box of the entries' positions (the hull), so the
// rounds never leave it. A neighbor outside it means that the input breaks preconditions 2 or 3; it is not added,
// which also keeps the rounds finite. Positions are kept as topology::detail::tileSortKey keys, which sort by
// position.

/// @brief The root-tile position (voxel coordinates >> 12, as in tileSortKey) whose topology::detail::tileSortKey
///        is @a key: each 21-bit field holds a coordinate biased by 2^19.
__hostdev__ inline nanovdb::Coord rootTileOf(uint64_t key)
{
    auto coord = [](uint64_t field) { return int32_t(field & ((uint64_t(1) << 21) - 1)) - (1 << 19); };
    return nanovdb::Coord(coord(key >> 42), coord(key >> 21), coord(key));
}

/// @brief A key of no root-tile position, which sorts after all of them.
constexpr uint64_t NoRootTile = ~uint64_t(0);

/// @brief The topology::detail::tileSortKey of root-table entry @a i, in table order.
template <typename BuildT>
struct RootEntryKeyFunctor
{
    __device__ void operator()(size_t i, const NanoGrid<BuildT>* grid, uint64_t* keys) const
    {
        keys[i] = topology::detail::tileSortKey(grid->tree().root().tile(uint32_t(i))->origin());
    }
}; // RootEntryKeyFunctor

/// @brief The candidates of position @a positions[i], at candidates[6 i + f]: the key of its neighbor across face
///        f if f is in @a faces[i] (in every face if @a faces is null) and the neighbor lies in @a hull, else
///        NoRootTile. A neighbor outside @a hull sets *escaped.
template <typename BuildT>
struct RootNeighborFunctor
{
    __device__ void operator()(size_t i, const uint64_t* positions, const uint32_t* faces, nanovdb::CoordBBox hull,
                               uint64_t* candidates, int* escaped) const
    {
        const nanovdb::Coord p = rootTileOf(positions[i]);
        const uint32_t emitted = faces ? faces[i] : (1u << faceCount) - 1;
        for (int f = 0; f < faceCount; ++f) {
            uint64_t key = NoRootTile;
            if ((emitted >> f) & 1u) {
                const nanovdb::Coord q = faceNeighbor(p, Face(f), 1);
                if (hull.isInside(q)) key = topology::detail::tileSortKey(q << NanoUpper<BuildT>::TOTAL);
                else *escaped = 1;
            }
            candidates[faceCount * i + f] = key;
        }
    }
}; // RootNeighborFunctor

/// @brief isNew[i] = 1 if candidate i, of the first *count candidates (sorted and unique), is a position that is
///        not in the sorted @a known, else 0.
struct NewRootTileFunctor
{
    __device__ void operator()(size_t i, const uint64_t* candidates, const int* count, const uint64_t* known,
                               uint32_t knownCount, uint8_t* isNew) const
    {
        isNew[i] = int(i) < *count && candidates[i] != NoRootTile &&
                   !thrust::binary_search(thrust::seq, known, known + knownCount, candidates[i]);
    }
}; // NewRootTileFunctor

/// @brief origins[i] = the origin of the root-tile position of @a keys[i].
template <typename BuildT>
struct RootTileOriginFunctor
{
    __device__ void operator()(size_t i, const uint64_t* keys, nanovdb::Coord* origins) const
    {
        origins[i] = rootTileOf(keys[i]) << NanoUpper<BuildT>::TOTAL;
    }
}; // RootTileOriginFunctor

} // namespace floodfill::detail

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillRootInvertTiles()
{
    // Decide: read the face seeds back; if none is set, no absent root-tile position is interior.
    const uint32_t entryCount = uint32_t(mRootFaceSeeds.size());
    if (entryCount == 0) return;
    std::vector<uint32_t> faceSeeds(entryCount);
    cudaCheck(cudaMemcpyAsync(faceSeeds.data(), mRootFaceSeeds.data(), mRootFaceSeeds.size_bytes(),
                              cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    if (std::none_of(faceSeeds.begin(), faceSeeds.end(), [](uint32_t seeds) { return seeds != 0; }))
        return;

    // Expand. The entries' keys, in table order so that round 1 pairs them with their face seeds, and their hull.
    static constexpr unsigned int threadsPerBlock = 128;
    BufferT<uint64_t> entryKeys(mStream, this->ref(), entryCount, nanovdb::cuda::noInit);
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(entryCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        entryCount, floodfill::detail::RootEntryKeyFunctor<BuildT>{}, mDeviceGrid, entryKeys.data());
    cudaCheckError();
    std::vector<uint64_t> hostEntryKeys(entryCount);
    cudaCheck(cudaMemcpyAsync(hostEntryKeys.data(), entryKeys.data(), entryKeys.size_bytes(), cudaMemcpyDeviceToHost,
                              mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    nanovdb::CoordBBox hull; // empty until expanded
    for (const uint64_t key : hostEntryKeys) hull.expand(floodfill::detail::rootTileOf(key));

    // known: the entries and the positions added so far, sorted; added: the positions added so far, sorted;
    // frontier: the positions a round emits candidates from, with their face seeds in round 1, all faces later.
    BufferT<uint64_t> known(mStream, this->ref(), entryCount, nanovdb::cuda::noInit);
    CALL_CUBS(DeviceRadixSort::SortKeys, entryKeys.data(), known.data(), int(entryCount), 0, 64);
    uint32_t          knownCount = entryCount;
    BufferT<uint64_t> added(mStream, this->ref(), 0, nanovdb::cuda::noInit);
    uint32_t          addedCount = 0;
    BufferT<uint64_t> frontier = std::move(entryKeys);
    uint32_t          frontierCount = entryCount;
    const uint32_t* frontierFaces = mRootFaceSeeds.data();

    // counts[0]: the number of keys a cub selection kept; counts[1]: set if a neighbor fell outside the hull
    BufferT<int> counts(mStream, this->ref(), 2, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(counts.data(), 0, counts.size_bytes(), mStream));
    int hostCounts[2] = {0, 0};
    while (true) {
        const uint32_t candidateCount = floodfill::detail::faceCount * frontierCount;
        BufferT<uint64_t> candidates(mStream, this->ref(), candidateCount, nanovdb::cuda::noInit);
        BufferT<uint64_t> sorted(mStream, this->ref(), candidateCount, nanovdb::cuda::noInit);
        BufferT<uint64_t> unique(mStream, this->ref(), candidateCount, nanovdb::cuda::noInit);
        BufferT<uint8_t>  isNew(mStream, this->ref(), candidateCount, nanovdb::cuda::noInit);
        BufferT<uint64_t> next(mStream, this->ref(), candidateCount, nanovdb::cuda::noInit);

        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(frontierCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            frontierCount, floodfill::detail::RootNeighborFunctor<BuildT>{}, frontier.data(), frontierFaces, hull,
            candidates.data(), counts.data() + 1);
        cudaCheckError();
        CALL_CUBS(DeviceRadixSort::SortKeys, candidates.data(), sorted.data(), int(candidateCount), 0, 64);
        CALL_CUBS(DeviceSelect::Unique, sorted.data(), unique.data(), counts.data(), int(candidateCount));
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(candidateCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            candidateCount, floodfill::detail::NewRootTileFunctor{}, unique.data(), counts.data(), known.data(),
            knownCount, isNew.data());
        cudaCheckError();
        CALL_CUBS(DeviceSelect::Flagged, unique.data(), isNew.data(), next.data(), counts.data(), int(candidateCount));

        // The host sizes the next round from this one, so each round reads its count back
        cudaCheck(cudaMemcpyAsync(hostCounts, counts.data(), sizeof(hostCounts), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        const uint32_t nextCount = uint32_t(hostCounts[0]);
        if (nextCount == 0) break;

        // Merge the new positions, sorted and disjoint from both, into known and into added
        BufferT<uint64_t> mergedKnown(mStream, this->ref(), knownCount + nextCount, nanovdb::cuda::noInit);
        CALL_CUBS(DeviceMerge::MergeKeys, known.data(), int64_t(knownCount), next.data(), int64_t(nextCount),
                  mergedKnown.data(), ::cuda::std::less<uint64_t>{});
        known = std::move(mergedKnown);
        knownCount += nextCount;
        BufferT<uint64_t> mergedAdded(mStream, this->ref(), addedCount + nextCount, nanovdb::cuda::noInit);
        CALL_CUBS(DeviceMerge::MergeKeys, added.data(), int64_t(addedCount), next.data(), int64_t(nextCount),
                  mergedAdded.data(), ::cuda::std::less<uint64_t>{});
        added = std::move(mergedAdded);
        addedCount += nextCount;
        frontier = std::move(next);
        frontierCount = nextCount;
        frontierFaces = nullptr;
    }
    NANOVDB_ASSERT(hostCounts[1] == 0); // a neighbor outside the hull: the input breaks preconditions 2 or 3

    if (addedCount == 0) return;
    mRootInvertTileOrigins = BufferT<nanovdb::Coord>(mStream, this->ref(), addedCount, nanovdb::cuda::noInit);
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(addedCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        addedCount, floodfill::detail::RootTileOriginFunctor<BuildT>{}, added.data(), mRootInvertTileOrigins.data());
    cudaCheckError();
}// PropagateSign<BuildT, ValueT, ResourceT>::fillRootInvertTiles

//----------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#ifdef CALL_CUBS
#undef CALL_CUBS
#endif

#endif // NANOVDB_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
