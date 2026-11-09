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
#include <thrust/binary_search.h>    // thrust::binary_search (cuda/std/algorithm needs CCCL 3.2)
#include <thrust/execution_policy.h> // thrust::seq

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

/// @brief Computes the invert masks of a ValueOnIndex level set: where its inactive voxels take the background
///        value with its sign inverted.
/// @details A ValueOnIndex level set stores values, in its sidecar, only for its active voxels, the narrow band.
///          PropagateSign adds a semantic trait for the inactive voxels beyond the band, wherever the tree holds
///          them (leaf voxels, childless lower and upper slots, and upper-node-sized regions without a root-table
///          entry): a bit that says whether the voxel takes the background value or its negation. The grid is not
///          modified. Rather than an inside/outside predicate, which would presume that infinity is always outside,
///          the masks only disambiguate the sign of the background, and the signed value carries the determination.
///
///          The input is assumed to have positive SDF values on the side that reaches infinity, so the bits are off
///          on the unbounded region. They are on for the inactive voxels connected, through inactive voxels, to
///          active voxels with negative values; the bits of active voxels are off.
///
///          The caller allocates the leaf, lower and upper masks, which propagate() fills, and propagate() returns
///          the root-level result:
///          - leaf: one Mask<3> per leaf; bit n is the bit of voxel n (off if the voxel is active).
///          - lower and upper: one Mask per node; bit n is the bit of slot n (off if the slot has a child).
///          - root: the origins of the upper-node-sized regions without a root-table entry whose bit is on, in
///            increasing (x, y, z) order.
///
///          Preconditions (PruneGrid and MeshToGrid output satisfies 1 and 4). Only 4 is checked, by propagate(),
///          which throws if it fails; the others are the caller's to ensure:
///          1. No empty nodes: every leaf has an active voxel, and every lower and upper node has a child.
///          2. Each face-connected region of inactive voxels touches active voxels of one sign only (zero counts
///             as positive).
///          3. The unbounded region touches no active voxel with a negative value.
///          4. Active values are leaf voxels only (no active tiles), and the root holds only child tiles.
/// @tparam BuildT The grid's build type, which must be ValueOnIndex.
/// @tparam ValueT Floating-point type of the SDF values.
/// @tparam ResourceT Stream-ordered resource used for the root-tile output and all scratch allocations.
template <typename BuildT, typename ValueT = float, typename ResourceT = nanovdb::cuda::DeviceResource>
class PropagateSign
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex, "PropagateSign requires a ValueOnIndex grid");
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
    /// @param deviceSdf SDF sidecar: the SDF value of each active voxel at its value index. Only these values are
    ///        read, and only their signs matter (zero counts as positive).
    /// @param stream Stream on which all work and allocations are ordered.
    /// @param resource Allocation resource; must outlive the returned buffer.
    /// todo: Accept root value tiles, which OpenVDB level sets converted to NanoVDB can have (rejected for now).
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

    /// @brief Compute the invert masks: fill the caller's leaf, lower and upper masks, and return the origins of the
    ///        upper-node-sized regions without a root-table entry that it finds to be interior.
    /// @param leafMasks, lowerMasks, upperMasks Device arrays of one mask per leaf, lower and upper node, indexed
    ///        like the grid's nodes; need not be initialized.
    /// @return { origins, hasInvertRootTiles }: a device buffer with those origins, and whether there is any.
    /// @throw std::runtime_error if the grid has active tiles or root value tiles (precondition 4), or if the
    ///        interior reaches past the bounding box of the root-table entries (preconditions 2 or 3).
    /// @note The stream is synchronized before this method returns.
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

        if (mVerbose==1) mTimer.start("Upper fill and seeding coarser levels");
        fillUpperAndSeedCoarserLevels();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Root tile expansion");
        fillRootInvertTiles();
        if (mVerbose==1) mTimer.stop();

        // The root face seeds are scratch: release them now, ordered on mStream behind their last reader.
        mRootFaceSeeds.destroy();

        cudaCheck(cudaStreamSynchronize(mStream));
        const bool hasInvertRootTiles = mRootInvertTileOrigins.size() > 0;
        return {std::move(mRootInvertTileOrigins), hasInvertRootTiles};
    }

private:

    // Pipeline stages, run in order by propagate().

    // Setup: check precondition 4, clear the caller's lower and upper masks, and allocate the root face seeds.
    void initializeMasks();

    // Stages 1-3: fill the leaf, lower and upper nodes, and seed coarser levels across their faces.
    void fillLeafAndSeedCoarserLevels();
    void fillLowerAndSeedCoarserLevels();
    void fillUpperAndSeedCoarserLevels();

    // Stage 4: if stages 1-3 set any root seed, expand the seeds through the tiles without a root-table entry. It
    // forwards to floodfill::detail::fillRootInvertTiles, which takes its inputs explicitly, so that tests reach it.
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

    // Outputs: the caller's invert masks, and the origins of the tiles without a root-table entry found to be interior.
    nanovdb::Mask<3>*         mLeafInvertMasks{nullptr};
    nanovdb::Mask<4>*         mLowerInvertMasks{nullptr};
    nanovdb::Mask<5>*         mUpperInvertMasks{nullptr};
    BufferT<nanovdb::Coord>   mRootInvertTileOrigins;

    // Root seeds, one word per root-table entry: bit f (see floodfill::detail::Face) is set if the tile across
    // face f, which has no root-table entry, is inverted.
    BufferT<uint32_t>         mRootFaceSeeds;
}; // tools::cuda::PropagateSign<BuildT, ValueT, ResourceT>

//----------------------------------------------------------------------------------------------------------------------
// Setup: check precondition 4, clear the caller's lower and upper masks, and allocate the root face seeds.

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::initializeMasks()
{
    // Read the grid's tree header on mStream (blocking).
    mTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceGrid, mStream);

    // Precondition 4, checked once so that the kernels can assume it. Every upper node has its own root-table
    // entry, so the root holds no value tile, active or not, if and only if the table has as many entries as there
    // are upper nodes.
    const uint32_t entryCount = util::cuda::DeviceGridTraits<BuildT>::getRootTableSize(mDeviceGrid, mTreeData, mStream);
    if (mTreeData.mTileCount[0] || mTreeData.mTileCount[1] || mTreeData.mTileCount[2])
        throw std::runtime_error("PropagateSign: the grid has active tiles");
    if (entryCount != mTreeData.mNodeCount[2])
        throw std::runtime_error("PropagateSign: the root holds value tiles");

    // Stage 4 bounds its search by the root's index bounding box, which NanoVDB's builders compute. An empty grid
    // has none, and needs none. Debug builds only: the flags read back (blocking) is compiled out with the assert.
    NANOVDB_ASSERT(entryCount == 0 ||
                   util::cuda::DeviceGridTraits<BuildT>::getFlags(mDeviceGrid, mStream).isMaskOn(GridFlags::HasBBox));

    // Stage 1 writes every leaf mask whole. Lower and upper masks receive seeds (atomic ORs) before their own
    // fill, so they start at zero.
    cudaCheck(cudaMemsetAsync(mLowerInvertMasks, 0, mTreeData.mNodeCount[1] * sizeof(nanovdb::Mask<4>), mStream));
    cudaCheck(cudaMemsetAsync(mUpperInvertMasks, 0, mTreeData.mNodeCount[2] * sizeof(nanovdb::Mask<5>), mStream));

    // The root seeds start at zero; stages 1-3 set them.
    mRootFaceSeeds = BufferT<uint32_t>(mStream, this->ref(), entryCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mRootFaceSeeds.data(), 0, mRootFaceSeeds.size_bytes(), mStream));
}// PropagateSign<BuildT, ValueT, ResourceT>::initializeMasks

//----------------------------------------------------------------------------------------------------------------------

namespace floodfill::detail {

// Stages 1-3: fill each leaf, lower or upper node, then seed coarser levels across its faces.
//
// The reference result is a flood fill at voxel resolution from the active voxels with a negative SDF value,
// through face-adjacent inactive voxels, stopped by the other active voxels. The inactive voxels it reaches are
// marked in the invert masks: the background value there is the negative of the one in slot 0 of the sidecar.
// The locations it covers, its seeds and the inactive voxels it reaches, are the interior (by preconditions 2 and 3,
// the bounded side of the zero crossing); a childless slot of a lower or upper node is interior if its voxels are.
// Under preconditions 1-4 the fill can be done node by node, finest level first:
// - A leaf's fill is exact on its own: a marked region within a leaf borders only negative voxels (2) and is not
//   the whole leaf (1), so it touches a negative voxel of that leaf.
// - A node next to a marked childless slot has its whole face toward it marked or negative (2), so the finest
//   nodes along that face seed the slot (seedAcrossFace); every marked region of childless slots borders such a
//   node (1, 3). So each level seeds every coarser level directly, and since seeds only go to strictly coarser
//   levels, each level is final before a coarser fill reads it.
//
// Each node is filled with a Jacobi flood fill: walls are the active voxels in a leaf and the slots with a child
// in a lower or upper node, and each sweep turns on every open slot next to a reached one. By preconditions 1 and
// 2 the flood fill never has to go around a wall, so the number of sweeps is bounded by the node's size (see
// BitFloodFunctor).
//
// One kernel per level, one warp per node: lane x holds x-slice x of the node's mask as a util::morphology::Mask2D
// (lanes x >= Dim hold empty slices), fills it with the warp MaskShift of util/cuda/MorphologyHelpers.cuh, stores
// its inactive part as the node's invert mask, and seeds coarser levels across the node's faces.

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
        const bool     seed   = index != 0 && sdf[index] < ValueT(0); // zero counts as positive
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
// Trigger: a face of a filled node that holds an interior slot, meaning either
// - an active voxel with a negative SDF value (leaves only; its invert bit is off), or
// - a slot marked in the node's invert mask by its fill.
// Any one interior slot on the face triggers it.
//
// Outcome: the coordinate across a trigger face (the node's origin, shifted by the node's size) is looked up in the
// tree. If the lookup ends at a level coarser than the node's, it ends at a slot without a child, and that slot is
// marked:
// - in an upper or lower node: its bit in the upper or lower masks, from which that level's fill starts.
// - at the root, where the lookup ends because there is no root-table entry: face bit f in the root seed word of the
//   node's own root-table entry, from which stage 4 starts.
// If the lookup instead reaches a node of the seeding node's level, nothing is marked: that node is filled on its own.
// Any coordinate of the node-sized block across the face gives the same result, since nodes are aligned to their
// size, so one lookup per face suffices.

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

/// @brief Seeds across @a face of the Level node at @a origin (see the outcome above). The tree is descended toward
///        the block across the face, until the lookup ends or reaches a node of the seeding node's level. At the root,
///        the lookup can leave the node's upper node only across the same face of the upper node, so bit @a face of
///        the node's own root-table entry stands for the upper-node-sized region across that face of the upper node.
///        Bits are set atomically: many nodes may set the same bit, and the nodes of one root-table entry share its
///        word.
template <typename BuildT, int Level>
__device__ inline void seedAcrossFace(const NanoGrid<BuildT>& grid, const nanovdb::Coord& origin, Face face,
                                      nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks,
                                      uint32_t* rootFaceSeeds)
{
    const auto& tree = grid.tree();
    const auto& root = tree.root();
    const nanovdb::Coord ijk = faceNeighbor(origin, face, NanoNode<BuildT, Level>::type::DIM);

    const auto* tile = root.probeTile(ijk);
    if (!tile) { // the lookup ends at the root: no root-table entry there
        const auto* ownTile = root.probeTile(origin); // the seeding node's own entry, which always exists
        NANOVDB_ASSERT(ownTile);
        atomicOr(&rootFaceSeeds[ownTile - root.tile(0)], 1u << face);
        return;
    }
    NANOVDB_ASSERT(tile->isChild()); // precondition 4, checked by initializeMasks(): no root value tiles
    if (uint32_t(Level) == NanoUpper<BuildT>::LEVEL) return; // reaches an upper node: our own level

    const auto* upper = root.getChild(tile);
    const auto* lower = upper->probeChild(ijk);
    if (!lower) { // ends at an upper slot without a child
        upperMasks[upper - tree.template getFirstNode<2>()].setOnAtomic(NanoUpper<BuildT>::CoordToOffset(ijk));
        return;
    }
    if (uint32_t(Level) == NanoLower<BuildT>::LEVEL) return; // reaches a lower node: our own level

    const uint32_t slot = NanoLower<BuildT>::CoordToOffset(ijk);
    if (!lower->childMask().isOn(slot)) // ends at a lower slot without a child
        lowerMasks[lower - tree.template getFirstNode<1>()].setOnAtomic(slot);
}

/// @brief Fills one leaf (Level 0), lower (1) or upper (2) node, one warp per node, stores its invert mask in
///        @a masks, then seeds across the node's trigger faces (see seedAcrossFace).
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

        // The interior starts as the seeds; open slots are those the fill may spread into.
        SliceT interior{}, open{};
        if constexpr (Level == 0) {
            // Leaf: seeds are active voxels with a negative SDF value; inactive voxels are open.
            interior = leafSeedSlice(node, sdf);
            if (hasSlice) {
                open = SliceT::load(node.valueMask(), x);
                open.toggle();
            }
        } else if (hasSlice) {
            // Lower/upper: seeds were marked in masks by finer levels (seedAcrossFace); childless slots are open.
            open = SliceT::load(node.childMask(), x);
            open.toggle();
            interior = SliceT::load(masks[nodeID], x) & open;
        }

        // Each sweep adds the open face neighbors of the interior slots so far. A sweep that changes nothing ends
        // the fill; any other sweep adds at least one slot, so SlotCount + 1 sweeps suffice for any input.
        // Under preconditions 1 and 2 (see PropagateSign), a Manhattan-shortest walk from an interior open slot to the
        // nearest wall crosses only interior open slots, since any wall on it would be nearer, and ends on a seed: the
        // wall itself in a leaf (an active voxel with a negative SDF value), or the slot just before it in a lower or
        // upper node (seeded across the child's face, see seedAcrossFace). So no slot is farther from a seed than the
        // node's Manhattan diameter, 3 * (Dim - 1): at most that many sweeps turn slots on in a leaf, one fewer in a
        // lower or upper node, and one more sweep changes nothing: MaxSweeps, i.e. 22, 45 and 93.
        int sweep = 0;
        for (; sweep <= SlotCount; ++sweep) {
            SliceT neighbors{};
            util::morphology::cuda::MaskShift<-1,  0,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 1,  0,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0, -1,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  1,  0>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  0, -1>(interior, neighbors);
            util::morphology::cuda::MaskShift< 0,  0,  1>(interior, neighbors);
            // Only open slots are entered, so active voxels keep their start: seeds if negative, walls otherwise.
            // By precondition 2, the face-connected open slots around a seed are all interior.
            const SliceT next = interior | (open & neighbors);
            if (!__any_sync(0xFFFFFFFFu, next != interior)) break;
            interior = next;
        }
        NANOVDB_ASSERT(sweep < MaxSweeps); // the input breaks preconditions 1 or 2

        // The invert mask holds only the open interior slots: a leaf's interior active voxels take their own
        // values, so their bits are off (in lower and upper nodes all interior slots are open). The face test below
        // still uses all of them, since any interior slot triggers a face (see seedAcrossFace).
        if (hasSlice) (interior & open).store(masks[nodeID], x);

        // Bit f is set if face f is triggered: it holds an interior slot. Each lane first finds the faces on which
        // its own slice has one; the xor-shuffle butterfly then ORs these sets over all 32 lanes, so every lane ends
        // up with the set for the whole node.
        uint32_t triggerFaces = facesWithOnSlots(interior, x);
        for (int offset = 16; offset > 0; offset >>= 1) triggerFaces |= __shfl_xor_sync(0xFFFFFFFFu, triggerFaces, offset);

        // Lane f seeds across face f if it is triggered.
        if (x < faceCount && ((triggerFaces >> x) & 1u))
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

// Stage 4: the interior upper-node-sized regions without a root-table entry.
//
// Here a tile means a root tile: an upper-node-sized region of index space aligned to its size. Besides the tiles of
// the root table, this includes the potential new entries needed to cover the interior, even though they hold no
// active voxels. In stages 1-3, whenever the flood fill needs to extend into a tile without a root-table entry, it
// records that need as a directional flag on the neighboring entry, the one that holds the trigger face: bit f of the
// entry's root face seeds stands for the tile across its face f (see seedAcrossFace). The flood fill can only enter
// such a tile across a trigger face, so if no flag is set, no such tile is interior. Otherwise the flagged tiles grow
// in rounds, one face-neighbor step at a time, filling the cavity of such tiles that they belong to: nothing stops
// the fill there, since these tiles hold no active voxels. By precondition 3 the cavity is bounded, and with each
// step spanning an upper node, the rounds are few compared with the work of stages 1-3 on the narrow band around it.
//
// So the cavity lies within the bounding box of the root-table entries (the hull), since every tile outside it
// connects to infinity. A candidate outside the hull means that the input breaks preconditions 2 or 3: it is not
// added, and fillRootInvertTiles() throws once its round ends, so the rounds end for any input. The hull is derived
// from the root's index bounding box (see RootNeighborFunctor). Tiles are kept as the
// topology::detail::tileSortKey of their origins, their keys, which sort by origin.

/// @brief The origin of the tile whose topology::detail::tileSortKey is @a key: the inverse of tileSortKey,
///        up to the tile, i.e. tileSortKeyToOrigin(tileSortKey(ijk)) == ijk & ~4095.
/// todo: Move next to tileSortKey, in a shared header for the tree sort keys that several CUDA tools duplicate,
///       so that the encoding and its inverse live in one place.
__hostdev__ inline nanovdb::Coord tileSortKeyToOrigin(uint64_t key)
{
    // unbias, then scale rather than shift (a left shift of a negative value is undefined before C++20)
    auto field = [](uint64_t f) { return (int32_t(f & ((uint64_t(1) << 21) - 1)) - (1 << 19)) * (1 << 12); };
    return nanovdb::Coord(field(key >> 42), field(key >> 21), field(key));
}

/// @brief Sentinel for a candidate slot that holds no candidate: it is no tile's key, and sorts after every key.
constexpr uint64_t NoRootTile = ~uint64_t(0);

/// @brief The device-side results of a round, read back together: uniqueCount, the candidates left after removing
///        repeats; newCount, the new tiles among them (both written by cub each round); and escaped, set if a
///        neighbor fell outside the hull (never cleared: the host throws once it is set).
struct RoundCounts
{
    int uniqueCount;
    int newCount;
    int escaped;
}; // RoundCounts

/// @brief keys[i] = the topology::detail::tileSortKey of root-table entry i's origin, in table order, so that
///        keys[i] pairs with the root face seeds of entry i.
template <typename BuildT>
struct RootEntryKeyFunctor
{
    __device__ void operator()(size_t i, const NanoGrid<BuildT>* grid, uint64_t* keys) const
    {
        keys[i] = topology::detail::tileSortKey(grid->tree().root().tile(uint32_t(i))->origin());
    }
}; // RootEntryKeyFunctor

/// @brief The candidates of the tile with key @a tiles[i], at candidates[6 i + f]: the key of its neighbor across
///        face f if f is in @a faces[i] (in every face if @a faces is null) and the neighbor lies in the hull, else
///        NoRootTile. A neighbor outside the hull sets *escaped. The hull, the bounding box of the root-table entries'
///        origins, is the root's index bounding box with its min rounded down to an origin; its max needs no
///        rounding, since only origins are tested against it.
template <typename BuildT>
struct RootNeighborFunctor
{
    __device__ void operator()(size_t i, const NanoGrid<BuildT>* grid, const uint64_t* tiles, const uint32_t* faces,
                               uint64_t* candidates, int* escaped) const
    {
        const nanovdb::CoordBBox& bbox = grid->tree().root().bbox();
        const nanovdb::CoordBBox  hull(bbox.min() & ~int(NanoUpper<BuildT>::DIM - 1), bbox.max());
        const nanovdb::Coord p = tileSortKeyToOrigin(tiles[i]);
        const uint32_t emitted = faces ? faces[i] : (1u << faceCount) - 1;
        for (int f = 0; f < faceCount; ++f) {
            uint64_t key = NoRootTile;
            if ((emitted >> f) & 1u) {
                const nanovdb::Coord q = faceNeighbor(p, Face(f), NanoUpper<BuildT>::DIM);
                if (hull.isInside(q)) key = topology::detail::tileSortKey(q);
                else *escaped = 1;
            }
            candidates[faceCount * i + f] = key;
        }
    }
}; // RootNeighborFunctor

/// @brief isNew[i] = 1 if candidate i, of the first *uniqueCount candidates (sorted and unique), is a tile that
///        is not in the sorted @a known, else 0.
struct NewRootTileFunctor
{
    __device__ void operator()(size_t i, const uint64_t* candidates, const int* uniqueCount, const uint64_t* known,
                               uint32_t knownCount, uint8_t* isNew) const
    {
        isNew[i] = int(i) < *uniqueCount && candidates[i] != NoRootTile &&
                   !thrust::binary_search(thrust::seq, known, known + knownCount, candidates[i]);
    }
}; // NewRootTileFunctor

/// @brief origins[i] = the origin of the tile whose key is @a keys[i].
struct RootTileOriginFunctor
{
    __device__ void operator()(size_t i, const uint64_t* keys, nanovdb::Coord* origins) const
    {
        origins[i] = tileSortKeyToOrigin(keys[i]);
    }
}; // RootTileOriginFunctor

/// @brief Stage 4 of PropagateSign: the origins of the interior tiles without a root-table entry, from the root face
///        seeds that stages 1-3 set (see seedAcrossFace).
/// @param deviceGrid Device grid; only its root table and its root's index bounding box are read.
/// @param deviceRootFaceSeeds Device array of one word per root-table entry, in table order: bit f (see Face) is set
///        if the tile across face f of the entry, which has no root-table entry, is interior.
/// @param entryCount The number of root-table entries.
/// @param stream Stream on which all work and allocations are ordered.
/// @param resource Resource that the result and the scratch are allocated from.
/// @param tempPool Pool for cub's temporary storage, drawing on @a resource.
/// @return The origins, in increasing (x, y, z) order; empty if there are none.
/// @throw std::runtime_error if the interior reaches past the bounding box of the root-table entries (preconditions 2
///        or 3 of PropagateSign).
template <typename BuildT, typename ResourceT>
nanovdb::cuda::Buffer<nanovdb::Coord, nanovdb::cuda::ResourceRef<ResourceT>>
fillRootInvertTiles(const NanoGrid<BuildT>* deviceGrid, const uint32_t* deviceRootFaceSeeds, uint32_t entryCount,
                    cudaStream_t stream, ResourceT& resource, nanovdb::cuda::TempPool<ResourceT>& tempPool)
{
    using RefT          = nanovdb::cuda::ResourceRef<ResourceT>;
    using KeyBufferT    = nanovdb::cuda::Buffer<uint64_t, RefT>;
    using FlagBufferT   = nanovdb::cuda::Buffer<uint8_t, RefT>;
    using CountsBufferT = nanovdb::cuda::Buffer<RoundCounts, RefT>;
    using CoordBufferT  = nanovdb::cuda::Buffer<nanovdb::Coord, RefT>;

    // CALL_CUBS draws on mTempDevicePool and orders its calls on mStream, by name.
    nanovdb::cuda::TempPool<ResourceT>& mTempDevicePool = tempPool;
    const cudaStream_t                  mStream = stream;
    const RefT                          ref(resource);
    CoordBufferT                        origins(mStream, ref, 0, nanovdb::cuda::noInit);

    // Decide: read the root face seeds back; if no flag is set, no tile without a root-table entry is interior.
    if (entryCount == 0) return origins; // null grid, no root tiles
    std::vector<uint32_t> faceSeeds(entryCount);
    cudaCheck(cudaMemcpyAsync(faceSeeds.data(), deviceRootFaceSeeds, entryCount * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    if (std::none_of(faceSeeds.begin(), faceSeeds.end(), [](uint32_t seeds) { return seeds != 0; }))
        return origins;

    // Expand. The entries' keys, in table order so that round 1 pairs them with their face seeds.
    static constexpr unsigned int threadsPerBlock = 128;
    KeyBufferT entryKeys(mStream, ref, entryCount, nanovdb::cuda::noInit);
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(entryCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        entryCount, RootEntryKeyFunctor<BuildT>{}, deviceGrid, entryKeys.data());
    cudaCheckError();

    // The search state, kept across rounds; each key list has its count.
    // - frontier: the tiles a round emits candidates from: the entries in round 1, in table order, then the
    //   tiles the previous round added. frontierFaces: the faces it emits across, the face seeds in round 1
    //   and all faces later (null).
    // - known: the entries and the tiles added so far, sorted; a candidate in it is not new.
    // - added: the tiles added so far, sorted: the result.
    // - counts: the round's device-side results (see RoundCounts), zeroed once; uniqueCount, newCount and escaped
    //   point to its members.
    KeyBufferT      frontier = std::move(entryKeys);
    KeyBufferT      known(mStream, ref, entryCount, nanovdb::cuda::noInit);
    KeyBufferT      added(mStream, ref, 0, nanovdb::cuda::noInit);
    CountsBufferT   counts(mStream, ref, 1, nanovdb::cuda::noInit);
    int*            uniqueCount = &counts.data()->uniqueCount;
    int*            newCount = &counts.data()->newCount;
    int*            escaped = &counts.data()->escaped;
    uint32_t        frontierCount = entryCount, knownCount = entryCount, addedCount = 0;
    const uint32_t* frontierFaces = deviceRootFaceSeeds;
    CALL_CUBS(DeviceRadixSort::SortKeys, frontier.data(), known.data(), int(entryCount), 0, 64);
    cudaCheck(cudaMemsetAsync(counts.data(), 0, counts.size_bytes(), mStream));
    while (true) {
        // The round's scratch, one slot per face of each frontier tile:
        // - candidates: the keys of the neighbors the round emits, NoRootTile in unused slots;
        // - sorted, unique: the candidates sorted, then without repeats (uniqueCount of them);
        // - isNew: 1 where a unique candidate is a tile not in known;
        // - next: the new tiles, sorted (newCount of them): the next frontier.
        const uint32_t candidateCount = faceCount * frontierCount;
        KeyBufferT     candidates(mStream, ref, candidateCount, nanovdb::cuda::noInit);
        KeyBufferT     sorted(mStream, ref, candidateCount, nanovdb::cuda::noInit);
        KeyBufferT     unique(mStream, ref, candidateCount, nanovdb::cuda::noInit);
        FlagBufferT    isNew(mStream, ref, candidateCount, nanovdb::cuda::noInit);
        KeyBufferT     next(mStream, ref, candidateCount, nanovdb::cuda::noInit);

        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(frontierCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            frontierCount, RootNeighborFunctor<BuildT>{}, deviceGrid, frontier.data(), frontierFaces, candidates.data(),
            escaped);
        cudaCheckError();
        CALL_CUBS(DeviceRadixSort::SortKeys, candidates.data(), sorted.data(), int(candidateCount), 0, 64);
        CALL_CUBS(DeviceSelect::Unique, sorted.data(), unique.data(), uniqueCount, int(candidateCount));
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(candidateCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            candidateCount, NewRootTileFunctor{}, unique.data(), uniqueCount, known.data(), knownCount, isNew.data());
        cudaCheckError();
        CALL_CUBS(DeviceSelect::Flagged, unique.data(), isNew.data(), next.data(), newCount, int(candidateCount));

        // The host sizes the next round from this one, so each round reads back its counts: the new count, and the
        // escaped flag.
        RoundCounts hostCounts{};
        cudaCheck(cudaMemcpyAsync(&hostCounts, counts.data(), counts.size_bytes(), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        if (hostCounts.escaped)
            throw std::runtime_error("PropagateSign: the interior reaches past the bounding box of the root-table "
                                     "entries");
        const uint32_t nextCount = uint32_t(hostCounts.newCount);
        if (nextCount == 0) break;

        // Merge the new tiles, sorted and disjoint from both, into known and into added: each merge writes a
        // new list, which replaces the old one.
        KeyBufferT mergedKnown(mStream, ref, knownCount + nextCount, nanovdb::cuda::noInit);
        KeyBufferT mergedAdded(mStream, ref, addedCount + nextCount, nanovdb::cuda::noInit);
        CALL_CUBS(DeviceMerge::MergeKeys, known.data(), int64_t(knownCount), next.data(), int64_t(nextCount),
                  mergedKnown.data(), ::cuda::std::less<uint64_t>{});
        CALL_CUBS(DeviceMerge::MergeKeys, added.data(), int64_t(addedCount), next.data(), int64_t(nextCount),
                  mergedAdded.data(), ::cuda::std::less<uint64_t>{});
        known = std::move(mergedKnown);
        added = std::move(mergedAdded);
        knownCount += nextCount;
        addedCount += nextCount;
        frontier = std::move(next);
        frontierCount = nextCount;
        frontierFaces = nullptr;
    }

    if (addedCount == 0) return origins;
    origins = CoordBufferT(mStream, ref, addedCount, nanovdb::cuda::noInit);
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(addedCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        addedCount, RootTileOriginFunctor{}, added.data(), origins.data());
    cudaCheckError();
    return origins;
}// fillRootInvertTiles

} // namespace floodfill::detail

template <typename BuildT, typename ValueT, typename ResourceT>
void PropagateSign<BuildT, ValueT, ResourceT>::fillRootInvertTiles()
{
    mRootInvertTileOrigins = floodfill::detail::fillRootInvertTiles(mDeviceGrid, mRootFaceSeeds.data(),
                                                                     uint32_t(mRootFaceSeeds.size()), mStream,
                                                                     *mResource, mTempDevicePool);
}// PropagateSign<BuildT, ValueT, ResourceT>::fillRootInvertTiles

//----------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#ifdef CALL_CUBS
#undef CALL_CUBS
#endif

#endif // NANOVDB_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
