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

        cudaCheck(cudaStreamSynchronize(mStream));
        return {std::move(mLeafInteriorMasks), std::move(mLowerInteriorMasks),
                std::move(mUpperInteriorMasks), std::move(mRootInteriorTileOrigins)};
    }

private:

    // --- Pipeline stages (run in order by propagate). ---
    // Stage 4 runs only when the active root bounds contain positions without Upper children.

    // Setup: allocate per-node masks and a dense root mask.
    void initializeMasks();

    // Stage 1: fill leaf interiors and seed coarser childless slots from their faces.
    void fillLeafAndSeedCoarserLevels();

    // Stage 2: fill lower interiors and seed coarser childless slots from their faces.
    void fillLowerAndSeedCoarserLevels();

    // Stage 3: fill upper interiors and seed coarser childless slots from their faces.
    void fillUpperAndSeedCoarserLevels();

    // Stage 4: fill root positions with Upper children as walls, then collect interior tile origins.
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

namespace signing::detail {

// Stages 1, 2 and 3: fill each leaf, lower or upper node, then seed coarser levels across its faces.
//
// In a leaf, active voxels keep their own sign and act as walls; the inside ones seed a flood through the
// leaf's inactive voxels. In a lower or upper node, slots with a child are walls, and the seeds are the
// slots that the faces of finer nodes set in the previous stages. The flood is Jacobi style: in each sweep
// every unreached open slot turns on if a face neighbor is already on, so a sweep is one parallel step with
// no ordering between threads. A scanline fill, which runs to the end of an open row in one sweep, needs
// fewer sweeps only when the inside region winds through long corridors; next to a surface the flood
// crosses a node in a few sweeps, so the simpler Jacobi sweep is no slower there.
//
// One kernel per level, one warp per node: the warp holds the node's mask in registers as a WarpSlicedMask,
// fills it, stores it, and then seeds coarser levels across the node's faces (seedAcrossFace).

/// @brief Repeat the low @a period bits of @a unit across a 64-bit word,
///        e.g. repeatBits(0x01, 8) == 0x0101010101010101.
constexpr uint64_t repeatBits(uint64_t unit, int period)
{
    uint64_t r = 0;
    for (int i = 0; i < 64; i += period) r |= unit << i;
    return r;
}

/// @brief A node mask (Mask<Log2Dim>, a Dim^3 cube of bits) spread over one warp. Lane x holds x-slice x,
///        which is WordsPerSlice consecutive words of the mask (leaf: 1, lower: 4, upper: 16); bit
///        (x * Dim + y) * Dim + z of the mask is slot (x, y, z). Lanes x >= Dim hold no slots and stay empty.
///
///        |, &, ~, load and store work on each lane's own words. fromSlots, faceNeighbors, facesWithBits and
///        differsAnywhere exchange data across the warp (shuffles and votes), so all 32 lanes of the warp must
///        call them together.
template <int Log2Dim>
struct WarpSlicedMask
{
    static constexpr int      Dim           = 1 << Log2Dim;   // slots per axis: 8, 16 or 32
    static constexpr int      WordsPerSlice = Dim * Dim / 64; // words per lane: 1, 4 or 16
    static constexpr int      SlotCount     = Dim * Dim * Dim;
    static constexpr unsigned FullMask      = 0xFFFFFFFFu;    // all 32 lanes take part
    static_assert(Dim <= 32, "WarpSlicedMask assigns one lane per x-slice");

    uint64_t words[WordsPerSlice];

    __device__ static int  lane() { return threadIdx.x & 31; }
    __device__ static bool ownsSlice() { return lane() < Dim; }

    __device__ static WarpSlicedMask empty()
    {
        WarpSlicedMask m;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) m.words[j] = 0;
        return m;
    }

    __device__ static WarpSlicedMask load(const nanovdb::Mask<Log2Dim>& mask)
    {
        WarpSlicedMask m;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j)
            m.words[j] = ownsSlice() ? mask.words()[lane() * WordsPerSlice + j] : 0;
        return m;
    }

    __device__ void store(nanovdb::Mask<Log2Dim>& mask) const
    {
        if (!ownsSlice()) return;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) mask.words()[lane() * WordsPerSlice + j] = words[j];
    }

    /// @brief The mask whose slot n is on iff isOn(n). The warp asks about 32 slots at a time (slot 32k + lane),
    ///        and __ballot_sync packs the 32 answers into half of word k/2, which the lane holding that word keeps.
    template <typename PredicateT>
    __device__ static WarpSlicedMask fromSlots(PredicateT isOn)
    {
        WarpSlicedMask m = empty();
        for (int k = 0; k < SlotCount / 32; ++k) {
            const uint32_t ballot = __ballot_sync(FullMask, isOn(uint32_t(32 * k + lane())));
            const int      word   = k >> 1;
#pragma unroll
            for (int j = 0; j < WordsPerSlice; ++j)
                if (word == lane() * WordsPerSlice + j) m.words[j] |= uint64_t(ballot) << (32 * (k & 1));
        }
        return m;
    }

    __device__ WarpSlicedMask operator|(const WarpSlicedMask& other) const
    {
        WarpSlicedMask r;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) r.words[j] = words[j] | other.words[j];
        return r;
    }

    __device__ WarpSlicedMask operator&(const WarpSlicedMask& other) const
    {
        WarpSlicedMask r;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) r.words[j] = words[j] & other.words[j];
        return r;
    }

    /// @brief The complement within the cube; lanes without a slice stay empty.
    __device__ WarpSlicedMask operator~() const
    {
        WarpSlicedMask r;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) r.words[j] = ownsSlice() ? ~words[j] : 0;
        return r;
    }

    /// @brief Every slot that has an on face neighbor (-x, +x, -y, +y, -z, +z) in this mask.
    ///
    /// Within a word, the rows of Dim bits run along z and consecutive rows along y:
    ///   z +/- 1 : shift by 1 bit. The last slot of a row (z = Dim - 1) would land on the first slot of the
    ///             next row (z = 0), which is not its neighbor, so the +z result is masked with NotZ0 (clears
    ///             z = 0 of every row) and the -z result with NotZLast (clears z = Dim - 1 of every row).
    ///   y +/- 1 : shift by Dim bits (one row). The lowest and highest rows of a word take their neighbor
    ///             row from the previous and next word of the same slice.
    ///   x +/- 1 : the same word of the neighboring x-slice, held by lane - 1 or lane + 1.
    __device__ WarpSlicedMask faceNeighbors() const
    {
        WarpSlicedMask r;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) {
            const uint64_t fromLowerX = __shfl_up_sync(FullMask, words[j], 1);
            const uint64_t fromUpperX = __shfl_down_sync(FullMask, words[j], 1);
            uint64_t n = ((words[j] << 1) & NotZ0) | ((words[j] >> 1) & NotZLast)
                       | (words[j] << Dim) | (words[j] >> Dim);
            if (j > 0)                 n |= words[j - 1] >> (64 - Dim);
            if (j < WordsPerSlice - 1) n |= words[j + 1] << (64 - Dim);
            if (lane() >= 1)           n |= fromLowerX;
            if (lane() < Dim - 1)      n |= fromUpperX;
            r.words[j] = ownsSlice() ? n : 0;
        }
        return r;
    }

    /// @brief Bit f is set if face f (0-5: -x, +x, -y, +y, -z, +z) of the cube holds an on slot:
    ///   -x / +x : x-slice 0 / Dim - 1, held by lane 0 / lane Dim - 1
    ///   -y / +y : the lowest row of the first word / the highest row of the last word, in every lane
    ///   -z / +z : the first / last bit of every row, in every word of every lane
    __device__ uint32_t facesWithBits() const
    {
        bool anyOn = false, zFirst = false, zLast = false;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) {
            anyOn  |= words[j] != 0;
            zFirst |= (words[j] & ZFirst) != 0;
            zLast  |= (words[j] & ZLast) != 0;
        }
        return uint32_t(__any_sync(FullMask, lane() == 0 && anyOn)) << 0 |
               uint32_t(__any_sync(FullMask, lane() == Dim - 1 && anyOn)) << 1 |
               uint32_t(__any_sync(FullMask, (words[0] & FirstRow) != 0)) << 2 |
               uint32_t(__any_sync(FullMask, (words[WordsPerSlice - 1] >> (64 - Dim)) != 0)) << 3 |
               uint32_t(__any_sync(FullMask, zFirst)) << 4 |
               uint32_t(__any_sync(FullMask, zLast)) << 5;
    }

    /// @brief Whether any lane's words differ from @a other's.
    __device__ bool differsAnywhere(const WarpSlicedMask& other) const
    {
        bool differs = false;
#pragma unroll
        for (int j = 0; j < WordsPerSlice; ++j) differs |= words[j] != other.words[j];
        return __any_sync(FullMask, differs);
    }

private:
    static constexpr uint64_t NotZ0    = repeatBits(((1ull << Dim) - 1) & ~1ull, Dim); // all but z = 0 of every row
    static constexpr uint64_t NotZLast = repeatBits((1ull << (Dim - 1)) - 1, Dim);     // all but z = Dim - 1 of every row
    static constexpr uint64_t ZFirst   = repeatBits(1ull, Dim);                        // z = 0 of every row
    static constexpr uint64_t ZLast    = repeatBits(1ull << (Dim - 1), Dim);           // z = Dim - 1 of every row
    static constexpr uint64_t FirstRow = (1ull << Dim) - 1;                            // lowest row of a word
}; // WarpSlicedMask

// Seeding coarser levels across node faces.
//
// Each face of a filled node that has an inside slot seeds the region across that face, if that region is
// a childless slot of a coarser level: a lower or upper slot, or a root-tile position. Pushing seeds from
// the filled nodes (scatter), rather than having every childless slot look at its neighbors (gather),
// keeps the work proportional to the filled nodes: most childless slots touch no node, and one childless
// slot can border many finer nodes. One probe per face suffices: nodes are aligned to their size, so the
// block of the same size across a face lies inside a single slot of every coarser level.

/// @brief A face of a node. WarpSlicedMask::facesWithBits sets bit f for face f.
enum Face : int { minusX, plusX, minusY, plusY, minusZ, plusZ, faceCount };

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

/// @brief Finds the childless element that contains @a ijk, i.e. where the tree stops descending.
/// @return 0 for a root value tile or leaf, 1 for lower, 2 for upper, and 3 for absent root space.
///         For 1 and 2, @a nodeID and @a slot say which node and which of its slots.
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

/// @brief Seeds the region across @a face of the Level node at @a origin, if it is a childless slot of a
///        coarser level: its bit in @a lowerMasks or @a upperMasks, set atomically since neighboring nodes
///        may seed the same slot, or its entry of @a rootMask if it is absent root space inside the dense
///        root-tile array.
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

/// @brief Fills one leaf (Level 0), lower (1) or upper (2) node's interior mask in @a masks, one warp per
///        node, then seeds the childless slots of coarser levels across the node's inside faces.
template <typename BuildT, int Level>
struct BitFloodFunctor
{
    using NodeT = typename NanoNode<BuildT, Level>::type;
    using MaskT = nanovdb::Mask<NodeT::LOG2DIM>;
    using SlicedMaskT = WarpSlicedMask<NodeT::LOG2DIM>;

    static constexpr int NodesPerBlock              = 4;
    static constexpr int MaxThreadsPerBlock         = 32 * NodesPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    // TODO: Rename signs to signSidecar?
    __device__ void operator()(const NanoGrid<BuildT>* grid, const int8_t* signs, MaskT* masks, uint32_t nodeCount,
                               nanovdb::Mask<4>* lowerMasks, nanovdb::Mask<5>* upperMasks, uint8_t* rootMask,
                               nanovdb::Coord tileMin, nanovdb::Coord tileDims)
    {
        // One warp per node; the block's last warps may have no node, and leave together.
        const int      lane   = threadIdx.x & 31;
        const uint32_t nodeID = blockIdx.x * NodesPerBlock + (threadIdx.x >> 5);
        if (nodeID >= nodeCount) return;
        const auto& node  = grid->tree().template getFirstNode<Level>()[nodeID];

        // Interior starts as the seeds; open slots are those the interior may spread into.
        SlicedMaskT interior, open;
        if constexpr (Level == 0) {
            // Leaf: seeds are active voxels whose sign is -1; inactive voxels are open.
            interior = SlicedMaskT::fromSlots([&](uint32_t n) {
                return node.isActive(n) && signs[node.getValue(n)] == int8_t(-1);
            });
            open = ~SlicedMaskT::load(node.valueMask());
        } else {
            // Lower/upper: seeds were written into masks by the finer level; childless slots are open.
            open     = ~SlicedMaskT::load(node.childMask());
            interior = SlicedMaskT::load(masks[nodeID]) & open;
        }

        // Each sweep adds the open face neighbors of the previous interior. A sweep that changes nothing
        // ends the fill; any other sweep adds at least one slot, so SlotCount + 1 sweeps suffice.
        int sweep = 0;
        for (; sweep <= SlicedMaskT::SlotCount; ++sweep) {
            const SlicedMaskT next = interior | (open & interior.faceNeighbors());
            if (!next.differsAnywhere(interior)) break;
            interior = next;
        }
        NANOVDB_ASSERT(sweep <= SlicedMaskT::SlotCount); // exhausted the backstop: the fill did not converge
        interior.store(masks[nodeID]);

        const uint32_t faceHasInside = interior.facesWithBits();

        // Lane f seeds across face f if that face has an inside slot.
        if (lane < faceCount && ((faceHasInside >> lane) & 1u))
            seedAcrossFace<BuildT, Level>(*grid, node.origin(), Face(lane),
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
        <<<(count + Op::NodesPerBlock - 1) / Op::NodesPerBlock, Op::MaxThreadsPerBlock, 0, mStream>>>(
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
        <<<(count + Op::NodesPerBlock - 1) / Op::NodesPerBlock, Op::MaxThreadsPerBlock, 0, mStream>>>(
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
        <<<(count + Op::NodesPerBlock - 1) / Op::NodesPerBlock, Op::MaxThreadsPerBlock, 0, mStream>>>(
        mDeviceGrid, noSigns, mUpperInteriorMasks.data(), count,
        mLowerInteriorMasks.data(), mUpperInteriorMasks.data(), mRootMask.data(), mRootTileMin, mRootTileDims);
    cudaCheckError();
}// PropagateSign<BuildT, ResourceT>::fillUpperAndSeedCoarserLevels

//----------------------------------------------------------------------------------------------------------------------

namespace signing::detail {

// Stage 4: fill the root-tile positions.
//
// Positions with an upper child are walls; the seeds from upper-node faces spread through the other
// positions of the dense root-tile array (see PropagateSign::mRootMask). A single block floods the whole
// array, so this stage is meant for active bounds that span few root tiles.

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

/// @brief Fills the dense root-tile array in place, in a single block, from the seeds that
///        BitFloodFunctor<BuildT, 2> set in it, around the walls from RootWallFunctor.
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

//----------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_PROPAGATESIGN_CUH_HAS_BEEN_INCLUDED
