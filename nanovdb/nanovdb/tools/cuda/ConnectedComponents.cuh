// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/ConnectedComponents.cuh

    \authors Efty Sifakis and JaeHyun Lee

    \brief Connected-components labeling of NanoVDB ValueOnIndex grids on the device.

           Identifies connected components of active voxels in a ValueOnIndex grid: two
           active voxels share a component label iff they are connected through a path of
           adjacent active voxels. As with the other topology operators, grids with active
           tiles are rejected; the grid must also index exactly its active voxels (no stats),
           as NanoVDB's CUDA builders produce.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel

#include <cub/cub.cuh>
#include <cuda/atomic> // cuda::atomic_ref

#include <utility>      // std::pair
#include <limits>       // std::numeric_limits
#include <stdexcept>    // std::runtime_error
#include <type_traits>  // std::is_unsigned

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

/// @brief Vocabulary shared by ConnectedComponents and the device code implementing it.
struct ConnectedComponentsBase
{
    /// @brief Identifies a connected component. Must accommodate the total number of per-leaf
    ///        components created across the grid; enforced by MaxComponentCount.
    using ComponentLabelT = uint32_t;
    static_assert(std::is_unsigned<ComponentLabelT>::value,
                  "ComponentLabelT must be unsigned: the background sentinel is its all-ones value");

    /// @brief Index of each of the 6 leaf faces into a component's face-mask array.
    enum LeafNeighborTap : int {
        minusX = 0,
        plusX  = 1,
        minusY = 2,
        plusY  = 3,
        minusZ = 4,
        plusZ  = 5
    };

    /// @brief Maximum number of components, across the whole grid, this operator can label.
    static constexpr uint64_t MaxComponentCount =
        uint64_t(std::numeric_limits<ComponentLabelT>::max());
};// tools::cuda::ConnectedComponentsBase

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class ConnectedComponents : public ConnectedComponentsBase
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex,
                  "ConnectedComponents labels the active voxels of a ValueOnIndex grid");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "ConnectedComponents allocates stream-ordered scratch and requires an AsyncResource");
    static_assert(ResourceT::DEFAULT_ALIGNMENT >= alignof(uint64_t),
                  "ConnectedComponents stores scratch up to uint64_t width and requires word-aligned allocations");

    using GridT = NanoGrid<BuildT>;
    using TreeT = NanoTree<BuildT>;
    using RootT = NanoRoot<BuildT>;

    /// @brief Stream-ordered, device-only scratch storage borrowed from the injected resource.
    template <typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:

    /// @brief Constructor
    /// @param d_srcGrid source device ValueOnIndex grid whose active voxels are to be labeled; it must
    ///        have no active tiles, and must index exactly its active voxels (no stats), which is
    ///        checked in debug builds
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param resource resource instance all device scratch is allocated from;
    ///        must outlive this operator (defaults to the per-type default resource)
    ConnectedComponents(const GridT* d_srcGrid, cudaStream_t stream = 0,
                        ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid)
        , mResource(&resource)
        , mTempDevicePool(resource)
        , mLeafComponentCounts(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafComponentOffsets(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafComponentMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mLeafComponentFaceMasks(stream, resource, 0, nanovdb::cuda::noInit)
        , mComponentParent(stream, resource, 0, nanovdb::cuda::noInit)
        , mVoxelLabel(stream, resource, 0, nanovdb::cuda::noInit) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Run the connected-components pipeline and return { labels, componentCount }:
    ///          - labels: an owning label sidecar indexed like the grid's values (one entry per value
    ///            index, i.e. the grid's value count). Each active voxel holds its component's dense
    ///            id in [0, N), so two active voxels share a value iff they are in the same connected
    ///            component; every other entry, including slot 0 (the background), holds the
    ///            all-ones sentinel. Ownership passes to the caller: the buffer outlives this operator and
    ///            frees itself on the stream it was allocated on. Only that free needs the
    ///            resource, which by default has program lifetime.
    ///          - componentCount: the number of connected components N.
    /// @note The source grid is read in order on the operator's stream, so it may still be in flight
    ///       there (e.g. after an asynchronous upload on that stream). The returned contents are
    ///       complete: the stream is synchronized before returning.
    /// @throw std::runtime_error if the grid has active tiles.
    std::pair<BufT<ComponentLabelT>, ComponentLabelT> getVoxelLabelsAndCount()
    {
        // The source's tree header is read once, in stream order; every stage sizes its work from it.
        mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid, mStream);

        // Only leaf voxels are labeled, so reject active tiles, as the other topology operators do.
        if (mSrcTreeData.mTileCount[0] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[2])
            throw std::runtime_error("Topological operations not supported on grids with value tiles");

        // Each scratch buffer is released right after its last use, to cut peak memory. The
        // frees are ordered on mStream behind the kernels that read them, so no host sync.
        processLeafConnectedComponents();
        mLeafComponentCounts.destroy();
        processGlobalConnectedComponents();
        mLeafComponentFaceMasks.destroy();
        processVoxelLabels();
        mLeafComponentOffsets.destroy();
        mLeafComponentMasks.destroy();
        mComponentParent.destroy();
        cudaCheck(cudaStreamSynchronize(mStream));
        return { std::move(mVoxelLabel), mGlobalComponentCount };
    }

private:

    // --- Pipeline stages (run in order by getVoxelLabelsAndCount). ---

    // Stage 1: per-leaf 6-connected components (each leaf in isolation) -> per-component count,
    // prefix-sum offset, active-voxel Mask<3>, and 6 face masks.
    void processLeafConnectedComponents();

    // Stage 2: global union-find, uniting every pair of face-adjacent leaf-local components whose
    // touching face masks intersect -> deviceComponentParent()[s] = component s's representative
    // (its class's minimum global slot).
    void processGlobalConnectedComponents();

    // Build the per-voxel label sidecar + component count N from the parent array.
    void processVoxelLabels();

    /// @brief Borrowed handle to the injected resource, for the scratch allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    // --- Internal device-array accessors (each valid after the stage that fills it). ---
    auto deviceLeafComponentCounts()    { return mLeafComponentCounts.data(); }
    auto deviceLeafComponentOffsets()   { return mLeafComponentOffsets.data(); }
    auto deviceLeafComponentMasks()     { return mLeafComponentMasks.data(); }
    auto deviceLeafComponentFaceMasks() { return mLeafComponentFaceMasks.data(); }
    auto deviceComponentParent()        { return mComponentParent.data(); }

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    const GridT                 *mDeviceSrcGrid;
    typename TreeT::DataType     mSrcTreeData{};              // host copy of the source's tree header, read by getVoxelLabelsAndCount()
    ResourceT                   *mResource;                   // non-owning; all device scratch routes through this instance
    nanovdb::cuda::TempPool<ResourceT> mTempDevicePool;

    uint64_t                     mLeafComponentAggregateCount{0}; // total leaf-local components across all leaves (= K = offsets[leafCount])

    // Scratch: getVoxelLabelsAndCount() releases each of these after its last use.
    BufT<uint16_t>               mLeafComponentCounts;      // leafCount                    x uint16_t:      per-leaf component count
    BufT<uint64_t>               mLeafComponentOffsets;     // (leafCount+1)                x uint64_t:      exclusive+inclusive prefix sums
    BufT<nanovdb::Mask<3>>       mLeafComponentMasks;       // mLeafComponentAggregateCount x Mask<3>:       per-component active-voxel footprint
    BufT<uint64_t[6]>            mLeafComponentFaceMasks;   // mLeafComponentAggregateCount x uint64_t[6]:   per-component face bitmasks (0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z)

    BufT<uint64_t>               mComponentParent;          // K x uint64_t: per-component global representative

    BufT<ComponentLabelT>        mVoxelLabel;               // valueCount x ComponentLabelT: dense component id in [0,N) per active voxel (index by leaf.getValue(n)); all other entries, incl. slot 0, = sentinel
    ComponentLabelT              mGlobalComponentCount{0};  // number of connected components N (distinct representatives)

}; // tools::cuda::ConnectedComponents<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace components::detail {

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stage 1: leaf-local connected components.

// Leaf-local connected components in shared memory, one warp per leaf, over the z-runs of the leaf
// instead of its voxels. A leaf's 8^3 voxels are 64 rows (x, y) of 8 bits (z): row r = 8x + y is
// byte r of the value mask. A z-run is a maximal stretch of active voxels in one row. Connectivity
// is 6-connected and strictly intra-leaf: two runs are adjacent iff their rows are face neighbors
// (x +- 1 or y +- 1) and their bits overlap, since voxels adjacent along z already lie in one run.

constexpr unsigned WarpFullMask = 0xFFFFFFFFu;

/// @brief The z-runs of one leaf, built by one warp. Runs are numbered in ascending order of their
///        first voxel offset (row by row, then by z), so the smallest run number in a component
///        holds the component's smallest voxel offset.
struct LeafRuns
{
    static constexpr int MaxRuns = 256;  // 64 rows, at most 4 runs in an 8-bit row
    uint16_t rowBase[65];                // first run number of each row; rowBase[64] = run count
    uint8_t  runRow[MaxRuns];
    uint8_t  runBits[MaxRuns];           // the run's voxels, as bits z of its row
    int      parentA[MaxRuns], parentB[MaxRuns];

    __device__ int count() const { return rowBase[64]; }

    // Lane l handles rows l and l + 32.
    __device__ void build(const uint64_t* leafWords, int lane)
    {
        uint32_t bytes[2], starts[2], counts[2], before[2];
        for (int h = 0; h < 2; ++h) {
            const int r = lane + 32 * h;
            bytes[h]  = uint32_t(leafWords[r >> 3] >> (8 * (r & 7))) & 0xFF;
            starts[h] = bytes[h] & ~(bytes[h] << 1) & 0xFF;   // first voxel of each run
            counts[h] = __popc(starts[h]);
        }
        // Exclusive prefix over rows 0..63: lanes hold rows 0..31 first, then 32..63.
        for (int h = 0; h < 2; ++h) {
            uint32_t inclusive = counts[h];
            for (int d = 1; d < 32; d *= 2) {
                const uint32_t v = __shfl_up_sync(WarpFullMask, inclusive, d);
                if (lane >= d) inclusive += v;
            }
            before[h] = inclusive - counts[h];
            if (h == 1) before[1] += __shfl_sync(WarpFullMask, before[0] + counts[0], 31);
        }
        // Run numbers and counts fit uint16_t: a leaf has at most MaxRuns runs.
        rowBase[lane]      = uint16_t(before[0]);
        rowBase[lane + 32] = uint16_t(before[1]);
        if (lane == 31) rowBase[64] = uint16_t(before[1] + counts[1]);
        for (int h = 0; h < 2; ++h) {
            int k = int(before[h]);
            for (uint32_t s = starts[h]; s; s &= s - 1) {
                const int z0     = __ffs(s) - 1;
                const int length = __ffs(~(bytes[h] >> z0)) - 1;
                runRow[k]  = uint8_t(lane + 32 * h);                  // a row index, < 64
                runBits[k] = uint8_t(((1u << length) - 1) << z0);     // bits of an 8-bit row
                ++k;
            }
        }
        __syncwarp();
    }

    // Minimum parent label over run i and the overlapping runs of its (up to 4) neighboring rows.
    __device__ int neighborMin(const int* parent, int i) const
    {
        const int r = runRow[i], x = r >> 3, y = r & 7;
        int m = parent[i];
        if (x > 0) m = rowMin(parent, r - 8, runBits[i], m);   // -X
        if (x < 7) m = rowMin(parent, r + 8, runBits[i], m);   // +X
        if (y > 0) m = rowMin(parent, r - 1, runBits[i], m);   // -Y
        if (y < 7) m = rowMin(parent, r + 1, runBits[i], m);   // +Y
        return m;
    }

    // Smallest of m and the parents of the runs of the given row that overlap bits.
    __device__ int rowMin(const int* parent, int row, uint32_t bits, int m) const
    {
        for (int k = rowBase[row]; k < rowBase[row + 1]; ++k)
            if (runBits[k] & bits) m = ::min(m, parent[k]);
        return m;
    }

    // Index of the component whose root is run `root`, among the components of the leaf in root
    // order: the number of roots before it, from the bit set isRoot of MaxRuns bits.
    __device__ static int componentIndex(const uint32_t* isRoot, int root)
    {
        int index = __popc(isRoot[root >> 5] & ((1u << (root & 31)) - 1));
        for (int w = 0; w < (root >> 5); ++w) index += __popc(isRoot[w]);
        return index;
    }
}; // LeafRuns

// Union-find over the runs of one leaf, over a parent array of run numbers. The primitives are
// algorithm P of S. C. Liu and R. E. Tarjan, "Simple Concurrent Connected Components Algorithms",
// ACM Transactions on Parallel Computing 9(2), 2022, becoming their algorithm R once solve()
// restricts hooking to roots.
//
// They are double-buffered (Jacobi): read "cur", write "nxt", then swap -- identically in every
// lane, so the per-lane pointer copies stay in sync. Every lane of the warp must call each of
// them, as each contains __syncwarp(). They return whether any lane of the warp changed an entry.
struct LeafUnionFind
{
    // Union-find hook: if the smallest parent m among run i's neighbors is below i's own parent
    // p, lower the parent of p toward m via atomicMin (many runs can target the same slot p).
    //
    // @param rootsOnly picks which of Liu & Tarjan's connect operations this performs:
    //        parent-connect when false, parent-root-connect when true, which hooks only through
    //        parents that are themselves roots. That restriction cannot move a subtree between
    //        trees, making the algorithm monotone -- the property their O(lg n) bound rests on.
    __device__ static bool hook(const LeafRuns& L, int*& cur, int*& nxt, int n, int lane, bool rootsOnly)
    {
        for (int i = lane; i < n; i += 32) nxt[i] = cur[i];   // seed nxt = cur (own slots, no race)
        __syncwarp();
        bool changed = false;
        for (int i = lane; i < n; i += 32) {
            const int pi = cur[i];
            if (rootsOnly && cur[pi] != pi) continue;         // a root is required
            const int m = L.neighborMin(cur, i);
            if (m < pi && atomicMin_block(&nxt[pi], m) > m) changed = true;  // block scope: nxt[] is shared
        }
        __syncwarp();
        int* t = cur; cur = nxt; nxt = t;                     // swap (identical on every lane)
        return __any_sync(WarpFullMask, changed);
    }

    // Union-find compress: parent[i] <- parent[parent[i]], one round of pointer jumping applied to
    // every run at once, halving tree depth per call. This is shortcut in Liu & Tarjan.
    __device__ static bool compress(int*& cur, int*& nxt, int n, int lane)
    {
        bool changed = false;
        for (int i = lane; i < n; i += 32) {
            const int v = cur[cur[i]];                        // i's grandparent
            changed |= v != cur[i];
            nxt[i] = v;                                       // own slot, no race
        }
        __syncwarp();
        int* t = cur; cur = nxt; nxt = t;                     // swap
        return __any_sync(WarpFullMask, changed);
    }

    // Round at which the main loop switches from algorithm P (parent-connect) to algorithm R
    // (parent-root-connect). P is cheaper per round but has no proven step bound; R is bounded by
    // O(lg n), which matches the lower bound. Leaves are observed to converge in far fewer rounds
    // than this, so the switch is a fallback that normally never runs, and the two cost the same
    // when it does not. Raising it favours P, lowering it reaches the bound sooner.
    static constexpr int SwitchToRootAfter = 8;

    // Run the full schedule to convergence. On return cur[i] holds run i's component root, and each
    // component's root is the minimum run number it contains.
    __device__ static void solve(const LeafRuns& L, int*& cur, int*& nxt, int lane)
    {
        // Init: every run starts as its own root.
        const int n = L.count();
        for (int i = lane; i < n; i += 32) cur[i] = i;
        __syncwarp();

        // Unconditional warm-up: one hook, then enough compresses to flatten the forest rather
        // than merely halve its depth (a leaf is DIM=8 across). Flatness keeps the
        // parent-root-connect phase below cheap: with nearly every run its own root, that
        // phase's extra test is almost always satisfied and costs nothing.
        hook    (L, cur, nxt, n, lane, false);
        compress(cur, nxt, n, lane);
        compress(cur, nxt, n, lane);
        compress(cur, nxt, n, lane);
        compress(cur, nxt, n, lane);

        // Then alternate (hook, compress) until a full iteration changes nothing, as algorithm P
        // and then as algorithm R once SwitchToRootAfter rounds have passed. The bound is a
        // backstop, not the mechanism that ends the loop: algorithm R converges within its O(lg n)
        // step bound, so only a bug could exhaust it.
        constexpr int MaxConvergenceIters = 64;
        for (int it = 0; it < MaxConvergenceIters; ++it) {
            bool changed = hook(L, cur, nxt, n, lane, it >= SwitchToRootAfter);
            changed |= compress(cur, nxt, n, lane);
            if (!changed) return;  // converged: no lane changed anything
        }
        NANOVDB_ASSERT(false);  // exhausted the backstop: algorithm R's bound was not met
    }
}; // LeafUnionFind

/// @brief Counts the 6-connected components of one leaf's active voxels, taken in isolation from
///        neighboring leaves. One warp per leaf, four leaves per block. The counts size the
///        per-leaf component slices that LeafComponentMaskFunctor then fills.
template <typename BuildT>
struct LeafComponentCountFunctor
{
    static constexpr int LeavesPerBlock             = 4;
    static constexpr int MaxThreadsPerBlock         = 32 * LeavesPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, uint16_t* d_counts, uint32_t leafCount)
    {
        __shared__ LeafRuns sRuns[LeavesPerBlock];

        const int      lane   = threadIdx.x & 31;
        const uint32_t leafID = blockIdx.x * LeavesPerBlock + (threadIdx.x >> 5);
        if (leafID >= leafCount) return;
        LeafRuns& L = sRuns[threadIdx.x >> 5];
        L.build(d_grid->tree().template getFirstNode<0>()[leafID].valueMask().words(), lane);

        int* cur = L.parentA;
        int* nxt = L.parentB;
        LeafUnionFind::solve(L, cur, nxt, lane);

        // Component count = number of surviving roots (cur[i] == i).
        int roots = 0;
        for (int i = lane; i < L.count(); i += 32) roots += cur[i] == i;
        roots = __reduce_add_sync(WarpFullMask, roots);
        if (lane == 0) d_counts[leafID] = uint16_t(roots);  // at most MaxRuns components per leaf
    }
}; // LeafComponentCountFunctor

/// @brief Re-solves one leaf's components and fills its slice [d_offsets[leafID], d_offsets[leafID+1]),
///        in ascending root order: each component's voxel Mask<3>, and its 6 face masks (the voxels
///        it has on each leaf face, which the cross-leaf stage intersects with the neighbor's).
template <typename BuildT>
struct LeafComponentMaskFunctor
{
    using LeafNeighborTap = ConnectedComponentsBase::LeafNeighborTap;

    static constexpr int LeavesPerBlock             = 4;
    static constexpr int MaxThreadsPerBlock         = 32 * LeavesPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                               const uint64_t*         d_offsets,
                               nanovdb::Mask<3>*       d_masks,
                               uint64_t                (*d_faces)[6],
                               uint32_t                leafCount)
    {
        __shared__ LeafRuns sRuns[LeavesPerBlock];
        __shared__ uint32_t sIsRoot[LeavesPerBlock][LeafRuns::MaxRuns / 32];

        const int      lane   = threadIdx.x & 31;
        const int      warp   = threadIdx.x >> 5;
        const uint32_t leafID = blockIdx.x * LeavesPerBlock + warp;
        if (leafID >= leafCount) return;
        LeafRuns& L = sRuns[warp];
        L.build(d_grid->tree().template getFirstNode<0>()[leafID].valueMask().words(), lane);

        int* cur = L.parentA;
        int* nxt = L.parentB;
        LeafUnionFind::solve(L, cur, nxt, lane);
        const int n = L.count();

        // Roots ascend with the smallest voxel offset, so a component's index in the leaf's slice
        // is the number of roots before its root.
        uint32_t* isRoot = sIsRoot[warp];
        if (lane < LeafRuns::MaxRuns / 32) isRoot[lane] = 0;
        __syncwarp();
        for (int i = lane; i < n; i += 32)
            if (cur[i] == i) atomicOr_block(&isRoot[i >> 5], 1u << (i & 31));
        __syncwarp();
        int components = 0;
        for (int w = 0; w < LeafRuns::MaxRuns / 32; ++w) components += __popc(isRoot[w]);

        // Mask-fill: clear the masks of the leaf's slice, then each run ORs its voxels into its
        // component's mask.
        nanovdb::Mask<3>* masks = d_masks + d_offsets[leafID];
        for (int t = lane; t < 8 * components; t += 32) masks[t >> 3].words()[t & 7] = 0;
        __syncwarp();
        for (int i = lane; i < n; i += 32) {
            const int r = L.runRow[i];
            atomicOr(reinterpret_cast<unsigned long long*>(&masks[LeafRuns::componentIndex(isRoot, cur[i])].words()[r >> 3]),
                     (unsigned long long)L.runBits[i] << (8 * (r & 7)));
        }
        __threadfence_block();
        __syncwarp();

        // Lane k extracts the 6 face masks of component k from its mask words and stores them.
        for (int k = lane; k < components; k += 32) {
            const uint64_t* words = masks[k].words();
            uint64_t*       face  = d_faces[d_offsets[leafID] + k];

            // +/-X: whole word 0 / word 7 are exactly the minusX / plusX face planes.
            face[LeafNeighborTap::minusX] = words[0];
            face[LeafNeighborTap::plusX]  = words[7];

            // +/-Y: bottom byte (y=0) and top byte (y=7) of each word x;
            //     shift-accumulate from x=7 down; result bit index = x*8 + z (x major, z minor).
            uint64_t mY = words[7] & 0xFF,  pY = (words[7] >> 56) & 0xFF;
            for (int x = 6; x >= 0; --x) {
                mY = (mY << 8) | (words[x] & 0xFF);
                pY = (pY << 8) | ((words[x] >> 56) & 0xFF);
            }
            face[LeafNeighborTap::minusY] = mY;
            face[LeafNeighborTap::plusY]  = pY;

            // +/-Z: bit 0 (z=0) and bit 7 (z=7) of each byte in each word;
            //     shift-accumulate from x=7 down; result bit index = y*8 + x (y major, x minor).
            uint64_t mZ = words[7] & UINT64_C(0x0101010101010101);
            uint64_t pZ = (words[7] >> 7) & UINT64_C(0x0101010101010101);
            for (int x = 6; x >= 0; --x) {
                mZ = (mZ << 1) | (words[x] & UINT64_C(0x0101010101010101));
                pZ = (pZ << 1) | ((words[x] >> 7) & UINT64_C(0x0101010101010101));
            }
            face[LeafNeighborTap::minusZ] = mZ;
            face[LeafNeighborTap::plusZ]  = pZ;
        }
    }
}; // LeafComponentMaskFunctor

// offsets[i+1] = counts[i]: upcast the per-leaf uint16 component counts into the uint64 offset array.
struct UpcastCountsFunctor {
    __device__ void operator()(size_t i, const uint16_t* counts, uint64_t* offsets) const {
        offsets[i + 1] = uint64_t(counts[i]);
    }
};

} // namespace components::detail

template <typename BuildT, typename ResourceT>
void ConnectedComponents<BuildT, ResourceT>::processLeafConnectedComponents()
{
    const uint32_t leafCount = mSrcTreeData.mNodeCount[0];

    // Allocate one component-count per leaf (device-only). At most 256 components per 8^3
    // leaf (6-connected worst case), so uint16_t is sufficient.
    if (mVerbose==1) mTimer.start("Allocating per-leaf component counts");
    mLeafComponentCounts = BufT<uint16_t>(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
    if (mVerbose==1) mTimer.stop();

    if (leafCount == 0) return;

    // One warp per leaf; counts the distinct 6-connected components of each leaf's active
    // voxels (in isolation) into mLeafComponentCounts.
    using Op = components::detail::LeafComponentCountFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf connected-component counting");
    util::cuda::operatorKernel<Op>
        <<<(leafCount + Op::LeavesPerBlock - 1) / Op::LeavesPerBlock, Op::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentCounts(), leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Prefix sum: mLeafComponentOffsets[0]=0, mLeafComponentOffsets[1..leafCount] = inclusive sum
    // of mLeafComponentCounts. mLeafComponentOffsets[leafCount] = K (total leaf-local components).
    if (mVerbose==1) mTimer.start("Allocating per-leaf component offsets");
    mLeafComponentOffsets = BufT<uint64_t>(mStream, this->ref(), std::size_t(leafCount) + 1, nanovdb::cuda::noInit);
    if (mVerbose==1) mTimer.stop();

    cudaCheck(cudaMemsetAsync(mLeafComponentOffsets.data(), 0, sizeof(uint64_t), mStream));

    // Upcast per-leaf uint16_t counts into offsets[1..leafCount] as uint64_t.
    uint16_t* d_counts  = deviceLeafComponentCounts();
    uint64_t* d_offsets = deviceLeafComponentOffsets();
    static constexpr unsigned int threadsPerBlock = 256;
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(leafCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        leafCount, components::detail::UpcastCountsFunctor{}, d_counts, d_offsets);
    cudaCheckError();

    // In-place inclusive sum over offsets[1..leafCount]; offsets[leafCount] = K (total components).
    if (mVerbose==1) mTimer.start("Per-leaf component offset prefix sum");
    CALL_CUBS(DeviceScan::InclusiveSum, d_offsets + 1, d_offsets + 1, leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Read the aggregate count from the sentinel at offsets[leafCount].
    // Use async copy on mStream (ordered after the scan) + stream sync to avoid
    // inadvertently synchronizing other streams.
    cudaCheck(cudaMemcpyAsync(&mLeafComponentAggregateCount, d_offsets + leafCount,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    // Everything downstream indexes the leaf-local components: the cross-leaf stage unites global
    // slots, the rank scan runs over K items, and the dense labels count them. K is carried as
    // uint64_t precisely so an out-of-range value survives to be tested here; past this point the
    // narrower types are known to fit, which is what lets N share ComponentLabelT with the labels.
    if (mLeafComponentAggregateCount > MaxComponentCount)
        throw std::runtime_error("nanovdb::tools::cuda::ConnectedComponents: leaf-local component "
                                 "count exceeds the indexable range");

    // Allocate mLeafComponentAggregateCount Mask<3> objects - one per leaf-local component.
    // No zero-init needed: the mask-fill kernel clears every mask before it sets its bits.
    if (mVerbose==1) mTimer.start("Allocating per-component leaf masks");
    mLeafComponentMasks = BufT<nanovdb::Mask<3>>(
        mStream, this->ref(), mLeafComponentAggregateCount, nanovdb::cuda::noInit);
    if (mVerbose==1) mTimer.stop();

    // Allocate 6 uint64_t face masks per component (one per LeafNeighborTap entry).
    // The mask-fill kernel writes all six of every component; no zero-init needed.
    if (mVerbose==1) mTimer.start("Allocating per-component face masks");
    mLeafComponentFaceMasks = BufT<uint64_t[6]>(
        mStream, this->ref(), mLeafComponentAggregateCount, nanovdb::cuda::noInit);
    if (mVerbose==1) mTimer.stop();

    // Re-run the leaf union-find and scatter each run's bits into its component's Mask<3>.
    using MaskOp = components::detail::LeafComponentMaskFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf component mask fill");
    util::cuda::operatorKernel<MaskOp>
        <<<(leafCount + MaskOp::LeavesPerBlock - 1) / MaskOp::LeavesPerBlock, MaskOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(),
            deviceLeafComponentMasks(), deviceLeafComponentFaceMasks(), leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT, ResourceT>::processLeafConnectedComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace components::detail {

// Stage 2: cross-leaf connectivity.
//
// One block per leaf. For each of the leaf's +X/+Y/+Z neighbor leaves (so each undirected leaf-leaf
// boundary is visited exactly once, from its -side), pair every local component of this leaf with
// every local component of the neighbor and test whether their touching face masks intersect. The
// component "global slot" is leafComponentOffsets[leaf] + localIdx.

/// @brief Linear leaf index of leaf's +axis neighbor (axis: 0=+X, 1=+Y, 2=+Z), or -1 if none.
template <typename BuildT>
__device__ inline int neighborLeafIndex(const NanoGrid<BuildT>* d_grid,
                                        const NanoLeaf<BuildT>&  leaf, int axis)
{
    const nanovdb::Coord leafOrigin     = leaf.origin();
    const nanovdb::Coord neighborOrigin = (axis == 0) ? leafOrigin.offsetBy(8, 0, 0)
                                        : (axis == 1) ? leafOrigin.offsetBy(0, 8, 0)
                                                      : leafOrigin.offsetBy(0, 0, 8);
    const auto* neighborLeaf = d_grid->tree().root().probeLeaf(neighborOrigin);
    if (!neighborLeaf) return -1;
    const auto* firstLeaf = d_grid->tree().template getFirstNode<0>();
    return int(neighborLeaf - firstLeaf);  // leaves are contiguous breadth-first; typed diff = linear index
}

/// @brief Enumerate this leaf's cross-leaf component pairs whose touching faces intersect, invoking
///        emit(globalSlotThisLeaf, globalSlotNeighbor) for each. Threads of the block stride the
///        pair grid; emit() must be safe under concurrent calls.
template <typename BuildT, typename EdgeFn>
__device__ inline void forEachCrossLeafEdge(
    const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets, const uint64_t (*d_faces)[6],
    int leafID, int tID, int nThreads, EdgeFn&& emit)
{
    using LeafNeighborTap = ConnectedComponentsBase::LeafNeighborTap;
    using ComponentLabelT = ConnectedComponentsBase::ComponentLabelT;

    // base* is a leaf's first global component slot, count* how many components it has.
    const auto&    leaf      = d_grid->tree().template getFirstNode<0>()[leafID];
    const uint64_t baseLeaf  = d_offsets[leafID];
    const int      countLeaf = int(d_offsets[leafID + 1] - baseLeaf);
    if (countLeaf == 0) return;

    // Only the +axis faces are walked; the neighbor on its -axis side covers the other direction,
    // so every adjacent pair is visited exactly once, by the lower leaf.
    const int faceLeaf[3]     = { LeafNeighborTap::plusX,  LeafNeighborTap::plusY,  LeafNeighborTap::plusZ  };  // this leaf's +axis face
    const int faceNeighbor[3] = { LeafNeighborTap::minusX, LeafNeighborTap::minusY, LeafNeighborTap::minusZ };  // neighbor's matching -axis face

    for (int axis = 0; axis < 3; ++axis) {
        const int neighborID = neighborLeafIndex<BuildT>(d_grid, leaf, axis);
        if (neighborID < 0) continue;
        const uint64_t baseNeighbor  = d_offsets[neighborID];
        const int      countNeighbor = int(d_offsets[neighborID + 1] - baseNeighbor);
        if (countNeighbor == 0) continue;

        // Every (this leaf's component, neighbor's component) pair, flattened so the block can
        // stride it; two components are adjacent iff their touching face masks intersect. The
        // slots narrow to ComponentLabelT, which is safe because the component count is checked
        // against MaxComponentCount before any of them is produced.
        const int fLeaf = faceLeaf[axis], fNeighbor = faceNeighbor[axis];
        const int total = countLeaf * countNeighbor;
        for (int p = tID; p < total; p += nThreads) {
            const int i = p / countNeighbor, j = p % countNeighbor;
            if (d_faces[baseLeaf + i][fLeaf] & d_faces[baseNeighbor + j][fNeighbor])
                emit(ComponentLabelT(baseLeaf + i), ComponentLabelT(baseNeighbor + j));
        }
    }
}

// Global union-find over the cross-leaf pairs (representative = min slot in the class).
//
// Design choice: this is a different algorithm from Stage 1's LeafUnionFind. A leaf's graph is an
// implicit 8^3 lattice owned by one warp, so synchronous hook/compress rounds cost only a few
// __syncwarp(). Here the graph is irregular and spans the whole grid, where a synchronous round
// would need a grid-wide barrier (a kernel launch plus a host convergence check per round). An
// asynchronous lock-free union-find (CAS root-to-root links) reaches the same answer in a single
// pass: each pair is united where forEachCrossLeafEdge finds it, so no edge list is built.
//
// Every access to the parent array that can overlap a write in the same pass (the loads in find(),
// the link in unite(), the store in flatten) is a device-scope atomic access, so the passes are free
// of data races. Relaxed ordering suffices: each entry only ever decreases, from its own slot to its
// class's minimum slot, and nothing else is published through it, so the per-entry modification
// order is all the algorithm relies on.
using ParentRef = ::cuda::atomic_ref<uint64_t, ::cuda::thread_scope_device>;

/// @brief Walk parent pointers to the root of x. Links always point larger->smaller slot, so the
///        forest is acyclic and this terminates; the root is the minimum slot in x's class.
///        No path compression: the walk length is the tree depth, which is data-dependent and
///        diverges across a warp. Trees are shallow in practice, but long chains of components
///        united in an unlucky order can deepen them. Path halving in unite()'s finds was tried,
///        and measured slower on an elongated grid.
__device__ inline uint64_t find(uint64_t* parent, uint64_t x)
{
    for (uint64_t p; (p = ParentRef(parent[x]).load(::cuda::std::memory_order_relaxed)) != x; ) x = p;
    return x;
}

/// @brief Lock-free union of the classes of a and b: find both roots and CAS-link the larger root
///        under the smaller (parent[hi] : hi -> lo). Retries if hi ceased to be a root meanwhile.
///        Order-independent result: a class's minimum slot is never the "hi", so it stays the root.
__device__ inline void unite(uint64_t* parent, uint64_t a, uint64_t b)
{
    while (true) {
        const uint64_t ra = find(parent, a);
        const uint64_t rb = find(parent, b);
        if (ra == rb) return;
        const uint64_t hi = (ra > rb) ? ra : rb;
        const uint64_t lo = (ra < rb) ? ra : rb;
        uint64_t expected = hi;
        if (ParentRef(parent[hi]).compare_exchange_strong(expected, lo, ::cuda::std::memory_order_relaxed))
            return;  // linked; else hi moved -> retry
    }
}

// Union-find init: every component is its own root. A plain store: each thread writes only its own
// entry, and nothing else reads the array during this pass.
struct LabelInitFunctor {
    __device__ void operator()(size_t s, uint64_t* p) const { p[s] = uint64_t(s); }
};

/// @brief Union-find unite: links the components of one leaf with those of its +X/+Y/+Z neighbors
///        whose touching faces intersect (each unite has its own CAS-retry). One block per leaf.
template <typename BuildT>
struct CrossLeafUniteFunctor
{
    using ComponentLabelT = ConnectedComponentsBase::ComponentLabelT;

    static constexpr int MaxThreadsPerBlock         = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    /// @param d_offsets  per-leaf component offsets (stage 1)
    /// @param d_faces    per-component face masks (stage 1)
    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets,
                               const uint64_t (*d_faces)[6], uint64_t* d_parent)
    {
        forEachCrossLeafEdge<BuildT>(d_grid, d_offsets, d_faces, blockIdx.x, threadIdx.x, blockDim.x,
            [&] __device__ (ComponentLabelT a, ComponentLabelT b) { unite(d_parent, uint64_t(a), uint64_t(b)); });
    }
};

// Union-find flatten: point every component directly at its representative (class minimum slot).
struct LabelFlattenFunctor {
    __device__ void operator()(size_t s, uint64_t* p) const {
        ParentRef(p[s]).store(find(p, uint64_t(s)), ::cuda::std::memory_order_relaxed);
    }
};

// flag[s] = 1 if component s is a self-root (parent[s]==s), else 0. An inclusive scan of these flags

} // namespace components::detail

template <typename BuildT, typename ResourceT>
void ConnectedComponents<BuildT, ResourceT>::processGlobalConnectedComponents()
{
    const uint64_t K = mLeafComponentAggregateCount;
    mComponentParent = BufT<uint64_t>(mStream, this->ref(), K, nanovdb::cuda::noInit);
    if (K == 0) return;
    uint64_t* d_parent = deviceComponentParent();

    static constexpr unsigned int threadsPerBlock = 256;

    // (a) init: every component is its own root.
    if (mVerbose==1) mTimer.start("Global union-find init");
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(K, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        K, components::detail::LabelInitFunctor{}, d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (b) unite: one block per leaf; each unite has its own CAS-retry, so one pass suffices.
    using UniteOp = components::detail::CrossLeafUniteFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Global union-find unite");
    util::cuda::operatorKernel<UniteOp>
        <<<mSrcTreeData.mNodeCount[0], UniteOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(), deviceLeafComponentFaceMasks(), d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (c) flatten: point every component directly at its representative (class minimum slot).
    if (mVerbose==1) mTimer.start("Global union-find flatten");
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(K, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        K, components::detail::LabelFlattenFunctor{}, d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT, ResourceT>::processGlobalConnectedComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace components::detail {

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stage 3: dense label assignment.

struct RootFlagFunctor {
    // LabelT is deduced from the call, so this stays independent of ConnectedComponents.
    template <typename LabelT>
    __device__ void operator()(size_t s, const uint64_t* p, LabelT* flag) const {
        flag[s] = (p[s] == uint64_t(s)) ? LabelT(1) : LabelT(0);
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Voxel-label scatter: write each active voxel's dense component id into the per-voxel sidecar.
//
// One warp per leaf. For each of the leaf's components, the dense id (rank[parent[s]] - 1) is read
// once and written at leaf.getValue(n) for every voxel n of the component's mask.

template <typename BuildT>
struct VoxelLabelScatterFunctor
{
    static constexpr int LeavesPerBlock             = 4;
    static constexpr int MaxThreadsPerBlock         = 32 * LeavesPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    // LabelT is deduced from the call, so this stays independent of ConnectedComponents.
    template <typename LabelT>
    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets,
                               const nanovdb::Mask<3>* d_masks, const uint64_t* d_parent,
                               const LabelT* d_rank, LabelT* d_voxelLabel, uint32_t leafCount)
    {
        const int      lane   = threadIdx.x & 31;
        const uint32_t leafID = blockIdx.x * LeavesPerBlock + (threadIdx.x >> 5);
        if (leafID >= leafCount) return;
        const auto&    leaf = d_grid->tree().template getFirstNode<0>()[leafID];
        const uint64_t base = d_offsets[leafID], end = d_offsets[leafID + 1];
        for (uint64_t s = base; s < end; ++s) {
            const LabelT    label = LabelT(d_rank[d_parent[s]] - 1);  // dense id in [0,N)
            const uint32_t* words = reinterpret_cast<const uint32_t*>(d_masks[s].words());
            for (int h = 0; h < 16; ++h)   // lane l handles bit l of each of the 16 32-bit words
                if ((words[h] >> lane) & 1u) d_voxelLabel[leaf.getValue(uint32_t(h * 32 + lane))] = label;
        }
    }
};

} // namespace components::detail

template <typename BuildT, typename ResourceT>
void ConnectedComponents<BuildT, ResourceT>::processVoxelLabels()
{
    const uint32_t leafCount = mSrcTreeData.mNodeCount[0];
    const uint64_t valueCount =
        util::cuda::DeviceGridTraits<BuildT>::getValueCount(mDeviceSrcGrid, mStream);
    // Supported grids index exactly their active voxels (no stats), as NanoVDB's CUDA builders
    // produce: a placeholder check, pending a decision on which index layouts to support.
    NANOVDB_ASSERT(valueCount == mSrcTreeData.mVoxelCount + 1);

    // Dense-label sidecar over the grid's value indices (indexed by leaf.getValue(n)). Every slot but
    // slot 0 (the background) is an active voxel and is written by the scatter below, so only slot 0
    // is set to the all-ones sentinel.
    mVoxelLabel = BufT<ComponentLabelT>(mStream, this->ref(), valueCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mVoxelLabel.data(), 0xFF, sizeof(ComponentLabelT), mStream)); // background = 0xFFFFFFFF

    mGlobalComponentCount = 0;
    const uint64_t K = mLeafComponentAggregateCount;
    if (leafCount == 0 || K == 0) return;

    // (a) Assign dense component ids: rank[s] = inclusive count of self-roots in [0,s], so a root's
    //     dense id is rank[root]-1 and rank[K-1] = N.
    //     rankBuf retains mStream, so the scatter below is ordered before the free at scope exit.
    BufT<ComponentLabelT> rankBuf(mStream, this->ref(), K, nanovdb::cuda::noInit);
    auto* d_rank = rankBuf.data();
    static constexpr unsigned int threadsPerBlock = 256;
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(K, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        K, components::detail::RootFlagFunctor{}, deviceComponentParent(), d_rank);
    cudaCheckError();
    if (mVerbose==1) mTimer.start("Component-rank prefix sum");
    CALL_CUBS(DeviceScan::InclusiveSum, d_rank, d_rank, K);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    cudaCheck(cudaMemcpyAsync(&mGlobalComponentCount, d_rank + (K - 1), sizeof(ComponentLabelT),
                              cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    // (b) Scatter: each active voxel's dense component id -> sidecar.
    using ScatterOp = components::detail::VoxelLabelScatterFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Voxel-label scatter");
    util::cuda::operatorKernel<ScatterOp>
        <<<(leafCount + ScatterOp::LeavesPerBlock - 1) / ScatterOp::LeavesPerBlock, ScatterOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(), deviceLeafComponentMasks(),
            deviceComponentParent(), d_rank, mVoxelLabel.data(), leafCount);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT, ResourceT>::processVoxelLabels

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#ifdef CALL_CUBS
#undef CALL_CUBS
#endif

#endif // NANOVDB_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
