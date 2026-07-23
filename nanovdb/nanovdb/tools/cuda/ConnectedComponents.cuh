// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/ConnectedComponents.cuh

    \authors Efty Sifakis and JaeHyun Lee

    \brief Connected-components labeling of NanoVDB indexGrids on the device.

           Identifies connected components of active voxels in a ValueOnIndex grid: two
           active voxels share a component label iff they are connected through a path of
           adjacent active voxels.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel

#include <cub/cub.cuh>

#include <utility>  // std::pair

// Private helper macro to call CUB functions that use dynamic temporary storage, in terms of the
// enclosing class's mTempDevicePool + mStream members. Uniquely named and #undef'd at the end of
// this header so it never leaks into or collides with similarly-named macros in other headers
// (e.g. TopologyBuilder.cuh's own CALL_CUBS, which is written in terms of a local `stream`).
#ifdef _WIN32
#define NANOVDB_CC_CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else
#define NANOVDB_CC_CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif

namespace nanovdb {

namespace tools::cuda {

/// @brief Index of each of the 6 leaf faces into a component's face-mask array. Each face is a
///        uint64_t bitmask over the 8x8 boundary plane; cross-leaf adjacency is one AND of the
///        touching faces (faceMasks[I][plusX] & faceMasks[J][minusX]). Bit index per axis:
///          ±X: y*8+z    ±Y: x*8+z    ±Z: y*8+x
enum LeafNeighborTap : int {
    minusX = 0,
    plusX  = 1,
    minusY = 2,
    plusY  = 3,
    minusZ = 4,
    plusZ  = 5
};

/// @brief Undirected edge between two leaf-local components (global slots) touching across a leaf
///        face, stored canonically with a < b. uint32_t: K <= 256*leafCount stays well below 2^32.
struct CrossLeafEdge { uint32_t a, b; };

template <typename BuildT>
class ConnectedComponents
{
    using GridT = NanoGrid<BuildT>;
    using TreeT = NanoTree<BuildT>;
    using RootT = NanoRoot<BuildT>;

public:

    /// @brief Constructor
    /// @param d_srcGrid source device indexGrid whose active voxels are to be labeled
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    ConnectedComponents(const GridT* d_srcGrid, cudaStream_t stream = 0)
        : mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Run the connected-components pipeline and return { d_labels, componentCount }:
    ///          - d_labels: a device array of activeVoxelCount+1 uint32_t, indexed by leaf.getValue(n)
    ///            (slot 0 = background, sentinel 0xFFFFFFFF). Each active voxel holds its component's
    ///            dense id in [0, N), so two active voxels share a value iff they are in the same
    ///            connected component. This is a non-owning view valid for this object's lifetime.
    ///          - componentCount: the number of connected components N.
    std::pair<uint32_t*, uint64_t> getVoxelLabelsAndCount()
    {
        processLeafConnectedComponents();
        processCrossLeafEdges();
        processComponentLabels();
        processVoxelLabels();
        return { static_cast<uint32_t*>(mVoxelLabel.deviceData()), mGlobalComponentCount };
    }

private:

    // --- Pipeline stages (run in order by getVoxelLabelsAndCount). ---

    // Stage 1: per-leaf 6-connected components (each leaf in isolation) -> per-component count,
    // prefix-sum offset, active-voxel Mask<3>, and 6 face masks.
    void processLeafConnectedComponents();

    // Stage 2: emit one edge (a<b) per pair of face-adjacent leaf-local components whose touching
    // face masks intersect. Unordered.
    void processCrossLeafEdges();

    // Stage 3: union-find over the edges -> deviceComponentParent()[s] = component s's representative
    // (its class's minimum global slot).
    void processComponentLabels();

    // Build the per-voxel label sidecar + component count N from the parent array.
    void processVoxelLabels();

    // --- Internal device-array accessors (each valid after the stage that fills it). ---
    auto deviceLeafComponentCounts()    { return static_cast<uint16_t*>(mLeafComponentCounts.deviceData()); }
    auto deviceLeafComponentOffsets()   { return static_cast<uint64_t*>(mLeafComponentOffsets.deviceData()); }
    auto deviceLeafComponentMasks()     { return static_cast<nanovdb::Mask<3>*>(mLeafComponentMasks.deviceData()); }
    auto deviceLeafComponentFaceMasks() { return reinterpret_cast<uint64_t(*)[6]>(mLeafComponentFaceMasks.deviceData()); }
    auto deviceCrossLeafEdges()         { return static_cast<CrossLeafEdge*>(mCrossLeafEdges.deviceData()); }
    auto deviceComponentParent()        { return static_cast<uint64_t*>(mComponentParent.deviceData()); }

    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    const GridT                 *mDeviceSrcGrid;
    nanovdb::cuda::TempDevicePool mTempDevicePool;

    uint64_t                     mLeafComponentAggregateCount{0}; // total leaf-local components across all leaves (= K = offsets[leafCount])

    nanovdb::cuda::DeviceBuffer  mLeafComponentCounts;      // leafCount                    × uint16_t:      per-leaf component count
    nanovdb::cuda::DeviceBuffer  mLeafComponentOffsets;     // (leafCount+1)                × uint64_t:      exclusive+inclusive prefix sums
    nanovdb::cuda::DeviceBuffer  mLeafComponentMasks;       // mLeafComponentAggregateCount × Mask<3>:       per-component active-voxel footprint
    nanovdb::cuda::DeviceBuffer  mLeafComponentFaceMasks;   // mLeafComponentAggregateCount × uint64_t[6]:   per-component face bitmasks (0=-X,1=+X,2=-Y,3=+Y,4=-Z,5=+Z)

    uint64_t                     mCrossLeafEdgeCount{0};    // total cross-leaf edges (E)
    nanovdb::cuda::DeviceBuffer  mCrossLeafEdgeOffsets;     // (leafCount+1) × uint64_t:  per-leaf edge prefix sums
    nanovdb::cuda::DeviceBuffer  mCrossLeafEdges;           // E × CrossLeafEdge (a<b)

    nanovdb::cuda::DeviceBuffer  mComponentParent;          // K × uint64_t: per-component global representative

    nanovdb::cuda::DeviceBuffer  mVoxelLabel;               // (activeVoxelCount+1) × uint32_t: per-active-voxel dense component id in [0,N) (index by leaf.getValue(n); slot 0 = background)
    uint64_t                     mGlobalComponentCount{0};  // number of connected components N (distinct representatives)

}; // tools::cuda::ConnectedComponents<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace cc_detail {

// Per-leaf connected-components via a Shiloach-Vishkin union-find run in shared memory,
// one CUDA block per leaf, one thread per voxel offset n in [0, 512). The forest is stored
// as a parent array of leaf-local voxel offsets: parent[n] = n for active roots, a smaller
// active offset for non-roots, and -1 for inactive voxels. Connectivity is 6-connected and
// strictly intra-leaf (cross-leaf edges are ignored at this stage).
//
// The three primitives are double-buffered (Jacobi): they read the "cur" buffer and write
// the "nxt" buffer, then swap. Inactive entries (-1) are carried through unchanged. The
// pointer swap is performed identically by every thread, so the register copies stay in sync.

constexpr int LEAF_DIM  = 8;            // NanoLeaf DIM
constexpr int LEAF_SIZE = 512;          // 8^3
constexpr int CC_INACTIVE = -1;         // parent sentinel for inactive voxels

// 3D view over a 512-entry parent buffer. The voxel offset is x-major
// (n = (x<<6)|(y<<3)|z), so a row-major int[8][8][8] indexed [x][y][z] aliases the flat
// buffer exactly: element [x][y][z] sits at linear offset x*64 + y*8 + z == n. Accessing
// individual int elements through this view is well-defined (the storage really is int).
using ParentsT = int[LEAF_DIM][LEAF_DIM][LEAF_DIM];

// Smallest parent among the (up to 6) active in-leaf face neighbors of offset n, floored at
// the supplied current value. Offset layout is x-major: n = (x<<6)|(y<<3)|z.
__device__ inline int ccNeighborMin(const int* parentsPtr, int n, int current)
{
    const auto& p = reinterpret_cast<const ParentsT&>(*parentsPtr);
    const int x =  n >> 6       ;
    const int y = (n >> 3) & 0x7;
    const int z =       n  & 0x7;
    int m = current;
    if (x > 0 && p[x-1][y][z] != CC_INACTIVE) m = ::min(m, p[x-1][y][z]);   // -X
    if (x < 7 && p[x+1][y][z] != CC_INACTIVE) m = ::min(m, p[x+1][y][z]);   // +X
    if (y > 0 && p[x][y-1][z] != CC_INACTIVE) m = ::min(m, p[x][y-1][z]);   // -Y
    if (y < 7 && p[x][y+1][z] != CC_INACTIVE) m = ::min(m, p[x][y+1][z]);   // +Y
    if (z > 0 && p[x][y][z-1] != CC_INACTIVE) m = ::min(m, p[x][y][z-1]);   // -Z
    if (z < 7 && p[x][y][z+1] != CC_INACTIVE) m = ::min(m, p[x][y][z+1]);   // +Z
    return m;
}

// SV root hook: every vertex v whose smallest active neighbor label m is below parent[v]
// lowers the slot of v's *parent* (its tree root, once flattened) toward m, via atomicMin.
// Sets *changed (when non-null) iff some root slot was actually lowered.
__device__ inline void ccHook(int*& cur, int*& nxt, int n, int* changed)
{
    const int pn = cur[n];
    nxt[n] = pn;                                  // Phase A: seed nxt = cur (own slot, no race)
    __syncthreads();
    if (pn != CC_INACTIVE) {                      // active voxel
        const int m = ccNeighborMin(cur, n, pn);
        if (m < pn) {                             // root slot is data-dependent -> atomicMin
            const int old = atomicMin_block(&nxt[pn], m);  // block scope: nxt[] is shared
            if (changed && old > m) *changed = 1;
        }
    }
    __syncthreads();
    int* t = cur; cur = nxt; nxt = t;             // swap (identical on every thread)
}

// Pointer-jumping compress: parent[v] <- parent[parent[v]]. Halves tree depth per call.
// Sets *changed (when non-null) iff some entry actually moved.
__device__ inline void ccCompress(int*& cur, int*& nxt, int n, int* changed)
{
    const int pn = cur[n];
    int v = CC_INACTIVE;
    if (pn != CC_INACTIVE) {                      // active: grandparent (cur[pn] is valid)
        v = cur[pn];
        if (changed && v != pn) *changed = 1;
    }
    nxt[n] = v;                                   // own slot, no race
    __syncthreads();
    int* t = cur; cur = nxt; nxt = t;             // swap
}

template <typename BuildT>
struct LeafComponentCountFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    // Safety cap on the convergence loop, well above the worst case for an 8^3 leaf
    // (~log2(depth) + log2(#local minima) <= ~18); guards against a non-terminating bug
    // rather than limiting any legitimate input.
    static constexpr int MaxConvergenceIters = 64;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, uint16_t* d_counts)
    {
        __shared__ int bufA[LEAF_SIZE];
        __shared__ int bufB[LEAF_SIZE];
        __shared__ int changed;
        __shared__ int compCount;

        const int   leafID = blockIdx.x;
        const int   tID    = threadIdx.x;
        const auto& leaf   = d_grid->tree().template getFirstNode<0>()[leafID];

        int* cur = bufA;
        int* nxt = bufB;

        // Init: active voxels label themselves, inactive get the sentinel.
        cur[tID] = leaf.isActive(uint32_t(tID)) ? tID : CC_INACTIVE;
        __syncthreads();

        // Unconditional warm-up: 1 hook + log2(DIM)=3 compresses.
        ccHook    (cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);

        // Then alternate (hook, compress) until a full iteration changes nothing.
        for (int it = 0; it < MaxConvergenceIters; ++it) {
            if (tID == 0) changed = 0;
            __syncthreads();
            ccHook    (cur, nxt, tID, &changed);
            ccCompress(cur, nxt, tID, &changed);
            __syncthreads();
            if (changed == 0) break;
            __syncthreads();  // all threads have read `changed`; safe for thread 0 to reset it next iteration
        }

        // Component count = number of surviving roots (cur[tID] == tID; inactive entries are -1).
        if (tID == 0) compCount = 0;
        __syncthreads();
        if (cur[tID] == tID) atomicAdd_block(&compCount, 1);  // block scope: compCount is shared
        __syncthreads();
        if (tID == 0) d_counts[leafID] = uint16_t(compCount);
    }
}; // LeafComponentCountFunctor

template <typename BuildT>
struct LeafComponentMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int MaxConvergenceIters        = 64;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid,
                                const uint64_t*         d_offsets,
                                nanovdb::Mask<3>*       d_masks,
                                uint64_t              (*d_faces)[6])
    {
        __shared__ int   bufA[LEAF_SIZE];
        __shared__ int   bufB[LEAF_SIZE];
        __shared__ typename cub::BlockReduce<uint32_t, LEAF_SIZE>::TempStorage reduceTmp;
        __shared__ int      changed;
        __shared__ uint32_t sMinLabel;
        // Ballot words (u32/warp) aliased to the Mask<3> u64 words. NAMED union: an anonymous
        // __shared__ union compiled to per-thread local storage, breaking cross-warp sharing.
        __shared__ union { uint32_t u32[16]; uint64_t u64[8]; } sMaskU;
        uint32_t* sMaskWords_u32 = sMaskU.u32;
        uint64_t* sMaskWords     = sMaskU.u64;

        const int   leafID = blockIdx.x;
        const int   tID    = threadIdx.x;
        const auto& leaf   = d_grid->tree().template getFirstNode<0>()[leafID];

        int* cur = bufA;
        int* nxt = bufB;

        // Init + SV convergence: identical schedule to LeafComponentCountFunctor.
        cur[tID] = leaf.isActive(uint32_t(tID)) ? tID : CC_INACTIVE;
        __syncthreads();

        ccHook    (cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);
        ccCompress(cur, nxt, tID, nullptr);

        for (int it = 0; it < MaxConvergenceIters; ++it) {
            if (tID == 0) changed = 0;
            __syncthreads();
            ccHook    (cur, nxt, tID, &changed);
            ccCompress(cur, nxt, tID, &changed);
            __syncthreads();
            if (changed == 0) break;
            __syncthreads();
        }

        // Mask-fill: iterate over leaf-local components in ascending root-label order.
        //
        // Each iteration finds the smallest unprocessed root label via a block-wide unsigned
        // min (CC_INACTIVE = -1 recasts to 0xFFFFFFFF and thus never wins), then collects
        // matching voxels via __ballot_sync and writes the 32-bit result for each warp
        // directly into the Mask<3> word array recast as uint32_t* (each of the 16 warps
        // covers exactly one 32-bit word, so all words are written unconditionally).
        // Processed entries are erased to CC_INACTIVE so they don't win a future min.
        // localCompIdx tracks the dense component index in lockstep across all threads.

        const uint64_t baseOffset = d_offsets[leafID];
        const int warpID = tID >> 5;   // tID / 32
        const int laneID = tID & 31;   // tID % 32

        int localCompIdx = 0;

        while (true) {
            uint32_t minLabel = cub::BlockReduce<uint32_t, LEAF_SIZE>(reduceTmp)
                                    .Reduce(uint32_t(cur[tID]), ::cuda::minimum<uint32_t>{});
            if (tID == 0) sMinLabel = minLabel;
            __syncthreads();
            if (sMinLabel == uint32_t(CC_INACTIVE)) break;

            const bool     match  = (uint32_t(cur[tID]) == sMinLabel);
            const uint32_t ballot = __ballot_sync(0xFFFFFFFF, match);
            if (laneID == 0) sMaskWords_u32[warpID] = ballot;

            if (match) cur[tID] = CC_INACTIVE;
            __syncthreads();  // [SYNC1]: sMaskWords_u32 fully written + cur erases visible

            // Coalesced write: first 8 threads store sMaskWords → mask.words(), one uint64_t each.
            if (tID < 8) {
                d_masks[baseOffset + localCompIdx].words()[tID] = sMaskWords[tID];
            }

            // Extract 6 face masks from sMaskWords (uint64_t view) and store them.
            if (tID == 0) {
                uint64_t* face = d_faces[baseOffset + localCompIdx];

                // ±X: whole word 0 / word 7 are exactly the minusX / plusX face planes.
                face[minusX] = sMaskWords[0];
                face[plusX]  = sMaskWords[7];

                // ±Y: bottom byte (y=0) and top byte (y=7) of each word x;
                //     shift-accumulate from x=7 down; result bit index = x*8 + z (x major, z minor).
                uint64_t mY = sMaskWords[7] & 0xFF,  pY = (sMaskWords[7] >> 56) & 0xFF;
                for (int x = 6; x >= 0; --x) {
                    mY = (mY << 8) | (sMaskWords[x] & 0xFF);
                    pY = (pY << 8) | ((sMaskWords[x] >> 56) & 0xFF);
                }
                face[minusY] = mY;
                face[plusY]  = pY;

                // ±Z: bit 0 (z=0) and bit 7 (z=7) of each byte in each word;
                //     shift-accumulate from x=7 down; result bit index = y*8 + x (y major, x minor).
                uint64_t mZ = sMaskWords[7] & UINT64_C(0x0101010101010101);
                uint64_t pZ = (sMaskWords[7] >> 7) & UINT64_C(0x0101010101010101);
                for (int x = 6; x >= 0; --x) {
                    mZ = (mZ << 1) | (sMaskWords[x] & UINT64_C(0x0101010101010101));
                    pZ = (pZ << 1) | ((sMaskWords[x] >> 7) & UINT64_C(0x0101010101010101));
                }
                face[minusZ] = mZ;
                face[plusZ]  = pZ;
            }
            __syncthreads();  // [SYNC2]: mask + face writes done; all threads safe for next BlockReduce

            ++localCompIdx;
        }
    }
}; // LeafComponentMaskFunctor

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stage 2: cross-leaf connectivity edges.
//
// One block per leaf. For each of the leaf's +X/+Y/+Z neighbor leaves (so each undirected leaf-leaf
// boundary is visited exactly once, from its -side), pair every local component of this leaf with
// every local component of the neighbor and test whether their touching face masks intersect. The
// component "global slot" is leafComponentOffsets[leaf] + localIdx.

/// @brief Linear leaf index of leaf's +axis neighbor (axis: 0=+X, 1=+Y, 2=+Z), or -1 if none.
template <typename BuildT>
__device__ inline int ccNeighborLeafIndex(const NanoGrid<BuildT>* d_grid,
                                           const NanoLeaf<BuildT>&  leaf, int axis)
{
    const nanovdb::Coord o  = leaf.origin();
    const nanovdb::Coord no = (axis == 0) ? o.offsetBy(8, 0, 0)
                            : (axis == 1) ? o.offsetBy(0, 8, 0)
                                          : o.offsetBy(0, 0, 8);
    const auto* nptr = d_grid->tree().root().probeLeaf(no);
    if (!nptr) return -1;
    const auto* base = d_grid->tree().template getFirstNode<0>();
    return int(nptr - base);  // leaves are contiguous breadth-first; typed diff = linear index
}

/// @brief Enumerate this leaf's cross-leaf component pairs whose touching faces intersect, invoking
///        emit(globalSlotThisLeaf, globalSlotNeighbor) for each. Shared by the count and scatter
///        passes so their iteration + AND test cannot drift. Threads of the block stride the pair
///        grid; emit() must be safe under concurrent calls (it uses a block-scoped atomic).
template <typename BuildT, typename EdgeFn>
__device__ inline void ccForEachCrossLeafEdge(
    const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets, const uint64_t (*d_faces)[6],
    int leafID, int tID, int nThreads, EdgeFn&& emit)
{
    const auto&    leaf   = d_grid->tree().template getFirstNode<0>()[leafID];
    const uint64_t baseL  = d_offsets[leafID];
    const int      countL = int(d_offsets[leafID + 1] - baseL);
    if (countL == 0) return;

    const int faceL[3] = { plusX,  plusY,  plusZ  };  // this leaf's +axis face
    const int faceN[3] = { minusX, minusY, minusZ };  // neighbor's matching -axis face

    for (int axis = 0; axis < 3; ++axis) {
        const int N = ccNeighborLeafIndex<BuildT>(d_grid, leaf, axis);
        if (N < 0) continue;
        const uint64_t baseN  = d_offsets[N];
        const int      countN = int(d_offsets[N + 1] - baseN);
        if (countN == 0) continue;
        const int fL = faceL[axis], fN = faceN[axis];
        const int total = countL * countN;
        for (int p = tID; p < total; p += nThreads) {
            const int i = p / countN, j = p % countN;
            if (d_faces[baseL + i][fL] & d_faces[baseN + j][fN])
                emit(uint32_t(baseL + i), uint32_t(baseN + j));
        }
    }
}

template <typename BuildT>
struct CrossLeafEdgeCountFunctor
{
    static constexpr int MaxThreadsPerBlock         = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    // d_outCount is (leafComponentEdgeOffsets + 1), so writing [leafID] lands at offsets[leafID+1].
    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets,
                               const uint64_t (*d_faces)[6], uint64_t* d_outCount)
    {
        __shared__ int sEdges;
        const int leafID = blockIdx.x, tID = threadIdx.x;
        if (tID == 0) sEdges = 0;
        __syncthreads();
        ccForEachCrossLeafEdge<BuildT>(d_grid, d_offsets, d_faces, leafID, tID, blockDim.x,
            [&] __device__ (uint32_t, uint32_t) { atomicAdd_block(&sEdges, 1); });
        __syncthreads();
        if (tID == 0) d_outCount[leafID] = uint64_t(sEdges);
    }
};

template <typename BuildT>
struct CrossLeafEdgeScatterFunctor
{
    static constexpr int MaxThreadsPerBlock         = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets,
                               const uint64_t (*d_faces)[6], const uint64_t* d_edgeOffsets,
                               CrossLeafEdge* d_edges)
    {
        __shared__ unsigned long long sWrite;
        const int leafID = blockIdx.x, tID = threadIdx.x;
        if (tID == 0) sWrite = (unsigned long long)d_edgeOffsets[leafID];
        __syncthreads();
        ccForEachCrossLeafEdge<BuildT>(d_grid, d_offsets, d_faces, leafID, tID, blockDim.x,
            [&] __device__ (uint32_t ga, uint32_t gb) {
                const unsigned long long slot = atomicAdd_block(&sWrite, 1ull);
                CrossLeafEdge e;
                e.a = (ga < gb) ? ga : gb;  // canonical a < b
                e.b = (ga < gb) ? gb : ga;
                d_edges[slot] = e;
            });
        __syncthreads();
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Stage 3: global union-find over the cross-leaf edge list (representative = min slot in the class).

/// @brief Walk parent pointers to the root of x. Links always point larger->smaller slot, so the
///        forest is acyclic and this terminates; the root is the minimum slot in x's class.
__device__ inline uint64_t ccFind(const uint64_t* parent, uint64_t x)
{
    while (parent[x] != x) x = parent[x];
    return x;
}

/// @brief Lock-free union of the classes of a and b: find both roots and CAS-link the larger root
///        under the smaller (parent[hi] : hi -> lo). Retries if hi ceased to be a root meanwhile.
///        Order-independent result: a class's minimum slot is never the "hi", so it stays the root.
__device__ inline void ccUnite(uint64_t* parent, uint64_t a, uint64_t b)
{
    while (true) {
        const uint64_t ra = ccFind(parent, a);
        const uint64_t rb = ccFind(parent, b);
        if (ra == rb) return;
        const uint64_t hi = (ra > rb) ? ra : rb;
        const uint64_t lo = (ra < rb) ? ra : rb;
        const unsigned long long old = atomicCAS(
            reinterpret_cast<unsigned long long*>(&parent[hi]),
            static_cast<unsigned long long>(hi),
            static_cast<unsigned long long>(lo));
        if (old == static_cast<unsigned long long>(hi)) return;  // linked; else hi moved -> retry
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Small element-wise functors driven by lambdaKernel. These are named structs (rather than inline
// extended __device__ lambdas) so their enclosing ConnectedComponents member functions can have
// private access — nvcc forbids extended __device__ lambdas inside private/protected methods — which
// also matches the sibling operators (PruneGrid/DilateGrid/... drive lambdaKernel with named functors).

// offsets[i+1] = counts[i]: upcast the per-leaf uint16 component counts into the uint64 offset array.
struct UpcastCountsFunctor {
    __device__ void operator()(size_t i, const uint16_t* counts, uint64_t* offsets) const {
        offsets[i + 1] = uint64_t(counts[i]);
    }
};

// Union-find init: every component is its own root.
struct LabelInitFunctor {
    __device__ void operator()(size_t s, uint64_t* p) const { p[s] = uint64_t(s); }
};

// Union-find unite: link the two endpoints of cross-leaf edge e (each ccUnite has its own CAS-retry).
struct LabelUniteFunctor {
    __device__ void operator()(size_t e, uint64_t* p, const CrossLeafEdge* edges) const {
        ccUnite(p, uint64_t(edges[e].a), uint64_t(edges[e].b));
    }
};

// Union-find flatten: point every component directly at its representative (class minimum slot).
struct LabelFlattenFunctor {
    __device__ void operator()(size_t s, uint64_t* p) const { p[s] = ccFind(p, uint64_t(s)); }
};

// flag[s] = 1 if component s is a self-root (parent[s]==s), else 0. An inclusive scan of these flags
// gives each root its dense id (scan[root]-1) and, in the last entry, the component count N.
struct RootFlagFunctor {
    __device__ void operator()(size_t s, const uint64_t* p, uint32_t* flag) const {
        flag[s] = (p[s] == uint64_t(s)) ? 1u : 0u;
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Voxel-label scatter: write each active voxel's dense component id into the per-voxel sidecar.
//
// One block per leaf, one thread per voxel offset n. The leaf's components partition its active
// voxels, so the short scan over the leaf's slots finds the unique component containing n; its dense
// id (rank[parent[s]] - 1) is written at leaf.getValue(n).

template <typename BuildT>
struct VoxelLabelScatterFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* d_grid, const uint64_t* d_offsets,
                               const nanovdb::Mask<3>* d_masks, const uint64_t* d_parent,
                               const uint32_t* d_rank, uint32_t* d_voxelLabel)
    {
        const int   leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf   = d_grid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;
        const uint64_t base = d_offsets[leafID], end = d_offsets[leafID + 1];
        for (uint64_t s = base; s < end; ++s)
            if (d_masks[s].isOn(uint32_t(n))) {
                d_voxelLabel[leaf.getValue(uint32_t(n))] = d_rank[d_parent[s]] - 1u;  // dense id in [0,N)
                return;
            }
    }
};

} // namespace cc_detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void ConnectedComponents<BuildT>::processLeafConnectedComponents()
{
    const uint32_t leafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid).mNodeCount[0];

    // Allocate one component-count per leaf (device-only). At most 256 components per 8^3
    // leaf (6-connected worst case), so uint16_t is sufficient.
    if (mVerbose==1) mTimer.start("Allocating per-leaf component counts");
    mLeafComponentCounts = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(leafCount) * sizeof(uint16_t), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    if (leafCount == 0) return;

    // One block per leaf, one thread per voxel offset; counts the distinct 6-connected
    // components of each leaf's active voxels (in isolation) into mLeafComponentCounts.
    using Op = cc_detail::LeafComponentCountFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf connected-component counting");
    util::cuda::operatorKernel<Op>
        <<<leafCount, Op::MaxThreadsPerBlock, 0, mStream>>>(mDeviceSrcGrid, deviceLeafComponentCounts());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Prefix sum: mLeafComponentOffsets[0]=0, mLeafComponentOffsets[1..leafCount] = inclusive sum
    // of mLeafComponentCounts. mLeafComponentOffsets[leafCount] = K (total leaf-local components).
    if (mVerbose==1) mTimer.start("Allocating per-leaf component offsets");
    mLeafComponentOffsets = nanovdb::cuda::DeviceBuffer::create(
        (std::size_t(leafCount) + 1) * sizeof(uint64_t), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    cudaCheck(cudaMemsetAsync(mLeafComponentOffsets.deviceData(), 0, sizeof(uint64_t), mStream));

    // Upcast per-leaf uint16_t counts into offsets[1..leafCount] as uint64_t.
    uint16_t* d_counts  = deviceLeafComponentCounts();
    uint64_t* d_offsets = deviceLeafComponentOffsets();
    util::cuda::lambdaKernel<<<(leafCount + 255) / 256, 256, 0, mStream>>>(
        leafCount, cc_detail::UpcastCountsFunctor{}, d_counts, d_offsets);
    cudaCheckError();

    // In-place inclusive sum over offsets[1..leafCount]; offsets[leafCount] = K (total components).
    if (mVerbose==1) mTimer.start("Per-leaf component offset prefix sum");
    NANOVDB_CC_CALL_CUBS(DeviceScan::InclusiveSum, d_offsets + 1, d_offsets + 1, int(leafCount));
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Read the aggregate count from the sentinel at offsets[leafCount].
    // Use async copy on mStream (ordered after the scan) + stream sync to avoid
    // inadvertently synchronizing other streams.
    cudaCheck(cudaMemcpyAsync(&mLeafComponentAggregateCount, d_offsets + leafCount,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));

    // Allocate mLeafComponentAggregateCount Mask<3> objects — one per leaf-local component.
    // No zero-init needed: the mask-fill kernel writes all 16 uint32_t words of every mask
    // unconditionally via warp ballot (zero ballot = no component voxels in that warp).
    if (mVerbose==1) mTimer.start("Allocating per-component leaf masks");
    mLeafComponentMasks = nanovdb::cuda::DeviceBuffer::create(
        mLeafComponentAggregateCount * sizeof(nanovdb::Mask<3>), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    // Allocate 6 uint64_t face masks per component (one per LeafNeighborTap entry).
    // The face-extraction kernel fills these; no zero-init needed for the same reason.
    if (mVerbose==1) mTimer.start("Allocating per-component face masks");
    mLeafComponentFaceMasks = nanovdb::cuda::DeviceBuffer::create(
        mLeafComponentAggregateCount * 6 * sizeof(uint64_t), nullptr, false);
    if (mVerbose==1) mTimer.stop();

    // Re-run SV per leaf and scatter each active voxel's bit into its component's Mask<3>.
    using MaskOp = cc_detail::LeafComponentMaskFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Per-leaf component mask fill");
    util::cuda::operatorKernel<MaskOp>
        <<<leafCount, MaskOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(),
            deviceLeafComponentMasks(), deviceLeafComponentFaceMasks());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT>::processLeafConnectedComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void ConnectedComponents<BuildT>::processCrossLeafEdges()
{
    const uint32_t leafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid).mNodeCount[0];
    mCrossLeafEdgeCount = 0;
    if (leafCount == 0) return;

    // Per-leaf edge offsets: offsets[0]=0, offsets[1..leafCount] filled by the count pass then
    // scanned in place; offsets[leafCount] = E (total cross-leaf edges).
    mCrossLeafEdgeOffsets = nanovdb::cuda::DeviceBuffer::create(
        (std::size_t(leafCount) + 1) * sizeof(uint64_t), nullptr, false);
    uint64_t* d_offsets = static_cast<uint64_t*>(mCrossLeafEdgeOffsets.deviceData());
    cudaCheck(cudaMemsetAsync(d_offsets, 0, sizeof(uint64_t), mStream));

    // Count pass: one block per leaf, writing each leaf's edge count into offsets[leafID+1].
    using CountOp = cc_detail::CrossLeafEdgeCountFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Cross-leaf edge count");
    util::cuda::operatorKernel<CountOp>
        <<<leafCount, CountOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(), deviceLeafComponentFaceMasks(), d_offsets + 1);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // Inclusive prefix sum over offsets[1..leafCount]; offsets[leafCount] = E.
    if (mVerbose==1) mTimer.start("Cross-leaf edge offset prefix sum");
    NANOVDB_CC_CALL_CUBS(DeviceScan::InclusiveSum, d_offsets + 1, d_offsets + 1, int(leafCount));
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    cudaCheck(cudaMemcpyAsync(&mCrossLeafEdgeCount, d_offsets + leafCount,
                              sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    if (mCrossLeafEdgeCount == 0) return;  // single-leaf grid / no touching components

    // Scatter pass: write each leaf's edges into [offsets[leafID], offsets[leafID+1]).
    mCrossLeafEdges = nanovdb::cuda::DeviceBuffer::create(
        mCrossLeafEdgeCount * sizeof(CrossLeafEdge), nullptr, false);
    using ScatterOp = cc_detail::CrossLeafEdgeScatterFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Cross-leaf edge scatter");
    util::cuda::operatorKernel<ScatterOp>
        <<<leafCount, ScatterOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(), deviceLeafComponentFaceMasks(),
            d_offsets, deviceCrossLeafEdges());
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT>::processCrossLeafEdges

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void ConnectedComponents<BuildT>::processComponentLabels()
{
    const uint64_t K = mLeafComponentAggregateCount;
    mComponentParent = nanovdb::cuda::DeviceBuffer::create(
        std::size_t(K ? K : 1) * sizeof(uint64_t), nullptr, false);  // avoid 0-byte alloc
    if (K == 0) return;
    uint64_t* d_parent = deviceComponentParent();

    auto blocks = [](uint64_t n) { return (unsigned int)((n + 255) / 256); };

    // (a) init: every component is its own root.
    if (mVerbose==1) mTimer.start("Component-label init");
    util::cuda::lambdaKernel<<<blocks(K), 256, 0, mStream>>>(
        K, cc_detail::LabelInitFunctor{}, d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    // (b) unite: one thread per edge; each ccUnite has its own CAS-retry, so one pass suffices.
    if (mCrossLeafEdgeCount) {
        if (mVerbose==1) mTimer.start("Component-label unite");
        util::cuda::lambdaKernel<<<blocks(mCrossLeafEdgeCount), 256, 0, mStream>>>(
            mCrossLeafEdgeCount, cc_detail::LabelUniteFunctor{}, d_parent, deviceCrossLeafEdges());
        cudaCheckError();
        if (mVerbose==1) mTimer.stop();
    }

    // (c) flatten: point every component directly at its representative (class minimum slot).
    if (mVerbose==1) mTimer.start("Component-label flatten");
    util::cuda::lambdaKernel<<<blocks(K), 256, 0, mStream>>>(
        K, cc_detail::LabelFlattenFunctor{}, d_parent);
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT>::processComponentLabels

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void ConnectedComponents<BuildT>::processVoxelLabels()
{
    const uint32_t leafCount =
        util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid).mNodeCount[0];
    const uint64_t activeCount =
        util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(mDeviceSrcGrid);

    // Per-active-voxel dense-label sidecar (indexed by leaf.getValue(n); slot 0 = background).
    mVoxelLabel = nanovdb::cuda::DeviceBuffer::create((activeCount + 1) * sizeof(uint32_t), nullptr, false);
    cudaCheck(cudaMemsetAsync(mVoxelLabel.deviceData(), 0xFF, (activeCount + 1) * sizeof(uint32_t), mStream)); // background = 0xFFFFFFFF

    mGlobalComponentCount = 0;
    const uint64_t K = mLeafComponentAggregateCount;
    if (leafCount == 0 || K == 0) return;

    // (a) Assign dense component ids: rank[s] = inclusive count of self-roots in [0,s], so a root's
    //     dense id is rank[root]-1 and rank[K-1] = N.
    auto rankBuf = nanovdb::cuda::DeviceBuffer::create(K * sizeof(uint32_t), nullptr, false);
    auto* d_rank = static_cast<uint32_t*>(rankBuf.deviceData());
    util::cuda::lambdaKernel<<<(unsigned int)((K + 255) / 256), 256, 0, mStream>>>(
        K, cc_detail::RootFlagFunctor{}, deviceComponentParent(), d_rank);
    cudaCheckError();
    NANOVDB_CC_CALL_CUBS(DeviceScan::InclusiveSum, d_rank, d_rank, int(K));
    cudaCheckError();

    uint32_t hN = 0;
    cudaCheck(cudaMemcpyAsync(&hN, d_rank + (K - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    mGlobalComponentCount = uint64_t(hN);

    // (b) Scatter: each active voxel's dense component id -> sidecar.
    using ScatterOp = cc_detail::VoxelLabelScatterFunctor<BuildT>;
    if (mVerbose==1) mTimer.start("Voxel-label scatter");
    util::cuda::operatorKernel<ScatterOp>
        <<<leafCount, ScatterOp::MaxThreadsPerBlock, 0, mStream>>>(
            mDeviceSrcGrid, deviceLeafComponentOffsets(), deviceLeafComponentMasks(),
            deviceComponentParent(), d_rank, static_cast<uint32_t*>(mVoxelLabel.deviceData()));
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();
}// ConnectedComponents<BuildT>::processVoxelLabels

} // namespace tools::cuda

} // namespace nanovdb

#undef NANOVDB_CC_CALL_CUBS  // don't leak the private helper macro

#endif // NVIDIA_TOOLS_CUDA_CONNECTEDCOMPONENTS_CUH_HAS_BEEN_INCLUDED
