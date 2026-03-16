// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file Rasterization.cuh

    \author Efty Sifakis

    \brief Implements GPU kernels for rasterizing triangle mesh geometry into
           NanoVDB sparse tree topology (upper/lower internal node masks and
           leaf voxel value masks).
*/

#ifndef NANOVDB_UTIL_CUDA_RASTERIZATION_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_RASTERIZATION_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Proximity.h>

namespace nanovdb::util {

namespace rasterization {

namespace cuda {

/// @brief Scatters leaf/triangle pair origins into the upper and lower internal
///        node topology masks of a rasterized NanoVDB tree.
///
///        Intended to be called via nanovdb::util::cuda::lambdaKernel.
///        Launched with 1 thread per leaf/triangle pair. Each thread:
///          1. Locates its root tile via probeTile() + pointer-difference index
///          2. Computes upper and lower child offsets via CoordToOffset()
///          3. Atomically sets the corresponding bits in mUpperMasks and mLowerMasks
///
///        Both setOnAtomic() calls are required since multiple pairs can share
///        the same upper node (competing for the upper mask) or the same lower
///        node (competing for the lower mask).
template<typename BuildT, typename PairT>
struct RasterizeInternalNodesFunctor
{
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;

    const PairT *dPairs;
    const RootT *dRoot;
    Mask<5>     *dUpperMasks;
    Mask<4>     (*dLowerMasks)[Mask<5>::SIZE];

    __device__ void operator()(size_t pairID) const
    {
        const auto &pair = dPairs[pairID];

        // Locate the root tile containing this leaf origin
        const auto *tile = dRoot->probeTile(pair.origin);
        uint64_t tileIdx = util::PtrDiff(tile, dRoot->tile(0))
                           / sizeof(typename RootT::Tile);

        // Offsets of the enclosing upper and lower nodes
        const uint32_t upperBit = UpperT::CoordToOffset(pair.origin);
        const uint32_t lowerBit = LowerT::CoordToOffset(pair.origin);

        dUpperMasks[tileIdx].setOnAtomic(upperBit);
        dLowerMasks[tileIdx][upperBit].setOnAtomic(lowerBit);
    }
};

/// @brief Fills leaf voxel value masks via exact point-to-triangle UDF.
///
///        Intended to be called via nanovdb::util::cuda::operatorKernelInstance.
///        Launched as <<<pairCount, MaxThreadsPerBlock>>> - 1 CTA per leaf/triangle
///        pair, 512 threads (one per voxel in the 8^3 leaf). Each CTA:
///          1. Each thread decodes its voxel local coords (lx, ly, lz) and
///             computes closestPointOnTriangleToPoint from the voxel center
///             to the pair's triangle.
///          2. Warp ballot builds a local 16-word mask without atomics (same
///             pattern as evaluateAndCountSubBoxesKernel).
///          3. Thread 0 locates the destination leaf via probeLeaf().
///          4. Threads 0..7 each pack two 32-bit ballots into one uint64_t and
///             atomicOr into the corresponding mask word, allowing multiple CTAs
///             writing to the same leaf to coexist.
///
/// @note  Degenerate triangles (zero area) are handled implicitly: the face
///        interior test naturally fails and the code falls through to the
///        nearest edge/vertex result, which is correct.
/// @tparam TriangleT  Any type with a __hostdev__ const Vec3f& operator[](int) const
///                    returning the i-th vertex (i = 0, 1, 2).
template<typename BuildT, typename PairT, typename TriangleT>
struct RasterizeLeafNodesFunctor
{
    static constexpr int MaxThreadsPerBlock = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    const PairT      *dPairs;
    const TriangleT  *dTriangles;
    NanoGrid<BuildT> *dGrid;
    float             bandWidthSqr;

    __device__ void operator()() const
    {
        const uint64_t pairID   = blockIdx.x;
        const int      threadID = threadIdx.x; // 0..511 = voxel index within leaf

        const auto &pair = dPairs[pairID];
        const auto &tri  = dTriangles[pair.triangleID];

        // Decode voxel local coords within the 8^3 leaf
        // Bit ordering: threadID = lx + ly*8 + lz*64 matches NanoVDB Mask<3> layout
        const int lx =  threadID       & 0x7;
        const int ly = (threadID >> 3) & 0x7;
        const int lz = (threadID >> 6) & 0x7;

        const nanovdb::Vec3f voxelCenter(
            float(pair.origin[0] + lx),
            float(pair.origin[1] + ly),
            float(pair.origin[2] + lz));

        const bool hit = nanovdb::math::pointToTriangleDistSqr(
            tri[0], tri[1], tri[2], voxelCenter) <= bandWidthSqr;

        // Build a per-block local mask via warp ballot (avoids per-voxel atomics).
        // 512 threads -> 16 warps -> 16 x 32-bit ballot words.
        // Mask<3> stores 512 bits as 8 x uint64_t words, so we pack pairs of ballots.
        __shared__ uint32_t s_ballots[16]; // one 32-bit ballot per warp
        const unsigned int ballot = __ballot_sync(0xFFFFFFFF, hit);
        if ((threadID & 31) == 0) s_ballots[threadID >> 5] = ballot;
        __syncthreads();

        // Threads 0..7 each pack two ballots into one uint64_t and atomicOr into the
        // corresponding mask word.
        auto *leaf = const_cast<nanovdb::NanoLeaf<BuildT>*>(
            dGrid->tree().root().probeLeaf(pair.origin));

        if (threadID < int(nanovdb::Mask<3>::WORD_COUNT)) {
            const uint64_t word = uint64_t(s_ballots[2*threadID])
                                | (uint64_t(s_ballots[2*threadID + 1]) << 32);
            auto &maskWord = const_cast<nanovdb::Mask<3>&>(leaf->valueMask()).words()[threadID];
            atomicOr(reinterpret_cast<unsigned long long*>(&maskWord),
                     static_cast<unsigned long long>(word));
        }
    }
};

/// @brief Computes unsigned distance field (UDF) values for all active voxels
///        by iterating over (leaf, triangle) pairs and accumulating the minimum
///        squared distance to each voxel's sidecar entry via an atomic min.
///
///        Intended to be called via nanovdb::util::cuda::operatorKernelInstance.
///        Launched as <<<pairCount, MaxThreadsPerBlock>>> - 1 CTA per leaf/triangle
///        pair, 512 threads (one per voxel in the 8^3 leaf). Each CTA:
///          1. Probes the leaf pointer from the grid using the pair's origin.
///          2. Each thread skips its voxel if inactive in the leaf mask.
///          3. Active threads compute pointToTriangleDistSqr from the voxel center
///             to the pair's triangle.
///          4. If distSqr < bandWidthSqr, an atomicMin via uint32 reinterpret
///             updates dSidecar[sidecarIdx] with the new minimum squared distance.
///
/// @note atomicMin via uint32 reinterpret is valid for all IEEE-754 non-negative
///       floats (including subnormals and +0.0) because their uint32 bit patterns
///       are ordered identically to the corresponding floating-point values.
/// @tparam TriangleT  Any type with a __hostdev__ const Vec3f& operator[](int) const
///                    returning the i-th vertex (i = 0, 1, 2).
template<typename BuildT, typename PairT, typename TriangleT>
struct ComputeUDFFunctor
{
    static constexpr int MaxThreadsPerBlock         = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    const PairT            *dPairs;
    const TriangleT        *dTriangles;
    const NanoGrid<BuildT> *dGrid;
    float                  *dSidecar;
    float                   bandWidthSqr;

    __device__ void operator()() const
    {
        const uint64_t pairID   = blockIdx.x;
        const int      threadID = threadIdx.x;

        const auto &pair = dPairs[pairID];
        const auto *leaf = dGrid->tree().root().probeLeaf(pair.origin);
        if (!leaf) return;
        if (!leaf->isActive(threadID)) return;

        const int lx =  threadID       & 0x7;
        const int ly = (threadID >> 3) & 0x7;
        const int lz = (threadID >> 6) & 0x7;

        const nanovdb::Vec3f voxelCenter(
            float(pair.origin[0] + lx),
            float(pair.origin[1] + ly),
            float(pair.origin[2] + lz));

        const auto &tri = dTriangles[pair.triangleID];
        const float distSqr = nanovdb::math::pointToTriangleDistSqr(
            tri[0], tri[1], tri[2], voxelCenter);

        if (distSqr >= bandWidthSqr) return;

        const uint64_t sidecarIdx = leaf->getValue(threadID);
        atomicMin(reinterpret_cast<uint32_t*>(&dSidecar[sidecarIdx]),
                  __float_as_uint(distSqr));
    }
};

} // namespace cuda

} // namespace rasterization

} // namespace nanovdb::util

#endif // NANOVDB_UTIL_CUDA_RASTERIZATION_CUH_HAS_BEEN_INCLUDED
