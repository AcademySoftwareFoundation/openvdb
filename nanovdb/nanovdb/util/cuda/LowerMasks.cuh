// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/util/cuda/LowerMasks.cuh

    \brief Device views of the lower-node child masks that tools::cuda::TopologyBuilder
           collects from a consumer's internal-node functors.

    A functor marks lower node @c bit of upper-node child @c uc in processed root tile
    @c tile. It sets the upper-mask bit itself and writes the lower mask through a view:

        upperMasks[tile].setOnAtomic(uc);
        lowerMasks.setOnAtomic(tile, uc, bit);

    - SparseLowerMasks stores a Mask<4> only for the (tile, uc) pairs the functors touch.
      TopologyBuilder::buildInternalMasks hands this view to the functors.
    - DenseLowerMasks stores Mask<5>::SIZE Mask<4> per tile (16 MiB), for the deprecated
      TopologyBuilder::allocateInternalMaskBuffers path.

    A view passed for one grid of a batch is shifted to that grid's first tile with
    offsetTiles(tileBase).

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_UTIL_CUDA_LOWERMASKS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_LOWERMASKS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

namespace nanovdb::util::morphology::cuda {

/// @brief Upper-mask words per processed tile
static constexpr uint32_t UpperMaskWords = Mask<5>::WORD_COUNT;

/// @brief SparseLowerMasks row index of a (tile, uc) no functor has touched
static constexpr uint32_t UnclaimedRow = 0xFFFFFFFFu;
/// @brief SparseLowerMasks row index while one warp claims a row for (tile, uc)
static constexpr uint32_t ClaimingRow  = 0xFFFFFFFEu;
/// @brief SparseLowerMasks row index of a (tile, uc) whose claim overflowed the pool
static constexpr uint32_t OverflowRow  = 0xFFFFFFFDu;

/// @brief Write calls shared by the views; Derived supplies row(tile, uc), which returns
///        nullptr when the write must be dropped
template <typename Derived>
struct LowerMasksWrites
{
    __device__ void setOnAtomic(uint64_t tile, uint32_t uc, uint32_t bit) const
    {
        if (Mask<4> *m = static_cast<const Derived&>(*this).row(tile, uc)) m->setOnAtomic(bit);
    }
    __device__ void atomicOrWord(uint64_t tile, uint32_t uc, uint32_t word, uint64_t bits) const
    {
        if (Mask<4> *m = static_cast<const Derived&>(*this).row(tile, uc)) util::atomicOr(m->words() + word, bits);
    }
    /// @brief Non-atomic OR, for callers whose threads own disjoint words of the row
    __device__ void orWord(uint64_t tile, uint32_t uc, uint32_t word, uint64_t bits) const
    {
        if (Mask<4> *m = static_cast<const Derived&>(*this).row(tile, uc)) m->words()[word] |= bits;
    }
};

/// @brief Mask<5>::SIZE Mask<4> per tile, indexed [tile][uc]
struct DenseLowerMasks : LowerMasksWrites<DenseLowerMasks>
{
    Mask<4> *rows;

    __hostdev__ DenseLowerMasks offsetTiles(uint64_t tileBase) const { return {{}, rows + tileBase * Mask<5>::SIZE}; }
    __device__ Mask<4>* row(uint64_t tile, uint32_t uc) const { return rows + tile * Mask<5>::SIZE + uc; }
};

/// @brief Rows claimed on first touch from a pool of @c capacity zeroed Mask<4>. @c index
///        holds one row index per (tile, uc), initialized to UnclaimedRow. A claim past the
///        pool sets @c *overflow and drops the write.
struct SparseLowerMasks : LowerMasksWrites<SparseLowerMasks>
{
    uint32_t *index;
    Mask<4>  *pool;
    uint32_t *claimed;// rows handed out so far
    uint32_t *overflow;
    uint32_t  capacity;

    __hostdev__ SparseLowerMasks offsetTiles(uint64_t tileBase) const
    {
        SparseLowerMasks v = *this;
        v.index += tileBase * Mask<5>::SIZE;
        return v;
    }

    __device__ Mask<4>* row(uint64_t tile, uint32_t uc) const
    {
        // Lanes after the same row elect one leader to claim it, so a row costs one claim
        // per warp rather than one per lane
        uint32_t *slot = index + tile * Mask<5>::SIZE + uc;
        const uint32_t peers = __match_any_sync(__activemask(), reinterpret_cast<unsigned long long>(slot));
        const int leader = __ffs(peers) - 1;
        uint32_t r = 0;
        if ((threadIdx.x & 31) == uint32_t(leader)) r = claim(slot);
        r = __shfl_sync(peers, r, leader);
        return r == OverflowRow ? nullptr : pool + r;
    }

private:
    /// @brief Pool row of @a slot, claiming one if the slot is unclaimed and waiting while
    ///        another warp claims it
    __device__ uint32_t claim(uint32_t *slot) const
    {
        uint32_t r = *reinterpret_cast<volatile uint32_t*>(slot);
        if (r == UnclaimedRow) {
            if (atomicCAS(slot, UnclaimedRow, ClaimingRow) == UnclaimedRow) {
                r = atomicAdd(claimed, 1u);
                if (r >= capacity) {
                    r = OverflowRow;
                    atomicExch(overflow, 1u);
                }
                __threadfence();
                atomicExch(slot, r);
                return r;
            }
            r = ClaimingRow;
        }
        while (r == ClaimingRow) {
            __nanosleep(32);
            r = *reinterpret_cast<volatile uint32_t*>(slot);
        }
        __threadfence();
        return r;
    }
};

} // namespace nanovdb::util::morphology::cuda

#endif // NANOVDB_UTIL_CUDA_LOWERMASKS_CUH_HAS_BEEN_INCLUDED
