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

    Claiming rows needs labeled coalesced groups, __nanosleep and independent thread scheduling, so a
    kernel that writes through SparseLowerMasks requires compute capability 7.0 or newer.
    DenseLowerMasks, and so the deprecated dense path, has no such requirement.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_UTIL_CUDA_LOWERMASKS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_LOWERMASKS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define NANOVDB_LOWERMASKS_CAN_CLAIM 1
#include <cooperative_groups.h>
#else
#define NANOVDB_LOWERMASKS_CAN_CLAIM 0
#endif

namespace nanovdb::util::morphology::cuda {

/// @brief Upper-mask words per processed tile
static constexpr uint32_t UpperMaskWords = Mask<5>::WORD_COUNT;

/// @brief SparseLowerMasks row index of a (tile, uc) no functor has touched
static constexpr uint32_t UnclaimedRow = 0xFFFFFFFFu;
/// @brief SparseLowerMasks row index while one warp claims a row for (tile, uc)
static constexpr uint32_t ClaimingRow  = 0xFFFFFFFEu;
/// @brief SparseLowerMasks row index of a (tile, uc) whose claim overflowed the pool
static constexpr uint32_t OverflowRow  = 0xFFFFFFFDu;
/// @brief Largest SparseLowerMasks pool, so that every row index stays below the sentinels
static constexpr uint32_t MaxLowerRows = OverflowRow;

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

/// @brief Mask<5>::SIZE Mask<4> per tile, indexed [tile][uc]. Converts implicitly from the
///        Mask<4>(*)[Mask<5>::SIZE] array of TopologyBuilder's deprecated dense path.
struct DenseLowerMasks : LowerMasksWrites<DenseLowerMasks>
{
    Mask<4> *rows;

    __hostdev__ DenseLowerMasks(Mask<4> *rows) : rows(rows) {}
    __hostdev__ DenseLowerMasks(Mask<4> (*rows)[Mask<5>::SIZE]) : rows(rows ? rows[0] : nullptr) {}

    __hostdev__ DenseLowerMasks offsetTiles(uint64_t tileBase) const { return DenseLowerMasks(rows + tileBase * Mask<5>::SIZE); }
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

    /// @note A template only so that the compute capability check below fires in kernels
    ///       that write through this view, not in every file that includes the header
    template <int Unused = 0>
    __device__ Mask<4>* row(uint64_t tile, uint32_t uc) const
    {
#if NANOVDB_LOWERMASKS_CAN_CLAIM
        uint32_t *slot = index + tile * Mask<5>::SIZE + uc;
        uint32_t r = *reinterpret_cast<volatile uint32_t*>(slot);
        if (r == UnclaimedRow || r == ClaimingRow) {
            // The lanes that reach here together for the same slot elect one leader to claim
            // it, so a row costs one claim per warp: the documented coalesced-group pattern
            namespace cg = cooperative_groups;
            const cg::coalesced_group peers = cg::labeled_partition(cg::coalesced_threads(), slot);
            if (peers.thread_rank() == 0) r = claim(slot);
            r = peers.shfl(r, 0);
        }
        return r < capacity ? pool + r : nullptr;
#else
        static_assert(Unused != 0, "SparseLowerMasks, and the CUDA topology tools that write through it, require compute capability 7.0 or newer");
        return nullptr;
#endif
    }

#if NANOVDB_LOWERMASKS_CAN_CLAIM
private:
    /// @brief Pool row of @a slot, claiming one if the slot is unclaimed. The claimant never
    ///        waits between its CAS and its store, so a lane waiting on another warp's claim
    ///        is released once independent thread scheduling runs that claimant.
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
#endif
};

} // namespace nanovdb::util::morphology::cuda

#undef NANOVDB_LOWERMASKS_CAN_CLAIM

#endif // NANOVDB_UTIL_CUDA_LOWERMASKS_CUH_HAS_BEEN_INCLUDED
