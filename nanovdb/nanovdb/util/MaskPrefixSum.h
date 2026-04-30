// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/util/MaskPrefixSum.h

    \author Efty Sifakis

    \date March 23, 2026

    \brief  Bit-parallel inclusive prefix-sum over a NanoVDB Mask<3>.

    \details Computes a 512-entry uint16_t table where entry i holds the
    inclusive prefix popcount of the leaf's valueMask up to and including
    voxel i.  That is:

        offsets[i] = number of active voxels at positions 0..i (inclusive)

    The function is intentionally CPU-only; a GPU equivalent would look
    fundamentally different and is not provided here.
*/

#ifndef NANOVDB_UTIL_MASKPREFIXSUM_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_MASKPREFIXSUM_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

namespace nanovdb {

namespace util {

namespace {

/// @brief Union giving byte-accessible view of a uint64_t.
union QWord {
    uint64_t ui64;
    uint8_t  ui8[8];
};

/// @brief Broadcast constant: LSB of each byte in a uint64_t.
static constexpr uint64_t kSpread = UINT64_C(0x0101010101010101);

/// @brief Transpose a single byte-row of an 8x8 bit matrix into a byte-column.
///
/// @details Treats the low 8 bits of \p src as the first row of an 8x8 bit
/// matrix and returns the result of transposing it: bit z of the input byte
/// becomes the LSB of output byte z.  Equivalently:
///
///     output.ui8[z] = (src >> z) & 1  for z = 0..7
///
/// Equivalent to _pdep_u64(src & 0xFF, kSpread) on x86. Implemented via
/// portable arithmetic so that the constituent shift/mask ops vectorize
/// across the y-loop in buildMaskPrefixSums; _pdep_u64 has no SIMD equivalent.
///
/// Two stages:
///   Stage 1: scatter bit-pairs (b1b0, b3b2, b5b4, b7b6) into 16-bit lanes.
///   Stage 2: space the odd bit of each pair into its own byte lane.
///
/// @param src  Any uint64_t; only the low 8 bits are used.
/// @return     uint64_t with bit z of (src & 0xFF) in byte z.
inline uint64_t transposeByteRow(uint64_t src)
{
    uint64_t v = src & 0xFFu;
    // Stage 1: scatter bit-pairs (b1b0, b3b2, b5b4, b7b6) into 16-bit lanes.
    v = (v | (v << 14) | (v << 28) | (v << 42)) & UINT64_C(0x0003000300030003);
    // Stage 2: space the odd bit of each pair into its own byte lane.
    v = (v | (v <<  7)) & UINT64_C(0x0101010101010101);
    return v;
}

} // anonymous namespace

// --------------------------> buildMaskPrefixSums <------------------------------------------

/// @brief Compute the 512-entry inclusive prefix-sum table for a NanoVDB
///        Mask<3> leaf, optionally over the inverted mask.
///
/// @details
/// When @p Invert is false (the default), each entry offsets[i] equals the
/// number of active (set) voxels at linearised positions 0..i (inclusive).
/// When @p Invert is true, offsets[i] equals the number of inactive (unset)
/// voxels at positions 0..i (inclusive) -- equivalently, the result of
/// running the algorithm on the bitwise complement of the mask.
///
/// The algorithm is a three-pass bit-parallel scan that operates entirely
/// on 64 uint64_t values (the 8x8 workspace data[x][y]) without any
/// hardware popcount instruction.
///
/// @tparam Invert     If true, count inactive (0) bits instead of active (1) bits.
/// @param mask        The 8x8x8 leaf value mask (512 bits, 8 uint64_t words).
/// @param prefixSum   Packed cross-word exclusive offsets: 7 x 9-bit fields,
///                    field x-1 = total active voxels in words 0..x-1, for
///                    x = 1..7.  This is the value stored in the leaf's
///                    mPrefixSum member.  Always the original (non-inverted)
///                    field; the function adjusts it internally when Invert=true.
/// @param offsets     Output array of 512 uint16_t values.  offsets[i] holds
///                    the inclusive prefix popcount (of active or inactive bits)
///                    at voxel i.
template <bool Invert = false>
inline void buildMaskPrefixSums(
    const Mask<3>& mask,
    uint64_t       prefixSum,
    uint16_t       offsets[512])
{
    const uint64_t *maskWords = mask.words();

    alignas(64) QWord data[8][8];

    // ------------------------------------------------------------------
    // Step 1: Indicator fill.
    //
    // data[x][y].ui8[z] = 1 if bit (y*8+z) of maskWords[x] is set, else 0.
    //
    // transposeByteRow(maskWords[x] >> (y*8)) extracts byte y of word x and
    // places bit z of that byte into byte z of the result.  The y-loop
    // is independent for fixed x and vectorizable via #pragma omp simd.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++) {
        const uint64_t word = Invert ? ~maskWords[x] : maskWords[x];
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            data[x][y].ui64 = transposeByteRow(word >> (y * 8));
    }

    // ------------------------------------------------------------------
    // Step 2: Z-pass -- Hillis-Steele inclusive prefix sum over z.
    //
    // Three shift-and-add steps perform a length-8 inclusive scan within
    // each byte lane independently.  No inter-byte carry occurs: values
    // enter as 0/1 and reach at most 8.
    //
    // After: data[x][y].ui8[z] = \sum_{z'=0..z} indicator(x, y, z')
    //   i.e. the partial popcount over z-positions 0..z within row (x, y).
    // data[x][y].ui8[7] = full popcount of row (x, y).
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++) {
            uint64_t& zRow = data[x][y].ui64;
            zRow += zRow << 8;
            zRow += zRow << 16;
            zRow += zRow << 32;
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Y-pass -- accumulate preceding-row popcounts.
    //
    // Extract full-row popcounts (byte 7 of each post-Z-pass word) into
    // byte 0, run an exclusive prefix scan over y (sequential in y,
    // independent across x), then broadcast each row offset to all byte
    // lanes via * kSpread and add to data[x][y].
    //
    // After: data[x][y].ui8[z] = (\sum_{y'<y} popcount(row y' of word x))
    //                           + (\sum_{z'=0..z} indicator(x, y, z'))
    //      = inclusive prefix sum within word x at position y*8+z.
    // ------------------------------------------------------------------

    // extract full-row popcounts into byte 0
    alignas(64) QWord shifts[8][8];
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            shifts[x][y].ui64 = data[x][y].ui64 >> 56;  // = data[x][y].ui8[7]
    }

    // exclusive y-prefix scan (sequential over y, independent over x)
    alignas(64) QWord rowOffset[8][8];
    for (int x = 0; x < 8; x++)
        rowOffset[x][0].ui64 = 0;
    for (int y = 1; y < 8; y++)
        for (int x = 0; x < 8; x++)
            rowOffset[x][y].ui64 = rowOffset[x][y-1].ui64 + shifts[x][y-1].ui64;

    // broadcast row offsets to all byte lanes and add
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            data[x][y].ui64 += rowOffset[x][y].ui64 * kSpread;
    }

    // ------------------------------------------------------------------
    // Step 4: Zero-extend bytes to uint16_t in linear index order.
    //
    // data[x][y].ui8[z]  ->  offsets[x*64 + y*8 + z]
    //
    // Values are at most 64 (at most 64 active bits per 64-bit word),
    // so zero-extending to uint16_t is always safe.  The output is already
    // in the correct linear order -- no reordering is required.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++)
            for (int z = 0; z < 8; z++)
                offsets[x*64 + y*8 + z] = data[x][y].ui8[z];

    // ------------------------------------------------------------------
    // Step 5: Add cross-word offsets decoded from prefixSum.
    //
    // Unpack prefixSum into xOffset[x]: the exclusive cumulative popcount
    // up to word x.  Each of the 64 uint16_t entries in slice x is
    // incremented by xOffset[x].
    // ------------------------------------------------------------------
    uint16_t xOffset[8];
    xOffset[0] = 0;
    for (int x = 1; x < 8; x++) {
        const uint16_t ones = static_cast<uint16_t>((prefixSum >> (9*(x-1))) & 0x1FFu);
        if constexpr (Invert)
            xOffset[x] = static_cast<uint16_t>(64 * x) - ones;
        else
            xOffset[x] = ones;
    }

    for (int x = 0; x < 8; x++) {
        uint16_t *p = offsets + x * 64;
        for (int i = 0; i < 64; i++)
            p[i] += xOffset[x];
    }
} // util::buildMaskPrefixSums

} // namespace util

} // namespace nanovdb

#endif // NANOVDB_UTIL_MASKPREFIXSUM_H_HAS_BEEN_INCLUDED
