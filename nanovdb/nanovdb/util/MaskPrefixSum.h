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

    To obtain the exclusive prefix (the value stored in mOffset for a given
    active voxel), subtract the active indicator: offsets[i] - isActive(i).

    The function is intentionally CPU-only.  A CUDA equivalent would be
    organized around the 32-thread warp (using __ballot_sync and prefix
    intrinsics) and would look fundamentally different; marking this
    __hostdev__ would be misleading about the intended usage.

    \par NanoVDB leaf-mask layout

    A NanoVDB Mask<3> for an 8x8x8 leaf stores 512 bits as 8 uint64_t words.
    Word x covers voxels at x-coordinates [8x, 8x+63] in the linearisation
    i = x*64 + y*8 + z, where y is the byte index within the word (0..7) and
    z is the bit index within each byte (0..7).

    The packed parameter \p prefixSum encodes the seven exclusive cumulative
    popcounts at the word boundaries:

        prefixSum[9*(x-1) .. 9*(x-1)+8]  =  total active voxels in words 0..x-1
                                             (exclusive offset for word x), x = 1..7

    This is the same encoding stored in nanovdb::LeafNode::mPrefixSum.

    \par Algorithm

    The algorithm works in five steps on an 8x8 workspace of uint64_t values
    (named data[x][y]) where x is the word index (0..7) and y is the byte index
    within each word (0..7).  The eight bits z (0..7) live as individual bytes
    inside each uint64.

    Step 1 -- Indicator fill (transposeByteRow).
        data[x][y].ui8[z] = 1 if bit (y*8+z) of maskWords[x] is set, else 0.
        Implemented via transposeByteRow, which transposes byte y of word x
        (a 1x8 bit row) into an 8x1 bit column stored as the LSB of each byte.

    Step 2 -- Z-pass: Hillis-Steele inclusive prefix sum over z (within each uint64).
        Three shift-and-add steps accumulate within each byte lane independently
        (values stay <= 8 so no inter-byte carry occurs):
            d += d << 8;   // stride 1
            d += d << 16;  // stride 2
            d += d << 32;  // stride 4
        After this pass, data[x][y].ui8[z] = number of active bits at
        z-positions 0..z within row (x, y).  As a free by-product,
        data[x][y].ui8[7] holds the full popcount of row y of word x.

    Step 3 -- Y-pass: exclusive row-prefix scan + broadcast.
        Sub-step 3a: extract the full row popcount (byte 7) into shifts[x][y].
        Sub-step 3b: exclusive scan over y (sequential y-loop, independent x):
            rowOffset[x][0] = 0
            rowOffset[x][y] = rowOffset[x][y-1] + shifts[x][y-1]
        Sub-step 3c+3d: broadcast rowOffset[x][y] to all 8 byte lanes (by
        multiplying by kSpread = 0x0101010101010101) and add to data[x][y].

        After this pass, data[x][y].ui8[z] is the inclusive prefix sum of the
        linearised voxel index x*64 + y*8 + z within word x.

    Step 4 -- Zero-extend to uint16_t.
        data[x][y].ui8[z]  ->  offsets[x*64 + y*8 + z]
        Values are at most 64 (one full 64-bit word), safe to zero-extend.

    Step 5 -- Add cross-word offsets.
        Decode xOffset[x] from \p prefixSum and add to all 64 entries of
        slice x, yielding the global inclusive prefix sum across the full leaf.
*/

#ifndef NANOVDB_UTIL_MASKPREFIXSUM_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_MASKPREFIXSUM_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

#include <cstdint>

namespace nanovdb {

namespace util {

// -----------------------------------------------------------------------
// File-internal helpers
// -----------------------------------------------------------------------

namespace {// anonymous

/// @brief Union giving byte-accessible view of a uint64_t.
union MaskPrefixSumQWord {
    uint64_t ui64;
    uint8_t  ui8[8];
};

/// @brief Broadcast constant: LSB of each byte in a uint64_t.
static constexpr uint64_t kMaskPSSpread = UINT64_C(0x0101010101010101);

/// @brief Transpose a single byte-row of an 8x8 bit matrix into a byte-column.
///
/// @details Treats the low 8 bits of \p src as the first row of an 8x8 bit
/// matrix and returns the result of transposing it: bit z of the input byte
/// becomes the LSB of output byte z.  Equivalently:
///
///     output.ui8[z] = (src >> z) & 1  for z = 0..7
///
/// This is the single-row specialization of transposeBits8x8: whereas the
/// full transpose maps an arbitrary 8x8 bit matrix, this function handles
/// the common case where only the first row is non-zero, and is cheaper.
/// It is also equivalent to _pdep_u64(src & 0xFF, kMaskPSSpread) on x86.
///
/// Two-stage multiply-and-mask:
///   Stage 1: replicate 4 bit-pairs into 16-bit lanes.
///     Multiply by 2^0 + 2^14 + 2^28 + 2^42 = 0x0000040010004001, mask with
///     0x0003000300030003 (bits 0-1, 16-17, 32-33, 48-49) to obtain pairs
///     [b1,b0], [b3,b2], [b5,b4], [b7,b6] at 16-bit boundaries.
///     Because the input is <= 8 bits the shifted copies never overlap, so
///     OR is equivalent to multiplication and the compiler emits
///     vpsllq + vpor + vpand rather than vpmuludq.
///   Stage 2: separate each pair into individual byte lanes.
///     (v | (v << 7)) & kMaskPSSpread extracts bit z into byte z.
///     Preferred over v * 129 because AVX2 has no 64x64->64 vector multiply;
///     both spellings reduce to vpsllq + vpaddq + vpand anyway.
///
/// @param src  Any uint64_t; only the low 8 bits are used.
/// @return     uint64_t with bit z of (src & 0xFF) in byte z.
inline uint64_t transposeByteRow(uint64_t src)
{
    uint64_t v = src & 0xFFu;
    // Stage 1: replicate into 16-bit pairs.
    v = (v | (v << 14) | (v << 28) | (v << 42)) & UINT64_C(0x0003000300030003);
    // Stage 2: separate each pair into its own byte lane.
    v = (v | (v <<  7))                          & UINT64_C(0x0101010101010101);
    return v;
}

} // anonymous namespace

// -----------------------------------------------------------------------
// buildMaskPrefixSums
// -----------------------------------------------------------------------

/// @brief Compute the 512-entry inclusive prefix-sum table for a NanoVDB
///        Mask<3> leaf.
///
/// @details
/// Each entry offsets[i] equals the number of active voxels at linearised
/// positions 0..i (inclusive) within the leaf.  The exclusive prefix sum
/// (used as the voxel-data index offset) is offsets[i] - isActive(i).
///
/// The algorithm is a three-pass bit-parallel scan that operates entirely
/// on 64 uint64_t values (the 8x8 workspace data[x][y]) without any
/// hardware popcount instruction.  See the file-level documentation for a
/// step-by-step description of all five passes.
///
/// @param mask        The 8x8x8 leaf value mask (512 bits, 8 uint64_t words).
/// @param prefixSum   Packed cross-word exclusive offsets: 7 x 9-bit fields,
///                    field x-1 = total active voxels in words 0..x-1, for
///                    x = 1..7.  This is the value stored in the leaf's
///                    mPrefixSum member.
/// @param offsets     Output array of 512 uint16_t values.  offsets[i] holds
///                    the inclusive prefix popcount at voxel i.
inline void buildMaskPrefixSums(
    const Mask<3>& mask,
    uint64_t       prefixSum,
    uint16_t       offsets[512])
{
    const uint64_t* maskWords = mask.words();

    alignas(64) MaskPrefixSumQWord data[8][8];

    // ------------------------------------------------------------------
    // Step 1: Indicator fill.
    //
    // data[x][y].ui8[z] = 1 if bit (y*8+z) of maskWords[x] is set, else 0.
    //
    // transposeByteRow(maskWords[x] >> (y*8)) extracts byte y of word x and
    // places bit z of that byte into byte z of the result.  The y-loop
    // is independent for fixed x and vectorisable via #pragma omp simd;
    // the 8 outer x-iterations are also independent, allowing the
    // out-of-order engine to overlap the multiply chains.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            data[x][y].ui64 = transposeByteRow(maskWords[x] >> (y * 8));
    }

    // ------------------------------------------------------------------
    // Step 2: Z-pass -- Hillis-Steele inclusive prefix sum over z.
    //
    // z is the bit index within each byte, which after the indicator fill
    // lives as the byte index of the uint64.  Three shift-and-add steps
    // perform a length-8 inclusive scan within each byte lane independently.
    // No inter-byte carry occurs: values enter as 0/1 and reach at most 8.
    //
    // After: data[x][y].ui8[z] = \sum_{z'=0..z} indicator(x, y, z')
    //   i.e. the partial popcount over z-positions 0..z within row (x, y).
    // Bonus: data[x][y].ui8[7] = full popcount of row (x, y) for free.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++) {
            data[x][y].ui64 += data[x][y].ui64 << 8;
            data[x][y].ui64 += data[x][y].ui64 << 16;
            data[x][y].ui64 += data[x][y].ui64 << 32;
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Y-pass -- accumulate preceding-row popcounts.
    //
    // After the Z-pass, data[x][y].ui8[z] only accounts for voxels within
    // the same row y.  The full linear prefix requires adding the total
    // popcount of all earlier rows y' < y.
    //
    // 3a. Extract full-row popcounts: byte 7 of each post-Z-pass word holds
    //     the complete popcount of row y.  Shift right by 56 to bring it
    //     into byte 0; all other bytes become zero.
    //
    // 3b. Exclusive y-prefix scan: sum the extracted popcounts sequentially
    //     over y.  The scan is sequential in y (loop-carried dependence)
    //     but independent across x -- GCC/Clang vectorise the inner x-loop
    //     via AVX2 (4 x uint64) or AVX-512 (8 x uint64) when the loops
    //     are written with x as the inner axis.
    //
    // 3c+3d. Broadcast rowOffset[x][y] (which lives in byte 0) to all 8
    //     bytes by multiplying by kMaskPSSpread = 0x0101010101010101, then
    //     add to data[x][y].ui64.
    //
    // After: data[x][y].ui8[z] = (\sum_{y'<y} popcount(row y' of word x))
    //                           + (\sum_{z'=0..z} indicator(x, y, z'))
    //      = inclusive prefix sum within word x at position y*8+z.
    // ------------------------------------------------------------------

    // 3a: extract full-row popcounts into byte 0
    alignas(64) MaskPrefixSumQWord shifts[8][8];
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            shifts[x][y].ui64 = data[x][y].ui64 >> 56;
    }

    // 3b: exclusive y-prefix scan (sequential over y, independent over x)
    alignas(64) MaskPrefixSumQWord rowOffset[8][8];
    for (int x = 0; x < 8; x++)
        rowOffset[x][0].ui64 = 0;
    for (int y = 1; y < 8; y++)
        for (int x = 0; x < 8; x++)
            rowOffset[x][y].ui64 = rowOffset[x][y-1].ui64 + shifts[x][y-1].ui64;

    // 3c+3d: broadcast to all byte lanes and add
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            data[x][y].ui64 += rowOffset[x][y].ui64 * kMaskPSSpread;
    }

    // ------------------------------------------------------------------
    // Step 4: Zero-extend bytes to uint16_t in linear index order.
    //
    // data[x][y].ui8[z]  ->  offsets[x*64 + y*8 + z]
    //
    // Values are at most 64 (at most 64 active bits per 64-bit word),
    // so zero-extending to uint16_t is always safe.  The output is already
    // in the correct linear order -- no reordering is required.
    // Compilers vectorise this loop as vpmovzxbw over 64 contiguous bytes
    // per x-slice.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++)
            for (int z = 0; z < 8; z++)
                offsets[x*64 + y*8 + z] = data[x][y].ui8[z];

    // ------------------------------------------------------------------
    // Step 5: Add cross-word offsets decoded from prefixSum.
    //
    // The packed parameter encodes 7 exclusive cumulative popcounts at the
    // word boundaries.  Field (x-1), occupying bits 9*(x-1)..9*(x-1)+8,
    // gives the total number of active voxels in words 0..x-1:
    //
    //   xOffset[0] = 0
    //   xOffset[x] = (prefixSum >> 9*(x-1)) & 0x1FF,  x = 1..7
    //
    // Each of the 64 uint16_t entries in slice x is incremented by
    // xOffset[x].  Each slice is 128 contiguous bytes, so compilers emit
    // 4 AVX2 vpbroadcastw + vpaddw instructions per slice.
    // ------------------------------------------------------------------
    uint16_t xOffset[8];
    xOffset[0] = 0;
    for (int x = 1; x < 8; x++)
        xOffset[x] = static_cast<uint16_t>((prefixSum >> (9*(x-1))) & 0x1FFu);

    for (int x = 0; x < 8; x++) {
        uint16_t* p = offsets + x * 64;
        for (int i = 0; i < 64; i++)
            p[i] += xOffset[x];
    }
}// util::buildMaskPrefixSums

} // namespace util

} // namespace nanovdb

#endif // NANOVDB_UTIL_MASKPREFIXSUM_H_HAS_BEEN_INCLUDED
