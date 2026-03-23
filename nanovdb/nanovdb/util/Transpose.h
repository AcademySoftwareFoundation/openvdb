// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/util/Transpose.h

    \brief  Bit- and byte-level 8×8 matrix transpose utilities.

    \details Two complementary operations are provided:

    - transposeBits8x8: transposes the 8×8 bit matrix packed inside a single
      uint64_t word (Knuth 3-round XOR/shift/mask algorithm).  Pure C++17,
      suitable for __hostdev__ use.

    - transposeBytes8x8: transposes an 8×8 matrix whose elements are individual
      bytes, stored as 8 consecutive uint64_t words (one row per word).  A 3-round
      butterfly using __builtin_shuffle (GCC) or __builtin_shufflevector (Clang)
      is used when NANOVDB_USE_INTRINSICS is defined and vector extensions are
      available; otherwise a portable scalar fallback is used.

    Together these functions implement the full bit-level permutation of a NanoVDB
    Mask<3> from its native x-major layout (word=x, byte=y, bit=z) to the y-major
    layout (word=y, byte=z, bit=x) needed by the bit-parallel prefix-sum algorithm
    in DecodeInverseMapsCPUPlan.md §13–14.
*/

#ifndef NANOVDB_UTIL_TRANSPOSE_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_TRANSPOSE_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Util.h>

namespace nanovdb {

namespace util {

// ---------------------------> transposeBits8x8 <-----------------------------------

/// @brief Transposes the 8×8 bit matrix packed in a single 64-bit word.
///
/// @details Bit (r, c) of the input (row r = bits 8r..8r+7, column c = bit c within
/// each byte) becomes bit (c, r) of the output.  Uses the Knuth 3-round XOR/shift/mask
/// algorithm; each round swaps 2^k × 2^k sub-blocks at stride 2^k for k = 0, 1, 2.
///
/// No intrinsics required: the three rounds compile to vpsrlq/vpxor/vpand/vpsllq
/// under #pragma omp simd, or to equivalent scalar shifts on any platform.
///
/// @param v   8×8 bit matrix packed in a uint64_t (row-major, LSB = row 0, col 0)
/// @return    transposed matrix, same packing
__hostdev__ inline uint64_t transposeBits8x8(uint64_t v)
{
    // Round 1: swap 1×1 blocks at stride 1 within 2×2 tiles
    uint64_t t = (v ^ (v >> 7)) & UINT64_C(0x00aa00aa00aa00aa);
    v ^= t ^ (t << 7);
    // Round 2: swap 2×2 blocks at stride 2 within 4×4 tiles
    t  = (v ^ (v >> 14)) & UINT64_C(0x0000cccc0000cccc);
    v ^= t ^ (t << 14);
    // Round 3: swap 4×4 blocks at stride 4 within 8×8 tiles
    t  = (v ^ (v >> 28)) & UINT64_C(0x00000000f0f0f0f0);
    v ^= t ^ (t << 28);
    return v;
}// util::transposeBits8x8

// ---------------------------> transposeBytes8x8 <----------------------------------

/// @brief Transposes an 8×8 matrix of bytes stored as 8 consecutive uint64_t words.
///
/// @details src[x] is one row of the matrix: byte y of src[x] = element (x, y).
/// After the call, dst[y] is a row of the transposed matrix: byte x of dst[y] =
/// element (x, y).  In other words, dst[y * 8 + x] == src[x * 8 + y] (viewing
/// the 64 bytes as a flat array).
///
/// A 3-round byte-interleave butterfly is used when vector extensions are available
/// (NANOVDB_USE_INTRINSICS on GCC or Clang, CPU only).  Otherwise a portable
/// scalar nested loop is used; GCC auto-vectorizes this to vpunpcklbw at -O3.
///
/// @param src  pointer to 8 uint64_t words (64 bytes, one row per word); may be
///             the same pointer as dst (in-place is safe if dst == src)
/// @param dst  pointer to 8 uint64_t words that receive the transposed result
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ inline void transposeBytes8x8(const uint64_t* __restrict__ src,
                                          uint64_t* __restrict__       dst)
{
// Vector-extension SIMD path: CPU-only (GCC or Clang), not available in CUDA/HIP.
#if !defined(__CUDA_ARCH__) && !defined(__HIP__) && defined(NANOVDB_USE_INTRINSICS) \
    && (defined(__GNUC__) || defined(__clang__))

    // 16-byte vector type: two uint64_t words viewed as 16 individual bytes.
    using u8x16 = uint8_t __attribute__((vector_size(16)));

    // Load 8 input words as four 128-bit (16-byte) registers, two words each.
    u8x16 v01, v23, v45, v67;
    __builtin_memcpy(&v01, src + 0, 16);
    __builtin_memcpy(&v23, src + 2, 16);
    __builtin_memcpy(&v45, src + 4, 16);
    __builtin_memcpy(&v67, src + 6, 16);

    // Round 1: interleave bytes of the two words within each 16-byte register.
    // Produces: [byte0_w0, byte0_w1, byte1_w0, byte1_w1, ...] per register pair.
    // GCC emits vpshufb for this pattern; Clang emits vpunpcklbw + vpunpckhbw.
#if defined(__clang__)
    u8x16 t01 = __builtin_shufflevector(v01,v01, 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15);
    u8x16 t23 = __builtin_shufflevector(v23,v23, 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15);
    u8x16 t45 = __builtin_shufflevector(v45,v45, 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15);
    u8x16 t67 = __builtin_shufflevector(v67,v67, 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15);
#else // GCC: __builtin_shuffle with same-type mask vector
    u8x16 t01 = __builtin_shuffle(v01,v01,(u8x16){ 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15});
    u8x16 t23 = __builtin_shuffle(v23,v23,(u8x16){ 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15});
    u8x16 t45 = __builtin_shuffle(v45,v45,(u8x16){ 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15});
    u8x16 t67 = __builtin_shuffle(v67,v67,(u8x16){ 0, 8, 1, 9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15});
#endif

    // Round 2: interleave 2-byte groups across register pairs.
    // Indices 0–15 select from the first argument, 16–31 from the second.
#if defined(__clang__)
    u8x16 q02lo = __builtin_shufflevector(t01,t23, 0, 1,16,17,  2, 3,18,19,  4, 5,20,21,  6, 7,22,23);
    u8x16 q02hi = __builtin_shufflevector(t01,t23, 8, 9,24,25, 10,11,26,27, 12,13,28,29, 14,15,30,31);
    u8x16 q46lo = __builtin_shufflevector(t45,t67, 0, 1,16,17,  2, 3,18,19,  4, 5,20,21,  6, 7,22,23);
    u8x16 q46hi = __builtin_shufflevector(t45,t67, 8, 9,24,25, 10,11,26,27, 12,13,28,29, 14,15,30,31);
#else
    u8x16 q02lo = __builtin_shuffle(t01,t23,(u8x16){ 0, 1,16,17,  2, 3,18,19,  4, 5,20,21,  6, 7,22,23});
    u8x16 q02hi = __builtin_shuffle(t01,t23,(u8x16){ 8, 9,24,25, 10,11,26,27, 12,13,28,29, 14,15,30,31});
    u8x16 q46lo = __builtin_shuffle(t45,t67,(u8x16){ 0, 1,16,17,  2, 3,18,19,  4, 5,20,21,  6, 7,22,23});
    u8x16 q46hi = __builtin_shuffle(t45,t67,(u8x16){ 8, 9,24,25, 10,11,26,27, 12,13,28,29, 14,15,30,31});
#endif

    // Round 3: interleave 4-byte groups across the half-results.
#if defined(__clang__)
    u8x16 r01 = __builtin_shufflevector(q02lo,q46lo, 0, 1, 2, 3,16,17,18,19,  4, 5, 6, 7,20,21,22,23);
    u8x16 r23 = __builtin_shufflevector(q02lo,q46lo, 8, 9,10,11,24,25,26,27, 12,13,14,15,28,29,30,31);
    u8x16 r45 = __builtin_shufflevector(q02hi,q46hi, 0, 1, 2, 3,16,17,18,19,  4, 5, 6, 7,20,21,22,23);
    u8x16 r67 = __builtin_shufflevector(q02hi,q46hi, 8, 9,10,11,24,25,26,27, 12,13,14,15,28,29,30,31);
#else
    u8x16 r01 = __builtin_shuffle(q02lo,q46lo,(u8x16){ 0, 1, 2, 3,16,17,18,19,  4, 5, 6, 7,20,21,22,23});
    u8x16 r23 = __builtin_shuffle(q02lo,q46lo,(u8x16){ 8, 9,10,11,24,25,26,27, 12,13,14,15,28,29,30,31});
    u8x16 r45 = __builtin_shuffle(q02hi,q46hi,(u8x16){ 0, 1, 2, 3,16,17,18,19,  4, 5, 6, 7,20,21,22,23});
    u8x16 r67 = __builtin_shuffle(q02hi,q46hi,(u8x16){ 8, 9,10,11,24,25,26,27, 12,13,14,15,28,29,30,31});
#endif

    __builtin_memcpy(dst + 0, &r01, 16);
    __builtin_memcpy(dst + 2, &r23, 16);
    __builtin_memcpy(dst + 4, &r45, 16);
    __builtin_memcpy(dst + 6, &r67, 16);

#else // scalar fallback: portable, __hostdev__-safe, auto-vectorizes under -O3

    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    uint8_t*       d = reinterpret_cast<uint8_t*>(dst);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            d[i * 8 + j] = s[j * 8 + i];

#endif
}// util::transposeBytes8x8

} // namespace util

} // namespace nanovdb

#endif // NANOVDB_UTIL_TRANSPOSE_H_HAS_BEEN_INCLUDED
