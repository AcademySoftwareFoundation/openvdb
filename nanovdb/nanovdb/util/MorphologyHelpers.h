// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file MorphologyHelpers.h

    \author Efty Sifakis

    \date March 17, 2025

    \brief This file implements helper methods used in morphology operations

*/

#ifndef NANOVDB_UTIL_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

#include <type_traits>

namespace nanovdb::tools::morphology {

enum NearestNeighbors { NN_FACE = 6, NN_FACE_EDGE = 18, NN_FACE_EDGE_VERTEX = 26 };

} // namespace nanovdb::tools::morphology

namespace nanovdb::util {

/// @brief Unmasked fixed-distance shuffle-down on a flat array: data[j] = data[j+Shift]
///        for j in [0, N-Shift), with the trailing Shift slots zero-filled.
///
/// Unmasked counterpart of shuffleDownMask (VoxelBlockManager.h); the name follows the
/// CUDA __shfl_down_sync convention (fixed-distance gather from higher indices). Used to
/// express the cross-word axis shifts of the host MaskShift in a SIMD-friendly form.
///
/// In-place safe and SIMD-vectorizable: each write data[j] reads data[j+Shift] with
/// j < j+Shift, so within and across vector chunks every source is read before its slot
/// is overwritten (chunk k writes [kW, kW+W), reads [kW+Shift, ...) which never aliases an
/// already-written lower chunk). This is why the shuffleDownMask family only shifts *down*.
///
/// @tparam N      Length of the data array.
/// @tparam Shift  Number of positions to shift; must satisfy 0 < Shift <= N.
/// @tparam DataT  Element type (any unsigned integer type).
template <int N, int Shift, typename DataT>
inline void shuffleDown(DataT* NANOVDB_RESTRICT data)
{
    static_assert(Shift > 0 && Shift <= N, "Shift must satisfy 0 < Shift <= N");
    static_assert(std::is_unsigned_v<DataT>, "DataT must be an unsigned integer type");
    #pragma omp simd
    for (int j = 0; j < N - Shift; j++)
        data[j] = data[j + Shift];
    for (int j = N - Shift; j < N; j++)
        data[j] = DataT{0};
}

/// @brief Unmasked fixed-distance shuffle-up on a flat array: data[j] = data[j-Shift]
///        for j in [Shift, N), with the leading Shift slots zero-filled.
///
/// Mirror of shuffleDown for the opposite shift direction. In-place safe only under
/// *descending* iteration (each write data[j] reads data[j-Shift] with j-Shift < j, so the
/// source is read before it is overwritten when j decreases). Unlike shuffleDown this is
/// NOT marked `omp simd`: an ascending vectorized pass would let a later chunk read slots an
/// earlier chunk already overwrote. If a SIMD shuffle-up is ever needed, write out-of-place
/// into a separate destination buffer instead.
///
/// @tparam N      Length of the data array.
/// @tparam Shift  Number of positions to shift; must satisfy 0 < Shift <= N.
/// @tparam DataT  Element type (any unsigned integer type).
template <int N, int Shift, typename DataT>
inline void shuffleUp(DataT* NANOVDB_RESTRICT data)
{
    static_assert(Shift > 0 && Shift <= N, "Shift must satisfy 0 < Shift <= N");
    static_assert(std::is_unsigned_v<DataT>, "DataT must be an unsigned integer type");
    for (int j = N - 1; j >= Shift; j--)
        data[j] = data[j - Shift];
    for (int j = 0; j < Shift; j++)
        data[j] = DataT{0};
}

namespace morphology {

template<int di, int dj, int dk>
struct NearestNeighborBitMask {
    static_assert( (di>=-1) && (di<=1) && (dj>=-1) && (dj<=1) && (dk>=-1) && (dk<=1) );
    static constexpr uint32_t value = 1u << (di+1)*9+(dj+1)*3+dk+1;
};

template<tools::morphology::NearestNeighbors nnType>
__hostdev__
uint32_t neighborMaskStencil(const nanovdb::Mask<3>& mask)
{
    using tools::morphology::NN_FACE;
    using tools::morphology::NN_FACE_EDGE;
    using tools::morphology::NN_FACE_EDGE_VERTEX;

    uint32_t result = 0;
    auto words = mask.words();
    uint64_t allWordsOr = 0;
    for (int i = 0; i < 8; i++)
        allWordsOr |= words[i];
    // Center
    if ( allWordsOr )                            result |= NearestNeighborBitMask< 0, 0, 0>::value;
    // Neighbors across faces
    if constexpr (nnType == NN_FACE || nnType == NN_FACE_EDGE || nnType == NN_FACE_EDGE_VERTEX) {
        if ( words[0] )                          result |= NearestNeighborBitMask<-1, 0, 0>::value;
        if ( words[7] )                          result |= NearestNeighborBitMask< 1, 0, 0>::value;
        if ( allWordsOr & UINT64_C(0x00000000000000ff) ) result |= NearestNeighborBitMask< 0,-1, 0>::value;
        if ( allWordsOr & UINT64_C(0xff00000000000000) ) result |= NearestNeighborBitMask< 0, 1, 0>::value;
        if ( allWordsOr & UINT64_C(0x0101010101010101) ) result |= NearestNeighborBitMask< 0, 0,-1>::value;
        if ( allWordsOr & UINT64_C(0x8080808080808080) ) result |= NearestNeighborBitMask< 0, 0, 1>::value; }
    // Neighbors across edges
    if constexpr (nnType == NN_FACE_EDGE || nnType == NN_FACE_EDGE_VERTEX) {
        if ( words[0]   & UINT64_C(0x00000000000000ff) ) result |= NearestNeighborBitMask<-1,-1, 0>::value;
        if ( words[0]   & UINT64_C(0xff00000000000000) ) result |= NearestNeighborBitMask<-1, 1, 0>::value;
        if ( words[0]   & UINT64_C(0x0101010101010101) ) result |= NearestNeighborBitMask<-1, 0,-1>::value;
        if ( words[0]   & UINT64_C(0x8080808080808080) ) result |= NearestNeighborBitMask<-1, 0, 1>::value;
        if ( allWordsOr & UINT64_C(0x0000000000000001) ) result |= NearestNeighborBitMask< 0,-1,-1>::value;
        if ( allWordsOr & UINT64_C(0x0000000000000080) ) result |= NearestNeighborBitMask< 0,-1, 1>::value;
        if ( allWordsOr & UINT64_C(0x0100000000000000) ) result |= NearestNeighborBitMask< 0, 1,-1>::value;
        if ( allWordsOr & UINT64_C(0x8000000000000000) ) result |= NearestNeighborBitMask< 0, 1, 1>::value;
        if ( words[7]   & UINT64_C(0x00000000000000ff) ) result |= NearestNeighborBitMask< 1,-1, 0>::value;
        if ( words[7]   & UINT64_C(0xff00000000000000) ) result |= NearestNeighborBitMask< 1, 1, 0>::value;
        if ( words[7]   & UINT64_C(0x0101010101010101) ) result |= NearestNeighborBitMask< 1, 0,-1>::value;
        if ( words[7]   & UINT64_C(0x8080808080808080) ) result |= NearestNeighborBitMask< 1, 0, 1>::value; }
        // Neighbors across vertices
    if constexpr (nnType == NN_FACE_EDGE_VERTEX) {
        if ( words[0]   & UINT64_C(0x0000000000000001) ) result |= NearestNeighborBitMask<-1,-1,-1>::value;
        if ( words[0]   & UINT64_C(0x0000000000000080) ) result |= NearestNeighborBitMask<-1,-1, 1>::value;
        if ( words[0]   & UINT64_C(0x0100000000000000) ) result |= NearestNeighborBitMask<-1, 1,-1>::value;
        if ( words[0]   & UINT64_C(0x8000000000000000) ) result |= NearestNeighborBitMask<-1, 1, 1>::value;
        if ( words[7]   & UINT64_C(0x0000000000000001) ) result |= NearestNeighborBitMask< 1,-1,-1>::value;
        if ( words[7]   & UINT64_C(0x0000000000000080) ) result |= NearestNeighborBitMask< 1,-1, 1>::value;
        if ( words[7]   & UINT64_C(0x0100000000000000) ) result |= NearestNeighborBitMask< 1, 1,-1>::value;
        if ( words[7]   & UINT64_C(0x8000000000000000) ) result |= NearestNeighborBitMask< 1, 1, 1>::value; }
    return result;
}

__hostdev__
inline Coord::ValueType
coarsenComponent(const Coord::ValueType n)
{return (n>=0) ? (n>>1) : -((-n+1)>>1);} // Round down for negative integers

__hostdev__
inline Coord
coarsenCoord(const Coord& coord)
{
    Coord result;
    result[0] = coarsenComponent(coord[0]);
    result[1] = coarsenComponent(coord[1]);
    result[2] = coarsenComponent(coord[2]);
    return result;
}

__hostdev__
inline Coord::ValueType
refineComponent(const Coord::ValueType n)
{return (n>=0) ? (n<<1) : -((-n)<<1);}

__hostdev__
inline Coord
refineCoord(const Coord& coord)
{
    Coord result;
    result[0] = refineComponent(coord[0]);
    result[1] = refineComponent(coord[1]);
    result[2] = refineComponent(coord[2]);
    return result;
}

/// @brief Host counterpart of the warp-cooperative CUDA MaskShift (util/cuda/MorphologyHelpers.cuh).
///        Shifts a Mask<4> by (di,dj,dk) within the 16x16x16 node: out(x,y,z) (|)= in(x+di, y+dj, z+dk)
///        for source coordinates inside [0,16), else zero. With combine=true the shifted mask is OR'd
///        into outMaskQ; with combine=false it overwrites.
///
/// Layout: a Mask<4> bit is n = (x<<8)|(y<<4)|z, so word index = x*4 + (y>>2) and the in-word bit is
/// 16*(y&3) + z (four 16-bit z-subwords per word). The three axes decompose cleanly on the host, where
/// the whole 64-word mask is in contiguous memory (no warp shuffles needed):
///   - X: words for fixed (y>>2) are strided by 4 with x, so out(x)=in(x+di) is a flat shuffle by 4*di
///        over the 64 words (zero-filled at the x boundary) -- util::shuffleDown/shuffleUp.
///   - Y: y = 4*(y>>2) + (y&3) splits across the word index (djj=dj>>2) and the subword (mjj=dj&3); a
///        per-x-group subword gather with carry between adjacent words reproduces out(y)=in(y+dj).
///   - Z: intra-16-bit-subword shift, masked to keep z within [0,16) (no cross-subword leak).
///
/// @note Host-only (uses util::shuffleDown's omp-simd loop). The intent matches the device MaskShift
///       bit-for-bit; the two are validated against each other and a brute-force coordinate oracle.
template<int di, int dj, int dk, bool combine = true>
inline void MaskShift(const nanovdb::Mask<4>& inMaskQ, nanovdb::Mask<4>& outMaskQ)
{
    static_assert( (di<=15) && (di>=-15) && (dj<=15) && (dj>=-15) && (dk<=15) && (dk>=-15) );
    constexpr int N = nanovdb::Mask<4>::WORD_COUNT; // 64
    uint64_t w[N];
    for (int i = 0; i < N; ++i) w[i] = inMaskQ.words()[i];

    // X-axis: out(x)=in(x+di). word = x*4 + (y>>2), so a flat shuffle by 4*di over the 64 words.
    if constexpr (di > 0) nanovdb::util::shuffleDown<N,  4*di >(w);
    else if constexpr (di < 0) nanovdb::util::shuffleUp <N, -4*di >(w);

    // Y-axis: out(y)=in(y+dj), y = 4*(y>>2) + (y&3). djj shifts whole words within the x-group of 4,
    // mjj shifts subwords (16 bits each) with a carry pulled from the next word.
    if constexpr (dj != 0) {
        constexpr int djj = dj >> 2;   // arithmetic shift; may be negative
        constexpr int mjj = dj & 3;    // in [0,3]
        uint64_t y[N];
        for (int i = 0; i < N; ++i) {
            const int jjw  = i & 3;         // (y>>2) slot of this word within its x-group
            const int base = i & ~3;        // first word of this x-group (x*4)
            const int kA = jjw + djj;
            const int kB = jjw + djj + 1;   // carry word
            const uint64_t termA = (kA >= 0 && kA <= 3) ? w[base + kA] : uint64_t(0);
            const uint64_t termB = (kB >= 0 && kB <= 3) ? w[base + kB] : uint64_t(0);
            uint64_t out = termA >> (16 * mjj);
            if constexpr (mjj > 0) out |= termB << (16 * (4 - mjj));
            y[i] = out;
        }
        for (int i = 0; i < N; ++i) w[i] = y[i];
    }

    // Z-axis: out(z)=in(z+dk), within each 16-bit subword; mask keeps z in [0,16) (no cross-subword leak).
    if constexpr (dk > 0) {
        constexpr uint64_t s16 = uint64_t(uint16_t(0xffff) >> dk);          // bits [0, 16-dk)
        constexpr uint64_t m = s16 | (s16<<16) | (s16<<32) | (s16<<48);
        for (int i = 0; i < N; ++i) w[i] = (w[i] >> dk) & m;
    } else if constexpr (dk < 0) {
        constexpr int a = -dk;
        constexpr uint64_t s16 = uint64_t(uint16_t(0xffff << a));           // bits [a, 16)
        constexpr uint64_t m = s16 | (s16<<16) | (s16<<32) | (s16<<48);
        for (int i = 0; i < N; ++i) w[i] = (w[i] << a) & m;
    }

    if constexpr (combine)
        for (int i = 0; i < N; ++i) outMaskQ.words()[i] |= w[i];
    else
        for (int i = 0; i < N; ++i) outMaskQ.words()[i]  = w[i];
}

} // namespace morphology

} // namespace nanovdb::util

#if defined(__CUDACC__)
#include <nanovdb/util/cuda/MorphologyHelpers.cuh>
#endif // defined(__CUDACC__)

#endif // NANOVDB_UTIL_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED

