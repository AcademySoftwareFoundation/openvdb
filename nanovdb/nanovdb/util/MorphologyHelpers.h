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

namespace nanovdb::tools::morphology {

enum NearestNeighbors { NN_FACE = 6, NN_FACE_EDGE = 18, NN_FACE_EDGE_VERTEX = 26 };

} // namespace nanovdb::tools::morphology

namespace nanovdb::util {

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

/// @brief One x-slice of a Mask<LOG2DIM>: the DIM x DIM plane of slots (x, y, z) with x fixed, in the
///        mask's own layout. Slot (y, z) is bit y * DIM + z of the slice, so the DIM slots of one y are
///        contiguous bits, and every 64-bit word holds whole y's. A plain aggregate: Mask2D m{} is all off.
template<int LOG2DIM>
struct Mask2D
{
    static_assert(LOG2DIM >= 3 && LOG2DIM <= 5,
                  "Mask2D supports only LOG2DIM 3, 4 and 5, the sizes of NanoVDB's leaf, lower and upper node masks");
    static constexpr int DIM        = 1 << LOG2DIM; // slots per axis: 8, 16, 32
    static constexpr int SIZE       = DIM * DIM;    // slots in the slice: 64, 256, 1024
    static constexpr int WORD_COUNT = SIZE >> 6;    // 64-bit words: 1, 4, 16

    uint64_t mWords[WORD_COUNT];

    /// @brief Slice @a x of @a mask, and its inverse.
    __hostdev__ static Mask2D load(const Mask<LOG2DIM>& mask, int x)
    {
        Mask2D s;
        for (int j = 0; j < WORD_COUNT; ++j) s.mWords[j] = mask.words()[x * WORD_COUNT + j];
        return s;
    }
    __hostdev__ void store(Mask<LOG2DIM>& mask, int x) const
    {
        for (int j = 0; j < WORD_COUNT; ++j) mask.words()[x * WORD_COUNT + j] = mWords[j];
    }

    // Whole-slice queries and updates, named as in Mask.
    __hostdev__ bool isOn() const  { uint64_t a = ~uint64_t(0); for (int j = 0; j < WORD_COUNT; ++j) a &= mWords[j]; return a == ~uint64_t(0); }
    __hostdev__ bool isOff() const { uint64_t a = 0; for (int j = 0; j < WORD_COUNT; ++j) a |= mWords[j]; return a == 0; }
    __hostdev__ void setOn()       { for (int j = 0; j < WORD_COUNT; ++j) mWords[j] = ~uint64_t(0); }
    __hostdev__ void setOff()      { for (int j = 0; j < WORD_COUNT; ++j) mWords[j] = 0; }
    __hostdev__ void toggle()      { for (int j = 0; j < WORD_COUNT; ++j) mWords[j] = ~mWords[j]; }

    __hostdev__ bool operator==(const Mask2D& o) const { bool eq = true; for (int j = 0; j < WORD_COUNT; ++j) eq &= mWords[j] == o.mWords[j]; return eq; }
    __hostdev__ bool operator!=(const Mask2D& o) const { return !(*this == o); }
    __hostdev__ Mask2D& operator&=(const Mask2D& o)    { for (int j = 0; j < WORD_COUNT; ++j) mWords[j] &= o.mWords[j]; return *this; }
    __hostdev__ Mask2D& operator|=(const Mask2D& o)    { for (int j = 0; j < WORD_COUNT; ++j) mWords[j] |= o.mWords[j]; return *this; }

    // Value-returning forms, for slices held in registers (Mask has only the compound forms).
    __hostdev__ Mask2D operator&(const Mask2D& o) const { Mask2D r = *this; return r &= o; }
    __hostdev__ Mask2D operator|(const Mask2D& o) const { Mask2D r = *this; return r |= o; }
}; // Mask2D<LOG2DIM>

namespace detail {

/// @brief Take the Mask2D<LOG2DIM> whose on slots are those with y == 0, and keep just its first word:
///        the DIM low bits (0xFF, 0xFFFF or 0xFFFFFFFF).
template<int LOG2DIM>
inline constexpr uint64_t firstWordOfY0 = (uint64_t(1) << (1 << LOG2DIM)) - 1;

/// @brief MaskShift<0, dk, false> of the all-on Mask2D<LOG2DIM>, restricted to its first word.
template<int LOG2DIM, int dk>
inline constexpr uint64_t zShiftBits =
    ((firstWordOfY0<LOG2DIM> >> (dk > 0 ? dk : 0)) & (firstWordOfY0<LOG2DIM> << (dk < 0 ? -dk : 0))) // one y: z in [-dk, DIM - dk)
    * (~uint64_t(0) / firstWordOfY0<LOG2DIM>);                                                       // copied into every y

} // namespace detail

/// @brief out(y, z) |= in(y + dj, z + dk), or = if !combine, with zeros from outside the slice: the 2D,
///        per-thread counterpart of util::morphology::cuda::MaskShift. @a in and @a out may be the same slice.
template<int dj, int dk, bool combine = true, int LOG2DIM>
__hostdev__ void MaskShift(const Mask2D<LOG2DIM>& in, Mask2D<LOG2DIM>& out)
{
    using SliceT = Mask2D<LOG2DIM>;
    constexpr int DIM = SliceT::DIM, WORD_COUNT = SliceT::WORD_COUNT;
    static_assert(dj > -DIM && dj < DIM && dk > -DIM && dk < DIM, "the shift must stay within the slice");

    // z: shift every word, then clear the slots whose source lies outside the slice (they received bits
    // from the neighboring y instead).
    uint64_t t[WORD_COUNT];
    for (int j = 0; j < WORD_COUNT; ++j) {
        if constexpr (dk > 0)      t[j] = (in.mWords[j] >>  dk) & detail::zShiftBits<LOG2DIM, dk>;
        else if constexpr (dk < 0) t[j] = (in.mWords[j] << -dk) & detail::zShiftBits<LOG2DIM, dk>;
        else                       t[j] = in.mWords[j];
    }

    // y: whole y's, so the slice moves as one (64 * WORD_COUNT)-bit integer by |dj| * DIM bits,
    // i.e. Q whole words plus R bits carried in from the next word over.
    constexpr int Q = (dj < 0 ? -dj : dj) * DIM / 64;
    constexpr int R = (dj < 0 ? -dj : dj) * DIM % 64;
    SliceT r;
    for (int i = 0; i < WORD_COUNT; ++i) {
        uint64_t w = 0;
        if constexpr (dj >= 0) { // toward lower bits: out bit b = t bit b + |dj| * DIM
            if (i + Q < WORD_COUNT) w = t[i + Q] >> R;
            if constexpr (R != 0) { if (i + Q + 1 < WORD_COUNT) w |= t[i + Q + 1] << (64 - R); }
        } else {                 // toward higher bits: out bit b = t bit b - |dj| * DIM
            if (i - Q >= 0) w = t[i - Q] << R;
            if constexpr (R != 0) { if (i - Q - 1 >= 0) w |= t[i - Q - 1] >> (64 - R); }
        }
        r.mWords[i] = w;
    }
    for (int i = 0; i < WORD_COUNT; ++i) out.mWords[i] = combine ? (out.mWords[i] | r.mWords[i]) : r.mWords[i];
}

} // namespace morphology

} // namespace nanovdb::util

#if defined(__CUDACC__)
#include <nanovdb/util/cuda/MorphologyHelpers.cuh>
#endif // defined(__CUDACC__)

#endif // NANOVDB_UTIL_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED

