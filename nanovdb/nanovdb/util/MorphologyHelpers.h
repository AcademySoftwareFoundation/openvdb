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

/// @brief One x-slice of a Mask<Log2Dim>: the Dim x Dim plane of slots (x, y, z) with x fixed, in the
///        mask's own layout. Slot (y, z) is bit y * Dim + z of the slice, so the Dim slots of one y are
///        contiguous bits, and every 64-bit word holds whole y's. A plain aggregate with no notion of
///        threads: Mask2D m{} is all off.
template<int Log2Dim>
struct Mask2D
{
    static_assert(Log2Dim >= 3 && Log2Dim <= 5, "every 64-bit word must hold whole y's of the slice");
    static constexpr int Dim       = 1 << Log2Dim;   // slots per axis: 8, 16, 32
    static constexpr int WordCount = Dim * Dim / 64; // 1, 4, 16

    uint64_t words[WordCount];

    /// @brief Slice @a x of @a mask, and its inverse.
    __hostdev__ static Mask2D load(const Mask<Log2Dim>& mask, int x)
    {
        Mask2D s;
        for (int j = 0; j < WordCount; ++j) s.words[j] = mask.words()[x * WordCount + j];
        return s;
    }
    __hostdev__ void store(Mask<Log2Dim>& mask, int x) const
    {
        for (int j = 0; j < WordCount; ++j) mask.words()[x * WordCount + j] = words[j];
    }

    // Whole-slice queries and updates, named as in Mask.
    __hostdev__ bool isOn() const  { uint64_t a = ~uint64_t(0); for (int j = 0; j < WordCount; ++j) a &= words[j]; return a == ~uint64_t(0); }
    __hostdev__ bool isOff() const { uint64_t a = 0; for (int j = 0; j < WordCount; ++j) a |= words[j]; return a == 0; }
    __hostdev__ void setOn()       { for (int j = 0; j < WordCount; ++j) words[j] = ~uint64_t(0); }
    __hostdev__ void setOff()      { for (int j = 0; j < WordCount; ++j) words[j] = 0; }
    __hostdev__ void toggle()      { for (int j = 0; j < WordCount; ++j) words[j] = ~words[j]; }

    __hostdev__ bool operator==(const Mask2D& o) const { bool eq = true; for (int j = 0; j < WordCount; ++j) eq &= words[j] == o.words[j]; return eq; }
    __hostdev__ bool operator!=(const Mask2D& o) const { return !(*this == o); }
    __hostdev__ Mask2D& operator&=(const Mask2D& o)    { for (int j = 0; j < WordCount; ++j) words[j] &= o.words[j]; return *this; }
    __hostdev__ Mask2D& operator|=(const Mask2D& o)    { for (int j = 0; j < WordCount; ++j) words[j] |= o.words[j]; return *this; }

    // Value-returning forms, for slices held in registers (Mask has only the compound forms).
    __hostdev__ Mask2D operator&(const Mask2D& o) const { Mask2D r = *this; return r &= o; }
    __hostdev__ Mask2D operator|(const Mask2D& o) const { Mask2D r = *this; return r |= o; }
}; // Mask2D<Log2Dim>

namespace detail {

/// @brief The bits of one y of a Mask2D<Log2Dim> word (its Dim low bits).
template<int Log2Dim>
inline constexpr uint64_t yBits = (uint64_t(1) << (1 << Log2Dim)) - 1;

/// @brief The bits of a Mask2D<Log2Dim> word, in every y of the word, whose slot z has a source under
///        a z shift by dk, i.e. 0 <= z + dk < Dim: the slots that MaskShift<dj, dk> does not zero-fill.
template<int Log2Dim, int dk>
inline constexpr uint64_t zShiftBits =
    ((yBits<Log2Dim> >> (dk > 0 ? dk : 0)) & (yBits<Log2Dim> << (dk < 0 ? -dk : 0))) // one y: z in [-dk, Dim - dk)
    * (~uint64_t(0) / yBits<Log2Dim>);                                               // repeated for every y

} // namespace detail

/// @brief out(y, z) |= in(y + dj, z + dk), or = if !combine, with zeros from outside the slice: the 2D,
///        per-thread counterpart of util::morphology::cuda::MaskShift. @a in and @a out may be the same slice.
template<int dj, int dk, bool combine = true, int Log2Dim>
__hostdev__ void MaskShift(const Mask2D<Log2Dim>& in, Mask2D<Log2Dim>& out)
{
    using SliceT = Mask2D<Log2Dim>;
    constexpr int Dim = SliceT::Dim, WordCount = SliceT::WordCount;
    static_assert(dj > -Dim && dj < Dim && dk > -Dim && dk < Dim, "the shift must stay within the slice");

    // z: shift every word, then clear the slots whose source lies outside the slice (they received bits
    // from the neighboring y instead).
    uint64_t t[WordCount];
    for (int j = 0; j < WordCount; ++j) {
        if constexpr (dk > 0)      t[j] = (in.words[j] >>  dk) & detail::zShiftBits<Log2Dim, dk>;
        else if constexpr (dk < 0) t[j] = (in.words[j] << -dk) & detail::zShiftBits<Log2Dim, dk>;
        else                       t[j] = in.words[j];
    }

    // y: whole y's, so the slice moves as one (64 * WordCount)-bit integer by |dj| * Dim bits,
    // i.e. Q whole words plus R bits carried in from the next word over.
    constexpr int Q = (dj < 0 ? -dj : dj) * Dim / 64;
    constexpr int R = (dj < 0 ? -dj : dj) * Dim % 64;
    SliceT r;
    for (int i = 0; i < WordCount; ++i) {
        uint64_t w = 0;
        if constexpr (dj >= 0) { // toward lower bits: out bit b = t bit b + |dj| * Dim
            if (i + Q < WordCount) w = t[i + Q] >> R;
            if constexpr (R != 0) { if (i + Q + 1 < WordCount) w |= t[i + Q + 1] << (64 - R); }
        } else {                 // toward higher bits: out bit b = t bit b - |dj| * Dim
            if (i - Q >= 0) w = t[i - Q] << R;
            if constexpr (R != 0) { if (i - Q - 1 >= 0) w |= t[i - Q - 1] >> (64 - R); }
        }
        r.words[i] = w;
    }
    for (int i = 0; i < WordCount; ++i) out.words[i] = combine ? (out.words[i] | r.words[i]) : r.words[i];
}

} // namespace morphology

} // namespace nanovdb::util

#if defined(__CUDACC__)
#include <nanovdb/util/cuda/MorphologyHelpers.cuh>
#endif // defined(__CUDACC__)

#endif // NANOVDB_UTIL_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED

