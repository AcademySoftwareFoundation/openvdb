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

/// @brief Axis-aligned box [lo, hi] of voxel offsets for a box dilation (a Minkowski sum): the
///        dilated topology holds ijk + o for every active voxel ijk and every offset o with
///        lo <= o <= hi componentwise. DilationBox(Coord(-1), Coord(1)) is the 26-neighbor
///        dilation (NN_FACE_EDGE_VERTEX). A box need not contain the origin, and a box with
///        lo > hi on any axis is empty, so its dilation is empty too.
///
///        The dilation of one source leaf is split by target leaf: target d is the leaf at the
///        source origin + LeafDim * d. leafOffsetCount() and leafOffset(i) enumerate the targets
///        the dilation of a leaf can reach, along each axis the leaves from the one holding a
///        local coordinate 0 + lo to the one holding LeafDim - 1 + hi, and dilateLeafMask
///        computes the voxels one source leaf contributes to one of them. A box within
///        [-LeafDim, LeafDim] reaches at most the source leaf and its 26 neighbors. As with any
///        Coord arithmetic, the dilated coordinates, ijk + lo and ijk + hi, must fit in int32.
struct DilationBox
{
    /// @brief Voxels along each axis of a leaf node
    static constexpr int LeafDim = 8;

    Coord lo, hi;

    __hostdev__ DilationBox(const Coord &lo, const Coord &hi) : lo(lo), hi(hi) {}

    /// @brief Whether lo > hi on any axis, so that the dilation is empty
    __hostdev__ bool empty() const { return lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]; }

    /// @brief Bounding box of the dilation of the voxels in @a bbox; empty if either is empty.
    ///        bbox.min() + lo and bbox.max() + hi must fit in int32.
    __hostdev__ CoordBBox dilate(const CoordBBox &bbox) const
    {
        return bbox.empty() || this->empty() ? CoordBBox() : CoordBBox(bbox.min() + lo, bbox.max() + hi);
    }

    /// @brief Number of target leaves the dilation of one source leaf can reach; 0 if empty()
    __hostdev__ uint64_t leafOffsetCount() const
    {
        if (this->empty()) return 0;
        uint64_t n = 1;
        for (int a = 0; a < 3; ++a) n *= axisCount(a);
        return n;
    }

    /// @brief Target @a i < leafOffsetCount(), with z varying fastest
    __hostdev__ Coord leafOffset(uint64_t i) const
    {
        NANOVDB_ASSERT(i < leafOffsetCount());
        Coord d;
        for (int a = 2; a >= 0; --a) {
            d[a] = Coord::ValueType(firstTarget(a) + int64_t(i % axisCount(a)));
            i /= axisCount(a);
        }
        return d;
    }

    /// @brief Sets @a dst to the voxels, in the local coordinates of the target leaf at offset
    ///        @a leafOffset, that the dilation of the leaf mask @a src turns on there, and
    ///        returns whether there are any
    __hostdev__ bool dilateLeafMask(const Mask<3> &src, const Coord &leafOffset, Mask<3> &dst) const
    {
        constexpr int W = int(Mask<3>::WORD_COUNT);
        static_assert(W == LeafDim && Mask<3>::SIZE == LeafDim * LeafDim * LeafDim,
                      "the shifts below assume 8^3 leaves of eight 64-bit words");
        // Word w of a leaf mask holds local x = w, and its byte y holds local z = 0..7 as bits.
        // The Minkowski sum factors by axis: dilate along z (bits), then y (bytes), then x
        // (words), each time OR-ing the shifts s = o - LeafDim * d, for the box offsets o, that
        // keep voxels in the target leaf, which are those with |s| < LeafDim.
        constexpr uint64_t byteLow = UINT64_C(0x0101010101010101);// bit 0 of every byte
        const uint64_t *in = src.words();
        uint64_t wz[W], wy[W];
        int sLo, sHi;
        shiftRange(2, leafOffset, sLo, sHi);
        for (int w = 0; w < W; ++w) {
            uint64_t acc = 0;
            for (int s = sLo; s <= sHi; ++s) {// z moves by s within each byte
                if (s >= 0) acc |= (in[w] & (byteLow * (0xFFu >> s))) << s;
                else acc |= (in[w] >> -s) & (byteLow * (0xFFu >> -s));
            }
            wz[w] = acc;
        }
        shiftRange(1, leafOffset, sLo, sHi);
        for (int w = 0; w < W; ++w) {
            uint64_t acc = 0;
            for (int s = sLo; s <= sHi; ++s)// y moves by s whole bytes
                acc |= s >= 0 ? wz[w] << (LeafDim * s) : wz[w] >> (-LeafDim * s);
            wy[w] = acc;
        }
        shiftRange(0, leafOffset, sLo, sHi);
        uint64_t any = 0;
        uint64_t *out = dst.words();
        for (int w = 0; w < W; ++w) {
            uint64_t acc = 0;
            for (int s = sLo; s <= sHi; ++s) {// x moves by s whole words
                const int from = w - s;
                if (from >= 0 && from < W) acc |= wy[from];
            }
            out[w] = acc;
            any |= acc;
        }
        return any != 0;
    }

private:
    /// @brief Floor of @a v / LeafDim, in 64 bits so that no int32 input overflows
    __hostdev__ static int64_t floorDiv(int64_t v) { return v >= 0 ? v / LeafDim : -((-v + LeafDim - 1) / LeafDim); }

    /// @brief Lowest target along axis @a a: the leaf holding local coordinate lo[a]
    __hostdev__ int64_t firstTarget(int a) const { return floorDiv(lo[a]); }

    /// @brief Targets along axis @a a, from the leaf holding local coordinate lo[a] to the
    ///        leaf holding LeafDim - 1 + hi[a]
    __hostdev__ uint64_t axisCount(int a) const { return uint64_t(floorDiv(int64_t(LeafDim - 1) + hi[a]) - firstTarget(a) + 1); }

    /// @brief The shifts o - LeafDim * d along axis @a a, for the offsets o of the box, that
    ///        keep voxels in target @a d; sLo > sHi when there are none
    __hostdev__ void shiftRange(int a, const Coord &d, int &sLo, int &sHi) const
    {
        const int64_t first = int64_t(lo[a]) - int64_t(LeafDim) * d[a], last = int64_t(hi[a]) - int64_t(LeafDim) * d[a];
        sLo = int(first < 1 - LeafDim ? 1 - LeafDim : first);
        sHi = int(last > LeafDim - 1 ? LeafDim - 1 : last);
    }
};

} // namespace morphology

} // namespace nanovdb::util

#if defined(__CUDACC__)
#include <nanovdb/util/cuda/MorphologyHelpers.cuh>
#endif // defined(__CUDACC__)

#endif // NANOVDB_UTIL_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED

