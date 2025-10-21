// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file MorphologyHelpers.cuh

    \author Efty Sifakis

    \date March 17, 2025

    \brief This file implements helper methods used in morphology operations
*/

#ifndef NANOVDB_UTIL_MORPHOLOGY_CUDA_MORPHOLOGYHELPERS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_MORPHOLOGY_CUDA_MORPHOLOGYHELPERS_CUH_HAS_BEEN_INCLUDED

namespace nanovdb::util {

namespace morphology {

namespace cuda {

template<int di, int dj, int dk, bool combine = true>
__device__
void MaskShift( const nanovdb::Mask<4>& inMaskQ, nanovdb::Mask<4>& outMaskQ )
{
    static_assert( (di<=15) && (di >= -15) && (dj<=15) && (dj >= -15) && (dk<=15) && (dk >= -15) );
    using WordBlockType = uint64_t [2][8][4];
    auto inMask = reinterpret_cast<const WordBlockType&>(*inMaskQ.words());
    auto outMask = reinterpret_cast<WordBlockType&>(*outMaskQ.words());

    int threadInWarpID = threadIdx.x & 0x1f;
    int ii = threadInWarpID >> 2;
    int jj = threadInWarpID & 0x3;

    uint64_t loWord = inMask[0][ii][jj];
    uint64_t hiWord = inMask[1][ii][jj];

    // Positive X-axis shift
    if constexpr (di < 0) {
        uint64_t loShfU = __shfl_up_sync  ( 0xffffffff, loWord,       ((-di)%8) << 2  );
        uint64_t loShfD = __shfl_down_sync( 0xffffffff, loWord, 32 - (((-di)%8) << 2) );
        uint64_t hiShfU = __shfl_up_sync  ( 0xffffffff, hiWord,       ((-di)%8) << 2  );

        loWord = ((ii+di) >= 0) ? loShfU : 0UL;
        if constexpr (di <= -8)
            hiWord = ((ii+di+8) >= 0) ? loShfU : 0UL;
        else
            hiWord = ((ii+di) >= 0) ? hiShfU : loShfD;
    }

    // Negative X-axis shift
    if constexpr (di > 0) {
        uint64_t loShfD = __shfl_down_sync( 0xffffffff, loWord,       (di%8) << 2  );
        uint64_t hiShfD = __shfl_down_sync( 0xffffffff, hiWord,       (di%8) << 2  );
        uint64_t hiShfU = __shfl_up_sync  ( 0xffffffff, hiWord, 32 - ((di%8) << 2) );

        if constexpr (di >= 8)
            loWord = ((ii+di-16) < 0) ? hiShfD : 0UL;
        else
            loWord = ((ii+di-8) < 0) ? loShfD : hiShfU;
        hiWord = ((ii+di) < 8) ? hiShfD : 0UL;
    }

    // Positive Y-axis shift
    if constexpr (dj > 0)
    {
        int djj = dj >> 2, mjj = dj & 0x3;
        uint64_t loLSW = __shfl_down_sync( 0xffffffff, loWord, djj ) >> ( 16 * mjj );
        uint64_t hiLSW = __shfl_down_sync( 0xffffffff, hiWord, djj ) >> ( 16 * mjj );
        if (jj + djj > 3) loLSW = hiLSW = 0UL;
        uint64_t loMSW = __shfl_down_sync( 0xffffffff, loWord, djj + 1 ) << (64 - ( 16 * mjj ) );
        uint64_t hiMSW = __shfl_down_sync( 0xffffffff, hiWord, djj + 1 ) << (64 - ( 16 * mjj ) );
        if (jj + djj > 2) loMSW = hiMSW = 0UL;
        loWord = loLSW | loMSW;
        hiWord = hiLSW | hiMSW;
    }

    // Negative Y-axis shift
    if constexpr (dj < 0)
    {
        int djj = (-dj) >> 2, mjj = (-dj) & 0x3;
        uint64_t loLSW = __shfl_up_sync( 0xffffffff, loWord, djj ) << ( 16 * mjj );
        uint64_t hiLSW = __shfl_up_sync( 0xffffffff, hiWord, djj ) << ( 16 * mjj );
        if (jj - djj < 0) loLSW = hiLSW = 0UL;
        uint64_t loMSW = __shfl_up_sync( 0xffffffff, loWord, djj + 1 ) >> (64 - ( 16 * mjj ) );
        uint64_t hiMSW = __shfl_up_sync( 0xffffffff, hiWord, djj + 1 ) >> (64 - ( 16 * mjj ) );
        if (jj - djj < 1) loMSW = hiMSW = 0UL;
        loWord = loLSW | loMSW;
        hiWord = hiLSW | hiMSW;
    }

    // Positive Z-axis shift
    if constexpr (dk > 0)
    {
        constexpr uint64_t shortMask = (uint16_t(0xffff) >> dk) << dk;
        static_assert( shortMask != 0xffffUL ); // Ensure the reciprocating shifts have not been optimized away
        constexpr uint64_t mask = (shortMask << 48) | (shortMask << 32) | (shortMask << 16) | shortMask;
        loWord = (loWord & mask) >> dk;
        hiWord = (hiWord & mask) >> dk;
    }

    // Negative Z-axis shift
    if constexpr (dk < 0)
    {
        // This awkward expression circumvents an inadvertent elimination by nvcc of the reciprocating shifts
        constexpr uint64_t shortMask = (uint64_t(uint16_t(0xffff) << (-dk)) & 0x000000000000ffffUL) >> (-dk);
        static_assert( shortMask != 0xffffUL ); // Ensure the reciprocating shifts have not been optimized away
        constexpr uint64_t mask = (shortMask << 48) | (shortMask << 32) | (shortMask << 16) | shortMask;
        loWord = (loWord & mask) << (-dk);
        hiWord = (hiWord & mask) << (-dk);
    }

    if constexpr (combine) {
        outMask[0][ii][jj] |= loWord;
        outMask[1][ii][jj] |= hiWord; }
    else {
        outMask[0][ii][jj] = loWord;
        outMask[1][ii][jj] = hiWord; }
}

} // namespace cuda

} // namespace morphology

} // namespace nanovdb::util

#endif // NANOVDB_UTIL_MORPHOLOGY_CUDA_MORPHOLOGYHELPERS_CUH_HAS_BEEN_INCLUDED
