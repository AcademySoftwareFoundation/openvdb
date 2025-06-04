// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file MorphologyHelpers.h

    \author Efty Sifakis

    \date March 17, 2025

    \brief This file implements helper methods used in morphology operations

*/

#ifndef NANOVDB_TOOLS_CUDA_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

namespace nanovdb::tools {

namespace morphology {

enum NearestNeighbors { NN_FACE = 6, NN_FACE_EDGE = 18, NN_FACE_EDGE_VERTEX = 26 };

template<int di, int dj, int dk>
struct NearestNeighborBitMask {
    static_assert( (di>=-1) && (di<=1) && (dj>=-1) && (dj<=1) && (dk>=-1) && (dk<=1) );
    static constexpr uint32_t value = 1u << (di+1)*9+(dj+1)*3+dk+1;
};

template<NearestNeighbors nnType>
__hostdev__
uint32_t neighborMaskStencil(const nanovdb::Mask<3>& mask)
{
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
        if ( allWordsOr & 0x00000000000000ffUL ) result |= NearestNeighborBitMask< 0,-1, 0>::value;
        if ( allWordsOr & 0xff00000000000000UL ) result |= NearestNeighborBitMask< 0, 1, 0>::value;
        if ( allWordsOr & 0x0101010101010101UL ) result |= NearestNeighborBitMask< 0, 0,-1>::value;
        if ( allWordsOr & 0x8080808080808080UL ) result |= NearestNeighborBitMask< 0, 0, 1>::value; }
    // Neighbors across edges
    if constexpr (nnType == NN_FACE_EDGE || nnType == NN_FACE_EDGE_VERTEX) {
        if ( words[0]   & 0x00000000000000ffUL ) result |= NearestNeighborBitMask<-1,-1, 0>::value;
        if ( words[0]   & 0xff00000000000000UL ) result |= NearestNeighborBitMask<-1, 1, 0>::value;
        if ( words[0]   & 0x0101010101010101UL ) result |= NearestNeighborBitMask<-1, 0,-1>::value;
        if ( words[0]   & 0x8080808080808080UL ) result |= NearestNeighborBitMask<-1, 0, 1>::value;
        if ( allWordsOr & 0x0000000000000001UL ) result |= NearestNeighborBitMask< 0,-1,-1>::value;
        if ( allWordsOr & 0x0000000000000080UL ) result |= NearestNeighborBitMask< 0,-1, 1>::value;
        if ( allWordsOr & 0x0100000000000000UL ) result |= NearestNeighborBitMask< 0, 1,-1>::value;
        if ( allWordsOr & 0x8000000000000000UL ) result |= NearestNeighborBitMask< 0, 1, 1>::value;
        if ( words[7]   & 0x00000000000000ffUL ) result |= NearestNeighborBitMask< 1,-1, 0>::value;
        if ( words[7]   & 0xff00000000000000UL ) result |= NearestNeighborBitMask< 1, 1, 0>::value;
        if ( words[7]   & 0x0101010101010101UL ) result |= NearestNeighborBitMask< 1, 0,-1>::value;
        if ( words[7]   & 0x8080808080808080UL ) result |= NearestNeighborBitMask< 1, 0, 1>::value; }
        // Neighbors across vertices
    if constexpr (nnType == NN_FACE_EDGE_VERTEX) {
        if ( words[0]   & 0x0000000000000001UL ) result |= NearestNeighborBitMask<-1,-1,-1>::value;
        if ( words[0]   & 0x0000000000000080UL ) result |= NearestNeighborBitMask<-1,-1, 1>::value;
        if ( words[0]   & 0x0100000000000000UL ) result |= NearestNeighborBitMask<-1, 1,-1>::value;
        if ( words[0]   & 0x8000000000000000UL ) result |= NearestNeighborBitMask<-1, 1, 1>::value;
        if ( words[7]   & 0x0000000000000001UL ) result |= NearestNeighborBitMask< 1,-1,-1>::value;
        if ( words[7]   & 0x0000000000000080UL ) result |= NearestNeighborBitMask< 1,-1, 1>::value;
        if ( words[7]   & 0x0100000000000000UL ) result |= NearestNeighborBitMask< 1, 1,-1>::value;
        if ( words[7]   & 0x8000000000000000UL ) result |= NearestNeighborBitMask< 1, 1, 1>::value; }
    return result;
}

} // namespace morphology

} // namespace nanovdb::tools

#if defined(__CUDACC__)
#include <nanovdb/util/cuda/MorphologyHelpers.cuh>
#endif // defined(__CUDACC__)

#endif // NANOVDB_TOOLS_CUDA_MORPHOLOGYHELPERS_H_HAS_BEEN_INCLUDED

