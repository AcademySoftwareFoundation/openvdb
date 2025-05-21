// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file Morphology.cuh

    \author Efty Sifakis

    \date March 17, 2025

    \brief Implements various stages of morphology dilation operators
*/

#include <cub/cub.cuh>

#include <nanovdb/util/MorphologyHelpers.h>

#ifndef NANOVDB_TOOLS_MORPHOLOGY_CUDA_MORPHOLOGY_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_MORPHOLOGY_CUDA_MORPHOLOGY_CUH_HAS_BEEN_INCLUDED

namespace nanovdb::tools {

namespace morphology {

namespace cuda {

template<class BuildT, NearestNeighbors nnType>
struct DilateInternalNodesFunctor
{
    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int WarpsPerBlock = MaxThreadsPerBlock >> 5;
    static constexpr int SlicesPerLowerNode = 8;
    static constexpr int LeafNodesPerSlice = 4096 / SlicesPerLowerNode;

    void __device__
    operator()(
        NanoGrid<BuildT> *srcGrid,
        NanoRoot<BuildT> *dilatedRoot,
        void *upperMasks_,
        void *lowerMasks_)
    {
        int tID = threadIdx.x;
        int lowerID = blockIdx.x;
        int sliceID = blockIdx.y;
        int threadInWarpID = threadIdx.x & 0x1f;
        int warpID = threadIdx.x >> 5;

        using UpperMaskArrayT = Mask<5>*;
        using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
        auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
        auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

        using LowerMaskT = Mask<4>;
        using LowerMaskStencilT = LowerMaskT (&)[3][3][3];
        __shared__ uint64_t sOffsetMasksRaw[LowerMaskT::WORD_COUNT*27];
        __shared__ uint64_t sNeighborMasksRaw[LowerMaskT::WORD_COUNT*27];
        auto sOffsetMasks = reinterpret_cast<LowerMaskStencilT>(sOffsetMasksRaw[0]);
        auto sNeighborMasks = reinterpret_cast<LowerMaskStencilT>(sNeighborMasksRaw[0]);

        // TODO: Use all available threads
        if (tID < LowerMaskT::WORD_COUNT)
            for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                const_cast<uint64_t*>(sOffsetMasks[i][j][k].words())[tID] = 0;
                const_cast<uint64_t*>(sNeighborMasks[i][j][k].words())[tID] = 0; }
        __syncthreads();

        const auto& srcTree = srcGrid->tree();
        const auto& lower = srcTree.template getFirstNode<1>()[lowerID];
        auto& valueMask = const_cast<LowerMaskT&>(lower.valueMask());

        for ( std::size_t jj = sliceID*LeafNodesPerSlice; jj < (sliceID+1)*LeafNodesPerSlice; jj += MaxThreadsPerBlock ) {

            // Compute the mask of affected lower nodes in packed uint32_t format
            uint32_t neighborMask = 0;
            if ( lower.childMask().isOn(jj+tID) ) {
                auto& leaf = *lower.data()->getChild(jj+tID);
                neighborMask = neighborMaskStencil<nnType>(leaf.valueMask()); }

            // Combine information from LeafNodes processed into an offset mask
            for (int bit = 0; bit < 27; bit++) {
                uint32_t mask = (neighborMask & (1u << bit)) ? (1u << threadInWarpID) : 0;
                mask = __reduce_or_sync(0xffffffffu, mask);
                auto warpMaskPtr = reinterpret_cast<uint32_t*>(sOffsetMasks[0][0][bit].words()) + (jj >> 5);
                // Do we need to guard this ??
                if (threadInWarpID == 0)
                    warpMaskPtr[warpID] = mask;
                __syncthreads(); }
        }

        // Compute neighbor masks from offset masks
        // This version is optimized for 128 threads (and requires at least that many)
        if (warpID == 0) {
            // Contribution to mask of own lower node
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,1,1)
            MaskShift<  1,  1,  1>( sOffsetMasks[0][0][0], sNeighborMasks[1][1][1] );
            MaskShift<  1,  1,  0>( sOffsetMasks[0][0][1], sNeighborMasks[1][1][1] );
            MaskShift<  1,  1, -1>( sOffsetMasks[0][0][2], sNeighborMasks[1][1][1] );
            MaskShift<  1,  0,  1>( sOffsetMasks[0][1][0], sNeighborMasks[1][1][1] );
            MaskShift<  1,  0,  0>( sOffsetMasks[0][1][1], sNeighborMasks[1][1][1] );
            MaskShift<  1,  0, -1>( sOffsetMasks[0][1][2], sNeighborMasks[1][1][1] );
            MaskShift<  1, -1,  1>( sOffsetMasks[0][2][0], sNeighborMasks[1][1][1] );
            MaskShift<  1, -1,  0>( sOffsetMasks[0][2][1], sNeighborMasks[1][1][1] );
            MaskShift<  1, -1, -1>( sOffsetMasks[0][2][2], sNeighborMasks[1][1][1] );
            MaskShift<  0,  1,  1>( sOffsetMasks[1][0][0], sNeighborMasks[1][1][1] );
            MaskShift<  0,  1,  0>( sOffsetMasks[1][0][1], sNeighborMasks[1][1][1] );
            MaskShift<  0,  1, -1>( sOffsetMasks[1][0][2], sNeighborMasks[1][1][1] );
            MaskShift<  0,  0,  1>( sOffsetMasks[1][1][0], sNeighborMasks[1][1][1] );
            MaskShift<  0,  0,  0>( sOffsetMasks[1][1][1], sNeighborMasks[1][1][1] );
            MaskShift<  0,  0, -1>( sOffsetMasks[1][1][2], sNeighborMasks[1][1][1] );
            MaskShift<  0, -1,  1>( sOffsetMasks[1][2][0], sNeighborMasks[1][1][1] );
            MaskShift<  0, -1,  0>( sOffsetMasks[1][2][1], sNeighborMasks[1][1][1] );
            MaskShift<  0, -1, -1>( sOffsetMasks[1][2][2], sNeighborMasks[1][1][1] );
            MaskShift< -1,  1,  1>( sOffsetMasks[2][0][0], sNeighborMasks[1][1][1] );
            MaskShift< -1,  1,  0>( sOffsetMasks[2][0][1], sNeighborMasks[1][1][1] );
            MaskShift< -1,  1, -1>( sOffsetMasks[2][0][2], sNeighborMasks[1][1][1] );
            MaskShift< -1,  0,  1>( sOffsetMasks[2][1][0], sNeighborMasks[1][1][1] );
            MaskShift< -1,  0,  0>( sOffsetMasks[2][1][1], sNeighborMasks[1][1][1] );
            MaskShift< -1,  0, -1>( sOffsetMasks[2][1][2], sNeighborMasks[1][1][1] );
            MaskShift< -1, -1,  1>( sOffsetMasks[2][2][0], sNeighborMasks[1][1][1] );
            MaskShift< -1, -1,  0>( sOffsetMasks[2][2][1], sNeighborMasks[1][1][1] );
            MaskShift< -1, -1, -1>( sOffsetMasks[2][2][2], sNeighborMasks[1][1][1] );
            // Contribution to mask of lower node at offset (-1,-1,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,-15,-15)
            MaskShift<-15,-15,-15>( sOffsetMasks[0][0][0], sNeighborMasks[0][0][0] );
            // Contribution to mask of lower node at offset (-1,-1,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,-15,17)
            MaskShift<-15,-15, 15>( sOffsetMasks[0][0][2], sNeighborMasks[0][0][2] );
            // Contribution to mask of lower node at offset (-1,1,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,17,-15)
            MaskShift<-15, 15,-15>( sOffsetMasks[0][2][0], sNeighborMasks[0][2][0] );
            // Contribution to mask of lower node at offset (-1,1,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,17,17)
            MaskShift<-15, 15, 15>( sOffsetMasks[0][2][2], sNeighborMasks[0][2][2] );
            // Contribution to mask of lower node at offset (1,-1,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,-15,-15)
            MaskShift< 15,-15,-15>( sOffsetMasks[2][0][0], sNeighborMasks[2][0][0] );
        }

        if (warpID == 1) {
            // Contribution to mask of lower node at offset (0,0,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,1,-15)
            MaskShift<  1,  1,-15>( sOffsetMasks[0][0][0], sNeighborMasks[1][1][0] );
            MaskShift<  1,  0,-15>( sOffsetMasks[0][1][0], sNeighborMasks[1][1][0] );
            MaskShift<  1, -1,-15>( sOffsetMasks[0][2][0], sNeighborMasks[1][1][0] );
            MaskShift<  0,  1,-15>( sOffsetMasks[1][0][0], sNeighborMasks[1][1][0] );
            MaskShift<  0,  0,-15>( sOffsetMasks[1][1][0], sNeighborMasks[1][1][0] );
            MaskShift<  0, -1,-15>( sOffsetMasks[1][2][0], sNeighborMasks[1][1][0] );
            MaskShift< -1,  1,-15>( sOffsetMasks[2][0][0], sNeighborMasks[1][1][0] );
            MaskShift< -1,  0,-15>( sOffsetMasks[2][1][0], sNeighborMasks[1][1][0] );
            MaskShift< -1, -1,-15>( sOffsetMasks[2][2][0], sNeighborMasks[1][1][0] );
            // Contribution to mask of lower node at offset (0,0,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,1,17)
            MaskShift<  1,  1, 15>( sOffsetMasks[0][0][2], sNeighborMasks[1][1][2] );
            MaskShift<  1,  0, 15>( sOffsetMasks[0][1][2], sNeighborMasks[1][1][2] );
            MaskShift<  1, -1, 15>( sOffsetMasks[0][2][2], sNeighborMasks[1][1][2] );
            MaskShift<  0,  1, 15>( sOffsetMasks[1][0][2], sNeighborMasks[1][1][2] );
            MaskShift<  0,  0, 15>( sOffsetMasks[1][1][2], sNeighborMasks[1][1][2] );
            MaskShift<  0, -1, 15>( sOffsetMasks[1][2][2], sNeighborMasks[1][1][2] );
            MaskShift< -1,  1, 15>( sOffsetMasks[2][0][2], sNeighborMasks[1][1][2] );
            MaskShift< -1,  0, 15>( sOffsetMasks[2][1][2], sNeighborMasks[1][1][2] );
            MaskShift< -1, -1, 15>( sOffsetMasks[2][2][2], sNeighborMasks[1][1][2] );
            // Contribution to mask of lower node at offset (-1,-1,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,-15,1)
            MaskShift<-15,-15,  1>( sOffsetMasks[0][0][0], sNeighborMasks[0][0][1] );
            MaskShift<-15,-15,  0>( sOffsetMasks[0][0][1], sNeighborMasks[0][0][1] );
            MaskShift<-15,-15, -1>( sOffsetMasks[0][0][2], sNeighborMasks[0][0][1] );
            // Contribution to mask of lower node at offset (-1,1,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,17,1)
            MaskShift<-15, 15,  1>( sOffsetMasks[0][2][0], sNeighborMasks[0][2][1] );
            MaskShift<-15, 15,  0>( sOffsetMasks[0][2][1], sNeighborMasks[0][2][1] );
            MaskShift<-15, 15, -1>( sOffsetMasks[0][2][2], sNeighborMasks[0][2][1] );
            // Contribution to mask of lower node at offset (1,-1,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,-15,1)
            MaskShift< 15,-15,  1>( sOffsetMasks[2][0][0], sNeighborMasks[2][0][1] );
            MaskShift< 15,-15,  0>( sOffsetMasks[2][0][1], sNeighborMasks[2][0][1] );
            MaskShift< 15,-15, -1>( sOffsetMasks[2][0][2], sNeighborMasks[2][0][1] );
            // Contribution to mask of lower node at offset (1,1,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,17,1)
            MaskShift< 15, 15,  1>( sOffsetMasks[2][2][0], sNeighborMasks[2][2][1] );
            MaskShift< 15, 15,  0>( sOffsetMasks[2][2][1], sNeighborMasks[2][2][1] );
            MaskShift< 15, 15, -1>( sOffsetMasks[2][2][2], sNeighborMasks[2][2][1] );
            // Contribution to mask of lower node at offset (1,-1,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,-15,17)
            MaskShift< 15,-15, 15>( sOffsetMasks[2][0][2], sNeighborMasks[2][0][2] );
        }

        if (warpID == 2) {
            // Contribution to mask of lower node at offset (0,-1,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,-15,1)
            MaskShift<  1,-15,  1>( sOffsetMasks[0][0][0], sNeighborMasks[1][0][1] );
            MaskShift<  1,-15,  0>( sOffsetMasks[0][0][1], sNeighborMasks[1][0][1] );
            MaskShift<  1,-15, -1>( sOffsetMasks[0][0][2], sNeighborMasks[1][0][1] );
            MaskShift<  0,-15,  1>( sOffsetMasks[1][0][0], sNeighborMasks[1][0][1] );
            MaskShift<  0,-15,  0>( sOffsetMasks[1][0][1], sNeighborMasks[1][0][1] );
            MaskShift<  0,-15, -1>( sOffsetMasks[1][0][2], sNeighborMasks[1][0][1] );
            MaskShift< -1,-15,  1>( sOffsetMasks[2][0][0], sNeighborMasks[1][0][1] );
            MaskShift< -1,-15,  0>( sOffsetMasks[2][0][1], sNeighborMasks[1][0][1] );
            MaskShift< -1,-15, -1>( sOffsetMasks[2][0][2], sNeighborMasks[1][0][1] );
            // Contribution to mask of lower node at offset (0,1,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,17,1)
            MaskShift<  1, 15,  1>( sOffsetMasks[0][2][0], sNeighborMasks[1][2][1] );
            MaskShift<  1, 15,  0>( sOffsetMasks[0][2][1], sNeighborMasks[1][2][1] );
            MaskShift<  1, 15, -1>( sOffsetMasks[0][2][2], sNeighborMasks[1][2][1] );
            MaskShift<  0, 15,  1>( sOffsetMasks[1][2][0], sNeighborMasks[1][2][1] );
            MaskShift<  0, 15,  0>( sOffsetMasks[1][2][1], sNeighborMasks[1][2][1] );
            MaskShift<  0, 15, -1>( sOffsetMasks[1][2][2], sNeighborMasks[1][2][1] );
            MaskShift< -1, 15,  1>( sOffsetMasks[2][2][0], sNeighborMasks[1][2][1] );
            MaskShift< -1, 15,  0>( sOffsetMasks[2][2][1], sNeighborMasks[1][2][1] );
            MaskShift< -1, 15, -1>( sOffsetMasks[2][2][2], sNeighborMasks[1][2][1] );
            // Contribution to mask of lower node at offset (-1,0,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,1,-15)
            MaskShift<-15,  1,-15>( sOffsetMasks[0][0][0], sNeighborMasks[0][1][0] );
            MaskShift<-15,  0,-15>( sOffsetMasks[0][1][0], sNeighborMasks[0][1][0] );
            MaskShift<-15, -1,-15>( sOffsetMasks[0][2][0], sNeighborMasks[0][1][0] );
            // Contribution to mask of lower node at offset (-1,0,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,1,17)
            MaskShift<-15,  1, 15>( sOffsetMasks[0][0][2], sNeighborMasks[0][1][2] );
            MaskShift<-15,  0, 15>( sOffsetMasks[0][1][2], sNeighborMasks[0][1][2] );
            MaskShift<-15, -1, 15>( sOffsetMasks[0][2][2], sNeighborMasks[0][1][2] );
            // Contribution to mask of lower node at offset (1,0,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,1,-15)
            MaskShift< 15,  1,-15>( sOffsetMasks[2][0][0], sNeighborMasks[2][1][0] );
            MaskShift< 15,  0,-15>( sOffsetMasks[2][1][0], sNeighborMasks[2][1][0] );
            MaskShift< 15, -1,-15>( sOffsetMasks[2][2][0], sNeighborMasks[2][1][0] );
            // Contribution to mask of lower node at offset (1,0,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,1,17)
            MaskShift< 15,  1, 15>( sOffsetMasks[2][0][2], sNeighborMasks[2][1][2] );
            MaskShift< 15,  0, 15>( sOffsetMasks[2][1][2], sNeighborMasks[2][1][2] );
            MaskShift< 15, -1, 15>( sOffsetMasks[2][2][2], sNeighborMasks[2][1][2] );
            // Contribution to mask of lower node at offset (1,1,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,17,-15)
            MaskShift< 15, 15,-15>( sOffsetMasks[2][2][0], sNeighborMasks[2][2][0] );
        }

        if (warpID == 3) {
            // Contribution to mask of lower node at offset (-1,0,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (-15,1,1)
            MaskShift<-15,  1,  1>( sOffsetMasks[0][0][0], sNeighborMasks[0][1][1] );
            MaskShift<-15,  1,  0>( sOffsetMasks[0][0][1], sNeighborMasks[0][1][1] );
            MaskShift<-15,  1, -1>( sOffsetMasks[0][0][2], sNeighborMasks[0][1][1] );
            MaskShift<-15,  0,  1>( sOffsetMasks[0][1][0], sNeighborMasks[0][1][1] );
            MaskShift<-15,  0,  0>( sOffsetMasks[0][1][1], sNeighborMasks[0][1][1] );
            MaskShift<-15,  0, -1>( sOffsetMasks[0][1][2], sNeighborMasks[0][1][1] );
            MaskShift<-15, -1,  1>( sOffsetMasks[0][2][0], sNeighborMasks[0][1][1] );
            MaskShift<-15, -1,  0>( sOffsetMasks[0][2][1], sNeighborMasks[0][1][1] );
            MaskShift<-15, -1, -1>( sOffsetMasks[0][2][2], sNeighborMasks[0][1][1] );
            // Contribution to mask of lower node at offset (1,0,0)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,1,1)
            MaskShift< 15,  1,  1>( sOffsetMasks[2][0][0], sNeighborMasks[2][1][1] );
            MaskShift< 15,  1,  0>( sOffsetMasks[2][0][1], sNeighborMasks[2][1][1] );
            MaskShift< 15,  1, -1>( sOffsetMasks[2][0][2], sNeighborMasks[2][1][1] );
            MaskShift< 15,  0,  1>( sOffsetMasks[2][1][0], sNeighborMasks[2][1][1] );
            MaskShift< 15,  0,  0>( sOffsetMasks[2][1][1], sNeighborMasks[2][1][1] );
            MaskShift< 15,  0, -1>( sOffsetMasks[2][1][2], sNeighborMasks[2][1][1] );
            MaskShift< 15, -1,  1>( sOffsetMasks[2][2][0], sNeighborMasks[2][1][1] );
            MaskShift< 15, -1,  0>( sOffsetMasks[2][2][1], sNeighborMasks[2][1][1] );
            MaskShift< 15, -1, -1>( sOffsetMasks[2][2][2], sNeighborMasks[2][1][1] );
            // Contribution to mask of lower node at offset (0,-1,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,-15,-15)
            MaskShift<  1,-15,-15>( sOffsetMasks[0][0][0], sNeighborMasks[1][0][0] );
            MaskShift<  0,-15,-15>( sOffsetMasks[1][0][0], sNeighborMasks[1][0][0] );
            MaskShift< -1,-15,-15>( sOffsetMasks[2][0][0], sNeighborMasks[1][0][0] );
            // Contribution to mask of lower node at offset (0,-1,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,-15,17)
            MaskShift<  1,-15, 15>( sOffsetMasks[0][0][2], sNeighborMasks[1][0][2] );
            MaskShift<  0,-15, 15>( sOffsetMasks[1][0][2], sNeighborMasks[1][0][2] );
            MaskShift< -1,-15, 15>( sOffsetMasks[2][0][2], sNeighborMasks[1][0][2] );
            // Contribution to mask of lower node at offset (0,1,-1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,17,-15)
            MaskShift<  1, 15,-15>( sOffsetMasks[0][2][0], sNeighborMasks[1][2][0] );
            MaskShift<  0, 15,-15>( sOffsetMasks[1][2][0], sNeighborMasks[1][2][0] );
            MaskShift< -1, 15,-15>( sOffsetMasks[2][2][0], sNeighborMasks[1][2][0] );
            // Contribution to mask of lower node at offset (0,1,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (1,17,17)
            MaskShift<  1, 15, 15>( sOffsetMasks[0][2][2], sNeighborMasks[1][2][2] );
            MaskShift<  0, 15, 15>( sOffsetMasks[1][2][2], sNeighborMasks[1][2][2] );
            MaskShift< -1, 15, 15>( sOffsetMasks[2][2][2], sNeighborMasks[1][2][2] );
            // Contribution to mask of lower node at offset (1,1,1)
            // Arguments to MaskShift plus indices to sOffsetMasks add up to (17,17,17)
            MaskShift< 15, 15, 15>( sOffsetMasks[2][2][2], sNeighborMasks[2][2][2] );
        }

        __syncthreads();

        // Compose contributions to the lower-node masks of the dilated tree

        for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
        for (int dk = -1; dk <= 1; dk++) {
            int neighborID = (di+1)*9+(dj+1)*3+dk+1;
            if ((neighborID % WarpsPerBlock) == warpID) {
                auto neighborOrigin = lower.origin().offsetBy(di*128,dj*128,dk*128);
                auto upperChildIndex = NanoUpper<BuildT>::CoordToOffset(neighborOrigin);
                auto& neighborMask = sNeighborMasks[di+1][dj+1][dk+1];

                for (int tOffset = 0; tOffset < Mask<4>::WORD_COUNT; tOffset += 32) {
                    unsigned long long int computedWord = neighborMask.words()[threadInWarpID+tOffset];
                    if (computedWord) {
                        auto dilatedTile = dilatedRoot->probeTile(neighborOrigin);
                        uint64_t tileChildIndex =
                            util::PtrDiff(dilatedRoot->getChild(dilatedTile), dilatedRoot->getChild(dilatedRoot->tile(0)))
                            / sizeof(NanoUpper<BuildT>); // TODO: consider some faster integer division? or a way to avoid this?
                        auto& outputUpperMask = upperMasks[tileChildIndex];
                        outputUpperMask.setOnAtomic(upperChildIndex);
                        auto& outputLowerMask = lowerMasks[tileChildIndex][upperChildIndex];
                        unsigned long long int *outputWord = reinterpret_cast<unsigned long long int*>(
                            const_cast<uint64_t*>(&outputLowerMask.words()[threadInWarpID+tOffset]));
                        atomicOr( outputWord, computedWord); } } } }
        __syncthreads();
    }

};

struct EnumerateNodesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int WarpsPerBlock = MaxThreadsPerBlock >> 5;
    static constexpr int SlicesPerUpperNode = 256;
    static constexpr int LowerNodesPerSlice = 32768 / SlicesPerUpperNode;

    void __device__
    operator()(
        void *upperMasks_,
        void *lowerMasks_,
        uint32_t (*lowerCounts)[Mask<5>::SIZE],
        uint32_t (*leafCounts)[Mask<5>::SIZE] )
    {
        int upperID = blockIdx.x;
        int sliceID = blockIdx.y;
        int threadInWarpID = threadIdx.x & 0x1f;
        int warpID = threadIdx.x >> 5;

        using UpperMaskArrayT = Mask<5>*;
        using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
        auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
        auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

        for ( std::size_t jj = sliceID*LowerNodesPerSlice + warpID; jj < (sliceID+1)*LowerNodesPerSlice; jj += WarpsPerBlock ) {
            if (upperMasks[upperID].isOn(jj)) {
                auto& lowerMask = lowerMasks[upperID][jj];
                uint32_t lowerCountOn = util::countOn(lowerMask.words()[2*threadInWarpID]) + util::countOn(lowerMask.words()[2*threadInWarpID+1]);
                lowerCountOn = __reduce_add_sync( 0xffffffff, lowerCountOn );
                if (threadInWarpID == 0) { // TODO: do we need to guard this?
                    lowerCounts[upperID][jj] = 1;
                    leafCounts[upperID][jj] = lowerCountOn; } }
            else if (threadInWarpID == 0) // TODO: do we need to guard this?
                lowerCounts[upperID][jj] = leafCounts[upperID][jj] = 0;
            __syncthreads(); }
    }
};

template<typename BuildT>
struct ProcessLowerNodesFunctor
{
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int WarpsPerBlock = MaxThreadsPerBlock >> 5;
    static constexpr int SlicesPerUpperNode = 256;
    static constexpr int LowerNodesPerSlice = 32768 / SlicesPerUpperNode;

    void __device__
    operator()(
        void *upperMasks_,
        void *lowerMasks_,
        uint32_t *upperOffsets,
        uint32_t (*lowerOffsets)[Mask<5>::SIZE],
        uint32_t (*leafOffsets)[Mask<5>::SIZE],
        NanoGrid<BuildT> *dstGrid,
        uint32_t *lowerParents,
        uint32_t *leafParents)
    {
        int dilatedTileID = blockIdx.x;
        int upperID = upperOffsets[dilatedTileID];
        int sliceID = blockIdx.y;
        int threadInWarpID = threadIdx.x & 0x1f;
        int warpID = threadIdx.x >> 5;

        using UpperMaskArrayT = Mask<5>*;
        using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
        auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
        auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

        using WarpScan = cub::WarpScan<uint32_t>;
        __shared__ typename WarpScan::TempStorage temp_storage[WarpsPerBlock];

        const auto& dstTree = dstGrid->tree();
        const auto upperOrigin = dstTree.root().tile(upperID)->origin();

        if (upperOffsets[dilatedTileID+1] > upperID) { // check that this particular dilated tile is not empty, i.e. it exists in the tree
            auto& upper = const_cast<NanoUpper<BuildT>&>(dstTree.template getFirstNode<2>()[upperID]);
            for ( int jj = sliceID*LowerNodesPerSlice + warpID; jj < (sliceID+1)*LowerNodesPerSlice; jj += WarpsPerBlock ) {
                if (upperMasks[dilatedTileID].isOn(jj)) {
                    const_cast<Mask<5>&>(upper.childMask()).setOnAtomic(jj);
                    auto lowerID = lowerOffsets[dilatedTileID][jj];
                    auto& lower = const_cast<NanoLower<BuildT>&>(dstTree.template getFirstNode<1>()[lowerID]);
                    const auto lowerOrigin = upperOrigin + (NanoUpper<BuildT>::OffsetToLocalCoord(jj) << NanoUpper<BuildT>::ChildNodeType::TOTAL);
                    upper.setChild(jj, &lower);
                    lowerParents[lowerID] = upperID;

                    auto lowerWords = lowerMasks[dilatedTileID][jj].words();
                    lower.mChildMask.words()[2*threadInWarpID  ] = lowerWords[2*threadInWarpID  ];
                    lower.mChildMask.words()[2*threadInWarpID+1] = lowerWords[2*threadInWarpID+1];
                    uint32_t prefixSum = util::countOn(lowerWords[2*threadInWarpID]) + util::countOn(lowerWords[2*threadInWarpID+1]);
                    WarpScan(temp_storage[warpID]).ExclusiveSum(prefixSum, prefixSum);
                    for ( int wordID = 2*threadInWarpID; wordID <= 2*threadInWarpID+1; wordID++ )
                        for ( int bitID = 0; bitID < 64; bitID++)
                            if ( lowerWords[wordID] & (1UL << bitID) ) {
                                int kk = (wordID << 6) + bitID;
                                int leafID = leafOffsets[dilatedTileID][jj] + prefixSum;
                                auto& leaf = const_cast<NanoLeaf<BuildT>&>(dstTree.template getFirstNode<0>()[leafID]);
                                lower.setChild(kk, &leaf);
                                leafParents[leafID] = lowerID;
                                const auto leafOrigin = lowerOrigin + (NanoLower<BuildT>::OffsetToLocalCoord(kk) << NanoLower<BuildT>::ChildNodeType::TOTAL);
                                leaf.mBBoxMin = leafOrigin; // To be further updated after the leaf-level dilation is complete
                                prefixSum++;
                                // TODO: Is this accurate? Any other flags that should be set?
                                leaf.mFlags = (uint64_t)GridFlags::HasBBox; }
                    lower.mBBox = CoordBBox(); // To be further updated after the leaf-level dilation is complete
                    // TODO: Is this accurate? Any other flags that should be set?
                    lower.mFlags = (uint64_t)GridFlags::HasBBox;
                }
            }
        }
    }
};

template<class BuildT, NearestNeighbors nnType>
struct DilateLeafNodesFunctor;

template<class BuildT>
struct DilateLeafNodesFunctor<BuildT, NN_FACE>
{
    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int SlicesPerLowerNode = 8;
    static constexpr int LeafNodesPerSlice = 4096 / SlicesPerLowerNode;

    __device__
    void operator()(
        NanoGrid<BuildT> *srcGrid,
        NanoGrid<BuildT> *dstGrid )
    {
        int tID = threadIdx.x;
        int lowerID = blockIdx.x;
        int sliceID = blockIdx.y;

        const auto& srcTree = srcGrid->tree();
        const auto& dstTree = dstGrid->tree();
        const auto& dstLower = dstTree.template getFirstNode<1>()[lowerID];
        for ( std::size_t jj = sliceID*LeafNodesPerSlice; jj < (sliceID+1)*LeafNodesPerSlice; jj += MaxThreadsPerBlock )
            if ( dstLower.childMask().isOn(jj+tID) )
            {
                auto& dstLeaf = *dstLower.data()->getChild(jj+tID);
                const auto leafOrigin = dstLeaf.origin();
                auto originalLeafPtr  = srcTree.root().probeLeaf(leafOrigin);
                uint64_t originalWords[8] = {}, dilatedWords[8]; // Keep these in registers
                if (originalLeafPtr)
                    for (int i = 0; i < 8; i++)
                        originalWords[i] = originalLeafPtr->valueMask().words()[i];

                for (int i = 0; i < 8; i++) {
                    dilatedWords[i] = originalWords[i];
                    // Activate voxel if the neighbor at stencil offset ( 1, 0, 0) is active
                    if (i < 7) dilatedWords[i] |= originalWords[i+1];
                    // Activate voxel if the neighbor at stencil offset (-1, 0, 0) is active
                    if (i > 0) dilatedWords[i] |= originalWords[i-1];

                    // Activate voxel if the neighbor at stencil offset ( 0, 1, 0) is active
                    dilatedWords[i] |= (originalWords[i] & 0xffffffffffffff00UL) >> 8;
                    // Activate voxel if the neighbor at stencil offset ( 0,-1, 0) is active
                    dilatedWords[i] |= (originalWords[i] & 0x00ffffffffffffffUL) << 8;

                    // Activate voxel if the neighbor at stencil offset ( 0, 0, 1) is active
                    dilatedWords[i] |= (originalWords[i] & 0xfefefefefefefefeUL) >> 1;
                    // Activate voxel if the neighbor at stencil offset ( 0, 0,-1) is active
                    dilatedWords[i] |= (originalWords[i] & 0x7f7f7f7f7f7f7f7fUL) << 1;
                }

                auto leafPlusXPtr  = srcTree.root().probeLeaf(leafOrigin.offsetBy( 8, 0, 0));
                auto leafMinusXPtr = srcTree.root().probeLeaf(leafOrigin.offsetBy(-8, 0, 0));
                auto leafPlusYPtr  = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0, 8, 0));
                auto leafMinusYPtr = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0,-8, 0));
                auto leafPlusZPtr  = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0, 0, 8));
                auto leafMinusZPtr = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0, 0,-8));

                // Activate voxel if the neighbor at stencil offset ( 1, 0, 0) is active
                if (leafPlusXPtr) dilatedWords[7] |= leafPlusXPtr->valueMask().words()[0];
                // Activate voxel if the neighbor at stencil offset (-1, 0, 0) is active
                if (leafMinusXPtr) dilatedWords[0] |= leafMinusXPtr->valueMask().words()[7];

                // Activate voxel if the neighbor at stencil offset ( 0, 1, 0) is active
                if (leafPlusYPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafPlusYPtr->valueMask().words()[i] & 0x00000000000000ffUL) << 56;
                // Activate voxel if the neighbor at stencil offset ( 0,-1, 0) is active
                if (leafMinusYPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafMinusYPtr->valueMask().words()[i] & 0xff00000000000000UL) >> 56;

                // Activate voxel if the neighbor at stencil offset ( 0, 0, 1) is active
                if (leafPlusZPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafPlusZPtr->valueMask().words()[i] & 0x0101010101010101UL) << 7;
                // Activate voxel if the neighbor at stencil offset ( 0, 0,-1) is active
                if (leafMinusZPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafMinusZPtr->valueMask().words()[i] & 0x8080808080808080UL) >> 7;

                auto& dilatedMask = const_cast<Mask<3>&>(dstLeaf.valueMask());
                for (int i = 0; i < 8; i++)
                    dilatedMask.words()[i] = dilatedWords[i];
            }
        return;
    }

};

template<class BuildT>
struct DilateLeafNodesFunctor<BuildT, NN_FACE_EDGE_VERTEX>
{
    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int SlicesPerLowerNode = 8;
    static constexpr int LeafNodesPerSlice = 4096 / SlicesPerLowerNode;

    __device__
    void operator()(NanoGrid<BuildT> *srcGrid, NanoGrid<BuildT> *dstGrid )
    {
        int tID = threadIdx.x;
        int lowerID = blockIdx.x;
        int sliceID = blockIdx.y;

        const auto& srcTree = srcGrid->tree();
        const auto& dstTree = dstGrid->tree();
        const auto& dstLower = dstTree.template getFirstNode<1>()[lowerID];
        for ( std::size_t jj = sliceID*LeafNodesPerSlice; jj < (sliceID+1)*LeafNodesPerSlice; jj += MaxThreadsPerBlock )
            if ( dstLower.childMask().isOn(jj+tID) )
            {
                auto& dstLeaf = *dstLower.data()->getChild(jj+tID);
                const auto leafOrigin = dstLeaf.origin();

                uint64_t originalWordsShifted[10][3][3] = {};
                using WordStencilT = uint64_t (&)[10][3][3]; // Dimensions: [x-voxel offset][y-block offset][z-block offset]
                auto& originalWords = reinterpret_cast<WordStencilT>(originalWordsShifted[1][1][1]); // Range: [-1,8][-1,1][-1,1]

                for (int dBi = -1; dBi <= 1; dBi++)
                for (int dBj = -1; dBj <= 1; dBj++)
                for (int dBk = -1; dBk <= 1; dBk++) {
                    auto neighborOrigin = leafOrigin.offsetBy( dBi*8, dBj*8, dBk*8);
                    if (auto neighborLeafPtr = srcTree.root().probeLeaf(neighborOrigin)) {
                        auto neighborWords = neighborLeafPtr->valueMask().words();
                        if (dBi == -1)
                            originalWords[-1][dBj][dBk] = neighborWords[7];
                        else if (dBi == 1)
                            originalWords[8][dBj][dBk] = neighborWords[0];
                        else
                            for (int i = 0; i < 8; i++)
                                originalWords[i][dBj][dBk] = neighborWords[i]; } }
                // Dilate along z-axis
                for (int i = -1; i <= 8; i++)
                for (int dBj = -1; dBj <= 1; dBj++) {
                    uint64_t dilatedWord = originalWords[i][dBj][0];
                    // Activate voxel if the neighbor at stencil offset ( 0, 0, 1) is active
                    dilatedWord |= (originalWords[i][dBj][ 0] & 0xfefefefefefefefeUL) >> 1;
                    dilatedWord |= (originalWords[i][dBj][ 1] & 0x0101010101010101UL) << 7;
                    // Activate voxel if the neighbor at stencil offset ( 0, 0,-1) is active
                    dilatedWord |= (originalWords[i][dBj][ 0] & 0x7f7f7f7f7f7f7f7fUL) << 1;
                    dilatedWord |= (originalWords[i][dBj][-1] & 0x8080808080808080UL) >> 7;
                    // Replace original with dilation result
                    originalWords[i][dBj][0] = dilatedWord; }

                // Dilate along y-axis
                for (int i = -1; i <= 8; i++) {
                    uint64_t dilatedWord = originalWords[i][0][0];
                    // Activate voxel if the neighbor at stencil offset ( 0, 1, 0) is active
                    dilatedWord |= (originalWords[i][ 0][0] & 0xffffffffffffff00UL) >> 8;
                    dilatedWord |= (originalWords[i][ 1][0] & 0x00000000000000ffUL) << 56;
                    // Activate voxel if the neighbor at stencil offset ( 0,-1, 0) is active
                    dilatedWord |= (originalWords[i][ 0][0] & 0x00ffffffffffffffUL) << 8;
                    dilatedWord |= (originalWords[i][-1][0] & 0xff00000000000000UL) >> 56;
                    // Replace original with dilation result
                    originalWords[i][0][0] = dilatedWord; }

                // Dilate along x-axis
                auto dilatedWords = const_cast<Mask<3>&>(dstLeaf.valueMask()).words();
                for (int i = 0; i <= 7; i++)
                    dilatedWords[i] = originalWords[i-1][0][0] | originalWords[i][0][0] | originalWords[i+1][0][0];
            }
        return;
    }

};

} // namespace cuda

} // namespace morphology

} // namespace nanovdb::tools

#endif // NANOVDB_TOOLS_MORPHOLOGY_CUDA_MORPHOLOGY_CUH_HAS_BEEN_INCLUDED
