// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*
    \file Injection.cuh

    \author Efty Sifakis

    \date Jun 3, 2025

    \brief This file implements data injection methods on sidecars corresponding to
           distinct indexGrid topologies.

*/

#ifndef NANOVDB_UTIL_CUDA_INECTION_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_INECTION_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

namespace nanovdb::util {

namespace cuda {

template <typename BuildT, typename ValueType, int64_t offset = 0>
struct InjectGridDataFunctor
{
    // Copies the sidecar data of a (source) grid into the sidecar of an overlapping (destination) grid
    // Intended to be called via nanovdb::util::cuda::operatorKernel
    // blockDim.x is presumed to be the leaf count of the source tree
    // Values of the destination sidecar that do not overlap with the source are left unchanged
    // NOTE: If the source voxels are not a subset of the destination voxels, the injection will be from
    // the intersection of the two active voxel sets into the destination
    // This version presumes that the sidecar contents are of compile-time known length,
    // and can be copied with the assignment operator.

    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__
    void operator()(
        const typename nanovdb::NanoGrid<BuildT> *d_srcGrid,
        const typename nanovdb::NanoGrid<BuildT> *d_dstGrid,
        const ValueType *d_srcData,
        ValueType *d_dstData)
    {
        int srcLeafID = blockIdx.x;
        int warpID = threadIdx.x >> 5;
        int threadInWarpID = threadIdx.x & 0x1f;

        const auto& srcTree = d_srcGrid->tree();
        const auto& dstTree = d_dstGrid->tree();
        const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
        auto dstLeafPtr = dstTree.root().probeLeaf(srcLeaf.origin());
        if (dstLeafPtr) {
            auto srcWord = srcLeaf.valueMask().words()[warpID];
            auto dstWord = dstLeafPtr->valueMask().words()[warpID];
            auto srcOffset = srcLeaf.firstOffset();
            if (warpID) srcOffset += (srcLeaf.mPrefixSum >> ((warpID-1) * 9)) & 0x1ff;
            auto dstOffset = dstLeafPtr->firstOffset();
            if (warpID) dstOffset += (dstLeafPtr->mPrefixSum >> ((warpID-1) * 9)) & 0x1ff;

            uint64_t loMask = 1UL << threadInWarpID;
            uint64_t hiMask = 0x100000000UL << threadInWarpID;
            uint64_t loSrcCnt = nanovdb::util::countOn( srcWord & (loMask-1UL) );
            uint64_t hiSrcCnt = nanovdb::util::countOn( srcWord & (hiMask-1UL) );
            uint64_t loDstCnt = nanovdb::util::countOn( dstWord & (loMask-1UL) );
            uint64_t hiDstCnt = nanovdb::util::countOn( dstWord & (hiMask-1UL) );

            if ( loMask & srcWord & dstWord )
                d_dstData[int64_t(dstOffset+loDstCnt)+offset] = d_srcData[int64_t(srcOffset+loSrcCnt)+offset];
            if ( hiMask & srcWord & dstWord )
                d_dstData[int64_t(dstOffset+hiDstCnt)+offset] = d_srcData[int64_t(srcOffset+hiSrcCnt)+offset];
        }
    }
};

template <typename BuildT, typename ValueType, int64_t offset = 0>
struct InjectGridFeatureFunctor
{
    // Copies the sidecar data of a (source) grid into the sidecar of an overlapping (destination) grid
    // Intended to be called via nanovdb::util::cuda::operatorKernel
    // blockDim.x is presumed to be the leaf count of the source tree
    // Values of the destination sidecar that do not overlap with the source are left unchanged
    // NOTE: If the source voxels are not a subset of the destination voxels, the injection will be from
    // the intersection of the two active voxel sets into the destination
    // This version presumes a runtime dimension parameter for input features
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__
    void operator()(
        const typename nanovdb::NanoGrid<BuildT> *d_srcGrid,
        const typename nanovdb::NanoGrid<BuildT> *d_dstGrid,
        const ValueType *d_srcData,
        ValueType *d_dstData,
        const std::size_t dim)
    {
        int srcLeafID = blockIdx.x;
        int warpID = threadIdx.x >> 5;
        int threadInWarpID = threadIdx.x & 0x1f;

        const auto& srcTree = d_srcGrid->tree();
        const auto& dstTree = d_dstGrid->tree();
        const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
        auto dstLeafPtr = dstTree.root().probeLeaf(srcLeaf.origin());
        if (dstLeafPtr) {
            auto srcWord = srcLeaf.valueMask().words()[warpID];
            auto dstWord = dstLeafPtr->valueMask().words()[warpID];
            auto srcOffset = srcLeaf.firstOffset();
            if (warpID) srcOffset += (srcLeaf.mPrefixSum >> ((warpID-1) * 9)) & 0x1ff;
            auto dstOffset = dstLeafPtr->firstOffset();
            if (warpID) dstOffset += (dstLeafPtr->mPrefixSum >> ((warpID-1) * 9)) & 0x1ff;

            uint64_t loMask = 1UL << threadInWarpID;
            uint64_t hiMask = 0x100000000UL << threadInWarpID;
            uint64_t loSrcCnt = nanovdb::util::countOn( srcWord & (loMask-1UL) );
            uint64_t hiSrcCnt = nanovdb::util::countOn( srcWord & (hiMask-1UL) );
            uint64_t loDstCnt = nanovdb::util::countOn( dstWord & (loMask-1UL) );
            uint64_t hiDstCnt = nanovdb::util::countOn( dstWord & (hiMask-1UL) );

            if ( loMask & srcWord & dstWord )
                for (int64_t w = 0; w < dim; w++)
                    d_dstData[int64_t(dim)*(offset+int64_t(dstOffset+loDstCnt))+w] = d_srcData[int64_t(dim)*(offset+int64_t(srcOffset+loSrcCnt))+w];
            if ( hiMask & srcWord & dstWord )
                for (int64_t w = 0; w < dim; w++)
                    d_dstData[int64_t(dim)*(offset+int64_t(dstOffset+hiDstCnt))+w] = d_srcData[int64_t(dim)*(offset+int64_t(srcOffset+hiSrcCnt))+w];
        }
    }
};

template <typename BuildT>
struct InjectGridMaskFunctor
{
    // Populates a sidecar of leaf-enumerated bitmasks (associated with a destination grid)
    // with those voxels that are also present in a source grid
    // Intended to be called via nanovdb::util::cuda::lambdaKernel
    // Note: if the source topology is not a subset of the destination topology, the result
    // will be equivalent to the source being taken as the intersection of the two grids

    __device__
    void operator()(
        const std::size_t dstLeafID,
        const typename nanovdb::NanoGrid<BuildT> *d_srcGrid,
        const typename nanovdb::NanoGrid<BuildT> *d_dstGrid,
        typename nanovdb::Mask<3> *d_dstLeafMasks)
    {
        const auto& srcTree = d_srcGrid->tree();
        const auto& dstTree = d_dstGrid->tree();
        const auto& dstLeaf = dstTree.template getFirstNode<0>()[dstLeafID];
        auto& resultMask = d_dstLeafMasks[dstLeafID];
        auto srcLeafPtr = srcTree.root().probeLeaf(dstLeaf.origin());
        if (srcLeafPtr) {
            resultMask = srcLeafPtr->valueMask();
            resultMask &= dstLeaf.valueMask();
        }
        else
            resultMask.setOff();
    }
};

template <typename BuildT, int64_t offset = 0>
struct InjectPredicateToMaskFunctor
{
    // Populates a sidecar of leaf-enumerated bitmasks (associated with a destination grid)
    // with those voxels that are also set to true in a boolean sidecar of values
    // Intended to be called via nanovdb::util::cuda::operatorKernel

    static constexpr int MaxThreadsPerBlock = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__
    void operator()(
        const typename nanovdb::NanoGrid<BuildT> *d_grid,
        const bool *d_predicate,
        typename nanovdb::Mask<3> *d_dstLeafMasks)
    {
        int leafID = blockIdx.x;
        int threadID = threadIdx.x;

        const auto &tree = d_grid->tree();
        const auto &leaf = tree.template getFirstNode<0>()[leafID];
        auto &resultMask = d_dstLeafMasks[leafID];
        if (threadID < nanovdb::Mask<3>::WORD_COUNT)
            resultMask.words()[threadID] = 0UL;
        __syncthreads();
        if (auto n = leaf.data()->getValue(threadID))
            if (d_predicate[int64_t(n)+offset])
                resultMask.setOnAtomic(threadID);
    }
};

} // namespace cuda

} // namespace nanovdb::util

#endif // NANOVDB_UTIL_CUDA_INECTION_CUH_HAS_BEEN_INCLUDED
