// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaNodeManager.cuh

    \author Ken Museth

    \date October 3, 2023

    \brief Contains cuda kernels for NodeManager

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_CUDA_NODE_MANAGER_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_NODE_MANAGER_CUH_HAS_BEEN_INCLUDED

#include "CudaUtils.h"// for cudaLambdaKernel
#include "CudaDeviceBuffer.h"
#include "../NodeManager.h"

namespace nanovdb {

/// @brief Construct a NodeManager from a device grid pointer
///
/// @param d_grid device grid pointer whose nodes will be accessed sequentially
/// @param buffer buffer from which to allocate the output handle
/// @param stream cuda stream
/// @return Handle that contains a device NodeManager
template <typename BuildT, typename BufferT = CudaDeviceBuffer>
inline typename enable_if<BufferTraits<BufferT>::hasDeviceDual, NodeManagerHandle<BufferT>>::type
cudaCreateNodeManager(const NanoGrid<BuildT> *d_grid,
                      const BufferT& pool = BufferT(),
                      cudaStream_t stream = 0)
{
    auto buffer = BufferT::create(sizeof(NodeManagerData), &pool, false, stream);
    auto *d_data = (NodeManagerData*)buffer.deviceData();
    size_t size = 0u, *d_size;
    cudaCheck(CUDA_MALLOC((void**)&d_size, sizeof(size_t), stream));
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
#ifdef NANOVDB_USE_NEW_MAGIC_NUMBERS
        *d_data = NodeManagerData{NANOVDB_MAGIC_NODE,   0u, (void*)d_grid, {0u,0u,0u}};
#else
        *d_data = NodeManagerData{NANOVDB_MAGIC_NUMBER, 0u, (void*)d_grid, {0u,0u,0u}};
#endif
        *d_size = sizeof(NodeManagerData);
        auto &tree = d_grid->tree();
        if (NodeManager<BuildT>::FIXED_SIZE && d_grid->isBreadthFirst()) {
            d_data->mLinear = uint8_t(1u);
            d_data->mOff[0] = PtrDiff(tree.template getFirstNode<0>(), d_grid);
            d_data->mOff[1] = PtrDiff(tree.template getFirstNode<1>(), d_grid);
            d_data->mOff[2] = PtrDiff(tree.template getFirstNode<2>(), d_grid);
        } else {
            *d_size += sizeof(uint64_t)*tree.totalNodeCount();
        }
    });
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(CUDA_FREE(d_size, stream));
    if (size > sizeof(NodeManagerData)) {
        auto tmp = BufferT::create(size, &pool, false, stream);// only allocate buffer on the device
        cudaCheck(cudaMemcpyAsync(tmp.deviceData(), buffer.deviceData(), sizeof(NodeManagerData), cudaMemcpyDeviceToDevice, stream));
        buffer = std::move(tmp);
        d_data = reinterpret_cast<NodeManagerData*>(buffer.deviceData());
        cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__ (size_t) {
            auto &tree = d_grid->tree();
            int64_t *ptr0 = d_data->mPtr[0] = reinterpret_cast<int64_t*>(d_data + 1);
            int64_t *ptr1 = d_data->mPtr[1] = d_data->mPtr[0] + tree.nodeCount(0);
            int64_t *ptr2 = d_data->mPtr[2] = d_data->mPtr[1] + tree.nodeCount(1);
            // Performs depth first traversal but breadth first insertion
            for (auto it2 = tree.root().cbeginChild(); it2; ++it2) {
                *ptr2++ = PtrDiff(&*it2, d_grid);
                for (auto it1 = it2->cbeginChild(); it1; ++it1) {
                    *ptr1++ = PtrDiff(&*it1, d_grid);
                    for (auto it0 = it1->cbeginChild(); it0; ++it0) {
                        *ptr0++ = PtrDiff(&*it0, d_grid);
                    }// loop over child nodes of the lower internal node
                }// loop over child nodes of the upper internal node
            }// loop over child nodes of the root node
        });
    }

    return NodeManagerHandle<BufferT>(mapToGridType<BuildT>(), std::move(buffer));
}// cudaCreateNodeManager

} // namespace nanovdb

#endif // NANOVDB_CUDA_NODE_MANAGER_CUH_HAS_BEEN_INCLUDED
