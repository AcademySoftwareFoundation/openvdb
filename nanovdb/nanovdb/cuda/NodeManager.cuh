// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/cuda/NodeManager.cuh

    \author Ken Museth

    \date October 3, 2023

    \brief Contains cuda kernels for NodeManager

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_CUDA_NODE_MANAGER_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_NODE_MANAGER_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/util/cuda/Util.h>// for cuda::lambdaKernel
#include <nanovdb/cuda/Buffer.h>// for the resource-aware buffers below
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/NodeManager.h>

namespace nanovdb {

namespace cuda {

namespace detail {

/// @brief Constructs a single-space buffer of @c count elements through
///        @c ref, on @c stream when the resource is stream-ordered.
template<typename BufferT, typename RefT>
inline BufferT makeNodeManagerStorage(cudaStream_t stream, const RefT& ref, uint64_t count)
{
    if constexpr (is_async_resource<typename BufferT::ResourceType>::value) {
        return BufferT(stream, ref, count, noInit);
    } else {
        (void)stream;
        return BufferT(ref, count, noInit);
    }
}

}// namespace detail

/// @brief Construct a NodeManager from a device grid pointer
///
/// @param d_grid device grid pointer whose nodes will be accessed sequentially
/// @param buffer buffer from which to allocate the output handle
/// @param stream cuda stream
/// @return Handle that contains a device NodeManager
template <typename BuildT, typename BufferT = DeviceBuffer>
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, NodeManagerHandle<BufferT>>::type
createNodeManager(const NanoGrid<BuildT> *d_grid,
                  const BufferT& pool = BufferT(),
                  cudaStream_t stream = 0)
{
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    auto buffer = BufferT::create(sizeof(NodeManagerData), &pool, device, stream);
    auto *d_data = (NodeManagerData*)buffer.deviceData();
    Buffer<size_t, DeviceResource> sizeBuf(stream, 1, noInit);
    size_t size = 0u, *d_size = sizeBuf.data();
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        *d_data = NodeManagerData((void*)d_grid);
        *d_size = sizeof(NodeManagerData);
        auto &tree = d_grid->tree();
        if (NodeManager<BuildT>::FIXED_SIZE && d_grid->isBreadthFirst()) {
            d_data->mLinear = uint8_t(1u);
            d_data->mOff[0] = util::PtrDiff(tree.template getFirstNode<0>(), d_grid);
            d_data->mOff[1] = util::PtrDiff(tree.template getFirstNode<1>(), d_grid);
            d_data->mOff[2] = util::PtrDiff(tree.template getFirstNode<2>(), d_grid);
        } else {
            *d_size += sizeof(uint64_t)*tree.totalNodeCount();
        }
    });
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));// the branch below reads the copied size on the host
    if (size > sizeof(NodeManagerData)) {
        auto tmp = BufferT::create(size, &pool, device, stream);// only allocate buffer on the device
        cudaCheck(cudaMemcpyAsync(tmp.deviceData(), buffer.deviceData(), sizeof(NodeManagerData), cudaMemcpyDeviceToDevice, stream));
        buffer = std::move(tmp);
        d_data = reinterpret_cast<NodeManagerData*>(buffer.deviceData());
        util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__ (size_t) {
            auto &tree = d_grid->tree();
            int64_t *ptr0 = d_data->mPtr[0] = reinterpret_cast<int64_t*>(d_data + 1);
            int64_t *ptr1 = d_data->mPtr[1] = d_data->mPtr[0] + tree.nodeCount(0);
            int64_t *ptr2 = d_data->mPtr[2] = d_data->mPtr[1] + tree.nodeCount(1);
            // Performs depth first traversal but breadth first insertion
            for (auto it2 = tree.root().cbeginChild(); it2; ++it2) {
                *ptr2++ = util::PtrDiff(&*it2, d_grid);
                for (auto it1 = it2->cbeginChild(); it1; ++it1) {
                    *ptr1++ = util::PtrDiff(&*it1, d_grid);
                    for (auto it0 = it1->cbeginChild(); it0; ++it0) {
                        *ptr0++ = util::PtrDiff(&*it0, d_grid);
                    }// loop over child nodes of the lower internal node
                }// loop over child nodes of the upper internal node
            }// loop over child nodes of the root node
        });
    }

    return NodeManagerHandle<BufferT>(toGridType<BuildT>(), std::move(buffer));
}// cuda::createNodeManager

/// @brief Construct a NodeManager over a single-space device buffer whose
///        storage allocates through @a resource, which the returned handle
///        borrows (it must outlive the handle) -- the same ownership contract
///        as the builders.
///
/// @param d_grid device grid pointer whose nodes will be accessed sequentially
/// @param resource device-resident memory resource the handle's storage
///        allocates through; a host-accessible resource (e.g. PinnedResource)
///        is excluded, since the result would carry no device NodeManager
/// @param stream cuda stream
/// @return Handle over cuda::Buffer<std::byte, ResourceRef<ResourceT>> that
///         contains a device NodeManager
template <typename BuildT, typename ResourceT>
inline typename util::enable_if<(is_resource<ResourceT>::value || is_async_resource<ResourceT>::value)
                                    && !is_host_accessible_resource<ResourceT>::value,
                                NodeManagerHandle<Buffer<std::byte, ResourceRef<ResourceT>>>>::type
createNodeManager(const NanoGrid<BuildT> *d_grid,
                  ResourceT& resource,
                  cudaStream_t stream = 0)
{
    using RefT    = ResourceRef<ResourceT>;
    using BufferT = Buffer<std::byte, RefT>;
    RefT ref(resource);
    auto buffer = detail::makeNodeManagerStorage<BufferT>(stream, ref, sizeof(NodeManagerData));
    auto *d_data = reinterpret_cast<NodeManagerData*>(buffer.data());
    auto sizeBuf = detail::makeNodeManagerStorage<Buffer<size_t, RefT>>(stream, ref, 1);
    size_t size = 0u, *d_size = sizeBuf.data();
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        *d_data = NodeManagerData((void*)d_grid);
        *d_size = sizeof(NodeManagerData);
        auto &tree = d_grid->tree();
        if (NodeManager<BuildT>::FIXED_SIZE && d_grid->isBreadthFirst()) {
            d_data->mLinear = uint8_t(1u);
            d_data->mOff[0] = util::PtrDiff(tree.template getFirstNode<0>(), d_grid);
            d_data->mOff[1] = util::PtrDiff(tree.template getFirstNode<1>(), d_grid);
            d_data->mOff[2] = util::PtrDiff(tree.template getFirstNode<2>(), d_grid);
        } else {
            *d_size += sizeof(uint64_t)*tree.totalNodeCount();
        }
    });
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&size, d_size, sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));// the branch below reads the copied size on the host
    if (size > sizeof(NodeManagerData)) {
        auto tmp = detail::makeNodeManagerStorage<BufferT>(stream, ref, size);
        cudaCheck(cudaMemcpyAsync(tmp.data(), buffer.data(), sizeof(NodeManagerData), cudaMemcpyDeviceToDevice, stream));
        buffer = std::move(tmp);
        d_data = reinterpret_cast<NodeManagerData*>(buffer.data());
        util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__ (size_t) {
            auto &tree = d_grid->tree();
            int64_t *ptr0 = d_data->mPtr[0] = reinterpret_cast<int64_t*>(d_data + 1);
            int64_t *ptr1 = d_data->mPtr[1] = d_data->mPtr[0] + tree.nodeCount(0);
            int64_t *ptr2 = d_data->mPtr[2] = d_data->mPtr[1] + tree.nodeCount(1);
            // Performs depth first traversal but breadth first insertion
            for (auto it2 = tree.root().cbeginChild(); it2; ++it2) {
                *ptr2++ = util::PtrDiff(&*it2, d_grid);
                for (auto it1 = it2->cbeginChild(); it1; ++it1) {
                    *ptr1++ = util::PtrDiff(&*it1, d_grid);
                    for (auto it0 = it1->cbeginChild(); it0; ++it0) {
                        *ptr0++ = util::PtrDiff(&*it0, d_grid);
                    }// loop over child nodes of the lower internal node
                }// loop over child nodes of the upper internal node
            }// loop over child nodes of the root node
        });
        cudaCheckError();
    }

    return NodeManagerHandle<BufferT>(toGridType<BuildT>(), std::move(buffer));
}// cuda::createNodeManager (single-space)

}// namespace cuda

template <typename BuildT, typename BufferT = cuda::DeviceBuffer>
[[deprecated("Use cuda::createNodeManager instead")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, NodeManagerHandle<BufferT>>::type
cudaCreateNodeManager(const NanoGrid<BuildT> *d_grid,
                      const BufferT& pool = BufferT(),
                      cudaStream_t stream = 0)
{
    return cuda::createNodeManager<BuildT, BufferT>(d_grid, pool, stream);
}

} // namespace nanovdb

#endif // NANOVDB_CUDA_NODE_MANAGER_CUH_HAS_BEEN_INCLUDED
