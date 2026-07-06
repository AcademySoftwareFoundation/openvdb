// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @brief Defines a simple memory pool used to call cub functions that use dynamic temporary storage
///
/// @details See nanovdb/tools/cuda/PointToGrid.cuh and nanovdb/tools/cuda/DistributedPointToGrid.cuh
///          for examples. Also note that this explains the somewhat unusual API with direct access to
///          private member data.
//
#ifndef NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED

#include <nanovdb/cuda/DeviceResource.h>

#include <cstddef>
#include <cuda_runtime_api.h>

namespace nanovdb {

namespace cuda {

template <class Resource>
class TempPool {
public:

    /// @brief Default c-tor of an empty memory pool that uses the default
    ///        instance of @c Resource for all allocations.
    TempPool() : mResource(&default_resource<Resource>()), mData(nullptr), mSize(0), mRequestedSize(0), mStream(nullptr) {}

    /// @brief C-tor of an empty memory pool that routes all allocations through
    ///        the supplied @c Resource instance.
    /// @param resource resource instance to allocate from; must outlive this pool.
    explicit TempPool(Resource& resource) : mResource(&resource), mData(nullptr), mSize(0), mRequestedSize(0), mStream(nullptr) {}

    /// @brief Destructor. Frees the managed memory on the stream of the most
    ///        recent reallocate(), so the stream-ordered free is ordered after
    ///        the work that used the memory (rather than on the null stream).
    ~TempPool() {
        mRequestedSize = 0;
        mResource->deallocate_async(mData, mSize, Resource::DEFAULT_ALIGNMENT, mStream);
        mData = nullptr;
        mSize = 0;
    }

    /// @brief Returns a non-const void pointer to the data managed by this instance.
    void* data() {return mData;}

    /// @brief Returns a non-const reference to the actual size of the data managed by this instance.
    size_t& size() {return mSize;}

    /// @brief Returns a non-const reference to the requested size of the data managed by this instance.
    /// @note This requested size should always be less than or smaller than the actual size().
    size_t& requestedSize() {return mRequestedSize;}

    /// @brief Returns the stream that the managed memory was last (re)allocated on,
    ///        i.e. the stream this pool will free on at destruction.
    cudaStream_t stream() const {return mStream;}

    /// @brief Re-allocation of the data managed by this instance. Only has affect if the pool in empty or
    ///        the requested memory is larger than the existing size.
    /// @param stream cuda stream used for asynchronous de-allocation and allocation.
    void reallocate(cudaStream_t stream) {
        if (!mData || mRequestedSize > mSize) {
            mResource->deallocate_async(mData, mSize, Resource::DEFAULT_ALIGNMENT, stream);
            mData = mResource->allocate_async(mRequestedSize, Resource::DEFAULT_ALIGNMENT, stream);
            mSize = mRequestedSize;
        }
        mStream = stream;// retained so the destructor frees on the most-recently-used stream
    }
private:
    Resource    *mResource;
    void        *mData;
    size_t       mSize;
    size_t       mRequestedSize;
    cudaStream_t mStream;
};// TempPool<Resource> class

using TempDevicePool = TempPool<DeviceResource>;

} // namespace cuda

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED
