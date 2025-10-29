// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @brief Defines a simple memory pool used to call cub functions that use dynamic temporary storage
///
/// @details See nanovdb/tools/cuda/PointToGrid.cuh and nanovdb/tools/cuda/DistributedPointToGrid.cuh
///          for examples. Also note that this explains the somewhat unusual API with direct access to
///          protected member data.
//
#ifndef NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED

#include <nanovdb/cuda/DeviceResource.h>

namespace nanovdb {

namespace cuda {

template <class Resource>
class TempPool {
public:

    /// @brief Default c-tor of an empty memory pool.
    TempPool() : mData(nullptr), mSize(0), mRequestedSize(0) {}

    /// @brief Destructor
    ~TempPool() {
        mRequestedSize = 0;
        Resource::deallocateAsync(mData, mSize, Resource::DEFAULT_ALIGNMENT, nullptr);
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

    /// @brief Re-allocation of the data managed by this instance. Only has affect if the pool in empty or
    ///        the requested memory is larger than the existing size.
    /// @param stream cuda stream used for asynchronous de-allocation and allocation.
    void reallocate(cudaStream_t stream) {
        if (!mData || mRequestedSize > mSize) {
            Resource::deallocateAsync(mData, mSize, Resource::DEFAULT_ALIGNMENT, stream);
            mData = Resource::allocateAsync(mRequestedSize, Resource::DEFAULT_ALIGNMENT, stream);
            mSize = mRequestedSize;
        }
    }
private:
    void  *mData;
    size_t mSize;
    size_t mRequestedSize;
};// TempPool<Resource> class

using TempDevicePool = TempPool<DeviceResource>;

} // namespace cuda

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED
