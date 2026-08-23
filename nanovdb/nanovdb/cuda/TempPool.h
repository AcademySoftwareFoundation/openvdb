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

#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>

#include <cstddef>
#include <cuda_runtime_api.h>

namespace nanovdb {

namespace cuda {

template <class Resource>
class TempPool {
    static_assert(is_async_resource<Resource>::value,
                  "TempPool allocates stream-ordered scratch and requires an AsyncResource");
    // The buffer borrows the pool's resource through a ResourceRef rather than
    // copying it, preserving the pool's contract that all traffic reaches the
    // caller's resource instance (which may be stateful).
    using BufferT = Buffer<std::byte, ResourceRef<Resource>>;
public:

    /// @brief Default c-tor of an empty memory pool that uses the default
    ///        instance of @c Resource for all allocations.
    TempPool() : TempPool(default_resource<Resource>()) {}

    /// @brief C-tor of an empty memory pool that routes all allocations through
    ///        the supplied @c Resource instance.
    /// @param resource resource instance to allocate from; must outlive this pool.
    explicit TempPool(Resource& resource)
        : mResource(&resource)
        , mBuffer(cudaStream_t{0}, ResourceRef<Resource>(resource), 0, noInit)
    {
    }

    /// @brief Returns a non-const void pointer to the data managed by this instance.
    void* data() {return mBuffer.data();}

    /// @brief Returns a non-const reference to the actual size of the data managed by this instance.
    /// @note Returned by reference because cub's two-pass API takes the storage
    ///       size as a size_t&, so this cannot forward Buffer::size() by value.
    size_t& size() {return mSize;}

    /// @brief Returns a non-const reference to the requested size of the data managed by this instance.
    /// @note This requested size should always be less than or smaller than the actual size().
    size_t& requestedSize() {return mRequestedSize;}

    /// @brief Returns the stream that the managed memory was last (re)allocated on,
    ///        i.e. the stream this pool will free on at destruction.
    cudaStream_t stream() const {return mBuffer.stream();}

    /// @brief Re-allocation of the data managed by this instance. Only has affect if the pool in empty or
    ///        the requested memory is larger than the existing size.
    /// @param stream cuda stream used for asynchronous de-allocation and allocation.
    /// @note Scratch is discarded, never resized: preserving a prefix of
    ///       temporary storage would be a wasted copy.
    void reallocate(cudaStream_t stream) {
        if (mBuffer.empty() || mRequestedSize > mSize) {
            mBuffer.destroy(stream);// free the outgrown block on this stream
            mBuffer = BufferT(stream, ResourceRef<Resource>(*mResource), mRequestedSize, noInit);
            mSize = mBuffer.size();
        } else {
            mBuffer.set_stream(stream);// retained so the d-tor frees on the most-recently-used stream
        }
    }
private:
    Resource *mResource;// non-owning; must outlive this pool and its buffer
    BufferT   mBuffer;
    size_t    mSize{0};
    size_t    mRequestedSize{0};
};// TempPool<Resource> class

using TempDevicePool = TempPool<DeviceResource>;

} // namespace cuda

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_TEMPPOOL_H_HAS_BEEN_INCLUDED
