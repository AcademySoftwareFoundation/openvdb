// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NANOVDB_CUDA_TEMPDEVICEPOOL_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_TEMPDEVICEPOOL_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>

namespace nanovdb {

namespace cuda {

class TempDevicePool {
public:
    TempDevicePool() : mData(nullptr), mSize(0), mRequestedSize(0) {}
    ~TempDevicePool() {
        mRequestedSize = 0;
        cudaFree(mData);
        mData = nullptr;
        mSize = 0;
    }

    void* data() {
        return mData;
    }

    size_t& size() {
        return mSize;
    }

    size_t& requestedSize() {
        return mRequestedSize;
    }

    void reallocate(cudaStream_t stream) {
        if (!mData || mRequestedSize > mSize) {
            cudaFreeAsync(mData, stream);
            cudaMallocAsync(&mData, mRequestedSize, stream);
            mSize = mRequestedSize;
        }
    }
private:
    void* mData;
    size_t mSize;
    size_t mRequestedSize;
};

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_TEMPDEVICEPOOL_H_HAS_BEEN_INCLUDED
