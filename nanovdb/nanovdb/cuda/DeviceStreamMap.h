// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file DeviceStreamMap.h

    \author Ken Museth

    \date October 15, 2024

    \brief nanovdb::cuda::DeviceStreamMap maps device IDs to CUDA streams,
           which is useful for multi-GPU applications.
*/

#ifndef NANOVDB_CUDA_DEVICESTREAMMAP_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICESTREAMMAP_H_HAS_BEEN_INCLUDED

#include <cuda.h>
#include <map>
#include <vector>
#include <nanovdb/util/cuda/Util.h>// for cudaCheck, deviceCount etc

namespace nanovdb {// ================================================================

namespace cuda {// ===================================================================

/// @brief map from a device ID to an associated cuda stream. Useful for multi-GPU applications.
class DeviceStreamMap : public std::map<int, cudaStream_t>
{
     using FuncT = CUresult(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
     FuncT *mFunctPtr;

public:

    enum DeviceType { Any = 0, PeerToPeer = 1, Unified = 3 };

    /// @brief Initiates a map between CUDA device IDs and corresponding streams that satisfy certain constraints.
    ///        All devices should be able to access memory on all the other devices!
    /// @param t Type of device to include in map. Any means all available devices, PeerToPeer means only devices that
    ///          access all another devices are included, and Unified means all devices support unified memory, concurrent access,
    ///          and can be access by all other devices.
    /// @param exclude optional list of device IDs to exclude from the map
    /// @param verbose  0 means quiet, 1 means print if a device is ignores and 2 means print is a device is included
    DeviceStreamMap(DeviceType t = DeviceType::Unified, std::vector<int> exclude = {}, int verbose = 0);

    /// @brief Destructor
    ~DeviceStreamMap();

    /// @brief returns the minimum page size of all the devices in this map
    size_t getMinPageSize() const;

    /// @brief Print information about all the devices included in this map
    void printDevInfo(std::FILE* file = stdout) const {for (auto &p : *this) util::cuda::printDevInfo(p.first, nullptr, file);}

    /// @brief Returns the number of device associated with this map
    int deviceCount() const {return this->size();}

};// DeviceStreamMap

DeviceStreamMap::DeviceStreamMap(DeviceType t, std::vector<int> exclude, int verbose)
{
    std::initializer_list<cudaDeviceAttr> filter = {cudaDevAttrUnifiedAddressing, cudaDevAttrConcurrentManagedAccess};
    const int devCount = util::cuda::deviceCount(), current = util::cuda::currentDevice();
    for (int dev = 0; dev < devCount; ++dev) {
        int check = 1;
        for (auto it=exclude.begin();         check && it!=exclude.end(); ++it) if (dev == *it) check = 0;
        for (auto it=filter.begin(); (t&2) && check && it!=filter.end(); ++it) cudaCheck(cudaDeviceGetAttribute( &check, *it, dev));
        for (auto it= this->begin(); (t&1) && check && it!= this->end(); ++it) cudaCheck(cudaDeviceCanAccessPeer(&check, dev, it->first));
        if (check) {
            cudaCheck(cudaSetDevice(dev));
            cudaStream_t stream;
            cudaCheck(cudaStreamCreate(&stream));
            if (verbose>1) util::cuda::printDevInfo(dev, "Using");
            (*this)[dev] = stream;
        } else if (verbose) util::cuda::printDevInfo(dev, "Ignoring");
    }
    cudaCheck(cudaSetDevice(current));// reset to the previous device

    void* entryPoint = nullptr;
#if CUDART_VERSION >= 13000
    cudaDriverEntryPointQueryResult queryResult;
    cudaCheck(cudaGetDriverEntryPointByVersion("cuMemGetAllocationGranularity", &entryPoint, 13000, cudaEnableDefault, &queryResult));
    NANOVDB_ASSERT(queryResult == cudaDriverEntryPointSuccess);
#elif CUDART_VERSION >= 12000// queryResult argument was added in CUDA 12
    cudaDriverEntryPointQueryResult queryResult;
    cudaCheck(cudaGetDriverEntryPoint("cuMemGetAllocationGranularity", &entryPoint, cudaEnableDefault, &queryResult));
    NANOVDB_ASSERT(queryResult == cudaDriverEntryPointSuccess);
#else
    cudaCheck(cudaGetDriverEntryPoint("cuMemGetAllocationGranularity", &entryPoint, cudaEnableDefault));
#endif
    mFunctPtr = reinterpret_cast<FuncT*>(entryPoint);
}// DeviceStreamMap::DeviceStreamMap

DeviceStreamMap::~DeviceStreamMap()
{
    const int current = util::cuda::currentDevice();
    for (auto& [device, stream] : *this) {
        cudaCheck(cudaSetDevice(device));
        cudaCheck(cudaStreamDestroy(stream));
    }
    cudaCheck(cudaSetDevice(current));// reset to the previous device
}

inline size_t DeviceStreamMap::getMinPageSize() const
{
    NANOVDB_ASSERT(mFunctPtr);
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    size_t minGranularity = 0;
    for (auto it = this->begin(); it!=this->end(); ++it) {
        prop.location.id = it->first;
        size_t granularity = 0;
        (*mFunctPtr)(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (minGranularity < granularity) minGranularity = granularity;
    }
    return minGranularity;
}// DeviceStreamMap::getMinPageSize

}// namespace cuda

}// namespace nanovdb

#endif // end of NANOVDB_CUDA_DEVICESTREAMMAP_H_HAS_BEEN_INCLUDED
