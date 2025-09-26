// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file DeviceMesh.h

    \brief nanovdb::cuda::DeviceMesh encapsulates device IDs, CUDA streams,
           and NCCL communicators in order to facilitate multi-GPU applications.
*/

#ifndef NANOVDB_CUDA_DEVICEMESH_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICEMESH_H_HAS_BEEN_INCLUDED

#include <vector>
#include <algorithm>
#include <thread>

#include <nanovdb/util/cuda/Util.h>
#ifdef NANOVDB_USE_NCCL
#include <nccl.h>
#endif

namespace nanovdb {

namespace cuda {

namespace detail {

/// @brief RAII class that caches/restores the current device at construction/destruction
class DeviceGuard {
    public:
        DeviceGuard() { cudaGetDevice(&deviceId); }
        ~DeviceGuard() { cudaSetDevice(deviceId); }

        /// @{
        /// @brief DeviceGuard is not copyable nor movable
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
        DeviceGuard(DeviceGuard&& other) = delete;
        DeviceGuard& operator=(DeviceGuard&& other) = delete;
        /// @}
    private:
        int deviceId = -1;
};

}

/// @brief POD struct representing a device id and a stream on that device
struct DeviceNode
{
    int id = -1;
    cudaStream_t stream = cudaStream_t(0);
};

/// @brief This class wraps a vector of per-device IDs and CUDA streams and holds inter-device connectivity information and NCCL comms.
class DeviceMesh
{
    using DeviceVecT = std::vector<DeviceNode>;
    using const_iterator = DeviceVecT::const_iterator;
    using size_type = DeviceVecT::size_type;
public:
    /// @brief Constructs a device mesh for all devices on the host and initializes a CUDA stream (and NCCL communicator) for each device
    DeviceMesh();
    /// @brief Destroys the per-device CUDA stream and finalizes and destroys the per-device NCCL communicators
    ~DeviceMesh();
    /// @brief Disallow copy-construction
    DeviceMesh(const DeviceMesh&) = delete;
    /// @brief Move constructor.  Underlying CUDA streams and NCCL communicators are not reinitialized.
    /// @param DeviceMesh instance that will be moved into this DeviceMesh.
    DeviceMesh(DeviceMesh&&) noexcept;
    /// @brief Disallow copy-assignment
    DeviceMesh& operator=(const DeviceMesh&) = delete;
    /// @brief Move assignment. Underlying CUDA streams and NCCL communicators are not reinitialized.
    /// @param DeviceMesh instance that will be moved into this DeviceMesh.
    DeviceMesh& operator=(DeviceMesh&&) noexcept;

    /// @brief Returns the number of devices
    size_type deviceCount() const { return mDeviceNodes.size(); };

    /// @brief Returns a const reference to the DeviceNode for a particular device
    const DeviceNode& operator [](int deviceId) const { return mDeviceNodes[deviceId]; }

    /// @brief Returns an iterator to the first device node
    const_iterator begin() const { return mDeviceNodes.begin(); };
    /// @brief Returns an iterator past the last device node
    const_iterator end() const { return mDeviceNodes.end(); };

#ifdef NANOVDB_USE_NCCL
    /// @brief Returns the NCCL communicator for a particular device
    ncclComm_t comm(int deviceId) const { return mComms[deviceId]; }
#endif

    /// @brief Returns whether or not peer to peer access is supported between two devices.
    bool canAccessPeer(int deviceId, int peerId) const { return mConnectivity[deviceId][peerId] & (1 << cudaDevP2PAttrAccessSupported); }

private:
    /// @brief Returns whether or not a device supports managed memory
    bool hasManagedMemory(int deviceId) const {
        int managedMemoryValue = 0;
        cudaDeviceGetAttribute(&managedMemoryValue, cudaDevAttrManagedMemory, deviceId);
        return static_cast<bool>(managedMemoryValue);
    }

    DeviceVecT mDeviceNodes;

#ifdef NANOVDB_USE_NCCL
    std::vector<ncclComm_t> mComms;
#endif
    std::vector<std::vector<uint32_t>> mConnectivity;
};

inline DeviceMesh::DeviceMesh()
{
    detail::DeviceGuard deviceGuard;

    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount);

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        NANOVDB_ASSERT(hasManagedMemory(deviceId));
        cudaSetDevice(deviceId);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        mDeviceNodes.push_back({deviceId, stream});
    }

#ifdef NANOVDB_USE_NCCL
    mComms.resize(deviceCount);
    ncclCommInitAll(mComms.data(), deviceCount, nullptr);
#endif

    mConnectivity.resize(deviceCount);
    for (const auto& deviceNode : mDeviceNodes) {
        mConnectivity[deviceNode.id].resize(deviceCount);
        for (const auto& peerNode : mDeviceNodes) {
            mConnectivity[deviceNode.id][peerNode.id] = 0;
            if (deviceNode.id != peerNode.id) {
                int peerAccessSupportedValue = 0;
                cudaDeviceGetP2PAttribute(&peerAccessSupportedValue, cudaDevP2PAttrAccessSupported, deviceNode.id, peerNode.id);
                if (peerAccessSupportedValue) {
                    mConnectivity[deviceNode.id][peerNode.id] |= (1 << cudaDevP2PAttrAccessSupported);
                }
            }
        }
    }
}

inline DeviceMesh::DeviceMesh(DeviceMesh&& other) noexcept
    : mDeviceNodes(std::move(other.mDeviceNodes)),
#ifdef NANOVDB_USE_NCCL
      mComms(std::move(other.mComms)),
#endif
      mConnectivity(std::move(other.mConnectivity))
{
}

inline DeviceMesh::~DeviceMesh()
{
    detail::DeviceGuard deviceGuard;

#ifdef NANOVDB_USE_NCCL
    std::for_each(mComms.begin(), mComms.end(), [](ncclComm_t comm) {
        ncclCommFinalize(comm);
        ncclCommDestroy(comm);
    });
#endif

    std::for_each(mDeviceNodes.begin(), mDeviceNodes.end(), [](DeviceNode& deviceNode) {
        cudaSetDevice(deviceNode.id);
        cudaStreamDestroy(deviceNode.stream);
    });
}

inline DeviceMesh& DeviceMesh::operator=(DeviceMesh&& other) noexcept
{
    mDeviceNodes = std::move(other.mDeviceNodes);
#ifdef NANOVDB_USE_NCCL
    mComms = std::move(other.mComms);
#endif
    mConnectivity = std::move(other.mConnectivity);
    return *this;
}

namespace detail {

inline void* queryAllocationGranularityEntryPoint()
{
    void* entryPoint = nullptr;
#if CUDART_VERSION >= 12500 // cudaGetDriverEntryPointByVersion was added in CUDA 12.5 with cudaGetDriverEntryPoint being potentially deprecated
    cudaDriverEntryPointQueryResult queryResult = cudaDriverEntryPointSymbolNotFound;
    cudaCheck(cudaGetDriverEntryPointByVersion("cuMemGetAllocationGranularity", &entryPoint, 12000, cudaEnableDefault, &queryResult));
    NANOVDB_ASSERT(queryResult == cudaDriverEntryPointSuccess);
#elif CUDART_VERSION >= 12000 // queryResult argument was added in CUDA 12
    cudaDriverEntryPointQueryResult queryResult = cudaDriverEntryPointSymbolNotFound;
    cudaCheck(cudaGetDriverEntryPoint("cuMemGetAllocationGranularity", &entryPoint, cudaEnableDefault, &queryResult));
    NANOVDB_ASSERT(queryResult == cudaDriverEntryPointSuccess);
#else
    cudaCheck(cudaGetDriverEntryPoint("cuMemGetAllocationGranularity", &entryPoint, cudaEnableDefault));
#endif
    return entryPoint;
}// queryAllocationGranularityEntryPoint

}

/// @brief Returns the minimum page size (in bytes) across all devices on the system
inline size_t minDevicePageSize(const DeviceMesh& mesh)
{
    using FuncT = CUresult(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
    static FuncT* functPtr = reinterpret_cast<FuncT*>(detail::queryAllocationGranularityEntryPoint());

    NANOVDB_ASSERT(functPtr);
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    size_t minGranularity = 0;
    for (const auto& node : mesh) {
        prop.location.id = node.id;
        size_t granularity = 0;
        (*functPtr)(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (minGranularity < granularity) minGranularity = granularity;
    }
    return minGranularity;
}// minDevicePageSize

} // namespace cuda

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_DEVICEMESH_H_HAS_BEEN_INCLUDED
