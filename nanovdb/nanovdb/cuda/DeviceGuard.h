// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file DeviceGuard.h

    \brief nanovdb::cuda::DeviceGuard is an RAII class for caching/restoring the
    current device associated with the host thread.
*/

#ifndef NANOVDB_CUDA_DEVICEGUARD_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICEGUARD_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>

namespace nanovdb {

namespace cuda {

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

} // namespace cuda

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_DEVICEGUARD_H_HAS_BEEN_INCLUDED
