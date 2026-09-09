// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyDeviceBuffer.h"

#include <cstdint>

#include <nanovdb/cuda/DeviceBuffer.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineDeviceBuffer(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    defineDeviceBufferLike<BufferT>(m, "DeviceBuffer")
        .def_static(
            "from_external",
            [](uint64_t size, uintptr_t gpuPtr, uintptr_t cpuPtr) {
                // Wrap externally-managed host + device memory in a NON-OWNING
                // DeviceBuffer (mManaged == 0). The buffer will NOT free either
                // pointer on destruction, upload, or download — the caller
                // retains ownership of both allocations.
                if (gpuPtr == 0)
                    throw nb::value_error(
                        "from_external: gpuPtr must be a non-null device pointer.");
                if (cpuPtr == 0)
                    throw nb::value_error(
                        "from_external: cpuPtr must be a non-null host pointer; the "
                        "externally-managed DeviceBuffer constructor requires both a "
                        "host and a device pointer.");
                return BufferT::create(size,
                                       reinterpret_cast<void*>(cpuPtr),
                                       reinterpret_cast<void*>(gpuPtr));
            },
            "size"_a,
            "gpuPtr"_a,
            "cpuPtr"_a,
            "Wrap externally-managed host and device memory in a NON-OWNING "
            "DeviceBuffer. size is the byte size of both allocations; gpuPtr "
            "and cpuPtr are raw pointers (Python ints). The returned buffer "
            "does NOT take ownership: it will never free either pointer, so "
            "the caller must keep both allocations alive for the buffer's "
            "lifetime. The device pointer is associated with the current CUDA "
            "device.")
        .def("recordUse", &recordUseChecked, "stream"_a, "device"_a = -1,
             kRecordUseDoc);
}

} // namespace pynanovdb

#endif
