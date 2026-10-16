// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyDeviceBuffer.h"
#include "PyValidate.h"

#include <cstdint>

#include <nanovdb/cuda/DeviceBuffer.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineDeviceBuffer(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DualDeviceBuffer;
    defineDeviceBufferLike<BufferT>(m, "DeviceBuffer")
        .def_static(
            "from_external",
            [](uint64_t size, uintptr_t gpuPtr, uintptr_t cpuPtr) {
                // Wrap externally-managed host + device memory in a NON-OWNING
                // DeviceBuffer (mManaged == 0). The buffer will NOT free either
                // pointer on destruction, upload, or download — the caller
                // retains ownership of both allocations.
                requireAlignedBuffer(reinterpret_cast<const void*>(gpuPtr), "from_external", "gpuPtr");
                requireAlignedBuffer(reinterpret_cast<const void*>(cpuPtr), "from_external", "cpuPtr");
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
            "device. Both pointers must be non-null and aligned to "
            "NANOVDB_DATA_ALIGNMENT (32 bytes), otherwise ValueError is raised.")
        .def("recordUse", &recordUseChecked, "stream"_a, "device"_a = -1,
             kRecordUseDoc);
}

} // namespace pynanovdb

#endif
