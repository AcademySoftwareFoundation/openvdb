// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyUnifiedBuffer.h"
#include "PyDeviceBuffer.h"  // for defineDeviceBufferLike (shared interop)

#include <cstdint>

#include <cuda_runtime.h>

#include <nanovdb/cuda/UnifiedBuffer.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

void defineUnifiedBuffer(nb::module_& m)
{
    using BufferT = nanovdb::cuda::UnifiedBuffer;

    // defineDeviceBufferLike gives size() / device_ptr / host_ptr /
    // __cuda_array_interface__ / __dlpack_device__ / __dlpack__ for free. For
    // UnifiedBuffer the managed pointer is valid on host AND device, so
    // host_ptr and device_ptr report the SAME address.
    defineDeviceBufferLike<BufferT>(m, "UnifiedBuffer")
        .def(
            "__init__",
            [](BufferT* self, size_t size) {
                // Allocate a managed page table of `size` bytes (size == capacity).
                nb::gil_scoped_release release;
                new (self) BufferT(size);
            },
            "size"_a,
            "Construct a UnifiedBuffer backed by `size` bytes of CUDA managed "
            "(unified) memory. The same pointer is valid on the host and on "
            "every device.")
        .def(
            "__init__",
            [](BufferT* self, size_t size, int device, uintptr_t stream) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                // The (size, device, stream) ctor allocates and applies a
                // preferred-location advise + prefetch to `device`.
                nb::gil_scoped_release release;
                new (self) BufferT(static_cast<uint64_t>(size), device, s);
            },
            "size"_a,
            "device"_a,
            "stream"_a = 0,
            "Construct a UnifiedBuffer of `size` bytes and set the preferred "
            "location advise plus an async prefetch to `device`. stream is a "
            "raw CUDA stream handle (Python int; 0 = default stream).")
        .def("capacity", &BufferT::capacity,
            "Number of bytes reserved in the managed page table (room for "
            "growth; may exceed size()).")
        .def("empty", &BufferT::empty,
            "True iff this buffer manages no memory.")
        .def("isEmpty", &BufferT::isEmpty,
            "Same as empty(). Retained for parity with the C++ "
            "UnifiedBuffer::isEmpty() member.")
        .def(
            "clear",
            [](BufferT& buf) {
                nb::gil_scoped_release release;
                buf.clear();
            },
            "Free all managed memory and reset this buffer to empty.")
        .def(
            "resize",
            [](BufferT& buf, size_t size, int device) {
                nb::gil_scoped_release release;
                buf.resize(size, device);
            },
            "size"_a, "device"_a = cudaCpuDeviceId,
            "Resize the managed memory block. If the new size fits inside the "
            "current capacity this only redefines size(); otherwise a new page "
            "table is allocated (with a preferred-location advise on `device`) "
            "and the old contents are copied over. device defaults to the host "
            "(cudaCpuDeviceId == -1).")
        .def(
            "advise",
            [](const BufferT& buf, ptrdiff_t byteOffset, size_t size, int device,
               int adv) {
                nb::gil_scoped_release release;
                buf.advise(byteOffset, size, device,
                           static_cast<cudaMemoryAdvise>(adv));
            },
            "byteOffset"_a, "size"_a, "device"_a, "advise"_a,
            "Apply a single cudaMemoryAdvise (passed as its integer enumerator, "
            "e.g. cudaMemAdviseSetPreferredLocation == 3) to the [byteOffset, "
            "byteOffset + size) range for `device` (cudaCpuDeviceId == -1 "
            "selects the host).")
        .def(
            "prefetch",
            [](const BufferT& buf, ptrdiff_t byteOffset, size_t size, int device,
               uintptr_t stream) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                buf.prefetch(byteOffset, size, device, s);
            },
            "byteOffset"_a = 0, "size"_a = 0, "device"_a = cudaCpuDeviceId,
            "stream"_a = 0,
            "Prefetch the [byteOffset, byteOffset + size) range to `device` "
            "(cudaCpuDeviceId == -1 selects the host). size == 0 prefetches all "
            "size() bytes. stream is a raw CUDA stream handle (Python int).")
        .def(
            "deviceUpload",
            [](const BufferT& buf, int device, uintptr_t stream, bool sync) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                buf.deviceUpload(device, s, sync);
            },
            "device"_a = 0, "stream"_a = 0, "sync"_a = false,
            "Prefetch all managed bytes to `device` (legacy DeviceBuffer-compat; "
            "internally a memPrefetchAsync). stream is a raw CUDA stream handle "
            "(Python int); if sync is True the call blocks until the prefetch "
            "completes.")
        .def(
            "deviceDownload",
            [](const BufferT& buf, uintptr_t stream, bool sync) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                buf.deviceDownload(s, sync);
            },
            "stream"_a = 0, "sync"_a = false,
            "Prefetch all managed bytes back to the host (legacy "
            "DeviceBuffer-compat). stream is a raw CUDA stream handle (Python "
            "int); if sync is True the call blocks until the prefetch "
            "completes.")
        .def_static(
            "create",
            [](size_t size) {
                nb::gil_scoped_release release;
                return BufferT::create(size);
            },
            "size"_a,
            "Create a UnifiedBuffer of `size` bytes (size == capacity).")
        .def_static(
            "create",
            [](size_t size, size_t capacity) {
                nb::gil_scoped_release release;
                return BufferT::create(size, capacity);
            },
            "size"_a, "capacity"_a,
            "Create a UnifiedBuffer with `size` used bytes and a managed page "
            "table of `capacity` bytes for future growth.");
}

} // namespace pynanovdb

#endif
