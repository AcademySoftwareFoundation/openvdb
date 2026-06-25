// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyTempPool.h"

#include <cstdint>

#include <cuda_runtime.h>

#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/TempPool.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

void defineTempPool(nb::module_& m)
{
    using DeviceResource = nanovdb::cuda::DeviceResource;
    using TempDevicePool = nanovdb::cuda::TempDevicePool;

    // DeviceResource: a stateless static async allocator over the current CUDA
    // device. Exposed in its current shape only — raw pointers are Python ints.
    nb::class_<DeviceResource>(m, "DeviceResource",
        "Stateless CUDA async allocator: allocateAsync / deallocateAsync issue "
        "cudaMallocAsync / cudaFreeAsync on a stream. Pointers are raw Python "
        "ints. Backs TempDevicePool.")
        .def_ro_static("DEFAULT_ALIGNMENT", &DeviceResource::DEFAULT_ALIGNMENT,
            "Default allocation alignment in bytes (256).")
        .def_static(
            "allocateAsync",
            [](size_t bytes, size_t alignment, uintptr_t stream) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                void* p = DeviceResource::allocateAsync(bytes, alignment, s);
                return reinterpret_cast<uintptr_t>(p);
            },
            "bytes"_a, "alignment"_a = DeviceResource::DEFAULT_ALIGNMENT, "stream"_a = 0,
            "Asynchronously allocate `bytes` of device memory on `stream` and "
            "return the raw device pointer as a Python int. The alignment "
            "argument is accepted for API parity but ignored by cudaMallocAsync.")
        .def_static(
            "deallocateAsync",
            [](uintptr_t ptr, size_t bytes, size_t alignment, uintptr_t stream) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                DeviceResource::deallocateAsync(reinterpret_cast<void*>(ptr),
                                                bytes, alignment, s);
            },
            "ptr"_a, "bytes"_a = 0, "alignment"_a = DeviceResource::DEFAULT_ALIGNMENT,
            "stream"_a = 0,
            "Asynchronously free a device pointer (Python int) on `stream`. The "
            "bytes / alignment arguments are accepted for API parity but ignored "
            "by cudaFreeAsync.");

    // TempDevicePool: a thin pool for CUB temporary storage. It owns a raw
    // device pointer, so it is intentionally non-copyable (the default copy
    // would double-free). Exposed in its current shape only.
    nb::class_<TempDevicePool>(m, "TempDevicePool",
        "Thin pool of CUB temporary device storage backed by DeviceResource. "
        "Owns a raw device pointer (non-copyable). reallocate(stream) grows it "
        "to requestedSize when needed.")
        .def(nb::init<>(),
            "Construct an empty pool (no device allocation yet).")
        .def(
            "data",
            [](TempDevicePool& pool) {
                return reinterpret_cast<uintptr_t>(pool.data());
            },
            "Raw device pointer to the pooled storage as a Python int "
            "(0 if nothing has been allocated yet).")
        .def(
            "size",
            [](TempDevicePool& pool) { return pool.size(); },
            "Currently allocated size of the pool in bytes.")
        .def(
            "requestedSize",
            [](TempDevicePool& pool) { return pool.requestedSize(); },
            "Size in bytes most recently requested for the pool. reallocate() "
            "grows the allocation to this value when it exceeds size().")
        .def(
            "setRequestedSize",
            [](TempDevicePool& pool, size_t value) { pool.requestedSize() = value; },
            "value"_a,
            "Set the requested size (in bytes) used by the next reallocate().")
        .def(
            "reallocate",
            [](TempDevicePool& pool, uintptr_t stream) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                pool.reallocate(s);
            },
            "stream"_a = 0,
            "Reallocate the pool on `stream` (a raw CUDA stream handle, Python "
            "int) if it is empty or requestedSize() exceeds the current size().");
}

} // namespace pynanovdb

#endif
