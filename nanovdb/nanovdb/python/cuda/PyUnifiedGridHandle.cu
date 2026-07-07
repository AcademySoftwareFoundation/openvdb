// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyUnifiedGridHandle.h"
#include "../PyGridHandle.h"

#include <cstdint>

#include <cuda_runtime.h>

#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/NanoVDB.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace pynanovdb {

// Polymorphic deviceGrid(n) for a unified GridHandle — same dispatch shape as
// pyHostGrid<BufferT> / the DeviceBuffer pyDeviceGrid, returning the grid via
// the unified (managed) pointer. With UnifiedBuffer host and device pointers
// coincide, so this is valid as soon as the handle holds a grid.
static nb::object pyUnifiedDeviceGrid(nb::handle py_handle, uint32_t n)
{
    using BufferT = nanovdb::cuda::UnifiedBuffer;
    auto& handle = nb::cast<nanovdb::GridHandle<BufferT>&>(py_handle);
    if (n >= handle.gridCount()) return nb::none();
    switch (handle.gridType(n)) {
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)              \
        case nanovdb::GridType::GridTypeEnum: {                                 \
            auto* grid = handle.template deviceGrid<T>(n);                      \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)   \
                        : nb::none();                                           \
        }
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
        case nanovdb::GridType::GridTypeEnum: {                                 \
            auto* grid = handle.template deviceGrid<T>(n);                      \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)   \
                        : nb::none();                                           \
        }
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)               \
        case nanovdb::GridType::GridTypeEnum: {                                 \
            auto* grid = handle.template deviceGrid<T>(n);                      \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)   \
                        : nb::none();                                           \
        }
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum)            \
        case nanovdb::GridType::GridTypeEnum: {                                 \
            auto* grid = handle.template deviceGrid<T>(n);                      \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)   \
                        : nb::none();                                           \
        }
#include "../BuildTypes.def"
        default:
            return nb::none();
    }
}

void defineUnifiedGridHandle(nb::module_& m)
{
    using BufferT = nanovdb::cuda::UnifiedBuffer;
    defineGridHandle<BufferT>(m, "UnifiedGridHandle")
        .def("deviceGrid", &pyUnifiedDeviceGrid, "n"_a = 0,
             nb::keep_alive<0, 1>(),
             "Return the n-th grid as a typed Grid subclass selected by "
             "gridType(n), accessed through the unified (managed) pointer, or "
             "None if the BuildT is not bound in Python. The returned grid "
             "keeps this handle alive.")
        .def(
            "deviceUpload",
            [](nanovdb::GridHandle<BufferT>& handle, uintptr_t stream, bool sync) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                handle.deviceUpload(reinterpret_cast<void*>(s), sync);
            },
            "stream"_a = 0, "sync"_a = true,
            "Prefetch the unified buffer to the current device. stream is a raw "
            "CUDA stream handle (Python int; 0 = default stream). If sync is "
            "True the call blocks until the prefetch completes.")
        .def(
            "deviceDownload",
            [](nanovdb::GridHandle<BufferT>& handle, uintptr_t stream, bool sync) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                nb::gil_scoped_release release;
                handle.deviceDownload(reinterpret_cast<void*>(s), sync);
            },
            "stream"_a = 0, "sync"_a = true,
            "Prefetch the unified buffer back to the host. stream is a raw CUDA "
            "stream handle (Python int; 0 = default stream). If sync is True "
            "the call blocks until the prefetch completes.")
        .def(
            "device_ptr",
            [](nanovdb::GridHandle<BufferT>& handle) {
                return reinterpret_cast<uintptr_t>(handle.buffer().deviceData());
            },
            "Raw device (managed) pointer to the base of the whole buffer as a "
            "Python int. For unified memory this equals the host pointer.");
}

} // namespace pynanovdb

#endif
