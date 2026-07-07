// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "../PyGridHandle.h"
#include <nanobind/ndarray.h>

#include <cstdint>

#include <cuda_runtime.h>

#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/NanoVDB.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// Device-side polymorphic deviceGrid(n) — same dispatch shape as
// pyHostGrid<BufferT> in PyGridHandle.h, but returns the device pointer.
// gridType(n) is read from the host-side GridData header (the handle keeps
// a host mirror), so this works whether or not the grid has been uploaded.
// Returns None if the device-side grid is null (i.e. no deviceUpload yet)
// or the BuildT is not Python-visible.
static nb::object pyDeviceGrid(nb::handle py_handle, uint32_t n)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    auto& handle = nb::cast<GridHandle<BufferT>&>(py_handle);
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

void defineDeviceGridHandle(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    defineGridHandle<BufferT>(m, "DeviceGridHandle")
        .def(
            "__init__",
            [](GridHandle<BufferT>&                                 handle,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cpu>  cpu_t,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cuda> cuda_t) {
                assert(cpu_t.size() == cuda_t.size());
                BufferT buffer(cpu_t.size() * sizeof(uint32_t), cpu_t.data(), cuda_t.data());
                new (&handle) GridHandle<BufferT>(std::move(buffer));
            },
            "cpu_t"_a.noconvert(),
            "cuda_t"_a.noconvert(),
            "Construct a DeviceGridHandle that wraps an existing pair of "
            "host and device uint32 arrays of equal length.")
        .def("deviceGrid", &pyDeviceGrid, "n"_a = 0,
             nb::keep_alive<0, 1>(),
             "Return the n-th device-resident grid as a typed Grid subclass "
             "selected by gridType(n), or None if the BuildT is not bound in "
             "Python or the device copy has not been uploaded yet. The "
             "returned grid keeps this handle alive.")
        .def(
            "deviceUpload",
            [](GridHandle<BufferT>& handle, uintptr_t stream, bool sync) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                // Use the current-device overload (void*, bool) — NOT the
                // (int device, void*, bool) form — so the targeted device
                // matches deviceData()/CAI/__dlpack__ (which use cudaGetDevice).
                handle.deviceUpload(reinterpret_cast<void*>(s), sync);
            },
            "stream"_a = 0, "sync"_a = true,
            "Copy the host-side buffer to the device. stream is a raw CUDA "
            "stream handle (Python int; 0 = default stream). If sync is True "
            "the call blocks until the transfer completes.")
        .def(
            "deviceDownload",
            [](GridHandle<BufferT>& handle, uintptr_t stream, bool sync) {
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                // Current-device overload, matching deviceData() (see deviceUpload).
                handle.deviceDownload(reinterpret_cast<void*>(s), sync);
            },
            "stream"_a = 0, "sync"_a = true,
            "Copy the device-side buffer back to the host. stream is a raw "
            "CUDA stream handle (Python int; 0 = default stream). If sync is "
            "True the call blocks until the transfer completes.")
        .def(
            "device_ptr",
            [](GridHandle<BufferT>& handle) {
                return reinterpret_cast<uintptr_t>(handle.buffer().deviceData());
            },
            "Raw device pointer to the base of the whole device buffer as a "
            "Python int (0 if the handle has not been uploaded to the device "
            "yet).")
        .def_prop_ro(
            "__cuda_array_interface__",
            [](GridHandle<BufferT>& handle) {
                // CUDA Array Interface (v3) over the whole device buffer as a
                // 1-D contiguous uint8 array. stream=1 selects the legacy
                // default stream per the CAI v3 spec.
                nb::dict iface;
                iface["shape"] = nb::make_tuple(handle.buffer().size());
                iface["typestr"] = "|u1";
                iface["data"] = nb::make_tuple(
                    reinterpret_cast<uintptr_t>(handle.buffer().deviceData()), false);
                iface["version"] = 3;
                iface["strides"] = nb::none();
                iface["stream"] = 1;
                return iface;
            },
            "CUDA Array Interface (v3) view of the whole device buffer as 1-D "
            "uint8 — lets CuPy / Numba / PyTorch consume the serialized grid "
            "bytes zero-copy. Returns a null data pointer until deviceUpload.")
        .def(
            "__dlpack_device__",
            [](GridHandle<BufferT>&) {
                int device = 0;
                cudaGetDevice(&device);
                return nb::make_tuple(2, device);  // 2 == kDLCUDA
            },
            "DLPack device tuple (kDLCUDA, device_id) for the device buffer.")
        .def(
            "__dlpack__",
            [](nb::handle self, nb::handle /*stream*/) {
                auto& handle = nb::cast<GridHandle<BufferT>&>(self);
                size_t shape[1] = {static_cast<size_t>(handle.buffer().size())};
                nb::ndarray<nb::device::cuda, uint8_t, nb::ndim<1>> arr(
                    handle.buffer().deviceData(), 1, shape, self);
                // nb::cast of a no-framework device ndarray IS the "dltensor"
                // capsule (what __dlpack__ must return); return it directly.
                return nb::cast(arr, nb::rv_policy::reference);
            },
            "stream"_a = nb::none(),
            "DLPack capsule exporting the whole device buffer as 1-D uint8, "
            "parented to this handle.")
        .def_static(
            "from_buffer",
            [](BufferT& buffer) {
                // Consumes (moves from) the buffer; the GridHandle ctor peeks
                // the GridData header (host side if present, else a D2H copy of
                // the device side) and throws std::runtime_error if it is not a
                // valid grid.
                return GridHandle<BufferT>(std::move(buffer));
            },
            "buffer"_a,
            "Build a DeviceGridHandle that takes ownership of a DeviceBuffer. "
            "The buffer is MOVED FROM (left empty), and its first GridData "
            "header is validated — a RuntimeError is raised if it does not hold "
            "a valid NanoVDB grid. Pair with DeviceBuffer.from_external to wrap "
            "externally-managed device/host memory zero-copy.");
    // NOTE: defineGridHandleUtilities<BufferT> intentionally NOT called for
    // DeviceBuffer. Registering nanovdb.splitGrids / nanovdb.mergeGrids as a
    // second overload taking a DeviceGridHandle list conflicts with the host
    // overload because both signatures take nb::list, and nanobind's
    // overload resolution can't disambiguate by element type — it picks one
    // and the inner cast fails with std::bad_cast. A properly typed device
    // variant (with its own name, or strongly-typed std::vector<HandleT>
    // args via nanobind/stl/vector.h) can land later if it's needed.
}

} // namespace pynanovdb

#endif
