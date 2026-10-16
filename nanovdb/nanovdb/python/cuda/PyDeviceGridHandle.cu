// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "../PyGridHandle.h"
#include <nanobind/ndarray.h>

#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/NanoVDB.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// Device-side polymorphic deviceGrid(n): the device analogue of pyHostGrid in
// PyGridHandle.h. gridType(n) is read from the host-side GridData header (the
// handle keeps a host mirror), so this works whether or not the grid has been
// uploaded. Returns None if the device-side grid is null (no deviceUpload yet)
// or the BuildT is not Python-visible.
struct PyDeviceGridOp
{
    using BufferT = nanovdb::cuda::DualDeviceBuffer;
    template<typename BuildT>
    static nb::object known(nb::handle py_handle, GridHandle<BufferT>& handle, uint32_t n)
    {
        auto* grid = handle.template deviceGrid<BuildT>(n);
        return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle) : nb::none();
    }
    static nb::object unknown(nb::handle, GridHandle<BufferT>&, uint32_t) { return nb::none(); }
};

static nb::object pyDeviceGrid(nb::handle py_handle, uint32_t n)
{
    auto& handle = nb::cast<GridHandle<PyDeviceGridOp::BufferT>&>(py_handle);
    if (n >= handle.gridCount()) return nb::none();
    return callPyBuildT<PyDeviceGridOp>(handle.gridType(n), py_handle, handle, n);
}

void defineDeviceGridHandle(nb::module_& m)
{
    using BufferT = nanovdb::cuda::DualDeviceBuffer;
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
            "deviceUpload", [](GridHandle<BufferT>& handle, bool sync) { handle.deviceUpload(nullptr, sync); }, "sync"_a = true,
            "Copy the host-side buffer to the device. If sync is True the "
            "call blocks until the transfer completes.")
        .def(
            "deviceDownload", [](GridHandle<BufferT>& handle, bool sync) { handle.deviceDownload(nullptr, sync); }, "sync"_a = true,
            "Copy the device-side buffer back to the host. If sync is True "
            "the call blocks until the transfer completes.");
    // splitGrids / mergeGrids are host-only: defineGridHandleUtilities is
    // registered for HostBuffer alone. A DeviceGridHandle overload cannot
    // share the module-level name because both would take a Python sequence
    // and nanobind cannot pick an overload by element type. The host
    // mergeGrids raises a TypeError naming this limitation when it is handed
    // a DeviceGridHandle.
}

} // namespace pynanovdb

#endif
