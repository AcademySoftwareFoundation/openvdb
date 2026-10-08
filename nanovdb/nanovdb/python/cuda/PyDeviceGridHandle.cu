// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyGridHandle.h"
#include "PyDeviceBuffer.h"
#include "PyValidate.h"  // for recordUseChecked / kRecordUseDoc
#include <nanobind/ndarray.h>

#include <cstdint>

#include <cuda_runtime.h>

#include <nanovdb/cuda/DeviceBuffer.h>
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
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cpu>  cpuT,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cuda> cudaT) {
                if (cpuT.size() != cudaT.size())
                    throw nb::value_error("DeviceGridHandle: cpuT and cudaT must have the same length.");
                requireAlignedBuffer(cpuT.data(), "DeviceGridHandle", "cpuT");
                requireAlignedBuffer(cudaT.data(), "DeviceGridHandle", "cudaT");
                BufferT buffer(cpuT.size() * sizeof(uint32_t), cpuT.data(), cudaT.data());
                new (&handle) GridHandle<BufferT>(std::move(buffer));
            },
            "cpuT"_a.noconvert(),
            "cudaT"_a.noconvert(),
            // The DualDeviceBuffer is non-owning, so the handle must keep
            // both source arrays alive for as long as it exists.
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>(),
            "Construct a DeviceGridHandle that wraps an existing pair of host "
            "and device uint32 arrays of equal length. The handle does not "
            "copy or own either allocation: it keeps both arrays alive for "
            "its own lifetime, reads the host grid in place, and moves bytes "
            "between the two with deviceUpload()/deviceDownload(). Both data "
            "pointers must be aligned to NANOVDB_DATA_ALIGNMENT (32 bytes); a "
            "plain NumPy allocation is usually only 16-byte aligned and raises "
            "ValueError. Raises RuntimeError if the host bytes do not form a "
            "valid grid.")
        .def("deviceGrid", &pyDeviceGrid, "n"_a = 0,
             nb::keep_alive<0, 1>(),
             "Return the n-th device-resident grid as a typed Grid subclass "
             "selected by gridType(n), or None if the BuildT is not bound in "
             "Python or the device copy has not been uploaded yet. The "
             "returned grid keeps this handle alive.")
        .def(
            "deviceUpload",
            [](GridHandle<BufferT>& handle, uintptr_t stream, bool sync) {
                requireHostCopy(handle.buffer().data(), "deviceUpload");
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                // Use the current-device overload (void*, bool) — NOT the
                // (int device, void*, bool) form — so the targeted device
                // matches deviceData()/CAI/__dlpack__ (which use cudaGetDevice).
                handle.deviceUpload(reinterpret_cast<void*>(s), sync);
            },
            "stream"_a = 0, "sync"_a = true,
            "Copy the host-side buffer to the device, allocating the device "
            "copy on first use. stream is a raw CUDA stream handle (Python "
            "int; 0 = default stream). If sync is True the call blocks until "
            "the transfer completes. Raises ValueError if the handle has no "
            "host copy (e.g. a grid built on the device by tools.cuda).")
        .def(
            "deviceDownload",
            [](GridHandle<BufferT>& handle, uintptr_t stream, bool sync) {
                requireDeviceCopy(deviceDataOrNull(handle.buffer()), "deviceDownload");
                cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
                // Current-device overload, matching deviceData() (see deviceUpload).
                handle.deviceDownload(reinterpret_cast<void*>(s), sync);
            },
            "stream"_a = 0, "sync"_a = true,
            "Copy the device-side buffer back to the host, allocating the host "
            "copy on first use. stream is a raw CUDA stream handle (Python "
            "int; 0 = default stream). If sync is True the call blocks until "
            "the transfer completes. Raises ValueError if the handle has no "
            "device copy yet (e.g. a handle from tools.cuda.createLevelSetSphere "
            "before deviceUpload()).")
        .def(
            "device_ptr",
            [](GridHandle<BufferT>& handle) {
                return reinterpret_cast<uintptr_t>(deviceDataOrNull(handle.buffer()));
            },
            "Raw device pointer to the base of the whole device buffer as a "
            "Python int (0 if the handle has not been uploaded to the device "
            "yet). Work you enqueue against this pointer on a non-blocking "
            "stream is invisible to the buffer's lifetime tracking: call "
            "recordUse(stream) afterwards, or synchronize before the handle "
            "is destroyed.")
        .def(
            "recordUse",
            [](GridHandle<BufferT>& handle, uintptr_t stream, int device) {
                recordUseChecked(handle.buffer(), stream, device);
            },
            "stream"_a, "device"_a = -1,
            kRecordUseDoc)
        .def_prop_ro(
            "__cuda_array_interface__",
            [](GridHandle<BufferT>& handle) {
                // CUDA Array Interface (v3) over the whole device buffer as a
                // 1-D contiguous uint8 array. stream=1 selects the legacy
                // default stream per the CAI v3 spec — make that claim true by
                // ordering the legacy default stream after the buffer's tracked
                // prior uses (async uploads, recordUse'd kernels).
                orderPriorUsesBefore(handle.buffer(), cudaStream_t(0), 0);
                nb::dict iface;
                iface["shape"] = nb::make_tuple(handle.buffer().size());
                iface["typestr"] = "|u1";
                iface["data"] = nb::make_tuple(
                    reinterpret_cast<uintptr_t>(deviceDataOrNull(handle.buffer())), false);
                iface["version"] = 3;
                iface["strides"] = nb::none();
                iface["stream"] = 1;
                return iface;
            },
            "CUDA Array Interface (v3) view of the whole device buffer as 1-D "
            "uint8 — lets CuPy / Numba / PyTorch consume the serialized grid "
            "bytes zero-copy. Returns a null data pointer until deviceUpload. "
            "After enqueuing work on this view from a non-blocking stream, "
            "call recordUse(stream) so the buffer's device free is ordered "
            "after it.")
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
            [](nb::handle self, nb::handle stream) {
                auto& handle = nb::cast<GridHandle<BufferT>&>(self);
                // Honor the consumer-provided stream per the DLPack protocol:
                // order it after the buffer's tracked prior uses so the
                // consumer cannot read a partially-uploaded buffer.
                cudaStream_t consumer;
                if (resolveDlpackStream(stream, consumer))
                    orderPriorUsesBefore(handle.buffer(), consumer, 0);
                size_t shape[1] = {static_cast<size_t>(handle.buffer().size())};
                nb::ndarray<nb::device::cuda, uint8_t, nb::ndim<1>> arr(
                    deviceDataOrNull(handle.buffer()), 1, shape, self);
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
    // splitGrids / mergeGrids are host-only: defineGridHandleUtilities is
    // registered for HostBuffer alone. A DeviceGridHandle overload cannot
    // share the module-level name because both would take a Python sequence
    // and nanobind cannot pick an overload by element type. The host
    // mergeGrids raises a TypeError naming this limitation when it is handed
    // a DeviceGridHandle.
}

} // namespace pynanovdb

#endif
