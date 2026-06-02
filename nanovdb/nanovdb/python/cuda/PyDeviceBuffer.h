// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICEBUFFER_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICEBUFFER_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

#ifdef NANOVDB_USE_CUDA
#include <nanobind/ndarray.h>

#include <cstdint>

#include <cuda_runtime.h>
#endif

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA

/// @brief Bind the device-interop surface (CUDA Array Interface / DLPack, raw
///        device/host pointers, streams) onto a device-buffer-like class.
///
/// @details This is the Phase-B interop hook. It is templated only on the
///          duck-typed buffer surface (size()/data()/deviceData()) so it can
///          be reused for any DeviceBuffer-like type; it must NOT reference
///          DeviceBuffer-specific members. It exposes the whole device buffer
///          as 1-D bytes through the CUDA Array Interface and DLPack so it can
///          be consumed zero-copy by CuPy / PyTorch / Numba, plus the raw
///          device/host pointers as Python ints.
template<typename BufferT>
void addDeviceInterop(nb::class_<BufferT>& cls)
{
    cls.def("size", &BufferT::size, "Total number of bytes managed by this buffer.");

    cls.def(
        "device_ptr",
        [](const BufferT& buf) {
            return reinterpret_cast<uintptr_t>(buf.deviceData());
        },
        "Raw device pointer to the current device's buffer as a Python int "
        "(0 if no device allocation exists yet).");

    cls.def(
        "host_ptr",
        [](const BufferT& buf) {
            return reinterpret_cast<uintptr_t>(buf.data());
        },
        "Raw host pointer to the buffer's host mirror as a Python int "
        "(0 if no host allocation exists).");

    cls.def_prop_ro(
        "__cuda_array_interface__",
        [](const BufferT& buf) {
            // Hand-rolled CUDA Array Interface (version 3) describing the whole
            // device buffer as a 1-D contiguous uint8 array. stream=1 selects
            // the legacy default stream per the CAI v3 spec.
            nb::dict iface;
            iface["shape"] = nb::make_tuple(buf.size());
            iface["typestr"] = "|u1";
            iface["data"] =
                nb::make_tuple(reinterpret_cast<uintptr_t>(buf.deviceData()), false);
            iface["version"] = 3;
            iface["strides"] = nb::none();
            iface["stream"] = 1;
            return iface;
        },
        "CUDA Array Interface (v3) view of the whole device buffer as 1-D "
        "uint8. Lets CuPy / Numba / PyTorch consume the device bytes "
        "zero-copy. Returns a null data pointer until the buffer is populated "
        "on the device.");

    cls.def(
        "__dlpack_device__",
        [](const BufferT&) {
            int device = 0;
            cudaGetDevice(&device);
            // 2 == kDLCUDA.
            return nb::make_tuple(2, device);
        },
        "DLPack device tuple (kDLCUDA, device_id) for the device buffer.");

    cls.def(
        "__dlpack__",
        [](nb::handle self, nb::handle /*stream*/) {
            const BufferT& buf = nb::cast<const BufferT&>(self);
            size_t shape[1] = {static_cast<size_t>(buf.size())};
            // Delegate the capsule construction to nanobind: build a device
            // ndarray view parented to this buffer (keep_alive via owner) and
            // forward to its own __dlpack__ producer.
            nb::ndarray<nb::device::cuda, uint8_t, nb::ndim<1>> arr(
                buf.deviceData(), 1, shape, self);
            // nb::cast of a no-framework device ndarray IS the "dltensor"
            // PyCapsule (nanobind ndarray_export), which is exactly what
            // __dlpack__ must return — so return it directly (do NOT call
            // .attr("__dlpack__") on it; a capsule has no such attribute).
            // ndarray_inc_ref + owner=self keep the device memory alive.
            return nb::cast(arr, nb::rv_policy::reference);
        },
        nb::arg("stream") = nb::none(),
        "DLPack capsule exporting the whole device buffer as 1-D uint8. "
        "Delegates to a nanobind device ndarray view parented to this "
        "buffer.");
}

/// @brief Create an @c nb::class_ for a device-buffer-like type, attach the
///        shared device-interop surface via addDeviceInterop, and return it so
///        callers may chain additional @c .def() bindings.
template<typename BufferT>
nb::class_<BufferT> defineDeviceBufferLike(nb::module_& m, const char* name)
{
    nb::class_<BufferT> cls(m, name,
        "CUDA device-side buffer used to back a DeviceGridHandle. Holds a "
        "host mirror and a device pointer; deviceUpload / deviceDownload on "
        "the handle move bytes between the two.");
    addDeviceInterop(cls);
    return cls;
}

void defineDeviceBuffer(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
