// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICEBUFFER_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICEBUFFER_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA

/// @brief Bind the device-interop surface (CUDA Array Interface / DLPack, raw
///        device/host pointers, streams) onto a device-buffer-like class.
///
/// @details This is the Phase-B interop hook. It is templated only on the
///          duck-typed buffer surface (size()/data()/deviceData()) so it can
///          be reused for any DeviceBuffer-like type; it must NOT reference
///          DeviceBuffer-specific members. For Phase A this is a minimal seam:
///          it binds only the trivial @c size() accessor. The CUDA Array
///          Interface, DLPack, raw pointer, and stream bindings land in Phase B.
template<typename BufferT>
void addDeviceInterop(nb::class_<BufferT>& cls)
{
    cls.def("size", &BufferT::size, "Total number of bytes managed by this buffer.");
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
