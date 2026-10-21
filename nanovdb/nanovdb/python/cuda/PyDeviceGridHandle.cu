// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyDeviceGridHandle.h"
#include "../PyGridHandle.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <stdexcept> // for std::invalid_argument

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineDeviceGridHandle(nb::module_& m)
{
    using HandleT = DeviceGridHandle;
    nb::class_<HandleT>(m, "DeviceGridHandle")
        .def(nb::init<>())
        .def(
            "__init__",
            [](HandleT&                                             handle,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>  cpu_t,
               nb::ndarray<uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cuda> cuda_t) {
                if (cpu_t.size() != cuda_t.size())
                    throw std::invalid_argument("DeviceGridHandle: cpu_t and cuda_t must have the same size");
                new (&handle) HandleT(cpu_t.data(), cuda_t.data(), cpu_t.size() * sizeof(uint32_t));
            },
            "cpu_t"_a.noconvert(),
            "cuda_t"_a.noconvert(),
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>(),
            "Construct a DeviceGridHandle that views an existing pair of contiguous host and device arrays of the same size, "
            "aligned to 32 bytes, the host one holding valid grids. The handle keeps both arrays alive.")
        .def("reset", &HandleT::reset)
        .def("size", &HandleT::bufferSize)
        .def("isEmpty", &HandleT::empty)
        .def("empty", &HandleT::empty)
        .def(
            "__bool__", [](const HandleT& handle) { return !handle.empty(); }, nb::is_operator())
        .def("floatGrid", &HandleT::grid<float>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("doubleGrid", &HandleT::grid<double>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("int32Grid", &HandleT::grid<int32_t>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("vec3fGrid", &HandleT::grid<Vec3f>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("rgba8Grid", &HandleT::grid<math::Rgba8>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("isPadded", &HandleT::isPadded)
        .def("gridCount", &HandleT::gridCount)
        .def("gridSize", &HandleT::gridSize, "n"_a = 0)
        .def("gridType", &HandleT::gridType, "n"_a = 0)
        .def(
            "gridData",
            [](const HandleT& handle, uint32_t n) {
                handle.checkIndex(n);
                return handle.withHost("gridData", [n](const auto& h) { return nb::bytes(h.gridData(n), h.gridSize(n)); });
            },
            "n"_a = 0)
        .def(
            "write",
            [](const HandleT& handle, const std::string& fileName) { handle.withHost("write", [&](const auto& h) { h.write(fileName); }); },
            "fileName"_a)
        .def(
            "write",
            [](const HandleT& handle, const std::string& fileName, uint32_t n) {
                handle.withHost("write", [&](const auto& h) { h.write(fileName, n); });
            },
            "fileName"_a,
            "n"_a)
        .def(
            "read",
            [](HandleT& handle, const std::string& fileName) {
                GridHandle<HostBuffer> host;
                host.read(fileName);
                handle = HandleT(host);
            },
            "fileName"_a)
        .def(
            "read",
            [](HandleT& handle, const std::string& fileName, uint32_t n) {
                GridHandle<HostBuffer> host;
                host.read(fileName, n);
                handle = HandleT(host);
            },
            "fileName"_a,
            "n"_a)
        .def(
            "read",
            [](HandleT& handle, const std::string& fileName, const std::string& gridName) {
                GridHandle<HostBuffer> host;
                host.read(fileName, gridName);
                handle = HandleT(host);
            },
            "fileName"_a,
            "gridName"_a)
        .def("deviceFloatGrid", &HandleT::deviceGrid<float>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceDoubleGrid", &HandleT::deviceGrid<double>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceInt32Grid", &HandleT::deviceGrid<int32_t>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceVec3fGrid", &HandleT::deviceGrid<Vec3f>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceRGBA8Grid", &HandleT::deviceGrid<math::Rgba8>, "n"_a = 0, nb::rv_policy::reference_internal)
        .def("deviceUpload", &HandleT::upload, "sync"_a = true)
        .def("deviceDownload", &HandleT::download, "sync"_a = true);
}

} // namespace pynanovdb

#endif
