// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDHANDLE_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDHANDLE_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <nanovdb/GridHandle.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BufferT> nb::class_<nanovdb::GridHandle<BufferT>> defineGridHandle(nb::module_& m, const char* name)
{
    return nb::class_<nanovdb::GridHandle<BufferT>>(m, name)
        .def(nb::init<>())
        .def("reset", &nanovdb::GridHandle<BufferT>::reset)
        .def("size", &nanovdb::GridHandle<BufferT>::bufferSize)
        .def("isEmpty", &nanovdb::GridHandle<BufferT>::isEmpty)
        .def("empty", &nanovdb::GridHandle<BufferT>::empty)
        .def(
            "__bool__", [](const nanovdb::GridHandle<BufferT>& handle) { handle.empty(); }, nb::is_operator())
        .def("floatGrid", nb::overload_cast<uint32_t>(&nanovdb::GridHandle<BufferT>::template grid<float>), nb::arg("n") = 0, nb::rv_policy::reference_internal)
        .def("doubleGrid",
             nb::overload_cast<uint32_t>(&nanovdb::GridHandle<BufferT>::template grid<double>),
             nb::arg("n") = 0,
             nb::rv_policy::reference_internal)
        .def("int32Grid",
             nb::overload_cast<uint32_t>(&nanovdb::GridHandle<BufferT>::template grid<int32_t>),
             nb::arg("n") = 0,
             nb::rv_policy::reference_internal)
        .def("vec3fGrid",
             nb::overload_cast<uint32_t>(&nanovdb::GridHandle<BufferT>::template grid<nanovdb::Vec3f>),
             nb::arg("n") = 0,
             nb::rv_policy::reference_internal)
        .def("rgba8Grid",
             nb::overload_cast<uint32_t>(&nanovdb::GridHandle<BufferT>::template grid<nanovdb::math::Rgba8>),
             nb::arg("n") = 0,
             nb::rv_policy::reference_internal)
        .def("isPadded", &nanovdb::GridHandle<BufferT>::isPadded)
        .def("gridCount", &nanovdb::GridHandle<BufferT>::gridCount)
        .def("gridSize", &nanovdb::GridHandle<BufferT>::gridSize, nb::arg("n") = 0)
        .def("gridType", &nanovdb::GridHandle<BufferT>::gridType, nb::arg("n") = 0)
        .def(
            "gridData",
            [](nanovdb::GridHandle<BufferT>& handle, uint32_t n) { return nb::bytes(handle.gridData(n), handle.gridSize(n)); },
            nb::arg("n") = 0,
            nb::rv_policy::reference_internal)
        .def("write", nb::overload_cast<const std::string&>(&nanovdb::GridHandle<BufferT>::write, nb::const_), nb::arg("fileName"))
        .def("write", nb::overload_cast<const std::string&, uint32_t>(&nanovdb::GridHandle<BufferT>::write, nb::const_), nb::arg("fileName"), nb::arg("n"))
        .def(
            "read", [](nanovdb::GridHandle<BufferT>& handle, const std::string& fileName) { handle.read(fileName); }, nb::arg("fileName"))
        .def(
            "read",
            [](nanovdb::GridHandle<BufferT>& handle, const std::string& fileName, uint32_t n) { handle.read(fileName, n); },
            nb::arg("fileName"),
            nb::arg("n"))
        .def(
            "read",
            [](nanovdb::GridHandle<BufferT>& handle, const std::string& fileName, const std::string& gridName) { handle.read(fileName, gridName); },
            nb::arg("fileName"),
            nb::arg("gridName"));

}

void defineHostGridHandle(nb::module_& m);
#ifdef NANOVDB_USE_CUDA
void defineDeviceGridHandle(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
