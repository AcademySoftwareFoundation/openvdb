// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDHANDLE_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDHANDLE_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>

#include <vector>

namespace nb = nanobind;

namespace pynanovdb {

// Polymorphic host-side `handle.grid(n)`: dispatch on gridType(n) to the
// matching NanoGrid<BuildT> subclass currently bound in Python. Returns
// None when the underlying BuildT is not yet Python-visible (e.g. Boolean,
// Half, Fp16 — they land in Phase 2). The returned object is parented to
// the handle so the handle is kept alive at least as long as the grid.
template<typename BufferT>
inline nb::object pyHostGrid(nb::handle py_handle, uint32_t n)
{
    auto& handle = nb::cast<nanovdb::GridHandle<BufferT>&>(py_handle);
    if (n >= handle.gridCount()) return nb::none();
    switch (handle.gridType(n)) {
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
        case nanovdb::GridType::GridTypeEnum: {                                \
            auto* grid = handle.template grid<T>(n);                           \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)  \
                        : nb::none();                                          \
        }
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
        case nanovdb::GridType::GridTypeEnum: {                                \
            auto* grid = handle.template grid<T>(n);                           \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)  \
                        : nb::none();                                          \
        }
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)              \
        case nanovdb::GridType::GridTypeEnum: {                                \
            auto* grid = handle.template grid<T>(n);                           \
            return grid ? nb::cast(grid, nb::rv_policy::reference, py_handle)  \
                        : nb::none();                                          \
        }
#include "BuildTypes.def"
        default:
            return nb::none();
    }
}

// Free functions splitGrids / mergeGrids exposed at module scope. Templated
// on BufferT so the same definitions work for both the host GridHandle and
// the device GridHandle bindings.
template<typename BufferT> void defineGridHandleUtilities(nb::module_& m)
{
    using HandleT = nanovdb::GridHandle<BufferT>;
    m.def("splitGrids", [](const HandleT& handle) {
        auto handles = nanovdb::splitGrids(handle);
        nb::list out;
        for (auto& h : handles) {
            out.append(std::move(h));
        }
        return out;
    }, nb::arg("handle"),
       "Split a multi-grid handle into a list of single-grid handles, "
       "each owning a freshly-allocated buffer.");
    m.def("mergeGrids", [](nb::list handles) {
        std::vector<HandleT> hs;
        hs.reserve(handles.size());
        for (auto h : handles) {
            hs.emplace_back(nb::cast<HandleT&&>(h));
        }
        return nanovdb::mergeGrids(hs);
    }, nb::arg("handles"),
       "Combine a list of GridHandles into a single multi-grid GridHandle.");
}

template<typename BufferT> nb::class_<nanovdb::GridHandle<BufferT>> defineGridHandle(nb::module_& m, const char* name)
{
    return nb::class_<nanovdb::GridHandle<BufferT>>(m, name)
        .def(nb::init<>())
        .def("reset", &nanovdb::GridHandle<BufferT>::reset)
        .def("size", &nanovdb::GridHandle<BufferT>::bufferSize)
        .def("isEmpty", &nanovdb::GridHandle<BufferT>::isEmpty)
        .def("empty", &nanovdb::GridHandle<BufferT>::empty)
        .def(
            "__bool__",
            [](const nanovdb::GridHandle<BufferT>& handle) { return !handle.empty(); },
            nb::is_operator())
        .def("copy",
             [](const nanovdb::GridHandle<BufferT>& handle) {
                 return handle.template copy<BufferT>();
             },
             "Return a deep copy of this GridHandle backed by a freshly-allocated buffer.")
        .def("grid", &pyHostGrid<BufferT>, nb::arg("n") = 0,
             "Return the n-th grid as a typed Grid subclass selected by "
             "gridType(n), or None if the BuildT is not bound in Python.")
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
