// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDHANDLE_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDHANDLE_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridChecksum.h>  // for tools::updateGridCount

#include <cstring>
#include <vector>

namespace nb = nanobind;

namespace pynanovdb {

// Polymorphic host-side `handle.grid(n)`: dispatch on gridType(n) to the
// matching NanoGrid<BuildT> subclass currently bound in Python. Returns
// None when the underlying BuildT is not bound in this build — see the
// list of bound types in BuildTypes.def. The returned object is parented
// to the handle so the handle is kept alive at least as long as the grid.
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
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum)           \
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
    // mergeGrids: walk the Python sequence by CONST ref to each handle and
    // concatenate buffer bytes into a freshly-allocated output. The original
    // nanovdb::mergeGrids takes a `const std::vector<GridHandle>&`, but
    // building such a vector from a Python list requires moving from each
    // wrapper (GridHandle is move-only) — which would silently empty the
    // caller's `h1`/`h2` Python objects. Instead we inline the merge logic
    // here so we never need to move from the inputs.
    m.def("mergeGrids", [](nb::sequence handles) {
        // Collect const refs so we touch each Python wrapper exactly once.
        std::vector<const HandleT*> sources;
        sources.reserve(nb::len(handles));
        for (nb::handle item : handles) {
            sources.push_back(&nb::cast<const HandleT&>(item));
        }

        uint64_t totalSize = 0;
        uint32_t totalGrids = 0;
        for (const HandleT* h : sources) {
            totalGrids += h->gridCount();
            for (uint32_t n = 0; n < h->gridCount(); ++n) {
                totalSize += h->gridSize(n);
            }
        }

        auto buffer = BufferT::create(totalSize);
        uint8_t* dst = static_cast<uint8_t*>(buffer.data());
        uint32_t writeIndex = 0;
        for (const HandleT* h : sources) {
            const uint8_t* src = static_cast<const uint8_t*>(h->data());
            for (uint32_t n = 0; n < h->gridCount(); ++n) {
                const uint64_t gs = h->gridSize(n);
                std::memcpy(dst, src, gs);
                auto* gd = reinterpret_cast<nanovdb::GridData*>(dst);
                nanovdb::tools::updateGridCount(gd, writeIndex++, totalGrids);
                dst += gs;
                src += gs;
            }
        }
        return HandleT(std::move(buffer));
    }, nb::arg("handles"),
       "Combine a list of GridHandles into a single multi-grid GridHandle. "
       "Input handles are read by const reference; the new handle owns a "
       "freshly-allocated buffer and the inputs are left untouched.");
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
