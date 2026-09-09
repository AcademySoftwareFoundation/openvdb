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
    // mergeGrids: collect non-owning pointers to each handle and forward to
    // the borrowing nanovdb::mergeGrids overload. The vector-of-handles
    // overload is unusable from a Python list because GridHandle is move-only
    // — building the vector would move from (and silently empty) the caller's
    // Python objects.
    m.def("mergeGrids", [](nb::sequence handles) {
        std::vector<const HandleT*> sources;
        sources.reserve(nb::len(handles));
        for (nb::handle item : handles) {
            sources.push_back(&nb::cast<const HandleT&>(item));
        }
        return nanovdb::mergeGrids<BufferT>(sources.data(), sources.size());
    }, nb::arg("handles"),
       "Combine a list of GridHandles into a single multi-grid GridHandle. "
       "Input handles are read by const reference; the new handle owns a "
       "freshly-allocated buffer and the inputs are left untouched.");
}

template<typename BufferT> nb::class_<nanovdb::GridHandle<BufferT>> defineGridHandle(nb::module_& m, const char* name)
{
    return nb::class_<nanovdb::GridHandle<BufferT>>(m, name,
            "Owns a buffer holding one or more serialized NanoVDB grids. "
            "Construct via nanovdb.tools.create* factories or nanovdb.io.readGrid(s); "
            "access individual grids via handle.grid(n), handle[i], or iteration.")
        .def(nb::init<>(),
            "Construct an empty handle. Use the nanovdb.tools.create* "
            "factories or nanovdb.io.readGrid(s) instead in normal use.")
        .def("reset", &nanovdb::GridHandle<BufferT>::reset,
            "Drop the underlying buffer; the handle becomes empty.")
        .def("size", &nanovdb::GridHandle<BufferT>::bufferSize,
            "Total byte size of the buffer backing this handle (sum of "
            "every grid plus any internal padding).")
        .def("isEmpty", &nanovdb::GridHandle<BufferT>::isEmpty,
            "True iff the handle owns no buffer.")
        .def("empty", &nanovdb::GridHandle<BufferT>::empty,
            "Same as isEmpty(). Retained for parity with the C++ "
            "GridHandle::empty() member.")
        .def(
            "__bool__",
            [](const nanovdb::GridHandle<BufferT>& handle) { return !handle.empty(); },
            nb::is_operator(),
            "True iff the handle owns a non-empty buffer (`not isEmpty()`).")
        .def("copy",
             [](const nanovdb::GridHandle<BufferT>& handle) {
                 return handle.template copy<BufferT>();
             },
             "Return a deep copy of this GridHandle backed by a freshly-allocated buffer.")
        .def("grid", &pyHostGrid<BufferT>, nb::arg("n") = 0,
             nb::keep_alive<0, 1>(),
             "Return the n-th grid as a typed Grid subclass selected by "
             "gridType(n), or None if the BuildT is not bound in Python. "
             "The returned grid keeps this handle alive.")
        .def("__len__", &nanovdb::GridHandle<BufferT>::gridCount,
            "Number of grids stored in this handle (same as gridCount()).")
        .def(
            "__getitem__",
            [](nb::handle py_handle, Py_ssize_t i) {
                auto& handle = nb::cast<nanovdb::GridHandle<BufferT>&>(py_handle);
                const Py_ssize_t count = static_cast<Py_ssize_t>(handle.gridCount());
                if (i < 0) i += count;
                if (i < 0 || i >= count)
                    throw nb::index_error("GridHandle index out of range [-gridCount(), gridCount()).");
                return pyHostGrid<BufferT>(py_handle, static_cast<uint32_t>(i));
            },
            nb::arg("i"),
            nb::keep_alive<0, 1>(),
            "handle[i] -> the i-th grid as a typed Grid subclass (same "
            "dispatch as grid(n)); negative indices count from the end. "
            "Raises IndexError when out of range, which also makes handles "
            "iterable via the sequence protocol (`for grid in handle`). "
            "Grids whose BuildT is not bound in Python are returned as "
            "None, matching grid(n).")
        .def("isPadded", &nanovdb::GridHandle<BufferT>::isPadded,
            "True iff this handle's buffer is aligned past the natural "
            "GridData alignment (used by the I/O code path).")
        .def("gridCount", &nanovdb::GridHandle<BufferT>::gridCount,
            "Number of grids stored in this handle.")
        .def(
            "gridSize",
            [](const nanovdb::GridHandle<BufferT>& handle, uint32_t n) {
                // GridHandle::gridSize(n) indexes mMetaData[n] unchecked, so an
                // out-of-range n (including any n on an empty handle) is UB.
                if (n >= handle.gridCount())
                    throw nb::index_error("gridSize: grid index out of range [0, gridCount()).");
                return handle.gridSize(n);
            },
            nb::arg("n") = 0,
            "Byte size of the n-th grid (without padding).")
        .def(
            "gridType",
            [](const nanovdb::GridHandle<BufferT>& handle, uint32_t n) {
                // GridHandle::gridType(n) indexes mMetaData[n] unchecked, so an
                // out-of-range n (including any n on an empty handle) is UB.
                if (n >= handle.gridCount())
                    throw nb::index_error("gridType: grid index out of range [0, gridCount()).");
                return handle.gridType(n);
            },
            nb::arg("n") = 0,
            "GridType enumerator of the n-th grid (e.g. GridType.Float). "
            "Cheap to query — does not require materializing the grid.")
        .def(
            "gridData",
            [](nanovdb::GridHandle<BufferT>& handle, uint32_t n) {
                // gridData(n) returns nullptr for out-of-range n, but gridSize(n)
                // below indexes mMetaData[n] unchecked — guard both here.
                if (n >= handle.gridCount())
                    throw nb::index_error("gridData: grid index out of range [0, gridCount()).");
                return nb::bytes(handle.gridData(n), handle.gridSize(n));
            },
            nb::arg("n") = 0,
            nb::rv_policy::reference_internal,
            "Raw byte contents of the n-th grid as a Python bytes object. "
            "Useful for hashing or for handing off to non-NanoVDB tooling.")
        .def("write", nb::overload_cast<const std::string&>(&nanovdb::GridHandle<BufferT>::write, nb::const_),
            nb::arg("fileName"),
            "Write every grid in this handle to the given .nvdb file.")
        .def("write", nb::overload_cast<const std::string&, uint32_t>(&nanovdb::GridHandle<BufferT>::write, nb::const_),
            nb::arg("fileName"), nb::arg("n"),
            "Write just the n-th grid in this handle to the given .nvdb file.")
        .def(
            "read", [](nanovdb::GridHandle<BufferT>& handle, const std::string& fileName) { handle.read(fileName); },
            nb::arg("fileName"),
            "Replace this handle's contents with every grid read from the "
            "given .nvdb file.")
        .def(
            "read",
            [](nanovdb::GridHandle<BufferT>& handle, const std::string& fileName, uint32_t n) { handle.read(fileName, n); },
            nb::arg("fileName"),
            nb::arg("n"),
            "Replace this handle's contents with the n-th grid read from "
            "the given .nvdb file.")
        .def(
            "read",
            [](nanovdb::GridHandle<BufferT>& handle, const std::string& fileName, const std::string& gridName) { handle.read(fileName, gridName); },
            nb::arg("fileName"),
            nb::arg("gridName"),
            "Replace this handle's contents with the grid of the given "
            "name read from the .nvdb file.");
}

void defineHostGridHandle(nb::module_& m);
#ifdef NANOVDB_USE_CUDA
void defineDeviceGridHandle(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
