// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyTree.h"

#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// Polymorphic mgr() that returns the right typed NodeManager based on the
// handle's stored gridType. Dispatch follows the same X-macro pattern as
// pyHostGrid / pyDeviceGrid; unbound BuildTs return None rather than the
// generic getMgr<BuildT>() ptr that would be reinterpreted.
template<typename BufferT>
static nb::object pyNodeMgr(nb::handle py_self)
{
    using HandleT = NodeManagerHandle<BufferT>;
    auto& handle = nb::cast<HandleT&>(py_self);
    if (!handle.data()) return nb::none();
    // We need to read the stored gridType, but it's private. The public
    // mgr<BuildT>() returns NULL for type mismatch, so iterate by BuildT.
    // The X-macro produces one case per bound BuildT; first non-null wins.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
    if (auto* m = handle.template mgr<T>()) {                                  \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    if (auto* m = handle.template mgr<T>()) {                                  \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)              \
    if (auto* m = handle.template mgr<T>()) {                                  \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum)           \
    if (auto* m = handle.template mgr<T>()) {                                  \
        return nb::cast(m, nb::rv_policy::reference, py_self);                 \
    }
#include "BuildTypes.def"
    return nb::none();
}

void defineNodeManagerHandle(nb::module_& m)
{
    using HandleT = NodeManagerHandle<HostBuffer>;
    nb::class_<HandleT>(m, "NodeManagerHandle",
        "Owns the memory backing a NodeManager. Move-only. "
        "Obtain via nanovdb.createNodeManager(grid).")
        .def("size",
             [](const HandleT& h) { return h.size(); })
        .def(
            "__bool__",
            [](const HandleT& h) { return h.data() != nullptr; },
            nb::is_operator())
        .def("mgr", &pyNodeMgr<HostBuffer>,
             "Return the typed NodeManager for the grid this handle was "
             "built from, or None if the BuildT is not Python-visible.");
}

// createNodeManager has one template instantiation per BuildT. We expose a
// single polymorphic `createNodeManager(grid)` that picks the right one
// based on the runtime type of `grid` (any nb::class_-bound NanoGrid<T>).
template<typename BuildT>
static nb::object tryCreateNodeManager(nb::handle py_grid)
{
    using GridT = NanoGrid<BuildT>;
    GridT* grid = nullptr;
    try {
        grid = &nb::cast<GridT&>(py_grid);
    } catch (const nb::cast_error&) {
        return nb::object();  // sentinel: "not this BuildT, try next"
    }
    return nb::cast(createNodeManager<BuildT, HostBuffer>(*grid));
}

void defineCreateNodeManager(nb::module_& m)
{
    m.def("createNodeManager",
        [](nb::handle py_grid) -> nb::object {
            // Try every bound BuildT; first successful cast wins.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
            if (auto obj = tryCreateNodeManager<T>(py_grid); obj.is_valid()) { \
                return obj;                                                    \
            }
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
            if (auto obj = tryCreateNodeManager<T>(py_grid); obj.is_valid()) { \
                return obj;                                                    \
            }
#define NANOVDB_PY_FOR_EACH_POINT_BUILDT(T, Suffix, GridTypeEnum)              \
            if (auto obj = tryCreateNodeManager<T>(py_grid); obj.is_valid()) { \
                return obj;                                                    \
            }
#define NANOVDB_PY_FOR_EACH_READONLY_BUILDT(T, Suffix, GridTypeEnum)           \
            if (auto obj = tryCreateNodeManager<T>(py_grid); obj.is_valid()) { \
                return obj;                                                    \
            }
#include "BuildTypes.def"
            throw nb::type_error(
                "createNodeManager: argument is not a NanoVDB grid of any "
                "bound BuildT");
        },
        "grid"_a,
        "Build a NodeManager for the given grid, returning a "
        "NodeManagerHandle that owns the underlying buffer. The handle's "
        "mgr() method returns the typed NodeManager.");
}

} // namespace pynanovdb
