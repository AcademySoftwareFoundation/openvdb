// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyTree.h"

#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// createNodeManager has one template instantiation per BuildT. We expose a
// single polymorphic `createNodeManager(grid)` that picks the right one
// based on the runtime type of `grid` (any nb::class_-bound NanoGrid<T>)
// and returns the typed NodeManager directly. nb::isinstance is a fast
// type check that avoids the exception-on-mismatch overhead that would
// come from trying nb::cast and catching cast_error for every
// non-matching BuildT.
//
// Lifetime: the C++ NodeManagerHandle that owns the node-index buffer is
// moved to the heap and owned by an nb::capsule; reference_internal
// parents the returned NodeManager to that capsule, so the buffer lives
// exactly as long as the manager (and, transitively, as long as any
// leaf(i)/lower(i)/upper(i) node view, which are reference_internal to
// the manager). The def-level keep_alive<0,1> on createNodeManager below
// additionally ties the manager to the source grid, whose memory the
// nodes point into.
template<typename BuildT>
static nb::object tryCreateNodeManager(nb::handle py_grid)
{
    using GridT = NanoGrid<BuildT>;
    using HandleT = NodeManagerHandle<HostBuffer>;
    if (!nb::isinstance<GridT>(py_grid)) {
        return nb::object();  // sentinel: "not this BuildT, try next"
    }
    auto& grid = nb::cast<GridT&>(py_grid);
    auto* handle = new HandleT(createNodeManager<BuildT, HostBuffer>(grid));
    nb::capsule owner(handle, [](void* p) noexcept {
        delete static_cast<HandleT*>(p);
    });
    // Non-null by construction: the handle was just built for this BuildT.
    NodeManager<BuildT>* mgr = handle->template mgr<BuildT>();
    return nb::cast(mgr, nb::rv_policy::reference_internal, owner);
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
        // The constructed NodeManager stores raw pointers back into the
        // grid; the returned manager must therefore keep the grid (and
        // transitively the GridHandle that owns the grid's buffer) alive.
        nb::keep_alive<0, 1>(),
        "Build and return the typed NodeManager (e.g. FloatNodeManager) "
        "for the given grid. The manager owns its node-index buffer "
        "internally and keeps the source grid (and transitively its "
        "GridHandle) alive for as long as it lives.");
}

} // namespace pynanovdb
