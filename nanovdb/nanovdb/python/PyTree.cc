// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyTree.h"
#include "PyBuildTypes.h"

#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// createNodeManager has one template instantiation per BuildT. We expose a
// single polymorphic `createNodeManager(grid)` that dispatches on the grid's
// GridType and returns the typed NodeManager directly.
//
// Lifetime: the C++ NodeManagerHandle that owns the node-index buffer is
// moved to the heap and owned by an nb::capsule; reference_internal
// parents the returned NodeManager to that capsule, so the buffer lives
// exactly as long as the manager (and, transitively, as long as any
// leaf(i)/lower(i)/upper(i) node view, which are reference_internal to
// the manager). The def-level keep_alive<0,1> on createNodeManager below
// additionally ties the manager to the source grid, whose memory the
// nodes point into.
struct CreateNodeManagerOp
{
    template<typename BuildT>
    static nb::object known(GridData* gridData)
    {
        using HandleT = NodeManagerHandle<HostBuffer>;
        auto& grid = *static_cast<NanoGrid<BuildT>*>(gridData);
        auto* handle = new HandleT(createNodeManager<BuildT, HostBuffer>(grid));
        nb::capsule owner(handle, [](void* p) noexcept {
            delete static_cast<HandleT*>(p);
        });
        // Non-null by construction: the handle was just built for this BuildT.
        NodeManager<BuildT>* mgr = handle->template mgr<BuildT>();
        return nb::cast(mgr, nb::rv_policy::reference_internal, owner);
    }
    static nb::object unknown(GridData*)
    {
        throw nb::type_error(
            "createNodeManager: the grid's BuildT is not bound in Python");
    }
};

void defineCreateNodeManager(nb::module_& m)
{
    m.def("createNodeManager",
        [](GridData* gridData) -> nb::object {
            return callPyBuildT<CreateNodeManagerOp>(gridData->mGridType, gridData);
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
