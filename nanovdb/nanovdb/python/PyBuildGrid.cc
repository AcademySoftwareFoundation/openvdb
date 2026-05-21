// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyBuildGrid.h"

#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <array>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

// Bind nanovdb::tools::build::Grid<BuildT> together with its
// ValueAccessor<BuildT> and Tree<BuildT>::WriteAccessor proxies, plus a
// .to_nanovdb() shortcut that bakes the build grid into a host NanoGrid
// via tools::createNanoGrid.
//
// One instantiation per writable BuildT in BuildTypes.def (scalars +
// vectors). All three classes live under the nanovdb.tools.build submodule;
// the Python class names mirror the existing typed-grid naming so
// nanovdb.tools.build.FloatGrid is the mutable counterpart of the read-only
// nanovdb.FloatGrid.
template<typename BuildT>
static void defineBuildGrid(nb::module_& m,
                            const char*  gridName,
                            const char*  valueAccName,
                            const char*  writeAccName)
{
    using GridT     = tools::build::Grid<BuildT>;
    using TreeT     = tools::build::Tree<BuildT>;
    using AccT      = tools::build::ValueAccessor<BuildT>;
    using WriteAccT = typename TreeT::WriteAccessor;
    using ValueT    = typename GridT::ValueType;

    // ----- build::Grid<BuildT> -----
    nb::class_<GridT>(m, gridName)
        .def(nb::init<const ValueT&, const std::string&, GridClass>(),
            "background"_a,
            "name"_a      = std::string(""),
            "gridClass"_a = GridClass::Unknown,
            "Construct an empty mutable build grid. Voxels read as "
            "background until written.")
        .def("getValue",
            [](const GridT& self, const Coord& ijk) -> ValueT {
                return self.getValue(ijk);
            },
            "ijk"_a,
            "Return the value at ijk (background if no leaf covers it).")
        .def("setValue",
            [](GridT& self, const Coord& ijk, const ValueT& value) {
                self.setValue(ijk, value);
            },
            "ijk"_a, "value"_a,
            "Set the voxel value at ijk and mark the voxel active.")
        // build::Grid has no top-level isActive(ijk); the read path is via
        // ValueAccessor. Spin up a fresh accessor for the single query so
        // Python callers don't have to.
        .def("isActive",
            [](GridT& self, const Coord& ijk) {
                AccT acc = self.getAccessor();
                return acc.isActive(ijk);
            },
            "ijk"_a,
            "Return True iff ijk is in an active voxel. Equivalent to "
            "self.getAccessor().isActive(ijk), but allocates a fresh "
            "accessor for each call — for repeated queries use "
            "self.getAccessor() and reuse it.")
        .def("setValueOn",
            [](GridT& self, const Coord& ijk) {
                AccT acc = self.getAccessor();
                acc.setValueOn(ijk);
            },
            "ijk"_a,
            "Mark ijk active without changing the stored value. Equivalent "
            "to self.getAccessor().setValueOn(ijk).")
        .def("nodeCount",
            [](const GridT& self) -> std::array<size_t, 3> {
                return self.nodeCount();
            },
            "Return a 3-tuple (leaf_count, lower_count, upper_count).")
        .def("gridType",  &GridT::gridType,
            "Return the GridType enumerator this BuildT carries.")
        .def("gridClass", &GridT::gridClass,
            "Return the GridClass assigned at construction time.")
        .def("getName",   &GridT::getName,
            "Return the grid name (as passed at construction).")
        .def("setName",   &GridT::setName, "name"_a,
            "Replace the grid name.")
        .def("setTransform", &GridT::setTransform,
            "scale"_a       = 1.0,
            "translation"_a = Vec3d(0.0),
            "Set an affine index-to-world map from a uniform scale and "
            "translation. Replaces any prior transform.")
        .def_prop_ro("background",
            [](const GridT& self) -> ValueT {
                return self.mRoot.background();
            },
            "The background value supplied at construction.")
        .def("getAccessor", &GridT::getAccessor,
            nb::keep_alive<0, 1>(),
            "Return a ValueAccessor wired to this grid's root. Thread-safe "
            "for reads, NOT thread-safe for writes. The accessor borrows "
            "from this grid — the grid must outlive it.")
        // WriteAccessor's defaulted move constructor would leave its
        // internal ValueAccessor's `mRoot&` reference dangling (it points
        // into the moved-from WriteAccessor's own mRoot field). Bypass
        // the move path entirely by heap-allocating and handing nanobind
        // ownership — the C++ object stays at a stable address for its
        // whole lifetime.
        .def("getWriteAccessor",
            [](GridT& self) -> WriteAccT* {
                return new WriteAccT(self.mRoot, self.mMutex);
            },
            nb::rv_policy::take_ownership,
            nb::keep_alive<0, 1>(),
            "Return a WriteAccessor for thread-safe writes; the accessor "
            "buffers changes into a private root and merges them into the "
            "parent grid on destruction (or on an explicit merge() call). "
            "Held by nanobind on the heap so the accessor's internal "
            "references stay valid.")
        // Baking a large grid is the most expensive operation on this
        // class; release the GIL so other Python threads can run during
        // the conversion (the lambda only touches C++ state).
        .def("to_nanovdb",
            [](const GridT& self,
               tools::StatsMode sMode,
               CheckMode        cMode,
               int              verbose) {
                return tools::createNanoGrid<GridT, BuildT, HostBuffer>(
                    self, sMode, cMode, verbose);
            },
            nb::call_guard<nb::gil_scoped_release>(),
            "sMode"_a   = tools::StatsMode::Default,
            "cMode"_a   = CheckMode::Default,
            "verbose"_a = 0,
            "Bake this mutable grid into a host NanoGrid<BuildT> and "
            "return its GridHandle. The build grid is unchanged. "
            "Releases the GIL during conversion.");

    // ----- build::ValueAccessor<BuildT> -----
    //
    // Move-only (copy is deleted) — returned by getAccessor(). Caches the
    // last leaf / lower / upper node it touched, so repeated access to
    // neighboring coordinates is fast.
    nb::class_<AccT>(m, valueAccName)
        .def("getValue",
            [](const AccT& self, const Coord& ijk) -> ValueT {
                return self.getValue(ijk);
            },
            "ijk"_a, "Return the value at ijk (uses the cache).")
        .def("setValue",
            [](AccT& self, const Coord& ijk, const ValueT& value) {
                self.setValue(ijk, value);
            },
            "ijk"_a, "value"_a,
            "Set the value at ijk and mark it active (uses the cache).")
        .def("setValueOn",
            [](AccT& self, const Coord& ijk) { self.setValueOn(ijk); },
            "ijk"_a,
            "Mark ijk active without changing the stored value.")
        .def("isActive",
            [](const AccT& self, const Coord& ijk) {
                return self.isActive(ijk);
            },
            "ijk"_a,
            "Return True iff ijk is in an active voxel.")
        .def("isValueOn",
            [](const AccT& self, const Coord& ijk) {
                return self.isValueOn(ijk);
            },
            "ijk"_a,
            "Alias for isActive(ijk).");

    // ----- build::Tree<BuildT>::WriteAccessor -----
    //
    // Move-only. Holds its own root node + a reference to the parent root's
    // mutex; on destruction (or explicit merge()) it locks the mutex and
    // splices its buffered nodes into the parent. Designed for multi-thread
    // writes — one WriteAccessor per thread, no shared mutable state.
    nb::class_<WriteAccT>(m, writeAccName)
        .def("setValue",
            [](WriteAccT& self, const Coord& ijk, const ValueT& value) {
                self.setValue(ijk, value);
            },
            "ijk"_a, "value"_a,
            "Buffer a set into this accessor's private root.")
        .def("setValueOn",
            [](WriteAccT& self, const Coord& ijk) { self.setValueOn(ijk); },
            "ijk"_a,
            "Buffer an active-state set into this accessor's private root.")
        .def("merge", &WriteAccT::merge,
            "Lock the parent mutex and splice this accessor's buffered "
            "nodes into the parent grid. Called automatically when the "
            "accessor is destroyed; calling it explicitly is only "
            "necessary if you want the changes visible to the parent "
            "before the accessor goes out of scope.");
}

void defineBuildGridModule(nb::module_& toolsModule)
{
    nb::module_ buildModule = toolsModule.def_submodule("build");
    buildModule.doc() =
        "Mutable, voxel-by-voxel CPU grid builder mirroring "
        "nanovdb::tools::build::*. Construct a typed Grid (e.g. "
        "FloatGrid(0.0, 'mygrid')), populate it with setValue / "
        "ValueAccessor / WriteAccessor, then call .to_nanovdb() to bake "
        "a host NanoGrid handle.";

    // X-macro instantiation over every writable BuildT: scalars (full
    // arithmetic) and vectors. Read-only special BuildTs (Boolean, Fp*,
    // Index, Mask) and Point are deliberately excluded — they have no
    // SetValue<T> specialization and can't be built voxel-by-voxel.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)            \
    defineBuildGrid<T>(buildModule, #Suffix "Grid",                           \
                       #Suffix "ValueAccessor", #Suffix "WriteAccessor");
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    defineBuildGrid<T>(buildModule, #Suffix "Grid",                              \
                       #Suffix "ValueAccessor", #Suffix "WriteAccessor");
#include "BuildTypes.def"
}

} // namespace pynanovdb
