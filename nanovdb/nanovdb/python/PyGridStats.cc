// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyGridStats.h"

#include <nanobind/stl/string.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridStats.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineStatsMode(nb::module_& m)
{
    nb::enum_<tools::StatsMode>(m, "StatsMode",
        "Selector controlling which per-node statistics are computed by "
        "tools.updateGridStats: Disable skips stats, BBox refreshes only "
        "bounding boxes, MinMax adds min/max, and All adds average and "
        "standard deviation as well.")
        .value("Disable", tools::StatsMode::Disable)
        .value("BBox", tools::StatsMode::BBox)
        .value("MinMax", tools::StatsMode::MinMax)
        .value("All", tools::StatsMode::All)
        .value("Default", tools::StatsMode::Default)
        .value("End", tools::StatsMode::End);
}

namespace {

// ----- Extrema<ValueT> binding (rank 0 and rank 1 share the same surface) -----
template<typename BuildT>
static void defineExtrema(nb::module_& m, const char* name)
{
    using ValueT   = typename NanoGrid<BuildT>::ValueType;
    using ExtremaT = tools::Extrema<ValueT>;

    nb::class_<ExtremaT>(m, name,
        "Running minimum / maximum accumulator over a stream of values. "
        "Build via repeated add(v) calls or via tools.getExtrema(grid, bbox).")
        .def(nb::init<>(),
            "Default-construct an Extrema with min = numeric_limits::max and "
            "max = numeric_limits::lowest, so any subsequent .add(v) gives "
            "exact min/max.")
        .def("min",
            [](const ExtremaT& self) -> ValueT { return self.min(); },
            "Return the minimum value observed so far.")
        .def("max",
            [](const ExtremaT& self) -> ValueT { return self.max(); },
            "Return the maximum value observed so far.")
        .def("add",
            [](ExtremaT& self, const ValueT& v) { self.add(v); },
            "value"_a,
            "Update min/max with a single sample.")
        .def("__bool__",
            [](const ExtremaT& self) { return bool(self); },
            "True iff the Extrema has accumulated at least one sample "
            "(i.e. min <= max).")
        .def_static("hasMinMax", &ExtremaT::hasMinMax,
            "True for value types where min/max is meaningful "
            "(everything except bool).")
        .def_static("hasAverage", &ExtremaT::hasAverage,
            "Always False — Extrema does not compute averages; "
            "use Stats for that.")
        .def_static("hasStdDeviation", &ExtremaT::hasStdDeviation,
            "Always False — Extrema does not compute standard "
            "deviation; use Stats for that.")
        .def_static("hasStats", &ExtremaT::hasStats,
            "True iff the value type supports the min/max bookkeeping "
            "(everything except bool).");
}

// ----- Stats<ValueT> binding (inherits Extrema) -----
template<typename BuildT>
static void defineStats(nb::module_& m, const char* name)
{
    using ValueT  = typename NanoGrid<BuildT>::ValueType;
    using BaseT   = tools::Extrema<ValueT>;
    using StatsT  = tools::Stats<ValueT>;

    nb::class_<StatsT, BaseT>(m, name,
        "Running min/max/mean/variance/std accumulator over a stream of "
        "values. Extends Extrema with sample-count-weighted moments.")
        .def(nb::init<>(),
            "Default-construct a Stats accumulator with zero samples.")
        .def("add",
            [](StatsT& self, const ValueT& v) { self.add(v); },
            "value"_a,
            "Add a single sample.")
        .def("size",
            [](const StatsT& self) -> size_t { return self.size(); },
            "Number of samples accumulated so far.")
        .def("avg",
            [](const StatsT& self) -> double { return self.avg(); },
            "Arithmetic mean of all samples.")
        .def("mean",
            [](const StatsT& self) -> double { return self.mean(); },
            "Alias for avg().")
        .def("var",
            [](const StatsT& self) -> double { return self.var(); },
            "Population variance (Sum(x-mean)^2 / N). Returns 0 if "
            "fewer than two samples have been added.")
        .def("variance",
            [](const StatsT& self) -> double { return self.variance(); },
            "Alias for var().")
        .def("std",
            [](const StatsT& self) -> double { return self.std(); },
            "Standard deviation = sqrt(var()).")
        .def("stdDev",
            [](const StatsT& self) -> double { return self.stdDev(); },
            "Alias for std().")
        .def_static("hasMinMax", &StatsT::hasMinMax,
            "True for value types where min/max is meaningful.")
        .def_static("hasAverage", &StatsT::hasAverage,
            "True for value types that support mean/variance.")
        .def_static("hasStdDeviation", &StatsT::hasStdDeviation,
            "True for value types that support standard deviation.")
        .def_static("hasStats", &StatsT::hasStats,
            "True for value types where full statistics is meaningful.");
}

// ----- updateGridStats polymorphic dispatch ----------------------------------
//
// Match the IsNanoGridValid/callNanoGrid pattern: an Op struct with `known<T>`
// (called for every BuildT that's in scope) and `unknown` (fallback). Only
// scalar + vector BuildTs have a meaningful Stats specialization, so the
// other arms raise.
struct UpdateGridStatsOp
{
    template<typename BuildT>
    static void known(GridData* gridData, tools::StatsMode mode)
    {
        using GridT  = NanoGrid<BuildT>;
        using ValueT = typename GridT::ValueType;
        if constexpr (BuildTraits<BuildT>::is_special &&
                      !util::is_same<bool, BuildT>::value) {
            // Special / quantized / index / mask BuildTs don't have an
            // arithmetic ValueT, so tools::updateGridStats's MinMax / All
            // branches would instantiate Stats<ValueT> / Extrema<ValueT>
            // with no meaningful semantics (and may not even compile).
            // The Disable and BBox branches use NoopStats, which works
            // for any ValueT — drive that directly here so the BBox path
            // remains available on special grids (it just recomputes
            // node bounding boxes without touching min/max/avg).
            if (mode == tools::StatsMode::Disable) {
                return;
            } else if (mode == tools::StatsMode::BBox) {
                tools::GridStats<GridT, tools::NoopStats<ValueT>> stats;
                stats.update(*static_cast<GridT*>(gridData));
            } else {
                throw nb::value_error(
                    "updateGridStats: this grid's BuildT (special / "
                    "quantized / index / mask) has no arithmetic value "
                    "type — only StatsMode.Disable and StatsMode.BBox "
                    "are supported.");
            }
        } else {
            tools::updateGridStats(static_cast<GridT*>(gridData), mode);
        }
    }
    static void unknown(GridData*, tools::StatsMode) {
        throw nb::value_error(
            "updateGridStats: unsupported GridType / BuildT combination.");
    }
};

// ----- getExtrema (per-BuildT factory) ---------------------------------------
template<typename BuildT>
static tools::Extrema<typename NanoGrid<BuildT>::ValueType>
pyGetExtrema(const NanoGrid<BuildT>& grid, const CoordBBox& bbox)
{
    return tools::getExtrema<BuildT>(grid, bbox);
}

} // namespace

void defineGridStatsModule(nb::module_& toolsModule)
{
    // Per-BuildT Extrema + Stats. One pair per scalar/vector BuildT — the
    // value types are all distinct so we get N pairs of new Python classes.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
    defineExtrema<T>(toolsModule, #Suffix "Extrema");                          \
    defineStats<T>(toolsModule,   #Suffix "Stats");
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    defineExtrema<T>(toolsModule, #Suffix "Extrema");                            \
    defineStats<T>(toolsModule,   #Suffix "Stats");
#include "BuildTypes.def"

    // Polymorphic updateGridStats. Accepts any bound NanoGrid<BuildT> (via
    // upcast to GridData*) and dispatches on its mGridType.
    toolsModule.def("updateGridStats",
        [](GridData* gridData, tools::StatsMode mode) {
            if (gridData == nullptr) {
                throw nb::value_error("updateGridStats: grid is None.");
            }
            callNanoGrid<UpdateGridStatsOp>(gridData, mode);
        },
        "grid"_a, "mode"_a = tools::StatsMode::Default,
        nb::call_guard<nb::gil_scoped_release>(),
        "Recompute and write per-node statistics into the given grid in "
        "place. Polymorphic over BuildT. Scalar, vector, and Boolean "
        "grids accept every StatsMode (Disable / BBox / MinMax / All); "
        "Boolean grids use the C++ NoopStats path internally regardless "
        "of mode because there's no arithmetic min/max/avg/dev on bool. "
        "Other special (quantized / index / mask) grids accept Disable "
        "and BBox (the latter recomputes node bounding boxes only); "
        "MinMax and All raise ValueError because their value type has "
        "no arithmetic semantics.");

    // Per-BuildT getExtrema. We expose one overload per scalar/vector
    // BuildT — they each return a Python-side Extrema<ValueT> of the
    // matching name.
#define NANOVDB_PY_FOR_EACH_SCALAR_BUILDT(T, Suffix, GridTypeEnum)             \
    toolsModule.def("getExtrema", &pyGetExtrema<T>,                            \
        "grid"_a, "bbox"_a, nb::call_guard<nb::gil_scoped_release>(),          \
        "Return the Extrema of all values in the grid that intersect "         \
        "the given bbox.");
#define NANOVDB_PY_FOR_EACH_VECTOR_BUILDT(T, Suffix, AccessorName, GridTypeEnum) \
    toolsModule.def("getExtrema", &pyGetExtrema<T>,                              \
        "grid"_a, "bbox"_a, nb::call_guard<nb::gil_scoped_release>(),            \
        "Return the Extrema of all values in the grid that intersect "           \
        "the given bbox.");
#include "BuildTypes.def"
}

} // namespace pynanovdb
