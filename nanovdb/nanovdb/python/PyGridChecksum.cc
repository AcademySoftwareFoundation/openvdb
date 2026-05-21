// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyGridChecksum.h"

#include <nanovdb/tools/GridChecksum.h>

#include <nanobind/operators.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

void defineCheckMode(nb::module_& m)
{
    nb::enum_<CheckMode>(m, "CheckMode")
        .value("Disable", CheckMode::Disable)
        .value("Partial", CheckMode::Partial)
        .value("Full", CheckMode::Full)
        .value("Default", CheckMode::Default)
        .value("End", CheckMode::End);
}

void defineChecksum(nb::module_& m)
{
    nb::class_<Checksum>(m, "Checksum").def(nb::self == nb::self, "rhs"_a).def(nb::self != nb::self, "rhs"_a);
}

void defineUpdateChecksum(nb::module_& m)
{
    m.def(
        "updateChecksum", [](GridData* gridData, CheckMode mode) { tools::updateChecksum(gridData, mode); }, "gridData"_a, "mode"_a);
}

void defineEvalChecksumModule(nb::module_& toolsModule)
{
    // tools.evalChecksum(grid, mode) — compute a fresh checksum for the
    // given grid without writing it back. Mirrors the GridData* overload
    // in tools/GridChecksum.h; the polymorphism over BuildT is implicit
    // because every NanoGrid<T> is-a GridData in the C++ hierarchy (and
    // the Python class binding declares NanoGrid<T> as derived from
    // GridData).
    toolsModule.def("evalChecksum",
        [](const GridData* gridData, CheckMode mode) -> Checksum {
            if (gridData == nullptr) {
                throw nb::value_error("evalChecksum: grid is None.");
            }
            return tools::evalChecksum(gridData, mode);
        },
        "grid"_a, "mode"_a = CheckMode::Default,
        nb::call_guard<nb::gil_scoped_release>(),
        "Compute and return the Checksum for the given grid using the "
        "specified CheckMode. Does not modify the grid.");

    // tools.validateChecksum(grid, mode) — compare the stored checksum
    // against a freshly computed one and return a bool.
    toolsModule.def("validateChecksum",
        [](const GridData* gridData, CheckMode mode) -> bool {
            if (gridData == nullptr) {
                throw nb::value_error("validateChecksum: grid is None.");
            }
            return tools::validateChecksum(gridData, mode);
        },
        "grid"_a, "mode"_a = CheckMode::Default,
        nb::call_guard<nb::gil_scoped_release>(),
        "Return True iff the grid's stored checksum matches a freshly "
        "computed one for the given CheckMode. A grid with no stored "
        "checksum (Checksum.isEmpty()) is considered valid.");
}

} // namespace pynanovdb
