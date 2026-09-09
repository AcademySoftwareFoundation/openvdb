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

} // namespace pynanovdb
