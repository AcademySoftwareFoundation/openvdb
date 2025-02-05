// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyGridStats.h"

#include <nanovdb/tools/GridStats.h>

namespace nb = nanobind;
using namespace nanovdb;

namespace pynanovdb {

void defineStatsMode(nb::module_& m)
{
    nb::enum_<tools::StatsMode>(m, "StatsMode")
        .value("Disable", tools::StatsMode::Disable)
        .value("BBox", tools::StatsMode::BBox)
        .value("MinMax", tools::StatsMode::MinMax)
        .value("All", tools::StatsMode::All)
        .value("Default", tools::StatsMode::Default)
        .value("End", tools::StatsMode::End);
}

} // namespace pynanovdb
