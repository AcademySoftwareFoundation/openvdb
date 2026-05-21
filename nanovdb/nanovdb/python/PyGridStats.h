// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDSTATS_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDSTATS_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineStatsMode(nb::module_& m);

/// @brief Register per-BuildT Extrema and Stats classes (one set per
///        scalar / vector BuildT in BuildTypes.def) and the polymorphic
///        tools.updateGridStats / tools.getExtrema helpers under the
///        nanovdb.tools submodule.
void defineGridStatsModule(nb::module_& toolsModule);

}

#endif
