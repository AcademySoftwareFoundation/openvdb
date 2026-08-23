// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYBUILDGRID_HAS_BEEN_INCLUDED
#define NANOVDB_PYBUILDGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

/// @brief Register the nanovdb.tools.build submodule and its per-BuildT
///        Grid / ValueAccessor / WriteAccessor classes (one set per writable
///        scalar and vector BuildT in BuildTypes.def). Constructs the submodule
///        as `toolsModule.def_submodule("build")`.
void defineBuildGridModule(nb::module_& toolsModule);

} // namespace pynanovdb

#endif
