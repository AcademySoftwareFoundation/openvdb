// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDVALIDATOR_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDVALIDATOR_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BufferT> void defineValidateGrids(nb::module_& m);

/// @brief Register tools.validateGrid (single grid in a handle),
///        tools.checkGrid (polymorphic, returns (bool, error_str)) and
///        tools.isValid (polymorphic shortcut) under the nanovdb.tools
///        submodule. Bound once; polymorphic dispatch uses callNanoGrid.
void defineGridValidatorModule(nb::module_& toolsModule);

} // namespace pynanovdb

#endif
