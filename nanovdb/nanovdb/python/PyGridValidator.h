// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDVALIDATOR_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDVALIDATOR_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BufferT> void defineValidateGrids(nb::module_& m);
#ifdef NANOVDB_USE_CUDA
void defineDeviceValidateGrids(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
