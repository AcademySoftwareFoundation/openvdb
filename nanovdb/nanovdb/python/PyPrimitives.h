// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYPRIMITIVES_HAS_BEEN_INCLUDED
#define NANOVDB_PYPRIMITIVES_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void definePrimitives(nb::module_& m);
#ifdef NANOVDB_USE_CUDA
void defineDevicePrimitives(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
