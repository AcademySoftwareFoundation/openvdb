// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDSTATS_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDSTATS_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineStatsMode(nb::module_& m);

}

#endif
