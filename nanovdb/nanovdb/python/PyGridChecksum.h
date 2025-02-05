// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYGRIDCHECKSUM_HAS_BEEN_INCLUDED
#define NANOVDB_PYGRIDCHECKSUM_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineCheckMode(nb::module_& m);
void defineChecksum(nb::module_& m);
void defineUpdateChecksum(nb::module_& m);

} // namespace pynanovdb

#endif
