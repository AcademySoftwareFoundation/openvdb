// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYSAMPLEFROMVOXELS_HAS_BEEN_INCLUDED
#define NANOVDB_PYSAMPLEFROMVOXELS_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BuildT> void defineNearestNeighborSampler(nb::module_& m, const char* name);
template<typename BuildT> void defineTrilinearSampler(nb::module_& m, const char* name);
template<typename BuildT> void defineTriquadraticSampler(nb::module_& m, const char* name);
template<typename BuildT> void defineTricubicSampler(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
