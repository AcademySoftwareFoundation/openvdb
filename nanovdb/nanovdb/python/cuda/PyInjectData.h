// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYINJECTDATA_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYINJECTDATA_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename T> void defineInject(nb::module_& m, const char* name);
template<typename T> void defineInjectFeatures(nb::module_& m, const char* name);
void defineInjectPredicateToMask(nb::module_& m, const char* name);
void defineInjectGridMask(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
