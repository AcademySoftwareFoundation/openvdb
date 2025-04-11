// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYPOINTSTOGRID_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYPOINTSTOGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

template<typename BuildT> void definePointsToGrid(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
