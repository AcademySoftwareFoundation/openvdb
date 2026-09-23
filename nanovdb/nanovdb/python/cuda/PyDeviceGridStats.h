// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICEGRIDSTATS_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICEGRIDSTATS_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

// Bind nanovdb::tools::cuda::updateGridStats for one grid BuildT. All
// instantiations register under the same Python name ("updateGridStats") and
// are disambiguated by nanobind on the (device) grid class.
template<typename BuildT>
void defineDeviceUpdateGridStats(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
