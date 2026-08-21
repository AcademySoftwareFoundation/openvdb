// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYADDBLINDDATA_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYADDBLINDDATA_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

// Bind nanovdb::tools::cuda::addBlindData for a (grid BuildT, blind-data
// element type) pair. The blind data is a flat 1-D device array; all
// instantiations register under the same Python name and are disambiguated by
// nanobind on the grid class and the blind-data ndarray dtype.
template<typename BuildT, typename BlindDataT>
void defineAddBlindData(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
