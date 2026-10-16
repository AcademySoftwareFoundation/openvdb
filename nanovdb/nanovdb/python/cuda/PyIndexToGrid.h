// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYINDEXTOGRID_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYINDEXTOGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

// Bind nanovdb::tools::cuda::indexToGrid for a (DstBuildT, SrcBuildT) pair.
// SrcBuildT must be an index build type (ValueIndex / ValueOnIndex); DstBuildT
// must be a non-special value type (float / double / Vec3f / ...). All
// instantiations are registered under the same Python name and disambiguated
// by nanobind on the source grid class and the value ndarray dtype/shape.
template<typename DstBuildT, typename SrcBuildT>
void defineIndexToGridScalar(nb::module_& m, const char* name);

template<typename DstBuildT, typename SrcBuildT>
void defineIndexToGridVec3(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
