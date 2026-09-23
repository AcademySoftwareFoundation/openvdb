// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDISTRIBUTEDPOINTSTOGRID_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDISTRIBUTEDPOINTSTOGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
/// @brief Register the multi-GPU nanovdb::tools::cuda::DistributedPointsToGrid<
///        BuildT> on the nanovdb.tools.cuda submodule. The Python class wraps a
///        nanovdb::cuda::DeviceMesh and a scale/translation (or unit map), and
///        getHandle(voxels, count) rasterizes an (N, 3) int32 unified-memory
///        array of index-space voxel coordinates into a UnifiedGridHandle.
/// @note  The DeviceMesh passed to the constructor must outlive the converter.
template<typename BuildT> void defineDistributedPointsToGrid(nb::module_& m, const char* name);
#endif

} // namespace pynanovdb

#endif
