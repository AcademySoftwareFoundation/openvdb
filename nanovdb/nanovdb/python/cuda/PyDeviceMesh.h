// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYDEVICEMESH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYDEVICEMESH_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
/// @brief Register nanovdb::cuda::DeviceNode and nanovdb::cuda::DeviceMesh on
///        the nanovdb.cuda submodule. DeviceMesh enumerates every CUDA device
///        on the host, creates a per-device stream, and caches P2P
///        connectivity; it is the multi-GPU context object consumed by
///        nanovdb.tools.cuda.DistributedPointsToGrid.
void defineDeviceMesh(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
