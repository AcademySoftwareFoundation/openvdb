// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYMESHTOGRID_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYMESHTOGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

// Bind nanovdb::tools::cuda::MeshToGrid: rasterize a triangle mesh (device
// vertex + triangle-index arrays) into a device ValueOnIndex GridHandle plus a
// per-value unsigned-distance-field sidecar buffer. Registered under `name`.
void defineMeshToGrid(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
