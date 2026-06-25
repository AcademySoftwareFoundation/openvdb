// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYUNIFIEDGRIDHANDLE_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYUNIFIEDGRIDHANDLE_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
/// @brief Register GridHandle<nanovdb::cuda::UnifiedBuffer> as
///        "UnifiedGridHandle" on the nanovdb.cuda submodule. This is the handle
///        type returned by nanovdb.tools.cuda.DistributedPointsToGrid.getHandle
///        (its default BufferT is UnifiedBuffer), so the class must be
///        registered for nanobind to cast the result.
void defineUnifiedGridHandle(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
