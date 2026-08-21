// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYUNIFIEDBUFFER_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYUNIFIEDBUFFER_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
/// @brief Register nanovdb::cuda::UnifiedBuffer as "UnifiedBuffer" on the
///        nanovdb.cuda submodule. UnifiedBuffer is CUDA Unified (managed)
///        memory: a single pointer valid on the host AND every device, so it
///        gets the shared device-interop surface (CAI / DLPack / device_ptr /
///        host_ptr / size) for free — note host_ptr and device_ptr are the
///        same managed pointer.
void defineUnifiedBuffer(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
