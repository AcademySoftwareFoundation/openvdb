// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYTEMPPOOL_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYTEMPPOOL_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

#ifdef NANOVDB_USE_CUDA
/// @brief Register nanovdb::cuda::DeviceResource (static async allocator) and
///        nanovdb::cuda::TempDevicePool (= TempPool<DeviceResource>) on the
///        nanovdb.cuda submodule, in their current shape only. These are the
///        low-level temp-storage primitives used by the CUDA tools; they are
///        NOT reshaped into any resource-concept abstraction.
void defineTempPool(nb::module_& m);
#endif

} // namespace pynanovdb

#endif
