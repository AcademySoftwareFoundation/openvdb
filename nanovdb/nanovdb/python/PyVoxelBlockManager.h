// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYVOXELBLOCKMANAGER_HAS_BEEN_INCLUDED
#define NANOVDB_PYVOXELBLOCKMANAGER_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

/// @brief Bind VoxelBlockManagerHandle<HostBuffer>,
///        tools.buildVoxelBlockManager, tools.decodeInverseMaps, and the
///        test-scaffold tools.createOnIndexGrid factory under the given
///        Python submodule (expected to be the existing nanovdb.tools).
void defineVoxelBlockManagerModule(nb::module_& toolsModule);

} // namespace pynanovdb

#endif
