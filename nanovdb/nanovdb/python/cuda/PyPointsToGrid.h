// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_CUDA_PYPOINTSTOGRID_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PYPOINTSTOGRID_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

// Rasterize an (N, 3) int32 device tensor of voxel (index-space) coordinates
// into a fresh device GridHandle of type NanoGrid<BuildT>. Used for the
// legacy pointsToRGBA8Grid binding and the new voxelsTo*Grid bindings.
template<typename BuildT> void defineVoxelsToGrid(nb::module_& m, const char* name);

// Rasterize an (N, 3) float OR double device tensor of WORLD-space point
// positions into a fresh device GridHandle of type NanoGrid<Point>, encoding
// the point coordinates as blind data. BuildT selects the world coordinate
// scalar type (float or double).
template<typename BuildT> void definePointsToGrid(nb::module_& m, const char* name);

} // namespace pynanovdb

#endif
