// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file pyFloatGrid.cc
/// @author Peter Cucka
/// @brief nanobind wrappers for scalar, floating-point openvdb::Grid types

#include "pyGrid.h"

/// Create a Python wrapper for each supported Grid type.
void
exportFloatGrid(nb::module_ m)
{
    pyGrid::exportScalarGrid<FloatGrid>(m);
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    pyGrid::exportScalarGrid<DoubleGrid>(m);
#endif

    m.def("createLevelSetSphere",
        &pyGrid::createLevelSetSphere<FloatGrid>,
        nb::arg("radius"), nb::arg("center")=openvdb::Coord(), nb::arg("voxelSize")=1.0,
             nb::arg("halfWidth")=openvdb::LEVEL_SET_HALF_WIDTH,
        "Return a grid containing a narrow-band level set representation\n"
        "of a sphere.");
}
