// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file pyFloatGrid.cc
/// @author Peter Cucka
/// @brief pybind11 wrappers for scalar, floating-point openvdb::Grid types

#include "pyGrid.h"

/// Create a Python wrapper for each supported Grid type.
void
exportFloatGrid(py::module_ m)
{
    pyGrid::exportGrid<FloatGrid>(m);
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    pyGrid::exportGrid<DoubleGrid>(m);
#endif

    m.def("createLevelSetSphere",
        &pyGrid::createLevelSetSphere<FloatGrid>,
        py::arg("radius"), py::arg("center")=openvdb::Coord(), py::arg("voxelSize")=1.0,
             py::arg("halfWidth")=openvdb::LEVEL_SET_HALF_WIDTH,
        "createLevelSetSphere(radius, center, voxelSize, halfWidth) -> FloatGrid\n\n"
        "Return a grid containing a narrow-band level set representation\n"
        "of a sphere.");
}
