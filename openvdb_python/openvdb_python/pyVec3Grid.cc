// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file pyVec3Grid.cc
/// @brief pybind11 wrappers for vector-valued openvdb::Grid types

#include "pyGrid.h"

void
exportVec3Grid(py::module_ m)
{
    pyGrid::exportGrid<Vec3SGrid>(m);
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    pyGrid::exportGrid<Vec3IGrid>(m);
    pyGrid::exportGrid<Vec3DGrid>(m);
#endif
}
