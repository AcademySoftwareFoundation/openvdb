// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file pyPointGrid.cc
/// @brief pybind11 wrappers for point openvdb::Grid types

#include <pybind11/pybind11.h>

#include "pyGrid.h"

namespace py = pybind11;


#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
void
exportPointGrid(py::module_ m)
{
    pyGrid::exportGrid<points::PointDataGrid>(m);
}
#else
void
exportPointGrid(py::module_)
{
}
#endif
