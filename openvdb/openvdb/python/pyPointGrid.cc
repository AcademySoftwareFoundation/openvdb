// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file pyPointGrid.cc
/// @brief nanobind wrappers for point openvdb::Grid types

#include <nanobind/nanobind.h>

#include "pyGrid.h"

namespace nb = nanobind;


#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
void
exportPointGrid(nb::module_ m)
{
    pyGrid::exportGrid<points::PointDataGrid>(m);
}
#else
void
exportPointGrid(nb::module_)
{
}
#endif
