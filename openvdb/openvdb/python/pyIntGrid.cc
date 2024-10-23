// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file pyIntGrid.cc
/// @brief nanobind wrappers for scalar, integer-valued openvdb::Grid types

#include "pyGrid.h"

void
exportIntGrid(nb::module_ m)
{
    pyGrid::exportScalarGrid<BoolGrid>(m);
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    pyGrid::exportScalarGrid<Int32Grid>(m);
    pyGrid::exportScalarGrid<Int64Grid>(m);
#endif
}
