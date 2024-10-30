// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file pyVec3Grid.cc
/// @brief nanobind wrappers for vector-valued openvdb::Grid types

#include "pyGrid.h"

void
exportVec3Grid(nb::module_ m)
{
    pyGrid::exportVectorGrid<Vec3SGrid>(m);
#ifdef PY_OPENVDB_WRAP_ALL_GRID_TYPES
    pyGrid::exportVectorGrid<Vec3IGrid>(m);
    pyGrid::exportVectorGrid<Vec3DGrid>(m);
#endif
}
