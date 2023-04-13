// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file pyIntGrid.cc
/// @brief pybind11 wrappers for scalar, integer-valued openvdb::Grid types

#include "pyGrid.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/OpenToNanoVDB.h>

#include <openvdb/tools/LevelSetSphere.h> // For testing

// TODO:
// #include <nanovdb/util/NanoToOpenVDB.h>

namespace py = pybind11;

namespace pyGrid {
    void fooBar() {
        const double voxelSize = 0.1, halfWidth = 3.0;
        const float radius = 10.0f;
        const nanovdb::Vec3f center(0);
        const nanovdb::Vec3d origin(0);
        const float tolerance = 0.5f * voxelSize;

        auto handle = nanovdb::createLevelSetSphere<float>(radius, center,
                                                           voxelSize, halfWidth,
                                                           origin, "sphere",
                                                           nanovdb::StatsMode::Default,
                                                           nanovdb::ChecksumMode::Default,
                                                           tolerance,
                                                           false);
        return;
    }

    void openToNanoVDBFooBar() {
        const float radius = 1.5f;
        const openvdb::Vec3f center(0.0f, 0.0f, 0.0f);
        const float voxelSize = 0.25f;
        const float halfWidth = 3.0f;

        openvdb::FloatGrid::Ptr srcGrid = openvdb::tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, halfWidth);

        auto handle = nanovdb::openToNanoVDB(*srcGrid);
        auto *grid = handle.grid<float>();
        std::cout << "openToNanoVDBFooBar::grid = " << grid << std::endl;
    }

} // namespace pyGrid


