// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/tools/CreateNanoGrid.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/io/IO.h>

/// @brief Convert an openvdb level set sphere into a nanovdb, access a single value in both grids, and save NanoVDB to file.
///
/// @note This example depends on both OpenVDB and NanoVDB.
int main()
{
    try {
        // Create an OpenVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto srcGrid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(100.0f, openvdb::Vec3f(0.0f), 1.0f);
        auto handle  = nanovdb::tools::createNanoGrid(*srcGrid); // Convert from OpenVDB to NanoVDB and return a shared pointer to a GridHandle.
        auto* dstGrid = handle.grid<float>(); // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        // Access and print out a single value (inside the level set) from both grids
        printf("OpenVDB cpu: %4.2f\n", srcGrid->tree().getValue(openvdb::Coord(99, 0, 0)));
        printf("NanoVDB cpu: %4.2f\n", dstGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));

        nanovdb::io::writeGrid("data/sphere2.nvdb", handle); // Write the NanoVDB grid to file and throw if writing fails
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}