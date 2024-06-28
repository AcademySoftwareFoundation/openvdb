// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/io/IO.h>

/// @brief Creates multiple NanoVDB grids, accesses a value in one, and saves all grids to file.
///
/// @note This example only depends on NanoVDB.
int main()
{
    try {
        std::vector<nanovdb::GridHandle<>> handles;
        // Create multiple NanoVDB grids of various types
        handles.push_back(nanovdb::tools::createLevelSetSphere<float>(100.0f));
        handles.push_back(nanovdb::tools::createLevelSetTorus<float>(100.0f, 50.0f));
        handles.push_back(nanovdb::tools::createLevelSetBox<float>(400.0f, 600.0f, 800.0f));
        handles.push_back(nanovdb::tools::createLevelSetBBox<float>(400.0f, 600.0f, 800.0f, 10.0f));
        handles.push_back(nanovdb::tools::createPointSphere<float>(1, 100.0f));

        auto* dstGrid = handles[0].grid<float>(); // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        // Access and print out a single value (inside the level set) from both grids
        printf("NanoVDB cpu: %4.2f\n", dstGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));

        nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/primitives.nvdb", handles); // Write the NanoVDB grids to file and throw if writing fails
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
