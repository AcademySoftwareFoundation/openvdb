// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/CreatePrimitives.h>

#include <iostream>

/// @brief Creates a NanoVDB grids of a level set sphere and accesses a value.
///
/// @note This example only depends on NanoVDB.
int main()
{
    try {
        auto handle = nanovdb::tools::createLevelSetSphere(100.0f);

        auto* dstGrid = handle.grid<float>(); // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        // Access and print out a single value (inside the level set) from both grids
        printf("NanoVDB cpu: %4.2f\n", dstGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}