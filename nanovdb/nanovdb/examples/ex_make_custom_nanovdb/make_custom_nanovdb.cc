// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

#include <iostream>

/// @brief Creates a NanoVDB grids with custom values and access them.
///
/// @note This example only depends on NanoVDB.
int main()
{
    try {
        nanovdb::tools::build::Grid<float> grid(0.0f);
        auto acc = grid.getAccessor();
        acc.setValue(nanovdb::Coord(1, 2, 3), 1.0f);

        printf("build::Grid: (%i,%i,%i)=%4.2f\t", 1, 2, 3, acc.getValue(nanovdb::Coord(1, 2, 3)));
        printf("build::Grid: (%i,%i,%i)=%4.2f\n", 1, 2,-3, acc.getValue(nanovdb::Coord(1, 2,-3)));

        auto handle = nanovdb::tools::createNanoGrid(grid);
        auto* dstGrid = handle.grid<float>(); // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        if (!dstGrid)
            throw std::runtime_error("GridHandle does not contain a grid with value type float");

        printf("NanoVDB cpu: (%i,%i,%i)=%4.2f\t", 1, 2, 3, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        printf("NanoVDB cpu: (%i,%i,%i)=%4.2f\n", 1, 2,-3, dstGrid->tree().getValue(nanovdb::Coord(1, 2,-3)));
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}