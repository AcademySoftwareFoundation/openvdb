// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/util/GridBuilder.h>

#include <iostream>

/// @brief Creates a NanoVDB grids with custom values and access them.
///
/// @note This example only depends on NanoVDB.
int main()
{
    try {
        nanovdb::GridBuilder<float> builder(0.0f);
        auto acc = builder.getAccessor();
        acc.setValue(nanovdb::Coord(1, 2, 3), 1.0f);

        printf("GridBuilder: (%i,%i,%i)=%4.2f\t", 1, 2, 3, acc.getValue(nanovdb::Coord(1, 2, 3)));
        printf("GridBuilder: (%i,%i,%i)=%4.2f\n", 1, 2,-3, acc.getValue(nanovdb::Coord(1, 2,-3)));

        auto handle = builder.getHandle<>();
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