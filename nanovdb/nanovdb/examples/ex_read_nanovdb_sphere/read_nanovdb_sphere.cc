// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/io/IO.h>

/// @brief Read a NanoVDB grid form file, check pointer and access a single value
///
/// @note This example does NOT dpend on OpenVDB (or CUDA), only NanoVDB.
int main()
{
    try {
        auto handle = nanovdb::io::readGrid("data/sphere.nvdb"); // read first grid in file

        auto* grid = handle.grid<float>(); // get a (raw) pointer to the first NanoVDB grid of value type float

        if (grid == nullptr)
            throw std::runtime_error("File did not contain a grid with value type float");

        // Access and print out a single value in the NanoVDB grid
        printf("NanoVDB cpu: %4.2f\n", grid->tree().getValue(nanovdb::Coord(99, 0, 0)));
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}