// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>

#include <iostream>

/// @brief Creates a NanoVDB grids with custom values and access them.
///
/// @note This example only depends on NanoVDB.
int main()
{
    using namespace nanovdb;
    try {
        const float background = 5.0f;
        const int size = 500;
        auto func = [&](const Coord &ijk){
            float v = 40.0f + 50.0f*(cos(ijk[0]*0.1f)*sin(ijk[1]*0.1f) +
                                     cos(ijk[1]*0.1f)*sin(ijk[2]*0.1f) +
                                     cos(ijk[2]*0.1f)*sin(ijk[0]*0.1f));
            v = math::Max(v, Vec3f(ijk).length() - size);// CSG intersection with a sphere
            return v > background ? background : v < -background ? -background : v;// clamp value
        };
        tools::build::Grid<float> grid(background, "funny", GridClass::LevelSet);
        grid(func, CoordBBox(Coord(-size), Coord(size)));
        io::writeGrid("data/funny.nvdb", tools::createNanoGrid(grid), io::Codec::BLOSC);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}