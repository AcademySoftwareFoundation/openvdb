// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>

#include <iostream>

/// @brief Creates a NanoVDB grids with custom values and access them.
///
/// @note This example only depends on NanoVDB.
int main()
{
    try {
        const float background = 5.0f;
        nanovdb::GridBuilder<float> builder(background, nanovdb::GridClass::LevelSet);
        auto acc = builder.getAccessor();
        const int size = 500;
        auto func = [&](const nanovdb::Coord &ijk){
            float v = 40.0f + 50.0f*(cos(ijk[0]*0.1f)*sin(ijk[1]*0.1f) + 
                                     cos(ijk[1]*0.1f)*sin(ijk[2]*0.1f) + 
                                     cos(ijk[2]*0.1f)*sin(ijk[0]*0.1f));
            v = nanovdb::Max(v, nanovdb::Vec3f(ijk).length() - size);// CSG intersection with a sphere
            return v > background ? background : v < -background ? -background : v;// clamp value
        };
        builder(func, nanovdb::CoordBBox(nanovdb::Coord(-size),nanovdb::Coord(size)));

        auto handle = builder.getHandle<>();
        nanovdb::io::writeGrid("data/funny.nvdb", handle, nanovdb::io::Codec::BLOSC);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}