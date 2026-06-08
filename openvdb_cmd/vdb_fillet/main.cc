// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Blend.h>
#include <openvdb/tools/MeshToVolume.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

const char* gProgName = "";

void
usage [[noreturn]] (int exitStatus = EXIT_FAILURE)
{
    std::cerr <<
"Usage: " << gProgName << " [out.vdb]\n" <<
"Which: writes a filleted union of two box level sets to a VDB file\n" <<
"Options:\n" <<
"    -version       print version information\n";
    exit(exitStatus);
}


/// @brief  Generate the 8 vertices and 6 quad faces of an axis-aligned box
///         centered at @a center with half-extents @a halfSize, then rotate
///         by Euler angles @a rotDeg (X, Y, Z order) in degrees.
///
/// @param center    World-space center of the box.
/// @param halfSize  Half-extents along each axis before rotation.
/// @param rotDeg    Euler rotation angles (X, Y, Z) in degrees.
/// @param points    Output vertex list (index-space, divided by voxelSize).
/// @param quads     Output quad-index list.
/// @param voxelSize Voxel size used to convert world coords to index space.
void
makeRotatedBox(
    const openvdb::Vec3s& center,
    const openvdb::Vec3s& halfSize,
    const openvdb::Vec3s& rotDeg,
    float voxelSize,
    std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec4I>& quads)
{
    using openvdb::Vec3s;

    Vec3s corners[8] = {
        Vec3s(-halfSize[0], -halfSize[1], -halfSize[2]),
        Vec3s( halfSize[0], -halfSize[1], -halfSize[2]),
        Vec3s( halfSize[0],  halfSize[1], -halfSize[2]),
        Vec3s(-halfSize[0],  halfSize[1], -halfSize[2]),
        Vec3s(-halfSize[0], -halfSize[1],  halfSize[2]),
        Vec3s( halfSize[0], -halfSize[1],  halfSize[2]),
        Vec3s( halfSize[0],  halfSize[1],  halfSize[2]),
        Vec3s(-halfSize[0],  halfSize[1],  halfSize[2]),
    };

    const float degToRad = openvdb::math::pi<float>() / 180.0f;
    const float rx = rotDeg[0] * degToRad;
    const float ry = rotDeg[1] * degToRad;
    const float rz = rotDeg[2] * degToRad;

    const float cx = std::cos(rx), sx = std::sin(rx);
    const float cy = std::cos(ry), sy = std::sin(ry);
    const float cz = std::cos(rz), sz = std::sin(rz);

    auto rotate = [&](const Vec3s& v) -> Vec3s {
        const float x1 = v[0];
        const float y1 = cx * v[1] - sx * v[2];
        const float z1 = sx * v[1] + cx * v[2];

        const float x2 =  cy * x1 + sy * z1;
        const float y2 =  y1;
        const float z2 = -sy * x1 + cy * z1;

        const float x3 = cz * x2 - sz * y2;
        const float y3 = sz * x2 + cz * y2;
        const float z3 = z2;
        return Vec3s(x3, y3, z3);
    };

    points.clear();
    points.reserve(8);
    for (int i = 0; i < 8; ++i) {
        const Vec3s r = rotate(corners[i]);
        points.emplace_back(
            (r[0] + center[0]) / voxelSize,
            (r[1] + center[1]) / voxelSize,
            (r[2] + center[2]) / voxelSize);
    }

    quads.clear();
    quads.reserve(6);
    quads.emplace_back(openvdb::Vec4I(0, 3, 2, 1));
    quads.emplace_back(openvdb::Vec4I(4, 5, 6, 7));
    quads.emplace_back(openvdb::Vec4I(0, 1, 5, 4));
    quads.emplace_back(openvdb::Vec4I(2, 3, 7, 6));
    quads.emplace_back(openvdb::Vec4I(0, 4, 7, 3));
    quads.emplace_back(openvdb::Vec4I(1, 2, 6, 5));
}


openvdb::FloatGrid::Ptr
makeBoxLevelSet(
    const openvdb::Vec3s& center,
    const openvdb::Vec3s& halfSize,
    const openvdb::Vec3s& rotDeg,
    const openvdb::math::Transform& transform,
    float voxelSize,
    float exteriorBand,
    float interiorBand)
{
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec4I> quads;
    makeRotatedBox(center, halfSize, rotDeg, voxelSize, points, quads);

    openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> mesh(points, quads);
    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::meshToVolume<openvdb::FloatGrid>(mesh, transform, exteriorBand, interiorBand);
    if (grid) grid->setGridClass(openvdb::GRID_LEVEL_SET);
    return grid;
}


void
fillet_two_boxes(const std::string& outFilename)
{
    const float voxelSize = 0.02f;
    const float exteriorBand = 6.0f;
    const float interiorBand = 3.0f;

    const float alpha = 2.0f;
    const float beta  = 80.0f;
    const float gamma = 1.0f;

    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(voxelSize);

    openvdb::FloatGrid::Ptr gridA = makeBoxLevelSet(
        openvdb::Vec3s(0.0f),
        openvdb::Vec3s(0.5f, 0.5f, 0.5f),
        openvdb::Vec3s(45.0f, 0.0f, 45.0f),
        *transform,
        voxelSize,
        exteriorBand,
        interiorBand);
    if (!gridA) throw std::runtime_error("failed to create box A level set");

    openvdb::FloatGrid::Ptr gridB = makeBoxLevelSet(
        openvdb::Vec3s(0.0f),
        openvdb::Vec3s(1.0f, 0.2f, 1.0f),
        openvdb::Vec3s(0.0f),
        *transform,
        voxelSize,
        exteriorBand,
        interiorBand);
    if (!gridB) throw std::runtime_error("failed to create box B level set");

    openvdb::FloatGrid::ConstPtr noMask;
    openvdb::FloatGrid::Ptr filletUnion =
        openvdb::tools::unionFillet<openvdb::FloatGrid>(
            *gridA, *gridB, noMask, alpha, beta, gamma);
    if (!filletUnion) throw std::runtime_error("failed to create fillet union");

    filletUnion->setName("fillet_union");
    filletUnion->setGridClass(openvdb::GRID_LEVEL_SET);

    openvdb::GridPtrVec grids;
    grids.push_back(filletUnion);

    openvdb::io::File file(outFilename);
    file.write(grids);
    file.close();

    std::cout << "Wrote " << outFilename << " with grid \""
        << filletUnion->getName() << "\" ("
        << filletUnion->activeVoxelCount() << " active voxels)\n";
}

} // unnamed namespace


int
main(int argc, char* argv[])
{
    gProgName = argv[0];
    if (const char* ptr = std::strrchr(gProgName, '/')) gProgName = ptr + 1;

    if (argc > 2) usage();

    std::string outFilename = "fillet_union.vdb";
    if (argc == 2) {
        const std::string arg = argv[1];
        if (arg == "-h" || arg == "-help" || arg == "--help") usage(EXIT_SUCCESS);
        if (arg == "-version" || arg == "--version") {
            std::cout << "OpenVDB " << openvdb::getLibraryVersionString() << "\n";
            return EXIT_SUCCESS;
        }
        outFilename = arg;
    }

    try {
        openvdb::initialize();
        fillet_two_boxes(outFilename);
        openvdb::uninitialize();
    } catch (const std::exception& e) {
        std::cerr << gProgName << ": " << e.what() << "\n";
        openvdb::uninitialize();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
