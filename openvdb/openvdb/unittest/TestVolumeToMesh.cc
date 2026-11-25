// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/Exceptions.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/MeshToVolume.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

class TestVolumeToMesh: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestVolumeToMesh, testAuxiliaryDataCollection)
{
    typedef openvdb::tree::Tree4<float, 5, 4, 3>::Type  FloatTreeType;
    typedef FloatTreeType::ValueConverter<bool>::Type   BoolTreeType;

    const float iso = 0.0f;
    const openvdb::Coord ijk(0,0,0);

    FloatTreeType inputTree(1.0f);
    inputTree.setValue(ijk, -1.0f);

    BoolTreeType intersectionTree(false);

    openvdb::tools::volume_to_mesh_internal::identifySurfaceIntersectingVoxels(
        intersectionTree, inputTree, iso);

    EXPECT_EQ(size_t(8), size_t(intersectionTree.activeVoxelCount()));

    typedef FloatTreeType::ValueConverter<openvdb::Int16>::Type   Int16TreeType;
    typedef FloatTreeType::ValueConverter<openvdb::Index32>::Type Index32TreeType;

    Int16TreeType signFlagsTree(0);
    Index32TreeType pointIndexTree(99999);

    openvdb::tools::volume_to_mesh_internal::computeAuxiliaryData(
         signFlagsTree, pointIndexTree, intersectionTree, inputTree, iso);

    const int flags = int(signFlagsTree.getValue(ijk));

    EXPECT_TRUE(bool(flags & openvdb::tools::volume_to_mesh_internal::INSIDE));
    EXPECT_TRUE(bool(flags & openvdb::tools::volume_to_mesh_internal::EDGES));
    EXPECT_TRUE(bool(flags & openvdb::tools::volume_to_mesh_internal::XEDGE));
    EXPECT_TRUE(bool(flags & openvdb::tools::volume_to_mesh_internal::YEDGE));
    EXPECT_TRUE(bool(flags & openvdb::tools::volume_to_mesh_internal::ZEDGE));
}


TEST_F(TestVolumeToMesh, testUniformMeshing)
{
    typedef openvdb::tree::Tree4<float, 5, 4, 3>::Type  FloatTreeType;
    typedef openvdb::Grid<FloatTreeType>                FloatGridType;

    FloatGridType grid(1.0f);

    // test voxel region meshing

    openvdb::CoordBBox bbox(openvdb::Coord(1), openvdb::Coord(6));

    grid.tree().fill(bbox, -1.0f);

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec4I> quads;
    std::vector<openvdb::Vec3I> triangles;

    openvdb::tools::volumeToMesh(grid, points, quads);

    EXPECT_TRUE(!points.empty());
    EXPECT_EQ(size_t(216), quads.size());


    points.clear();
    quads.clear();
    triangles.clear();
    grid.clear();


    // test tile region meshing

    grid.tree().addTile(FloatTreeType::LeafNodeType::LEVEL + 1, openvdb::Coord(0), -1.0f, true);

    openvdb::tools::volumeToMesh(grid, points, quads);

    EXPECT_TRUE(!points.empty());
    EXPECT_EQ(size_t(384), quads.size());


    points.clear();
    quads.clear();
    triangles.clear();
    grid.clear();


    // test tile region and bool volume meshing

    typedef FloatTreeType::ValueConverter<bool>::Type   BoolTreeType;
    typedef openvdb::Grid<BoolTreeType>                 BoolGridType;

    BoolGridType maskGrid(false);

    maskGrid.tree().addTile(BoolTreeType::LeafNodeType::LEVEL + 1, openvdb::Coord(0), true, true);

    openvdb::tools::volumeToMesh(maskGrid, points, quads);

    EXPECT_TRUE(!points.empty());
    EXPECT_EQ(size_t(384), quads.size());
}


////////////////////////////////////////


namespace {

// Euler characteristic of the mesh (vertices - edges + faces)
int
eulerCharacteristic(const std::vector<openvdb::Vec3s>& points,
    const std::vector<openvdb::Vec4I>& quads, const std::vector<openvdb::Vec3I>& triangles)
{
    std::set<std::pair<openvdb::Index32, openvdb::Index32>> edges;
    const auto addEdge = [&](openvdb::Index32 a, openvdb::Index32 b) {
        edges.insert(std::minmax(a, b));
    };
    for (const auto& quad : quads) {
        for (int i = 0; i < 4; ++i) addEdge(quad[i], quad[(i + 1) % 4]);
    }
    for (const auto& triangle : triangles) {
        for (int i = 0; i < 3; ++i) addEdge(triangle[i], triangle[(i + 1) % 3]);
    }
    return int(points.size()) - int(edges.size()) + int(quads.size()) + int(triangles.size());
}

} // anonymous namespace


// A sphere is closed, so a missing neighbor across a leaf boundary shows up as a hole
TEST_F(TestVolumeToMesh, testClosedSurfaceAcrossLeafBoundaries)
{
    using namespace openvdb;

    // voxel sizes chosen so the surface crosses many leaf boundaries at different offsets
    for (const float voxelSize : {0.1f, 0.13f}) {
        for (const Vec3f& center : {Vec3f(0.0f), Vec3f(0.37f, -0.21f, 0.5f)}) {
            FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(
                4.0f, center, voxelSize);

            for (const double adaptivity : {0.0, 0.5}) {
                std::vector<Vec3s> points;
                std::vector<Vec4I> quads;
                std::vector<Vec3I> triangles;
                tools::volumeToMesh(*grid, points, triangles, quads, 0.0, adaptivity);

                EXPECT_TRUE(!points.empty());
                EXPECT_EQ(2, eulerCharacteristic(points, quads, triangles));
            }
        }
    }
}


// Converting a closed mesh to a level set and back gives a closed mesh again
TEST_F(TestVolumeToMesh, testMeshToVolumeToMesh)
{
    using namespace openvdb;

    FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(3.0f, Vec3f(0.0f), 0.1f);

    std::vector<Vec3s> points;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(*sphere, points, quads, 0.0);
    EXPECT_TRUE(!points.empty());

    math::Transform::Ptr transform = math::Transform::createLinearTransform(0.07);
    FloatGrid::Ptr grid = tools::meshToLevelSet<FloatGrid>(*transform, points, quads);
    EXPECT_TRUE(grid->activeVoxelCount() > 0);

    std::vector<Vec3s> outPoints;
    std::vector<Vec4I> outQuads;
    std::vector<Vec3I> outTriangles;
    tools::volumeToMesh(*grid, outPoints, outTriangles, outQuads, 0.0, 0.0);
    EXPECT_TRUE(!outPoints.empty());
    EXPECT_EQ(2, eulerCharacteristic(outPoints, outQuads, outTriangles));
}
