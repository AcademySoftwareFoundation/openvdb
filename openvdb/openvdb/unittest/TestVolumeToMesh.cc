// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/Exceptions.h>

#include <gtest/gtest.h>

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
