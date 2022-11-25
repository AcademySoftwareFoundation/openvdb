// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/io/Stream.h>
#include <openvdb/Metadata.h>
#include <openvdb/math/Maps.h>
#include <openvdb/math/Transform.h>
#include <openvdb/version.h>
#include <openvdb/openvdb.h>
#include <gtest/gtest.h>
#include <cstdio> // for remove()
#include <fstream>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(a, b) \
    EXPECT_NEAR((a), (b), /*tolerance=*/0.0);


class TestStream: public ::testing::Test
{
public:
    void SetUp() override;
    void TearDown() override;

    void testFileReadFromStream();

protected:
    static openvdb::GridPtrVecPtr createTestGrids(openvdb::MetaMap::Ptr&);
    static void verifyTestGrids(openvdb::GridPtrVecPtr, openvdb::MetaMap::Ptr);
};


////////////////////////////////////////


void
TestStream::SetUp()
{
    openvdb::uninitialize();

    openvdb::Int32Grid::registerGrid();
    openvdb::FloatGrid::registerGrid();

    openvdb::StringMetadata::registerType();
    openvdb::Int32Metadata::registerType();
    openvdb::Int64Metadata::registerType();
    openvdb::Vec3IMetadata::registerType();
    openvdb::io::DelayedLoadMetadata::registerType();

    // Register maps
    openvdb::math::MapRegistry::clear();
    openvdb::math::AffineMap::registerMap();
    openvdb::math::ScaleMap::registerMap();
    openvdb::math::UniformScaleMap::registerMap();
    openvdb::math::TranslationMap::registerMap();
    openvdb::math::ScaleTranslateMap::registerMap();
    openvdb::math::UniformScaleTranslateMap::registerMap();
    openvdb::math::NonlinearFrustumMap::registerMap();
}


void
TestStream::TearDown()
{
    openvdb::uninitialize();
}


////////////////////////////////////////


openvdb::GridPtrVecPtr
TestStream::createTestGrids(openvdb::MetaMap::Ptr& metadata)
{
    using namespace openvdb;

    // Create trees
    Int32Tree::Ptr tree1(new Int32Tree(1));
    FloatTree::Ptr tree2(new FloatTree(2.0));

    // Set some values
    tree1->setValue(Coord(0, 0, 0), 5);
    tree1->setValue(Coord(100, 0, 0), 6);
    tree2->setValue(Coord(0, 0, 0), 10);
    tree2->setValue(Coord(0, 100, 0), 11);

    // Create grids
    GridBase::Ptr
        grid1 = createGrid(tree1),
        grid2 = createGrid(tree1), // instance of grid1
        grid3 = createGrid(tree2);
    grid1->setName("density");
    grid2->setName("density_copy");
    grid3->setName("temperature");

    // Create transforms
    math::Transform::Ptr trans1 = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid1->setTransform(trans1);
    grid2->setTransform(trans2);
    grid3->setTransform(trans2);

    metadata.reset(new MetaMap);
    metadata->insertMeta("author", StringMetadata("Einstein"));
    metadata->insertMeta("year", Int32Metadata(2009));

    GridPtrVecPtr grids(new GridPtrVec);
    grids->push_back(grid1);
    grids->push_back(grid2);
    grids->push_back(grid3);

    return grids;
}


void
TestStream::verifyTestGrids(openvdb::GridPtrVecPtr grids, openvdb::MetaMap::Ptr meta)
{
    using namespace openvdb;

    EXPECT_TRUE(grids.get() != nullptr);
    EXPECT_TRUE(meta.get() != nullptr);

    // Verify the metadata.
    EXPECT_EQ(2, int(meta->metaCount()));
    EXPECT_EQ(std::string("Einstein"), meta->metaValue<std::string>("author"));
    EXPECT_EQ(2009, meta->metaValue<int32_t>("year"));

    // Verify the grids.
    EXPECT_EQ(3, int(grids->size()));

    GridBase::Ptr grid = findGridByName(*grids, "density");
    EXPECT_TRUE(grid.get() != nullptr);
    Int32Tree::Ptr density = gridPtrCast<Int32Grid>(grid)->treePtr();
    EXPECT_TRUE(density.get() != nullptr);

    grid.reset();
    grid = findGridByName(*grids, "density_copy");
    EXPECT_TRUE(grid.get() != nullptr);
    EXPECT_TRUE(gridPtrCast<Int32Grid>(grid)->treePtr().get() != nullptr);
    // Verify that "density_copy" is an instance of (i.e., shares a tree with) "density".
    EXPECT_EQ(density, gridPtrCast<Int32Grid>(grid)->treePtr());

    grid.reset();
    grid = findGridByName(*grids, "temperature");
    EXPECT_TRUE(grid.get() != nullptr);
    FloatTree::Ptr temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    EXPECT_TRUE(temperature.get() != nullptr);

    ASSERT_DOUBLES_EXACTLY_EQUAL(5, density->getValue(Coord(0, 0, 0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(6, density->getValue(Coord(100, 0, 0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, temperature->getValue(Coord(0, 0, 0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(11, temperature->getValue(Coord(0, 100, 0)));
}


////////////////////////////////////////


TEST_F(TestStream, testWrite)
{
    using namespace openvdb;

    // Create test grids and stream them to a string.
    MetaMap::Ptr meta;
    GridPtrVecPtr grids = createTestGrids(meta);
    std::ostringstream ostr(std::ios_base::binary);
    io::Stream(ostr).write(*grids, *meta);
    //std::ofstream file("debug.vdb2", std::ios_base::binary);
    //file << ostr.str();

    // Stream the grids back in.
    std::istringstream is(ostr.str(), std::ios_base::binary);
    io::Stream strm(is);
    meta = strm.getMetadata();
    grids = strm.getGrids();

    verifyTestGrids(grids, meta);
}


TEST_F(TestStream, testRead)
{
    using namespace openvdb;

    // Create test grids and write them to a file.
    MetaMap::Ptr meta;
    GridPtrVecPtr grids = createTestGrids(meta);
    const char* filename = "something.vdb2";
    io::File(filename).write(*grids, *meta);
    SharedPtr<const char> scopedFile(filename, ::remove);

    // Stream the grids back in.
    std::ifstream is(filename, std::ios_base::binary);
    io::Stream strm(is);
    meta = strm.getMetadata();
    grids = strm.getGrids();

    verifyTestGrids(grids, meta);
}


/// Stream grids to a file using io::Stream, then read the file back using io::File.
void
TestStream::testFileReadFromStream()
{
    using namespace openvdb;

    MetaMap::Ptr meta;
    GridPtrVecPtr grids;

    // Create test grids and stream them to a file (and then close the file).
    const char* filename = "something.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);
    {
        std::ofstream os(filename, std::ios_base::binary);
        grids = createTestGrids(meta);
        io::Stream(os).write(*grids, *meta);
    }

    // Read the grids back in.
    io::File file(filename);
    EXPECT_TRUE(file.inputHasGridOffsets());
    EXPECT_THROW(file.getGrids(), IoError);

    file.open();
    meta = file.getMetadata();
    grids = file.getGrids();

    EXPECT_TRUE(!file.inputHasGridOffsets());
    EXPECT_TRUE(meta.get() != nullptr);
    EXPECT_TRUE(grids.get() != nullptr);
    EXPECT_TRUE(!grids->empty());

    verifyTestGrids(grids, meta);
}
TEST_F(TestStream, testFileReadFromStream) { testFileReadFromStream(); }
