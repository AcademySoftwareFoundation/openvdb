// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/io/File.h>
#include <openvdb/io/io.h>
#include <openvdb/io/Queue.h>
#include <openvdb/io/Stream.h>
#include <openvdb/Metadata.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/Count.h>
#include <openvdb/tools/LevelSetUtil.h> // for tools::sdfToFogVolume()
#include <openvdb/util/logging.h>
#include <openvdb/version.h>
#include <openvdb/openvdb.h>
#include "util.h" // for unittest_util::makeSphere()

#include <gtest/gtest.h>

#include <thread>
#include <chrono>
#include <algorithm> // for std::sort()
#include <cstdio> // for remove() and rename()
#include <fstream>
#include <functional> // for std::bind()
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <sys/types.h> // for stat()
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#include <cstring> // for memset()
#endif


class TestFile: public ::testing::Test
{
public:
    void SetUp() override {}
    void TearDown() override { openvdb::uninitialize(); }

    void testHeader();
    void testWriteGrid();
    void testWriteMultipleGrids();
    void testReadGridDescriptors();
    void testEmptyGridIO();
    void testOpen();
    void testDelayedLoadMetadata();
    void testNonVdbOpen();
};


////////////////////////////////////////

void
TestFile::testHeader()
{
    using namespace openvdb::io;

    File file("something.vdb2");

    std::ostringstream
        ostr(std::ios_base::binary),
        ostr2(std::ios_base::binary);

    file.writeHeader(ostr2, /*seekable=*/true);
    std::string uuidStr = file.getUniqueTag();

    file.writeHeader(ostr, /*seekable=*/true);
    // Verify that a file gets a new UUID each time it is written.
    EXPECT_TRUE(!file.isIdentical(uuidStr));
    uuidStr = file.getUniqueTag();

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    bool unique=true;
    EXPECT_NO_THROW(unique=file.readHeader(istr));

    EXPECT_TRUE(!unique);//reading same file again

    uint32_t version = openvdb::OPENVDB_FILE_VERSION;

    EXPECT_EQ(version, file.fileVersion());
    EXPECT_EQ(openvdb::OPENVDB_LIBRARY_MAJOR_VERSION, file.libraryVersion().first);
    EXPECT_EQ(openvdb::OPENVDB_LIBRARY_MINOR_VERSION, file.libraryVersion().second);
    EXPECT_EQ(uuidStr, file.getUniqueTag());

    //std::cerr << "\nuuid=" << uuidStr << std::endl;

    EXPECT_TRUE(file.isIdentical(uuidStr));

    remove("something.vdb2");
}
TEST_F(TestFile, testHeader) { testHeader(); }


void
TestFile::testWriteGrid()
{
    using namespace openvdb;
    using namespace openvdb::io;

    using TreeType = Int32Tree;
    using GridType = Grid<TreeType>;

    logging::LevelScope suppressLogging{logging::Level::Fatal};

    File file("something.vdb2");

    std::ostringstream ostr(std::ios_base::binary);

    // Create a grid with transform.
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);

    GridType::Ptr grid = createGrid<GridType>(/*bg=*/1);
    TreeType& tree = grid->tree();
    grid->setTransform(trans);
    tree.setValue(Coord(10, 1, 2), 10);
    tree.setValue(Coord(0, 0, 0), 5);

    // Add some metadata.
    Metadata::clearRegistry();
    StringMetadata::registerType();
    const std::string meta0Val, meta1Val("Hello, world.");
    Metadata::Ptr stringMetadata = Metadata::createMetadata(typeNameAsString<std::string>());
    EXPECT_TRUE(stringMetadata);
    if (stringMetadata) {
        grid->insertMeta("meta0", *stringMetadata);
        grid->metaValue<std::string>("meta0") = meta0Val;
        grid->insertMeta("meta1", *stringMetadata);
        grid->metaValue<std::string>("meta1") = meta1Val;
    }

    // Create the grid descriptor out of this grid.
    GridDescriptor gd(Name("temperature"), grid->type());

    // Write out the grid.
    file.writeGrid(gd, grid, ostr, /*seekable=*/true);

    EXPECT_TRUE(gd.getGridPos() != 0);
    EXPECT_TRUE(gd.getBlockPos() != 0);
    EXPECT_TRUE(gd.getEndPos() != 0);

    // Read in the grid descriptor.
    GridDescriptor gd2;
    std::istringstream istr(ostr.str(), std::ios_base::binary);

    // Since the input is only a fragment of a VDB file (in particular,
    // it doesn't have a header), set the file format version number explicitly.
    io::setCurrentVersion(istr);

    GridBase::Ptr gd2_grid;
    EXPECT_THROW(gd2.read(istr), openvdb::LookupError);

    // Register the grid and the transform and the blocks.
    GridBase::clearRegistry();
    GridType::registerGrid();

    // Register transform maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::ScaleMap::registerMap();
    math::UniformScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();
    math::UniformScaleTranslateMap::registerMap();
    math::NonlinearFrustumMap::registerMap();

    istr.seekg(0, std::ios_base::beg);
    EXPECT_NO_THROW(gd2_grid = gd2.read(istr));

    EXPECT_EQ(gd.gridName(), gd2.gridName());
    EXPECT_EQ(GridType::gridType(), gd2_grid->type());
    EXPECT_EQ(gd.getGridPos(), gd2.getGridPos());
    EXPECT_EQ(gd.getBlockPos(), gd2.getBlockPos());
    EXPECT_EQ(gd.getEndPos(), gd2.getEndPos());

    // Position the stream to beginning of the grid storage and read the grid.
    gd2.seekToGrid(istr);
    Archive::readGridCompression(istr);
    gd2_grid->readMeta(istr);
    gd2_grid->readTransform(istr);
    gd2_grid->readTopology(istr);

    // Remove delay load metadata if it exists.
    if ((*gd2_grid)["file_delayed_load"]) {
        gd2_grid->removeMeta("file_delayed_load");
    }

    // Ensure that we have the same metadata.
    EXPECT_EQ(grid->metaCount(), gd2_grid->metaCount());
    EXPECT_TRUE((*gd2_grid)["meta0"]);
    EXPECT_TRUE((*gd2_grid)["meta1"]);
    EXPECT_EQ(meta0Val, gd2_grid->metaValue<std::string>("meta0"));
    EXPECT_EQ(meta1Val, gd2_grid->metaValue<std::string>("meta1"));

    // Ensure that we have the same topology and transform.
    EXPECT_EQ(
        grid->baseTree().leafCount(), gd2_grid->baseTree().leafCount());
    EXPECT_EQ(
        grid->baseTree().nonLeafCount(), gd2_grid->baseTree().nonLeafCount());
    EXPECT_EQ(
        grid->baseTree().treeDepth(), gd2_grid->baseTree().treeDepth());

    //EXPECT_EQ(0.1, gd2_grid->getTransform()->getVoxelSizeX());
    //EXPECT_EQ(0.1, gd2_grid->getTransform()->getVoxelSizeY());
    //EXPECT_EQ(0.1, gd2_grid->getTransform()->getVoxelSizeZ());

    // Read in the data blocks.
    gd2.seekToBlocks(istr);
    gd2_grid->readBuffers(istr);
    TreeType::Ptr tree2 = DynamicPtrCast<TreeType>(gd2_grid->baseTreePtr());
    EXPECT_TRUE(tree2.get() != nullptr);
    EXPECT_EQ(10, tree2->getValue(Coord(10, 1, 2)));
    EXPECT_EQ(5, tree2->getValue(Coord(0, 0, 0)));

    EXPECT_EQ(1, tree2->getValue(Coord(1000, 1000, 16000)));
    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    remove("something.vdb2");
}
TEST_F(TestFile, testWriteGrid) { testWriteGrid(); }


void
TestFile::testWriteMultipleGrids()
{
    using namespace openvdb;
    using namespace openvdb::io;

    using TreeType = Int32Tree;
    using GridType = Grid<TreeType>;

    logging::LevelScope suppressLogging{logging::Level::Fatal};

    File file("something.vdb2");

    std::ostringstream ostr(std::ios_base::binary);

    // Create a grid with transform.
    GridType::Ptr grid = createGrid<GridType>(/*bg=*/1);
    TreeType& tree = grid->tree();
    tree.setValue(Coord(10, 1, 2), 10);
    tree.setValue(Coord(0, 0, 0), 5);
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    grid->setTransform(trans);

    GridType::Ptr grid2 = createGrid<GridType>(/*bg=*/2);
    TreeType& tree2 = grid2->tree();
    tree2.setValue(Coord(0, 0, 0), 10);
    tree2.setValue(Coord(1000, 1000, 1000), 50);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.2);
    grid2->setTransform(trans2);

    // Create the grid descriptor out of this grid.
    GridDescriptor gd(Name("temperature"), grid->type());
    GridDescriptor gd2(Name("density"), grid2->type());

    // Write out the grids.
    file.writeGrid(gd, grid, ostr, /*seekable=*/true);
    file.writeGrid(gd2, grid2, ostr, /*seekable=*/true);

    EXPECT_TRUE(gd.getGridPos() != 0);
    EXPECT_TRUE(gd.getBlockPos() != 0);
    EXPECT_TRUE(gd.getEndPos() != 0);

    EXPECT_TRUE(gd2.getGridPos() != 0);
    EXPECT_TRUE(gd2.getBlockPos() != 0);
    EXPECT_TRUE(gd2.getEndPos() != 0);

    // register the grid
    GridBase::clearRegistry();
    GridType::registerGrid();

    // register maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::ScaleMap::registerMap();
    math::UniformScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();
    math::UniformScaleTranslateMap::registerMap();
    math::NonlinearFrustumMap::registerMap();

    // Read in the first grid descriptor.
    GridDescriptor gd_in;
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    io::setCurrentVersion(istr);

    GridBase::Ptr gd_in_grid;
    EXPECT_NO_THROW(gd_in_grid = gd_in.read(istr));

    // Ensure read in the right values.
    EXPECT_EQ(gd.gridName(), gd_in.gridName());
    EXPECT_EQ(GridType::gridType(), gd_in_grid->type());
    EXPECT_EQ(gd.getGridPos(), gd_in.getGridPos());
    EXPECT_EQ(gd.getBlockPos(), gd_in.getBlockPos());
    EXPECT_EQ(gd.getEndPos(), gd_in.getEndPos());

    // Position the stream to beginning of the grid storage and read the grid.
    gd_in.seekToGrid(istr);
    Archive::readGridCompression(istr);
    gd_in_grid->readMeta(istr);
    gd_in_grid->readTransform(istr);
    gd_in_grid->readTopology(istr);

    // Ensure that we have the same topology and transform.
    EXPECT_EQ(
        grid->baseTree().leafCount(), gd_in_grid->baseTree().leafCount());
    EXPECT_EQ(
        grid->baseTree().nonLeafCount(), gd_in_grid->baseTree().nonLeafCount());
    EXPECT_EQ(
        grid->baseTree().treeDepth(), gd_in_grid->baseTree().treeDepth());

    // EXPECT_EQ(0.1, gd_in_grid->getTransform()->getVoxelSizeX());
    // EXPECT_EQ(0.1, gd_in_grid->getTransform()->getVoxelSizeY());
    // EXPECT_EQ(0.1, gd_in_grid->getTransform()->getVoxelSizeZ());

    // Read in the data blocks.
    gd_in.seekToBlocks(istr);
    gd_in_grid->readBuffers(istr);
    TreeType::Ptr grid_in = DynamicPtrCast<TreeType>(gd_in_grid->baseTreePtr());
    EXPECT_TRUE(grid_in.get() != nullptr);
    EXPECT_EQ(10, grid_in->getValue(Coord(10, 1, 2)));
    EXPECT_EQ(5, grid_in->getValue(Coord(0, 0, 0)));
    EXPECT_EQ(1, grid_in->getValue(Coord(1000, 1000, 16000)));

    /////////////////////////////////////////////////////////////////
    // Now read in the second grid descriptor. Make use of hte end offset.
    ///////////////////////////////////////////////////////////////

    gd_in.seekToEnd(istr);

    GridDescriptor gd2_in;
    GridBase::Ptr gd2_in_grid;
    EXPECT_NO_THROW(gd2_in_grid = gd2_in.read(istr));

    // Ensure that we read in the right values.
    EXPECT_EQ(gd2.gridName(), gd2_in.gridName());
    EXPECT_EQ(TreeType::treeType(), gd2_in_grid->type());
    EXPECT_EQ(gd2.getGridPos(), gd2_in.getGridPos());
    EXPECT_EQ(gd2.getBlockPos(), gd2_in.getBlockPos());
    EXPECT_EQ(gd2.getEndPos(), gd2_in.getEndPos());

    // Position the stream to beginning of the grid storage and read the grid.
    gd2_in.seekToGrid(istr);
    Archive::readGridCompression(istr);
    gd2_in_grid->readMeta(istr);
    gd2_in_grid->readTransform(istr);
    gd2_in_grid->readTopology(istr);

    // Ensure that we have the same topology and transform.
    EXPECT_EQ(
        grid2->baseTree().leafCount(), gd2_in_grid->baseTree().leafCount());
    EXPECT_EQ(
        grid2->baseTree().nonLeafCount(), gd2_in_grid->baseTree().nonLeafCount());
    EXPECT_EQ(
        grid2->baseTree().treeDepth(), gd2_in_grid->baseTree().treeDepth());
    // EXPECT_EQ(0.2, gd2_in_grid->getTransform()->getVoxelSizeX());
    // EXPECT_EQ(0.2, gd2_in_grid->getTransform()->getVoxelSizeY());
    // EXPECT_EQ(0.2, gd2_in_grid->getTransform()->getVoxelSizeZ());

    // Read in the data blocks.
    gd2_in.seekToBlocks(istr);
    gd2_in_grid->readBuffers(istr);
    TreeType::Ptr grid2_in = DynamicPtrCast<TreeType>(gd2_in_grid->baseTreePtr());
    EXPECT_TRUE(grid2_in.get() != nullptr);
    EXPECT_EQ(50, grid2_in->getValue(Coord(1000, 1000, 1000)));
    EXPECT_EQ(10, grid2_in->getValue(Coord(0, 0, 0)));
    EXPECT_EQ(2, grid2_in->getValue(Coord(100000, 100000, 16000)));

    // Clear registries.
    GridBase::clearRegistry();

    math::MapRegistry::clear();
    remove("something.vdb2");
}
TEST_F(TestFile, testWriteMultipleGrids) { testWriteMultipleGrids(); }


TEST_F(TestFile, testWriteFloatAsHalf)
{
    using namespace openvdb;
    using namespace openvdb::io;

    using TreeType = Vec3STree;
    using GridType = Grid<TreeType>;

    // Register all grid types.
    initialize();
    // Ensure that the registry is cleared on exit.
    struct Local { static void uninitialize(char*) { openvdb::uninitialize(); } };
    SharedPtr<char> onExit(nullptr, Local::uninitialize);

    // Create two test grids.
    GridType::Ptr grid1 = createGrid<GridType>(/*bg=*/Vec3s(1, 1, 1));
    TreeType& tree1 = grid1->tree();
    EXPECT_TRUE(grid1.get() != nullptr);
    grid1->setTransform(math::Transform::createLinearTransform(0.1));
    grid1->setName("grid1");

    GridType::Ptr grid2 = createGrid<GridType>(/*bg=*/Vec3s(2, 2, 2));
    EXPECT_TRUE(grid2.get() != nullptr);
    TreeType& tree2 = grid2->tree();
    grid2->setTransform(math::Transform::createLinearTransform(0.2));
    // Flag this grid for 16-bit float output.
    grid2->setSaveFloatAsHalf(true);
    grid2->setName("grid2");

    for (int x = 0; x < 40; ++x) {
        for (int y = 0; y < 40; ++y) {
            for (int z = 0; z < 40; ++z) {
                tree1.setValue(Coord(x, y, z), Vec3s(float(x), float(y), float(z)));
                tree2.setValue(Coord(x, y, z), Vec3s(float(x), float(y), float(z)));
            }
        }
    }

    GridPtrVec grids;
    grids.push_back(grid1);
    grids.push_back(grid2);

    const char* filename = "something.vdb2";
    {
        // Write both grids to a file.
        File vdbFile(filename);
        vdbFile.write(grids);
    }
    {
        // Verify that both grids can be read back successfully from the file.
        File vdbFile(filename);
        vdbFile.open();
        GridBase::Ptr
            bgrid1 = vdbFile.readGrid("grid1"),
            bgrid2 = vdbFile.readGrid("grid2");
        vdbFile.close();

        EXPECT_TRUE(bgrid1.get() != nullptr);
        EXPECT_TRUE(bgrid1->isType<GridType>());
        EXPECT_TRUE(bgrid2.get() != nullptr);
        EXPECT_TRUE(bgrid2->isType<GridType>());

        const TreeType& btree1 = StaticPtrCast<GridType>(bgrid1)->tree();
        EXPECT_EQ(Vec3s(10, 10, 10), btree1.getValue(Coord(10, 10, 10)));
        const TreeType& btree2 = StaticPtrCast<GridType>(bgrid2)->tree();
        EXPECT_EQ(Vec3s(10, 10, 10), btree2.getValue(Coord(10, 10, 10)));
    }
}


TEST_F(TestFile, testWriteInstancedGrids)
{
    using namespace openvdb;

    // Register data types.
    openvdb::initialize();

    // Remove something.vdb2 when done. We must declare this here before the
    // other grid smart_ptr's because we re-use them in the test several times.
    // We will not be able to remove something.vdb2 on Windows if the pointers
    // are still referencing data opened by the "file" variable.
    const char* filename = "something.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);

    // Create grids.
    Int32Tree::Ptr tree1(new Int32Tree(1));
    FloatTree::Ptr tree2(new FloatTree(2.0));
    GridBase::Ptr
        grid1 = createGrid(tree1),
        grid2 = createGrid(tree1), // instance of grid1
        grid3 = createGrid(tree2),
        grid4 = createGrid(tree2); // instance of grid3
    grid1->setName("density");
    grid2->setName("density_copy");
    // Leave grid3 and grid4 unnamed.

    // Create transforms.
    math::Transform::Ptr trans1 = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid1->setTransform(trans1);
    grid2->setTransform(trans2);
    grid3->setTransform(trans2);
    grid4->setTransform(trans1);

    // Set some values.
    tree1->setValue(Coord(0, 0, 0), 5);
    tree1->setValue(Coord(100, 0, 0), 6);
    tree2->setValue(Coord(0, 0, 0), 10);
    tree2->setValue(Coord(0, 100, 0), 11);

    MetaMap::Ptr meta(new MetaMap);
    meta->insertMeta("author", StringMetadata("Einstein"));
    meta->insertMeta("year", Int32Metadata(2009));

    GridPtrVecPtr grids(new GridPtrVec);
    grids->push_back(grid1);
    grids->push_back(grid2);
    grids->push_back(grid3);
    grids->push_back(grid4);

    // Write the grids to a file and then close the file.
    {
        io::File vdbFile(filename);
        vdbFile.write(*grids, *meta);
    }
    meta.reset();

    // Read the grids back in.
    io::File file(filename);
    file.open();
    grids = file.getGrids();
    meta = file.getMetadata();

    // Verify the metadata.
    EXPECT_TRUE(meta.get() != nullptr);
    EXPECT_EQ(2, int(meta->metaCount()));
    EXPECT_EQ(std::string("Einstein"), meta->metaValue<std::string>("author"));
    EXPECT_EQ(2009, meta->metaValue<int32_t>("year"));

    // Verify the grids.
    EXPECT_TRUE(grids.get() != nullptr);
    EXPECT_EQ(4, int(grids->size()));

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
    grid = findGridByName(*grids, "");
    EXPECT_TRUE(grid.get() != nullptr);
    FloatTree::Ptr temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    EXPECT_TRUE(temperature.get() != nullptr);

    grid.reset();
    for (GridPtrVec::reverse_iterator it = grids->rbegin(); !grid && it != grids->rend(); ++it) {
        // Search for the second unnamed grid starting from the end of the list.
        if ((*it)->getName() == "") grid = *it;
    }
    EXPECT_TRUE(grid.get() != nullptr);
    EXPECT_TRUE(gridPtrCast<FloatGrid>(grid)->treePtr().get() != nullptr);
    // Verify that the second unnamed grid is an instance of the first.
    EXPECT_EQ(temperature, gridPtrCast<FloatGrid>(grid)->treePtr());

    EXPECT_NEAR(5, density->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    EXPECT_NEAR(6, density->getValue(Coord(100, 0, 0)), /*tolerance=*/0);
    EXPECT_NEAR(10, temperature->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    EXPECT_NEAR(11, temperature->getValue(Coord(0, 100, 0)), /*tolerance=*/0);

    // Reread with instancing disabled.
    file.close();
    file.setInstancingEnabled(false);
    file.open();
    grids = file.getGrids();
    EXPECT_EQ(4, int(grids->size()));

    grid = findGridByName(*grids, "density");
    EXPECT_TRUE(grid.get() != nullptr);
    density = gridPtrCast<Int32Grid>(grid)->treePtr();
    EXPECT_TRUE(density.get() != nullptr);
    grid = findGridByName(*grids, "density_copy");
    EXPECT_TRUE(grid.get() != nullptr);
    EXPECT_TRUE(gridPtrCast<Int32Grid>(grid)->treePtr().get() != nullptr);
    // Verify that "density_copy" is *not* an instance of "density".
    EXPECT_TRUE(gridPtrCast<Int32Grid>(grid)->treePtr() != density);

    // Verify that the two unnamed grids are not instances of each other.
    grid = findGridByName(*grids, "");
    EXPECT_TRUE(grid.get() != nullptr);
    temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    EXPECT_TRUE(temperature.get() != nullptr);
    grid.reset();
    for (GridPtrVec::reverse_iterator it = grids->rbegin(); !grid && it != grids->rend(); ++it) {
        // Search for the second unnamed grid starting from the end of the list.
        if ((*it)->getName() == "") grid = *it;
    }
    EXPECT_TRUE(grid.get() != nullptr);
    EXPECT_TRUE(gridPtrCast<FloatGrid>(grid)->treePtr().get() != nullptr);
    EXPECT_TRUE(gridPtrCast<FloatGrid>(grid)->treePtr() != temperature);

    // Rewrite with instancing disabled, then reread with instancing enabled.
    file.close();
    {
        /// @todo (FX-7063) For now, write to a new file, then, when there's
        /// no longer a need for delayed load from the old file, replace it
        /// with the new file.
        const char* tempFilename = "somethingelse.vdb";
        SharedPtr<const char> scopedTempFile(tempFilename, ::remove);
        io::File vdbFile(tempFilename);
        vdbFile.setInstancingEnabled(false);
        vdbFile.write(*grids, *meta);
        grids.reset();
        // Note: Windows requires that the destination not exist, before we can rename to it.
        std::remove(filename);
        std::rename(tempFilename, filename);
    }
    file.setInstancingEnabled(true);
    file.open();
    grids = file.getGrids();
    EXPECT_EQ(4, int(grids->size()));

    // Verify that "density_copy" is not an instance of "density".
    grid = findGridByName(*grids, "density");
    EXPECT_TRUE(grid.get() != nullptr);
    density = gridPtrCast<Int32Grid>(grid)->treePtr();
    EXPECT_TRUE(density.get() != nullptr);
#ifdef OPENVDB_USE_DELAYED_LOADING
    EXPECT_TRUE(density->unallocatedLeafCount() > 0);
    EXPECT_EQ(density->leafCount(), density->unallocatedLeafCount());
#endif // OPENVDB_USE_DELAYED_LOADING
    grid = findGridByName(*grids, "density_copy");
    EXPECT_TRUE(grid.get() != nullptr);
    EXPECT_TRUE(gridPtrCast<Int32Grid>(grid)->treePtr().get() != nullptr);
    EXPECT_TRUE(gridPtrCast<Int32Grid>(grid)->treePtr() != density);

    // Verify that the two unnamed grids are not instances of each other.
    grid = findGridByName(*grids, "");
    EXPECT_TRUE(grid.get() != nullptr);
    temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    EXPECT_TRUE(temperature.get() != nullptr);
    grid.reset();
    for (GridPtrVec::reverse_iterator it = grids->rbegin(); !grid && it != grids->rend(); ++it) {
        // Search for the second unnamed grid starting from the end of the list.
        if ((*it)->getName() == "") grid = *it;
    }
    EXPECT_TRUE(grid.get() != nullptr);
    EXPECT_TRUE(gridPtrCast<FloatGrid>(grid)->treePtr().get() != nullptr);
    EXPECT_TRUE(gridPtrCast<FloatGrid>(grid)->treePtr() != temperature);
}


void
TestFile::testReadGridDescriptors()
{
    using namespace openvdb;
    using namespace openvdb::io;

    using GridType = Int32Grid;
    using TreeType = GridType::TreeType;

    File file("something.vdb2");

    std::ostringstream ostr(std::ios_base::binary);

    // Create a grid with transform.
    GridType::Ptr grid = createGrid<GridType>(1);
    TreeType& tree = grid->tree();
    tree.setValue(Coord(10, 1, 2), 10);
    tree.setValue(Coord(0, 0, 0), 5);
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    grid->setTransform(trans);

    // Create another grid with transform.
    GridType::Ptr grid2 = createGrid<GridType>(2);
    TreeType& tree2 = grid2->tree();
    tree2.setValue(Coord(0, 0, 0), 10);
    tree2.setValue(Coord(1000, 1000, 1000), 50);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.2);
    grid2->setTransform(trans2);

    // Create the grid descriptor out of this grid.
    GridDescriptor gd(Name("temperature"), grid->type());
    GridDescriptor gd2(Name("density"), grid2->type());

    // Write out the number of grids.
    int32_t gridCount = 2;
    ostr.write(reinterpret_cast<char*>(&gridCount), sizeof(int32_t));
    // Write out the grids.
    file.writeGrid(gd, grid, ostr, /*seekable=*/true);
    file.writeGrid(gd2, grid2, ostr, /*seekable=*/true);

    // Register the grid and the transform and the blocks.
    GridBase::clearRegistry();
    GridType::registerGrid();
    // register maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::ScaleMap::registerMap();
    math::UniformScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();
    math::UniformScaleTranslateMap::registerMap();
    math::NonlinearFrustumMap::registerMap();

    // Read in the grid descriptors.
    File file2("something.vdb2");
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    io::setCurrentVersion(istr);
    file2.readGridDescriptors(istr);

    // Compare with the initial grid descriptors.
    File::NameMapCIter it = file2.findDescriptor("temperature");
    EXPECT_TRUE(it != file2.gridDescriptors().end());
    GridDescriptor file2gd = it->second;
    EXPECT_EQ(gd.gridName(), file2gd.gridName());
    EXPECT_EQ(gd.getGridPos(), file2gd.getGridPos());
    EXPECT_EQ(gd.getBlockPos(), file2gd.getBlockPos());
    EXPECT_EQ(gd.getEndPos(), file2gd.getEndPos());

    it = file2.findDescriptor("density");
    EXPECT_TRUE(it != file2.gridDescriptors().end());
    file2gd = it->second;
    EXPECT_EQ(gd2.gridName(), file2gd.gridName());
    EXPECT_EQ(gd2.getGridPos(), file2gd.getGridPos());
    EXPECT_EQ(gd2.getBlockPos(), file2gd.getBlockPos());
    EXPECT_EQ(gd2.getEndPos(), file2gd.getEndPos());

    // Clear registries.
    GridBase::clearRegistry();
    math::MapRegistry::clear();

    remove("something.vdb2");
}
TEST_F(TestFile, testReadGridDescriptors) { testReadGridDescriptors(); }


TEST_F(TestFile, testGridNaming)
{
    using namespace openvdb;
    using namespace openvdb::io;

    using TreeType = Int32Tree;

    // Register data types.
    openvdb::initialize();

    logging::LevelScope suppressLogging{logging::Level::Fatal};

    // Create several grids that share a single tree.
    TreeType::Ptr tree(new TreeType(1));
    tree->setValue(Coord(10, 1, 2), 10);
    tree->setValue(Coord(0, 0, 0), 5);
    GridBase::Ptr
        grid1 = openvdb::createGrid(tree),
        grid2 = openvdb::createGrid(tree),
        grid3 = openvdb::createGrid(tree);

    std::vector<GridBase::Ptr> gridVec;
    gridVec.push_back(grid1);
    gridVec.push_back(grid2);
    gridVec.push_back(grid3);

    // Give all grids the same name, but also some metadata to distinguish them.
    for (int n = 0; n <= 2; ++n) {
        gridVec[n]->setName("grid");
        gridVec[n]->insertMeta("index", Int32Metadata(n));
    }

    const char* filename = "testGridNaming.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);

    // Test first with grid instancing disabled, then with instancing enabled.
    for (int instancing = 0; instancing <= 1; ++instancing) {
        {
            // Write the grids out to a file.
            File file(filename);
            file.setInstancingEnabled(instancing);
            file.write(gridVec);
        }

        // Open the file for reading.
        File file(filename);
        file.setInstancingEnabled(instancing);
        file.open();

        int n = 0;
        for (File::NameIterator i = file.beginName(), e = file.endName(); i != e; ++i, ++n) {
            EXPECT_TRUE(file.hasGrid(i.gridName()));
        }
        // Verify that the file contains three grids.
        EXPECT_EQ(3, n);

        // Read each grid.
        for (n = -1; n <= 2; ++n) {
            openvdb::Name name("grid");

            // On the first iteration, read the grid named "grid", then read "grid[0]"
            // (which is synonymous with "grid"), then "grid[1]", then "grid[2]".
            if (n >= 0) {
                name = GridDescriptor::nameAsString(GridDescriptor::addSuffix(name, n));
            }

            EXPECT_TRUE(file.hasGrid(name));

            // Read the current grid.
            GridBase::ConstPtr grid = file.readGrid(name);
            EXPECT_TRUE(grid.get() != nullptr);

            // Verify that the grid is named "grid".
            EXPECT_EQ(openvdb::Name("grid"), grid->getName());
            EXPECT_EQ((n < 0 ? 0 : n), grid->metaValue<openvdb::Int32>("index"));
        }

        // Read all three grids at once.
        GridPtrVecPtr allGrids = file.getGrids();
        EXPECT_TRUE(allGrids.get() != nullptr);
        EXPECT_EQ(3, int(allGrids->size()));

        GridBase::ConstPtr firstGrid;
        std::vector<int> indices;
        for (GridPtrVecCIter i = allGrids->begin(), e = allGrids->end(); i != e; ++i) {
            GridBase::ConstPtr grid = *i;
            EXPECT_TRUE(grid.get() != nullptr);

            indices.push_back(grid->metaValue<openvdb::Int32>("index"));

            // If instancing is enabled, verify that all grids share the same tree.
            if (instancing) {
                if (!firstGrid) firstGrid = grid;
                EXPECT_EQ(firstGrid->baseTreePtr(), grid->baseTreePtr());
            }
        }
        // Verify that three distinct grids were read,
        // by examining their "index" metadata.
        EXPECT_EQ(3, int(indices.size()));
        std::sort(indices.begin(), indices.end());
        EXPECT_EQ(0, indices[0]);
        EXPECT_EQ(1, indices[1]);
        EXPECT_EQ(2, indices[2]);
    }

    {
        // Try writing and then reading a grid with a weird name
        // that might conflict with the grid name indexing scheme.
        const openvdb::Name weirdName("grid[4]");
        gridVec[0]->setName(weirdName);
        {
            File file(filename);
            file.write(gridVec);
        }
        File file(filename);
        file.open();

        // Verify that the grid can be read and that its index is 0.
        GridBase::ConstPtr grid = file.readGrid(weirdName);
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_EQ(weirdName, grid->getName());
        EXPECT_EQ(0, grid->metaValue<openvdb::Int32>("index"));

        // Verify that the other grids can still be read successfully.
        grid = file.readGrid("grid[0]");
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_EQ(openvdb::Name("grid"), grid->getName());
        // Because there are now only two grids named "grid", the one with
        // index 1 is now "grid[0]".
        EXPECT_EQ(1, grid->metaValue<openvdb::Int32>("index"));

        grid = file.readGrid("grid[1]");
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_EQ(openvdb::Name("grid"), grid->getName());
        // Because there are now only two grids named "grid", the one with
        // index 2 is now "grid[1]".
        EXPECT_EQ(2, grid->metaValue<openvdb::Int32>("index"));

        // Verify that there is no longer a third grid named "grid".
        EXPECT_THROW(file.readGrid("grid[2]"), openvdb::KeyError);
    }
}


TEST_F(TestFile, testEmptyFile)
{
    using namespace openvdb;
    using namespace openvdb::io;

    const char* filename = "testEmptyFile.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);

    {
        File file(filename);
        file.write(GridPtrVec(), MetaMap());
    }
    File file(filename);
    file.open();

    GridPtrVecPtr grids = file.getGrids();
    MetaMap::Ptr meta = file.getMetadata();

    EXPECT_TRUE(grids.get() != nullptr);
    EXPECT_TRUE(grids->empty());

    EXPECT_TRUE(meta.get() != nullptr);
    EXPECT_EQ(0, int(meta->metaCount()));
}


void
TestFile::testEmptyGridIO()
{
    using namespace openvdb;
    using namespace openvdb::io;

    using GridType = Int32Grid;

    logging::LevelScope suppressLogging{logging::Level::Fatal};

    const char* filename = "something.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);

    File file(filename);

    std::ostringstream ostr(std::ios_base::binary);

    // Create a grid with transform.
    GridType::Ptr grid = createGrid<GridType>(/*bg=*/1);
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    grid->setTransform(trans);

    // Create another grid with transform.
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.2);
    GridType::Ptr grid2 = createGrid<GridType>(/*bg=*/2);
    grid2->setTransform(trans2);

    // Create the grid descriptor out of this grid.
    GridDescriptor gd(Name("temperature"), grid->type());
    GridDescriptor gd2(Name("density"), grid2->type());

    // Write out the number of grids.
    int32_t gridCount = 2;
    ostr.write(reinterpret_cast<char*>(&gridCount), sizeof(int32_t));
    // Write out the grids.
    file.writeGrid(gd, grid, ostr, /*seekable=*/true);
    file.writeGrid(gd2, grid2, ostr, /*seekable=*/true);

    // Ensure that the block offset and the end offsets are equivalent.
    EXPECT_EQ(0, int(grid->baseTree().leafCount()));
    EXPECT_EQ(0, int(grid2->baseTree().leafCount()));
    EXPECT_EQ(gd.getEndPos(), gd.getBlockPos());
    EXPECT_EQ(gd2.getEndPos(), gd2.getBlockPos());

    // Register the grid and the transform and the blocks.
    GridBase::clearRegistry();
    GridType::registerGrid();
    // register maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::ScaleMap::registerMap();
    math::UniformScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();
    math::UniformScaleTranslateMap::registerMap();
    math::NonlinearFrustumMap::registerMap();

    // Read in the grid descriptors.
    File file2(filename);
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    io::setCurrentVersion(istr);
    file2.readGridDescriptors(istr);

    // Compare with the initial grid descriptors.
    File::NameMapCIter it = file2.findDescriptor("temperature");
    EXPECT_TRUE(it != file2.gridDescriptors().end());
    GridDescriptor file2gd = it->second;
    file2gd.seekToGrid(istr);
    GridBase::Ptr gd_grid = GridBase::createGrid(file2gd.gridType());
    Archive::readGridCompression(istr);
    gd_grid->readMeta(istr);
    gd_grid->readTransform(istr);
    gd_grid->readTopology(istr);
    EXPECT_EQ(gd.gridName(), file2gd.gridName());
    EXPECT_TRUE(gd_grid.get() != nullptr);
    EXPECT_EQ(0, int(gd_grid->baseTree().leafCount()));
    //EXPECT_EQ(8, int(gd_grid->baseTree().nonLeafCount()));
    EXPECT_EQ(4, int(gd_grid->baseTree().treeDepth()));
    EXPECT_EQ(gd.getGridPos(), file2gd.getGridPos());
    EXPECT_EQ(gd.getBlockPos(), file2gd.getBlockPos());
    EXPECT_EQ(gd.getEndPos(), file2gd.getEndPos());

    it = file2.findDescriptor("density");
    EXPECT_TRUE(it != file2.gridDescriptors().end());
    file2gd = it->second;
    file2gd.seekToGrid(istr);
    gd_grid = GridBase::createGrid(file2gd.gridType());
    Archive::readGridCompression(istr);
    gd_grid->readMeta(istr);
    gd_grid->readTransform(istr);
    gd_grid->readTopology(istr);
    EXPECT_EQ(gd2.gridName(), file2gd.gridName());
    EXPECT_TRUE(gd_grid.get() != nullptr);
    EXPECT_EQ(0, int(gd_grid->baseTree().leafCount()));
    //EXPECT_EQ(8, int(gd_grid->nonLeafCount()));
    EXPECT_EQ(4, int(gd_grid->baseTree().treeDepth()));
    EXPECT_EQ(gd2.getGridPos(), file2gd.getGridPos());
    EXPECT_EQ(gd2.getBlockPos(), file2gd.getBlockPos());
    EXPECT_EQ(gd2.getEndPos(), file2gd.getEndPos());

    // Clear registries.
    GridBase::clearRegistry();
    math::MapRegistry::clear();
}
TEST_F(TestFile, testEmptyGridIO) { testEmptyGridIO(); }


void TestFile::testOpen()
{
    using namespace openvdb;

    using FloatGrid = openvdb::FloatGrid;
    using IntGrid = openvdb::Int32Grid;
    using FloatTree = FloatGrid::TreeType;
    using IntTree = Int32Grid::TreeType;

    // Create a VDB to write.

    // Create grids
    IntGrid::Ptr grid = createGrid<IntGrid>(/*bg=*/1);
    IntTree& tree = grid->tree();
    grid->setName("density");

    FloatGrid::Ptr grid2 = createGrid<FloatGrid>(/*bg=*/2.0);
    FloatTree& tree2 = grid2->tree();
    grid2->setName("temperature");

    // Create transforms
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid->setTransform(trans);
    grid2->setTransform(trans2);

    // Set some values
    tree.setValue(Coord(0, 0, 0), 5);
    tree.setValue(Coord(100, 0, 0), 6);
    tree2.setValue(Coord(0, 0, 0), 10);
    tree2.setValue(Coord(0, 100, 0), 11);

    MetaMap meta;
    meta.insertMeta("author", StringMetadata("Einstein"));
    meta.insertMeta("year", Int32Metadata(2009));

    GridPtrVec grids;
    grids.push_back(grid);
    grids.push_back(grid2);

    EXPECT_TRUE(findGridByName(grids, "density") == grid);
    EXPECT_TRUE(findGridByName(grids, "temperature") == grid2);
    EXPECT_TRUE(meta.metaValue<std::string>("author") == "Einstein");
    EXPECT_EQ(2009, meta.metaValue<int32_t>("year"));

    // Register grid and transform.
    GridBase::clearRegistry();
    IntGrid::registerGrid();
    FloatGrid::registerGrid();
    Metadata::clearRegistry();
    StringMetadata::registerType();
    Int32Metadata::registerType();
    // register maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::ScaleMap::registerMap();
    math::UniformScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();
    math::UniformScaleTranslateMap::registerMap();
    math::NonlinearFrustumMap::registerMap();

    // Write the vdb out to a file.
    io::File vdbfile("something.vdb2");
    vdbfile.write(grids, meta);

    // Now we can read in the file.
    EXPECT_TRUE(!vdbfile.open());//opening the same file
    // Can't open same file multiple times without closing.
    EXPECT_THROW(vdbfile.open(), openvdb::IoError);
    vdbfile.close();
    EXPECT_TRUE(!vdbfile.open());//opening the same file

    EXPECT_TRUE(vdbfile.isOpen());

    uint32_t version = OPENVDB_FILE_VERSION;

    EXPECT_EQ(version, vdbfile.fileVersion());
    EXPECT_EQ(version, io::getFormatVersion(vdbfile.inputStream()));
    EXPECT_EQ(OPENVDB_LIBRARY_MAJOR_VERSION, vdbfile.libraryVersion().first);
    EXPECT_EQ(OPENVDB_LIBRARY_MINOR_VERSION, vdbfile.libraryVersion().second);
    EXPECT_EQ(OPENVDB_LIBRARY_MAJOR_VERSION,
        io::getLibraryVersion(vdbfile.inputStream()).first);
    EXPECT_EQ(OPENVDB_LIBRARY_MINOR_VERSION,
        io::getLibraryVersion(vdbfile.inputStream()).second);

    // Ensure that we read in the vdb metadata.
    EXPECT_TRUE(vdbfile.getMetadata());
    EXPECT_TRUE(vdbfile.getMetadata()->metaValue<std::string>("author") == "Einstein");
    EXPECT_EQ(2009, vdbfile.getMetadata()->metaValue<int32_t>("year"));

    // Ensure we got the grid descriptors.
    EXPECT_EQ(1, int(vdbfile.gridDescriptors().count("density")));
    EXPECT_EQ(1, int(vdbfile.gridDescriptors().count("temperature")));

    io::File::NameMapCIter it = vdbfile.findDescriptor("density");
    EXPECT_TRUE(it != vdbfile.gridDescriptors().end());
    io::GridDescriptor gd = it->second;
    EXPECT_EQ(IntTree::treeType(), gd.gridType());

    it = vdbfile.findDescriptor("temperature");
    EXPECT_TRUE(it != vdbfile.gridDescriptors().end());
    gd = it->second;
    EXPECT_EQ(FloatTree::treeType(), gd.gridType());

    // Ensure we throw an error if there is no file.
    io::File vdbfile2("somethingelses.vdb2");
    EXPECT_THROW(vdbfile2.open(), openvdb::IoError);
    EXPECT_THROW(vdbfile2.inputStream(), openvdb::IoError);

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    // Test closing the file.
    vdbfile.close();
    EXPECT_TRUE(vdbfile.isOpen() == false);
    EXPECT_TRUE(vdbfile.fileMetadata().get() == nullptr);
    EXPECT_EQ(0, int(vdbfile.gridDescriptors().size()));
    EXPECT_THROW(vdbfile.inputStream(), openvdb::IoError);

    remove("something.vdb2");
}
TEST_F(TestFile, testOpen) { testOpen(); }


void
TestFile::testNonVdbOpen()
{
    std::ofstream file("dummy.vdb2", std::ios_base::binary);

    int64_t something = 1;
    file.write(reinterpret_cast<char*>(&something), sizeof(int64_t));

    file.close();

    openvdb::io::File vdbfile("dummy.vdb2");
    EXPECT_THROW(vdbfile.open(), openvdb::IoError);
    EXPECT_THROW(vdbfile.inputStream(), openvdb::IoError);

    remove("dummy.vdb2");
}
TEST_F(TestFile, testNonVdbOpen) { testNonVdbOpen(); }


TEST_F(TestFile, testGetMetadata)
{
    using namespace openvdb;

    GridPtrVec grids;
    MetaMap meta;

    meta.insertMeta("author", StringMetadata("Einstein"));
    meta.insertMeta("year", Int32Metadata(2009));

    // Adjust registry before writing.
    Metadata::clearRegistry();
    StringMetadata::registerType();
    Int32Metadata::registerType();

    // Write the vdb out to a file.
    io::File vdbfile("something.vdb2");
    vdbfile.write(grids, meta);

    // Check if reading without opening the file
    EXPECT_THROW(vdbfile.getMetadata(), openvdb::IoError);

    vdbfile.open();

    MetaMap::Ptr meta2 = vdbfile.getMetadata();

    EXPECT_EQ(2, int(meta2->metaCount()));

    EXPECT_TRUE(meta2->metaValue<std::string>("author") == "Einstein");
    EXPECT_EQ(2009, meta2->metaValue<int32_t>("year"));

    // Clear registry.
    Metadata::clearRegistry();

    remove("something.vdb2");
}


TEST_F(TestFile, testReadAll)
{
    using namespace openvdb;

    using FloatGrid = openvdb::FloatGrid;
    using IntGrid = openvdb::Int32Grid;
    using FloatTree = FloatGrid::TreeType;
    using IntTree = Int32Grid::TreeType;

    // Create a vdb to write.

    // Create grids
    IntGrid::Ptr grid1 = createGrid<IntGrid>(/*bg=*/1);
    IntTree& tree = grid1->tree();
    grid1->setName("density");

    FloatGrid::Ptr grid2 = createGrid<FloatGrid>(/*bg=*/2.0);
    FloatTree& tree2 = grid2->tree();
    grid2->setName("temperature");

    // Create transforms
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid1->setTransform(trans);
    grid2->setTransform(trans2);

    // Set some values
    tree.setValue(Coord(0, 0, 0), 5);
    tree.setValue(Coord(100, 0, 0), 6);
    tree2.setValue(Coord(0, 0, 0), 10);
    tree2.setValue(Coord(0, 100, 0), 11);

    MetaMap meta;
    meta.insertMeta("author", StringMetadata("Einstein"));
    meta.insertMeta("year", Int32Metadata(2009));

    GridPtrVec grids;
    grids.push_back(grid1);
    grids.push_back(grid2);

    // Register grid and transform.
    openvdb::initialize();

    // Write the vdb out to a file.
    io::File vdbfile("something.vdb2");
    vdbfile.write(grids, meta);

    io::File vdbfile2("something.vdb2");
    EXPECT_THROW(vdbfile2.getGrids(), openvdb::IoError);

    vdbfile2.open();
    EXPECT_TRUE(vdbfile2.isOpen());

    GridPtrVecPtr grids2 = vdbfile2.getGrids();
    MetaMap::Ptr meta2 = vdbfile2.getMetadata();

    // Ensure we have the metadata.
    EXPECT_EQ(2, int(meta2->metaCount()));
    EXPECT_TRUE(meta2->metaValue<std::string>("author") == "Einstein");
    EXPECT_EQ(2009, meta2->metaValue<int32_t>("year"));

    // Ensure we got the grids.
    EXPECT_EQ(2, int(grids2->size()));

    GridBase::Ptr grid;
    grid.reset();
    grid = findGridByName(*grids2, "density");
    EXPECT_TRUE(grid.get() != nullptr);
    IntTree::Ptr density = gridPtrCast<IntGrid>(grid)->treePtr();
    EXPECT_TRUE(density.get() != nullptr);

    grid.reset();
    grid = findGridByName(*grids2, "temperature");
    EXPECT_TRUE(grid.get() != nullptr);
    FloatTree::Ptr temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    EXPECT_TRUE(temperature.get() != nullptr);

    EXPECT_NEAR(5, density->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    EXPECT_NEAR(6, density->getValue(Coord(100, 0, 0)), /*tolerance=*/0);
    EXPECT_NEAR(10, temperature->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    EXPECT_NEAR(11, temperature->getValue(Coord(0, 100, 0)), /*tolerance=*/0);

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    vdbfile2.close();

    remove("something.vdb2");
}


TEST_F(TestFile, testWriteOpenFile)
{
    using namespace openvdb;

    MetaMap::Ptr meta(new MetaMap);
    meta->insertMeta("author", StringMetadata("Einstein"));
    meta->insertMeta("year", Int32Metadata(2009));

    // Register metadata
    Metadata::clearRegistry();
    StringMetadata::registerType();
    Int32Metadata::registerType();

    // Write the metadata out to a file.
    io::File vdbfile("something.vdb2");
    vdbfile.write(GridPtrVec(), *meta);

    io::File vdbfile2("something.vdb2");
    EXPECT_THROW(vdbfile2.getGrids(), openvdb::IoError);

    vdbfile2.open();
    EXPECT_TRUE(vdbfile2.isOpen());

    GridPtrVecPtr grids = vdbfile2.getGrids();
    meta = vdbfile2.getMetadata();

    // Ensure we have the metadata.
    EXPECT_TRUE(meta.get() != nullptr);
    EXPECT_EQ(2, int(meta->metaCount()));
    EXPECT_TRUE(meta->metaValue<std::string>("author") == "Einstein");
    EXPECT_EQ(2009, meta->metaValue<int32_t>("year"));

    // Ensure we got the grids.
    EXPECT_TRUE(grids.get() != nullptr);
    EXPECT_EQ(0, int(grids->size()));

    // Cannot write an open file.
    EXPECT_THROW(vdbfile2.write(*grids), openvdb::IoError);

    vdbfile2.close();

    EXPECT_NO_THROW(vdbfile2.write(*grids));

    // Clear registries.
    Metadata::clearRegistry();

    remove("something.vdb2");
}


TEST_F(TestFile, testReadGridMetadata)
{
    using namespace openvdb;

    openvdb::initialize();

    const char* filename = "testReadGridMetadata.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);

    // Create grids
    Int32Grid::Ptr igrid = createGrid<Int32Grid>(/*bg=*/1);
    FloatGrid::Ptr fgrid = createGrid<FloatGrid>(/*bg=*/2.0);

    // Add metadata.
    igrid->setName("igrid");
    igrid->insertMeta("author", StringMetadata("Einstein"));
    igrid->insertMeta("year", Int32Metadata(2012));

    fgrid->setName("fgrid");
    fgrid->insertMeta("author", StringMetadata("Einstein"));
    fgrid->insertMeta("year", Int32Metadata(2012));

    // Add transforms.
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    igrid->setTransform(trans);
    fgrid->setTransform(trans);

    // Set some values.
    igrid->tree().setValue(Coord(0, 0, 0), 5);
    igrid->tree().setValue(Coord(100, 0, 0), 6);
    fgrid->tree().setValue(Coord(0, 0, 0), 10);
    fgrid->tree().setValue(Coord(0, 100, 0), 11);

    GridPtrVec srcGrids;
    srcGrids.push_back(igrid);
    srcGrids.push_back(fgrid);
    std::map<std::string, GridBase::Ptr> srcGridMap;
    srcGridMap[igrid->getName()] = igrid;
    srcGridMap[fgrid->getName()] = fgrid;

    enum { OUTPUT_TO_FILE = 0, OUTPUT_TO_STREAM = 1 };
    for (int outputMethod = OUTPUT_TO_FILE; outputMethod <= OUTPUT_TO_STREAM; ++outputMethod)
    {
        if (outputMethod == OUTPUT_TO_FILE) {
            // Write the grids to a file.
            io::File vdbfile(filename);
            vdbfile.write(srcGrids);
        } else {
            // Stream the grids to a file (i.e., without file offsets).
            std::ofstream ostrm(filename, std::ios_base::binary);
            io::Stream(ostrm).write(srcGrids);
        }

        // Read just the grid-level metadata from the file.
        io::File vdbfile(filename);

        // Verify that reading from an unopened file generates an exception.
        EXPECT_THROW(vdbfile.readGridMetadata("igrid"), openvdb::IoError);
        EXPECT_THROW(vdbfile.readGridMetadata("noname"), openvdb::IoError);
        EXPECT_THROW(vdbfile.readAllGridMetadata(), openvdb::IoError);

        vdbfile.open();

        EXPECT_TRUE(vdbfile.isOpen());

        // Verify that reading a nonexistent grid generates an exception.
        EXPECT_THROW(vdbfile.readGridMetadata("noname"), openvdb::KeyError);

        // Read all grids and store them in a list.
        GridPtrVecPtr gridMetadata = vdbfile.readAllGridMetadata();
        EXPECT_TRUE(gridMetadata.get() != nullptr);
        EXPECT_EQ(2, int(gridMetadata->size()));

        // Read individual grids and append them to the list.
        GridBase::Ptr grid = vdbfile.readGridMetadata("igrid");
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_EQ(std::string("igrid"), grid->getName());
        gridMetadata->push_back(grid);

        grid = vdbfile.readGridMetadata("fgrid");
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_EQ(std::string("fgrid"), grid->getName());
        gridMetadata->push_back(grid);

        // Verify that the grids' metadata and transforms match the original grids'.
        for (size_t i = 0, N = gridMetadata->size(); i < N; ++i) {
            grid = (*gridMetadata)[i];

            EXPECT_TRUE(grid.get() != nullptr);
            EXPECT_TRUE(grid->getName() == "igrid" || grid->getName() == "fgrid");
            EXPECT_TRUE(grid->baseTreePtr().get() != nullptr);

            // Since we didn't read the grid's topology, the tree should be empty.
            EXPECT_EQ(0, int(grid->constBaseTreePtr()->leafCount()));
            EXPECT_EQ(0, int(grid->constBaseTreePtr()->activeVoxelCount()));

            // Retrieve the source grid of the same name.
            GridBase::ConstPtr srcGrid = srcGridMap[grid->getName()];

            // Compare grid types and transforms.
            EXPECT_EQ(srcGrid->type(), grid->type());
            EXPECT_EQ(srcGrid->transform(), grid->transform());

            // Compare metadata, ignoring fields that were added when the file was written.
            MetaMap::Ptr
                statsMetadata = grid->getStatsMetadata(),
                otherMetadata = grid->copyMeta(); // shallow copy
            EXPECT_TRUE(statsMetadata->metaCount() != 0);
            statsMetadata->insertMeta(GridBase::META_FILE_COMPRESSION, StringMetadata(""));
            for (MetaMap::ConstMetaIterator it = grid->beginMeta(), end = grid->endMeta();
                it != end; ++it)
            {
                // Keep all fields that exist in the source grid.
                if ((*srcGrid)[it->first]) continue;
                // Remove any remaining grid statistics fields.
                if ((*statsMetadata)[it->first]) {
                    otherMetadata->removeMeta(it->first);
                }
                // Remove delay load metadata if it exists.
                if ((*otherMetadata)["file_delayed_load"]) {
                    otherMetadata->removeMeta("file_delayed_load");
                }
            }
            EXPECT_EQ(srcGrid->str(), otherMetadata->str());

            const CoordBBox srcBBox = srcGrid->evalActiveVoxelBoundingBox();
            EXPECT_EQ(srcBBox.min().asVec3i(), grid->metaValue<Vec3i>("file_bbox_min"));
            EXPECT_EQ(srcBBox.max().asVec3i(), grid->metaValue<Vec3i>("file_bbox_max"));
            EXPECT_EQ(srcGrid->activeVoxelCount(),
                Index64(grid->metaValue<Int64>("file_voxel_count")));
            EXPECT_EQ(srcGrid->memUsage(),
                Index64(grid->metaValue<Int64>("file_mem_bytes")));
        }
    }
}


TEST_F(TestFile, testReadGrid)
{
    using namespace openvdb;

    using FloatGrid = openvdb::FloatGrid;
    using IntGrid = openvdb::Int32Grid;
    using FloatTree = FloatGrid::TreeType;
    using IntTree = Int32Grid::TreeType;

    // Create a vdb to write.

    // Create grids
    IntGrid::Ptr grid = createGrid<IntGrid>(/*bg=*/1);
    IntTree& tree = grid->tree();
    grid->setName("density");

    FloatGrid::Ptr grid2 = createGrid<FloatGrid>(/*bg=*/2.0);
    FloatTree& tree2 = grid2->tree();
    grid2->setName("temperature");

    // Create transforms
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid->setTransform(trans);
    grid2->setTransform(trans2);

    // Set some values
    tree.setValue(Coord(0, 0, 0), 5);
    tree.setValue(Coord(100, 0, 0), 6);
    tree2.setValue(Coord(0, 0, 0), 10);
    tree2.setValue(Coord(0, 100, 0), 11);

    MetaMap meta;
    meta.insertMeta("author", StringMetadata("Einstein"));
    meta.insertMeta("year", Int32Metadata(2009));

    GridPtrVec grids;
    grids.push_back(grid);
    grids.push_back(grid2);

    // Register grid and transform.
    openvdb::initialize();

    // Write the vdb out to a file.
    io::File vdbfile("something.vdb2");
    vdbfile.write(grids, meta);

    io::File vdbfile2("something.vdb2");

    vdbfile2.open();

    EXPECT_TRUE(vdbfile2.isOpen());

    // Get Temperature
    GridBase::Ptr temperature = vdbfile2.readGrid("temperature");

    EXPECT_TRUE(temperature.get() != nullptr);

    FloatTree::Ptr typedTemperature = gridPtrCast<FloatGrid>(temperature)->treePtr();

    EXPECT_TRUE(typedTemperature.get() != nullptr);

    EXPECT_NEAR(10, typedTemperature->getValue(Coord(0, 0, 0)), 0);
    EXPECT_NEAR(11, typedTemperature->getValue(Coord(0, 100, 0)), 0);

    // Get Density
    GridBase::Ptr density = vdbfile2.readGrid("density");

    EXPECT_TRUE(density.get() != nullptr);

    IntTree::Ptr typedDensity = gridPtrCast<IntGrid>(density)->treePtr();

    EXPECT_TRUE(typedDensity.get() != nullptr);

    EXPECT_NEAR(5,typedDensity->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    EXPECT_NEAR(6,typedDensity->getValue(Coord(100, 0, 0)), /*tolerance=*/0);

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    vdbfile2.close();

    remove("something.vdb2");
}


////////////////////////////////////////


template<typename GridT>
void
validateClippedGrid(const GridT& clipped, const typename GridT::ValueType& fg)
{
    using namespace openvdb;

    using ValueT = typename GridT::ValueType;

    const CoordBBox bbox = clipped.evalActiveVoxelBoundingBox();
    EXPECT_EQ(4, bbox.min().x());
    EXPECT_EQ(4, bbox.min().y());
    EXPECT_EQ(-6, bbox.min().z());
    EXPECT_EQ(4, bbox.max().x());
    EXPECT_EQ(4, bbox.max().y());
    EXPECT_EQ(6, bbox.max().z());
    EXPECT_EQ(6 + 6 + 1, int(clipped.activeVoxelCount()));
    EXPECT_EQ(2, int(clipped.constTree().leafCount()));

    typename GridT::ConstAccessor acc = clipped.getConstAccessor();
    const ValueT bg = clipped.background();
    Coord xyz;
    int &x = xyz[0], &y = xyz[1], &z = xyz[2];
    for (x = -10; x <= 10; ++x) {
        for (y = -10; y <= 10; ++y) {
            for (z = -10; z <= 10; ++z) {
                if (x == 4 && y == 4 && z >= -6 && z <= 6) {
                    EXPECT_EQ(fg, acc.getValue(Coord(4, 4, z)));
                } else {
                    EXPECT_EQ(bg, acc.getValue(Coord(x, y, z)));
                }
            }
        }
    }
}


// See also TestGrid::testClipping()
TEST_F(TestFile, testReadClippedGrid)
{
    using namespace openvdb;

    // Register types.
    openvdb::initialize();

    // World-space clipping region
    const BBoxd clipBox(Vec3d(4.0, 4.0, -6.0), Vec3d(4.9, 4.9, 6.0));

    // Create grids of several types and fill a cubic region of each with a foreground value.

    const bool bfg = true;
    BoolGrid::Ptr bgrid = BoolGrid::create(/*bg=*/zeroVal<bool>());
    bgrid->setName("bgrid");
    bgrid->fill(CoordBBox(Coord(-10), Coord(10)), /*value=*/bfg, /*active=*/true);

    const float ffg = 5.f;
    FloatGrid::Ptr fgrid = FloatGrid::create(/*bg=*/zeroVal<float>());
    fgrid->setName("fgrid");
    fgrid->fill(CoordBBox(Coord(-10), Coord(10)), /*value=*/ffg, /*active=*/true);

    const Vec3s vfg(1.f, -2.f, 3.f);
    Vec3SGrid::Ptr vgrid = Vec3SGrid::create(/*bg=*/zeroVal<Vec3s>());
    vgrid->setName("vgrid");
    vgrid->fill(CoordBBox(Coord(-10), Coord(10)), /*value=*/vfg, /*active=*/true);

    GridPtrVec srcGrids;
    srcGrids.push_back(bgrid);
    srcGrids.push_back(fgrid);
    srcGrids.push_back(vgrid);

    const char* filename = "testReadClippedGrid.vdb";
    SharedPtr<const char> scopedFile(filename, ::remove);

    enum { OUTPUT_TO_FILE = 0, OUTPUT_TO_STREAM = 1 };
    for (int outputMethod = OUTPUT_TO_FILE; outputMethod <= OUTPUT_TO_STREAM; ++outputMethod)
    {
        if (outputMethod == OUTPUT_TO_FILE) {
            // Write the grids to a file.
            io::File vdbfile(filename);
            vdbfile.write(srcGrids);
        } else {
            // Stream the grids to a file (i.e., without file offsets).
            std::ofstream ostrm(filename, std::ios_base::binary);
            io::Stream(ostrm).write(srcGrids);
        }

        // Open the file for reading.
        io::File vdbfile(filename);
        vdbfile.open();

        GridBase::Ptr grid;

        // Read and clip each grid.

        EXPECT_NO_THROW(grid = vdbfile.readGrid("bgrid", clipBox));
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_NO_THROW(bgrid = gridPtrCast<BoolGrid>(grid));
        validateClippedGrid(*bgrid, bfg);

        EXPECT_NO_THROW(grid = vdbfile.readGrid("fgrid", clipBox));
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_NO_THROW(fgrid = gridPtrCast<FloatGrid>(grid));
        validateClippedGrid(*fgrid, ffg);

        EXPECT_NO_THROW(grid = vdbfile.readGrid("vgrid", clipBox));
        EXPECT_TRUE(grid.get() != nullptr);
        EXPECT_NO_THROW(vgrid = gridPtrCast<Vec3SGrid>(grid));
        validateClippedGrid(*vgrid, vfg);
    }
}


////////////////////////////////////////


namespace {

template<typename T, openvdb::Index Log2Dim> struct MultiPassLeafNode; // forward declaration

// Dummy value type
using MultiPassValue = openvdb::PointIndex<openvdb::Index32, 1000>;

// Tree configured to match the default OpenVDB configuration
using MultiPassTree = openvdb::tree::Tree<
    openvdb::tree::RootNode<
    openvdb::tree::InternalNode<
    openvdb::tree::InternalNode<
    MultiPassLeafNode<MultiPassValue, 3>, 4>, 5>>>;

using MultiPassGrid = openvdb::Grid<MultiPassTree>;


template<typename T, openvdb::Index Log2Dim>
struct MultiPassLeafNode: public openvdb::tree::LeafNode<T, Log2Dim>, openvdb::io::MultiPass
{
    // The following had to be copied from the LeafNode class
    // to make the derived class compatible with the tree structure.

    using LeafNodeType  = MultiPassLeafNode;
    using Ptr           = openvdb::SharedPtr<MultiPassLeafNode>;
    using BaseLeaf      = openvdb::tree::LeafNode<T, Log2Dim>;
    using NodeMaskType  = openvdb::util::NodeMask<Log2Dim>;
    using ValueType     = T;
    using ValueOnCIter  = typename BaseLeaf::template ValueIter<typename NodeMaskType::OnIterator,
        const MultiPassLeafNode, const ValueType, typename BaseLeaf::ValueOn>;
    using ChildOnIter = typename BaseLeaf::template ChildIter<typename NodeMaskType::OnIterator,
        MultiPassLeafNode, typename BaseLeaf::ChildOn>;
    using ChildOnCIter = typename BaseLeaf::template ChildIter<
        typename NodeMaskType::OnIterator, const MultiPassLeafNode, typename BaseLeaf::ChildOn>;

    MultiPassLeafNode(const openvdb::Coord& coords, const T& value, bool active = false)
        : BaseLeaf(coords, value, active) {}
    MultiPassLeafNode(openvdb::PartialCreate, const openvdb::Coord& coords, const T& value,
        bool active = false): BaseLeaf(openvdb::PartialCreate(), coords, value, active) {}
    MultiPassLeafNode(const MultiPassLeafNode& rhs): BaseLeaf(rhs) {}

    ValueOnCIter cbeginValueOn() const { return ValueOnCIter(this->getValueMask().beginOn(),this); }
    ChildOnCIter cbeginChildOn() const { return ChildOnCIter(this->getValueMask().endOn(), this); }
    ChildOnIter   beginChildOn()       { return ChildOnIter(this->getValueMask().endOn(), this); }

    // Methods in use for reading and writing multiple buffers

    void readBuffers(std::istream& is, const openvdb::CoordBBox&, bool fromHalf = false)
    {
        this->readBuffers(is, fromHalf);
    }

    void readBuffers(std::istream& is, bool /*fromHalf*/ = false)
    {
        const openvdb::io::StreamMetadata::Ptr meta = openvdb::io::getStreamMetadataPtr(is);
        if (!meta) {
            OPENVDB_THROW(openvdb::IoError,
                "Cannot write out a MultiBufferLeaf without StreamMetadata.");
        }

        // clamp pass to 16-bit integer
        const uint32_t pass(static_cast<uint16_t>(meta->pass()));

        // Read in the stored pass number.
        uint32_t readPass;
        is.read(reinterpret_cast<char*>(&readPass), sizeof(uint32_t));
        EXPECT_EQ(pass, readPass);
        // Record the pass number.
        mReadPasses.push_back(readPass);

        if (pass == 0) {
            // Read in the node's origin.
            openvdb::Coord origin;
            is.read(reinterpret_cast<char*>(&origin), sizeof(openvdb::Coord));
            EXPECT_EQ(origin, this->origin());
        }
    }

    void writeBuffers(std::ostream& os, bool /*toHalf*/ = false) const
    {
        const openvdb::io::StreamMetadata::Ptr meta = openvdb::io::getStreamMetadataPtr(os);
        if (!meta) {
            OPENVDB_THROW(openvdb::IoError,
                "Cannot read in a MultiBufferLeaf without StreamMetadata.");
        }

        // clamp pass to 16-bit integer
        const uint32_t pass(static_cast<uint16_t>(meta->pass()));

        // Leaf traversal analysis deduces the number of passes to perform for this leaf
        // then updates the leaf traversal value to ensure all passes will be written.
        if (meta->countingPasses()) {
            if (mNumPasses > pass) meta->setPass(mNumPasses);
            return;
        }

        // Record the pass number.
        EXPECT_TRUE(mWritePassesPtr);
        const_cast<std::vector<int>&>(*mWritePassesPtr).push_back(pass);

        // Write out the pass number.
        os.write(reinterpret_cast<const char*>(&pass), sizeof(uint32_t));
        if (pass == 0) {
            // Write out the node's origin and the pass number.
            const auto origin = this->origin();
            os.write(reinterpret_cast<const char*>(&origin), sizeof(openvdb::Coord));
        }
    }


    uint32_t mNumPasses = 0;
    // Pointer to external vector in which to record passes as they are written
    std::vector<int>* mWritePassesPtr = nullptr;
    // Vector in which to record passes as they are read
    // (this needs to be internal, because leaf nodes are constructed as a grid is read)
    std::vector<int> mReadPasses;
}; // struct MultiPassLeafNode

} // anonymous namespace


TEST_F(TestFile, testMultiPassIO)
{
    using namespace openvdb;

    openvdb::initialize();
    MultiPassGrid::registerGrid();

    // Create a multi-buffer grid.
    const MultiPassGrid::Ptr grid = openvdb::createGrid<MultiPassGrid>();
    grid->setName("test");
    grid->setTransform(math::Transform::createLinearTransform(1.0));
    MultiPassGrid::TreeType& tree = grid->tree();
    tree.setValue(Coord(0, 0, 0), 5);
    tree.setValue(Coord(0, 10, 0), 5);
    EXPECT_EQ(2, int(tree.leafCount()));

    const GridPtrVec grids{grid};

    // Vector in which to record pass numbers (to ensure blocked ordering)
    std::vector<int> writePasses;
    {
        // Specify the required number of I/O passes for each leaf node.
        MultiPassGrid::TreeType::LeafIter leafIter = tree.beginLeaf();
        leafIter->mNumPasses = 3;
        leafIter->mWritePassesPtr = &writePasses;
        ++leafIter;
        leafIter->mNumPasses = 2;
        leafIter->mWritePassesPtr = &writePasses;
    }

    const char* filename = "testMultiPassIO.vdb";
    SharedPtr<const char> scopedFile(filename, ::remove);
    {
        // Verify that passes are written to a file in the correct order.
        io::File(filename).write(grids);
        EXPECT_EQ(6, int(writePasses.size()));
        EXPECT_EQ(0, writePasses[0]); // leaf 0
        EXPECT_EQ(0, writePasses[1]); // leaf 1
        EXPECT_EQ(1, writePasses[2]); // leaf 0
        EXPECT_EQ(1, writePasses[3]); // leaf 1
        EXPECT_EQ(2, writePasses[4]); // leaf 0
        EXPECT_EQ(2, writePasses[5]); // leaf 1
    }
    {
        // Verify that passes are read in the correct order.
        io::File file(filename);
        file.open();
        const auto newGrid = GridBase::grid<MultiPassGrid>(file.readGrid("test"));

        auto leafIter = newGrid->tree().beginLeaf();
        EXPECT_EQ(3, int(leafIter->mReadPasses.size()));
        EXPECT_EQ(0, leafIter->mReadPasses[0]);
        EXPECT_EQ(1, leafIter->mReadPasses[1]);
        EXPECT_EQ(2, leafIter->mReadPasses[2]);
        ++leafIter;
        EXPECT_EQ(3, int(leafIter->mReadPasses.size()));
        EXPECT_EQ(0, leafIter->mReadPasses[0]);
        EXPECT_EQ(1, leafIter->mReadPasses[1]);
        EXPECT_EQ(2, leafIter->mReadPasses[2]);
    }
    {
        // Verify that when using multi-pass and bbox clipping that each leaf node
        // is still being read before being clipped
        io::File file(filename);
        file.open();
        const auto newGrid = GridBase::grid<MultiPassGrid>(
            file.readGrid("test", BBoxd(Vec3d(0), Vec3d(1))));
        EXPECT_EQ(Index32(1), newGrid->tree().leafCount());

        auto leafIter = newGrid->tree().beginLeaf();
        EXPECT_EQ(3, int(leafIter->mReadPasses.size()));
        EXPECT_EQ(0, leafIter->mReadPasses[0]);
        EXPECT_EQ(1, leafIter->mReadPasses[1]);
        EXPECT_EQ(2, leafIter->mReadPasses[2]);
        ++leafIter;
        EXPECT_TRUE(!leafIter); // second leaf node has now been clipped
    }

    // Clear the pass data.
    writePasses.clear();

    {
        // Verify that passes are written to and read from a non-seekable stream
        // in the correct order.
        std::ostringstream ostr(std::ios_base::binary);
        io::Stream(ostr).write(grids);

        EXPECT_EQ(6, int(writePasses.size()));
        EXPECT_EQ(0, writePasses[0]); // leaf 0
        EXPECT_EQ(0, writePasses[1]); // leaf 1
        EXPECT_EQ(1, writePasses[2]); // leaf 0
        EXPECT_EQ(1, writePasses[3]); // leaf 1
        EXPECT_EQ(2, writePasses[4]); // leaf 0
        EXPECT_EQ(2, writePasses[5]); // leaf 1

        std::istringstream is(ostr.str(), std::ios_base::binary);
        io::Stream strm(is);
        const auto streamedGrids = strm.getGrids();
        EXPECT_EQ(1, int(streamedGrids->size()));

        const auto newGrid = gridPtrCast<MultiPassGrid>(*streamedGrids->begin());
        EXPECT_TRUE(bool(newGrid));
        auto leafIter = newGrid->tree().beginLeaf();
        EXPECT_EQ(3, int(leafIter->mReadPasses.size()));
        EXPECT_EQ(0, leafIter->mReadPasses[0]);
        EXPECT_EQ(1, leafIter->mReadPasses[1]);
        EXPECT_EQ(2, leafIter->mReadPasses[2]);
        ++leafIter;
        EXPECT_EQ(3, int(leafIter->mReadPasses.size()));
        EXPECT_EQ(0, leafIter->mReadPasses[0]);
        EXPECT_EQ(1, leafIter->mReadPasses[1]);
        EXPECT_EQ(2, leafIter->mReadPasses[2]);
    }
}


////////////////////////////////////////


TEST_F(TestFile, testHasGrid)
{
    using namespace openvdb;
    using namespace openvdb::io;

    using FloatGrid = openvdb::FloatGrid;
    using IntGrid = openvdb::Int32Grid;
    using FloatTree = FloatGrid::TreeType;
    using IntTree = Int32Grid::TreeType;

    // Create a vdb to write.

    // Create grids
    IntGrid::Ptr grid = createGrid<IntGrid>(/*bg=*/1);
    IntTree& tree = grid->tree();
    grid->setName("density");

    FloatGrid::Ptr grid2 = createGrid<FloatGrid>(/*bg=*/2.0);
    FloatTree& tree2 = grid2->tree();
    grid2->setName("temperature");

    // Create transforms
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid->setTransform(trans);
    grid2->setTransform(trans2);

    // Set some values
    tree.setValue(Coord(0, 0, 0), 5);
    tree.setValue(Coord(100, 0, 0), 6);
    tree2.setValue(Coord(0, 0, 0), 10);
    tree2.setValue(Coord(0, 100, 0), 11);

    MetaMap meta;
    meta.insertMeta("author", StringMetadata("Einstein"));
    meta.insertMeta("year", Int32Metadata(2009));

    GridPtrVec grids;
    grids.push_back(grid);
    grids.push_back(grid2);

    // Register grid and transform.
    GridBase::clearRegistry();
    IntGrid::registerGrid();
    FloatGrid::registerGrid();
    Metadata::clearRegistry();
    StringMetadata::registerType();
    Int32Metadata::registerType();
    // register maps
    math::MapRegistry::clear();
    math::AffineMap::registerMap();
    math::ScaleMap::registerMap();
    math::UniformScaleMap::registerMap();
    math::TranslationMap::registerMap();
    math::ScaleTranslateMap::registerMap();
    math::UniformScaleTranslateMap::registerMap();
    math::NonlinearFrustumMap::registerMap();

    // Write the vdb out to a file.
    io::File vdbfile("something.vdb2");
    vdbfile.write(grids, meta);

    io::File vdbfile2("something.vdb2");

    EXPECT_THROW(vdbfile2.hasGrid("density"), openvdb::IoError);

    vdbfile2.open();

    EXPECT_TRUE(vdbfile2.hasGrid("density"));
    EXPECT_TRUE(vdbfile2.hasGrid("temperature"));
    EXPECT_TRUE(!vdbfile2.hasGrid("Temperature"));
    EXPECT_TRUE(!vdbfile2.hasGrid("densitY"));

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    vdbfile2.close();

    remove("something.vdb2");
}


TEST_F(TestFile, testNameIterator)
{
    using namespace openvdb;
    using namespace openvdb::io;

    using FloatGrid = openvdb::FloatGrid;
    using FloatTree = FloatGrid::TreeType;
    using IntTree = Int32Grid::TreeType;

    // Create trees.
    IntTree::Ptr itree(new IntTree(1));
    itree->setValue(Coord(0, 0, 0), 5);
    itree->setValue(Coord(100, 0, 0), 6);
    FloatTree::Ptr ftree(new FloatTree(2.0));
    ftree->setValue(Coord(0, 0, 0), 10.0);
    ftree->setValue(Coord(0, 100, 0), 11.0);

    // Create grids.
    GridPtrVec grids;
    GridBase::Ptr grid = createGrid(itree);
    grid->setName("density");
    grids.push_back(grid);

    grid = createGrid(ftree);
    grid->setName("temperature");
    grids.push_back(grid);

    // Create two unnamed grids.
    grids.push_back(createGrid(ftree));
    grids.push_back(createGrid(ftree));

    // Create two grids with the same name.
    grid = createGrid(ftree);
    grid->setName("level_set");
    grids.push_back(grid);
    grid = createGrid(ftree);
    grid->setName("level_set");
    grids.push_back(grid);

    // Register types.
    openvdb::initialize();

    const char* filename = "testNameIterator.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);

    // Write the grids out to a file.
    {
        io::File vdbfile(filename);
        vdbfile.write(grids);
    }

    io::File vdbfile(filename);

    // Verify that name iteration fails if the file is not open.
    EXPECT_THROW(vdbfile.beginName(), openvdb::IoError);

    vdbfile.open();

    // Names should appear in lexicographic order.
    Name names[6] = { "[0]", "[1]", "density", "level_set[0]", "level_set[1]", "temperature" };
    int count = 0;
    for (io::File::NameIterator iter = vdbfile.beginName(); iter != vdbfile.endName(); ++iter) {
        EXPECT_EQ(names[count], *iter);
        EXPECT_EQ(names[count], iter.gridName());
        ++count;
        grid = vdbfile.readGrid(*iter);
        EXPECT_TRUE(grid);
    }
    EXPECT_EQ(6, count);

    vdbfile.close();
}


TEST_F(TestFile, testReadOldFileFormat)
{
    /// @todo Save some old-format (prior to OPENVDB_FILE_VERSION) .vdb2 files
    /// to /work/rd/fx_tools/vdb_unittest/TestFile::testReadOldFileFormat/
    /// Verify that the files can still be read correctly.
}


TEST_F(TestFile, testCompression)
{
    using namespace openvdb;
    using namespace openvdb::io;

    using IntGrid = openvdb::Int32Grid;

    // Register types.
    openvdb::initialize();

    // Create reference grids.
    IntGrid::Ptr intGrid = IntGrid::create(/*background=*/0);
    intGrid->fill(CoordBBox(Coord(0), Coord(49)), /*value=*/999, /*active=*/true);
    intGrid->fill(CoordBBox(Coord(6), Coord(43)), /*value=*/0, /*active=*/false);
    intGrid->fill(CoordBBox(Coord(21), Coord(22)), /*value=*/1, /*active=*/false);
    intGrid->fill(CoordBBox(Coord(23), Coord(24)), /*value=*/2, /*active=*/false);
    EXPECT_EQ(8, int(IntGrid::TreeType::LeafNodeType::DIM));

    FloatGrid::Ptr lsGrid = createLevelSet<FloatGrid>();
    unittest_util::makeSphere(/*dim=*/Coord(100), /*ctr=*/Vec3f(50, 50, 50), /*r=*/20.0,
        *lsGrid, unittest_util::SPHERE_SPARSE_NARROW_BAND);
    EXPECT_EQ(int(GRID_LEVEL_SET), int(lsGrid->getGridClass()));

    FloatGrid::Ptr fogGrid = lsGrid->deepCopy();
    tools::sdfToFogVolume(*fogGrid);
    EXPECT_EQ(int(GRID_FOG_VOLUME), int(fogGrid->getGridClass()));


    GridPtrVec grids;
    grids.push_back(intGrid);
    grids.push_back(lsGrid);
    grids.push_back(fogGrid);

    const char* filename = "testCompression.vdb2";
    SharedPtr<const char> scopedFile(filename, ::remove);

    size_t uncompressedSize = 0;
    {
        // Write the grids out to a file with compression disabled.
        io::File vdbfile(filename);
        vdbfile.setCompression(io::COMPRESS_NONE);
        vdbfile.write(grids);
        vdbfile.close();

        // Get the size of the file in bytes.
        struct stat buf;
        buf.st_size = 0;
        EXPECT_EQ(0, ::stat(filename, &buf));
        uncompressedSize = buf.st_size;
    }

    // Write the grids out with various combinations of compression options
    // and verify that they can be read back successfully.
    // See io/Compression.h for the flag values.

#ifdef OPENVDB_USE_BLOSC
    #ifdef OPENVDB_USE_ZLIB
        std::vector<uint32_t> validFlags{0x0,0x1,0x2,0x3,0x4,0x6};
    #else
        std::vector<uint32_t> validFlags{0x0,0x2,0x4,0x6};
    #endif
#else
    #ifdef OPENVDB_USE_ZLIB
        std::vector<uint32_t> validFlags{0x0,0x1,0x2,0x3};
    #else
        std::vector<uint32_t> validFlags{0x0,0x2};
    #endif
#endif
    for (uint32_t flags : validFlags) {

        if (flags != io::COMPRESS_NONE) {
            io::File vdbfile(filename);
            vdbfile.setCompression(flags);
            vdbfile.write(grids);
            vdbfile.close();
        }
        if (flags != io::COMPRESS_NONE) {
            // Verify that the compressed file is significantly smaller than
            // the uncompressed file.
            size_t compressedSize = 0;
            struct stat buf;
            buf.st_size = 0;
            EXPECT_EQ(0, ::stat(filename, &buf));
            compressedSize = buf.st_size;
            EXPECT_TRUE(compressedSize < size_t(0.75 * double(uncompressedSize)));
        }
        {
            // Verify that the grids can be read back successfully.

            io::File vdbfile(filename);
            vdbfile.open();

            GridPtrVecPtr inGrids = vdbfile.getGrids();
            EXPECT_EQ(3, int(inGrids->size()));

            // Verify that the original and input grids are equal.
            {
                const IntGrid::Ptr grid = gridPtrCast<IntGrid>((*inGrids)[0]);
                EXPECT_TRUE(grid.get() != nullptr);
                EXPECT_EQ(int(intGrid->getGridClass()), int(grid->getGridClass()));

                EXPECT_TRUE(grid->tree().hasSameTopology(intGrid->tree()));

                EXPECT_EQ(
                    intGrid->tree().getValue(Coord(0)),
                    grid->tree().getValue(Coord(0)));
                // Verify that leaf nodes with more than two distinct inactive values
                // are handled correctly (FX-7085).
                EXPECT_EQ(
                    intGrid->tree().getValue(Coord(6)),
                    grid->tree().getValue(Coord(6)));
                EXPECT_EQ(
                    intGrid->tree().getValue(Coord(21)),
                    grid->tree().getValue(Coord(21)));
                EXPECT_EQ(
                    intGrid->tree().getValue(Coord(23)),
                    grid->tree().getValue(Coord(23)));

                // Verify that the only active value in this grid is 999.
                const math::MinMax<Int32> extrema = tools::minMax(grid->tree());
                EXPECT_EQ(999, extrema.min());
                EXPECT_EQ(999, extrema.max());
            }
            for (int idx = 1; idx <= 2; ++idx) {
                const FloatGrid::Ptr
                    grid = gridPtrCast<FloatGrid>((*inGrids)[idx]),
                    refGrid = gridPtrCast<FloatGrid>(grids[idx]);
                EXPECT_TRUE(grid.get() != nullptr);
                EXPECT_EQ(int(refGrid->getGridClass()), int(grid->getGridClass()));

                EXPECT_TRUE(grid->tree().hasSameTopology(refGrid->tree()));

                FloatGrid::ConstAccessor refAcc = refGrid->getConstAccessor();
                for (FloatGrid::ValueAllCIter it = grid->cbeginValueAll(); it; ++it) {
                    EXPECT_EQ(refAcc.getValue(it.getCoord()), *it);
                }
            }
        }
    }
}


////////////////////////////////////////


namespace {

using namespace openvdb;

struct TestAsyncHelper
{
    std::set<io::Queue::Id> ids;
    std::map<io::Queue::Id, std::string> filenames;
    size_t refFileSize;
    bool verbose;

    TestAsyncHelper(size_t _refFileSize): refFileSize(_refFileSize), verbose(false) {}

    ~TestAsyncHelper()
    {
        // Remove output files.
        for (std::map<io::Queue::Id, std::string>::iterator it = filenames.begin();
            it != filenames.end(); ++it)
        {
            ::remove(it->second.c_str());
        }
        filenames.clear();
        ids.clear();
    }

    io::Queue::Notifier notifier()
    {
        return std::bind(&TestAsyncHelper::validate, this,
            std::placeholders::_1, std::placeholders::_2);
    }

    void insert(io::Queue::Id id, const std::string& filename)
    {
        ids.insert(id);
        filenames[id] = filename;
        if (verbose) std::cerr << "queued " << filename << " as task " << id << "\n";
    }

    void validate(io::Queue::Id id, io::Queue::Status status)
    {
        if (verbose) {
            std::ostringstream ostr;
            ostr << "task " << id;
            switch (status) {
                case io::Queue::UNKNOWN:   ostr << " is unknown"; break;
                case io::Queue::PENDING:   ostr << " is pending"; break;
                case io::Queue::SUCCEEDED: ostr << " succeeded"; break;
                case io::Queue::FAILED:    ostr << " failed"; break;
            }
            std::cerr << ostr.str() << "\n";
        }

        if (status == io::Queue::SUCCEEDED) {
            // If the task completed successfully, verify that the output file's
            // size matches the reference file's size.
            struct stat buf;
            buf.st_size = 0;
            EXPECT_EQ(0, ::stat(filenames[id].c_str(), &buf));
            EXPECT_EQ(Index64(refFileSize), Index64(buf.st_size));
        }

        if (status == io::Queue::SUCCEEDED || status == io::Queue::FAILED) {
            ids.erase(id);
        }
    }
}; // struct TestAsyncHelper

} // unnamed namespace


TEST_F(TestFile, testAsync)
{
    using namespace openvdb;

    // Register types.
    openvdb::initialize();

    // Create a grid.
    FloatGrid::Ptr lsGrid = createLevelSet<FloatGrid>();
    unittest_util::makeSphere(/*dim=*/Coord(100), /*ctr=*/Vec3f(50, 50, 50), /*r=*/20.0,
        *lsGrid, unittest_util::SPHERE_SPARSE_NARROW_BAND);

    MetaMap fileMetadata;
    fileMetadata.insertMeta("author", StringMetadata("Einstein"));
    fileMetadata.insertMeta("year", Int32Metadata(2013));

    GridPtrVec grids;
    grids.push_back(lsGrid);
    grids.push_back(lsGrid->deepCopy());
    grids.push_back(lsGrid->deepCopy());

    size_t refFileSize = 0;
    {
        // Write a reference file without using asynchronous I/O.
        const char* filename = "testAsyncref.vdb";
        SharedPtr<const char> scopedFile(filename, ::remove);
        io::File f(filename);
        f.write(grids, fileMetadata);

        // Record the size of the reference file.
        struct stat buf;
        buf.st_size = 0;
        EXPECT_EQ(0, ::stat(filename, &buf));
        refFileSize = buf.st_size;
    }

    {
        // Output multiple files using asynchronous I/O.
        // Use polling to get the status of the I/O tasks.

        TestAsyncHelper helper(refFileSize);

        io::Queue queue;
        for (int i = 1; i < 10; ++i) {
            std::ostringstream ostr;
            ostr << "testAsync." << i << ".vdb";
            const std::string filename = ostr.str();
            io::Queue::Id id = queue.write(grids, io::File(filename), fileMetadata);
            helper.insert(id, filename);
        }

        auto start = std::chrono::steady_clock::now();
        while (!helper.ids.empty()) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start);
            if (size_t(duration.count()) > 60)  break; // time out after 1 minute

            // Wait one second for tasks to complete.
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // Poll each task in the pending map.
            std::set<io::Queue::Id> ids = helper.ids; // iterate over a copy
            for (std::set<io::Queue::Id>::iterator it = ids.begin(); it != ids.end(); ++it) {
                const io::Queue::Id id = *it;
                const io::Queue::Status status = queue.status(id);
                helper.validate(id, status);
            }
        }
        EXPECT_TRUE(helper.ids.empty());
        EXPECT_TRUE(queue.empty());
    }
    {
        // Output multiple files using asynchronous I/O.
        // Use notifications to get the status of the I/O tasks.

        TestAsyncHelper helper(refFileSize);

        io::Queue queue(/*capacity=*/2);
        queue.addNotifier(helper.notifier());

        for (int i = 1; i < 10; ++i) {
            std::ostringstream ostr;
            ostr << "testAsync" << i << ".vdb";
            const std::string filename = ostr.str();
            io::Queue::Id id = queue.write(grids, io::File(filename), fileMetadata);
            helper.insert(id, filename);
        }
        while (!queue.empty()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    {
        // Test queue timeout.

        io::Queue queue(/*capacity=*/1);
        queue.setTimeout(0/*sec*/);

        SharedPtr<const char>
            scopedFile1("testAsyncIOa.vdb", ::remove),
            scopedFile2("testAsyncIOb.vdb", ::remove);
        std::ofstream
            file1(scopedFile1.get()),
            file2(scopedFile2.get());

        queue.write(grids, io::Stream(file1));

        // With the queue length restricted to 1 and the timeout to 0 seconds,
        // the next write() call should time out immediately with an exception.
        // (It is possible, though highly unlikely, for the previous task to complete
        // in time for this write() to actually succeed.)
        EXPECT_THROW(queue.write(grids, io::Stream(file2)), openvdb::RuntimeError);

        while (!queue.empty()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}


#ifdef OPENVDB_USE_BLOSC
// This tests for a data corruption bug that existed in versions of Blosc prior to 1.5.0
// (see https://github.com/Blosc/c-blosc/pull/63).
TEST_F(TestFile, testBlosc)
{
    openvdb::initialize();

    const unsigned char rawdata[] = {
        0x93, 0xb0, 0x49, 0xaf, 0x62, 0xad, 0xe3, 0xaa, 0xe4, 0xa5, 0x43, 0x20, 0x24,
        0x29, 0xc9, 0xaf, 0xee, 0xad, 0x0b, 0xac, 0x3d, 0xa8, 0x1f, 0x99, 0x53, 0x27,
        0xb6, 0x2b, 0x16, 0xb0, 0x5f, 0xae, 0x89, 0xac, 0x51, 0xa9, 0xfc, 0xa1, 0xc9,
        0x24, 0x59, 0x2a, 0x2f, 0x2d, 0xb4, 0xae, 0xeb, 0xac, 0x2f, 0xaa, 0xec, 0xa4,
        0x53, 0x21, 0x31, 0x29, 0x8f, 0x2c, 0x8e, 0x2e, 0x31, 0xad, 0xd6, 0xaa, 0x6d,
        0xa6, 0xad, 0x1b, 0x3e, 0x28, 0x0a, 0x2c, 0xfd, 0x2d, 0xf8, 0x2f, 0x45, 0xab,
        0x81, 0xa7, 0x1f, 0x95, 0x02, 0x27, 0x3d, 0x2b, 0x85, 0x2d, 0x75, 0x2f, 0xb6,
        0x30, 0x13, 0xa8, 0xb2, 0x9c, 0xf3, 0x25, 0x9c, 0x2a, 0x28, 0x2d, 0x0b, 0x2f,
        0x7b, 0x30, 0x68, 0x9e, 0x51, 0x25, 0x31, 0x2a, 0xe6, 0x2c, 0xbc, 0x2e, 0x4e,
        0x30, 0x5a, 0xb0, 0xe6, 0xae, 0x0e, 0xad, 0x59, 0xaa, 0x08, 0xa5, 0x89, 0x21,
        0x59, 0x29, 0xb0, 0x2c, 0x57, 0xaf, 0x8c, 0xad, 0x6f, 0xab, 0x65, 0xa7, 0xd3,
        0x12, 0xf5, 0x27, 0xeb, 0x2b, 0xf6, 0x2d, 0xee, 0xad, 0x27, 0xac, 0xab, 0xa8,
        0xb1, 0x9f, 0xa2, 0x25, 0xaa, 0x2a, 0x4a, 0x2d, 0x47, 0x2f, 0x7b, 0xac, 0x6d,
        0xa9, 0x45, 0xa3, 0x73, 0x23, 0x9d, 0x29, 0xb7, 0x2c, 0xa8, 0x2e, 0x51, 0x30,
        0xf7, 0xa9, 0xec, 0xa4, 0x79, 0x20, 0xc5, 0x28, 0x3f, 0x2c, 0x24, 0x2e, 0x09,
        0x30, 0xc8, 0xa5, 0xb1, 0x1c, 0x23, 0x28, 0xc3, 0x2b, 0xba, 0x2d, 0x9c, 0x2f,
        0xc3, 0x30, 0x44, 0x18, 0x6e, 0x27, 0x3d, 0x2b, 0x6b, 0x2d, 0x40, 0x2f, 0x8f,
        0x30, 0x02, 0x27, 0xed, 0x2a, 0x36, 0x2d, 0xfe, 0x2e, 0x68, 0x30, 0x66, 0xae,
        0x9e, 0xac, 0x96, 0xa9, 0x7c, 0xa3, 0xa9, 0x23, 0xc5, 0x29, 0xd8, 0x2c, 0xd7,
        0x2e, 0x0e, 0xad, 0x90, 0xaa, 0xe4, 0xa5, 0xf8, 0x1d, 0x82, 0x28, 0x2b, 0x2c,
        0x1e, 0x2e, 0x0c, 0x30, 0x53, 0xab, 0x9c, 0xa7, 0xd4, 0x96, 0xe7, 0x26, 0x30,
        0x2b, 0x7f, 0x2d, 0x6e, 0x2f, 0xb3, 0x30, 0x74, 0xa8, 0xb1, 0x9f, 0x36, 0x25,
        0x3e, 0x2a, 0xfa, 0x2c, 0xdd, 0x2e, 0x65, 0x30, 0xfc, 0xa1, 0xe0, 0x23, 0x82,
        0x29, 0x8f, 0x2c, 0x66, 0x2e, 0x23, 0x30, 0x2d, 0x22, 0xfb, 0x28, 0x3f, 0x2c,
        0x0a, 0x2e, 0xde, 0x2f, 0xaa, 0x28, 0x0a, 0x2c, 0xc8, 0x2d, 0x8f, 0x2f, 0xb0,
        0x30, 0xde, 0x2b, 0xa0, 0x2d, 0x5a, 0x2f, 0x8f, 0x30, 0x12, 0xac, 0x9d, 0xa8,
        0x0f, 0xa0, 0x51, 0x25, 0x66, 0x2a, 0x1b, 0x2d, 0x0b, 0x2f, 0x82, 0x30, 0x7b,
        0xa9, 0xea, 0xa3, 0x63, 0x22, 0x3f, 0x29, 0x7b, 0x2c, 0x60, 0x2e, 0x26, 0x30,
        0x76, 0xa5, 0xf8, 0x1d, 0x4c, 0x28, 0xeb, 0x2b, 0xce, 0x2d, 0xb0, 0x2f, 0xd3,
        0x12, 0x1d, 0x27, 0x15, 0x2b, 0x57, 0x2d, 0x2c, 0x2f, 0x85, 0x30, 0x0e, 0x26,
        0x74, 0x2a, 0xfa, 0x2c, 0xc3, 0x2e, 0x4a, 0x30, 0x08, 0x2a, 0xb7, 0x2c, 0x74,
        0x2e, 0x1d, 0x30, 0x8f, 0x2c, 0x3f, 0x2e, 0xf8, 0x2f, 0x24, 0x2e, 0xd0, 0x2f,
        0xc3, 0x30, 0xdb, 0xa6, 0xd3, 0x0e, 0x38, 0x27, 0x3d, 0x2b, 0x78, 0x2d, 0x5a,
        0x2f, 0xa3, 0x30, 0x68, 0x9e, 0x51, 0x25, 0x31, 0x2a, 0xe6, 0x2c, 0xbc, 0x2e,
        0x4e, 0x30, 0xa9, 0x23, 0x59, 0x29, 0x6e, 0x2c, 0x38, 0x2e, 0x06, 0x30, 0xb8,
        0x28, 0x10, 0x2c, 0xce, 0x2d, 0x95, 0x2f, 0xb3, 0x30, 0x9b, 0x2b, 0x7f, 0x2d,
        0x39, 0x2f, 0x7f, 0x30, 0x4a, 0x2d, 0xf8, 0x2e, 0x58, 0x30, 0xd0, 0x2e, 0x3d,
        0x30, 0x30, 0x30, 0x53, 0x21, 0xc5, 0x28, 0x24, 0x2c, 0xef, 0x2d, 0xc3, 0x2f,
        0xda, 0x27, 0x58, 0x2b, 0x6b, 0x2d, 0x33, 0x2f, 0x82, 0x30, 0x9c, 0x2a, 0x00,
        0x2d, 0xbc, 0x2e, 0x41, 0x30, 0xb0, 0x2c, 0x60, 0x2e, 0x0c, 0x30, 0x1e, 0x2e,
        0xca, 0x2f, 0xc0, 0x30, 0x95, 0x2f, 0x9f, 0x30, 0x8c, 0x30, 0x23, 0x2a, 0xc4,
        0x2c, 0x81, 0x2e, 0x23, 0x30, 0x5a, 0x2c, 0x0a, 0x2e, 0xc3, 0x2f, 0xc3, 0x30,
        0xad, 0x2d, 0x5a, 0x2f, 0x88, 0x30, 0x0b, 0x2f, 0x5b, 0x30, 0x3a, 0x30, 0x7f,
        0x2d, 0x2c, 0x2f, 0x72, 0x30, 0xc3, 0x2e, 0x37, 0x30, 0x09, 0x30, 0xb6, 0x30
    };

    const char* indata = reinterpret_cast<const char*>(rawdata);
    size_t inbytes = sizeof(rawdata);

    const int
        compbufbytes = int(inbytes + BLOSC_MAX_OVERHEAD),
        decompbufbytes = int(inbytes + BLOSC_MAX_OVERHEAD);

    std::unique_ptr<char[]>
        compresseddata(new char[compbufbytes]),
        outdata(new char[decompbufbytes]);

    for (int compcode = 0; compcode <= BLOSC_ZLIB; ++compcode) {
        char* compname = nullptr;
#if BLOSC_VERSION_MAJOR > 1 || (BLOSC_VERSION_MAJOR == 1 && BLOSC_VERSION_MINOR >= 15)
        if (0 > blosc_compcode_to_compname(compcode, const_cast<const char**>(&compname)))
#else
        if (0 > blosc_compcode_to_compname(compcode, &compname))
#endif
            continue;
        /// @todo This changes the compressor setting globally.
        if (blosc_set_compressor(compname) < 0) continue;

        for (int typesize = 1; typesize <= 4; ++typesize) {

            // Compress the data.
            ::memset(compresseddata.get(), 0, compbufbytes);
            int compressedbytes = blosc_compress(
                /*clevel=*/9,
                /*doshuffle=*/true,
                typesize,
                /*srcsize=*/inbytes,
                /*src=*/indata,
                /*dest=*/compresseddata.get(),
                /*destsize=*/compbufbytes);

            EXPECT_TRUE(compressedbytes > 0);

            // Decompress the data.
            ::memset(outdata.get(), 0, decompbufbytes);
            int outbytes = blosc_decompress(
                compresseddata.get(), outdata.get(), decompbufbytes);

            EXPECT_TRUE(outbytes > 0);
            EXPECT_EQ(int(inbytes), outbytes);

            // Compare original and decompressed data.
            int diff = 0;
            for (size_t i = 0; i < inbytes; ++i) {
                if (outdata[i] != indata[i]) ++diff;
            }
            if (diff > 0) {
                if (diff != 0) {
                    FAIL() << "Your version of the Blosc library is most likely"
                    " out of date; please install the latest version.  "
                    "(Earlier versions have a bug that can cause data corruption.)";
                }
                return;
            }
        }
    }
}
#endif


void
TestFile::testDelayedLoadMetadata()
{
    openvdb::initialize();

    using namespace openvdb;

    io::File file("something.vdb2");

    // Create a level set grid.
    auto lsGrid = createLevelSet<FloatGrid>();
    lsGrid->setName("sphere");
    unittest_util::makeSphere(/*dim=*/Coord(100), /*ctr=*/Vec3f(50, 50, 50), /*r=*/20.0,
        *lsGrid, unittest_util::SPHERE_SPARSE_NARROW_BAND);

    // Write the VDB to a string stream.
    std::ostringstream ostr(std::ios_base::binary);

    // Create the grid descriptor out of this grid.
    io::GridDescriptor gd(Name("sphere"), lsGrid->type());

    // Write out the grid.
    file.writeGrid(gd, lsGrid, ostr, /*seekable=*/true);

    // Duplicate VDB string stream.
    std::ostringstream ostr2(std::ios_base::binary);

    { // Read back in, clip and write out again to verify metadata is rebuilt.
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        io::setVersion(istr, file.libraryVersion(), file.fileVersion());

        io::GridDescriptor gd2;
        GridBase::Ptr grid = gd2.read(istr);
        gd2.seekToGrid(istr);

        const BBoxd clipBbox(Vec3d(-10.0,-10.0,-10.0), Vec3d(10.0,10.0,10.0));
        io::Archive::readGrid(grid, gd2, istr, clipBbox);

        // Verify clipping is working as expected.
        EXPECT_TRUE(grid->baseTreePtr()->leafCount() < lsGrid->tree().leafCount());

        file.writeGrid(gd, grid, ostr2, /*seekable=*/true);
    }

    // Since the input is only a fragment of a VDB file (in particular,
    // it doesn't have a header), set the file format version number explicitly.
    // On read, the delayed load metadata for OpenVDB library versions less than 6.1
    // should be removed to ensure correctness as it possible for the metadata to
    // have been treated as unknown and blindly copied over when read and re-written
    // using this library version resulting in out-of-sync metadata.

    // By default, DelayedLoadMetadata is dropped from the grid during read so
    // as not to be exposed to the user.

    { // read using current library version
        std::istringstream istr(ostr2.str(), std::ios_base::binary);
        io::setVersion(istr, file.libraryVersion(), file.fileVersion());

        io::GridDescriptor gd2;
        GridBase::Ptr grid = gd2.read(istr);
        gd2.seekToGrid(istr);
        io::Archive::readGrid(grid, gd2, istr);

        EXPECT_TRUE(!((*grid)[GridBase::META_FILE_DELAYED_LOAD]));
    }

    // To test the version mechanism, a stream metadata object is created with
    // a non-zero test value and set on the input stream. This disables the
    // behaviour where the DelayedLoadMetadata is dropped from the grid.

    io::StreamMetadata::Ptr streamMetadata(new io::StreamMetadata);
    streamMetadata->__setTest(uint32_t(1));

    { // read using current library version
        std::istringstream istr(ostr2.str(), std::ios_base::binary);
        io::setVersion(istr, file.libraryVersion(), file.fileVersion());
        io::setStreamMetadataPtr(istr, streamMetadata, /*transfer=*/false);

        io::GridDescriptor gd2;
        GridBase::Ptr grid = gd2.read(istr);
        gd2.seekToGrid(istr);
        io::Archive::readGrid(grid, gd2, istr);

        EXPECT_TRUE(((*grid)[GridBase::META_FILE_DELAYED_LOAD]));
    }

    { // read using library version of 5.0
        std::istringstream istr(ostr2.str(), std::ios_base::binary);
        io::setVersion(istr, VersionId(5,0), file.fileVersion());
        io::setStreamMetadataPtr(istr, streamMetadata, /*transfer=*/false);

        io::GridDescriptor gd2;
        GridBase::Ptr grid = gd2.read(istr);
        gd2.seekToGrid(istr);
        io::Archive::readGrid(grid, gd2, istr);

        EXPECT_TRUE(!((*grid)[GridBase::META_FILE_DELAYED_LOAD]));
    }

    { // read using library version of 4.9
        std::istringstream istr(ostr2.str(), std::ios_base::binary);
        io::setVersion(istr, VersionId(4,9), file.fileVersion());
        io::setStreamMetadataPtr(istr, streamMetadata, /*transfer=*/false);

        io::GridDescriptor gd2;
        GridBase::Ptr grid = gd2.read(istr);
        gd2.seekToGrid(istr);
        io::Archive::readGrid(grid, gd2, istr);

        EXPECT_TRUE(!((*grid)[GridBase::META_FILE_DELAYED_LOAD]));
    }

    { // read using library version of 6.1
        std::istringstream istr(ostr2.str(), std::ios_base::binary);
        io::setVersion(istr, VersionId(6,1), file.fileVersion());
        io::setStreamMetadataPtr(istr, streamMetadata, /*transfer=*/false);

        io::GridDescriptor gd2;
        GridBase::Ptr grid = gd2.read(istr);
        gd2.seekToGrid(istr);
        io::Archive::readGrid(grid, gd2, istr);

        EXPECT_TRUE(!((*grid)[GridBase::META_FILE_DELAYED_LOAD]));
    }

    { // read using library version of 6.2
        std::istringstream istr(ostr2.str(), std::ios_base::binary);
        io::setVersion(istr, VersionId(6,2), file.fileVersion());
        io::setStreamMetadataPtr(istr, streamMetadata, /*transfer=*/false);

        io::GridDescriptor gd2;
        GridBase::Ptr grid = gd2.read(istr);
        gd2.seekToGrid(istr);
        io::Archive::readGrid(grid, gd2, istr);

        EXPECT_TRUE(((*grid)[GridBase::META_FILE_DELAYED_LOAD]));
    }

    remove("something.vdb2");
}
TEST_F(TestFile, testDelayedLoadMetadata) { testDelayedLoadMetadata(); }

