///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <boost/uuid/uuid_generators.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/io/File.h>
#include <openvdb/io/Stream.h>
#include <openvdb/Metadata.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetUtil.h> // for tools::sdfToFogVolume()
#include <openvdb/version.h>
#include <openvdb/openvdb.h>
#include "util.h" // for unittest_util::makeSphere()
#include <sys/types.h> // for stat()
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif


class TestFile: public CppUnit::TestCase
{
public:
    virtual void setUp() {}
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestFile);
    CPPUNIT_TEST(testHeader);
    CPPUNIT_TEST(testWriteGrid);
    CPPUNIT_TEST(testWriteMultipleGrids);
    CPPUNIT_TEST(testWriteFloatAsHalf);
    CPPUNIT_TEST(testWriteInstancedGrids);
    CPPUNIT_TEST(testReadGridDescriptors);
    CPPUNIT_TEST(testGridNaming);
    CPPUNIT_TEST(testEmptyFile);
    CPPUNIT_TEST(testEmptyGridIO);
    CPPUNIT_TEST(testOpen);
    CPPUNIT_TEST(testNonVdbOpen);
    CPPUNIT_TEST(testGetMetadata);
    CPPUNIT_TEST(testReadAll);
    CPPUNIT_TEST(testWriteOpenFile);
    CPPUNIT_TEST(testReadGridMetadata);
    CPPUNIT_TEST(testReadGridPartial);
    CPPUNIT_TEST(testReadGrid);
    CPPUNIT_TEST(testMultipleBufferIO);
    CPPUNIT_TEST(testHasGrid);
    CPPUNIT_TEST(testNameIterator);
    CPPUNIT_TEST(testReadOldFileFormat);
    CPPUNIT_TEST(testCompression);
    CPPUNIT_TEST_SUITE_END();

    void testHeader();
    void testWriteGrid();
    void testWriteMultipleGrids();
    void testWriteFloatAsHalf();
    void testWriteInstancedGrids();
    void testReadGridDescriptors();
    void testGridNaming();
    void testEmptyFile();
    void testEmptyGridIO();
    void testOpen();
    void testNonVdbOpen();
    void testGetMetadata();
    void testReadAll();
    void testWriteOpenFile();
    void testReadGridMetadata();
    void testReadGridPartial();
    void testReadGrid();
    void testMultipleBufferIO();
    void testHasGrid();
    void testNameIterator();
    void testReadOldFileFormat();
    void testCompression();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFile);


////////////////////////////////////////


void
TestFile::testHeader()
{
    using namespace openvdb::io;

    File file("something.vdb2");

    std::ostringstream ostr(std::ios_base::binary);

    file.writeHeader(ostr, /*seekable=*/true);

    std::string uuid_str=file.getUniqueTag();

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    bool unique=true;
    CPPUNIT_ASSERT_NO_THROW(unique=file.readHeader(istr));

    CPPUNIT_ASSERT(!unique);//reading same file again

    CPPUNIT_ASSERT_EQUAL(openvdb::OPENVDB_FILE_VERSION, file.fileVersion());
    CPPUNIT_ASSERT_EQUAL(openvdb::OPENVDB_LIBRARY_MAJOR_VERSION, file.libraryVersion().first);
    CPPUNIT_ASSERT_EQUAL(openvdb::OPENVDB_LIBRARY_MINOR_VERSION, file.libraryVersion().second);
    CPPUNIT_ASSERT_EQUAL(uuid_str, file.getUniqueTag());

    //std::cerr << "\nuuid=" << uuid_str << std::endl;

    CPPUNIT_ASSERT(file.isIdentical(uuid_str));

    remove("something.vdb2");
}


void
TestFile::testWriteGrid()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef Int32Tree TreeType;
    typedef Grid<TreeType> GridType;

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
    CPPUNIT_ASSERT(stringMetadata);
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

    CPPUNIT_ASSERT(gd.getGridPos() != 0);
    CPPUNIT_ASSERT(gd.getBlockPos() != 0);
    CPPUNIT_ASSERT(gd.getEndPos() != 0);

    // Read in the grid descriptor.
    GridDescriptor gd2;
    std::istringstream istr(ostr.str(), std::ios_base::binary);

    // Since the input is only a fragment of a VDB file (in particular,
    // it doesn't have a header), set the file format version number explicitly.
    io::setCurrentVersion(istr);

    GridBase::Ptr gd2_grid;
    CPPUNIT_ASSERT_THROW(gd2.read(istr), openvdb::LookupError);

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
    CPPUNIT_ASSERT_NO_THROW(gd2_grid = gd2.read(istr));

    CPPUNIT_ASSERT_EQUAL(gd.gridName(), gd2.gridName());
    CPPUNIT_ASSERT_EQUAL(GridType::gridType(), gd2_grid->type());
    CPPUNIT_ASSERT_EQUAL(gd.getGridPos(), gd2.getGridPos());
    CPPUNIT_ASSERT_EQUAL(gd.getBlockPos(), gd2.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd.getEndPos(), gd2.getEndPos());

    // Position the stream to beginning of the grid storage and read the grid.
    gd2.seekToGrid(istr);
    Archive::readGridCompression(istr);
    gd2_grid->readMeta(istr);
    gd2_grid->readTransform(istr);
    gd2_grid->readTopology(istr);

    // Ensure that we have the same metadata.
    CPPUNIT_ASSERT_EQUAL(grid->metaCount(), gd2_grid->metaCount());
    CPPUNIT_ASSERT((*gd2_grid)["meta0"]);
    CPPUNIT_ASSERT((*gd2_grid)["meta1"]);
    CPPUNIT_ASSERT_EQUAL(meta0Val, gd2_grid->metaValue<std::string>("meta0"));
    CPPUNIT_ASSERT_EQUAL(meta1Val, gd2_grid->metaValue<std::string>("meta1"));

    // Ensure that we have the same topology and transform.
    CPPUNIT_ASSERT_EQUAL(
        grid->baseTree().leafCount(), gd2_grid->baseTree().leafCount());
    CPPUNIT_ASSERT_EQUAL(
        grid->baseTree().nonLeafCount(), gd2_grid->baseTree().nonLeafCount());
    CPPUNIT_ASSERT_EQUAL(
        grid->baseTree().treeDepth(), gd2_grid->baseTree().treeDepth());

    //CPPUNIT_ASSERT_EQUAL(0.1, gd2_grid->getTransform()->getVoxelSizeX());
    //CPPUNIT_ASSERT_EQUAL(0.1, gd2_grid->getTransform()->getVoxelSizeY());
    //CPPUNIT_ASSERT_EQUAL(0.1, gd2_grid->getTransform()->getVoxelSizeZ());

    // Read in the data blocks.
    gd2.seekToBlocks(istr);
    gd2_grid->readBuffers(istr);
    TreeType::Ptr tree2 = boost::dynamic_pointer_cast<TreeType>(gd2_grid->baseTreePtr());
    CPPUNIT_ASSERT(tree2.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(10, tree2->getValue(Coord(10, 1, 2)));
    CPPUNIT_ASSERT_EQUAL(5, tree2->getValue(Coord(0, 0, 0)));

    CPPUNIT_ASSERT_EQUAL(1, tree2->getValue(Coord(1000, 1000, 16000)));
    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    remove("something.vdb2");
}


void
TestFile::testWriteMultipleGrids()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef Int32Tree TreeType;
    typedef Grid<TreeType> GridType;

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

    CPPUNIT_ASSERT(gd.getGridPos() != 0);
    CPPUNIT_ASSERT(gd.getBlockPos() != 0);
    CPPUNIT_ASSERT(gd.getEndPos() != 0);

    CPPUNIT_ASSERT(gd2.getGridPos() != 0);
    CPPUNIT_ASSERT(gd2.getBlockPos() != 0);
    CPPUNIT_ASSERT(gd2.getEndPos() != 0);

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
    CPPUNIT_ASSERT_NO_THROW(gd_in_grid = gd_in.read(istr));

    // Ensure read in the right values.
    CPPUNIT_ASSERT_EQUAL(gd.gridName(), gd_in.gridName());
    CPPUNIT_ASSERT_EQUAL(GridType::gridType(), gd_in_grid->type());
    CPPUNIT_ASSERT_EQUAL(gd.getGridPos(), gd_in.getGridPos());
    CPPUNIT_ASSERT_EQUAL(gd.getBlockPos(), gd_in.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd.getEndPos(), gd_in.getEndPos());

    // Position the stream to beginning of the grid storage and read the grid.
    gd_in.seekToGrid(istr);
    Archive::readGridCompression(istr);
    gd_in_grid->readMeta(istr);
    gd_in_grid->readTransform(istr);
    gd_in_grid->readTopology(istr);

    // Ensure that we have the same topology and transform.
    CPPUNIT_ASSERT_EQUAL(
        grid->baseTree().leafCount(), gd_in_grid->baseTree().leafCount());
    CPPUNIT_ASSERT_EQUAL(
        grid->baseTree().nonLeafCount(), gd_in_grid->baseTree().nonLeafCount());
    CPPUNIT_ASSERT_EQUAL(
        grid->baseTree().treeDepth(), gd_in_grid->baseTree().treeDepth());

    // CPPUNIT_ASSERT_EQUAL(0.1, gd_in_grid->getTransform()->getVoxelSizeX());
    // CPPUNIT_ASSERT_EQUAL(0.1, gd_in_grid->getTransform()->getVoxelSizeY());
    // CPPUNIT_ASSERT_EQUAL(0.1, gd_in_grid->getTransform()->getVoxelSizeZ());

    // Read in the data blocks.
    gd_in.seekToBlocks(istr);
    gd_in_grid->readBuffers(istr);
    TreeType::Ptr grid_in = boost::dynamic_pointer_cast<TreeType>(gd_in_grid->baseTreePtr());
    CPPUNIT_ASSERT(grid_in.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(10, grid_in->getValue(Coord(10, 1, 2)));
    CPPUNIT_ASSERT_EQUAL(5, grid_in->getValue(Coord(0, 0, 0)));
    CPPUNIT_ASSERT_EQUAL(1, grid_in->getValue(Coord(1000, 1000, 16000)));

    /////////////////////////////////////////////////////////////////
    // Now read in the second grid descriptor. Make use of hte end offset.
    ///////////////////////////////////////////////////////////////

    gd_in.seekToEnd(istr);

    GridDescriptor gd2_in;
    GridBase::Ptr gd2_in_grid;
    CPPUNIT_ASSERT_NO_THROW(gd2_in_grid = gd2_in.read(istr));

    // Ensure that we read in the right values.
    CPPUNIT_ASSERT_EQUAL(gd2.gridName(), gd2_in.gridName());
    CPPUNIT_ASSERT_EQUAL(TreeType::treeType(), gd2_in_grid->type());
    CPPUNIT_ASSERT_EQUAL(gd2.getGridPos(), gd2_in.getGridPos());
    CPPUNIT_ASSERT_EQUAL(gd2.getBlockPos(), gd2_in.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd2.getEndPos(), gd2_in.getEndPos());

    // Position the stream to beginning of the grid storage and read the grid.
    gd2_in.seekToGrid(istr);
    Archive::readGridCompression(istr);
    gd2_in_grid->readMeta(istr);
    gd2_in_grid->readTransform(istr);
    gd2_in_grid->readTopology(istr);

    // Ensure that we have the same topology and transform.
    CPPUNIT_ASSERT_EQUAL(
        grid2->baseTree().leafCount(), gd2_in_grid->baseTree().leafCount());
    CPPUNIT_ASSERT_EQUAL(
        grid2->baseTree().nonLeafCount(), gd2_in_grid->baseTree().nonLeafCount());
    CPPUNIT_ASSERT_EQUAL(
        grid2->baseTree().treeDepth(), gd2_in_grid->baseTree().treeDepth());
    // CPPUNIT_ASSERT_EQUAL(0.2, gd2_in_grid->getTransform()->getVoxelSizeX());
    // CPPUNIT_ASSERT_EQUAL(0.2, gd2_in_grid->getTransform()->getVoxelSizeY());
    // CPPUNIT_ASSERT_EQUAL(0.2, gd2_in_grid->getTransform()->getVoxelSizeZ());

    // Read in the data blocks.
    gd2_in.seekToBlocks(istr);
    gd2_in_grid->readBuffers(istr);
    TreeType::Ptr grid2_in =
        boost::dynamic_pointer_cast<TreeType>(gd2_in_grid->baseTreePtr());
    CPPUNIT_ASSERT(grid2_in.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(50, grid2_in->getValue(Coord(1000, 1000, 1000)));
    CPPUNIT_ASSERT_EQUAL(10, grid2_in->getValue(Coord(0, 0, 0)));
    CPPUNIT_ASSERT_EQUAL(2, grid2_in->getValue(Coord(100000, 100000, 16000)));

    // Clear registries.
    GridBase::clearRegistry();

    math::MapRegistry::clear();
    remove("something.vdb2");
}


void
TestFile::testWriteFloatAsHalf()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef Vec3STree TreeType;
    typedef Grid<TreeType> GridType;

    // Register all grid types.
    initialize();
    // Ensure that the registry is cleared on exit.
    struct Local { static void uninitialize(char*) { openvdb::uninitialize(); } };
    boost::shared_ptr<char> onExit((char*)(0), Local::uninitialize);

    // Create two test grids.
    GridType::Ptr grid1 = createGrid<GridType>(/*bg=*/Vec3s(1, 1, 1));
    TreeType& tree1 = grid1->tree();
    CPPUNIT_ASSERT(grid1.get() != NULL);
    grid1->setTransform(math::Transform::createLinearTransform(0.1));
    grid1->setName("grid1");

    GridType::Ptr grid2 = createGrid<GridType>(/*bg=*/Vec3s(2, 2, 2));
    CPPUNIT_ASSERT(grid2.get() != NULL);
    TreeType& tree2 = grid2->tree();
    grid2->setTransform(math::Transform::createLinearTransform(0.2));
    // Flag this grid for 16-bit float output.
    grid2->setSaveFloatAsHalf(true);
    grid2->setName("grid2");

    for (int x = 0; x < 40; ++x) {
        for (int y = 0; y < 40; ++y) {
            for (int z = 0; z < 40; ++z) {
                tree1.setValue(Coord(x, y, z), Vec3s(x, y, z));
                tree2.setValue(Coord(x, y, z), Vec3s(x, y, z));
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

        CPPUNIT_ASSERT(bgrid1.get() != NULL);
        CPPUNIT_ASSERT(bgrid1->isType<GridType>());
        CPPUNIT_ASSERT(bgrid2.get() != NULL);
        CPPUNIT_ASSERT(bgrid2->isType<GridType>());

        const TreeType& btree1 = boost::static_pointer_cast<GridType>(bgrid1)->tree();
        CPPUNIT_ASSERT_EQUAL(Vec3s(10, 10, 10), btree1.getValue(Coord(10, 10, 10)));
        const TreeType& btree2 = boost::static_pointer_cast<GridType>(bgrid2)->tree();
        CPPUNIT_ASSERT_EQUAL(Vec3s(10, 10, 10), btree2.getValue(Coord(10, 10, 10)));
    }
}


void
TestFile::testWriteInstancedGrids()
{
    using namespace openvdb;

    // Register data types.
    openvdb::initialize();

    // Create grids.
    Int32Tree::Ptr tree1(new Int32Tree(1));
    FloatTree::Ptr tree2(new FloatTree(2.0));
    GridBase::Ptr
        grid1 = createGrid(tree1),
        grid2 = createGrid(tree1), // instance of grid1
        grid3 = createGrid(tree2);
    grid1->setName("density");
    grid2->setName("density_copy");
    grid3->setName("temperature");

    // Create transforms.
    math::Transform::Ptr trans1 = math::Transform::createLinearTransform(0.1);
    math::Transform::Ptr trans2 = math::Transform::createLinearTransform(0.1);
    grid1->setTransform(trans1);
    grid2->setTransform(trans2);
    grid3->setTransform(trans2);

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

    // Write the grids to a file and then close the file.
    const char* filename = "something.vdb2";
    boost::shared_ptr<const char> scopedFile(filename, ::remove);
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
    CPPUNIT_ASSERT(meta.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(2, int(meta->metaCount()));
    CPPUNIT_ASSERT_EQUAL(std::string("Einstein"), meta->metaValue<std::string>("author"));
    CPPUNIT_ASSERT_EQUAL(2009, meta->metaValue<int32_t>("year"));

    // Verify the grids.
    CPPUNIT_ASSERT(grids.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(3, int(grids->size()));

    GridBase::Ptr grid = findGridByName(*grids, "density");
    CPPUNIT_ASSERT(grid.get() != NULL);
    Int32Tree::Ptr density = gridPtrCast<Int32Grid>(grid)->treePtr();
    CPPUNIT_ASSERT(density.get() != NULL);

    grid.reset();
    grid = findGridByName(*grids, "density_copy");
    CPPUNIT_ASSERT(grid.get() != NULL);
    CPPUNIT_ASSERT(gridPtrCast<Int32Grid>(grid)->treePtr().get() != NULL);
    // Verify that "density_copy" is an instance of (i.e., shares a tree with) "density".
    CPPUNIT_ASSERT_EQUAL(density, gridPtrCast<Int32Grid>(grid)->treePtr());

    grid.reset();
    grid = findGridByName(*grids, "temperature");
    CPPUNIT_ASSERT(grid.get() != NULL);
    FloatTree::Ptr temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    CPPUNIT_ASSERT(temperature.get() != NULL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, density->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6, density->getValue(Coord(100, 0, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10, temperature->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11, temperature->getValue(Coord(0, 100, 0)), /*tolerance=*/0);

    // Reread with instancing disabled.
    file.close();
    file.setInstancingEnabled(false);
    file.open();
    grids = file.getGrids();
    CPPUNIT_ASSERT_EQUAL(3, int(grids->size()));
    grid = findGridByName(*grids, "density_copy");
    CPPUNIT_ASSERT(grid.get() != NULL);
    CPPUNIT_ASSERT(gridPtrCast<Int32Grid>(grid)->treePtr().get() != NULL);
    // Verify that "density_copy" is *not* an instance of "density".
    CPPUNIT_ASSERT(gridPtrCast<Int32Grid>(grid)->treePtr() != density);

    // Rewrite with instancing disabled, then reread with instancing enabled.
    file.close();
    {
        io::File vdbFile(filename);
        vdbFile.setInstancingEnabled(false);
        vdbFile.write(*grids, *meta);
    }
    file.setInstancingEnabled(true);
    file.open();
    grids = file.getGrids();
    CPPUNIT_ASSERT_EQUAL(3, int(grids->size()));
    grid = findGridByName(*grids, "density_copy");
    CPPUNIT_ASSERT(grid.get() != NULL);
    CPPUNIT_ASSERT(gridPtrCast<Int32Grid>(grid)->treePtr().get() != NULL);
    // Verify that "density_copy" is *not* an instance of "density".
    CPPUNIT_ASSERT(gridPtrCast<Int32Grid>(grid)->treePtr() != density);
}


void
TestFile::testReadGridDescriptors()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef Int32Grid GridType;
    typedef GridType::TreeType TreeType;

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
    ostr.write((char*)&gridCount, sizeof(int32_t));
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
    CPPUNIT_ASSERT(it != file2.mGridDescriptors.end());
    GridDescriptor file2gd = it->second;
    CPPUNIT_ASSERT_EQUAL(gd.gridName(), file2gd.gridName());
    CPPUNIT_ASSERT_EQUAL(gd.getGridPos(), file2gd.getGridPos());
    CPPUNIT_ASSERT_EQUAL(gd.getBlockPos(), file2gd.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd.getEndPos(), file2gd.getEndPos());

    it = file2.findDescriptor("density");
    CPPUNIT_ASSERT(it != file2.mGridDescriptors.end());
    file2gd = it->second;
    CPPUNIT_ASSERT_EQUAL(gd2.gridName(), file2gd.gridName());
    CPPUNIT_ASSERT_EQUAL(gd2.getGridPos(), file2gd.getGridPos());
    CPPUNIT_ASSERT_EQUAL(gd2.getBlockPos(), file2gd.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd2.getEndPos(), file2gd.getEndPos());

    // Clear registries.
    GridBase::clearRegistry();
    math::MapRegistry::clear();

    remove("something.vdb2");
}


void
TestFile::testGridNaming()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef Int32Tree TreeType;
    typedef Grid<TreeType> GridType;

    // Register data types.
    openvdb::initialize();

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

    const char* filename = "/tmp/testGridNaming.vdb2";
    boost::shared_ptr<const char> scopedFile(filename, ::remove);

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
            CPPUNIT_ASSERT(file.hasGrid(i.gridName()));
        }
        // Verify that the file contains three grids.
        CPPUNIT_ASSERT_EQUAL(3, n);

        // Read each grid.
        for (n = -1; n <= 2; ++n) {
            openvdb::Name name("grid");

            // On the first iteration, read the grid named "grid", then read "grid[0]"
            // (which is synonymous with "grid"), then "grid[1]", then "grid[2]".
            if (n >= 0) {
                name = GridDescriptor::nameAsString(GridDescriptor::addSuffix(name, n));
            }

            CPPUNIT_ASSERT(file.hasGrid(name));

            // Partially read the current grid.
            GridBase::ConstPtr grid = file.readGridPartial(name);
            CPPUNIT_ASSERT(grid.get() != NULL);

            // Verify that the grid is named "grid".
            CPPUNIT_ASSERT_EQUAL(openvdb::Name("grid"), grid->getName());

            CPPUNIT_ASSERT_EQUAL((n < 0 ? 0 : n), grid->metaValue<openvdb::Int32>("index"));

            // Fully read the current grid.
            grid = file.readGrid(name);
            CPPUNIT_ASSERT(grid.get() != NULL);
            CPPUNIT_ASSERT_EQUAL(openvdb::Name("grid"), grid->getName());
            CPPUNIT_ASSERT_EQUAL((n < 0 ? 0 : n), grid->metaValue<openvdb::Int32>("index"));
        }

        // Read all three grids at once.
        GridPtrVecPtr allGrids = file.getGrids();
        CPPUNIT_ASSERT(allGrids.get() != NULL);
        CPPUNIT_ASSERT_EQUAL(3, int(allGrids->size()));

        GridBase::ConstPtr firstGrid;
        std::vector<int> indices;
        for (GridPtrVecCIter i = allGrids->begin(), e = allGrids->end(); i != e; ++i) {
            GridBase::ConstPtr grid = *i;
            CPPUNIT_ASSERT(grid.get() != NULL);

            indices.push_back(grid->metaValue<openvdb::Int32>("index"));

            // If instancing is enabled, verify that all grids share the same tree.
            if (instancing) {
                if (!firstGrid) firstGrid = grid;
                CPPUNIT_ASSERT_EQUAL(firstGrid->baseTreePtr(), grid->baseTreePtr());
            }
        }
        // Verify that three distinct grids were read,
        // by examining their "index" metadata.
        CPPUNIT_ASSERT_EQUAL(3, int(indices.size()));
        std::sort(indices.begin(), indices.end());
        CPPUNIT_ASSERT_EQUAL(0, indices[0]);
        CPPUNIT_ASSERT_EQUAL(1, indices[1]);
        CPPUNIT_ASSERT_EQUAL(2, indices[2]);
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
        CPPUNIT_ASSERT(grid.get() != NULL);
        CPPUNIT_ASSERT_EQUAL(weirdName, grid->getName());
        CPPUNIT_ASSERT_EQUAL(0, grid->metaValue<openvdb::Int32>("index"));

        // Verify that the other grids can still be read successfully.
        grid = file.readGrid("grid[0]");
        CPPUNIT_ASSERT(grid.get() != NULL);
        CPPUNIT_ASSERT_EQUAL(openvdb::Name("grid"), grid->getName());
        // Because there are now only two grids named "grid", the one with
        // index 1 is now "grid[0]".
        CPPUNIT_ASSERT_EQUAL(1, grid->metaValue<openvdb::Int32>("index"));

        grid = file.readGrid("grid[1]");
        CPPUNIT_ASSERT(grid.get() != NULL);
        CPPUNIT_ASSERT_EQUAL(openvdb::Name("grid"), grid->getName());
        // Because there are now only two grids named "grid", the one with
        // index 2 is now "grid[1]".
        CPPUNIT_ASSERT_EQUAL(2, grid->metaValue<openvdb::Int32>("index"));

        // Verify that there is no longer a third grid named "grid".
        CPPUNIT_ASSERT_THROW(file.readGrid("grid[2]"), openvdb::KeyError);
    }
}


void
TestFile::testEmptyFile()
{
    using namespace openvdb;
    using namespace openvdb::io;

    const char* filename = "/tmp/testEmptyFile.vdb2";
    boost::shared_ptr<const char> scopedFile(filename, ::remove);

    {
        File file(filename);
        file.write(GridPtrVec(), MetaMap());
    }
    File file(filename);
    file.open();

    GridPtrVecPtr grids = file.getGrids();
    MetaMap::Ptr meta = file.getMetadata();

    CPPUNIT_ASSERT(grids.get() != NULL);
    CPPUNIT_ASSERT(grids->empty());

    CPPUNIT_ASSERT(meta.get() != NULL);
    CPPUNIT_ASSERT(meta->empty());
}


void
TestFile::testEmptyGridIO()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef Int32Grid GridType;
    typedef GridType::TreeType TreeType;

    const char* filename = "/tmp/something.vdb2";
    boost::shared_ptr<const char> scopedFile(filename, ::remove);

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
    ostr.write((char*)&gridCount, sizeof(int32_t));
    // Write out the grids.
    file.writeGrid(gd, grid, ostr, /*seekable=*/true);
    file.writeGrid(gd2, grid2, ostr, /*seekable=*/true);

    // Ensure that the block offset and the end offsets are equivalent.
    CPPUNIT_ASSERT_EQUAL(0, int(grid->baseTree().leafCount()));
    CPPUNIT_ASSERT_EQUAL(0, int(grid2->baseTree().leafCount()));
    CPPUNIT_ASSERT_EQUAL(gd.getEndPos(), gd.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd2.getEndPos(), gd2.getBlockPos());

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
    CPPUNIT_ASSERT(it != file2.mGridDescriptors.end());
    GridDescriptor file2gd = it->second;
    file2gd.seekToGrid(istr);
    GridBase::Ptr gd_grid = GridBase::createGrid(file2gd.gridType());
    Archive::readGridCompression(istr);
    gd_grid->readMeta(istr);
    gd_grid->readTransform(istr);
    gd_grid->readTopology(istr);
    CPPUNIT_ASSERT_EQUAL(gd.gridName(), file2gd.gridName());
    CPPUNIT_ASSERT(gd_grid.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(0, int(gd_grid->baseTree().leafCount()));
    //CPPUNIT_ASSERT_EQUAL(8, int(gd_grid->baseTree().nonLeafCount()));
    CPPUNIT_ASSERT_EQUAL(4, int(gd_grid->baseTree().treeDepth()));
    CPPUNIT_ASSERT_EQUAL(gd.getGridPos(), file2gd.getGridPos());
    CPPUNIT_ASSERT_EQUAL(gd.getBlockPos(), file2gd.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd.getEndPos(), file2gd.getEndPos());

    it = file2.findDescriptor("density");
    CPPUNIT_ASSERT(it != file2.mGridDescriptors.end());
    file2gd = it->second;
    file2gd.seekToGrid(istr);
    gd_grid = GridBase::createGrid(file2gd.gridType());
    Archive::readGridCompression(istr);
    gd_grid->readMeta(istr);
    gd_grid->readTransform(istr);
    gd_grid->readTopology(istr);
    CPPUNIT_ASSERT_EQUAL(gd2.gridName(), file2gd.gridName());
    CPPUNIT_ASSERT(gd_grid.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(0, int(gd_grid->baseTree().leafCount()));
    //CPPUNIT_ASSERT_EQUAL(8, int(gd_grid->nonLeafCount()));
    CPPUNIT_ASSERT_EQUAL(4, int(gd_grid->baseTree().treeDepth()));
    CPPUNIT_ASSERT_EQUAL(gd2.getGridPos(), file2gd.getGridPos());
    CPPUNIT_ASSERT_EQUAL(gd2.getBlockPos(), file2gd.getBlockPos());
    CPPUNIT_ASSERT_EQUAL(gd2.getEndPos(), file2gd.getEndPos());

    // Clear registries.
    GridBase::clearRegistry();
    math::MapRegistry::clear();
}


void
TestFile::testOpen()
{
    using namespace openvdb;

    typedef openvdb::FloatGrid FloatGrid;
    typedef openvdb::Int32Grid IntGrid;
    typedef FloatGrid::TreeType FloatTree;
    typedef Int32Grid::TreeType IntTree;

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

    CPPUNIT_ASSERT(findGridByName(grids, "density") == grid);
    CPPUNIT_ASSERT(findGridByName(grids, "temperature") == grid2);
    CPPUNIT_ASSERT(meta.metaValue<std::string>("author") == "Einstein");
    CPPUNIT_ASSERT_EQUAL(2009, meta.metaValue<int32_t>("year"));

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
    CPPUNIT_ASSERT(!vdbfile.open());//opening the same file
    //Can't open same file multiple times without cloasing
    CPPUNIT_ASSERT_THROW(vdbfile.open(), openvdb::IoError);
    vdbfile.close();
    CPPUNIT_ASSERT(!vdbfile.open());//opening the same file

    CPPUNIT_ASSERT(vdbfile.isOpen());
    CPPUNIT_ASSERT_EQUAL(OPENVDB_FILE_VERSION, vdbfile.fileVersion());
    CPPUNIT_ASSERT_EQUAL(OPENVDB_FILE_VERSION, io::getFormatVersion(vdbfile.mInStream));
    CPPUNIT_ASSERT_EQUAL(OPENVDB_LIBRARY_MAJOR_VERSION, vdbfile.libraryVersion().first);
    CPPUNIT_ASSERT_EQUAL(OPENVDB_LIBRARY_MINOR_VERSION, vdbfile.libraryVersion().second);
    CPPUNIT_ASSERT_EQUAL(OPENVDB_LIBRARY_MAJOR_VERSION,
        io::getLibraryVersion(vdbfile.mInStream).first);
    CPPUNIT_ASSERT_EQUAL(OPENVDB_LIBRARY_MINOR_VERSION,
        io::getLibraryVersion(vdbfile.mInStream).second);

    // Ensure that we read in the vdb metadata.
    CPPUNIT_ASSERT(vdbfile.getMetadata());
    CPPUNIT_ASSERT(vdbfile.getMetadata()->metaValue<std::string>("author") == "Einstein");
    CPPUNIT_ASSERT_EQUAL(2009, vdbfile.getMetadata()->metaValue<int32_t>("year"));

    // Ensure we got the grid descriptors.
    CPPUNIT_ASSERT_EQUAL(1, int(vdbfile.mGridDescriptors.count("density")));
    CPPUNIT_ASSERT_EQUAL(1, int(vdbfile.mGridDescriptors.count("temperature")));

    io::File::NameMapCIter it = vdbfile.findDescriptor("density");
    CPPUNIT_ASSERT(it != vdbfile.mGridDescriptors.end());
    io::GridDescriptor gd = it->second;
    CPPUNIT_ASSERT_EQUAL(IntTree::treeType(), gd.gridType());

    it = vdbfile.findDescriptor("temperature");
    CPPUNIT_ASSERT(it != vdbfile.mGridDescriptors.end());
    gd = it->second;
    CPPUNIT_ASSERT_EQUAL(FloatTree::treeType(), gd.gridType());

    // Ensure we throw an error if there is no file.
    io::File vdbfile2("somethingelses.vdb2");
    CPPUNIT_ASSERT_THROW(vdbfile2.open(), openvdb::IoError);
    CPPUNIT_ASSERT(vdbfile2.mInStream.is_open() == false);

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    // Test closing the file.
    vdbfile.close();
    CPPUNIT_ASSERT(vdbfile.isOpen() == false);
    CPPUNIT_ASSERT(vdbfile.mMeta.get() == NULL);
    CPPUNIT_ASSERT_EQUAL(0, int(vdbfile.mGridDescriptors.size()));
    CPPUNIT_ASSERT(vdbfile.mInStream.is_open() == false);

    remove("something.vdb2");
}


void
TestFile::testNonVdbOpen()
{
    std::ofstream file("dummy.vdb2", std::ios_base::binary);

    int64_t something = 1;
    file.write((char*)&something, sizeof(int64_t));

    file.close();

    openvdb::io::File vdbfile("dummy.vdb2");
    CPPUNIT_ASSERT_THROW(vdbfile.open(), openvdb::IoError);
    CPPUNIT_ASSERT(vdbfile.mInStream.is_open() == false);

    remove("dummy.vdb2");
}


void
TestFile::testGetMetadata()
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
    CPPUNIT_ASSERT_THROW(vdbfile.getMetadata(), openvdb::IoError);

    vdbfile.open();

    MetaMap::Ptr meta2 = vdbfile.getMetadata();

    CPPUNIT_ASSERT_EQUAL(2, int(meta2->metaCount()));

    CPPUNIT_ASSERT(meta2->metaValue<std::string>("author") == "Einstein");
    CPPUNIT_ASSERT_EQUAL(2009, meta2->metaValue<int32_t>("year"));

    // Clear registry.
    Metadata::clearRegistry();

    remove("something.vdb2");
}


void
TestFile::testReadAll()
{
    using namespace openvdb;

    typedef openvdb::FloatGrid FloatGrid;
    typedef openvdb::Int32Grid IntGrid;
    typedef FloatGrid::TreeType FloatTree;
    typedef Int32Grid::TreeType IntTree;

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
    CPPUNIT_ASSERT_THROW(vdbfile2.getGrids(), openvdb::IoError);

    vdbfile2.open();
    CPPUNIT_ASSERT(vdbfile2.isOpen());

    GridPtrVecPtr grids2 = vdbfile2.getGrids();
    MetaMap::Ptr meta2 = vdbfile2.getMetadata();

    // Ensure we have the metadata.
    CPPUNIT_ASSERT_EQUAL(2, int(meta2->metaCount()));
    CPPUNIT_ASSERT(meta2->metaValue<std::string>("author") == "Einstein");
    CPPUNIT_ASSERT_EQUAL(2009, meta2->metaValue<int32_t>("year"));

    // Ensure we got the grids.
    CPPUNIT_ASSERT_EQUAL(2, int(grids2->size()));

    GridBase::Ptr grid;
    grid.reset();
    grid = findGridByName(*grids2, "density");
    CPPUNIT_ASSERT(grid.get() != NULL);
    IntTree::Ptr density = gridPtrCast<IntGrid>(grid)->treePtr();
    CPPUNIT_ASSERT(density.get() != NULL);

    grid.reset();
    grid = findGridByName(*grids2, "temperature");
    CPPUNIT_ASSERT(grid.get() != NULL);
    FloatTree::Ptr temperature = gridPtrCast<FloatGrid>(grid)->treePtr();
    CPPUNIT_ASSERT(temperature.get() != NULL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, density->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6, density->getValue(Coord(100, 0, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10, temperature->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11, temperature->getValue(Coord(0, 100, 0)), /*tolerance=*/0);

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    vdbfile2.close();

    remove("something.vdb2");
}


void
TestFile::testWriteOpenFile()
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
    CPPUNIT_ASSERT_THROW(vdbfile2.getGrids(), openvdb::IoError);

    vdbfile2.open();
    CPPUNIT_ASSERT(vdbfile2.isOpen());

    GridPtrVecPtr grids = vdbfile2.getGrids();
    meta = vdbfile2.getMetadata();

    // Ensure we have the metadata.
    CPPUNIT_ASSERT(meta.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(2, int(meta->metaCount()));
    CPPUNIT_ASSERT(meta->metaValue<std::string>("author") == "Einstein");
    CPPUNIT_ASSERT_EQUAL(2009, meta->metaValue<int32_t>("year"));

    // Ensure we got the grids.
    CPPUNIT_ASSERT(grids.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(0, int(grids->size()));

    // Cannot write an open file.
    CPPUNIT_ASSERT_THROW(vdbfile2.write(*grids), openvdb::IoError);

    vdbfile2.close();

    CPPUNIT_ASSERT_NO_THROW(vdbfile2.write(*grids));

    // Clear registries.
    Metadata::clearRegistry();

    remove("something.vdb2");
}


void
TestFile::testReadGridMetadata()
{
    using namespace openvdb;

    openvdb::initialize();

    const char* filename = "/tmp/testReadGridMetadata.vdb2";
    boost::shared_ptr<const char> scopedFile(filename, ::remove);

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
            io::Stream().write(ostrm, srcGrids);
        }

        // Read just the grid-level metadata from the file.
        io::File vdbfile(filename);

        // Verify that reading from an unopened file generates an exception.
        CPPUNIT_ASSERT_THROW(vdbfile.readGridMetadata("igrid"), openvdb::IoError);
        CPPUNIT_ASSERT_THROW(vdbfile.readGridMetadata("noname"), openvdb::IoError);
        CPPUNIT_ASSERT_THROW(vdbfile.readAllGridMetadata(), openvdb::IoError);

        vdbfile.open();

        CPPUNIT_ASSERT(vdbfile.isOpen());

        // Verify that reading a nonexistent grid generates an exception.
        CPPUNIT_ASSERT_THROW(vdbfile.readGridMetadata("noname"), openvdb::KeyError);

        // Read all grids and store them in a list.
        GridPtrVecPtr gridMetadata = vdbfile.readAllGridMetadata();
        CPPUNIT_ASSERT(gridMetadata.get() != NULL);
        CPPUNIT_ASSERT_EQUAL(2, int(gridMetadata->size()));

        // Read individual grids and append them to the list.
        GridBase::Ptr grid = vdbfile.readGridMetadata("igrid");
        CPPUNIT_ASSERT(grid.get() != NULL);
        CPPUNIT_ASSERT_EQUAL(std::string("igrid"), grid->getName());
        gridMetadata->push_back(grid);

        grid = vdbfile.readGridMetadata("fgrid");
        CPPUNIT_ASSERT(grid.get() != NULL);
        CPPUNIT_ASSERT_EQUAL(std::string("fgrid"), grid->getName());
        gridMetadata->push_back(grid);

        // Verify that the grids' metadata and transforms match the original grids'.
        for (size_t i = 0, N = gridMetadata->size(); i < N; ++i) {
            grid = (*gridMetadata)[i];

            CPPUNIT_ASSERT(grid.get() != NULL);
            CPPUNIT_ASSERT(grid->getName() == "igrid" || grid->getName() == "fgrid");
            CPPUNIT_ASSERT(grid->baseTreePtr().get() != NULL);

            // Since we didn't read the grid's topology, the tree should be empty.
            CPPUNIT_ASSERT_EQUAL(0, int(grid->constBaseTreePtr()->leafCount()));
            CPPUNIT_ASSERT_EQUAL(0, int(grid->constBaseTreePtr()->activeVoxelCount()));

            // Retrieve the source grid of the same name.
            GridBase::ConstPtr srcGrid = srcGridMap[grid->getName()];

            // Compare grid types and transforms.
            CPPUNIT_ASSERT_EQUAL(srcGrid->type(), grid->type());
            CPPUNIT_ASSERT_EQUAL(srcGrid->transform(), grid->transform());

            // Compare metadata, ignoring fields that were added when the file was written.
            MetaMap::Ptr
                statsMetadata = grid->getStatsMetadata(),
                otherMetadata = grid->copyMeta(); // shallow copy
            CPPUNIT_ASSERT(!statsMetadata->empty());
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
            }
            CPPUNIT_ASSERT_EQUAL(srcGrid->str(), otherMetadata->str());

            const CoordBBox srcBBox = srcGrid->evalActiveVoxelBoundingBox();
            CPPUNIT_ASSERT_EQUAL(srcBBox.min().asVec3i(), grid->metaValue<Vec3i>("file_bbox_min"));
            CPPUNIT_ASSERT_EQUAL(srcBBox.max().asVec3i(), grid->metaValue<Vec3i>("file_bbox_max"));
            CPPUNIT_ASSERT_EQUAL(srcGrid->activeVoxelCount(),
                Index64(grid->metaValue<Int64>("file_voxel_count")));
            CPPUNIT_ASSERT_EQUAL(srcGrid->memUsage(),
                Index64(grid->metaValue<Int64>("file_mem_bytes")));
        }
    }
}


void
TestFile::testReadGridPartial()
{
    using namespace openvdb;

    typedef openvdb::FloatGrid FloatGrid;
    typedef openvdb::Int32Grid IntGrid;
    typedef FloatGrid::TreeType FloatTree;
    typedef Int32Grid::TreeType IntTree;

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

    CPPUNIT_ASSERT_THROW(vdbfile2.readGridPartial("density"), openvdb::IoError);

    vdbfile2.open();

    CPPUNIT_ASSERT(vdbfile2.isOpen());

    CPPUNIT_ASSERT_THROW(vdbfile2.readGridPartial("noname"), openvdb::KeyError);

    GridBase::ConstPtr density = vdbfile2.readGridPartial("density");

    CPPUNIT_ASSERT(density.get() != NULL);

    IntTree::ConstPtr typedDensity = gridConstPtrCast<IntGrid>(density)->treePtr();

    CPPUNIT_ASSERT(typedDensity.get() != NULL);

    // the following should cause a compiler error.
    // typedDensity->setValue(0, 0, 0, 0);

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    vdbfile2.close();

    remove("something.vdb2");
}


void
TestFile::testReadGrid()
{
    using namespace openvdb;

    typedef openvdb::FloatGrid FloatGrid;
    typedef openvdb::Int32Grid IntGrid;
    typedef FloatGrid::TreeType FloatTree;
    typedef Int32Grid::TreeType IntTree;

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

    CPPUNIT_ASSERT_THROW(vdbfile2.readGridPartial("density"), openvdb::IoError);

    vdbfile2.open();

    CPPUNIT_ASSERT(vdbfile2.isOpen());

    CPPUNIT_ASSERT_THROW(vdbfile2.readGridPartial("noname"), openvdb::KeyError);

    // Get Temperature
    GridBase::Ptr temperature = vdbfile2.readGrid("temperature");

    CPPUNIT_ASSERT(temperature.get() != NULL);

    FloatTree::Ptr typedTemperature = gridPtrCast<FloatGrid>(temperature)->treePtr();

    CPPUNIT_ASSERT(typedTemperature.get() != NULL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(10, typedTemperature->getValue(Coord(0, 0, 0)), 0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(11, typedTemperature->getValue(Coord(0, 100, 0)), 0);

    // Get Density
    GridBase::Ptr density = vdbfile2.readGrid("density");

    CPPUNIT_ASSERT(density.get() != NULL);

    IntTree::Ptr typedDensity = gridPtrCast<IntGrid>(density)->treePtr();

    CPPUNIT_ASSERT(typedDensity.get() != NULL);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5,typedDensity->getValue(Coord(0, 0, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6,typedDensity->getValue(Coord(100, 0, 0)), /*tolerance=*/0);

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    vdbfile2.close();

    remove("something.vdb2");
}


void
TestFile::testMultipleBufferIO()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef Int32Grid GridType;
    typedef GridType::TreeType TreeType;

    File file("something.vdb2");

    std::ostringstream ostr(std::ios_base::binary);

    // Create a grid with transform.
    GridType::Ptr grid = createGrid<GridType>(/*bg=*/1);
    TreeType& tree = grid->tree();
    grid->setName("temperature");
    math::Transform::Ptr trans = math::Transform::createLinearTransform(0.1);
    grid->setTransform(trans);
    tree.setValue(Coord(10, 1, 2), 10);
    tree.setValue(Coord(0, 0, 0), 5);

    GridPtrVec grids;
    grids.push_back(grid);

    // Register grid and transform.
    openvdb::initialize();

    // write the vdb to the file.
    file.write(grids);

    // read into a different grid.
    File file2("something.vdb2");
    file2.open();

    GridBase::Ptr temperature = file2.readGrid("temperature");

    CPPUNIT_ASSERT(temperature.get() != NULL);

    // Clear registries.
    GridBase::clearRegistry();
    math::MapRegistry::clear();

    remove("something.vdb2");
}


void
TestFile::testHasGrid()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef openvdb::FloatGrid FloatGrid;
    typedef openvdb::Int32Grid IntGrid;
    typedef FloatGrid::TreeType FloatTree;
    typedef Int32Grid::TreeType IntTree;

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

    CPPUNIT_ASSERT_THROW(vdbfile2.hasGrid("density"), openvdb::IoError);

    vdbfile2.open();

    CPPUNIT_ASSERT(vdbfile2.hasGrid("density"));
    CPPUNIT_ASSERT(vdbfile2.hasGrid("temperature"));
    CPPUNIT_ASSERT(!vdbfile2.hasGrid("Temperature"));
    CPPUNIT_ASSERT(!vdbfile2.hasGrid("densitY"));

    // Clear registries.
    GridBase::clearRegistry();
    Metadata::clearRegistry();
    math::MapRegistry::clear();

    vdbfile2.close();

    remove("something.vdb2");
}


void
TestFile::testNameIterator()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef openvdb::FloatGrid FloatGrid;
    typedef openvdb::Int32Grid IntGrid;
    typedef FloatGrid::TreeType FloatTree;
    typedef Int32Grid::TreeType IntTree;

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

    const char* filename = "/tmp/testNameIterator.vdb2";
    boost::shared_ptr<const char> scopedFile(filename, ::remove);

    // Write the grids out to a file.
    {
        io::File vdbfile(filename);
        vdbfile.write(grids);
    }

    io::File vdbfile(filename);

    // Verify that name iteration fails if the file is not open.
    CPPUNIT_ASSERT_THROW(vdbfile.beginName(), openvdb::IoError);

    vdbfile.open();

    // Names should appear in alphabetical order.
    Name names[6] = { "", "[1]", "density", "level_set", "level_set[1]", "temperature" };
    int count = 0;
    for (io::File::NameIterator iter = vdbfile.beginName(); iter != vdbfile.endName(); ++iter) {
        CPPUNIT_ASSERT_EQUAL(names[count], *iter);
        CPPUNIT_ASSERT_EQUAL(names[count], iter.gridName());
        ++count;
        grid = vdbfile.readGrid(*iter);
        CPPUNIT_ASSERT(grid);
    }
    CPPUNIT_ASSERT_EQUAL(6, count);

    vdbfile.close();
}


void
TestFile::testReadOldFileFormat()
{
    /// @todo Save some old-format (prior to OPENVDB_FILE_VERSION) .vdb2 files
    /// to /work/rd/fx_tools/vdb_unittest/TestFile::testReadOldFileFormat/
    /// Verify that the files can still be read correctly.
}


void
TestFile::testCompression()
{
    using namespace openvdb;
    using namespace openvdb::io;

    typedef openvdb::Int32Grid IntGrid;

    // Register types.
    openvdb::initialize();

    // Create reference grids.
    IntGrid::Ptr intGrid = IntGrid::create(/*background=*/0);
    intGrid->fill(CoordBBox(Coord(0), Coord(49)), /*value=*/999, /*active=*/true);
    intGrid->fill(CoordBBox(Coord(6), Coord(43)), /*value=*/0, /*active=*/false);

    FloatGrid::Ptr lsGrid = createLevelSet<FloatGrid>();
    unittest_util::makeSphere(/*dim=*/Coord(100), /*ctr=*/Vec3f(50, 50, 50), /*r=*/20.0,
        *lsGrid, unittest_util::SPHERE_SPARSE_NARROW_BAND);
    CPPUNIT_ASSERT_EQUAL(int(GRID_LEVEL_SET), int(lsGrid->getGridClass()));

    FloatGrid::Ptr fogGrid = lsGrid->deepCopy();
    tools::sdfToFogVolume(*fogGrid);
    CPPUNIT_ASSERT_EQUAL(int(GRID_FOG_VOLUME), int(fogGrid->getGridClass()));


    GridPtrVec grids;
    grids.push_back(intGrid);
    grids.push_back(lsGrid);
    grids.push_back(fogGrid);

    const char* filename = "/tmp/testCompression.vdb2";
    boost::shared_ptr<const char> scopedFile(filename, ::remove);

    size_t uncompressedSize = 0;
    {
        // Write the grids out to a file with compression disabled.
        io::File vdbfile(filename);
        vdbfile.setCompressionFlags(io::COMPRESS_NONE);
        vdbfile.write(grids);
        vdbfile.close();

        // Get the size of the file in bytes.
        struct stat buf;
        buf.st_size = 0;
        CPPUNIT_ASSERT_EQUAL(0, ::stat(filename, &buf));
        uncompressedSize = buf.st_size;
    }

    // Write the grids out with various combinations of compression options
    // and verify that they can be read back successfully.
    // Currently, only bits 0 and 1 have meaning as compression flags
    // (see io/Compression.h), so the valid combinations range from 0x0 to 0x3.
    for (uint32_t flags = 0x0; flags <= 0x3; ++flags) {

        if (flags != io::COMPRESS_NONE) {
            io::File vdbfile(filename);
            vdbfile.setCompressionFlags(flags);
            vdbfile.write(grids);
            vdbfile.close();
        }
        if (flags != io::COMPRESS_NONE) {
            // Verify that the compressed file is significantly smaller than
            // the uncompressed file.
            size_t compressedSize = 0;
            struct stat buf;
            buf.st_size = 0;
            CPPUNIT_ASSERT_EQUAL(0, ::stat(filename, &buf));
            compressedSize = buf.st_size;
            CPPUNIT_ASSERT(compressedSize < size_t(0.75 * uncompressedSize));
        }
        {
            // Verify that the grids can be read back successfully.

            io::File vdbfile(filename);
            vdbfile.open();

            GridPtrVecPtr inGrids = vdbfile.getGrids();
            CPPUNIT_ASSERT_EQUAL(3, int(inGrids->size()));

            // Verify that the original and input grids are equal.
            {
                const IntGrid::Ptr grid = gridPtrCast<IntGrid>((*inGrids)[0]);
                CPPUNIT_ASSERT(grid.get() != NULL);
                CPPUNIT_ASSERT_EQUAL(int(intGrid->getGridClass()), int(grid->getGridClass()));

                CPPUNIT_ASSERT(grid->tree().hasSameTopology(intGrid->tree()));

                CPPUNIT_ASSERT_EQUAL(
                    intGrid->tree().getValue(Coord(0)),
                    grid->tree().getValue(Coord(0)));

                // Verify that the only active value in this grid is 999.
                Int32 minVal = -1, maxVal = -1;
                grid->evalMinMax(minVal, maxVal);
                CPPUNIT_ASSERT_EQUAL(999, minVal);
                CPPUNIT_ASSERT_EQUAL(999, maxVal);
            }
            for (int idx = 1; idx <= 2; ++idx) {
                const FloatGrid::Ptr
                    grid = gridPtrCast<FloatGrid>((*inGrids)[idx]),
                    refGrid = gridPtrCast<FloatGrid>(grids[idx]);
                CPPUNIT_ASSERT(grid.get() != NULL);
                CPPUNIT_ASSERT_EQUAL(int(refGrid->getGridClass()), int(grid->getGridClass()));

                CPPUNIT_ASSERT(grid->tree().hasSameTopology(refGrid->tree()));

                FloatGrid::ConstAccessor refAcc = refGrid->getConstAccessor();
                for (FloatGrid::ValueAllCIter it = grid->cbeginValueAll(); it; ++it) {
                    CPPUNIT_ASSERT_EQUAL(refAcc.getValue(it.getCoord()), *it);
                }
            }
        }
    }
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
