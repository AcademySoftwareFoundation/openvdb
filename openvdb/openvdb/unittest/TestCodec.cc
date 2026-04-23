// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/io/Codec.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <openvdb/tools/Clip.h>
#include <gtest/gtest.h>

class TestCodec: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};

struct MockCodec : public openvdb::io::Codec
{
    static std::string name() { return "mock"; }

    openvdb::io::CodecData::Ptr createData() final { return nullptr; }
};

TEST_F(TestCodec, testCodecRegistry)
{
    using namespace openvdb::io;

    // Start clean
    CodecRegistry::clear();

    // Test isRegistered on empty registry
    EXPECT_FALSE(CodecRegistry::isRegistered("mock"));

    // Test registerCodecByName
    EXPECT_NO_THROW(
        CodecRegistry::registerCodecByName("mock", std::make_unique<MockCodec>())
    );

    EXPECT_TRUE(CodecRegistry::isRegistered("mock"));
    EXPECT_FALSE(CodecRegistry::isRegistered("nonexistent"));

    // Test duplicate registration throws KeyError
    EXPECT_THROW(
        CodecRegistry::registerCodecByName("mock", std::make_unique<MockCodec>()),
        openvdb::KeyError
    );

    // Test registerCodec template form also throws on duplicate
    EXPECT_THROW(
        CodecRegistry::registerCodec<MockCodec>(),
        openvdb::KeyError
    );

    // Test get
    EXPECT_NE(CodecRegistry::get("mock"), nullptr);
    EXPECT_EQ(CodecRegistry::get("nonexistent"), nullptr);

    // Test clear
    CodecRegistry::clear();
    EXPECT_FALSE(CodecRegistry::isRegistered("mock"));
    EXPECT_NO_THROW(CodecRegistry::clear());  // Clear on empty registry

    // Test registerCodec template form on fresh registry
    EXPECT_NO_THROW(CodecRegistry::registerCodec<MockCodec>());
    EXPECT_TRUE(CodecRegistry::isRegistered("mock"));

    // Test io::initialize and io::uninitialize
    CodecRegistry::clear();
    EXPECT_FALSE(CodecRegistry::isRegistered(openvdb::BoolGrid::gridType()));

    EXPECT_NO_THROW(internal::initialize());
    EXPECT_TRUE(CodecRegistry::isRegistered(openvdb::BoolGrid::gridType()));

    EXPECT_NO_THROW(internal::uninitialize());
    EXPECT_FALSE(CodecRegistry::isRegistered(openvdb::BoolGrid::gridType()));
}


TEST_F(TestCodec, testReadDiagnostics)
{
    using namespace openvdb;
    using namespace openvdb::io;

    // ReadDiagnostics struct: disabled by default, addWarning is a no-op until enabled
    {
        ReadDiagnostics diags;
        EXPECT_FALSE(diags.enabled());
        diags.addWarning("grid_a", "something went wrong");
        EXPECT_TRUE(diags.diagnostics().empty());

        diags.enable();
        diags.addWarning("grid_a", "something went wrong");
        ASSERT_EQ(diags.diagnostics().size(), size_t(1));
        EXPECT_EQ(diags.diagnostics()[0].severity, DiagnosticSeverity::Warning);

        diags.clear();
        EXPECT_TRUE(diags.diagnostics().empty());
    }

    CodecRegistry::clear();
    openvdb::io::internal::initialize();

    // Archive API and getGrids() with diagnostics

    BoolGrid::Ptr srcGrid = BoolGrid::create(false);
    srcGrid->setName("bool_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), true, true);

    const std::string codecPath = "testReadDiagnostics.vdb";
    {
        io::File f(codecPath);
        f.write(GridPtrVec{srcGrid});
    }

    // Disabled by default; enabling produces no warnings on a clean read
    {
        io::File f(codecPath);
        f.open();
        EXPECT_FALSE(f.readDiagnostics().enabled());
        f.enableReadDiagnostics();
        EXPECT_TRUE(f.readDiagnostics().enabled());
        f.readGrid("bool_grid");
        EXPECT_TRUE(f.readDiagnostics().diagnostics().empty());
        f.close();
    }

    // clearReadDiagnostics() resets entries but keeps diagnostics enabled
    {
        io::File f(codecPath);
        f.open();
        f.enableReadDiagnostics();
        GridPtrVecPtr grids = f.getGrids();
        ASSERT_TRUE(grids && !grids->empty());
        f.clearReadDiagnostics();
        EXPECT_TRUE(f.readDiagnostics().enabled());
        EXPECT_TRUE(f.readDiagnostics().diagnostics().empty());
        f.close();
    }

    std::remove(codecPath.c_str());
}


template <typename GridT>
void testIOImpl(
    const std::string& gridName,
    const typename GridT::ValueType& bgValue,
    const typename GridT::ValueType& fillValue)
{
    using namespace openvdb;
    using namespace openvdb::io;

    typename GridT::Ptr srcGrid = GridT::create(bgValue);
    srcGrid->setName(gridName);
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), fillValue, true);

    std::stringstream ss("test");
    if (CodecRegistry::isRegistered(GridT::gridType())) {
        ss << "_codec";
    } else {
        ss << "_tree";
    }
    ss << "_" << GridT::gridType() << ".vdb";
    const std::string path = ss.str();
    {
        io::File f(path);
        f.write(GridPtrVec{srcGrid});
    }

    typename GridT::Ptr readGrid;
    {
        io::File f(path);
        f.open();
        readGrid = gridPtrCast<GridT>(f.readGrid(gridName));
        f.close();
    }
    ASSERT_TRUE(readGrid);
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(readGrid->tree()));
    {
        auto readAcc = readGrid->getConstAccessor();
        for (typename GridT::ValueOnCIter it = srcGrid->cbeginValueOn(); it; ++it) {
            EXPECT_EQ(*it, readAcc.getValue(it.getCoord()));
        }
    }

    // clip read
    const BBoxd clipBBox(Vec3d(0.0), Vec3d(3.5));
    auto srcClipped = tools::clip(*srcGrid, clipBBox);

    typename GridT::Ptr readClipped;
    {
        io::File f(path);
        f.open();
        readClipped = gridPtrCast<GridT>(f.readGrid(gridName, clipBBox));
        f.close();
    }
    ASSERT_TRUE(readClipped);
    EXPECT_TRUE(srcClipped->tree().hasSameTopology(readClipped->tree()));
    {
        auto readAcc = readClipped->getConstAccessor();
        for (typename GridT::ValueOnCIter it = srcClipped->cbeginValueOn(); it; ++it) {
            EXPECT_EQ(*it, readAcc.getValue(it.getCoord()));
        }
    }

    // topology-only read
    ReadOptions topoOpts;
    topoOpts.readMode = ReadMode::TopologyOnly;

    typename GridT::Ptr readTopo;
    {
        io::File f(path);
        f.open();
        GridBase::Ptr base;
        EXPECT_NO_THROW(base = f.readGrid(gridName, topoOpts));
        readTopo = gridPtrCast<GridT>(base);
        f.close();
    }
    ASSERT_TRUE(readTopo);
    EXPECT_EQ(readTopo->activeVoxelCount(), Index64(0));
    EXPECT_TRUE(readTopo->tree().leafCount() == 0);
    EXPECT_EQ(readTopo->getName(), gridName);

    // Cleanup
    std::remove(path.c_str());
}

template <typename GridT>
void testCodecIOImpl(
    const std::string& gridName,
    const typename GridT::ValueType& bgValue,
    const typename GridT::ValueType& fillValue)
{
    // initialize to register all the codecs
    openvdb::io::CodecRegistry::clear();
    openvdb::io::internal::initialize();
    // ensure the codec is registered
    ASSERT_TRUE(openvdb::io::CodecRegistry::isRegistered(GridT::gridType()));
    // test the io implementation (codec)
    testIOImpl<GridT>(gridName, bgValue, fillValue);
    // clear the codec registry (now read/write falls back to Tree I/O)
    openvdb::io::CodecRegistry::clear();
    // ensure the codec is not registered
    ASSERT_FALSE(openvdb::io::CodecRegistry::isRegistered(GridT::gridType()));
    // test the io implementation (tree I/O)
    testIOImpl<GridT>(gridName, bgValue, fillValue);
}

TEST_F(TestCodec, testFloatCodecIO) { testCodecIOImpl<openvdb::FloatGrid>("float_grid", 0.0f, 1.0f); }
TEST_F(TestCodec, testDoubleCodecIO) { testCodecIOImpl<openvdb::DoubleGrid>("double_grid", 0.0, 1.0); }
TEST_F(TestCodec, testInt32CodecIO) { testCodecIOImpl<openvdb::Int32Grid>("int32_grid", 0, 1); }
TEST_F(TestCodec, testInt64CodecIO) { testCodecIOImpl<openvdb::Int64Grid>("int64_grid", openvdb::Int64(0), openvdb::Int64(1)); }
TEST_F(TestCodec, testHalfCodecIO) { testCodecIOImpl<openvdb::HalfGrid>("half_grid", openvdb::Half(0.0), openvdb::Half(1.5)); }
TEST_F(TestCodec, testVec3ICodecIO) { testCodecIOImpl<openvdb::Vec3IGrid>("vec3i_grid", openvdb::Vec3i(0), openvdb::Vec3i(1, 2, 3)); }
TEST_F(TestCodec, testVec3SCodecIO) { testCodecIOImpl<openvdb::Vec3SGrid>("vec3s_grid", openvdb::Vec3s(0.0f), openvdb::Vec3s(1.0f, 2.0f, 3.0f)); }
TEST_F(TestCodec, testVec3DCodecIO) { testCodecIOImpl<openvdb::Vec3DGrid>("vec3d_grid", openvdb::Vec3d(0.0), openvdb::Vec3d(1.0, 2.0, 3.0)); }
TEST_F(TestCodec, testBoolCodecIO) { testCodecIOImpl<openvdb::BoolGrid>("bool_grid", false, true); }
TEST_F(TestCodec, testMaskCodecIO) { testCodecIOImpl<openvdb::MaskGrid>("mask_grid", false, true); }
