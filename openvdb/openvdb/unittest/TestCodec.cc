// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/io/Codec.h>
#include <openvdb/io/File.h>
#include <openvdb/io/Stream.h>
#include <openvdb/openvdb.h>
#include <openvdb/codecs/ScalarCodec.h>
#include <openvdb/Exceptions.h>
#include <openvdb/tools/Clip.h>
#include <gtest/gtest.h>
#include <cstdio> // for remove()
#include <fstream>
#include <sstream>

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


TEST_F(TestCodec, testInitializeIdempotent)
{
    using namespace openvdb::io;

    // Calling initialize() twice without uninitialize() in between must not throw.
    // Previously registerCodecByName() threw KeyError on the duplicate registration.
    CodecRegistry::clear();
    EXPECT_NO_THROW(internal::initialize());
    EXPECT_NO_THROW(internal::initialize());

    // Codecs must still be registered after the second call.
    EXPECT_TRUE(CodecRegistry::isRegistered(openvdb::BoolGrid::gridType()));
    EXPECT_TRUE(CodecRegistry::isRegistered(openvdb::FloatGrid::gridType()));

    internal::uninitialize();
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
    // TopologyOnly: full tree structure is read (topology + active-voxel masks),
    // leaf buffers are allocated and zero-filled, values are not read.
    EXPECT_EQ(readTopo->tree().leafCount(), srcGrid->tree().leafCount());
    EXPECT_TRUE(readTopo->tree().leafCount() > 0);
    EXPECT_EQ(readTopo->activeVoxelCount(), srcGrid->activeVoxelCount());
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(readTopo->tree()));
    // verify leaf buffers are allocated (bool/mask buffers are always present, skip empty() check)
    if constexpr (!std::is_same_v<typename GridT::ValueType, bool>) {
        for (auto leafIter = readTopo->tree().cbeginLeaf(); leafIter; ++leafIter) {
            EXPECT_FALSE(leafIter->buffer().empty());
        }
    }
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

#if OPENVDB_ABI_VERSION_NUMBER < 14
    // clear the codec registry (now read/write falls back to Tree I/O)
    openvdb::io::CodecRegistry::clear();
    // ensure the codec is not registered
    ASSERT_FALSE(openvdb::io::CodecRegistry::isRegistered(GridT::gridType()));
    // test the io implementation (tree I/O)
    testIOImpl<GridT>(gridName, bgValue, fillValue);
#endif // OPENVDB_ABI_VERSION_NUMBER < 14
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

TEST_F(TestCodec, testFloatToHalfCodecConversion)
{
    using namespace openvdb;
    using namespace openvdb::io;

    openvdb::initialize();
    CodecRegistry::clear();

    // Verify the conversion codec name
    const std::string expectedName = FloatGrid::gridType() + "_to_half";
    EXPECT_EQ((codecs::ScalarCodec<HalfGrid, FloatGrid, CodecMode::ReadOnly>::name()),
              expectedName);

    // Verify the codec is registered after initialize()
    io::internal::initialize();
    EXPECT_TRUE(CodecRegistry::isRegistered(expectedName));

    // Write a FloatGrid with a known fill value (1.5f is exactly representable in half)
    const std::string floatPath = "test_float_to_half.vdb";
    const std::string gridName = "float_grid";
    FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
    srcGrid->setName(gridName);
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    {
        io::File f(floatPath);
        f.write(GridPtrVec{srcGrid});
    }

    // Read back with ReadMode::Half — triggers the float-to-half conversion codec
    ReadOptions halfOpts;
    halfOpts.readMode = ReadMode::Half;

    GridBase::Ptr base;
    {
        io::File f(floatPath);
        f.open();
        base = f.readGrid(gridName, halfOpts);
        f.close();
    }
    ASSERT_TRUE(base);

    // The returned grid must be a HalfGrid
    EXPECT_TRUE(base->isType<HalfGrid>());
    HalfGrid::Ptr halfGrid = gridPtrCast<HalfGrid>(base);
    ASSERT_TRUE(halfGrid);

    // Topology must match the source FloatGrid
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(halfGrid->tree()));

    // All active voxel values must equal Half(1.5f)
    for (HalfGrid::ValueOnCIter it = halfGrid->cbeginValueOn(); it; ++it) {
        EXPECT_EQ(*it, Half(1.5f));
    }

    // Background value must equal Half(0.0f)
    EXPECT_EQ(halfGrid->background(), Half(0.0f));

    // Cleanup
    std::remove(floatPath.c_str());
}

TEST_F(TestCodec, testFloatToHalfCodecConversionNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(3.25f);
    srcGrid->setName("float_to_half");
    srcGrid->tree().setValue(Coord(0, 0, 0), 1.0f / 3.0f);

    const std::string path = "test_float_to_half_no_offsets.vdb";

    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::Half;

    {
        io::File f(path);
        f.open();
        GridPtrVecPtr grids = f.getGrids(readOptions);
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(1), grids->size());
        EXPECT_TRUE((*grids)[0]->isType<HalfGrid>());
        HalfGrid::Ptr halfGrid = gridPtrCast<HalfGrid>((*grids)[0]);
        ASSERT_TRUE(halfGrid);
        EXPECT_EQ(halfGrid->tree().getValue(Coord(0, 0, 0)), Half(1.0f / 3.0f));
        EXPECT_EQ(halfGrid->background(), Half(3.25f));
        f.close();
    }

    {
        io::File f(path);
        f.open();
        GridBase::Ptr grid = f.readGrid(srcGrid->getName(), readOptions);
        ASSERT_TRUE(grid);
        EXPECT_TRUE(grid->isType<HalfGrid>());
        HalfGrid::Ptr halfGrid = gridPtrCast<HalfGrid>(grid);
        ASSERT_TRUE(halfGrid);
        EXPECT_EQ(halfGrid->tree().getValue(Coord(0, 0, 0)), Half(1.0f / 3.0f));
        EXPECT_EQ(halfGrid->background(), Half(3.25f));
        f.close();
    }

    std::remove(path.c_str());
}

TEST_F(TestCodec, testBoolAndMaskConversionNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    auto runCase = [](ReadMode mode) {
        FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
        srcGrid->setName("float_grid");
        srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

        const std::string path =
            "test_bool_mask_no_offsets_" + std::to_string(int(mode)) + ".vdb";
        {
            std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
            io::Stream(os).write(GridPtrVec{srcGrid});
        }

        ReadOptions readOptions;
        readOptions.readMode = mode;

        auto checkGrid = [&](const GridBase::Ptr& grid) {
            ASSERT_TRUE(grid);
            if (mode == ReadMode::Bool) {
                EXPECT_TRUE(grid->isType<BoolGrid>());
                BoolGrid::Ptr dstGrid = gridPtrCast<BoolGrid>(grid);
                ASSERT_TRUE(dstGrid);
                EXPECT_TRUE(srcGrid->tree().hasSameTopology(dstGrid->tree()));
                for (BoolGrid::ValueOnCIter it = dstGrid->cbeginValueOn(); it; ++it) {
                    EXPECT_EQ(*it, true);
                }
            } else {
                EXPECT_TRUE(grid->isType<MaskGrid>());
                MaskGrid::Ptr dstGrid = gridPtrCast<MaskGrid>(grid);
                ASSERT_TRUE(dstGrid);
                EXPECT_TRUE(srcGrid->tree().hasSameTopology(dstGrid->tree()));
                for (MaskGrid::ValueOnCIter it = dstGrid->cbeginValueOn(); it; ++it) {
                    EXPECT_EQ(*it, true);
                }
            }
        };

        {
            io::File f(path);
            f.open();
            GridPtrVecPtr grids = f.getGrids(readOptions);
            ASSERT_TRUE(grids);
            ASSERT_EQ(size_t(1), grids->size());
            checkGrid((*grids)[0]);
            f.close();
        }

        {
            io::File f(path);
            f.open();
            checkGrid(f.readGrid(srcGrid->getName(), readOptions));
            f.close();
        }

        std::remove(path.c_str());
    };

    runCase(ReadMode::Bool);
    runCase(ReadMode::Mask);
}

#if OPENVDB_ABI_VERSION_NUMBER < 14

TEST_F(TestCodec, testVec3FallsBackWithWarningNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    Vec3SGrid::Ptr srcGrid = Vec3SGrid::create(Vec3s(0.0f));
    srcGrid->setName("vec3_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), Vec3s(1.0f), true);

    const std::string path = "test_vec3_fallback_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::Bool;

    // Determine the warning the offsets path produces for the same request,
    // so the two paths can be compared for parity.
    std::string offsetsWarning;
    {
        const std::string offsetsPath = "test_vec3_fallback_offsets.vdb";
        {
            io::File f(offsetsPath);
            f.write(GridPtrVec{srcGrid});
        }
        io::File f(offsetsPath);
        f.open();
        f.enableReadDiagnostics();
        GridBase::Ptr grid = f.readGrid(srcGrid->getName(), readOptions);
        ASSERT_TRUE(grid);
        EXPECT_TRUE(grid->isType<Vec3SGrid>());
        ASSERT_EQ(size_t(1), f.readDiagnostics().diagnostics().size());
        offsetsWarning = f.readDiagnostics().diagnostics()[0].message;
        f.close();
        std::remove(offsetsPath.c_str());
    }

    {
        io::File f(path);
        f.open();
        f.enableReadDiagnostics();
        GridBase::Ptr grid;
        EXPECT_NO_THROW(grid = f.readGrid(srcGrid->getName(), readOptions));
        ASSERT_TRUE(grid);
        EXPECT_TRUE(grid->isType<Vec3SGrid>());
        ASSERT_EQ(size_t(1), f.readDiagnostics().diagnostics().size());
        EXPECT_EQ(f.readDiagnostics().diagnostics()[0].message, offsetsWarning);
        f.close();
    }

    std::remove(path.c_str());
}

#endif // OPENVDB_ABI_VERSION_NUMBER < 14

#if OPENVDB_ABI_VERSION_NUMBER < 14

// Grids with no codec use the tree I/O path, which keeps its own per-grid stream metadata.
// Several grids and repeated reads from one open file catch a stale stream metadata pointer.
TEST_F(TestCodec, testNoCodecMultipleGridsRepeatedReads)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    ASSERT_FALSE(CodecRegistry::isRegistered(FloatGrid::gridType()));

    FloatGrid::Ptr floatGrid = FloatGrid::create(0.0f);
    floatGrid->setName("float_grid");
    floatGrid->tree().setValue(Coord(1, 2, 3), 1.0f);

    Int32Grid::Ptr intGrid = Int32Grid::create(0);
    intGrid->setName("int_grid");
    intGrid->tree().setValue(Coord(4, 5, 6), 7);

    FloatGrid::Ptr halfGrid = FloatGrid::create(0.0f);
    halfGrid->setName("half_grid");
    halfGrid->setSaveFloatAsHalf(true);
    halfGrid->tree().setValue(Coord(7, 8, 9), 3.0f);

    const GridPtrVec grids{floatGrid, intGrid, halfGrid};

    auto check = [&](const GridPtrVec& read) {
        ASSERT_EQ(size_t(3), read.size());
        for (const GridBase::Ptr& grid : read) {
            if (grid->getName() == "float_grid") {
                EXPECT_EQ(1.0f, gridPtrCast<FloatGrid>(grid)->tree().getValue(Coord(1, 2, 3)));
            } else if (grid->getName() == "int_grid") {
                EXPECT_EQ(7, gridPtrCast<Int32Grid>(grid)->tree().getValue(Coord(4, 5, 6)));
            } else {
                EXPECT_EQ("half_grid", grid->getName());
                EXPECT_EQ(3.0f, gridPtrCast<FloatGrid>(grid)->tree().getValue(Coord(7, 8, 9)));
            }
        }
    };

    // File with grid offsets: read every way, more than once, on one open file.
    const std::string path = "test_no_codec_multiple_grids.vdb";
    {
        io::File file(path);
        file.write(grids);
    }
    {
        io::File file(path);
        file.open();
        for (int repeat = 0; repeat < 2; ++repeat) {
            check(*file.getGrids());
            GridPtrVec byName;
            for (const GridBase::Ptr& grid : grids) byName.push_back(file.readGrid(grid->getName()));
            check(byName);
        }
        file.close();
    }
    std::remove(path.c_str());

    // Stream (no grid offsets): write and read several grids.
    {
        std::ostringstream ostr(std::ios_base::binary);
        io::Stream(ostr).write(grids);
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        io::Stream stream(istr);
        check(*stream.getGrids());
    }
}

#endif // OPENVDB_ABI_VERSION_NUMBER < 14

#if OPENVDB_ABI_VERSION_NUMBER >= 14

// Without a codec there is no tree I/O to fall back on, so reading and writing must fail
// instead of producing a grid with no topology or buffers.
TEST_F(TestCodec, testMissingCodecThrows)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr grid = FloatGrid::create(0.0f);
    grid->setName("density");
    grid->fill(CoordBBox(Coord(0), Coord(7)), 1.0f, true);

    const std::string path = "test_missing_codec.vdb";
    {
        io::File file(path);
        file.write(GridPtrVec{grid});
    }

    CodecRegistry::clear();

    // Reading
    {
        io::File file(path);
        file.open();
        EXPECT_THROW(file.readGrid("density"), IoError);
        file.close();
    }

    // Writing, which must fail before anything is written
    {
        std::ostringstream stream(std::ios_base::binary);
        EXPECT_THROW(io::Stream(stream).write(GridPtrVec{grid}), IoError);
        EXPECT_EQ(std::string::size_type(0), stream.str().size());
    }

    std::remove(path.c_str());
}

#endif // OPENVDB_ABI_VERSION_NUMBER >= 14

namespace {

// Writes the same bytes as the scalar codec and records the section size that
// readBuffers() is given.
struct SizeRecordingCodec final : public openvdb::codecs::TopologyCodec<openvdb::FloatGrid>
{
    static int64_t lastSize;
    static int64_t sectionSize;

    void readBuffers(std::istream& is, int64_t size, openvdb::io::CodecData& data,
        const openvdb::io::ReadOptions& options, openvdb::io::ReadDiagnostics&) const override
    {
        using namespace openvdb;
        lastSize = size;
        auto& topologyData = static_cast<codecs::TopologyCodecData<float>&>(data);
        codecs::internal::scalarCodecReadBuffers<FloatGrid, FloatGrid>(
            static_cast<FloatGrid&>(*data.grid), is, options, &topologyData.storageBackground);
    }

    void writeBuffers(std::ostream& os, const openvdb::GridBase& grid,
        const openvdb::io::WriteOptions&) const override
    {
        const std::streampos start = os.tellp();
        openvdb::codecs::internal::scalarCodecWriteBuffers(
            static_cast<const openvdb::FloatGrid&>(grid), os);
        sectionSize = int64_t(os.tellp() - start);
    }
};

int64_t SizeRecordingCodec::lastSize = -2;
int64_t SizeRecordingCodec::sectionSize = -2;

} // anonymous namespace


// The size given to readBuffers() is the byte count of the buffer section alone, and -1
// when the size is not known.
TEST_F(TestCodec, testReadBuffersSize)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    CodecRegistry::registerCodecByName(openvdb::FloatGrid::gridType(), std::make_unique<SizeRecordingCodec>());

    FloatGrid::Ptr grid = FloatGrid::create(0.0f);
    grid->setName("density");
    // Not aligned to a leaf, so the voxels are stored in leaf buffers and not as a tile.
    grid->fill(CoordBBox(Coord(1), Coord(10)), 1.0f, true);

    const std::string path = "test_read_buffers_size.vdb";
    {
        io::File file(path);
        file.write(GridPtrVec{grid});
    }
    ASSERT_GT(SizeRecordingCodec::sectionSize, 0);

    SizeRecordingCodec::lastSize = -2;
    {
        io::File file(path);
        file.open();
        file.readGrid("density");
        file.close();
    }
    EXPECT_EQ(SizeRecordingCodec::sectionSize, SizeRecordingCodec::lastSize);

    // A stream has no positions, so the size is not known.
    std::ostringstream output(std::ios_base::binary);
    io::Stream(output).write(GridPtrVec{grid});
    SizeRecordingCodec::lastSize = -2;
    std::istringstream input(output.str(), std::ios_base::binary);
    io::Stream(input).getGrids();
    EXPECT_EQ(int64_t(-1), SizeRecordingCodec::lastSize);

    std::remove(path.c_str());
}

namespace {

// Writes the same bytes as the scalar codec and records the calls to beginRead() and finishRead().
struct FinishReadRecordingCodec final : public openvdb::codecs::TopologyCodec<openvdb::FloatGrid>
{
    static int beginReadCalls;
    static int finishReadCalls;
    static bool throwInBeginRead;
    static bool throwInFinishRead;
    // 'B' for each beginRead() call, 'b' for each readBuffers() call and 'f' for each
    // finishRead() call, in order.
    static std::string events;

    static void reset()
    {
        beginReadCalls = 0;
        finishReadCalls = 0;
        throwInBeginRead = false;
        throwInFinishRead = false;
        events.clear();
    }

    void readBuffers(std::istream& is, int64_t, openvdb::io::CodecData& data,
        const openvdb::io::ReadOptions& options, openvdb::io::ReadDiagnostics&) const override
    {
        using namespace openvdb;
        events += 'b';
        auto& topologyData = static_cast<codecs::TopologyCodecData<float>&>(data);
        codecs::internal::scalarCodecReadBuffers<FloatGrid, FloatGrid>(
            static_cast<FloatGrid&>(*data.grid), is, options, &topologyData.storageBackground);
    }

    void beginRead(openvdb::io::CodecData&, const openvdb::io::ReadOptions&,
        openvdb::io::ReadDiagnostics&) const override
    {
        ++beginReadCalls;
        events += 'B';
        if (throwInBeginRead) OPENVDB_THROW(openvdb::IoError, "beginRead failed");
    }

    void finishRead(openvdb::io::CodecData&, const openvdb::io::ReadOptions&,
        openvdb::io::ReadDiagnostics&) const override
    {
        ++finishReadCalls;
        events += 'f';
        if (throwInFinishRead) OPENVDB_THROW(openvdb::IoError, "finishRead failed");
    }

    void writeBuffers(std::ostream& os, const openvdb::GridBase& grid,
        const openvdb::io::WriteOptions&) const override
    {
        openvdb::codecs::internal::scalarCodecWriteBuffers(
            static_cast<const openvdb::FloatGrid&>(grid), os);
    }
};

int FinishReadRecordingCodec::beginReadCalls = 0;
int FinishReadRecordingCodec::finishReadCalls = 0;
bool FinishReadRecordingCodec::throwInBeginRead = false;
bool FinishReadRecordingCodec::throwInFinishRead = false;
std::string FinishReadRecordingCodec::events;

} // anonymous namespace


// beginRead() and finishRead() are called once for each grid that has sections, for every way
// of reading and every read mode, and never for an instance.
TEST_F(TestCodec, testFinishReadIsCalledOncePerGrid)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    CodecRegistry::registerCodecByName(FloatGrid::gridType(),
        std::make_unique<FinishReadRecordingCodec>());

    FloatGrid::Ptr apple = FloatGrid::create(0.0f);
    apple->setName("apple");
    apple->fill(CoordBBox(Coord(1), Coord(10)), 1.0f, true);
    // A shallow copy shares the tree, so it is written as an instance of apple.
    GridBase::Ptr mango = apple->copyGrid();
    mango->setName("mango");
    const GridPtrVec grids{apple, mango};

    const std::string path = "test_finish_read_once.vdb";
    {
        io::File file(path);
        file.write(grids);
    }
    std::ostringstream output(std::ios_base::binary);
    io::Stream(output).write(grids);

    ReadOptions topologyOnly;
    topologyOnly.readMode = ReadMode::TopologyOnly;
    ReadOptions metadataOnly;
    metadataOnly.readMode = ReadMode::MetadataOnly;
    ReadOptions clipped;
    clipped.clipBBox = BBoxd(Vec3d(0.0), Vec3d(3.0));

    auto expectCalls = [](int expected, const char* what) {
        EXPECT_EQ(expected, FinishReadRecordingCodec::beginReadCalls) << what;
        EXPECT_EQ(expected, FinishReadRecordingCodec::finishReadCalls) << what;
        FinishReadRecordingCodec::reset();
    };

    FinishReadRecordingCodec::reset();
    {
        io::File file(path);
        file.open();
        file.readGrid("apple");
        expectCalls(1, "readGrid");
        file.readGrid("apple", topologyOnly);
        expectCalls(1, "readGrid, topology only");
        file.readGrid("apple", metadataOnly);
        expectCalls(1, "readGrid, metadata only");
        file.readGrid("apple", clipped);
        expectCalls(1, "readGrid, clipped");
        // An instance has no phases of its own, so only its parent is finished.
        file.readGrid("mango");
        expectCalls(1, "readGrid, instance");
        file.readGrids("apple");
        expectCalls(1, "readGrids");
        file.readGridMetadata("apple");
        expectCalls(1, "readGridMetadata");
        file.readAllGridMetadata();
        expectCalls(1, "readAllGridMetadata");
        file.getGrids();
        expectCalls(1, "getGrids");
        file.getGrids(topologyOnly);
        expectCalls(1, "getGrids, topology only");
        file.getGrids(metadataOnly);
        expectCalls(1, "getGrids, metadata only");

        // beginRead() comes before the buffers of the grid are read and finishRead() after.
        file.getGrids();
        EXPECT_EQ(std::string("Bbf"), FinishReadRecordingCodec::events);
        FinishReadRecordingCodec::reset();
        // Metadata only reads no buffers, but is still begun and finished.
        file.getGrids(metadataOnly);
        EXPECT_EQ(std::string("Bf"), FinishReadRecordingCodec::events);
        FinishReadRecordingCodec::reset();
        file.close();
    }

    {
        std::istringstream input(output.str(), std::ios_base::binary);
        io::Stream stream(input);
        expectCalls(1, "stream");
    }

    std::remove(path.c_str());
}


// An error from beginRead() is thrown to the caller and no other phase is called.
TEST_F(TestCodec, testBeginReadError)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    CodecRegistry::registerCodecByName(FloatGrid::gridType(),
        std::make_unique<FinishReadRecordingCodec>());

    FloatGrid::Ptr apple = FloatGrid::create(0.0f);
    apple->setName("apple");
    apple->fill(CoordBBox(Coord(1), Coord(10)), 1.0f, true);

    const std::string path = "test_begin_read_error.vdb";
    {
        io::File file(path);
        file.write(GridPtrVec{apple});
    }
    std::ostringstream output(std::ios_base::binary);
    io::Stream(output).write(GridPtrVec{apple});

    FinishReadRecordingCodec::reset();
    FinishReadRecordingCodec::throwInBeginRead = true;
    {
        io::File file(path);
        file.open();
        EXPECT_THROW(file.readGrid("apple"), IoError);
        // No other phase runs after beginRead() throws.
        EXPECT_EQ(std::string("B"), FinishReadRecordingCodec::events);
        EXPECT_EQ(0, FinishReadRecordingCodec::finishReadCalls);
        EXPECT_THROW(file.readGrids("apple"), IoError);
        EXPECT_THROW(file.getGrids(), IoError);
        file.close();
    }
    {
        std::istringstream input(output.str(), std::ios_base::binary);
        EXPECT_THROW(io::Stream{input}, IoError);
    }
    FinishReadRecordingCodec::reset();

    std::remove(path.c_str());
}


// An error from finishRead() is thrown to the caller and no grid is returned.
TEST_F(TestCodec, testFinishReadError)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    CodecRegistry::registerCodecByName(FloatGrid::gridType(),
        std::make_unique<FinishReadRecordingCodec>());

    FloatGrid::Ptr apple = FloatGrid::create(0.0f);
    apple->setName("apple");
    apple->fill(CoordBBox(Coord(1), Coord(10)), 1.0f, true);

    const std::string path = "test_finish_read_error.vdb";
    {
        io::File file(path);
        file.write(GridPtrVec{apple});
    }
    std::ostringstream output(std::ios_base::binary);
    io::Stream(output).write(GridPtrVec{apple});

    FinishReadRecordingCodec::reset();
    FinishReadRecordingCodec::throwInFinishRead = true;
    {
        io::File file(path);
        file.open();
        EXPECT_THROW(file.readGrid("apple"), IoError);
        EXPECT_THROW(file.readGrids("apple"), IoError);
        EXPECT_THROW(file.getGrids(), IoError);
        file.close();
    }
    {
        std::istringstream input(output.str(), std::ios_base::binary);
        EXPECT_THROW(io::Stream{input}, IoError);
    }
    FinishReadRecordingCodec::reset();

    std::remove(path.c_str());
}


TEST_F(TestCodec, testOffsetsAndNoOffsetsParity)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(2.0f);
    srcGrid->setName("parity_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 0.25f, true);

    const std::string offsetsPath = "test_parity_offsets.vdb";
    const std::string noOffsetsPath = "test_parity_no_offsets.vdb";
    {
        io::File f(offsetsPath);
        f.write(GridPtrVec{srcGrid});
    }
    {
        std::ofstream os(noOffsetsPath, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::Half;

    HalfGrid::Ptr offsetsGrid;
    {
        io::File f(offsetsPath);
        f.open();
        offsetsGrid = gridPtrCast<HalfGrid>(f.readGrid(srcGrid->getName(), readOptions));
        f.close();
    }
    HalfGrid::Ptr noOffsetsGrid;
    {
        io::File f(noOffsetsPath);
        f.open();
        noOffsetsGrid = gridPtrCast<HalfGrid>(f.readGrid(srcGrid->getName(), readOptions));
        f.close();
    }

    ASSERT_TRUE(offsetsGrid);
    ASSERT_TRUE(noOffsetsGrid);
    EXPECT_EQ(offsetsGrid->type(), noOffsetsGrid->type());
    EXPECT_TRUE(offsetsGrid->tree().hasSameTopology(noOffsetsGrid->tree()));
    for (HalfGrid::ValueOnCIter it = offsetsGrid->cbeginValueOn(); it; ++it) {
        EXPECT_EQ(*it, noOffsetsGrid->tree().getValue(it.getCoord()));
    }

    std::remove(offsetsPath.c_str());
    std::remove(noOffsetsPath.c_str());
}

TEST_F(TestCodec, testClipBBoxInGetGridsNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
    srcGrid->setName("clip_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    const std::string path = "test_clip_getgrids_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    const BBoxd clipBBox(Vec3d(0.0), Vec3d(3.5));
    auto srcClipped = tools::clip(*srcGrid, clipBBox);

    io::File f(path);
    f.open();
    f.enableReadDiagnostics();

    ReadOptions clipOptions;
    clipOptions.clipBBox = clipBBox;

    {
        GridPtrVecPtr grids = f.getGrids(clipOptions);
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(1), grids->size());
        FloatGrid::Ptr clipped = gridPtrCast<FloatGrid>((*grids)[0]);
        ASSERT_TRUE(clipped);
        EXPECT_TRUE(srcClipped->tree().hasSameTopology(clipped->tree()));
        EXPECT_EQ(size_t(1), f.readDiagnostics().diagnostics().size());
    }

    // mGrids must be unmutated: a default-options call still sees the full topology.
    {
        GridPtrVecPtr grids = f.getGrids();
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(1), grids->size());
        FloatGrid::Ptr full = gridPtrCast<FloatGrid>((*grids)[0]);
        ASSERT_TRUE(full);
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(full->tree()));
    }

    f.close();
    std::remove(path.c_str());
}

TEST_F(TestCodec, testClipOnOffsetsFileRecordsNoDiagnostic)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
    srcGrid->setName("clip_grid_offsets");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    const std::string path = "test_clip_getgrids_offsets.vdb";
    {
        io::File f(path);
        f.write(GridPtrVec{srcGrid});
    }

    const BBoxd clipBBox(Vec3d(0.0), Vec3d(3.5));

    ReadOptions clipOptions;
    clipOptions.clipBBox = clipBBox;

    io::File f(path);
    f.open();
    f.enableReadDiagnostics();

    GridPtrVecPtr grids = f.getGrids(clipOptions);
    ASSERT_TRUE(grids);
    ASSERT_EQ(size_t(1), grids->size());
    EXPECT_EQ(size_t(0), f.readDiagnostics().diagnostics().size());

    GridBase::Ptr grid = f.readGrid(srcGrid->getName(), clipOptions);
    ASSERT_TRUE(grid);
    EXPECT_EQ(size_t(0), f.readDiagnostics().diagnostics().size());

    f.close();
    std::remove(path.c_str());
}

TEST_F(TestCodec, testClipAndConversionTogetherNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
    srcGrid->setName("clip_convert_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    const std::string path = "test_clip_and_convert_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    const BBoxd clipBBox(Vec3d(0.0), Vec3d(3.5));
    auto srcClipped = tools::clip(*srcGrid, clipBBox);

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::Half;
    readOptions.clipBBox = clipBBox;

    {
        io::File f(path);
        f.open();
        f.enableReadDiagnostics();
        GridPtrVecPtr grids = f.getGrids(readOptions);
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(1), grids->size());
        EXPECT_TRUE((*grids)[0]->isType<HalfGrid>());
        HalfGrid::Ptr clipped = gridPtrCast<HalfGrid>((*grids)[0]);
        ASSERT_TRUE(clipped);
        EXPECT_TRUE(srcClipped->tree().hasSameTopology(clipped->tree()));
        // Only the clip diagnostic is expected, the Half conversion succeeded.
        EXPECT_EQ(size_t(1), f.readDiagnostics().diagnostics().size());
        f.close();
    }

    {
        io::File f(path);
        f.open();
        GridBase::Ptr grid = f.readGrid(srcGrid->getName(), readOptions);
        ASSERT_TRUE(grid);
        EXPECT_TRUE(grid->isType<HalfGrid>());
        HalfGrid::Ptr clipped = gridPtrCast<HalfGrid>(grid);
        ASSERT_TRUE(clipped);
        EXPECT_TRUE(srcClipped->tree().hasSameTopology(clipped->tree()));
        f.close();
    }

    std::remove(path.c_str());
}

TEST_F(TestCodec, testTopologyOnlyWarnsAndIgnoresNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
    srcGrid->setName("topology_only_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    const std::string path = "test_topology_only_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::TopologyOnly;

    io::File f(path);
    f.open();
    f.enableReadDiagnostics();

    GridBase::Ptr grid;
    EXPECT_NO_THROW(grid = f.readGrid(srcGrid->getName(), readOptions));
    ASSERT_TRUE(grid);
    EXPECT_TRUE(grid->isType<FloatGrid>());
    EXPECT_EQ(size_t(1), f.readDiagnostics().diagnostics().size());

    // Whatever happens to the values, the topology must match the source grid.
    FloatGrid::Ptr floatGrid = gridPtrCast<FloatGrid>(grid);
    ASSERT_TRUE(floatGrid);
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(floatGrid->tree()));

    f.close();
    std::remove(path.c_str());
}

TEST_F(TestCodec, testGetGridsTopologyOnlyWarnsNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
    srcGrid->setName("topology_only_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    const std::string path = "test_getgrids_topology_only_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::TopologyOnly;

    // Pin getGrids() and readGrid() to the same diagnostic message.
    std::string readGridMessage;
    {
        io::File f(path);
        f.open();
        f.enableReadDiagnostics();
        GridBase::Ptr grid;
        EXPECT_NO_THROW(grid = f.readGrid(srcGrid->getName(), readOptions));
        ASSERT_TRUE(grid);
        EXPECT_TRUE(grid->isType<FloatGrid>());
        ASSERT_EQ(size_t(1), f.readDiagnostics().diagnostics().size());
        readGridMessage = f.readDiagnostics().diagnostics()[0].message;
        FloatGrid::Ptr floatGrid = gridPtrCast<FloatGrid>(grid);
        ASSERT_TRUE(floatGrid);
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(floatGrid->tree()));
        f.close();
    }

    {
        io::File f(path);
        f.open();
        f.enableReadDiagnostics();
        GridPtrVecPtr grids;
        EXPECT_NO_THROW(grids = f.getGrids(readOptions));
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(1), grids->size());
        EXPECT_TRUE((*grids)[0]->isType<FloatGrid>());
        ASSERT_EQ(size_t(1), f.readDiagnostics().diagnostics().size());
        EXPECT_EQ(f.readDiagnostics().diagnostics()[0].message, readGridMessage);
        FloatGrid::Ptr floatGrid = gridPtrCast<FloatGrid>((*grids)[0]);
        ASSERT_TRUE(floatGrid);
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(floatGrid->tree()));
        f.close();
    }

    std::remove(path.c_str());
}

TEST_F(TestCodec, testMetadataOnlyNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatGrid::Ptr srcGrid = FloatGrid::create(0.0f);
    srcGrid->setName("metadata_only_grid");
    srcGrid->insertMeta("author", StringMetadata("Einstein"));
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    const std::string path = "test_metadata_only_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{srcGrid});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::MetadataOnly;

    GridBase::Ptr expected;
    {
        io::File f(path);
        f.open();
        expected = f.readGridMetadata(srcGrid->getName());
        f.close();
    }

    {
        io::File f(path);
        f.open();
        f.enableReadDiagnostics();
        GridPtrVecPtr grids = f.getGrids(readOptions);
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(1), grids->size());
        GridBase::Ptr grid = (*grids)[0];
        EXPECT_TRUE(grid->isType<FloatGrid>());
        EXPECT_TRUE(gridPtrCast<FloatGrid>(grid)->tree().empty());
        EXPECT_EQ(std::string("Einstein"), grid->metaValue<std::string>("author"));
        EXPECT_EQ(expected->transform(), grid->transform());
        EXPECT_EQ(size_t(0), f.readDiagnostics().diagnostics().size());
        f.close();
    }

    {
        io::File f(path);
        f.open();
        f.enableReadDiagnostics();
        GridBase::Ptr grid = f.readGrid(srcGrid->getName(), readOptions);
        ASSERT_TRUE(grid);
        EXPECT_TRUE(grid->isType<FloatGrid>());
        EXPECT_TRUE(gridPtrCast<FloatGrid>(grid)->tree().empty());
        EXPECT_EQ(std::string("Einstein"), grid->metaValue<std::string>("author"));
        EXPECT_EQ(expected->transform(), grid->transform());
        EXPECT_EQ(size_t(0), f.readDiagnostics().diagnostics().size());
        f.close();
    }

    std::remove(path.c_str());
}

TEST_F(TestCodec, testClipInstancingNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatTree::Ptr tree(new FloatTree(0.0f));
    tree->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    GridBase::Ptr grid1 = createGrid(tree);
    grid1->setName("parent");
    GridBase::Ptr grid2 = createGrid(tree); // instance of grid1
    grid2->setName("instance");

    const std::string path = "test_clip_instancing_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{grid1, grid2});
    }

    ReadOptions readOptions;
    readOptions.clipBBox = BBoxd(Vec3d(0.0), Vec3d(3.5));

    io::File f(path);
    f.open();
    GridPtrVecPtr grids = f.getGrids(readOptions);
    ASSERT_TRUE(grids);
    ASSERT_EQ(size_t(2), grids->size());

    FloatGrid::Ptr resultParent = gridPtrCast<FloatGrid>(findGridByName(*grids, "parent"));
    FloatGrid::Ptr resultInstance = gridPtrCast<FloatGrid>(findGridByName(*grids, "instance"));
    ASSERT_TRUE(resultParent);
    ASSERT_TRUE(resultInstance);

    // Same transform, so the clipped tree is shared.
    EXPECT_EQ(resultParent->treePtr(), resultInstance->treePtr());

    f.close();
    std::remove(path.c_str());
}

TEST_F(TestCodec, testConversionInstancingNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatTree::Ptr tree(new FloatTree(0.0f));
    tree->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    GridBase::Ptr grid1 = createGrid(tree);
    grid1->setName("parent");
    GridBase::Ptr grid2 = createGrid(tree); // instance of grid1
    grid2->setName("instance");

    const std::string path = "test_conversion_instancing_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{grid1, grid2});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::Half;

    {
        io::File f(path);
        f.open();
        GridPtrVecPtr grids = f.getGrids(readOptions);
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(2), grids->size());

        HalfGrid::Ptr resultParent = gridPtrCast<HalfGrid>(findGridByName(*grids, "parent"));
        HalfGrid::Ptr resultInstance = gridPtrCast<HalfGrid>(findGridByName(*grids, "instance"));
        ASSERT_TRUE(resultParent);
        ASSERT_TRUE(resultInstance);

        // Conversion doesn't break instancing, both results share the tree.
        EXPECT_EQ(resultParent->treePtr(), resultInstance->treePtr());
        f.close();
    }

    // Instancing disabled, so the two results don't share a tree.
    {
        io::File f(path);
        f.setInstancingEnabled(false);
        f.open();
        GridPtrVecPtr grids = f.getGrids(readOptions);
        ASSERT_TRUE(grids);
        ASSERT_EQ(size_t(2), grids->size());

        HalfGrid::Ptr resultParent = gridPtrCast<HalfGrid>(findGridByName(*grids, "parent"));
        HalfGrid::Ptr resultInstance = gridPtrCast<HalfGrid>(findGridByName(*grids, "instance"));
        ASSERT_TRUE(resultParent);
        ASSERT_TRUE(resultInstance);
        EXPECT_NE(resultParent->treePtr(), resultInstance->treePtr());
        f.close();
    }

    std::remove(path.c_str());
}

TEST_F(TestCodec, testInstanceKeepsOwnTransformNoGridOffsets)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    FloatTree::Ptr tree(new FloatTree(0.0f));
    tree->fill(CoordBBox(Coord(-5), Coord(5)), 1.5f, true);

    GridBase::Ptr grid1 = createGrid(tree);
    grid1->setName("parent");
    grid1->setTransform(math::Transform::createLinearTransform(1.0));
    GridBase::Ptr grid2 = createGrid(tree); // instance of grid1
    grid2->setName("instance");
    grid2->setTransform(math::Transform::createLinearTransform(2.0));

    const std::string path = "test_instance_own_transform_no_offsets.vdb";
    {
        std::ofstream os(path, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{grid1, grid2});
    }

    ReadOptions readOptions;
    readOptions.readMode = ReadMode::Half;

    io::File f(path);
    f.open();
    GridPtrVecPtr grids = f.getGrids(readOptions);
    ASSERT_TRUE(grids);
    ASSERT_EQ(size_t(2), grids->size());

    HalfGrid::Ptr resultParent = gridPtrCast<HalfGrid>(findGridByName(*grids, "parent"));
    HalfGrid::Ptr resultInstance = gridPtrCast<HalfGrid>(findGridByName(*grids, "instance"));
    ASSERT_TRUE(resultParent);
    ASSERT_TRUE(resultInstance);

    EXPECT_EQ(resultParent->treePtr(), resultInstance->treePtr());
    EXPECT_EQ(1.0, resultParent->voxelSize()[0]);
    EXPECT_EQ(2.0, resultInstance->voxelSize()[0]);

    f.close();
    std::remove(path.c_str());
}

template <typename SrcGridT, typename DstGridT, openvdb::io::ReadMode mode>
void testConvertCodecImpl()
{
    using namespace openvdb;
    using namespace openvdb::io;

    openvdb::io::CodecRegistry::clear();
    openvdb::io::internal::initialize();

    // Verify the conversion codec name
    const std::string expectedName =
        SrcGridT::gridType() + "_to_" + typeNameAsString<typename DstGridT::BuildType>();
    EXPECT_EQ((codecs::ScalarCodec<DstGridT, SrcGridT, CodecMode::ReadOnly>::name()),
              expectedName);

    // Verify the codec is registered after initialize()
    EXPECT_TRUE(CodecRegistry::isRegistered(expectedName));

    // Write a SrcGridT with a non-zero fill value
    const std::string testPath = "test_" + expectedName + ".vdb";
    const std::string gridName = SrcGridT::gridType();
    typename SrcGridT::Ptr srcGrid =
        SrcGridT::create(typename SrcGridT::ValueType(0));
    srcGrid->setName(gridName);
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)),
                  typename SrcGridT::ValueType(1), true);

    {
        io::File f(testPath);
        f.write(GridPtrVec{srcGrid});
    }

    // Read back with the specified ReadMode — triggers the conversion codec
    ReadOptions opts;
    opts.readMode = mode;

    GridBase::Ptr base;
    {
        io::File f(testPath);
        f.open();
        base = f.readGrid(gridName, opts);
        f.close();
    }
    ASSERT_TRUE(base);

    // The returned grid must be a DstGridT
    EXPECT_TRUE(base->isType<DstGridT>());
    typename DstGridT::Ptr dstGrid = gridPtrCast<DstGridT>(base);
    ASSERT_TRUE(dstGrid);

    // Topology must match the source grid
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(dstGrid->tree()));

    // All active voxel values must be true
    for (typename DstGridT::ValueOnCIter it = dstGrid->cbeginValueOn(); it; ++it) {
        EXPECT_EQ(*it, typename DstGridT::ValueType(true));
    }

    // Background value must be false
    EXPECT_EQ(dstGrid->background(), typename DstGridT::ValueType(false));

    // Cleanup
    std::remove(testPath.c_str());
}

TEST_F(TestCodec, testNumericToBoolCodecConversion)
{
    testConvertCodecImpl<openvdb::FloatGrid,  openvdb::BoolGrid, openvdb::io::ReadMode::Bool>();
    testConvertCodecImpl<openvdb::DoubleGrid, openvdb::BoolGrid, openvdb::io::ReadMode::Bool>();
    testConvertCodecImpl<openvdb::Int32Grid,  openvdb::BoolGrid, openvdb::io::ReadMode::Bool>();
    testConvertCodecImpl<openvdb::Int64Grid,  openvdb::BoolGrid, openvdb::io::ReadMode::Bool>();
    testConvertCodecImpl<openvdb::HalfGrid,   openvdb::BoolGrid, openvdb::io::ReadMode::Bool>();
}

TEST_F(TestCodec, testNumericToMaskCodecConversion)
{
    testConvertCodecImpl<openvdb::FloatGrid,  openvdb::MaskGrid, openvdb::io::ReadMode::Mask>();
    testConvertCodecImpl<openvdb::DoubleGrid, openvdb::MaskGrid, openvdb::io::ReadMode::Mask>();
    testConvertCodecImpl<openvdb::Int32Grid,  openvdb::MaskGrid, openvdb::io::ReadMode::Mask>();
    testConvertCodecImpl<openvdb::Int64Grid,  openvdb::MaskGrid, openvdb::io::ReadMode::Mask>();
    testConvertCodecImpl<openvdb::HalfGrid,   openvdb::MaskGrid, openvdb::io::ReadMode::Mask>();
}

// Regression test for the dangling storageBackground pointer bug.
//
// ReadTopologyOp stores storageBackground on its stack frame and registers
// &storageBackground with the stream.  Before the fix, topologyCodecReadTopology
// returned and destroyed ReadTopologyOp before readBuffers() ran; readCompressedValues
// then dereferenced the dead pointer to reconstruct inactive voxels under
// COMPRESS_ACTIVE_MASK, producing garbage inactive values.
//
// The test is deliberately structured to maximize the chance that the freed
// stack frame has been overwritten: a non-zero background (3.0f / 5) forces the
// reconstructed inactive value to be wrong if the pointer is stale, and
// COMPRESS_ACTIVE_MASK (flag 0x2, always on by default) is the code path that
// uses the background pointer.
TEST_F(TestCodec, testInactiveValuesAfterReadBuffers)
{
    using namespace openvdb;
    using namespace openvdb::io;

    openvdb::io::CodecRegistry::clear();
    openvdb::io::internal::initialize();

    // Float: non-zero background, active region surrounded by inactive background voxels.
    {
        const float bg = 3.0f;
        FloatGrid::Ptr src = FloatGrid::create(bg);
        src->setName("float_bg");
        src->fill(CoordBBox(Coord(0), Coord(15)), 1.0f, /*active=*/true);
        src->fill(CoordBBox(Coord(4), Coord(11)), bg,   /*active=*/false);

        const std::string path = "testInactiveVals_float.vdb";
        {
            io::File f(path);
            f.setCompression(COMPRESS_ACTIVE_MASK);
            f.write(GridPtrVec{src});
        }
        FloatGrid::Ptr result;
        {
            io::File f(path);
            f.open();
            result = gridPtrCast<FloatGrid>(f.readGrid("float_bg"));
            f.close();
        }
        ASSERT_TRUE(result);
        EXPECT_EQ(result->background(), bg);
        FloatGrid::ConstAccessor refAcc = src->getConstAccessor();
        for (FloatGrid::ValueAllCIter it = result->cbeginValueAll(); it; ++it) {
            EXPECT_EQ(*it, refAcc.getValue(it.getCoord()));
        }
        std::remove(path.c_str());
    }

    // Int32: non-zero background (5), verify inactive values round-trip.
    {
        const int bg = 5;
        Int32Grid::Ptr src = Int32Grid::create(bg);
        src->setName("int_bg");
        src->fill(CoordBBox(Coord(0), Coord(15)), 99, /*active=*/true);
        src->fill(CoordBBox(Coord(4), Coord(11)), bg, /*active=*/false);

        const std::string path = "testInactiveVals_int.vdb";
        {
            io::File f(path);
            f.setCompression(COMPRESS_ACTIVE_MASK);
            f.write(GridPtrVec{src});
        }
        Int32Grid::Ptr result;
        {
            io::File f(path);
            f.open();
            result = gridPtrCast<Int32Grid>(f.readGrid("int_bg"));
            f.close();
        }
        ASSERT_TRUE(result);
        EXPECT_EQ(result->background(), bg);
        Int32Grid::ConstAccessor refAcc = src->getConstAccessor();
        for (Int32Grid::ValueAllCIter it = result->cbeginValueAll(); it; ++it) {
            EXPECT_EQ(*it, refAcc.getValue(it.getCoord()));
        }
        std::remove(path.c_str());
    }
}

// A MaskGrid has no ValueMask specialization of InternalNode or RootNode, so its
// tiles carry a value separate from the active state. TopologyOnly resets those
// values to the background, which must leave the active topology untouched.
TEST_F(TestCodec, testTopologyOnlyPreservesMaskTiles)
{
    using namespace openvdb;
    using namespace openvdb::io;

    const std::string gridName = "mask_tiles";
    const std::string path = "testTopologyOnlyMaskTiles.vdb";

    // Active tiles at both internal levels and at the root, plus a region of
    // individual voxels so that leaf nodes are exercised too.
    MaskGrid::Ptr src = MaskGrid::create();
    src->setName(gridName);
    src->tree().addTile(/*level=*/1, Coord(0), true, /*active=*/true);
    src->tree().addTile(/*level=*/2, Coord(4096), true, /*active=*/true);
    src->tree().addTile(/*level=*/3, Coord(-8192), true, /*active=*/true);
    src->fill(CoordBBox(Coord(1024), Coord(1030)), true, /*active=*/true);

    const Index64 srcActiveVoxels = src->activeVoxelCount();
    const Index64 srcActiveTiles = src->tree().activeTileCount();
    ASSERT_TRUE(srcActiveTiles > 0);

    {
        io::File f(path);
        f.write(GridPtrVec{src});
    }

    ReadOptions topoOpts;
    topoOpts.readMode = ReadMode::TopologyOnly;

    MaskGrid::Ptr readTopo, readOriginal;
    {
        io::File f(path);
        f.open();
        readTopo = gridPtrCast<MaskGrid>(f.readGrid(gridName, topoOpts));
        readOriginal = gridPtrCast<MaskGrid>(f.readGrid(gridName, ReadOptions{}));
        f.close();
    }
    ASSERT_TRUE(readTopo);
    ASSERT_TRUE(readOriginal);

    // TopologyOnly must not lose the active tiles.
    EXPECT_EQ(readTopo->tree().activeTileCount(), srcActiveTiles);
    EXPECT_EQ(readTopo->activeVoxelCount(), srcActiveVoxels);
    EXPECT_TRUE(src->tree().hasSameTopology(readTopo->tree()));
    EXPECT_TRUE(readOriginal->tree().hasSameTopology(readTopo->tree()));

    // Leaf voxels keep their values because value and active state share one bit.
    auto topoAcc = readTopo->getConstAccessor();
    for (MaskGrid::ValueOnCIter it = readOriginal->cbeginValueOn(); it; ++it) {
        if (it.isVoxelValue()) {
            EXPECT_TRUE(topoAcc.isValueOn(it.getCoord()));
        }
    }

    // Tile values are reset to the background, matching a TopologyCopy, while the
    // tiles stay active.
    for (MaskGrid::ValueOnCIter it = readTopo->cbeginValueOn(); it; ++it) {
        if (!it.isVoxelValue()) {
            EXPECT_FALSE(*it);
        }
    }

    std::remove(path.c_str());
}

// A BoolGrid stores the value and the active state separately, so TopologyOnly is
// expected to discard values while keeping topology.
TEST_F(TestCodec, testTopologyOnlyClearsBoolTiles)
{
    using namespace openvdb;
    using namespace openvdb::io;

    const std::string gridName = "bool_tiles";
    const std::string path = "testTopologyOnlyBoolTiles.vdb";

    // Active tiles at both internal levels, plus a region of individual voxels.
    BoolGrid::Ptr src = BoolGrid::create(/*background=*/false);
    src->setName(gridName);
    src->tree().addTile(/*level=*/1, Coord(0), true, /*active=*/true);
    src->tree().addTile(/*level=*/2, Coord(4096), true, /*active=*/true);
    src->fill(CoordBBox(Coord(1024), Coord(1030)), true, /*active=*/true);

    const Index64 srcActiveVoxels = src->activeVoxelCount();
    const Index64 srcActiveTiles = src->tree().activeTileCount();
    ASSERT_TRUE(srcActiveTiles > 0);

    {
        io::File f(path);
        f.write(GridPtrVec{src});
    }

    ReadOptions topoOpts;
    topoOpts.readMode = ReadMode::TopologyOnly;

    BoolGrid::Ptr readTopo;
    {
        io::File f(path);
        f.open();
        readTopo = gridPtrCast<BoolGrid>(f.readGrid(gridName, topoOpts));
        f.close();
    }
    ASSERT_TRUE(readTopo);

    // Topology is preserved, including active tiles.
    EXPECT_EQ(readTopo->tree().activeTileCount(), srcActiveTiles);
    EXPECT_EQ(readTopo->activeVoxelCount(), srcActiveVoxels);
    EXPECT_TRUE(src->tree().hasSameTopology(readTopo->tree()));

    // Values are discarded: tiles are reset to the background even where active.
    for (BoolGrid::ValueAllCIter it = readTopo->cbeginValueAll(); it; ++it) {
        if (!it.isVoxelValue()) {
            EXPECT_FALSE(*it);
        }
    }

    std::remove(path.c_str());
}

// Compare two grids of the same type for background, topology, values and active
// states, so that a conversion done on read can be checked against a reference.
template <typename GridT>
void expectGridsMatch(const GridT& reference, const GridT& other, const std::string& context)
{
    using namespace openvdb;

    SCOPED_TRACE(context);

    EXPECT_EQ(reference.background(), other.background());
    EXPECT_TRUE(reference.constTree().hasSameTopology(other.constTree()));
    EXPECT_EQ(reference.activeVoxelCount(), other.activeVoxelCount());
    EXPECT_EQ(reference.constTree().activeTileCount(), other.constTree().activeTileCount());

    // Walk each grid against an accessor on the other so that a tile or voxel
    // present in only one of them is still compared.
    auto otherAccessor = other.getConstAccessor();
    for (typename GridT::ValueAllCIter it = reference.cbeginValueAll(); it; ++it) {
        const Coord ijk = it.getCoord();
        EXPECT_EQ(*it, otherAccessor.getValue(ijk));
        EXPECT_EQ(it.isValueOn(), otherAccessor.isValueOn(ijk));
    }
    auto referenceAccessor = reference.getConstAccessor();
    for (typename GridT::ValueAllCIter it = other.cbeginValueAll(); it; ++it) {
        const Coord ijk = it.getCoord();
        EXPECT_EQ(*it, referenceAccessor.getValue(ijk));
        EXPECT_EQ(it.isValueOn(), referenceAccessor.isValueOn(ijk));
    }
}

// A conversion readMode must produce the same grid as reading the grid in its
// original type and converting it in memory.  That in-memory conversion is the
// reference for the conversion codec (files with grid offsets) and for the cached
// conversion in File::resolveCachedGrid (files without).  This holds for Bool and
// Half, whose targets have genuine values distinct from their active state; Mask
// is the exception and is tested separately in testMaskConversionMatchesInMemory.
template <typename DstGridT, openvdb::io::ReadMode mode>
void testConversionMatchesInMemoryImpl(const std::string& label)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    const std::string gridName = "conversion_parity";

    // A non-zero background, an inactive non-zero tile, an active zero tile and a
    // region of active voxels.  Casting values and copying activity disagree on
    // the background and on both tiles, which is what makes this a useful case.
    // The two tiles sit in different internal nodes so that neither node holds
    // more than two distinct inactive values, keeping them exact under the
    // default active mask compression.
    FloatGrid::Ptr src = FloatGrid::create(/*background=*/1.0f);
    src->setName(gridName);
    src->tree().addTile(/*level=*/2, Coord(4096), 0.1f, /*active=*/false);
    src->tree().addTile(/*level=*/2, Coord(8192), 0.0f, /*active=*/true);
    src->fill(CoordBBox(Coord(0), Coord(6)), 5.0f, /*active=*/true);

    // io::File writes grid offsets, so a conversion readMode goes through the
    // conversion codec.  io::Stream writes none, so every grid is cached on open()
    // and converted in memory instead.
    const std::string offsetsPath = "test_conversion_parity_offsets_" + label + ".vdb";
    const std::string noOffsetsPath = "test_conversion_parity_no_offsets_" + label + ".vdb";
    {
        io::File f(offsetsPath);
        f.write(GridPtrVec{src});
    }
    {
        std::ofstream os(noOffsetsPath, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{src});
    }

    ReadOptions convertOptions;
    convertOptions.readMode = mode;

    for (const std::string& path : {offsetsPath, noOffsetsPath}) {
        FloatGrid::Ptr readOriginal;
        typename DstGridT::Ptr readConverted;
        {
            io::File f(path);
            f.open();
            readOriginal = gridPtrCast<FloatGrid>(f.readGrid(gridName, ReadOptions{}));
            readConverted = gridPtrCast<DstGridT>(f.readGrid(gridName, convertOptions));
            f.close();
        }
        ASSERT_TRUE(readOriginal);
        ASSERT_TRUE(readConverted);

        // Read as the original type, then convert in memory: the reference.
        const DstGridT reference(*readOriginal);

        // Pin the reference semantics for the bool-valued targets, so that a change
        // to the core grid conversion is caught here rather than silently redefining
        // what the read paths are compared against.  Values come from a cast, active
        // state from the source topology, and the two disagree on both tiles.
        if constexpr (std::is_same_v<typename DstGridT::ValueType, bool>) {
            auto accessor = reference.getConstAccessor();
            EXPECT_TRUE(reference.background());                    // bool(1.0f)
            EXPECT_TRUE(accessor.getValue(Coord(4096)));            // bool(0.1f)
            EXPECT_FALSE(accessor.isValueOn(Coord(4096)));
            EXPECT_FALSE(accessor.getValue(Coord(8192)));           // bool(0.0f)
            EXPECT_TRUE(accessor.isValueOn(Coord(8192)));
            EXPECT_TRUE(accessor.getValue(Coord(0)));               // bool(5.0f)
            EXPECT_TRUE(accessor.isValueOn(Coord(0)));
        }

        expectGridsMatch(reference, *readConverted, path);
    }

    std::remove(offsetsPath.c_str());
    std::remove(noOffsetsPath.c_str());
}

// ReadMode::Mask does not match MaskGrid(readOriginal): a mask records active
// state rather than a cast value, so the reference here is a topology copy of
// the source rather than the in-memory conversion used for Bool and Half.
TEST_F(TestCodec, testMaskConversionMatchesInMemory)
{
    using namespace openvdb;
    using namespace openvdb::io;

    CodecRegistry::clear();
    io::internal::initialize();

    const std::string gridName = "conversion_parity";

    // Same source grid as the generic parity test: a non-zero background, an
    // inactive non-zero tile and an active zero tile disagree between a value
    // cast and a topology copy, which is the case worth pinning here.
    FloatGrid::Ptr src = FloatGrid::create(/*background=*/1.0f);
    src->setName(gridName);
    src->tree().addTile(/*level=*/2, Coord(4096), 0.1f, /*active=*/false);
    src->tree().addTile(/*level=*/2, Coord(8192), 0.0f, /*active=*/true);
    src->fill(CoordBBox(Coord(0), Coord(6)), 5.0f, /*active=*/true);

    const std::string offsetsPath = "test_conversion_parity_offsets_mask.vdb";
    const std::string noOffsetsPath = "test_conversion_parity_no_offsets_mask.vdb";
    {
        io::File f(offsetsPath);
        f.write(GridPtrVec{src});
    }
    {
        std::ofstream os(noOffsetsPath, std::ios_base::out | std::ios_base::binary);
        io::Stream(os).write(GridPtrVec{src});
    }

    ReadOptions maskOptions;
    maskOptions.readMode = ReadMode::Mask;

    for (const std::string& path : {offsetsPath, noOffsetsPath}) {
        FloatGrid::Ptr readOriginal;
        MaskGrid::Ptr readConverted;
        {
            io::File f(path);
            f.open();
            readOriginal = gridPtrCast<FloatGrid>(f.readGrid(gridName, ReadOptions{}));
            readConverted = gridPtrCast<MaskGrid>(f.readGrid(gridName, maskOptions));
            f.close();
        }
        ASSERT_TRUE(readOriginal);
        ASSERT_TRUE(readConverted);

        // The reference is a topology copy of the source, not MaskGrid(readOriginal):
        // value equals active state everywhere, background is false, and inactive
        // tiles are preserved rather than dropped.
        MaskGrid::Ptr reference = MaskGrid::create(static_cast<const GridBase&>(*readOriginal));
        reference->setTree(MaskGrid::TreeType::Ptr(
            new MaskGrid::TreeType(readOriginal->constTree(),
                /*inactiveValue=*/false, /*activeValue=*/true, TopologyCopy())));

        auto accessor = reference->getConstAccessor();
        EXPECT_FALSE(reference->background());
        EXPECT_FALSE(accessor.getValue(Coord(4096)));
        EXPECT_FALSE(accessor.isValueOn(Coord(4096)));
        EXPECT_TRUE(accessor.getValue(Coord(8192)));
        EXPECT_TRUE(accessor.isValueOn(Coord(8192)));
        EXPECT_TRUE(accessor.getValue(Coord(0)));
        EXPECT_TRUE(accessor.isValueOn(Coord(0)));
        EXPECT_EQ(reference->activeVoxelCount(), readOriginal->activeVoxelCount());
        EXPECT_EQ(reference->constTree().activeTileCount(), readOriginal->constTree().activeTileCount());

        // Value equals active state at every position for a mask, which is what
        // the topology-copy reference buys over a plain value cast.
        for (auto it = reference->cbeginValueAll(); it; ++it) {
            EXPECT_EQ(*it, it.isValueOn());
        }

        expectGridsMatch(*reference, *readConverted, path);
    }

    std::remove(offsetsPath.c_str());
    std::remove(noOffsetsPath.c_str());
}

TEST_F(TestCodec, testBoolConversionMatchesInMemory)
{
    testConversionMatchesInMemoryImpl<openvdb::BoolGrid, openvdb::io::ReadMode::Bool>("bool");
}

TEST_F(TestCodec, testHalfConversionMatchesInMemory)
{
    testConversionMatchesInMemoryImpl<openvdb::HalfGrid, openvdb::io::ReadMode::Half>("half");
}
