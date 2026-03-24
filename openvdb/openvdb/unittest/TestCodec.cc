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

TEST_F(TestCodec, testBoolCodecIO) { testCodecIOImpl<openvdb::BoolGrid>("bool_grid", false, true); }
