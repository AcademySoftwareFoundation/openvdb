// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/io/Codec.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <gtest/gtest.h>

namespace {

struct MockCodec : public openvdb::io::Codec
{
    static std::string name() { return "mock"; }

    openvdb::io::CodecData::Ptr createData() final { return nullptr; }
};

class TestCodec: public ::testing::Test
{
};

} // unnamed namespace


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

    // Clean up
    CodecRegistry::clear();
}

TEST_F(TestCodec, testBoolCodecIO)
{
    using namespace openvdb;
    using namespace openvdb::io;

    openvdb::initialize();
    CodecRegistry::clear();

    BoolGrid::Ptr srcGrid = BoolGrid::create(false);
    srcGrid->setName("bool_grid");
    srcGrid->fill(CoordBBox(Coord(-5), Coord(5)), true, true);

    const std::string rawPath = "testBoolCodec_raw.vdb";
    const std::string codecPath = "testBoolCodec_codec.vdb";

    // Phase 1: write/read without codec
    {
        io::File f(rawPath);
        f.write(GridPtrVec{srcGrid});
    }

    BoolGrid::Ptr rawGrid;
    {
        io::File f(rawPath);
        f.open();
        rawGrid = gridPtrCast<BoolGrid>(f.readGrid("bool_grid"));
        f.close();
    }
    ASSERT_TRUE(rawGrid);

    // Phase 2: register codec, write/read with codec
    io::internal::initialize();
    ASSERT_TRUE(CodecRegistry::isRegistered(BoolGrid::gridType()));

    {
        io::File f(codecPath);
        f.write(GridPtrVec{srcGrid});
    }

    BoolGrid::Ptr codecGrid;
    {
        io::File f(codecPath);
        f.open();
        codecGrid = gridPtrCast<BoolGrid>(f.readGrid("bool_grid"));
        f.close();
    }
    ASSERT_TRUE(codecGrid);

    // Phase 3: full read comparison
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(rawGrid->tree()));
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(codecGrid->tree()));
    {
        auto codecAcc = codecGrid->getConstAccessor();
        for (BoolGrid::ValueOnCIter it = rawGrid->cbeginValueOn(); it; ++it) {
            EXPECT_EQ(*it, codecAcc.getValue(it.getCoord()));
        }
    }

    // Phase 4: clip read
    const BBoxd clipBBox(Vec3d(0.0), Vec3d(3.5));

    BoolGrid::Ptr rawClipped;
    {
        io::File f(rawPath);
        f.open();
        rawClipped = gridPtrCast<BoolGrid>(f.readGrid("bool_grid", clipBBox));
        f.close();
    }
    ASSERT_TRUE(rawClipped);

    BoolGrid::Ptr codecClipped;
    {
        io::File f(codecPath);
        f.open();
        codecClipped = gridPtrCast<BoolGrid>(f.readGrid("bool_grid", clipBBox));
        f.close();
    }
    ASSERT_TRUE(codecClipped);

    EXPECT_TRUE(rawClipped->tree().hasSameTopology(codecClipped->tree()));

    {
        auto codecAcc = codecClipped->getConstAccessor();
        for (BoolGrid::ValueOnCIter it = rawClipped->cbeginValueOn(); it; ++it) {
            EXPECT_EQ(*it, codecAcc.getValue(it.getCoord()));
        }
    }

    for (BoolGrid::ValueOnCIter it = rawClipped->cbeginValueOn(); it; ++it) {
        EXPECT_TRUE(clipBBox.isInside(rawClipped->indexToWorld(it.getCoord())));
    }

    for (BoolGrid::ValueOnCIter it = codecClipped->cbeginValueOn(); it; ++it) {
        EXPECT_TRUE(clipBBox.isInside(codecClipped->indexToWorld(it.getCoord())));
    }

    // Phase 5: TopologyOnly read
    ReadOptions topoOpts;
    topoOpts.readMode = ReadMode::TopologyOnly;

    BoolGrid::Ptr rawTopo;
    {
        io::File f(rawPath);
        f.open();
        GridBase::Ptr base;
        EXPECT_NO_THROW(base = f.readGrid("bool_grid", topoOpts));
        rawTopo = gridPtrCast<BoolGrid>(base);
        f.close();
    }
    ASSERT_TRUE(rawTopo);
    EXPECT_EQ(rawTopo->activeVoxelCount(), Index64(0));
    EXPECT_TRUE(rawTopo->tree().leafCount() == 0);
    EXPECT_EQ(rawTopo->getName(), std::string("bool_grid"));

    BoolGrid::Ptr codecTopo;
    {
        io::File f(codecPath);
        f.open();
        GridBase::Ptr base;
        EXPECT_NO_THROW(base = f.readGrid("bool_grid", topoOpts));
        codecTopo = gridPtrCast<BoolGrid>(base);
        f.close();
    }
    ASSERT_TRUE(codecTopo);
    EXPECT_EQ(codecTopo->activeVoxelCount(), Index64(0));
    EXPECT_TRUE(codecTopo->tree().leafCount() == 0);
    EXPECT_EQ(codecTopo->getName(), std::string("bool_grid"));

    // Cleanup
    CodecRegistry::clear();
    std::remove(rawPath.c_str());
    std::remove(codecPath.c_str());
}
