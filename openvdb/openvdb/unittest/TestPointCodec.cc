// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/io/Codec.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <gtest/gtest.h>
#include "util.h" // for unittest_util::genPoints

namespace {

class PointList {
public:
    using PosType = openvdb::Vec3R;
    PointList(const std::vector<PosType>& points) : mPoints(&points) {}
    size_t size() const { return mPoints->size(); }
    void getPos(size_t n, PosType& xyz) const { xyz = (*mPoints)[n]; }
private:
    std::vector<PosType> const * const mPoints;
};

} // namespace

class TestPointCodec: public ::testing::Test
{
};

TEST_F(TestPointCodec, testPointIndexCodecIO)
{
    using namespace openvdb;
    using namespace openvdb::io;
    using PointIndexGrid = tools::PointIndexGrid;

    openvdb::initialize();
    CodecRegistry::clear();

    // Generate points on a unit sphere and build a PointIndexGrid
    std::vector<Vec3R> points;
    unittest_util::genPoints(100, points);
    PointList pointList(points);

    const double voxelSize = 0.1;
    math::Transform::Ptr transform = math::Transform::createLinearTransform(voxelSize);

    PointIndexGrid::Ptr srcGrid =
        tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);
    srcGrid->setName("point_index_grid");

    const std::string rawPath = "testPointIndexCodec_raw.vdb";

    // Phase 1: write/read without codec
    {
        io::File f(rawPath);
        f.write(GridPtrVec{srcGrid});
    }

    PointIndexGrid::Ptr rawGrid;
    {
        io::File f(rawPath);
        f.open();
        rawGrid = gridPtrCast<PointIndexGrid>(f.readGrid("point_index_grid"));
        f.close();
    }
    ASSERT_TRUE(rawGrid);

    PointIndexGrid::Ptr rawTopo;
    {
        io::File f(rawPath);
        f.open();
        GridBase::Ptr base;
        EXPECT_NO_THROW(base = f.readGrid("point_index_grid"));
        rawTopo = gridPtrCast<PointIndexGrid>(base);
        f.close();
    }
    ASSERT_TRUE(rawTopo);
    EXPECT_EQ(rawTopo->activeVoxelCount(), Index64(97));
    EXPECT_EQ(rawTopo->getName(), std::string("point_index_grid"));

    std::remove(rawPath.c_str());

    const std::string codecPath = "testPointIndexCodec_codec.vdb";

    // Phase 2: register codec, write/read with codec
    io::internal::initialize();
    ASSERT_TRUE(CodecRegistry::isRegistered(PointIndexGrid::gridType()));

    {
        io::File f(codecPath);
        f.write(GridPtrVec{srcGrid});
    }

    PointIndexGrid::Ptr codecGrid;
    {
        io::File f(codecPath);
        f.open();
        codecGrid = gridPtrCast<PointIndexGrid>(f.readGrid("point_index_grid"));
        f.close();
    }
    ASSERT_TRUE(codecGrid);

    // Phase 3: full read comparison
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(srcGrid->tree()));
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(codecGrid->tree()));
    {
        auto codecAcc = codecGrid->getConstAccessor();
        for (PointIndexGrid::ValueOnCIter it = srcGrid->cbeginValueOn(); it; ++it) {
            EXPECT_EQ(*it, codecAcc.getValue(it.getCoord()));
        }
    }

    // Compare leaf indices arrays
    {
        auto srcLeafIt = srcGrid->tree().cbeginLeaf();
        auto codecLeafIt = codecGrid->tree().cbeginLeaf();
        for (; srcLeafIt; ++srcLeafIt, ++codecLeafIt) {
            ASSERT_TRUE(codecLeafIt);
            EXPECT_EQ(srcLeafIt->indices().size(), codecLeafIt->indices().size());
            for (size_t i = 0; i < srcLeafIt->indices().size(); ++i) {
                EXPECT_EQ(srcLeafIt->indices()[i], codecLeafIt->indices()[i]);
            }
        }
        EXPECT_TRUE(!codecLeafIt);
    }

    // Phase 4: TopologyOnly read
    ReadOptions topoOpts;
    topoOpts.readMode = ReadMode::TopologyOnly;

    PointIndexGrid::Ptr codecTopo;
    {
        io::File f(codecPath);
        f.open();
        GridBase::Ptr base;
        EXPECT_NO_THROW(base = f.readGrid("point_index_grid", topoOpts));
        codecTopo = gridPtrCast<PointIndexGrid>(base);
        f.close();
    }
    ASSERT_TRUE(codecTopo);
    EXPECT_EQ(codecTopo->activeVoxelCount(), Index64(0));
    EXPECT_TRUE(codecTopo->tree().leafCount() == 0);
    EXPECT_EQ(codecTopo->getName(), std::string("point_index_grid"));

    // Cleanup
    CodecRegistry::clear();
    std::remove(codecPath.c_str());
}
