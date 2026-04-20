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
    const std::string codecPath = "testPointIndexCodec_codec.vdb";

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
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(rawGrid->tree()));
    EXPECT_TRUE(srcGrid->tree().hasSameTopology(codecGrid->tree()));
    {
        auto codecAcc = codecGrid->getConstAccessor();
        for (PointIndexGrid::ValueOnCIter it = rawGrid->cbeginValueOn(); it; ++it) {
            EXPECT_EQ(*it, codecAcc.getValue(it.getCoord()));
        }
    }

    // Compare leaf indices arrays
    {
        auto rawLeafIt = rawGrid->tree().cbeginLeaf();
        auto codecLeafIt = codecGrid->tree().cbeginLeaf();
        for (; rawLeafIt; ++rawLeafIt, ++codecLeafIt) {
            ASSERT_TRUE(codecLeafIt);
            EXPECT_EQ(rawLeafIt->indices().size(), codecLeafIt->indices().size());
            for (size_t i = 0; i < rawLeafIt->indices().size(); ++i) {
                EXPECT_EQ(rawLeafIt->indices()[i], codecLeafIt->indices()[i]);
            }
        }
        EXPECT_TRUE(!codecLeafIt);
    }

    // Phase 4: TopologyOnly read
    ReadOptions topoOpts;
    topoOpts.readMode = ReadMode::TopologyOnly;

    PointIndexGrid::Ptr rawTopo;
    {
        io::File f(rawPath);
        f.open();
        GridBase::Ptr base;
        EXPECT_NO_THROW(base = f.readGrid("point_index_grid", topoOpts));
        rawTopo = gridPtrCast<PointIndexGrid>(base);
        f.close();
    }
    ASSERT_TRUE(rawTopo);
    EXPECT_EQ(rawTopo->activeVoxelCount(), Index64(0));
    EXPECT_TRUE(rawTopo->tree().leafCount() == 0);
    EXPECT_EQ(rawTopo->getName(), std::string("point_index_grid"));

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
    std::remove(rawPath.c_str());
    std::remove(codecPath.c_str());
}

TEST_F(TestPointCodec, testPointDataCodecIO)
{
    using namespace openvdb;
    using namespace openvdb::io;
    using namespace openvdb::points;
    using PointDataTree = PointDataGrid::TreeType;

    openvdb::initialize();
    CodecRegistry::clear();

    // Helper: compare P attribute values leaf-by-leaf between two PointDataGrids
    auto comparePositions = [](const PointDataGrid& a, const PointDataGrid& b) {
        auto aIt = a.tree().cbeginLeaf();
        auto bIt = b.tree().cbeginLeaf();
        for (; aIt && bIt; ++aIt, ++bIt) {
            EXPECT_EQ(aIt->pointCount(), bIt->pointCount());
            AttributeHandle<Vec3f> aH(aIt->constAttributeArray("P"));
            AttributeHandle<Vec3f> bH(bIt->constAttributeArray("P"));
            for (Index i = 0; i < aIt->pointCount(); ++i) {
                const Vec3f av = aH.get(i);
                const Vec3f bv = bH.get(i);
                EXPECT_NEAR(av.x(), bv.x(), 1e-6f);
                EXPECT_NEAR(av.y(), bv.y(), 1e-6f);
                EXPECT_NEAR(av.z(), bv.z(), 1e-6f);
            }
        }
        EXPECT_TRUE(!aIt && !bIt);
    };

    // -----------------------------------------------------------------------
    // Section A: Positions only
    // -----------------------------------------------------------------------
    {
        const std::vector<Vec3f> positions = {
            Vec3f(0.0f, 1.0f, 0.0f),
            Vec3f(1.5f, 3.5f, 1.0f),
            Vec3f(-1.0f, 6.0f, -2.0f),
            Vec3f(1.1f, 1.25f, 0.06f)
        };

        const float voxelSize = 0.5f;
        math::Transform::Ptr transform = math::Transform::createLinearTransform(voxelSize);

        PointDataGrid::Ptr srcGrid =
            createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
        srcGrid->setName("pdg_positions");

        const std::string rawPath = "testPDG_A_raw.vdb";
        const std::string codecPath = "testPDG_A_codec.vdb";

        // Phase 1: write/read without codec
        {
            io::File f(rawPath);
            f.write(GridPtrVec{srcGrid});
        }

        PointDataGrid::Ptr rawGrid;
        {
            io::File f(rawPath);
            f.open();
            rawGrid = gridPtrCast<PointDataGrid>(f.readGrid("pdg_positions"));
            f.close();
        }
        ASSERT_TRUE(rawGrid);
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(rawGrid->tree()));

        // Phase 2: register codec, write/read with codec
        io::internal::initialize();
        ASSERT_TRUE(CodecRegistry::isRegistered(PointDataGrid::gridType()));

        {
            io::File f(codecPath);
            f.write(GridPtrVec{srcGrid});
        }

        PointDataGrid::Ptr codecGrid;
        {
            io::File f(codecPath);
            f.open();
            codecGrid = gridPtrCast<PointDataGrid>(f.readGrid("pdg_positions"));
            f.close();
        }
        ASSERT_TRUE(codecGrid);

        // Phase 3: compare raw vs codec
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(codecGrid->tree()));
        EXPECT_EQ(pointCount(rawGrid->tree()), pointCount(codecGrid->tree()));
        comparePositions(*rawGrid, *codecGrid);

        // Phase 4: TopologyOnly read
        ReadOptions topoOpts;
        topoOpts.readMode = ReadMode::TopologyOnly;

        PointDataGrid::Ptr rawTopo;
        {
            io::File f(rawPath);
            f.open();
            GridBase::Ptr base;
            EXPECT_NO_THROW(base = f.readGrid("pdg_positions", topoOpts));
            rawTopo = gridPtrCast<PointDataGrid>(base);
            f.close();
        }
        ASSERT_TRUE(rawTopo);
        EXPECT_EQ(rawTopo->activeVoxelCount(), Index64(0));
        EXPECT_TRUE(rawTopo->tree().leafCount() == 0);

        PointDataGrid::Ptr codecTopo;
        {
            io::File f(codecPath);
            f.open();
            GridBase::Ptr base;
            EXPECT_NO_THROW(base = f.readGrid("pdg_positions", topoOpts));
            codecTopo = gridPtrCast<PointDataGrid>(base);
            f.close();
        }
        ASSERT_TRUE(codecTopo);
        EXPECT_EQ(codecTopo->activeVoxelCount(), Index64(0));
        EXPECT_TRUE(codecTopo->tree().leafCount() == 0);

        CodecRegistry::clear();
        std::remove(rawPath.c_str());
        std::remove(codecPath.c_str());
    }

    // -----------------------------------------------------------------------
    // Section B: Multiple attributes
    // -----------------------------------------------------------------------
    {
        const std::vector<Vec3f> positions = {
            Vec3f(0.0f, 1.0f, 0.0f),
            Vec3f(1.5f, 3.5f, 1.0f),
            Vec3f(-1.0f, 6.0f, -2.0f),
            Vec3f(1.1f, 1.25f, 0.06f)
        };
        const std::vector<Vec3f> velocities = {
            Vec3f(1.0f, 0.0f, 0.0f),
            Vec3f(0.0f, 1.0f, 0.0f),
            Vec3f(0.0f, 0.0f, 1.0f),
            Vec3f(1.0f, 1.0f, 0.5f)
        };
        const std::vector<int> ids = {0, 1, 2, 3};

        const float voxelSize = 0.5f;
        math::Transform::Ptr transform = math::Transform::createLinearTransform(voxelSize);

        PointAttributeVector<Vec3f> posWrapper(positions);
        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(posWrapper, *transform);

        PointDataGrid::Ptr srcGrid =
            createPointDataGrid<NullCodec, PointDataGrid>(
                *pointIndexGrid, posWrapper, *transform);
        srcGrid->setName("pdg_multi");

        PointDataTree& tree = srcGrid->tree();
        tools::PointIndexTree& indexTree = pointIndexGrid->tree();

        appendAttribute<Vec3f>(tree, "velocity");
        populateAttribute<PointDataTree, tools::PointIndexTree,
            PointAttributeVector<Vec3f>>(
                tree, indexTree, "velocity",
                PointAttributeVector<Vec3f>(velocities));

        appendAttribute<int>(tree, "id");
        populateAttribute<PointDataTree, tools::PointIndexTree,
            PointAttributeVector<int>>(
                tree, indexTree, "id",
                PointAttributeVector<int>(ids));

        // Verify attribute count on src grid (P, velocity, id)
        {
            auto leafIt = srcGrid->tree().cbeginLeaf();
            ASSERT_TRUE(leafIt);
            EXPECT_EQ(leafIt->attributeSet().size(), size_t(3));
        }

        const std::string rawPath = "testPDG_B_raw.vdb";
        const std::string codecPath = "testPDG_B_codec.vdb";

        // Phase 1: write/read without codec
        {
            io::File f(rawPath);
            f.write(GridPtrVec{srcGrid});
        }

        PointDataGrid::Ptr rawGrid;
        {
            io::File f(rawPath);
            f.open();
            rawGrid = gridPtrCast<PointDataGrid>(f.readGrid("pdg_multi"));
            f.close();
        }
        ASSERT_TRUE(rawGrid);

        // Phase 2: register codec, write/read with codec
        io::internal::initialize();

        {
            io::File f(codecPath);
            f.write(GridPtrVec{srcGrid});
        }

        PointDataGrid::Ptr codecGrid;
        {
            io::File f(codecPath);
            f.open();
            codecGrid = gridPtrCast<PointDataGrid>(f.readGrid("pdg_multi"));
            f.close();
        }
        ASSERT_TRUE(codecGrid);

        EXPECT_TRUE(rawGrid->tree().hasSameTopology(codecGrid->tree()));
        EXPECT_EQ(pointCount(rawGrid->tree()), pointCount(codecGrid->tree()));

        // Verify attribute count on codec grid
        {
            auto leafIt = codecGrid->tree().cbeginLeaf();
            ASSERT_TRUE(leafIt);
            EXPECT_EQ(leafIt->attributeSet().size(), size_t(3));
        }

        // Compare all three attributes leaf-by-leaf
        {
            auto rawIt = rawGrid->tree().cbeginLeaf();
            auto codecIt = codecGrid->tree().cbeginLeaf();
            for (; rawIt && codecIt; ++rawIt, ++codecIt) {
                EXPECT_EQ(rawIt->pointCount(), codecIt->pointCount());
                AttributeHandle<Vec3f> rawP(rawIt->constAttributeArray("P"));
                AttributeHandle<Vec3f> codecP(codecIt->constAttributeArray("P"));
                AttributeHandle<Vec3f> rawVel(rawIt->constAttributeArray("velocity"));
                AttributeHandle<Vec3f> codecVel(codecIt->constAttributeArray("velocity"));
                AttributeHandle<int> rawId(rawIt->constAttributeArray("id"));
                AttributeHandle<int> codecId(codecIt->constAttributeArray("id"));
                for (Index i = 0; i < rawIt->pointCount(); ++i) {
                    const Vec3f rp = rawP.get(i);
                    const Vec3f cp = codecP.get(i);
                    EXPECT_NEAR(rp.x(), cp.x(), 1e-6f);
                    EXPECT_NEAR(rp.y(), cp.y(), 1e-6f);
                    EXPECT_NEAR(rp.z(), cp.z(), 1e-6f);
                    const Vec3f rv = rawVel.get(i);
                    const Vec3f cv = codecVel.get(i);
                    EXPECT_NEAR(rv.x(), cv.x(), 1e-6f);
                    EXPECT_NEAR(rv.y(), cv.y(), 1e-6f);
                    EXPECT_NEAR(rv.z(), cv.z(), 1e-6f);
                    EXPECT_EQ(rawId.get(i), codecId.get(i));
                }
            }
            EXPECT_TRUE(!rawIt && !codecIt);
        }

        CodecRegistry::clear();
        std::remove(rawPath.c_str());
        std::remove(codecPath.c_str());
    }

    // -----------------------------------------------------------------------
    // Section C: Shared vs non-shared descriptors
    // -----------------------------------------------------------------------
    {
        std::vector<Vec3R> pts;
        unittest_util::genPoints(100, pts);

        std::vector<Vec3f> positions;
        positions.reserve(pts.size());
        for (const auto& p : pts) {
            positions.emplace_back(float(p.x()), float(p.y()), float(p.z()));
        }

        const double voxelSize = 0.1;
        math::Transform::Ptr transform = math::Transform::createLinearTransform(voxelSize);

        PointDataGrid::Ptr srcGrid =
            createPointDataGrid<NullCodec, PointDataGrid>(positions, *transform);
        srcGrid->setName("pdg_desc");

        // All leaves should share one Descriptor::Ptr initially
        ASSERT_GT(srcGrid->tree().leafCount(), Index32(1));
        {
            auto leafIt = srcGrid->tree().cbeginLeaf();
            auto firstDescPtr = leafIt->attributeSet().descriptorPtr();
            for (; leafIt; ++leafIt) {
                EXPECT_EQ(leafIt->attributeSet().descriptorPtr(), firstDescPtr);
            }
        }

        io::internal::initialize();

        // -- C1: Shared descriptors (default) --
        // All leaves already share one pointer; this exercises the header=1 write path.
        const std::string sharedPath = "testPDG_C_shared.vdb";
        {
            io::File f(sharedPath);
            f.write(GridPtrVec{srcGrid});
        }

        PointDataGrid::Ptr sharedGrid;
        {
            io::File f(sharedPath);
            f.open();
            sharedGrid = gridPtrCast<PointDataGrid>(f.readGrid("pdg_desc"));
            f.close();
        }
        ASSERT_TRUE(sharedGrid);
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(sharedGrid->tree()));
        EXPECT_EQ(pointCount(srcGrid->tree()), pointCount(sharedGrid->tree()));
        comparePositions(*srcGrid, *sharedGrid);

        // -- C2: After makeDescriptorUnique --
        // makeDescriptorUnique() creates ONE new descriptor and assigns it to every
        // leaf, so all leaves still share a single pointer (the new copy).
        // The codec still detects matching descriptors and writes header=1;
        // this is a regression check that the round-trip remains correct.
        makeDescriptorUnique(srcGrid->tree());
        {
            auto leafIt = srcGrid->tree().cbeginLeaf();
            auto firstDescPtr = leafIt->attributeSet().descriptorPtr();
            ++leafIt;
            if (leafIt) {
                // All leaves share the same new pointer
                EXPECT_EQ(leafIt->attributeSet().descriptorPtr(), firstDescPtr);
            }
        }

        const std::string nonSharedPath = "testPDG_C_nonshared.vdb";
        {
            io::File f(nonSharedPath);
            f.write(GridPtrVec{srcGrid});
        }

        PointDataGrid::Ptr nonSharedGrid;
        {
            io::File f(nonSharedPath);
            f.open();
            nonSharedGrid = gridPtrCast<PointDataGrid>(f.readGrid("pdg_desc"));
            f.close();
        }
        ASSERT_TRUE(nonSharedGrid);
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(nonSharedGrid->tree()));
        EXPECT_EQ(pointCount(srcGrid->tree()), pointCount(nonSharedGrid->tree()));
        comparePositions(*sharedGrid, *nonSharedGrid);

        // -- C3: Genuinely different descriptors (exercises the header=0 write path) --
        // Add "extra" to all leaves, then drop it from only the first leaf so that
        // leaf descriptors differ by value, triggering matching=false in the codec.
        appendAttribute<float>(srcGrid->tree(), "extra");
        makeDescriptorUnique(srcGrid->tree());
        srcGrid->setName("pdg_desc_diff");

        {
            auto leafIt = srcGrid->tree().beginLeaf();
            ASSERT_TRUE(leafIt);
            const size_t extraIdx =
                leafIt->attributeSet().descriptor().find("extra");
            ASSERT_NE(extraIdx, AttributeSet::INVALID_POS);
            const std::vector<size_t> dropIndices = {extraIdx};
            AttributeSet::Descriptor::Ptr newDesc =
                leafIt->attributeSet().descriptor().duplicateDrop(dropIndices);
            leafIt->dropAttributes(
                dropIndices, leafIt->attributeSet().descriptor(), newDesc);
        }

        const std::string diffPath = "testPDG_C_diff.vdb";
        {
            io::File f(diffPath);
            f.write(GridPtrVec{srcGrid});
        }

        PointDataGrid::Ptr diffGrid;
        {
            io::File f(diffPath);
            f.open();
            diffGrid = gridPtrCast<PointDataGrid>(f.readGrid("pdg_desc_diff"));
            f.close();
        }
        ASSERT_TRUE(diffGrid);
        EXPECT_TRUE(srcGrid->tree().hasSameTopology(diffGrid->tree()));
        EXPECT_EQ(pointCount(srcGrid->tree()), pointCount(diffGrid->tree()));

        // First leaf has {P} only; remaining leaves have {P, extra}
        {
            auto diffIt = diffGrid->tree().cbeginLeaf();
            ASSERT_TRUE(diffIt);
            EXPECT_EQ(diffIt->attributeSet().size(), size_t(1));
            ++diffIt;
            if (diffIt) {
                EXPECT_EQ(diffIt->attributeSet().size(), size_t(2));
            }
        }

        std::remove(sharedPath.c_str());
        std::remove(nonSharedPath.c_str());
        std::remove(diffPath.c_str());
    }
}

