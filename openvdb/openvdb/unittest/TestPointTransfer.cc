// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/points/PointTransfer.h>
#include <openvdb/util/Assert.h>

#include <gtest/gtest.h>

using namespace openvdb;

class TestPointTransfer: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointTransfer

struct DefaultTransfer
{
    inline bool startPointLeaf(const points::PointDataTree::LeafNodeType&) { return true; }
    inline bool endPointLeaf(const points::PointDataTree::LeafNodeType&) { return true; }
    inline bool finalize(const Coord&, size_t) { return true; }
};

TEST_F(TestPointTransfer, testRasterizeWithFilter)
{
    struct CountPointsTransferScheme
        : public DefaultTransfer
        , public points::VolumeTransfer<Int32Tree>
        , public points::FilteredTransfer<points::GroupFilter>
    {
        using FilterT = points::FilteredTransfer<points::GroupFilter>;
        using FilterT::startPointLeaf;

        CountPointsTransferScheme(Int32Tree& tree, const points::GroupFilter& filter)
            : points::VolumeTransfer<Int32Tree>(&tree)
            , points::FilteredTransfer<points::GroupFilter>(filter) {}

        inline void initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
        {
            points::VolumeTransfer<Int32Tree>::initialize(origin, idx, bounds);
            points::FilteredTransfer<points::GroupFilter>::initialize(origin, idx, bounds);
        }

        inline Int32 range(const Coord&, size_t) const { return 0; }

        inline void rasterizePoint(const Coord& ijk,
                        const Index id,
                        const CoordBBox&)
        {
            if (!this->FilterT::filter(id)) return;
            const Index offset = points::PointDataTree::LeafNodeType::coordToOffset(ijk);
            auto* const data = this->template buffer<0>();
            data[offset] += 1;
        }
    };

    std::vector<Vec3f> positions;
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));

    const double voxelSize = 0.1;
    math::Transform::Ptr transform =
        math::Transform::createLinearTransform(voxelSize);

    points::PointDataGrid::Ptr points =
        points::createPointDataGrid<points::NullCodec,
            points::PointDataGrid>(positions, *transform);

    points::PointDataTree& tree = points->tree();
    points::appendGroup(tree, "test");

    auto groupHandle = tree.beginLeaf()->groupWriteHandle("test");
    groupHandle.set(0, true);
    groupHandle.set(1, false);
    groupHandle.set(2, true);
    groupHandle.set(3, false);

    Int32Tree::Ptr intTree(new Int32Tree);
    intTree->setValueOn(Coord(0,0,0));

    points::GroupFilter filter("test", tree.cbeginLeaf()->attributeSet());
    CountPointsTransferScheme transfer(*intTree, filter);

    points::rasterize(*points, transfer);
    const int count = intTree->getValue(Coord(0,0,0));
    EXPECT_EQ(2, count);
}

TEST_F(TestPointTransfer, testRasterizeWithInitializeAndFinalize)
{
    struct LinearFunctionPointCountTransferScheme
        : public DefaultTransfer
        , public points::VolumeTransfer<Int32Tree>
    {
        LinearFunctionPointCountTransferScheme(Int32Tree& tree) :
            points::VolumeTransfer<Int32Tree>(&tree) {}

        inline Int32 range(const Coord&, size_t) const { return 0; }

        inline void initialize(const Coord& c, size_t i, const CoordBBox& b)
        {
            this->points::VolumeTransfer<Int32Tree>::initialize(c,i,b);
            auto* const data = this->template buffer<0>();
            const auto& mask = *(this->template mask<0>());
            for (auto iter = mask.beginOn(); iter; ++iter) {
                data[iter.pos()] += 10;
            }
        }

        inline bool finalize(const Coord&, size_t)
        {
            auto* const data = this->template buffer<0>();
            const auto& mask = *(this->template mask<0>());
            for (auto iter = mask.beginOn(); iter; ++iter) {
                data[iter.pos()] *= 2;
            }
            return true;
        }

        inline void rasterizePoint(const Coord& ijk, const Index, const CoordBBox&)
        {
            const Index offset = points::PointDataTree::LeafNodeType::coordToOffset(ijk);
            auto* const data = this->template buffer<0>();
            data[offset] += 1;
        }
    };

    std::vector<Vec3f> positions;
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));

    const double voxelSize = 0.1;
    math::Transform::Ptr transform =
        math::Transform::createLinearTransform(voxelSize);

    points::PointDataGrid::Ptr points =
        points::createPointDataGrid<points::NullCodec,
            points::PointDataGrid>(positions, *transform);

    Int32Tree::Ptr intTree(new Int32Tree);
    intTree->setValueOn(Coord(0,0,0));

    LinearFunctionPointCountTransferScheme transfer(*intTree);
    points::rasterize(*points, transfer);

    const int value = intTree->getValue(Coord(0,0,0));
    EXPECT_EQ(/*(10+4)*2*/28, value);
}

TEST_F(TestPointTransfer, tetsSingleTreeRasterize)
{
    struct CountPoints27TransferScheme
        : public DefaultTransfer
        , public points::VolumeTransfer<Int32Tree>
    {
        CountPoints27TransferScheme(Int32Tree& tree)
            : points::VolumeTransfer<Int32Tree>(&tree) {}

        /// @brief  The maximum lookup range of this transfer scheme in voxels.
        inline Int32 range(const Coord&, size_t) const { return 1; }

        /// @brief  The point stamp function. Each point which contributes to the
        ///         current leaf that the thread has access to will call this function
        ///         exactly once.
        /// @param ijk     The current voxel containing the point being rasterized. May
        ///                not be inside the destination leaf nodes.
        /// @param id      The point index being rasterized
        /// @param bounds  The active bounds of the leaf node(s) being written to.
        void rasterizePoint(const Coord& ijk, const Index, const CoordBBox& bounds)
        {
            static const Index DIM = Int32Tree::LeafNodeType::DIM;
            static const Index LOG2DIM = Int32Tree::LeafNodeType::LOG2DIM;

            CoordBBox intersectBox(ijk.offsetBy(-1), ijk.offsetBy(1));
            intersectBox.intersect(bounds);
            if (intersectBox.empty()) return;
            auto* const data = this->template buffer<0>();
            const auto& mask = *(this->template mask<0>());

            // loop over voxels in this leaf which are affected by this point

            const Coord& a(intersectBox.min());
            const Coord& b(intersectBox.max());
            for (Coord c = a; c.x() <= b.x(); ++c.x()) {
                const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM);
                for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                    const Index j = ((c.y() & (DIM-1u)) << LOG2DIM);
                    for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                        OPENVDB_ASSERT(bounds.isInside(c));
                        const Index offset = i + j + /*k*/(c.z() & (DIM-1u));
                        if (!mask.isOn(offset)) continue;
                        data[offset] += 1;

                    }
                }
            }
        }
    };

    std::vector<Vec3f> positions;
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(0.1f, 0.1f, 0.1f));
    positions.emplace_back(Vec3f(0.2f, 0.2f, 0.2f));
    positions.emplace_back(Vec3f(-0.1f,-0.1f,-0.1f));

    const double voxelSize = 0.1;
    math::Transform::Ptr transform =
        math::Transform::createLinearTransform(voxelSize);

    points::PointDataGrid::Ptr points =
        points::createPointDataGrid<points::NullCodec,
            points::PointDataGrid>(positions, *transform);

    Int32Tree::Ptr intTree(new Int32Tree);
    intTree->setValueOn(Coord(0,0,0));
    intTree->setValueOn(Coord(1,1,1));
    intTree->setValueOn(Coord(-1,-1,-1));

    CountPoints27TransferScheme transfer(*intTree);
    points::rasterize(*points, transfer);

    int count = intTree->getValue(Coord(0,0,0));
    EXPECT_EQ(3, count);
    count = intTree->getValue(Coord(1,1,1));
    EXPECT_EQ(3, count);
    count = intTree->getValue(Coord(-1,-1,-1));
    EXPECT_EQ(2, count);
}

TEST_F(TestPointTransfer, testMultiTreeRasterize)
{
    /// @brief  An example multi tree transfer scheme which rasterizes a vector
    ///         attribute into an int grid using length, and an int attribute
    ///         into a vector grid. Based on a 27 lookup stencil.
    ///
    struct MultiTransferScheme
        : public DefaultTransfer
        , public points::VolumeTransfer<Int32Tree, Vec3DTree>
    {
        using BaseT = points::VolumeTransfer<Int32Tree, Vec3DTree>;

        MultiTransferScheme(const size_t vIdx,const size_t iIdx,
                            Int32Tree& t1, Vec3DTree& t2)
            : BaseT(t1, t2)
            , mVIdx(vIdx), mIIdx(iIdx)
            , mVHandle(), mIHandle() {}

        MultiTransferScheme(const MultiTransferScheme& other)
            : BaseT(other), mVIdx(other.mVIdx), mIIdx(other.mIIdx)
            , mVHandle(), mIHandle() {}

        inline Int32 range(const Coord&, size_t) const { return 1; }

        inline bool startPointLeaf(const points::PointDataTree::LeafNodeType& leaf)
        {
            mVHandle = points::AttributeHandle<Vec3d>::create(leaf.constAttributeArray(mVIdx));
            mIHandle = points::AttributeHandle<int>::create(leaf.constAttributeArray(mIIdx));
            return true;
        }

        void rasterizePoint(const Coord& ijk, const Index id, const CoordBBox& bounds)
        {
            static const Index DIM = Int32Tree::LeafNodeType::DIM;
            static const Index LOG2DIM = Int32Tree::LeafNodeType::LOG2DIM;

            CoordBBox intersectBox(ijk.offsetBy(-1), ijk.offsetBy(1));
            intersectBox.intersect(bounds);
            if (intersectBox.empty()) return;

            auto* const data1 = this->template buffer<0>();
            auto* const data2 = this->template buffer<1>();
            const auto& mask = *(this->template mask<0>());

            // loop over voxels in this leaf which are affected by this point
            const Vec3d& vec = mVHandle->get(id);
            const int integer = mIHandle->get(id);

            const Coord& a(intersectBox.min());
            const Coord& b(intersectBox.max());
            for (Coord c = a; c.x() <= b.x(); ++c.x()) {
                const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM);
                for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                    const Index j = ((c.y() & (DIM-1u)) << LOG2DIM);
                    for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                        OPENVDB_ASSERT(bounds.isInside(c));
                        const Index offset = i + j + /*k*/(c.z() & (DIM-1u));
                        if (!mask.isOn(offset)) continue;
                        data1[offset] += static_cast<int>(vec.length());
                        data2[offset] += Vec3d(integer);
                    }
                }
            }
        }

    private:
        const size_t mVIdx;
        const size_t mIIdx;
        points::AttributeHandle<Vec3d>::Ptr mVHandle;
        points::AttributeHandle<int>::Ptr mIHandle;
    };

    std::vector<Vec3f> positions;
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));

    const double voxelSize = 0.1;
    math::Transform::Ptr transform =
        math::Transform::createLinearTransform(voxelSize);

    points::PointDataGrid::Ptr points =
        points::createPointDataGrid<points::NullCodec,
            points::PointDataGrid>(positions, *transform);

    points::PointDataTree& tree = points->tree();

    const Vec3d vectorValue(444.0f,555.0f,666.0f);
    points::appendAttribute<Vec3d>(tree, "testVec3d", vectorValue);
    points::appendAttribute<int>(tree, "testInt", 7);

    Vec3DTree::Ptr vecTree(new Vec3DTree);
    Int32Tree::Ptr intTree(new Int32Tree);
    vecTree->topologyUnion(tree);
    intTree->topologyUnion(tree);

    const points::PointDataTree::LeafNodeType& leaf = *(tree.cbeginLeaf());
    const size_t vidx = leaf.attributeSet().descriptor().find("testVec3d");
    const size_t iidx = leaf.attributeSet().descriptor().find("testInt");

    MultiTransferScheme transfer(vidx, iidx, *intTree, *vecTree);
    points::rasterize(*points, transfer);

    const Vec3d vecResult = vecTree->getValue(Coord(0,0,0));
    const int intResult = intTree->getValue(Coord(0,0,0));

    EXPECT_EQ(7.0, vecResult[0]);
    EXPECT_EQ(7.0, vecResult[1]);
    EXPECT_EQ(7.0, vecResult[2]);
    const int expected = static_cast<int>(vectorValue.length());
    EXPECT_EQ(expected, intResult);
}

TEST_F(TestPointTransfer, testMultiTreeRasterizeWithMask)
{
    struct CountPointsMaskTransferScheme
        : public DefaultTransfer
        , public points::VolumeTransfer<BoolTree, Int32Tree>
    {
        using BaseT = points::VolumeTransfer<BoolTree, Int32Tree>;
        CountPointsMaskTransferScheme(BoolTree& t1, Int32Tree& t2)
            : BaseT(t1, t2) {}

        inline Int32 range(const Coord&, size_t) const { return 0; }
        inline void rasterizePoint(const Coord& ijk, const Index, const CoordBBox&)
        {
            auto* const data = this->template buffer<1>();
            const Index offset = points::PointDataTree::LeafNodeType::coordToOffset(ijk);
            data[offset] += 1;
        }
    };

    // Setup test data
    std::vector<Vec3f> positions;
    positions.emplace_back(Vec3f(0.0f, 0.0f, 0.0f));
    positions.emplace_back(Vec3f(1.0f, 1.0f, 1.0f));

    const double voxelSize = 0.1;
    math::Transform::Ptr transform =
        math::Transform::createLinearTransform(voxelSize);

    points::PointDataGrid::Ptr points =
        points::createPointDataGrid<points::NullCodec,
            points::PointDataGrid>(positions, *transform);

    Int32Tree::Ptr intTree(new Int32Tree);
    BoolTree::Ptr topology(new BoolTree);
    intTree->topologyUnion(points->tree());
    topology->setValueOn(Coord(0,0,0));

    EXPECT_TRUE(intTree->isValueOn(Coord(10,10,10)));

    CountPointsMaskTransferScheme transfer(*topology, *intTree);
    points::rasterize(*points, transfer);

    int count = intTree->getValue(Coord(0,0,0));
    EXPECT_EQ(1, count);
    count = intTree->getValue(Coord(10,10,10));
    EXPECT_EQ(0, count);
}
