// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"

#include <openvdb_ax/ax.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/AttributeArray.h>

#include <openvdb/math/Transform.h>
#include <openvdb/openvdb.h>

#include <limits>

using namespace openvdb::points;

class TestWorldSpaceAccessors: public unittest_util::AXTestCase
{
};


TEST_F(TestWorldSpaceAccessors, testWorldSpaceAssign)
{
    std::vector<openvdb::Vec3d> positions =
        {openvdb::Vec3d(0.0, 0.0, 0.0),
         openvdb::Vec3d(0.0, 0.0, 0.05),
         openvdb::Vec3d(0.0, 1.0, 0.0),
         openvdb::Vec3d(1.0, 1.0, 0.0)};

    ASSERT_TRUE(mHarness.mInputPointGrids.size() > 0);
    PointDataGrid::Ptr grid = mHarness.mInputPointGrids.back();

    openvdb::points::PointDataTree* tree = &(grid->tree());

    // @note  snippet moves all points to a single leaf node
    ASSERT_EQ(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    const std::string code = unittest_util::loadText("test/snippets/worldspace/worldSpaceAssign");
    ASSERT_NO_THROW(openvdb::ax::run(code.c_str(), *grid));

    // Tree is modified if points are moved
    tree = &(grid->tree());
    ASSERT_EQ(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    // test that P_original has the world-space value of the P attribute prior to running this snippet.
    // test that P_new has the expected world-space P value

    PointDataTree::LeafCIter leaf = tree->cbeginLeaf();
    const openvdb::math::Transform& transform = grid->transform();
    for (; leaf; ++leaf)
    {
        ASSERT_TRUE(leaf->pointCount() == 4);

        AttributeHandle<openvdb::Vec3f>::Ptr pOriginalHandle = AttributeHandle<openvdb::Vec3f>::create(leaf->attributeArray("P_original"));
        AttributeHandle<openvdb::Vec3f>::Ptr pNewHandle = AttributeHandle<openvdb::Vec3f>::create(leaf->attributeArray("P_new"));
        AttributeHandle<openvdb::Vec3f>::Ptr pHandle = AttributeHandle<openvdb::Vec3f>::create(leaf->attributeArray("P"));

        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel) {
            const openvdb::Coord& coord = voxel.getCoord();
            auto iter = leaf->beginIndexVoxel(coord);
            for (; iter; ++iter) {

                const openvdb::Index idx = *iter;

                // test that the value for P_original
                const openvdb::Vec3f& oldPosition = positions[idx];
                const openvdb::Vec3f& pOriginal = pOriginalHandle->get(idx);

                ASSERT_EQ(oldPosition.x(), pOriginal.x());
                ASSERT_EQ(oldPosition.y(), pOriginal.y());
                ASSERT_EQ(oldPosition.z(), pOriginal.z());

                // test that the value for P_new, which should be the world space value of the points
                const openvdb::Vec3f newPosition = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f& pNew = pNewHandle->get(idx);

                ASSERT_EQ(newPosition.x(), pNew.x());
                ASSERT_EQ(newPosition.y(), pNew.y());
                ASSERT_EQ(newPosition.z(), pNew.z());

                // test that the value for P, which should be the updated voxel space value of the points
                const openvdb::Vec3f voxelSpacePosition = openvdb::Vec3f(0.2f, 0.3f, 0.4f);
                const openvdb::Vec3f& pVoxelSpace = pHandle->get(idx);
                // @todo: look at improving precision
                ASSERT_NEAR(voxelSpacePosition.x(), pVoxelSpace.x(), 1e-5);
                ASSERT_NEAR(voxelSpacePosition.y(), pVoxelSpace.y(), 1e-5);
                ASSERT_NEAR(voxelSpacePosition.z(), pVoxelSpace.z(), 1e-5);

                // test that the value for P, which should be the updated world space value of the points
                const openvdb::Vec3f positionWS = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f pWS = transform.indexToWorld(coord.asVec3d() + pHandle->get(idx));
                ASSERT_NEAR(positionWS.x(), pWS.x(), std::numeric_limits<float>::epsilon());
                ASSERT_NEAR(positionWS.y(), pWS.y(), std::numeric_limits<float>::epsilon());
                ASSERT_NEAR(positionWS.z(), pWS.z(), std::numeric_limits<float>::epsilon());
            }
        }
    }
}


TEST_F(TestWorldSpaceAccessors, testWorldSpaceAssignComponent)
{
    std::vector<openvdb::Vec3d> positions =
        {openvdb::Vec3d(0.0, 0.0, 0.0),
         openvdb::Vec3d(0.0, 0.0, 0.05),
         openvdb::Vec3d(0.0, 1.0, 0.0),
         openvdb::Vec3d(1.0, 1.0, 0.0)};

    ASSERT_TRUE(mHarness.mInputPointGrids.size() > 0);
    PointDataGrid::Ptr grid = mHarness.mInputPointGrids.back();

    openvdb::points::PointDataTree& tree = grid->tree();

    const openvdb::Index64 originalCount = pointCount(tree);
    ASSERT_TRUE(originalCount > 0);

    const std::string code = unittest_util::loadText("test/snippets/worldspace/worldSpaceAssignComponent");
    ASSERT_NO_THROW(openvdb::ax::run(code.c_str(), *grid));

    // test that P_original has the world-space value of the P attribute prior to running this snippet.
    // test that P_new has the expected world-space P value

    PointDataTree::LeafCIter leaf = grid->tree().cbeginLeaf();
    const openvdb::math::Transform& transform = grid->transform();
    for (; leaf; ++leaf)
    {
        AttributeHandle<float>::Ptr pXOriginalHandle = AttributeHandle<float>::create(leaf->attributeArray("Px_original"));
        AttributeHandle<float>::Ptr pNewHandle = AttributeHandle<float>::create(leaf->attributeArray("Px_new"));
        AttributeHandle<openvdb::Vec3f>::Ptr pHandle = AttributeHandle<openvdb::Vec3f>::create(leaf->attributeArray("P"));

        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel) {
            const openvdb::Coord& coord = voxel.getCoord();
            auto iter = leaf->beginIndexVoxel(coord);
            for (; iter; ++iter) {
                const openvdb::Index idx = *iter;

                //@todo: requiring the point order, we should check the values of the px_original
                // test that the value for P_original
                // const float oldPosition = positions[idx].x();
                // const float pXOriginal = pXOriginalHandle->get(idx);

                // ASSERT_EQ(oldPosition, pOriginal.x());

                // test that the value for P_new, which should be the world space value of the points
                const float newX = 5.22f;
                const float pNewX = pNewHandle->get(idx);

                ASSERT_EQ(newX, pNewX);

                // test that the value for P, which should be the updated voxel space value of the points
                const float voxelSpacePosition = 0.2f;
                const openvdb::Vec3f& pVoxelSpace = pHandle->get(idx);
                // @todo: look at improving precision
                ASSERT_NEAR(voxelSpacePosition, pVoxelSpace.x(), 1e-5);
                //@todo: requiring point order, check the y and z components are unchanged
                // ASSERT_NEAR(voxelSpacePosition.y(), pVoxelSpace.y(), 1e-6);
                // ASSERT_NEAR(voxelSpacePosition.z(), pVoxelSpace.z(), 1e-6);

                // test that the value for P, which should be the updated world space value of the points
                const float positionWSX = 5.22f;
                const openvdb::Vec3f pWS = transform.indexToWorld(coord.asVec3d() + pHandle->get(idx));
                ASSERT_NEAR(positionWSX, pWS.x(), std::numeric_limits<float>::epsilon());
                //@todo: requiring point order, check the y and z components are unchanged
                // ASSERT_NEAR(positionWS.y(), pWS.y(), std::numeric_limits<float>::epsilon());
                // ASSERT_NEAR(positionWS.z(), pWS.z(), std::numeric_limits<float>::epsilon());
            }
        }
    }
}

TEST_F(TestWorldSpaceAccessors, testWorldSpaceAssignBound)
{
    std::vector<openvdb::Vec3d> positions =
        {openvdb::Vec3d(0.0, 0.0, 0.0),
         openvdb::Vec3d(0.0, 0.0, 0.05),
         openvdb::Vec3d(0.0, 1.0, 0.0),
         openvdb::Vec3d(1.0, 1.0, 0.0)};

    ASSERT_TRUE(mHarness.mInputPointGrids.size() > 0);
    PointDataGrid::Ptr grid = mHarness.mInputPointGrids.back();

    openvdb::points::PointDataTree* tree = &(grid->tree());

    // @note  snippet moves all points to a single leaf node
    ASSERT_EQ(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    const std::string code = unittest_util::loadText("test/snippets/worldspace/worldSpaceAssignBound");
    ASSERT_NO_THROW(openvdb::ax::run(code.c_str(), *grid, {{"pos","P"}}));

    // Tree is modified if points are moved
    tree = &(grid->tree());
    ASSERT_EQ(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    // test that P_original has the world-space value of the P attribute prior to running this snippet.
    // test that P_new has the expected world-space P value

    PointDataTree::LeafCIter leaf = tree->cbeginLeaf();
    const openvdb::math::Transform& transform = grid->transform();
    for (; leaf; ++leaf)
    {
        ASSERT_TRUE(leaf->pointCount() == 4);

        AttributeHandle<openvdb::Vec3f>::Ptr pOriginalHandle = AttributeHandle<openvdb::Vec3f>::create(leaf->attributeArray("P_original"));
        AttributeHandle<openvdb::Vec3f>::Ptr pNewHandle = AttributeHandle<openvdb::Vec3f>::create(leaf->attributeArray("P_new"));
        AttributeHandle<openvdb::Vec3f>::Ptr pHandle = AttributeHandle<openvdb::Vec3f>::create(leaf->attributeArray("P"));

        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel) {
            const openvdb::Coord& coord = voxel.getCoord();
            auto iter = leaf->beginIndexVoxel(coord);
            for (; iter; ++iter) {

                const openvdb::Index idx = *iter;

                // test that the value for P_original
                const openvdb::Vec3f& oldPosition = positions[idx];
                const openvdb::Vec3f& pOriginal = pOriginalHandle->get(idx);

                ASSERT_EQ(oldPosition.x(), pOriginal.x());
                ASSERT_EQ(oldPosition.y(), pOriginal.y());
                ASSERT_EQ(oldPosition.z(), pOriginal.z());

                // test that the value for P_new, which should be the world space value of the points
                const openvdb::Vec3f newPosition = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f& pNew = pNewHandle->get(idx);

                ASSERT_EQ(newPosition.x(), pNew.x());
                ASSERT_EQ(newPosition.y(), pNew.y());
                ASSERT_EQ(newPosition.z(), pNew.z());

                // test that the value for P, which should be the updated voxel space value of the points
                const openvdb::Vec3f voxelSpacePosition = openvdb::Vec3f(0.2f, 0.3f, 0.4f);
                const openvdb::Vec3f& pVoxelSpace = pHandle->get(idx);
                // @todo: look at improving precision
                ASSERT_NEAR(voxelSpacePosition.x(), pVoxelSpace.x(), 1e-5);
                ASSERT_NEAR(voxelSpacePosition.y(), pVoxelSpace.y(), 1e-5);
                ASSERT_NEAR(voxelSpacePosition.z(), pVoxelSpace.z(), 1e-5);

                // test that the value for P, which should be the updated world space value of the points
                const openvdb::Vec3f positionWS = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f pWS = transform.indexToWorld(coord.asVec3d() + pHandle->get(idx));
                ASSERT_NEAR(positionWS.x(), pWS.x(), std::numeric_limits<float>::epsilon());
                ASSERT_NEAR(positionWS.y(), pWS.y(), std::numeric_limits<float>::epsilon());
                ASSERT_NEAR(positionWS.z(), pWS.z(), std::numeric_limits<float>::epsilon());
            }
        }
    }
}



