// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include <openvdb_ax/ax.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/AttributeArray.h>

#include <openvdb/math/Transform.h>
#include <openvdb/openvdb.h>

#include <cppunit/extensions/HelperMacros.h>

#include <limits>

using namespace openvdb::points;

class TestWorldSpaceAccessors: public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestWorldSpaceAccessors);
    CPPUNIT_TEST(testWorldSpaceAssign);
    CPPUNIT_TEST(testWorldSpaceAssignComponent);
    CPPUNIT_TEST(testWorldSpaceAssignBound);

    CPPUNIT_TEST_SUITE_END();

    void testWorldSpaceAssign();
    void testWorldSpaceAssignComponent();
    void testWorldSpaceAssignBound();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestWorldSpaceAccessors);

void
TestWorldSpaceAccessors::testWorldSpaceAssign()
{
    std::vector<openvdb::Vec3d> positions =
        {openvdb::Vec3d(0.0, 0.0, 0.0),
         openvdb::Vec3d(0.0, 0.0, 0.05),
         openvdb::Vec3d(0.0, 1.0, 0.0),
         openvdb::Vec3d(1.0, 1.0, 0.0)};

    CPPUNIT_ASSERT(mHarness.mInputPointGrids.size() > 0);
    PointDataGrid::Ptr grid = mHarness.mInputPointGrids.back();

    openvdb::points::PointDataTree* tree = &(grid->tree());

    // @note  snippet moves all points to a single leaf node
    CPPUNIT_ASSERT_EQUAL(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    const std::string code = unittest_util::loadText("test/snippets/worldspace/worldSpaceAssign");
    CPPUNIT_ASSERT_NO_THROW(openvdb::ax::run(code.c_str(), *grid));

    // Tree is modified if points are moved
    tree = &(grid->tree());
    CPPUNIT_ASSERT_EQUAL(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    // test that P_original has the world-space value of the P attribute prior to running this snippet.
    // test that P_new has the expected world-space P value

    PointDataTree::LeafCIter leaf = tree->cbeginLeaf();
    const openvdb::math::Transform& transform = grid->transform();
    for (; leaf; ++leaf)
    {
        CPPUNIT_ASSERT(leaf->pointCount() == 4);

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

                CPPUNIT_ASSERT_EQUAL(oldPosition.x(), pOriginal.x());
                CPPUNIT_ASSERT_EQUAL(oldPosition.y(), pOriginal.y());
                CPPUNIT_ASSERT_EQUAL(oldPosition.z(), pOriginal.z());

                // test that the value for P_new, which should be the world space value of the points
                const openvdb::Vec3f newPosition = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f& pNew = pNewHandle->get(idx);

                CPPUNIT_ASSERT_EQUAL(newPosition.x(), pNew.x());
                CPPUNIT_ASSERT_EQUAL(newPosition.y(), pNew.y());
                CPPUNIT_ASSERT_EQUAL(newPosition.z(), pNew.z());

                // test that the value for P, which should be the updated voxel space value of the points
                const openvdb::Vec3f voxelSpacePosition = openvdb::Vec3f(0.2f, 0.3f, 0.4f);
                const openvdb::Vec3f& pVoxelSpace = pHandle->get(idx);
                // @todo: look at improving precision
                CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.x(), pVoxelSpace.x(), 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.y(), pVoxelSpace.y(), 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.z(), pVoxelSpace.z(), 1e-5);

                // test that the value for P, which should be the updated world space value of the points
                const openvdb::Vec3f positionWS = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f pWS = transform.indexToWorld(coord.asVec3d() + pHandle->get(idx));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.x(), pWS.x(), std::numeric_limits<float>::epsilon());
                CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.y(), pWS.y(), std::numeric_limits<float>::epsilon());
                CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.z(), pWS.z(), std::numeric_limits<float>::epsilon());
            }
        }
    }
}


void
TestWorldSpaceAccessors::testWorldSpaceAssignComponent()
{
    std::vector<openvdb::Vec3d> positions =
        {openvdb::Vec3d(0.0, 0.0, 0.0),
         openvdb::Vec3d(0.0, 0.0, 0.05),
         openvdb::Vec3d(0.0, 1.0, 0.0),
         openvdb::Vec3d(1.0, 1.0, 0.0)};

    CPPUNIT_ASSERT(mHarness.mInputPointGrids.size() > 0);
    PointDataGrid::Ptr grid = mHarness.mInputPointGrids.back();

    openvdb::points::PointDataTree& tree = grid->tree();

    const openvdb::Index64 originalCount = pointCount(tree);
    CPPUNIT_ASSERT(originalCount > 0);

    const std::string code = unittest_util::loadText("test/snippets/worldspace/worldSpaceAssignComponent");
    CPPUNIT_ASSERT_NO_THROW(openvdb::ax::run(code.c_str(), *grid));

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

                // CPPUNIT_ASSERT_EQUAL(oldPosition, pOriginal.x());

                // test that the value for P_new, which should be the world space value of the points
                const float newX = 5.22f;
                const float pNewX = pNewHandle->get(idx);

                CPPUNIT_ASSERT_EQUAL(newX, pNewX);

                // test that the value for P, which should be the updated voxel space value of the points
                const float voxelSpacePosition = 0.2f;
                const openvdb::Vec3f& pVoxelSpace = pHandle->get(idx);
                // @todo: look at improving precision
                CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition, pVoxelSpace.x(), 1e-5);
                //@todo: requiring point order, check the y and z components are unchanged
                // CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.y(), pVoxelSpace.y(), 1e-6);
                // CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.z(), pVoxelSpace.z(), 1e-6);

                // test that the value for P, which should be the updated world space value of the points
                const float positionWSX = 5.22f;
                const openvdb::Vec3f pWS = transform.indexToWorld(coord.asVec3d() + pHandle->get(idx));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWSX, pWS.x(), std::numeric_limits<float>::epsilon());
                //@todo: requiring point order, check the y and z components are unchanged
                // CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.y(), pWS.y(), std::numeric_limits<float>::epsilon());
                // CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.z(), pWS.z(), std::numeric_limits<float>::epsilon());
            }
        }
    }
}

void
TestWorldSpaceAccessors::testWorldSpaceAssignBound()
{
    std::vector<openvdb::Vec3d> positions =
        {openvdb::Vec3d(0.0, 0.0, 0.0),
         openvdb::Vec3d(0.0, 0.0, 0.05),
         openvdb::Vec3d(0.0, 1.0, 0.0),
         openvdb::Vec3d(1.0, 1.0, 0.0)};

    CPPUNIT_ASSERT(mHarness.mInputPointGrids.size() > 0);
    PointDataGrid::Ptr grid = mHarness.mInputPointGrids.back();

    openvdb::points::PointDataTree* tree = &(grid->tree());

    // @note  snippet moves all points to a single leaf node
    CPPUNIT_ASSERT_EQUAL(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    const std::string code = unittest_util::loadText("test/snippets/worldspace/worldSpaceAssignBound");
    CPPUNIT_ASSERT_NO_THROW(openvdb::ax::run(code.c_str(), *grid, {{"pos","P"}}));

    // Tree is modified if points are moved
    tree = &(grid->tree());
    CPPUNIT_ASSERT_EQUAL(openvdb::points::pointCount(*tree), openvdb::Index64(4));

    // test that P_original has the world-space value of the P attribute prior to running this snippet.
    // test that P_new has the expected world-space P value

    PointDataTree::LeafCIter leaf = tree->cbeginLeaf();
    const openvdb::math::Transform& transform = grid->transform();
    for (; leaf; ++leaf)
    {
        CPPUNIT_ASSERT(leaf->pointCount() == 4);

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

                CPPUNIT_ASSERT_EQUAL(oldPosition.x(), pOriginal.x());
                CPPUNIT_ASSERT_EQUAL(oldPosition.y(), pOriginal.y());
                CPPUNIT_ASSERT_EQUAL(oldPosition.z(), pOriginal.z());

                // test that the value for P_new, which should be the world space value of the points
                const openvdb::Vec3f newPosition = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f& pNew = pNewHandle->get(idx);

                CPPUNIT_ASSERT_EQUAL(newPosition.x(), pNew.x());
                CPPUNIT_ASSERT_EQUAL(newPosition.y(), pNew.y());
                CPPUNIT_ASSERT_EQUAL(newPosition.z(), pNew.z());

                // test that the value for P, which should be the updated voxel space value of the points
                const openvdb::Vec3f voxelSpacePosition = openvdb::Vec3f(0.2f, 0.3f, 0.4f);
                const openvdb::Vec3f& pVoxelSpace = pHandle->get(idx);
                // @todo: look at improving precision
                CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.x(), pVoxelSpace.x(), 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.y(), pVoxelSpace.y(), 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(voxelSpacePosition.z(), pVoxelSpace.z(), 1e-5);

                // test that the value for P, which should be the updated world space value of the points
                const openvdb::Vec3f positionWS = openvdb::Vec3f(2.22f, 3.33f, 4.44f);
                const openvdb::Vec3f pWS = transform.indexToWorld(coord.asVec3d() + pHandle->get(idx));
                CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.x(), pWS.x(), std::numeric_limits<float>::epsilon());
                CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.y(), pWS.y(), std::numeric_limits<float>::epsilon());
                CPPUNIT_ASSERT_DOUBLES_EQUAL(positionWS.z(), pWS.z(), std::numeric_limits<float>::epsilon());
            }
        }
    }
}



