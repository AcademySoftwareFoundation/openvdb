///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
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

#include "TestHarness.h"

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

    CPPUNIT_TEST_SUITE_END();

    void testWorldSpaceAssign();
    void testWorldSpaceAssignComponent();
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
    CPPUNIT_ASSERT_NO_THROW(unittest_util::wrapExecution(*grid, "test/snippets/worldspace/worldSpaceAssign"));

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

    CPPUNIT_ASSERT_NO_THROW(unittest_util::wrapExecution(*grid, "test/snippets/worldspace/worldSpaceAssignComponent"));

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



// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )

