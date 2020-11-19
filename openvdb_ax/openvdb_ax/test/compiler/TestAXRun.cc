// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb_ax/ax.h>
#include <openvdb_ax/Exceptions.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>

#include <cppunit/extensions/HelperMacros.h>

class TestAXRun : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestAXRun);
    CPPUNIT_TEST(singleRun);
    CPPUNIT_TEST(multiRun);
    CPPUNIT_TEST_SUITE_END();

    void singleRun();
    void multiRun();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAXRun);

void
TestAXRun::singleRun()
{
    openvdb::FloatGrid f;
    f.setName("a");
    f.tree().setValueOn({0,0,0}, 0.0f);
    openvdb::ax::run("@a = 1.0f;", f);
    CPPUNIT_ASSERT_EQUAL(1.0f, f.tree().getValue({0,0,0}));

    openvdb::math::Transform::Ptr defaultTransform =
        openvdb::math::Transform::createLinearTransform();
    const std::vector<openvdb::Vec3d> singlePointZero = {openvdb::Vec3d::zero()};
    openvdb::points::PointDataGrid::Ptr
        points = openvdb::points::createPointDataGrid
            <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);

    openvdb::ax::run("@a = 1.0f;", *points);
    const auto leafIter = points->tree().cbeginLeaf();
    const auto& descriptor = leafIter->attributeSet().descriptor();

    CPPUNIT_ASSERT_EQUAL(size_t(2), descriptor.size());
    const size_t idx = descriptor.find("a");
    CPPUNIT_ASSERT(idx != openvdb::points::AttributeSet::INVALID_POS);
    CPPUNIT_ASSERT(descriptor.valueType(idx) == openvdb::typeNameAsString<float>());
    openvdb::points::AttributeHandle<float> handle(leafIter->constAttributeArray(idx));
    CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));
}

void
TestAXRun::multiRun()
{
    {
        // test error on points and volumes
        openvdb::FloatGrid::Ptr f(new openvdb::FloatGrid);
        openvdb::points::PointDataGrid::Ptr p(new openvdb::points::PointDataGrid);
        std::vector<openvdb::GridBase::Ptr> v1 { f, p };
        CPPUNIT_ASSERT_THROW(openvdb::ax::run("@a = 1.0f;", v1), openvdb::AXCompilerError);

        std::vector<openvdb::GridBase::Ptr> v2 { p, f };
        CPPUNIT_ASSERT_THROW(openvdb::ax::run("@a = 1.0f;", v2), openvdb::AXCompilerError);
    }

    {
        // multi volumes
        openvdb::FloatGrid::Ptr f1(new openvdb::FloatGrid);
        openvdb::FloatGrid::Ptr f2(new openvdb::FloatGrid);
        f1->setName("a");
        f2->setName("b");
        f1->tree().setValueOn({0,0,0}, 0.0f);
        f2->tree().setValueOn({0,0,0}, 0.0f);
        std::vector<openvdb::GridBase::Ptr> v { f1, f2 };
        openvdb::ax::run("@a = @b = 1;", v);
        CPPUNIT_ASSERT_EQUAL(1.0f, f1->tree().getValue({0,0,0}));
        CPPUNIT_ASSERT_EQUAL(1.0f, f2->tree().getValue({0,0,0}));
    }

    {
        // multi points
        openvdb::math::Transform::Ptr defaultTransform =
            openvdb::math::Transform::createLinearTransform();
        const std::vector<openvdb::Vec3d> singlePointZero = {openvdb::Vec3d::zero()};
        openvdb::points::PointDataGrid::Ptr
            p1 = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::points::PointDataGrid::Ptr
            p2 = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);

        std::vector<openvdb::GridBase::Ptr> v { p1, p2 };
        openvdb::ax::run("@a = @b = 1;", v);

        const auto leafIter1 = p1->tree().cbeginLeaf();
        const auto leafIter2 = p2->tree().cbeginLeaf();
        const auto& descriptor1 = leafIter1->attributeSet().descriptor();
        const auto& descriptor2 = leafIter1->attributeSet().descriptor();

        CPPUNIT_ASSERT_EQUAL(size_t(3), descriptor1.size());
        CPPUNIT_ASSERT_EQUAL(size_t(3), descriptor2.size());
        const size_t idx1 = descriptor1.find("a");
        CPPUNIT_ASSERT_EQUAL(idx1, descriptor2.find("a"));
        const size_t idx2 = descriptor1.find("b");
        CPPUNIT_ASSERT_EQUAL(idx2, descriptor2.find("b"));
        CPPUNIT_ASSERT(idx1 != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(idx2 != openvdb::points::AttributeSet::INVALID_POS);

        CPPUNIT_ASSERT(descriptor1.valueType(idx1) == openvdb::typeNameAsString<float>());
        CPPUNIT_ASSERT(descriptor1.valueType(idx2) == openvdb::typeNameAsString<float>());
        CPPUNIT_ASSERT(descriptor2.valueType(idx1) == openvdb::typeNameAsString<float>());
        CPPUNIT_ASSERT(descriptor2.valueType(idx2) == openvdb::typeNameAsString<float>());

        openvdb::points::AttributeHandle<float> handle(leafIter1->constAttributeArray(idx1));
        CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));
        handle = openvdb::points::AttributeHandle<float>(leafIter1->constAttributeArray(idx2));
        CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));

        handle = openvdb::points::AttributeHandle<float>(leafIter2->constAttributeArray(idx1));
        CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));
        handle = openvdb::points::AttributeHandle<float>(leafIter2->constAttributeArray(idx2));
        CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));
    }
}

