// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>


class TestVec3Metadata : public ::testing::Test
{
};


TEST_F(TestVec3Metadata, testVec3i)
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec3IMetadata(openvdb::Vec3i(1, 1, 1)));
    Metadata::Ptr m3 = m->copy();

    EXPECT_TRUE(dynamic_cast<Vec3IMetadata*>( m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Vec3IMetadata*>(m3.get()) != 0);

    EXPECT_TRUE( m->typeName().compare("vec3i") == 0);
    EXPECT_TRUE(m3->typeName().compare("vec3i") == 0);

    Vec3IMetadata *s = dynamic_cast<Vec3IMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::Vec3i(1, 1, 1));
    s->value() = openvdb::Vec3i(3, 3, 3);
    EXPECT_TRUE(s->value() == openvdb::Vec3i(3, 3, 3));

    m3->copy(*s);

    s = dynamic_cast<Vec3IMetadata*>(m3.get());
    EXPECT_TRUE(s->value() == openvdb::Vec3i(3, 3, 3));
}

TEST_F(TestVec3Metadata, testVec3s)
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec3SMetadata(openvdb::Vec3s(1, 1, 1)));
    Metadata::Ptr m3 = m->copy();

    EXPECT_TRUE(dynamic_cast<Vec3SMetadata*>( m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Vec3SMetadata*>(m3.get()) != 0);

    EXPECT_TRUE( m->typeName().compare("vec3s") == 0);
    EXPECT_TRUE(m3->typeName().compare("vec3s") == 0);

    Vec3SMetadata *s = dynamic_cast<Vec3SMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::Vec3s(1, 1, 1));
    s->value() = openvdb::Vec3s(3, 3, 3);
    EXPECT_TRUE(s->value() == openvdb::Vec3s(3, 3, 3));

    m3->copy(*s);

    s = dynamic_cast<Vec3SMetadata*>(m3.get());
    EXPECT_TRUE(s->value() == openvdb::Vec3s(3, 3, 3));
}

TEST_F(TestVec3Metadata, testVec3d)
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec3DMetadata(openvdb::Vec3d(1, 1, 1)));
    Metadata::Ptr m3 = m->copy();

    EXPECT_TRUE(dynamic_cast<Vec3DMetadata*>( m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Vec3DMetadata*>(m3.get()) != 0);

    EXPECT_TRUE( m->typeName().compare("vec3d") == 0);
    EXPECT_TRUE(m3->typeName().compare("vec3d") == 0);

    Vec3DMetadata *s = dynamic_cast<Vec3DMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::Vec3d(1, 1, 1));
    s->value() = openvdb::Vec3d(3, 3, 3);
    EXPECT_TRUE(s->value() == openvdb::Vec3d(3, 3, 3));

    m3->copy(*s);

    s = dynamic_cast<Vec3DMetadata*>(m3.get());
    EXPECT_TRUE(s->value() == openvdb::Vec3d(3, 3, 3));
}
