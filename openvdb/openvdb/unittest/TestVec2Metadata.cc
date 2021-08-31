// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>


class TestVec2Metadata : public ::testing::Test
{
};


TEST_F(TestVec2Metadata, testVec2i)
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec2IMetadata(openvdb::Vec2i(1, 1)));
    Metadata::Ptr m2 = m->copy();

    EXPECT_TRUE(dynamic_cast<Vec2IMetadata*>(m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Vec2IMetadata*>(m2.get()) != 0);

    EXPECT_TRUE(m->typeName().compare("vec2i") == 0);
    EXPECT_TRUE(m2->typeName().compare("vec2i") == 0);

    Vec2IMetadata *s = dynamic_cast<Vec2IMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::Vec2i(1, 1));
    s->value() = openvdb::Vec2i(2, 2);
    EXPECT_TRUE(s->value() == openvdb::Vec2i(2, 2));

    m2->copy(*s);

    s = dynamic_cast<Vec2IMetadata*>(m2.get());
    EXPECT_TRUE(s->value() == openvdb::Vec2i(2, 2));
}

TEST_F(TestVec2Metadata, testVec2s)
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec2SMetadata(openvdb::Vec2s(1, 1)));
    Metadata::Ptr m2 = m->copy();

    EXPECT_TRUE(dynamic_cast<Vec2SMetadata*>(m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Vec2SMetadata*>(m2.get()) != 0);

    EXPECT_TRUE(m->typeName().compare("vec2s") == 0);
    EXPECT_TRUE(m2->typeName().compare("vec2s") == 0);

    Vec2SMetadata *s = dynamic_cast<Vec2SMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::Vec2s(1, 1));
    s->value() = openvdb::Vec2s(2, 2);
    EXPECT_TRUE(s->value() == openvdb::Vec2s(2, 2));

    m2->copy(*s);

    s = dynamic_cast<Vec2SMetadata*>(m2.get());
    EXPECT_TRUE(s->value() == openvdb::Vec2s(2, 2));
}

TEST_F(TestVec2Metadata, testVec2d)
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec2DMetadata(openvdb::Vec2d(1, 1)));
    Metadata::Ptr m2 = m->copy();

    EXPECT_TRUE(dynamic_cast<Vec2DMetadata*>(m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Vec2DMetadata*>(m2.get()) != 0);

    EXPECT_TRUE(m->typeName().compare("vec2d") == 0);
    EXPECT_TRUE(m2->typeName().compare("vec2d") == 0);

    Vec2DMetadata *s = dynamic_cast<Vec2DMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::Vec2d(1, 1));
    s->value() = openvdb::Vec2d(2, 2);
    EXPECT_TRUE(s->value() == openvdb::Vec2d(2, 2));

    m2->copy(*s);

    s = dynamic_cast<Vec2DMetadata*>(m2.get());
    EXPECT_TRUE(s->value() == openvdb::Vec2d(2, 2));
}
