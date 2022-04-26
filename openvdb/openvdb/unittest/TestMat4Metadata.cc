// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>
#include <gtest/gtest.h>

class TestMat4Metadata : public ::testing::Test
{
};

TEST_F(TestMat4Metadata, testMat4s)
{
    using namespace openvdb;

    Metadata::Ptr m(new Mat4SMetadata(openvdb::math::Mat4s(1.0f, 1.0f, 1.0f, 1.0f,
                                                           1.0f, 1.0f, 1.0f, 1.0f,
                                                           1.0f, 1.0f, 1.0f, 1.0f,
                                                           1.0f, 1.0f, 1.0f, 1.0f)));
    Metadata::Ptr m3 = m->copy();

    EXPECT_TRUE(dynamic_cast<Mat4SMetadata*>( m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Mat4SMetadata*>(m3.get()) != 0);

    EXPECT_TRUE( m->typeName().compare("mat4s") == 0);
    EXPECT_TRUE(m3->typeName().compare("mat4s") == 0);

    Mat4SMetadata *s = dynamic_cast<Mat4SMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::math::Mat4s(1.0f, 1.0f, 1.0f, 1.0f,
                                                      1.0f, 1.0f, 1.0f, 1.0f,
                                                      1.0f, 1.0f, 1.0f, 1.0f,
                                                      1.0f, 1.0f, 1.0f, 1.0f));
    s->value() = openvdb::math::Mat4s(3.0f, 3.0f, 3.0f, 3.0f,
                                      3.0f, 3.0f, 3.0f, 3.0f,
                                      3.0f, 3.0f, 3.0f, 3.0f,
                                      3.0f, 3.0f, 3.0f, 3.0f);
    EXPECT_TRUE(s->value() == openvdb::math::Mat4s(3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f));

    m3->copy(*s);

    s = dynamic_cast<Mat4SMetadata*>(m3.get());
    EXPECT_TRUE(s->value() == openvdb::math::Mat4s(3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f));
}

TEST_F(TestMat4Metadata, testMat4d)
{
    using namespace openvdb;

    Metadata::Ptr m(new Mat4DMetadata(openvdb::math::Mat4d(1.0, 1.0, 1.0, 1.0,
                                                           1.0, 1.0, 1.0, 1.0,
                                                           1.0, 1.0, 1.0, 1.0,
                                                           1.0, 1.0, 1.0, 1.0)));
    Metadata::Ptr m3 = m->copy();

    EXPECT_TRUE(dynamic_cast<Mat4DMetadata*>( m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Mat4DMetadata*>(m3.get()) != 0);

    EXPECT_TRUE( m->typeName().compare("mat4d") == 0);
    EXPECT_TRUE(m3->typeName().compare("mat4d") == 0);

    Mat4DMetadata *s = dynamic_cast<Mat4DMetadata*>(m.get());
    EXPECT_TRUE(s->value() == openvdb::math::Mat4d(1.0, 1.0, 1.0, 1.0,
                                                      1.0, 1.0, 1.0, 1.0,
                                                      1.0, 1.0, 1.0, 1.0,
                                                      1.0, 1.0, 1.0, 1.0));
    s->value() = openvdb::math::Mat4d(3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0);
    EXPECT_TRUE(s->value() == openvdb::math::Mat4d(3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0));

    m3->copy(*s);

    s = dynamic_cast<Mat4DMetadata*>(m3.get());
    EXPECT_TRUE(s->value() == openvdb::math::Mat4d(3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0));
}
