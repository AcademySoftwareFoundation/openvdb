// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>


class TestFloatMetadata : public ::testing::Test
{
};

TEST_F(TestFloatMetadata, test)
{
    using namespace openvdb;

    Metadata::Ptr m(new FloatMetadata(1.0));
    Metadata::Ptr m2 = m->copy();

    EXPECT_TRUE(dynamic_cast<FloatMetadata*>(m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<FloatMetadata*>(m2.get()) != 0);

    EXPECT_TRUE(m->typeName().compare("float") == 0);
    EXPECT_TRUE(m2->typeName().compare("float") == 0);

    FloatMetadata *s = dynamic_cast<FloatMetadata*>(m.get());
    //EXPECT_TRUE(s->value() == 1.0);
    EXPECT_NEAR(1.0f,s->value(),0);
    s->value() = 2.0;
    //EXPECT_TRUE(s->value() == 2.0);
    EXPECT_NEAR(2.0f,s->value(),0);
    m2->copy(*s);

    s = dynamic_cast<FloatMetadata*>(m2.get());
    //EXPECT_TRUE(s->value() == 2.0);
    EXPECT_NEAR(2.0f,s->value(),0);
}
