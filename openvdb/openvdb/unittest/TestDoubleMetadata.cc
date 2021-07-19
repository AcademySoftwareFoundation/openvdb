// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>

class TestDoubleMetadata : public ::testing::Test
{
};

TEST_F(TestDoubleMetadata, test)
{
    using namespace openvdb;

    Metadata::Ptr m(new DoubleMetadata(1.23));
    Metadata::Ptr m2 = m->copy();

    EXPECT_TRUE(dynamic_cast<DoubleMetadata*>(m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<DoubleMetadata*>(m2.get()) != 0);

    EXPECT_TRUE(m->typeName().compare("double") == 0);
    EXPECT_TRUE(m2->typeName().compare("double") == 0);

    DoubleMetadata *s = dynamic_cast<DoubleMetadata*>(m.get());
    //EXPECT_TRUE(s->value() == 1.23);
    EXPECT_NEAR(1.23,s->value(),0);
    s->value() = 4.56;
    //EXPECT_TRUE(s->value() == 4.56);
    EXPECT_NEAR(4.56,s->value(),0);

    m2->copy(*s);

    s = dynamic_cast<DoubleMetadata*>(m2.get());
    //EXPECT_TRUE(s->value() == 4.56);
    EXPECT_NEAR(4.56,s->value(),0);
}
