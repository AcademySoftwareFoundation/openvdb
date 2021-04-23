// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>


class TestInt64Metadata : public ::testing::Test
{
};


TEST_F(TestInt64Metadata, test)
{
    using namespace openvdb;

    Metadata::Ptr m(new Int64Metadata(123));
    Metadata::Ptr m2 = m->copy();

    EXPECT_TRUE(dynamic_cast<Int64Metadata*>(m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<Int64Metadata*>(m2.get()) != 0);

    EXPECT_TRUE(m->typeName().compare("int64") == 0);
    EXPECT_TRUE(m2->typeName().compare("int64") == 0);

    Int64Metadata *s = dynamic_cast<Int64Metadata*>(m.get());
    EXPECT_TRUE(s->value() == 123);
    s->value() = 456;
    EXPECT_TRUE(s->value() == 456);

    m2->copy(*s);

    s = dynamic_cast<Int64Metadata*>(m2.get());
    EXPECT_TRUE(s->value() == 456);
}
