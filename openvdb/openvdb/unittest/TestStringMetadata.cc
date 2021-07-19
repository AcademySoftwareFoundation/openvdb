// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>


class TestStringMetadata : public ::testing::Test
{
};


TEST_F(TestStringMetadata, test)
{
    using namespace openvdb;

    Metadata::Ptr m(new StringMetadata("testing"));
    Metadata::Ptr m2 = m->copy();

    EXPECT_TRUE(dynamic_cast<StringMetadata*>(m.get()) != 0);
    EXPECT_TRUE(dynamic_cast<StringMetadata*>(m2.get()) != 0);

    EXPECT_TRUE(m->typeName().compare("string") == 0);
    EXPECT_TRUE(m2->typeName().compare("string") == 0);

    StringMetadata *s = dynamic_cast<StringMetadata*>(m.get());
    EXPECT_TRUE(s->value().compare("testing") == 0);
    s->value() = "testing2";
    EXPECT_TRUE(s->value().compare("testing2") == 0);

    m2->copy(*s);

    s = dynamic_cast<StringMetadata*>(m2.get());
    EXPECT_TRUE(s->value().compare("testing2") == 0);
}
