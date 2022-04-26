// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/util/Name.h>

#include <gtest/gtest.h>


class TestName : public ::testing::Test
{
};


TEST_F(TestName, test)
{
    using namespace openvdb;

    Name name;
    Name name2("something");
    Name name3 = std::string("something2");
    name = "something";

    EXPECT_TRUE(name == name2);
    EXPECT_TRUE(name != name3);
    EXPECT_TRUE(name != Name("testing"));
    EXPECT_TRUE(name == Name("something"));
}

TEST_F(TestName, testIO)
{
    using namespace openvdb;

    Name name("some name that i made up");

    std::ostringstream ostr(std::ios_base::binary);

    openvdb::writeString(ostr, name);

    name = "some other name";

    EXPECT_TRUE(name == Name("some other name"));

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    name = openvdb::readString(istr);

    EXPECT_TRUE(name == Name("some name that i made up"));
}

TEST_F(TestName, testMultipleIO)
{
    using namespace openvdb;

    Name name("some name that i made up");
    Name name2("something else");

    std::ostringstream ostr(std::ios_base::binary);

    openvdb::writeString(ostr, name);
    openvdb::writeString(ostr, name2);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    Name n = openvdb::readString(istr), n2 = openvdb::readString(istr);

    EXPECT_TRUE(name == n);
    EXPECT_TRUE(name2 == n2);
}
