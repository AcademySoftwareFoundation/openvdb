// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/util/Name.h>

#include <gtest/gtest.h>

#include <sstream>


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


TEST_F(TestName, testCorruptLength)
{
    // Length of 4 GiB - 1 followed by only a few bytes.
    std::string bytes("\xff\xff\xff\xff", 4);
    bytes += "0123456789";
    std::istringstream istr(bytes, std::ios_base::binary);
    EXPECT_THROW(openvdb::readString(istr), openvdb::IoError);
}

TEST_F(TestName, testTruncatedLength)
{
    std::istringstream istr(std::string("\x05\x00", 2), std::ios_base::binary);
    EXPECT_THROW(openvdb::readString(istr), openvdb::IoError);
}

TEST_F(TestName, testLargeStrings)
{
    // Sizes around the chunk boundary and well past it.
    for (const size_t size : {size_t(0), size_t(1), size_t(65535), size_t(65536),
        size_t(65537), size_t(200000), size_t(1) << 20}) {
        const openvdb::Name name(size, 'x');
        std::ostringstream ostr(std::ios_base::binary);
        openvdb::writeString(ostr, name);
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        EXPECT_TRUE(name == openvdb::readString(istr));
    }
}
