// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Metadata.h>
#include <openvdb/Types.h>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

class TestMetadataIO: public ::testing::Test
{
public:
    template <typename T>
    void test();
    template <typename T>
    void testMultiple();
};


namespace {

template<typename T> struct Value { static T create(int i) { return T(i); } };

template<> struct Value<std::string> {
    static std::string create(int i) { return "test" + std::to_string(i); }
};

template<typename T> struct Value<openvdb::math::Vec2<T>> {
    using ValueType = openvdb::math::Vec2<T>;
    static ValueType create(int i) { return ValueType(i, i+1); }
};
template<typename T> struct Value<openvdb::math::Vec3<T>> {
    using ValueType = openvdb::math::Vec3<T>;
    static ValueType create(int i) { return ValueType(i, i+1, i+2); }
};
template<typename T> struct Value<openvdb::math::Vec4<T>> {
    using ValueType = openvdb::math::Vec4<T>;
    static ValueType create(int i) { return ValueType(i, i+1, i+2, i+3); }
};

}


template <typename T>
void
TestMetadataIO::test()
{
    using namespace openvdb;

    const T val = Value<T>::create(1);
    TypedMetadata<T> m(val);

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<T> tm;
    tm.read(istr);

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN

    EXPECT_EQ(val, tm.value());

    OPENVDB_NO_FP_EQUALITY_WARNING_END
}

template <typename T>
void
TestMetadataIO::testMultiple()
{
    using namespace openvdb;

    const T val1 = Value<T>::create(1), val2 = Value<T>::create(2);
    TypedMetadata<T> m1(val1);
    TypedMetadata<T> m2(val2);

    std::ostringstream ostr(std::ios_base::binary);

    m1.write(ostr);
    m2.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<T> tm1, tm2;
    tm1.read(istr);
    tm2.read(istr);

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN

    EXPECT_EQ(val1, tm1.value());
    EXPECT_EQ(val2, tm2.value());

    OPENVDB_NO_FP_EQUALITY_WARNING_END
}

TEST_F(TestMetadataIO, testInt) { test<int>(); }
TEST_F(TestMetadataIO, testMultipleInt) { testMultiple<int>(); }

TEST_F(TestMetadataIO, testInt64) { test<int64_t>(); }
TEST_F(TestMetadataIO, testMultipleInt64) { testMultiple<int64_t>(); }

TEST_F(TestMetadataIO, testFloat) { test<float>(); }
TEST_F(TestMetadataIO, testMultipleFloat) { testMultiple<float>(); }

TEST_F(TestMetadataIO, testDouble) { test<double>(); }
TEST_F(TestMetadataIO, testMultipleDouble) { testMultiple<double>(); }

TEST_F(TestMetadataIO, testString) { test<std::string>(); }
TEST_F(TestMetadataIO, testMultipleString) { testMultiple<std::string>(); }

TEST_F(TestMetadataIO, testVec3R) { test<openvdb::Vec3R>(); }
TEST_F(TestMetadataIO, testMultipleVec3R) { testMultiple<openvdb::Vec3R>(); }

TEST_F(TestMetadataIO, testVec2i) { test<openvdb::Vec2i>(); }
TEST_F(TestMetadataIO, testMultipleVec2i) { testMultiple<openvdb::Vec2i>(); }

TEST_F(TestMetadataIO, testVec4d) { test<openvdb::Vec4d>(); }
TEST_F(TestMetadataIO, testMultipleVec4d) { testMultiple<openvdb::Vec4d>(); }
