// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

// Uncomment to temporarily disable testing of PNanoVDB
//#define DISABLE_PNANOVDB

#include <iostream>
#include <cstdlib>
#include <sstream> // for std::stringstream
#include <vector>
#include <limits.h> // CHAR_BIT
#include <algorithm> // for std::is_sorted
#include <cmath>
#include <cstdlib>

#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/GridValidator.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/DitherLUT.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <nanovdb/util/Stencils.h>
#include <nanovdb/util/Range.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/util/Invoke.h>
#include <nanovdb/util/Reduce.h>
#include <nanovdb/util/GridChecksum.h>
#include <nanovdb/util/NodeManager.h>
#include "../examples/ex_util/CpuTimer.h"

#if !defined(_MSC_VER) // does not compile in msvc c++ due to zero-sized arrays.
#include <nanovdb/CNanoVDB.h>
#endif

#if !defined(DISABLE_PNANOVDB)
#define PNANOVDB_C
#include <nanovdb/PNanoVDB.h>
#include "pnanovdb_validate_strides.h"
#endif

#include <gtest/gtest.h>


namespace nanovdb {// this namespace is required by gtest
std::ostream&
operator<<(std::ostream& os, const Coord& ijk)
{
    os << "(" << ijk[0] << "," << ijk[1] << "," << ijk[2] << ")";
    return os;
}

std::ostream&
operator<<(std::ostream& os, const CoordBBox& b)
{
    os << b[0] << " -> " << b[1];
    return os;
}

template<typename T>
std::ostream&
operator<<(std::ostream& os, const Vec3<T>& v)
{
    os << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
    return os;
}
}// namespace nanovdb

namespace {
template<typename ValueT>
struct Sphere
{
    Sphere(const nanovdb::Vec3<ValueT>& center,
           ValueT                       radius,
           ValueT                       voxelSize = 1.0,
           ValueT                       halfWidth = 3.0)
        : mCenter(center)
        , mRadius(radius)
        , mVoxelSize(voxelSize)
        , mBackground(voxelSize * halfWidth)
    {
    }

    ValueT background() const { return mBackground; }

    /// @brief Only method required by GridBuilder
    ValueT operator()(const nanovdb::Coord& ijk) const
    {
        const ValueT dst = this->sdf(ijk);
        return dst >= mBackground ? mBackground : dst <= -mBackground ? -mBackground : dst;
    }
    ValueT operator()(const nanovdb::Vec3<ValueT>& p) const
    {
        const ValueT dst = this->sdf(p);
        return dst >= mBackground ? mBackground : dst <= -mBackground ? -mBackground : dst;
    }
    bool isInside(const nanovdb::Coord& ijk) const
    {
        return this->sdf(ijk) < 0;
    }
    bool isOutside(const nanovdb::Coord& ijk) const
    {
        return this->sdf(ijk) > 0;
    }
    bool inNarrowBand(const nanovdb::Coord& ijk) const
    {
        const ValueT d = this->sdf(ijk);
        return d < mBackground && d > -mBackground;
    }

private:
    ValueT sdf(nanovdb::Vec3<ValueT> xyz) const
    {
        xyz *= mVoxelSize;
        xyz -= mCenter;
        return xyz.length() - mRadius;
    }
    ValueT sdf(const nanovdb::Coord& ijk) const { return this->sdf(nanovdb::Vec3<ValueT>(ijk[0], ijk[1], ijk[2])); }
    static_assert(nanovdb::is_floating_point<float>::value, "Sphere: expect floating point");
    const nanovdb::Vec3<ValueT> mCenter;
    const ValueT                mRadius, mVoxelSize, mBackground;
}; // Sphere
} // namespace

// The fixture for testing class.
class TestNanoVDB : public ::testing::Test
{
protected:
    TestNanoVDB() {}

    ~TestNanoVDB() override {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    std::string getEnvVar(const std::string& name) const
    {
        const char* str = std::getenv(name.c_str());
        return str == nullptr ? std::string("") : std::string(str);
    }

    template<typename T>
    void printType(const std::string& s)
    {
        const auto n = sizeof(T);
        std::cerr << "Size of " << s << ": " << n << " bytes which is" << (n % 32 == 0 ? " " : " NOT ") << "32 byte aligned" << std::endl;
    }
    nanovdb::CpuTimer<> mTimer;
}; // TestNanoVDB

template <typename T>
class TestOffsets : public ::testing::Test
{
protected:
    TestOffsets() {}

    ~TestOffsets() override {}

    void SetUp() override
    {
    }

    void TearDown() override
    {
    }

}; // TestOffsets<T>

using MyTypes = ::testing::Types<float,
                                 double,
                                 nanovdb::Fp4,
                                 nanovdb::Fp8,
                                 nanovdb::Fp16,
                                 nanovdb::FpN,
                                 int16_t,
                                 int32_t,
                                 int64_t,
                                 nanovdb::Vec3f,
                                 nanovdb::Vec3d,
                                 nanovdb::ValueMask,
                                 bool,
                                 int16_t,
                                 uint32_t>;

TYPED_TEST_SUITE(TestOffsets, MyTypes);

TEST_F(TestNanoVDB, Version)
{
    EXPECT_EQ( 4u, sizeof(uint32_t));
    EXPECT_EQ( 4u, sizeof(nanovdb::Version));
    {// default constructor
        nanovdb::Version v;
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), v.getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), v.getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), v.getPatch());
        std::stringstream ss;
        ss << NANOVDB_MAJOR_VERSION_NUMBER << "."
           << NANOVDB_MINOR_VERSION_NUMBER << "."
           << NANOVDB_PATCH_VERSION_NUMBER;
        auto c_str = v.c_str();
        EXPECT_EQ(ss.str(), std::string(c_str));
        std::free(const_cast<char*>(c_str));
        //std::cerr << v.c_str() << std::endl;
    }
    {// detailed constructor
        const uint32_t major = (1u << 11) - 1;// maximum allowed value
        const uint32_t minor = (1u << 11) - 1;// maximum allowed value
        const uint32_t patch = (1u << 10) - 1;// maximum allowed value
        nanovdb::Version v( major, minor, patch);
        EXPECT_EQ(major, v.getMajor());
        EXPECT_EQ(minor, v.getMinor());
        EXPECT_EQ(patch, v.getPatch());
        std::stringstream ss;
        ss << major << "." << minor << "." << patch;
        auto c_str = v.c_str();
        EXPECT_EQ(ss.str(), std::string(c_str));
        std::free(const_cast<char*>(c_str));
        //std::cerr << v.c_str() << std::endl;
    }
    {// smallest possible version number
        const uint32_t major = 1u;
        const uint32_t minor = 0u;
        const uint32_t patch = 0u;
        nanovdb::Version v( major, minor, patch);
        EXPECT_EQ(major, v.getMajor());
        EXPECT_EQ(minor, v.getMinor());
        EXPECT_EQ(patch, v.getPatch());
        std::stringstream ss;
        ss << major << "." << minor << "." << patch;
        auto c_str = v.c_str();
        EXPECT_EQ(ss.str(), std::string(c_str));
        std::free(const_cast<char*>(c_str));
        //std::cerr << "version.data = " << v.id() << std::endl;
    }
    {// test comparison operators
        EXPECT_EQ( nanovdb::Version(28, 2, 7), nanovdb::Version( 28, 2, 7) );
        EXPECT_LE( nanovdb::Version(28, 2, 7), nanovdb::Version( 28, 2, 7) );
        EXPECT_GE( nanovdb::Version(28, 2, 7), nanovdb::Version( 28, 2, 7) );
        EXPECT_LT( nanovdb::Version(28, 2, 7), nanovdb::Version( 28, 2, 8) );
        EXPECT_LT( nanovdb::Version(28, 2, 7), nanovdb::Version( 28, 3, 7) );
        EXPECT_LT( nanovdb::Version(28, 2, 7), nanovdb::Version( 29, 2, 7) );
        EXPECT_LT( nanovdb::Version(28, 2, 7), nanovdb::Version( 29, 3, 8) );
        EXPECT_GT( nanovdb::Version(29, 0, 0), nanovdb::Version( 28, 2, 8) );
    }

    // nanovdb::Version was introduce in 29.0.0! For backwards compatibility with 28.X.X
    // we need to distinguish between old formats (either uint32_t or two uint16_t) and
    // the new format (nanovdb::Version which internally stored a single uint32_t).
    {
        // Define a struct that memory-maps all three possible representations
        struct T {
            union {
                nanovdb::Version version;
                uint32_t id;
                struct {uint16_t major, minor;};
            };
            T(uint32_t _major) : id(_major) {}
            T(uint32_t _major, uint32_t _minor) : major(_major), minor(_minor) {}
            T(uint32_t _major, uint32_t _minor, uint32_t _patch) : version(_major, _minor, _patch) {}
        };
        EXPECT_EQ( sizeof(uint32_t), sizeof(T) );
        // Verify that T(1,0,0).id() is the smallest instance of Version
        for (uint32_t major = 1; major < 30; ++major) {
            for (uint32_t minor = 0; minor < 10; ++minor) {
                for (uint32_t patch = 0; patch < 10; ++patch) {
                    T tmp(major, minor, patch);
                    EXPECT_LE(T(1,0,0).id, tmp.id);
                    EXPECT_LE(T(1,0,0).version, tmp.version);
                }
            }
        }
        // Verify that all relevant "uint16_t major,minor" instances are smaller than T(29,0,0)
        for (uint32_t major = 1; major <= 28; ++major) {
            for (uint32_t minor = 0; minor < 30; ++minor) {
                T tmp(major, minor);
                EXPECT_LT(tmp.id, T(29,0,0).id);
                EXPECT_LT(tmp.version, T(29,0,0).version);
            }
        }
        // Verify that all relevant "uint32_t major" instances are smaller than T(29,0,0)
        for (uint32_t major = 1; major <= 28; ++major) {
            T tmp(major);
            EXPECT_LT(tmp.id, T(29,0,0).id);
            EXPECT_LT(tmp.version, T(29,0,0).version);
        }
    }
}

TEST_F(TestNanoVDB, Basic)
{
    { // CHAR_BIT
        EXPECT_EQ(8, CHAR_BIT);
    }
    {
        std::vector<int> v = {3, 1, 7, 0};
        EXPECT_FALSE(std::is_sorted(v.begin(), v.end()));
        std::map<int, void*> m;
        for (const auto& i : v)
            m[i] = nullptr;
        v.clear();
        for (const auto& i : m)
            v.push_back(i.first);
        EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
    }
    {
        enum tmp { a = 0,
                   b,
                   c,
                   d } t;
        EXPECT_EQ(sizeof(int), sizeof(t));
    }
    {// Check size of io::MetaData
        EXPECT_EQ(176u, sizeof(nanovdb::io::MetaData));
        //std::cerr << "sizeof(MetaData) = " << sizeof(nanovdb::io::MetaData) << std::endl;
    }
}

TEST_F(TestNanoVDB, Assumptions)
{
    struct A
    {
        int32_t i;
    };
    EXPECT_EQ(sizeof(int32_t), sizeof(A));
    A a{-1};
    EXPECT_EQ(-1, a.i);
    EXPECT_EQ(reinterpret_cast<uint8_t*>(&a), reinterpret_cast<uint8_t*>(&a.i));
    struct B
    {
        A a;
    };
    B b{-1};
    EXPECT_EQ(-1, b.a.i);
    EXPECT_EQ(reinterpret_cast<uint8_t*>(&b), reinterpret_cast<uint8_t*>(&(b.a)));
    EXPECT_EQ(reinterpret_cast<uint8_t*>(&(b.a)), reinterpret_cast<uint8_t*>(&(b.a.i)));
    EXPECT_EQ(nanovdb::AlignUp<32>(48), 64U);
    EXPECT_EQ(nanovdb::AlignUp<8>(16), 16U);
}

TEST_F(TestNanoVDB, Magic)
{
    EXPECT_EQ(0x304244566f6e614eUL, NANOVDB_MAGIC_NUMBER); // Magic number: "NanoVDB0" in hex)
    EXPECT_EQ(0x4e616e6f56444230UL, nanovdb::io::reverseEndianness(NANOVDB_MAGIC_NUMBER));

    // Verify little endian representation
    const char* str = "NanoVDB0"; // note it's exactly 8 bytes
    EXPECT_EQ(8u, strlen(str));
    std::stringstream ss1;
    ss1 << "0x";
    for (int i = 7; i >= 0; --i)
        ss1 << std::hex << unsigned(str[i]);
    ss1 << "UL";
    //std::cerr << ss1.str() << std::endl;
    EXPECT_EQ("0x304244566f6e614eUL", ss1.str());

    uint64_t magic;
    ss1 >> magic;
    EXPECT_EQ(magic, NANOVDB_MAGIC_NUMBER);

    // Verify big endian representation
    std::stringstream ss2;
    ss2 << "0x";
    for (size_t i = 0; i < 8; ++i)
        ss2 << std::hex << unsigned(str[i]);
    ss2 << "UL";
    //std::cerr << ss2.str() << std::endl;
    EXPECT_EQ("0x4e616e6f56444230UL", ss2.str());

    ss2 >> magic;
    EXPECT_EQ(magic, nanovdb::io::reverseEndianness(NANOVDB_MAGIC_NUMBER));
}

TEST_F(TestNanoVDB, FindBits)
{
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t word = uint32_t(1) << i;
        EXPECT_EQ(i, nanovdb::FindLowestOn(word));
        EXPECT_EQ(i, nanovdb::FindHighestOn(word));
    }
    for (uint32_t i = 0; i < 64; ++i) {
        uint64_t word = uint64_t(1) << i;
        EXPECT_EQ(i, nanovdb::FindLowestOn(word));
        EXPECT_EQ(i, nanovdb::FindHighestOn(word));
    }
}

TEST_F(TestNanoVDB, CRC32)
{
    { // test function that uses iterators
        const std::string s{"The quick brown fox jumps over the lazy dog"};
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << nanovdb::crc32(s.begin(), s.end());
        EXPECT_EQ("414fa339", ss.str());
    }
    { // test the checksum for a modified string
        const std::string s{"The quick brown Fox jumps over the lazy dog"};
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << nanovdb::crc32(s.begin(), s.end());
        EXPECT_NE("414fa339", ss.str());
    }
    { // test function that uses void pointer and byte size
        const std::string s{"The quick brown fox jumps over the lazy dog"};
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << nanovdb::crc32(s.data(), s.size());
        EXPECT_EQ("414fa339", ss.str());
    }
    { // test accumulation
        nanovdb::CRC32    crc;
        const std::string s1{"The quick brown fox jum"};
        crc(s1.begin(), s1.end());
        const std::string s2{"ps over the lazy dog"};
        crc(s2.begin(), s2.end());
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << crc.checksum();
        EXPECT_EQ("414fa339", ss.str());
    }
}

TEST_F(TestNanoVDB, Range1D)
{
    nanovdb::Range1D r1(0, 20, 2);
    EXPECT_FALSE(r1.empty());
    EXPECT_EQ(2U, r1.grainsize());
    EXPECT_EQ(20U, r1.size());
    EXPECT_EQ(10U, r1.middle());
    EXPECT_TRUE(r1.is_divisible());
    EXPECT_EQ(0U, r1.begin());
    EXPECT_EQ(20U, r1.end());

    nanovdb::Range1D r2(r1, nanovdb::Split());

    EXPECT_FALSE(r1.empty());
    EXPECT_EQ(2U, r1.grainsize());
    EXPECT_EQ(10U, r1.size());
    EXPECT_EQ(5U, r1.middle());
    EXPECT_TRUE(r1.is_divisible());
    EXPECT_EQ(0U, r1.begin());
    EXPECT_EQ(10U, r1.end());

    EXPECT_FALSE(r2.empty());
    EXPECT_EQ(2U, r2.grainsize());
    EXPECT_EQ(10U, r2.size());
    EXPECT_EQ(15U, r2.middle());
    EXPECT_TRUE(r2.is_divisible());
    EXPECT_EQ(10U, r2.begin());
    EXPECT_EQ(20U, r2.end());
}

TEST_F(TestNanoVDB, Range2D)
{
    nanovdb::Range<2, int> r1(-20, 20, 1u, 0, 20, 2u);

    EXPECT_FALSE(r1.empty());
    EXPECT_EQ(1U, r1[0].grainsize());
    EXPECT_EQ(40U, r1[0].size());
    EXPECT_EQ(0, r1[0].middle());
    EXPECT_TRUE(r1[0].is_divisible());
    EXPECT_EQ(-20, r1[0].begin());
    EXPECT_EQ(20, r1[0].end());

    EXPECT_EQ(2U, r1[1].grainsize());
    EXPECT_EQ(20U, r1[1].size());
    EXPECT_EQ(10, r1[1].middle());
    EXPECT_TRUE(r1[1].is_divisible());
    EXPECT_EQ(0, r1[1].begin());
    EXPECT_EQ(20, r1[1].end());

    nanovdb::Range<2, int> r2(r1, nanovdb::Split());

    EXPECT_FALSE(r1.empty());
    EXPECT_EQ(1U, r1[0].grainsize());
    EXPECT_EQ(20U, r1[0].size());
    EXPECT_EQ(-10, r1[0].middle());
    EXPECT_TRUE(r1[0].is_divisible());
    EXPECT_EQ(-20, r1[0].begin());
    EXPECT_EQ(0, r1[0].end());

    EXPECT_EQ(2U, r1[1].grainsize());
    EXPECT_EQ(20U, r1[1].size());
    EXPECT_EQ(10, r1[1].middle());
    EXPECT_TRUE(r1[1].is_divisible());
    EXPECT_EQ(0, r1[1].begin());
    EXPECT_EQ(20, r1[1].end());

    EXPECT_FALSE(r2.empty());
    EXPECT_EQ(1U, r2[0].grainsize());
    EXPECT_EQ(20U, r2[0].size());
    EXPECT_EQ(10, r2[0].middle());
    EXPECT_TRUE(r2[0].is_divisible());
    EXPECT_EQ(0, r2[0].begin());
    EXPECT_EQ(20, r2[0].end());

    EXPECT_EQ(2U, r2[1].grainsize());
    EXPECT_EQ(20U, r2[1].size());
    EXPECT_EQ(10, r2[1].middle());
    EXPECT_TRUE(r2[1].is_divisible());
    EXPECT_EQ(0, r2[1].begin());
    EXPECT_EQ(20, r2[1].end());
    EXPECT_EQ(r1[1], r2[1]);
}

TEST_F(TestNanoVDB, Range3D)
{
    nanovdb::Range<3, int> r1(-20, 20, 1u, 0, 20, 2u, 0, 10, 5);

    EXPECT_FALSE(r1.empty());
    EXPECT_EQ(1U, r1[0].grainsize());
    EXPECT_EQ(40U, r1[0].size());
    EXPECT_EQ(0, r1[0].middle());
    EXPECT_TRUE(r1[0].is_divisible());
    EXPECT_EQ(-20, r1[0].begin());
    EXPECT_EQ(20, r1[0].end());

    EXPECT_EQ(2U, r1[1].grainsize());
    EXPECT_EQ(20U, r1[1].size());
    EXPECT_EQ(10, r1[1].middle());
    EXPECT_TRUE(r1[1].is_divisible());
    EXPECT_EQ(0, r1[1].begin());
    EXPECT_EQ(20, r1[1].end());

    EXPECT_EQ(5U, r1[2].grainsize());
    EXPECT_EQ(10U, r1[2].size());
    EXPECT_EQ(5, r1[2].middle());
    EXPECT_TRUE(r1[2].is_divisible());
    EXPECT_EQ(0, r1[2].begin());
    EXPECT_EQ(10, r1[2].end());

    nanovdb::Range<3, int> r2(r1, nanovdb::Split());

    EXPECT_FALSE(r1.empty());
    EXPECT_EQ(1U, r1[0].grainsize());
    EXPECT_EQ(20U, r1[0].size());
    EXPECT_EQ(-10, r1[0].middle());
    EXPECT_TRUE(r1[0].is_divisible());
    EXPECT_EQ(-20, r1[0].begin());
    EXPECT_EQ(0, r1[0].end());

    EXPECT_EQ(2U, r1[1].grainsize());
    EXPECT_EQ(20U, r1[1].size());
    EXPECT_EQ(10, r1[1].middle());
    EXPECT_TRUE(r1[1].is_divisible());
    EXPECT_EQ(0, r1[1].begin());
    EXPECT_EQ(20, r1[1].end());

    EXPECT_EQ(5U, r1[2].grainsize());
    EXPECT_EQ(10U, r1[2].size());
    EXPECT_EQ(5, r1[2].middle());
    EXPECT_TRUE(r1[2].is_divisible());
    EXPECT_EQ(0, r1[2].begin());
    EXPECT_EQ(10, r1[2].end());

    EXPECT_FALSE(r2.empty());
    EXPECT_EQ(1U, r2[0].grainsize());
    EXPECT_EQ(20U, r2[0].size());
    EXPECT_EQ(10, r2[0].middle());
    EXPECT_TRUE(r2[0].is_divisible());
    EXPECT_EQ(0, r2[0].begin());
    EXPECT_EQ(20, r2[0].end());

    EXPECT_EQ(2U, r2[1].grainsize());
    EXPECT_EQ(20U, r2[1].size());
    EXPECT_EQ(10, r2[1].middle());
    EXPECT_TRUE(r2[1].is_divisible());
    EXPECT_EQ(0, r2[1].begin());
    EXPECT_EQ(20, r2[1].end());
    EXPECT_EQ(r1[1], r2[1]);

    EXPECT_EQ(5U, r2[2].grainsize());
    EXPECT_EQ(10U, r2[2].size());
    EXPECT_EQ(5, r2[2].middle());
    EXPECT_TRUE(r2[2].is_divisible());
    EXPECT_EQ(0, r2[2].begin());
    EXPECT_EQ(10, r2[2].end());
    EXPECT_EQ(r1[2], r2[2]);
}

TEST_F(TestNanoVDB, invoke)
{
    const int size = 4;
    std::vector<int> array(size, 0);
    for (int i=0; i<size; ++i) {
        EXPECT_EQ(0, array[i]);
    }
    auto kernel0 = [&array](){array[0]=0; };
    auto kernel1 = [&array](){array[1]=1; };
    auto kernel2 = [&array](){array[2]=2; };
    auto kernel3 = [&array](){array[3]=3; };
    nanovdb::invoke(kernel0, kernel1, kernel2, kernel3);
    for (int i=0; i<size; ++i) {
        EXPECT_EQ(i, array[i]);
    }
}

TEST_F(TestNanoVDB, forEach)
{
    const int size = 1000;
    std::vector<int> array(size, 0);
    for (int i=0; i<size; ++i) {
        EXPECT_EQ(0, array[i]);
    }
    auto kernel = [&array](const nanovdb::Range1D &r){for (auto i=r.begin(); i!=r.end(); ++i) array[i]=i; };
    nanovdb::forEach(array, kernel);
    for (int i=0; i<size; ++i) {
        EXPECT_EQ(i, array[i]);
    }
}

TEST_F(TestNanoVDB, reduce)
{
    const int size = 1000;
    std::vector<int> array(size);
    int expected = 0;
    for (int i=0; i<size; ++i) {
        array[i] = i;
        expected += i;
    }
    const int identity = 0;
    auto func = [&array](const nanovdb::Range1D &r, int a){for (auto i=r.begin(); i!=r.end(); ++i) a+=array[i]; return a; };
    auto join = [](int a, int b){return a + b;};
    EXPECT_EQ(expected, nanovdb::reduce(nanovdb::Range1D(0, size), identity, func, join));
    EXPECT_EQ(expected, nanovdb::reduce(array, identity, func, join));
    EXPECT_EQ(expected, nanovdb::reduce(array, 8, identity, func, join));
    for (int i=0; i<size; ++i) {
        EXPECT_EQ(i, array[i]);
    }
}

TEST_F(TestNanoVDB, DitherLUT)
{
    nanovdb::DitherLUT lut;
    float min = 1.0f, max = 0.0f;
    for (int i=-10; i<1024; ++i) {
        const float offset = lut(i);
        if (offset < min) min = offset;
        if (offset > max) max = offset;
        EXPECT_TRUE( offset > 0.0f);
        EXPECT_TRUE( offset < 1.0f);
    }
    //std::cout << "Dither: min = " << min << ", max = " << max << std::endl;
}

TEST_F(TestNanoVDB, Rgba8)
{
    EXPECT_EQ(sizeof(uint32_t), sizeof(nanovdb::Rgba8));
    {
        nanovdb::Rgba8 p;
        EXPECT_EQ(0u, p[0]);
        EXPECT_EQ(0u, p[1]);
        EXPECT_EQ(0u, p[2]);
        EXPECT_EQ(0u, p[3]);
        EXPECT_EQ(0u, p.r());
        EXPECT_EQ(0u, p.g());
        EXPECT_EQ(0u, p.b());
        EXPECT_EQ(0u, p.a());
        EXPECT_EQ(0u, p.packed());
        EXPECT_EQ(nanovdb::Rgba8(), p);
    }
    {
        nanovdb::Rgba8 p(uint8_t(1));
        EXPECT_EQ(1u, p[0]);
        EXPECT_EQ(1u, p[1]);
        EXPECT_EQ(1u, p[2]);
        EXPECT_EQ(1u, p[3]);
        EXPECT_EQ(1u, p.r());
        EXPECT_EQ(1u, p.g());
        EXPECT_EQ(1u, p.b());
        EXPECT_EQ(1u, p.a());
        EXPECT_LT(nanovdb::Rgba8(), p);
    }
    {
        nanovdb::Rgba8 p(uint8_t(1), uint8_t(2), uint8_t(3), uint8_t(4));
        EXPECT_EQ(1u, p[0]);
        EXPECT_EQ(2u, p[1]);
        EXPECT_EQ(3u, p[2]);
        EXPECT_EQ(4u, p[3]);
        EXPECT_EQ(1u, p.r());
        EXPECT_EQ(2u, p.g());
        EXPECT_EQ(3u, p.b());
        EXPECT_EQ(4u, p.a());
        EXPECT_LT(nanovdb::Rgba8(), p);
    }
    {
        nanovdb::Rgba8 p(uint8_t(255), uint8_t(255), uint8_t(255), uint8_t(255));
        EXPECT_EQ(255u, p[0]);
        EXPECT_EQ(255u, p[1]);
        EXPECT_EQ(255u, p[2]);
        EXPECT_EQ(255u, p[3]);
        EXPECT_EQ(255u, p.r());
        EXPECT_EQ(255u, p.g());
        EXPECT_EQ(255u, p.b());
        EXPECT_EQ(255u, p.a());
        EXPECT_LT(nanovdb::Rgba8(), p);
        EXPECT_NEAR(p.lengthSqr(), 3.0f, 1e-6);
        EXPECT_NEAR(p.length(), sqrt(3.0f), 1e-6);
    }
    {
        nanovdb::Rgba8 p(1.0f, 0.0f, 0.0f, 1.0f);
        EXPECT_EQ(255u, p[0]);
        EXPECT_EQ(0u,   p[1]);
        EXPECT_EQ(0u,   p[2]);
        EXPECT_EQ(255u, p[3]);
        EXPECT_EQ(255u, p.r());
        EXPECT_EQ(0u,   p.g());
        EXPECT_EQ(0u,   p.b());
        EXPECT_EQ(255u, p.a());
        EXPECT_LT(nanovdb::Rgba8(), p);
        EXPECT_NEAR(p.lengthSqr(), 1.0f, 1e-6);
        EXPECT_NEAR(p.length(), 1.0f, 1e-6);
    }
    {
        nanovdb::Rgba8 p(0.0f, 1.0f, 0.5f, 0.1f);
        EXPECT_EQ(0u,   p[0]);
        EXPECT_EQ(255u, p[1]);
        EXPECT_EQ(128u, p[2]);
        EXPECT_EQ(26u,  p[3]);
        EXPECT_EQ(0u,   p.r());
        EXPECT_EQ(255u, p.g());
        EXPECT_EQ(128u, p.b());
        EXPECT_EQ(26u,  p.a());
        EXPECT_LT(nanovdb::Rgba8(), p);
    }
}

TEST_F(TestNanoVDB, Coord)
{
    EXPECT_EQ(size_t(3 * 4), nanovdb::Coord::memUsage()); // due to padding
    {
        nanovdb::Coord ijk;
        EXPECT_EQ(sizeof(ijk), size_t(3 * 4));
        EXPECT_EQ(0, ijk[0]);
        EXPECT_EQ(0, ijk[1]);
        EXPECT_EQ(0, ijk[2]);
        EXPECT_EQ(0, ijk.x());
        EXPECT_EQ(0, ijk.y());
        EXPECT_EQ(0, ijk.z());
    }
    {
        nanovdb::Coord ijk(1, 2, 3);
        EXPECT_EQ(1, ijk[0]);
        EXPECT_EQ(2, ijk[1]);
        EXPECT_EQ(3, ijk[2]);
        EXPECT_EQ(1, ijk.x());
        EXPECT_EQ(2, ijk.y());
        EXPECT_EQ(3, ijk.z());
        ijk[1] = 4;
        EXPECT_EQ(1, ijk[0]);
        EXPECT_EQ(4, ijk[1]);
        EXPECT_EQ(3, ijk[2]);
        ijk.x() += -2;
        EXPECT_EQ(-1, ijk[0]);
        EXPECT_EQ(4, ijk[1]);
        EXPECT_EQ(3, ijk[2]);
    }
    { // hash
        EXPECT_EQ(0, nanovdb::Coord(1, 2, 3).octant());
        EXPECT_EQ(0, nanovdb::Coord(1, 9, 3).octant());
        EXPECT_EQ(1, nanovdb::Coord(-1, 2, 3).octant());
        EXPECT_EQ(2, nanovdb::Coord(1, -2, 3).octant());
        EXPECT_EQ(3, nanovdb::Coord(-1, -2, 3).octant());
        EXPECT_EQ(4, nanovdb::Coord(1, 2, -3).octant());
        EXPECT_EQ(5, nanovdb::Coord(-1, 2, -3).octant());
        EXPECT_EQ(6, nanovdb::Coord(1, -2, -3).octant());
        EXPECT_EQ(7, nanovdb::Coord(-1, -2, -3).octant());
        for (int i = 0; i < 5; ++i)
            EXPECT_EQ(i / 2, i >> 1);
    }
}

TEST_F(TestNanoVDB, BBox)
{
    nanovdb::BBox<nanovdb::Vec3f> bbox;
    EXPECT_EQ(sizeof(bbox), size_t(2 * 3 * 4));
    EXPECT_EQ(std::numeric_limits<float>::max(), bbox[0][0]);
    EXPECT_EQ(std::numeric_limits<float>::max(), bbox[0][1]);
    EXPECT_EQ(std::numeric_limits<float>::max(), bbox[0][2]);
    EXPECT_EQ(-std::numeric_limits<float>::max(), bbox[1][0]);
    EXPECT_EQ(-std::numeric_limits<float>::max(), bbox[1][1]);
    EXPECT_EQ(-std::numeric_limits<float>::max(), bbox[1][2]);
    EXPECT_TRUE(bbox.empty());

    bbox.expand(nanovdb::Vec3f(57.0f, -31.0f, 60.0f));
    EXPECT_TRUE(bbox.empty());
    EXPECT_EQ(nanovdb::Vec3f(0.0f), bbox.dim());
    EXPECT_EQ(57.0f, bbox[0][0]);
    EXPECT_EQ(-31.0f, bbox[0][1]);
    EXPECT_EQ(60.0f, bbox[0][2]);
    EXPECT_EQ(57.0f, bbox[1][0]);
    EXPECT_EQ(-31.0f, bbox[1][1]);
    EXPECT_EQ(60.0f, bbox[1][2]);

    bbox.expand(nanovdb::Vec3f(58.0f, 0.0f, 62.0f));
    EXPECT_FALSE(bbox.empty());
    EXPECT_EQ(nanovdb::Vec3f(1.0f, 31.0f, 2.0f), bbox.dim());
    EXPECT_EQ(57.0f, bbox[0][0]);
    EXPECT_EQ(-31.0f, bbox[0][1]);
    EXPECT_EQ(60.0f, bbox[0][2]);
    EXPECT_EQ(58.0f, bbox[1][0]);
    EXPECT_EQ(0.0f, bbox[1][1]);
    EXPECT_EQ(62.0f, bbox[1][2]);
}

TEST_F(TestNanoVDB, CoordBBox)
{
    nanovdb::CoordBBox bbox;
    EXPECT_EQ(sizeof(bbox), size_t(2 * 3 * 4));
    EXPECT_EQ(std::numeric_limits<int32_t>::max(), bbox[0][0]);
    EXPECT_EQ(std::numeric_limits<int32_t>::max(), bbox[0][1]);
    EXPECT_EQ(std::numeric_limits<int32_t>::max(), bbox[0][2]);
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), bbox[1][0]);
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), bbox[1][1]);
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), bbox[1][2]);
    EXPECT_TRUE(bbox.empty());

    bbox.expand(nanovdb::Coord(57, -31, 60));
    EXPECT_FALSE(bbox.empty());
    EXPECT_EQ(nanovdb::Coord(1), bbox.dim());
    EXPECT_EQ(57, bbox[0][0]);
    EXPECT_EQ(-31, bbox[0][1]);
    EXPECT_EQ(60, bbox[0][2]);
    EXPECT_EQ(57, bbox[1][0]);
    EXPECT_EQ(-31, bbox[1][1]);
    EXPECT_EQ(60, bbox[1][2]);

    bbox.expand(nanovdb::Coord(58, 0, 62));
    EXPECT_FALSE(bbox.empty());
    EXPECT_EQ(nanovdb::Coord(2, 32, 3), bbox.dim());
    EXPECT_EQ(57, bbox[0][0]);
    EXPECT_EQ(-31, bbox[0][1]);
    EXPECT_EQ(60, bbox[0][2]);
    EXPECT_EQ(58, bbox[1][0]);
    EXPECT_EQ(0, bbox[1][1]);
    EXPECT_EQ(62, bbox[1][2]);

    { // test convert
        auto bbox2 = bbox.asReal<float>();
        EXPECT_FALSE(bbox2.empty());
        EXPECT_EQ(nanovdb::Vec3f(57.0f, -31.0f, 60.0f), bbox2.min());
        EXPECT_EQ(nanovdb::Vec3f(59.0f, 1.0f, 63.0f), bbox2.max());
    }

    { // test prefix iterator
        auto iter = bbox.begin();
        EXPECT_TRUE(iter);
        for (int i = bbox.min()[0]; i <= bbox.max()[0]; ++i) {
            for (int j = bbox.min()[1]; j <= bbox.max()[1]; ++j) {
                for (int k = bbox.min()[2]; k <= bbox.max()[2]; ++k) {
                    EXPECT_TRUE(bbox.isInside(*iter));
                    EXPECT_TRUE(iter);
                    const auto& ijk = *iter; // note, copy by reference
                    EXPECT_EQ(ijk[0], i);
                    EXPECT_EQ(ijk[1], j);
                    EXPECT_EQ(ijk[2], k);
                    ++iter;
                }
            }
        }
        EXPECT_FALSE(iter);
    }

    { // test postfix iterator
        auto iter = bbox.begin();
        EXPECT_TRUE(iter);
        for (int i = bbox.min()[0]; i <= bbox.max()[0]; ++i) {
            for (int j = bbox.min()[1]; j <= bbox.max()[1]; ++j) {
                for (int k = bbox.min()[2]; k <= bbox.max()[2]; ++k) {
                    EXPECT_TRUE(iter);
                    const auto ijk = *iter++; // note, copy by value!
                    EXPECT_EQ(ijk[0], i);
                    EXPECT_EQ(ijk[1], j);
                    EXPECT_EQ(ijk[2], k);
                }
            }
        }
        EXPECT_FALSE(iter);
    }
}

TEST_F(TestNanoVDB, Vec3)
{
    bool test = nanovdb::is_specialization<double, nanovdb::Vec3>::value;
    EXPECT_FALSE(test);
    test = nanovdb::TensorTraits<double>::IsVector;
    EXPECT_FALSE(test);
    test = nanovdb::is_specialization<nanovdb::Vec3R, nanovdb::Vec3>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<double, nanovdb::Vec3R::ValueType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::TensorTraits<nanovdb::Vec3R>::IsVector;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<double, nanovdb::TensorTraits<nanovdb::Vec3R>::ElementType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<double, nanovdb::FloatTraits<nanovdb::Vec3R>::FloatType>::value;
    EXPECT_TRUE(test);
    EXPECT_EQ(size_t(3 * 8), sizeof(nanovdb::Vec3R));

    nanovdb::Vec3R xyz(1.0, 2.0, 3.0);
    EXPECT_EQ(1.0, xyz[0]);
    EXPECT_EQ(2.0, xyz[1]);
    EXPECT_EQ(3.0, xyz[2]);

    xyz[1] = -2.0;
    EXPECT_EQ(1.0, xyz[0]);
    EXPECT_EQ(-2.0, xyz[1]);
    EXPECT_EQ(3.0, xyz[2]);

    EXPECT_EQ(1.0 + 4.0 + 9.0, xyz.lengthSqr());
    EXPECT_EQ(sqrt(1.0 + 4.0 + 9.0), xyz.length());

    EXPECT_EQ(nanovdb::Vec3f(1, 2, 3), nanovdb::Vec3f(1, 2, 3));
    EXPECT_NE(nanovdb::Vec3f(1, 2, 3), nanovdb::Vec3f(1, 2, 4));

    {// alignment to largest type
        EXPECT_EQ(size_t(3 * 4), sizeof(nanovdb::Vec3f));
        union {uint64_t a; nanovdb::Vec3f b;} c;
        EXPECT_EQ(2 * sizeof(uint64_t), sizeof(c));
        EXPECT_EQ(nanovdb::AlignUp<8>(sizeof(nanovdb::Vec3f)), sizeof(c));
    }
}

TEST_F(TestNanoVDB, Vec4)
{
    bool test = nanovdb::is_specialization<double, nanovdb::Vec4>::value;
    EXPECT_FALSE(test);
    test = nanovdb::TensorTraits<double>::IsVector;
    EXPECT_FALSE(test);
    test = nanovdb::TensorTraits<double>::IsScalar;
    EXPECT_TRUE(test);
    int rank = nanovdb::TensorTraits<double>::Rank;
    EXPECT_EQ(0, rank);
    rank = nanovdb::TensorTraits<nanovdb::Vec3R>::Rank;
    EXPECT_EQ(1, rank);
    test = nanovdb::is_same<double, nanovdb::FloatTraits<float>::FloatType>::value;
    EXPECT_FALSE(test);
    test = nanovdb::is_same<double, nanovdb::FloatTraits<double>::FloatType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<float, nanovdb::FloatTraits<uint32_t>::FloatType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<double, nanovdb::FloatTraits<uint64_t>::FloatType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_specialization<nanovdb::Vec4R, nanovdb::Vec4>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_specialization<nanovdb::Vec3R, nanovdb::Vec4>::value;
    EXPECT_FALSE(test);
    test = nanovdb::is_same<double, nanovdb::Vec4R::ValueType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::TensorTraits<nanovdb::Vec3R>::IsVector;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<double, nanovdb::TensorTraits<nanovdb::Vec4R>::ElementType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<double, nanovdb::TensorTraits<double>::ElementType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<float, nanovdb::TensorTraits<float>::ElementType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<uint32_t, nanovdb::TensorTraits<uint32_t>::ElementType>::value;
    EXPECT_TRUE(test);
    test = nanovdb::is_same<double, nanovdb::FloatTraits<nanovdb::Vec4R>::FloatType>::value;
    EXPECT_TRUE(test);
    EXPECT_EQ(size_t(4 * 8), sizeof(nanovdb::Vec4R));

    nanovdb::Vec4R xyz(1.0, 2.0, 3.0, 4.0);
    EXPECT_EQ(1.0, xyz[0]);
    EXPECT_EQ(2.0, xyz[1]);
    EXPECT_EQ(3.0, xyz[2]);
    EXPECT_EQ(4.0, xyz[3]);

    xyz[1] = -2.0;
    EXPECT_EQ(1.0, xyz[0]);
    EXPECT_EQ(-2.0, xyz[1]);
    EXPECT_EQ(3.0, xyz[2]);
    EXPECT_EQ(4.0, xyz[3]);

    EXPECT_EQ(1.0 + 4.0 + 9.0 + 16.0, xyz.lengthSqr());
    EXPECT_EQ(sqrt(1.0 + 4.0 + 9.0 + 16.0), xyz.length());

    EXPECT_EQ(nanovdb::Vec4f(1, 2, 3, 4), nanovdb::Vec4f(1, 2, 3, 4));
    EXPECT_NE(nanovdb::Vec4f(1, 2, 3, 4), nanovdb::Vec4f(1, 2, 3, 5));
}// Vec4

TEST_F(TestNanoVDB, Extrema)
{
    { // int
        nanovdb::Extrema<int> e(-1);
        EXPECT_EQ(-1, e.min());
        EXPECT_EQ(-1, e.max());
        e.add(-2);
        e.add(5);
        EXPECT_TRUE(e);
        EXPECT_EQ(-2, e.min());
        EXPECT_EQ(5, e.max());
    }
    { // float
        nanovdb::Extrema<float> e(-1.0f);
        EXPECT_EQ(-1.0f, e.min());
        EXPECT_EQ(-1.0f, e.max());
        e.add(-2.0f);
        e.add(5.0f);
        EXPECT_TRUE(e);
        EXPECT_EQ(-2.0f, e.min());
        EXPECT_EQ(5.0f, e.max());
    }
    { // Vec3f
        nanovdb::Extrema<nanovdb::Vec3f> e(nanovdb::Vec3f(1.0f, 1.0f, 0.0f));
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 1.0f, 0.0f), e.min());
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 1.0f, 0.0f), e.max());
        e.add(nanovdb::Vec3f(1.0f, 0.0f, 0.0f));
        e.add(nanovdb::Vec3f(1.0f, 1.0f, 1.0f));
        EXPECT_TRUE(e);
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), e.min());
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 1.0f, 1.0f), e.max());
    }
}

TEST_F(TestNanoVDB, RayEmptyBBox)
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using CoordBBoxT = nanovdb::BBox<CoordT>;
    using BBoxT = nanovdb::BBox<Vec3T>;
    using RayT = nanovdb::Ray<RealT>;

    // test bbox clip
    const Vec3T dir(1.0, 0.0, 0.0);
    const Vec3T eye(-1.0, 0.5, 0.5);
    RealT       t0 = 0.0, t1 = 10000.0;
    RayT        ray(eye, dir, t0, t1);

    const CoordBBoxT bbox1;
    EXPECT_TRUE(bbox1.empty());
    EXPECT_FALSE(ray.intersects(bbox1, t0, t1));

    const BBoxT bbox2;
    EXPECT_TRUE(bbox2.empty());
    EXPECT_FALSE(ray.intersects(bbox2, t0, t1));
}

TEST_F(TestNanoVDB, RayBasic)
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using CoordBBoxT = nanovdb::BBox<CoordT>;
    using BBoxT = nanovdb::BBox<Vec3T>;
    using RayT = nanovdb::Ray<RealT>;

    // test bbox clip
    const Vec3T dir(1.0, 0.0, 0.0);
    const Vec3T eye(-1.0, 0.5, 0.5);
    RealT       t0 = 0.0, t1 = 10000.0;
    RayT        ray(eye, dir, t0, t1);

    const CoordBBoxT bbox(CoordT(0, 0, 0), CoordT(0, 0, 0)); // only contains a single point (0,0,0)
    EXPECT_FALSE(bbox.empty());
    EXPECT_EQ(bbox.dim(), CoordT(1, 1, 1));

    const BBoxT bbox2(CoordT(0, 0, 0), CoordT(0, 0, 0));
    EXPECT_EQ(bbox2, bbox.asReal<float>());
    EXPECT_FALSE(bbox2.empty());
    EXPECT_EQ(bbox2.dim(), Vec3T(1.0f, 1.0f, 1.0f));
    EXPECT_EQ(bbox2[0], Vec3T(0.0f, 0.0f, 0.0f));
    EXPECT_EQ(bbox2[1], Vec3T(1.0f, 1.0f, 1.0f));

    EXPECT_TRUE(ray.clip(bbox)); // ERROR: how can a non-empty bbox have no intersections!?
    //EXPECT_TRUE( ray.clip(bbox.asReal<float>()));// correct!

    // intersects the two faces of the box perpendicular to the x-axis!
    EXPECT_EQ(1.0f, ray.t0());
    EXPECT_EQ(2.0f, ray.t1());
    EXPECT_EQ(ray(1.0f), Vec3T(0.0f, 0.5f, 0.5f)); //lower y component of intersection
    EXPECT_EQ(ray(2.0f), Vec3T(1.0f, 0.5f, 0.5f)); //higher y component of intersection
} // RayBasic

TEST_F(TestNanoVDB, Ray)
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using CoordBBoxT = nanovdb::BBox<CoordT>;
    using BBoxT = nanovdb::BBox<Vec3T>;
    using RayT = nanovdb::Ray<RealT>;

    // test bbox clip
    const Vec3T dir(-1.0, 2.0, 3.0);
    const Vec3T eye(2.0, 1.0, 1.0);
    RealT       t0 = 0.1, t1 = 12589.0;
    RayT        ray(eye, dir, t0, t1);

    // intersects the two faces of the box perpendicular to the y-axis!
    EXPECT_TRUE(ray.clip(CoordBBoxT(CoordT(0, 2, 2), CoordT(2, 4, 6))));
    //std::cerr << ray(0.5) << ", " << ray(2.0) << std::endl;
    EXPECT_EQ(0.5, ray.t0());
    EXPECT_EQ(2.0, ray.t1());
    EXPECT_EQ(ray(0.5)[1], 2); //lower y component of intersection
    EXPECT_EQ(ray(2.0)[1], 5); //higher y component of intersection

    ray.reset(eye, dir, t0, t1);
    // intersects the lower edge anlong the z-axis of the box
    EXPECT_TRUE(ray.clip(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0))));
    //std::cerr << ray(0.5) << ", " << ray(2.0) << std::endl;
    EXPECT_EQ(0.5, ray.t0());
    EXPECT_EQ(0.5, ray.t1());
    EXPECT_EQ(ray(0.5)[0], 1.5); //lower y component of intersection
    EXPECT_EQ(ray(0.5)[1], 2.0); //higher y component of intersection

    ray.reset(eye, dir, t0, t1);
    // no intersections
    EXPECT_TRUE(!ray.clip(CoordBBoxT(CoordT(4, 2, 2), CoordT(6, 4, 6))));
    EXPECT_EQ(t0, ray.t0());
    EXPECT_EQ(t1, ray.t1());
}

TEST_F(TestNanoVDB, HDDA)
{
    using RealT = float;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;
    using Vec3T = RayT::Vec3T;
    using DDAT = nanovdb::HDDA<RayT, CoordT>;

    { // basic test
        const RayT::Vec3T dir(1.0, 0.0, 0.0);
        const RayT::Vec3T eye(-1.0, 0.0, 0.0);
        const RayT        ray(eye, dir);
        DDAT              dda(ray, 1 << (3 + 4 + 5));
        EXPECT_EQ(nanovdb::Delta<RealT>::value(), dda.time());
        EXPECT_EQ(1.0, dda.next());
        dda.step();
        EXPECT_EQ(1.0, dda.time());
        EXPECT_EQ(4096 + 1.0, dda.next());
    }
    { // Check for the notorious +-0 issue!

        const Vec3T dir1(1.0, 0.0, 0.0);
        const Vec3T eye1(2.0, 0.0, 0.0);
        const RayT  ray1(eye1, dir1);
        DDAT        dda1(ray1, 1 << 3);
        dda1.step();

        const Vec3T dir2(1.0, -0.0, -0.0);
        const Vec3T eye2(2.0, 0.0, 0.0);
        const RayT  ray2(eye2, dir2);
        DDAT        dda2(ray2, 1 << 3);
        dda2.step();

        const Vec3T dir3(1.0, -1e-9, -1e-9);
        const Vec3T eye3(2.0, 0.0, 0.0);
        const RayT  ray3(eye3, dir3);
        DDAT        dda3(ray3, 1 << 3);
        dda3.step();

        const Vec3T dir4(1.0, -1e-9, -1e-9);
        const Vec3T eye4(2.0, 0.0, 0.0);
        const RayT  ray4(eye3, dir4);
        DDAT        dda4(ray4, 1 << 3);
        dda4.step();

        EXPECT_EQ(dda1.time(), dda2.time());
        EXPECT_EQ(dda2.time(), dda3.time());
        EXPECT_EQ(dda3.time(), dda4.time());
        EXPECT_EQ(dda1.next(), dda2.next());
        EXPECT_EQ(dda2.next(), dda3.next());
        EXPECT_EQ(dda3.next(), dda4.next());
    }
    { // test voxel traversal along both directions of each axis
        const Vec3T eye(0, 0, 0);
        for (int s = -1; s <= 1; s += 2) {
            for (int a = 0; a < 3; ++a) {
                const int   d[3] = {s * (a == 0), s * (a == 1), s * (a == 2)};
                const Vec3T dir(d[0], d[1], d[2]);
                RayT        ray(eye, dir);
                DDAT        dda(ray, 1 << 0);
                for (int i = 1; i <= 10; ++i) {
                    EXPECT_TRUE(dda.step());
                    EXPECT_EQ(i, dda.time());
                }
            }
        }
    }
    { // test Node traversal along both directions of each axis
        const Vec3T eye(0, 0, 0);

        for (int s = -1; s <= 1; s += 2) {
            for (int a = 0; a < 3; ++a) {
                const int   d[3] = {s * (a == 0), s * (a == 1), s * (a == 2)};
                const Vec3T dir(d[0], d[1], d[2]);
                RayT        ray(eye, dir);
                DDAT        dda(ray, 1 << 3);
                for (int i = 1; i <= 10; ++i) {
                    EXPECT_TRUE(dda.step());
                    EXPECT_EQ(8 * i, dda.time());
                }
            }
        }
    }
    { // test accelerated Node traversal along both directions of each axis
        const Vec3T eye(0, 0, 0);

        for (int s = -1; s <= 1; s += 2) {
            for (int a = 0; a < 3; ++a) {
                const int   d[3] = {s * (a == 0), s * (a == 1), s * (a == 2)};
                const Vec3T dir(2.0f * d[0], 2.0f * d[1], 2.0f * d[2]);
                RayT        ray(eye, dir);
                DDAT        dda(ray, 1 << 3);
                double      next = 0;
                for (int i = 1; i <= 10; ++i) {
                    EXPECT_TRUE(dda.step());
                    EXPECT_EQ(4 * i, dda.time());
                    if (i > 1) {
                        EXPECT_EQ(dda.time(), next);
                    }
                    next = dda.next();
                }
            }
        }
    }
#if 0
  if (!this->getEnvVar( "VDB_DATA_PATH" ).empty()) {// bug when ray-tracing dragon model
    const Vec3T eye(1563.350342,-390.161621,697.749023);
    const Vec3T dir(-0.963871,0.048393,-0.626651);
    RayT ray( eye, dir );
    auto srcGrid = this->getSrcGrid();
    auto handle = nanovdb::openToNanoVDB( *srcGrid );
    EXPECT_TRUE( handle );
    auto *grid = handle.grid<float>();
    EXPECT_TRUE( grid );
    EXPECT_TRUE( grid->isLevelSet() );
    //ray = ray.applyMapF( map );
    auto acc = grid->getAccessor();
    CoordT ijk;
    float v0;
    EXPECT_TRUE(nanovdb::ZeroCrossing( ray, acc, ijk, v0 ) );
    std::cerr << "hit with v0 =" << v0 << " background = " << grid->tree().background() << std::endl;
  }
#endif
} // HDDA

TEST_F(TestNanoVDB, Mask)
{
    using MaskT = nanovdb::Mask<3>;
    EXPECT_EQ(8u, MaskT::wordCount());
    EXPECT_EQ(512u, MaskT::bitCount());
    EXPECT_EQ(size_t(8 * 8), MaskT::memUsage());

    MaskT mask;
    EXPECT_EQ(0U, mask.countOn());
    EXPECT_TRUE(mask.isOff());
    EXPECT_FALSE(mask.isOn());
    EXPECT_FALSE(mask.beginOn());
    for (uint32_t i = 0; i < MaskT::bitCount(); ++i)
        EXPECT_FALSE(mask.isOn(i));

    mask.setOn(256);
    EXPECT_FALSE(mask.isOff());
    EXPECT_FALSE(mask.isOn());
    auto iter = mask.beginOn();
    EXPECT_TRUE(iter);
    EXPECT_EQ(256U, *iter);
    EXPECT_FALSE(++iter);
    for (uint32_t i = 0; i < MaskT::bitCount(); ++i) {
        if (i != 256)
            EXPECT_FALSE(mask.isOn(i));
        else
            EXPECT_TRUE(mask.isOn(i));
    }

    mask.set(256, false);
    EXPECT_TRUE(mask.isOff());
    EXPECT_FALSE(mask.isOn());
    EXPECT_FALSE(mask.isOn(256));

    mask.set(256, true);
    EXPECT_FALSE(mask.isOff());
    EXPECT_FALSE(mask.isOn());
    EXPECT_TRUE(mask.isOn(256));
}

TEST_F(TestNanoVDB, LeafNode)
{
    using LeafT = nanovdb::LeafNode<float>;
    //EXPECT_FALSE(LeafT::IgnoreValues);
    EXPECT_EQ(8u, LeafT::dim());
    EXPECT_EQ(512u, LeafT::voxelCount());
    EXPECT_EQ(size_t(
                  3 * 4 + // mBBoxMin
                  4 * 1 + // mBBoxDif[3] + mFlags
                  8 * 8 + // mValueMask,
                  2 * 4 + // mMinimum, mMaximum
                  2 * 4 + // mAverage, mVariance
                  512 * 4 // mValues[512]
                  ),
              sizeof(LeafT));
    // this particular value type happens to be exactly 32B aligned!
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(
                  3 * 4 + // mBBoxMin
                  4 * 1 + // mBBoxDif[3] + mFlags
                  8 * 8 + // mValueMask,
                  2 * 4 + // mMinimum, mMaximum
                  2 * 4 + // mAverage, mVariance
                  512 * 4 // mValues[512]
                  ),
              sizeof(LeafT));

    // allocate buffer
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[LeafT::memUsage()]);
    std::memset(buffer.get(), 0, LeafT::memUsage());
    LeafT*                     leaf = reinterpret_cast<LeafT*>(buffer.get());

    { // set members of the leaf node
        auto& data = *reinterpret_cast<LeafT::DataType*>(buffer.get());
        data.mValueMask.setOff();
        auto* values = data.mValues;
        for (int i = 0; i < 256; ++i)
            *values++ = 0.0f;
        for (uint32_t i = 256; i < LeafT::voxelCount(); ++i) {
            data.mValueMask.setOn(i);
            *values++ = 1.234f;
        }
        data.mMinimum = 0.0f;
        data.mMaximum = 1.234f;
        data.mFlags = uint8_t(2);// set bit # 1 on since leaf contains active values
    }

    EXPECT_TRUE( leaf->isActive() );

    { // compute BBox
        auto& data = *reinterpret_cast<LeafT::DataType*>(buffer.get());
        EXPECT_EQ(8u, data.mValueMask.wordCount());

        nanovdb::CoordBBox bbox(nanovdb::Coord(-1), nanovdb::Coord(-1));
        uint64_t           word = 0u;
        for (int i = 0; i < 8; ++i) {
            if (uint64_t w = data.mValueMask.getWord<uint64_t>(i)) {
                word |= w;
                if (bbox[0][0] == -1)
                    bbox[0][0] = i;
                bbox[1][0] = i;
            }
        }
        EXPECT_TRUE(word != 0u);
        bbox[0][1] = nanovdb::FindLowestOn(word) >> 3;
        bbox[1][1] = nanovdb::FindHighestOn(word) >> 3;

        const uint8_t* p = reinterpret_cast<const uint8_t*>(&word);
        uint32_t       b = p[0] | p[1] | p[2] | p[3] | p[4] | p[5] | p[6] | p[7];
        EXPECT_TRUE(b != 0u);
        bbox[0][2] = nanovdb::FindLowestOn(b);
        bbox[1][2] = nanovdb::FindHighestOn(b);
        //std::cerr << bbox << std::endl;
        EXPECT_EQ(bbox[0], nanovdb::Coord(4, 0, 0));
        EXPECT_EQ(bbox[1], nanovdb::Coord(7, 7, 7));
    }

    EXPECT_TRUE( leaf->isActive() );

    // check values
    auto* ptr = reinterpret_cast<LeafT::DataType*>(buffer.get())->mValues;
    for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
        if (i < 256) {
            EXPECT_FALSE(leaf->valueMask().isOn(i));
            EXPECT_EQ(0.0f, *ptr++);
        } else {
            EXPECT_TRUE(leaf->valueMask().isOn(i));
            EXPECT_EQ(1.234f, *ptr++);
        }
    }
    EXPECT_EQ(0.0f, leaf->minimum());
    EXPECT_EQ(1.234f, leaf->maximum());

    { // test stand-alone implementation
        auto localBBox = [](const LeafT* leaf) {
            // static_assert(8u == LeafT::dim(), "Expected dim = 8");
            nanovdb::CoordBBox bbox(nanovdb::Coord(-1, 0, 0), nanovdb::Coord(-1, 7, 7));
            uint64_t           word64 = 0u;
            for (int i = 0; i < 8; ++i) {
                if (uint64_t w = leaf->valueMask().getWord<uint64_t>(i)) {
                    word64 |= w;
                    if (bbox[0][0] == -1)
                        bbox[0][0] = i; // only set once
                    bbox[1][0] = i;
                }
            }
            assert(word64);
            if (word64 == ~uint64_t(0))
                return bbox; // early out of dense leaf
            bbox[0][1] = nanovdb::FindLowestOn(word64) >> 3;
            bbox[1][1] = nanovdb::FindHighestOn(word64) >> 3;
            const uint32_t *p = reinterpret_cast<const uint32_t*>(&word64), word32 = p[0] | p[1];
            const uint16_t *q = reinterpret_cast<const uint16_t*>(&word32), word16 = q[0] | q[1];
            const uint8_t * b = reinterpret_cast<const uint8_t*>(&word16), byte = b[0] | b[1];
            assert(byte);
            bbox[0][2] = nanovdb::FindLowestOn(uint32_t(byte));
            bbox[1][2] = nanovdb::FindHighestOn(uint32_t(byte));
            return bbox;
        }; // bboxOp

        // test
        leaf->data()->mValueMask.setOff();
        const nanovdb::Coord min(1, 2, 3), max(5, 6, 7);
        leaf->setValue(min, 1.0f);
        leaf->setValue(max, 2.0f);
        EXPECT_EQ(1.0f, leaf->getValue(min));
        EXPECT_EQ(2.0f, leaf->getValue(max));
        const auto bbox = localBBox(leaf);
        //std::cerr << "bbox = " << bbox << std::endl;
        EXPECT_EQ(bbox[0], min);
        EXPECT_EQ(bbox[1], max);
    }

    { // test LeafNode::updateBBox
        leaf->data()->mValueMask.setOff();
        const nanovdb::Coord min(1, 2, 3), max(5, 6, 7);
        leaf->setValue(min, 1.0f);
        leaf->setValue(max, 2.0f);
        EXPECT_EQ(1.0f, leaf->getValue(min));
        EXPECT_EQ(2.0f, leaf->getValue(max));
        leaf->updateBBox();
        const auto bbox = leaf->bbox();
        //std::cerr << "bbox = " << bbox << std::endl;
        EXPECT_EQ(bbox[0], min);
        EXPECT_EQ(bbox[1], max);
    }

} // LeafNode

TEST_F(TestNanoVDB, LeafNodeBool)
{
    using LeafT = nanovdb::LeafNode<bool>;
    EXPECT_EQ(8u, LeafT::dim());
    EXPECT_EQ(512u, LeafT::voxelCount());
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(8 * 8 + // mValueMask
                                                       8 * 8 + // mMask
                                                       3 * 4 + // mBBoxMin
                                                       4 * 1), // mBBoxDif[3] + mFlags
              sizeof(LeafT));

    // allocate buffer
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[LeafT::memUsage()]);
    LeafT*                     leaf = reinterpret_cast<LeafT*>(buffer.get());

    { // set members of the leaf node
        auto& data = *reinterpret_cast<LeafT::DataType*>(buffer.get());
        data.mValueMask.setOff();
        data.mValues.setOn();

        for (uint32_t i = 256; i < LeafT::voxelCount(); ++i) {
            data.mValueMask.setOn(i);
            data.mValues.setOff(i);
        }
    }

    // check values
    for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
        if (i < 256) {
            EXPECT_FALSE(leaf->valueMask().isOn(i));
            EXPECT_TRUE(leaf->getValue(i));
        } else {
            EXPECT_TRUE(leaf->valueMask().isOn(i));
            EXPECT_FALSE(leaf->getValue(i));
        }
    }
    EXPECT_EQ(false, leaf->minimum());
    EXPECT_EQ(false, leaf->maximum());
} // LeafNodeBool

TEST_F(TestNanoVDB, LeafNodeValueMask)
{
    using LeafT = nanovdb::LeafNode<nanovdb::ValueMask>;
    //EXPECT_TRUE(LeafT::IgnoreValues);
    EXPECT_EQ(8u, LeafT::dim());
    EXPECT_EQ(512u, LeafT::voxelCount());
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(8 * 8 + // mValueMask
                                                       3 * 4 + // mBBoxMin
                                                       4 * 1), // mBBoxDif[3] + mFlags
              sizeof(LeafT));

    // allocate buffer
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[LeafT::memUsage()]);
    LeafT*                     leaf = reinterpret_cast<LeafT*>(buffer.get());

    { // set members of the leaf node
        auto& data = *reinterpret_cast<LeafT::DataType*>(buffer.get());
        data.mValueMask.setOff();

        for (uint32_t i = 256; i < LeafT::voxelCount(); ++i) {
            data.mValueMask.setOn(i);
        }
    }

    // check values
    for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
        if (i < 256) {
            EXPECT_FALSE(leaf->valueMask().isOn(i));
            EXPECT_FALSE(leaf->getValue(i));
        } else {
            EXPECT_TRUE(leaf->valueMask().isOn(i));
            EXPECT_TRUE(leaf->getValue(i));
        }
    }
    EXPECT_EQ(false, leaf->minimum());
    EXPECT_EQ(false, leaf->maximum());
} // LeafNodeValueMask

TEST_F(TestNanoVDB, InternalNode)
{
    using LeafT = nanovdb::LeafNode<float>;
    using NodeT = nanovdb::InternalNode<LeafT>;
    EXPECT_EQ(8 * 16u, NodeT::dim());
    //         2 x bit-masks         tiles    Vmin&Vmax offset + bbox + padding
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(size_t(2 * (16 * 16 * 16 / 64) * 8 + 16 * 16 * 16 * 8 + 2 * 4 + 4 + 2 * 3 * 4 + 4)), NodeT::memUsage());

    // an empty InternalNode
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[NodeT::memUsage()]);
    NodeT*                     node = reinterpret_cast<NodeT*>(buffer.get());

    { // set members of the node
        auto& data = *reinterpret_cast<NodeT::DataType*>(buffer.get());
        data.mValueMask.setOff();
        data.mChildMask.setOff();
        auto* tiles = data.mTable;
        for (uint32_t i = 0; i < NodeT::SIZE / 2; ++i, ++tiles)
            tiles->value = 0.0f;
        for (uint32_t i = NodeT::SIZE / 2; i < NodeT::SIZE; ++i, ++tiles) {
            data.mValueMask.setOn(i);
            tiles->value = 1.234f;
        }
        data.mMinimum = 0.0f;
        data.mMaximum = 1.234f;
    }

    // check values
    auto* ptr = reinterpret_cast<NodeT::DataType*>(buffer.get())->mTable;
    for (uint32_t i = 0; i < NodeT::SIZE; ++i, ++ptr) {
        EXPECT_FALSE(node->childMask().isOn(i));
        if (i < NodeT::SIZE / 2) {
            EXPECT_FALSE(node->valueMask().isOn(i));
            EXPECT_EQ(0.0f, ptr->value);
        } else {
            EXPECT_TRUE(node->valueMask().isOn(i));
            EXPECT_EQ(1.234f, ptr->value);
        }
    }
    EXPECT_EQ(0.0f, node->minimum());
    EXPECT_EQ(1.234f, node->maximum());
} //  InternalNode

TEST_F(TestNanoVDB, InternalNodeValueMask)
{
    using LeafT = nanovdb::LeafNode<nanovdb::ValueMask>;
    using NodeT = nanovdb::InternalNode<LeafT>;
    //EXPECT_TRUE(LeafT::IgnoreValues);
    //EXPECT_TRUE(NodeT::IgnoreValues);
    EXPECT_EQ(8 * 16u, NodeT::dim());

    /*
    BBox<CoordT> mBBox; // 24B. node bounding box.
    uint64_t     mFlags; // 8B. node flags.
    MaskT        mValueMask; // LOG2DIM(5): 4096B, LOG2DIM(4): 512B
    MaskT        mChildMask; // LOG2DIM(5): 4096B, LOG2DIM(4): 512B

    ValueT mMinimum;
    ValueT mMaximum;
    alignas(32) Tile mTable[1u << (3 * LOG2DIM)];
    */
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(size_t(24 + 4 + 4 + 512 + 512 + 4 + 4 + (16 * 16 * 16) * 8)), NodeT::memUsage());

    // an empty InternalNode
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[NodeT::memUsage()]);
    NodeT*                     node = reinterpret_cast<NodeT*>(buffer.get());

    { // set members of the node
        auto& data = *reinterpret_cast<NodeT::DataType*>(buffer.get());
        data.mValueMask.setOff();
        data.mChildMask.setOff();
        auto* tiles = data.mTable;
        for (uint32_t i = 0; i < NodeT::SIZE / 2; ++i, ++tiles)
            tiles->value = 0u;
        for (uint32_t i = NodeT::SIZE / 2; i < NodeT::SIZE; ++i, ++tiles) {
            data.mValueMask.setOn(i);
            tiles->value = 1u;
        }
        data.mMinimum = 0u;
        data.mMaximum = 1u;
    }

    // check values
    auto* ptr = reinterpret_cast<NodeT::DataType*>(buffer.get())->mTable;
    for (uint32_t i = 0; i < NodeT::SIZE; ++i, ++ptr) {
        EXPECT_FALSE(node->childMask().isOn(i));
        if (i < NodeT::SIZE / 2) {
            EXPECT_FALSE(node->valueMask().isOn(i));
            EXPECT_EQ(0u, ptr->value);
        } else {
            EXPECT_TRUE(node->valueMask().isOn(i));
            EXPECT_EQ(1u, ptr->value);
        }
    }
    EXPECT_EQ(0u, node->minimum());
    EXPECT_EQ(1u, node->maximum());
} //  InternalNodeValueMask

TEST_F(TestNanoVDB, RootNode)
{
    using NodeT0 = nanovdb::LeafNode<float>;
    using NodeT1 = nanovdb::InternalNode<NodeT0>;
    using NodeT2 = nanovdb::InternalNode<NodeT1>;
    using NodeT3 = nanovdb::RootNode<NodeT2>;
    using CoordT = NodeT3::CoordType;
    using KeyT   = NodeT3::DataType::KeyT;

    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(sizeof(nanovdb::CoordBBox) + sizeof(uint32_t) + (5 * sizeof(float))), NodeT3::memUsage(0));

    // an empty RootNode
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[NodeT3::memUsage(0)]);
    NodeT3*                    root = reinterpret_cast<NodeT3*>(buffer.get());

    { // set members of the node
        auto& data = *reinterpret_cast<NodeT3::DataType*>(buffer.get());
        data.mBackground = data.mMinimum = data.mMaximum = 1.234f;
        data.mTableSize = 0;
    }

    EXPECT_EQ(1.234f, root->background());
    EXPECT_EQ(1.234f, root->minimum());
    EXPECT_EQ(1.234f, root->maximum());
    EXPECT_EQ(0u, root->tileCount());
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(sizeof(nanovdb::CoordBBox) + sizeof(uint32_t) + (5 * sizeof(float))), root->memUsage()); // background, min, max, tileCount + bbox
    EXPECT_EQ(1.234f, root->getValue(CoordT(1, 2, 3)));

    { // test RootData::CoordToKey and RotData::KetToCoord
        const int dim = NodeT2::DIM;// dimension of the root's child nodes
        EXPECT_EQ(4096, dim);
        auto coordToKey = [](int i, int j, int k) { return NodeT3::DataType::CoordToKey(CoordT(i, j, k)); };
        auto keyToCoord = [](KeyT key) { return NodeT3::DataType::KeyToCoord(key); };
        EXPECT_TRUE(coordToKey(0, 0, 0) <  coordToKey(dim, 0, 0));
        EXPECT_TRUE(coordToKey(0, 0, 0) <  coordToKey(0, dim, 0));
        EXPECT_TRUE(coordToKey(0, 0, 0) <  coordToKey(0, 0, dim));
        EXPECT_TRUE(coordToKey(0, 0, 0) == coordToKey(dim-1, dim-1, dim-1));
#ifdef USE_SINGLE_ROOT_KEY
        EXPECT_TRUE((std::is_same<uint64_t, KeyT>::value));

        EXPECT_EQ(uint64_t(0), coordToKey(0, 0, 0));
        EXPECT_EQ(uint64_t(0), coordToKey(dim-1, dim-1, dim-1));

        EXPECT_EQ(uint64_t(1), coordToKey(0, 0, dim));
        EXPECT_EQ(uint64_t(1), coordToKey(dim-1, dim-1, dim));

        EXPECT_EQ(uint64_t(1)<<21, coordToKey(0, dim, 0));
        EXPECT_EQ(uint64_t(1)<<21, coordToKey(dim-1, dim, dim-1));

        EXPECT_EQ(uint64_t(1)<<42, coordToKey(dim, 0, 0));
        EXPECT_EQ(uint64_t(1)<<42, coordToKey(dim, dim-1, dim-1));

        EXPECT_EQ(CoordT(0,0,0), keyToCoord(0u));
        //std::cerr << "keyToCoord(1u) = " << keyToCoord(1u) << std::endl;
        EXPECT_EQ(CoordT(0, 0, dim), keyToCoord(1u));
        EXPECT_EQ(CoordT(0, dim, 0), keyToCoord(uint64_t(1)<<21));
        EXPECT_EQ(CoordT(dim, 0, 0), keyToCoord(uint64_t(1)<<42));
#endif
    }
} // RootNode

TEST_F(TestNanoVDB, Offsets)
{
    { // check GridData memory alignment, total 672 bytes
    /*
        static const int MaxNameSize = 256;// due to NULL termination the maximum length is one less
        uint64_t         mMagic; // 8B magic to validate it is valid grid data.
        uint64_t         mChecksum; // 8B. Checksum of grid buffer.
        Version          mVersion;// 4B major, minor, and patch version numbers
        uint32_t         mFlags; // 4B. flags for grid.
        uint32_t         mGridIndex;// 4B. Index of this grid in the buffer
        uint32_t         mGridCount; // 4B. Total number of grids in the buffer
        uint64_t         mGridSize; // 8B. byte count of this entire grid occupied in the buffer.
        char             mGridName[MaxNameSize]; // 256B
        Map              mMap; // 264B. affine transformation between index and world space in both single and double precision
        BBox<Vec3R>      mWorldBBox; // 48B. floating-point AABB of active values in WORLD SPACE (2 x 3 doubles)
        Vec3R            mVoxelSize; // 24B. size of a voxel in world units
        GridClass        mGridClass; // 4B.
        GridType         mGridType; //  4B.
        int64_t          mBlindMetadataOffset; // 8B. offset of GridBlindMetaData structures that follow this grid.
        uint32_t         mBlindMetadataCount; // 4B. count of GridBlindMetaData structures that follow this grid.
    */
        int offset = 0;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mMagic), offset);
        offset += 8;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mChecksum), offset);
        offset += 8;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mVersion), offset);
        offset += 4;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mFlags), offset);
        offset += 4;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridIndex), offset);
        offset += 4;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridCount), offset);
        offset += 4;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridSize), offset);
        offset += 8;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridName), offset);
        offset += 256;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mMap), offset);
        offset += 264;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mWorldBBox), offset);
        offset += 48;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mVoxelSize), offset);
        offset += 24;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridClass), offset);
        offset += 4;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridType), offset);
        offset += 4;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mBlindMetadataOffset), offset);
        offset += 8;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mBlindMetadataCount), offset);
        offset += 4;
        offset = nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(offset);
        //std::cerr << "GridData: Offset = " << offset << std::endl;
        EXPECT_EQ(offset, (int)sizeof(nanovdb::GridData));
    }
    {// check TreeData memory alignment, total 64 bytes
        /*
            uint64_t mNodeOffset[4];//32B, byte offset from this tree to first leaf, lower, upper and root node
            uint32_t mNodeCount[3];// 12B, total number of nodes of type: leaf, lower internal, upper internal
            uint32_t mTileCount[3];// 12B, total number of tiles of type: leaf, lower internal, upper internal (node, only active tiles!)
            uint64_t mVoxelCount;//    8B, total number of active voxels in the root and all its child nodes.
        */
        int offset = 0;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::TreeData<>, mNodeOffset), offset);
        offset += 4*8;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::TreeData<>, mNodeCount), offset);
        offset += 12;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::TreeData<>, mTileCount), offset);
        offset += 12;
        EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::TreeData<>, mVoxelCount), offset);
        offset += 8;
        offset = nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(offset);
        //std::cerr << "TreeData: Offset = " << offset << std::endl;
        EXPECT_EQ(offset, (int)sizeof(nanovdb::TreeData<>));
    }
}

template <typename ValueT>
void checkLeaf(int &offset);

TYPED_TEST(TestOffsets, NanoVDB)
{
    using ValueType = typename nanovdb::BuildToValueMap<TypeParam>::Type;
    using T = typename nanovdb::TensorTraits<ValueType>::ElementType;
    using StatsT = typename nanovdb::FloatTraits<ValueType>::FloatType;
    static const size_t ALIGNMENT = sizeof(T) > sizeof(StatsT) ? sizeof(T) : sizeof(StatsT);
    //std::cerr << "Alignment = " << ALIGNMENT << " sizeof(ValueType) = " << sizeof(ValueType) << std::endl;
    {// check memory layout of RootData
        using DataT = typename nanovdb::NanoRoot<ValueType>::DataType;
        bool test = nanovdb::is_same<StatsT, typename DataT::StatsT>::value;
        EXPECT_TRUE(test);
        int offsets[] = {
            NANOVDB_OFFSETOF(DataT, mBBox),
            NANOVDB_OFFSETOF(DataT, mTableSize),
            NANOVDB_OFFSETOF(DataT, mBackground),
            NANOVDB_OFFSETOF(DataT, mMinimum),
            NANOVDB_OFFSETOF(DataT, mMaximum),
            NANOVDB_OFFSETOF(DataT, mAverage),
            NANOVDB_OFFSETOF(DataT, mStdDevi)
        };
        //for (int i : offsets) std::cout << i << " ";
        const int *p = offsets;
        int offset = 0;// first data member
        EXPECT_EQ(*p++, offset);// mBBox
        offset += 24;// 2 * 3 * 4 bytes = 24 bytes
        EXPECT_EQ(*p++, offset);// mTableSize
        offset += sizeof(uint32_t);
        offset = nanovdb::AlignUp<ALIGNMENT>(offset);
        EXPECT_EQ(*p++, offset);// mBackground
        offset += sizeof(ValueType);
        EXPECT_EQ(*p++, offset);// mMinimum
        offset += sizeof(ValueType);
        EXPECT_EQ(*p++, offset);// mMaximum
        offset += sizeof(ValueType);
        offset = nanovdb::AlignUp<ALIGNMENT>(offset);
        EXPECT_EQ(*p++, offset);// mAverage
        offset += sizeof(StatsT);
        offset = nanovdb::AlignUp<ALIGNMENT>(offset);
        EXPECT_EQ(*p++, offset);// mStdDevi
        offset += sizeof(StatsT);
        offset = nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(offset);
        EXPECT_EQ(offset, (int)sizeof(DataT));// size of RootData
    }
    {// check  memory layout of upper internal nodes
        using DataT = typename nanovdb::NanoUpper<ValueType>::DataType;
        bool test = nanovdb::is_same<StatsT, typename DataT::StatsT>::value;
        EXPECT_TRUE(test);
        int offsets[] = {
            NANOVDB_OFFSETOF(DataT, mBBox),
            NANOVDB_OFFSETOF(DataT, mFlags),
            NANOVDB_OFFSETOF(DataT, mValueMask),
            NANOVDB_OFFSETOF(DataT, mChildMask),
            NANOVDB_OFFSETOF(DataT, mMinimum),
            NANOVDB_OFFSETOF(DataT, mMaximum),
            NANOVDB_OFFSETOF(DataT, mAverage),
            NANOVDB_OFFSETOF(DataT, mStdDevi),
            NANOVDB_OFFSETOF(DataT, mTable),
        };
        //for (int i : offsets) std::cout << i << " ";
        int offset = 0, *p = offsets;
        EXPECT_EQ(*p++, offset);
        offset += 24;
        EXPECT_EQ(*p++, offset);
        offset += 8;
        EXPECT_EQ(*p++, offset);
        offset += 4096;// = 32*32*32/8
        EXPECT_EQ(*p++, offset);
        offset += 4096;// = 32*32*32/8
        EXPECT_EQ(*p++, offset);
        offset += sizeof(ValueType);
        EXPECT_EQ(*p++, offset);
        offset += sizeof(ValueType);
        offset = nanovdb::AlignUp<ALIGNMENT>(offset);
        EXPECT_EQ(*p++, offset);
        offset += sizeof(StatsT);
        offset = nanovdb::AlignUp<ALIGNMENT>(offset);
        EXPECT_EQ(*p++, offset);
        offset += sizeof(StatsT);
        offset = nanovdb::AlignUp<32>(offset);
        EXPECT_EQ(*p++, offset);
        const size_t tile_size = nanovdb::AlignUp<8>(sizeof(ValueType));
        EXPECT_EQ(sizeof(typename DataT::Tile), tile_size);
        offset += (32*32*32)*tile_size;
        offset = nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(offset);
        EXPECT_EQ(sizeof(DataT), (size_t)offset);
    }
    {// check  memory lower of upper internal nodes
        using DataT = typename nanovdb::NanoLower<ValueType>::DataType;
        bool test = nanovdb::is_same<StatsT, typename DataT::StatsT>::value;
        EXPECT_TRUE(test);
        int offsets[] = {
            NANOVDB_OFFSETOF(DataT, mBBox),
            NANOVDB_OFFSETOF(DataT, mFlags),
            NANOVDB_OFFSETOF(DataT, mValueMask),
            NANOVDB_OFFSETOF(DataT, mChildMask),
            NANOVDB_OFFSETOF(DataT, mMinimum),
            NANOVDB_OFFSETOF(DataT, mMaximum),
            NANOVDB_OFFSETOF(DataT, mAverage),
            NANOVDB_OFFSETOF(DataT, mStdDevi),
            NANOVDB_OFFSETOF(DataT, mTable),
        };
        //for (int i : offsets) std::cout << i << " ";
        int offset = 0, *p = offsets;
        EXPECT_EQ(*p++, offset);
        offset += 24;
        EXPECT_EQ(*p++, offset);
        offset += 8;
        EXPECT_EQ(*p++, offset);
        offset += 512;// = 16*16*16/8
        EXPECT_EQ(*p++, offset);
        offset += 512;// = 16*16*16/8
        EXPECT_EQ(*p++, offset);
        offset += sizeof(ValueType);
        EXPECT_EQ(*p++, offset);
        offset += sizeof(ValueType);
        offset = nanovdb::AlignUp<ALIGNMENT>(offset);
        EXPECT_EQ(*p++, offset);
        offset += sizeof(StatsT);
        offset = nanovdb::AlignUp<ALIGNMENT>(offset);
        EXPECT_EQ(*p++, offset);
        offset += sizeof(StatsT);
        offset = nanovdb::AlignUp<32>(offset);
        EXPECT_EQ(*p++, offset);
        const size_t tile_size = nanovdb::AlignUp<8>(sizeof(ValueType));
        EXPECT_EQ(sizeof(typename DataT::Tile), tile_size);
        offset += (16*16*16)*tile_size;
        offset = nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(offset);
        EXPECT_EQ(sizeof(DataT), (size_t)offset);
    }
    {// check  memory of leaf nodes
        using DataT = typename nanovdb::LeafNode<ValueType>::DataType;
        bool test = nanovdb::is_same<StatsT, typename DataT::FloatType>::value;
        EXPECT_TRUE(test);
        int offsets[] = {
            NANOVDB_OFFSETOF(DataT, mBBoxMin),
            NANOVDB_OFFSETOF(DataT, mBBoxDif),
            NANOVDB_OFFSETOF(DataT, mFlags),
            NANOVDB_OFFSETOF(DataT, mValueMask),
        };
        //for (int i : offsets) std::cout << i << " ";
        int offset = 0, *p = offsets;
        EXPECT_EQ(*p++, offset);
        offset += 12;
        EXPECT_EQ(*p++, offset);
        offset += 3;
        EXPECT_EQ(*p++, offset);
        offset += 1;
        EXPECT_EQ(*p++, offset);
        offset += 64;// = 8*8*8/8
        checkLeaf<ValueType>(offset);
        offset = nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(offset);
        EXPECT_EQ(sizeof(DataT), (size_t)offset);
    }
}// TestOffsets NanoVDB

template<typename ValueType>
void checkLeaf(int &offset)
{
    using DataT = typename nanovdb::LeafNode<ValueType>::DataType;
    using T = typename nanovdb::TensorTraits<ValueType>::ElementType;
    using StatsT = typename nanovdb::FloatTraits<ValueType>::FloatType;
    static const size_t ALIGNMENT = sizeof(T) > sizeof(StatsT) ? sizeof(T) : sizeof(StatsT);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMinimum), offset);
    offset += sizeof(ValueType);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMaximum), offset);
    offset += sizeof(ValueType);
    offset = nanovdb::AlignUp<ALIGNMENT>(offset);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mAverage), offset);
    offset += sizeof(StatsT);
    offset = nanovdb::AlignUp<ALIGNMENT>(offset);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mStdDevi), offset);
    offset += sizeof(StatsT);
    offset = nanovdb::AlignUp<32>(offset);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mValues), offset);
    offset += (8*8*8)*sizeof(ValueType);
}

template<>
void checkLeaf<bool>(int &offset)
{
    using DataT = typename nanovdb::LeafNode<bool>::DataType;
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mValues), offset);
    offset += 64;// = 8*8*8/8
}

template<>
void checkLeaf<nanovdb::Fp4>(int &offset)
{
    using DataT = typename nanovdb::LeafNode<nanovdb::Fp4>::DataType;
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMinimum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mQuantum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMin), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMax), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mAvg), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mDev), offset);
    offset += sizeof(uint16_t);
    offset = nanovdb::AlignUp<32>(offset);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mCode), offset);
    offset += 256*sizeof(uint8_t);
}

template<>
void checkLeaf<nanovdb::Fp8>(int &offset)
{
    using DataT = typename nanovdb::LeafNode<nanovdb::Fp8>::DataType;
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMinimum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mQuantum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMin), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMax), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mAvg), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mDev), offset);
    offset += sizeof(uint16_t);
    offset = nanovdb::AlignUp<32>(offset);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mCode), offset);
    offset += 512*sizeof(uint8_t);
}

template<>
void checkLeaf<nanovdb::Fp16>(int &offset)
{
    using DataT = typename nanovdb::LeafNode<nanovdb::Fp16>::DataType;
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMinimum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mQuantum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMin), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMax), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mAvg), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mDev), offset);
    offset += sizeof(uint16_t);
    offset = nanovdb::AlignUp<32>(offset);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mCode), offset);
    offset += 512*sizeof(uint16_t);
}

template<>
void checkLeaf<nanovdb::FpN>(int &offset)
{
    using DataT = typename nanovdb::LeafNode<nanovdb::FpN>::DataType;
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMinimum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mQuantum), offset);
    offset += sizeof(float);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMin), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mMax), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mAvg), offset);
    offset += sizeof(uint16_t);
    EXPECT_EQ(NANOVDB_OFFSETOF(DataT, mDev), offset);
    offset += sizeof(uint16_t);
    offset = nanovdb::AlignUp<32>(offset);
}

template<>
void checkLeaf<nanovdb::ValueMask>(int &) {}

TEST_F(TestNanoVDB, BasicGrid)
{
    using LeafT  = nanovdb::LeafNode<float>;
    using NodeT1 = nanovdb::InternalNode<LeafT>;
    using NodeT2 = nanovdb::InternalNode<NodeT1>;
    using RootT  = nanovdb::RootNode<NodeT2>;
    using TreeT  = nanovdb::Tree<RootT>;
    using GridT  = nanovdb::Grid<TreeT>;
    using CoordT = LeafT::CoordType;

    const std::string name("test name");
    {
        // This is just for visual inspection
        /*
        this->printType<GridT>("Grid");
        this->printType<TreeT>("Tree");
        this->printType<NodeT2>("Upper InternalNode");
        this->printType<NodeT1>("Lower InternalNode");
        this->printType<LeafT>("Leaf");
        Old: W/O mAverage and mVariance
            Size of Grid: 672 bytes which is 32 byte aligned
            Size of Tree: 64 bytes which is 32 byte aligned
            Size of Upper InternalNode: 139328 bytes which is 32 byte aligned
            Size of Lower InternalNode: 17472 bytes which is 32 byte aligned
            Size of Leaf: 2144 bytes which is 32 byte aligned

        New: WITH mAverage and mVariance
            Size of Grid: 672 bytes which is 32 byte aligned
            Size of Tree: 64 bytes which is 32 byte aligned
            Size of Upper InternalNode: 139328 bytes which is 32 byte aligned
            Size of Lower InternalNode: 17472 bytes which is 32 byte aligned
            Size of Leaf: 2144 bytes which is 32 byte aligned
        */
    }

    EXPECT_EQ(sizeof(GridT), nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(8 + 8 + 4 + 4 + 8 + nanovdb::GridData::MaxNameSize + 48 + sizeof(nanovdb::Map) + 24 + 4 + 4 + 8 + 4));
    EXPECT_EQ(sizeof(TreeT), nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(4*8 + 3*4 + 3*4 + 8));
    EXPECT_EQ(sizeof(TreeT), size_t(4*8 + 3*4 + 3*4 + 8));// should already be 32 byte aligned

    size_t bytes[6] = {GridT::memUsage(), TreeT::memUsage(), RootT::memUsage(1), NodeT2::memUsage(), NodeT1::memUsage(), LeafT::memUsage()};
    for (int i = 1; i < 6; ++i)
        bytes[i] += bytes[i - 1]; // Byte offsets to: tree, root, internal nodes, leafs, total
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[bytes[5]]);

    // init leaf
    LeafT* leaf = reinterpret_cast<LeafT*>(buffer.get() + bytes[4]);
    { // set members of the leaf node
        auto* data = leaf->data();
        data->mValueMask.setOff();
        auto* voxels = data->mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount() / 2; ++i)
            *voxels++ = 0.0f;
        for (uint32_t i = LeafT::voxelCount() / 2; i < LeafT::voxelCount(); ++i) {
            data->mValueMask.setOn(i);
            *voxels++ = 1.0f;
        }
        data->mMinimum = 1.0f;
        data->mMaximum = 1.0f;
    }

    // lower internal node
    NodeT1* node1 = reinterpret_cast<NodeT1*>(buffer.get() + bytes[3]);
    { // set members of the  internal node
        auto *data = node1->data();
        data->mValueMask.setOff();
        data->mChildMask.setOff();
        data->mChildMask.setOn(0);
        data->setChild(0, leaf);
        for (uint32_t i = 1; i < NodeT1::SIZE / 2; ++i)
            data->mTable[i].value = 0.0f;
        for (uint32_t i = NodeT1::SIZE / 2; i < NodeT1::SIZE; ++i) {
            data->mValueMask.setOn(i);
            data->mTable[i].value = 2.0f;
        }
        data->mMinimum = 1.0f;
        data->mMaximum = 2.0f;
        EXPECT_EQ(leaf, data->getChild(0));
    }

    // upper internal node
    NodeT2* node2 = reinterpret_cast<NodeT2*>(buffer.get() + bytes[2]);
    { // set members of the  internal node
        auto *data = node2->data();
        data->mValueMask.setOff();
        data->mChildMask.setOff();
        data->mChildMask.setOn(0);
        data->setChild(0, node1);
        for (uint32_t i = 1; i < NodeT2::SIZE / 2; ++i)
            data->mTable[i].value = 0.0f;
        for (uint32_t i = NodeT2::SIZE / 2; i < NodeT2::SIZE; ++i) {
            data->mValueMask.setOn(i);
            data->mTable[i].value = 3.0f;
        }
        data->mMinimum = 1.0f;
        data->mMaximum = 3.0f;
        EXPECT_EQ(node1, data->getChild(0));
    }

    // init root
    RootT* root = reinterpret_cast<RootT*>(buffer.get() + bytes[1]);
    { // set members of the root node
        auto* data = root->data();
        data->mBackground = 0.0f;
        data->mMinimum = 1.0f;
        data->mMaximum = 3.0f;
        data->mTableSize = 1;
        data->tile(0)->setChild(RootT::CoordType(0), node2, data);
    }

    // init tree
    TreeT* tree = reinterpret_cast<TreeT*>(buffer.get() + bytes[0]);
    {
        auto* data = tree->data();
        data->setRoot(root);
        data->setFirstNode(node2);
        data->setFirstNode(node1);
        data->setFirstNode(leaf);
        data->mNodeCount[0] = data->mNodeCount[1] = data->mNodeCount[2] = 1;
    }

    GridT* grid = reinterpret_cast<GridT*>(buffer.get());
    { // init Grid
        auto* data = grid->data();
        {
            const double dx = 2.0, Tx = 0.0, Ty = 0.0, Tz = 0.0;
            const double mat[4][4] = {
                {dx, 0.0, 0.0, 0.0}, // row 0
                {0.0, dx, 0.0, 0.0}, // row 1
                {0.0, 0.0, dx, 0.0}, // row 2
                {Tx, Ty, Tz, 1.0}, // row 3
            };
            const double invMat[4][4] = {
                {1 / dx, 0.0, 0.0, 0.0}, // row 0
                {0.0, 1 / dx, 0.0, 0.0}, // row 1
                {0.0, 0.0, 1 / dx, 0.0}, // row 2
                {-Tx, -Ty, -Tz, 1.0}, // row 3
            };
            data->setFlagsOff();
            data->setMinMaxOn();
            data->mGridIndex = 0;
            data->mGridCount = 1;
            data->mBlindMetadataOffset = 0;
            data->mBlindMetadataCount = 0;
            data->mVoxelSize = nanovdb::Vec3R(dx);
            data->mMap.set(mat, invMat, 1.0);
            data->mGridClass = nanovdb::GridClass::Unknown;
            data->mGridType = nanovdb::GridType::Float;
            data->mMagic = NANOVDB_MAGIC_NUMBER;
            data->mVersion = nanovdb::Version();
            memcpy(data->mGridName, name.c_str(), name.size() + 1);
        }

        EXPECT_EQ(tree, &grid->tree());
        const nanovdb::Vec3R p1(1.0, 2.0, 3.0);
        const auto           p2 = grid->worldToIndex(p1);
        EXPECT_EQ(nanovdb::Vec3R(0.5, 1.0, 1.5), p2);
        const auto p3 = grid->indexToWorld(p2);
        EXPECT_EQ(p1, p3);
        {
            const double dx = 2.0, Tx = p1[0], Ty = p1[1], Tz = p1[2];
            const double mat[4][4] = {
                {dx, 0.0, 0.0, 0.0}, // row 0
                {0.0, dx, 0.0, 0.0}, // row 1
                {0.0, 0.0, dx, 0.0}, // row 2
                {Tx, Ty, Tz, 1.0}, // row 3
            };
            const double invMat[4][4] = {
                {1 / dx, 0.0, 0.0, 0.0}, // row 0
                {0.0, 1 / dx, 0.0, 0.0}, // row 1
                {0.0, 0.0, 1 / dx, 0.0}, // row 2
                {-1 / Tx, -1 / Ty, -1 / Tz, 1.0}, // row 3
            };
            data->mVoxelSize = nanovdb::Vec3R(dx);
            data->mMap.set(mat, invMat, 1.0);
        }


        // Start actual tests

        auto const p4 = grid->worldToIndex(p3);
        EXPECT_EQ(nanovdb::Vec3R(0.0, 0.0, 0.0), p4);
        const auto p5 = grid->indexToWorld(p4);
        EXPECT_EQ(p1, p5);
    }

    { // check leaf node
        auto* ptr = reinterpret_cast<LeafT::DataType*>(buffer.get() + bytes[4])->mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
            if (i < 256) {
                EXPECT_FALSE(leaf->valueMask().isOn(i));
                EXPECT_EQ(0.0f, *ptr++);
            } else {
                EXPECT_TRUE(leaf->valueMask().isOn(i));
                EXPECT_EQ(1.0f, *ptr++);
            }
        }
        EXPECT_EQ(1.0f, leaf->minimum());
        EXPECT_EQ(1.0f, leaf->maximum());
        EXPECT_EQ(0.0f, leaf->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, leaf->getValue(CoordT(8-1)));
    }

    { // check lower internal node
        auto& data = *reinterpret_cast<NodeT1::DataType*>(buffer.get() + bytes[3]);
        EXPECT_TRUE(node1->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT1::SIZE; ++i) {
            EXPECT_FALSE(node1->childMask().isOn(i));
            if (i < NodeT1::SIZE / 2) {
                EXPECT_FALSE(node1->valueMask().isOn(i));
                EXPECT_EQ(0.0f, data.mTable[i].value);
            } else {
                EXPECT_TRUE(node1->valueMask().isOn(i));
                EXPECT_EQ(2.0f, data.mTable[i].value);
            }
        }
        EXPECT_EQ(1.0f, node1->minimum());
        EXPECT_EQ(2.0f, node1->maximum());
        EXPECT_EQ(0.0f, node1->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, node1->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, node1->getValue(CoordT(8*16-1)));
    }
    { // check upper internal node
        auto& data = *reinterpret_cast<NodeT2::DataType*>(buffer.get() + bytes[2]);
        EXPECT_TRUE(node2->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT2::SIZE; ++i) {
            EXPECT_FALSE(node2->childMask().isOn(i));
            if (i < NodeT2::SIZE / 2) {
                EXPECT_FALSE(node2->valueMask().isOn(i));
                EXPECT_EQ(0.0f, data.mTable[i].value);
            } else {
                EXPECT_TRUE(node2->valueMask().isOn(i));
                EXPECT_EQ(3.0f, data.mTable[i].value);
            }
        }
        EXPECT_EQ(1.0f, node2->minimum());
        EXPECT_EQ(3.0f, node2->maximum());
        EXPECT_EQ(0.0f, node2->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, node2->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, node2->getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, node2->getValue(CoordT(8*16*32-1)));
    }
    { // check root
        EXPECT_EQ(0.0f, root->background());
        EXPECT_EQ(1.0f, root->minimum());
        EXPECT_EQ(3.0f, root->maximum());
        EXPECT_EQ(1u,   root->tileCount());
        EXPECT_EQ(0.0f, root->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, root->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, root->getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, root->getValue(CoordT(8*16*32-1)));
    }
    { // check tree
        EXPECT_EQ(0.0f, tree->background());
        float a, b;
        tree->extrema(a, b);
        EXPECT_EQ(1.0f, a);
        EXPECT_EQ(3.0f, b);
        EXPECT_EQ(0.0f, tree->getValue(CoordT(0)));
        EXPECT_EQ(1.0f, tree->getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, tree->getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, tree->getValue(CoordT(8*16*32-1)));
        EXPECT_EQ(1u, tree->nodeCount<LeafT>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT1>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT2>());
    }
    {// check grid
        EXPECT_EQ(nanovdb::Version(), grid->version());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), grid->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), grid->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), grid->version().getPatch());
        EXPECT_TRUE(grid->isValid());
        EXPECT_EQ(grid->gridType(), nanovdb::GridType::Float);
        EXPECT_EQ(grid->gridClass(), nanovdb::GridClass::Unknown);
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isStaggered());
        EXPECT_FALSE(grid->isPointIndex());
        EXPECT_FALSE(grid->isPointData());
        EXPECT_FALSE(grid->isMask());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_TRUE(grid->hasMinMax());
        EXPECT_FALSE(grid->hasBBox());
        EXPECT_FALSE(grid->hasLongGridName());
        EXPECT_FALSE(grid->hasAverage());
        EXPECT_FALSE(grid->hasStdDeviation());
        //std::cerr << "\nName = \"" << grid->gridName() << "\"" << std::endl;
        EXPECT_EQ(name, std::string(grid->gridName()));
    }
    {// check ReadAccessor
        auto acc = grid->getAccessor();
        EXPECT_EQ(0.0f, acc.getValue(CoordT(0)));
        EXPECT_EQ(1.0f, acc.getValue(CoordT(8-1)));
        EXPECT_EQ(2.0f, acc.getValue(CoordT(8*16-1)));
        EXPECT_EQ(3.0f, acc.getValue(CoordT(8*16*32-1)));
        EXPECT_FALSE(acc.isActive(CoordT(0)));
        EXPECT_TRUE(acc.isActive(CoordT(8-1)));
        EXPECT_TRUE(acc.isActive(CoordT(16*8-1)));
        EXPECT_TRUE(acc.isActive(CoordT(32*16*8-1)));
    }
} // BasicGrid

TEST_F(TestNanoVDB, GridBuilderEmpty)
{
    { // empty grid
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        auto                        handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_TRUE(meta->isEmpty());
        EXPECT_EQ("test", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("test", std::string(dstGrid->gridName()));
        EXPECT_EQ(0u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_EQ(0.0f, srcAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_FALSE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(dstGrid->tree().root().minimum(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().maximum(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().average(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().variance(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().stdDeviation(), 0.0f);
    }
} // GridBuilderEmpty

TEST_F(TestNanoVDB, GridBuilderBasic1)
{
    { // 1 grid point
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(1, 2, 3), 1.0f);
        EXPECT_EQ(1.0f, srcAcc.getValue(nanovdb::Coord(1, 2, 3)));
        auto handle = builder.getHandle<>();
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        //EXPECT_EQ(1u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(nanovdb::Coord(1, 2, 3), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord(1, 2, 3), dstGrid->indexBBox()[1]);
        EXPECT_EQ(dstGrid->tree().root().minimum(), 1.0f);// minimum active value
        EXPECT_EQ(dstGrid->tree().root().maximum(), 1.0f);// maximum active value
        EXPECT_NEAR(dstGrid->tree().root().average(), 1.0f, 1e-6);
        EXPECT_NEAR(dstGrid->tree().root().variance(), 0.0f,1e-6);
        EXPECT_NEAR(dstGrid->tree().root().stdDeviation(), 0.0f, 1e-6);
    }
} // GridBuilderBasic1

TEST_F(TestNanoVDB, GridBuilderBasic2)
{
    { // 2 grid points
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(1, 2, 3),  1.0f);
        srcAcc.setValue(nanovdb::Coord(2, -2, 9),-1.0f);
        //srcAcc.setValue(nanovdb::Coord(20,-20,90), 0.0f);// same as background
        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        //EXPECT_TRUE(meta->version().isValid());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("test", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("test", std::string(dstGrid->gridName()));
        EXPECT_EQ(2u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_EQ( 1.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(-1.0f, dstAcc.getValue(nanovdb::Coord(2, -2, 9)));

        const nanovdb::BBox<nanovdb::Vec3R> indexBBox = dstGrid->indexBBox();
        EXPECT_DOUBLE_EQ( 1.0, indexBBox[0][0]);
        EXPECT_DOUBLE_EQ(-2.0, indexBBox[0][1]);
        EXPECT_DOUBLE_EQ( 3.0, indexBBox[0][2]);
        EXPECT_DOUBLE_EQ( 3.0, indexBBox[1][0]);
        EXPECT_DOUBLE_EQ( 3.0, indexBBox[1][1]);
        EXPECT_DOUBLE_EQ(10.0, indexBBox[1][2]);

        EXPECT_EQ(nanovdb::Coord(1, -2, 3), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord(2, 2, 9), dstGrid->indexBBox()[1]);

        EXPECT_EQ(dstGrid->tree().root().minimum(),-1.0f);
        EXPECT_EQ(dstGrid->tree().root().maximum(), 1.0f);
        EXPECT_NEAR(dstGrid->tree().root().average(), 0.0f, 1e-6);
        EXPECT_NEAR(dstGrid->tree().root().variance(),1.0f, 1e-6);// Sim (x_i - Avg)^2/N = ((-1)^2 + 1^2)/2 =  1
        EXPECT_NEAR(dstGrid->tree().root().stdDeviation(), 1.0f, 1e-6);// stdDev = Sqrt(var)
    }
} // GridBuilderBasic2

TEST_F(TestNanoVDB, GridBuilderPrune)
{
    {
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        const nanovdb::CoordBBox    bbox(nanovdb::Coord(0), nanovdb::Coord(8*16-1));
        auto                        func = [](const nanovdb::Coord&) { return 1.0f; };
        //auto                        func = [](const nanovdb::Coord&, float &v) { v = 1.0f; return true; };
        builder(func, bbox);
        for (auto ijk = bbox.begin(); ijk; ++ijk) {
            EXPECT_EQ(1.0f, srcAcc.getValue(*ijk));
            EXPECT_TRUE(srcAcc.isActive(*ijk));
        }

        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        //EXPECT_TRUE(meta->version().isValid());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("test", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("test", std::string(dstGrid->gridName()));
        EXPECT_EQ(512*16*16*16u, dstGrid->activeVoxelCount());//
        auto dstAcc = dstGrid->getAccessor();

        for (nanovdb::Coord ijk = bbox[0]; ijk[0] <= bbox[1][0]; ++ijk[0]) {
            for (ijk[1] = bbox[0][1]; ijk[1] <= bbox[1][1]; ++ijk[1]) {
                for (ijk[2] = bbox[0][2]; ijk[2] <= bbox[1][2]; ++ijk[2]) {
                    EXPECT_EQ(1.0f, dstAcc.getValue(ijk));
                }
            }
        }
        EXPECT_EQ( 0.0f, dstAcc.getValue(nanovdb::Coord(2, -2, 9)));

        const nanovdb::BBox<nanovdb::Vec3R> indexBBox = dstGrid->indexBBox();
        EXPECT_DOUBLE_EQ(   0.0, indexBBox[0][0]);
        EXPECT_DOUBLE_EQ(   0.0, indexBBox[0][1]);
        EXPECT_DOUBLE_EQ(   0.0, indexBBox[0][2]);
        EXPECT_DOUBLE_EQ(8*16.0, indexBBox[1][0]);
        EXPECT_DOUBLE_EQ(8*16.0, indexBBox[1][1]);
        EXPECT_DOUBLE_EQ(8*16.0, indexBBox[1][2]);

        EXPECT_EQ(nanovdb::Coord(0), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord(8*16-1), dstGrid->indexBBox()[1]);

        EXPECT_EQ(0u, dstGrid->tree().nodeCount(0));// all pruned away
        EXPECT_EQ(0u, dstGrid->tree().nodeCount(1));// all pruned away
        EXPECT_EQ(1u, dstGrid->tree().nodeCount(2));

        EXPECT_EQ(dstGrid->tree().root().minimum(), 1.0f);
        EXPECT_EQ(dstGrid->tree().root().maximum(), 1.0f);
        EXPECT_NEAR(dstGrid->tree().root().average(), 1.0f, 1e-6);
        EXPECT_NEAR(dstGrid->tree().root().variance(),0.0f, 1e-6);// Sim (x_i - Avg)^2/N = 0
        EXPECT_NEAR(dstGrid->tree().root().stdDeviation(), 0.0f, 1e-6);// stdDev = Sqrt(var)
    }
} // GridBuilderPrune

TEST_F(TestNanoVDB, GridBuilder_Vec3f)
{
    using VoxelT = nanovdb::Vec3f;
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(12 + 3 + 1 + 2*4 + 64 + 3*(2*4 + 512*4)), sizeof(nanovdb::NanoLeaf<VoxelT>));
    { // 3 grid point
        nanovdb::GridBuilder<VoxelT> builder(nanovdb::Vec3f(0.0f));
        auto srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(  1,  2,  3), nanovdb::Vec3f(1.0f));
        srcAcc.setValue(nanovdb::Coord(-10, 20,-50), nanovdb::Vec3f(2.0f));
        srcAcc.setValue(nanovdb::Coord( 50,-12, 30), nanovdb::Vec3f(3.0f));
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isValueOn(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(nanovdb::Vec3f(1.0f), srcAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(nanovdb::Vec3f(2.0f), srcAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(nanovdb::Vec3f(3.0f), srcAcc.getValue(nanovdb::Coord( 50,-12, 30)));

        builder.setStats(nanovdb::StatsMode::All);
        auto handle = builder.getHandle();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::mapToGridType<VoxelT>(), meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(nanovdb::Vec3f(1.0f), dstGrid->tree().root().minimum());
        EXPECT_EQ(nanovdb::Vec3f(3.0f), dstGrid->tree().root().maximum());
        EXPECT_EQ((nanovdb::Vec3f(1.0f).lengthSqr() +
                   nanovdb::Vec3f(2.0f).lengthSqr() +
                   nanovdb::Vec3f(3.0f).lengthSqr())/3.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(1, 2, 3));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->data()->mBBoxMin);

        EXPECT_EQ(nanovdb::Vec3f(1.0f), dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(nanovdb::Vec3f(1.0f), dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(nanovdb::Vec3f(2.0f), dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(nanovdb::Vec3f(3.0f), dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        //std::cerr << dstGrid->indexBBox() << std::endl;
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
} // GridBuilder_Vec3f

TEST_F(TestNanoVDB, GridBuilder_Vec4f)
{
    using VoxelT = nanovdb::Vec4f;
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(12 + 3 + 1 + 2*4 + 64 + 4*(2*4 + 512*4)), sizeof(nanovdb::NanoLeaf<VoxelT>));
    { // 3 grid point
        nanovdb::GridBuilder<VoxelT> builder(nanovdb::Vec4f(0.0f));
        auto srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(  1,  2,  3), nanovdb::Vec4f(1.0f));
        srcAcc.setValue(nanovdb::Coord(-10, 20,-50), nanovdb::Vec4f(2.0f));
        srcAcc.setValue(nanovdb::Coord( 50,-12, 30), nanovdb::Vec4f(3.0f));
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isValueOn(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(nanovdb::Vec4f(1.0f), srcAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(nanovdb::Vec4f(2.0f), srcAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(nanovdb::Vec4f(3.0f), srcAcc.getValue(nanovdb::Coord( 50,-12, 30)));

        builder.setStats(nanovdb::StatsMode::All);
        auto handle = builder.getHandle();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::mapToGridType<VoxelT>(), meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(nanovdb::Vec4f(1.0f), dstGrid->tree().root().minimum());
        EXPECT_EQ(nanovdb::Vec4f(3.0f), dstGrid->tree().root().maximum());
        EXPECT_EQ((nanovdb::Vec4f(1.0f).lengthSqr() +
                   nanovdb::Vec4f(2.0f).lengthSqr() +
                   nanovdb::Vec4f(3.0f).lengthSqr())/3.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(1, 2, 3));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->data()->mBBoxMin);

        EXPECT_EQ(nanovdb::Vec4f(1.0f), dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(nanovdb::Vec4f(1.0f), dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(nanovdb::Vec4f(2.0f), dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(nanovdb::Vec4f(3.0f), dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        //std::cerr << dstGrid->indexBBox() << std::endl;
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
} // GridBuilder_Vec4f


TEST_F(TestNanoVDB, GridBuilder_Fp4)
{
    using VoxelT = nanovdb::Fp4;
    EXPECT_EQ(96u + 512u/2, sizeof(nanovdb::NanoLeaf<VoxelT>));
    { // 3 grid point
        nanovdb::GridBuilder<float, VoxelT> builder(0.0f);
        auto srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(nanovdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(nanovdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isValueOn(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(nanovdb::Coord( 50,-12, 30)));

        builder.setStats(nanovdb::StatsMode::All);
        auto handle = builder.getHandle();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::mapToGridType<VoxelT>(), meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(1, 2, 3));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->data()->mBBoxMin);
        //const auto offset = nanovdb::NanoLeaf<nanovdb::Fp4>::CoordToOffset(nanovdb::Coord(1, 2, 3));
        //std::cerr << "offset = " << offset << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1]) << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1] >> 4) << std::endl;

        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        //std::cerr << dstGrid->indexBBox() << std::endl;
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
    {// Sphere
        const double voxelSize = 0.1, halfWidth = 3.0;
        const float radius = 10.0f;
        const nanovdb::Vec3f center(0);
        const nanovdb::Vec3d origin(0);
        const float tolerance = 0.5f * voxelSize;

        auto handle = nanovdb::createLevelSetSphere<float, VoxelT>(radius, center,
                                                                   voxelSize, halfWidth,
                                                                   origin, "sphere",
                                                                   nanovdb::StatsMode::Default,
                                                                   nanovdb::ChecksumMode::Default,
                                                                   tolerance,
                                                                   false);
        auto* nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);
        Sphere<float> sphere(center, radius, float(voxelSize), float(halfWidth));
        auto kernel = [&](const nanovdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const nanovdb::Coord p = *it;
                EXPECT_NEAR(nanoAcc.getValue(p), sphere(p), tolerance);
            }
        };
        nanovdb::forEach(nanoGrid->indexBBox(), kernel);

        nanovdb::io::writeGrid("data/sphere_fp4.nvdb", handle);
        handle = nanovdb::io::readGrid("data/sphere_fp4.nvdb");
        nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);

        nanovdb::forEach(nanoGrid->indexBBox(), kernel);
    }
} // GridBuilder_Fp4

TEST_F(TestNanoVDB, GridBuilder_Fp8)
{
    using VoxelT = nanovdb::Fp8;
    EXPECT_EQ(96u + 512u, sizeof(nanovdb::NanoLeaf<VoxelT>));
    { // 3 grid point
        nanovdb::GridBuilder<float, VoxelT> builder(0.0f);
        auto srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(nanovdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(nanovdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isValueOn(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(nanovdb::Coord( 50,-12, 30)));

        builder.setStats(nanovdb::StatsMode::All);
        auto handle = builder.getHandle();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::mapToGridType<VoxelT>(), meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(1, 2, 3));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->data()->mBBoxMin);
        //const auto offset = nanovdb::NanoLeaf<nanovdb::Fp4>::CoordToOffset(nanovdb::Coord(1, 2, 3));
        //std::cerr << "offset = " << offset << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1]) << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1] >> 4) << std::endl;

        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        //std::cerr << dstGrid->indexBBox() << std::endl;
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
    {// Sphere
        const double voxelSize = 0.1, halfWidth = 3.0;
        const float radius = 10.0f;
        const nanovdb::Vec3f center(0);
        const nanovdb::Vec3d origin(0);
        const float tolerance = 0.05f * voxelSize;

        auto handle = nanovdb::createLevelSetSphere<float, VoxelT>(radius, center,
                                                                   voxelSize, halfWidth,
                                                                   origin, "sphere",
                                                                   nanovdb::StatsMode::Default,
                                                                   nanovdb::ChecksumMode::Default,
                                                                   tolerance,
                                                                   false);
        auto* nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);
        Sphere<float> sphere(center, radius, float(voxelSize), float(halfWidth));
        auto kernel = [&](const nanovdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const nanovdb::Coord p = *it;
                EXPECT_NEAR(nanoAcc.getValue(p), sphere(p), tolerance);
            }
        };
        nanovdb::forEach(nanoGrid->indexBBox(), kernel);

        nanovdb::io::writeGrid("data/sphere_fp8.nvdb", handle);
        handle = nanovdb::io::readGrid("data/sphere_fp8.nvdb");
        nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);

        nanovdb::forEach(nanoGrid->indexBBox(), kernel);
    }
} // GridBuilder_Fp8

TEST_F(TestNanoVDB, GridBuilder_Fp16)
{
    using VoxelT = nanovdb::Fp16;
    EXPECT_EQ(96u + 512u*2, sizeof(nanovdb::NanoLeaf<VoxelT>));
    { // 3 grid point
        nanovdb::GridBuilder<float, VoxelT> builder(0.0f);
        auto srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(nanovdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(nanovdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isValueOn(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(nanovdb::Coord( 50,-12, 30)));

        builder.setStats(nanovdb::StatsMode::All);
        auto handle = builder.getHandle();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::mapToGridType<VoxelT>(), meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_TRUE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(1, 2, 3));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->data()->mBBoxMin);
        //const auto offset = nanovdb::NanoLeaf<nanovdb::Fp4>::CoordToOffset(nanovdb::Coord(1, 2, 3));
        //std::cerr << "offset = " << offset << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1]) << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1] >> 4) << std::endl;

        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        //std::cerr << dstGrid->indexBBox() << std::endl;
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
    {// Sphere
        const double voxelSize = 0.1, halfWidth = 3.0;
        const float radius = 10.0f;
        const nanovdb::Vec3f center(0);
        const nanovdb::Vec3d origin(0);
        const float tolerance = 0.005f * voxelSize;

        auto handle = nanovdb::createLevelSetSphere<float, VoxelT>(radius, center,
                                                                   voxelSize, halfWidth,
                                                                   origin, "sphere",
                                                                   nanovdb::StatsMode::Default,
                                                                   nanovdb::ChecksumMode::Default,
                                                                   tolerance,
                                                                   false);
        auto* nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);
        Sphere<float> sphere(center, radius, float(voxelSize), float(halfWidth));
        auto kernel = [&](const nanovdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const nanovdb::Coord p = *it;
                EXPECT_NEAR(nanoAcc.getValue(p), sphere(p), tolerance);
            }
        };
        nanovdb::forEach(nanoGrid->indexBBox(), kernel);

        nanovdb::io::writeGrid("data/sphere_fp16.nvdb", handle);
        handle = nanovdb::io::readGrid("data/sphere_fp16.nvdb");
        nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);

        nanovdb::forEach(nanoGrid->indexBBox(), kernel);
    }
} // GridBuilder_Fp16

TEST_F(TestNanoVDB, GridBuilder_FpN_Basic1)
{
    using VoxelT = nanovdb::FpN;
    EXPECT_EQ(96u, sizeof(nanovdb::NanoLeaf<VoxelT>));
    { // 1 grid point
        nanovdb::GridBuilder<float, VoxelT> builder(0.0f);
        auto srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(  0,  0,  0), 1.0f);
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(0, 0, 0)));
        EXPECT_TRUE(srcAcc.isValueOn(nanovdb::Coord(0, 0, 0)));
        EXPECT_EQ(1.0f, srcAcc.getValue(nanovdb::Coord(  0,  0,  0)));

        builder.setStats(nanovdb::StatsMode::All);
        //builder.setVerbose(true);
        auto handle = builder.getHandle();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::mapToGridType<VoxelT>(), meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(0, 0, 0)));
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(1.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(1.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_FALSE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_FALSE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(0, 0, 0));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->data()->mBBoxMin);

        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(0, 0, 0)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(0, 0, 0)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord( 0,  0,  0)));
        EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        //std::cerr << dstGrid->indexBBox() << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord(0,0,0), dstGrid->indexBBox()[1]);
    }
}// GridBuilder_FpN_Basic1

TEST_F(TestNanoVDB, GridBuilder_FpN_Basic3)
{
    using VoxelT = nanovdb::FpN;
    EXPECT_EQ(96u, sizeof(nanovdb::NanoLeaf<VoxelT>));
    { // 3 grid point
        nanovdb::GridBuilder<float, VoxelT> builder(0.0f);
        auto srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(  1,  2,  3), 1.0f);
        srcAcc.setValue(nanovdb::Coord(-10, 20,-50), 2.0f);
        srcAcc.setValue(nanovdb::Coord( 50,-12, 30), 3.0f);
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isValueOn(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, srcAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, srcAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, srcAcc.getValue(nanovdb::Coord( 50,-12, 30)));

        builder.setStats(nanovdb::StatsMode::All);
        //builder.setVerbose(true);
        auto handle = builder.getHandle();

        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::mapToGridType<VoxelT>(), meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("", std::string(dstGrid->gridName()));
        EXPECT_EQ((const char*)handle.data(), (const char*)dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().root().minimum());
        EXPECT_EQ(3.0f, dstGrid->tree().root().maximum());
        EXPECT_EQ(2.0f, dstGrid->tree().root().average());
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_FALSE(dstGrid->isSequential<GridT::TreeType::Node0>());
        EXPECT_TRUE(dstGrid->isSequential<2>());
        EXPECT_TRUE(dstGrid->isSequential<1>());
        EXPECT_FALSE(dstGrid->isSequential<0>());

        EXPECT_EQ(nanovdb::Vec3R(1.0), dstGrid->voxelSize());
        auto *leaf = dstGrid->tree().root().probeLeaf(nanovdb::Coord(1, 2, 3));
        EXPECT_TRUE(leaf);
        //std::cerr << leaf->origin() << ", " << leaf->data()->mBBoxMin << std::endl;
        EXPECT_EQ(nanovdb::Coord(0,0,0), leaf->origin());
        EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->data()->mBBoxMin);
        //const auto offset = nanovdb::NanoLeaf<nanovdb::Fp4>::CoordToOffset(nanovdb::Coord(1, 2, 3));
        //std::cerr << "offset = " << offset << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1]) << std::endl;
        //std::cerr << "code = " <<  int(leaf->data()->mCode[offset>>1] >> 4) << std::endl;

        EXPECT_EQ(1.0f, dstGrid->tree().getValue(nanovdb::Coord(1, 2, 3)));
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(  1,  2,  3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(-10, 20,-50)));
        EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord( 50,-12, 30)));
        //std::cerr << dstGrid->indexBBox() << std::endl;
        EXPECT_EQ(nanovdb::Coord(-10,-12,-50), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord( 50, 20, 30), dstGrid->indexBBox()[1]);
    }
}// GridBuilder_FpN_Basic3

TEST_F(TestNanoVDB, GridBuilder_FpN_Sphere)
{
    using VoxelT = nanovdb::FpN;
    EXPECT_EQ(96u, sizeof(nanovdb::NanoLeaf<VoxelT>));
    {// Sphere
        const double voxelSize = 0.1, halfWidth = 3.0;
        const float radius = 10.0f;
        const nanovdb::Vec3f center(0);
        const nanovdb::Vec3d origin(0);
        const float tolerance = 0.5f * voxelSize;

        auto handle = nanovdb::createLevelSetSphere<float, VoxelT>(radius, center,
                                                                   voxelSize, halfWidth,
                                                                   origin, "sphere",
                                                                   nanovdb::StatsMode::Default,
                                                                   nanovdb::ChecksumMode::Default,
                                                                   tolerance,
                                                                   false);
        auto* nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);
        Sphere<float> sphere(center, radius, float(voxelSize), float(halfWidth));
        auto kernel = [&](const nanovdb::CoordBBox& bbox) {
            auto nanoAcc = nanoGrid->getAccessor();
            for (auto it = bbox.begin(); it; ++it) {
                const nanovdb::Coord p = *it;
                EXPECT_NEAR(nanoAcc.getValue(p), sphere(p), tolerance);
            }
        };
        nanovdb::forEach(nanoGrid->indexBBox(), kernel);

        nanovdb::io::writeGrid("data/sphere_fpN.nvdb", handle);
        handle = nanovdb::io::readGrid("data/sphere_fpN.nvdb");
        nanoGrid = handle.grid<VoxelT>();
        EXPECT_TRUE(nanoGrid);

        nanovdb::forEach(nanoGrid->indexBBox(), kernel);
    }
} // GridBuilder_FpN_Sphere

TEST_F(TestNanoVDB, NodeManager)
{
     { // 1 active voxel
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        builder.setGridClass(nanovdb::GridClass::LevelSet);
        const nanovdb::Coord x0(1, 2, 3), x1(1, 2, 4);
        srcAcc.setValue(x1, 1.0f);
        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
        EXPECT_TRUE(handle);
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());

        auto nodeMgr = nanovdb::createNodeMgr(*dstGrid);
        auto leafMgr = nanovdb::createLeafMgr(*dstGrid);
        EXPECT_FALSE(nodeMgr.empty());
        EXPECT_FALSE(leafMgr.empty());
        EXPECT_EQ(1u, nodeMgr.nodeCount(2));
        EXPECT_EQ(1u, nodeMgr.nodeCount(1));
        EXPECT_EQ(1u, nodeMgr.nodeCount(0));
        EXPECT_EQ(1u, leafMgr.size());

        EXPECT_EQ(0.0f, nodeMgr.grid()->tree().getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.tree()->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.root()->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.upper(0)->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.lower(0)->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.leaf(0)->getValue(x0));
        EXPECT_EQ(0.0f, leafMgr[0]->getValue(x0));

        EXPECT_EQ(1.0f, nodeMgr.grid()->tree().getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.tree()->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.root()->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.upper(0)->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.lower(0)->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.leaf(0)->getValue(x1));
        EXPECT_EQ(1.0f, leafMgr[0]->getValue(x1));

        EXPECT_EQ(leafMgr[0], nodeMgr.leaf(0));
        EXPECT_EQ(leafMgr[0], dstGrid->tree().getFirstNode< 0 >());
        EXPECT_EQ(leafMgr[0], dstGrid->tree().getFirstNode< nanovdb::NanoLeaf<float> >());
        EXPECT_EQ(nodeMgr.leaf(0),  dstGrid->tree().getFirstNode< nanovdb::NanoLeaf< float> >());
        EXPECT_EQ(nodeMgr.lower(0), dstGrid->tree().getFirstNode< nanovdb::NanoLower<float> >());
        EXPECT_EQ(nodeMgr.upper(0), dstGrid->tree().getFirstNode< nanovdb::NanoUpper<float> >());
        EXPECT_EQ(nodeMgr.leaf(0),  dstGrid->tree().getFirstNode< 0 >());
        EXPECT_EQ(nodeMgr.lower(0), dstGrid->tree().getFirstNode< 1 >());
        EXPECT_EQ(nodeMgr.upper(0), dstGrid->tree().getFirstNode< 2 >());
    }
    { // 2 active voxels
        nanovdb::GridBuilder<float> builder(0.0f, nanovdb::GridClass::LevelSet);
        auto                        srcAcc = builder.getAccessor();
        const nanovdb::Coord x0(1, 2, 3), x1(2,-2, 9), x2(1, 2, 4);
        srcAcc.setValue(x1, 1.0f);
        srcAcc.setValue(x2, 2.0f);
        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
        EXPECT_TRUE(handle);
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ(1.0f, dstGrid->tree().getValue(x1));
        EXPECT_EQ(2.0f, dstGrid->tree().getValue(x2));
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());

        nanovdb::NodeManager<nanovdb::FloatGrid> nodeMgr;
        EXPECT_TRUE(nodeMgr.empty());
        nanovdb::NodeManager<nanovdb::FloatGrid> tmp(*dstGrid);
        EXPECT_FALSE(tmp.empty());
        //mgr = tmp;// fails to compile since assignment operator is deleted
        nodeMgr = std::move(tmp);
        auto leafMgr = nanovdb::createLeafMgr(*dstGrid);
        EXPECT_FALSE(nodeMgr.empty());
        EXPECT_FALSE(leafMgr.empty());
        EXPECT_EQ(2u, nodeMgr.nodeCount(2));
        EXPECT_EQ(2u, nodeMgr.nodeCount(1));
        EXPECT_EQ(2u, nodeMgr.nodeCount(0));
        EXPECT_EQ(2u, leafMgr.size());

        EXPECT_EQ(0.0f, nodeMgr.grid()->tree().getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.tree()->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.root()->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.upper(0)->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.lower(0)->getValue(x0));
        EXPECT_EQ(0.0f, nodeMgr.leaf(0)->getValue(x0));
        EXPECT_EQ(0.0f, leafMgr[0]->getValue(x0));

        EXPECT_EQ(1.0f, nodeMgr.grid()->tree().getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.tree()->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.root()->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.upper(0)->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.lower(0)->getValue(x1));
        EXPECT_EQ(1.0f, nodeMgr.leaf(0)->getValue(x1));
        EXPECT_EQ(1.0f, leafMgr[0]->getValue(x1));

        EXPECT_EQ(2.0f, nodeMgr.grid()->tree().getValue(x2));
        EXPECT_EQ(2.0f, nodeMgr.tree()->getValue(x2));
        EXPECT_EQ(2.0f, nodeMgr.root()->getValue(x2));
        EXPECT_EQ(2.0f, nodeMgr.upper(1)->getValue(x2));
        EXPECT_EQ(2.0f, nodeMgr.lower(1)->getValue(x2));
        EXPECT_EQ(2.0f, nodeMgr.leaf(1)->getValue(x2));
        EXPECT_EQ(2.0f, leafMgr[1]->getValue(x2));

        EXPECT_EQ(leafMgr[0], nodeMgr.leaf(0));
        EXPECT_EQ(leafMgr[1], nodeMgr.leaf(1));
    }
    {// random points
        const size_t voxelCount = 512;
        const int min = -10000, max = 10000;
        std::vector<nanovdb::Coord> voxels;
        std::srand(98765);
        auto op = [&](){return rand() % (max - min) + min;};
        while (voxels.size() <  voxelCount) {
            const nanovdb::Coord ijk(op(), op(), op());
            if (voxels.end() == std::find(voxels.begin(), voxels.end(), ijk)) {
                voxels.push_back(ijk);
            }
        }
        EXPECT_EQ(voxelCount, voxels.size());
        nanovdb::GridBuilder<float> builder(-1.0f, nanovdb::GridClass::LevelSet);
        auto                        srcAcc = builder.getAccessor();
        for (size_t i=0; i<voxelCount; ++i) {
            srcAcc.setValue(voxels[i], float(i));
        }
        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
        EXPECT_TRUE(handle);
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_TRUE(dstGrid->isBreadthFirst());
        using GridT = std::remove_pointer<decltype(dstGrid)>::type;
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node2>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node1>());
        EXPECT_TRUE(dstGrid->isSequential<GridT::TreeType::Node0>());

        auto nodeMgr = nanovdb::createNodeMgr(*dstGrid);
        auto leafMgr = nanovdb::createLeafMgr(*dstGrid);
        EXPECT_FALSE(nodeMgr.empty());
        EXPECT_FALSE(leafMgr.empty());
        EXPECT_EQ(nodeMgr.nodeCount(0), leafMgr.size());
        auto dstAcc = dstGrid->getAccessor();
        for (size_t i=0; i<voxelCount; ++i) {
            EXPECT_EQ(float(i), dstAcc.getValue(voxels[i]));
            EXPECT_EQ(nodeMgr.leaf(i), leafMgr[i]);
        }
    }
} // NodeManager

TEST_F(TestNanoVDB, GridBuilderBasicDense)
{
    { // dense functor
        nanovdb::GridBuilder<float> builder(0.0f, nanovdb::GridClass::LevelSet);
        auto                        srcAcc = builder.getAccessor();
        const nanovdb::CoordBBox    bbox(nanovdb::Coord(0), nanovdb::Coord(100));
        auto                        func = [](const nanovdb::Coord&) { return 1.0f; };
        //auto                        func = [](const nanovdb::Coord&, float &v) { v = 1.0f; return true; };
        builder(func, bbox);
        for (auto ijk = bbox.begin(); ijk; ++ijk) {
            EXPECT_EQ(1.0f, srcAcc.getValue(*ijk));
            EXPECT_TRUE(srcAcc.isActive(*ijk));
        }
        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_FALSE(meta->isEmpty());
        //EXPECT_TRUE(meta->version().isValid());
        EXPECT_EQ(uint32_t(NANOVDB_MAJOR_VERSION_NUMBER), meta->version().getMajor());
        EXPECT_EQ(uint32_t(NANOVDB_MINOR_VERSION_NUMBER), meta->version().getMinor());
        EXPECT_EQ(uint32_t(NANOVDB_PATCH_VERSION_NUMBER), meta->version().getPatch());
        EXPECT_EQ("test", std::string(meta->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ("test", std::string(dstGrid->gridName()));
        const nanovdb::Coord dim = bbox.dim();
        EXPECT_EQ(nanovdb::Coord(101), dim);
        EXPECT_EQ(101u * 101u * 101u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        for (nanovdb::Coord ijk = bbox[0]; ijk[0] <= bbox[1][0]; ++ijk[0]) {
            for (ijk[1] = bbox[0][1]; ijk[1] <= bbox[1][1]; ++ijk[1]) {
                for (ijk[2] = bbox[0][2]; ijk[2] <= bbox[1][2]; ++ijk[2]) {
                    EXPECT_EQ(1.0f, dstAcc.getValue(ijk));
                }
            }
        }
        EXPECT_EQ(bbox[0], dstGrid->indexBBox()[0]);
        EXPECT_EQ(bbox[1], dstGrid->indexBBox()[1]);

        EXPECT_EQ(dstGrid->tree().root().minimum(), 1.0f);// smallest active value
        EXPECT_EQ(dstGrid->tree().root().maximum(), 1.0f);// largest active value
        EXPECT_EQ(dstGrid->tree().root().average(),  1.0f);
        EXPECT_EQ(dstGrid->tree().root().variance(), 0.0f);
        EXPECT_EQ(dstGrid->tree().root().stdDeviation(), 0.0f);
    }
} // GridBuilderDense

TEST_F(TestNanoVDB, GridBuilderBackground)
{
    {
        nanovdb::GridBuilder<float> builder(0.5f);
        auto                        acc = builder.getAccessor();

        acc.setValue(nanovdb::Coord(1), 1);
        acc.setValue(nanovdb::Coord(2), 0);

        EXPECT_EQ(0.5f, acc.getValue(nanovdb::Coord(0)));
        EXPECT_FALSE(acc.isActive(nanovdb::Coord(0)));
        EXPECT_EQ(1, acc.getValue(nanovdb::Coord(1)));
        EXPECT_TRUE(acc.isActive(nanovdb::Coord(1)));
        EXPECT_EQ(0, acc.getValue(nanovdb::Coord(2)));
        EXPECT_TRUE(acc.isActive(nanovdb::Coord(1)));

        auto gridHdl = builder.getHandle<>();
        auto grid = gridHdl.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_FALSE(grid->isEmpty());
        EXPECT_EQ(0.5, grid->tree().getValue(nanovdb::Coord(0)));
        EXPECT_EQ(1, grid->tree().getValue(nanovdb::Coord(1)));
        EXPECT_EQ(0, grid->tree().getValue(nanovdb::Coord(2)));
    }
} // GridBuilderBackground

TEST_F(TestNanoVDB, GridBuilderSphere)
{
    Sphere<float> sphere(nanovdb::Vec3<float>(50), 20.0f);
    EXPECT_EQ(3.0f, sphere.background());
    EXPECT_EQ(3.0f, sphere(nanovdb::Coord(100)));
    EXPECT_EQ(-3.0f, sphere(nanovdb::Coord(50)));
    EXPECT_EQ(0.0f, sphere(nanovdb::Coord(50, 50, 70)));
    EXPECT_EQ(-1.0f, sphere(nanovdb::Coord(50, 50, 69)));
    EXPECT_EQ(2.0f, sphere(nanovdb::Coord(50, 50, 72)));

    nanovdb::GridBuilder<float> builder(sphere.background(), nanovdb::GridClass::LevelSet);
    auto                        srcAcc = builder.getAccessor();

    const nanovdb::CoordBBox bbox(nanovdb::Coord(-100), nanovdb::Coord(100));
    //mTimer.start("GridBulder Sphere");
    builder(sphere, bbox);
    //mTimer.stop();


    auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("test", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("test", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    //mTimer.start("GridBulder NodeMananger");
    nanovdb::NodeManager<nanovdb::FloatGrid> mgr(*dstGrid);
    //mTimer.stop();
    EXPECT_EQ(dstGrid->tree().nodeCount(0), mgr.nodeCount(0));
    EXPECT_EQ(dstGrid->tree().nodeCount(1), mgr.nodeCount(1));
    EXPECT_EQ(dstGrid->tree().nodeCount(2), mgr.nodeCount(2));

    //std::cerr << "bbox.min = (" << dstGrid->indexBBox()[0][0] << ", " <<  dstGrid->indexBBox()[0][1] << ", " <<  dstGrid->indexBBox()[0][2] << ")" << std::endl;
    //std::cerr << "bbox.max = (" << dstGrid->indexBBox()[1][0] << ", " <<  dstGrid->indexBBox()[1][1] << ", " <<  dstGrid->indexBBox()[1][2] << ")" << std::endl;

    uint64_t count = 0;
    auto    dstAcc = dstGrid->getAccessor();
    for (nanovdb::Coord ijk = bbox[0]; ijk[0] <= bbox[1][0]; ++ijk[0]) {
        for (ijk[1] = bbox[0][1]; ijk[1] <= bbox[1][1]; ++ijk[1]) {
            for (ijk[2] = bbox[0][2]; ijk[2] <= bbox[1][2]; ++ijk[2]) {
                if (dstAcc.isActive(ijk))
                    ++count;
                EXPECT_EQ(sphere(ijk), dstAcc.getValue(ijk));
                EXPECT_EQ(srcAcc.getValue(ijk), dstAcc.getValue(ijk));
            }
        }
    }

    EXPECT_EQ(count, dstGrid->activeVoxelCount());

} // GridBuilderSphere

TEST_F(TestNanoVDB, createLevelSetSphere)
{
    Sphere<float> sphere(nanovdb::Vec3<float>(50), 20.0f);
    EXPECT_EQ(3.0f, sphere.background());
    EXPECT_EQ(3.0f, sphere(nanovdb::Coord(100)));
    EXPECT_EQ(-3.0f, sphere(nanovdb::Coord(50)));
    EXPECT_EQ(0.0f, sphere(nanovdb::Coord(50, 50, 70)));
    EXPECT_EQ(-1.0f, sphere(nanovdb::Coord(50, 50, 69)));
    EXPECT_EQ(2.0f, sphere(nanovdb::Coord(50, 50, 72)));

    auto handle = nanovdb::createLevelSetSphere(20.0f, nanovdb::Vec3f(50),
                                                1.0, 3.0, nanovdb::Vec3d(0), "sphere_20");

    const nanovdb::CoordBBox bbox(nanovdb::Coord(0), nanovdb::Coord(100));

    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("sphere_20", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("sphere_20", std::string(dstGrid->gridName()));

    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    EXPECT_NEAR( -3.0f, dstGrid->tree().root().minimum(), 0.04f);
    EXPECT_NEAR(  3.0f, dstGrid->tree().root().maximum(), 0.04f);
    EXPECT_NEAR(  0.0f, dstGrid->tree().root().average(), 0.3f);
    //std::cerr << dstGrid->tree().root().minimum() << std::endl;
    //std::cerr << dstGrid->tree().root().maximum() << std::endl;
    //std::cerr << dstGrid->tree().root().average() << std::endl;
    //std::cerr << dstGrid->tree().root().stdDeviation() << std::endl;


    EXPECT_EQ(nanovdb::Coord(50 - 20 - 2), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 20 + 2), dstGrid->indexBBox()[1]);

    //std::cerr << "bbox.min = (" << dstGrid->indexBBox()[0][0] << ", " <<  dstGrid->indexBBox()[0][1] << ", " <<  dstGrid->indexBBox()[0][2] << ")" << std::endl;
    //std::cerr << "bbox.max = (" << dstGrid->indexBBox()[1][0] << ", " <<  dstGrid->indexBBox()[1][1] << ", " <<  dstGrid->indexBBox()[1][2] << ")" << std::endl;
    uint64_t count = 0;
    auto     dstAcc = dstGrid->getAccessor();
    for (nanovdb::Coord ijk = bbox[0]; ijk[0] <= bbox[1][0]; ++ijk[0]) {
        for (ijk[1] = bbox[0][1]; ijk[1] <= bbox[1][1]; ++ijk[1]) {
            for (ijk[2] = bbox[0][2]; ijk[2] <= bbox[1][2]; ++ijk[2]) {
                if (sphere.inNarrowBand(ijk))
                    ++count;
                EXPECT_EQ(sphere(ijk), dstAcc.getValue(ijk));
                EXPECT_EQ(sphere.inNarrowBand(ijk), dstAcc.isActive(ijk));
            }
        }
    }

    EXPECT_EQ(count, dstGrid->activeVoxelCount());

} // createLevelSetSphere

TEST_F(TestNanoVDB, createFogVolumeSphere)
{
    auto                     handle = nanovdb::createFogVolumeSphere(20.0f, nanovdb::Vec3f(50),
                                                                     1.0, 3.0, nanovdb::Vec3d(0), "sphere_20");
    const nanovdb::CoordBBox bbox(nanovdb::Coord(0), nanovdb::Coord(100));

    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("sphere_20", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::FogVolume, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("sphere_20", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    EXPECT_NEAR(  0.0f, dstGrid->tree().root().minimum(),1e-6);
    EXPECT_NEAR(  1.0f, dstGrid->tree().root().maximum(), 1e-6);
    EXPECT_NEAR(  0.8f, dstGrid->tree().root().average(), 1e-3);
    EXPECT_NEAR(  0.3f, dstGrid->tree().root().stdDeviation(), 1e-2);
    //std::cerr << dstGrid->tree().root().minimum() << std::endl;
    //std::cerr << dstGrid->tree().root().maximum() << std::endl;
    //std::cerr << dstGrid->tree().root().average() << std::endl;
    //std::cerr << dstGrid->tree().root().stdDeviation() << std::endl;

    //std::cerr << "bbox.min = (" << dstGrid->indexBBox()[0][0] << ", " <<  dstGrid->indexBBox()[0][1] << ", " <<  dstGrid->indexBBox()[0][2] << ")" << std::endl;
    //std::cerr << "bbox.max = (" << dstGrid->indexBBox()[1][0] << ", " <<  dstGrid->indexBBox()[1][1] << ", " <<  dstGrid->indexBBox()[1][2] << ")" << std::endl;

    EXPECT_EQ(nanovdb::Coord(50 - 20), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 20), dstGrid->indexBBox()[1]);

    Sphere<float> sphere(nanovdb::Vec3<float>(50), 20.0f);
    uint64_t      count = 0;
    auto          dstAcc = dstGrid->getAccessor();
    for (nanovdb::Coord ijk = bbox[0]; ijk[0] <= bbox[1][0]; ++ijk[0]) {
        for (ijk[1] = bbox[0][1]; ijk[1] <= bbox[1][1]; ++ijk[1]) {
            for (ijk[2] = bbox[0][2]; ijk[2] <= bbox[1][2]; ++ijk[2]) {
                if (sphere(ijk) > 0) {
                    EXPECT_FALSE(dstAcc.isActive(ijk));
                    EXPECT_EQ(0, dstAcc.getValue(ijk));
                } else {
                    ++count;
                    EXPECT_TRUE(dstAcc.isActive(ijk));
                    EXPECT_TRUE(dstAcc.getValue(ijk) >= 0);
                    EXPECT_TRUE(dstAcc.getValue(ijk) <= 1);
                }
            }
        }
    }

    EXPECT_EQ(count, dstGrid->activeVoxelCount());

} // createFogVolumeSphere

TEST_F(TestNanoVDB, createPointSphere)
{
    Sphere<float> sphere(nanovdb::Vec3<float>(0), 100.0f, 1.0f, 1.0f);
    EXPECT_EQ(1.0f, sphere.background());
    EXPECT_EQ(1.0f, sphere(nanovdb::Coord(101, 0, 0)));
    EXPECT_EQ(-1.0f, sphere(nanovdb::Coord(0)));
    EXPECT_EQ(0.0f, sphere(nanovdb::Coord(0, 0, 100)));
    EXPECT_EQ(-1.0f, sphere(nanovdb::Coord(0, 0, 99)));
    EXPECT_EQ(1.0f, sphere(nanovdb::Coord(0, 0, 101)));

    auto handle = nanovdb::createPointSphere<float>(1,// pointer per voxel
                                                    100.0f,// radius of sphere
                                                    nanovdb::Vec3f(0),// center sphere
                                                    1.0,// voxel size
                                                    nanovdb::Vec3d(0),// origin of grid
                                                    "point_sphere");

    const nanovdb::CoordBBox bbox(nanovdb::Coord(-100), nanovdb::Coord(100));

    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("point_sphere", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::PointData, meta->gridClass());
    auto* dstGrid = handle.grid<uint32_t>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("point_sphere", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_FALSE(dstGrid->hasAverage());
    EXPECT_FALSE(dstGrid->hasStdDeviation());

    EXPECT_EQ(bbox[0], dstGrid->indexBBox()[0]);
    EXPECT_EQ(bbox[1], dstGrid->indexBBox()[1]);

    //std::cerr << "bbox.min = (" << dstGrid->indexBBox()[0][0] << ", " <<  dstGrid->indexBBox()[0][1] << ", " <<  dstGrid->indexBBox()[0][2] << ")" << std::endl;
    //std::cerr << "bbox.max = (" << dstGrid->indexBBox()[1][0] << ", " <<  dstGrid->indexBBox()[1][1] << ", " <<  dstGrid->indexBBox()[1][2] << ")" << std::endl;

    uint64_t                               count = 0;
    nanovdb::PointAccessor<nanovdb::Vec3f> acc(*dstGrid);
    const nanovdb::Vec3f *                 begin = nullptr, *end = nullptr;
    for (nanovdb::Coord ijk = bbox[0]; ijk[0] <= bbox[1][0]; ++ijk[0]) {
        for (ijk[1] = bbox[0][1]; ijk[1] <= bbox[1][1]; ++ijk[1]) {
            for (ijk[2] = bbox[0][2]; ijk[2] <= bbox[1][2]; ++ijk[2]) {
                if (nanovdb::Abs(sphere(ijk)) < 0.5f) {
                    ++count;
                    EXPECT_TRUE(acc.isActive(ijk));
                    EXPECT_TRUE(acc.getValue(ijk) != std::numeric_limits<uint32_t>::max());
                    EXPECT_EQ(1u, acc.voxelPoints(ijk, begin, end)); // exactly one point per voxel
                    const nanovdb::Vec3f p = *begin + ijk.asVec3s();// local voxel coordinate + global index coordinates
                    EXPECT_TRUE(nanovdb::Abs(sphere(p)) <= 1.0f);
                } else {
                    EXPECT_FALSE(acc.isActive(ijk));
                    EXPECT_TRUE(acc.getValue(ijk) < 512 || acc.getValue(ijk) == std::numeric_limits<uint32_t>::max());
                }
            }
        }
    }
    EXPECT_EQ(count, dstGrid->activeVoxelCount());
} // createPointSphere

TEST_F(TestNanoVDB, createLevelSetTorus)
{
    auto handle = nanovdb::createLevelSetTorus(100.0f, 50.0f, nanovdb::Vec3f(50),
                                               1.0, 3.0, nanovdb::Vec3d(0), "torus_100");

    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("torus_100", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("torus_100", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    EXPECT_EQ(nanovdb::Coord(50 - 100 - 50 - 2, 50 - 50 - 2, 50 - 100 - 50 - 2), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 100 + 50 + 2, 50 + 50 + 2, 50 + 100 + 50 + 2), dstGrid->indexBBox()[1]);

    auto dstAcc = dstGrid->getAccessor();
    EXPECT_EQ(3.0f, dstAcc.getValue(nanovdb::Coord(50)));
    EXPECT_FALSE(dstAcc.isActive(nanovdb::Coord(50)));
    EXPECT_EQ(-3.0f, dstAcc.getValue(nanovdb::Coord(150, 50, 50)));
    EXPECT_FALSE(dstAcc.isActive(nanovdb::Coord(150, 50, 50)));
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(200, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(200, 50, 50)));
    EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(201, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(201, 50, 50)));
    EXPECT_EQ(-2.0f, dstAcc.getValue(nanovdb::Coord(198, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(198, 50, 50)));

} // createLevelSetTorus

TEST_F(TestNanoVDB, createFogVolumeTorus)
{
    auto handle = nanovdb::createFogVolumeTorus(100.0f, 50.0f, nanovdb::Vec3f(50),
                                                1.0, 3.0, nanovdb::Vec3d(0), "torus_100");

    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("torus_100", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::FogVolume, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("torus_100", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    //std::cerr << "bbox.min = (" << dstGrid->indexBBox()[0][0] << ", " <<  dstGrid->indexBBox()[0][1] << ", " <<  dstGrid->indexBBox()[0][2] << ")" << std::endl;
    //std::cerr << "bbox.max = (" << dstGrid->indexBBox()[1][0] << ", " <<  dstGrid->indexBBox()[1][1] << ", " <<  dstGrid->indexBBox()[1][2] << ")" << std::endl;

    EXPECT_EQ(nanovdb::Coord(50 - 100 - 50, 50 - 50, 50 - 100 - 50), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 100 + 50, 50 + 50, 50 + 100 + 50), dstGrid->indexBBox()[1]);

    auto dstAcc = dstGrid->getAccessor();
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(50)));
    EXPECT_FALSE(dstAcc.isActive(nanovdb::Coord(50)));
    EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(150, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(150, 50, 50)));
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(200, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(200, 50, 50)));
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(201, 50, 50)));
    EXPECT_FALSE(dstAcc.isActive(nanovdb::Coord(201, 50, 50)));
    EXPECT_EQ(1.0f / 3.0f, dstAcc.getValue(nanovdb::Coord(199, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(199, 50, 50)));
    EXPECT_EQ(2.0f / 3.0f, dstAcc.getValue(nanovdb::Coord(198, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(198, 50, 50)));
} // createFogVolumeTorus

TEST_F(TestNanoVDB, createLevelSetBox)
{
    auto handle = nanovdb::createLevelSetBox<float>(40.0f, 60.0f, 80.0f, nanovdb::Vec3f(50),
                                                    1.0, 3.0, nanovdb::Vec3d(0), "box");
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("box", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
     EXPECT_EQ("box", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    EXPECT_EQ(nanovdb::Coord(50 - 20 - 2, 50 - 30 - 2, 50 - 40 - 2), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 20 + 2, 50 + 30 + 2, 50 + 40 + 2), dstGrid->indexBBox()[1]);

    auto dstAcc = dstGrid->getAccessor();
    EXPECT_EQ(-3.0f, dstAcc.getValue(nanovdb::Coord(50)));
    EXPECT_FALSE(dstAcc.isActive(nanovdb::Coord(50)));
    EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(72, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(72, 50, 50)));
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(70, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(70, 50, 50)));
    EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(71, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(71, 50, 50)));
    EXPECT_EQ(-2.0f, dstAcc.getValue(nanovdb::Coord(68, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(68, 50, 50)));

} // createLevelSetBox

TEST_F(TestNanoVDB, createFogVolumeBox)
{
    auto handle = nanovdb::createFogVolumeBox<float>(40.0f, 60.0f, 80.0f, nanovdb::Vec3f(50),
                                                     1.0, 3.0, nanovdb::Vec3d(0), "box");
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("box", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::FogVolume, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("box", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    EXPECT_EQ(nanovdb::Coord(50 - 20, 50 - 30, 50 - 40), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 20, 50 + 30, 50 + 40), dstGrid->indexBBox()[1]);

    auto dstAcc = dstGrid->getAccessor();
    EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(50)));
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(72, 50, 50)));
    EXPECT_FALSE(dstAcc.isActive(nanovdb::Coord(72, 50, 50)));
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(70, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(70, 50, 50)));
    EXPECT_EQ(1.0f / 3.0f, dstAcc.getValue(nanovdb::Coord(69, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(69, 50, 50)));
    EXPECT_EQ(2.0f / 3.0f, dstAcc.getValue(nanovdb::Coord(68, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(68, 50, 50)));

} // createFogVolumeBox

TEST_F(TestNanoVDB, createLevelSetOctahedron)
{
    auto handle = nanovdb::createLevelSetOctahedron<float>(100.0f, nanovdb::Vec3f(50),
                                                           1.0f, 3.0f, nanovdb::Vec3d(0), "octahedron");
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("octahedron", std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ("octahedron", std::string(dstGrid->gridName()));
    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    EXPECT_EQ(nanovdb::Coord(50 - 100/2 - 2), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 100/2 + 2), dstGrid->indexBBox()[1]);

    auto dstAcc = dstGrid->getAccessor();
    EXPECT_EQ(-3.0f, dstAcc.getValue(nanovdb::Coord(50)));
    EXPECT_FALSE(dstAcc.isActive(nanovdb::Coord(50)));
    EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(102, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(102, 50, 50)));
    EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(100, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(100, 50, 50)));
    EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(101, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(101, 50, 50)));
    EXPECT_EQ(-nanovdb::Sqrt(4.0f/3.0f), dstAcc.getValue(nanovdb::Coord(98, 50, 50)));
    EXPECT_TRUE(dstAcc.isActive(nanovdb::Coord(98, 50, 50)));

} // createLevelSetOctahedron

#if !defined(_MSC_VER)
TEST_F(TestNanoVDB, CNanoVDBSize)
{
    // Verify the sizes of structures are what we expect.
    EXPECT_EQ(sizeof(cnanovdb_mask3), sizeof(nanovdb::Mask<3>));
    EXPECT_EQ(sizeof(cnanovdb_mask4), sizeof(nanovdb::Mask<4>));
    EXPECT_EQ(sizeof(cnanovdb_mask5), sizeof(nanovdb::Mask<5>));
    EXPECT_EQ(sizeof(cnanovdb_map), sizeof(nanovdb::Map));
    EXPECT_EQ(sizeof(cnanovdb_coord), sizeof(nanovdb::Coord));
    EXPECT_EQ(sizeof(cnanovdb_Vec3F), sizeof(nanovdb::Vec3f));

    EXPECT_EQ(sizeof(cnanovdb_node0F), sizeof(nanovdb::LeafNode<float>));
    EXPECT_EQ(sizeof(cnanovdb_node1F), sizeof(nanovdb::InternalNode<nanovdb::LeafNode<float>>));
    EXPECT_EQ(sizeof(cnanovdb_node2F), sizeof(nanovdb::InternalNode<nanovdb::InternalNode<nanovdb::LeafNode<float>>>));
    EXPECT_EQ(sizeof(cnanovdb_rootdataF), sizeof(nanovdb::NanoRoot<float>));

    EXPECT_EQ(sizeof(cnanovdb_node0F3), sizeof(nanovdb::LeafNode<nanovdb::Vec3f>));
    EXPECT_EQ(sizeof(cnanovdb_node1F3), sizeof(nanovdb::InternalNode<nanovdb::LeafNode<nanovdb::Vec3f>>));
    EXPECT_EQ(sizeof(cnanovdb_node2F3), sizeof(nanovdb::InternalNode<nanovdb::InternalNode<nanovdb::LeafNode<nanovdb::Vec3f>>>));
    EXPECT_EQ(sizeof(cnanovdb_rootdataF3), sizeof(nanovdb::NanoRoot<nanovdb::Vec3f>));
    EXPECT_EQ(sizeof(cnanovdb_treedata), sizeof(nanovdb::NanoTree<float>));
    EXPECT_EQ(sizeof(cnanovdb_gridblindmetadata), sizeof(nanovdb::GridBlindMetaData));
    EXPECT_EQ(sizeof(cnanovdb_griddata), sizeof(nanovdb::NanoGrid<float>));
} // CNanoVDBSize
#endif

#if !defined(DISABLE_PNANOVDB) && !defined(_MSC_VER)
TEST_F(TestNanoVDB, PNanoVDB_Basic)
{
    EXPECT_EQ(NANOVDB_MAGIC_NUMBER, PNANOVDB_MAGIC_NUMBER);

    EXPECT_EQ(NANOVDB_MAJOR_VERSION_NUMBER, PNANOVDB_MAJOR_VERSION_NUMBER);
    EXPECT_EQ(NANOVDB_MINOR_VERSION_NUMBER, PNANOVDB_MINOR_VERSION_NUMBER);
    EXPECT_EQ(NANOVDB_PATCH_VERSION_NUMBER, PNANOVDB_PATCH_VERSION_NUMBER);

    EXPECT_EQ((int)nanovdb::GridType::Unknown, PNANOVDB_GRID_TYPE_UNKNOWN);
    EXPECT_EQ((int)nanovdb::GridType::Float,   PNANOVDB_GRID_TYPE_FLOAT);
    EXPECT_EQ((int)nanovdb::GridType::Double,  PNANOVDB_GRID_TYPE_DOUBLE);
    EXPECT_EQ((int)nanovdb::GridType::Int16,   PNANOVDB_GRID_TYPE_INT16);
    EXPECT_EQ((int)nanovdb::GridType::Int32,   PNANOVDB_GRID_TYPE_INT32);
    EXPECT_EQ((int)nanovdb::GridType::Int64,   PNANOVDB_GRID_TYPE_INT64);
    EXPECT_EQ((int)nanovdb::GridType::Vec3f,   PNANOVDB_GRID_TYPE_VEC3F);
    EXPECT_EQ((int)nanovdb::GridType::Vec3d,   PNANOVDB_GRID_TYPE_VEC3D);
    EXPECT_EQ((int)nanovdb::GridType::Mask,    PNANOVDB_GRID_TYPE_MASK);
    EXPECT_EQ((int)nanovdb::GridType::Half,    PNANOVDB_GRID_TYPE_HALF);
    EXPECT_EQ((int)nanovdb::GridType::UInt32,  PNANOVDB_GRID_TYPE_UINT32);
    EXPECT_EQ((int)nanovdb::GridType::Boolean, PNANOVDB_GRID_TYPE_BOOLEAN);
    EXPECT_EQ((int)nanovdb::GridType::RGBA8,   PNANOVDB_GRID_TYPE_RGBA8);
    EXPECT_EQ((int)nanovdb::GridType::Fp4,     PNANOVDB_GRID_TYPE_FP4);
    EXPECT_EQ((int)nanovdb::GridType::Fp8,     PNANOVDB_GRID_TYPE_FP8);
    EXPECT_EQ((int)nanovdb::GridType::Fp16,    PNANOVDB_GRID_TYPE_FP16);
    EXPECT_EQ((int)nanovdb::GridType::FpN,     PNANOVDB_GRID_TYPE_FPN);
    EXPECT_EQ((int)nanovdb::GridType::Vec4f,   PNANOVDB_GRID_TYPE_VEC4F);
    EXPECT_EQ((int)nanovdb::GridType::Vec4d,   PNANOVDB_GRID_TYPE_VEC4D);
    EXPECT_EQ((int)nanovdb::GridType::End,     PNANOVDB_GRID_TYPE_END);

    EXPECT_EQ((int)nanovdb::GridClass::Unknown,    PNANOVDB_GRID_CLASS_UNKNOWN);
    EXPECT_EQ((int)nanovdb::GridClass::LevelSet,   PNANOVDB_GRID_CLASS_LEVEL_SET);
    EXPECT_EQ((int)nanovdb::GridClass::FogVolume,  PNANOVDB_GRID_CLASS_FOG_VOLUME);
    EXPECT_EQ((int)nanovdb::GridClass::Staggered,  PNANOVDB_GRID_CLASS_STAGGERED);
    EXPECT_EQ((int)nanovdb::GridClass::PointIndex, PNANOVDB_GRID_CLASS_POINT_INDEX);
    EXPECT_EQ((int)nanovdb::GridClass::PointData,  PNANOVDB_GRID_CLASS_POINT_DATA);
    EXPECT_EQ((int)nanovdb::GridClass::Topology,   PNANOVDB_GRID_CLASS_TOPOLOGY);
    EXPECT_EQ((int)nanovdb::GridClass::VoxelVolume,PNANOVDB_GRID_CLASS_VOXEL_VOLUME);
    EXPECT_EQ((int)nanovdb::GridClass::End,        PNANOVDB_GRID_CLASS_END);

    // check some basic types
    EXPECT_EQ(sizeof(pnanovdb_map_t),   sizeof(nanovdb::Map));
    EXPECT_EQ(sizeof(pnanovdb_coord_t), sizeof(nanovdb::Coord));
    EXPECT_EQ(sizeof(pnanovdb_vec3_t),  sizeof(nanovdb::Vec3f));

    // check nanovdb::Map
    EXPECT_EQ((int)sizeof(nanovdb::Map), PNANOVDB_MAP_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mMatF),    PNANOVDB_MAP_OFF_MATF);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mInvMatF), PNANOVDB_MAP_OFF_INVMATF);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mVecF),    PNANOVDB_MAP_OFF_VECF);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mTaperF),  PNANOVDB_MAP_OFF_TAPERF);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mMatD),    PNANOVDB_MAP_OFF_MATD);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mInvMatD), PNANOVDB_MAP_OFF_INVMATD);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mVecD),    PNANOVDB_MAP_OFF_VECD);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::Map, mTaperD),  PNANOVDB_MAP_OFF_TAPERD);

    EXPECT_EQ((int)sizeof(pnanovdb_map_t), PNANOVDB_MAP_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, matf),    PNANOVDB_MAP_OFF_MATF);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, invmatf), PNANOVDB_MAP_OFF_INVMATF);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, vecf),    PNANOVDB_MAP_OFF_VECF);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, taperf),  PNANOVDB_MAP_OFF_TAPERF);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, matd),    PNANOVDB_MAP_OFF_MATD);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, invmatd), PNANOVDB_MAP_OFF_INVMATD);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, vecd),    PNANOVDB_MAP_OFF_VECD);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_map_t, taperd),  PNANOVDB_MAP_OFF_TAPERD);

    EXPECT_TRUE(validate_strides(printf));// checks strides and prints out new ones if they have changed
}

template <typename ValueT>
void validateLeaf(pnanovdb_grid_type_t grid_type)
{
    using nodedata0_t = typename nanovdb::LeafData<ValueT, nanovdb::Coord, nanovdb::Mask, 3>;
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMinimum), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_min);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMaximum), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_max);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mAverage), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_ave);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mStdDevi), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_stddev);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mValues),  (int)pnanovdb_grid_type_constants[grid_type].leaf_off_table);
}

template <>
void validateLeaf<nanovdb::Fp4>(pnanovdb_grid_type_t grid_type)
{
    using nodedata0_t = typename nanovdb::LeafData<nanovdb::Fp4, nanovdb::Coord, nanovdb::Mask, 3>;
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMin), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_min);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMax), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_max);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mAvg), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_ave);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mDev), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_stddev);
}

template <>
void validateLeaf<nanovdb::Fp8>(pnanovdb_grid_type_t grid_type)
{
    using nodedata0_t = typename nanovdb::LeafData<nanovdb::Fp8, nanovdb::Coord, nanovdb::Mask, 3>;
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMin), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_min);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMax), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_max);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mAvg), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_ave);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mDev), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_stddev);
}

template <>
void validateLeaf<nanovdb::Fp16>(pnanovdb_grid_type_t grid_type)
{
    using nodedata0_t = typename nanovdb::LeafData<nanovdb::Fp16, nanovdb::Coord, nanovdb::Mask, 3>;
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMin), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_min);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMax), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_max);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mAvg), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_ave);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mDev), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_stddev);
}

template <>
void validateLeaf<nanovdb::FpN>(pnanovdb_grid_type_t grid_type)
{
    using nodedata0_t = typename nanovdb::LeafData<nanovdb::FpN, nanovdb::Coord, nanovdb::Mask, 3>;
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMin), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_min);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mMax), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_max);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mAvg), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_ave);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodedata0_t, mDev), (int)pnanovdb_grid_type_constants[grid_type].leaf_off_stddev);
}

// template specializations for bool types
template <>
void validateLeaf<bool>(pnanovdb_grid_type_t grid_type)
{
    using nodeLeaf_t = typename nanovdb::LeafData<bool, nanovdb::Coord, nanovdb::Mask, 3>;
    using leaf_t = typename nanovdb::LeafNode<bool>;

    EXPECT_EQ(sizeof(leaf_t), (pnanovdb_grid_type_constants[grid_type].leaf_size));
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mBBoxMin), PNANOVDB_LEAF_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mBBoxDif), PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mValueMask), PNANOVDB_LEAF_OFF_VALUE_MASK);
}

// template specializations for nanovdb::ValueMask types
template <>
void validateLeaf<nanovdb::ValueMask>(pnanovdb_grid_type_t grid_type)
{
    using nodeLeaf_t = typename nanovdb::LeafData<nanovdb::ValueMask, nanovdb::Coord, nanovdb::Mask, 3>;
    using leaf_t = typename nanovdb::LeafNode<nanovdb::ValueMask>;

    EXPECT_EQ(sizeof(leaf_t), (pnanovdb_grid_type_constants[grid_type].leaf_size));
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mBBoxMin), PNANOVDB_LEAF_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mBBoxDif), PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mValueMask), PNANOVDB_LEAF_OFF_VALUE_MASK);
}

TYPED_TEST(TestOffsets, PNanoVDB)
{
    using ValueType = TypeParam;
    pnanovdb_grid_type_t grid_type;
    if (std::is_same<float, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_FLOAT;
    } else if (std::is_same<double, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_DOUBLE;
    } else if (std::is_same<int16_t, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_INT16;
    } else if (std::is_same<int32_t, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_INT32;
    } else if (std::is_same<int64_t, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_INT64;
    } else if (std::is_same<nanovdb::Vec3f, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_VEC3F;
    } else if (std::is_same<nanovdb::Vec3d, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_VEC3D;
    } else if (std::is_same<uint32_t, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_UINT32;
    } else if (std::is_same<nanovdb::ValueMask, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_MASK;
    } else if (std::is_same<bool, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_BOOLEAN;
    } else if (std::is_same<nanovdb::Fp4, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_FP4;
    } else if (std::is_same<nanovdb::Fp8, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_FP8;
    } else if (std::is_same<nanovdb::Fp16, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_FP16;
    } else if (std::is_same<nanovdb::FpN, TypeParam>::value) {
        grid_type = PNANOVDB_GRID_TYPE_FPN;
    } else {
        EXPECT_TRUE(false);
    }
    static const uint32_t rootLevel = 3u;
    using nodeLeaf_t = typename nanovdb::LeafData<ValueType, nanovdb::Coord, nanovdb::Mask, 3>;
    using leaf_t = typename nanovdb::LeafNode<ValueType>;
    using nodeLower_t = typename nanovdb::InternalData<leaf_t, leaf_t::LOG2DIM + 1>;
    using lower_t = typename nanovdb::InternalNode<leaf_t>;
    using nodeUpper_t = typename nanovdb::InternalData<lower_t, lower_t::LOG2DIM + 1>;
    using upper_t = typename nanovdb::InternalNode<lower_t>;
    using rootdata_t = typename nanovdb::RootData<upper_t>;
    using root_t = typename nanovdb::RootNode<upper_t>;
    using rootdata_tile_t = typename nanovdb::RootData<upper_t>::Tile;
    using root_tile_t = typename nanovdb::RootNode<upper_t>::Tile;
    using treedata_t = typename nanovdb::TreeData<rootLevel>;
    using tree_t = typename nanovdb::Tree<root_t>;

    // grid
    EXPECT_EQ((int)sizeof(nanovdb::GridData), PNANOVDB_GRID_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mMagic), PNANOVDB_GRID_OFF_MAGIC);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mChecksum), PNANOVDB_GRID_OFF_CHECKSUM);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mVersion), PNANOVDB_GRID_OFF_VERSION);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mFlags), PNANOVDB_GRID_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridSize), PNANOVDB_GRID_OFF_GRID_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridName), PNANOVDB_GRID_OFF_GRID_NAME);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mWorldBBox), PNANOVDB_GRID_OFF_WORLD_BBOX);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mMap), PNANOVDB_GRID_OFF_MAP);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mVoxelSize), PNANOVDB_GRID_OFF_VOXEL_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridClass), PNANOVDB_GRID_OFF_GRID_CLASS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mGridType), PNANOVDB_GRID_OFF_GRID_TYPE);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mBlindMetadataOffset), PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridData, mBlindMetadataCount), PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT);

    EXPECT_EQ((int)sizeof(pnanovdb_grid_t), PNANOVDB_GRID_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, magic), PNANOVDB_GRID_OFF_MAGIC);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, checksum), PNANOVDB_GRID_OFF_CHECKSUM);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, version), PNANOVDB_GRID_OFF_VERSION);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, flags), PNANOVDB_GRID_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, grid_size), PNANOVDB_GRID_OFF_GRID_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, grid_name), PNANOVDB_GRID_OFF_GRID_NAME);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, map), PNANOVDB_GRID_OFF_MAP);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, world_bbox), PNANOVDB_GRID_OFF_WORLD_BBOX);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, voxel_size), PNANOVDB_GRID_OFF_VOXEL_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, grid_class), PNANOVDB_GRID_OFF_GRID_CLASS);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, grid_type), PNANOVDB_GRID_OFF_GRID_TYPE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, blind_metadata_offset), PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_grid_t, blind_metadata_count), PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT);

    // test GridBlindMetaData
    EXPECT_EQ((int)sizeof(nanovdb::GridBlindMetaData), PNANOVDB_GRIDBLINDMETADATA_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridBlindMetaData, mByteOffset), PNANOVDB_GRIDBLINDMETADATA_OFF_BYTE_OFFSET);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridBlindMetaData, mElementCount), PNANOVDB_GRIDBLINDMETADATA_OFF_ELEMENT_COUNT);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridBlindMetaData, mFlags), PNANOVDB_GRIDBLINDMETADATA_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridBlindMetaData, mSemantic), PNANOVDB_GRIDBLINDMETADATA_OFF_SEMANTIC);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridBlindMetaData, mDataClass), PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_CLASS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridBlindMetaData, mDataType), PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_TYPE);
    EXPECT_EQ(NANOVDB_OFFSETOF(nanovdb::GridBlindMetaData, mName), PNANOVDB_GRIDBLINDMETADATA_OFF_NAME);

    EXPECT_EQ((int)sizeof(pnanovdb_gridblindmetadata_t), PNANOVDB_GRIDBLINDMETADATA_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_gridblindmetadata_t, byte_offset), PNANOVDB_GRIDBLINDMETADATA_OFF_BYTE_OFFSET);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_gridblindmetadata_t, element_count), PNANOVDB_GRIDBLINDMETADATA_OFF_ELEMENT_COUNT);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_gridblindmetadata_t, flags), PNANOVDB_GRIDBLINDMETADATA_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_gridblindmetadata_t, semantic), PNANOVDB_GRIDBLINDMETADATA_OFF_SEMANTIC);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_gridblindmetadata_t, data_class), PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_CLASS);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_gridblindmetadata_t, data_type), PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_TYPE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_gridblindmetadata_t, name), PNANOVDB_GRIDBLINDMETADATA_OFF_NAME);

    // test tree
    EXPECT_EQ((int)sizeof(tree_t), PNANOVDB_TREE_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mNodeOffset[0]), PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mNodeOffset[1]), PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mNodeOffset[2]), PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mNodeOffset[3]), PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mNodeCount[0]), PNANOVDB_TREE_OFF_NODE_COUNT_LEAF);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mNodeCount[1]), PNANOVDB_TREE_OFF_NODE_COUNT_LOWER);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mNodeCount[2]), PNANOVDB_TREE_OFF_NODE_COUNT_UPPER);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mTileCount[0]), PNANOVDB_TREE_OFF_TILE_COUNT_LEAF);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mTileCount[1]), PNANOVDB_TREE_OFF_TILE_COUNT_LOWER);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mTileCount[2]), PNANOVDB_TREE_OFF_TILE_COUNT_UPPER);
    EXPECT_EQ(NANOVDB_OFFSETOF(treedata_t, mVoxelCount), PNANOVDB_TREE_OFF_VOXEL_COUNT);

    EXPECT_EQ((int)sizeof(pnanovdb_tree_t), PNANOVDB_TREE_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, node_offset_leaf), PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, node_offset_lower), PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, node_offset_upper), PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, node_offset_root), PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, node_count_leaf), PNANOVDB_TREE_OFF_NODE_COUNT_LEAF);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, node_count_lower), PNANOVDB_TREE_OFF_NODE_COUNT_LOWER);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, node_count_upper), PNANOVDB_TREE_OFF_NODE_COUNT_UPPER);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, tile_count_leaf), PNANOVDB_TREE_OFF_TILE_COUNT_LEAF);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, tile_count_lower), PNANOVDB_TREE_OFF_TILE_COUNT_LOWER);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, tile_count_upper), PNANOVDB_TREE_OFF_TILE_COUNT_UPPER);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_tree_t, voxel_count), PNANOVDB_TREE_OFF_VOXEL_COUNT);

    // background value can start at pad1
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_t, pad1), PNANOVDB_ROOT_BASE_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_t, bbox_min), PNANOVDB_ROOT_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_t, bbox_max), PNANOVDB_ROOT_OFF_BBOX_MAX);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_t, table_size), PNANOVDB_ROOT_OFF_TABLE_SIZE);

    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_tile_t, pad1), PNANOVDB_ROOT_TILE_BASE_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_tile_t, key), PNANOVDB_ROOT_TILE_OFF_KEY);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_tile_t, child), PNANOVDB_ROOT_TILE_OFF_CHILD);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_root_tile_t, state), PNANOVDB_ROOT_TILE_OFF_STATE);

    EXPECT_EQ((int)sizeof(pnanovdb_upper_t), PNANOVDB_UPPER_BASE_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_upper_t, bbox_min), PNANOVDB_UPPER_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_upper_t, bbox_max), PNANOVDB_UPPER_OFF_BBOX_MAX);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_upper_t, flags), PNANOVDB_UPPER_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_upper_t, value_mask), PNANOVDB_UPPER_OFF_VALUE_MASK);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_upper_t, child_mask), PNANOVDB_UPPER_OFF_CHILD_MASK);

    EXPECT_EQ((int)sizeof(pnanovdb_lower_t), PNANOVDB_LOWER_BASE_SIZE);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_lower_t, bbox_min), PNANOVDB_LOWER_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_lower_t, bbox_max), PNANOVDB_LOWER_OFF_BBOX_MAX);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_lower_t, flags), PNANOVDB_LOWER_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_lower_t, value_mask), PNANOVDB_LOWER_OFF_VALUE_MASK);
    EXPECT_EQ(NANOVDB_OFFSETOF(pnanovdb_lower_t, child_mask), PNANOVDB_LOWER_OFF_CHILD_MASK);

    EXPECT_EQ((uint)sizeof(root_t), (pnanovdb_grid_type_constants[grid_type].root_size));
    EXPECT_EQ(NANOVDB_OFFSETOF(rootdata_t, mBBox.mCoord[0]), PNANOVDB_ROOT_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(rootdata_t, mBBox.mCoord[1]), PNANOVDB_ROOT_OFF_BBOX_MAX);
    EXPECT_EQ(NANOVDB_OFFSETOF(rootdata_t, mTableSize), PNANOVDB_ROOT_OFF_TABLE_SIZE);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(rootdata_t, mBackground), pnanovdb_grid_type_constants[grid_type].root_off_background);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(rootdata_t, mMinimum), pnanovdb_grid_type_constants[grid_type].root_off_min);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(rootdata_t, mMaximum), pnanovdb_grid_type_constants[grid_type].root_off_max);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(rootdata_t, mAverage), pnanovdb_grid_type_constants[grid_type].root_off_ave);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(rootdata_t, mStdDevi), pnanovdb_grid_type_constants[grid_type].root_off_stddev);

    EXPECT_EQ((uint)sizeof(root_tile_t), (pnanovdb_grid_type_constants[grid_type].root_tile_size));
    EXPECT_EQ(NANOVDB_OFFSETOF(rootdata_tile_t, key), PNANOVDB_ROOT_TILE_OFF_KEY);
    EXPECT_EQ(NANOVDB_OFFSETOF(rootdata_tile_t, child), PNANOVDB_ROOT_TILE_OFF_CHILD);
    EXPECT_EQ(NANOVDB_OFFSETOF(rootdata_tile_t, state), PNANOVDB_ROOT_TILE_OFF_STATE);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(rootdata_tile_t, value), pnanovdb_grid_type_constants[grid_type].root_tile_off_value);

    EXPECT_EQ((uint)sizeof(upper_t), (pnanovdb_grid_type_constants[grid_type].upper_size));
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeUpper_t, mBBox.mCoord[0]), PNANOVDB_UPPER_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeUpper_t, mBBox.mCoord[1]), PNANOVDB_UPPER_OFF_BBOX_MAX);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeUpper_t, mFlags), PNANOVDB_UPPER_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeUpper_t, mValueMask), PNANOVDB_UPPER_OFF_VALUE_MASK);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeUpper_t, mChildMask), PNANOVDB_UPPER_OFF_CHILD_MASK);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeUpper_t, mMinimum), pnanovdb_grid_type_constants[grid_type].upper_off_min);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeUpper_t, mMaximum), pnanovdb_grid_type_constants[grid_type].upper_off_max);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeUpper_t, mAverage), pnanovdb_grid_type_constants[grid_type].upper_off_ave);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeUpper_t, mStdDevi), pnanovdb_grid_type_constants[grid_type].upper_off_stddev);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeUpper_t, mTable), pnanovdb_grid_type_constants[grid_type].upper_off_table);

    EXPECT_EQ((uint)sizeof(lower_t), (pnanovdb_grid_type_constants[grid_type].lower_size));
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLower_t, mBBox.mCoord[0]), PNANOVDB_LOWER_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLower_t, mBBox.mCoord[1]), PNANOVDB_LOWER_OFF_BBOX_MAX);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLower_t, mFlags), PNANOVDB_LOWER_OFF_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLower_t, mValueMask), PNANOVDB_LOWER_OFF_VALUE_MASK);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLower_t, mChildMask), PNANOVDB_LOWER_OFF_CHILD_MASK);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeLower_t, mMinimum), pnanovdb_grid_type_constants[grid_type].lower_off_min);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeLower_t, mMaximum), pnanovdb_grid_type_constants[grid_type].lower_off_max);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeLower_t, mAverage), pnanovdb_grid_type_constants[grid_type].lower_off_ave);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeLower_t, mStdDevi), pnanovdb_grid_type_constants[grid_type].lower_off_stddev);
    EXPECT_EQ((uint)NANOVDB_OFFSETOF(nodeLower_t, mTable), pnanovdb_grid_type_constants[grid_type].lower_off_table);

    EXPECT_EQ(8u*sizeof(rootdata_t::mAverage),  pnanovdb_grid_type_stat_strides_bits[grid_type]);
    EXPECT_EQ(8u*sizeof(rootdata_t::mStdDevi),  pnanovdb_grid_type_stat_strides_bits[grid_type]);
    EXPECT_EQ(8u*sizeof(nodeUpper_t::mAverage), pnanovdb_grid_type_stat_strides_bits[grid_type]);
    EXPECT_EQ(8u*sizeof(nodeUpper_t::mStdDevi), pnanovdb_grid_type_stat_strides_bits[grid_type]);
    EXPECT_EQ(8u*sizeof(nodeLower_t::mAverage), pnanovdb_grid_type_stat_strides_bits[grid_type]);
    EXPECT_EQ(8u*sizeof(nodeLower_t::mStdDevi), pnanovdb_grid_type_stat_strides_bits[grid_type]);

    // leaf nodes
    EXPECT_EQ(sizeof(leaf_t), (pnanovdb_grid_type_constants[grid_type].leaf_size));
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mBBoxMin), PNANOVDB_LEAF_OFF_BBOX_MIN);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mBBoxDif), PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS);
    EXPECT_EQ(NANOVDB_OFFSETOF(nodeLeaf_t, mValueMask), PNANOVDB_LEAF_OFF_VALUE_MASK);
    validateLeaf<ValueType>(grid_type);
}// PNanoVDB
#endif

TEST_F(TestNanoVDB, GridStats)
{
    using GridT = nanovdb::NanoGrid<float>;
    Sphere<float>               sphere(nanovdb::Vec3<float>(50), 50.0f);
    nanovdb::GridBuilder<float> builder(sphere.background(), nanovdb::GridClass::LevelSet);
    const nanovdb::CoordBBox    bbox(nanovdb::Coord(-100), nanovdb::Coord(100));
    //mTimer.start("GridBuilder");
    builder(sphere, bbox);
    //mTimer.stop();
    auto handle1 = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
    auto handle2 = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test");
    EXPECT_TRUE(handle1);
    EXPECT_TRUE(handle2);
    GridT* grid1 = handle1.grid<float>();
    GridT* grid2 = handle2.grid<float>();
    EXPECT_TRUE(grid1);
    EXPECT_TRUE(grid2);

    //std::cerr << "grid1 = " << grid1->indexBBox() << ", grid2 = " << grid2->indexBBox() << std::endl;
    EXPECT_EQ(grid1->activeVoxelCount(), grid2->activeVoxelCount());
    EXPECT_EQ(grid1->worldBBox(), grid2->worldBBox());
    EXPECT_EQ(grid1->indexBBox(), grid2->indexBBox());
    nanovdb::NodeManager<nanovdb::FloatGrid> mgr1(*grid1);
    nanovdb::NodeManager<nanovdb::FloatGrid> mgr2(*grid2);

    { // reset stats in grid2
        //grid2->tree().data()->mVoxelCount = uint64_t(0);
        grid2->data()->mWorldBBox = nanovdb::BBox<nanovdb::Vec3R>();
        grid2->tree().root().data()->mBBox = nanovdb::BBox<nanovdb::Coord>();
        for (uint32_t i = 0; i < grid2->tree().nodeCount(0); ++i) {
            auto* leaf = mgr2.leaf(i);
            auto* data = leaf->data();
            data->mMinimum = data->mMaximum = 0.0f;
            data->mBBoxMin &= ~nanovdb::NanoLeaf<float>::MASK; /// set to origin!
            data->mBBoxDif[0] = data->mBBoxDif[1] = data->mBBoxDif[2] = uint8_t(255);
            EXPECT_EQ(data->mBBoxDif[0], uint8_t(255));
        }
        for (uint32_t i = 0; i < grid2->tree().nodeCount(1); ++i) {
            auto* node = mgr2.lower(i);
            auto* data = node->data();
            data->mMinimum = data->mMaximum = 0.0f;
            data->mBBox[0] &= ~nanovdb::NanoLower<float>::MASK; /// set to origin!
            data->mBBox[1] = nanovdb::Coord(0);
        }
        for (uint32_t i = 0; i < grid2->tree().nodeCount(2); ++i) {
            auto* node = mgr2.upper(i);
            auto* data = node->data();
            data->mMinimum = data->mMaximum = 0.0f;
            data->mBBox[0] &= ~nanovdb::NanoUpper<float>::MASK; /// set to origin!
            data->mBBox[1] = nanovdb::Coord(0);
        }
    }
    //std::cerr << "grid1 = " << grid1->indexBBox() << ", grid2 = " << grid2->indexBBox() << std::endl;
    //EXPECT_NE(grid1->activeVoxelCount(), grid2->activeVoxelCount());
    EXPECT_NE(grid1->indexBBox(), grid2->indexBBox());
    EXPECT_NE(grid1->worldBBox(), grid2->worldBBox());

    { // check stats in grid2
        EXPECT_EQ(grid1->tree().nodeCount(0), grid2->tree().nodeCount(0));

        for (uint32_t i = 0; i < grid2->tree().nodeCount(0); ++i) {
            auto* leaf1 = mgr1.leaf(i);
            auto* leaf2 = mgr2.leaf(i);
            EXPECT_NE(leaf1->minimum(), leaf2->minimum());
            EXPECT_NE(leaf1->maximum(), leaf2->maximum());
            EXPECT_NE(leaf1->bbox(), leaf2->bbox());
        }

        EXPECT_EQ(grid1->tree().nodeCount(1), grid2->tree().nodeCount(1));
        for (uint32_t i = 0; i < grid2->tree().nodeCount(1); ++i) {
            auto* node1 = mgr1.lower(i);
            auto* node2 = mgr2.lower(i);
            EXPECT_NE(node1->minimum(), node2->minimum());
            EXPECT_NE(node1->maximum(), node2->maximum());
            EXPECT_NE(node1->bbox(), node2->bbox());
        }
        EXPECT_EQ(grid1->tree().nodeCount(2), grid2->tree().nodeCount(2));
        for (uint32_t i = 0; i < grid2->tree().nodeCount(2); ++i) {
            auto* node1 = mgr1.upper(i);
            auto* node2 = mgr2.upper(i);
            EXPECT_NE(node1->minimum(), node2->minimum());
            EXPECT_NE(node1->maximum(), node2->maximum());
            EXPECT_NE(node1->bbox(), node2->bbox());
        }
    }

    //mTimer.start("GridStats");
    nanovdb::gridStats(*grid2);
    //mTimer.stop();

    { // check stats in grid2
        EXPECT_EQ(grid1->tree().nodeCount(0), grid2->tree().nodeCount(0));
        for (uint32_t i = 0; i < grid2->tree().nodeCount(0); ++i) {
            auto* leaf1 = mgr1.leaf(i);
            auto* leaf2 = mgr2.leaf(i);
            EXPECT_EQ(leaf1->minimum(), leaf2->minimum());
            EXPECT_EQ(leaf1->maximum(), leaf2->maximum());
            EXPECT_EQ(leaf1->bbox(), leaf2->bbox());
        }
        EXPECT_EQ(grid1->tree().nodeCount(1), grid2->tree().nodeCount(1));
        for (uint32_t i = 0; i < grid2->tree().nodeCount(1); ++i) {
            auto* node1 = mgr1.lower(i);
            auto* node2 = mgr2.lower(i);
            EXPECT_EQ(node1->minimum(), node2->minimum());
            EXPECT_EQ(node1->maximum(), node2->maximum());
            EXPECT_EQ(node1->bbox(), node2->bbox());
        }
        EXPECT_EQ(grid1->tree().nodeCount(2), grid2->tree().nodeCount(2));
        for (uint32_t i = 0; i < grid2->tree().nodeCount(2); ++i) {
            auto* node1 = mgr1.upper(i);
            auto* node2 = mgr2.upper(i);
            EXPECT_EQ(node1->minimum(), node2->minimum());
            EXPECT_EQ(node1->maximum(), node2->maximum());
            EXPECT_EQ(node1->bbox(), node2->bbox());
        }
    }

    //std::cerr << "grid1 = " << grid1->indexBBox() << ", grid2 = " << grid2->indexBBox() << std::endl;
    EXPECT_EQ(grid1->activeVoxelCount(), grid2->activeVoxelCount());
    EXPECT_EQ(grid1->indexBBox(), grid2->indexBBox());
    EXPECT_EQ(grid1->worldBBox(), grid2->worldBBox());

} // GridStats

TEST_F(TestNanoVDB, ScalarSampleFromVoxels)
{
    // create a grid so sample from
    const float dx = 0.5f; // voxel size
    auto        trilinearWorld = [&](const nanovdb::Vec3d& xyz) -> float {
        return 0.34f + 1.6f * xyz[0] + 6.7f * xyz[1] - 3.5f * xyz[2]; // index coordinates
    };
    auto trilinearIndex = [&](const nanovdb::Coord& ijk) -> float {
        return 0.34f + 1.6f * dx * ijk[0] + 6.7f * dx * ijk[1] - 3.5f * dx * ijk[2]; // index coordinates
    };

    nanovdb::GridBuilder<float> builder(1.0f);
    const nanovdb::CoordBBox    bbox(nanovdb::Coord(0), nanovdb::Coord(128));
    builder(trilinearIndex, bbox);
    auto handle = builder.getHandle<>(dx);
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* grid = handle.grid<float>();
    EXPECT_TRUE(grid);

    const nanovdb::Vec3d xyz(13.4, 24.67, 5.23); // in index space
    const nanovdb::Coord ijk(13, 25, 5); // in index space (nearest)
    const auto           exact = trilinearWorld(grid->indexToWorld(xyz));
    const auto           approx = trilinearIndex(ijk);
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;

    auto acc = grid->getAccessor();
    auto sampler0 = nanovdb::createSampler<0>(grid->tree());
    auto sampler1 = nanovdb::createSampler<1>(acc);
    auto sampler2 = nanovdb::createSampler<2>(acc);
    auto sampler3 = nanovdb::createSampler<3>(acc);
    //std::cerr << "0'th order: v = " << sampler0(xyz) << std::endl;
    EXPECT_EQ(approx, sampler0(xyz));
    EXPECT_NE(exact, sampler0(xyz));
    //std::cerr << "1'th order: v = " << sampler1(xyz) << std::endl;
    EXPECT_NEAR(exact, sampler1(xyz), 1e-5);
    //std::cerr << "2'th order: v = " << sampler2(xyz) << std::endl;
    EXPECT_NEAR(exact, sampler2(xyz), 1e-4);
    //std::cerr << "3'rd order: v = " << sampler3(xyz) << std::endl;
    EXPECT_NEAR(exact, sampler3(xyz), 1e-5);

    EXPECT_FALSE(sampler1.zeroCrossing());
    const auto gradIndex = sampler1.gradient(xyz); //in index space
    EXPECT_NEAR(1.6f, gradIndex[0] / dx, 2e-5);
    EXPECT_NEAR(6.7f, gradIndex[1] / dx, 2e-5);
    EXPECT_NEAR(-3.5f, gradIndex[2] / dx, 2e-5);
    const auto gradWorld = grid->indexToWorldGrad(gradIndex); // in world units
    EXPECT_NEAR(1.6f, gradWorld[0], 2e-5);
    EXPECT_NEAR(6.7f, gradWorld[1], 2e-5);
    EXPECT_NEAR(-3.5f, gradWorld[2], 2e-5);

    EXPECT_EQ(grid->tree().getValue(ijk), sampler0.accessor().getValue(ijk));
    EXPECT_EQ(grid->tree().getValue(ijk), sampler1.accessor().getValue(ijk));
    EXPECT_EQ(grid->tree().getValue(ijk), sampler2.accessor().getValue(ijk));
    EXPECT_EQ(grid->tree().getValue(ijk), sampler3.accessor().getValue(ijk));
} // ScalarSampleFromVoxels

TEST_F(TestNanoVDB, VectorSampleFromVoxels)
{
    // create a grid so sample from
    const float dx = 0.5f; // voxel size
    auto        trilinearWorld = [&](const nanovdb::Vec3d& xyz) -> nanovdb::Vec3f {
        return nanovdb::Vec3f(0.34f, 1.6f * xyz[0] + 6.7f * xyz[1], -3.5f * xyz[2]); // index coordinates
    };
    auto trilinearIndex = [&](const nanovdb::Coord& ijk) -> nanovdb::Vec3f {
        return nanovdb::Vec3f(0.34f, 1.6f * dx * ijk[0] + 6.7f * dx * ijk[1], -3.5f * dx * ijk[2]); // index coordinates
    };

    nanovdb::GridBuilder<nanovdb::Vec3f> builder(nanovdb::Vec3f(1.0f));
    const nanovdb::CoordBBox             bbox(nanovdb::Coord(0), nanovdb::Coord(128));
    builder(trilinearIndex, bbox);
    auto handle = builder.getHandle<>(dx);
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* grid = handle.grid<nanovdb::Vec3f>();
    EXPECT_TRUE(grid);

    const nanovdb::Vec3d ijk(13.4, 24.67, 5.23); // in index space
    const auto           exact = trilinearWorld(grid->indexToWorld(ijk));
    const auto           approx = trilinearIndex(nanovdb::Coord(13, 25, 5));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;

    auto acc = grid->getAccessor();
    auto sampler0 = nanovdb::createSampler<0>(acc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_EQ(approx, sampler0(ijk));

    auto sampler1 = nanovdb::createSampler<1>(acc); // faster since it's using an accessor!!!
    //std::cerr << "1'th order: v = " << sampler1(ijk) << std::endl;
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(exact[i], sampler1(ijk)[i], 1e-5);
    //EXPECT_FALSE(sampler1.zeroCrossing());// triggeres a static_assert error
    //EXPECT_FALSE(sampler1.gradient(grid->indexToWorld(ijk)));// triggeres a static_assert error

    nanovdb::SampleFromVoxels<nanovdb::NanoTree<nanovdb::Vec3f>, 3> sampler3(grid->tree());
    //auto sampler3 = nanovdb::createSampler<3>( acc );
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(exact[i], sampler3(ijk)[i], 1e-5);

} // VectorSampleFromVoxels

TEST_F(TestNanoVDB, GridChecksum)
{
    EXPECT_TRUE(nanovdb::ChecksumMode::Disable < nanovdb::ChecksumMode::End);
    EXPECT_TRUE(nanovdb::ChecksumMode::Partial < nanovdb::ChecksumMode::End);
    EXPECT_TRUE(nanovdb::ChecksumMode::Full < nanovdb::ChecksumMode::End);
    EXPECT_TRUE(nanovdb::ChecksumMode::Default < nanovdb::ChecksumMode::End);
    EXPECT_NE(nanovdb::ChecksumMode::Disable, nanovdb::ChecksumMode::Partial);
    EXPECT_NE(nanovdb::ChecksumMode::Disable, nanovdb::ChecksumMode::Full);
    EXPECT_NE(nanovdb::ChecksumMode::Full, nanovdb::ChecksumMode::Partial);
    EXPECT_NE(nanovdb::ChecksumMode::Default, nanovdb::ChecksumMode::Disable);
    EXPECT_EQ(nanovdb::ChecksumMode::Default, nanovdb::ChecksumMode::Partial);
    EXPECT_NE(nanovdb::ChecksumMode::Default, nanovdb::ChecksumMode::Full);

    nanovdb::CpuTimer<> timer;
    //timer.start("nanovdb::createLevelSetSphere");
    auto handle = nanovdb::createLevelSetSphere<float>(100.0f,
                                                       nanovdb::Vec3f(50),
                                                       1.0,
                                                       3.0,
                                                       nanovdb::Vec3d(0),
                                                       "sphere_20",
                                                       nanovdb::StatsMode::Disable,
                                                       nanovdb::ChecksumMode::Disable);
    //timer.stop();
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* grid = handle.grid<float>();
    EXPECT_TRUE(grid);

    nanovdb::GridChecksum checksum1, checksum2, checksum3;

    EXPECT_EQ(checksum1, checksum3);

    //timer.start("Partial checksum");
    checksum3(*grid, nanovdb::ChecksumMode::Partial);
    //timer.stop();

    EXPECT_NE(checksum1, checksum3);

    //timer.start("Full checksum");
    checksum1(*grid, nanovdb::ChecksumMode::Full);
    //timer.stop();

    checksum2(*grid, nanovdb::ChecksumMode::Full);

    EXPECT_EQ(checksum1, checksum2);

    auto* leaf = grid->tree().getFirstNode<0>();
    EXPECT_EQ(leaf, nanovdb::createLeafMgr(*grid)[0]);

    leaf->data()->mValues[0] += 0.00001f; // slightly modify a single voxel value

    checksum2(*grid, nanovdb::ChecksumMode::Full);
    EXPECT_NE(checksum1, checksum2);

    leaf->data()->mValues[0] -= 0.00001f; // change back the single voxel value to it's original value

    checksum2(*grid, nanovdb::ChecksumMode::Full);
    EXPECT_EQ(checksum1, checksum2);

    leaf->data()->mValueMask.toggle(0); // change a single bit in a value mask

    checksum2(*grid, nanovdb::ChecksumMode::Full);
    EXPECT_NE(checksum1, checksum2);

    //timer.start("Incomplete checksum");
    checksum2(*grid, nanovdb::ChecksumMode::Partial);
    //timer.stop();
    EXPECT_EQ(checksum2, checksum3);
} // GridChecksum

TEST_F(TestNanoVDB, GridValidator)
{
    nanovdb::CpuTimer<> timer;
    //timer.start("nanovdb::createLevelSetSphere");
    auto handle = nanovdb::createLevelSetSphere<float>(100.0f,
                                                       nanovdb::Vec3f(50),
                                                       1.0, 3.0,
                                                       nanovdb::Vec3d(0),
                                                       "sphere_20",
                                                       nanovdb::StatsMode::All,
                                                       nanovdb::ChecksumMode::Full);
    //timer.stop();
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* grid = handle.grid<float>();
    EXPECT_TRUE(grid);

    //timer.start("isValid - not detailed");
    EXPECT_TRUE(nanovdb::isValid(*grid, false, true));
    //timer.stop();

    //timer.start("isValid - detailed");
    EXPECT_TRUE(nanovdb::isValid(*grid, true, true));
    //timer.stop();

    //timer.start("Full checksum");
    auto fastChecksum = nanovdb::checksum(*grid, nanovdb::ChecksumMode::Full);
    //timer.stop();
    EXPECT_EQ(fastChecksum, nanovdb::checksum(*grid, nanovdb::ChecksumMode::Full));

    auto mgr = nanovdb::createLeafMgr(*grid);
    auto* leaf = mgr[0];

    leaf->data()->mValues[0] += 0.00001f; // slightly modify a single voxel value

    EXPECT_NE(fastChecksum, nanovdb::checksum(*grid, nanovdb::ChecksumMode::Full));
    EXPECT_FALSE(nanovdb::isValid(*grid, true, false));

    leaf->data()->mValues[0] -= 0.00001f; // change back the single voxel value to it's original value

    EXPECT_EQ(fastChecksum, nanovdb::checksum(*grid, nanovdb::ChecksumMode::Full));
    EXPECT_TRUE(nanovdb::isValid(*grid, true, true));

    leaf->data()->mValueMask.toggle(0); // change a singel bit in a value mask

    EXPECT_NE(fastChecksum, nanovdb::checksum(*grid, nanovdb::ChecksumMode::Full));
    EXPECT_FALSE(nanovdb::isValid(*grid, true, false));
} // GridValidator

TEST_F(TestNanoVDB, RandomReadAccessor)
{
    const float background = 0.0f;
    const int voxelCount = 512, min = -10000, max = 10000;
    std::srand(98765);
    auto op = [&](){return rand() % (max - min) + min;};

    for (int i=0; i<10; ++i) {
        nanovdb::GridBuilder<float> builder(background);
        auto acc = builder.getAccessor();
        std::vector<nanovdb::Coord> voxels(voxelCount);
        for (int j=0; j<voxelCount; ++j) {
            auto &ijk = voxels[j];
            ijk[0] = op();
            ijk[1] = op();
            ijk[2] = op();
            acc.setValue(ijk, 1.0f*j);
        }
        auto gridHdl = builder.getHandle<>();
        EXPECT_TRUE(gridHdl);
        EXPECT_EQ(1u, gridHdl.gridCount());
        auto grid = gridHdl.grid<float>();
        EXPECT_TRUE(grid);
        const auto &root = grid->tree().root();
#if 1
        auto acc0a = nanovdb::createAccessor<>(root);// no node caching
        auto acc1a = nanovdb::createAccessor<0>(root);// cache leaf node only
        auto acc1b = nanovdb::createAccessor<1>(root);// cache lower internal node only
        auto acc1c = nanovdb::createAccessor<2>(root);// cache upper internal node only
        auto acc2a = nanovdb::createAccessor<0, 1>(root);// cache leaf and lower internal nodes
        auto acc2b = nanovdb::createAccessor<1, 2>(root);// cache lower and upper internal nodes
        auto acc2c = nanovdb::createAccessor<0, 2>(root);// cache leaf and upper internal nodes
        auto acc3a = nanovdb::createAccessor<0, 1, 2>(root);// cache leaf and both intern node levels
        auto acc3b = root.getAccessor();// same as the one above where all levels are cached
        auto acc3c = nanovdb::DefaultReadAccessor<float>(root);// same as the one above where all levels are cached
#else
        // Alternative (more verbose) way to create accessors
        auto acc0a = nanovdb::ReadAccessor<float>(root);// no node caching
        auto acc1a = nanovdb::ReadAccessor<float, 0>(root);// cache leaf node only
        auto acc1b = nanovdb::ReadAccessor<float, 1>(root);// cache lower internal node only
        auto acc1c = nanovdb::ReadAccessor<float, 2>(root);// cache upper internal node only
        auto acc2a = nanovdb::ReadAccessor<float, 0, 1>(root);// cache leaf and lower internal nodes
        auto acc2b = nanovdb::ReadAccessor<float, 1, 2>(root);// cache lower and upper internal nodes
        auto acc2c = nanovdb::ReadAccessor<float, 0, 2>(root);// cache leaf and upper internal nodes
        auto acc3a = nanovdb::ReadAccessor<float, 0, 1, 2>(root);// cache leaf and both intern node levels
        auto acc3b = nanovdb::DefaultReadAccessor<float>(root);// same as the one above where all levels are cached
#endif
        for (int j=0; j<voxelCount; ++j) {
            const float v = 1.0f * j;
            const auto &ijk = voxels[j];
            //if (j<5) std::cerr << ijk << std::endl;
            EXPECT_EQ( v, acc0a.getValue(ijk) );
            EXPECT_EQ( v, acc1a.getValue(ijk) );
            EXPECT_EQ( v, acc1b.getValue(ijk) );
            EXPECT_EQ( v, acc1c.getValue(ijk) );
            EXPECT_EQ( v, acc2a.getValue(ijk) );
            EXPECT_EQ( v, acc2b.getValue(ijk) );
            EXPECT_EQ( v, acc2c.getValue(ijk) );
            EXPECT_EQ( v, acc3a.getValue(ijk) );
            EXPECT_EQ( v, acc3b.getValue(ijk) );
            EXPECT_EQ( v, acc3c.getValue(ijk) );
        }
    }
}

TEST_F(TestNanoVDB, StandardDeviation)
{
    nanovdb::GridBuilder<float> builder(0.5f);

    {
        auto acc = builder.getAccessor();
        acc.setValue(nanovdb::Coord(-1), 1.0f);
        acc.setValue(nanovdb::Coord(0), 2.0f);
        acc.setValue(nanovdb::Coord(1), 3.0f);
        acc.setValue(nanovdb::Coord(2), 0.0f);
    }

    auto gridHdl = builder.getHandle<>();
    EXPECT_TRUE(gridHdl);
    auto grid = gridHdl.grid<float>();
    EXPECT_TRUE(grid);
    nanovdb::gridStats(*grid);

    auto acc  = grid->tree().getAccessor();
    {
        EXPECT_EQ( 1.0f,  acc.getValue(nanovdb::Coord(-1)) );
        EXPECT_EQ( 2.0f,  acc.getValue(nanovdb::Coord( 0)) );
        EXPECT_EQ( 3.0f,  acc.getValue(nanovdb::Coord( 1)) );
        EXPECT_EQ( 0.0f,  acc.getValue(nanovdb::Coord( 2)) );
        auto nodeInfo = acc.getNodeInfo(nanovdb::Coord(-1));
        EXPECT_EQ(nodeInfo.mAverage, 1.f);
        EXPECT_EQ(nodeInfo.mLevel, 0u);
        EXPECT_EQ(nodeInfo.mDim, 8u);
    }
    {
        auto nodeInfo = acc.getNodeInfo(nanovdb::Coord(1));
        EXPECT_EQ(nodeInfo.mAverage, (2.0f + 3.0f) / 3.0f);
        auto getStdDev = [&](int n, float a, float b, float c) {
            float m = (a + b + c) / n;
            float sd = sqrtf(((a - m) * (a - m) +
                              (b - m) * (b - m) +
                              (c - m) * (c - m)) /
                             n);
            return sd;
        };
        EXPECT_NEAR(nodeInfo.mStdDevi, getStdDev(3.0f, 2.0f, 3.0f, 0), 1e-5);
        EXPECT_EQ(nodeInfo.mLevel, 0u);
        EXPECT_EQ(nodeInfo.mDim, 8u);
    }
} // ReadAccessor

TEST_F(TestNanoVDB, BoxStencil)
{
    const float a = 0.54f, b[3]={0.12f, 0.78f,-0.34f};
    const nanovdb::Coord min(-17, -10, -8), max(10, 21, 13);
    const nanovdb::CoordBBox bbox(min, max), bbox2(min, max.offsetBy(-1));
    nanovdb::GridBuilder<float> builder(0.0f);
    auto func = [&](const nanovdb::Coord &ijk) {
        return a + b[0]*ijk[0] + b[1]*ijk[1] + b[2]*ijk[2];
    };
    builder(func, bbox);
    auto handle = builder.getHandle();
    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    auto* grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    auto acc = grid->getAccessor();
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_EQ(func(*it), acc.getValue(*it));
    }
    auto func2 = [&](const nanovdb::Vec3f &xyz) {
        return a + b[0]*xyz[0] + b[1]*xyz[1] + b[2]*xyz[2];
    };
    nanovdb::BoxStencil<nanovdb::FloatGrid> s(*grid);
    for (auto it = bbox2.begin(); it; ++it) {
        const nanovdb::Coord p = *it;
        s.moveTo(p);
        const nanovdb::Vec3f xyz(p[0] + 0.12f, p[1] + 0.34f, p[2] + 0.07f);
        EXPECT_NEAR(func2(xyz), s.interpolation(xyz), 5e-6f);
        const auto grad = s.gradient(xyz);
        EXPECT_NEAR( b[0], grad[0], 3e-6f);
        EXPECT_NEAR( b[1], grad[1], 3e-6f);
        EXPECT_NEAR( b[2], grad[2], 3e-6f);
    }
}// BoxStencil

TEST_F(TestNanoVDB, CurvatureStencil)
{
    {// test of level set to sphere at (6,8,10) with R=10 and dx=0.5
        const float radius = 10.0f;
        const nanovdb::Vec3f center(6.0, 8.0, 10.0);//i.e. (12,16,20) in index space
        auto handle = nanovdb::createLevelSetSphere<float>(radius,
                                                    center,
                                                    0.5, // dx
                                                    20.0); // half-width so dense inside

        EXPECT_TRUE(handle);
        EXPECT_EQ(1u, handle.gridCount());
        auto* grid = handle.grid<float>();
        EXPECT_TRUE(grid);

        nanovdb::CurvatureStencil<nanovdb::FloatGrid> cs(*grid);
        nanovdb::Coord xyz(20,16,20);//i.e. 8 voxel or 4 world units away from the center
        cs.moveTo(xyz);

        EXPECT_NEAR(1.0/4.0, cs.meanCurvature(), 0.01);// 1/distance from center
        EXPECT_NEAR(1.0/4.0, cs.meanCurvatureNormGrad(), 0.01);// 1/distance from center

        EXPECT_NEAR(1.0/16.0, cs.gaussianCurvature(), 0.01);// 1/distance^2 from center
        EXPECT_NEAR(1.0/16.0, cs.gaussianCurvatureNormGrad(), 0.01);// 1/distance^2 from center

        //std::cerr << cs.gradient() << std::endl;
        EXPECT_NEAR( 1.0f, cs.gradient()[0], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient()[1], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient()[2], 1e-6f);

        float mean, gaussian;
        cs.curvatures(mean, gaussian);
        EXPECT_NEAR(1.0/4.0, mean, 0.01);// 1/distance from center
        EXPECT_NEAR(1.0/16.0, gaussian, 0.01);// 1/distance^2 from center

        float minCurv, maxCurv;
        cs.principalCurvatures(minCurv, maxCurv);
        EXPECT_NEAR(1.0/4.0, minCurv, 0.01);// 1/distance from center
        EXPECT_NEAR(1.0/4.0, maxCurv, 0.01);// 1/distance from center

        xyz = nanovdb::Coord(12,16,10);//i.e. 10 voxel or 5 world units away from the center
        cs.moveTo(xyz);
        EXPECT_NEAR(1.0/5.0, cs.meanCurvature(), 0.01);// 1/distance from center
        EXPECT_NEAR(
            1.0/5.0, cs.meanCurvatureNormGrad(), 0.01);// 1/distance from center

        EXPECT_NEAR(1.0/25.0, cs.gaussianCurvature(), 0.01);// 1/distance^2 from center
        EXPECT_NEAR(
            1.0/25.0, cs.gaussianCurvatureNormGrad(), 0.01);// 1/distance^2 from center

       cs.principalCurvatures(minCurv, maxCurv);
        EXPECT_NEAR(1.0/5.0, minCurv,  0.01);// 1/distance from center
        EXPECT_NEAR(1.0/5.0, maxCurv, 0.01);// 1/distance from center
        EXPECT_NEAR(
            1.0/5.0, minCurv,  0.01);// 1/distance from center
            EXPECT_NEAR(
            1.0/5.0, maxCurv, 0.01);// 1/distance from center

        cs.curvaturesNormGrad(mean, gaussian);
        EXPECT_NEAR(1.0/5.0, mean, 0.01);// 1/distance from center
        EXPECT_NEAR(1.0/25.0, gaussian, 0.01);// 1/distance^2 from center
    }

    {// test sparse level set sphere
      const double percentage = 0.1/100.0;//i.e. 0.1%
      const int dim = 256;

      // sparse level set sphere
      nanovdb::Vec3f C(0.35f, 0.35f, 0.35f);
      double r = 0.15, voxelSize = 1.0/(dim-1);
      auto handle = nanovdb::createLevelSetSphere(float(r), C, voxelSize);
      EXPECT_TRUE(handle);
      EXPECT_EQ(1u, handle.gridCount());
      auto* sphere = handle.grid<float>();
      EXPECT_TRUE(sphere);

      nanovdb::CurvatureStencil<nanovdb::FloatGrid> cs(*sphere);
      const auto ijk = nanovdb::RoundDown<nanovdb::Coord>(sphere->worldToIndex(nanovdb::Vec3d(0.35, 0.35, 0.35 + 0.15)));
      const nanovdb::Vec3d tmp(ijk[0],ijk[1],ijk[2]);
      const double radius = (sphere->indexToWorld(tmp)-nanovdb::Vec3d(0.35)).length();
      //std::cerr << "\rRadius = " << radius << std::endl;
      //std::cerr << "Index coord =" << ijk << std::endl;
      cs.moveTo(ijk);
      auto acc = sphere->getAccessor();
      auto v = cs.getValue< 0, 0, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0,  0,  0)), v);
      v = cs.getValue<-1, 0, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy(-1,  0,  0)), v);
      v = cs.getValue< 1, 0, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 1,  0,  0)), v);
      v = cs.getValue< 0,-1, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0, -1,  0)), v);
      v = cs.getValue< 0, 1, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0,  1,  0)), v);
      v = cs.getValue< 0, 0,-1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0,  0, -1)), v);
      v = cs.getValue< 0, 0, 1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0,  0,  1)), v);

      v = cs.getValue<-1,-1, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy(-1, -1,  0)), v);
      v = cs.getValue< 1,-1, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 1, -1,  0)), v);
      v = cs.getValue<-1, 1, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy(-1,  1,  0)), v);
      v = cs.getValue< 1,1, 0>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 1,  1,  0)), v);

      v = cs.getValue<-1, 0, -1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy(-1, 0, -1)), v);
      v = cs.getValue< 1, 0, -1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 1, 0, -1)), v);
      v = cs.getValue<-1, 0, 1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy(-1, 0, 1)), v);
      v = cs.getValue< 1, 0, 1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 1, 0, 1)), v);

      v = cs.getValue< 0, -1, -1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0, -1, -1)), v);
      v = cs.getValue< 0, 1, -1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0, 1, -1)), v);
      v = cs.getValue< 0, -1, 1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0, -1, 1)), v);
      v = cs.getValue< 0, 1, 1>();
      EXPECT_EQ(acc.getValue(ijk.offsetBy( 0, 1, 1)), v);

      //std::cerr << "Mean curvature = "     << cs.meanCurvature()     << ", 1/r=" << 1.0/radius << std::endl;
      //std::cerr << "Gaussian curvature = " << cs.gaussianCurvature() << ", 1/(r*r)=" << 1.0/(radius*radius) << std::endl;
      EXPECT_NEAR(1.0/radius,  cs.meanCurvature(), percentage*1.0/radius);
      EXPECT_NEAR(1.0/(radius*radius),  cs.gaussianCurvature(), percentage*1.0/(radius*radius));
      float mean, gauss;
      cs.curvatures(mean, gauss);
      //std::cerr << "Mean curvature = "     << mean     << ", 1/r=" << 1.0/radius << std::endl;
      //std::cerr << "Gaussian curvature = " << gauss << ", 1/(r*r)=" << 1.0/(radius*radius) << std::endl;
      EXPECT_NEAR(1.0/radius,  mean, percentage*1.0/radius);
      EXPECT_NEAR(1.0/(radius*radius),  gauss, percentage*1.0/(radius*radius));
    }

}// CurvatureStencil

TEST_F(TestNanoVDB, GradStencil)
{
    {// test of level set to sphere at (6,8,10) with R=10 and dx=0.5
        const float radius = 10.0f;// 20 voxels
        const nanovdb::Vec3f center(6.0, 8.0, 10.0);//i.e. (12,16,20) in index space
        auto handle = nanovdb::createLevelSetSphere(radius,
                                                    center,
                                                    0.5, // dx
                                                    20.0);// width, so dense inside

        EXPECT_TRUE(handle);
        EXPECT_EQ(1u, handle.gridCount());
        auto* grid = handle.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(0.5f, grid->voxelSize()[0]);

        nanovdb::GradStencil<nanovdb::FloatGrid> cs(*grid);

        nanovdb::Coord ijk(12, 16, 20);// on the surface in the +x direction
        const nanovdb::Vec3d xyz(ijk[0], ijk[1], ijk[2]);
        EXPECT_NEAR(center[0], grid->indexToWorld(xyz)[0], 1e-6);
        EXPECT_NEAR(center[1], grid->indexToWorld(xyz)[1], 1e-6);
        EXPECT_NEAR(center[2], grid->indexToWorld(xyz)[2], 1e-6);
        cs.moveTo(ijk.offsetBy(20, 0, 0));// on the sphere
        const float val = cs.getValue<0,0,0>();
        EXPECT_NEAR( 0.0f, val, 1e-6);// on the sphere

        EXPECT_NEAR( 1.0f, cs.normSqGrad(), 2e-3f);

        EXPECT_TRUE( cs.zeroCrossing() );

        //std::cerr << cs.gradient() << std::endl;//second order
        EXPECT_NEAR( 1.0f, cs.gradient()[0], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient()[1], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient()[2], 1e-6f);

        const nanovdb::Vec3f v(-1, 0, 0);// upwind direction
        //std::cerr << cs.gradient(v) << std::endl;// first order
        EXPECT_NEAR( 1.0f, cs.gradient(v)[0], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient(v)[1], 2.5e-2f);
        EXPECT_NEAR( 0.0f, cs.gradient(v)[2], 2.5e-2f);

        // mean curvature = 0.5 * laplace of SDF => laplacian = 2 * mean curvature = 2 / radius
        //std::cerr << "Laplacian = " << cs.laplacian() << " " << (2/radius) << std::endl;
        EXPECT_NEAR(cs.laplacian(), 2/radius, 1e-3);
    }
}// GradStencil

TEST_F(TestNanoVDB, WenoStencil)
{
    {// test of level set to sphere at (6,8,10) with R=10 and dx=0.5
        const float radius = 10.0f;// 20 voxels
        const nanovdb::Vec3f center(6.0, 8.0, 10.0);//i.e. (12,16,20) in index space
        auto handle = nanovdb::createLevelSetSphere(radius,
                                                    center,
                                                    0.5, // dx
                                                    20.0);// width, so dense inside

        EXPECT_TRUE(handle);
        EXPECT_EQ(1u, handle.gridCount());
        auto* grid = handle.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(0.5f, grid->voxelSize()[0]);

        nanovdb::WenoStencil<nanovdb::FloatGrid> cs(*grid);

        nanovdb::Coord ijk(12, 16, 20);// on the surface in the +x direction
        const nanovdb::Vec3d xyz(ijk[0], ijk[1], ijk[2]);
        EXPECT_NEAR(center[0], grid->indexToWorld(xyz)[0], 1e-6);
        EXPECT_NEAR(center[1], grid->indexToWorld(xyz)[1], 1e-6);
        EXPECT_NEAR(center[2], grid->indexToWorld(xyz)[2], 1e-6);
        cs.moveTo(ijk.offsetBy(20, 0, 0));// on the sphere
        const float val = cs.getValue<0,0,0>();
        EXPECT_NEAR( 0.0f, val, 1e-6);// on the sphere

        EXPECT_NEAR( 1.0f, cs.normSqGrad(), 1e-6f);

        EXPECT_TRUE( cs.zeroCrossing() );

        //std::cerr << cs.gradient() << std::endl;
        EXPECT_NEAR( 1.0f, cs.gradient()[0], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient()[1], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient()[2], 1e-6f);

        const nanovdb::Vec3f v(-1, 0, 0);// upwind direction
        //std::cerr << cs.gradient(v) << std::endl;
        EXPECT_NEAR( 1.0f, cs.gradient(v)[0], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient(v)[1], 1e-6f);
        EXPECT_NEAR( 0.0f, cs.gradient(v)[2], 1e-6f);

        // mean curvature = 0.5 * laplace of SDF => laplacian = 2 * mean curvature = 2 / radius
        //std::cerr << "Laplacian = " << cs.laplacian() << " " << (2/radius) << std::endl;
        EXPECT_NEAR(cs.laplacian(), 2/radius, 1e-3);
    }
}// WenoStencil

TEST_F(TestNanoVDB, StencilIntersection)
{
  const nanovdb::Coord ijk(1,4,-9);
  nanovdb::GridBuilder<float> builder(0.0f);
  auto acc = builder.getAccessor();
  acc.setValue(ijk,-1.0f);
  int cases = 0;
  for (int mx=0; mx<2; ++mx) {
    acc.setValue(ijk.offsetBy(-1,0,0), mx ? 1.0f : -1.0f);
    for (int px=0; px<2; ++px) {
      acc.setValue(ijk.offsetBy(1,0,0), px ? 1.0f : -1.0f);
      for (int my=0; my<2; ++my) {
        acc.setValue(ijk.offsetBy(0,-1,0), my ? 1.0f : -1.0f);
        for (int py=0; py<2; ++py) {
          acc.setValue(ijk.offsetBy(0,1,0), py ? 1.0f : -1.0f);
          for (int mz=0; mz<2; ++mz) {
            acc.setValue(ijk.offsetBy(0,0,-1), mz ? 1.0f : -1.0f);
            for (int pz=0; pz<2; ++pz) {
              acc.setValue(ijk.offsetBy(0,0,1), pz ? 1.0f : -1.0f);
              ++cases;
              auto handle = builder.getHandle<>();
              EXPECT_TRUE(handle);
              auto grid = handle.grid<float>();
              EXPECT_TRUE(grid);
              EXPECT_EQ(7, int(grid->activeVoxelCount()));
              nanovdb::GradStencil<nanovdb::FloatGrid> stencil(*grid);
              stencil.moveTo(ijk);
              const int count = mx + px + my + py + mz + pz;// number of intersections
              EXPECT_TRUE(stencil.intersects() == (count > 0));
              auto mask = stencil.intersectionMask();
              EXPECT_TRUE(mask.none() == (count == 0));
              EXPECT_TRUE(mask.any() == (count > 0));
              EXPECT_EQ(count, mask.count());
              EXPECT_TRUE(mask.test(0) == mx);
              EXPECT_TRUE(mask.test(1) == px);
              EXPECT_TRUE(mask.test(2) == my);
              EXPECT_TRUE(mask.test(3) == py);
              EXPECT_TRUE(mask.test(4) == mz);
              EXPECT_TRUE(mask.test(5) == pz);
            }//pz
          }//mz
        }//py
      }//my
    }//px
  }//mx
  EXPECT_EQ(64, cases);// = 2^6
}// StencilIntersection

TEST_F(TestNanoVDB, MultiFile)
{
    std::vector<nanovdb::GridHandle<>> handles;
    { // add an int32_t grid
        nanovdb::GridBuilder<int> builder(-1);
        auto acc = builder.getAccessor();
        acc.setValue(nanovdb::Coord(-256), 10);
        handles.push_back(builder.getHandle(1.0, nanovdb::Vec3d(0), "Int32 grid"));
    }
    { // add an empty int32_t grid
        nanovdb::GridBuilder<int> builder(-4);
        handles.push_back(builder.getHandle(1.0, nanovdb::Vec3d(0), "Int32 grid, empty"));
    }
    { // add a Vec3f grid
        nanovdb::GridBuilder<nanovdb::Vec3f> builder(nanovdb::Vec3f(0.0f, 0.0f, -1.0f));
        builder.setGridClass(nanovdb::GridClass::Staggered);
        auto acc = builder.getAccessor();
        acc.setValue(nanovdb::Coord(-256), nanovdb::Vec3f(1.0f, 0.0f, 0.0f));
        handles.push_back(builder.getHandle(1.0, nanovdb::Vec3d(0), "Float vector grid"));
    }
    { // add an int64_t grid
        nanovdb::GridBuilder<int64_t> builder(0);
        auto acc = builder.getAccessor();
        acc.setValue(nanovdb::Coord(0), 10);
        handles.push_back(builder.getHandle(1.0, nanovdb::Vec3d(0), "Int64 grid"));
    }
    for (int i = 0; i < 10; ++i) {
        const float          radius = 100.0f;
        const float          voxelSize = 1.0f, width = 3.0f;
        const nanovdb::Vec3f center(i * 10.0f, 0.0f, 0.0f);
        handles.push_back(nanovdb::createLevelSetSphere(radius, center, voxelSize, width,
                          nanovdb::Vec3d(0), "Level set sphere at (" + std::to_string(i * 10) + ",0,0)"));
    }
    { // add a double grid
        nanovdb::GridBuilder<double> builder(0.0);
        builder.setGridClass(nanovdb::GridClass::FogVolume);
        auto acc = builder.getAccessor();
        acc.setValue(nanovdb::Coord(6000), 1.0);
        handles.push_back(builder.getHandle(1.0, nanovdb::Vec3d(0), "Double grid"));
    }
#if defined(NANOVDB_USE_BLOSC)
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/multi1.nvdb", handles, nanovdb::io::Codec::BLOSC);
#elif defined(NANOVDB_USE_ZIP)
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/multi1.nvdb", handles, nanovdb::io::Codec::ZIP);
#else
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("data/multi1.nvdb", handles, nanovdb::io::Codec::NONE);
#endif
    { // read grid meta data and test it
        //mTimer.start("nanovdb::io::readGridMetaData");
        auto meta = nanovdb::io::readGridMetaData("data/multi1.nvdb");
        //mTimer.stop();
        EXPECT_EQ(15u, meta.size());
        EXPECT_EQ(std::string("Double grid"), meta.back().gridName);
    }
    { // read in32 grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi1.nvdb");
        //mTimer.stop();
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles.front();
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_EQ(std::string("Int32 grid"), handle.gridMetaData()->shortGridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        auto* grid = handle.grid<int32_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        const auto&          tree = grid->tree();
        EXPECT_EQ(10, tree.getValue(ijk));
        EXPECT_EQ(-1, tree.getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(10, tree.root().minimum());
        EXPECT_EQ(10, tree.root().maximum());
        EXPECT_TRUE(grid->tree().isActive(ijk));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord( 10, 450, 90)));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord(-10,-450,-90)));
        EXPECT_FALSE(grid->tree().isActive(ijk + nanovdb::Coord(1, 0, 0)));
        const nanovdb::CoordBBox bbox(ijk, ijk);
        EXPECT_EQ(bbox, grid->indexBBox());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(1u, tree.nodeCount(0));
        EXPECT_EQ(1u, tree.nodeCount(1));
        EXPECT_EQ(1u, tree.nodeCount(2));
        const auto* leaf = tree.getFirstNode<0>();
        EXPECT_TRUE(leaf);
        EXPECT_EQ(bbox, leaf->bbox());
        const auto* node1 = tree.getFirstNode<1>();
        EXPECT_TRUE(node1);
        EXPECT_EQ(bbox, node1->bbox());
        const auto* node2 = tree.getFirstNode<2>();
        EXPECT_TRUE(node2);
        EXPECT_EQ(bbox, node2->bbox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read empty in32 grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi1.nvdb");
        //mTimer.stop();
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles[1];
        EXPECT_TRUE(handle);
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_EQ(std::string("Int32 grid, empty"), handle.gridMetaData()->shortGridName());
        EXPECT_FALSE(handle.grid<float>());
        EXPECT_FALSE(handle.grid<double>());
        EXPECT_FALSE(handle.grid<int64_t>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3f>());
        EXPECT_FALSE(handle.grid<nanovdb::Vec3d>());
        auto* grid = handle.grid<int32_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(0u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(-4, grid->tree().getValue(ijk));
        EXPECT_EQ(-4, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_FALSE(grid->tree().isActive(ijk));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord( 10, 450, 90)));
        EXPECT_FALSE(grid->tree().isActive(nanovdb::Coord(-10,-450,-90)));
        EXPECT_FALSE(grid->tree().isActive(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(-4, grid->tree().root().minimum());
        EXPECT_EQ(-4, grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::Coord(std::numeric_limits<int>::max()), grid->indexBBox().min());
        EXPECT_EQ(nanovdb::Coord(std::numeric_limits<int>::min()), grid->indexBBox().max());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(0u, grid->tree().nodeCount(0));
        EXPECT_EQ(0u, grid->tree().nodeCount(1));
        EXPECT_EQ(0u, grid->tree().nodeCount(2));
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isMask());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read int64 grid and test values
        //mTimer.start("Reading multiple grids from file");
        auto handles = nanovdb::io::readGrids("data/multi1.nvdb");
        //mTimer.stop();
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles[3];
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_TRUE(handle);
        EXPECT_EQ(std::string("Int64 grid"), handle.gridMetaData()->shortGridName());
        auto* grid = handle.grid<int64_t>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(0);
        EXPECT_EQ(10, grid->tree().getValue(ijk));
        EXPECT_EQ(0, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(10, grid->tree().root().minimum());
        EXPECT_EQ(10, grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_TRUE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
    { // read vec3f grid and test values
        auto handles = nanovdb::io::readGrids("data/multi1.nvdb");
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles[2];
        EXPECT_TRUE(handle);
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_EQ(std::string("Float vector grid"), handle.gridMetaData()->shortGridName());
        auto* grid = handle.grid<nanovdb::Vec3f>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(1u, grid->activeVoxelCount());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        const nanovdb::Coord ijk(-256);
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().getValue(ijk));
        EXPECT_EQ(nanovdb::Vec3f(0.0f, 0.0f, -1.0f), grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().root().minimum());
        EXPECT_EQ(nanovdb::Vec3f(1.0f, 0.0f, 0.0f), grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_FALSE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_TRUE(grid->isStaggered());
    }
    { // read double grid and test values
        auto handles = nanovdb::io::readGrids("data/multi1.nvdb");
        EXPECT_EQ(15u, handles.size());
        auto& handle = handles.back();
        EXPECT_TRUE(handle);
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_EQ(std::string("Double grid"), handle.gridMetaData()->shortGridName());
        auto* grid = handle.grid<double>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(1u, grid->activeVoxelCount());
        const nanovdb::Coord ijk(6000);
        EXPECT_EQ(1.0, grid->tree().getValue(ijk));
        EXPECT_EQ(0.0, grid->tree().getValue(ijk + nanovdb::Coord(1, 0, 0)));
        EXPECT_EQ(1.0, grid->tree().root().minimum());
        EXPECT_EQ(1.0, grid->tree().root().maximum());
        EXPECT_EQ(nanovdb::CoordBBox(ijk, ijk), grid->tree().bbox());
        EXPECT_EQ(handle.gridMetaData()->indexBBox(), grid->indexBBox());
        EXPECT_FALSE(grid->isLevelSet());
        EXPECT_TRUE(grid->isFogVolume());
        EXPECT_FALSE(grid->isUnknown());
        EXPECT_FALSE(grid->isStaggered());
    }
} // MultiFile

TEST_F(TestNanoVDB, HostBuffer)
{
    {// internal memory - HostBuffer
        std::vector<nanovdb::GridHandle<> > gridHdls;

        // create two grids...
        gridHdls.push_back(nanovdb::createLevelSetSphere(100.0f, nanovdb::Vec3f(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3R(0), "spheref"));
        gridHdls.push_back(nanovdb::createLevelSetSphere(100.0,  nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3R(0), "sphered"));

        EXPECT_TRUE(gridHdls[0]);
        auto* meta0 = gridHdls[0].gridMetaData();
        EXPECT_TRUE(meta0);
        EXPECT_FALSE(meta0->isEmpty());
        EXPECT_EQ("spheref", std::string(meta0->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta0->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta0->gridClass());
        auto* grid0 = gridHdls[0].grid<float>();
        EXPECT_TRUE(grid0);
        auto acc0 = grid0->getAccessor();
        EXPECT_EQ(0.0f, acc0.getValue(nanovdb::Coord(-20+100, 0, 0)));

        EXPECT_TRUE(gridHdls[1]);
        auto* meta1 = gridHdls[1].gridMetaData();
        EXPECT_TRUE(meta1);
        EXPECT_FALSE(meta1->isEmpty());
        EXPECT_EQ("sphered", std::string(meta1->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Double, meta1->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta1->gridClass());
        auto* grid1 = gridHdls[1].grid<double>();
        EXPECT_TRUE(grid1);
        auto acc1 = grid1->getAccessor();
        EXPECT_EQ(0.0, acc1.getValue(nanovdb::Coord( 20+100, 0, 0)));
    }
    {// internal memory - bump pool
        const size_t poolSize = 1 << 26;// 64 MB
        auto pool = nanovdb::HostBuffer::createPool(poolSize);
        EXPECT_TRUE(pool.isManaged());
        EXPECT_EQ(64ULL * 1024 * 1024, pool.poolSize());
        EXPECT_TRUE(pool.isPool());
        EXPECT_TRUE(pool.isEmpty());
        EXPECT_FALSE(pool.isFull());

        std::vector<nanovdb::GridHandle<> > gridHdls;

        // create two grids...
        gridHdls.push_back(nanovdb::createLevelSetSphere(100.0f, nanovdb::Vec3f(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3R(0), "spheref", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool));
        gridHdls.push_back(nanovdb::createLevelSetSphere(100.0,  nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3R(0), "sphered", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool));

        EXPECT_TRUE(gridHdls[0]);
        auto* meta0 = gridHdls[0].gridMetaData();
        EXPECT_TRUE(meta0);
        EXPECT_FALSE(meta0->isEmpty());
        EXPECT_EQ("spheref", std::string(meta0->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta0->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta0->gridClass());
        auto* grid0 = gridHdls[0].grid<float>();
        //printf("Before resize: address of grid0 = %p\n", (void*)grid0);
        EXPECT_TRUE(grid0);
        auto acc0 = grid0->getAccessor();
        EXPECT_EQ(0.0f, acc0.getValue(nanovdb::Coord(-20+100, 0, 0)));

        EXPECT_TRUE(gridHdls[1]);
        auto* meta1 = gridHdls[1].gridMetaData();
        EXPECT_TRUE(meta1);
        EXPECT_FALSE(meta1->isEmpty());
        EXPECT_EQ("sphered", std::string(meta1->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Double, meta1->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta1->gridClass());
        auto* grid1 = gridHdls[1].grid<double>();
        EXPECT_TRUE(grid1);
        auto acc1 = grid1->getAccessor();
        EXPECT_EQ(0.0, acc1.getValue(nanovdb::Coord( 20+100, 0, 0)));

        pool.resizePool( 2*poolSize );
        EXPECT_TRUE(pool.isManaged());
        EXPECT_EQ(128ULL * 1024 * 1024, pool.poolSize());
        EXPECT_TRUE(pool.isPool());
        EXPECT_TRUE(pool.isEmpty());// because this buffer does not use the pool
        EXPECT_FALSE(pool.isFull());

        EXPECT_TRUE(gridHdls[0]);
        meta0 = gridHdls[0].gridMetaData();
        EXPECT_TRUE(meta0);
        EXPECT_FALSE(meta0->isEmpty());
        EXPECT_EQ("spheref", std::string(meta0->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta0->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta0->gridClass());
        grid0 = gridHdls[0].grid<float>();
        //printf("After  resize: address of grid0 = %p\n", (void*)grid0);
        EXPECT_TRUE(grid0);
        acc0 = grid0->getAccessor();
        EXPECT_EQ(0.0f, acc0.getValue(nanovdb::Coord(-20+100, 0, 0)));

        EXPECT_TRUE(gridHdls[1]);
        meta1 = gridHdls[1].gridMetaData();
        EXPECT_TRUE(meta1);
        EXPECT_FALSE(meta1->isEmpty());
        EXPECT_EQ("sphered", std::string(meta1->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Double, meta1->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta1->gridClass());
        grid1 = gridHdls[1].grid<double>();
        EXPECT_TRUE(grid1);
        acc1 = grid1->getAccessor();
        EXPECT_EQ(0.0, acc1.getValue(nanovdb::Coord( 20+100, 0, 0)));

        pool.reset();
        EXPECT_TRUE(pool.isManaged());
        EXPECT_EQ(128ULL * 1024 * 1024, pool.poolSize());
        EXPECT_TRUE(pool.isPool());
        EXPECT_TRUE(pool.isEmpty());// because this buffer does not use the pool
        EXPECT_FALSE(pool.isFull());

        EXPECT_FALSE(gridHdls[0]);
        EXPECT_FALSE(gridHdls[1]);
    }
    {// insufficient internal memory
        const size_t poolSize = 1 << 6;// 64 B
        auto pool = nanovdb::HostBuffer::createPool(poolSize);
        EXPECT_EQ(64ULL, pool.poolSize());
        EXPECT_TRUE(pool.isPool());
        EXPECT_TRUE(pool.isEmpty());
        EXPECT_FALSE(pool.isFull());

        std::vector<nanovdb::GridHandle<> > gridHdls;

        // create two grids...
        ASSERT_THROW(gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0f, nanovdb::Vec3f(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "spheref", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool)), std::runtime_error);
        ASSERT_THROW(gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0,  nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphered", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool)), std::runtime_error);
    }
    {// zero internal memory size
        ASSERT_THROW(nanovdb::HostBuffer::createPool(0), std::runtime_error);
        ASSERT_THROW(nanovdb::HostBuffer::createFull(0), std::runtime_error);
    }
    {// external memory

        const size_t poolSize = 1 << 26;// 64 MB
        std::unique_ptr<uint8_t[]> buffer(new uint8_t[poolSize]);
        auto pool = nanovdb::HostBuffer::createPool(poolSize, buffer.get());
        EXPECT_EQ(64ULL * 1024 * 1024, pool.poolSize());
        EXPECT_FALSE(pool.isManaged());
        EXPECT_TRUE(pool.isPool());
        EXPECT_TRUE(pool.isEmpty());
        EXPECT_FALSE(pool.isFull());

        std::vector<nanovdb::GridHandle<> > gridHdls;

        // create two grids...
        gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0f, nanovdb::Vec3f(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "spheref", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool));
        gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0,  nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphered", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool));

        EXPECT_TRUE(gridHdls[0]);
        auto* meta0 = gridHdls[0].gridMetaData();
        EXPECT_TRUE(meta0);
        EXPECT_FALSE(meta0->isEmpty());
        EXPECT_EQ("spheref", std::string(meta0->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta0->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta0->gridClass());
        auto* grid0 = gridHdls[0].grid<float>();
        EXPECT_TRUE(grid0);
        auto acc0 = grid0->getAccessor();
        EXPECT_EQ(0.0f, acc0.getValue(nanovdb::Coord(-20+100, 0, 0)));

        EXPECT_TRUE(gridHdls[1]);
        auto* meta1 = gridHdls[1].gridMetaData();
        EXPECT_TRUE(meta1);
        EXPECT_FALSE(meta1->isEmpty());
        EXPECT_EQ("sphered", std::string(meta1->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Double, meta1->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta1->gridClass());
        auto* grid1 = gridHdls[1].grid<double>();
        EXPECT_TRUE(grid1);
        auto acc1 = grid1->getAccessor();
        EXPECT_EQ(0.0, acc1.getValue(nanovdb::Coord( 20+100, 0, 0)));

        pool.reset();

        EXPECT_FALSE(gridHdls[0]);
        EXPECT_FALSE(gridHdls[1]);
    }
    {// insufficient external memory
        const size_t poolSize = 64;// 64 B
        uint8_t *data = static_cast<uint8_t*>(std::malloc(poolSize));
        auto pool = nanovdb::HostBuffer::createPool(poolSize, data);
        EXPECT_EQ(0ULL, pool.size());
        EXPECT_EQ(64ULL, pool.poolSize());
        EXPECT_EQ(0ULL, pool.poolUsage());
        EXPECT_TRUE(pool.isPool());
        EXPECT_TRUE(pool.isEmpty());
        EXPECT_FALSE(pool.isFull());
        EXPECT_FALSE(pool.isManaged());

        auto buffer = nanovdb::HostBuffer::create(32, &pool);
        EXPECT_EQ(32ULL, buffer.size());
        EXPECT_EQ(64ULL, buffer.poolSize());
        EXPECT_EQ(32ULL,  pool.poolUsage());
        EXPECT_FALSE(buffer.isPool());
        EXPECT_FALSE(buffer.isEmpty());
        EXPECT_FALSE(buffer.isFull());
        EXPECT_FALSE(buffer.isManaged());

        std::vector<nanovdb::GridHandle<> > gridHdls;

        // create two grids...
        ASSERT_THROW(gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0f, nanovdb::Vec3f(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "spheref", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool)), std::runtime_error);
        ASSERT_THROW(gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0,  nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphered", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool)), std::runtime_error);

        EXPECT_FALSE(pool.isManaged());
        pool.resizePool(1<<26);// resize to 64 MB
        EXPECT_TRUE(pool.isManaged());
        std::free(data);

        EXPECT_EQ(0ULL, pool.size());
        EXPECT_EQ(1ULL<<26, pool.poolSize());
        EXPECT_TRUE(pool.isPool());
        EXPECT_TRUE(pool.isEmpty());
        EXPECT_FALSE(pool.isFull());
        EXPECT_TRUE(pool.isManaged());

        EXPECT_EQ(32ULL, buffer.size());
        EXPECT_EQ(1ULL<<26, buffer.poolSize());
        EXPECT_FALSE(buffer.isPool());
        EXPECT_FALSE(buffer.isEmpty());
        EXPECT_FALSE(buffer.isFull());
        EXPECT_TRUE(buffer.isManaged());

        gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0f, nanovdb::Vec3f(-20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "spheref", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool));
        gridHdls.push_back(nanovdb::createLevelSetSphere( 100.0,  nanovdb::Vec3d( 20, 0, 0), 1.0, 3.0, nanovdb::Vec3d(0), "sphered", nanovdb::StatsMode::BBox, nanovdb::ChecksumMode::Partial, -1.0f, false, pool));

        EXPECT_TRUE(gridHdls[0]);
        auto* meta0 = gridHdls[0].gridMetaData();
        EXPECT_TRUE(meta0);
        EXPECT_FALSE(meta0->isEmpty());
        EXPECT_EQ("spheref", std::string(meta0->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta0->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta0->gridClass());
        auto* grid0 = gridHdls[0].grid<float>();
        EXPECT_TRUE(grid0);
        auto acc0 = grid0->getAccessor();
        EXPECT_EQ(0.0f, acc0.getValue(nanovdb::Coord(-20+100, 0, 0)));

        EXPECT_TRUE(gridHdls[1]);
        auto* meta1 = gridHdls[1].gridMetaData();
        EXPECT_TRUE(meta1);
        EXPECT_FALSE(meta1->isEmpty());
        EXPECT_EQ("sphered", std::string(meta1->shortGridName()));
        EXPECT_EQ(nanovdb::GridType::Double, meta1->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta1->gridClass());
        auto* grid1 = gridHdls[1].grid<double>();
        EXPECT_TRUE(grid1);
        auto acc1 = grid1->getAccessor();
        EXPECT_EQ(0.0, acc1.getValue(nanovdb::Coord( 20+100, 0, 0)));

        pool.reset();
        EXPECT_EQ(0ULL, pool.poolUsage());
        EXPECT_TRUE(pool.isManaged());

        EXPECT_FALSE(gridHdls[0]);
        EXPECT_FALSE(gridHdls[1]);
    }
    {// zero external memory size
        const size_t poolSize = 1 << 6;// 64 B
        uint8_t *data = static_cast<uint8_t*>(std::malloc(poolSize));
        ASSERT_THROW(nanovdb::HostBuffer::createPool(0, data), std::runtime_error);
        std::free(data);
    }
    try {// reading multiple grids into a HostBuffer with external memory
        const size_t poolSize = 1 << 27;// 128 MB
        std::unique_ptr<uint8_t[]> array(new uint8_t[poolSize]);// scoped buffer
        auto pool = nanovdb::HostBuffer::createPool(poolSize, array.get());
        EXPECT_EQ(128ULL * 1024 * 1024, pool.poolSize());
        auto handles = nanovdb::io::readGrids("data/multi1.nvdb", 0, pool);
        EXPECT_EQ(15u, handles.size());
        for (auto &h : handles) EXPECT_TRUE(h);
        EXPECT_EQ(std::string("Int32 grid"), handles[0].grid<int>()->gridName());
        EXPECT_EQ(std::string("Int32 grid, empty"), handles[1].grid<int>()->gridName());
        EXPECT_EQ(std::string("Float vector grid"), handles[2].grid<nanovdb::Vec3f>()->gridName());
        EXPECT_EQ(std::string("Int64 grid"), handles[3].grid<int64_t>()->gridName());
        EXPECT_EQ(std::string("Double grid"), handles[14].grid<double>()->gridName());
        pool.reset();
        for (auto &h : handles) EXPECT_FALSE(h);
        handles = nanovdb::io::readGrids("data/multi1.nvdb", 0, pool);
        EXPECT_EQ(15u, handles.size());
        for (auto &h : handles) EXPECT_TRUE(h);
        EXPECT_EQ(std::string("Int32 grid"), handles[0].grid<int>()->gridName());
        EXPECT_EQ(std::string("Int32 grid, empty"), handles[1].grid<int>()->gridName());
        EXPECT_EQ(std::string("Float vector grid"), handles[2].grid<nanovdb::Vec3f>()->gridName());
        EXPECT_EQ(std::string("Int64 grid"), handles[3].grid<int64_t>()->gridName());
        EXPECT_EQ(std::string("Double grid"), handles[14].grid<double>()->gridName());
    } catch(const std::exception& e) {
        std::cout << "Unable to read \"data/multi1.nvdb\" for unit-test\n" << e.what() << std::endl;
    }
}// HostBuffer

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
