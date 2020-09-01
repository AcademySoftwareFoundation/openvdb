// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <iostream>
#include <cstdlib>
#include <sstream> // for std::stringstream
#include <vector>
#include <algorithm> // for std::is_sorted
#define _USE_MATH_DEFINES
#include <cmath>

#include "gtest/gtest.h"

#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <nanovdb/CNanoVDB.h>
#include <nanovdb/util/Range.h>
#include "../examples/ex_util/CpuTimer.h"

inline std::ostream&
operator<<(std::ostream& os, const nanovdb::CoordBBox& b)
{
    os << "(" << b[0][0] << "," << b[0][1] << "," << b[0][2] << ") ->"
       << "(" << b[1][0] << "," << b[1][1] << "," << b[1][2] << ")";
    return os;
}

inline std::ostream&
operator<<(std::ostream& os, const nanovdb::Coord& ijk)
{
    os << "(" << ijk[0] << "," << ijk[1] << "," << ijk[2] << ")";
    return os;
}

template <typename T>
inline std::ostream&
operator<<(std::ostream& os, const nanovdb::Vec3<T>& v)
{
    os << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
    return os;
}

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
    /*
    nanovdb::GridHandle<> getSrcGrid(int verbose = 1)
    {
    }
*/
    nanovdb::CpuTimer<> mTimer;
}; // TestNanoVDB

TEST_F(TestNanoVDB, Basic)
{
    {
        std::vector<int> v = {3, 1, 7, 0};
        EXPECT_FALSE(std::is_sorted(v.begin(), v.end()));
        std::map<int, void*> m;
        for (const auto& i : v) m[i] = nullptr;
        v.clear();
        for (const auto& i : m) v.push_back(i.first);
        EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
    }
    {
        enum tmp { a = 0,
                   b,
                   c,
                   d } t;
        EXPECT_EQ(sizeof(int), sizeof(t));
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
    EXPECT_EQ(21, NANOVDB_MAJOR_VERSION_NUMBER);
    EXPECT_EQ(0, NANOVDB_MINOR_VERSION_NUMBER);
    EXPECT_EQ(0x304244566f6e614eUL, NANOVDB_MAGIC_NUMBER); // Magic number: "NanoVDB0" in hex)
    EXPECT_EQ(0x4e616e6f56444230UL, nanovdb::io::reverseEndianness(NANOVDB_MAGIC_NUMBER));

    // Verify little endian representation
    const char* str = "NanoVDB0"; // note it's exactly 8 bytes
    EXPECT_EQ(8u, strlen(str));
    std::stringstream ss1;
    ss1 << "0x";
    for (int i = 7; i>=0; --i)
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
    EXPECT_EQ(  0, r1[0].end());

    EXPECT_EQ(2U, r1[1].grainsize());
    EXPECT_EQ(20U, r1[1].size());
    EXPECT_EQ(10, r1[1].middle());
    EXPECT_TRUE(r1[1].is_divisible());
    EXPECT_EQ(0, r1[1].begin());
    EXPECT_EQ(20, r1[1].end());

    EXPECT_FALSE(r2.empty());
    EXPECT_EQ(1U, r2[0].grainsize());
    EXPECT_EQ(20U, r2[0].size());
    EXPECT_EQ( 10, r2[0].middle());
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
    EXPECT_EQ(  0, r1[0].end());

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
    EXPECT_EQ( 10, r2[0].middle());
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

TEST_F(TestNanoVDB, Coord)
{
    EXPECT_EQ(size_t(3 * 4), nanovdb::Coord::memUsage()); // due to padding
    {
        nanovdb::Coord ijk;
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
    EXPECT_EQ( std::numeric_limits<float>::max(), bbox[0][0]);
    EXPECT_EQ( std::numeric_limits<float>::max(), bbox[0][1]);
    EXPECT_EQ( std::numeric_limits<float>::max(), bbox[0][2]);
    EXPECT_EQ(-std::numeric_limits<float>::max(), bbox[1][0]);
    EXPECT_EQ(-std::numeric_limits<float>::max(), bbox[1][1]);
    EXPECT_EQ(-std::numeric_limits<float>::max(), bbox[1][2]);
    EXPECT_TRUE(bbox.empty());

    bbox.expand(nanovdb::Vec3f(57.0f, -31.0f, 60.0f));
    EXPECT_TRUE(bbox.empty());
    EXPECT_EQ(nanovdb::Vec3f(0.0f), bbox.dim());
    EXPECT_EQ( 57.0f, bbox[0][0]);
    EXPECT_EQ(-31.0f, bbox[0][1]);
    EXPECT_EQ( 60.0f, bbox[0][2]);
    EXPECT_EQ( 57.0f, bbox[1][0]);
    EXPECT_EQ(-31.0f, bbox[1][1]);
    EXPECT_EQ( 60.0f, bbox[1][2]);

    bbox.expand(nanovdb::Vec3f(58.0f, 0.0f, 62.0f));
    EXPECT_FALSE(bbox.empty());
    EXPECT_EQ(nanovdb::Vec3f(1.0f, 31.0f, 2.0f), bbox.dim());
    EXPECT_EQ( 57.0f, bbox[0][0]);
    EXPECT_EQ(-31.0f, bbox[0][1]);
    EXPECT_EQ( 60.0f, bbox[0][2]);
    EXPECT_EQ( 58.0f, bbox[1][0]);
    EXPECT_EQ(  0.0f, bbox[1][1]);
    EXPECT_EQ( 62.0f, bbox[1][2]);
}

TEST_F(TestNanoVDB, CoordBBox)
{
    nanovdb::CoordBBox bbox;
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

    {// test convert
      auto bbox2 = bbox.asReal<float>();
      EXPECT_FALSE( bbox2.empty() );
      EXPECT_EQ(nanovdb::Vec3f(57.0f,-31.0f,60.0f), bbox2.min());
      EXPECT_EQ(nanovdb::Vec3f(59.0f,  1.0f,63.0f), bbox2.max());
    }

    {// test prefix iterator
        auto iter = bbox.begin();
        EXPECT_TRUE(iter);
        for (int i = bbox.min()[0]; i <= bbox.max()[0]; ++i) {
            for (int j = bbox.min()[1]; j <= bbox.max()[1]; ++j) {
                for (int k = bbox.min()[2]; k <= bbox.max()[2]; ++k) {
                    EXPECT_TRUE(bbox.isInside(*iter));
                    EXPECT_TRUE(iter);
                    const auto &ijk = *iter;// note, copy by reference
                    EXPECT_EQ( ijk[0] , i);
                    EXPECT_EQ( ijk[1] , j);
                    EXPECT_EQ( ijk[2] , k);
                    ++iter;
                }
            }
        }
        EXPECT_FALSE(iter);
    }
    
    {// test postfix iterator 
        auto iter = bbox.begin();
        EXPECT_TRUE(iter);
        for (int i = bbox.min()[0]; i <= bbox.max()[0]; ++i) {
            for (int j = bbox.min()[1]; j <= bbox.max()[1]; ++j) {
                for (int k = bbox.min()[2]; k <= bbox.max()[2]; ++k) {
                    EXPECT_TRUE(iter);
                    const auto ijk = *iter++;// note, copy by value!
                    EXPECT_EQ( ijk[0] , i);
                    EXPECT_EQ( ijk[1] , j);
                    EXPECT_EQ( ijk[2] , k);
                }
            }   
        }
        EXPECT_FALSE(iter);
    }
}

TEST_F(TestNanoVDB, Vec3R)
{
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
}

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
    EXPECT_TRUE(ray.clip(CoordBBoxT(CoordT(0, 2, 2), CoordT(2, 4, 6) - CoordT(1))));
    EXPECT_EQ(0.5, ray.t0());
    EXPECT_EQ(1.5, ray.t1());
    EXPECT_EQ(ray(0.5)[1], 2); //lower y component of intersection
    EXPECT_EQ(ray(1.5)[1], 4); //higher y component of intersection

    ray.reset(eye, dir, t0, t1);
    // intersects the lower edge anlong the z-axis of the box
    EXPECT_TRUE(ray.clip(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0))));
    EXPECT_EQ(0.5, ray.t0());
    EXPECT_EQ(0.5, ray.t1());
    EXPECT_EQ(ray(0.5)[0], 1.5); //lower y component of intersection
    EXPECT_EQ(ray(0.5)[1], 2.0); //higher y component of intersection

    ray.reset(eye, dir, t0, t1);
    // no intersections
    EXPECT_TRUE(!ray.clip(CoordBBoxT(CoordT(4, 2, 2), CoordT(6, 4, 6) - CoordT(1))));
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
    EXPECT_EQ(8u, LeafT::dim());
    EXPECT_EQ(512u, LeafT::voxelCount());
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(8 * 8 + // mValueMask
                                                       4 * 512 + // mValues[512]
                                                       2 * 4 + // mValueMin, mValueMax
                                                       3 * 4 + // mBBoxMin
                                                       4 * 1), /// mBBoxDif[3] + mFalgs
              sizeof(LeafT));

    // allocate buffer
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[LeafT::memUsage()]);
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
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
    }

    // check values
    auto* ptr = reinterpret_cast<LeafT::DataType*>(buffer.get())->values();
    for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
        if (i < 256) {
            EXPECT_FALSE(leaf->valueMask().isOn(i));
            EXPECT_EQ(0.0f, *ptr++);
        } else {
            EXPECT_TRUE(leaf->valueMask().isOn(i));
            EXPECT_EQ(1.234f, *ptr++);
        }
    }
    EXPECT_EQ(0.0f, leaf->valueMin());
    EXPECT_EQ(1.234f, leaf->valueMax());
} // LeafNode

TEST_F(TestNanoVDB, LeafNodeValueMask)
{
    using LeafT = nanovdb::LeafNode<nanovdb::ValueMask>;
    EXPECT_EQ(8u, LeafT::dim());
    EXPECT_EQ(512u, LeafT::voxelCount());
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(8 * 8 + // mValueMask
                                                       3 * 4 + // mBBoxMin
                                                       4 * 1), /// mBBoxDif[3] + mFalgs
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
    auto* ptr = reinterpret_cast<LeafT::DataType*>(buffer.get())->values();
    EXPECT_EQ(nullptr, ptr);
    for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
        if (i < 256) {
            EXPECT_FALSE(leaf->valueMask().isOn(i));
            EXPECT_EQ(true, leaf->getValue(i));
        } else {
            EXPECT_TRUE(leaf->valueMask().isOn(i));
            EXPECT_EQ(true, leaf->getValue(i));
        }
    }
    EXPECT_EQ(true, leaf->valueMin());
    EXPECT_EQ(true, leaf->valueMax());
} // LeafNodeValueMask

TEST_F(TestNanoVDB, InternalNode)
{
    using LeafT = nanovdb::LeafNode<float>;
    using NodeT = nanovdb::InternalNode<LeafT>;
    EXPECT_EQ(8 * 16u, NodeT::dim());
    //         2 x bit-masks         tiles    Vmin&Vmax offset + bbox + padding
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(size_t(2 * (16 * 16 * 16 / 64) * 8 + 16 * 16 * 16 * 4 + 2 * 4 + 4 + 2 * 3 * 4 + 4)), NodeT::memUsage());

    // an empty InternalNode
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[NodeT::memUsage()]);
    NodeT*                     node = reinterpret_cast<NodeT*>(buffer.get());

    { // set members of the node
        auto& data = *reinterpret_cast<NodeT::DataType*>(buffer.get());
        data.mValueMask.setOff();
        data.mChildMask.setOff();
        data.mOffset = 1;
        auto* tiles = data.mTable;
        for (uint32_t i = 0; i < NodeT::SIZE / 2; ++i, ++tiles)
            tiles->value = 0.0f;
        for (uint32_t i = NodeT::SIZE / 2; i < NodeT::SIZE; ++i, ++tiles) {
            data.mValueMask.setOn(i);
            tiles->value = 1.234f;
        }
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
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
    EXPECT_EQ(0.0f, node->valueMin());
    EXPECT_EQ(1.234f, node->valueMax());
} //  InternalNode

TEST_F(TestNanoVDB, RootNode)
{
    using NodeT0 = nanovdb::LeafNode<float>;
    using NodeT1 = nanovdb::InternalNode<NodeT0>;
    using NodeT2 = nanovdb::InternalNode<NodeT1>;
    using NodeT3 = nanovdb::RootNode<NodeT2>;
    using CoordT = NodeT3::CoordType;

    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(sizeof(nanovdb::CoordBBox) + sizeof(uint64_t) + size_t(3 * 4 + 4)), NodeT3::memUsage(0));

    // an empty RootNode
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[NodeT3::memUsage(0)]);
    NodeT3*                    root = reinterpret_cast<NodeT3*>(buffer.get());

    { // set members of the node
        auto& data = *reinterpret_cast<NodeT3::DataType*>(buffer.get());
        data.mBackground = data.mValueMin = data.mValueMax = 1.234f;
        data.mTileCount = 0;
    }

    EXPECT_EQ(1.234f, root->background());
    EXPECT_EQ(1.234f, root->valueMin());
    EXPECT_EQ(1.234f, root->valueMax());
    EXPECT_EQ(0u, root->tileCount());
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(sizeof(nanovdb::CoordBBox) + sizeof(uint64_t) + size_t(3 * 4 + 4)), root->memUsage()); // background, min, max, tileCount + bbox
    EXPECT_EQ(1.234f, root->getValue(CoordT(1, 2, 3)));

    { // test RootData::CoordToKey
        auto key = [](int i, int j, int k) { return NodeT3::DataType::CoordToKey(CoordT(i, j, k)); };
        EXPECT_TRUE(key(0, 0, 0) < key(4096, 0, 0));
        EXPECT_TRUE(key(0, 0, 0) < key(0, 4096, 0));
        EXPECT_TRUE(key(0, 0, 0) < key(0, 0, 4096));
        EXPECT_TRUE(key(0, 0, 0) == key(4095, 4095, 4095));
#ifdef USE_SINGLE_ROOT_KEY
        EXPECT_EQ(0u, key(0, 0, 0));
        EXPECT_EQ(1u, key(4095, 4095, 4096));
#endif
    }
} // RootNode

TEST_F(TestNanoVDB, Grid)
{
    using LeafT = nanovdb::LeafNode<float>;
    using NodeT1 = nanovdb::InternalNode<LeafT>;
    using NodeT2 = nanovdb::InternalNode<NodeT1>;
    using RootT = nanovdb::RootNode<NodeT2>;
    using TreeT = nanovdb::Tree<RootT>;
    using GridT = nanovdb::Grid<TreeT>;
    using CoordT = LeafT::CoordType;

    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(nanovdb::GridData::MaxNameSize + 8 + 8 * 2 * 3 + sizeof(nanovdb::Map) + 8 + 2 + 2 + 4), sizeof(GridT));
    EXPECT_EQ(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>((4 + 8) * (RootT::LEVEL + 1)), sizeof(TreeT));

    size_t bytes[6] = {GridT::memUsage(0), TreeT::memUsage(), RootT::memUsage(1), NodeT2::memUsage(), NodeT1::memUsage(), LeafT::memUsage()};
    for (int i = 1; i < 6; ++i)
        bytes[i] += bytes[i - 1]; // Byte offsets to: tree, root, internal nodes, leafs, total
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[bytes[5]]);

    // init leaf
    const LeafT* leaf = reinterpret_cast<LeafT*>(buffer.get() + bytes[4]);
    { // set members of the leaf node
        auto& data = *reinterpret_cast<LeafT::DataType*>(buffer.get() + bytes[4]);
        data.mValueMask.setOff();
        auto* voxels = data.mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount() / 2; ++i)
            *voxels++ = 0.0f;
        for (uint32_t i = LeafT::voxelCount() / 2; i < LeafT::voxelCount(); ++i) {
            data.mValueMask.setOn(i);
            *voxels++ = 1.234f;
        }
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
    }

    // lower internal node
    const NodeT1* node1 = reinterpret_cast<NodeT1*>(buffer.get() + bytes[3]);
    { // set members of the  internal node
        auto& data = *reinterpret_cast<NodeT1::DataType*>(buffer.get() + bytes[3]);
        auto* tiles = data.mTable;
        data.mValueMask.setOff();
        data.mChildMask.setOff();
        data.mChildMask.setOn(0);
        tiles->childID = 0; // the leaf node resides right after this node
        for (uint32_t i = 1; i < NodeT1::SIZE / 2; ++i, ++tiles)
            tiles->value = 0.0f;
        for (uint32_t i = NodeT1::SIZE / 2; i < NodeT1::SIZE; ++i, ++tiles) {
            data.mValueMask.setOn(i);
            tiles->value = 1.234f;
        }
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
        data.mOffset = 1;
        EXPECT_EQ(leaf, data.child(0));
    }

    // upper internal node
    const NodeT2* node2 = reinterpret_cast<NodeT2*>(buffer.get() + bytes[2]);
    { // set members of the  internal node
        auto& data = *reinterpret_cast<NodeT2::DataType*>(buffer.get() + bytes[2]);
        auto* tiles = data.mTable;
        data.mValueMask.setOff();
        data.mChildMask.setOff();
        data.mChildMask.setOn(0);
        tiles->childID = 0; // the leaf node resides right after this node
        for (uint32_t i = 1; i < NodeT2::SIZE / 2; ++i, ++tiles)
            tiles->value = 0.0f;
        for (uint32_t i = NodeT2::SIZE / 2; i < NodeT2::SIZE; ++i, ++tiles) {
            data.mValueMask.setOn(i);
            tiles->value = 1.234f;
        }
        data.mValueMin = 0.0f;
        data.mValueMax = 1.234f;
        data.mOffset = 1;
        EXPECT_EQ(node1, data.child(0));
    }

    // init root
    RootT* root = reinterpret_cast<RootT*>(buffer.get() + bytes[1]);
    { // set members of the root node
        auto& data = *reinterpret_cast<RootT::DataType*>(buffer.get() + bytes[1]);
        data.mBackground = data.mValueMin = data.mValueMax = 1.234f;
        data.mTileCount = 1;
        auto& tile = data.tile(0);
        tile.setChild(RootT::CoordType(0), 0);
    }

    // init tree
    TreeT* tree = reinterpret_cast<TreeT*>(buffer.get() + bytes[0]);
    {
        auto& data = *reinterpret_cast<TreeT::DataType*>(buffer.get() + bytes[0]);
        data.mCount[0] = data.mCount[1] = data.mCount[2] = data.mCount[3] = 1;
        data.mBytes[0] = bytes[4] - bytes[0];
        data.mBytes[1] = bytes[3] - bytes[0];
        data.mBytes[2] = bytes[2] - bytes[0];
        data.mBytes[3] = bytes[1] - bytes[0];
    }

    GridT* grid = reinterpret_cast<GridT*>(buffer.get());
    { // init Grid
        auto* data = reinterpret_cast<GridT::DataType*>(buffer.get());
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
            data->mBlindDataCount = 0;
            data->mUniformScale = dx;
            data->mMap.set(mat, invMat, 1.0);
            data->mGridClass = nanovdb::GridClass::Unknown;
            data->mGridType = nanovdb::GridType::Float;
            const std::string name("");
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
            data->mUniformScale = dx;
            data->mMap.set(mat, invMat, 1.0);
        }

        auto const p4 = grid->worldToIndex(p3);
        EXPECT_EQ(nanovdb::Vec3R(0.0, 0.0, 0.0), p4);
        const auto p5 = grid->indexToWorld(p4);
        EXPECT_EQ(p1, p5);

        EXPECT_EQ(nanovdb::GridClass::Unknown, grid->gridClass());
        EXPECT_EQ(nanovdb::GridType::Float, grid->gridType());
        //std::cerr << "\nName = \"" << grid->getName() << "\"" << std::endl;
        EXPECT_EQ("", std::string(grid->gridName()));
    }

    { // check leaf node
        auto* ptr = reinterpret_cast<LeafT::DataType*>(buffer.get() + bytes[4])->mValues;
        for (uint32_t i = 0; i < LeafT::voxelCount(); ++i) {
            if (i < 256) {
                EXPECT_FALSE(leaf->valueMask().isOn(i));
                EXPECT_EQ(0.0f, *ptr++);
            } else {
                EXPECT_TRUE(leaf->valueMask().isOn(i));
                EXPECT_EQ(1.234f, *ptr++);
            }
        }
        EXPECT_EQ(0.0f, leaf->valueMin());
        EXPECT_EQ(1.234f, leaf->valueMax());
    }

    { // check lower internal node
        auto* ptr = reinterpret_cast<NodeT1::DataType*>(buffer.get() + bytes[3])->mTable;
        EXPECT_TRUE(node1->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT1::SIZE; ++i, ++ptr) {
            EXPECT_FALSE(node1->childMask().isOn(i));
            if (i < NodeT1::SIZE / 2) {
                EXPECT_FALSE(node1->valueMask().isOn(i));
                EXPECT_EQ(0.0f, ptr->value);
            } else {
                EXPECT_TRUE(node1->valueMask().isOn(i));
                EXPECT_EQ(1.234f, ptr->value);
            }
        }
        EXPECT_EQ(0.0f, node1->valueMin());
        EXPECT_EQ(1.234f, node1->valueMax());
    }
    { // check upper internal node
        auto* ptr = reinterpret_cast<NodeT2::DataType*>(buffer.get() + bytes[2])->mTable;
        EXPECT_TRUE(node2->childMask().isOn(0));
        for (uint32_t i = 1; i < NodeT2::SIZE; ++i, ++ptr) {
            EXPECT_FALSE(node2->childMask().isOn(i));
            if (i < NodeT2::SIZE / 2) {
                EXPECT_FALSE(node2->valueMask().isOn(i));
                EXPECT_EQ(0.0f, ptr->value);
            } else {
                EXPECT_TRUE(node2->valueMask().isOn(i));
                EXPECT_EQ(1.234f, ptr->value);
            }
        }
        EXPECT_EQ(0.0f, node2->valueMin());
        EXPECT_EQ(1.234f, node2->valueMax());
    }
    { // check root
        EXPECT_EQ(1.234f, root->background());
        EXPECT_EQ(1.234f, root->valueMin());
        EXPECT_EQ(1.234f, root->valueMax());
        EXPECT_EQ(1u, root->tileCount());
        EXPECT_EQ(0.0f, root->getValue(CoordT(0, 0, 0)));
        EXPECT_EQ(1.234f, root->getValue(CoordT(7, 7, 7)));
    }
    { // check tree
        EXPECT_EQ(1.234f, tree->background());
        float a, b;
        tree->extrema(a, b);
        EXPECT_EQ(1.234f, a);
        EXPECT_EQ(1.234f, b);
        EXPECT_EQ(0.0f, tree->getValue(CoordT(0, 0, 0)));
        EXPECT_EQ(1.234f, tree->getValue(CoordT(7, 7, 7)));
        EXPECT_EQ(1u, tree->nodeCount<LeafT>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT1>());
        EXPECT_EQ(1u, tree->nodeCount<NodeT2>());
        EXPECT_EQ(1u, tree->nodeCount<RootT>());
        EXPECT_EQ(reinterpret_cast<LeafT*>(buffer.get() + bytes[4]), tree->getNode<LeafT>(0));
        EXPECT_EQ(reinterpret_cast<NodeT1*>(buffer.get() + bytes[3]), tree->getNode<NodeT1>(0));
        EXPECT_EQ(reinterpret_cast<NodeT2*>(buffer.get() + bytes[2]), tree->getNode<NodeT2>(0));
    }

} // Grid

TEST_F(TestNanoVDB, GridBuilderBasic0)
{
    { // empty grid
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        auto                        handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test", nanovdb::GridClass::LevelSet);
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_EQ("test", std::string(meta->gridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ(0u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_EQ(0.0f, srcAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_FALSE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(0.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
    }
} // GridBuilderBasic0

TEST_F(TestNanoVDB, GridBuilderBasic1)
{
    { // 1 grid point
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(1, 2, 3), 1.0f);
        auto handle = builder.getHandle<>();
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_EQ("", std::string(meta->gridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::Unknown, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ(1.0 , dstGrid->voxelSize());
        EXPECT_EQ(1u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_TRUE(srcAcc.isActive(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(nanovdb::Coord(1, 2, 3), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord(1, 2, 3), dstGrid->indexBBox()[1]);
    }
} // GridBuilderBasic1

TEST_F(TestNanoVDB, GridBuilderBasic2)
{
    { // 2 grid points
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        srcAcc.setValue(nanovdb::Coord(1, 2, 3), 1.0f);
        srcAcc.setValue(nanovdb::Coord(2, -2, 9), 2.0f);
        //srcAcc.setValue(nanovdb::Coord(20,-20,90), 0.0f);// same as background
        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test", nanovdb::GridClass::LevelSet);
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_EQ("test", std::string(meta->gridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
        EXPECT_EQ(2u, dstGrid->activeVoxelCount());
        auto dstAcc = dstGrid->getAccessor();
        EXPECT_EQ(1.0f, dstAcc.getValue(nanovdb::Coord(1, 2, 3)));
        EXPECT_EQ(2.0f, dstAcc.getValue(nanovdb::Coord(2, -2, 9)));
        EXPECT_EQ(nanovdb::Coord(1, -2, 3), dstGrid->indexBBox()[0]);
        EXPECT_EQ(nanovdb::Coord(2, 2, 9), dstGrid->indexBBox()[1]);
    }
} // GridBuilderBasic2

TEST_F(TestNanoVDB, GridBuilderBasicDense)
{
    { // dense functor
        nanovdb::GridBuilder<float> builder(0.0f);
        auto                        srcAcc = builder.getAccessor();
        const nanovdb::CoordBBox    bbox(nanovdb::Coord(0), nanovdb::Coord(100));
        auto                        func = [](const nanovdb::Coord&) { return 1.0f; };
        //auto                        func = [](const nanovdb::Coord&, float &v) { v = 1.0f; return true; };
        builder(func, bbox);
        for (auto ijk = bbox.begin(); ijk; ++ijk) {
            EXPECT_EQ(1.0f, srcAcc.getValue(*ijk));
            EXPECT_TRUE(srcAcc.isActive(*ijk));
        }
        auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test", nanovdb::GridClass::LevelSet);
        EXPECT_TRUE(handle);
        auto* meta = handle.gridMetaData();
        EXPECT_TRUE(meta);
        EXPECT_EQ("test", std::string(meta->gridName()));
        EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
        EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
        auto* dstGrid = handle.grid<float>();
        EXPECT_TRUE(dstGrid);
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
        EXPECT_EQ(0.5, grid->tree().getValue(nanovdb::Coord(0)));
        EXPECT_EQ(1, grid->tree().getValue(nanovdb::Coord(1)));
        EXPECT_EQ(0, grid->tree().getValue(nanovdb::Coord(2)));
    }
} // GridBuilderDense

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
    ValueT sdf(const nanovdb::Coord& ijk) const
    {
        nanovdb::Vec3<ValueT> xyz(ijk[0], ijk[1], ijk[2]);
        xyz *= mVoxelSize;
        xyz -= mCenter;
        return xyz.length() - mRadius;
    }
    static_assert(nanovdb::is_floating_point<float>::value, "Sphere: expect floating point");
    const nanovdb::Vec3<ValueT> mCenter;
    const ValueT                mRadius, mVoxelSize, mBackground;
}; // Sphere
} // namespace

TEST_F(TestNanoVDB, GridBuilderSphere)
{
    Sphere<float> sphere(nanovdb::Vec3<float>(50), 20.0f);
    EXPECT_EQ(3.0f, sphere.background());
    EXPECT_EQ(3.0f, sphere(nanovdb::Coord(100)));
    EXPECT_EQ(-3.0f, sphere(nanovdb::Coord(50)));
    EXPECT_EQ(0.0f, sphere(nanovdb::Coord(50, 50, 70)));
    EXPECT_EQ(-1.0f, sphere(nanovdb::Coord(50, 50, 69)));
    EXPECT_EQ(2.0f, sphere(nanovdb::Coord(50, 50, 72)));

    nanovdb::GridBuilder<float> builder(sphere.background());
    auto                        srcAcc = builder.getAccessor();

    const nanovdb::CoordBBox bbox(nanovdb::Coord(-100), nanovdb::Coord(100));
    //mTimer.start("GridBulder Sphere");
    builder(sphere, bbox);
    //mTimer.stop();

    auto handle = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test", nanovdb::GridClass::LevelSet);
    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("test", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);

    //std::cerr << "bbox.min = (" << dstGrid->indexBBox()[0][0] << ", " <<  dstGrid->indexBBox()[0][1] << ", " <<  dstGrid->indexBBox()[0][2] << ")" << std::endl;
    //std::cerr << "bbox.max = (" << dstGrid->indexBBox()[1][0] << ", " <<  dstGrid->indexBBox()[1][1] << ", " <<  dstGrid->indexBBox()[1][2] << ")" << std::endl;

    uint64_t count = 0;
    auto     dstAcc = dstGrid->getAccessor();
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

    auto handle = nanovdb::createLevelSetSphere<float>(20.0f, nanovdb::Vec3d(50), 1.0f, 3.0f, nanovdb::Vec3d(0), "sphere_20");

    const nanovdb::CoordBBox bbox(nanovdb::Coord(0), nanovdb::Coord(100));

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("sphere_20", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);

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
    auto                     handle = nanovdb::createFogVolumeSphere(20.0f, nanovdb::Vec3d(50), 1.0f, 3.0f, nanovdb::Vec3d(0), "sphere_20");
    const nanovdb::CoordBBox bbox(nanovdb::Coord(0), nanovdb::Coord(100));

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("sphere_20", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::FogVolume, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);

    EXPECT_EQ(nanovdb::Coord(50 - 20), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(50 + 20), dstGrid->indexBBox()[1]);

    //std::cerr << "bbox.min = (" << dstGrid->indexBBox()[0][0] << ", " <<  dstGrid->indexBBox()[0][1] << ", " <<  dstGrid->indexBBox()[0][2] << ")" << std::endl;
    //std::cerr << "bbox.max = (" << dstGrid->indexBBox()[1][0] << ", " <<  dstGrid->indexBBox()[1][1] << ", " <<  dstGrid->indexBBox()[1][2] << ")" << std::endl;
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

    auto handle = nanovdb::createPointSphere<float>(1, 100.0f, nanovdb::Vec3d(0), 1.0f, nanovdb::Vec3d(0), "point_sphere");

    const nanovdb::CoordBBox bbox(nanovdb::Coord(-100), nanovdb::Coord(100));

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("point_sphere", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::UInt32, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::PointData, meta->gridClass());
    auto* dstGrid = handle.grid<uint32_t>();
    EXPECT_TRUE(dstGrid);

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
                    const auto  p = begin->round();
                    const float diff = nanovdb::Vec3f(ijk - p).lengthSqr();
                    EXPECT_TRUE(diff <= 1.0f); // allow for floating point rounding error by one voxel
                    //if (diff > 1.0f) std::cerr << "p = ("<<p[0]<<", "<<p[1]<<", "<<p[2]<<")"<<"ijk = ("<<ijk[0]<<", "<<ijk[1]<<", "<<ijk[2]<<")"<<std::endl;
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
    auto handle = nanovdb::createLevelSetTorus<float>(100.0f, 50.0f, nanovdb::Vec3d(50), 1.0f, 3.0f, nanovdb::Vec3d(0), "torus_100");

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("torus_100", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);

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
    auto handle = nanovdb::createFogVolumeTorus<float>(100.0f, 50.0f, nanovdb::Vec3d(50), 1.0f, 3.0f, nanovdb::Vec3d(0), "torus_100");

    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("torus_100", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::FogVolume, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);

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
    auto handle = nanovdb::createLevelSetBox<float>(40.0f, 60.0f, 80.0f, nanovdb::Vec3d(50), 1.0f, 3.0f, nanovdb::Vec3d(0), "box");
    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("box", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);

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
    auto handle = nanovdb::createFogVolumeBox<float>(40.0f, 60.0f, 80.0f, nanovdb::Vec3d(50), 1.0f, 3.0f, nanovdb::Vec3d(0), "box");
    EXPECT_TRUE(handle);
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ("box", std::string(meta->gridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::FogVolume, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);

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

TEST_F(TestNanoVDB, GridStats)
{
    Sphere<float> sphere(nanovdb::Vec3<float>(50), 50.0f);
    nanovdb::GridBuilder<float> builder(sphere.background());
    const nanovdb::CoordBBox bbox(nanovdb::Coord(-100), nanovdb::Coord(100));
    //mTimer.start("GridBuilder");
    builder(sphere, bbox);
    //mTimer.stop();
    auto handle1 = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test", nanovdb::GridClass::LevelSet);
    auto handle2 = builder.getHandle<>(1.0, nanovdb::Vec3d(0.0), "test", nanovdb::GridClass::LevelSet);
    EXPECT_TRUE(handle1);
    EXPECT_TRUE(handle2);
    auto* grid1 = handle1.grid<float>();
    auto* grid2 = handle2.grid<float>();
    EXPECT_TRUE(grid1);
    EXPECT_TRUE(grid2);

    //std::cerr << "grid1 = " << grid1->indexBBox() << ", grid2 = " << grid2->indexBBox() << std::endl;
    EXPECT_EQ( grid1->activeVoxelCount(), grid2->activeVoxelCount() );
    EXPECT_EQ( grid1->worldBBox(), grid2->worldBBox() );
    EXPECT_EQ( grid1->worldBBox(), grid2->worldBBox() );

    {// reset stats in grid2
        const_cast<size_t&>(grid2->activeVoxelCount()) = size_t(0);
        const_cast<nanovdb::BBox<nanovdb::Vec3R>&>(grid2->worldBBox()) = nanovdb::BBox<nanovdb::Vec3R>();
        const_cast<nanovdb::BBox<nanovdb::Coord>&>(grid2->indexBBox()) = nanovdb::BBox<nanovdb::Coord>();
        for (uint32_t i=0; i< grid2->tree().nodeCount(0); ++i) {
            auto *leaf = grid2->tree().getNode<0>(i);
            auto *data = const_cast<nanovdb::NanoLeaf<float>*>(leaf)->data();
            data->mValueMin = data->mValueMax = 0.0f;
            data->mBBoxMin &= ~nanovdb::NanoLeaf<float>::MASK;/// set to origin!
            data->mBBoxDif[0] = data->mBBoxDif[1] = data->mBBoxDif[2] = uint8_t(255);
            EXPECT_EQ(data->mBBoxDif[0], uint8_t(255));
        }
        for (uint32_t i=0; i< grid2->tree().nodeCount(1); ++i) {
             auto *node = grid2->tree().getNode<1>(i);
             auto *data = const_cast<nanovdb::NanoNode1<float>*>(node)->data();
             data->mValueMin = data->mValueMax = 0.0f;
             data->mBBox[0] &= ~nanovdb::NanoNode1<float>::MASK;/// set to origin!
             data->mBBox[1] = nanovdb::Coord(0);
        }
        for (uint32_t i=0; i< grid2->tree().nodeCount(2); ++i) {
             auto *node = grid2->tree().getNode<2>(i);
             auto *data = const_cast<nanovdb::NanoNode2<float>*>(node)->data();
             data->mValueMin = data->mValueMax = 0.0f;
             data->mBBox[0] &= ~nanovdb::NanoNode2<float>::MASK;/// set to origin!
             data->mBBox[1] = nanovdb::Coord(0);
        }
    }

    //std::cerr << "grid1 = " << grid1->indexBBox() << ", grid2 = " << grid2->indexBBox() << std::endl;
    EXPECT_NE( grid1->activeVoxelCount(), grid2->activeVoxelCount() );
    EXPECT_NE( grid1->indexBBox(), grid2->indexBBox() );
    EXPECT_NE( grid1->worldBBox(), grid2->worldBBox() );

    {// check stats in grid2
        EXPECT_EQ( grid1->tree().nodeCount(0), grid2->tree().nodeCount(0));
        for (uint32_t i=0; i< grid2->tree().nodeCount(0); ++i) {
            auto *leaf1 = grid1->tree().getNode<0>(i);
            auto *leaf2 = grid2->tree().getNode<0>(i);
            EXPECT_NE(leaf1->valueMin(), leaf2->valueMin());
            EXPECT_NE(leaf1->valueMax(), leaf2->valueMax());
            EXPECT_NE(leaf1->bbox(), leaf2->bbox());
        }
        EXPECT_EQ( grid1->tree().nodeCount(1), grid2->tree().nodeCount(1));
        for (uint32_t i=0; i< grid2->tree().nodeCount(1); ++i) {
            auto *node1 = grid1->tree().getNode<1>(i);
            auto *node2 = grid2->tree().getNode<1>(i);
            EXPECT_NE(node1->valueMin(), node2->valueMin());
            EXPECT_NE(node1->valueMax(), node2->valueMax());
            EXPECT_NE(node1->bbox(), node2->bbox());
        }
        EXPECT_EQ( grid1->tree().nodeCount(2), grid2->tree().nodeCount(2));
        for (uint32_t i=0; i< grid2->tree().nodeCount(2); ++i) {
            auto *node1 = grid1->tree().getNode<2>(i);
            auto *node2 = grid2->tree().getNode<2>(i);
            EXPECT_NE(node1->valueMin(), node2->valueMin());
            EXPECT_NE(node1->valueMax(), node2->valueMax());
            EXPECT_NE(node1->bbox(), node2->bbox());
        }
    }
    
    //mTimer.start("GridStats");
    nanovdb::gridStats( *grid2 );
    //mTimer.stop();

    {// check stats in grid2
        EXPECT_EQ( grid1->tree().nodeCount(0), grid2->tree().nodeCount(0));
        for (uint32_t i=0; i< grid2->tree().nodeCount(0); ++i) {
            auto *leaf1 = grid1->tree().getNode<0>(i);
            auto *leaf2 = grid2->tree().getNode<0>(i);
            EXPECT_EQ(leaf1->valueMin(), leaf2->valueMin());
            EXPECT_EQ(leaf1->valueMax(), leaf2->valueMax());
            EXPECT_EQ(leaf1->bbox(), leaf2->bbox());
        }
        EXPECT_EQ( grid1->tree().nodeCount(1), grid2->tree().nodeCount(1));
        for (uint32_t i=0; i< grid2->tree().nodeCount(1); ++i) {
            auto *node1 = grid1->tree().getNode<1>(i);
            auto *node2 = grid2->tree().getNode<1>(i);
            EXPECT_EQ(node1->valueMin(), node2->valueMin());
            EXPECT_EQ(node1->valueMax(), node2->valueMax());
            EXPECT_EQ(node1->bbox(), node2->bbox());
        }
        EXPECT_EQ( grid1->tree().nodeCount(2), grid2->tree().nodeCount(2));
        for (uint32_t i=0; i< grid2->tree().nodeCount(2); ++i) {
            auto *node1 = grid1->tree().getNode<2>(i);
            auto *node2 = grid2->tree().getNode<2>(i);
            EXPECT_EQ(node1->valueMin(), node2->valueMin());
            EXPECT_EQ(node1->valueMax(), node2->valueMax());
            EXPECT_EQ(node1->bbox(), node2->bbox());
        }
    }

    //std::cerr << "grid1 = " << grid1->indexBBox() << ", grid2 = " << grid2->indexBBox() << std::endl; 
    EXPECT_EQ( grid1->activeVoxelCount(), grid2->activeVoxelCount() );
    EXPECT_EQ( grid1->indexBBox(), grid2->indexBBox() );
    EXPECT_EQ( grid1->worldBBox(), grid2->worldBBox() );
}

TEST_F(TestNanoVDB, ScalarSampleFromVoxels)
{
    // create a grid so sample from
    const float dx = 0.5f;// voxel size
    auto trilinearWorld = [&](const nanovdb::Vec3d& xyz) -> float {
        return 0.34f + 1.6f * xyz[0] + 6.7f * xyz[1] - 3.5f * xyz[2]; // index coordinates
    };
    auto trilinearIndex = [&](const nanovdb::Coord& ijk) -> float {
        return 0.34f + 1.6f * dx* ijk[0] + 6.7f * dx * ijk[1] - 3.5f * dx * ijk[2]; // index coordinates
    };
    
    nanovdb::GridBuilder<float> builder(1.0f);
    const nanovdb::CoordBBox bbox(nanovdb::Coord(0), nanovdb::Coord(128));
    builder(trilinearIndex, bbox);
    auto handle = builder.getHandle<>(dx);
    EXPECT_TRUE(handle);
    auto* grid = handle.grid<float>();
    EXPECT_TRUE(grid);

    const nanovdb::Vec3d ijk(13.4, 24.67, 5.23); // in index space
    const auto exact = trilinearWorld(grid->indexToWorld(ijk));
    const auto approx = trilinearIndex(nanovdb::Coord(13, 25, 5));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;

    auto acc = grid->getAccessor();
    auto sampler0 = nanovdb::createSampler<0>(acc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_EQ(approx, sampler0(ijk));

    auto sampler1 = nanovdb::createSampler<1>(acc); // faster since it's using an accessor!!!
    //std::cerr << "1'th order: v = " << sampler1(ijk) << std::endl;
    EXPECT_NEAR(exact, sampler1(ijk), 1e-5);

    EXPECT_FALSE(sampler1.zeroCrossing());
    const auto gradIndex = sampler1.gradient(ijk); //in index space
    EXPECT_NEAR( 1.6f, gradIndex[0] / dx, 2e-5);
    EXPECT_NEAR( 6.7f, gradIndex[1] / dx, 2e-5);
    EXPECT_NEAR(-3.5f, gradIndex[2] / dx, 2e-5);
    const auto gradWorld = grid->indexToWorldDir(gradIndex); // in world units
    EXPECT_NEAR( 1.6f, gradWorld[0], 2e-5);
    EXPECT_NEAR( 6.7f, gradWorld[1], 2e-5);
    EXPECT_NEAR(-3.5f, gradWorld[2], 2e-5);

    nanovdb::SampleFromVoxels<nanovdb::NanoTree<float>, 3> sampler3(grid->tree());
    //auto sampler3 = nanovdb::createSampler<3>( dstAcc );
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    EXPECT_NEAR(exact, sampler3(ijk), 1e-5);
    
} // ScalarSampleFromVoxels

TEST_F(TestNanoVDB, VectorSampleFromVoxels)
{
    // create a grid so sample from
    const float dx = 0.5f;// voxel size
    auto trilinearWorld = [&](const nanovdb::Vec3d& xyz) -> nanovdb::Vec3f {
        return nanovdb::Vec3f(0.34f, 1.6f * xyz[0] + 6.7f * xyz[1], - 3.5f * xyz[2]); // index coordinates
    };
    auto trilinearIndex = [&](const nanovdb::Coord& ijk) -> nanovdb::Vec3f {
        return nanovdb::Vec3f(0.34f, 1.6f * dx* ijk[0] + 6.7f * dx * ijk[1], - 3.5f * dx * ijk[2]); // index coordinates
    };
    
    nanovdb::GridBuilder<nanovdb::Vec3f> builder(nanovdb::Vec3f(1.0f));
    const nanovdb::CoordBBox bbox(nanovdb::Coord(0), nanovdb::Coord(128));
    builder(trilinearIndex, bbox);
    auto handle = builder.getHandle<>(dx);
    EXPECT_TRUE(handle);
    auto* grid = handle.grid<nanovdb::Vec3f>();
    EXPECT_TRUE(grid);

    const nanovdb::Vec3d ijk(13.4, 24.67, 5.23); // in index space
    const auto exact = trilinearWorld(grid->indexToWorld(ijk));
    const auto approx = trilinearIndex(nanovdb::Coord(13, 25, 5));
    //std::cerr << "Trilinear: exact = " << exact << ", approx = " << approx << std::endl;

    auto acc = grid->getAccessor();
    auto sampler0 = nanovdb::createSampler<0>(acc);
    //std::cerr << "0'th order: v = " << sampler0(ijk) << std::endl;
    EXPECT_EQ(approx, sampler0(ijk));

    auto sampler1 = nanovdb::createSampler<1>(acc); // faster since it's using an accessor!!!
    //std::cerr << "1'th order: v = " << sampler1(ijk) << std::endl;
    for (int i=0; i<3; ++i) EXPECT_NEAR(exact[i], sampler1(ijk)[i], 1e-5);
    //EXPECT_FALSE(sampler1.zeroCrossing());// triggeres a static_assert error
    //EXPECT_FALSE(sampler1.gradient(grid->indexToWorld(ijk)));// triggeres a static_assert error

    nanovdb::SampleFromVoxels<nanovdb::NanoTree<nanovdb::Vec3f>, 3> sampler3(grid->tree());
    //auto sampler3 = nanovdb::createSampler<3>( acc );
    //std::cerr << "3'rd order: v = " << sampler3(ijk) << std::endl;
    for (int i=0; i<3; ++i) EXPECT_NEAR(exact[i], sampler3(ijk)[i], 1e-5);
    
} // VectorSampleFromVoxels

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}