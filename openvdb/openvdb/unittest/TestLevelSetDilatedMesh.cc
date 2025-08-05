// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>

#include <openvdb/tools/Composite.h>
#include <openvdb/tools/Count.h>
#include <openvdb/tools/LevelSetDilatedMesh.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetTubes.h>

#include <gtest/gtest.h>

class TestLevelSetDilatedMesh: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};

namespace {

template<typename GridT>
void
testDilatedConvexPolygonMeasures(const std::vector<openvdb::Vec3s>& points,
                                 const float& r, typename GridT::Ptr& grid, float error = 0.01f)
{
    using namespace openvdb;

    const size_t n = points.size();
    const float pi = math::pi<float>(), r2 = math::Pow2(r), r3 = math::Pow3(r);

    float l = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const Vec3s &p = points[i], &q = points[(i+1) % n];
        l += (p - q).length();
    }

    const Vec3s &p = points[0];
    float a = 0.0f;
    for (size_t i = 1; i < n-1; ++i) {
        const Vec3s &q = points[i], &r = points[i+1];
        a += 0.5f * (q - p).cross(r - p).length();
    }

    const float area         = pi*r*l + 4.0f*pi*r2 + 2.0f*a,
                volume       = 0.5f*pi*r2*l + 4.0f/3.0f*pi*r3 + 2.0f*a*r,
                totGaussCurv = 4.0f*pi, // Gauss-Bonnet
                totMeanCurv  = 0.5f*pi*l + 4.0f*pi*r;

    tools::LevelSetMeasure<GridT> m(*grid);

    EXPECT_NEAR(m.area(true),                 area,         area*error);
    EXPECT_NEAR(m.volume(true),               volume,       volume*error);
    EXPECT_NEAR(m.totGaussianCurvature(true), totGaussCurv, totGaussCurv*error);
    EXPECT_NEAR(m.totMeanCurvature(true),     totMeanCurv,  totMeanCurv*error);
}

}

TEST_F(TestLevelSetDilatedMesh, testGeneric)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // triangle mesh
    {
        const float r = 2.9f;
        const Vec3s p0(15.8f, 13.2f, 16.7f), p1(4.3f, 7.9f, -4.8f), p2(-3.0f, -7.4f, 8.9f),
                    p3(-2.7f, 8.9f, 30.4f), p4(23.0f, 17.4f, -10.9f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3, p4});
        const std::vector<Vec3I> tris({Vec3I(0, 1, 2), Vec3I(0, 1, 3), Vec3I(0, 1, 4)});

        const float voxelSize = 0.1f, width = 3.25f;
        const Coord ijk(int(p1[0]/voxelSize),
                        int(p1[1]/voxelSize),
                        int(p1[2]/voxelSize));// inside

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tris, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(voxelSize*width, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(30, 0, -50)), 1e-6);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // quad mesh
    {
        const float r = 2.9f;
        const Vec3s p0(15.8f, 13.2f, 16.7f), p1(4.3f, 7.9f, -4.8f),    p2(-3.0f, -7.4f, 8.9f),
                    p3(-2.7f, 8.9f, 30.4f),  p4(23.0f, 17.4f, -10.9f), p5(5.2f, -5.7f, 29.0f),
                    p6(-14.6f, 3.7f, 10.9f), p7(35.8f, 23.4f, 5.8f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3, p4, p5, p6, p7});
        const std::vector<Vec4I> quads({Vec4I(0, 1, 2, 5), Vec4I(0, 1, 6, 3), Vec4I(0, 1, 4, 7)});

        const float voxelSize = 0.1f, width = 3.25f;
        const Coord ijk(int(p1[0]/voxelSize),
                        int(p1[1]/voxelSize),
                        int(p1[2]/voxelSize));// inside

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, quads, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(voxelSize*width, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(30, 0, -50)), 1e-6);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // mixed triangle and quad mesh
    {
        const float r = 2.9f;
        const Vec3s p0(15.8f, 13.2f, 16.7f), p1(4.3f, 7.9f, -4.8f), p2(-3.0f, -7.4f, 8.9f),
                    p3(35.8f, 23.4f, 5.8f), p4(23.0f, 17.4f, -10.9f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3, p4});
        const std::vector<Vec3I> tris({Vec3I(0, 1, 2)});
        const std::vector<Vec4I> quads({Vec4I(0, 1, 4, 3)});

        const float voxelSize = 0.1f, width = 3.25f;
        const Coord ijk(int(p1[0]/voxelSize),
                        int(p1[1]/voxelSize),
                        int(p1[2]/voxelSize));// inside

        FloatGrid::Ptr ls = tools::createLevelSetDilatedMesh<FloatGrid>(
            vertices, tris, quads, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(voxelSize*width, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(30, 0, -50)), 1e-6);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // test closed surface mesh has a void (non-zero Betti number b2)
    {
        const float r = 0.3f, voxelSize = 0.05f, width = 2.0f;

        const std::vector<Vec3s> vertices({
            Vec3s(-0.5f, 0.5f, -0.5f),  Vec3s(0.5f, 0.5f, -0.5f), Vec3s(0.5f, -0.5f, -0.5f),
            Vec3s(-0.5f, -0.5f, -0.5f), Vec3s(-0.5f, 0.5f, 0.5f), Vec3s(0.5f, 0.5f, 0.5f),
            Vec3s(0.5f, -0.5f, 0.5f), Vec3s(-0.5f, -0.5f, 0.5f)
        });

        const std::vector<Vec4I> quads({Vec4I(0, 1, 2, 3), Vec4I(4, 5, 1, 0), Vec4I(5, 6, 2, 1),
                                        Vec4I(6, 7, 3, 2), Vec4I(7, 4, 0, 3), Vec4I(7, 6, 5, 4)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, quads, r, voxelSize, width);

        EXPECT_NEAR(ls->background(), ls->tree().getValue(Coord(0, 0, 0)), 1e-6);
    }

}// testGeneric

TEST_F(TestLevelSetDilatedMesh, testMeasures)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // test measures of a dilated triangle
    {
        const float r = 1.1f, voxelSize = 0.05f;
        const Vec3s p0(9.4f, 7.6f, -0.9f), p1(-1.4f, -3.5f, -1.4f), p2(-8.5f, 9.7f, -5.6f);

        const std::vector<Vec3s> vertices({p0, p1, p2});
        const std::vector<Vec3I> tri({Vec3I(0, 1, 2)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tri, r, voxelSize);

        testDilatedConvexPolygonMeasures<GridT>(vertices, r, ls);
    }

    // test measures of a dilated quad
    {
        const float r = 1.3f, voxelSize = 0.05f;
        const Vec3s p0(9.1f, 5.1f, -0.5f), p1(-1.4f, -3.5f, -1.4f),
                    p2(-8.5f, 9.7f, -5.6f), p3(9.4f, 7.6f, -0.9f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3});
        const std::vector<Vec4I> quad({Vec4I(0, 1, 2, 3)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, quad, r, voxelSize);

        testDilatedConvexPolygonMeasures<GridT>(vertices, r, ls);
    }

    // test measures of a dilated convex polygon from a triangle mesh
    {
        const float r = 0.08f, voxelSize = 0.025f;
        const Vec3s p0(0.0f, 0.0f, 0.0f), p1(1.0f, 0.0f, 0.0f), p2(1.0f, 1.1f, 0.0f),
                    p3(0.0f, 1.0f, 0.0f), p4(-0.5f, 0.5f, 0.0f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3, p4});
        const std::vector<Vec3I> tris({Vec3I(0, 1, 2), Vec3I(0, 2, 3), Vec3I(0, 3, 4)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tris, r, voxelSize);

        testDilatedConvexPolygonMeasures<GridT>(vertices, r, ls, 0.02f);
    }

    // test measures of a rigatoni noodle
    {
        const float pi = math::pi<float>(), a = 0.25, c = 1.4f,
                    z1 = 0.0f, z2 = 8.0f, voxelSize = 0.05f;

        const Index n = 256;
        std::vector<Vec3s> vertices(2*n);
        std::vector<Vec4I> quads(n);
        const float delta = 2.0f*pi/static_cast<float>(n);

        float theta = 0.0f;
        for (Index32 i = 0; i < n; ++i, theta += delta) {
            const float x = c*math::Cos(theta), y = c*math::Sin(theta);
            vertices[i]   = Vec3s(x, y, z1);
            vertices[i+n] = Vec3s(x, y, z2);
            quads[i]      = Vec4I(i, (i+1)%n, n + ((i+1)%n), n + (i%n));
        }

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, quads, a, voxelSize);

        const float error = 0.02f, h = math::Abs(z2-z1);

        const float area         = 4.0f*pi*c * (h + pi*a),
                    volume       = 2.0f*a*c*pi * (2.0f*h + a*pi),
                    totGaussCurv = 0.0f, // Gauss-Bonnet
                    totMeanCurv  = 2.0f*c*math::Pow2(pi);

        tools::LevelSetMeasure<GridT> m(*ls);

        EXPECT_NEAR(m.area(true),                 area,         area*error);
        EXPECT_NEAR(m.volume(true),               volume,       volume*error);
        EXPECT_NEAR(m.totGaussianCurvature(true), totGaussCurv, 10.0f*error);
        EXPECT_NEAR(m.totMeanCurvature(true),     totMeanCurv,  totMeanCurv*error);
    }

}// testMeasures

TEST_F(TestLevelSetDilatedMesh, testDegeneracies)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // degenerate case: capsule
    {
        const float r = 1.4f, voxelSize = 0.1f;
        const Vec3s p0(0.0f, 0.0f, 0.0f), p1(1.0f, -0.9f, 2.0f), p2(0.0f, 0.0f, 0.0f);

        const std::vector<Vec3s> vertices({p0, p1, p2});
        const std::vector<Vec3I> tris({Vec3I(0, 1, 2)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tris, r, voxelSize);

        const float error = 0.01f, pi = math::pi<float>(), l = (p1-p0).length();

        const float area         = 2.0f*l*pi*r + 4.0f*pi*math::Pow2(r),
                    volume       = l*pi*math::Pow2(r) + 4.0f*pi*math::Pow3(r)/3.0f,
                    totGaussCurv = 4.0f*pi, // Gauss-Bonnet
                    totMeanCurv  = pi*l + 4.0f*pi*r;

        tools::LevelSetMeasure<GridT> m(*ls);

        EXPECT_NEAR(m.area(true),                 area,         area*error);
        EXPECT_NEAR(m.volume(true),               volume,       volume*error);
        EXPECT_NEAR(m.totGaussianCurvature(true), totGaussCurv, totGaussCurv*error);
        EXPECT_NEAR(m.totMeanCurvature(true),     totMeanCurv,  totMeanCurv*error);
    }

    // degenerate case: sphere
    {
        const float r = 1.7f, voxelSize = 0.1f;
        const Vec3s p0(0.0f, 0.0f, 0.0f), p1(0.0f, 0.0f, 0.0f),
                    p2(0.0f, 0.0f, 0.0f), p3(0.0f, 0.0f, 0.0f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3});
        const std::vector<Vec4I> quads({Vec4I(0, 1, 2, 3)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, quads, r, voxelSize);

        const float error = 0.01f, pi = math::pi<float>();

        const float area         = 4.0f*pi*math::Pow2(r),
                    volume       = 4.0f*pi*math::Pow3(r)/3.0f,
                    totGaussCurv = 4.0f*pi, // Gauss-Bonnet
                    totMeanCurv  = 4.0f*pi*r;

        tools::LevelSetMeasure<GridT> m(*ls);

        EXPECT_NEAR(m.area(true),                 area,         area*error);
        EXPECT_NEAR(m.volume(true),               volume,       volume*error);
        EXPECT_NEAR(m.totGaussianCurvature(true), totGaussCurv, totGaussCurv*error);
        EXPECT_NEAR(m.totMeanCurvature(true),     totMeanCurv,  totMeanCurv*error);
    }

    // degenerate case: polygon
    {
        const float r = 0.0f, voxelSize = 0.1f, width = 2.0f;
        const Vec3s p0(9.4f, 7.6f, -0.9f), p1(-1.4f, -3.5f, -1.4f), p2(-8.5f, 9.7f, -5.6f);

        const std::vector<Vec3s> vertices({p0, p1, p2});
        const std::vector<Vec3I> tri({Vec3I(0, 1, 2)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tri, r, voxelSize, width);

        EXPECT_TRUE(tools::minMax(ls->tree()).min() >= 0.0f); // no zero crossings
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // degenerate case: line
    {
        const float r = 0.0f, voxelSize = 0.1f, width = 2.0f;
        const Vec3s p0(0.0f, 0.0f, 0.0f),  p1(0.5f, 0.0f, 0.0f),
                    p2(0.75f, 0.0f, 0.0f), p3(1.0f, 0.0f, 0.0f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3});
        const std::vector<Vec4I> quad({Vec4I(0, 1, 2, 3)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, quad, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 117 /* 90 + 27 */);
        EXPECT_TRUE(tools::minMax(ls->tree()).min() >= 0.0f); // no zero crossings
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // degenerate case: point
    {
        const float r = 0.0f, voxelSize = 0.1f, width = 2.0f;
        const Vec3s p0(0.0f, 0.0f, 0.0f), p1(0.0f, 0.0f, 0.0f), p2(0.0f, 0.0f, 0.0f);

        const std::vector<Vec3s> vertices({p0, p1, p2});
        const std::vector<Vec3I> tri({Vec3I(0, 1, 2)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tri, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 27 /* 3x3x3 grid */);
        EXPECT_TRUE(tools::minMax(ls->tree()).min() >= 0.0f); // no zero crossings
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // degenerate case: negative radius --> empty grid
    {
        const float r = -10.0f, voxelSize = 0.1f, width = 2.0f;
        const Vec3s p0(9.4f, 7.6f, -0.9f), p1(-1.4f, -3.5f, -1.4f), p2(-8.5f, 9.7f, -5.6f);

        const std::vector<Vec3s> vertices({p0, p1, p2});
        const std::vector<Vec3I> tri({Vec3I(0, 1, 2)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tri, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

}// testDegeneracies

TEST_F(TestLevelSetDilatedMesh, testFaceTopologies)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // test measures of a dilated triangle
    {
        const float r = 1.1f, voxelSize = 0.05f;
        const Vec3s p0(9.4f, 7.6f, -0.9f), p1(-1.4f, -3.5f, -1.4f), p2(-8.5f, 9.7f, -5.6f);

        const std::vector<Vec3s> vertices({p0, p1, p2});
        const std::vector<Vec3I> tri1({Vec3I(0, 1, 2)});

        GridPtr ls1 = tools::createLevelSetDilatedMesh<GridT>(vertices, tri1, r, voxelSize);

        testDilatedConvexPolygonMeasures<GridT>(vertices, r, ls1);

        // change in face orientation doesn't effect result

        const std::vector<Vec3I> tri2({Vec3I(0, 2, 1)});

        GridPtr ls2 = tools::createLevelSetDilatedMesh<GridT>(vertices, tri2, r, voxelSize);

        testDilatedConvexPolygonMeasures<GridT>(vertices, r, ls2);

        // nor does multiple copies of the same face

        const std::vector<Vec3I> tri3({Vec3I(0, 1, 2), Vec3I(0, 1, 2),
                                       Vec3I(0, 1, 2), Vec3I(0, 1, 2)});

        GridPtr ls3 = tools::createLevelSetDilatedMesh<GridT>(vertices, tri3, r, voxelSize);

        testDilatedConvexPolygonMeasures<GridT>(vertices, r, ls3);
    }

    // test singular edge
    {
        const float r = 0.1f, voxelSize = 0.025f;
        const Vec3s p0(0.0f, 0.0f, 0.0f), p1(0.0f, 1.0f, 0.0f), p2(0.0f, 0.5f, 0.6f),
                    p3(-0.5f, 0.5f, -0.2f), p4(0.5f, 0.5f, -0.2f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3, p4});
        const std::vector<Vec3I> tris({Vec3I(0, 1, 2), Vec3I(0, 1, 3), Vec3I(0, 1, 4)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tris, r, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        // The dilated mesh should contain the capsule along the singular edge

        GridPtr capsule = tools::createLevelSetCapsule<GridT>(p0, p1, r, voxelSize);
        tools::csgDifference(*capsule, *ls);

        EXPECT_TRUE(tools::minMax(capsule->tree()).min() >= 0.0f); // no zero crossings
    }

    // test singular vertex
    {
        const float r = 0.1f, voxelSize = 0.025f;
        const Vec3s p0(-1.0f, -1.0f, 0.0f), p1(1.0f, -1.0f, 0.0f), p2(0.0f, 0.0f, 0.0f),
                    p3(-1.0f, 1.0f, -0.2f), p4(1.0f, 1.0f, 0.2f);

        const std::vector<Vec3s> vertices({p0, p1, p2, p3, p4});
        const std::vector<Vec3I> tris({Vec3I(0, 1, 2), Vec3I(2, 3, 4)});

        GridPtr ls = tools::createLevelSetDilatedMesh<GridT>(vertices, tris, r, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        // The dilated mesh should contain the sphere at the singular vertex

        GridPtr sphere = tools::createLevelSetCapsule<GridT>(p2, p2, r, voxelSize);
        tools::csgDifference(*sphere, *ls);

        EXPECT_TRUE(tools::minMax(sphere->tree()).min() >= 0.0f); // no zero crossings
    }

    // test self-intersecting faces
    {
        const float r = 0.2f, voxelSize = 0.025f;
        const Vec3s p0(0.0f, 0.0f, 0.0f),   p1(1.0f, 0.0f, 0.0f),     p2(0.0f, 1.0f, 0.0f),
                    p3(0.25f, 0.25f, 0.5f), p4(0.75f, -0.25f, -0.5f), p5(-0.25f, 0.75f, -0.5f);

        const std::vector<Vec3s> vertices1({p0, p1, p2, p3, p4, p5});
        const std::vector<Vec3I> tris1({Vec3I(0, 1, 2), Vec3I(3, 4, 5)});

        // level set from self-intersecting triangles
        GridPtr ls_int = tools::createLevelSetDilatedMesh<GridT>(vertices1, tris1, r, voxelSize);

        const Vec3s q0(0.0f, 0.0f, 0.0f),   q1(1.0f, 0.0f, 0.0f),     q2(0.0f, 1.0f, 0.0f),
                    q3(0.25f, 0.25f, 0.5f), q4(0.75f, -0.25f, -0.5f), q5(-0.25f, 0.75f, -0.5f),
                    q6(0.5f, 0.0f, 0.0f),   q7(0.0f, 0.5f, 0.0f);

        const std::vector<Vec3s> vertices2({q0, q1, q2, q3, q4, q5, q6, q7});
        const std::vector<Vec3I> tris2({Vec3I(7, 0, 6), Vec3I(2, 6, 1), Vec3I(6, 2, 7),
                                        Vec3I(7, 3, 6), Vec3I(7, 4, 5), Vec3I(4, 7, 6)});

        // level set from split triangles that resolve the intersections
        GridPtr ls_split = tools::createLevelSetDilatedMesh<GridT>(vertices2, tris2, r, voxelSize);

        EXPECT_EQ(ls_int->activeVoxelCount(),       ls_split->activeVoxelCount());
        EXPECT_NEAR(tools::levelSetArea(*ls_int),   tools::levelSetArea(*ls_split),   1e-6);
        EXPECT_NEAR(tools::levelSetVolume(*ls_int), tools::levelSetVolume(*ls_split), 1e-6);
    }

}// testFaceTopologies

