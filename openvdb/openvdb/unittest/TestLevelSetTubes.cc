// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>

#include <openvdb/tools/Count.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetTubes.h>

#include <gtest/gtest.h>

class TestLevelSetTubes: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};

namespace {

template<typename GridT>
void
testCapsuleMeasures(const openvdb::Vec3s& p1, const openvdb::Vec3s& p2, const float& r,
                    typename GridT::Ptr& grid, float error = 0.01f)
{
    using namespace openvdb;

    const float pi = math::pi<float>(),
                l = (p2-p1).length();

    const float area         = 2.0f*l*pi*r + 4.0f*pi*math::Pow2(r),
                volume       = l*pi*math::Pow2(r) + 4.0f*pi*math::Pow3(r)/3.0f,
                totGaussCurv = 4.0f*pi, // Gauss-Bonnet
                totMeanCurv  = pi*l + 4.0f*pi*r;

    tools::LevelSetMeasure<GridT> m(*grid);

    EXPECT_NEAR(m.area(true),                 area,         area*error);
    EXPECT_NEAR(m.volume(true),               volume,       volume*error);
    EXPECT_NEAR(m.totGaussianCurvature(true), totGaussCurv, totGaussCurv*error);
    EXPECT_NEAR(m.totMeanCurvature(true),     totMeanCurv,  totMeanCurv*error);
}

template<typename GridT>
void
testTaperedCapsuleMeasures(const openvdb::Vec3s& p1, const openvdb::Vec3s& p2,
                           const float& r1, const float& r2, typename GridT::Ptr& grid,
                           float error = 0.01f, bool test_gauss_curvature = true)
{
    using namespace openvdb;

    const float pi = math::pi<float>(), epsilon = math::Tolerance<float>().value(),
                l = (p2-p1).length(), R = math::Max(r1, r2);

    const float l2 = math::Pow2(l), r12 = math::Pow2(r1),
                r22 = math::Pow2(r2), rdiff2 = math::Pow2(r1-r2);

    float area, volume, totGaussCurv, totMeanCurv;

    if (l <= math::Max(epsilon, R)) {
        // degenerate case: sphere
        area         = 4.0f*pi*math::Pow2(R),
        volume       = 4.0f/3.0f*pi*math::Pow3(R),
        totGaussCurv = 4.0f*pi, // Gauss-Bonnet
        totMeanCurv  = 4.0f*pi*R;
    } else {
        // formulas assume non-degenerate object, i.e. l > 0 && r1 > 0 && r2 > 0
        area         = pi/l * (l + r1 + r2) * (rdiff2 + l*(r1 + r2)),
        volume       = pi/(3.0f*l) * ((l2 + rdiff2)*(r12 + r1*r2 + r22) + 2.0f*l*(r1*r12 + r2*r22)),
        totGaussCurv = 4.0f*pi, // Gauss-Bonnet
        totMeanCurv  = pi/l * (math::Pow2(l) + math::Pow2(r1 - r2) + 2.0f*l*(r1 + r2));
    }

    tools::LevelSetMeasure<GridT> m(*grid);

    EXPECT_NEAR(m.area(true),             area,         area*error);
    EXPECT_NEAR(m.volume(true),           volume,       volume*error);
    EXPECT_NEAR(m.totMeanCurvature(true), totMeanCurv,  totMeanCurv*error);

    // objects with sharp corners or edges tend to give answers far from the analytical solution
    // in cases like these, e.g. cone-tip, we can toggle this test off
    if (test_gauss_curvature) {
        EXPECT_NEAR(m.totGaussianCurvature(true), totGaussCurv, totGaussCurv*error);
    }
}

}


TEST_F(TestLevelSetTubes, testCapsule)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // generic tests
    {
        const Vec3s p1(15.8f, 13.2f, 16.7f), p2(4.3f, 7.9f, -4.8f);
        const float r = 4.3f;

        const float voxelSize = 0.1f, width = 3.25f;
        const Coord ijk(int(p1[0]/voxelSize),
                        int(p1[1]/voxelSize),
                        int(p1[2]/voxelSize));// inside

        GridPtr ls = tools::createLevelSetCapsule<GridT>(p1, p2, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(voxelSize*width, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // test measures
    {
        const Vec3s p1(15.8f, 13.2f, 16.7f), p2(4.3f, 7.9f, -4.8f),
                    p3(-3.0f, -7.4f, 8.9f), p4(-2.7f, 8.9f, 30.4f);
        const float r1 = 4.3f, r2 = 1.2f, r3 = 2.1f, r4 = 3.6f;

        const float voxelSize = 0.1f;

        GridPtr ls1 = tools::createLevelSetCapsule<GridT>(p1, p2, r1, voxelSize);
        GridPtr ls2 = tools::createLevelSetCapsule<GridT>(p1, p3, r2, voxelSize);
        GridPtr ls3 = tools::createLevelSetCapsule<GridT>(p3, p2, r3, voxelSize);
        GridPtr ls4 = tools::createLevelSetCapsule<GridT>(p2, p4, r4, voxelSize);

        testCapsuleMeasures<GridT>(p1, p2, r1, ls1);
        testCapsuleMeasures<GridT>(p1, p3, r2, ls2);
        testCapsuleMeasures<GridT>(p3, p2, r3, ls3);
        testCapsuleMeasures<GridT>(p2, p4, r4, ls4);
    }

    // degenerate case: sphere
    {
        const Vec3s p1(4.3f, 7.9f, -4.8f), p2(4.3f, 7.9f, -4.8f);
        const float r = 4.3f;

        const float voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetCapsule<GridT>(p1, p2, r, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        testCapsuleMeasures<GridT>(p1, p2, r, ls);
    }

    // degenerate case: line
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(1.0f, 0.0f, 0.0f);
        const float r = 0.0f;

        const float voxelSize = 0.1f, width = 2.0f;

        GridPtr ls = tools::createLevelSetCapsule<GridT>(p1, p2, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 117 /* 90 + 27 */);
        EXPECT_TRUE(tools::minMax(ls->tree()).min() >= 0.0f); // no zero crossings
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // degenerate case: point
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(0.0f, 0.0f, 0.0f);
        const float r = 0.0f;

        const float voxelSize = 0.1f, width = 2.0f;

        GridPtr ls = tools::createLevelSetCapsule<GridT>(p1, p2, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 27 /* 3x3x3 grid */);
        EXPECT_TRUE(tools::minMax(ls->tree()).min() >= 0.0f); // no zero crossings
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // degenerate case: negative radius --> empty grid
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(0.0f, 0.0f, 0.0f);
        const float r = -10.0f;

        const float voxelSize = 0.1f, width = 2.0f;

        GridPtr ls = tools::createLevelSetCapsule<GridT>(p1, p2, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

}// testCapsule

TEST_F(TestLevelSetTubes, testTaperedCapsule)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // generic tests
    {
        const Vec3s p1(15.8f, 13.2f, 16.7f), p2(4.3f, 7.9f, -4.8f);
        const float r1 = 4.3f, r2 = 1.2f;

        const float voxelSize = 0.1f, width = 3.25f;
        const Coord ijk(int(p1[0]/voxelSize),
                        int(p1[1]/voxelSize),
                        int(p1[2]/voxelSize));// inside

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r1, r2, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(voxelSize*width, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // test measures
    {
        const Vec3s p1(15.8f, 13.2f, 16.7f), p2(4.3f, 7.9f, -4.8f),
                    p3(-3.0f, -7.4f, 8.9f), p4(-2.7f, 8.9f, 30.4f);
        const float r1 = 4.3f, r2 = 1.2f, r3 = 2.1f, r4 = 3.6f;

        const float voxelSize = 0.1f;

        GridPtr ls1 = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r1, r2, voxelSize);
        GridPtr ls2 = tools::createLevelSetTaperedCapsule<GridT>(p1, p3, r2, r3, voxelSize);
        GridPtr ls3 = tools::createLevelSetTaperedCapsule<GridT>(p3, p2, r3, r4, voxelSize);
        GridPtr ls4 = tools::createLevelSetTaperedCapsule<GridT>(p2, p4, r4, r3, voxelSize);

        testTaperedCapsuleMeasures<GridT>(p1, p2, r1, r2, ls1);
        testTaperedCapsuleMeasures<GridT>(p1, p3, r2, r3, ls2);
        testTaperedCapsuleMeasures<GridT>(p3, p2, r3, r4, ls3);
        testTaperedCapsuleMeasures<GridT>(p2, p4, r4, r3, ls4);
    }

    // degenerate case: capsule
    {
        const Vec3s p1(4.3f, 7.9f, -4.8f), p2(4.3f, 7.9f, -4.8f);
        const float r = 4.3f;

        const float voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r, r, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        testTaperedCapsuleMeasures<GridT>(p1, p2, r, r, ls);
    }

    // degenerate case: sphere by equivalent endpoints
    {
        const Vec3s p1(4.3f, 7.9f, -4.8f), p2(4.3f, 7.9f, -4.8f);
        const float r1 = 4.3f, r2 = 1.2f;

        const float voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r1, r2, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        testTaperedCapsuleMeasures<FloatGrid>(p1, p2, r1, r2, ls);
    }

    // degenerate case: sphere by nested spheres
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(1.0f, 0.0f, 0.0f);
        const float r1 = 1.5f, r2 = 0.5f;

        const float voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r1, r2, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        testTaperedCapsuleMeasures<FloatGrid>(p1, p2, r1, r2, ls);
    }

    // degenerate case: cone with sphere cap (tear drop)
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(4.0f, 0.0f, 0.0f);
        const float r1 = 1.0f, r2 = 0.0f;

        const float voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r1, r2, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        // don't test total Gaussian curvature
        testTaperedCapsuleMeasures<FloatGrid>(p1, p2, r1, r2, ls, 0.05f, false);
    }

    // degenerate case: cone with sphere cap (tear drop) and tip between endpoints
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(4.0f, 0.0f, 0.0f), p2_equiv(2.0f, 0.0f, 0.0f);
        const float r1 = 1.0f, r2 = -1.0f, r2_equiv = 0.0f;

        const float voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r1, r2, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        // don't test total Gaussian curvature
        testTaperedCapsuleMeasures<FloatGrid>(p1, p2_equiv, r1, r2_equiv, ls, 0.05f, false);
    }

    // degenerate case: line
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(1.0f, 0.0f, 0.0f);
        const float r = 0.0f;

        const float voxelSize = 0.1f, width = 2.0f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 117 /* 90 + 27 */);
        EXPECT_TRUE(tools::minMax(ls->tree()).min() >= 0.0f); // no zero crossings
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // degenerate case: point
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(0.0f, 0.0f, 0.0f);
        const float r = 0.0f;

        const float voxelSize = 0.1f, width = 2.0f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 27 /* 3x3x3 grid */);
        EXPECT_TRUE(tools::minMax(ls->tree()).min() >= 0.0f); // no zero crossings
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // degenerate case: negative radius --> empty grid
    {
        const Vec3s p1(0.0f, 0.0f, 0.0f), p2(0.0f, 0.0f, 0.0f);
        const float r = -10.0f;

        const float voxelSize = 0.1f, width = 2.0f;

        GridPtr ls = tools::createLevelSetTaperedCapsule<GridT>(p1, p2, r, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() == 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

}// testTaperedCapsule

TEST_F(TestLevelSetTubes, testTubeComplexConstantRadius)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // generic tests
    {
        const Vec3s p1(15.8f, 13.2f, 16.7f), p2(4.3f, 7.9f, -4.8f),
                    p3(-3.0f, -7.4f, 8.9f), p4(-2.7f, 8.9f, 30.4f);
        const float r = 2.1f;

        const std::vector<Vec3s> vertices({p1, p2, p3, p4});
        const std::vector<Vec2I> segments({Vec2I(0, 1), Vec2I(0, 2), Vec2I(0, 3)});

        const float voxelSize = 0.1f, width = 3.25f;
        const Coord ijk(int(p1[0]/voxelSize),
                        int(p1[1]/voxelSize),
                        int(p1[2]/voxelSize));// inside

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(vertices, segments, r, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(voxelSize*width, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // test measures of complex that forms a single straight capsule
    {
        const std::vector<Vec3s> vertices({Vec3s(0.0f, 0.0f, 0.0f), Vec3s(1.0f, 0.0f, 0.0f),
                                           Vec3s(1.5f, 0.0f, 0.0f), Vec3s(3.0f, 0.0f, 0.0f)});
        const std::vector<Vec2I> segments({Vec2I(0, 1), Vec2I(1, 2), Vec2I(2, 3)});

        const float r = 0.6f, voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(vertices, segments, r, voxelSize);

        // this complex is equivalent to a capsule from first vertex to last vertex with radius r
        testCapsuleMeasures<GridT>(vertices[0], vertices[3], r, ls);
    }

    // test measures of an approximate torus
    {
        const float pi = math::pi<float>(), error = 0.05f,
                    a = 1.2f, c = 3.0f, voxelSize = 0.1f;

        const Index32 n = 128;
        std::vector<Vec3s> vertices;
        std::vector<Vec2I> segments;
        const float delta = 2.0f*pi/static_cast<float>(n);

        float theta = 0.0f;
        for (Index32 i = 0; i < n; ++i, theta += delta) {
            vertices.push_back(Vec3s(c*math::Cos(theta), c*math::Sin(theta), 0.0f));
            segments.push_back(Vec2I(i, (i+1) % n));
        }

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(vertices, segments, a, voxelSize);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        const float area         = 4.0f*math::Pow2(pi)*a*c,
                    volume       = 2.0f*math::Pow2(pi*a)*c,
                    totGaussCurv = 0.0f, // Gauss-Bonnet
                    totMeanCurv  = 2.0f*c*math::Pow2(pi);

        tools::LevelSetMeasure<GridT> m(*ls);

        EXPECT_NEAR(m.area(true),                 area,         area*error);
        EXPECT_NEAR(m.volume(true),               volume,       volume*error);
        EXPECT_NEAR(m.totGaussianCurvature(true), totGaussCurv, 0.1f);
        EXPECT_NEAR(m.totMeanCurvature(true),     totMeanCurv,  totMeanCurv*error);
    }

    // genus of a cube wireframe
    {
        using FilterT = tools::LevelSetFilter<GridT>;

        const std::vector<Vec3s> vertices({
            Vec3s(-0.5, 0.5, -0.5f), Vec3s(0.5, 0.5, -0.5f), Vec3s(0.5, -0.5, -0.5f),
            Vec3s(-0.5, -0.5, -0.5f), Vec3s(-0.5, 0.5, 0.5f), Vec3s(0.5, 0.5, 0.5f),
            Vec3s(0.5, -0.5, 0.5f), Vec3s(-0.5, -0.5, 0.5f)
        });

        const std::vector<Vec2I> segments({
            Vec2I(0, 1), Vec2I(0, 3), Vec2I(0, 4), Vec2I(1, 2), Vec2I(1, 5), Vec2I(2, 3),
            Vec2I(2, 6), Vec2I(3, 7), Vec2I(4, 5), Vec2I(4, 7), Vec2I(5, 6), Vec2I(6, 7)
        });

        const float r = 0.1f, voxelSize = 0.025f;

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(vertices, segments, r, voxelSize);

        // smooth slightly to ensure numerical integration when determining the genus is correct
        FilterT filter(*ls);
        for (Index i = 0; i < 5; ++i)
            filter.meanCurvature();

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        EXPECT_EQ(tools::levelSetGenus(*ls), int(5));
    }

}// testTubeComplexConstantRadius

TEST_F(TestLevelSetTubes, testTubeComplexVariableRadius)
{
    using namespace openvdb;

    using GridT = FloatGrid;
    using GridPtr = typename GridT::Ptr;

    // generic tests
    {
        const Vec3s p1(15.8f, 13.2f, 16.7f), p2(4.3f, 7.9f, -4.8f),
                    p3(-3.0f, -7.4f, 8.9f), p4(-2.7f, 8.9f, 30.4f);
        const float r1 = 4.3f, r2 = 1.2f, r3 = 2.1f, r4 = 3.6f;

        const std::vector<Vec3s> vertices({p1, p2, p3, p4});
        const std::vector<Vec2I> segments({Vec2I(0, 1), Vec2I(0, 2), Vec2I(0, 3)});
        const std::vector<float> radii({r1, r2, r3, r4});

        const float voxelSize = 0.1f, width = 3.25f;
        const Coord ijk(int(p1[0]/voxelSize),
                        int(p1[1]/voxelSize),
                        int(p1[2]/voxelSize));// inside

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(vertices, segments, radii, voxelSize, width);

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_TRUE(ls->tree().isValueOff(ijk));
        EXPECT_NEAR(-ls->background(), ls->tree().getValue(ijk), 1e-6);
        EXPECT_NEAR(voxelSize*width, ls->background(), 1e-6);
        EXPECT_NEAR(ls->background(),ls->tree().getValue(Coord(0)), 1e-6);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));
    }

    // test measures of complex that forms a single straight tapered capsule
    {
        const std::vector<Vec3s> vertices({Vec3s(0.0f, 0.0f, 0.0f), Vec3s(1.0f, 0.0f, 0.0f),
                                           Vec3s(1.5f, 0.0f, 0.0f), Vec3s(3.0f, 0.0f, 0.0f)});
        const std::vector<Vec2I> segments({Vec2I(0, 1), Vec2I(1, 2), Vec2I(2, 3)});
        const std::vector<float> radii({0.3f, 0.4f, 0.45f, 0.6f});

        const float voxelSize = 0.1f;

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(vertices, segments, radii, voxelSize);

        // this complex is equivalent to a tapered capsule from first vertex to last vertex
        testTaperedCapsuleMeasures<GridT>(vertices[0], vertices[3], radii[0], radii[3], ls, 0.02f);
    }

    // test genus of a torus with per-vertex radii
    {
        using FilterT = tools::LevelSetFilter<GridT>;

        const float pi = math::pi<float>(),
                    a = 1.2f, c = 3.0f, voxelSize = 0.1f, width = 3.0f;

        const Index32 n = 128;
        std::vector<Vec3s> vertices;
        std::vector<Vec2I> segments;
        std::vector<float> radii;
        const float delta = 2.0f*pi/static_cast<float>(n);

        float theta = 0.0f;
        for (Index32 i = 0; i < n; ++i, theta += delta) {
            vertices.push_back(Vec3s(c*math::Cos(theta), c*math::Sin(theta), 0.0f));
            segments.push_back(Vec2I(i, (i+1) % n));
            radii.push_back(a + 0.15f*a*math::Sin(5.0f*theta));
        }

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(
            vertices, segments, radii, voxelSize, width, tools::TUBE_VERTEX_RADII);

        // smooth slightly to ensure numerical integration of the genus is correct
        FilterT filter(*ls);
        for (Index i = 0; i < 5; ++i)
            filter.meanCurvature();

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        EXPECT_EQ(tools::levelSetGenus(*ls), int(1));
    }

    // test genus of a torus with per-segment radii
    {
        using FilterT = tools::LevelSetFilter<GridT>;

        const float pi = math::pi<float>(),
                    a = 1.2f, c = 3.0f, voxelSize = 0.1f, width = 3.0f;

        const Index32 n = 10;
        std::vector<Vec3s> vertices;
        std::vector<Vec2I> segments;
        std::vector<float> radii;
        const float delta = 2.0f*pi/static_cast<float>(n);

        float theta = 0.0f;
        for (Index32 i = 0; i < n; ++i, theta += delta) {
            vertices.push_back(Vec3s(c*math::Cos(theta), c*math::Sin(theta), 0.0f));
            segments.push_back(Vec2I(i, (i+1) % n));
            radii.push_back(0.7f*a + 0.3f*a*math::Sin(9.0f*theta));
        }

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(
            vertices, segments, radii, voxelSize, width, tools::TUBE_SEGMENT_RADII);

        // smooth slightly to ensure numerical integration of the genus is correct
        FilterT filter(*ls);
        for (Index i = 0; i < 5; ++i)
            filter.meanCurvature();

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        EXPECT_EQ(tools::levelSetGenus(*ls), int(1));
    }

    // genus of a cube wireframe
    {
        using FilterT = tools::LevelSetFilter<GridT>;

        const std::vector<Vec3s> vertices({
            Vec3s(-0.5, 0.5, -0.5f), Vec3s(0.5, 0.5, -0.5f), Vec3s(0.5, -0.5, -0.5f),
            Vec3s(-0.5, -0.5, -0.5f), Vec3s(-0.5, 0.5, 0.5f), Vec3s(0.5, 0.5, 0.5f),
            Vec3s(0.5, -0.5, 0.5f), Vec3s(-0.5, -0.5, 0.5f)
        });

        const std::vector<Vec2I> segments({
            Vec2I(0, 1), Vec2I(0, 3), Vec2I(0, 4), Vec2I(1, 2), Vec2I(1, 5), Vec2I(2, 3),
            Vec2I(2, 6), Vec2I(3, 7), Vec2I(4, 5), Vec2I(4, 7), Vec2I(5, 6), Vec2I(6, 7)
        });

        const std::vector<float> radii({0.1f, 0.15f, 0.2f, 0.175f, 0.08f, 0.11f, 0.3f, 0.09f});

        const float voxelSize = 0.025f;

        GridPtr ls = tools::createLevelSetTubeComplex<GridT>(vertices, segments, radii, voxelSize);

        // smooth slightly to ensure numerical integration of the genus is correct
        FilterT filter(*ls);
        for (Index i = 0; i < 5; ++i)
            filter.meanCurvature();

        EXPECT_TRUE(ls->activeVoxelCount() > 0);
        EXPECT_EQ(int(GRID_LEVEL_SET), int(ls->getGridClass()));

        EXPECT_EQ(tools::levelSetGenus(*ls), int(5));
    }

}// testTubeComplexConstantRadius
