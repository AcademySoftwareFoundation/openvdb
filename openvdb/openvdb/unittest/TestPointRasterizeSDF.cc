// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/points/PointRasterizeSDF.h>
#include <openvdb/points/PrincipalComponentAnalysis.h>
#include "PointBuilder.h"

#include <gtest/gtest.h>

using namespace openvdb;

class TestPointRasterizeSDF: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointRasterizeSDF


template <typename FilterT>
struct FixedSurfacing
{
    /// @note  The surface and surfaceSmooth methods use the old API. Once this
    ///   has been deprecated they can be swicthed over to the new AI (as
    ///   demonstrated by surfaceEllips).
    FixedSurfacing(const FilterT& f = FilterT()) : filter(f) {}

    //

    FloatGrid::Ptr surface(const Real radius)
    {
        auto sdf = points::rasterizeSpheres(*points, radius, halfband, transform, filter);
        sdf->setName("fixed");
        return sdf;
    }

    template <typename AttList>
    GridPtrVec surface(const Real radius, const std::vector<std::string>& attrs)
    {
        auto grids = points::rasterizeSpheres<points::PointDataGrid, AttList>
            (*points, radius, attrs, halfband, transform, filter);
        grids.front()->setName("fixed");
        return grids;
    }

    //

    FloatGrid::Ptr surfaceSmooth(const Real radius, const Real search)
    {
        auto sdf = points::rasterizeSmoothSpheres(*points, radius, search, halfband, transform, filter);
        sdf->setName("fixed_avg");
        return sdf;
    }

    template <typename AttList>
    GridPtrVec surfaceSmooth(const Real radius, const Real search, const std::vector<std::string>& attrs)
    {
        auto grids = points::rasterizeSmoothSpheres<points::PointDataGrid, AttList>
            (*points, radius, search, attrs, halfband, transform, filter);
        grids.front()->setName("fixed_avg");
        return grids;
    }

    FilterT filter;
    double halfband = LEVEL_SET_HALF_WIDTH;
    points::PointDataGrid::Ptr points = nullptr;
    math::Transform::Ptr transform = nullptr;
};

template <typename FilterT>
struct VariableSurfacing : public FixedSurfacing<FilterT>
{
    VariableSurfacing(const FilterT& f = FilterT()) : FixedSurfacing<FilterT>(f) {}

    FloatGrid::Ptr surface(const Real scale = 1.0, const std::string& pscale = "pscale")
    {
        auto sdf = points::rasterizeSpheres(*(this->points), pscale, scale,
            this->halfband, this->transform, this->filter);
        sdf->setName("variable");
        return sdf;
    }

    template <typename AttList>
    GridPtrVec surface(const Real scale, const std::string& pscale, const std::vector<std::string>& attrs)
    {
        auto grids = points::rasterizeSpheres<points::PointDataGrid, AttList>
            (*(this->points), pscale, attrs, scale, this->halfband, this->transform, this->filter);
        grids.front()->setName("variable");
        return grids;
    }

    FloatGrid::Ptr surfaceSmooth(const Real scale, const Real search, const std::string& pscale)
    {
        auto sdf = points::rasterizeSmoothSpheres(*(this->points), pscale, scale, search,
                        this->halfband, this->transform, this->filter);
        sdf->setName("variable_avg");
        return sdf;
    }

    template <typename AttList>
    GridPtrVec surfaceSmooth(const Real scale, const Real search, const std::string& pscale, const std::vector<std::string>& attrs)
    {
        auto grids = points::rasterizeSmoothSpheres<points::PointDataGrid, AttList>
            (*(this->points), pscale, scale, search, attrs, this->halfband, this->transform, this->filter);
        grids.front()->setName("variable_avg");
        return grids;
    }
};


TEST_F(TestPointRasterizeSDF, testRasterizeSpheres)
{
    // Test no points
    {
        float radius = 0.2f;
        FixedSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({}).voxelsize(0.1).get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());
    }

    // Test single point
    {
        FixedSurfacing<points::NullFilter> s;

        // small radius, small voxel size
        float radius = 0.2f;
        s.halfband = 3;
        s.points = PointBuilder({Vec3f(0)}).voxelsize(0.1).get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(8), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(485), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float(ws.length()); // dist to center
            length -= radius; // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        // larger radius, larger voxel size
        radius = 1.3f;
        s.halfband = 3;
        s.points = PointBuilder({Vec3f(0)}).voxelsize(0.5).get();
        s.transform = nullptr;

        sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(8), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(739), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float(ws.length()); // dist to center
            length -= radius; // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        // offset position, different transform, larger half band
        Vec3f center(-1.2f, 3.4f,-5.6f);
        radius = 2.0f;
        s.halfband = 4;
        s.points = PointBuilder({center}).voxelsize(0.1).get();
        s.transform = math::Transform::createLinearTransform(0.3);

        sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == *s.transform);
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.transform->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(27), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(5005), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord()) - center;
            float length = float(ws.length()); // dist to center
            length -= radius; // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord()) - center;
            float length = float(ws.length()); // dist to center
            // if length is <= the (rad - halfbandws), voxel is inside the surface
            const bool interior = (length <= (radius - (s.halfband * s.transform->voxelSize()[0])));
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(80), interiorOff);
        EXPECT_EQ(size_t(82438), exteriorOff);
    }

    // Test multiple points - 8 points at cube corner positions
    {
        FixedSurfacing<points::NullFilter> s;
        float radius = 0.2f;
        auto positions = getBoxPoints(/*scale*/0.0f);

        // test points overlapping all at 0,0,0 - should produce the same grid as first test
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(8), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(485), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float(ws.length()); // dist to center
            length -= radius; // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        // test points from a box with coords at 0.5
        positions = getBoxPoints(/*scale*/0.5f);
        radius = 0.2f;
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).get();
        s.transform = nullptr;

        sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(38), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(485*8), sdf->tree().activeVoxelCount()); // 485 per sphere

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            // get closest dist from all points
            float length = std::numeric_limits<float>::max();
            for (auto& pos : positions) length = std::min(length, float((ws-pos).length()));
            length -= radius; // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        // test points from a box with coords at 3, rad 10 so
        // overlapping and rotated/scaled/translated transform

        const double deg2rad = math::pi<double>() / 180.0;
        Mat4R mat(Mat4R::identity()); // voxelsize = 1
        mat.preRotate(math::Z_AXIS, deg2rad*99.0);
        mat.preRotate(math::Y_AXIS, deg2rad*66.0);
        mat.preRotate(math::X_AXIS, deg2rad*33.0);
        mat.preScale(Vec3d(1.5));
        mat.preTranslate(Vec3d(-1,2,-3));

        positions = getBoxPoints(/*scale*/3.0f);
        radius = 10.0f;
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).get();
        s.transform = math::Transform::createLinearTransform(mat);

        sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == *s.transform);
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.transform->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(46), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(7198), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            // get closest dist from all points
            float length = std::numeric_limits<float>::max();
            for (auto& pos : positions) length = std::min(length, float((ws-pos).length()));
            length -= radius; // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            // get closest dist from all points
            float length = std::numeric_limits<float>::max();
            for (auto& pos : positions) length = std::min(length, float((ws-pos).length()));
            // if length is <= the (rad - halfbandws), voxel is inside the surface
            const bool interior = (length <= (radius - (s.halfband * s.transform->voxelSize()[0])));
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(1049), interiorOff);
        EXPECT_EQ(size_t(310163), exteriorOff);
    }

    // Test point filter
    {
        // test alternativing points from a box with coords at 0.5
        // spheres end up on a single face and don't overlap
        std::vector<Vec3f> positions = getBoxPoints(/*scale*/0.5f);
        float radius = 0.2f;
        auto points = PointBuilder(positions)
            .voxelsize(0.1)
            .group({1,0,1,0,1,0,1,0}, "test")
            .get();
        points::GroupFilter filter("test", points->tree().cbeginLeaf()->attributeSet());
        FixedSurfacing<points::GroupFilter> s(filter);
        s.halfband = 3;
        s.points = points;
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface(radius);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(17), sdf->tree().leafCount()); // less leaf nodes, active points are on a single face
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(485*4), sdf->tree().activeVoxelCount()); // 485 per sphere

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            // get closest dist from all points
            float length = std::numeric_limits<float>::max();
            for (auto& pos : positions) length = std::min(length, float((ws-pos).length()));
            length -= radius; // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }
    }
}


TEST_F(TestPointRasterizeSDF, testRasterizeVariableSurfacing)
{
    // First few tests check that the results are fp equivalent to fixed spheres

    // Test no points
    {
        VariableSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({}).voxelsize(0.1).get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface();
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());
    }

    // Test single point
    {
        VariableSurfacing<points::NullFilter> s;
        float radius = 0.2f;

        // small radius, small voxel size
        s.halfband = 3;
        s.points = PointBuilder({ Vec3f(0) }).voxelsize(0.1).attribute(radius, "pscale").get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface();
        FloatGrid::Ptr comp = s.FixedSurfacing<points::NullFilter>::surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });

        // larger radius, larger voxel size
        radius = 1.3f;
        s.halfband = 3;
        s.points = PointBuilder({ Vec3f(0) }).voxelsize(0.5).attribute(radius, "pscale").get();
        s.transform = nullptr;

        sdf = s.surface();
        comp = s.FixedSurfacing<points::NullFilter>::surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });

        // offset position, different transform, larger half band
        Vec3f center(-1.2f, 3.4f,-5.6f);
        radius = 2.0f;
        s.halfband = 4;
        s.points =PointBuilder({ center }).voxelsize(0.1).attribute(radius, "pscale").get();
        s.transform = math::Transform::createLinearTransform(0.3);

        sdf = s.surface();
        comp = s.FixedSurfacing<points::NullFilter>::surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });
    }

    // Test multiple points - 8 points at cube corner positions
    {
        VariableSurfacing<points::NullFilter> s;
        float radius = 0.2f;
        auto positions = getBoxPoints(/*scale*/0.0f);
        // test points overlapping all at 0,0,0 - should produce the same grid as first test
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).attribute(radius, "pscale").get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface();
        FloatGrid::Ptr comp = s.FixedSurfacing<points::NullFilter>::surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });

        // test points from a box with coords at 0.5
        radius = 0.2f;
        positions = getBoxPoints(/*scale*/0.5f);
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).attribute(radius, "pscale").get();
        s.transform = nullptr;

        sdf = s.surface();
        comp = s.FixedSurfacing<points::NullFilter>::surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });

        // test points from a box with coords at 3, rad 10 so
        // overlapping and rotated/scaled/translated transform

        const double deg2rad = math::pi<double>() / 180.0;
        Mat4R mat(Mat4R::identity()); // voxelsize = 1
        mat.preRotate(math::Z_AXIS, deg2rad*99.0);
        mat.preRotate(math::Y_AXIS, deg2rad*66.0);
        mat.preRotate(math::X_AXIS, deg2rad*33.0);
        mat.preScale(Vec3d(1.5));
        mat.preTranslate(Vec3d(-1,2,-3));

        radius = 10.0f;
        positions = getBoxPoints(/*scale*/3.0f);
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).attribute(radius, "pscale").get();
        s.transform = math::Transform::createLinearTransform(mat);

        sdf = s.surface();
        comp = s.FixedSurfacing<points::NullFilter>::surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });
    }

    // Test point filter
    {
        // test alternativing points from a box with coords at 0.5
        // spheres end up on a single face and don't overlap
        std::vector<Vec3f> positions = getBoxPoints(/*scale*/0.5f);
        float radius = 0.2f;
        auto points = PointBuilder(positions)
            .voxelsize(0.1)
            .group({1,0,1,0,1,0,1,0}, "test")
            .attribute(radius, "pscale")
            .get();
        points::GroupFilter filter("test", points->tree().cbeginLeaf()->attributeSet());
        VariableSurfacing<points::GroupFilter> s(filter);
        s.halfband = 3;
        s.points = points;
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface();
        FloatGrid::Ptr comp = s.FixedSurfacing<points::GroupFilter>::surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });
    }

    // Test multiple points with different radius values
    {
        std::vector<Vec3f> positions = getBoxPoints(/*scale*/0.5f);
        std::vector<float> rads = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f};
        float radscale = 0.5f;

        auto points = PointBuilder(positions)
            .voxelsize(0.2)
            .attribute(rads, "myrad")
            .get();

        VariableSurfacing<points::NullFilter> s;
        s.halfband = 1; // small half band
        s.points = points;
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface(radscale, "myrad");
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(8), sdf->tree().leafCount()); // less leaf nodes, active points are on a single face
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(1454), sdf->tree().activeVoxelCount()); // 485 per sphere

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            // get closest dist from all points including rad
            float length = std::numeric_limits<float>::max();
            for (size_t i = 0; i < positions.size(); ++i) {
                length = std::min(length, float((ws-positions[i]).length()) - (rads[i] * radscale));
            }
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            // get closest dist from all points
            float length = std::numeric_limits<float>::max();
            for (size_t i = 0; i < positions.size(); ++i) {
                length = std::min(length, float((ws-positions[i]).length()) - (rads[i] * radscale));
            }
            // if length is <= the (rad - halfbandws), voxel is inside the surface
            const bool interior = (length <= (s.halfband * sdf->voxelSize()[0]));
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(664), interiorOff);
        EXPECT_EQ(size_t(296874), exteriorOff);
    }
}


TEST_F(TestPointRasterizeSDF, testRasterizeSmoothSpheres)
{
    // Test no points
    {
        float radius = 0.2f, search = 0.4f;
        FixedSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({}).voxelsize(0.1).get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surfaceSmooth(radius, search);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());
    }

    // Test single point
    {
        FixedSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({ Vec3f(0) }).voxelsize(0.1).get();
        s.transform = nullptr;
        double radius = 0.2, search = 0.0; // 0.0 search = no result

        FloatGrid::Ptr sdf = s.surfaceSmooth(radius, search);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());

        // result should be exactly the same as normal rasterizeSphere
        s.halfband = 3; // half band equal to search radius to ensure same topology
        s.transform = nullptr;
        radius = 0.2, search = 0.6;

        sdf = s.surfaceSmooth(radius, search);
        FloatGrid::Ptr comp = s.surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });

        // Test exterior half band limit
        s.halfband = 3;
        s.transform = nullptr;
        radius = 0.5, search = 5; // search of 5 allows for halfband size to up 10
        sdf = s.surfaceSmooth(radius, search);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            EXPECT_TRUE(*iter > -sdf->background());
            EXPECT_TRUE(*iter < sdf->background());
        }
    }

    // Test point filter
    {
        std::vector<Vec3f> positions = getBoxPoints(/*scale*/1.0f);
        double radius = 0.6, search = 2.0;
        auto points = PointBuilder(positions)
            .voxelsize(0.1)
            .group({1,0,0,0,0,0,0,0}, "test") // only first point
            .get();
        points::GroupFilter filter("test", points->tree().cbeginLeaf()->attributeSet());
        FixedSurfacing<points::GroupFilter> s(filter);
        s.halfband = 3;
        s.points = points;
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surfaceSmooth(radius, search);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");

        // voxels values should be based on first position
        Vec3f pos = positions.front();
        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float((ws-pos).length() - radius); // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_TRUE(sdf->background() == *iter ||
                -sdf->background() == *iter);
        }
    }

    // Test two points qhich create a ghost particle outside of their
    // radii (to test that the surface topology correctly accounts for the
    // search distance)
    {
        FixedSurfacing<points::NullFilter> s;
        s.halfband = 2;
        s.points = PointBuilder({ Vec3f(0), Vec3f(0,5,0) }).voxelsize(0.1).get();
        s.transform = nullptr;
        double radius = 0.5, search = 5; // large search

        FloatGrid::Ptr sdf = s.surfaceSmooth(radius, search);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        // @todo  regression test. find a way to better test these values
        size_t interiorOn = 0, exteriorOn = 0;
        EXPECT_EQ(Index32(36), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(4188), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            EXPECT_TRUE(*iter > -sdf->background());
            EXPECT_TRUE(*iter < sdf->background());
            if (*iter > 0) ++exteriorOn;
            else           ++interiorOn;
        }
        EXPECT_EQ(size_t(1244), interiorOn);
        EXPECT_EQ(size_t(2944), exteriorOn);

        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_TRUE((*iter <= -sdf->background()) || (*iter >= sdf->background()))
                << *iter << " " << sdf->background();;
            if (*iter > 0) ++exteriorOff;
            else           ++interiorOff;
        }
        EXPECT_EQ(size_t(323), interiorOff);
        EXPECT_EQ(size_t(308789), exteriorOff);
    }

    // Test multiple points - 8 points at cube corner positions
    {
        FixedSurfacing<points::NullFilter> s;
        // test points overlapping all at 0,0,0 - should produce the same grid as first test
        auto positions = getBoxPoints(/*scale*/0.0f);
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).get();
        s.transform = nullptr;
        double radius = 0.2, search = 0.6; // radius * 3

        FloatGrid::Ptr sdf = s.surfaceSmooth(radius, search);
        FloatGrid::Ptr comp = s.surface(radius);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");
        EXPECT_TRUE(comp && comp->getName() == "fixed");
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });

        // test points from a box with coords at 0.5 and a search radius
        // large enough to create a smoothed box
        // @todo  regression test. find a way to better test these values
        positions = getBoxPoints(/*scale*/0.5f);
        radius = 0.2, search = 1.2;
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).get();
        s.transform = nullptr;

        sdf = s.surfaceSmooth(radius, search);
        EXPECT_TRUE(sdf && sdf->getName() == "fixed_avg");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index64(44), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(6303), sdf->tree().activeVoxelCount());
        const CoordBBox bounds(Coord(-7), Coord(7));
        for (CoordBBox::XYZIterator iter(bounds); iter; ++iter) {
            EXPECT_TRUE(sdf->tree().isValueOn(*iter));
        }
        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            EXPECT_TRUE(*iter > -sdf->background());
            EXPECT_TRUE(*iter < sdf->background());
        }
        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }
    }
}


TEST_F(TestPointRasterizeSDF, testRasterizeVariableSmoothSpheres)
{
    // Test no points
    {
        float radius = 0.2f, search = 0.4f;
        VariableSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({}).voxelsize(0.1).get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surfaceSmooth(radius, search, "pscale");
        EXPECT_TRUE(sdf && sdf->getName() == "variable_avg");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());
    }

    // Test single point
    {
        VariableSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({ Vec3f(0) }).voxelsize(0.1).attribute(0.2f, "rad").get();
        s.transform = math::Transform::createLinearTransform(0.1);
        double scale = 1.0, search = 0.0; // 0.0 search = no result

        FloatGrid::Ptr sdf = s.surfaceSmooth(scale, search, "rad");
        EXPECT_TRUE(sdf && sdf->getName() == "variable_avg");
        EXPECT_TRUE(sdf->transform() == *s.transform);
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.transform->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());

        // result should be exactly the same as normal rasterizeSphere
        s.halfband = 3; // half band equal to search radius to ensure same topology
        s.transform = math::Transform::createLinearTransform(0.1);
        scale = 1.0;
        search = 0.6;

        sdf = s.surfaceSmooth(scale, search, "rad");
        FloatGrid::Ptr comp = s.FixedSurfacing<points::NullFilter>::surfaceSmooth(0.2, search);
        EXPECT_TRUE(sdf && sdf->getName() == "variable_avg");
        EXPECT_TRUE(comp && comp->getName() == "fixed_avg");
        EXPECT_TRUE(sdf->transform() == *s.transform);
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.transform->voxelSize()[0]), sdf->background());
                EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });

        // Test exterior half band limit and radius scale
        s.halfband = 3;
        s.points = PointBuilder({ Vec3f(0) }).voxelsize(0.1).attribute(0.4f, "rad").get();
        s.transform = math::Transform::createLinearTransform(0.1);
        scale = 0.5; // 0.4*0.5 = 0.2
        search = 5; // search of 5 allows for halfband size to up 10
        sdf = s.surfaceSmooth(scale, search, "rad");
        comp = s.FixedSurfacing<points::NullFilter>::surfaceSmooth(0.2, search);
        EXPECT_TRUE(sdf && sdf->getName() == "variable_avg");
        EXPECT_TRUE(comp && comp->getName() == "fixed_avg");
        EXPECT_TRUE(sdf->transform() == *s.transform);
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.transform->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->tree().hasSameTopology(comp->tree()));
        tools::foreach(sdf->cbeginValueAll(), [&comp](auto iter) {
            auto val = comp->tree().getValue(iter.getCoord());
            EXPECT_NEAR(val, *iter, 1e-6f);
        });
    }

    // Test point filter
    {
        std::vector<Vec3f> positions = getBoxPoints(/*scale*/1.0f);
        double radius = 1.0, scale = 0.6, search = 2.0;
        auto points = PointBuilder(positions)
            .voxelsize(0.1)
            .group({1,0,0,0,0,0,0,0}, "test") // only first point
            .attribute(float(radius), "pscale")
            .get();
        points::GroupFilter filter("test", points->tree().cbeginLeaf()->attributeSet());
        VariableSurfacing<points::GroupFilter> s(filter);
        s.halfband = 3;
        s.points = points;
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surfaceSmooth(scale, search, "pscale");
        EXPECT_TRUE(sdf && sdf->getName() == "variable_avg");

        // voxels values should be based on first position
        Vec3f pos = positions.front();
        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float((ws-pos).length() - (radius*scale)); // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_TRUE(sdf->background() == *iter ||
                -sdf->background() == *iter);
        }
    }

    // Test multiple points - 8 points at cube corner positions
    {
        // test points from a box with coords at 1.0 and a search radius
        // large enough to create a smoothed box with each corner having
        // a sphere of a different radius
        // @todo find a way to better test these values
        std::vector<Vec3f> positions = getBoxPoints(/*scale*/1.0f);
        std::vector<float> rads = {1.1f, 1.3f, 1.5f, 1.7f, 2.1f, 2.3f, 2.5f, 2.7f};
        double scale = 0.6, search = 2.0;

        VariableSurfacing<points::NullFilter> s;
        s.halfband = 4; // large enough to fill interior of the cube
        s.points = PointBuilder(positions).voxelsize(0.2).attribute(rads, "myrad").get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surfaceSmooth(scale, search, "myrad");
        EXPECT_TRUE(sdf && sdf->getName() == "variable_avg");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_EQ(Index64(64), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(15011), sdf->tree().activeVoxelCount());
        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            EXPECT_TRUE(*iter > -sdf->background());
            EXPECT_TRUE(*iter < sdf->background());
        }
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_TRUE(sdf->background() == *iter ||
                -sdf->background() == *iter);
        }
    }
}

TEST_F(TestPointRasterizeSDF, testRasterizeEllipsoids)
{
    // @brief  Small helper struct to represent a single ellipsoid
    struct Ellipse
    {
        Ellipse(const Vec3d center, const Vec3d stretch, const Mat3s rotation)
            : mCenter(center), mStretch(stretch), mRotation(rotation)
            , mInverseRotation(mRotation.transpose())
            , mInverseTransform([&](){
                // construct inverse transformation to create sphere out of an ellipsoid
                // The transform that defines how we go from each voxel back to the source point
                const Vec3d inverseStretch = 1.0f / mStretch;
                math::Mat3s invDiag;
                invDiag.setSymmetric(inverseStretch, Vec3f(0));
                return invDiag * mInverseRotation;
            }())
            , mMinRadius(std::min(mStretch.x(), std::min(mStretch.y(), mStretch.z()))) {}

#if OPENVDB_ELLIPSOID_KERNEL_MODE == 1
        double project(const Vec3d P) const
        {
            const Vec3d dir = (P-mCenter);
            const Vec3d pUnitSphere = mInverseTransform * dir;
            const double len = math::Sqrt(pUnitSphere.lengthSqr());

            if (len == 0) { // exactly at center
                return -mMinRadius;
            }
            else {
                Vec3d ellipsNormal = (mInverseTransform.transpose() * pUnitSphere);
                ellipsNormal.normalize();
                // project back
                return static_cast<double>(dir.dot(ellipsNormal) * (1.0 - (float(1.0)/len)));
            }
        }
#elif OPENVDB_ELLIPSOID_KERNEL_MODE == 2
        double project(const Vec3d P) const
        {
            const Vec3d dir = (P-mCenter);
            const Vec3d pUnitSphere = mInverseRotation * dir;
            const Vec3d radInv = 1.0f / mStretch;
            const Vec3d radInv2 = 1.0f / math::Pow2(mStretch);
            const double k1 = (pUnitSphere * radInv).length();
            if (k1 == 0) { // exactly at center
                return -mMinRadius;
            }
            const double k2 = (pUnitSphere * radInv2).length();
            return static_cast<double>((k1 * (k1 - double(1.0)) / k2));
        }
#endif

    private:
        Vec3d mCenter;
        Vec3d mStretch;
        math::Mat3s mRotation;
        math::Mat3s mInverseRotation;
        math::Mat3s mInverseTransform;
        double mMinRadius;
    };

    //

    const points::PcaAttributes pcaAttrs;

    // Test no points
    {
        points::EllipsoidSettings<> s;
        s.radiusScale = Vec3f(0.2f);
        s.halfband = 3;
        s.transform = nullptr;

        auto points = PointBuilder({}).voxelsize(0.1).get();
        auto grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids.front());

        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());
    }

    // Test single point which is not treated as an ellips. This should give
    // identicle results to SphereSettings<>
    {
        points::EllipsoidSettings<> s;
        s.radiusScale = Vec3f(0.2f);
        s.halfband = 3;
        s.transform = nullptr;
        s.radius = pcaAttrs.stretch;
        s.xform = pcaAttrs.xform;
        s.pws = pcaAttrs.positionWS;

        /// 1) test with a single point with uniform stretch
        auto points = PointBuilder({Vec3f(0)})
            .voxelsize(0.1)
            .attribute(points::PcaAttributes::StretchT(1.0), pcaAttrs.stretch) // uniform stretch
            .attribute(points::PcaAttributes::RotationT::identity(), pcaAttrs.xform)
            .attribute(points::PcaAttributes::PosWsT(0.0), pcaAttrs.positionWS)
            .get();

        auto grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(8), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(485), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float(ws.length()); // dist to center
            length -= float(s.radiusScale.x()); // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        /// 2) larger radius, larger voxel size
        s.radiusScale = Vec3f(1.3f);
        float stretch = 1.6f;
        points = PointBuilder({Vec3f(0)})
            .voxelsize(0.5)
            .attribute(points::PcaAttributes::StretchT(stretch), pcaAttrs.stretch) // uniform stretch
            .attribute(points::PcaAttributes::RotationT::identity(), pcaAttrs.xform)
            .attribute(points::PcaAttributes::PosWsT(0.0), pcaAttrs.positionWS)
            .get();

        grids = points::rasterizeSdf(*points, s);
        sdf = StaticPtrCast<FloatGrid>(grids.front());

        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(8), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(1544), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float(ws.length()); // dist to center
            length -= float(s.radiusScale.x() * stretch); // account for radius and sphere scale
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float(ws.length()); // dist to center
            // if length is <= the (rad - halfbandws), voxel is inside the surface
            const bool interior = (length <= ((s.radiusScale.x() * stretch) - (s.halfband * sdf->voxelSize()[0])));
            if (interior) EXPECT_EQ(-(sdf->background()), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(7), interiorOff);
        EXPECT_EQ(size_t(297441), exteriorOff);

        /// 3) offset position, different transform, larger half band
        stretch = 1.0f;
        s.radiusScale = Vec3f(2.0f);
        s.halfband = 4;
        s.transform = math::Transform::createLinearTransform(0.3);
        points::PcaAttributes::RotationT rot;
        rot.setToRotation({0,1,0}, 45); // arbitrary rotation should have no effect

        const Vec3f center(-1.2f, 3.4f,-5.6f);
        points = PointBuilder({Vec3f(center)})
            .voxelsize(0.1)
            .attribute(points::PcaAttributes::StretchT(stretch), pcaAttrs.stretch)
            .attribute(rot, pcaAttrs.xform)
            .attribute(points::PcaAttributes::PosWsT(center), pcaAttrs.positionWS)
            .get();

        grids = points::rasterizeSdf(*points, s);
        sdf = StaticPtrCast<FloatGrid>(grids.front());

        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == *s.transform);
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.transform->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(27), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(5005), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord()) - center;
            float length = float(ws.length()); // dist to center
            length -= float(s.radiusScale.x()); // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values
        interiorOff = 0;
        exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord()) - center;
            float length = float(ws.length()); // dist to center
            // if length is <= the (rad - halfbandws), voxel is inside the surface
            const bool interior = (length <= (s.radiusScale.x() - (s.halfband * s.transform->voxelSize()[0])));
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(80), interiorOff);
        EXPECT_EQ(size_t(82438), exteriorOff);
    }

    // Test single point which is treated as an ellips with scale and rotation
    // along one principal axis
    {
        points::EllipsoidSettings<> s;
        s.radiusScale = Vec3f(0.8f);
        s.halfband = 3;
        s.transform = nullptr;
        s.radius = pcaAttrs.stretch;
        s.xform = pcaAttrs.xform;
        s.pws = pcaAttrs.positionWS;

        // Design an ellips that is squashed in XYZ and then rotated
        const points::PcaAttributes::StretchT stretch(1.0f, 0.2f, 1.0f);
        points::PcaAttributes::RotationT rot;
         // 45 degree rotation about Y (we only squash in Y so this should be a no-op)
        rot.setToRotation({0,1,0}, 45);

        /// 1) test with a single ellips with Y stretch/rotation
        auto points = PointBuilder({Vec3f(0)})
            .voxelsize(0.1)
            .attribute(stretch, pcaAttrs.stretch)
            .attribute(rot, pcaAttrs.xform)
            .attribute(points::PcaAttributes::PosWsT(0.0), pcaAttrs.positionWS)
            .get();

        auto grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(32), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(3705), sdf->tree().activeVoxelCount());

        Ellipse ellipse(Vec3d(0), (stretch * (s.radiusScale/sdf->voxelSize()[0])), rot);

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const float distance = float(ellipse.project(iter.getCoord().asVec3d()) * float(sdf->voxelSize()[0]));
            EXPECT_NEAR(distance, *iter, 1e-6f);
        }

        // check off values (because we're also squashing the halfband we
        // just compre the overal length to 0.0)
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const float distance = float(ellipse.project(iter.getCoord().asVec3d()) * float(sdf->voxelSize()[0]));
            const bool interior = (distance <= 0.0);
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(80), interiorOff);
        EXPECT_EQ(size_t(307471), exteriorOff);

        // Run again with a different Y rotation, should be the same as the above
        // as we're only squashing in Y
        rot.setToRotation({0,1,0}, -88);
        points = PointBuilder({Vec3f(0)})
            .voxelsize(0.1)
            .group({1}, pcaAttrs.ellipses) // surface as an ellips
            .attribute(stretch, pcaAttrs.stretch)
            .attribute(rot, pcaAttrs.xform)
            .attribute(points::PcaAttributes::PosWsT(0.0), pcaAttrs.positionWS)
            .get();

        grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf2 = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf2);
        EXPECT_TRUE(sdf2->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf2->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf2->background());
        EXPECT_TRUE(sdf->tree().hasSameTopology(sdf2->tree()));

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            EXPECT_NEAR(sdf2->tree().getValue(iter.getCoord()), *iter, 1e-6f);
        }
    }

    // Test single point which is treated as an ellips with scale and rotation
    // along multiple axis
    {
        points::EllipsoidSettings<> s;
        s.halfband = 5;
        s.radius = pcaAttrs.stretch;
        s.xform = pcaAttrs.xform;
        s.pws = pcaAttrs.positionWS;

        // Design an ellips that is squashed in XYZ and then rotated
        const points::PcaAttributes::StretchT stretch(0.3f, 0.6f, 1.8f);
        points::PcaAttributes::RotationT a,b,c,rot;
        a.setToRotation({1,0,0}, 20);
        b.setToRotation({0,1,0}, 45);
        c.setToRotation({0,0,1}, 66);
        rot = a * b * c;

        //

        /// 1) test with a single ellips with Y stretch/rotation
        const Vec3f center(-1.2f, 3.4f,-5.6f);
        auto points = PointBuilder({center})
            .voxelsize(0.2)
            .attribute(stretch, pcaAttrs.stretch)
            .attribute(rot, pcaAttrs.xform)
            .attribute(points::PcaAttributes::PosWsT(center), pcaAttrs.positionWS)
            .get();

        auto grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(32), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(4715), sdf->tree().activeVoxelCount());

        const Ellipse ellipse(sdf->worldToIndex(center), stretch * (1/sdf->voxelSize()[0]), rot);

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const float distance = float(ellipse.project(iter.getCoord().asVec3d()) * float(sdf->voxelSize()[0]));
            EXPECT_NEAR(distance, *iter, 1e-6f);
        }

        // check off values (because we're also squashing the halfband we
        // just compre the overal length to 0.0)
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const float distance = float(ellipse.project(iter.getCoord().asVec3d()) * float(sdf->voxelSize()[0]));
            const bool interior = (distance <= 0.0);
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(0), interiorOff);
        EXPECT_EQ(size_t(85363), exteriorOff);
    }
}


TEST_F(TestPointRasterizeSDF, testAttrTransfer)
{
    // Test no points
    {
        float radius = 0.2f, search = 0.4f;
        FixedSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({}).voxelsize(0.1).get();
        s.transform = nullptr;

        GridPtrVec grids = s.surface<TypeList<>>(radius, {"test1", "test2"});
        EXPECT_EQ(size_t(1), grids.size());
        auto sdf = DynamicPtrCast<FloatGrid>(grids[0]);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());

        grids = s.surfaceSmooth<TypeList<>>(radius, search, {"test1", "test2"});
        EXPECT_EQ(size_t(1), grids.size());
        sdf = DynamicPtrCast<FloatGrid>(grids[0]);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());
    }

    // Test 8 point attribute transfers for normal spheres and smooth spheres
    {
        FixedSurfacing<points::NullFilter> s;

        std::vector<Vec3f> positions = getBoxPoints(/*scale*/0.5f);
        const std::vector<int64_t> test1data = {9,10,11,12,13,14,15,16};
        const std::vector<double> test2data = {-1.1,-2.2,-3.3,-4.4,-5.5,-6.6,-7.7,-8.8};

        // test cloest point transfer
        float radius = 0.2f, search = 0.4f;
        s.halfband = 1;
        s.transform = nullptr;
        s.points = PointBuilder(positions)
            .voxelsize(0.1)
            .attribute(test1data, "test1")
            .attribute(test2data, "test2")
            .get();

        // check throw on invalid attr type list
        EXPECT_THROW(s.surface<TypeList<>>(radius, {"test1", "test2"}), RuntimeError);
        EXPECT_THROW(s.surface<TypeList<char>>(radius, {"test1", "test2"}), RuntimeError);
        EXPECT_THROW(s.surfaceSmooth<TypeList<>>(radius, search, {"test1", "test2"}), RuntimeError);
        EXPECT_THROW(s.surfaceSmooth<TypeList<char>>(radius, search, {"test1", "test2"}), RuntimeError);

        // test both transfers
        GridPtrVec grids1 = s.surface<TypeList<int64_t, double>>(radius, {"test1", "test2"});
        GridPtrVec grids2 = s.surfaceSmooth<TypeList<int64_t, double>>(radius, search, {"test1", "test2"});
        EXPECT_EQ(size_t(3), grids1.size());
        EXPECT_EQ(size_t(3), grids2.size());

        for (int i = 0; i < 2; ++ i) {
            GridPtrVec grids = i == 0 ? grids1 : grids2;

            auto sdf = DynamicPtrCast<FloatGrid>(grids[0]);
            EXPECT_TRUE(sdf);
            if (i == 0) EXPECT_TRUE(sdf->getName() == "fixed");
            else        EXPECT_TRUE(sdf->getName() == "fixed_avg");
            EXPECT_TRUE(sdf->transform() == s.points->transform());
            EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
            EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());

            if (i == 0) EXPECT_EQ(Index64(928), sdf->tree().activeVoxelCount());
            else        EXPECT_EQ(Index64(856), sdf->tree().activeVoxelCount());

            auto test1 = DynamicPtrCast<Int64Grid>(grids[1]);
            EXPECT_TRUE(test1 && test1->getName() == "test1");
            EXPECT_TRUE(sdf->transform() == test1->transform());
            EXPECT_EQ(GRID_UNKNOWN, test1->getGridClass());
            EXPECT_EQ(zeroVal<int64_t>(), test1->background());

            auto test2 = DynamicPtrCast<DoubleGrid>(grids[2]);
            EXPECT_TRUE(test2 && test2->getName() == "test2");
            EXPECT_TRUE(sdf->transform() == test2->transform());
            EXPECT_EQ(GRID_UNKNOWN, test2->getGridClass());
            EXPECT_EQ(zeroVal<double>(), test2->background());

            EXPECT_TRUE(sdf->tree().hasSameTopology(test1->tree()));
            EXPECT_TRUE(sdf->tree().hasSameTopology(test2->tree()));

            // check atributes that have been transfered are correct
            for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
                const Coord ijk = iter.getCoord();
                const Vec3d ws = sdf->transform().indexToWorld(ijk);
                // get closest dist from all points including rad
                float length = std::numeric_limits<float>::max();
                size_t idx = 0;
                for (size_t i = 0; i < positions.size(); ++i) {
                    float min = float((ws-positions[i]).length()) - radius;
                    if (min < length) {
                        idx = i;
                        length = min;
                    }
                }

                EXPECT_TRUE(test1->tree().isValueOn(ijk));
                EXPECT_TRUE(test2->tree().isValueOn(ijk));
                EXPECT_EQ(test1data[idx], test1->tree().getValue(ijk));
                EXPECT_EQ(test2data[idx], test2->tree().getValue(ijk));
            }

            tools::foreach(test1->cbeginValueOff(), [&sdf](auto iter) {
                EXPECT_TRUE(!sdf->tree().isValueOn(iter.getCoord()));
                EXPECT_EQ(zeroVal<int64_t>(), *iter);
            });
            tools::foreach(test2->cbeginValueOff(), [&sdf](auto iter) {
                EXPECT_TRUE(!sdf->tree().isValueOn(iter.getCoord()));
                EXPECT_EQ(zeroVal<double>(), *iter);
            });
        }
    }
}


TEST_F(TestPointRasterizeSDF, testVariableAttrTransfer)
{
    // Test no points
    {
        float radius = 0.2f, search = 0.4f;
        VariableSurfacing<points::NullFilter> s;
        s.halfband = 3;
        s.points = PointBuilder({}).voxelsize(0.1).get();
        s.transform = nullptr;

        GridPtrVec grids = s.surface<TypeList<>>(radius, "pscale", {"test1", "test2"});
        EXPECT_EQ(size_t(1), grids.size());
        auto sdf = DynamicPtrCast<FloatGrid>(grids[0]);
        EXPECT_TRUE(sdf && sdf->getName() == "variable");
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());

        grids = s.surfaceSmooth<TypeList<>>(radius, search, "pscale", {"test1", "test2"});
        EXPECT_EQ(size_t(1), grids.size());
        sdf = DynamicPtrCast<FloatGrid>(grids[0]);
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == s.points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * s.points->voxelSize()[0]), sdf->background());
        EXPECT_TRUE(sdf->empty());
    }

    // Test 8 point attribute transfers - overlapping spheres with varying
    // radii and a different target transform
    {
        VariableSurfacing<points::NullFilter> s;

        std::vector<Vec3f> positions = getBoxPoints(/*scale*/0.35f);
        const std::vector<int64_t> test1data = {0,1,2,3,4,5,6,7};
        const std::vector<double> test2data = {-1.1,-2.2,-3.3,-4.4,-5.5,-6.6,-7.7,-8.8};
        const std::vector<float> rads = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f};
        // @note search needs to be high enough here to pick up all
        //   contributions to match normal sphere attribute transfer
        const float radscale = 0.5, search = 3.0;

        // test cloest point transfer
        s.halfband = 3;
        s.transform = math::Transform::createLinearTransform(0.3);
        s.points = PointBuilder(positions)
            .voxelsize(0.7)
            .attribute(rads, "myrad")
            .attribute(test1data, "test1")
            .attribute(test2data, "test2")
            .get();

        // check throw on invalid attr type list
        EXPECT_THROW(s.surface<TypeList<>>(radscale, "myrad", {"test1", "test2"}), RuntimeError);
        EXPECT_THROW(s.surface<TypeList<char>>(radscale, "myrad", {"test1", "test2"}), RuntimeError);
        EXPECT_THROW(s.surfaceSmooth<TypeList<>>(radscale, search, "myrad", {"test1", "test2"}), RuntimeError);
        EXPECT_THROW(s.surfaceSmooth<TypeList<char>>(radscale, search, "myrad", {"test1", "test2"}), RuntimeError);

        // test both transfers
        GridPtrVec grids1 = s.surface<TypeList<int64_t, double>>(radscale, "myrad", {"test1", "test2"});
        GridPtrVec grids2 = s.surfaceSmooth<TypeList<int64_t, double>>(radscale, search, "myrad", {"test1", "test2"});
        EXPECT_EQ(size_t(3), grids1.size());
        EXPECT_EQ(size_t(3), grids2.size());

        for (int i = 0; i < 2; ++ i) {
            GridPtrVec grids = i == 0 ? grids1 : grids2;

            auto sdf = DynamicPtrCast<FloatGrid>(grids[0]);
            EXPECT_TRUE(sdf);
            if (i == 0) EXPECT_TRUE(sdf->getName() == "variable");
            else        EXPECT_TRUE(sdf->getName() == "variable_avg");
            EXPECT_TRUE(sdf->transform() == *s.transform);
            EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
            EXPECT_EQ(float(s.halfband * s.transform->voxelSize()[0]), sdf->background());

            if (i == 0) EXPECT_EQ(Index64(1525), sdf->tree().activeVoxelCount());
            else        EXPECT_EQ(Index64(998), sdf->tree().activeVoxelCount());

            auto test1 = DynamicPtrCast<Int64Grid>(grids[1]);
            EXPECT_TRUE(test1 && test1->getName() == "test1");
            EXPECT_TRUE(sdf->transform() == test1->transform());
            EXPECT_EQ(GRID_UNKNOWN, test1->getGridClass());
            EXPECT_EQ(zeroVal<int64_t>(), test1->background());

            auto test2 = DynamicPtrCast<DoubleGrid>(grids[2]);
            EXPECT_TRUE(test2 && test2->getName() == "test2");
            EXPECT_TRUE(sdf->transform() == test2->transform());
            EXPECT_EQ(GRID_UNKNOWN, test2->getGridClass());
            EXPECT_EQ(zeroVal<double>(), test2->background());

            EXPECT_TRUE(sdf->tree().hasSameTopology(test1->tree()));
            EXPECT_TRUE(sdf->tree().hasSameTopology(test2->tree()));

            // check atributes that have been transfered are correct
            for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
                const Coord ijk = iter.getCoord();
                const Vec3d ws = sdf->transform().indexToWorld(ijk);
                // get closest dist from all points including rad
                float length = std::numeric_limits<float>::max();
                size_t idx = 0;
                for (size_t i = 0; i < positions.size(); ++i) {
                    float min = float((ws-positions[i]).length()) - (rads[i] * radscale);
                    if (min < length) {
                        idx = i;
                        length = min;
                    }
                }

                EXPECT_TRUE(test1->tree().isValueOn(ijk));
                EXPECT_TRUE(test2->tree().isValueOn(ijk));
                EXPECT_EQ(test1data[idx], test1->tree().getValue(ijk));
                EXPECT_EQ(test2data[idx], test2->tree().getValue(ijk));
            }

            tools::foreach(test1->cbeginValueOff(), [&sdf](auto iter) {
                EXPECT_TRUE(!sdf->tree().isValueOn(iter.getCoord()));
                EXPECT_EQ(zeroVal<int64_t>(), *iter);
            });
            tools::foreach(test2->cbeginValueOff(), [&sdf](auto iter) {
                EXPECT_TRUE(!sdf->tree().isValueOn(iter.getCoord()));
                EXPECT_EQ(zeroVal<double>(), *iter);
            });
        }
    }
}


TEST_F(TestPointRasterizeSDF, testEllipsXforms)
{
    // Test single point which is treated as an ellips with different transforms

    const math::Vec3<double> center(-1.2, 3.4f,-5.6f);
    const math::Vec3<float> stretch(0.3f, 0.6f, 1.8f);

    auto points1 = PointBuilder({center})
        .voxelsize(0.2)
        .attribute(stretch, "radius")
        .attribute(center, "pws")
        .get();
    auto points2 = PointBuilder({center})
        .voxelsize(0.2)
        .attribute(stretch, "radius")
        .attribute(center, "pws")
        .get();
    // unform radius, this one has the stretch embed into the xform
    auto points3 = PointBuilder({center})
        .voxelsize(0.2)
        .attribute(center, "pws")
        .get();

    points::EllipsoidSettings<> s;
    s.radius = "radius";
    s.xform = "xform";
    s.pws = "pws";

    // Design an ellips that is squashed in XYZ and then rotated
    math::Quat<float> a({1,0,0}, 20);
    math::Quat<float> b({0,1,0}, 45);
    math::Quat<float> c({0,0,1}, 66);
    math::Quat<float> qrot = a * b * c;
    math::Mat3<float> rot(qrot);

    // First grid gets a rotation
    points::appendAttribute<math::Mat3<float>>(points1->tree(), "xform", rot);
    // Second gets rotation as a quaternion
    points::appendAttribute<math::Quat<float>>(points2->tree(), "xform", qrot);
    // Third gets a combined xform
    points::appendAttribute<math::Mat3<float>>(points3->tree(), "xform", rot * math::scale<math::Mat3s>(stretch));

    auto grids1 = points::rasterizeSdf(*points1, s);
    auto grids2 = points::rasterizeSdf(*points2, s);
    s.radius = ""; // no radius for this one
    auto grids3 = points::rasterizeSdf(*points3, s);

    FloatGrid::Ptr sdf1 = StaticPtrCast<FloatGrid>(grids1.front());
    FloatGrid::Ptr sdf2 = StaticPtrCast<FloatGrid>(grids2.front());
    FloatGrid::Ptr sdf3 = StaticPtrCast<FloatGrid>(grids3.front());
    EXPECT_TRUE(sdf1);
    EXPECT_TRUE(sdf2);
    EXPECT_TRUE(sdf3);
    EXPECT_TRUE(sdf1->transform() == points1->transform());
    EXPECT_TRUE(sdf2->transform() == points2->transform());
    EXPECT_TRUE(sdf3->transform() == points3->transform());
    EXPECT_EQ(GRID_LEVEL_SET, sdf1->getGridClass());
    EXPECT_EQ(GRID_LEVEL_SET, sdf2->getGridClass());
    EXPECT_EQ(GRID_LEVEL_SET, sdf3->getGridClass());
    EXPECT_EQ(float(s.halfband * points1->voxelSize()[0]), sdf1->background());
    EXPECT_EQ(float(s.halfband * points2->voxelSize()[0]), sdf2->background());
    EXPECT_EQ(float(s.halfband * points3->voxelSize()[0]), sdf3->background());

    EXPECT_EQ(Index32(20), sdf1->tree().leafCount());
    EXPECT_EQ(Index64(0), sdf1->tree().activeTileCount());
    EXPECT_EQ(Index64(1921), sdf1->tree().activeVoxelCount());

    EXPECT_TRUE(sdf1->tree().hasSameTopology(sdf2->tree()));
    EXPECT_TRUE(sdf1->tree().hasSameTopology(sdf3->tree()));

    // All grid should match
    for (auto iter = sdf1->cbeginValueAll(); iter; ++iter)
    {
        auto val1 = *iter;
        auto val2 = sdf2->tree().getValue(iter.getCoord());
        auto val3 = sdf3->tree().getValue(iter.getCoord());
        ASSERT_NEAR(val1, val2, 1e-6f) << iter.getCoord();
        ASSERT_NEAR(val1, val3, 1e-6f) << iter.getCoord();
    }
}
