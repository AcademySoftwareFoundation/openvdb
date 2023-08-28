// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/points/PointRasterizeSDF.h>
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

        EXPECT_EQ(Index32(8), sdf->tree().leafCount());
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

        EXPECT_EQ(Index32(8), sdf->tree().leafCount());
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

        EXPECT_EQ(Index32(27), sdf->tree().leafCount());
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

        EXPECT_EQ(Index32(8), sdf->tree().leafCount());
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

        EXPECT_EQ(Index32(38), sdf->tree().leafCount());
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

        EXPECT_EQ(Index32(46), sdf->tree().leafCount());
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

        EXPECT_EQ(Index32(17), sdf->tree().leafCount()); // less leaf nodes, active points are on a single face
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

        EXPECT_EQ(Index32(8), sdf->tree().leafCount()); // less leaf nodes, active points are on a single face
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

        EXPECT_EQ(Index32(44), sdf->tree().leafCount());
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
        EXPECT_EQ(Index32(64), sdf->tree().leafCount());
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
    // Test no points
    {
        points::EllipsoidSettings<> s;
        s.radiusScale = 0.2f;
        s.sphereScale = 1.0f;
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
    // identicle results to SphereSettings<> (as long as the sphereScale is 1)
    {
        points::EllipsoidSettings<> s;
        s.radiusScale = 0.2f;
        s.sphereScale = 1.0f;
        s.halfband = 3;
        s.transform = nullptr;

        /// 1) test with a single point but that is not in the ellips ellipses group
        auto points = PointBuilder({Vec3f(0)})
            .voxelsize(0.1)
            .group({0}, s.pca.ellipses) // surface as a sphere
            .attribute(points::PcaAttributes::StretchT(1.0), s.pca.stretch)
            .attribute(points::PcaAttributes::RotationT::identity(), s.pca.rotation)
            .attribute(points::PcaAttributes::PosWsT(0.0), s.pca.positionWS)
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
            length -= float(s.radiusScale); // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        /// 2) test with a single point that is marked as an ellips, but has an identity
        ///    transformation matrix (and so is essentially a sphere)
        points = PointBuilder({Vec3f(0)})
            .voxelsize(0.1)
            .group({1}, s.pca.ellipses) // surface as an ellips
            .attribute(points::PcaAttributes::StretchT(1.0), s.pca.stretch)
            .attribute(points::PcaAttributes::RotationT::identity(), s.pca.rotation)
            .attribute(points::PcaAttributes::PosWsT(0.0), s.pca.positionWS)
            .get();

        grids = points::rasterizeSdf(*points, s);
        sdf = StaticPtrCast<FloatGrid>(grids.front());
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
            length -= float(s.radiusScale); // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // should only have exterior background
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            EXPECT_EQ(sdf->background(), *iter);
        }

        /// 3) larger radius, larger voxel size, with sphere scale
        s.radiusScale = 1.3f;
        s.sphereScale = 1.6f;
        points = PointBuilder({Vec3f(0)})
            .voxelsize(0.5)
            .group({0}, s.pca.ellipses) // surface as a sphere
            .attribute(points::PcaAttributes::StretchT(1.0), s.pca.stretch)
            .attribute(points::PcaAttributes::RotationT::identity(), s.pca.rotation)
            .attribute(points::PcaAttributes::PosWsT(0.0), s.pca.positionWS)
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
            length -= float(s.radiusScale * s.sphereScale); // account for radius and sphere scale
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord());
            float length = float(ws.length()); // dist to center
            // if length is <= the (rad - halfbandws), voxel is inside the surface
            const bool interior = (length <= ((s.radiusScale * s.sphereScale) - (s.halfband * sdf->voxelSize()[0])));
            if (interior) EXPECT_EQ(-(sdf->background()), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(7), interiorOff);
        EXPECT_EQ(size_t(297441), exteriorOff);

        /// 4) offset position, different transform, larger half band
        s.sphereScale = 1.0f;
        s.radiusScale = 2.0f;
        s.halfband = 4;
        s.transform = math::Transform::createLinearTransform(0.3);

        const Vec3f center(-1.2f, 3.4f,-5.6f);
        points = PointBuilder({Vec3f(center)})
            .voxelsize(0.1)
            .group({0}, s.pca.ellipses) // surface as a sphere
            .attribute(points::PcaAttributes::StretchT(1.0), s.pca.stretch)
            .attribute(points::PcaAttributes::RotationT::identity(), s.pca.rotation)
            .attribute(points::PcaAttributes::PosWsT(center), s.pca.positionWS)
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
            length -= float(s.radiusScale); // account for radius
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values
        interiorOff = 0;
        exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d ws = sdf->transform().indexToWorld(iter.getCoord()) - center;
            float length = float(ws.length()); // dist to center
            // if length is <= the (rad - halfbandws), voxel is inside the surface
            const bool interior = (length <= (s.radiusScale - (s.halfband * s.transform->voxelSize()[0])));
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
        s.radiusScale = 0.8f;
        s.sphereScale = 1.0f; // should have no effect
        s.halfband = 3;
        s.transform = nullptr;

        // Design an ellips that is squashed in XYZ and then rotated
        const points::PcaAttributes::StretchT stretch(1.0f, 0.2f, 1.0f);
        points::PcaAttributes::RotationT rot;
         // 45 degree rotation about Y (we only squash in Y so this should be a no-op)
        rot.setToRotation({0,1,0}, 45);
        // The transform that defines how we go from each voxel back to the source point
        const math::Mat3s inv = rot.timesDiagonal(1.0 / stretch) * rot.transpose();
        //

        /// 1) test with a single ellips with Y stretch/rotation
        auto points = PointBuilder({Vec3f(0)})
            .voxelsize(0.1)
            .group({1}, s.pca.ellipses) // surface as an ellips
            .attribute(stretch, s.pca.stretch)
            .attribute(rot, s.pca.rotation)
            .attribute(points::PcaAttributes::PosWsT(0.0), s.pca.positionWS)
            .get();

        auto grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(24), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(1018), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d is = inv * iter.getCoord().asVec3d();
            float length = float(is.length()); // dist to center in index space
            length -= float(s.radiusScale / sdf->voxelSize()[0]); // account for radius
            length *= float(sdf->voxelSize()[0]); // dist to center in world space
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values (because we're also squashing the halfband we
        // just compre the overal length to 0.0)
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d is = inv * iter.getCoord().asVec3d();
            float length = float(is.length()); // dist to center in index space
            length -= float(s.radiusScale / sdf->voxelSize()[0]); // account for radius
            length *= float(sdf->voxelSize()[0]); // dist to center in world space
            const bool interior = (length <= 0.0);
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(83), interiorOff);
        EXPECT_EQ(size_t(306067), exteriorOff);

        // Run again with a different Y rotation, should be the same as the above
        // as we're only squashing in Y. Also test sphereScale has no impact
        s.sphereScale = 10.0f; // should have no effect
        rot.setToRotation({0,1,0}, -88);
        points = PointBuilder({Vec3f(0)})
            .voxelsize(0.1)
            .group({1}, s.pca.ellipses) // surface as an ellips
            .attribute(stretch, s.pca.stretch)
            .attribute(rot, s.pca.rotation)
            .attribute(points::PcaAttributes::PosWsT(0.0), s.pca.positionWS)
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
        s.sphereScale = 3.0f; // should have no effect
        s.halfband = 5;

        // Design an ellips that is squashed in XYZ and then rotated
        const points::PcaAttributes::StretchT stretch(0.3f, 0.6f, 1.8f);
        points::PcaAttributes::RotationT a,b,c,rot;
        a.setToRotation({1,0,0}, 20);
        b.setToRotation({0,1,0}, 45);
        c.setToRotation({0,0,1}, 66);
        rot = a * b * c;

        // The transform that defines how we go from each voxel back to the source point
        const math::Mat3s inv = rot.timesDiagonal(1.0 / stretch) * rot.transpose();
        //

        /// 1) test with a single ellips with Y stretch/rotation
        const Vec3f center(-1.2f, 3.4f,-5.6f);
        auto points = PointBuilder({center})
            .voxelsize(0.2)
            .group({1}, s.pca.ellipses) // surface as an ellips
            .attribute(stretch, s.pca.stretch)
            .attribute(rot, s.pca.rotation)
            .attribute(points::PcaAttributes::PosWsT(center), s.pca.positionWS)
            .get();

        auto grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(20), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(1337), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const Vec3d is = inv * (iter.getCoord().asVec3d() - sdf->transform().worldToIndex(center));
            float length = float(is.length()); // dist to center in index space
            length -= float(s.radiusScale / sdf->voxelSize()[0]); // account for radius
            length *= float(sdf->voxelSize()[0]); // dist to center in world space
            EXPECT_NEAR(length, *iter, 1e-6f);
        }

        // check off values (because we're also squashing the halfband we
        // just compre the overal length to 0.0)
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            const Vec3d is = inv * (iter.getCoord().asVec3d() - sdf->transform().worldToIndex(center));
            float length = float(is.length()); // dist to center in index space
            length -= float(s.radiusScale / sdf->voxelSize()[0]); // account for radius
            length *= float(sdf->voxelSize()[0]); // dist to center in world space
            const bool interior = (length <= 0.0);
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(0), interiorOff);
        EXPECT_EQ(size_t(82609), exteriorOff);
    }

    // Test multiple ellips and spheres with different transformations and radii
    {
        points::EllipsoidSettings<> s;
        s.radiusScale = 2.0f;
        s.sphereScale = 0.8f;
        s.halfband = 5;
        s.transform = nullptr;

        const auto positions = getBoxPoints();
        const std::vector<Vec3d> positionsVec3d(positions.begin(), positions.end());
        const std::vector<short> ellips {0,1,0,1, 0,1,1,1};
        const std::vector<points::PcaAttributes::StretchT> stretches {
            {0.0f, 0.0f, 0.0f}, // sphere (should be ignored)
            {1.0f, 1.0f, 1.0f}, // ellips
            {5.0f, 5.0f, 5.0f}, // sphere  (should be ignored)
            {2.0f, 0.3f, 1.5f}, // ellips

            {1.0f, 1.0f, 1.0f}, // sphere  (should be ignored)
            {1.1f, 0.8f, 1.0f}, // ellips
            {0.8f, 4.0f, 1.5f}, // ellips
            {0.4f, 1.1f, 1.0f}  // ellips
        };
        const std::vector<points::PcaAttributes::RotationT> rotations {
            Mat3s::zero(),                      // sphere (should be ignored)
            Mat3s::identity(),
            math::rotation<Mat3s>({0,0,1}, 45), // sphere (should be ignored)
            Mat3s::identity(),

            math::rotation<Mat3s>({1,0,0}, -20), // sphere (should be ignored)
            math::rotation<Mat3s>({0,1,0},   5),
            math::rotation<Mat3s>({0,0,1}, 143),
            math::rotation<Mat3s>({1,0,0},  49)
        };

        /// 1) test with uniform radius
        auto points = PointBuilder(positions)
            .voxelsize(0.3)
            .group(ellips, s.pca.ellipses) // surface as an ellips
            .attribute(stretches, s.pca.stretch)
            .attribute(rotations, s.pca.rotation)
            .attribute(positionsVec3d, s.pca.positionWS)
            .get();

        auto grids = points::rasterizeSdf(*points, s);
        FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(147), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(33252), sdf->tree().activeVoxelCount());

        // Small lambda that finds the cloest length to a point that could
        // either be an ellips of sphere, for a given voxel
        const auto getClosestLength = [&](const Coord& ijk, const std::vector<float>* radii = nullptr)
        {
            double length = std::numeric_limits<double>::max();
            size_t idx = 0;
            for (auto& pos : positionsVec3d)
            {
                math::Mat3s inv = math::Mat3s::identity();
                double scale = s.radiusScale / sdf->voxelSize()[0];
                if (radii) scale *= double((*radii)[idx]);

                if (ellips[idx]) {
                    inv = rotations[idx].timesDiagonal(1.0 / stretches[idx]) *
                        rotations[idx].transpose();
                }
                else {
                    scale *= s.sphereScale;
                }

                const Vec3d is = inv * (ijk.asVec3d() - sdf->transform().worldToIndex(pos));
                length = std::min(length, (is.length() - scale));
                ++idx;
            }

            length *= (sdf->voxelSize()[0]); // dist to center in world space
            return float(length);
        };

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            // get closest dist from all points
            const float length = getClosestLength(iter.getCoord());
            EXPECT_NEAR(length, *iter, 1e-6f) << iter.getCoord();
        }

        // check off values (because we're also squashing the halfband we
        // just compre the overal length to 0.0)
        size_t interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            // get closest dist from all points
            const float length = getClosestLength(iter.getCoord());
            const bool interior = (length <= 0.0);
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(147), interiorOff);
        EXPECT_EQ(size_t(336622), exteriorOff);

        /// 2) test with varying radius

        const std::vector<float> radii {
            1.0f, 0.0f, 2.0f, 1.1f,
            0.2f, 0.5f, 0.8f, 3.0f
        };

        s.radiusScale = 1.2;
        s.sphereScale = 0.3;
        s.radius = "pscale";
        s.halfband = 1;

        points = PointBuilder(positions)
            .voxelsize(0.3)
            .group(ellips, s.pca.ellipses) // surface as an ellips
            .attribute(radii, s.radius)
            .attribute(stretches, s.pca.stretch)
            .attribute(rotations, s.pca.rotation)
            .attribute(positionsVec3d, s.pca.positionWS)
            .get();

        grids = points::rasterizeSdf(*points, s);
        sdf = StaticPtrCast<FloatGrid>(grids.front());
        EXPECT_TRUE(sdf);
        EXPECT_TRUE(sdf->transform() == points->transform());
        EXPECT_EQ(GRID_LEVEL_SET, sdf->getGridClass());
        EXPECT_EQ(float(s.halfband * points->voxelSize()[0]), sdf->background());

        EXPECT_EQ(Index32(39), sdf->tree().leafCount());
        EXPECT_EQ(Index64(0), sdf->tree().activeTileCount());
        EXPECT_EQ(Index64(2613), sdf->tree().activeVoxelCount());

        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            // get closest dist from all points
            const float length = getClosestLength(iter.getCoord(), &radii);
            EXPECT_NEAR(length, *iter, 1e-6f) << iter.getCoord();
        }

        // check off values (because we're also squashing the halfband we
        // just compre the overal length to 0.0)
        interiorOff = 0, exteriorOff = 0;
        for (auto iter = sdf->cbeginValueOff(); iter; ++iter) {
            // get closest dist from all points
            const float length = getClosestLength(iter.getCoord(), &radii);
            const bool interior = (length <= 0.0);
            if (interior) EXPECT_EQ(-sdf->background(), *iter);
            else          EXPECT_EQ(sdf->background(), *iter);
            interior ? ++interiorOff : ++exteriorOff;
        }

        EXPECT_EQ(size_t(2137), interiorOff);
        EXPECT_EQ(size_t(310083), exteriorOff);
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
