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
struct FixedSpheres
{
    FixedSpheres(const FilterT& f = FilterT()) : filter(f) {}

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
struct VariableSpheres : public FixedSpheres<FilterT>
{
    VariableSpheres(const FilterT& f = FilterT()) : FixedSpheres<FilterT>(f) {}

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
        FixedSpheres<points::NullFilter> s;
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
        FixedSpheres<points::NullFilter> s;

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
        FixedSpheres<points::NullFilter> s;
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
        FixedSpheres<points::GroupFilter> s(filter);
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


TEST_F(TestPointRasterizeSDF, testRasterizeVariableSpheres)
{
    // First few tests check that the results are fp equivalent to fixed spheres

    // Test no points
    {
        VariableSpheres<points::NullFilter> s;
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
        VariableSpheres<points::NullFilter> s;
        float radius = 0.2f;

        // small radius, small voxel size
        s.halfband = 3;
        s.points = PointBuilder({ Vec3f(0) }).voxelsize(0.1).attribute(radius, "pscale").get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface();
        FloatGrid::Ptr comp = s.FixedSpheres<points::NullFilter>::surface(radius);
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
        comp = s.FixedSpheres<points::NullFilter>::surface(radius);
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
        comp = s.FixedSpheres<points::NullFilter>::surface(radius);
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
        VariableSpheres<points::NullFilter> s;
        float radius = 0.2f;
        auto positions = getBoxPoints(/*scale*/0.0f);
        // test points overlapping all at 0,0,0 - should produce the same grid as first test
        s.halfband = 3;
        s.points = PointBuilder(positions).voxelsize(0.1).attribute(radius, "pscale").get();
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface();
        FloatGrid::Ptr comp = s.FixedSpheres<points::NullFilter>::surface(radius);
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
        comp = s.FixedSpheres<points::NullFilter>::surface(radius);
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
        comp = s.FixedSpheres<points::NullFilter>::surface(radius);
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
        VariableSpheres<points::GroupFilter> s(filter);
        s.halfband = 3;
        s.points = points;
        s.transform = nullptr;

        FloatGrid::Ptr sdf = s.surface();
        FloatGrid::Ptr comp = s.FixedSpheres<points::GroupFilter>::surface(radius);
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

        VariableSpheres<points::NullFilter> s;
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
        FixedSpheres<points::NullFilter> s;
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
        FixedSpheres<points::NullFilter> s;
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
        FixedSpheres<points::GroupFilter> s(filter);
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

    // Test multiple points - 8 points at cube corner positions
    {
        FixedSpheres<points::NullFilter> s;
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
        // @todo find a way to better test these values
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
        VariableSpheres<points::NullFilter> s;
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
        VariableSpheres<points::NullFilter> s;
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
        FloatGrid::Ptr comp = s.FixedSpheres<points::NullFilter>::surfaceSmooth(0.2, search);
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
        comp = s.FixedSpheres<points::NullFilter>::surfaceSmooth(0.2, search);
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
        VariableSpheres<points::GroupFilter> s(filter);
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

        VariableSpheres<points::NullFilter> s;
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


TEST_F(TestPointRasterizeSDF, testAttrTransfer)
{
    // Test no points
    {
        float radius = 0.2f, search = 0.4f;
        FixedSpheres<points::NullFilter> s;
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
        FixedSpheres<points::NullFilter> s;

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
        VariableSpheres<points::NullFilter> s;
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
        VariableSpheres<points::NullFilter> s;

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
