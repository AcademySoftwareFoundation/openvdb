// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>

#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Mask.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointRasterizeFrustum.h>

// enable this flag to reduce compilation time by only unit testing float rasterization
// #define ONLY_RASTER_FLOAT

// enable this flag to perform some simple benchmarking using CpuTimer
// #define PROFILE

#ifdef PROFILE
#include <openvdb/util/CpuTimer.h>
#endif

#include <gtest/gtest.h>

using namespace openvdb;
using namespace openvdb::points;

class TestPointRasterizeFrustum: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointRasterizeFrustum

TEST_F(TestPointRasterizeFrustum, testScaleByVoxelVolume)
{
    const double tolerance = 1.0e-6;

    // four our of five positions live in (0,0,0) voxels

    std::vector<Vec3s> positions =  {
                                        {0.0f, -0.2f, 0.0f},
                                        {0.0f, 0.0f, 0.0f},
                                        {0.12f, 0.12f, 0.12f},
                                        {100.5f, 98.6f, 103.0f}, /*not in (0,0,0)*/
                                        {-0.24f, 0.15f, 0.24f},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    // leaf containing (0,0,0) voxel lives at (-0.25 => 3.75) in each axis world space for 0.5 voxel size

    const double voxelSize = 0.5;
    math::Transform::Ptr transform(
        math::Transform::createLinearTransform(voxelSize));
    const double voxelVolume = transform->voxelSize(Vec3d(0,0,0)).product();
    EXPECT_EQ(voxelVolume, (voxelSize * voxelSize * voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
        createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    auto& tree = points->tree();

    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;
    using Mask = FrustumRasterizerMask;

    Settings settings(*transform);
    settings.velocityAttribute = "velocityAttr";
    settings.scaleByVoxelVolume = true;
    settings.threaded = false;
    settings.threshold = 1e-6f;

    Rasterizer rasterizer(settings);

    // add points to rasterizer

    rasterizer.addPoints(points, /*stream=*/false);

    // accumulate density

    auto density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE);

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *transform);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    auto iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/4.0f / static_cast<float>(voxelVolume), *iter);

    density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE);
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(/*value=*/4.0f / static_cast<float>(voxelVolume), *iter);

    // accumulate scaled density

    float scale = 13.9f;
    density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE, false, scale);

    EXPECT_TRUE(density);
    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/(4.0f * scale) / static_cast<float>(voxelVolume), *iter);

    // average density

    density = rasterizer.rasterizeUniformDensity(RasterMode::AVERAGE);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/8.0f, *iter);

    // maximum density

    density = rasterizer.rasterizeUniformDensity(RasterMode::MAXIMUM);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/1.0f / static_cast<float>(voxelVolume), *iter);

    // add density attribute

    appendAttribute<float>(tree, "densityAttr");

    std::vector<float> densities{2.0f, 3.1f, 8.9f, 6.7f, 4.2f};
    PointAttributeVector<float> densityWrapper(densities);
    populateAttribute(tree, pointIndexGrid->tree(), "densityAttr", densityWrapper);

    // accumulate density

    float sum = densities[0] + densities[1] + densities[2] + densities[4];

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE);

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *transform);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/sum / static_cast<float>(voxelVolume), *iter);

    // accumulate scaled density

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE, false, scale);
    EXPECT_TRUE(density);
    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    float scaledSum = scale * densities[0] + scale * densities[1] +
        scale * densities[2] + scale * densities[4];
    EXPECT_NEAR(scaledSum / static_cast<float>(voxelVolume), *iter, tolerance);

    // add temperature attribute (an arbitrary float attribute)

    appendAttribute<float>(tree, "temperatureAttr");

    std::vector<float> temperatures{4.2f, 6.7f, 8.9f, 3.1f, 2.0f};
    PointAttributeVector<float> temperatureWrapper(temperatures);
    populateAttribute(tree, pointIndexGrid->tree(), "temperatureAttr", temperatureWrapper);

    // accumulate temperature

    sum = temperatures[0] + temperatures[1] + temperatures[2] + temperatures[4];

    auto temperatureBase = rasterizer.rasterizeAttribute("temperatureAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(temperatureBase);
    auto temperature = GridBase::grid<FloatGrid>(temperatureBase);
    EXPECT_TRUE(temperature);
    EXPECT_EQ(openvdb::Name("temperatureAttr"), temperature->getName());
    EXPECT_EQ(temperature->transform(), *transform);

    EXPECT_EQ(Index64(2), temperature->tree().activeVoxelCount());
    iter = temperature->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/sum / static_cast<float>(voxelVolume), *iter);

#ifndef ONLY_RASTER_FLOAT
    // add velocity attribute (an arbitrary vec3s attribute)

    appendAttribute<Vec3s>(tree, "velocityAttr");

    std::vector<Vec3s> velocities = {
                                        {0.0f, 10.0f, 0.0f},
                                        {1.0f, 2.0f, 3.0f},
                                        {-3.0f, -2.0f, -1.0f},
                                        {4.0f, 5.0f, 6.0f},
                                        {4.2f, 6.7f, 8.9f},
                                    };
    PointAttributeVector<Vec3s> velocityWrapper(velocities);
    populateAttribute(tree, pointIndexGrid->tree(), "velocityAttr", velocityWrapper);

    // accumulate velocity

    auto sumV = velocities[0] + velocities[1] + velocities[2] + velocities[4];

    auto velocityBase = rasterizer.rasterizeAttribute("velocityAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(velocityBase);
    auto velocity = GridBase::grid<Vec3fGrid>(velocityBase);
    EXPECT_EQ(openvdb::Name("velocityAttr"), velocity->getName());
    EXPECT_EQ(velocity->transform(), *transform);

    EXPECT_EQ(Index64(2), velocity->tree().activeVoxelCount());
    auto iterV = velocity->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iterV.getCoord());
    EXPECT_EQ(/*value=*/sumV / static_cast<float>(voxelVolume), *iterV);

    // accumulate scaled velocity

    velocityBase = rasterizer.rasterizeAttribute("velocityAttr", RasterMode::ACCUMULATE, false, scale);
    EXPECT_TRUE(velocityBase);
    velocity = GridBase::grid<Vec3fGrid>(velocityBase);
    EXPECT_EQ(Index64(2), velocity->tree().activeVoxelCount());
    iterV = velocity->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iterV.getCoord());
    auto scaledSumV = (scale * velocities[0] + scale * velocities[1] +
        scale * velocities[2] + scale * velocities[4]) / voxelVolume;
    EXPECT_NEAR(scaledSumV[0], (*iterV)[0], tolerance);
    EXPECT_NEAR(scaledSumV[1], (*iterV)[1], tolerance);
    EXPECT_NEAR(scaledSumV[2], (*iterV)[2], tolerance);
#endif

    ////////////////////////////

    // manually build frustum transform using camera API

    math::NonlinearFrustumMap frustumMap(
        /*position*/Vec3d(0, 0, -10), /*direction*/Vec3d(0, 0, 1), /*up*/Vec3d(0, 1, 0),
        /*aspect*/1.5, /*znear*/0.05, /*depth*/20, /*xcount*/50, /*zcount*/50);

    math::Transform::Ptr frustum = math::Transform(frustumMap.copy()).copy();

    Mask mask(*frustum, nullptr, BBoxd(), false);

    settings.transform = frustum->copy();

    Rasterizer frustumRasterizer(settings, mask);
    frustumRasterizer.addPoints(points, /*stream=*/false);

    // accumulate density into frustum grid

    // point 0 is rasterized into (25, 16, 25)
    // points 1, 2, 4 are all rasterized into (25, 17, 25)
    // point 3 is rasterized into (26, 17, 282)

    density = frustumRasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE);
    EXPECT_EQ(openvdb::Name("density"), density->getName());

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *frustum);

    EXPECT_EQ(Index64(3), density->activeVoxelCount());

    { // point 0
        Coord xyz(25, 16, 25);
        const float sum = densities[0];
        const float voxelVolume = static_cast<float>(frustum->voxelSize(xyz.asVec3d()).product());
        EXPECT_NEAR(sum / voxelVolume, density->tree().getValue(xyz), 1e-6);
    }

    { // point 1, 2, 4
        Coord xyz(25, 17, 25);
        const float sum = densities[1] + densities[2] + densities[4];
        const float voxelVolume = static_cast<float>(frustum->voxelSize(xyz.asVec3d()).product());
        EXPECT_NEAR(sum / voxelVolume, density->tree().getValue(xyz), 1e-6);
    }

    { // point 3
        Coord xyz(26, 17, 282);
        const float sum = densities[3];
        const float voxelVolume = static_cast<float>(frustum->voxelSize(xyz.asVec3d()).product());
        EXPECT_NEAR(sum / voxelVolume, density->tree().getValue(xyz), 1e-6);
    }

    // use a clipped frustum grid (point 3 is clipped)

    Mask clippedMask(*frustum);

    Rasterizer clippedFrustumRasterizer(settings, clippedMask);
    clippedFrustumRasterizer.addPoints(points, /*stream=*/false);

    auto clippedDensity = clippedFrustumRasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE);

    EXPECT_EQ(Index64(2), clippedDensity->tree().activeVoxelCount());

    { // point 0
        Coord xyz(25, 16, 25);
        EXPECT_NEAR(density->tree().getValue(xyz),
            clippedDensity->tree().getValue(xyz), 1e-6);
    }

    { // point 1, 2, 4
        Coord xyz(25, 17, 25);
        EXPECT_NEAR(density->tree().getValue(xyz),
            clippedDensity->tree().getValue(xyz), 1e-6);
    }

    { // point 3 (outside of frustum)
        Coord xyz(26, 17, 282);
        EXPECT_NEAR(0.0, clippedDensity->tree().getValue(xyz), 1e-6);
    }
}


TEST_F(TestPointRasterizeFrustum, testPointRasterization)
{
    const double tolerance = 1.0e-5;

    // four our of five positions live in (0,0,0) voxels

    std::vector<Vec3s> positions =  {
                                        {0.0f, -0.2f, 0.0f},
                                        {0.0f, 0.0f, 0.0f},
                                        {0.12f, 0.12f, 0.12f},
                                        {100.5f, 98.6f, 103.0f}, /*not in (0,0,0)*/
                                        {-0.24f, 0.15f, 0.24f},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    // leaf containing (0,0,0) voxel lives at (-0.25 => 3.75) in each axis world space for 0.5 voxel size

    const double voxelSize = 0.5;
    math::Transform::Ptr transform(
        math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
        createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    auto& tree = points->tree();

    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;
    using Mask = FrustumRasterizerMask;

    Settings settings(*transform);
    settings.velocityAttribute = "velocityAttr";
    settings.velocityMotionBlur = false;

    Rasterizer rasterizer(settings);

    // add points to rasterizer

    rasterizer.addPoints(points, /*stream=*/false);

    // accumulate density

    auto density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE);

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *transform);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    auto iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/4.0f, *iter);

    density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE);
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(/*value=*/4.0f, *iter);

    // accumulate scaled density

    float scale = 13.9f;
    density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE, false, scale);

    EXPECT_TRUE(density);
    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/(4.0f * scale), *iter);

    // average density

    density = rasterizer.rasterizeUniformDensity(RasterMode::AVERAGE);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/1.0f, *iter);

    // maximum density

    density = rasterizer.rasterizeUniformDensity(RasterMode::MAXIMUM);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/1.0f, *iter);

    // add density attribute

    appendAttribute<float>(tree, "densityAttr");

    std::vector<float> densities{2.0f, 3.1f, 8.9f, 6.7f, 4.2f};
    PointAttributeVector<float> densityWrapper(densities);
    populateAttribute(tree, pointIndexGrid->tree(), "densityAttr", densityWrapper);

    // accumulate density

    float sum = densities[0] + densities[1] + densities[2] + densities[4];

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE);

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *transform);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/sum, *iter);

    // accumulate scaled density

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE, false, scale);
    EXPECT_TRUE(density);
    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    float scaledSum = scale * densities[0] + scale * densities[1] +
        scale * densities[2] + scale * densities[4];
    EXPECT_NEAR(scaledSum, *iter, tolerance);

    // average density

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::AVERAGE);

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *transform);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/sum / /*weight=*/4, *iter);

    // maximum density

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::MAXIMUM);

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *transform);

    EXPECT_EQ(Index64(2), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/(densities[2]), *iter);

    // add temperature attribute (an arbitrary float attribute)

    appendAttribute<float>(tree, "temperatureAttr");

    std::vector<float> temperatures{4.2f, 6.7f, 8.9f, 3.1f, 2.0f};
    PointAttributeVector<float> temperatureWrapper(temperatures);
    populateAttribute(tree, pointIndexGrid->tree(), "temperatureAttr", temperatureWrapper);

    // accumulate temperature

    sum = temperatures[0] + temperatures[1] + temperatures[2] + temperatures[4];

    auto temperatureBase = rasterizer.rasterizeAttribute("temperatureAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(temperatureBase);
    auto temperature = GridBase::grid<FloatGrid>(temperatureBase);
    EXPECT_TRUE(temperature);
    EXPECT_EQ(openvdb::Name("temperatureAttr"), temperature->getName());
    EXPECT_EQ(temperature->transform(), *transform);

    EXPECT_EQ(Index64(2), temperature->tree().activeVoxelCount());
    iter = temperature->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/sum, *iter);

    // explicitly specify types in rasterization

    auto temperatureF = rasterizer.rasterizeAttribute<FloatGrid, float>("temperatureAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(temperatureF);

    EXPECT_EQ(Index64(2), temperatureF->tree().activeVoxelCount());
    iter = temperatureF->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/sum, *iter);

    // rasterize float attribute into double grid

    auto temperatureD = rasterizer.rasterizeAttribute<DoubleGrid, float>("temperatureAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(temperatureD);

    EXPECT_EQ(Index64(2), temperatureD->tree().activeVoxelCount());
    auto iterD = temperatureD->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iterD.getCoord());
    EXPECT_EQ(/*value=*/double(sum), *iterD);

    // rasterize float attribute into bool grid

    auto boolGrid = rasterizer.rasterizeAttribute<BoolGrid, float>("temperatureAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(boolGrid);

    EXPECT_EQ(Index64(2), boolGrid->tree().activeVoxelCount());
    EXPECT_TRUE(boolGrid->tree().getValue(Coord(0,0,0)));
    EXPECT_TRUE(boolGrid->tree().isValueOn(Coord(0,0,0)));

    // average temperature

    temperatureBase = rasterizer.rasterizeAttribute("temperatureAttr", RasterMode::AVERAGE);
    EXPECT_TRUE(temperatureBase);
    temperature = GridBase::grid<FloatGrid>(temperatureBase);
    EXPECT_TRUE(temperature);
    EXPECT_EQ(openvdb::Name("temperatureAttr"), temperature->getName());
    EXPECT_EQ(temperature->transform(), *transform);

    EXPECT_EQ(Index64(2), temperature->tree().activeVoxelCount());
    iter = temperature->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/sum / /*weight=*/4, *iter);

    // maximum temperature

    temperatureBase = rasterizer.rasterizeAttribute("temperatureAttr", RasterMode::MAXIMUM);
    EXPECT_TRUE(temperatureBase);
    temperature = GridBase::grid<FloatGrid>(temperatureBase);
    EXPECT_TRUE(temperature);
    EXPECT_EQ(openvdb::Name("temperatureAttr"), temperature->getName());
    EXPECT_EQ(temperature->transform(), *transform);

    EXPECT_EQ(Index64(2), temperature->tree().activeVoxelCount());
    iter = temperature->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(/*value=*/(temperatures[2]), *iter);

#ifndef ONLY_RASTER_FLOAT
    // add arbitrary float[3] attribute and verify it rasterizes all components
    // into a vec3s grid

    appendAttribute<float>(tree, "stridedAttr", /*uniformValue=*/0.0f, /*stride=*/3);

    std::vector<float> stridedValues = {
                                            1.0f, 2.0f, 3.0f,
                                            1.0f, 2.0f, 3.0f,
                                            1.0f, 2.0f, 3.0f,
                                            1.0f, 2.0f, 3.0f,
                                            1.0f, 2.0f, 3.0f
                                        };
    PointAttributeVector<float> stridedWrapper(stridedValues, /*stride=*/3);
    populateAttribute(tree, pointIndexGrid->tree(), "stridedAttr", stridedWrapper, /*stride=*/3);

    auto stridedBase = rasterizer.rasterizeAttribute("stridedAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(stridedBase);
    auto strided = GridBase::grid<Vec3fGrid>(stridedBase);
    EXPECT_TRUE(strided);
    auto iterS = strided->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Vec3s(4.0f, 8.0f, 12.0f), *iterS);

    // add velocity attribute (an arbitrary vec3s attribute)

    appendAttribute<Vec3s>(tree, "velocityAttr");

    std::vector<Vec3s> velocities = {
                                        {0.0f, 10.0f, 0.0f},
                                        {1.0f, 2.0f, 3.0f},
                                        {-3.0f, -2.0f, -1.0f},
                                        {4.0f, 5.0f, 6.0f},
                                        {4.2f, 6.7f, 8.9f},
                                    };
    PointAttributeVector<Vec3s> velocityWrapper(velocities);
    populateAttribute(tree, pointIndexGrid->tree(), "velocityAttr", velocityWrapper);

    // accumulate velocity

    auto sumV = velocities[0] + velocities[1] + velocities[2] + velocities[4];

    auto velocityBase = rasterizer.rasterizeAttribute("velocityAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(velocityBase);
    auto velocity = GridBase::grid<Vec3fGrid>(velocityBase);
    EXPECT_EQ(openvdb::Name("velocityAttr"), velocity->getName());
    EXPECT_EQ(velocity->transform(), *transform);

    EXPECT_EQ(Index64(2), velocity->tree().activeVoxelCount());
    auto iterV = velocity->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iterV.getCoord());
    EXPECT_EQ(/*value=*/sumV, *iterV);

    // accumulate scaled velocity

    velocityBase = rasterizer.rasterizeAttribute("velocityAttr", RasterMode::ACCUMULATE, false, scale);
    EXPECT_TRUE(velocityBase);
    velocity = GridBase::grid<Vec3fGrid>(velocityBase);
    EXPECT_EQ(Index64(2), velocity->tree().activeVoxelCount());
    iterV = velocity->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iterV.getCoord());
    auto scaledSumV = (scale * velocities[0] + scale * velocities[1] +
        scale * velocities[2] + scale * velocities[4]);
    EXPECT_NEAR(scaledSumV[0], (*iterV)[0], tolerance);
    EXPECT_NEAR(scaledSumV[1], (*iterV)[1], tolerance);
    EXPECT_NEAR(scaledSumV[2], (*iterV)[2], tolerance);

    // explicitly specify Vec3f grid

    velocity = rasterizer.rasterizeAttribute<Vec3fGrid, Vec3f>("velocityAttr", RasterMode::ACCUMULATE);
    EXPECT_TRUE(velocity);

    EXPECT_EQ(Index64(2), velocity->tree().activeVoxelCount());
    iterV = velocity->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_NEAR((sumV)[0], (*iterV)[0], tolerance);
    EXPECT_NEAR((sumV)[1], (*iterV)[1], tolerance);
    EXPECT_NEAR((sumV)[2], (*iterV)[2], tolerance);

    // explicitly specify Vec3f grid and scale

    velocity = rasterizer.rasterizeAttribute<Vec3fGrid, Vec3f>("velocityAttr", RasterMode::ACCUMULATE, false, scale);
    EXPECT_TRUE(velocity);

    EXPECT_EQ(Index64(2), velocity->tree().activeVoxelCount());
    iterV = velocity->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_NEAR(scaledSumV[0], (*iterV)[0], tolerance);
    EXPECT_NEAR(scaledSumV[1], (*iterV)[1], tolerance);
    EXPECT_NEAR(scaledSumV[2], (*iterV)[2], tolerance);

    // rasterize float attribute into double grid

    auto velocityD = rasterizer.rasterizeAttribute<Vec3dGrid, Vec3f>("velocityAttr", RasterMode::ACCUMULATE, false, scale);
    EXPECT_TRUE(velocityD);

    EXPECT_EQ(Index64(2), velocityD->tree().activeVoxelCount());
    auto iterVD = velocityD->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iterVD.getCoord());
    // note that the order of operations and the precision being used for each one is important
    // to result in the exact same value within a tolerance of 1e-6
    auto scaledSumVD = Vec3d(scale * velocities[0]) + Vec3d(scale * velocities[1]) +
        Vec3d(scale * velocities[2]) + Vec3d(scale * velocities[4]);
    EXPECT_NEAR(scaledSumVD[0], (*iterVD)[0], tolerance);
    EXPECT_NEAR(scaledSumVD[1], (*iterVD)[1], tolerance);
    EXPECT_NEAR(scaledSumVD[2], (*iterVD)[2], tolerance);

    // average velocity

    velocityBase = rasterizer.rasterizeAttribute("velocityAttr", RasterMode::AVERAGE);
    EXPECT_TRUE(velocityBase);
    velocity = GridBase::grid<Vec3fGrid>(velocityBase);
    EXPECT_TRUE(velocity);
    EXPECT_EQ(openvdb::Name("velocityAttr"), velocity->getName());
    EXPECT_EQ(velocity->transform(), *transform);

    EXPECT_EQ(Index64(2), velocity->tree().activeVoxelCount());
    iterV = velocity->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iterV.getCoord());
    // note that the order of operations and the precision being used for each one is important
    // to result in the exact same value within a tolerance of 1e-6
    auto sumVD = velocities[0] + velocities[1] + velocities[2] + velocities[4];
    EXPECT_EQ(/*value=*/(sumVD / 4.0f), *iterV);

    // maximum velocity (not supported for vector attributes)

    EXPECT_THROW(rasterizer.rasterizeAttribute("velocityAttr", RasterMode::MAXIMUM),
        ValueError);
#endif

    ////////////////////////////

    // point filtering (select one point only)

    appendGroup(tree, "test");

    std::vector<short> membership{0, 1, 0, 0, 0};

    setGroup(tree, pointIndexGrid->tree(), membership, "test");

    std::vector<Name> includeGroups{"test"};
    std::vector<Name> excludeGroups;
    MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE, false, 1.0f, filter);

    EXPECT_TRUE(density);
    EXPECT_EQ(Index64(1), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(densities[1], *iter);

    // use RasterGroups object that is resolved per-grid

    RasterGroups rasterGroups;
    rasterGroups.includeNames.push_back("test");

    density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE, false, 1.0f, rasterGroups);

    EXPECT_TRUE(density);
    EXPECT_EQ(Index64(1), density->activeVoxelCount());
    iter = density->tree().cbeginLeaf()->cbeginValueOn();
    EXPECT_EQ(Coord(0,0,0), iter.getCoord());
    EXPECT_EQ(densities[1], *iter);

    ////////////////////////////

    // manually build frustum transform using camera API

    math::NonlinearFrustumMap frustumMap(
        /*position*/Vec3d(0, 0, -10), /*direction*/Vec3d(0, 0, 1), /*up*/Vec3d(0, 1, 0),
        /*aspect*/1.5, /*znear*/0.05, /*depth*/20, /*xcount*/50, /*zcount*/50);

    math::Transform::Ptr frustum = math::Transform(frustumMap.copy()).copy();

    Mask mask(*frustum, nullptr, BBoxd(), false);

    settings.transform = frustum->copy();

    Rasterizer frustumRasterizer(settings, mask);
    frustumRasterizer.addPoints(points, /*stream=*/false);

    // accumulate density into frustum grid

    // point 0 is rasterized into (25, 16, 25)
    // points 1, 2, 4 are all rasterized into (25, 17, 25)
    // point 3 is rasterized into (26, 17, 282)

    density = frustumRasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE);
    EXPECT_EQ(openvdb::Name("density"), density->getName());

    EXPECT_TRUE(density);
    EXPECT_EQ(openvdb::Name("density"), density->getName());
    EXPECT_EQ(density->transform(), *frustum);

    EXPECT_EQ(Index64(3), density->activeVoxelCount());

    { // point 0
        Coord xyz(25, 16, 25);
        const float sum = densities[0];
        EXPECT_NEAR(sum, density->tree().getValue(xyz), 1e-6);
    }

    { // point 1, 2, 4
        Coord xyz(25, 17, 25);
        const float sum = densities[1] + densities[2] + densities[4];
        EXPECT_NEAR(sum, density->tree().getValue(xyz), 1e-6);
    }

    { // point 3
        Coord xyz(26, 17, 282);
        const float sum = densities[3];
        EXPECT_NEAR(sum, density->tree().getValue(xyz), 1e-6);
    }

    // use a clipped frustum grid (point 3 is clipped)

    Mask clipMask(*frustum, nullptr, BBoxd(), true);
    Rasterizer clippedFrustumRasterizer(settings, clipMask);
    clippedFrustumRasterizer.addPoints(points, /*stream=*/false);

    auto clippedDensity = clippedFrustumRasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE);

    EXPECT_EQ(Index64(2), clippedDensity->tree().activeVoxelCount());

    { // point 0
        Coord xyz(25, 16, 25);
        EXPECT_NEAR(density->tree().getValue(xyz),
            clippedDensity->tree().getValue(xyz), 1e-6);
    }

    { // point 1, 2, 4
        Coord xyz(25, 17, 25);
        EXPECT_NEAR(density->tree().getValue(xyz),
            clippedDensity->tree().getValue(xyz), 1e-6);
    }

    { // point 3 (outside of frustum)
        Coord xyz(26, 17, 282);
        EXPECT_NEAR(0.0, clippedDensity->tree().getValue(xyz), 1e-6);
    }
}

TEST_F(TestPointRasterizeFrustum, testSphereRasterization)
{
    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;

    { // single point that lives at (0,0.2,0)

        Vec3s center(0.0f, 0.2f, 0.0f);

        std::vector<Vec3s> positions =  {
                                            center
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        const double voxelSize = 0.5;
        math::Transform::Ptr transform(
            math::Transform::createLinearTransform(voxelSize));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
        auto& tree = points->tree();

        { // no radius
            Settings settings(*transform);
            Rasterizer rasterizer(settings);

            // add points to rasterizer

            rasterizer.addPoints(points, /*stream=*/false);

            // accumulate density

            auto density = rasterizer.rasterizeUniformDensity();

            EXPECT_EQ(Index64(1), density->tree().activeVoxelCount());
        }

        { // use radius (no pscale so default to radius of 1.0)
            Settings settings(*transform);
            settings.useRadius = true;
            Rasterizer rasterizer(settings);

            // add points to rasterizer

            rasterizer.addPoints(points, /*stream=*/false);

            // accumulate density

            auto density = rasterizer.rasterizeUniformDensity();

            EXPECT_EQ(Index64(32), density->tree().activeVoxelCount());

            for (auto leaf = density->tree().cbeginLeaf(); leaf; ++leaf) {
                for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                    // other values should be less than one
                    EXPECT_TRUE(iter.getValue() < 1.0f);
                }
            }
        }

        { // verify falloff is monotonically decreasing in one axis (with radius scaling)
            Settings settings(*transform);
            settings.useRadius = true;
            settings.radiusScale = 10.0f;
            Rasterizer rasterizer(settings);

            // add points to rasterizer

            rasterizer.addPoints(points, /*stream=*/false);

            // accumulate density

            auto density = rasterizer.rasterizeUniformDensity();

            EXPECT_EQ(Index64(33484), density->tree().activeVoxelCount());

            float value = density->tree().getValue(Coord(0,0,0));

            for (int j = 1; j < 100; j++) {
                Coord ijk(0, j, 0);
                if (!density->tree().isValueOn(ijk))    break;
                float previousValue = value;
                value = density->tree().getValue(Coord(0, j, 0));
                EXPECT_TRUE(value < previousValue);
            }

            // last falloff value is almost zero
            EXPECT_TRUE(value < 0.05f);
        }

        { // add "radius" attribute, but leave radius set to expect "pscale" attribute
            appendAttribute<float>(tree, "radius");

            // set radius to 2.0f
            auto handle = AttributeWriteHandle<float>(tree.beginLeaf()->attributeArray("radius"));
            handle.set(0, 2.0f);

            Settings settings(*transform);
            settings.useRadius = true;
            Rasterizer rasterizer(settings);
            rasterizer.addPoints(points, /*stream=*/false);
            auto density = rasterizer.rasterizeUniformDensity();
            EXPECT_EQ(Index64(32), density->tree().activeVoxelCount());
        }

        { // change radius attribute to "radius"
            Settings settings(*transform);
            settings.useRadius = true;
            settings.radiusAttribute = "radius";
            Rasterizer rasterizer(settings);
            rasterizer.addPoints(points, /*stream=*/false);
            auto density = rasterizer.rasterizeUniformDensity();
            EXPECT_EQ(Index64(268), density->tree().activeVoxelCount());
        }

        { // use a higher threshold
            Settings settings(*transform);
            settings.useRadius = true;
            settings.radiusAttribute = "radius";
            settings.threshold = 0.1f;
            Rasterizer rasterizer(settings);
            rasterizer.addPoints(points, /*stream=*/false);
            auto density = rasterizer.rasterizeUniformDensity();
            EXPECT_EQ(Index64(196), density->tree().activeVoxelCount());

            for (auto leaf = density->tree().cbeginLeaf(); leaf; ++leaf) {
                for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                    EXPECT_TRUE(*iter >= settings.threshold);
                }
            }
        }
    }

    { // frustum sphere rasterization
        Vec3s center(0.25, 0.25, 1.5);

        std::vector<Vec3s> positions =  {
                                            center
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        const double voxelSize = 0.5;
        math::Transform::Ptr transform(
            math::Transform::createLinearTransform(voxelSize));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

        Mat4d mat4( 0.5, 0.0, 0.0, 0.0,
                    0.0, 0.5, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0, 1.0);
        math::AffineMap affineMap(mat4);

        math::NonlinearFrustumMap frustumMap(
            BBoxd(Vec3d(-0.5, -0.5, -0.5), Vec3d(99.5, 99.5, 99.5)),
            /*taper=*/0.5, /*depth=*/1.0, /*affineMap=*/affineMap.copy());

        math::Transform::Ptr frustum = math::Transform(frustumMap.copy()).copy();

        { // accurate method
            Settings settings(*frustum);
            settings.useRadius = true;
            settings.accurateFrustumRadius = true;
            settings.radiusScale = 0.1f;
            Rasterizer rasterizer(settings);

            // add points to rasterizer

            rasterizer.addPoints(points, /*stream=*/false);

            // accumulate density

            auto density = rasterizer.rasterizeUniformDensity();

            Index64 voxelCount = density->tree().activeVoxelCount();
            EXPECT_TRUE(voxelCount > Index64(7000) && voxelCount < Index64(8000));
        }

        { // approximate method
            Settings settings(*frustum);
            settings.useRadius = true;
            settings.accurateFrustumRadius = false;
            settings.radiusScale = 0.1f;
            Rasterizer rasterizer(settings);

            // add points to rasterizer

            rasterizer.addPoints(points, /*stream=*/false);

            // accumulate density

            auto density = rasterizer.rasterizeUniformDensity();

            Index64 voxelCount = density->tree().activeVoxelCount();
            EXPECT_TRUE(voxelCount > Index64(7000) && voxelCount < Index64(8000));
        }
    }

    { // two overlapping spheres
        std::vector<Vec3s> positions =  {
                                            {-1.1f, 0.0, 0},
                                            {1.1f, 0.0, 0}
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        const double voxelSize = 1.0;
        math::Transform::Ptr transform(
            math::Transform::createLinearTransform(voxelSize));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

        Settings settings(*transform);
        settings.useRadius = true;
        settings.radiusScale = std::sqrt(3.0f)+0.001f;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

        { // uniform density with default mode (maximum)
            auto density = rasterizer.rasterizeUniformDensity();

            auto value1 = density->tree().getValue(Coord(-1, 0, 0));
            auto value2 = density->tree().getValue(Coord(1, 0, 0));

            EXPECT_TRUE(value1 > 0.9f && value1 < 1.0f);
            EXPECT_TRUE(value2 > 0.9f && value2 < 1.0f);

            auto center = density->tree().getValue(Coord(0, 0, 0));

            EXPECT_TRUE(center > 0.3f && center < 0.4f);
        }

        { // accumulate uniform density
            auto density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE);

            auto value1 = density->tree().getValue(Coord(-1, 0, 0));
            auto value2 = density->tree().getValue(Coord(1, 0, 0));

            EXPECT_TRUE(value1 > 0.9f && value1 < 1.0f);
            EXPECT_TRUE(value2 > 0.9f && value2 < 1.0f);

            auto center = density->tree().getValue(Coord(0, 0, 0));

            EXPECT_TRUE(center > 0.7f && center < 0.8f);
        }
    }
}

TEST_F(TestPointRasterizeFrustum, testVelocityMotionBlur)
{
    // particle 0 and particle 2 rasterize into two of the same voxels

    std::vector<Vec3s> positions =  {
                                        {0.0f, 1.2f, 0.0f},
                                        {100.0f, 100.0f, 50.0f},
                                        {0.0f, 1.55f, 0.0f},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    // leaf containing (0,1,0) voxel lives at (-0.25 => 3.75) in each axis world space for 0.5 voxel size

    const double voxelSize = 0.5;
    math::Transform::Ptr transform(
        math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
        createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    auto& pointsTree = points->tree();

    // add velocity attribute

    const float fps = 24.0f;

    appendAttribute<Vec3s>(pointsTree, "v");

    std::vector<Vec3s> velocities = {
                                        {0, fps, 0},
                                        {0, fps/2.5f, 0},
                                        {0, fps*4, 0}
                                    };

    PointAttributeVector<Vec3s> velocityWrapper(velocities);
    populateAttribute(pointsTree, pointIndexGrid->tree(), "v", velocityWrapper);

    // particle 0: (0, 0.95, 0) =>  (0, 1.45, 0)
    // particle 1: (100, 99.9, 50) => (100, 100.1, 50)
    // particle 2: (0, 0.55, 0) => (0, 2.55, 0)

    // voxels 0: (0, 2, 0) (0, 3, 0)
    // voxels 1: (100, 100, 50)
    // voxels 2: (0, 1, 0) (0, 2, 0) (0, 3, 0) (0, 4, 0) (0, 5, 0)

    // per-voxel contribution 0: 0.5
    // per-voxel contribution 1: 1.0
    // per-voxel contribution 2: 0.2

    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;
    Settings settings(*transform);
    settings.velocityMotionBlur = true;
    settings.threaded = false;
    Rasterizer rasterizer(settings);

    // add points to rasterizer

    rasterizer.addPoints(points, /*stream=*/false);

    { // accumulate uniform density

        auto density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE);

        EXPECT_TRUE(density);
        EXPECT_EQ(openvdb::Name("density"), density->getName());
        EXPECT_EQ(density->transform(), *transform);

        const auto& tree = density->constTree();

        EXPECT_EQ(Index64(6), tree.activeVoxelCount());
    }

    { // average uniform density

        auto density = rasterizer.rasterizeUniformDensity(RasterMode::AVERAGE);

        EXPECT_TRUE(density);

        const auto& tree = density->constTree();

        EXPECT_EQ(Index64(6), tree.activeVoxelCount());
    }

    { // maximum uniform density

        auto density = rasterizer.rasterizeUniformDensity(RasterMode::MAXIMUM);

        EXPECT_TRUE(density);

        const auto& tree = density->constTree();

        EXPECT_EQ(Index64(6), tree.activeVoxelCount());
    }

    // add density attribute

    appendAttribute<float>(pointsTree, "densityAttr");

    std::vector<float> densities{2.0f, 3.1f, 8.9f};
    PointAttributeVector<float> densityWrapper(densities);
    populateAttribute(pointsTree, pointIndexGrid->tree(), "densityAttr", densityWrapper);

    // float w0 = densities[0] / 2.0f;
    // float w1 = densities[1];
    // float w2 = densities[2] / 5.0f;

    { // accumulate density

        auto density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE);

        EXPECT_TRUE(density);
        EXPECT_EQ(openvdb::Name("density"), density->getName());
        EXPECT_EQ(density->transform(), *transform);

        const auto& tree = density->constTree();

        EXPECT_EQ(Index64(6), tree.activeVoxelCount());
    }

    { // average density

        auto density = rasterizer.rasterizeDensity("densityAttr", RasterMode::AVERAGE);

        EXPECT_TRUE(density);
        EXPECT_EQ(openvdb::Name("density"), density->getName());
        EXPECT_EQ(density->transform(), *transform);

        const auto& tree = density->constTree();

        EXPECT_EQ(Index64(6), tree.activeVoxelCount());
    }

    { // maximum density

        auto density = rasterizer.rasterizeDensity("densityAttr", RasterMode::MAXIMUM);

        EXPECT_TRUE(density);
        EXPECT_EQ(openvdb::Name("density"), density->getName());
        EXPECT_EQ(density->transform(), *transform);

        const auto& tree = density->constTree();

        EXPECT_EQ(Index64(6), tree.activeVoxelCount());
    }

    ////////////////////////////

    // point filtering (select one point only which rasterizes to two voxels)

    appendGroup(pointsTree, "test");

    std::vector<short> membership{1, 0, 0};

    setGroup(pointsTree, pointIndexGrid->tree(), membership, "test");

    std::vector<Name> includeGroups{"test"};
    std::vector<Name> excludeGroups;
    MultiGroupFilter filter(includeGroups, excludeGroups, pointsTree.cbeginLeaf()->attributeSet());

    auto density = rasterizer.rasterizeDensity("densityAttr", RasterMode::ACCUMULATE, false, 1.0f, filter);

    EXPECT_TRUE(density);
    EXPECT_EQ(Index64(2), density->activeVoxelCount());

    ////////////////////////////

    // scale contribution by voxel volume

    settings.transform = transform->copy();
    settings.scaleByVoxelVolume = true;
    Rasterizer scaleRasterizer(settings);

    const double voxelVolume = transform->voxelSize(Vec3d(0,0,0)).product();
    EXPECT_EQ(voxelVolume, (voxelSize * voxelSize * voxelSize));

    // add points to rasterizer

    scaleRasterizer.addPoints(points, /*stream=*/false);

    { // accumulate uniform density

        auto density = scaleRasterizer.rasterizeUniformDensity();

        EXPECT_TRUE(density);
        EXPECT_EQ(openvdb::Name("density"), density->getName());
        EXPECT_EQ(density->transform(), *transform);

        const auto& tree = density->constTree();

        EXPECT_EQ(Index64(6), tree.activeVoxelCount());
    }
}

TEST_F(TestPointRasterizeFrustum, testCameraMotionBlur)
{
    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;

    { // test RasterCamera API
        math::Transform::Ptr transform1a(
            math::Transform::createLinearTransform(0.5));
        math::Transform::Ptr transform1b(
            math::Transform::createLinearTransform(0.5));
        math::Transform::Ptr transform2(
            math::Transform::createLinearTransform(0.75));

        // manually build frustum transform using camera API

        math::NonlinearFrustumMap frustumMap(
            /*position*/Vec3d(0, 0, -10), /*direction*/Vec3d(0, 0, 1), /*up*/Vec3d(0, 1, 0),
            /*aspect*/1.5, /*znear*/0.001, /*depth*/20, /*xcount*/1000, /*zcount*/1000);

        math::Transform::Ptr transform3 = math::Transform(frustumMap.copy()).copy();

        RasterCamera camera(*transform1a);
        EXPECT_TRUE(camera.isStatic());

        EXPECT_NEAR(-0.25, camera.shutterStart(), 1e-6);
        EXPECT_NEAR(0.25, camera.shutterEnd(), 1e-6);

        camera.setShutter(-0.3f, 0.8f);
        EXPECT_NEAR(-0.3, camera.shutterStart(), 1e-6);
        EXPECT_NEAR(0.8, camera.shutterEnd(), 1e-6);

        EXPECT_EQ(size_t(1), camera.size());

        EXPECT_EQ(*transform1a, camera.firstTransform());
        EXPECT_EQ(*transform1a, camera.lastTransform());
        EXPECT_EQ(*transform1a, camera.transform(0));

        camera.clear();
        EXPECT_TRUE(camera.isStatic());

        EXPECT_EQ(size_t(0), camera.size());

        camera.appendTransform(*transform1a);
        camera.appendTransform(*transform2);
        camera.appendTransform(*transform1b);
        camera.appendTransform(*transform3);

        EXPECT_TRUE(!camera.isStatic());
        EXPECT_EQ(size_t(4), camera.size());

        EXPECT_EQ(*transform1a, camera.firstTransform());
        EXPECT_EQ(*transform3, camera.lastTransform());
        EXPECT_EQ(*transform2, camera.transform(1));

        // simplify does nothing as the stored transforms do not match

        camera.simplify();
        EXPECT_EQ(size_t(4), camera.size());

        camera.clear();
        camera.appendTransform(*transform1a);
        camera.appendTransform(*transform1b);

        EXPECT_TRUE(!camera.isStatic());
        EXPECT_EQ(size_t(2), camera.size());

        // as the two transforms are identical, only keep one for efficiency

        camera.simplify();

        EXPECT_TRUE(camera.isStatic());
        EXPECT_EQ(size_t(1), camera.size());
    }

    { // rasterize points with zero velocity to test multiple voxel contribution
      // that derives solely from moving the camera

        std::vector<Vec3s> positions =  {
                                            {0.0f, 1.2f, 0.0f},
                                            {100.0f, 100.0f, 50.0f},
                                            {1.1f, 1.23f, 0.0f},
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        // leaf containing (0,1,0) voxel lives at (-0.25 => 3.75) in each axis world space for 0.5 voxel size

        const double voxelSize = 0.5;
        math::Transform::Ptr transform(
            math::Transform::createLinearTransform(voxelSize));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

        Settings settings(*transform);

        { // accumulate uniform density, static linear camera transform

            Rasterizer rasterizer(settings);
            rasterizer.addPoints(points, /*stream=*/false);

            auto density = rasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(density);
            EXPECT_EQ(openvdb::Name("density"), density->getName());
            EXPECT_EQ(density->transform(), *transform);

            const auto& tree = density->constTree();

            EXPECT_NEAR(0.0, tree.background(), 1e-6);

            EXPECT_EQ(Index64(3), tree.activeVoxelCount());

            EXPECT_NEAR(1, tree.getValue(Coord(0, 2, 0)), 1e-6);
            EXPECT_NEAR(1, tree.getValue(Coord(2, 2, 0)), 1e-6);
            EXPECT_NEAR(1, tree.getValue(Coord(200, 200, 100)), 1e-6);
        }

        { // accumulate uniform density, linear camera transform with X translation

            math::Transform startTransform(*transform);
            startTransform.preTranslate(openvdb::Vec3d(-1, 0, 0));
            math::Transform endTransform(*transform);
            endTransform.preTranslate(openvdb::Vec3d(1, 0, 0));

            auto& camera = settings.camera;
            EXPECT_EQ(size_t(1), camera.size());
            camera.clear();
            camera.appendTransform(startTransform);
            camera.appendTransform(endTransform);

            Rasterizer rasterizer(settings);
            rasterizer.addPoints(points, /*stream=*/false);

            auto density = rasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(density);
            EXPECT_EQ(openvdb::Name("density"), density->getName());
            EXPECT_EQ(density->transform(), *transform);

            const auto& tree = density->constTree();

            EXPECT_NEAR(0.0, tree.background(), 1e-6);

            EXPECT_EQ(Index64(8), tree.activeVoxelCount());

            // point 0 - the rasterization is from the center of (-1, 2, 0) to the
            // center of (1, 2, 0), this means the contribution of these two end
            // voxels is half the contribution of (0, 2, 0) which the rasterization
            // ray passes entirely though. As the total contribution must add up to 1.0,
            // this means the values across these three voxels is 0.25, 0.5, 0.25

            // point 1 - the rasterization is from the center of (199, 200, 100) to the
            // center of (201, 200, 100), this means the contribution of these two end
            // voxels is half the contribution of (200, 200, 100) which the rasterization
            // ray passes entirely though. As the total contribution must add up to 1.0,
            // this means the values across these three voxels is 0.25, 0.5, 0.25

            // point 2 - the rasterization is between (1,2,0) and (3,2,0) but the ray
            // spends more time in the former than the latter with the rasterization
            // values for each voxel being 0.15, 0.5, 0.35

            // note that voxel (1,2,0) receives contributions from point 0 and point 2

            // point 0 contribution
            EXPECT_NEAR(0.25, tree.getValue(Coord(-1, 2, 0)), 1e-6);
            EXPECT_NEAR(0.5, tree.getValue(Coord(0, 2, 0)), 1e-6);

            // point 0 and point 1 contribution (max of 0.25 and 0.15)
            EXPECT_NEAR(0.25, tree.getValue(Coord(1, 2, 0)), 1e-6);

            // point 1 contribution
            EXPECT_NEAR(0.5, tree.getValue(Coord(2, 2, 0)), 1e-6);
            EXPECT_NEAR(0.35, tree.getValue(Coord(3, 2, 0)), 1e-6);

            // point 2 contribution
            EXPECT_NEAR(0.25, tree.getValue(Coord(199, 200, 100)), 1e-6);
            EXPECT_NEAR(0.5, tree.getValue(Coord(200, 200, 100)), 1e-6);
            EXPECT_NEAR(0.25, tree.getValue(Coord(201, 200, 100)), 1e-6);

            // re-rasterize using accumulate mode and verify voxel value that receives
            // contribution from two points

            auto density2 = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE);

            EXPECT_NEAR(0.25 + 0.15, density2->tree().getValue(Coord(1, 2, 0)), 1e-6);
        }
    }

    { // rasterize a single point with three camera transforms
        std::vector<Vec3s> positions =  {
                                            {0.0f, 1.2f, 0.0f},
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        const double voxelSize = 0.5;
        math::Transform::Ptr transform(
            math::Transform::createLinearTransform(voxelSize));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

        Settings settings(*transform);

        math::Transform startTransform(*transform);
        startTransform.preTranslate(openvdb::Vec3d(0, 0, 0));
        math::Transform middleTransform(*transform);
        middleTransform.preTranslate(openvdb::Vec3d(2, 0, 0));
        math::Transform endTransform(*transform);
        endTransform.preTranslate(openvdb::Vec3d(2, 2, 0));

        auto& camera = settings.camera;
        camera.clear();
        camera.appendTransform(startTransform);
        camera.appendTransform(middleTransform);
        camera.appendTransform(endTransform);
        camera.simplify();
        EXPECT_EQ(size_t(3), camera.size());

        { // point rasterize with two motion samples
            Rasterizer rasterizer(settings);
            rasterizer.addPoints(points, /*stream=*/false);

            auto density = rasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(density);
            EXPECT_EQ(openvdb::Name("density"), density->getName());
            EXPECT_EQ(density->transform(), *transform);

            const auto& tree = density->constTree();

            EXPECT_NEAR(0.0, tree.background(), 1e-6);

            float total = 0.0f;
            for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf) {
                for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                    total += iter.getValue();
                }
            }

            EXPECT_NEAR(1.0, total, 1e-6);
        }

        { // point rasterize with three motion samples
            // this value must be set explicitly and must be less than or equal to the
            // number of camera transforms, if motion samples is left at the default value of 2
            // this simply ignores the middle transform (as above)
            settings.motionSamples = 3;

            Rasterizer rasterizer(settings);
            rasterizer.addPoints(points, /*stream=*/false);

            auto density = rasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(density);
            EXPECT_EQ(openvdb::Name("density"), density->getName());
            EXPECT_EQ(density->transform(), *transform);

            const auto& tree = density->constTree();

            EXPECT_NEAR(0.0, tree.background(), 1e-6);

            float total = 0.0f;
            for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf) {
                for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                    total += iter.getValue();
                }
            }

            EXPECT_NEAR(1.0, total, 1e-6);
        }
    }
}

#ifndef ONLY_RASTER_FLOAT
TEST_F(TestPointRasterizeFrustum, testBool)
{
    std::vector<Vec3s> positions =  {
                                        {0.0f, 1.2f, 0.0f},
                                        {100.0f, 100.0f, 50.0f},
                                        {1.1f, 1.23f, 0.0f},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    // leaf containing (0,1,0) voxel lives at (-0.25 => 3.75) in each axis world space for 0.5 voxel size

    const double voxelSize = 0.5;
    math::Transform::Ptr transform(
        math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
        createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    auto& pointsTree = points->tree();

    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;

    {
        Settings settings(*transform);
        settings.velocityMotionBlur = false;
        Rasterizer rasterizer(settings);

        // add points to rasterizer

        rasterizer.addPoints(points, /*stream=*/false);

        // verify ability to rasterize bool or mask grids

        auto boolGrid = rasterizer.rasterizeMask<BoolGrid>();

        EXPECT_TRUE(boolGrid->isType<BoolGrid>());
        EXPECT_EQ(Name("mask"), boolGrid->getName());
        EXPECT_EQ(Index64(3), boolGrid->activeVoxelCount());

        auto maskGrid = rasterizer.rasterizeMask<MaskGrid>();

        EXPECT_TRUE(maskGrid->isType<MaskGrid>());
        EXPECT_EQ(Name("mask"), maskGrid->getName());
        EXPECT_EQ(Index64(3), maskGrid->activeVoxelCount());

        // throw if attempting to use average mode when rasterizing a mask
        EXPECT_THROW((rasterizer.rasterizeAttribute<BoolGrid, bool>("", RasterMode::AVERAGE)),
            ValueError);

        // throw if attempting to use maximum mode when rasterizing a mask
        EXPECT_THROW((rasterizer.rasterizeAttribute<BoolGrid, bool>("", RasterMode::MAXIMUM)),
            ValueError);
    }

    // append velocity attribute

    appendAttribute<Vec3s>(pointsTree, "v");

    std::vector<Vec3s> velocities = {
                                        {0.0f, 10.0f, 0.0f},
                                        {1.0f, 2.0f, 3.0f},
                                        {-3.0f, -2.0f, -1.0f},
                                    };
    PointAttributeVector<Vec3s> velocityWrapper(velocities);
    populateAttribute(pointsTree, pointIndexGrid->tree(), "v", velocityWrapper);

    // use velocity motion blur

    {
        Settings settings(*transform);
        settings.velocityMotionBlur = true;
        Rasterizer rasterizer(settings);

        // add points to rasterizer

        rasterizer.addPoints(points, /*stream=*/false);

        // verify ability to rasterize bool or mask grids

        auto boolGrid = rasterizer.rasterizeMask<BoolGrid>();

        EXPECT_TRUE(boolGrid->isType<BoolGrid>());
        EXPECT_EQ(Name("mask"), boolGrid->getName());
        EXPECT_EQ(Index64(5), boolGrid->activeVoxelCount());

        auto maskGrid = rasterizer.rasterizeMask<MaskGrid>();

        EXPECT_TRUE(maskGrid->isType<MaskGrid>());
        EXPECT_EQ(Name("mask"), maskGrid->getName());
        EXPECT_EQ(Index64(5), maskGrid->activeVoxelCount());
    }
}

TEST_F(TestPointRasterizeFrustum, testInt)
{
    std::vector<Vec3s> positions =  {
                                        {0.0f, 1.2f, 0.0f},
                                        {0.0f, 0.6f, 0.1f},
                                        {1.1f, 1.23f, 0.0f},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    // leaf containing (0,1,0) voxel lives at (-0.25 => 3.75) in each axis world space for 0.5 voxel size

    const double voxelSize = 0.5;
    math::Transform::Ptr transform(
        math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
        createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    auto& pointsTree = points->tree();

    // append id attribute

    appendAttribute<int>(pointsTree, "id");

    std::vector<int> ids = { 1, 2, 3 };
    PointAttributeVector<int> idWrapper(ids);
    populateAttribute(pointsTree, pointIndexGrid->tree(), "id", idWrapper);

    // append velocity attribute

    appendAttribute<Vec3s>(pointsTree, "v");

    std::vector<Vec3s> velocities = {
                                        {0.0f, 10.0f, 0.0f},
                                        {0.0f, 24.0f, 0.0f},
                                        {0.0f, -2.0f, 0.0f},
                                    };
    PointAttributeVector<Vec3s> velocityWrapper(velocities);
    populateAttribute(pointsTree, pointIndexGrid->tree(), "v", velocityWrapper);

    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;

    {
        Settings settings(*transform);
        settings.velocityMotionBlur = true;
        Rasterizer rasterizer(settings);

        // add points to rasterizer

        rasterizer.addPoints(points, /*stream=*/false);

        { // rasterize int grid with accumulate mode
            auto intGrid = rasterizer.rasterizeAttribute("id", RasterMode::ACCUMULATE);

            EXPECT_TRUE(intGrid->isType<Int32Grid>());
            EXPECT_EQ(Name("id"), intGrid->getName());
            EXPECT_EQ(Index64(5), intGrid->activeVoxelCount());

            auto intGridTyped = GridBase::grid<Int32Grid>(intGrid);

            EXPECT_EQ(2, intGridTyped->tree().getValue(Coord(0,1,0)));
            EXPECT_EQ(3, intGridTyped->tree().getValue(Coord(0,2,0)));
        }

        { // rasterize int grid with maximum mode
            auto intGrid = rasterizer.rasterizeAttribute("id", RasterMode::MAXIMUM);

            EXPECT_EQ(Index64(5), intGrid->activeVoxelCount());

            auto intGridTyped = GridBase::grid<Int32Grid>(intGrid);

            EXPECT_EQ(2, intGridTyped->tree().getValue(Coord(0,1,0)));
            EXPECT_EQ(2, intGridTyped->tree().getValue(Coord(0,2,0)));
        }
    }
}
#endif

TEST_F(TestPointRasterizeFrustum, testInputs)
{
    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;

    auto empty = PointDataGrid::create();

    const double voxelSize = 0.5;
    math::Transform::Ptr transform(
        math::Transform::createLinearTransform(voxelSize));

    // attempt to create a rasterizer with velocity motion blur enabled,
    // but no velocity attribute name throws

    Settings settings(*transform);
    settings.velocityAttribute = "";
    settings.velocityMotionBlur = true;

    EXPECT_THROW(Rasterizer{settings}, ValueError);

    // reset velocity attribute
    settings.velocityAttribute = "v";

    // adding an empty points grid still produces an empty grid

    Rasterizer rasterizer(settings);
    EXPECT_NO_THROW(rasterizer.addPoints(empty));

    auto density = rasterizer.rasterizeUniformDensity();
    EXPECT_EQ(Index64(0), density->activeVoxelCount());

#ifndef ONLY_RASTER_FLOAT
    // attempting to rasterize position is not allowed

    EXPECT_THROW((rasterizer.rasterizeAttribute<Vec3SGrid, Vec3s>("P")), ValueError);
#endif

    // add a points grid that contains points

    std::vector<Vec3s> positions =  {
                                        {0.0f, 1.2f, 0.0f},
                                        {100.0f, 100.0f, 50.0f},
                                        {1.1f, 1.23f, 0.0f},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
        createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

    rasterizer.addPoints(points);

    // attempting to rasterize an attribute that doesn't exist simply ignores that grid

    FloatGrid::Ptr invalidAttributeGrid;
    EXPECT_NO_THROW((invalidAttributeGrid =
        rasterizer.rasterizeAttribute<FloatGrid, float>("invalidAttribute")));
    EXPECT_TRUE(invalidAttributeGrid);

#ifndef ONLY_RASTER_FLOAT
    // add density attribute

    appendAttribute<float>(points->tree(), "density");

    std::vector<float> densities{2.0f, 3.1f, 8.9f};
    PointAttributeVector<float> densityWrapper(densities);
    populateAttribute(points->tree(), pointIndexGrid->tree(), "density", densityWrapper);

    // attempting to rasterize an attribute with the wrong value type should error

    EXPECT_THROW((rasterizer.rasterizeAttribute<DoubleGrid, double>("density")),
        TypeError);
#endif
}


namespace
{

// Test thread-safe Interrupter that halts on the second interrupt check
struct HaltOnSecondInterrupt : public util::NullInterrupter
{
    /// Default constructor
    HaltOnSecondInterrupt() = default;
    void start(const char* name = NULL) override { (void)name; }
    void end() override {}
    /// Check if an interruptible operation should be aborted.
    inline bool wasInterrupted(int percent = -1) override
    {
        (void)percent;
        if (mInterrupt)     return true;
        mInterrupt = true;
        return false;
    }
    std::atomic<bool> mInterrupt{false};
};

} // namespace


TEST_F(TestPointRasterizeFrustum, testInterrupter)
{
    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;
    using Mask = FrustumRasterizerMask;
    using InterruptRasterizer = FrustumRasterizer<PointDataGrid>;

    // manually build frustum transform using camera API

    Vec3d position(9.5, 7.8, 21.7);

    math::NonlinearFrustumMap frustumMap(
        /*position*/position, /*direction*/-position.unit(), /*up*/-position.unit().cross(Vec3d(0,1,0)) * 10,
        /*aspect*/0.75, /*znear*/10, /*depth*/100, /*xcount*/100, /*zcount*/400);

    math::Transform::Ptr frustum = math::Transform(frustumMap.copy()).copy();

    // build a level set sphere

    Vec3s center(0, 0, 0);
    // float radius = 1;
    float radius = 10;
    float voxelSize = 0.2f;

    auto surface = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);

    // scatter points inside the sphere

    auto points = points::denseUniformPointScatter(*surface, /*pointsPerVoxel=*/8);

    // rasterize without an interrupter

    Settings settings(*frustum);
    settings.velocityAttribute = "";
    settings.velocityMotionBlur = false;
    Mask mask;
    { // rasterize as points
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points);
        auto density = rasterizer.rasterizeUniformDensity();

        { // verify this test interrupter interrupts the second time
            HaltOnSecondInterrupt interrupter;
            EXPECT_TRUE(!interrupter.wasInterrupted());
            EXPECT_TRUE(interrupter.wasInterrupted());
            EXPECT_TRUE(interrupter.wasInterrupted());
        }

        { // rasterize with the interrupter with threading enabled (relies on TBB's cancel_group_execution())
            HaltOnSecondInterrupt interrupter;
            InterruptRasterizer interruptRasterizer(settings, mask, &interrupter);
            interruptRasterizer.addPoints(points);
            auto interruptDensity = interruptRasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(interrupter.wasInterrupted());
            EXPECT_TRUE(interruptDensity->activeVoxelCount() > Index64(0));
            EXPECT_TRUE(interruptDensity->activeVoxelCount() < Index64(density->activeVoxelCount()));
        }

        { // rasterize with the interrupter with threading disabled
            HaltOnSecondInterrupt interrupter;
            settings.threaded = false;
            InterruptRasterizer interruptRasterizer(settings, mask, &interrupter);
            interruptRasterizer.addPoints(points);
            auto interruptDensity = interruptRasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(interrupter.wasInterrupted());
            EXPECT_TRUE(interruptDensity->activeVoxelCount() > Index64(0));
            EXPECT_TRUE(interruptDensity->activeVoxelCount() < Index64(density->activeVoxelCount()));
        }
    }

    { // rasterize as spheres
        std::vector<Vec3s> positions =  {
                                            {0,  0,  0},
                                            {10, 10, 10}
                                        };

        const PointAttributeVector<Vec3s> pointList(positions);

        math::Transform::Ptr transform(
            math::Transform::createLinearTransform(/*voxelSize=*/1.0f));

        tools::PointIndexGrid::Ptr pointIndexGrid =
            tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

        PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);

        Settings settings(*frustum);
        settings.useRadius = true;
        settings.radiusScale = 2.0f;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points);
        auto density = rasterizer.rasterizeUniformDensity();

        { // rasterize with the interrupter with threading enabled (relies on TBB's cancel_group_execution())
            HaltOnSecondInterrupt interrupter;
            InterruptRasterizer interruptRasterizer(settings, mask, &interrupter);
            interruptRasterizer.addPoints(points);
            auto interruptDensity = interruptRasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(interrupter.wasInterrupted());
            EXPECT_TRUE(interruptDensity->activeVoxelCount() > Index64(0));
            EXPECT_TRUE(interruptDensity->activeVoxelCount() < Index64(density->activeVoxelCount()));
        }

        { // rasterize with a very large radius to ensure interrupt is triggered early
            settings.radiusScale = 100.0f;

            HaltOnSecondInterrupt interrupter;
            InterruptRasterizer interruptRasterizer(settings, mask, &interrupter);
            interruptRasterizer.addPoints(points);
            auto interruptDensity = interruptRasterizer.rasterizeUniformDensity();

            EXPECT_TRUE(interrupter.wasInterrupted());

            // when a large sphere is used, interrupt is done per XY slice,
            // which results in an empty density grid
            EXPECT_EQ(Index64(0), interruptDensity->activeVoxelCount());
        }
    }
}

TEST_F(TestPointRasterizeFrustum, testClipping)
{
    std::vector<Vec3s> positions =  {
                                        {0, 0, 0},
                                        {0, 2, -2},
                                        {3, 4, 0},
                                        {1, 4, 1},
                                        {2, 5, 2},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    math::Transform::Ptr transform(
        math::Transform::createLinearTransform(/*voxelSize=*/0.5f));

    math::Transform::Ptr outputTransform(
        math::Transform::createLinearTransform(/*voxelSize=*/1.0f));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
        createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid, pointList, *transform);
    auto& tree = points->tree();

    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;
    using Mask = FrustumRasterizerMask;

    { // default settings (except no motion-blur), no clip bbox or clip mask
        Settings settings(*outputTransform);
        settings.velocityMotionBlur = false;

        Mask mask;

        Rasterizer rasterizer(settings, mask);
        rasterizer.addPoints(points, /*stream=*/false);
        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_EQ(Index64(5), density->tree().activeVoxelCount());
    }

    { // clip bbox that eliminates fifth point
        Settings settings(*outputTransform);
        settings.velocityMotionBlur = false;

        Mask mask(*outputTransform, nullptr, BBoxd(Vec3d(-0.5,-0.5,-2.5), Vec3d(4.0,4.0,3)));

        Rasterizer rasterizer(settings, mask);
        rasterizer.addPoints(points, /*stream=*/false);
        auto density = rasterizer.rasterizeUniformDensity();

        EXPECT_EQ(Index64(4), density->tree().activeVoxelCount());
    }

    { // clip bbox that only keeps first two points
        Settings settings(*outputTransform);
        settings.velocityMotionBlur = false;

        Mask mask(*outputTransform, nullptr, BBoxd(Vec3d(-0.5,-0.5,-2.5), Vec3d(1,3,0.5)));

        Rasterizer rasterizer(settings, mask);
        rasterizer.addPoints(points, /*stream=*/false);
        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_EQ(Index64(2), density->tree().activeVoxelCount());
    }

    { // clip mask that keeps all but the fifth point
        auto sphere = tools::createLevelSetSphere<FloatGrid>(5.5f, openvdb::Vec3f(0), 1.0f);
        auto sphereBool = tools::interiorMask(*sphere);
        auto sphereMask = MaskGrid::Ptr(new MaskGrid(*sphereBool));

        Settings settings(*outputTransform);
        settings.velocityMotionBlur = false;

        Mask mask(*outputTransform, sphereMask.get());

        Rasterizer rasterizer(settings, mask);
        rasterizer.addPoints(points, /*stream=*/false);
        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_EQ(Index64(4), density->tree().activeVoxelCount());
    }

    { // clip mask that keeps all but the first two points
        auto sphere = tools::createLevelSetSphere<FloatGrid>(5.0f, openvdb::Vec3f(0), 1.0f);
        auto sphereBool = tools::interiorMask(*sphere);
        auto sphereMask = MaskGrid::Ptr(new MaskGrid(*sphereBool));

        Settings settings(*outputTransform);
        settings.velocityMotionBlur = false;

        Mask mask(*outputTransform, sphereMask.get());

        Rasterizer rasterizer(settings, mask);
        rasterizer.addPoints(points, /*stream=*/false);
        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_EQ(Index64(3), density->tree().activeVoxelCount());
    }

    { // clip mask that only keeps first two points
        auto sphere = tools::createLevelSetSphere<FloatGrid>(5.0f, openvdb::Vec3f(0), 1.0f);
        auto sphereBool = tools::interiorMask(*sphere);
        auto sphereMask = MaskGrid::Ptr(new MaskGrid(*sphereBool));

        // third point coord for lookup
        Coord coord(1, 4, 1);

        sphereMask->tree().setValueOff(coord);

        Settings settings(*outputTransform);
        settings.velocityMotionBlur = false;

        Mask mask(*outputTransform, sphereMask.get());

        settings.threaded = false;
        Rasterizer rasterizer(settings, mask);
        rasterizer.addPoints(points, /*stream=*/false);

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_EQ(Index64(2), density->tree().activeVoxelCount());

        // introduce group where third point is disabled

        appendGroup(tree, "test");
        std::vector<short> membership{1, 1, 0, 1, 1};
        setGroup(tree, pointIndexGrid->tree(), membership, "test");

        std::vector<Name> includeGroups{"test"};
        std::vector<Name> excludeGroups;
        MultiGroupFilter filter(includeGroups, excludeGroups, tree.cbeginLeaf()->attributeSet());

        // re-enable mask voxel for third point

        sphereMask->tree().setValueOn(coord);

        // filter with group filtering and mask clipping

        density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE, false, 1.0f, filter);
        EXPECT_EQ(Index64(2), density->tree().activeVoxelCount());

        sphereMask->tree().setValueOff(coord);

        // invert the mask

        Mask invertMask(*outputTransform, sphereMask.get(), BBoxd(),
            /*clipToFrustum=*/false, /*invert=*/true);

        Rasterizer rasterizer2(settings, invertMask);
        rasterizer2.addPoints(points, /*stream=*/false);

        density = rasterizer2.rasterizeUniformDensity(RasterMode::ACCUMULATE, false, 1.0f);
        EXPECT_EQ(Index64(3), density->tree().activeVoxelCount());
    }
}

TEST_F(TestPointRasterizeFrustum, testStreaming)
{
    Name filename("rasterpoints.vdb");

    // manually build frustum transform using camera API

    Vec3d position(9.5, 7.8, 21.7);

    math::NonlinearFrustumMap frustumMap(
        /*position*/position, /*direction*/-position.unit(), /*up*/-position.unit().cross(Vec3d(0,1,0)) * 10,
        /*aspect*/0.75, /*znear*/10, /*depth*/100, /*xcount*/100, /*zcount*/400);

    math::Transform::Ptr frustum = math::Transform(frustumMap.copy()).copy();

    // build the level set sphere

    Vec3s center(0, 0, 0);
    float radius = 10;
    float voxelSize = 0.2f;

    auto surface = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);

    // scatter points inside the sphere

    auto points = points::denseUniformPointScatter(*surface, /*pointsPerVoxel=*/8);

    // append and expand attributes, as they're uniform when created

    points::appendAttribute<float>(points->tree(), "density");
    points::appendAttribute<Vec3f>(points->tree(), "v");

    for (auto leaf = points->tree().beginLeaf(); leaf; ++leaf) {
        AttributeWriteHandle<float> densityHandle(leaf->attributeArray("density"));
        densityHandle.collapse(1.0f);
        densityHandle.expand();

        AttributeWriteHandle<Vec3s> velocityHandle(leaf->attributeArray("v"));
        velocityHandle.expand();
    }

    // set name and transform

    const math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);

    points->setName("points");
    points->setTransform(xform);

    // write points to file

    GridCPtrVec grids;
    grids.push_back(points);

    {
        io::File file(filename);
        file.write(grids);
        file.close();
    }

    // read points from file (using delayed loading)

    {
        io::File file(filename);
        file.open();
        openvdb::GridBase::Ptr baseGrid = file.readGrid("points");
        file.close();

        points = openvdb::gridPtrCast<PointDataGrid>(baseGrid);
    }

    auto leaf = points->tree().cbeginLeaf();
    EXPECT_TRUE(leaf);
#ifdef OPENVDB_USE_DELAYED_LOADING
    EXPECT_TRUE(leaf->buffer().isOutOfCore());
#endif

    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;
    using Mask = FrustumRasterizerMask;

    Settings settings(*frustum);
    settings.velocityMotionBlur = true;
    settings.threshold = 0.0f; // disable threshold when testing streaming
    Mask mask(*frustum);

    Rasterizer rasterizer(settings, mask);

    // add points to rasterizer

    rasterizer.addPoints(points, /*stream=*/false);

    EXPECT_TRUE(!leaf->constAttributeArray("P").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isStreaming());

    EXPECT_TRUE(!leaf->constAttributeArray("P").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isUniform());

    auto density = rasterizer.rasterizeUniformDensity();
    EXPECT_EQ(Index64(74242), density->activeVoxelCount());

    // streaming is disabled, so all attributes should still be non-uniform

    EXPECT_TRUE(!leaf->constAttributeArray("P").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isStreaming());

    EXPECT_TRUE(!leaf->constAttributeArray("P").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isUniform());

    rasterizer.clear();
    rasterizer.addPoints(points, /*stream=*/true);

    density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE, /*reduceMemory=*/false);
    EXPECT_EQ(Index64(74242), density->activeVoxelCount());

    // streaming is enabled, but reduce memory is false,
    // so all attributes should still be non-uniform

    EXPECT_TRUE(!leaf->constAttributeArray("P").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isStreaming());

    EXPECT_TRUE(!leaf->constAttributeArray("P").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isUniform());

    // reopen file
    {
        io::File file(filename);
        file.open();
        openvdb::GridBase::Ptr baseGrid = file.readGrid("points");
        file.close();

        points = openvdb::gridPtrCast<PointDataGrid>(baseGrid);
    }
    leaf = points->tree().cbeginLeaf();

    rasterizer.clear();
    rasterizer.addPoints(points, /*stream=*/true);

    density = rasterizer.rasterizeUniformDensity(RasterMode::ACCUMULATE, /*reduceMemory=*/true);
    EXPECT_EQ(Index64(74242), density->activeVoxelCount());

    // with streaming and reduce memory both true, position and velocity should be uniform

    EXPECT_TRUE(leaf->constAttributeArray("P").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isStreaming());
    EXPECT_TRUE(leaf->constAttributeArray("v").isStreaming());

    EXPECT_TRUE(leaf->constAttributeArray("P").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isUniform());
    EXPECT_TRUE(leaf->constAttributeArray("v").isUniform());

    // reopen file
    {
        io::File file(filename);
        file.open();
        openvdb::GridBase::Ptr baseGrid = file.readGrid("points");
        file.close();

        points = openvdb::gridPtrCast<PointDataGrid>(baseGrid);
    }
    leaf = points->tree().cbeginLeaf();

    rasterizer.clear();
    rasterizer.addPoints(points, /*stream=*/true);

    // when rasterizing density, all attributes now become uniform

    density = rasterizer.rasterizeDensity("density", RasterMode::ACCUMULATE, /*reduceMemory=*/true);
    EXPECT_EQ(Index64(74242), density->activeVoxelCount());

    EXPECT_TRUE(leaf->constAttributeArray("P").isStreaming());
    EXPECT_TRUE(leaf->constAttributeArray("density").isStreaming());
    EXPECT_TRUE(leaf->constAttributeArray("v").isStreaming());

    EXPECT_TRUE(leaf->constAttributeArray("P").isUniform());
    EXPECT_TRUE(leaf->constAttributeArray("density").isUniform());
    EXPECT_TRUE(leaf->constAttributeArray("v").isUniform());

#ifndef ONLY_RASTER_FLOAT
    // reopen file
    {
        io::File file(filename);
        file.open();
        openvdb::GridBase::Ptr baseGrid = file.readGrid("points");
        file.close();

        points = openvdb::gridPtrCast<PointDataGrid>(baseGrid);
    }
    leaf = points->tree().cbeginLeaf();

    rasterizer.clear();
    rasterizer.addPoints(points, /*stream=*/true);

    // now verify that rasterizing velocity then density with streaming
    // enabled completes successfully

    auto velocity = rasterizer.rasterizeAttribute("v", RasterMode::ACCUMULATE, /*reduceMemory=*/false);
    EXPECT_EQ(Index64(74242), velocity->activeVoxelCount());

    EXPECT_TRUE(!leaf->constAttributeArray("P").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isStreaming());

    EXPECT_TRUE(!leaf->constAttributeArray("P").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("v").isUniform());

    velocity = rasterizer.rasterizeAttribute("v", RasterMode::ACCUMULATE, /*reduceMemory=*/true);
    EXPECT_EQ(Index64(74242), velocity->activeVoxelCount());

    EXPECT_TRUE(leaf->constAttributeArray("P").isStreaming());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isStreaming());
    EXPECT_TRUE(leaf->constAttributeArray("v").isStreaming());

    EXPECT_TRUE(leaf->constAttributeArray("P").isUniform());
    EXPECT_TRUE(!leaf->constAttributeArray("density").isUniform());
    EXPECT_TRUE(leaf->constAttributeArray("v").isUniform());
#endif

    // reopen file
    {
        io::File file(filename);
        file.open();
        openvdb::GridBase::Ptr baseGrid = file.readGrid("points");
        file.close();

        points = openvdb::gridPtrCast<PointDataGrid>(baseGrid);
    }
    leaf = points->tree().cbeginLeaf();

    // copy and translate the transform

    auto transform = points->transform().copy();
    transform->preTranslate(Vec3d(0, 100, 0));

    // deep-copy the points grid and apply the new transform

    auto points2 = points->deepCopy();
    points2->setTransform(transform);

#ifdef OPENVDB_USE_DELAYED_LOADING
    // verify both grids are out-of-core

    EXPECT_TRUE(points->tree().cbeginLeaf()->buffer().isOutOfCore());
    EXPECT_TRUE(points2->tree().cbeginLeaf()->buffer().isOutOfCore());
#endif

#ifndef ONLY_RASTER_FLOAT
    // memory tests

    if (io::Archive::isDelayedLoadingEnabled() && io::Archive::hasBloscCompression()) {

        FloatGrid::Ptr density1, density2, density3;
        Vec3SGrid::Ptr velocity1, velocity2, velocity3;

        const size_t mb = 1024*1024;
        const size_t tinyMemory = static_cast<size_t>(0.1*mb);

        size_t initialMemory;

        { // memory test 1 - retain caches and streaming disabled
            Rasterizer rasterizer(settings);

            rasterizer.addPoints(points, /*stream=*/false);
            rasterizer.addPoints(points2, /*stream=*/false);

            initialMemory = rasterizer.memUsage();

            EXPECT_TRUE(initialMemory > size_t(4*mb) && initialMemory < size_t(16*mb));

            EXPECT_EQ(size_t(2), rasterizer.size());

            velocity1 = rasterizer.rasterizeAttribute<Vec3SGrid, Vec3s>("v");
            EXPECT_EQ(Index64(219780), velocity1->activeVoxelCount());

            EXPECT_TRUE(rasterizer.memUsage() > size_t(71*mb) &&
                rasterizer.memUsage() < size_t(91*mb));

            density1 = rasterizer.rasterizeDensity("density");
            EXPECT_EQ(Index64(219780), density1->activeVoxelCount());

            // no data is discarded so expect a fairly high memory footprint

            EXPECT_TRUE(rasterizer.memUsage() > size_t(80*mb) &&
                rasterizer.memUsage() < size_t(100*mb));
        }

        { // memory test 2 - retain caches and streaming enabled

            { // reopen file and deep copy while setting transform
                io::File file(filename);
                file.open();
                openvdb::GridBase::Ptr baseGrid = file.readGrid("points");
                file.close();

                points = openvdb::gridPtrCast<PointDataGrid>(baseGrid);
                points2 = points->deepCopy();
                points2->setTransform(transform);
            }

            Rasterizer rasterizer(settings);

            rasterizer.addPoints(points, /*stream=*/true);
            rasterizer.addPoints(points2, /*stream=*/true);

            EXPECT_EQ(initialMemory, rasterizer.memUsage());

            EXPECT_EQ(size_t(2), rasterizer.size());

            velocity2 = rasterizer.rasterizeAttribute<Vec3SGrid, Vec3s>("v");
            EXPECT_EQ(Index64(219780), velocity2->activeVoxelCount());

            size_t postRasterMemory = rasterizer.memUsage();

            EXPECT_TRUE(postRasterMemory > size_t(70*mb) && postRasterMemory < size_t(85*mb));

            density2 = rasterizer.rasterizeDensity("density");
            EXPECT_EQ(Index64(219780), density2->activeVoxelCount());

            // as data is being streamed, second attribute shouldn't change memory usage very much

            EXPECT_TRUE(rasterizer.memUsage() < (postRasterMemory + tinyMemory));
        }

        { // memory test 3 - release caches and streaming enabled

            { // reopen file and deep copy while setting transform
                io::File file(filename);
                file.open();
                openvdb::GridBase::Ptr baseGrid = file.readGrid("points");
                file.close();

                points = openvdb::gridPtrCast<PointDataGrid>(baseGrid);
                points2 = points->deepCopy();
                points2->setTransform(transform);
            }

            auto points3 = points->deepCopy();
            auto points4 = points2->deepCopy();

            Settings settings2(*frustum);
            settings2.threshold = 0.0f;

            Mask mask2(*frustum, nullptr, BBoxd(), /*clipToFrustum=*/false);

            Rasterizer rasterizer(settings2, mask2);

            rasterizer.addPoints(points, /*stream=*/true);
            rasterizer.addPoints(points2, /*stream=*/true);

            EXPECT_EQ(initialMemory, rasterizer.memUsage());
            EXPECT_EQ(size_t(2), rasterizer.size());

            density3 = rasterizer.rasterizeDensity("density", RasterMode::ACCUMULATE, true);
            EXPECT_EQ(Index64(219780), density3->activeVoxelCount());

            // all voxel data, attribute data and caches are being discarded,
            // so memory after rasterizing shouldn't change very much

            EXPECT_TRUE(rasterizer.memUsage() < (initialMemory + tinyMemory));

            // deep-copies of delay-loaded point grids need to be used for repeat rasterization

            rasterizer.clear();
            rasterizer.addPoints(points3, /*stream=*/true);
            rasterizer.addPoints(points4, /*stream=*/true);

            EXPECT_EQ(size_t(2), rasterizer.size());

            EXPECT_TRUE(rasterizer.memUsage() < (initialMemory + tinyMemory));

            velocity3 = rasterizer.rasterizeAttribute<Vec3SGrid, Vec3s>("v", RasterMode::ACCUMULATE, true);
            EXPECT_EQ(Index64(219780), velocity3->activeVoxelCount());
        }
    }
#endif

    // clean up file
    remove(filename.c_str());
}

TEST_F(TestPointRasterizeFrustum, testProfile)
{
    using Rasterizer = FrustumRasterizer<PointDataGrid>;
    using Settings = FrustumRasterizerSettings;

    // fill a sphere with points to use for rasterization

    Vec3s center(0.0f, 0.0f, 25.0f);
#ifdef PROFILE
    float radius = 40;
#else
    float radius = 3;
#endif
    float voxelSize = 0.1f;

    auto surface = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);

    // scatter points inside the sphere

    auto referencePoints = points::denseUniformPointScatter(*surface, /*pointsPerVoxel=*/8);

#ifdef PROFILE
    std::cerr << std::endl;
    std::cerr << "---- Profiling Rasterization ----" << std::endl;
    std::cerr << "Points To Rasterize: " << points::pointCount(referencePoints->constTree()) << std::endl;
#endif

    math::Transform::Ptr linearTransform = math::Transform::createLinearTransform(voxelSize);

    Mat4d mat4( 100.0, 0.0, 0.0, 0.0,
                0.0, 100.0, 0.0, 0.0,
                0.0, 0.0, 100.0, 0.0,
                0.0, 0.0, -0.5, 1.0);
    math::AffineMap affineMap(mat4);

    BBoxd frustumBBox(Vec3d(-0.5,-0.5,-0.5), Vec3d(999.5, 999.5, 999.5));

    math::NonlinearFrustumMap nonLinearFrustumMap(
        frustumBBox, /*taper=*/0.5, /*depth=*/1.0, /*affineMap=*/affineMap.copy());

    math::Transform::Ptr frustumTransform(math::Transform(nonLinearFrustumMap.copy()).copy());

    { // point rasterize, linear transform
        auto points = referencePoints->deepCopy();

        Settings settings(*linearTransform);
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Point Rasterize, Linear Transform");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // point rasterize, frustum transform
        auto points = referencePoints->deepCopy();

        Settings settings(*frustumTransform);
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Point Rasterize, Frustum Transform");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // point rasterize, linear transform, low velocity
        auto points = referencePoints->deepCopy();

        points::appendAttribute<Vec3s>(points->tree(), "v", Vec3s(0.0f, 0.0001f, 0.0f));

        Settings settings(*linearTransform);
        settings.velocityMotionBlur = true;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Point Rasterize, Linear Transform, Low Velocity");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // point rasterize, linear transform, high velocity
        auto points = referencePoints->deepCopy();

        points::appendAttribute<Vec3s>(points->tree(), "v", Vec3s(0.0f, 1.0f, 0.0f));

        Settings settings(*linearTransform);
        settings.velocityMotionBlur = true;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Point Rasterize, Linear Transform, High Velocity");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // point rasterize, frustum transform
        auto points = referencePoints->deepCopy();

        points::appendAttribute<Vec3s>(points->tree(), "v", Vec3s(0.0f, 1.0f, 0.0f));

        Settings settings(*frustumTransform);
        settings.velocityMotionBlur = true;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Point Rasterize, Frustum Transform, High Velocity");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // sphere rasterize, frustum transform, approximate algorithm
        auto points = referencePoints->deepCopy();

        points::appendAttribute<float>(points->tree(), "pscale", 0.25f);

        Settings settings(*linearTransform);
        settings.useRadius = true;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Sphere Rasterize, Linear Transform");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // sphere rasterize, frustum transform, approximate algorithm
        auto points = referencePoints->deepCopy();

        points::appendAttribute<float>(points->tree(), "pscale", 0.25f);

        Settings settings(*frustumTransform);
        settings.useRadius = true;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Sphere Rasterize, Frustum Transform, Approximate");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // sphere rasterize, frustum transform, high velocity, approximate algorithm
        auto points = referencePoints->deepCopy();

        points::appendAttribute<Vec3s>(points->tree(), "v", Vec3s(0.0f, 1.0f, 0.0f));
        points::appendAttribute<float>(points->tree(), "pscale", 0.25f);

        Settings settings(*frustumTransform);
        settings.velocityMotionBlur = true;
        settings.useRadius = true;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Sphere Rasterize, Frustum Transform, High Velocity, Approximate");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

    { // sphere rasterize, frustum transform, high velocity, accurate algorithm with motion samples
        auto points = referencePoints->deepCopy();

        points::appendAttribute<Vec3s>(points->tree(), "v", Vec3s(0.0f, 1.0f, 0.0f));
        points::appendAttribute<float>(points->tree(), "pscale", 0.25f);

        Settings settings(*frustumTransform);
        settings.velocityMotionBlur = true;
        settings.useRadius = true;
        settings.accurateFrustumRadius = true;
        settings.motionSamples = 10;
        Rasterizer rasterizer(settings);
        rasterizer.addPoints(points, /*stream=*/false);

#ifdef PROFILE
        openvdb::util::CpuTimer timer("Sphere Rasterize, Frustum Transform, High Velocity, Accurate");
#endif

        auto density = rasterizer.rasterizeUniformDensity();
        EXPECT_TRUE(density);

#ifdef PROFILE
        timer.stop();

        std::cerr << "Voxel Count: " << density->tree().activeVoxelCount() << std::endl;
#endif
    }

#ifdef PROFILE
    std::cerr << "----------------------------------" << std::endl;
    std::cerr << std::endl;
#endif
}
