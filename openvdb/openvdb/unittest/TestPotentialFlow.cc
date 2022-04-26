// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file unittest/TestPotentialFlow.cc

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/PotentialFlow.h>

#include <gtest/gtest.h>


class TestPotentialFlow: public ::testing::Test
{
};


TEST_F(TestPotentialFlow, testMask)
{
    using namespace openvdb;

    const float radius = 1.5f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 0.25f;
    const float halfWidth = 3.0f;

    FloatGrid::Ptr sphere =
        tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, halfWidth);

    const int dilation = 5;

    MaskGrid::Ptr mask = tools::createPotentialFlowMask(*sphere, dilation);
    MaskGrid::Ptr defaultMask = tools::createPotentialFlowMask(*sphere);
    EXPECT_TRUE(*mask == *defaultMask);

    auto acc = mask->getAccessor();

    // the isosurface of this sphere is at y = 6
    // this mask forms a band dilated outwards from the isosurface by 5 voxels

    EXPECT_TRUE(!acc.isValueOn(Coord(0, 5, 0)));
    EXPECT_TRUE(acc.isValueOn(Coord(0, 6, 0)));
    EXPECT_TRUE(acc.isValueOn(Coord(0, 10, 0)));
    EXPECT_TRUE(!acc.isValueOn(Coord(0, 11, 0)));

    { // error on non-uniform voxel size
        FloatGrid::Ptr nonUniformSphere =
            tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, halfWidth);
        math::Transform::Ptr nonUniformTransform(new math::Transform(
            math::MapBase::Ptr(new math::ScaleMap(Vec3d(0.1, 0.2, 0.3)))));
        nonUniformSphere->setTransform(nonUniformTransform);

        EXPECT_THROW(tools::createPotentialFlowMask(*nonUniformSphere, dilation),
            openvdb::ValueError);
    }

    // this is the minimum mask of one voxel either side of the isosurface

    mask = tools::createPotentialFlowMask(*sphere, 2);

    acc = mask->getAccessor();

    EXPECT_TRUE(!acc.isValueOn(Coord(0, 5, 0)));
    EXPECT_TRUE(acc.isValueOn(Coord(0, 6, 0)));
    EXPECT_TRUE(acc.isValueOn(Coord(0, 7, 0)));
    EXPECT_TRUE(!acc.isValueOn(Coord(0, 8, 0)));

    // these should all produce the same masks as the dilation value is clamped

    MaskGrid::Ptr negativeMask = tools::createPotentialFlowMask(*sphere, -1);
    MaskGrid::Ptr zeroMask = tools::createPotentialFlowMask(*sphere, 0);
    MaskGrid::Ptr oneMask = tools::createPotentialFlowMask(*sphere, 1);

    EXPECT_TRUE(*negativeMask == *mask);
    EXPECT_TRUE(*zeroMask == *mask);
    EXPECT_TRUE(*oneMask == *mask);
}


TEST_F(TestPotentialFlow, testNeumannVelocities)
{
    using namespace openvdb;

    const float radius = 1.5f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 0.25f;
    const float halfWidth = 3.0f;

    FloatGrid::Ptr sphere =
        tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, halfWidth);

    MaskGrid::Ptr domain = tools::createPotentialFlowMask(*sphere);

    {
        // test identical potential from a wind velocity supplied through grid or background value

        Vec3d windVelocityValue(0, 0, 10);

        Vec3dTree::Ptr windTree(new Vec3dTree(sphere->tree(), zeroVal<Vec3d>(), TopologyCopy()));
        dilateActiveValues(*windTree, 2, tools::NN_FACE_EDGE_VERTEX, tools::IGNORE_TILES);
        windTree->voxelizeActiveTiles();

        for (auto leaf = windTree->beginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->beginValueOn(); iter; ++iter) {
                iter.setValue(windVelocityValue);
            }
        }

        Vec3dGrid::Ptr windGrid(Vec3dGrid::create(windTree));
        windGrid->setTransform(sphere->transform().copy());

        auto windPotentialFromGrid = tools::createPotentialFlowNeumannVelocities(
            *sphere, *domain, windGrid, Vec3d(0));

        EXPECT_EQ(windPotentialFromGrid->transform(), sphere->transform());

        auto windPotentialFromBackground = tools::createPotentialFlowNeumannVelocities(
            *sphere, *domain, Vec3dGrid::Ptr(), windVelocityValue);

        auto accessor = windPotentialFromGrid->getConstAccessor();
        auto accessor2 = windPotentialFromBackground->getConstAccessor();

        EXPECT_EQ(windPotentialFromGrid->activeVoxelCount(),
            windPotentialFromBackground->activeVoxelCount());

        for (auto leaf = windPotentialFromGrid->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                EXPECT_EQ(accessor.isValueOn(iter.getCoord()),
                    accessor2.isValueOn(iter.getCoord()));
                EXPECT_EQ(accessor.getValue(iter.getCoord()),
                    accessor2.getValue(iter.getCoord()));
            }
        }

        // test potential from a wind velocity supplied through grid background value

        Vec3dTree::Ptr emptyWindTree(
            new Vec3dTree(sphere->tree(), windVelocityValue, TopologyCopy()));
        Vec3dGrid::Ptr emptyWindGrid(Vec3dGrid::create(emptyWindTree));
        emptyWindGrid->setTransform(sphere->transform().copy());

        auto windPotentialFromGridBackground = tools::createPotentialFlowNeumannVelocities(
            *sphere, *domain, emptyWindGrid, Vec3d(0));

        EXPECT_EQ(windPotentialFromGridBackground->transform(), sphere->transform());

        accessor = windPotentialFromGridBackground->getConstAccessor();
        accessor2 = windPotentialFromBackground->getConstAccessor();

        EXPECT_EQ(windPotentialFromGridBackground->activeVoxelCount(),
            windPotentialFromBackground->activeVoxelCount());

        for (auto leaf = windPotentialFromGridBackground->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                EXPECT_EQ(accessor.isValueOn(iter.getCoord()),
                    accessor2.isValueOn(iter.getCoord()));
                EXPECT_EQ(accessor.getValue(iter.getCoord()),
                    accessor2.getValue(iter.getCoord()));
            }
        }

        // test potential values are double when applying wind velocity
        // through grid and background values

        auto windPotentialFromBoth = tools::createPotentialFlowNeumannVelocities(
            *sphere, *domain, windGrid, windVelocityValue);

        tools::prune(windPotentialFromBoth->tree(), Vec3d(1e-3));
        tools::prune(windPotentialFromBackground->tree(), Vec3d(1e-3));

        accessor = windPotentialFromBoth->getConstAccessor();
        accessor2 = windPotentialFromBackground->getConstAccessor();

        for (auto leaf = windPotentialFromBoth->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                EXPECT_EQ(accessor.isValueOn(iter.getCoord()),
                    accessor2.isValueOn(iter.getCoord()));
                EXPECT_EQ(accessor.getValue(iter.getCoord()),
                    accessor2.getValue(iter.getCoord()) * 2);
            }
        }

        EXPECT_TRUE(*windPotentialFromBoth == *windPotentialFromBackground);
    }

    Vec3dGrid::Ptr zeroVelocity = Vec3dGrid::create(Vec3d(0));

    { // error if grid is not a levelset
        FloatGrid::Ptr nonLevelSetSphere =
            tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, halfWidth);
        nonLevelSetSphere->setGridClass(GRID_FOG_VOLUME);

        EXPECT_THROW(tools::createPotentialFlowNeumannVelocities(
            *nonLevelSetSphere, *domain, zeroVelocity, Vec3d(5)), openvdb::TypeError);
    }

    { // accept double level set grid
        DoubleGrid::Ptr doubleSphere =
            tools::createLevelSetSphere<DoubleGrid>(radius, center, voxelSize, halfWidth);

        EXPECT_NO_THROW(tools::createPotentialFlowNeumannVelocities(
            *doubleSphere, *domain, zeroVelocity, Vec3d(5)));
    }

    { // zero boundary velocities and background velocity
        Vec3d zeroVelocityValue(zeroVal<Vec3d>());
        auto neumannVelocities = tools::createPotentialFlowNeumannVelocities(
            *sphere, *domain, zeroVelocity, zeroVelocityValue);
        EXPECT_EQ(neumannVelocities->activeVoxelCount(), Index64(0));
    }
}


TEST_F(TestPotentialFlow, testUniformStream)
{
    // this unit test checks the scalar potential and velocity flow field
    // for a uniform stream which consists of a 100x100x100 cube of
    // neumann voxels with constant velocity (0, 0, 1)

    using namespace openvdb;

    auto transform = math::Transform::createLinearTransform(1.0);

    auto mask = MaskGrid::create(false);
    mask->setTransform(transform);
    auto maskAccessor = mask->getAccessor();

    auto neumann = Vec3dGrid::create(Vec3d(0));
    auto neumannAccessor = neumann->getAccessor();

    for (int i = -50; i < 50; i++) {
        for (int j = -50; j < 50; j++) {
            for (int k = -50; k < 50; k++) {
                Coord ijk(i, j, k);
                maskAccessor.setValueOn(ijk, true);
                neumannAccessor.setValueOn(ijk, Vec3d(0, 0, 1));
            }
        }
    }

    openvdb::math::pcg::State state = math::pcg::terminationDefaults<float>();

    state.iterations = 2000;
    state.absoluteError = 1e-8;

    auto potential = tools::computeScalarPotential(*mask, *neumann, state);

    // check convergence

    EXPECT_TRUE(state.success);
    EXPECT_TRUE(state.iterations > 0 && state.iterations < 1000);
    EXPECT_TRUE(state.absoluteError < 1e-6);

    EXPECT_EQ(potential->activeVoxelCount(), mask->activeVoxelCount());

    // for uniform flow along the z-axis, the scalar potential should be equal to the z co-ordinate

    for (auto leaf = potential->tree().cbeginLeaf(); leaf; ++leaf) {
        for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
            const double staggeredZ = iter.getCoord().z() + 0.5;
            EXPECT_TRUE(math::isApproxEqual(iter.getValue(), staggeredZ, /*tolerance*/0.1));
        }
    }

    auto flow = tools::computePotentialFlow(*potential, *neumann);

    EXPECT_EQ(flow->activeVoxelCount(), mask->activeVoxelCount());

    // flow velocity should be equal to the input velocity (0, 0, 1)

    for (auto leaf = flow->tree().cbeginLeaf(); leaf; ++leaf) {
        for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
            EXPECT_TRUE(math::isApproxEqual(iter.getValue().x(), 0.0, /*tolerance*/1e-6));
            EXPECT_TRUE(math::isApproxEqual(iter.getValue().y(), 0.0, /*tolerance*/1e-6));
            EXPECT_TRUE(math::isApproxEqual(iter.getValue().z(), 1.0, /*tolerance*/1e-6));
        }
    }
}


TEST_F(TestPotentialFlow, testFlowAroundSphere)
{
    using namespace openvdb;

    const float radius = 1.5f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 0.25f;
    const float halfWidth = 3.0f;

    const int dilation = 50;

    FloatGrid::Ptr sphere =
        tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, halfWidth);

    MaskGrid::Ptr domain = tools::createPotentialFlowMask(*sphere, dilation);

    { // compute potential flow for a global wind velocity around a sphere

        Vec3f windVelocity(0, 0, 1);
        Vec3fGrid::Ptr neumann = tools::createPotentialFlowNeumannVelocities(*sphere,
            *domain, Vec3fGrid::Ptr(), windVelocity);

        openvdb::math::pcg::State state = math::pcg::terminationDefaults<float>();

        state.iterations = 2000;
        state.absoluteError = 1e-8;

        FloatGrid::Ptr potential = tools::computeScalarPotential(*domain, *neumann, state);

        // compute a laplacian of the potential within the domain (excluding neumann voxels)
        // and ensure it evaluates to zero

        auto mask = BoolGrid::create(/*background=*/false);
        mask->setTransform(potential->transform().copy());
        mask->topologyUnion(*potential);

        auto dilatedSphereMask = tools::interiorMask(*sphere);
        tools::dilateActiveValues(dilatedSphereMask->tree(), 1);
        mask->topologyDifference(*dilatedSphereMask);

        FloatGrid::Ptr laplacian = tools::laplacian(*potential, *mask);

        for (auto leaf = laplacian->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                EXPECT_TRUE(math::isApproxEqual(iter.getValue(), 0.0f, /*tolerance*/1e-3f));
            }
        }

        Vec3fGrid::Ptr flowVel = tools::computePotentialFlow(*potential, *neumann);

        // compute the divergence of the flow velocity within the domain
        // (excluding neumann voxels and exterior voxels)
        // and ensure it evaluates to zero

        tools::erodeActiveValues(mask->tree(), 2, tools::NN_FACE, tools::IGNORE_TILES);

        FloatGrid::Ptr divergence = tools::divergence(*flowVel, *mask);

        for (auto leaf = divergence->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                EXPECT_TRUE(math::isApproxEqual(iter.getValue(), 0.0f, /*tolerance*/0.1f));
            }
        }

        // check the background velocity has been applied correctly

        Vec3fGrid::Ptr flowVelBackground =
            tools::computePotentialFlow(*potential, *neumann, windVelocity);

        EXPECT_EQ(flowVelBackground->activeVoxelCount(),
            flowVelBackground->activeVoxelCount());

        auto maskAccessor = mask->getConstAccessor();

        auto accessor = flowVel->getConstAccessor();
        auto accessor2 = flowVelBackground->getConstAccessor();

        for (auto leaf = flowVelBackground->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                // ignore values near the neumann boundary
                if (!maskAccessor.isValueOn(iter.getCoord()))    continue;

                const Vec3f value1 = accessor.getValue(iter.getCoord());
                const Vec3f value2 = accessor2.getValue(iter.getCoord()) + windVelocity;

                EXPECT_TRUE(math::isApproxEqual(value1.x(), value2.x(), /*tolerance=*/1e-3f));
                EXPECT_TRUE(math::isApproxEqual(value1.y(), value2.y(), /*tolerance=*/1e-3f));
                EXPECT_TRUE(math::isApproxEqual(value1.z(), value2.z(), /*tolerance=*/1e-3f));
            }
        }
    }

    { // check double-precision solve
        DoubleGrid::Ptr sphereDouble =
            tools::createLevelSetSphere<DoubleGrid>(radius, center, voxelSize, halfWidth);

        Vec3d windVelocity(0, 0, 1);
        Vec3dGrid::Ptr neumann = tools::createPotentialFlowNeumannVelocities(*sphereDouble,
            *domain, Vec3dGrid::Ptr(), windVelocity);

        openvdb::math::pcg::State state = math::pcg::terminationDefaults<float>();

        state.iterations = 2000;
        state.absoluteError = 1e-8;

        DoubleGrid::Ptr potential = tools::computeScalarPotential(*domain, *neumann, state);

        EXPECT_TRUE(potential);

        // compute a laplacian of the potential within the domain (excluding neumann voxels)
        // and ensure it evaluates to zero

        auto mask = BoolGrid::create(/*background=*/false);
        mask->setTransform(potential->transform().copy());
        mask->topologyUnion(*potential);

        auto dilatedSphereMask = tools::interiorMask(*sphereDouble);
        tools::dilateActiveValues(dilatedSphereMask->tree(), 1);
        mask->topologyDifference(*dilatedSphereMask);

        DoubleGrid::Ptr laplacian = tools::laplacian(*potential, *mask);

        for (auto leaf = laplacian->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                EXPECT_TRUE(math::isApproxEqual(iter.getValue(), 0.0, /*tolerance*/1e-5));
            }
        }

        Vec3dGrid::Ptr flowVel = tools::computePotentialFlow(*potential, *neumann);

        EXPECT_TRUE(flowVel);
    }
}
