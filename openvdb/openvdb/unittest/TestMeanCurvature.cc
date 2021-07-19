// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/math/Operators.h>
#include <openvdb/tools/GridOperators.h>
#include "util.h" // for unittest_util::makeSphere()
#include <openvdb/tools/LevelSetSphere.h>

#include <gtest/gtest.h>


class TestMeanCurvature: public ::testing::Test
{
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


TEST_F(TestMeanCurvature, testISMeanCurvature)
{
    using namespace openvdb;

    typedef FloatGrid::ConstAccessor  AccessorType;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    AccessorType inAccessor = grid->getConstAccessor();
    AccessorType::ValueType alpha, beta, meancurv, normGrad;
    Coord xyz(35,30,30);

    // First test an empty grid
    EXPECT_TRUE(tree.empty());
    typedef math::ISMeanCurvature<math::CD_SECOND, math::CD_2ND> SecondOrder;
    EXPECT_TRUE(!SecondOrder::result(inAccessor, xyz, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_FOURTH, math::CD_4TH> FourthOrder;
    EXPECT_TRUE(!FourthOrder::result(inAccessor, xyz, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_SIXTH, math::CD_6TH> SixthOrder;
    EXPECT_TRUE(!SixthOrder::result(inAccessor, xyz, alpha, beta));

    // Next test a level set sphere
    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    EXPECT_TRUE(!tree.empty());

    SecondOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

    FourthOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

    SixthOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

    xyz.reset(35,10,40);

    SecondOrder::result(inAccessor, xyz, alpha, beta);

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/20.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/20.0, normGrad, 0.001);
}


TEST_F(TestMeanCurvature, testISMeanCurvatureStencil)
{
    using namespace openvdb;

    typedef FloatGrid::ConstAccessor  AccessorType;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();
    math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
    math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
    math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
    AccessorType::ValueType alpha, beta;
    Coord xyz(35,30,30);
    dense_2nd.moveTo(xyz);
    dense_4th.moveTo(xyz);
    dense_6th.moveTo(xyz);

    // First test on an empty grid
    EXPECT_TRUE(tree.empty());

    typedef math::ISMeanCurvature<math::CD_SECOND, math::CD_2ND> SecondOrder;
    EXPECT_TRUE(!SecondOrder::result(dense_2nd, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_FOURTH, math::CD_4TH> FourthOrder;
    EXPECT_TRUE(!FourthOrder::result(dense_4th, alpha, beta));

    typedef math::ISMeanCurvature<math::CD_SIXTH, math::CD_6TH> SixthOrder;
    EXPECT_TRUE(!SixthOrder::result(dense_6th, alpha, beta));

    // Next test on a level set sphere
    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);
    dense_2nd.moveTo(xyz);
    dense_4th.moveTo(xyz);
    dense_6th.moveTo(xyz);

    EXPECT_TRUE(!tree.empty());

    EXPECT_TRUE(SecondOrder::result(dense_2nd, alpha, beta));

    AccessorType::ValueType meancurv = alpha/(2*math::Pow3(beta) );
    AccessorType::ValueType normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

    EXPECT_TRUE(FourthOrder::result(dense_4th, alpha, beta));

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

    EXPECT_TRUE(SixthOrder::result(dense_6th, alpha, beta));

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

    xyz.reset(35,10,40);
    dense_2nd.moveTo(xyz);
    EXPECT_TRUE(SecondOrder::result(dense_2nd, alpha, beta));

    meancurv = alpha/(2*math::Pow3(beta) );
    normGrad = alpha/(2*math::Pow2(beta) );

    EXPECT_NEAR(1.0/20.0, meancurv, 0.001);
    EXPECT_NEAR(1.0/20.0, normGrad, 0.001);
}


TEST_F(TestMeanCurvature, testWSMeanCurvature)
{
    using namespace openvdb;
    using math::AffineMap;
    using math::TranslationMap;
    using math::UniformScaleMap;

    typedef FloatGrid::ConstAccessor  AccessorType;

    {// Empty grid test
        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();
        AccessorType inAccessor = grid->getConstAccessor();
        Coord xyz(35,30,30);
        EXPECT_TRUE(tree.empty());

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, inAccessor, xyz);
        EXPECT_NEAR(0.0, meancurv, 0.0);
        EXPECT_NEAR(0.0, normGrad, 0.0);

        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, inAccessor, xyz);
        EXPECT_NEAR(0.0, meancurv, 0.0);
        EXPECT_NEAR(0.0, normGrad, 0.0);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, inAccessor, xyz);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, inAccessor, xyz);
        EXPECT_NEAR(0.0, meancurv, 0.0);
        EXPECT_NEAR(0.0, normGrad, 0.0);

        xyz.reset(35,10,40);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, inAccessor, xyz);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, inAccessor, xyz);
        EXPECT_NEAR(0.0, meancurv, 0.0);
        EXPECT_NEAR(0.0, normGrad, 0.0);
    }

    { // unit size voxel test
        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();

        const openvdb::Coord dim(64,64,64);
        const openvdb::Vec3f center(35.0f ,30.0f, 40.0f);
        const float radius=0.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        EXPECT_TRUE(!tree.empty());
        Coord xyz(35,30,30);

        AccessorType inAccessor = grid->getConstAccessor();

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, inAccessor, xyz);

        EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/10.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, inAccessor, xyz);


        EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, inAccessor, xyz);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, inAccessor, xyz);

        EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

        xyz.reset(35,10,40);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, inAccessor, xyz);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, inAccessor, xyz);


        EXPECT_NEAR(1.0/20.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/20.0, normGrad, 0.001);
    }
    { // non-unit sized voxel

        double voxel_size = 0.5;
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(voxel_size));
        EXPECT_TRUE(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        AccessorType inAccessor = grid->getConstAccessor();

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        AffineMap affine(voxel_size*math::Mat3d::identity());
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, inAccessor, xyz);

        EXPECT_NEAR(1.0/4.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/4.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, inAccessor, xyz);


        EXPECT_NEAR(1.0/4.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/4.0, normGrad, 0.001);

        UniformScaleMap uniform(voxel_size);
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, inAccessor, xyz);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, inAccessor, xyz);

        EXPECT_NEAR(1.0/4.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/4.0, normGrad, 0.001);

    }
    { // NON-UNIFORM SCALING AND ROTATION

        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        FloatGrid::Ptr grid = FloatGrid::create();
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        math::MapBase::Ptr rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        grid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));
        EXPECT_TRUE(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        AccessorType inAccessor = grid->getConstAccessor();

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        Vec3d location = grid->indexToWorld(xyz);
        double dist = (center - location).length();
        AffineMap::ConstPtr affine = grid->transform().map<AffineMap>();
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            *affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            *affine, inAccessor, xyz);

        EXPECT_NEAR(1.0/dist, meancurv, 0.001);
        EXPECT_NEAR(1.0/dist, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            *affine, inAccessor, xyz);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            *affine, inAccessor, xyz);


        EXPECT_NEAR(1.0/dist, meancurv, 0.001);
        EXPECT_NEAR(1.0/dist, normGrad, 0.001);
    }
}


TEST_F(TestMeanCurvature, testWSMeanCurvatureStencil)
{
    using namespace openvdb;
    using math::AffineMap;
    using math::TranslationMap;
    using math::UniformScaleMap;

    typedef FloatGrid::ConstAccessor AccessorType;

    {// empty grid test
        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();
        EXPECT_TRUE(tree.empty());
        Coord xyz(35,30,30);

        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);
        dense_6th.moveTo(xyz);

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, dense_2nd);
        EXPECT_NEAR(0.0, meancurv, 0.0);
        EXPECT_NEAR(0.0, normGrad, 0.00);

        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, dense_4th);
        EXPECT_NEAR(0.0, meancurv, 0.00);
        EXPECT_NEAR(0.0, normGrad, 0.00);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, dense_6th);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, dense_6th);
        EXPECT_NEAR(0.0, meancurv, 0.0);
        EXPECT_NEAR(0.0, normGrad, 0.0);

        xyz.reset(35,10,40);
        dense_6th.moveTo(xyz);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, dense_6th);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, dense_6th);
        EXPECT_NEAR(0.0, meancurv, 0.0);
        EXPECT_NEAR(0.0, normGrad, 0.0);
    }

    { // unit-sized voxels

        FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
        FloatTree& tree = grid->tree();

        const openvdb::Coord dim(64,64,64);
        const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);//i.e. (35,30,40) in index space
        const float radius=0.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        EXPECT_TRUE(!tree.empty());
        Coord xyz(35,30,30);
        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);
        dense_6th.moveTo(xyz);

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        AffineMap affine;
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, dense_2nd);

        EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/10.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, dense_4th);

        EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

        UniformScaleMap uniform;
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, dense_6th);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, dense_6th);

        EXPECT_NEAR(1.0/10.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/10.0, normGrad, 0.001);

        xyz.reset(35,10,40);
        dense_6th.moveTo(xyz);

        TranslationMap trans;
        meancurv = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::result(
            trans, dense_6th);
        normGrad = math::MeanCurvature<TranslationMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            trans, dense_6th);


        EXPECT_NEAR(1.0/20.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/20.0, normGrad, 0.001);
    }
    { // non-unit sized voxel

        double voxel_size = 0.5;
        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(voxel_size));
        EXPECT_TRUE(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);


        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        math::SixthOrderDenseStencil<FloatGrid> dense_6th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);
        dense_6th.moveTo(xyz);

        AffineMap affine(voxel_size*math::Mat3d::identity());
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            affine, dense_2nd);

        EXPECT_NEAR(1.0/4.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/4.0, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            affine, dense_4th);


        EXPECT_NEAR(1.0/4.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/4.0, normGrad, 0.001);

        UniformScaleMap uniform(voxel_size);
        meancurv = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::result(
            uniform, dense_6th);
        normGrad = math::MeanCurvature<UniformScaleMap, math::CD_SIXTH, math::CD_6TH>::normGrad(
            uniform, dense_6th);

        EXPECT_NEAR(1.0/4.0, meancurv, 0.001);
        EXPECT_NEAR(1.0/4.0, normGrad, 0.001);
    }
    { // NON-UNIFORM SCALING AND ROTATION

        Vec3d voxel_sizes(0.25, 0.45, 0.75);
        FloatGrid::Ptr grid = FloatGrid::create();
        math::MapBase::Ptr base_map( new math::ScaleMap(voxel_sizes));
        // apply rotation
        math::MapBase::Ptr rotated_map = base_map->preRotate(1.5, math::X_AXIS);
        grid->setTransform(math::Transform::Ptr(new math::Transform(rotated_map)));
        EXPECT_TRUE(grid->empty());

        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        AccessorType::ValueType meancurv;
        AccessorType::ValueType normGrad;

        Coord xyz(20,16,20);
        math::SecondOrderDenseStencil<FloatGrid> dense_2nd(*grid);
        math::FourthOrderDenseStencil<FloatGrid> dense_4th(*grid);
        dense_2nd.moveTo(xyz);
        dense_4th.moveTo(xyz);


        Vec3d location = grid->indexToWorld(xyz);
        double dist = (center - location).length();
        AffineMap::ConstPtr affine = grid->transform().map<AffineMap>();
        meancurv = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::result(
            *affine, dense_2nd);
        normGrad = math::MeanCurvature<AffineMap, math::CD_SECOND, math::CD_2ND>::normGrad(
            *affine, dense_2nd);

        EXPECT_NEAR(1.0/dist, meancurv, 0.001);
        EXPECT_NEAR(1.0/dist, normGrad, 0.001);
        meancurv = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::result(
            *affine, dense_4th);
        normGrad = math::MeanCurvature<AffineMap, math::CD_FOURTH, math::CD_4TH>::normGrad(
            *affine, dense_4th);


        EXPECT_NEAR(1.0/dist, meancurv, 0.001);
        EXPECT_NEAR(1.0/dist, normGrad, 0.001);
    }
}


TEST_F(TestMeanCurvature, testMeanCurvatureTool)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);//i.e. (35,30,40) in index space
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    EXPECT_TRUE(!tree.empty());
    FloatGrid::Ptr curv = tools::meanCurvature(*grid);
    FloatGrid::ConstAccessor accessor = curv->getConstAccessor();

    Coord xyz(35,30,30);
    EXPECT_NEAR(1.0/10.0, accessor.getValue(xyz), 0.001);

    xyz.reset(35,10,40);
    EXPECT_NEAR(1.0/20.0, accessor.getValue(xyz), 0.001);
}


TEST_F(TestMeanCurvature, testMeanCurvatureMaskedTool)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*background=*/5.0);
    FloatTree& tree = grid->tree();

    const openvdb::Coord dim(64,64,64);
    const openvdb::Vec3f center(35.0f, 30.0f, 40.0f);//i.e. (35,30,40) in index space
    const float radius=0.0f;
    unittest_util::makeSphere<FloatGrid>(dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

    EXPECT_TRUE(!tree.empty());


    const openvdb::CoordBBox maskbbox(openvdb::Coord(35, 30, 30), openvdb::Coord(41, 41, 41));
    BoolGrid::Ptr maskGrid = BoolGrid::create(false);
    maskGrid->fill(maskbbox, true/*value*/, true/*activate*/);


    FloatGrid::Ptr curv = tools::meanCurvature(*grid, *maskGrid);
    FloatGrid::ConstAccessor accessor = curv->getConstAccessor();

    // test inside
    Coord xyz(35,30,30);
    EXPECT_TRUE(maskbbox.isInside(xyz));
    EXPECT_NEAR(1.0/10.0, accessor.getValue(xyz), 0.001);

    // test outside
    xyz.reset(35,10,40);
    EXPECT_TRUE(!maskbbox.isInside(xyz));
    EXPECT_NEAR(0.0, accessor.getValue(xyz), 0.001);
}


TEST_F(TestMeanCurvature, testCurvatureStencil)
{
    using namespace openvdb;

    {// test of level set to sphere at (6,8,10) with R=10 and dx=0.5

        FloatGrid::Ptr grid = FloatGrid::create(/*backgroundValue=*/5.0);
        grid->setTransform(math::Transform::createLinearTransform(/*voxel size=*/0.5));
        EXPECT_TRUE(grid->empty());
        math::CurvatureStencil<FloatGrid> cs(*grid);
        Coord xyz(20,16,20);//i.e. 8 voxel or 4 world units away from the center
        cs.moveTo(xyz);

        // First test on an empty grid
        EXPECT_NEAR(0.0, cs.meanCurvature(), 0.0);
        EXPECT_NEAR(0.0, cs.meanCurvatureNormGrad(), 0.0);

        // Next test on a level set sphere
        const openvdb::Coord dim(32,32,32);
        const openvdb::Vec3f center(6.0f, 8.0f, 10.0f);//i.e. (12,16,20) in index space
        const float radius=10.0f;
        unittest_util::makeSphere<FloatGrid>(
            dim, center, radius, *grid, unittest_util::SPHERE_DENSE);

        EXPECT_TRUE(!grid->empty());
        EXPECT_EQ(dim[0]*dim[1]*dim[2], int(grid->activeVoxelCount()));
        cs.moveTo(xyz);

        EXPECT_NEAR(1.0/4.0, cs.meanCurvature(), 0.01);// 1/distance from center
        EXPECT_NEAR(1.0/4.0, cs.meanCurvatureNormGrad(), 0.01);// 1/distance from center

        EXPECT_NEAR(1.0/16.0, cs.gaussianCurvature(), 0.01);// 1/distance^2 from center
        EXPECT_NEAR(1.0/16.0, cs.gaussianCurvatureNormGrad(), 0.01);// 1/distance^2 from center

        float mean, gaussian;
        cs.curvatures(mean, gaussian);
        EXPECT_NEAR(1.0/4.0, mean, 0.01);// 1/distance from center
        EXPECT_NEAR(1.0/16.0, gaussian, 0.01);// 1/distance^2 from center

        auto principalCurvatures = cs.principalCurvatures();
        EXPECT_NEAR(1.0/4.0, principalCurvatures.first,  0.01);// 1/distance from center
        EXPECT_NEAR(1.0/4.0, principalCurvatures.second, 0.01);// 1/distance from center

        xyz.reset(12,16,10);//i.e. 10 voxel or 5 world units away from the center
        cs.moveTo(xyz);
        EXPECT_NEAR(1.0/5.0, cs.meanCurvature(), 0.01);// 1/distance from center
        EXPECT_NEAR(
            1.0/5.0, cs.meanCurvatureNormGrad(), 0.01);// 1/distance from center

        EXPECT_NEAR(1.0/25.0, cs.gaussianCurvature(), 0.01);// 1/distance^2 from center
        EXPECT_NEAR(
            1.0/25.0, cs.gaussianCurvatureNormGrad(), 0.01);// 1/distance^2 from center

        principalCurvatures = cs.principalCurvatures();
        EXPECT_NEAR(1.0/5.0, principalCurvatures.first,  0.01);// 1/distance from center
        EXPECT_NEAR(1.0/5.0, principalCurvatures.second, 0.01);// 1/distance from center
        EXPECT_NEAR(
            1.0/5.0, principalCurvatures.first,  0.01);// 1/distance from center
            EXPECT_NEAR(
            1.0/5.0, principalCurvatures.second, 0.01);// 1/distance from center

        cs.curvaturesNormGrad(mean, gaussian);
        EXPECT_NEAR(1.0/5.0, mean, 0.01);// 1/distance from center
        EXPECT_NEAR(1.0/25.0, gaussian, 0.01);// 1/distance^2 from center
    }
    {// test sparse level set sphere
      const double percentage = 0.1/100.0;//i.e. 0.1%
      const int dim = 256;

      // sparse level set sphere
      Vec3f C(0.35f, 0.35f, 0.35f);
      Real r = 0.15, voxelSize = 1.0/(dim-1);
      FloatGrid::Ptr sphere = tools::createLevelSetSphere<FloatGrid>(float(r), C, float(voxelSize));

      math::CurvatureStencil<FloatGrid> cs(*sphere);
      const Coord ijk = Coord::round(sphere->worldToIndex(Vec3d(0.35, 0.35, 0.35 + 0.15)));
      const double radius = (sphere->indexToWorld(ijk)-Vec3d(0.35)).length();
      //std::cerr << "\rRadius = " << radius << std::endl;
      //std::cerr << "Index coord =" << ijk << std::endl;
      cs.moveTo(ijk);

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
}

TEST_F(TestMeanCurvature, testIntersection)
{
  using namespace openvdb;
  const Coord ijk(1,4,-9);
  FloatGrid grid(0.0f);
  auto acc = grid.getAccessor();
  math::GradStencil<FloatGrid> stencil(grid);
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
              EXPECT_EQ(7, int(grid.activeVoxelCount()));
              stencil.moveTo(ijk);
              const size_t count = mx + px + my + py + mz + pz;// number of intersections
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
}//testIntersection
