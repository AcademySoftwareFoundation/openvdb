// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/math/Maps.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/MapsUtil.h>
#include <gtest/gtest.h>


class TestMaps: public ::testing::Test
{
};


// Work-around for a macro expansion bug in debug mode that presents as an
// undefined reference linking error. This happens in cases where template
// instantiation of a type trait is prevented from occurring as the template
// instantiation is deemed to be in an unreachable code block.
// The work-around is to wrap the EXPECT_TRUE macro in another which holds
// the expected value in a temporary.
#define EXPECT_TRUE_TEMP(expected) \
    { bool result = expected; EXPECT_TRUE(result); }


TEST_F(TestMaps, testApproxInverse)
{
    using namespace openvdb::math;

    Mat4d singular = Mat4d::identity();
    singular[1][1] = 0.f;
    {
        Mat4d singularInv = approxInverse(singular);

        EXPECT_TRUE( singular == singularInv );
    }
    {
        Mat4d rot = Mat4d::identity();
        rot.setToRotation(X_AXIS, openvdb::math::pi<double>()/4.);

        Mat4d rotInv = rot.inverse();
        Mat4d mat = rotInv * singular * rot;

        Mat4d singularInv = approxInverse(mat);

        // this matrix is equal to its own singular inverse
        EXPECT_TRUE( mat.eq(singularInv) );

    }
    {
        Mat4d m = Mat4d::identity();
        m[0][1] = 1;

        // should give true inverse, since this matrix has det=1
        Mat4d minv = approxInverse(m);

        Mat4d prod = m * minv;
        EXPECT_TRUE( prod.eq( Mat4d::identity() ) );
    }
    {
        Mat4d m = Mat4d::identity();
        m[0][1] = 1;
        m[1][1] = 0;
        // should give true inverse, since this matrix has det=1
        Mat4d minv = approxInverse(m);

        Mat4d expected = Mat4d::zero();
        expected[3][3] = 1;
        EXPECT_TRUE( minv.eq(expected ) );
    }


}


TEST_F(TestMaps, testUniformScale)
{
    using namespace openvdb::math;

    AffineMap map;

    EXPECT_TRUE(map.hasUniformScale());

    // Apply uniform scale: should still have square voxels
    map.accumPreScale(Vec3d(2, 2, 2));

    EXPECT_TRUE(map.hasUniformScale());

    // Apply a rotation, should still have squaure voxels.
    map.accumPostRotation(X_AXIS, 2.5);

    EXPECT_TRUE(map.hasUniformScale());

    // non uniform scaling will stretch the voxels
    map.accumPostScale(Vec3d(1, 3, 1) );

    EXPECT_TRUE(!map.hasUniformScale());
}

TEST_F(TestMaps, testTranslation)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    TranslationMap::Ptr translation(new TranslationMap(Vec3d(1,1,1)));
    EXPECT_TRUE_TEMP(is_linear<TranslationMap>::value);

    TranslationMap another_translation(Vec3d(1,1,1));
    EXPECT_TRUE(another_translation == *translation);

    TranslationMap::Ptr translate_by_two(new TranslationMap(Vec3d(2,2,2)));

    EXPECT_TRUE(*translate_by_two != *translation);

    EXPECT_NEAR(translate_by_two->determinant(), 1, TOL);

    EXPECT_TRUE(translate_by_two->hasUniformScale());

    /// apply the map forward
    Vec3d unit(1,0,0);
    Vec3d result = translate_by_two->applyMap(unit);
    EXPECT_NEAR(result(0), 3, TOL);
    EXPECT_NEAR(result(1), 2, TOL);
    EXPECT_NEAR(result(2), 2, TOL);

    /// invert the map
    result = translate_by_two->applyInverseMap(result);
    EXPECT_NEAR(result(0), 1, TOL);
    EXPECT_NEAR(result(1), 0, TOL);
    EXPECT_NEAR(result(2), 0, TOL);

    /// Inverse Jacobian Transpose
    result = translate_by_two->applyIJT(result);
    EXPECT_NEAR(result(0), 1, TOL);
    EXPECT_NEAR(result(1), 0, TOL);
    EXPECT_NEAR(result(2), 0, TOL);

    /// Jacobian Transpose
    result = translate_by_two->applyJT(translate_by_two->applyIJT(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);


    MapBase::Ptr inverse = translation->inverseMap();
    EXPECT_TRUE(inverse->type() == TranslationMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(translation->applyMap(unit));
    EXPECT_NEAR(result(0), 1, TOL);
    EXPECT_NEAR(result(1), 0, TOL);
    EXPECT_NEAR(result(2), 0, TOL);


}

TEST_F(TestMaps, testScaleDefault)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    // testing default constructor
    // should be the identity
    ScaleMap::Ptr  scale(new ScaleMap());
    Vec3d unit(1, 1, 1);

    Vec3d result = scale->applyMap(unit);

    EXPECT_NEAR(unit(0), result(0), TOL);
    EXPECT_NEAR(unit(1), result(1), TOL);
    EXPECT_NEAR(unit(2), result(2), TOL);

    result = scale->applyInverseMap(unit);

    EXPECT_NEAR(unit(0), result(0), TOL);
    EXPECT_NEAR(unit(1), result(1), TOL);
    EXPECT_NEAR(unit(2), result(2), TOL);


    MapBase::Ptr inverse = scale->inverseMap();
    EXPECT_TRUE(inverse->type() == ScaleMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(scale->applyMap(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);


}

TEST_F(TestMaps, testRotation)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    double pi = 4.*atan(1.);
    UnitaryMap::Ptr  rotation(new UnitaryMap(Vec3d(1,0,0), pi/2));

    EXPECT_TRUE_TEMP(is_linear<UnitaryMap>::value);

    UnitaryMap another_rotation(Vec3d(1,0,0), pi/2.);
    EXPECT_TRUE(another_rotation == *rotation);

    UnitaryMap::Ptr rotation_two(new UnitaryMap(Vec3d(1,0,0), pi/4.));

    EXPECT_TRUE(*rotation_two != *rotation);

    EXPECT_NEAR(rotation->determinant(), 1, TOL);

    EXPECT_TRUE(rotation_two->hasUniformScale());

    /// apply the map forward
    Vec3d unit(0,1,0);
    Vec3d result = rotation->applyMap(unit);
    EXPECT_NEAR(0, result(0), TOL);
    EXPECT_NEAR(0, result(1), TOL);
    EXPECT_NEAR(1, result(2), TOL);

    /// invert the map
    result = rotation->applyInverseMap(result);
    EXPECT_NEAR(0, result(0), TOL);
    EXPECT_NEAR(1, result(1), TOL);
    EXPECT_NEAR(0, result(2), TOL);

    /// Inverse Jacobian Transpose
    result = rotation_two->applyIJT(result); // rotate backwards
    EXPECT_NEAR(0, result(0), TOL);
    EXPECT_NEAR(sqrt(2.)/2, result(1), TOL);
    EXPECT_NEAR(sqrt(2.)/2, result(2), TOL);

    /// Jacobian Transpose
    result = rotation_two->applyJT(rotation_two->applyIJT(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);


    // Test inverse map
    MapBase::Ptr inverse = rotation->inverseMap();
    EXPECT_TRUE(inverse->type() == UnitaryMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(rotation->applyMap(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);
}


TEST_F(TestMaps, testScaleTranslate)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    EXPECT_TRUE_TEMP(is_linear<ScaleTranslateMap>::value);

    TranslationMap::Ptr  translation(new TranslationMap(Vec3d(1,1,1)));
    ScaleMap::Ptr  scale(new ScaleMap(Vec3d(1,2,3)));

    ScaleTranslateMap::Ptr scaleAndTranslate(
        new ScaleTranslateMap(*scale, *translation));

    TranslationMap translate_by_two(Vec3d(2,2,2));
    ScaleTranslateMap another_scaleAndTranslate(*scale, translate_by_two);

    EXPECT_TRUE(another_scaleAndTranslate != *scaleAndTranslate);

    EXPECT_TRUE(!scaleAndTranslate->hasUniformScale());
    //EXPECT_NEAR(scaleAndTranslate->determinant(), 6, TOL);

    /// apply the map forward
    Vec3d unit(1,0,0);
    Vec3d result = scaleAndTranslate->applyMap(unit);
    EXPECT_NEAR(2, result(0), TOL);
    EXPECT_NEAR(1, result(1), TOL);
    EXPECT_NEAR(1, result(2), TOL);

    /// invert the map
    result = scaleAndTranslate->applyInverseMap(result);
    EXPECT_NEAR(1, result(0), TOL);
    EXPECT_NEAR(0, result(1), TOL);
    EXPECT_NEAR(0, result(2), TOL);

    /// Inverse Jacobian Transpose
    result = Vec3d(0,2,0);
    result = scaleAndTranslate->applyIJT(result );
    EXPECT_NEAR(0, result(0), TOL);
    EXPECT_NEAR(1, result(1), TOL);
    EXPECT_NEAR(0, result(2), TOL);


    /// Jacobian Transpose
    result = scaleAndTranslate->applyJT(scaleAndTranslate->applyIJT(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);


    // Test inverse map
    MapBase::Ptr inverse = scaleAndTranslate->inverseMap();
    EXPECT_TRUE(inverse->type() == ScaleTranslateMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(scaleAndTranslate->applyMap(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);

}


TEST_F(TestMaps, testUniformScaleTranslate)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    EXPECT_TRUE_TEMP(is_linear<UniformScaleMap>::value);
    EXPECT_TRUE_TEMP(is_linear<UniformScaleTranslateMap>::value);

    TranslationMap::Ptr  translation(new TranslationMap(Vec3d(1,1,1)));
    UniformScaleMap::Ptr  scale(new UniformScaleMap(2));

    UniformScaleTranslateMap::Ptr scaleAndTranslate(
        new UniformScaleTranslateMap(*scale, *translation));

    TranslationMap translate_by_two(Vec3d(2,2,2));
    UniformScaleTranslateMap another_scaleAndTranslate(*scale, translate_by_two);

    EXPECT_TRUE(another_scaleAndTranslate != *scaleAndTranslate);
    EXPECT_TRUE(scaleAndTranslate->hasUniformScale());
    //EXPECT_NEAR(scaleAndTranslate->determinant(), 6, TOL);

    /// apply the map forward
    Vec3d unit(1,0,0);
    Vec3d result = scaleAndTranslate->applyMap(unit);
    EXPECT_NEAR(3, result(0), TOL);
    EXPECT_NEAR(1, result(1), TOL);
    EXPECT_NEAR(1, result(2), TOL);

    /// invert the map
    result = scaleAndTranslate->applyInverseMap(result);
    EXPECT_NEAR(1, result(0), TOL);
    EXPECT_NEAR(0, result(1), TOL);
    EXPECT_NEAR(0, result(2), TOL);

    /// Inverse Jacobian Transpose
    result = Vec3d(0,2,0);
    result = scaleAndTranslate->applyIJT(result );
    EXPECT_NEAR(0, result(0), TOL);
    EXPECT_NEAR(1, result(1), TOL);
    EXPECT_NEAR(0, result(2), TOL);


    /// Jacobian Transpose
    result = scaleAndTranslate->applyJT(scaleAndTranslate->applyIJT(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);



    // Test inverse map
    MapBase::Ptr inverse = scaleAndTranslate->inverseMap();
    EXPECT_TRUE(inverse->type() == UniformScaleTranslateMap::mapType());
    // apply the map forward and the inverse map back
    result = inverse->applyMap(scaleAndTranslate->applyMap(unit));
    EXPECT_NEAR(result(0), unit(0), TOL);
    EXPECT_NEAR(result(1), unit(1), TOL);
    EXPECT_NEAR(result(2), unit(2), TOL);

}


TEST_F(TestMaps, testDecomposition)
{
    using namespace openvdb::math;

    //double TOL = 1e-7;

    EXPECT_TRUE_TEMP(is_linear<UnitaryMap>::value);
    EXPECT_TRUE_TEMP(is_linear<SymmetricMap>::value);
    EXPECT_TRUE_TEMP(is_linear<PolarDecomposedMap>::value);
    EXPECT_TRUE_TEMP(is_linear<FullyDecomposedMap>::value);

    Mat4d matrix(Mat4d::identity());
    Vec3d input_translation(0,0,1);
    matrix.setTranslation(input_translation);


    matrix(0,0) =  1.8930039;
    matrix(1,0) = -0.120080537;
    matrix(2,0) = -0.497615212;

    matrix(0,1) = -0.120080537;
    matrix(1,1) =  2.643265436;
    matrix(2,1) = 0.6176957495;

    matrix(0,2) = -0.497615212;
    matrix(1,2) =  0.6176957495;
    matrix(2,2) = 1.4637305884;

    FullyDecomposedMap::Ptr decomp = createFullyDecomposedMap(matrix);

    /// the singular values
    const Vec3<double>& singular_values =
        decomp->firstMap().firstMap().secondMap().getScale();
    /// expected values
    Vec3d expected_values(2, 3, 1);

    EXPECT_TRUE( isApproxEqual(singular_values, expected_values) );

    const Vec3<double>& the_translation = decomp->secondMap().secondMap().getTranslation();
    EXPECT_TRUE( isApproxEqual(the_translation, input_translation));
}


TEST_F(TestMaps, testFrustum)
{
    using namespace openvdb::math;

    openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
    NonlinearFrustumMap frustum(bbox, 1./6., 5);
    /// frustum will have depth, far plane - near plane = 5
    /// the frustum has width 1 in the front and 6 in the back

    Vec3d trans(2,2,2);
    NonlinearFrustumMap::Ptr map =
        openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(
            frustum.preScale(Vec3d(10,10,10))->postTranslate(trans));

    EXPECT_TRUE(!map->hasUniformScale());

    Vec3d result;
    result = map->voxelSize();

    EXPECT_TRUE( isApproxEqual(result.x(), 0.1));
    EXPECT_TRUE( isApproxEqual(result.y(), 0.1));
    EXPECT_TRUE( isApproxEqual(result.z(), 0.5, 0.0001));
    //--------- Front face
    Vec3d corner(0,0,0);
    result  = map->applyMap(corner);
    EXPECT_TRUE(isApproxEqual(result, Vec3d(-5, -5, 0) + trans));

    corner = Vec3d(100,0,0);
    result  = map->applyMap(corner);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(5, -5, 0) + trans));

    corner = Vec3d(0,100,0);
    result  = map->applyMap(corner);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(-5, 5, 0) + trans));

    corner = Vec3d(100,100,0);
    result  = map->applyMap(corner);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(5, 5, 0) + trans));

    //--------- Back face
    corner = Vec3d(0,0,100);
    result  = map->applyMap(corner);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(-30, -30, 50) + trans)); // 10*(5/2 + 1/2) = 30

    corner = Vec3d(100,0,100);
    result  = map->applyMap(corner);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(30, -30, 50) + trans));

    corner = Vec3d(0,100,100);
    result  = map->applyMap(corner);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(-30, 30, 50) + trans));

    corner = Vec3d(100,100,100);
    result  = map->applyMap(corner);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(30, 30, 50) + trans));


    // invert a single corner
    result = map->applyInverseMap(Vec3d(30,30,50) + trans);
    EXPECT_TRUE( isApproxEqual(result, Vec3d(100, 100, 100)));

    EXPECT_TRUE(map->hasSimpleAffine());

    /// create a frustum from from camera type information

    // the location of the camera
    Vec3d position(100,10,1);
    // the direction the camera is pointing
    Vec3d direction(0,1,1);
    direction.normalize();

    // the up-direction for the camera
    Vec3d up(10,3,-3);

    // distance from camera to near-plane measured in the direction 'direction'
    double z_near = 100.;
    // depth of frustum to far-plane to near-plane
    double depth = 500.;
    //aspect ratio of frustum: width/height
    double aspect = 2;

    // voxel count in frustum.  the y_count = x_count / aspect
    Coord::ValueType x_count = 500;
    Coord::ValueType z_count = 5000;


    NonlinearFrustumMap frustumMap_from_camera(
        position, direction, up, aspect, z_near, depth, x_count, z_count);
    Vec3d center;
    // find the center of the near plane and make sure it is in the correct place
    center = Vec3d(0,0,0);
    center += frustumMap_from_camera.applyMap(Vec3d(0,0,0));
    center += frustumMap_from_camera.applyMap(Vec3d(500,0,0));
    center += frustumMap_from_camera.applyMap(Vec3d(0,250,0));
    center +=  frustumMap_from_camera.applyMap(Vec3d(500,250,0));
    center = center /4.;
    EXPECT_TRUE( isApproxEqual(center, position + z_near * direction));
    // find the center of the far plane and make sure it is in the correct place
    center = Vec3d(0,0,0);
    center += frustumMap_from_camera.applyMap(Vec3d(  0,  0,5000));
    center += frustumMap_from_camera.applyMap(Vec3d(500,  0,5000));
    center += frustumMap_from_camera.applyMap(Vec3d(  0,250,5000));
    center += frustumMap_from_camera.applyMap(Vec3d(500,250,5000));
    center = center /4.;
    EXPECT_TRUE( isApproxEqual(center, position + (z_near+depth) * direction));
    // check that the frustum has the correct heigh on the near plane
    Vec3d corner1  = frustumMap_from_camera.applyMap(Vec3d(0,0,0));
    Vec3d corner2  = frustumMap_from_camera.applyMap(Vec3d(0,250,0));
    Vec3d side = corner2-corner1;
    EXPECT_TRUE( isApproxEqual( side.length(), 2 * up.length()));
    // check that the frustum is correctly oriented w.r.t up
    side.normalize();
    EXPECT_TRUE( isApproxEqual( side * (up.length()), up));
    // check that the linear map inside the frustum is a simple affine map (i.e. has no shear)
    EXPECT_TRUE(frustumMap_from_camera.hasSimpleAffine());
}


TEST_F(TestMaps, testCalcBoundingBox)
{
    using namespace openvdb::math;

    openvdb::BBoxd world_bbox(Vec3d(0,0,0), Vec3d(1,1,1));
    openvdb::BBoxd voxel_bbox;
    openvdb::BBoxd expected;
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,2,2));

        openvdb::util::calculateBounds<AffineMap>(affine, world_bbox, voxel_bbox);

        expected = openvdb::BBoxd(Vec3d(0,0,0), Vec3d(0.5, 0.5, 0.5));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.min(), expected.min()));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.max(), expected.max()));

        affine.accumPostTranslation(Vec3d(1,1,1));
        openvdb::util::calculateBounds<AffineMap>(affine, world_bbox, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-0.5,-0.5,-0.5), Vec3d(0, 0, 0));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.min(), expected.min()));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.max(), expected.max()));
    }
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,2,2));
        affine.accumPostTranslation(Vec3d(1,1,1));
        // test a sphere:
        Vec3d center(0,0,0);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5.5,-5.5,-5.5), Vec3d(4.5, 4.5, 4.5));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.min(), expected.min()));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.max(), expected.max()));
    }
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,2,2));
        double pi = 4.*atan(1.);
        affine.accumPreRotation(X_AXIS, pi/4.);
        Vec3d center(0,0,0);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5,-5,-5), Vec3d(5, 5, 5));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.min(), expected.min()));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.max(), expected.max()));
    }
    {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,1,1));
        double pi = 4.*atan(1.);
        affine.accumPreRotation(X_AXIS, pi/4.);
        Vec3d center(0,0,0);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5,-10,-10), Vec3d(5, 10, 10));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.min(), expected.min()));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.max(), expected.max()));
     }
     {
        AffineMap affine;
        affine.accumPreScale(Vec3d(2,1,1));
        double pi = 4.*atan(1.);
        affine.accumPreRotation(X_AXIS, pi/4.);
        affine.accumPostTranslation(Vec3d(1,1,1));
        Vec3d center(1,1,1);
        double radius = 10;

        openvdb::util::calculateBounds<AffineMap>(affine, center, radius, voxel_bbox);
        expected = openvdb::BBoxd(Vec3d(-5,-10,-10), Vec3d(5, 10, 10));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.min(), expected.min()));
        EXPECT_TRUE(isApproxEqual(voxel_bbox.max(), expected.max()));
     }
     {
         openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
         NonlinearFrustumMap frustum(bbox, 2, 5);
         NonlinearFrustumMap::Ptr map =
             openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(
                 frustum.preScale(Vec3d(2,2,2)));
         Vec3d center(20,20,10);
         double radius(1);

         openvdb::util::calculateBounds<NonlinearFrustumMap>(*map, center, radius, voxel_bbox);
     }
}
TEST_F(TestMaps, testJacobians)
{
    using namespace openvdb::math;
    const double TOL = 1e-7;
    {
        AffineMap affine;

        const int n = 10;
        const double dtheta = openvdb::math::pi<double>() / n;

        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);

        for (int i = 0; i < n; ++i) {
            double theta = i * dtheta;

            affine.accumPostRotation(X_AXIS, theta);

            Vec3d result = affine.applyJacobian(test);
            Vec3d expected = affine.applyMap(test) - affine.applyMap(origin);

            EXPECT_NEAR(result(0), expected(0), TOL);
            EXPECT_NEAR(result(1), expected(1), TOL);
            EXPECT_NEAR(result(2), expected(2), TOL);

            Vec3d tmp = affine.applyInverseJacobian(result);

            EXPECT_NEAR(tmp(0), test(0), TOL);
            EXPECT_NEAR(tmp(1), test(1), TOL);
            EXPECT_NEAR(tmp(2), test(2), TOL);
        }
    }

    {
        UniformScaleMap scale(3);
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = scale.applyJacobian(test);
        Vec3d expected = scale.applyMap(test) - scale.applyMap(origin);

        EXPECT_NEAR(result(0), expected(0), TOL);
        EXPECT_NEAR(result(1), expected(1), TOL);
        EXPECT_NEAR(result(2), expected(2), TOL);

        Vec3d tmp = scale.applyInverseJacobian(result);

        EXPECT_NEAR(tmp(0), test(0), TOL);
        EXPECT_NEAR(tmp(1), test(1), TOL);
        EXPECT_NEAR(tmp(2), test(2), TOL);
    }

    {
        ScaleMap scale(Vec3d(1,2,3));
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = scale.applyJacobian(test);
        Vec3d expected = scale.applyMap(test) - scale.applyMap(origin);

        EXPECT_NEAR(result(0), expected(0), TOL);
        EXPECT_NEAR(result(1), expected(1), TOL);
        EXPECT_NEAR(result(2), expected(2), TOL);

        Vec3d tmp = scale.applyInverseJacobian(result);

        EXPECT_NEAR(tmp(0), test(0), TOL);
        EXPECT_NEAR(tmp(1), test(1), TOL);
        EXPECT_NEAR(tmp(2), test(2), TOL);
    }
    {
        TranslationMap map(Vec3d(1,2,3));
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = map.applyJacobian(test);
        Vec3d expected = map.applyMap(test) - map.applyMap(origin);

        EXPECT_NEAR(result(0), expected(0), TOL);
        EXPECT_NEAR(result(1), expected(1), TOL);
        EXPECT_NEAR(result(2), expected(2), TOL);

        Vec3d tmp = map.applyInverseJacobian(result);

        EXPECT_NEAR(tmp(0), test(0), TOL);
        EXPECT_NEAR(tmp(1), test(1), TOL);
        EXPECT_NEAR(tmp(2), test(2), TOL);
    }
    {
        ScaleTranslateMap map(Vec3d(1,2,3), Vec3d(3,5,4));
        const Vec3d test(1,2,3);
        const Vec3d origin(0,0,0);


        Vec3d result = map.applyJacobian(test);
        Vec3d expected = map.applyMap(test) - map.applyMap(origin);

        EXPECT_NEAR(result(0), expected(0), TOL);
        EXPECT_NEAR(result(1), expected(1), TOL);
        EXPECT_NEAR(result(2), expected(2), TOL);

        Vec3d tmp = map.applyInverseJacobian(result);

        EXPECT_NEAR(tmp(0), test(0), TOL);
        EXPECT_NEAR(tmp(1), test(1), TOL);
        EXPECT_NEAR(tmp(2), test(2), TOL);
    }
    {
        openvdb::BBoxd bbox(Vec3d(0), Vec3d(100));
        NonlinearFrustumMap frustum(bbox, 1./6., 5);
        /// frustum will have depth, far plane - near plane = 5
        /// the frustum has width 1 in the front and 6 in the back

        Vec3d trans(2,2,2);
        NonlinearFrustumMap::Ptr map =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(
                frustum.preScale(Vec3d(10,10,10))->postTranslate(trans));

        const Vec3d test(1,2,3);
        const Vec3d origin(0, 0, 0);

        // these two drop down to just the linear part
        Vec3d lresult = map->applyJacobian(test);
        Vec3d ltmp = map->applyInverseJacobian(lresult);

        EXPECT_NEAR(ltmp(0), test(0), TOL);
        EXPECT_NEAR(ltmp(1), test(1), TOL);
        EXPECT_NEAR(ltmp(2), test(2), TOL);

        Vec3d isloc(4,5,6);
        // these two drop down to just the linear part
        Vec3d result = map->applyJacobian(test, isloc);
        Vec3d tmp = map->applyInverseJacobian(result, isloc);

        EXPECT_NEAR(tmp(0), test(0), TOL);
        EXPECT_NEAR(tmp(1), test(1), TOL);
        EXPECT_NEAR(tmp(2), test(2), TOL);



    }


}
