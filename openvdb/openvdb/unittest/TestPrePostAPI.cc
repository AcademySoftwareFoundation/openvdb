// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/math/Mat4.h>
#include <openvdb/math/Maps.h>
#include <openvdb/math/Transform.h>
#include <openvdb/util/MapsUtil.h>

#include <gtest/gtest.h>


class TestPrePostAPI: public ::testing::Test
{
};


TEST_F(TestPrePostAPI, testMat4)
{
    using namespace openvdb::math;

    double TOL = 1e-7;


    Mat4d m = Mat4d::identity();
    Mat4d minv = Mat4d::identity();

    // create matrix with pre-API
    // Translate Shear Rotate Translate Scale matrix
    m.preScale(Vec3d(1, 2, 3));
    m.preTranslate(Vec3d(2, 3, 4));
    m.preRotate(X_AXIS, 20);
    m.preShear(X_AXIS, Y_AXIS, 2);
    m.preTranslate(Vec3d(2, 2, 2));

    // create inverse using the post-API
    minv.postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
    minv.postTranslate(-Vec3d(2, 3, 4));
    minv.postRotate(X_AXIS,-20);
    minv.postShear(X_AXIS, Y_AXIS, -2);
    minv.postTranslate(-Vec3d(2, 2, 2));

    Mat4d mtest = minv * m;

    // verify that the results is an identity
    EXPECT_NEAR(mtest[0][0], 1, TOL);
    EXPECT_NEAR(mtest[1][1], 1, TOL);
    EXPECT_NEAR(mtest[2][2], 1, TOL);

    EXPECT_NEAR(mtest[0][1], 0, TOL);
    EXPECT_NEAR(mtest[0][2], 0, TOL);
    EXPECT_NEAR(mtest[0][3], 0, TOL);
    EXPECT_NEAR(mtest[1][0], 0, TOL);
    EXPECT_NEAR(mtest[1][2], 0, TOL);
    EXPECT_NEAR(mtest[1][3], 0, TOL);
    EXPECT_NEAR(mtest[2][0], 0, TOL);
    EXPECT_NEAR(mtest[2][1], 0, TOL);
    EXPECT_NEAR(mtest[2][3], 0, TOL);

    EXPECT_NEAR(mtest[3][0], 0, TOL);
    EXPECT_NEAR(mtest[3][1], 0, TOL);
    EXPECT_NEAR(mtest[3][2], 0, TOL);

    EXPECT_NEAR(mtest[3][3], 1, TOL);
}


TEST_F(TestPrePostAPI, testMat4Rotate)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    Mat4d rx, ry, rz;
    const double angle1 = 20. * M_PI / 180.;
    const double angle2 = 64. * M_PI / 180.;
    const double angle3 = 125. *M_PI / 180.;
    rx.setToRotation(Vec3d(1,0,0), angle1);
    ry.setToRotation(Vec3d(0,1,0), angle2);
    rz.setToRotation(Vec3d(0,0,1), angle3);

    Mat4d shear = Mat4d::identity();
    shear.setToShear(X_AXIS, Z_AXIS, 2.0);
    shear.preShear(Y_AXIS, X_AXIS, 3.0);
    shear.preTranslate(Vec3d(2,4,1));

    const Mat4d preResult = rz*ry*rx*shear;
    Mat4d mpre = shear;
    mpre.preRotate(X_AXIS, angle1);
    mpre.preRotate(Y_AXIS, angle2);
    mpre.preRotate(Z_AXIS, angle3);

    EXPECT_TRUE( mpre.eq(preResult, TOL) );

    const Mat4d postResult = shear*rx*ry*rz;
    Mat4d mpost = shear;
    mpost.postRotate(X_AXIS, angle1);
    mpost.postRotate(Y_AXIS, angle2);
    mpost.postRotate(Z_AXIS, angle3);

    EXPECT_TRUE( mpost.eq(postResult, TOL) );

    EXPECT_TRUE( !mpost.eq(mpre, TOL));

}


TEST_F(TestPrePostAPI, testMat4Scale)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    Mat4d mpre, mpost;
    double* pre  = mpre.asPointer();
    double* post = mpost.asPointer();
    for (int i = 0; i < 16; ++i) {
        pre[i] = double(i);
        post[i] = double(i);
    }

    Mat4d scale = Mat4d::identity();
    scale.setToScale(Vec3d(2, 3, 5.5));
    Mat4d preResult = scale * mpre;
    Mat4d postResult = mpost * scale;

    mpre.preScale(Vec3d(2, 3, 5.5));
    mpost.postScale(Vec3d(2, 3, 5.5));

    EXPECT_TRUE( mpre.eq(preResult, TOL) );
    EXPECT_TRUE( mpost.eq(postResult, TOL) );
}


TEST_F(TestPrePostAPI, testMat4Shear)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    Mat4d mpre, mpost;
    double* pre  = mpre.asPointer();
    double* post = mpost.asPointer();
    for (int i = 0; i < 16; ++i) {
        pre[i] = double(i);
        post[i] = double(i);
    }

    Mat4d shear = Mat4d::identity();
    shear.setToShear(X_AXIS, Z_AXIS, 13.);
    Mat4d preResult = shear * mpre;
    Mat4d postResult = mpost * shear;

    mpre.preShear(X_AXIS, Z_AXIS, 13.);
    mpost.postShear(X_AXIS, Z_AXIS, 13.);

    EXPECT_TRUE( mpre.eq(preResult, TOL) );
    EXPECT_TRUE( mpost.eq(postResult, TOL) );
}


TEST_F(TestPrePostAPI, testMaps)
{
    using namespace openvdb::math;

    double TOL = 1e-7;

    { // pre translate
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d trans(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.preTranslate(trans);
        {
            MapBase::Ptr base = usm.preTranslate(trans);
            Mat4d result = (base->getAffineMap())->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preTranslate(trans)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preTranslate(trans)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preTranslate(trans)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.preTranslate(trans)->getAffineMap()->getConstMat4();
        EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
    { // post translate
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d trans(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.postTranslate(trans);
        {
            const Mat4d result = usm.postTranslate(trans)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.postTranslate(trans)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postTranslate(trans)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postTranslate(trans)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.postTranslate(trans)->getAffineMap()->getConstMat4();
        EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
    { // pre scale
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d scale(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.preScale(scale);
        {
            const Mat4d result = usm.preScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.preScale(scale)->getAffineMap()->getConstMat4();
        EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
    { // post scale
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        const Vec3d scale(1,2,3);
        Mat4d correct = Mat4d::identity();
        correct.postScale(scale);
        {
            const Mat4d result = usm.postScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.postScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postScale(scale)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.postScale(scale)->getAffineMap()->getConstMat4();
        EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
    { // pre shear
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.preShear(X_AXIS, Z_AXIS, 13.);
        {
            const Mat4d result = usm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
        const Mat4d result = am.preShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
        EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
    { // post shear
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.postShear(X_AXIS, Z_AXIS, 13.);
        {
            const Mat4d result = usm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result =
                ustm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = am.postShear(13., X_AXIS, Z_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
    { // pre rotate
        const double angle1 = 20. * M_PI / 180.;
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.preRotate(X_AXIS, angle1);
        {
            const Mat4d result = usm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = am.preRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
    { // post rotate
        const double angle1 = 20. * M_PI / 180.;
        UniformScaleMap usm;
        UniformScaleTranslateMap ustm;
        ScaleMap sm;
        ScaleTranslateMap stm;
        AffineMap am;

        Mat4d correct = Mat4d::identity();
        correct.postRotate(X_AXIS, angle1);
        {
            const Mat4d result = usm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = ustm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = sm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = stm.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
        {
            const Mat4d result = am.postRotate(angle1, X_AXIS)->getAffineMap()->getConstMat4();
            EXPECT_TRUE( correct.eq(result, TOL));
        }
    }
}


TEST_F(TestPrePostAPI, testLinearTransform)
{
    using namespace openvdb::math;

    double TOL = 1e-7;
    {
        Transform::Ptr t = Transform::createLinearTransform(1.f);
        Transform::Ptr tinv = Transform::createLinearTransform(1.f);

        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        t->preScale(Vec3d(1, 2, 3));
        t->preTranslate(Vec3d(2, 3, 4));
        t->preRotate(20);
        t->preShear(2, X_AXIS, Y_AXIS);
        t->preTranslate(Vec3d(2, 2, 2));

        // create inverse using the post-API
        tinv->postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        tinv->postTranslate(-Vec3d(2, 3, 4));
        tinv->postRotate(-20);
        tinv->postShear(-2, X_AXIS, Y_AXIS);
        tinv->postTranslate(-Vec3d(2, 2, 2));


        // test this by verifying that equvilent interal matrix
        // represenations are inverses
        Mat4d m = t->baseMap()->getAffineMap()->getMat4();
        Mat4d minv = tinv->baseMap()->getAffineMap()->getMat4();

        Mat4d mtest = minv * m;

        // verify that the results is an identity
        EXPECT_NEAR(mtest[0][0], 1, TOL);
        EXPECT_NEAR(mtest[1][1], 1, TOL);
        EXPECT_NEAR(mtest[2][2], 1, TOL);

        EXPECT_NEAR(mtest[0][1], 0, TOL);
        EXPECT_NEAR(mtest[0][2], 0, TOL);
        EXPECT_NEAR(mtest[0][3], 0, TOL);
        EXPECT_NEAR(mtest[1][0], 0, TOL);
        EXPECT_NEAR(mtest[1][2], 0, TOL);
        EXPECT_NEAR(mtest[1][3], 0, TOL);
        EXPECT_NEAR(mtest[2][0], 0, TOL);
        EXPECT_NEAR(mtest[2][1], 0, TOL);
        EXPECT_NEAR(mtest[2][3], 0, TOL);

        EXPECT_NEAR(mtest[3][0], 0, TOL);
        EXPECT_NEAR(mtest[3][1], 0, TOL);
        EXPECT_NEAR(mtest[3][2], 0, TOL);

        EXPECT_NEAR(mtest[3][3], 1, TOL);
    }

    {
        Transform::Ptr t = Transform::createLinearTransform(1.f);

        Mat4d m = Mat4d::identity();

        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        m.preScale(Vec3d(1, 2, 3));
        m.preTranslate(Vec3d(2, 3, 4));
        m.preRotate(X_AXIS, 20);
        m.preShear(X_AXIS, Y_AXIS, 2);
        m.preTranslate(Vec3d(2, 2, 2));

        t->preScale(Vec3d(1,2,3));
        t->preMult(m);
        t->postMult(m);

        Mat4d minv = Mat4d::identity();

        // create inverse using the post-API
        minv.postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        minv.postTranslate(-Vec3d(2, 3, 4));
        minv.postRotate(X_AXIS,-20);
        minv.postShear(X_AXIS, Y_AXIS, -2);
        minv.postTranslate(-Vec3d(2, 2, 2));

        t->preMult(minv);
        t->postMult(minv);

        Mat4d mtest = t->baseMap()->getAffineMap()->getMat4();


        // verify that the results is the scale
        EXPECT_NEAR(mtest[0][0], 1, TOL);
        EXPECT_NEAR(mtest[1][1], 2, TOL);
        EXPECT_NEAR(mtest[2][2], 3, 1e-6);

        EXPECT_NEAR(mtest[0][1], 0, TOL);
        EXPECT_NEAR(mtest[0][2], 0, TOL);
        EXPECT_NEAR(mtest[0][3], 0, TOL);
        EXPECT_NEAR(mtest[1][0], 0, TOL);
        EXPECT_NEAR(mtest[1][2], 0, TOL);
        EXPECT_NEAR(mtest[1][3], 0, TOL);
        EXPECT_NEAR(mtest[2][0], 0, TOL);
        EXPECT_NEAR(mtest[2][1], 0, TOL);
        EXPECT_NEAR(mtest[2][3], 0, TOL);

        EXPECT_NEAR(mtest[3][0], 0, 1e-6);
        EXPECT_NEAR(mtest[3][1], 0, 1e-6);
        EXPECT_NEAR(mtest[3][2], 0, TOL);

        EXPECT_NEAR(mtest[3][3], 1, TOL);
    }


}


TEST_F(TestPrePostAPI, testFrustumTransform)
{
    using namespace openvdb::math;

    using BBoxd = BBox<Vec3d>;

    double TOL = 1e-7;
    {

        BBoxd bbox(Vec3d(-5,-5,0), Vec3d(5,5,10));
        Transform::Ptr t = Transform::createFrustumTransform(
            bbox, /* taper*/ 1, /*depth*/10, /* voxel size */1.f);
        Transform::Ptr tinv = Transform::createFrustumTransform(
            bbox, /* taper*/ 1, /*depth*/10, /* voxel size */1.f);


        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        t->preScale(Vec3d(1, 2, 3));
        t->preTranslate(Vec3d(2, 3, 4));
        t->preRotate(20);
        t->preShear(2, X_AXIS, Y_AXIS);
        t->preTranslate(Vec3d(2, 2, 2));

        // create inverse using the post-API
        tinv->postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        tinv->postTranslate(-Vec3d(2, 3, 4));
        tinv->postRotate(-20);
        tinv->postShear(-2, X_AXIS, Y_AXIS);
        tinv->postTranslate(-Vec3d(2, 2, 2));


        // test this by verifying that equvilent interal matrix
        // represenations are inverses
        NonlinearFrustumMap::Ptr frustum =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(t->baseMap());
        NonlinearFrustumMap::Ptr frustuminv =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(tinv->baseMap());

        Mat4d m = frustum->secondMap().getMat4();
        Mat4d minv = frustuminv->secondMap().getMat4();

        Mat4d mtest = minv * m;

        // verify that the results is an identity
        EXPECT_NEAR(mtest[0][0], 1, TOL);
        EXPECT_NEAR(mtest[1][1], 1, TOL);
        EXPECT_NEAR(mtest[2][2], 1, TOL);

        EXPECT_NEAR(mtest[0][1], 0, TOL);
        EXPECT_NEAR(mtest[0][2], 0, TOL);
        EXPECT_NEAR(mtest[0][3], 0, TOL);
        EXPECT_NEAR(mtest[1][0], 0, TOL);
        EXPECT_NEAR(mtest[1][2], 0, TOL);
        EXPECT_NEAR(mtest[1][3], 0, TOL);
        EXPECT_NEAR(mtest[2][0], 0, TOL);
        EXPECT_NEAR(mtest[2][1], 0, TOL);
        EXPECT_NEAR(mtest[2][3], 0, TOL);

        EXPECT_NEAR(mtest[3][0], 0, TOL);
        EXPECT_NEAR(mtest[3][1], 0, TOL);
        EXPECT_NEAR(mtest[3][2], 0, TOL);

        EXPECT_NEAR(mtest[3][3], 1, TOL);
    }

    {

        BBoxd bbox(Vec3d(-5,-5,0), Vec3d(5,5,10));
        Transform::Ptr t = Transform::createFrustumTransform(
            bbox, /* taper*/ 1, /*depth*/10, /* voxel size */1.f);


        Mat4d m = Mat4d::identity();

        // create matrix with pre-API
        // Translate Shear Rotate Translate Scale matrix
        m.preScale(Vec3d(1, 2, 3));
        m.preTranslate(Vec3d(2, 3, 4));
        m.preRotate(X_AXIS, 20);
        m.preShear(X_AXIS, Y_AXIS, 2);
        m.preTranslate(Vec3d(2, 2, 2));

        t->preScale(Vec3d(1,2,3));
        t->preMult(m);
        t->postMult(m);

        Mat4d minv = Mat4d::identity();

        // create inverse using the post-API
        minv.postScale(Vec3d(1.f, 1.f/2.f, 1.f/3.f));
        minv.postTranslate(-Vec3d(2, 3, 4));
        minv.postRotate(X_AXIS,-20);
        minv.postShear(X_AXIS, Y_AXIS, -2);
        minv.postTranslate(-Vec3d(2, 2, 2));

        t->preMult(minv);
        t->postMult(minv);

        NonlinearFrustumMap::Ptr frustum =
            openvdb::StaticPtrCast<NonlinearFrustumMap, MapBase>(t->baseMap());
        Mat4d mtest = frustum->secondMap().getMat4();

        // verify that the results is the scale
        EXPECT_NEAR(mtest[0][0], 1, TOL);
        EXPECT_NEAR(mtest[1][1], 2, TOL);
        EXPECT_NEAR(mtest[2][2], 3, 1e-6);

        EXPECT_NEAR(mtest[0][1], 0, TOL);
        EXPECT_NEAR(mtest[0][2], 0, TOL);
        EXPECT_NEAR(mtest[0][3], 0, TOL);
        EXPECT_NEAR(mtest[1][0], 0, TOL);
        EXPECT_NEAR(mtest[1][2], 0, TOL);
        EXPECT_NEAR(mtest[1][3], 0, TOL);
        EXPECT_NEAR(mtest[2][0], 0, TOL);
        EXPECT_NEAR(mtest[2][1], 0, TOL);
        EXPECT_NEAR(mtest[2][3], 0, TOL);

        EXPECT_NEAR(mtest[3][0], 0, 1e-6);
        EXPECT_NEAR(mtest[3][1], 0, 1e-6);
        EXPECT_NEAR(mtest[3][2], 0, TOL);

        EXPECT_NEAR(mtest[3][3], 1, TOL);
    }


}
