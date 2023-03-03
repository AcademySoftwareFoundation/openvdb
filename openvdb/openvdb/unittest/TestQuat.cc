// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Quat.h>
#include <openvdb/math/Mat4.h>

#include <gtest/gtest.h>


using namespace openvdb::math;

class TestQuat: public ::testing::Test
{
};


TEST_F(TestQuat, testConstructor)
{
    {
        Quat<float> qq(1.23f, 2.34f, 3.45f, 4.56f);
        EXPECT_TRUE( isExactlyEqual(qq.x(), 1.23f) );
        EXPECT_TRUE( isExactlyEqual(qq.y(), 2.34f) );
        EXPECT_TRUE( isExactlyEqual(qq.z(), 3.45f) );
        EXPECT_TRUE( isExactlyEqual(qq.w(), 4.56f) );
    }

    {
        float a[] = { 1.23f, 2.34f, 3.45f, 4.56f };
        Quat<float> qq(a);
        EXPECT_TRUE( isExactlyEqual(qq.x(), 1.23f) );
        EXPECT_TRUE( isExactlyEqual(qq.y(), 2.34f) );
        EXPECT_TRUE( isExactlyEqual(qq.z(), 3.45f) );
        EXPECT_TRUE( isExactlyEqual(qq.w(), 4.56f) );
    }
}


TEST_F(TestQuat, testAxisAngle)
{
    float TOL = 1e-6f;

    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Vec3s v(1, 2, 3);
    v.normalize();
    float a = float(openvdb::math::pi<float>() / 4.f);

    Quat<float> q(v,a);
    float b = q.angle();
    Vec3s vv = q.axis();

    EXPECT_TRUE( isApproxEqual(a, b, TOL) );
    EXPECT_TRUE( v.eq(vv, TOL) );

    q1.setAxisAngle(v,a);
    b = q1.angle();
    vv = q1.axis();
    EXPECT_TRUE( isApproxEqual(a, b, TOL) );
    EXPECT_TRUE( v.eq(vv, TOL) );
}


TEST_F(TestQuat, testOpPlus)
{
    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Quat<float> q = q1 + q2;

    float
        x=q1.x()+q2.x(), y=q1.y()+q2.y(), z=q1.z()+q2.z(), w=q1.w()+q2.w();
    EXPECT_TRUE( isExactlyEqual(q.x(), x) );
    EXPECT_TRUE( isExactlyEqual(q.y(), y) );
    EXPECT_TRUE( isExactlyEqual(q.z(), z) );
    EXPECT_TRUE( isExactlyEqual(q.w(), w) );

    q = q1;
    q += q2;
    EXPECT_TRUE( isExactlyEqual(q.x(), x) );
    EXPECT_TRUE( isExactlyEqual(q.y(), y) );
    EXPECT_TRUE( isExactlyEqual(q.z(), z) );
    EXPECT_TRUE( isExactlyEqual(q.w(), w) );

    q.add(q1,q2);
    EXPECT_TRUE( isExactlyEqual(q.x(), x) );
    EXPECT_TRUE( isExactlyEqual(q.y(), y) );
    EXPECT_TRUE( isExactlyEqual(q.z(), z) );
    EXPECT_TRUE( isExactlyEqual(q.w(), w) );
}


TEST_F(TestQuat, testOpMinus)
{
    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Quat<float> q = q1 - q2;

    float
        x=q1.x()-q2.x(), y=q1.y()-q2.y(), z=q1.z()-q2.z(), w=q1.w()-q2.w();
    EXPECT_TRUE( isExactlyEqual(q.x(), x) );
    EXPECT_TRUE( isExactlyEqual(q.y(), y) );
    EXPECT_TRUE( isExactlyEqual(q.z(), z) );
    EXPECT_TRUE( isExactlyEqual(q.w(), w) );

    q = q1;
    q -= q2;
    EXPECT_TRUE( isExactlyEqual(q.x(), x) );
    EXPECT_TRUE( isExactlyEqual(q.y(), y) );
    EXPECT_TRUE( isExactlyEqual(q.z(), z) );
    EXPECT_TRUE( isExactlyEqual(q.w(), w) );

    q.sub(q1,q2);
    EXPECT_TRUE( isExactlyEqual(q.x(), x) );
    EXPECT_TRUE( isExactlyEqual(q.y(), y) );
    EXPECT_TRUE( isExactlyEqual(q.z(), z) );
    EXPECT_TRUE( isExactlyEqual(q.w(), w) );
}


TEST_F(TestQuat, testOpMultiply)
{
    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);

    Quat<float> q = q1 * 1.5f;

    EXPECT_TRUE( isExactlyEqual(q.x(), float(1.5f)*q1.x()) );
    EXPECT_TRUE( isExactlyEqual(q.y(), float(1.5f)*q1.y()) );
    EXPECT_TRUE( isExactlyEqual(q.z(), float(1.5f)*q1.z()) );
    EXPECT_TRUE( isExactlyEqual(q.w(), float(1.5f)*q1.w()) );

    q = q1;
    q *= 1.5f;
    EXPECT_TRUE( isExactlyEqual(q.x(), float(1.5f)*q1.x()) );
    EXPECT_TRUE( isExactlyEqual(q.y(), float(1.5f)*q1.y()) );
    EXPECT_TRUE( isExactlyEqual(q.z(), float(1.5f)*q1.z()) );
    EXPECT_TRUE( isExactlyEqual(q.w(), float(1.5f)*q1.w()) );

    q.scale(1.5f, q1);
    EXPECT_TRUE( isExactlyEqual(q.x(), float(1.5f)*q1.x()) );
    EXPECT_TRUE( isExactlyEqual(q.y(), float(1.5f)*q1.y()) );
    EXPECT_TRUE( isExactlyEqual(q.z(), float(1.5f)*q1.z()) );
    EXPECT_TRUE( isExactlyEqual(q.w(), float(1.5f)*q1.w()) );
}


TEST_F(TestQuat, testInvert)
{
    float TOL = 1e-6f;

    Quat<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quat<float> q2(1.2f, 2.3f, 3.4f, 4.5f);


    q1 = q2;
    q2 = q2.inverse();

    Quat<float> q = q1*q2;

    EXPECT_TRUE( q.eq( Quat<float>(0,0,0,1), TOL ) );

    q1.normalize();
    q2 = q1.conjugate();
    q = q1*q2;
    EXPECT_TRUE( q.eq( Quat<float>(0,0,0,1), TOL ) );
}


TEST_F(TestQuat, testEulerAngles)
{

    {
        double TOL = 1e-7;

        Mat4d rx, ry, rz;
        const double angle1 = 20. * openvdb::math::pi<double>() / 180.;
        const double angle2 = 64. * openvdb::math::pi<double>() / 180.;
        const double angle3 = 125. *openvdb::math::pi<double>() / 180.;
        rx.setToRotation(Vec3d(1,0,0), angle1);
        ry.setToRotation(Vec3d(0,1,0), angle2);
        rz.setToRotation(Vec3d(0,0,1), angle3);

        Mat4d r = rx * ry * rz;

        const Quat<double> rot(r.getMat3());
        Vec3d result = rot.eulerAngles(ZYX_ROTATION);

        rx.setToRotation(Vec3d(1,0,0), result[0]);
        ry.setToRotation(Vec3d(0,1,0), result[1]);
        rz.setToRotation(Vec3d(0,0,1), result[2]);

        Mat4d rtest = rx * ry * rz;

        EXPECT_TRUE(r.eq(rtest, TOL));
    }

    {
        double TOL = 1e-7;

        Mat4d rx, ry, rz;
        const double angle1 = 20. * openvdb::math::pi<double>() / 180.;
        const double angle2 = 64. * openvdb::math::pi<double>() / 180.;
        const double angle3 = 125. *openvdb::math::pi<double>() / 180.;
        rx.setToRotation(Vec3d(1,0,0), angle1);
        ry.setToRotation(Vec3d(0,1,0), angle2);
        rz.setToRotation(Vec3d(0,0,1), angle3);

        Mat4d r = rz * ry * rx;

        const Quat<double> rot(r.getMat3());
        Vec3d result = rot.eulerAngles(XYZ_ROTATION);

        rx.setToRotation(Vec3d(1,0,0), result[0]);
        ry.setToRotation(Vec3d(0,1,0), result[1]);
        rz.setToRotation(Vec3d(0,0,1), result[2]);

        Mat4d rtest = rz * ry * rx;

        EXPECT_TRUE(r.eq(rtest, TOL));
    }

    {
        double TOL = 1e-7;

        Mat4d rx, ry, rz;
        const double angle1 = 20. * openvdb::math::pi<double>() / 180.;
        const double angle2 = 64. * openvdb::math::pi<double>() / 180.;
        const double angle3 = 125. *openvdb::math::pi<double>() / 180.;
        rx.setToRotation(Vec3d(1,0,0), angle1);
        ry.setToRotation(Vec3d(0,1,0), angle2);
        rz.setToRotation(Vec3d(0,0,1), angle3);

        Mat4d r = rz * rx * ry;

        const Quat<double> rot(r.getMat3());
        Vec3d result = rot.eulerAngles(YXZ_ROTATION);

        rx.setToRotation(Vec3d(1,0,0), result[0]);
        ry.setToRotation(Vec3d(0,1,0), result[1]);
        rz.setToRotation(Vec3d(0,0,1), result[2]);

        Mat4d rtest = rz * rx * ry;

        EXPECT_TRUE(r.eq(rtest, TOL));
    }

    {
        const Quat<float> rot(X_AXIS, 1.0);
        Vec3s result = rot.eulerAngles(XZY_ROTATION);
        EXPECT_EQ(result, Vec3s(1,0,0));
    }

}
