// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/BBox.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetSphere.h>

#include <gtest/gtest.h>


#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/0.0);

#define ASSERT_DOUBLES_APPROX_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/1.e-6);

class TestRay : public ::testing::Test
{
};


//  the Ray class makes use of infinity=1/0 so we test for it
TEST_F(TestRay, testInfinity)
{
    // This code generates compiler warnings which is why it's not
    // enabled by default.
    /*
    const double one=1, zero = 0, infinity = one / zero;
    EXPECT_NEAR( infinity , infinity,0);//not a NAN
    EXPECT_NEAR( infinity , infinity+1,0);//not a NAN
    EXPECT_NEAR( infinity , infinity*10,0);//not a NAN
    EXPECT_TRUE( zero <   infinity);
    EXPECT_TRUE( zero >  -infinity);
    EXPECT_NEAR( zero ,  one/infinity,0);
    EXPECT_NEAR( zero , -one/infinity,0);
    EXPECT_NEAR( infinity  ,  one/zero,0);
    EXPECT_NEAR(-infinity  , -one/zero,0);

    std::cerr << "inf:        "   << infinity << "\n";
    std::cerr << "1 / inf:    "   << one / infinity << "\n";
    std::cerr << "1 / (-inf): "   << one / (-infinity) << "\n";
    std::cerr << " inf * 0:   "   << infinity * 0 << "\n";
    std::cerr << "-inf * 0:   "   << (-infinity) * 0 << "\n";
    std::cerr << "(inf):      "   << (bool)(infinity) << "\n";
    std::cerr << "inf == inf: "   << (infinity == infinity) << "\n";
    std::cerr << "inf > 0:    "   << (infinity > 0) << "\n";
    std::cerr << "-inf > 0:   "   << ((-infinity) > 0) << "\n";
    */
}

TEST_F(TestRay, testRay)
{
    using namespace openvdb;
    typedef double             RealT;
    typedef math::Ray<RealT>   RayT;
    typedef RayT::Vec3T        Vec3T;
    typedef math::BBox<Vec3T>  BBoxT;

    {//default constructor
        RayT ray;
        EXPECT_TRUE(ray.eye() == Vec3T(0,0,0));
        EXPECT_TRUE(ray.dir() == Vec3T(1,0,0));
        ASSERT_DOUBLES_APPROX_EQUAL( math::Delta<RealT>::value(), ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( std::numeric_limits<RealT>::max(), ray.t1());
    }

    {// simple construction

        Vec3T eye(1.5,1.5,1.5), dir(1.5,1.5,1.5); dir.normalize();
        RealT t0=0.1, t1=12589.0;

        RayT ray(eye, dir, t0, t1);
        EXPECT_TRUE(ray.eye()==eye);
        EXPECT_TRUE(ray.dir()==dir);
        ASSERT_DOUBLES_APPROX_EQUAL( t0, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( t1, ray.t1());
    }

    {// test transformation
        math::Transform::Ptr xform = math::Transform::createLinearTransform();

        xform->preRotate(M_PI, math::Y_AXIS );
        xform->postTranslate(math::Vec3d(1, 2, 3));
        xform->preScale(Vec3R(0.1, 0.2, 0.4));

        Vec3T eye(9,1,1), dir(1,2,0);
        dir.normalize();
        RealT t0=0.1, t1=12589.0;

        RayT ray0(eye, dir, t0, t1);
        EXPECT_TRUE( ray0.test(t0));
        EXPECT_TRUE( ray0.test(t1));
        EXPECT_TRUE( ray0.test(0.5*(t0+t1)));
        EXPECT_TRUE(!ray0.test(t0-1));
        EXPECT_TRUE(!ray0.test(t1+1));
        //std::cerr << "Ray0: " << ray0 << std::endl;
        RayT ray1 = ray0.applyMap( *(xform->baseMap()) );
        //std::cerr << "Ray1: " << ray1 << std::endl;
        RayT ray2 = ray1.applyInverseMap( *(xform->baseMap()) );
        //std::cerr << "Ray2: " << ray2 << std::endl;

        ASSERT_DOUBLES_APPROX_EQUAL( eye[0], ray2.eye()[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( eye[1], ray2.eye()[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( eye[2], ray2.eye()[2]);

        ASSERT_DOUBLES_APPROX_EQUAL( dir[0], ray2.dir()[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[1], ray2.dir()[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[2], ray2.dir()[2]);

        ASSERT_DOUBLES_APPROX_EQUAL( dir[0], 1.0/ray2.invDir()[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[1], 1.0/ray2.invDir()[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[2], 1.0/ray2.invDir()[2]);

        ASSERT_DOUBLES_APPROX_EQUAL( t0, ray2.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( t1, ray2.t1());
    }

    {// test transformation

        // This is the index to world transform
        math::Transform::Ptr xform = math::Transform::createLinearTransform();
        xform->postRotate(M_PI, math::Y_AXIS );
        xform->postTranslate(math::Vec3d(1, 2, 3));
        xform->postScale(Vec3R(0.1, 0.1, 0.1));//voxel size

        // Define a ray in world space
        Vec3T eye(9,1,1), dir(1,2,0);
        dir.normalize();
        RealT t0=0.1, t1=12589.0;
        RayT ray0(eye, dir, t0, t1);
        //std::cerr << "\nWorld Ray0: " << ray0 << std::endl;
        EXPECT_TRUE( ray0.test(t0));
        EXPECT_TRUE( ray0.test(t1));
        EXPECT_TRUE( ray0.test(0.5*(t0+t1)));
        EXPECT_TRUE(!ray0.test(t0-1));
        EXPECT_TRUE(!ray0.test(t1+1));
        Vec3T xyz0[3] = {ray0.start(), ray0.mid(), ray0.end()};

        // Transform the ray to index space
        RayT ray1 = ray0.applyInverseMap( *(xform->baseMap()) );
        //std::cerr << "\nIndex Ray1: " << ray1 << std::endl;
        Vec3T xyz1[3] = {ray1.start(), ray1.mid(), ray1.end()};

        for (int i=0; i<3; ++i) {
            Vec3T pos = xform->baseMap()->applyMap(xyz1[i]);
            //std::cerr << "world0 ="<<xyz0[i] << " transformed index="<< pos << std::endl;
            for (int j=0; j<3; ++j) ASSERT_DOUBLES_APPROX_EQUAL(xyz0[i][j], pos[j]);
        }

        // Transform the ray back to world pace
        RayT ray2 = ray1.applyMap( *(xform->baseMap()) );
        //std::cerr << "\nWorld Ray2: " << ray2 << std::endl;

        ASSERT_DOUBLES_APPROX_EQUAL( eye[0], ray2.eye()[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( eye[1], ray2.eye()[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( eye[2], ray2.eye()[2]);

        ASSERT_DOUBLES_APPROX_EQUAL( dir[0], ray2.dir()[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[1], ray2.dir()[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[2], ray2.dir()[2]);

        ASSERT_DOUBLES_APPROX_EQUAL( dir[0], 1.0/ray2.invDir()[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[1], 1.0/ray2.invDir()[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( dir[2], 1.0/ray2.invDir()[2]);

        ASSERT_DOUBLES_APPROX_EQUAL( t0, ray2.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( t1, ray2.t1());

        Vec3T xyz2[3] = {ray0.start(), ray0.mid(), ray0.end()};

        for (int i=0; i<3; ++i) {
            //std::cerr << "world0 ="<<xyz0[i] << " world2 ="<< xyz2[i] << std::endl;
            for (int j=0; j<3; ++j) ASSERT_DOUBLES_APPROX_EQUAL(xyz0[i][j], xyz2[i][j]);
        }
    }

    {// test bbox intersection

        const Vec3T eye( 2.0, 1.0, 1.0), dir(-1.0, 2.0, 3.0);
        RayT ray(eye, dir);
        RealT t0=0, t1=0;


        // intersects the two faces of the box perpendicular to the y-axis!
        EXPECT_TRUE(ray.intersects(CoordBBox(Coord(0, 2, 2), Coord(2, 4, 6)), t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 1.5, t1);
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(1.5)[1], 4);//higher y component of intersection

        // intersects the lower edge anlong the z-axis of the box
        EXPECT_TRUE(ray.intersects(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0)), t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, t1);
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[0], 1.5);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2.0);//higher y component of intersection

        // no intersections
        EXPECT_TRUE(!ray.intersects(CoordBBox(Coord(4, 2, 2), Coord(6, 4, 6))));
    }

    {// test sphere intersection
        const Vec3T dir(-1.0, 2.0, 3.0);
        const Vec3T eye( 2.0, 1.0, 1.0);
        RayT ray(eye, dir);
        RealT t0=0, t1=0;

        // intersects twice - second intersection exits sphere in lower y-z-plane
        Vec3T center(2.0,3.0,4.0);
        RealT radius = 1.0f;
        EXPECT_TRUE(ray.intersects(center, radius, t0, t1));
        EXPECT_TRUE(t0 < t1);
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t1);
        ASSERT_DOUBLES_APPROX_EQUAL(ray(t1)[1], center[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(ray(t1)[2], center[2]);
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t0)-center).length()-radius, 0);
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t1)-center).length()-radius, 0);

        // no intersection
        center = Vec3T(3.0,3.0,4.0);
        radius = 1.0f;
        EXPECT_TRUE(!ray.intersects(center, radius, t0, t1));
    }

    {// test bbox clip
        const Vec3T dir(-1.0, 2.0, 3.0);
        const Vec3T eye( 2.0, 1.0, 1.0);
        RealT t0=0.1, t1=12589.0;
        RayT ray(eye, dir, t0, t1);

        // intersects the two faces of the box perpendicular to the y-axis!
        EXPECT_TRUE(ray.clip(CoordBBox(Coord(0, 2, 2), Coord(2, 4, 6))));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( 1.5, ray.t1());
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(1.5)[1], 4);//higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // intersects the lower edge anlong the z-axis of the box
        EXPECT_TRUE(ray.clip(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0))));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, ray.t1());
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[0], 1.5);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2.0);//higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // no intersections
        EXPECT_TRUE(!ray.clip(CoordBBox(Coord(4, 2, 2), Coord(6, 4, 6))));
        ASSERT_DOUBLES_APPROX_EQUAL( t0, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( t1, ray.t1());
    }

    {// test plane intersection

        const Vec3T dir(-1.0, 0.0, 0.0);
        const Vec3T eye( 0.5, 4.7,-9.8);
        RealT t0=1.0, t1=12589.0;
        RayT ray(eye, dir, t0, t1);

        Real t = 0.0;
        EXPECT_TRUE(!ray.intersects(Vec3T( 1.0, 0.0, 0.0), 4.0, t));
        EXPECT_TRUE(!ray.intersects(Vec3T(-1.0, 0.0, 0.0),-4.0, t));
        EXPECT_TRUE( ray.intersects(Vec3T( 1.0, 0.0, 0.0),-4.0, t));
        ASSERT_DOUBLES_APPROX_EQUAL(4.5, t);
        EXPECT_TRUE( ray.intersects(Vec3T(-1.0, 0.0, 0.0), 4.0, t));
        ASSERT_DOUBLES_APPROX_EQUAL(4.5, t);
        EXPECT_TRUE(!ray.intersects(Vec3T( 1.0, 0.0, 0.0),-0.4, t));
    }

    {// test plane intersection

        const Vec3T dir( 0.0, 1.0, 0.0);
        const Vec3T eye( 4.7, 0.5,-9.8);
        RealT t0=1.0, t1=12589.0;
        RayT ray(eye, dir, t0, t1);

        Real t = 0.0;
        EXPECT_TRUE(!ray.intersects(Vec3T( 0.0,-1.0, 0.0), 4.0, t));
        EXPECT_TRUE(!ray.intersects(Vec3T( 0.0, 1.0, 0.0),-4.0, t));
        EXPECT_TRUE( ray.intersects(Vec3T( 0.0, 1.0, 0.0), 4.0, t));
        ASSERT_DOUBLES_APPROX_EQUAL(3.5, t);
        EXPECT_TRUE( ray.intersects(Vec3T( 0.0,-1.0, 0.0),-4.0, t));
        ASSERT_DOUBLES_APPROX_EQUAL(3.5, t);
        EXPECT_TRUE(!ray.intersects(Vec3T( 1.0, 0.0, 0.0), 0.4, t));
    }

}

TEST_F(TestRay, testTimeSpan)
{
    using namespace openvdb;
    typedef double             RealT;
    typedef math::Ray<RealT>::TimeSpan   TimeSpanT;
    TimeSpanT t(2.0, 5.0);
    ASSERT_DOUBLES_EXACTLY_EQUAL(2.0, t.t0);
    ASSERT_DOUBLES_EXACTLY_EQUAL(5.0, t.t1);
    ASSERT_DOUBLES_APPROX_EQUAL(3.5, t.mid());
    EXPECT_TRUE(t.valid());
    t.set(-1, -1);
    EXPECT_TRUE(!t.valid());
    t.scale(5);
    ASSERT_DOUBLES_EXACTLY_EQUAL(-5.0, t.t0);
    ASSERT_DOUBLES_EXACTLY_EQUAL(-5.0, t.t1);
    ASSERT_DOUBLES_APPROX_EQUAL(-5.0, t.mid());
}

TEST_F(TestRay, testDDA)
{
    using namespace openvdb;
    typedef math::Ray<double>  RayType;

    {
        typedef math::DDA<RayType,3+4+5> DDAType;
        const RayType::Vec3T dir( 1.0, 0.0, 0.0);
        const RayType::Vec3T eye(-1.0, 0.0, 0.0);
        const RayType ray(eye, dir);
        //std::cerr << ray << std::endl;
        DDAType dda(ray);
        ASSERT_DOUBLES_APPROX_EQUAL(math::Delta<double>::value(), dda.time());
        ASSERT_DOUBLES_APPROX_EQUAL(1.0, dda.next());
        //dda.print();
        dda.step();
        ASSERT_DOUBLES_APPROX_EQUAL(1.0, dda.time());
        ASSERT_DOUBLES_APPROX_EQUAL(4096+1.0, dda.next());
        //dda.print();
    }

    {// Check for the notorious +-0 issue!
        typedef math::DDA<RayType,3> DDAType;

        //std::cerr << "\nPositive zero ray" << std::endl;
        const RayType::Vec3T dir1(1.0, 0.0, 0.0);
        const RayType::Vec3T eye1(2.0, 0.0, 0.0);
        const RayType ray1(eye1, dir1);
        //std::cerr << ray1 << std::endl;
        DDAType dda1(ray1);
        //dda1.print();
        dda1.step();
        //dda1.print();

        //std::cerr << "\nNegative zero ray" << std::endl;
        const RayType::Vec3T dir2(1.0,-0.0,-0.0);
        const RayType::Vec3T eye2(2.0, 0.0, 0.0);
        const RayType ray2(eye2, dir2);
        //std::cerr << ray2 << std::endl;
        DDAType dda2(ray2);
        //dda2.print();
        dda2.step();
        //dda2.print();

        //std::cerr << "\nNegative epsilon ray" << std::endl;
        const RayType::Vec3T dir3(1.0,-1e-9,-1e-9);
        const RayType::Vec3T eye3(2.0, 0.0, 0.0);
        const RayType ray3(eye3, dir3);
        //std::cerr << ray3 << std::endl;
        DDAType dda3(ray3);
        //dda3.print();
        dda3.step();
        //dda3.print();

        //std::cerr << "\nPositive epsilon ray" << std::endl;
        const RayType::Vec3T dir4(1.0,-1e-9,-1e-9);
        const RayType::Vec3T eye4(2.0, 0.0, 0.0);
        const RayType ray4(eye3, dir4);
        //std::cerr << ray4 << std::endl;
        DDAType dda4(ray4);
        //dda4.print();
        dda4.step();
        //dda4.print();

        ASSERT_DOUBLES_APPROX_EQUAL(dda1.time(), dda2.time());
        ASSERT_DOUBLES_APPROX_EQUAL(dda2.time(), dda3.time());
        ASSERT_DOUBLES_APPROX_EQUAL(dda3.time(), dda4.time());
        ASSERT_DOUBLES_APPROX_EQUAL(dda1.next(), dda2.next());
        ASSERT_DOUBLES_APPROX_EQUAL(dda2.next(), dda3.next());
        ASSERT_DOUBLES_APPROX_EQUAL(dda3.next(), dda4.next());
    }

    {// test voxel traversal along both directions of each axis
        typedef math::DDA<RayType> DDAType;
        const RayType::Vec3T eye( 0, 0, 0);
        for (int s = -1; s<=1; s+=2) {
            for (int a = 0; a<3; ++a) {
                const int d[3]={s*(a==0), s*(a==1), s*(a==2)};
                const RayType::Vec3T dir(d[0], d[1], d[2]);
                RayType ray(eye, dir);
                DDAType dda(ray);
                //std::cerr << "\nray: "<<ray<<std::endl;
                //dda.print();
                for (int i=1; i<=10; ++i) {
                    //std::cerr << "i="<<i<<" voxel="<<dda.voxel()<<" time="<<dda.time()<<std::endl;
                    //EXPECT_TRUE(dda.voxel()==Coord(i*d[0], i*d[1], i*d[2]));
                    EXPECT_TRUE(dda.step());
                    ASSERT_DOUBLES_APPROX_EQUAL(i,dda.time());
                }
            }
        }
    }
    {// test Node traversal along both directions of each axis
        typedef math::DDA<RayType,3> DDAType;
        const RayType::Vec3T eye(0, 0, 0);

        for (int s = -1; s<=1; s+=2) {
            for (int a = 0; a<3; ++a) {
                const int d[3]={s*(a==0), s*(a==1), s*(a==2)};
                const RayType::Vec3T dir(d[0], d[1], d[2]);
                RayType ray(eye, dir);
                DDAType dda(ray);
                //std::cerr << "\nray: "<<ray<<std::endl;
                for (int i=1; i<=10; ++i) {
                    //std::cerr << "i="<<i<<" voxel="<<dda.voxel()<<" time="<<dda.time()<<std::endl;
                    //EXPECT_TRUE(dda.voxel()==Coord(8*i*d[0],8*i*d[1],8*i*d[2]));
                    EXPECT_TRUE(dda.step());
                    ASSERT_DOUBLES_APPROX_EQUAL(8*i,dda.time());
                }
            }
        }
    }

    {// test accelerated Node traversal along both directions of each axis
        typedef math::DDA<RayType,3> DDAType;
        const RayType::Vec3T eye(0, 0, 0);

        for (int s = -1; s<=1; s+=2) {
            for (int a = 0; a<3; ++a) {
                const int d[3]={s*(a==0), s*(a==1), s*(a==2)};
                const RayType::Vec3T dir(2*d[0], 2*d[1], 2*d[2]);
                RayType ray(eye, dir);
                DDAType dda(ray);
                //ASSERT_DOUBLES_APPROX_EQUAL(0.0, dda.time());
                //EXPECT_TRUE(dda.voxel()==Coord(0,0,0));
                double next=0;
                //std::cerr << "\nray: "<<ray<<std::endl;
                for (int i=1; i<=10; ++i) {
                    //std::cerr << "i="<<i<<" voxel="<<dda.voxel()<<" time="<<dda.time()<<std::endl;
                    //EXPECT_TRUE(dda.voxel()==Coord(8*i*d[0],8*i*d[1],8*i*d[2]));
                    EXPECT_TRUE(dda.step());
                    ASSERT_DOUBLES_APPROX_EQUAL(4*i, dda.time());
                    if (i>1) {
                        ASSERT_DOUBLES_APPROX_EQUAL(dda.time(), next);
                    }
                    next = dda.next();
                }
            }
        }
    }

}
