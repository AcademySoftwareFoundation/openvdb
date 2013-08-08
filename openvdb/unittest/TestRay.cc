///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/BBox.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetRayIntersector.h>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);

#define ASSERT_DOUBLES_APPROX_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/1.e-6);

class TestRay : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestRay);
    CPPUNIT_TEST(testInfinity);
    CPPUNIT_TEST(testRay);
    CPPUNIT_TEST(testDDA);
    CPPUNIT_TEST_SUITE_END();

    void testInfinity();
    void testRay();
    void testDDA();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestRay);

//  the Ray class makes use of infinity=1/0 so we test for it
void
TestRay::testInfinity()
{
    // This code generates compiler warnings which is why it's not
    // enabled by default.
    /*   
    const double one=1, zero = 0, infinity = one / zero;
    CPPUNIT_ASSERT_DOUBLES_EQUAL( infinity , infinity,0);//not a NAN
    CPPUNIT_ASSERT_DOUBLES_EQUAL( infinity , infinity+1,0);//not a NAN
    CPPUNIT_ASSERT_DOUBLES_EQUAL( infinity , infinity*10,0);//not a NAN
    CPPUNIT_ASSERT( zero <   infinity);
    CPPUNIT_ASSERT( zero >  -infinity);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( zero ,  one/infinity,0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( zero , -one/infinity,0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL( infinity  ,  one/zero,0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-infinity  , -one/zero,0);
    */
    //std::cerr << "inf:        "   << infinity << "\n";
    //std::cerr << "1 / inf:    "   << one / infinity << "\n";
    //std::cerr << "1 / (-inf): "   << one / (-infinity) << "\n";
    //std::cerr << " inf * 0:   "   << infinity * 0 << "\n";
    //std::cerr << "-inf * 0:   "   << (-infinity) * 0 << "\n";
    //std::cerr << "(inf):      "   << (bool)(infinity) << "\n";
    //std::cerr << "inf == inf: "   << (infinity == infinity) << "\n";
    //std::cerr << "inf > 0:    "   << (infinity > 0) << "\n";
    //std::cerr << "-inf > 0:   "   << ((-infinity) > 0) << "\n";

}

void TestRay::testRay()
{
    using namespace openvdb;
    typedef double             RealT;
    typedef math::Ray<RealT>   RayT;
    typedef RayT::Vec3T        Vec3T;
    typedef math::BBox<Vec3T>  BBoxT;
    
    {// simple construction
        
        const Vec3T dir(1.5,1.5,1.5);
        const Vec3T eye(1.5,1.5,1.5);
        RealT t0=0.1, t1=12589.0;
        
        RayT ray(eye, dir, t0, t1);
        CPPUNIT_ASSERT(ray.eye()==eye);
        CPPUNIT_ASSERT(ray.dir()==dir);
        ASSERT_DOUBLES_APPROX_EQUAL( t0, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( t1, ray.t1());
    }
    
    {// test transformation
        math::Transform::Ptr xform = math::Transform::createLinearTransform();
        
        xform->preRotate(M_PI, math::Y_AXIS );
        xform->postTranslate(math::Vec3d(1, 2, 3));
        xform->preScale(Vec3R(0.1, 0.2, 0.4));
        
        Vec3T eye(9,1,1), dir(1,2,0);
        RealT t0=0.1, t1=12589.0;

        RayT ray0(eye, dir, t0, t1);
        CPPUNIT_ASSERT( ray0.test(t0));
        CPPUNIT_ASSERT( ray0.test(t1));
        CPPUNIT_ASSERT( ray0.test(0.5*(t0+t1)));
        CPPUNIT_ASSERT(!ray0.test(t0-1));
        CPPUNIT_ASSERT(!ray0.test(t1+1));
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

    {// test bbox intersection
        
        const Vec3T dir(-1.0, 2.0, 3.0);
        const Vec3T eye( 2.0, 1.0, 1.0);
        RayT ray(eye, dir);
        RealT t0=0, t1=0;
       

        // intersects the two faces of the box perpendicular to the y-axis!
        CPPUNIT_ASSERT(ray.intersects(CoordBBox(Coord(0, 2, 2), Coord(2, 4, 6)), t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 1.5, t1);
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(1.5)[1], 4);//higher y component of intersection

        // intersects the lower edge anlong the z-axis of the box
        CPPUNIT_ASSERT(ray.intersects(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0)), t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, t1);
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[0], 1.5);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2.0);//higher y component of intersection
        
        // no intersections
        CPPUNIT_ASSERT(!ray.intersects(CoordBBox(Coord(4, 2, 2), Coord(6, 4, 6))));
    }
    
    {// test sphere intersection
        const Vec3T dir(-1.0, 2.0, 3.0);
        const Vec3T eye( 2.0, 1.0, 1.0);
        RayT ray(eye, dir);
        RealT t0=0, t1=0;
        
        // intersects twice - second intersection exits sphere in lower y-z-plane
        Vec3T center(2.0,3.0,4.0);
        RealT radius = 1.0f;
        CPPUNIT_ASSERT(ray.intersects(center, radius, t0, t1));
        CPPUNIT_ASSERT(t0 < t1);
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t1);
        ASSERT_DOUBLES_APPROX_EQUAL(ray(t1)[1], center[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(ray(t1)[2], center[2]);
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t0)-center).length()-radius, 0);
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t1)-center).length()-radius, 0);
        
        // no intersection
        center = Vec3T(3.0,3.0,4.0);
        radius = 1.0f;
        CPPUNIT_ASSERT(!ray.intersects(center, radius, t0, t1));
    }

    {// test bbox clip
        const Vec3T dir(-1.0, 2.0, 3.0);
        const Vec3T eye( 2.0, 1.0, 1.0);
        RealT t0=0.1, t1=12589.0;
        RayT ray(eye, dir, t0, t1);

        // intersects the two faces of the box perpendicular to the y-axis!
        CPPUNIT_ASSERT(ray.clip(CoordBBox(Coord(0, 2, 2), Coord(2, 4, 6))));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( 1.5, ray.t1());
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(1.5)[1], 4);//higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // intersects the lower edge anlong the z-axis of the box
        CPPUNIT_ASSERT(ray.clip(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0))));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( 0.5, ray.t1());
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[0], 1.5);//lower y component of intersection
        ASSERT_DOUBLES_APPROX_EQUAL( ray(0.5)[1], 2.0);//higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // no intersections
        CPPUNIT_ASSERT(!ray.clip(CoordBBox(Coord(4, 2, 2), Coord(6, 4, 6))));
        ASSERT_DOUBLES_APPROX_EQUAL( t0, ray.t0());
        ASSERT_DOUBLES_APPROX_EQUAL( t1, ray.t1());
    }
   
}

void TestRay::testDDA()
{
    using namespace openvdb;
    typedef math::Ray<double>  RayType;
    
    {// test voxel traversal along both directions of each axis
        typedef math::DDA<RayType> DDAType;
        const RayType::Vec3T eye( 0, 0, 0);
        for (int s = -1; s<=1; s+=2) {
            for (int a = 0; a<3; ++a) {
                const int d[3]={s*(a==0), s*(a==1), s*(a==2)};
                const RayType::Vec3T dir(d[0], d[1], d[2]);
                RayType ray(eye, dir);
                DDAType dda(ray);
                CPPUNIT_ASSERT(dda.voxel()==Coord(0,0,0));
                //std::cerr << "\nray: "<<ray<<std::endl; 
                for (int i=0; i<10; ++i) {
                    //std::cerr << "i="<<i<<" voxel="<<dda.voxel()<<" time="<<dda.time()<<std::endl;
                    CPPUNIT_ASSERT(dda.voxel()==Coord(i*d[0], i*d[1], i*d[2]));
                    ASSERT_DOUBLES_APPROX_EQUAL(1.0+i,dda.step());
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
                CPPUNIT_ASSERT(dda.voxel()==Coord(0,0,0));
                //std::cerr << "\nray: "<<ray<<std::endl; 
                for (int i=0; i<10; ++i) {
                    //std::cerr << "i="<<i<<" voxel="<<dda.voxel()<<" time="<<dda.time()<<std::endl;
                    CPPUNIT_ASSERT(dda.voxel()==Coord(8*i*d[0],8*i*d[1],8*i*d[2]));
                    ASSERT_DOUBLES_APPROX_EQUAL(8.0+8*i,dda.step());
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
                CPPUNIT_ASSERT(dda.voxel()==Coord(0,0,0));
                //std::cerr << "\nray: "<<ray<<std::endl; 
                for (int i=0; i<10; ++i) {
                    //std::cerr << "i="<<i<<" voxel="<<dda.voxel()<<" time="<<dda.time()<<std::endl;
                    CPPUNIT_ASSERT(dda.voxel()==Coord(8*i*d[0],8*i*d[1],8*i*d[2]));
                    ASSERT_DOUBLES_APPROX_EQUAL(4.0+4*i,dda.step());
                }
            }
        }
    }
    
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
