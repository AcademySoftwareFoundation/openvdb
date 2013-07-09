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

//#define BENCHMARK_TEST

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetRayIntersector.h>

#ifdef BENCHMARK_TEST
#include <openvdb/math/Stats.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/GridOperators.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#endif

#include <stdlib.h> // for exit
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);

#define ASSERT_DOUBLES_APPROX_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/1.e-6);

class TestLevelSetRayIntersector : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestLevelSetRayIntersector);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

    void test();
};

/// @brief Extremely naive and bare-bone Portable Pixel Map implementation
/// class intended for debugging and unittesting only!
class PPM
{
  public:
    struct RGB
    {
        typedef unsigned char ValueType;
        RGB() : r(0), g(0), b(0) {}
        RGB(ValueType intensity) : r(intensity), g(intensity), b(intensity) {}
        RGB(ValueType _r, ValueType _g, ValueType _b) : r(_r), g(_g), b(_b) {}
        ValueType r, g, b;
    };
    
    PPM(size_t width, size_t height)
        : mWidth(width), mHeight(height), mSize(width*height), mPixels(new RGB[mSize]) {}
    ~PPM() { delete mPixels; }
    void setRGB(size_t w, size_t h, const RGB& rgb)
      {
          assert(w < mWidth);
          assert(h < mHeight);
          mPixels[w + h*mWidth] = rgb;
      }
    const RGB& getRGB(size_t w, size_t h) const
      {
          assert(w < mWidth);
          assert(h < mHeight);
          return mPixels[w + h*mWidth];
      }
    void fill(const RGB& rgb) { for (size_t i=0; i<mSize; ++i) mPixels[i] = rgb; }
    void save(const std::string& fileName)
      {
          std::string name(fileName + ".ppm");
          std::ofstream os(name.c_str(), std::ios_base::binary);
          if (!os.is_open()) {
              std::cerr << "Error opening PPM file \"" << name << "\"" << std::endl;
              exit(1);
          }
          os << "P6\n" << "# Created by PPM\n" << mWidth << " " << mHeight << "\n255\n";
          os.write((const char *)&(*mPixels), mSize*sizeof(RGB));
      }
    size_t width()  const { return mWidth; }
    size_t height() const { return mHeight; }
  private:
    size_t mWidth, mHeight, mSize;
    RGB* mPixels;
};

/// @brief For testing of multi-threaded ray-intersection
/// Super naive implementation using z-aligned orthographic projection
/// @warning ONLY INTENDED FOR DEBUGGING AND UNIT-TESTING!!!!!!!!!!
template<typename RayIntersectorT, typename ShaderT>
struct TestTracer
{
    typedef typename RayIntersectorT::GridType GridType;
    typedef typename RayIntersectorT::Vec3Type Vec3Type;
    typedef typename RayIntersectorT::RayType  RayType;

    // For the placement of the image plane
    struct OrthoCamera {
        Vec3Type origin;
        double width, height;
    };
    TestTracer(const RayIntersectorT& inter, PPM& ppm,
               const ShaderT& shader, const OrthoCamera& camera)
        : mInter(&inter), mPPM(&ppm), mShader(&shader), mCamera(&camera) {}
    void run(bool threaded = true)
    {
        tbb::blocked_range<size_t> range(0,mPPM->width());
        threaded ?  tbb::parallel_for(range, *this) : (*this)(range);
    }
    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        Vec3Type xyz(0), eye(mCamera->origin);
        RayType ray(eye, Vec3Type(0,0,1));
        openvdb::tools::LinearIntersector<GridType> tester(mInter->grid());
        const double di = mCamera->width/mPPM->width(), dj=mCamera->height/mPPM->height();
        for (size_t i=range.begin(), ie = range.end(); i<ie; ++i) {
            eye[0] = di*i + mCamera->origin[0];
            for (size_t j=0, je = mPPM->height(); j<je; ++j) {
                eye[1] = dj*j + mCamera->origin[1],
                ray.setEye(eye);
                if (mInter->intersectsWS(ray,tester,xyz)) mPPM->setRGB(i,j,(*mShader)(xyz,ray));
            }
        }
    }
    const RayIntersectorT* mInter;
    PPM*                   mPPM;
    const ShaderT*         mShader;
    const OrthoCamera*     mCamera;
};// TestTracer

#ifdef BENCHMARK_TEST
// Super simple test shaders
template <typename RealType>
struct DepthShader
{
    DepthShader(RealType _min, RealType _max) : min(_min), den(1.0/(_max-_min)) {}
    template <typename RayType>
    PPM::RGB operator()(const openvdb::math::Vec3<RealType>& xyz,
                        const RayType&) const
    {
        const double delta = den*(xyz[2]-min);
        return PPM::RGB(int(255*(1.0-delta)), int(255*delta),255);
    }
    RealType min, den;
};

struct MatteShader
{
    template <typename RealType, typename RayType>
    PPM::RGB operator()(const openvdb::math::Vec3<RealType>&,
                        const RayType&) const { return PPM::RGB(255, 255, 255); }
};

// Color shading that treats the normal (x, y, z) values as (r, g, b) color components.
template <typename GridType>
struct MyShader1
{
    typedef typename GridType::ValueType Vec3T;
    MyShader1(const GridType& grid) : mSampler(grid) {}
    template <typename RealType, typename RayType>
    PPM::RGB operator()(const openvdb::math::Vec3<RealType>& xyz,
                        const RayType& ray) const
    {
        Vec3T grad = mSampler.wsSample(xyz);
        return PPM::RGB(int(255*openvdb::math::Clamp01((grad[0]+1.0)*0.5)),
                        int(255*openvdb::math::Clamp01((grad[1]+1.0)*0.5)),
                        int(255*openvdb::math::Clamp01((grad[2]+1.0)*0.5)));
    }
    openvdb::tools::GridSampler<GridType, openvdb::tools::BoxSampler> mSampler;
};

// Gray-scale shading based on angle between camera ray and surface normal
template <typename GridType>
struct MyShader2
{
    typedef typename GridType::ValueType Vec3T;
    MyShader2(const GridType& grid) : mSampler(grid) {}
    template <typename RealType, typename RayType>
    PPM::RGB operator()(const openvdb::math::Vec3<RealType>& xyz,
                        const RayType& ray) const
    {
        Vec3T grad = mSampler.wsSample(xyz); 
        const float alpha = openvdb::math::Abs(grad.dot(ray.dir()));
        return PPM::RGB(int(255*alpha));
    }
    openvdb::tools::GridSampler<GridType, openvdb::tools::BoxSampler> mSampler;
};
#endif

CPPUNIT_TEST_SUITE_REGISTRATION(TestLevelSetRayIntersector);

void TestLevelSetRayIntersector::test()
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;  

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(20.0f, 0.0f, 0.0f);
        const float s = 0.5f, w = 2.0f;
        
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);
        tools::LinearIntersector<FloatGrid> tester(*ls);
        
        const Vec3T dir(1, 0, 0);
        const Vec3T eye(2, 0, 0);
        const RayT ray(eye, dir);
        Vec3T xyz(0);
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, tester, xyz));
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        CPPUNIT_ASSERT(ray(t0) == xyz);
    }
    
    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 20.0f, 0.0f);
        const float s = 1.5f, w = 2.0f;
        
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);
        tools::LinearIntersector<FloatGrid> tester(*ls);
        
        const Vec3T dir(0, 1, 0);
        const Vec3T eye(0,-2, 0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, tester, xyz));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[2]);
    }
    
    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;
        
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid, -1> lsri(*ls);
        tools::LinearIntersector<FloatGrid> tester(*ls);
        
        const Vec3T dir(0, 0, 1);
        const Vec3T eye(0, 0, 4);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, tester, xyz));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[2]);
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 10.0f);
        const float s = 1.0f, w = 3.0f;
        
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);
        tools::LinearIntersector<FloatGrid> tester(*ls);
        
        const Vec3T dir(1, 1, 1);
        const Vec3T eye(0, 0, 0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        CPPUNIT_ASSERT(lsri.intersectsWS(ray, tester, xyz));
        //analytical intersection test
        double t0=0, t1=0;
        CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t0)-c).length()-r, 0);
        ASSERT_DOUBLES_APPROX_EQUAL((ray(t1)-c).length()-r, 0);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        const Vec3T delta = xyz - ray(t0);
        //std::cerr << "delta = " << delta << std::endl;
        //std::cerr << "|delta|/dx=" << (delta.length()/ls->voxelSize()[0]) << std::endl;
        CPPUNIT_ASSERT( delta.length() < 0.5*ls->voxelSize()[0] );
    }
   
    {// generate and image and benchmark

        // Generate a high-resolution level set sphere @1000^3
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 20.0f);
        const float s = 0.01f, w = 2.0f;
        double t0=0, t1=0;
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);
        tools::LinearIntersector<FloatGrid, 0> tester(*ls);

        Vec3T xyz(0);
        const size_t width = 1024;
        const double dx = 20.0/width;
        PPM ppm(width, width);
        const Vec3T dir(0, 0, 1);
#ifdef BENCHMARK_TEST
        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();
        openvdb::math::Stats stats;
        openvdb::math::Histogram hist(0.0, 0.2, 20);
#endif
        for (size_t i=0; i<width; ++i) {
            for (size_t j=0; j<width; ++j) {
                const Vec3T eye(dx*i, dx*j, 0.0);
                const RayT ray(eye, dir);
                if (lsri.intersectsWS(ray, tester, xyz)) {
                    CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
                    double delta = (ray(t0)-xyz).length()/s;//in voxel units
                    if (delta > 0.1) {
                        ppm.setRGB(i, j, PPM::RGB(255,  0,  0));
                    } else {
#ifdef BENCHMARK_TEST
                        stats.add(delta);
                        hist.add(delta);
#endif
                        ppm.setRGB(i, j, PPM::RGB(0,255,0));
                    }
                }
            }
        }
#ifdef BENCHMARK_TEST
        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nRay-tracing took " << msdiff.total_milliseconds() << " ms\n";

        ppm.save("/tmp/sphere_serial");
        stats.print("First hit");
        hist.print("First hit");
#endif
    }
    
#ifdef BENCHMARK_TEST
     {// generate an image and benchmark numbers from a sampled sphere

        // Generate a high-resolution level set sphere @1000^3
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 20.0f);
        const float s = 0.01f, w = 2.0f;
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);
        
        PPM ppm(1024, 1024);
        
        DepthShader<double> shader(15.0, 20.0);
        typedef TestTracer<tools::LevelSetRayIntersector<FloatGrid>, DepthShader<double> > TracerT;

        TracerT::OrthoCamera camera;
        camera.origin = Vec3T(0);
        camera.width  = 20.0;
        camera.height = 20.0;

        TracerT tracer(lsri, ppm, shader, camera);        

        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();

        tracer.run(/*multi-threading = */true);
        
        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nRay-tracing took " << msdiff.total_milliseconds() << " ms\n";

        ppm.save("/tmp/sphere_threaded");
     }
#endif

#ifdef BENCHMARK_TEST
     {// generate an image from a VDB loaded from a file
        openvdb::initialize();
        io::File file("/usr/pic1/Data/OpenVDB/LevelSetModels/crawler.vdb");
        file.open();
        FloatGrid::Ptr ls = gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));//get first grid
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);

        Vec3fGrid::Ptr grad = tools::gradient(*ls);
        
        PPM ppm(2048, 1024);

        //typedef MyShader1<Vec3fGrid> ShaderT;
        typedef MyShader2<Vec3fGrid> ShaderT;
        ShaderT shader(*grad);
        
        typedef TestTracer<tools::LevelSetRayIntersector<FloatGrid>, ShaderT > TracerT;

        TracerT::OrthoCamera camera;
        camera.origin = Vec3T(-200,-100,-100);
        camera.width  = 400.0;
        camera.height = 200.0;
        TracerT tracer(lsri, ppm, shader, camera);        

        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();

        tracer.run(/*multi-threading = */true);        

        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nMulti-threaded ray-tracing took " << msdiff.total_milliseconds() << " ms\n";

        ppm.save("/tmp/crawler");
     }
#endif
}

#undef BENCHMARK_TEST

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
