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

// Uncomment to enable bechmarks of test ray-tracer
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

#ifdef BENCHMARK_TEST

/// @brief Extremely naive and bare-bone Portable Pixel Map implementation
/// class intended for debugging and unittesting only!
class Film
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
    
    Film(size_t width, size_t height)
        : mWidth(width), mHeight(height), mSize(width*height), mPixels(new RGB[mSize]) {}
    ~Film() { delete mPixels; }
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
    void savePPM(const std::string& fileName)
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

/// @brief For testing of multi-threaded ray-intersection.
/// @details Super naive implementation using z-aligned orthographic projection.
/// @warning ONLY INTENDED FOR DEBUGGING AND UNIT-TESTING!!!!!!!!!!
template<typename RayIntersectorT, typename ShaderT>
class TestTracer
{
public:
    typedef typename RayIntersectorT::GridType GridType;
    typedef typename RayIntersectorT::Vec3Type Vec3Type;
    typedef typename RayIntersectorT::RayType  RayType;
    
    struct OrthoCamera {
        Vec3Type origin;
        double width, height;
    };
    TestTracer(const RayIntersectorT& inter, const ShaderT& shader,
               Film& film, const OrthoCamera& camera)
        : mInter(&inter), mShader(&shader), mFilm(&film), mCamera(&camera) {}
    void run(bool threaded = true)
    {
        tbb::blocked_range<size_t> range(0, mFilm->width());
        if (threaded) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }
    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        Vec3Type xyz(0), eye(mCamera->origin), normal(0);
        RayType ray(eye, Vec3Type(0.0, 0.0, 1.0));//world space ray
        openvdb::tools::LinearIntersector<GridType, /*iterations=*/0> tester(mInter->grid());
        ShaderT shader(*mShader);//local deep copy for thread-safety
        const double di = mCamera->width/mFilm->width(), dj=mCamera->height/mFilm->height();
        for (size_t i=range.begin(), ie = range.end(); i<ie; ++i) {
            eye[0] = di*i + mCamera->origin[0];
            for (size_t j=0, je = mFilm->height(); j<je; ++j) {
                eye[1] = dj*j + mCamera->origin[1],
                ray.setEye(eye);
                if (mInter->intersectsWS(ray, tester, xyz, normal))
                    mFilm->setRGB(i, j, shader(xyz, normal, ray));
                //if (mInter->intersectsWS(ray, tester))
                //    mFilm->setRGB(i, j, Film::RGB(255, 255, 255));
            }
        }
    }
private:
    const RayIntersectorT* mInter;
    const ShaderT*         mShader;
    Film*                  mFilm;
    const OrthoCamera*     mCamera;
};// TestTracer

// Color shading that treats the normal (x, y, z) values as (r, g, b) color components.
struct NormalShader
{
    NormalShader() {}
    template <typename RayType>
    Film::RGB operator()(const openvdb::Vec3R&,
                         const openvdb::Vec3R& normal,
                         const RayType&) const
    {
        return Film::RGB(int(255*openvdb::math::Clamp01((normal[0]+1.0)*0.5)),
                         int(255*openvdb::math::Clamp01((normal[1]+1.0)*0.5)),
                         int(255*openvdb::math::Clamp01((normal[2]+1.0)*0.5)));
    }
};

// Gray-scale shading based on angle between camera ray and surface normal
struct SurfaceShader
{
    SurfaceShader() {}
    template <typename RayType>
    Film::RGB operator()(const openvdb::Vec3R&,
                         const openvdb::Vec3R& normal,
                         const RayType& ray) const
    {
        const float alpha = openvdb::math::Abs(normal.dot(ray.dir()));
        return Film::RGB(int(255*alpha));
    }
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
        
        const Vec3T dir(1.0, 0.0, 0.0);
        const Vec3T eye(2.0, 0.0, 0.0);
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
        const Vec3f c(20.0f, 0.0f, 0.0f);
        const float s = 0.5f, w = 2.0f;
        
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);
        tools::LinearIntersector<FloatGrid> tester(*ls);
        
        const Vec3T dir(1.0,-0.0,-0.0);
        const Vec3T eye(2.0, 0.0, 0.0);
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
        
        const Vec3T dir(0.0, 1.0, 0.0);
        const Vec3T eye(0.0,-2.0, 0.0);
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
        const Vec3f c(0.0f, 20.0f, 0.0f);
        const float s = 1.5f, w = 2.0f;
        
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);
        tools::LinearIntersector<FloatGrid> tester(*ls);
        
        const Vec3T dir(-0.0, 1.0,-0.0);
        const Vec3T eye( 0.0,-2.0, 0.0);
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
        
        const Vec3T dir(0.0, 0.0, 1.0);
        const Vec3T eye(0.0, 0.0, 4.0);
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
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;
        
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
        
        tools::LevelSetRayIntersector<FloatGrid, -1> lsri(*ls);
        tools::LinearIntersector<FloatGrid> tester(*ls);
        
        const Vec3T dir(-0.0,-0.0, 1.0);
        const Vec3T eye( 0.0, 0.0, 4.0);
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
        
        const Vec3T dir(1.0, 1.0, 1.0);
        const Vec3T eye(0.0, 0.0, 0.0);
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
        tools::LinearIntersector<FloatGrid, /*iterations=*/2> tester(*ls);

        Vec3T xyz(0);
        const size_t width = 1024;
        const double dx = 20.0/width;
        const Vec3T dir(0.0, 0.0, 1.0);
#ifdef BENCHMARK_TEST
        Film film(width, width);
        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();
        openvdb::math::Stats stats;
        openvdb::math::Histogram hist(0.0, 0.1, 20);
#endif
        for (size_t i=0; i<width; ++i) {
            for (size_t j=0; j<width; ++j) {
                const Vec3T eye(dx*i, dx*j, 0.0);
                const RayT ray(eye, dir);
                if (lsri.intersectsWS(ray, tester, xyz)){
                    CPPUNIT_ASSERT(ray.intersects(c, r, t0, t1));
#ifdef BENCHMARK_TEST
                    double delta = (ray(t0)-xyz).length()/s;//in voxel units
                    stats.add(delta);
                    hist.add(delta);
                    if (delta > 0.01) {
                        film.setRGB(i, j, Film::RGB(255,  0,  0));
                    } else {
                        film.setRGB(i, j, Film::RGB(0,255,0));
                    }
#endif
                }
            }
        }
#ifdef BENCHMARK_TEST
        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nRay-tracing took " << msdiff.total_milliseconds() << " ms\n";

        film.savePPM("/tmp/sphere_serial");
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
        
        Film film(1024, 1024);
        
        typedef NormalShader ShaderT;
        //typedef SurfaceShader ShaderT;
        ShaderT shader;
        
        typedef TestTracer<tools::LevelSetRayIntersector<FloatGrid>, ShaderT> TracerT;

        TracerT::OrthoCamera camera;
        camera.origin = Vec3T(0);
        camera.width  = 20.0;
        camera.height = 20.0;

        TracerT tracer(lsri, shader, film, camera);        

        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();

        tracer.run(/*multi-threading = */true);
        
        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nRay-tracing the sphere took " << msdiff.total_milliseconds() << " ms\n";

        film.savePPM("/tmp/sphere_threaded");
    }
#endif

#ifdef BENCHMARK_TEST
    {// generate an image from a VDB loaded from a file
        openvdb::initialize();
        io::File file("/usr/pic1/Data/OpenVDB/LevelSetModels/crawler.vdb");
        file.open();
        FloatGrid::Ptr ls = gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));//get first grid
        typedef tools::LevelSetRayIntersector<FloatGrid> IntersectorT;
        IntersectorT lsri(*ls);
        
        Film film(2048, 1024);
        
        //typedef NormalShader ShaderT;
        typedef SurfaceShader ShaderT;
        ShaderT shader;
        
        typedef TestTracer<IntersectorT, ShaderT > TracerT;

        TracerT::OrthoCamera camera;
        camera.origin = Vec3T(-200,-100,-100);
        camera.width  = 400.0;
        camera.height = 200.0;
        TracerT tracer(lsri, shader, film, camera);

        boost::posix_time::ptime mst1 = boost::posix_time::microsec_clock::local_time();

        tracer.run(/*multi-threading = */true);        

        boost::posix_time::ptime mst2 = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration msdiff = mst2 - mst1;
        std::cerr << "\nRay-tracing the crawler took " << msdiff.total_milliseconds() << " ms\n";

        film.savePPM("/tmp/crawler");
    }
#endif
}

#undef BENCHMARK_TEST

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )


