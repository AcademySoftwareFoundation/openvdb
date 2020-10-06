// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Benchmark.cpp
///
/// @author Ken Museth
///
/// @brief A simple ray-tracing benchmark test.

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "gtest/gtest.h"

#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include "Image.h"
#include "Camera.h"
#include "../ex_util/CpuTimer.h"

#if defined(NANOVDB_USE_CUDA)
#include <nanovdb/util/CudaDeviceBuffer.h>
#endif

#if defined(NANOVDB_USE_OPENVDB)
#include <nanovdb/util/OpenToNanoVDB.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/Transform.h>
#else
#include <nanovdb/util/GridBuilder.h>
#endif

#if defined(NANOVDB_USE_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#endif

inline std::ostream&
operator<<(std::ostream& os, const nanovdb::CoordBBox& b)
{
    os << "(" << b[0][0] << "," << b[0][1] << "," << b[0][2] << ") ->"
       << "(" << b[1][0] << "," << b[1][1] << "," << b[1][2] << ")";
    return os;
}

inline std::ostream&
operator<<(std::ostream& os, const nanovdb::Coord& ijk)
{
    os << "(" << ijk[0] << "," << ijk[1] << "," << ijk[2] << ")";
    return os;
}

template <typename T>
inline std::ostream&
operator<<(std::ostream& os, const nanovdb::Vec3<T>& v)
{
    os << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
    return os;
}

// define the enviroment variable VDB_DATA_PATH to use models from the web
// e.g. setenv VDB_DATA_PATH /home/kmu/dev/data/vdb
// or   export VDB_DATA_PATH=/Users/ken/dev/data/vdb

// define the enviroment variable VDB_SCRATCH_PATH to specify the directory where image are saved

// The fixture for testing class.
class Benchmark : public ::testing::Test
{
protected:
    Benchmark() {}

    ~Benchmark() override {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }
    std::string getEnvVar(const std::string& name, const std::string def = "") const
    {
        const char* str = std::getenv(name.c_str());
        return str == nullptr ? def : std::string(str);
    }

#if defined(NANOVDB_USE_OPENVDB)
    openvdb::FloatGrid::Ptr getSrcGrid(int verbose = 1)
    {
        openvdb::FloatGrid::Ptr grid;
        const std::string       path = this->getEnvVar("VDB_DATA_PATH");
        if (path.empty()) { // create a narrow-band level set sphere
            const float          radius = 500.0f;
            const openvdb::Vec3f center(0.0f, 0.0f, 0.0f);
            const float          voxelSize = 1.0f, width = 3.0f;
            if (verbose > 0) {
                std::stringstream ss;
                ss << "Generating level set sphere with a radius of " << radius << " voxels";
                mTimer.start(ss.str());
            }
#if 1 // choose between a sphere or one of five platonic solids
            grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);
#else
            const int faces[5] = {4, 6, 8, 12, 20};
            grid = openvdb::tools::createLevelSetPlatonic<openvdb::FloatGrid>(faces[4], radius, center, voxelSize, width);
#endif
        } else {
            openvdb::initialize();
            const std::vector<std::string> models = {"armadillo.vdb", "buddha.vdb", "bunny.vdb", "crawler.vdb", "dragon.vdb", "iss.vdb", "space.vdb", "torus_knot_helix.vdb", "utahteapot.vdb", "bunny_cloud.vdb", "wdas_cloud.vdb"};
            const std::string              fileName = path + "/" + models[4]; //
            if (verbose > 0)
                mTimer.start("Reading grid from the file \"" + fileName + "\"");
            openvdb::io::File file(fileName);
            file.open(false); //disable delayed loading
            grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(file.beginName().gridName()));
        }
        if (verbose > 0)
            mTimer.stop();
        if (verbose > 1)
            grid->print(std::cout, 3);
        return grid;
    }
#endif
    nanovdb::CpuTimer<> mTimer;
}; // Benchmark

TEST_F(Benchmark, Ray)
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using CoordBBoxT = nanovdb::BBox<CoordT>;
    using BBoxT = nanovdb::BBox<Vec3T>;
    using RayT = nanovdb::Ray<RealT>;

    {// clip ray against an index bbox
        // test bbox clip
        const Vec3T dir(-1.0, 2.0, 3.0);
        const Vec3T eye(2.0, 1.0, 1.0);
        RealT       t0 = 0.1, t1 = 12589.0;
        RayT        ray(eye, dir, t0, t1);

        // intersects the two faces of the box perpendicular to the y-axis!
        EXPECT_TRUE(ray.clip(CoordBBoxT(CoordT(0, 2, 2), CoordT(2, 4, 6))));
        //std::cerr << "t0 = " << ray.t0() << ", ray.t1() = " << ray.t1() << std::endl;
        //std::cerr << "ray(0.5) = " << ray(0.5) << std::endl;
        //std::cerr << "ray(1.5) = " << ray(1.5) << std::endl;
        //std::cerr << "ray(2.0) = " << ray(2.0) << std::endl;
        EXPECT_EQ(0.5, ray.t0());
        EXPECT_EQ(1.5, ray.t1());
        EXPECT_EQ(ray(0.5)[1], 2); //lower y component of intersection
        EXPECT_EQ(ray(1.5)[1], 4); //higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // intersects the lower edge anlong the z-axis of the box
        EXPECT_TRUE(ray.clip(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0))));
        EXPECT_EQ(0.5, ray.t0());
        EXPECT_EQ(0.5, ray.t1());
        EXPECT_EQ(ray(0.5)[0], 1.5); //lower y component of intersection
        EXPECT_EQ(ray(0.5)[1], 2.0); //higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // no intersections
        EXPECT_TRUE(!ray.clip(CoordBBoxT(CoordT(4, 2, 2), CoordT(6, 4, 6))));
        EXPECT_EQ(t0, ray.t0());
        EXPECT_EQ(t1, ray.t1());
    }
    {// clip ray against an real bbox
        // test bbox clip
        const Vec3T dir(-1.0, 2.0, 3.0);
        const Vec3T eye(2.0, 1.0, 1.0);
        RealT       t0 = 0.1, t1 = 12589.0;
        RayT        ray(eye, dir, t0, t1);

        // intersects the two faces of the box perpendicular to the y-axis!
        EXPECT_TRUE( ray.clip(CoordBBoxT(CoordT(0, 2, 2), CoordT(2, 4, 6)).asReal<double>()) );
        //std::cerr << "t0 = " << ray.t0() << ", ray.t1() = " << ray.t1() << std::endl;
        //std::cerr << "ray(0.5) = " << ray(0.5) << std::endl;
        //std::cerr << "ray(1.5) = " << ray(1.5) << std::endl;
        //std::cerr << "ray(2.0) = " << ray(2.0) << std::endl;
        EXPECT_EQ(0.5, ray.t0());
        EXPECT_EQ(2.0, ray.t1());
        EXPECT_EQ(ray(0.5)[1], 2); //lower y component of intersection
        EXPECT_EQ(ray(1.5)[1], 4); //higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // intersects the lower edge anlong the z-axis of the box
        EXPECT_TRUE( ray.clip(BBoxT(Vec3T(1.5, 2.0, 2.0), Vec3T(4.5, 4.0, 6.0))) );
        EXPECT_EQ(0.5, ray.t0());
        EXPECT_EQ(0.5, ray.t1());
        EXPECT_EQ(ray(0.5)[0], 1.5); //lower y component of intersection
        EXPECT_EQ(ray(0.5)[1], 2.0); //higher y component of intersection

        ray.reset(eye, dir, t0, t1);
        // no intersections
        EXPECT_TRUE(!ray.clip(CoordBBoxT(CoordT(4, 2, 2), CoordT(6, 4, 6)).asReal<double>()) );
        EXPECT_EQ(t0, ray.t0());
        EXPECT_EQ(t1, ray.t1());
    }
}

TEST_F(Benchmark, HDDA)
{
    using RealT = float;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;
    using Vec3T = RayT::Vec3T;

    { // basic test
        using DDAT = nanovdb::HDDA<RayT, CoordT>;
        const RayT::Vec3T dir(1.0, 0.0, 0.0);
        const RayT::Vec3T eye(-1.0, 0.0, 0.0);
        const RayT        ray(eye, dir);
        DDAT              dda(ray, 1 << (3 + 4 + 5));
        EXPECT_EQ(nanovdb::Delta<RealT>::value(), dda.time());
        EXPECT_EQ(1.0, dda.next());
        dda.step();
        EXPECT_EQ(1.0, dda.time());
        EXPECT_EQ(4096 + 1.0, dda.next());
    }
    { // Check for the notorious +-0 issue!
        using DDAT = nanovdb::HDDA<RayT, CoordT>;

        const Vec3T dir1(1.0, 0.0, 0.0);
        const Vec3T eye1(2.0, 0.0, 0.0);
        const RayT  ray1(eye1, dir1);
        DDAT        dda1(ray1, 1 << 3);
        dda1.step();

        const Vec3T dir2(1.0, -0.0, -0.0);
        const Vec3T eye2(2.0, 0.0, 0.0);
        const RayT  ray2(eye2, dir2);
        DDAT        dda2(ray2, 1 << 3);
        dda2.step();

        const Vec3T dir3(1.0, -1e-9, -1e-9);
        const Vec3T eye3(2.0, 0.0, 0.0);
        const RayT  ray3(eye3, dir3);
        DDAT        dda3(ray3, 1 << 3);
        dda3.step();

        const Vec3T dir4(1.0, -1e-9, -1e-9);
        const Vec3T eye4(2.0, 0.0, 0.0);
        const RayT  ray4(eye3, dir4);
        DDAT        dda4(ray4, 1 << 3);
        dda4.step();

        EXPECT_EQ(dda1.time(), dda2.time());
        EXPECT_EQ(dda2.time(), dda3.time());
        EXPECT_EQ(dda3.time(), dda4.time());
        EXPECT_EQ(dda1.next(), dda2.next());
        EXPECT_EQ(dda2.next(), dda3.next());
        EXPECT_EQ(dda3.next(), dda4.next());
    }
    { // test voxel traversal along both directions of each axis
        using DDAT = nanovdb::HDDA<RayT>;
        const Vec3T eye(0, 0, 0);
        for (int s = -1; s <= 1; s += 2) {
            for (int a = 0; a < 3; ++a) {
                const int   d[3] = {s * (a == 0), s * (a == 1), s * (a == 2)};
                const Vec3T dir(d[0], d[1], d[2]);
                RayT        ray(eye, dir);
                DDAT        dda(ray, 1 << 0);
                for (int i = 1; i <= 10; ++i) {
                    EXPECT_TRUE(dda.step());
                    EXPECT_EQ(i, dda.time());
                }
            }
        }
    }
    { // test Node traversal along both directions of each axis
        using DDAT = nanovdb::HDDA<RayT, CoordT>;
        const Vec3T eye(0, 0, 0);

        for (int s = -1; s <= 1; s += 2) {
            for (int a = 0; a < 3; ++a) {
                const int   d[3] = {s * (a == 0), s * (a == 1), s * (a == 2)};
                const Vec3T dir(d[0], d[1], d[2]);
                RayT        ray(eye, dir);
                DDAT        dda(ray, 1 << 3);
                for (int i = 1; i <= 10; ++i) {
                    EXPECT_TRUE(dda.step());
                    EXPECT_EQ(8 * i, dda.time());
                }
            }
        }
    }
    { // test accelerated Node traversal along both directions of each axis
        using DDAT = nanovdb::HDDA<RayT, CoordT>;
        const Vec3T eye(0, 0, 0);

        for (int s = -1; s <= 1; s += 2) {
            for (int a = 0; a < 3; ++a) {
                const int   d[3] = {s * (a == 0), s * (a == 1), s * (a == 2)};
                const Vec3T dir(2 * d[0], 2 * d[1], 2 * d[2]);
                RayT        ray(eye, dir);
                DDAT        dda(ray, 1 << 3);
                double      next = 0;
                for (int i = 1; i <= 10; ++i) {
                    EXPECT_TRUE(dda.step());
                    EXPECT_EQ(4 * i, dda.time());
                    if (i > 1) {
                        EXPECT_EQ(dda.time(), next);
                    }
                    next = dda.next();
                }
            }
        }
    }
} // HDDA

#if defined(NANOVDB_USE_OPENVDB)
TEST_F(Benchmark, OpenVDB_CPU)
{
    using GridT = openvdb::FloatGrid;
    using CoordT = openvdb::Coord;
    using ColorRGB = nanovdb::Image::ColorRGB;
    using RealT = float;
    using Vec3T = openvdb::math::Vec3<RealT>;
    using RayT = openvdb::math::Ray<RealT>;

    const std::string image_path = this->getEnvVar("VDB_SCRATCH_PATH", ".");

    auto srcGrid = this->getSrcGrid();
    mTimer.start("Generating NanoVDB grid");
    auto handle = nanovdb::openToNanoVDB(*srcGrid, /*mortonSort=*/false, /*verbose=*/0);
    mTimer.restart("Writing NanoVDB grid");
#if defined(NANOVDB_USE_BLOSC)
    nanovdb::io::writeGrid("data/test.nvdb", handle, nanovdb::io::Codec::BLOSC);
#elif defined(NANOVDB_USE_ZIP)
    nanovdb::io::writeGrid("data/test.nvdb", handle, nanovdb::io::Codec::ZIP);
#else
    nanovdb::io::writeGrid("data/test.nvdb", handle, nanovdb::io::Codec::NONE);
#endif
    mTimer.stop();

    const int            width = 1280, height = 720;
    const RealT          vfov = 25.0f, aspect = RealT(width) / height, radius = 300.0f;
    const auto           bbox = srcGrid->evalActiveVoxelBoundingBox();
    const openvdb::Vec3d center(0.5 * (bbox.max()[0] + bbox.min()[0]),
                                0.5 * (bbox.max()[1] + bbox.min()[1]),
                                0.5 * (bbox.max()[2] + bbox.min()[2]));
    const Vec3T          lookat = srcGrid->indexToWorld(center), up(0, -1, 0);
    auto                 eye = [&lookat, &radius](int angle) {
        const RealT theta = angle * M_PI / 180.0f;
        return lookat + radius * Vec3T(sin(theta), 0, cos(theta));
    };

    nanovdb::Camera<RealT, Vec3T, RayT> camera(eye(0), lookat, up, vfov, aspect);

    nanovdb::ImageHandle<> imgHandle(width, height);
    auto*                  img = imgHandle.image();

    auto kernel2D = [&](const tbb::blocked_range2d<int>& r) {
        openvdb::tools::LevelSetRayIntersector<GridT, openvdb::tools::LinearSearchImpl<GridT, 0, RealT>, GridT::TreeType::RootNodeType::ChildNodeType::LEVEL, RayT> tester(*srcGrid);
        const RealT                                                                                                                                                 wScale = 1.0f / width, hScale = 1.0f / height;
        auto                                                                                                                                                        acc = srcGrid->getAccessor();
        CoordT                                                                                                                                                      ijk;
        Vec3T                                                                                                                                                       xyz;
        float                                                                                                                                                       v;
        for (int w = r.rows().begin(); w != r.rows().end(); ++w) {
            for (int h = r.cols().begin(); h != r.cols().end(); ++h) {
                const RayT wRay = camera.getRay(w * wScale, h * hScale);
                RayT       iRay = wRay.applyInverseMap(*srcGrid->transform().baseMap());
                if (tester.intersectsIS(iRay, xyz)) {
                    ijk = openvdb::Coord::floor(xyz);
                    v = acc.getValue(ijk);
                    Vec3T grad(-v);
                    ijk[0] += 1;
                    grad[0] += acc.getValue(ijk);
                    ijk[0] -= 1;
                    ijk[1] += 1;
                    grad[1] += acc.getValue(ijk);
                    ijk[1] -= 1;
                    ijk[2] += 1;
                    grad[2] += acc.getValue(ijk);
                    grad.normalize();
                    (*img)(w, h) = ColorRGB(std::abs(grad.dot(iRay.dir())), 0, 0);
                } else {
                    const int checkerboard = 1 << 7;
                    (*img)(w, h) = ((h & checkerboard) ^ (w & checkerboard)) ? ColorRGB(1, 1, 1) : ColorRGB(0, 0, 0);
                }
            }
        }
    }; // kernel

    for (int angle = 0; angle < 6; ++angle) {
        camera.update(eye(angle), lookat, up, vfov, aspect);
        std::stringstream ss;
        ss << "OpenVDB: CPU kernel with " << img->size() << " rays";
        tbb::blocked_range2d<int> range2D(0, img->width(), 0, img->height());
        mTimer.start(ss.str());
#if 1
        tbb::parallel_for(range2D, kernel2D);
#else
        kernel2D(range2D);
#endif
        mTimer.stop();

        //mTimer.restart("Write image to file");
        ss.str("");
        ss.clear();
        ss << image_path << "/openvdb_cpu_" << std::setfill('0') << std::setw(3) << angle << ".ppm";
        img->writePPM(ss.str(), "Benchmark test");
        //mTimer.stop();
    } // loop over angle
} // OpenVDB_CPU
#endif

TEST_F(Benchmark, NanoVDB_CPU)
{
    using CoordT = nanovdb::Coord;
    using ColorRGB = nanovdb::Image::ColorRGB;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

    const std::string image_path = this->getEnvVar("VDB_SCRATCH_PATH", ".");

    mTimer.start("Reading and allocating nvdb");
#if defined(NANOVDB_USE_OPENVDB)
    auto handles = nanovdb::io::readGrids("data/test.nvdb");
#else
    std::vector<nanovdb::GridHandle<>> handles;
    handles.push_back(nanovdb::createLevelSetTorus<float>(100.0f, 50.0f));
#endif
    mTimer.stop();
    auto* grid = handles[0].grid<float>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->isLevelSet());

    const int   width = 1280, height = 720;
    const RealT vfov = 25.0f, aspect = RealT(width) / height, radius = 300.0f;
    const auto  bbox = grid->worldBBox();
    const Vec3T lookat(0.5 * (bbox.min() + bbox.max())), up(0, -1, 0);
    auto        eye = [&lookat, &radius](int angle) {
        const RealT theta = angle * M_PI / 180.0f;
        return lookat + radius * Vec3T(sin(theta), 0, cos(theta));
    };

    nanovdb::Camera<RealT> camera(eye(0), lookat, up, vfov, aspect);

    nanovdb::ImageHandle<> imgHandle(width, height);
    auto*                  img = imgHandle.image();

    auto kernel2D = [&](int x0, int y0, int x1, int y1) {
        const RealT wScale = 1.0f / width, hScale = 1.0f / height;
        //std::cerr << "\nActive voxel count = " << grid->activeVoxelCount() << std::endl;
        auto        acc = grid->getAccessor();
        CoordT      ijk;
        float       v;
        float       t;
        for (int w = x0; w != x1; ++w) {
            for (int h = y0; h != y1; ++h) {
                const RayT wRay = camera.getRay(w * wScale, h * hScale);
                RayT       iRay = wRay.worldToIndexF(*grid);
                if (nanovdb::ZeroCrossing(iRay, acc, ijk, v, t)) {
                    Vec3T grad(-v);
                    ijk[0] += 1;
                    grad[0] += acc.getValue(ijk);
                    ijk[0] -= 1;
                    ijk[1] += 1;
                    grad[1] += acc.getValue(ijk);
                    ijk[1] -= 1;
                    ijk[2] += 1;
                    grad[2] += acc.getValue(ijk);
                    grad.normalize();
                    (*img)(w, h) = ColorRGB(std::abs(grad.dot(iRay.dir())), 0, 0);
                } else {
                    const int checkerboard = 1 << 7;
                    (*img)(w, h) = ((h & checkerboard) ^ (w & checkerboard)) ? ColorRGB(1, 1, 1) : ColorRGB(0, 0, 0);
                }
            }
        }
    }; // kernel

    for (int angle = 0; angle < 360; ++angle) {
        camera.update(eye(angle), lookat, up, vfov, aspect);
        std::stringstream ss;
        ss << "NanoVDB: CPU kernel with " << img->size() << " rays";
        //mTimer.start(ss.str());
#if defined(NANOVDB_USE_TBB)
        tbb::blocked_range2d<int> range(0, img->width(), 0, img->height());
        tbb::parallel_for(range, [&](const tbb::blocked_range2d<int>& r) {
            kernel2D(r.rows().begin(), r.cols().begin(), r.rows().end(), r.cols().end());
        });
#else
        kernel2D(0, 0, img->width(), img->height());
#endif
        //mTimer.restart("Write image to file");
        ss.str("");
        ss.clear();
        ss << image_path << "/nanovdb_cpu_" << std::setfill('0') << std::setw(3) << angle << ".ppm";
        img->writePPM(ss.str(), "Benchmark test");
        //mTimer.stop();
    } // loop over angle
} // NanoVDB_CPU

#if defined(NANOVDB_USE_CUDA)

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
#define cudaCheck(ans) \
    { \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

static inline bool gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
        return false;
    }
#endif
    return true;
}

extern "C" void launch_kernels(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&,
                               nanovdb::ImageHandle<nanovdb::CudaDeviceBuffer>&,
                               const nanovdb::Camera<float>*,
                               cudaStream_t stream);

TEST_F(Benchmark, NanoVDB_GPU)
{
    using BufferT = nanovdb::CudaDeviceBuffer;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CameraT = nanovdb::Camera<RealT>;

    const std::string image_path = this->getEnvVar("VDB_SCRATCH_PATH", ".");

    // The first CUDA run time call initializes the CUDA sub-system (loads the runtime API) which takes time!
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
               device,
               deviceProp.major,
               deviceProp.minor);
    }
    cudaSetDevice(0);

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

#if defined(NANOVDB_USE_OPENVDB)
    auto handle = nanovdb::io::readGrid<BufferT>("data/test.nvdb");
#else
    auto handle = nanovdb::createLevelSetTorus<float, BufferT>(100.0f, 50.0f);
#endif
    //auto        handle = nanovdb::io::readGrid<BufferT>("data/test.nvdb");
    const auto* grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->isLevelSet());
    EXPECT_FALSE(grid->isFogVolume());
    handle.deviceUpload(stream, false);
    std::cout << "\nRay-tracing NanoVDB grid named \"" << grid->gridName() << "\"" << std::endl;

    const int   width = 1280, height = 720;
    const RealT vfov = 25.0f, aspect = RealT(width) / height, radius = 300.0f;
    const auto  bbox = grid->worldBBox();
    const Vec3T lookat(0.5 * (bbox.min() + bbox.max())), up(0, -1, 0);
    auto        eye = [&lookat, &radius](int angle) {
        const RealT theta = angle * M_PI / 180.0f;
        return lookat + radius * Vec3T(sin(theta), 0, cos(theta));
    };
    CameraT *host_camera, *dev_camera;
    cudaCheck(cudaMalloc((void**)&dev_camera, sizeof(CameraT))); // un-managed memory on the device
    cudaCheck(cudaMallocHost((void**)&host_camera, sizeof(CameraT)));

    nanovdb::ImageHandle<BufferT> imgHandle(width, height);
    auto*                         img = imgHandle.image();
    imgHandle.deviceUpload(stream, false);

    for (int angle = 0; angle < 360; ++angle) {
        host_camera->update(eye(angle), lookat, up, vfov, aspect);
        cudaCheck(cudaMemcpyAsync(dev_camera, host_camera, sizeof(CameraT), cudaMemcpyHostToDevice, stream));
        launch_kernels(handle, imgHandle, dev_camera, stream);

        //mTimer.start("Write image to file");
        imgHandle.deviceDownload(stream);
        std::stringstream ss;
        ss << image_path << "/nanovdb_gpu_" << std::setfill('0') << std::setw(3) << angle << ".ppm";
        img->writePPM(ss.str(), "Benchmark test");
        //mTimer.stop();

    } //frame number angle

    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(host_camera));
    cudaCheck(cudaFree(dev_camera));
} // NanoVDB_GPU
#endif

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}