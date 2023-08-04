// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file TestBenchmark.cu
///
/// @author Ken Museth
///
/// @brief A simple ray-tracing benchmark test.

#include <nanovdb/util/IO.h>// io::readGrid
#include <nanovdb/util/Primitives.h>// createLevelSetTorus
#include "Image.h"
#include "Camera.h"
#include <nanovdb/util/CpuTimer.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>

#include <gtest/gtest.h>

extern "C" void launch_kernels(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&,
                               nanovdb::ImageHandle<nanovdb::CudaDeviceBuffer>&,
                               const nanovdb::Camera<float>*,
                               cudaStream_t stream);

std::string getEnvVar(const std::string& name, const std::string def = "")
{
    const char* str = std::getenv(name.c_str());
    return str == nullptr ? def : std::string(str);
}

TEST(TestBenchmark, NanoVDB_GPU)
{
    using BufferT = nanovdb::CudaDeviceBuffer;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CameraT = nanovdb::Camera<RealT>;
    nanovdb::CpuTimer timer;

    const std::string image_path = getEnvVar("VDB_SCRATCH_PATH", ".");

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
    EXPECT_TRUE(handle.deviceGrid<float>());

    std::cout << "\nRay-tracing NanoVDB grid named \"" << grid->gridName() << "\"" << std::endl;

    const int   width = 1280, height = 720;
    const RealT vfov = 25.0f, aspect = RealT(width) / height, radius = 300.0f;
    const auto  bbox = grid->worldBBox();
    const Vec3T lookat(0.5 * (bbox.min() + bbox.max())), up(0, -1, 0);
    auto        eye = [&lookat, &radius](int angle) {
        const RealT theta = angle * nanovdb::pi<RealT>() / 180.0f;
        return lookat + radius * Vec3T(sin(theta), 0, cos(theta));
    };
    CameraT *host_camera, *dev_camera;
    cudaCheck(cudaMalloc((void**)&dev_camera, sizeof(CameraT))); // un-managed memory on the device
    cudaCheck(cudaMallocHost((void**)&host_camera, sizeof(CameraT)));

    nanovdb::ImageHandle<BufferT> imgHandle(width, height);
    auto*                         img = imgHandle.image();
    imgHandle.deviceUpload(stream, false);

    for (int angle = 0; angle < 6; ++angle) {
        std::stringstream ss;
        ss << "NanoVDB: GPU kernel with " << img->size() << " rays";
        host_camera->update(eye(angle), lookat, up, vfov, aspect);
        cudaCheck(cudaMemcpyAsync(dev_camera, host_camera, sizeof(CameraT), cudaMemcpyHostToDevice, stream));
        timer.start(ss.str());
        launch_kernels(handle, imgHandle, dev_camera, stream);// defined in BenchKernels_nano.cu
        timer.stop();

        //timer.start("Write image to file");
        imgHandle.deviceDownload(stream);
        ss.str("");
        ss.clear();
        ss << image_path << "/nanovdb_gpu_" << std::setfill('0') << std::setw(3) << angle << ".ppm";
        img->writePPM(ss.str(), "Benchmark test");
        //timer.stop();

    } //frame number angle

    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFreeHost(host_camera));
    cudaCheck(cudaFree(dev_camera));
} // NanoVDB_GPU