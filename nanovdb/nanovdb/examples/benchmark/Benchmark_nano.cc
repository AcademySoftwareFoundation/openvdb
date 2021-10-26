// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Benchmark_nano.cc
///
/// @author Ken Museth
///
/// @brief A super lightweight and portable ray-tracing benchmark
///        that only depends on NanoVDB (not OpenVDB) and CUDA.

#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include "Image.h"
#include "Camera.h"
#include "../ex_util/CpuTimer.h"

#include <iomanip>// for std::setfill and std::setw


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

extern "C" float launch_kernels(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&,
                                nanovdb::ImageHandle<nanovdb::CudaDeviceBuffer>&,
                                const nanovdb::Camera<float>*,
                                cudaStream_t stream);

int main(int argc, char** argv)
{
    using BufferT = nanovdb::CudaDeviceBuffer;
    using ValueT  = float;
    using BuildT  = nanovdb::FpN;
    using Vec3T   = nanovdb::Vec3<ValueT>;
    using CameraT = nanovdb::Camera<ValueT>;
    nanovdb::CpuTimer<> timer;

    if (argc!=2) {
        std::cerr << "Usage: " << argv[0] << " path/level_set.nvdb" << std::endl;
        std::cerr << "To generate an input file: nanovdb_convert dragon.vdb dragon.nvdb\n";
        return 1;
    }

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
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA driver version:\t%i.%i\n",  driverVersion/1000,  (driverVersion%1000)/10);
    printf("CUDA runtime version:\t%i.%i\n", runtimeVersion/1000, (runtimeVersion%1000)/10);

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

    auto handle = nanovdb::io::readGrid<BufferT>(argv[1]);

    const auto* grid = handle.grid<BuildT>();
    if (!grid || !grid->isLevelSet()) {
        std::cerr << "Error loading NanoVDB level set from file" << std::endl;
        return 1;
    }
    handle.deviceUpload(stream, false);
    std::cout << "\nRay-tracing NanoVDB grid named \"" << grid->gridName() << "\" of size "
              << (grid->gridSize() >> 20) << " MB" << std::endl;

    const int   width = 1280, height = 720;
    const ValueT vfov = 25.0f, aspect = ValueT(width) / height, radius = 300.0f;
    const auto  bbox = grid->worldBBox();
    const Vec3T lookat(0.5 * (bbox.min() + bbox.max())), up(0, -1, 0);
    auto        eye = [&lookat, &radius](int angle) {
        const ValueT theta = angle * M_PI / 180.0f;
        return lookat + radius * Vec3T(sin(theta), 0, cos(theta));
    };
    CameraT *host_camera, *dev_camera;
    cudaCheck(cudaMalloc((void**)&dev_camera, sizeof(CameraT))); // un-managed memory on the device
    cudaCheck(cudaMallocHost((void**)&host_camera, sizeof(CameraT)));

    nanovdb::ImageHandle<BufferT> imgHandle(width, height);
    auto*                         img = imgHandle.image();
    imgHandle.deviceUpload(stream, false);

    float elapsedTime = 0.0f;
    const int maxAngle = 360;
    for (int angle = 0; angle < maxAngle; ++angle) {
        host_camera->update(eye(angle), lookat, up, vfov, aspect);
        cudaCheck(cudaMemcpyAsync(dev_camera, host_camera, sizeof(CameraT), cudaMemcpyHostToDevice, stream));
        elapsedTime += launch_kernels(handle, imgHandle, dev_camera, stream);

        //timer.start("Write image to file");
        imgHandle.deviceDownload(stream);
#if 1
        std::stringstream ss;
        ss << "./nanovdb_gpu_" << std::setfill('0') << std::setw(3) << angle << ".ppm";
        img->writePPM(ss.str(), "Benchmark test");
#endif
        //timer.stop();

    } //frame number angle

    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(host_camera));
    cudaCheck(cudaFree(dev_camera));

    printf("\nRay-traced %i different frames, each with %i rays, in %5.3f ms.\nThis corresponds to an average of %5.3f ms per frame or %5.3f FPS!\n",
           maxAngle, imgHandle.image()->size(), elapsedTime, elapsedTime/maxAngle, 1000.0f*maxAngle/elapsedTime);

    return 0;
}
