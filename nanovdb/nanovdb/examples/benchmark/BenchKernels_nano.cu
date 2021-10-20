
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file BenchKernels.cu
///
/// @author Ken Museth
///
/// @brief CUDA kernel for a simple ray-tracing benchmark test.

#include <nanovdb/util/GridHandle.h> // for nanovdb::GridHandle
#include <nanovdb/util/CudaDeviceBuffer.h> // for CUDA memory management
#include <nanovdb/util/Ray.h> // for nanovdb::Ray
#include <nanovdb/util/HDDA.h> // for nanovdb::ZeroCrossing

#include "Image.h"
#include "Camera.h"

// Comment out to disable timing of the CUDA kernel
#define CUDA_TIMING

// This is called by the device
template <typename T>
__global__ void render_kernel(const nanovdb::NanoGrid<T>& grid,
                              const nanovdb::Camera<float>&   camera,
                              nanovdb::Image&                 img)
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;
    using ColorRGB = nanovdb::Image::ColorRGB;

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (w >= img.width() || h >= img.height())
        return;

    const auto& tree = grid.tree();
    const auto& bbox = tree.bbox();
    RayT        ray = camera.getRay(img.u(w), img.v(h));
    ray = ray.worldToIndexF(grid);

    auto   acc = tree.getAccessor();
    CoordT ijk;
    float  t;
    float  v0;
    if (nanovdb::ZeroCrossing(ray, acc, ijk, v0, t)) {
#if 1// second-order central difference
        Vec3T grad(acc.getValue(ijk.offsetBy(1,0,0)) - acc.getValue(ijk.offsetBy(-1,0,0)),
                   acc.getValue(ijk.offsetBy(0,1,0)) - acc.getValue(ijk.offsetBy(0,-1,0)),
                   acc.getValue(ijk.offsetBy(0,0,1)) - acc.getValue(ijk.offsetBy(0,0,-1)));
#else// first order single-sided difference
        Vec3T grad(-v0);
        ijk[0] += 1;
        grad[0] += acc.getValue(ijk);
        ijk[0] -= 1;
        ijk[1] += 1;
        grad[1] += acc.getValue(ijk);
        ijk[1] -= 1;
        ijk[2] += 1;
        grad[2] += acc.getValue(ijk);
#endif
        grad *= rnorm3df(grad[0], grad[1], grad[2]);
        img(w, h) = ColorRGB(abs(grad.dot(ray.dir())), 0, 0);
    } else {
        const int checkerboard = 1 << 7;
        img(w, h) = ((h & checkerboard) ^ (w & checkerboard)) ? ColorRGB(1, 1, 1) : ColorRGB(0, 0, 0);
    }
}

// This is called by the host
extern "C" float launch_kernels(const nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>&  gridHandle,
                               nanovdb::ImageHandle<nanovdb::CudaDeviceBuffer>& imgHandle,
                               const nanovdb::Camera<float>*                          camera,
                               cudaStream_t                                           stream)
{
    using BuildT = nanovdb::FpN;
    const auto* img = imgHandle.image(); // host image!
    auto        round = [](int a, int b) { return (a + b - 1) / b; };
    const dim3  threadsPerBlock(8, 8), numBlocks(round(img->width(), threadsPerBlock.x), round(img->height(), threadsPerBlock.y));
    auto*       deviceGrid = gridHandle.deviceGrid<BuildT>(); // note this cannot be de-referenced since it points to a memory address on the GPU!
    auto*       deviceImage = imgHandle.deviceImage(); // note this cannot be de-referenced since it points to a memory address on the GPU!
    assert(deviceGrid && deviceImage);

#ifdef CUDA_TIMING
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
#endif

    // kernal syntax:  <<<blocks per grid, threads per block, dynamic shared memory per block, stream >>>
    render_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(*deviceGrid, *camera, *deviceImage);

    float elapsedTime = 0.0f;
#ifdef CUDA_TIMING
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    //printf("NanoVDB: GPU kernel with %i rays ... completed in %5.3f milliseconds\n", imgHandle.image()->size(), elapsedTime);
    cudaError_t errCode = cudaGetLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(errCode), __FILE__, __LINE__);
        exit(errCode);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    return elapsedTime;
}