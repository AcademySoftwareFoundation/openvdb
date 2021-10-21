
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file BenchKernels_dense.cu
///
/// @author Ken Museth
///
/// @brief CUDA kernel for a simple ray-tracing benchmark test.

#include "DenseGrid.h"
#include <nanovdb/util/CudaDeviceBuffer.h> // for CUDA memory management
#include <nanovdb/util/Ray.h> // for nanovdb::Ray
#include <nanovdb/util/HDDA.h> // for nanovdb::DDA

#include "Image.h"
#include "Camera.h"

// Comment out to disable timing of the CUDA kernel
#define CUDA_TIMING

// This is called by the device
__global__ void render_kernel(const nanovdb::DenseGrid<float>& grid,
                              const nanovdb::Camera<float>&    camera,
                              nanovdb::Image&                  img)
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;
    using ColorRGB = nanovdb::Image::ColorRGB;

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (w >= img.width() || h >= img.height()) return;
    RayT ray = camera.getRay(img.u(w), img.v(h));// ray in world space
    ray = ray.worldToIndexF(grid);// ray in index space
    if (ray.clip(grid.indexBBox().expandBy(-1))) {// clip to the index bounding box
        nanovdb::DDA<RayT> dda(ray);
        const float v0 = grid.getValue(dda.voxel());
        while( dda.step() ) {
            CoordT ijk = dda.voxel();
            const float v1 = grid.getValue(ijk);
            if (v0*v1>0) continue;
#if 1// second-order central difference
            Vec3T grad(grid.getValue(ijk.offsetBy(1,0,0)) - grid.getValue(ijk.offsetBy(-1,0,0)),
                       grid.getValue(ijk.offsetBy(0,1,0)) - grid.getValue(ijk.offsetBy(0,-1,0)),
                       grid.getValue(ijk.offsetBy(0,0,1)) - grid.getValue(ijk.offsetBy(0,0,-1)));
#else// first order single-sided difference
            Vec3T grad(-v0);
            ijk[0] += 1;
            grad[0] += grid.getValue(ijk);
            ijk[0] -= 1;
            ijk[1] += 1;
            grad[1] += grid.getValue(ijk);
            ijk[1] -= 1;
            ijk[2] += 1;
            grad[2] += grid.getValue(ijk);
#endif
            grad *= rnorm3df(grad[0], grad[1], grad[2]);
            img(w, h) = ColorRGB(abs(grad.dot(ray.dir())), 0, 0);
            return;
        }
    }
    const int checkerboard = 1 << 7;
    img(w, h) = ((h & checkerboard) ^ (w & checkerboard)) ? ColorRGB(1, 1, 1) : ColorRGB(0, 0, 0);
}

// This is called by the host
extern "C" float launch_kernels(const nanovdb::DenseGridHandle<nanovdb::CudaDeviceBuffer>&  gridHandle,
                               nanovdb::ImageHandle<nanovdb::CudaDeviceBuffer>&             imgHandle,
                               const nanovdb::Camera<float>*                                camera,
                               cudaStream_t                                                 stream)
{
    const auto* img = imgHandle.image(); // host image!
    auto        round = [](int a, int b) { return (a + b - 1) / b; };
    const dim3  threadsPerBlock(8, 8), numBlocks(round(img->width(), threadsPerBlock.x), round(img->height(), threadsPerBlock.y));
    auto*       deviceGrid = gridHandle.deviceGrid<float>(); // note this cannot be de-referenced since it points to a memory address on the GPU!
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
    //printf("DenseGrid: GPU kernel with %i rays ... completed in %5.3f milliseconds\n", imgHandle.image()->size(), elapsedTime);
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