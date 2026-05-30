// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <nanovdb/NanoVDB.h>
#include "ComputePrimitives.h"

struct RenderOp;
template<typename GridT>
__global__ void renderIsoSurfacePersistentKernel(RenderOp renderOp, float* image, const GridT* grid, int numPixels, int* nextPixel);

struct RenderOp
{
    using Vec3T  = nanovdb::math::Vec3<float>;
    using RayT   = nanovdb::math::Ray<float>;
    int mWidth, mHeight;
    float mDx, mIso, mWBBoxDimZ;
    Vec3T mWBBoxCenter;

    template<typename BufferT> 
    RenderOp(nanovdb::GridHandle<BufferT>& handle, int width, int height)
    {
        mWidth = width;
        mHeight = height;
        const auto *metaData = handle.gridMetaData();
        mDx = float(metaData->voxelSize()[0]);
        mIso = mDx;
        mWBBoxDimZ = (float)metaData->worldBBox().dim()[2] * 2;
        mWBBoxCenter = Vec3T(metaData->worldBBox().min() + metaData->worldBBox().dim() * 0.5f);
    }

    template<typename GridT>
    inline float renderImage(bool useCuda, float* image, const GridT* grid)
    {
        using ClockT = std::chrono::high_resolution_clock;
        auto t0 = ClockT::now();

        computeForEach(
            useCuda, mWidth * mHeight, 512, __FILE__, __LINE__, [this, image, grid] __hostdev__(int start, int end) {
            (*this)(start, end, image, grid);
        });
        computeSync(useCuda, __FILE__, __LINE__);

        auto t1 = ClockT::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
        return duration;
    }

    template <typename GridT>
    inline __hostdev__ void operator()(int start, int end, float* image, const GridT* grid) const
    {
        static_assert(nanovdb::util::is_same<typename GridT::BuildType, float, nanovdb::ValueOnIndex, nanovdb::ValueIndex>::value, "only works for float and OnIndex grids");
        auto acc = nanovdb::getAccessor<GridT, float>(*grid);
        for (int i = start; i < end; ++i) {
            this->renderPixel(i, image, grid, acc);
        }
    }

    template <typename GridT, typename AccT>
    inline __hostdev__ void renderPixel(int i, float* image, const GridT* grid, AccT& acc) const
    {
        float          t0, v;
        nanovdb::Coord ijk;
        RayT           iRay = this->getIndexRay(i, grid);
        if (nanovdb::math::isoCrossing(iRay, acc, ijk, v, t0, mIso)) {// intersect...
            this->composite(image, i, (t0 * mDx) / (mWBBoxDimZ * 2), 1.0f);
        } else {
            this->composite(image, i, 0.0f, 0.0f);// write background value.
        }
    }

    template<typename GridT>
    inline float renderImagePersistent(float* image, const GridT* grid, int* nextPixel) const
    {
        int device = 0;
        NANOVDB_CUDA_CHECK_ERROR(cudaGetDevice(&device), __FILE__, __LINE__);

        cudaDeviceProp properties;
        NANOVDB_CUDA_CHECK_ERROR(cudaGetDeviceProperties(&properties, device), __FILE__, __LINE__);

        constexpr int blockSize = 256;
        // Launch a small, fixed pool of blocks that persists on the GPU and
        // pulls pixel work from a global counter instead of launching one
        // logical thread per pixel up front.
        int           blockCount = properties.multiProcessorCount * 4;
        if (blockCount < 1) blockCount = 1;

        // Reset the work queue before each timed render. The kernel advances
        // this counter by one warp of pixels at a time.
        NANOVDB_CUDA_CHECK_ERROR(cudaMemset(nextPixel, 0, sizeof(int)), __FILE__, __LINE__);

        using ClockT = std::chrono::high_resolution_clock;
        auto t0 = ClockT::now();

        const int numPixels = mWidth * mHeight;
        renderIsoSurfacePersistentKernel<GridT><<<blockCount, blockSize>>>(*this, image, grid, numPixels, nextPixel);
        NANOVDB_CUDA_CHECK_ERROR(cudaGetLastError(), __FILE__, __LINE__);
        NANOVDB_CUDA_CHECK_ERROR(cudaDeviceSynchronize(), __FILE__, __LINE__);

        auto t1 = ClockT::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
        return duration;
    }

    template <typename GridT>  
    inline __hostdev__ RayT getIndexRay(int i, const GridT *grid) const
    {
        // perspective camera along Z-axis...
        const uint32_t x = i % mWidth, y = i / mWidth;
        const float fov = 45.f;
        const float u = (float(x) + 0.5f) / mWidth;
        const float v = (float(y) + 0.5f) / mHeight;
        const float aspect = mWidth / float(mHeight);
        const float Px = (2.f * u - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f) * aspect;
        const float Py = (2.f * v - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f);
        const Vec3T origin = mWBBoxCenter + Vec3T(0, 0, mWBBoxDimZ);
        Vec3T       dir(Px, Py, -1.f);
        dir.normalize();
        RayT wRay(origin, dir);
        return wRay.worldToIndexF(*grid);// transform the ray to the grid's index-space.
    }

    inline __hostdev__ void composite(float* outImage, int offset, float value, float alpha) const
    {
        const uint32_t x = offset % mWidth, y = offset / mWidth;

        // checkerboard background...
        const int   mask = 1 << 7;
        const float bg = ((x & mask) ^ (y & mask)) ? 1.0f : 0.5f;
        outImage[offset] = alpha * value + (1.0f - alpha) * bg;
    }

    inline void saveImage(const std::string& filename, const float* image) const
    {
        const auto isLittleEndian = []() -> bool {
            static int  x = 1;
            static bool result = reinterpret_cast<uint8_t*>(&x)[0] == 1;
            return result;
        };

        float scale = 1.0f;
        if (isLittleEndian()) scale = -scale;

        std::fstream fs(filename, std::ios::out | std::ios::binary);
        if (!fs.is_open()) throw std::runtime_error("Unable to open file: " + filename);

        fs << "Pf\n"
           << mWidth << "\n"
           << mHeight << "\n"
           << scale << "\n";

        for (int i = 0; i < mWidth * mHeight; ++i) {
            float r = image[i];
            fs.write((char*)&r, sizeof(float));
        }
    }
};

template<typename GridT>
__global__ void renderIsoSurfacePersistentKernel(RenderOp renderOp, float* image, const GridT* grid, int numPixels, int* nextPixel)
{
    static_assert(nanovdb::util::is_same<typename GridT::BuildType, float, nanovdb::ValueOnIndex, nanovdb::ValueIndex>::value, "only works for float and OnIndex grids");
    auto acc = nanovdb::getAccessor<GridT, float>(*grid);
    const unsigned int lane = threadIdx.x & 31u;

    // Keep the fixed set of launched threads busy until all pixels have been assigned.
    while (true) {
        int base = 0;
        // Each warp asks the shared counter for the next batch of 32 pixels.
        // Only lane 0 updates the counter; __shfl_sync copies lane 0's result
        // to the other lanes in the warp.
        if (lane == 0) base = atomicAdd(nextPixel, 32);
        base = __shfl_sync(0xFFFFFFFFu, base, 0);

        // Each lane renders one pixel from the batch: lane 0 renders base,
        // lane 1 renders base + 1, and so on.
        const int i = base + int(lane);
        if (i >= numPixels) break;

        renderOp.renderPixel(i, image, grid, acc);
    }
}
