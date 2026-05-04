// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <nanovdb/NanoVDB.h>
#include "ComputePrimitives.h"

inline __hostdev__ uint32_t CompactBy1(uint32_t x)
{
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

inline __hostdev__ uint32_t SeparateBy1(uint32_t x)
{
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

inline __hostdev__ void mortonDecode(uint32_t code, uint32_t& x, uint32_t& y)
{
    x = CompactBy1(code);
    y = CompactBy1(code >> 1);
}

inline __hostdev__ void mortonEncode(uint32_t& code, uint32_t x, uint32_t y)
{
    code = SeparateBy1(x) | (SeparateBy1(y) << 1);
}

template<typename RenderFn, typename GridT>
inline float renderImage(bool useCuda, const RenderFn renderOp, int width, int height, float* image, const GridT* grid)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    computeForEach(
        useCuda, width * height, 512, __FILE__, __LINE__, [renderOp, image, grid] __hostdev__(int start, int end) {
            renderOp(start, end, image, grid);
        });
    computeSync(useCuda, __FILE__, __LINE__);

    auto t1 = ClockT::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    return duration;
}

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
        using BuildT = typename GridT::BuildType;
        static_assert(nanovdb::util::is_same<BuildT, float, nanovdb::ValueOnIndex>::value, "only works for float and OnIndex grids");
        using AccT = nanovdb::AccType<BuildT, float>;
        
        AccT acc(*grid);
        float  t0, v;
        nanovdb::Coord ijk;
        for (int i = start; i < end; ++i) {
            RayT iRay = this->getIndexRay(i, grid);   
            if (nanovdb::math::isoCrossing(iRay, acc, ijk, v, t0, mIso)) {// intersect...
                this->composite(image, i, (t0 * mDx) / (mWBBoxDimZ * 2), 1.0f);
            } else {
                this->composite(image, i, 0.0f, 0.0f);// write background value.
            }
        }
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
