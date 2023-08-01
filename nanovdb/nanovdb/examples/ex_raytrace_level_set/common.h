// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

inline void saveImage(const std::string& filename, int width, int height, const float* image)
{
    const auto isLittleEndian = []() -> bool {
        static int  x = 1;
        static bool result = reinterpret_cast<uint8_t*>(&x)[0] == 1;
        return result;
    };

    float scale = 1.0f;
    if (isLittleEndian())
        scale = -scale;

    std::fstream fs(filename, std::ios::out | std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    fs << "Pf\n"
       << width << "\n"
       << height << "\n"
       << scale << "\n";

    for (int i = 0; i < width * height; ++i) {
        float r = image[i];
        fs.write((char*)&r, sizeof(float));
    }
}

template<typename Vec3T>
struct RayGenOp
{
    float mWBBoxDimZ;
    Vec3T mWBBoxCenter;

    inline RayGenOp(float wBBoxDimZ, Vec3T wBBoxCenter)
        : mWBBoxDimZ(wBBoxDimZ)
        , mWBBoxCenter(wBBoxCenter)
    {
    }

    inline __hostdev__ void operator()(int i, int w, int h, Vec3T& outOrigin, Vec3T& outDir) const
    {
        // perspective camera along Z-axis...
        uint32_t x, y;
#if 0
        mortonDecode(i, x, y);
#else
        x = i % w;
        y = i / w;
#endif
        const float fov = 45.f;
        const float u = (float(x) + 0.5f) / w;
        const float v = (float(y) + 0.5f) / h;
        const float aspect = w / float(h);
        const float Px = (2.f * u - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f) * aspect;
        const float Py = (2.f * v - 1.f) * tanf(fov / 2 * 3.14159265358979323846f / 180.f);
        const Vec3T origin = mWBBoxCenter + Vec3T(0, 0, mWBBoxDimZ);
        Vec3T       dir(Px, Py, -1.f);
        dir.normalize();
        outOrigin = origin;
        outDir = dir;
    }
};

struct CompositeOp
{
    inline __hostdev__ void operator()(float* outImage, int i, int w, int h, float value, float alpha) const
    {
        uint32_t x, y;
        int      offset;
#if 0
        mortonDecode(i, x, y);
        offset = x + y * w;
#else
        x = i % w;
        y = i / w;
        offset = i;
#endif

        // checkerboard background...
        const int   mask = 1 << 7;
        const float bg = ((x & mask) ^ (y & mask)) ? 1.0f : 0.5f;
        outImage[offset] = alpha * value + (1.0f - alpha) * bg;
    }
};
