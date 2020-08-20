// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file FrameBuffer.cpp

	\author Wil Braithwaite

	\date May 26, 2020

	\brief Implementation of FrameBufferBase.
*/

#include "FrameBuffer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>

// Save a PFM file.
// http://netpbm.sourceforge.net/doc/pfm.html
static bool savePFM(const float* buffer, int width, int height, FrameBufferBase::InternalFormat format, const char* filePath)
{
    std::fstream file(filePath, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filePath << std::endl;
        return false;
    }

    const auto isLittleEndian = []() -> bool {
        static int  x = 1;
        static bool result = reinterpret_cast<uint8_t*>(&x)[0] == 1;
        return result;
    };

    int numComponents = 0;

    switch (format) {
    case FrameBufferBase::InternalFormat::DEPTH_COMPONENT32F:
    case FrameBufferBase::InternalFormat::DEPTH_COMPONENT32:
    case FrameBufferBase::InternalFormat::R32F:
        numComponents = 1;
        break;
    case FrameBufferBase::InternalFormat::RGBA32F:
        numComponents = 4;
        break;
    case FrameBufferBase::InternalFormat::RGB32F:
        numComponents = 3;
    case FrameBufferBase::InternalFormat::RGBA8UI:
        numComponents = 4;
        break;
    default:
        numComponents = 0;
    }

    if (numComponents == 0) {
        std::cerr << "Unable to save PFM file due to unrecognized type: " << (int)format << std::endl;
        return false;
    }

    std::string method = "Pf";
    if (numComponents > 1)
        method = "PF";

    float scale = 1.0f;
    if (isLittleEndian())
        scale = -scale;

    auto applyColorProfile = [](float x) -> float {
        return (x <= 0.04045f)
                   ? (x / 12.92f)
                   : (powf((x + 0.055f) / 1.055f, 2.4f));
    };

    file << method << "\n"
         << width << "\n"
         << height << "\n"
         << scale << "\n";

    if (method == "Pf") {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float v0 = buffer[(x + y * width) * numComponents + 0];
                file.write((char*)&v0, sizeof(float));
            }
        }
    } else if (method == "PF") {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float v0 = buffer[(x + y * width) * numComponents + 0];
                float v1 = buffer[(x + y * width) * numComponents + 1];
                float v2 = buffer[(x + y * width) * numComponents + 2];
#if 0
                v0 = applyColorProfile(v0);
                v1 = applyColorProfile(v1);
                v2 = applyColorProfile(v2);
#endif
                file.write((char*)&v0, sizeof(float));
                file.write((char*)&v1, sizeof(float));
                file.write((char*)&v2, sizeof(float));
            }
        }
    }

    return true;
}

// Load a PFM file.
// http://netpbm.sourceforge.net/doc/pfm.html
static bool loadPFM(const char* filePath, FrameBufferBase::InternalFormat format, int& outWidth, int& outHeight, float* buffer)
{
    std::fstream file(filePath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filePath << std::endl;
        return false;
    }

    const auto isLittleEndian = []() -> bool {
        static int  x = 1;
        static bool result = reinterpret_cast<uint8_t*>(&x)[0] == 1;
        return result;
    };

    auto applyColorProfile = [](float x) -> float {
        return (x <= 0.0031308f)
                   ? (12.92f * x)
                   : (1.055f * powf(x, 1.0f / 2.4f) - 0.055f);
    };

    int numComponents = 0;

    switch (format) {
    case FrameBufferBase::InternalFormat::DEPTH_COMPONENT32F:
    case FrameBufferBase::InternalFormat::DEPTH_COMPONENT32:
    case FrameBufferBase::InternalFormat::R32F:
        numComponents = 1;
        break;
    case FrameBufferBase::InternalFormat::RGBA32F:
        numComponents = 4;
        break;
    case FrameBufferBase::InternalFormat::RGB32F:
        numComponents = 3;
    case FrameBufferBase::InternalFormat::RGBA8UI:
        numComponents = 4;
        break;
    default:
        numComponents = 0;
    }

    if (numComponents == 0) {
        std::cerr << "Unable to load PFM file due to unrecognized type: " << (int)format << std::endl;
        return false;
    }

    std::string srcMethod = "";
    float       srcScale = 0.0f;
    int         srcWidth;
    int         srcHeight;
    int         srcNumComponents = 1;

    file >> srcMethod >> srcWidth >> srcHeight >> srcScale;

    if (srcMethod == "PF") {
        srcNumComponents = 3;
    }

    file.seekg(0, std::ios_base::end);
    auto lSize = file.tellg();
    auto pos = lSize - std::streampos(srcWidth * srcHeight * srcNumComponents * sizeof(float));
    file.seekg(pos, std::ios_base::beg);

    if (buffer != nullptr) {
        if (srcMethod == "Pf") {
            for (int y = 0; y < srcHeight; ++y) {
                for (int x = 0; x < srcWidth; ++x) {
                    float* v0 = &buffer[(x + y * srcWidth) * numComponents + 0];
                    file.read((char*)v0, sizeof(float));
                }
            }
        } else if (srcMethod == "PF") {
            for (int y = 0; y < srcHeight; ++y) {
                for (int x = 0; x < srcWidth; ++x) {
                    float v0, v1, v2;
                    file.read((char*)&v0, sizeof(float));
                    file.read((char*)&v1, sizeof(float));
                    file.read((char*)&v2, sizeof(float));
#if 0
                    v0 = applyColorProfile(v0);
                    v1 = applyColorProfile(v1);
                    v2 = applyColorProfile(v2);
#endif
                    buffer[(x + y * srcWidth) * numComponents + 0] = v0;
                    buffer[(x + y * srcWidth) * numComponents + 1] = v1;
                    buffer[(x + y * srcWidth) * numComponents + 2] = v2;
                }
            }
        }
    }

    outWidth = srcWidth;
    outHeight = srcHeight;

    return true;
}

bool FrameBufferBase::save(const char* filename)
{
    auto buffer = map(AccessType::READ_ONLY);
    if (!buffer)
        return false;
    std::cout << "Saving framebuffer(" << mWidth << "x" << mHeight << ") to file: " << filename << " ..." << std::endl;
    savePFM((const float*)buffer, mWidth, mHeight, mInternalFormat, filename);
    unmap();
    return true;
}

bool FrameBufferBase::load(const char* filename)
{
    std::cout << "Loading framebuffer from file: " << filename << " ..." << std::endl;

    int w, h;
    if (loadPFM(filename, InternalFormat::RGBA32F, w, h, nullptr)) {
        bool rc = setup(w, h, InternalFormat::RGBA32F);
        if (rc) {
            auto buffer = map(AccessType::WRITE_ONLY);
            if (!buffer)
                return false;

            loadPFM(filename, InternalFormat::RGBA32F, w, h, (float*)buffer);
            unmap();
            return true;
        }
    }
    return false;
}

float FrameBufferBase::computePSNR(FrameBufferBase& other)
{
    auto otherBuffer = other.map(AccessType::READ_ONLY);
    if (!otherBuffer)
        return -1.f;

    auto source = map(AccessType::READ_ONLY);
    if (!source) {
        other.unmap();
        return -1.f;
    }

    double mse = 0;
    if (mInternalFormat == InternalFormat::RGBA32F) {
        int numComponents = 4;
        for (int y = 0; y < mHeight; y++)
            for (int x = 0; x < mWidth; x++) {
                const float* p0 = &reinterpret_cast<const float*>(source)[(y * mWidth + x) * numComponents];
                const float* p1 = &reinterpret_cast<const float*>(otherBuffer)[(y * mWidth + x) * numComponents];

                auto d0 = std::abs(double(p0[0]) - double(p1[0]));
                auto d1 = std::abs(double(p0[1]) - double(p1[1]));
                auto d2 = std::abs(double(p0[2]) - double(p1[2]));
                mse += (d0 * d0 + d1 * d1 + d2 * d2) / 3.0;
            }

        mse /= mWidth * mHeight;
    }

    unmap();
    other.unmap();

    printf("mse: %f\n", mse);

    if (mse > 0.0001) {
        double maxValue = 1.0;
        return float(10.0 * log10((maxValue * maxValue) / mse));
    }
    return -1.f;
}