// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file FrameBuffer.cpp
	\brief Implementation of FrameBufferBase.
*/

#include "FrameBuffer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

// Save a PFM file.
// http://netpbm.sourceforge.net/doc/pfm.html
static bool savePFM(const float* buffer, int numComponents, int width, int height, const char* filePath)
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

    if (numComponents == 0) {
        std::cerr << "Unable to save PFM file due to invalid number of components: " << (int)numComponents << std::endl;
        return false;
    }

    std::string method = "Pf";
    if (numComponents > 1)
        method = "PF";

    float scale = 1.0f;
    if (isLittleEndian())
        scale = -scale;
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
                float v2 = (numComponents <= 2) ? 0.f : buffer[(x + y * width) * numComponents + 2];
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
static bool loadPFM(const char* filePath, int numComponents, int& outWidth, int& outHeight, float* buffer)
{
    std::fstream file(filePath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filePath << std::endl;
        return false;
    }
    /*
    const auto isLittleEndian = []() -> bool {
        static int  x = 1;
        static bool result = reinterpret_cast<uint8_t*>(&x)[0] == 1;
        return result;
    };
    */

    if (numComponents == 0) {
        std::cerr << "Unable to load PFM file due to invalid number of components: " << (int)numComponents << std::endl;
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
                    buffer[(x + y * srcWidth) * numComponents + 0] = v0;
                    buffer[(x + y * srcWidth) * numComponents + 1] = v1;
                    if (numComponents > 2)
                        buffer[(x + y * srcWidth) * numComponents + 2] = v2;
                }
            }
        }
    }

    outWidth = srcWidth;
    outHeight = srcHeight;

    return true;
}

bool FrameBufferBase::save(const char* filename, const char* fileformat, int quality)
{
    auto buffer = map(AccessType::READ_ONLY);
    if (!buffer)
        return false;
    //std::cout << "Saving framebuffer(" << mWidth << "x" << mHeight << ") to file: " << filename << " ..." << std::endl;

    std::string fn(filename);
    std::string ext(fileformat);

    bool targetIsFloat = false;
    if (ext == "pfm" || ext == "hdr") {
        targetIsFloat = true;
    }

    int                  numComponents = formatGetNumComponents(mInternalFormat);
    int                  typeSize = formatGetElementSize(mInternalFormat) / numComponents;
    std::vector<uint8_t> tmpBuffer;

    void* srcBuffer = buffer;

    auto byteToFloatFn = [](uint8_t in) -> float { return in / 255.f; };
    auto floatToByteFn = [](float in) -> uint8_t { return std::min(255, int(in * 255)); };

    if (targetIsFloat == false && formatIsFloat(mInternalFormat)) {
        tmpBuffer.resize(sizeof(uint8_t) * mWidth * mHeight * numComponents);
        for (int y = 0; y < mHeight; ++y) {
            auto* src = ((const float*)buffer) + ((mHeight - 1) - y) * mWidth * numComponents;
            auto* dst = ((uint8_t*)tmpBuffer.data()) + y * mWidth * numComponents;
            for (int i = 0; i < mWidth * numComponents; ++i) {
                dst[i] = floatToByteFn(src[i]);
            }
        }
        srcBuffer = tmpBuffer.data();
    } else if (targetIsFloat == true && formatIsFloat(mInternalFormat) == false) {
        tmpBuffer.resize(sizeof(float) * mWidth * mHeight * numComponents);
        for (int y = 0; y < mHeight; ++y) {
            auto* src = ((const uint8_t*)buffer) + ((mHeight - 1) - y) * mWidth * numComponents;
            auto* dst = ((float*)tmpBuffer.data()) + y * mWidth * numComponents;
            for (int i = 0; i < mWidth * numComponents; ++i) {
                dst[i] = byteToFloatFn(src[i]);
            }
        }
        srcBuffer = tmpBuffer.data();
    } else {
        // no type conversion.
        tmpBuffer.resize(typeSize * mWidth * mHeight * numComponents);
        for (int y = 0; y < mHeight; ++y) {
            auto* src = ((const uint8_t*)buffer) + ((mHeight - 1) - y) * mWidth * numComponents * typeSize;
            auto* dst = ((uint8_t*)tmpBuffer.data()) + y * mWidth * numComponents * typeSize;
            std::memcpy(dst, src, typeSize * numComponents * mWidth);
        }
        srcBuffer = tmpBuffer.data();
    }

    if (srcBuffer != buffer)
        unmap();

    bool hasError = false;

    if (ext == "pfm") {
        savePFM((const float*)srcBuffer, numComponents, mWidth, mHeight, filename);
    } else if (ext == "hdr") {
        stbi_write_hdr(fn.c_str(), mWidth, mHeight, numComponents, (const float*)srcBuffer);
    } else if (ext == "png") {
        stbi_write_png(fn.c_str(), mWidth, mHeight, numComponents, srcBuffer, mWidth * numComponents);
    } else if (ext == "bmp") {
        stbi_write_bmp(fn.c_str(), mWidth, mHeight, numComponents, srcBuffer);
    } else if (ext == "tga") {
        stbi_write_tga(fn.c_str(), mWidth, mHeight, numComponents, srcBuffer);
    } else if (ext == "jpg") {
        stbi_write_jpg(fn.c_str(), mWidth, mHeight, numComponents, srcBuffer, quality);
    } else {
        hasError = true;
    }

    if (srcBuffer == buffer)
        unmap();
    return hasError == false;
}

bool FrameBufferBase::load(const char* filename, const char* fileformat)
{
    std::string ext(fileformat);

    // TODO: only pfm support at the momemt. support other formats.
    
    if(ext == "pfm") {
        int numComponents = formatGetNumComponents(InternalFormat::RGBA32F);
        int w, h;
        if (loadPFM(filename, numComponents, w, h, nullptr)) {
            bool rc = setup(w, h, InternalFormat::RGBA32F);
            if (rc) {
                auto buffer = map(AccessType::WRITE_ONLY);
                if (!buffer)
                    return false;

                loadPFM(filename, numComponents, w, h, (float*)buffer);
                unmap();
                return true;
            }
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