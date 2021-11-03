// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file Image.h

    \author Ken Museth

    \date January 8, 2020

    \brief A simple image class that uses pinned memory for fast GPU transfer

    \warning This class is only included to support benchmark-tests.
*/

#ifndef NANOVDB_IMAGE_H_HAS_BEEN_INCLUDED
#define NANOVDB_IMAGE_H_HAS_BEEN_INCLUDED

#include <stdint.h> // for uint8_t
#include <string> //   for std::string
#include <fstream> //  for std::ofstream
#include <cassert>

#include <nanovdb/util/HostBuffer.h>

#if defined(NANOVDB_USE_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#endif

namespace nanovdb {

struct ImageData
{
    int   mWidth, mHeight, mSize;
    float mScale[2];
    ImageData(int w, int h)
        : mWidth(w)
        , mHeight(h)
        , mSize(w * h)
        , mScale{1.0f / w, 1.0f / h}
    {
    }
};

/// @note Can only be constructed by an ImageHandle
class Image : private ImageData
{
    using DataT = ImageData;

public:
    struct ColorRGB
    {
        uint8_t     r, g, b;
        __hostdev__ ColorRGB(float _r, float _g, float _b)
            : r(uint8_t(_r * 255.0f))
            , g(uint8_t(_g * 255.0f))
            , b(uint8_t(_b * 255.0f))
        {
        }
    };
    void                         clear(int log2 = 7);
    __hostdev__ int              width() const { return DataT::mWidth; }
    __hostdev__ int              height() const { return DataT::mHeight; }
    __hostdev__ int              size() const { return DataT::mSize; }
    __hostdev__ float            u(int w) const { return w * mScale[0]; }
    __hostdev__ float            v(int h) const { return h * mScale[1]; }
    __hostdev__ inline ColorRGB& operator()(int w, int h);
    void                         writePPM(const std::string& fileName, const std::string& comment = "width  height 255");
}; // Image

template<typename BufferT = HostBuffer>
class ImageHandle
{
    BufferT mBuffer;

public:
    ImageHandle(int width, int height, int log2 = 7);

    const Image* image() const { return reinterpret_cast<const Image*>(mBuffer.data()); }

    Image* image() { return reinterpret_cast<Image*>(mBuffer.data()); }

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, const Image*>::type
    deviceImage() const { return reinterpret_cast<const Image*>(mBuffer.deviceData()); }

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, Image*>::type
    deviceImage() { return reinterpret_cast<Image*>(mBuffer.deviceData()); }

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* stream = nullptr, bool sync = true) { mBuffer.deviceUpload(stream, sync); }

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true) { mBuffer.deviceDownload(stream, sync); }
};

template<typename BufferT>
ImageHandle<BufferT>::ImageHandle(int width, int height, int log2)
    : mBuffer(sizeof(ImageData) + width * height * sizeof(Image::ColorRGB))
{
    ImageData data(width, height);
    *reinterpret_cast<ImageData*>(mBuffer.data()) = data;
    this->image()->clear(log2); // clear pixels or set background
}

inline void Image::clear(int log2)
{
    ColorRGB* ptr = &(*this)(0, 0);
    if (log2 < 0) {
        for (auto* end = ptr + ImageData::mSize; ptr != end;)
            *ptr++ = ColorRGB(0, 0, 0);
    } else {
        const int checkerboard = 1 << log2;

        auto kernel2D = [&](int x0, int y0, int x1, int y1) {
            for (int h = y0; h != y1; ++h) {
                const int n = h & checkerboard;
                ColorRGB* p = ptr + h * ImageData::mWidth;
                for (int w = x0; w != x1; ++w) {
                    *(p + w) = (n ^ (w & checkerboard)) ? ColorRGB(1, 1, 1) : ColorRGB(0, 0, 0);
                }
            }
        };

#if defined(NANOVDB_USE_TBB)
        tbb::blocked_range2d<int> range(0, ImageData::mWidth, 0, ImageData::mHeight);
        tbb::parallel_for(range, [&](const tbb::blocked_range2d<int>& r) {
            kernel2D(r.rows().begin(), r.cols().begin(), r.rows().end(), r.cols().end());
        });
#else
        kernel2D(0, 0, ImageData::mWidth, ImageData::mHeight);
#endif
    }
}

inline Image::ColorRGB& Image::operator()(int w, int h)
{
    assert(w < ImageData::mWidth);
    assert(h < ImageData::mHeight);
    return *(reinterpret_cast<ColorRGB*>((uint8_t*)this + sizeof(ImageData)) + w + h * ImageData::mWidth);
}

inline void Image::writePPM(const std::string& fileName, const std::string& comment)
{
    std::ofstream os(fileName, std::ios::out | std::ios::binary);
    if (os.fail())
        throw std::runtime_error("Unable to open file named \"" + fileName + "\" for output");
    os << "P6\n#" << comment << "\n"
       << this->width() << " " << this->height() << "\n255\n";
    os.write((const char*)&(*this)(0, 0), this->size() * sizeof(ColorRGB));
}

} // namespace nanovdb

#endif // end of NANOVDB_IMAGE_H_HAS_BEEN_INCLUDED