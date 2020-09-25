// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file FrameBuffer.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Class definition for a platform-agnostic Framebuffer.
*/

#pragma once

#include <cstdint>

class FrameBufferBase
{
protected:
    static const int BUFFER_COUNT = 1;

public:
    enum class AccessType : int { READ_ONLY = 0,
                                  READ_WRITE = 1,
                                  WRITE_ONLY = 2 };

    enum class InternalFormat : int { UNKNOWN = -1,
                                      RGBA32F = 0,
                                      RGB32F,
                                      RGBA8UI,
                                      DEPTH_COMPONENT32F,
                                      DEPTH_COMPONENT32,
                                      R32F,
                                      COUNT };

    FrameBufferBase()
        : mIndex(0)
        , mWidth(0)
        , mHeight(0)
        , mSize(0)
        , mElementSize(0)
    {
    }

    virtual ~FrameBufferBase() {}

    //! return true if the resources have been allocated.
    bool valid() const { return mWidth > 0; }

    //! return the byteSize of the buffer.
    int size() const { return mSize; }

    //! return the width of the image.
    int width() const { return mWidth; }

    //! return the height of the image.
    int height() const { return mHeight; }

    //! return the internal texture format.
    InternalFormat internalFormat() const { return mInternalFormat; }

    static int getElementSizeForFormat(InternalFormat format)
    {
        switch (format) {
        case InternalFormat::RGBA32F:
            return sizeof(float) * 4;
        case InternalFormat::RGB32F:
            return sizeof(float) * 3;
        case InternalFormat::RGBA8UI:
            return sizeof(uint8_t) * 4;
        case InternalFormat::DEPTH_COMPONENT32F:
            return sizeof(float);
        case InternalFormat::DEPTH_COMPONENT32:
            return sizeof(uint32_t);
        case InternalFormat::R32F:
            return sizeof(float);
        default:
            return 0;
        }
    }

    void  invalidate() const { ++mBufferUpdateId; }
    bool  save(const char* filename);
    bool  load(const char* filename);
    float computePSNR(FrameBufferBase& other);

    virtual void* map(AccessType access) = 0;
    virtual void  unmap() = 0;
    virtual void* cudaMap(AccessType access, void* streamCUDA = nullptr) = 0;
    virtual void  cudaUnmap(void* streamCUDA = nullptr) = 0;
    virtual void* clMap(AccessType access, void* commandQueueCL) = 0;
    virtual void  clUnmap(void* commandQueueCL) = 0;
    virtual bool  setup(int w, int h, InternalFormat format) = 0;
    virtual bool  cleanup() = 0;
    virtual bool  render(int /*x*/, int /*y*/, int /*w*/, int /*h*/) { return false; }

    int              mIndex;
    int              mWidth;
    int              mHeight;
    int              mSize;
    int              mElementSize;
    InternalFormat   mInternalFormat;
    mutable uint32_t mBufferUpdateId;
};
