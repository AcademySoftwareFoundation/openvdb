// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file FrameBufferGL.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of OpenGL image renderer help class.
*/

#include "FrameBufferGL.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <exception>

#if defined(NANOVDB_USE_CUDA) && defined(NANOVDB_USE_CUDA_GL)
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#endif

#if defined(NANOVDB_USE_OPENCL) && 0
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif
#endif

#define _GL_ERROR_STR(code) \
    case code: return #code;

char const* getErrorStringGL(GLenum code)
{
    switch (code) {
        _GL_ERROR_STR(GL_INVALID_VALUE);
        _GL_ERROR_STR(GL_INVALID_ENUM);
        _GL_ERROR_STR(GL_INVALID_OPERATION);
        _GL_ERROR_STR(GL_INVALID_INDEX);
        _GL_ERROR_STR(GL_OUT_OF_MEMORY);
        _GL_ERROR_STR(GL_INVALID_FRAMEBUFFER_OPERATION);
#if !defined(__EMSCRIPTEN__)
        _GL_ERROR_STR(GL_STACK_OVERFLOW);
        _GL_ERROR_STR(GL_STACK_UNDERFLOW);
#endif
    }
    return "Unknown error";
}
#undef _GL_ERROR_STR

bool checkGL(const char* file, const int line)
{
    GLenum            err;
    bool              foundError = false;
    std::stringstream errStringStream;
    while ((err = glGetError()) != GL_NO_ERROR) {
        foundError = true;
        errStringStream << err << " (" << getErrorStringGL(err) << ")";
        std::cerr << errStringStream.str().c_str() << " in " << file << ":" << line << std::endl;
    }
    return !foundError;
}

#if defined(NANOVDB_USE_CUDA) && defined(NANOVDB_USE_CUDA_GL)
#define NANOVDB_CUDA_SAFE_CALL(x) checkCUDA(x, __FILE__, __LINE__)

static bool checkCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        return false;
    }
    return true;
}
#endif

FrameBufferGL::FrameBufferGL(void* context, void* display)
    : mBufferResourceId(0)
    , mBufferTypeGL(GL_STREAM_DRAW)
    , mTexture(0)
    , mFbo(0)
    , mContext(context)
    , mDisplay(display)
    , mTextureBufferId(0)
    , mTempBuffer(nullptr)
{
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        mPixelPackBuffers[i] = 0;
        mBufferResourcesCUDA[i] = nullptr;
        mBufferResourcesCL[i] = nullptr;
    }
}

FrameBufferGL::~FrameBufferGL()
{
    reset();
}

void FrameBufferGL::swapBuffers()
{
    mIndex = (mIndex + 1) % BUFFER_COUNT;
}

void* FrameBufferGL::context() const
{
    return mContext;
}

void* FrameBufferGL::display() const
{
    return mDisplay;
}

void* FrameBufferGL::cudaMap(AccessType /*access*/, void* /*stream*/)
{
#if defined(NANOVDB_USE_CUDA) && defined(NANOVDB_USE_CUDA_GL)

    int writeIndex = (mIndex + 1) % BUFFER_COUNT;

    if (!mSize)
        return nullptr;

    NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPixelPackBuffers[writeIndex]));

    if (BUFFER_COUNT > 1 && access == AccessType::WRITE_ONLY) {
        // discard buffer
        NANOVDB_GL_SAFE_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, mElementSize * mWidth * mHeight, 0, mBufferTypeGL));
    }

    if (access == AccessType::READ_ONLY)
        NANOVDB_CUDA_SAFE_CALL(cudaGraphicsResourceSetMapFlags(
            (cudaGraphicsResource*)mBufferResourcesCUDA[writeIndex], cudaGraphicsMapFlagsReadOnly));
    else if (access == AccessType::WRITE_ONLY)
        NANOVDB_CUDA_SAFE_CALL(cudaGraphicsResourceSetMapFlags(
            (cudaGraphicsResource*)mBufferResourcesCUDA[writeIndex], cudaGraphicsMapFlagsWriteDiscard));
    else
        NANOVDB_CUDA_SAFE_CALL(cudaGraphicsResourceSetMapFlags(
            (cudaGraphicsResource*)mBufferResourcesCUDA[writeIndex], cudaGraphicsMapFlagsNone));

    void*  ptr = nullptr;
    size_t size = 0;
    NANOVDB_CUDA_SAFE_CALL(cudaGraphicsMapResources(
        1, (cudaGraphicsResource**)&mBufferResourcesCUDA[writeIndex], (cudaStream_t)stream));
    NANOVDB_CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
        &ptr, &size, (cudaGraphicsResource*)mBufferResourcesCUDA[writeIndex]));
    assert(size == mSize);

    NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

    return ptr;
#else
    return nullptr;
#endif
}

void FrameBufferGL::cudaUnmap(void* /*stream*/)
{
#if defined(NANOVDB_USE_CUDA) && defined(NANOVDB_USE_CUDA_GL)
    int writeIndex = (mIndex + 1) % BUFFER_COUNT;
    NANOVDB_CUDA_SAFE_CALL(cudaGraphicsUnmapResources(
        1, (cudaGraphicsResource**)&mBufferResourcesCUDA[writeIndex], (cudaStream_t)stream));
    invalidate();
#endif
}

void* FrameBufferGL::clMap(AccessType /*access*/, void* /*stream*/)
{
#if defined(NANOVDB_USE_OPENCL) && 0

    int writeIndex = (mIndex + 1) % BUFFER_COUNT;

    if (!mSize)
        return nullptr;

    NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPixelPackBuffers[writeIndex]));

    if (BUFFER_COUNT > 1 && access == AccessType::WRITE_ONLY) {
        // discard buffer
        NANOVDB_GL_SAFE_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, mElementSize * mWidth * mHeight, 0, mBufferTypeGL));
    }

    glFinish();

    cl_mem memCL = cl_mem(mBufferResourcesCL[writeIndex]);
    cl_int err = clEnqueueAcquireGLObjects(cl_command_queue(stream), 1, &memCL, 0, 0, NULL);

    NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

    if (err != CL_SUCCESS)
        return nullptr;

    return (void*)memCL;
#else
    return nullptr;
#endif
}

void FrameBufferGL::clUnmap(void* /*stream*/)
{
#if defined(NANOVDB_USE_OPENCL) && 0
    int    writeIndex = (mIndex + 1) % BUFFER_COUNT;
    cl_mem bufferCL = cl_mem(mBufferResourcesCL[writeIndex]);
    cl_int err = clEnqueueReleaseGLObjects(cl_command_queue(stream), 1, &bufferCL, 0, 0, NULL);
    invalidate();
#endif
}

void* FrameBufferGL::map(AccessType access)
{
    int writeIndex = (mIndex + 1) % BUFFER_COUNT;

    if (!mSize)
        return nullptr;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPixelPackBuffers[writeIndex]);

#if 0
    if (access == AccessType::WRITE_ONLY) {
        // discard buffer
        glBufferData(GL_PIXEL_UNPACK_BUFFER, mSize, 0, mBufferTypeGL);
        NANOVDB_GL_CHECKERRORS();
    }
#endif

    void* ptr = nullptr;
#if defined(__EMSCRIPTEN__)
    mTempBufferAccess = access;
    if (mTempBufferAccess == AccessType::READ_WRITE || mTempBufferAccess == AccessType::READ_ONLY) {
        //printf("Downloading %d bytes to host-buffer(%p)\n", mSize, mTempBuffer);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, mPixelPackBuffers[writeIndex]);
        //glGetBufferSubData(GL_PIXEL_PACK_BUFFER, 0, mSize, mTempBuffer);
        EM_ASM_(
            {
                Module.ctx.getBufferSubData($0, $1, HEAPU8.subarray($2, $2 + $3));
            },
            GL_PIXEL_PACK_BUFFER,
            0,
            mTempBuffer,
            mSize);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        NANOVDB_GL_CHECKERRORS();
    }
    ptr = mTempBuffer;
#else
    GLbitfield accessGL = 0;
    if (access == AccessType::READ_WRITE)
        accessGL |= GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
    else if (access == AccessType::WRITE_ONLY)
        accessGL |= GL_MAP_WRITE_BIT;
    else if (access == AccessType::READ_ONLY)
        accessGL |= GL_MAP_READ_BIT;

    ptr = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, mSize, accessGL);
#endif
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    NANOVDB_GL_CHECKERRORS();

    return ptr;
}

void FrameBufferGL::unmap()
{
    int writeIndex = (mIndex + 1) % BUFFER_COUNT;
    NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPixelPackBuffers[writeIndex]));
#if defined(__EMSCRIPTEN__)
    if (mTempBufferAccess == AccessType::READ_WRITE || mTempBufferAccess == AccessType::WRITE_ONLY) {
        auto f = (float*)mTempBuffer;
        //printf("Uploading %d bytes from host-buffer(%p) (%f,%f,%f,...)\n", mSize, mTempBuffer, f[0], f[1], f[2]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, mSize, mTempBuffer, mBufferTypeGL);
    }
#else
    NANOVDB_GL_SAFE_CALL(glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER));
#endif
    NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    invalidate();
}

void FrameBufferGL::updateTextureGL()
{
    if (mBufferUpdateId == mTextureBufferId)
        return;
    mTextureBufferId = mBufferUpdateId;

    if (mWidth == 0 || mHeight == 0 || !glIsTexture(mTexture) || !glIsBuffer(mPixelPackBuffers[0])) {
        return;
    }

    //printf("Updating GL texture(%d) format(%d)\n", mTexture, (int)mInternalFormat);

    swapBuffers();

    // bind the texture and PBO
    glBindTexture(GL_TEXTURE_2D, mTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPixelPackBuffers[mIndex]);

    // copy pixels from PBO to texture object
    if (mInternalFormat == InternalFormat::RGBA32F)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_FLOAT, 0);
    else if (mInternalFormat == InternalFormat::RGB32F)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGB, GL_FLOAT, 0);
    else if (mInternalFormat == InternalFormat::RGBA8UI)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    else if (mInternalFormat == InternalFormat::R32F)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_FLOAT, 0);
    else if (mInternalFormat == InternalFormat::DEPTH_COMPONENT32)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);
    else if (mInternalFormat == InternalFormat::DEPTH_COMPONENT32F)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    else {
        std::stringstream msg;
        msg << "Unsupported internalFormat: " << (int)mInternalFormat;
        throw std::runtime_error(msg.str().c_str());
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    NANOVDB_GL_CHECKERRORS();
}

void FrameBufferGL::begin()
{
    glBindFramebuffer(GL_FRAMEBUFFER, mFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureGL(), 0);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

void FrameBufferGL::end()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool FrameBufferGL::render(int x, int y, int w, int h)
{
    if (w == 0 || h == 0)
        return false;

    updateTextureGL();

    float color[4] = {0, 0, 1, 1};
    glClearBufferfv(GL_COLOR, 0, color);

    // blit the texture to the destination GL framebuffer...
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureGL(), 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    assert(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

#if defined(__APPLE__)
    // TODO: we multiply by 2 for retina display! Find a better way.
    glBlitFramebuffer(0, 0, mWidth, mHeight, x, y, w*2, h*2, GL_COLOR_BUFFER_BIT, GL_NEAREST);
#else
    glBlitFramebuffer(0, 0, mWidth, mHeight, x, y, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
#endif

    NANOVDB_GL_CHECKERRORS();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    return true;
}

bool FrameBufferGL::genTextureGL(int w, int h, GLenum internalFormat)
{
    if (mTexture && mWidth == w && mHeight == h && mInternalFormat == formatFromGL(internalFormat))
        return true;

    NANOVDB_GL_CHECKERRORS();

    if (glIsTexture(mTexture)) {
        glDeleteTextures(1, &mTexture);
    }

    glGenTextures(1, &mTexture);
    NANOVDB_GL_CHECKERRORS();

    glBindTexture(GL_TEXTURE_2D, mTexture);
    if (!mTexture)
        throw "Error: Unable to create texture";

    NANOVDB_GL_CHECKERRORS();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

#ifdef __APPLE__
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
#else
    glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, w, h);
#endif

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (!NANOVDB_GL_CHECKERRORS())
        throw "Error: Unable to create texture storage";

    mWidth = w;
    mHeight = h;
    mInternalFormat = formatFromGL(internalFormat);

    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}

FrameBufferBase::InternalFormat FrameBufferGL::formatFromGL(uint32_t formatGL)
{
    if (formatGL == GL_RGBA32F)
        return InternalFormat::RGBA32F;
    else if (formatGL == GL_RGB32F)
        return InternalFormat::RGB32F;
    else if (formatGL == GL_RGBA8UI || formatGL == GL_RGBA8)
        return InternalFormat::RGBA8UI;
    else if (formatGL == GL_DEPTH_COMPONENT32F)
        return InternalFormat::DEPTH_COMPONENT32F;
    else if (formatGL == GL_R32F)
        return InternalFormat::R32F;
    else {
        return InternalFormat::UNKNOWN;
    }
}

GLenum FrameBufferGL::formatToGL(InternalFormat format)
{
    if (format == InternalFormat::RGBA32F)
        return GL_RGBA32F;
    else if (format == InternalFormat::RGB32F)
        return GL_RGB32F;
    else if (format == InternalFormat::RGBA8UI)
        return GL_RGBA8UI;
    else if (format == InternalFormat::DEPTH_COMPONENT32F)
        return GL_DEPTH_COMPONENT32F;
    else if (format == InternalFormat::R32F)
        return GL_R32F;
    else {
        return GL_NONE;
    }
}

bool FrameBufferGL::genPixelPackBufferGL(int w, int h, GLenum internalFormatGL, GLenum bufferType, void* /*contextCL*/)
{
    mInternalFormat = formatFromGL(internalFormatGL);
    mBufferTypeGL = bufferType;

    mElementSize = getElementSizeForFormat(mInternalFormat);

    if (mElementSize == 0) {
        std::stringstream msg;
        msg << "Unsupported GL internalFormat: " << internalFormatGL;
        throw std::runtime_error(msg.str().c_str());
    }

    auto s = mElementSize * w * h;

    if (s == 0) {
        reset();
        return true;
    }

    if (mPixelPackBuffers[0] && mSize == s)
        return true;

    if (glIsBuffer(mPixelPackBuffers[0]) == GL_TRUE)
        glDeleteBuffers(BUFFER_COUNT, &mPixelPackBuffers[0]);

    glGenBuffers(BUFFER_COUNT, &mPixelPackBuffers[0]);

    for (int i = 0; i < BUFFER_COUNT; ++i) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPixelPackBuffers[i]);
        if (glIsBuffer(mPixelPackBuffers[i]) == GL_FALSE)
            return false;

        glBufferData(GL_PIXEL_UNPACK_BUFFER, s, nullptr, bufferType);
        if (!NANOVDB_GL_CHECKERRORS())
            return false;

        mBufferResourcesCUDA[i] = nullptr;
        mBufferResourcesCL[i] = nullptr;

#if defined(NANOVDB_USE_CUDA) && defined(NANOVDB_USE_CUDA_GL)
        bool rc =
            NANOVDB_CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&mBufferResourcesCUDA[i],
                                                                mPixelPackBuffers[i],
                                                                cudaGraphicsMapFlagsWriteDiscard));
        if (!rc) {
            std::cerr << "Failed to register GL PBO with CUDA" << std::endl;
            return false;
        }
#endif

#if defined(NANOVDB_USE_OPENCL) && 0
        if (contextCL) {
            bool rc =
                clCreateFromGLBuffer(cl_context(contextCL), CL_MEM_READ_WRITE, mPixelPackBuffers[i], NULL);
            if (!rc) {
                std::cerr << "Failed to register GL PBO with OpenCL" << std::endl;
                return false;
            }
        }
#endif
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    NANOVDB_GL_CHECKERRORS();

    mBufferResourceId++;
    mSize = static_cast<GLsizei>(s);
    return true;
}

bool FrameBufferGL::cleanup()
{
    reset();
    return true;
}

bool FrameBufferGL::setup(int w, int h, InternalFormat format)
{
    return setupGL(w, h, nullptr, formatToGL(format), mBufferTypeGL, mContext);
}

bool FrameBufferGL::setupGL(int w, int h, void* /*buffer*/, GLenum texFormat, GLenum bufferType, void* contextCL)
{
    if (w == 0 || h == 0) {
        reset();
        return true;
    }

    bool rc = true;
    rc &= genTextureGL(w, h, texFormat);
    if (!rc) {
        reset();
        return false;
    }

    rc &= genPixelPackBufferGL(w, h, texFormat, bufferType, contextCL);
    if (!rc) {
        reset();
        return false;
    }

    if (!mFbo) {
        glGenFramebuffers(1, &mFbo);
    }
    if (!mFbo) {
        reset();
        return false;
    }

#if defined(__EMSCRIPTEN__)
    mTempBuffer = new uint8_t[mSize];
#endif

    return true;
}

void FrameBufferGL::clearTexture()
{
    if (mSize > 0) {
        auto ptr = map(FrameBufferGL::AccessType::WRITE_ONLY);
        std::memset(ptr, 0, size());
        unmap();
        updateTextureGL();
    }
}

void FrameBufferGL::reset()
{
#if defined(__EMSCRIPTEN__)
    delete[] mTempBuffer;
    mTempBuffer = nullptr;
#endif

    if (glIsFramebuffer(mFbo))
        glDeleteFramebuffers(1, &mFbo);

    if (glIsBuffer(mPixelPackBuffers[0]) == GL_TRUE)
        glDeleteBuffers(BUFFER_COUNT, &mPixelPackBuffers[0]);
    if (glIsTexture(mTexture) == GL_TRUE)
        glDeleteTextures(1, &mTexture);

    NANOVDB_GL_CHECKERRORS();

    for (int i = 0; i < BUFFER_COUNT; ++i)
        mPixelPackBuffers[i] = 0;
    mTexture = 0;
    mFbo = 0;
    mTextureBufferId = 0;

    mWidth = 0;
    mHeight = 0;
    mElementSize = 0;
    mSize = 0;
    mIndex = 0;
    mInternalFormat = InternalFormat::UNKNOWN;
    mBufferUpdateId = 0;
}

