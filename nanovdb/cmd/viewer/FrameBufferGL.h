// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file FrameBufferGL.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Declaration of OpenGL framebuffer class.
*/

#pragma once

#include <string>
#include <vector>
#include "FrameBuffer.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define GLFW_INCLUDE_ES3
#include <GLFW/glfw3.h>
#elif defined(NANOVDB_USE_GLAD)
#include <glad/glad.h>
#endif

bool checkGL(const char* file, const int line);

#define NANOVDB_GL_CHECKERRORS() checkGL(__FILE__, __LINE__)

#define NANOVDB_GL_SAFE_CALL(x) \
    do { \
        x; \
    } while (0)

class FrameBufferGL : public FrameBufferBase
{
public:
    FrameBufferGL(void* context, void* display);
    ~FrameBufferGL();

    static InternalFormat formatFromGL(uint32_t formatGL);
    static uint32_t       formatToGL(InternalFormat format);

    void* context() const;

    void* display() const;

    //! return the buffer resource ID.
    uint32_t bufferGL(int i = 0) const { return mPixelPackBuffers[(mIndex + i) % BUFFER_COUNT]; }

    //! return the texture resource ID.
    uint32_t textureGL(int = 0) const { return mTexture; }

    //! this can be called multiple times and will be a NOP if there
    //! are no changes.
    bool setupGL(int w, int h, void* buffer, uint32_t texFormat, uint32_t bufferType = 0x88E0, void* contextCL = nullptr);

    bool setup(int w, int h, InternalFormat format) override;
    bool cleanup() override;

    void* cudaMap(AccessType access, void* streamCUDA = nullptr) override;
    void  cudaUnmap(void* streamCUDA = nullptr) override;

    void* clMap(AccessType access, void* commandQueueCL) override;
    void  clUnmap(void* commandQueueCL) override;

    void* map(AccessType access) override;
    void  unmap() override;

    void swapBuffers();
    void reset();

    bool render(int x, int y, int w, int h) override;

    void clearTexture();

    int resourceId() const { return mBufferResourceId; }

    void begin();
    void end();

    void updateTextureGL();

private:
    bool genTextureGL(int w, int h, uint32_t internalFormatGL);
    bool genPixelPackBufferGL(int w, int h, uint32_t internalFormatGL, uint32_t bufferType, void* contextCL);

    int        mBufferResourceId;
    uint32_t   mBufferTypeGL;
    uint32_t   mTexture;
    uint32_t   mFbo;
    uint32_t   mPixelPackBuffers[BUFFER_COUNT];
    void*      mBufferResourcesCUDA[BUFFER_COUNT];
    void*      mBufferResourcesCL[BUFFER_COUNT];
    void*      mContext;
    void*      mDisplay;
    uint32_t   mTextureBufferId;
    uint8_t*   mTempBuffer;
    AccessType mTempBufferAccess;
};
