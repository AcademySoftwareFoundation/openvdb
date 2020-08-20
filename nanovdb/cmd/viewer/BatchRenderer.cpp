// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file BatchRenderer.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of BatchRenderer.
*/

#include "BatchRenderer.h"
#include <sstream>
#include <iomanip>

#if defined(NANOVDB_USE_EGL)
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#if defined(NANOVDB_USE_OPENGL)
#include "FrameBufferGL.h"
#endif
#include "FrameBufferHost.h"

BatchRenderer::BatchRenderer(const RendererParams& params)
    : RendererBase(params)
{
    mParams.mUseAccumulation = false;
}

bool initializeEGL(const RendererParams& params, void** glContext, void** glDisplay)
{
    *glContext = nullptr;
    *glDisplay = nullptr;

#if defined(NANOVDB_USE_EGL)
    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 16, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};

    static const int pbufferWidth = params.mWidth;
    static const int pbufferHeight = params.mHeight;

    static const EGLint pbufferAttribs[] = {
        EGL_WIDTH,
        pbufferWidth,
        EGL_HEIGHT,
        pbufferHeight,
        EGL_NONE,
    };

    EGLBoolean rc;

    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;
    rc = eglInitialize(eglDpy, &major, &minor);
    if (rc == EGL_FALSE) {
        throw std::runtime_error("eglInitialize failed.");
    }

    EGLint    numConfigs;
    EGLConfig eglCfg;
    rc = eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    if (rc == EGL_FALSE) {
        std::cerr << "eglChooseConfig failed.\n";
        return false;
    }

    if (numConfigs == 0) {
        std::cerr << "eglChooseConfig found no configurations.\n";
        return false;
    }

    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    if (eglSurf == EGL_NO_SURFACE) {
        std::cerr << "eglCreatePbufferSurface failed to create surface.\n";
        return false;
    }

    rc = eglBindAPI(EGL_OPENGL_API);
    if (rc == EGL_FALSE) {
        std::cerr << "eglBindAPI failed.\n";
        return false;
    }

    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
    if (eglCtx == EGL_NO_CONTEXT) {
        std::cerr << "eglCreateContext failed to create context.\n";
        return false;
    }

    rc = eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
    if (rc == EGL_FALSE) {
        std::cerr << "eglMakeCurrent failed.\n";
        return false;
    }

    *glContext = eglCtx;
    *glDisplay = eglDpy;

    return true;
#else
    return false;
#endif
}

#if defined(NANOVDB_USE_OPENGL)
static void initializeGL()
{
#if !defined(__EMSCRIPTEN__)
#if defined(NANOVDB_VIEWER_USE_GLES)
    if (!gladLoadGLES2Loader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error("Error: GLAD loader failed");
#elif defined(NANOVDB_USE_GLAD)
    if (!gladLoadGL())
        throw std::runtime_error("Error: GLAD loader failed");
#endif
#endif
    std::cout << "GL_VERSION  : " << glGetString(GL_VERSION) << "\n";
    std::cout << "GL_RENDERER : " << glGetString(GL_RENDERER) << "\n";
    std::cout << "GL_VENDOR   : " << glGetString(GL_VENDOR) << "\n";
}
#endif

void BatchRenderer::open()
{
    void* glContext = nullptr;
    void* glDisplay = nullptr;

#if defined(NANOVDB_USE_OPENGL)
    initializeEGL(mParams, &glContext, &glDisplay);

    if (glContext && glDisplay) {
        initializeGL();
        auto fb = new FrameBufferGL(glContext, glDisplay);
        fb->setupGL(mParams.mWidth, mParams.mHeight, nullptr, GL_RGBA32F, GL_DYNAMIC_DRAW);
        mFrameBuffer.reset(fb);
    } else {
        mFrameBuffer.reset(new FrameBufferHost());
    }
#else
    mFrameBuffer.reset(new FrameBufferHost());
#endif

    resize(mParams.mWidth, mParams.mHeight);

    printHelp();
}

void BatchRenderer::run()
{
    int breakWidth = 80;

    auto printBreak = [breakWidth] {
        std::cout.width(breakWidth);
        std::cout.fill('-');
        std::cout << '-' << std::endl;
        std::cout.fill(' ');
    };

    mFrame = 0;

    bool isSingleFrame = (mParams.mFrameCount <= 0);
    bool isComputingPSNR = (!mParams.mGoldPrefix.empty());

    FrameBufferHost goldImageBuffer;

    auto loadGoldImage = [&](int frame) -> bool {
        std::stringstream ss;
        if (!isSingleFrame) {
            ss << mParams.mGoldPrefix << '.' << std::setfill('0') << std::setw(4) << frame << std::setfill('\0') << ".pfm";
        } else {
            ss << mParams.mGoldPrefix << ".pfm";
        }
        return goldImageBuffer.load(ss.str().c_str());
    };

    int   count = (isSingleFrame) ? 1 : mParams.mFrameCount;
    float totalDuration = 0;
    float totalPSNR = 0;

    for (int i = 0; i < count; ++i) {
        render(mFrame);
        float renderDuration = mRenderStats.mDuration;

        float renderPSNR = -1;
        if (isComputingPSNR) {
            if (loadGoldImage(mFrame)) {
                renderPSNR = computePSNR(goldImageBuffer);
            }
        }

        printBreak();
        if (!isSingleFrame) {
            std::cout << "Frame    : " << mFrame << "/" << count << std::endl;
        }
        std::cout << "Duration : " << renderDuration << " ms" << std::endl;
        if (isComputingPSNR) {
            std::cout << "PSNR     : " << renderPSNR << " dB" << std::endl;
        }

        renderViewOverlay();

        saveFrameBuffer(!isSingleFrame, mFrame);

        ++mFrame;
        totalDuration += renderDuration;
        totalPSNR += renderPSNR;
    }

    if (!isSingleFrame) {
        printBreak();
        std::cout << "Average duration : ";
        std::cout << (totalDuration / count) << " s" << std::endl;
        if (isComputingPSNR) {
            std::cout << "Average PSNR     : ";
            std::cout << (totalPSNR / count) << " dB" << std::endl;
        }
        printBreak();
    }
}
