// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file BatchRenderer.cpp
	\brief Implementation of BatchRenderer.
*/

#include "BatchRenderer.h"
#include <sstream>
#include <iomanip>
#include <iostream>

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
    mParams.mUseAccumulation = true;
    mPendingSceneFrame = params.mFrameStart;
}

bool initializeEGL(const RendererParams& params, void** glContext, void** glDisplay)
{
    (void)params;
    
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

    resizeFrameBuffer(mParams.mWidth, mParams.mHeight);
    resetCamera();
}

void BatchRenderer::run()
{
    if (mSelectedSceneNodeIndex < 0) {
        logError("Nothing to render");
        return;
    }

    int breakWidth = 80;

    auto printBreak = [breakWidth] {
        std::cout.width(breakWidth);
        std::cout.fill('-');
        std::cout << '-' << std::endl;
        std::cout.fill(' ');
    };

    bool isSingleFrame = (mParams.mFrameEnd <= mParams.mFrameStart);
    bool isComputingPSNR = (!mParams.mGoldPrefix.empty());

    FrameBufferHost goldImageBuffer;

    auto loadGoldImage = [&](int frame) -> bool {
        return goldImageBuffer.load(updateFilePathWithFrame(mParams.mGoldPrefix, frame).c_str(), "pfm");
    };

    std::stringstream ss;
    if (isSingleFrame) {
        ss << "Rendering frame " << mParams.mFrameStart;
    } else {
        ss << "Rendering frames " << mParams.mFrameStart << " - " << mParams.mFrameEnd;
    }
    if (mParams.mOutputFilePath.length()) {
        ss << " to " << mParams.mOutputFilePath;
        if (mParams.mOutputExtension.length()) {
            ss << " as " << mParams.mOutputExtension;
        }
    }
    logInfo(ss.str());

    int   count = (isSingleFrame) ? 1 : (mParams.mFrameEnd - mParams.mFrameStart + 1);
    float totalDuration = 0;
    float totalPSNR = 0;
    int   frame = mParams.mFrameStart;

    // go to the first frame and ensure it is loaded to get the bounds.
    // we use this for the bounds for resetCamera.
    setSceneFrame(mParams.mFrameStart);

    bool hasError = !updateNodeAttachmentRequests(mSceneNodes[mSelectedSceneNodeIndex], true, mIsDumpingLog);
    if (hasError) {
        logError("Some assets have errors. Unable to render frame " + std::to_string(frame) + "; bad asset");
        return;
    }

    // reset the camera to home.
    // TODO: we should allow setting the camera in the command-line arguments.
    resetCamera();

    for (; frame <= mParams.mFrameEnd; ++frame) {
        setSceneFrame(frame);

        hasError = !updateNodeAttachmentRequests(mSceneNodes[mSelectedSceneNodeIndex], true, mIsDumpingLog);
        if (hasError) {
            logError("Unable to render frame " + std::to_string(frame) + "; bad asset");
            break;
        }

        updateScene();

        for (int i = (mParams.mUseAccumulation) ? mParams.mMaxProgressiveSamples : 1; i > 0; --i) {
            render(frame);
        }

        float renderDuration = mRenderStats.mDuration;

        float renderPSNR = -1;
        if (isComputingPSNR) {
            if (loadGoldImage(frame)) {
                renderPSNR = computePSNR(goldImageBuffer);
            }
        }

        printBreak();
        if (!isSingleFrame) {
            std::cout << "Frame    : " << frame << std::endl;
        }
        std::cout << "Duration : " << renderDuration << " ms" << std::endl;
        if (isComputingPSNR) {
            std::cout << "PSNR     : " << renderPSNR << " dB" << std::endl;
        }

        renderViewOverlay();

        if (mParams.mOutputFilePath.empty() == false) {
            hasError = (saveFrameBuffer(frame) == false);
            if (hasError) {
                break;
            }
        }

        totalDuration += renderDuration;
        totalPSNR += renderPSNR;
    }

    if (hasError == false) {
        logInfo("Rendering complete.");
    } else {
        logError("Rendering failed.");
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
