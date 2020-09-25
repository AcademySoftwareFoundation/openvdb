// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file Viewer.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of Viewer.
*/

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#include <cinttypes>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>

#if defined(__EMSCRIPTEN__)
#define GLFW_INCLUDE_ES3
#elif defined(NANOVDB_USE_GLAD)
#include <glad/glad.h>
#endif

#include <GLFW/glfw3.h>

#if !defined(__EMSCRIPTEN__)
#if defined(NANOVDB_USE_EGL)
#define GLFW_EXPOSE_NATIVE_EGL
#endif

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#define GLFW_EXPOSE_NATIVE_NSGL
#endif

#include <GLFW/glfw3native.h>
#endif

#if defined(NANOVDB_USE_EGL)
#include <EGL/eglplatform.h>
#include <EGL/egl.h>
#endif

#if defined(NANOVDB_VIEWER_USE_GLES)
#undef NANOVDB_USE_IMGUI
#endif

#if defined(NANOVDB_USE_IMGUI)
#include <imgui.h>
#include <imgui_internal.h>
#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#endif

#include "Viewer.h"
#include "RenderLauncher.h"

#include "FrameBufferHost.h"
#include "FrameBufferGL.h"

#include <nanovdb/util/IO.h> // for NanoVDB file import
#if defined(NANOVDB_USE_OPENVDB)
#include <nanovdb/util/OpenToNanoVDB.h>
#endif

#if defined(NANOVDB_USE_NFD)
#include <nfd.h>
#endif

static void keyCB(GLFWwindow* w, int key, int /*scancode*/, int action, int /*modifiers*/)
{
    reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(w))->onKey(key, action);
}

static void mouseButtonCB(GLFWwindow* w, int button, int action, int /*modifiers*/)
{
    reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(w))->onMouseButton(button, action);
}

static void mousePosCB(GLFWwindow* w, double x, double y)
{
    reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(w))->onMouseMove(int(x), int(y));
}

static void mouseWheelCB(GLFWwindow* w, double /*x*/, double y)
{
    reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(w))->onMouseWheel(int(y));
}

static void windowSizeCB(GLFWwindow* w, int width, int height)
{
    reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(w))->onResize(width, height);
}

static void windowRefreshCB(GLFWwindow* w)
{
    reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(w))->onRefresh();
}

static void windowDropCB(GLFWwindow* w, int count, const char** paths)
{
    reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(w))->onDrop(count, paths);
}

static bool openFileDialog(std::string& outPathStr)
{
#if defined(NANOVDB_USE_NFD)
    nfdchar_t*  outPath = NULL;
    nfdresult_t result = NFD_OpenDialog("vdb,nvdb", NULL, &outPath);
    if (result == NFD_OKAY) {
        outPathStr = outPath;
        free(outPath);
    } else if (result == NFD_CANCEL) {
        return false;
    } else {
        throw std::runtime_error(std::string(NFD_GetError()));
    }
    return true;
#else
    return false;
#endif
}

static bool openFolderDialog(std::string& outPathStr, const std::string& pathStr)
{
#if defined(NANOVDB_USE_NFD)
    nfdchar_t*  outPath = NULL;
    nfdresult_t result = NFD_PickFolder(pathStr.c_str(), &outPath);
    if (result == NFD_OKAY) {
        outPathStr = outPath;
        free(outPath);
    } else if (result == NFD_CANCEL) {
        return false;
    } else {
        throw std::runtime_error(std::string(NFD_GetError()));
    }
    return true;
#else
    return false;
#endif
}

Viewer::Viewer(const RendererParams& params)
    : RendererBase(params)
{
#if defined(NANOVDB_USE_OPENVDB)
    openvdb::initialize();
#endif

    setRenderPlatform(0);
    mFps = 0;
}

Viewer::~Viewer()
{
}

void Viewer::close()
{
    // ensure the GL context is active for GL resource destruction.
    glfwMakeContextCurrent((GLFWwindow*)mWindow);
    mFrameBuffer.reset();

#if defined(NANOVDB_USE_IMGUI)
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
#endif

    if (glfwGetCurrentContext() == mWindow)
        glfwMakeContextCurrent(nullptr);

    glfwDestroyWindow((GLFWwindow*)mWindow);
    mWindow = nullptr;
    glfwTerminate();
}

void Viewer::updateWindowTitle()
{
    if (!mWindow)
        return;

    if (mRenderGroupIndex < 0 || mGridGroups.size() == 0)
        return;

    auto group = mGridGroups[mRenderGroupIndex];
    auto instance = group->mInstances[group->mCurrentGridIndex];

    std::ostringstream ss;
    ss << "Viewer: " << group->mName << "[" << instance->mGridName << "] - " << mRenderLauncher.name() << " @ " << mFps << " fps";
    glfwSetWindowTitle((GLFWwindow*)mWindow, ss.str().c_str());
}

double Viewer::getTime()
{
    return glfwGetTime();
}

static void errorCallbackGLFW(int, const char* msg)
{
    throw std::runtime_error(msg);
}

static void initializeGL()
{
#if defined(NANOVDB_USE_GLAD)
#if defined(NANOVDB_VIEWER_USE_GLES)
    if (!gladLoadGLES2Loader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error("Error: GLAD loader failed");
#else
    if (!gladLoadGL())
        throw std::runtime_error("Error: GLAD loader failed");
#endif
#endif
    std::cout << "GL_VERSION  : " << glGetString(GL_VERSION) << "\n";
    std::cout << "GL_RENDERER : " << glGetString(GL_RENDERER) << "\n";
    std::cout << "GL_VENDOR   : " << glGetString(GL_VENDOR) << "\n";
}

void Viewer::open()
{
    if (!glfwInit())
        throw std::runtime_error("Error: Unable to initialize GLFW");

    glfwSetErrorCallback(errorCallbackGLFW);

#if defined(__EMSCRIPTEN__)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
#elif defined(NANOVDB_VIEWER_USE_GLES)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
#elif __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
#endif

    mWindow = glfwCreateWindow(mParams.mWidth, mParams.mHeight, "", NULL, NULL);
    if (!mWindow) {
        glfwTerminate();
        throw std::runtime_error("Error: Unable to create GLFW window");
    }
    updateWindowTitle();

    glfwSetWindowUserPointer((GLFWwindow*)mWindow, this);
    glfwSetKeyCallback((GLFWwindow*)mWindow, keyCB);
    glfwSetMouseButtonCallback((GLFWwindow*)mWindow, mouseButtonCB);
    glfwSetCursorPosCallback((GLFWwindow*)mWindow, mousePosCB);
    glfwSetScrollCallback((GLFWwindow*)mWindow, mouseWheelCB);
    glfwSetWindowSizeCallback((GLFWwindow*)mWindow, windowSizeCB);
    glfwSetWindowRefreshCallback((GLFWwindow*)mWindow, windowRefreshCB);
    glfwSetDropCallback((GLFWwindow*)mWindow, windowDropCB);

    void* glContext = nullptr;
    void* glDisplay = nullptr;

#if defined(_WIN32)
    glContext = glfwGetWGLContext((GLFWwindow*)mWindow);
    glDisplay = glfwGetWin32Window((GLFWwindow*)mWindow);
#elif defined(__linux__)
    glContext = glfwGetGLXContext((GLFWwindow*)mWindow);
    glDisplay = glfwGetX11Display();
#elif defined(__APPLE__)
    glContext = glfwGetNSGLContext((GLFWwindow*)mWindow);
    glDisplay = glfwGetCocoaWindow((GLFWwindow*)mWindow);
#endif

    glfwMakeContextCurrent((GLFWwindow*)mWindow);

    initializeGL();

    mFrameBuffer.reset(new FrameBufferGL(glContext, glDisplay));
    resize(mParams.mWidth, mParams.mHeight);
    resetAccumulationBuffer();

#ifdef NANOVDB_USE_IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*)mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 100");

    NANOVDB_GL_CHECKERRORS();
#else
    printHelp();
#endif
}

void Viewer::mainLoop(void* userData)
{
    auto viewerPtr = reinterpret_cast<Viewer*>(userData);

    bool rc = viewerPtr->runLoop();
    if (rc == false) {
#if defined(__EMSCRIPTEN__)
        emscripten_cancel_main_loop();
#endif
    }
}

bool Viewer::runLoop()
{
    //printf("frame: %d\n", mFrame);

    render(mFrame);
    renderViewOverlay();

#if defined(NANOVDB_USE_IMGUI) && !defined(__EMSCRIPTEN__)

    // workaround for bad GL state in ImGui...
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    NANOVDB_GL_CHECKERRORS();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //ImGui::ShowDemoWindow();
    drawMenuBar();
    drawGridOutliner();
    drawRenderOptionsDialog();
    drawRenderStatsOverlay();
    drawGridStatsOverlay();
    drawAboutDialog();
    drawHelpDialog();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    ImGui::EndFrame();
#endif

    // eval fps...
    ++mFpsFrame;
    double elapsed = glfwGetTime() - mTime;
    if (elapsed > 1.0) {
        mTime = glfwGetTime();
        mFps = (int)(double(mFpsFrame) / elapsed);
        updateWindowTitle();
        mFpsFrame = 0;
    }

    glfwSwapBuffers((GLFWwindow*)mWindow);

    glfwPollEvents();

    bool stop = (glfwWindowShouldClose((GLFWwindow*)mWindow) > 0);

    ++mFrame;
    return !stop;
}

void Viewer::run()
{
    glfwMakeContextCurrent((GLFWwindow*)mWindow);
    glfwSwapInterval(0);

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glDepthFunc(GL_LESS);
    glDisable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glfwSwapBuffers((GLFWwindow*)mWindow);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mFpsFrame = 0;
    mTime = glfwGetTime();

#if defined(__EMSCRIPTEN__)
    emscripten_set_main_loop_arg(Viewer::mainLoop, this, -1, 1);
#else
    while (runLoop()) {
    }
#endif
}

void Viewer::renderViewOverlay()
{
}

void Viewer::resize(int width, int height)
{
    auto fb = static_cast<FrameBufferGL*>(mFrameBuffer.get());
    fb->setupGL(width, height, nullptr, GL_RGBA32F, GL_DYNAMIC_DRAW);
    resetAccumulationBuffer();
}

void Viewer::render(int frame)
{
    if (mWindow == nullptr)
        return;

    glfwMakeContextCurrent((GLFWwindow*)mWindow);

    if (mGridGroups.size() == 0) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }

    if (mRenderGroupIndex < 0)
        return;

    auto        group = mGridGroups[mRenderGroupIndex];
    auto        instance = group->mInstances[group->mCurrentGridIndex];
    const auto& gridHdl = instance->mGridHandle;

    bool hasCameraChanged = updateCamera(frame);

    if (hasCameraChanged) {
        resetAccumulationBuffer();
    }

    size_t gridByteSize = gridHdl.size();
    assert(gridByteSize);

    // modify RenderConstants...
    auto renderConstants = mParams.mOptions;
    renderConstants.useGroundReflections = false;

    auto wBbox = group->mBounds;
    auto wBboxSize = wBbox.max() - wBbox.min();
    renderConstants.groundHeight = wBbox.min()[1];
    renderConstants.groundFalloff = 1000.f * float(wBboxSize.length());
    int w = mFrameBuffer->width();
    int h = mFrameBuffer->height();
    int numAccumulations = (mParams.mUseAccumulation) ? ++mNumAccumulations : 0;

    Camera<float> camera(mCurrentCameraState->eye(), mCurrentCameraState->target(), mCurrentCameraState->V(), mCurrentCameraState->mFovY, float(w) / h);

    // prevent progressive rendering to happen on turntable (as it looks displeasing)
    renderConstants.useOcclusion *= (mParams.mUseTurntable || hasCameraChanged || mMouseDown) ? 0 : 1;
    numAccumulations = (!mParams.mUseTurntable) ? numAccumulations : 0;

    bool renderRc = false;

    auto renderMethod = group->mRenderMethod;
    if (renderMethod == RenderMethod::AUTO) {
        if (instance->mGridClassOverride == nanovdb::GridClass::FogVolume)
            renderMethod = RenderMethod::FOG_VOLUME;
        else if (instance->mGridClassOverride == nanovdb::GridClass::LevelSet)
            renderMethod = RenderMethod::LEVELSET;
        else if (instance->mGridClassOverride == nanovdb::GridClass::PointData)
            renderMethod = RenderMethod::POINTS;
        else if (instance->mGridClassOverride == nanovdb::GridClass::PointIndex)
            renderMethod = RenderMethod::POINTS;
        else
            renderMethod = RenderMethod::GRID;
    }

    renderRc = mRenderLauncher.render(renderMethod, w, h, mFrameBuffer.get(), camera, gridHdl, numAccumulations, renderConstants, nullptr);

    if (!renderRc)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    else
        mFrameBuffer->render(0, 0, w, h);
}

void Viewer::printHelp() const
{
    auto platforms = mRenderLauncher.getPlatformNames();

    assert(platforms.size() > 0);
    std::stringstream platformList;
    platformList << "[" << platforms[0];
    for (int i = 1; i < platforms.size(); ++i)
        platformList << "," << platforms[i];
    platformList << "]";

    std::cout << "-------------------------------------\n";
    std::cout << "Hot Keys:\n";
    std::cout << "-------------------------------------\n";
    std::cout << "\n";
    std::cout << "- Show Hot-Keys                [I]\n";
    std::cout << "- Renderer-platform            [1-9] = (" << (mParams.mRenderLauncherType) << " / " << platformList.str() << ")\n";
    std::cout << "- Next grid                    [+/-] = (" << (mRenderGroupIndex) << ")\n";
    std::cout << "\n";
    std::cout << "View options ------------------------\n";
    std::cout << "- Camera Home                  [H]\n";
    std::cout << "- Camera Move                  [WASD]\n";
    std::cout << "\n";
    std::cout << "Render options ----------------------\n";
    std::cout << "- Toggle Render Progressive    [P] = (" << (mParams.mUseAccumulation ? "ON" : "OFF") << ")\n";
    std::cout << "- Toggle Render Lighting       [L] = (" << (mParams.mOptions.useLighting ? "ON" : "OFF") << ")\n";
    std::cout << "- Toggle Render Shadows        [B] = (" << (mParams.mOptions.useShadows ? "ON" : "OFF") << ")\n";
    std::cout << "- Toggle Render Ground-plane   [G] = (" << (mParams.mOptions.useGround ? "ON" : "OFF") << ")\n";
    std::cout << "- Toggle Render Occlusion      [O] = (" << (mParams.mOptions.useOcclusion ? "ON" : "OFF") << ")\n";
    std::cout << "-------------------------------------\n";
    std::cout << "\n";
}

bool Viewer::updateCamera(int frame)
{
    if (mRenderGroupIndex < mGridGroups.size()) {
        auto group = mGridGroups[mRenderGroupIndex];
        if (mPlaybackState == PlaybackState::PLAY) {
            setGridIndex(mRenderGroupIndex, ++group->mCurrentGridIndex);
        }
    }

    if (mParams.mUseTurntable) {
        int count = (mParams.mFrameCount == 0) ? 100 : mParams.mFrameCount;
        mCurrentCameraState->mCameraRotation[1] = (frame * 2.0f * 3.14159265f) / count;
        mCurrentCameraState->mIsViewChanged = true;
    } else {
#if defined(NANOVDB_USE_IMGUI)
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard)
            return false;
#endif

        const float cameraSpeed = 5.0f;
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_W) == GLFW_PRESS) {
            mCurrentCameraState->mCameraDistance -= cameraSpeed * 100;
            mCurrentCameraState->mIsViewChanged = true;
        }
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_S) == GLFW_PRESS) {
            mCurrentCameraState->mCameraDistance += cameraSpeed * 100;
            mCurrentCameraState->mIsViewChanged = true;
        }
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_A) == GLFW_PRESS) {
            mCurrentCameraState->mCameraLookAt = mCurrentCameraState->mCameraLookAt + (cameraSpeed * mCurrentCameraState->U());
            mCurrentCameraState->mIsViewChanged = true;
        }
        if (glfwGetKey((GLFWwindow*)mWindow, GLFW_KEY_D) == GLFW_PRESS) {
            mCurrentCameraState->mCameraLookAt = mCurrentCameraState->mCameraLookAt - (cameraSpeed * mCurrentCameraState->U());
            mCurrentCameraState->mIsViewChanged = true;
        }
    }
    return mCurrentCameraState->update();
}

void Viewer::onDrop(int numPaths, const char** paths)
{
    for (int i = 0; i < numPaths; i++) {
        try {
            addGrid(paths[i], paths[i]);
        }
        catch (const std::exception& e) {
            std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
        }
    }
}

void Viewer::onKey(int key, int action)
{
#if defined(NANOVDB_USE_IMGUI)
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard)
        return;
#endif

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE || key == 'Q') {
            glfwSetWindowShouldClose((GLFWwindow*)mWindow, true);
        } else if (key == 'I') {
            printHelp();
        } else if (key == 'H') {
            resetCamera();
            resetAccumulationBuffer();
        } else if (key == '`') {
            mIsDrawingOutliner = !mIsDrawingOutliner;
        } else if (key == 'B') {
            mParams.mOptions.useShadows = (mParams.mOptions.useShadows > 0) ? 0.f : 1.f;
            resetAccumulationBuffer();
        } else if (key == 'G') {
            mParams.mOptions.useGround = (mParams.mOptions.useGround > 0) ? 0.f : 1.f;
            resetAccumulationBuffer();
        } else if (key == 'O') {
            mParams.mOptions.useOcclusion = (mParams.mOptions.useOcclusion > 0) ? 0.f : 1.f;
            resetAccumulationBuffer();
        } else if (key == 'L') {
            mParams.mOptions.useLighting = (mParams.mOptions.useLighting > 0) ? 0.f : 1.f;
            resetAccumulationBuffer();
        } else if (key == 'P') {
            mParams.mUseAccumulation = !mParams.mUseAccumulation;
            resetAccumulationBuffer();
        } else if (key == 'T') {
            mParams.mUseTurntable = !mParams.mUseTurntable;
            resetAccumulationBuffer();
        } else if (key == GLFW_KEY_MINUS) {
            setGridIndex(mRenderGroupIndex, mGridGroups[mRenderGroupIndex]->mCurrentGridIndex - 1);
            updateWindowTitle();
            resetAccumulationBuffer();
            mFps = 0;
        } else if (key == GLFW_KEY_EQUAL) {
            setGridIndex(mRenderGroupIndex, mGridGroups[mRenderGroupIndex]->mCurrentGridIndex + 1);
            updateWindowTitle();
            resetAccumulationBuffer();
            mFps = 0;
        } else if (key == GLFW_KEY_LEFT_BRACKET) {
            mRenderGroupIndex = (mRenderGroupIndex - 1) % mGridGroups.size();
            setGridIndex(mRenderGroupIndex, mGridGroups[mRenderGroupIndex]->mCurrentGridIndex);
            updateWindowTitle();
            resetAccumulationBuffer();
            mFps = 0;
        } else if (key == GLFW_KEY_RIGHT_BRACKET) {
            mRenderGroupIndex = (mRenderGroupIndex + 1) % mGridGroups.size();
            setGridIndex(mRenderGroupIndex, mGridGroups[mRenderGroupIndex]->mCurrentGridIndex);
            updateWindowTitle();
            resetAccumulationBuffer();
            mFps = 0;
        } else if (key == GLFW_KEY_PRINT_SCREEN) {
            saveFrameBuffer(false);
        } else if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
            setRenderPlatform((key - GLFW_KEY_1));
            updateWindowTitle();
            mFps = 0;
        }
    }
}

void Viewer::onMouseButton(int button, int action)
{
#if defined(NANOVDB_USE_IMGUI)
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;
#endif

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mMouseDown = true;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mMouseDown = true;
            mIsMouseRightDown = true;
        }
    }

    if (action == GLFW_RELEASE) {
        mMouseDown = false;
        mIsMouseRightDown = false;
    }

    mIsFirstMouseMove = true;
    mCurrentCameraState->mIsViewChanged = true;

    //printf("mouse(%f, %f, %d)\n", mMouseX, mMouseY, mMouseDown?1:0);
}

void Viewer::onMouseMove(int x, int y)
{
#if defined(NANOVDB_USE_IMGUI)
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;
#endif

    const float orbitSpeed = 0.01f;
    const float strafeSpeed = mCurrentCameraState->mCameraDistance * 0.00005f;

    if (mIsFirstMouseMove) {
        mMouseX = float(x);
        mMouseY = float(y);
        mIsFirstMouseMove = false;
    }

    float dx = float(x) - mMouseX;
    float dy = float(y) - mMouseY;

    if (mMouseDown) {
        if (!mIsMouseRightDown) {
            // orbit...
            mCurrentCameraState->mCameraRotation[1] -= dx * orbitSpeed;
            mCurrentCameraState->mCameraRotation[0] += dy * orbitSpeed;

            mCurrentCameraState->mIsViewChanged = true;
        } else {
            // strafe...
            mCurrentCameraState->mCameraLookAt = mCurrentCameraState->mCameraLookAt + (dy * mCurrentCameraState->V() + dx * mCurrentCameraState->U()) * strafeSpeed;

            mCurrentCameraState->mIsViewChanged = true;
        }
    }

    mMouseX = float(x);
    mMouseY = float(y);

    //printf("mouse(%f, %f)\n", mMouseX, mMouseY);
}

void Viewer::onMouseWheel(int pos)
{
#if defined(NANOVDB_USE_IMGUI)
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;
#endif

    const float zoomSpeed = mCurrentCameraState->mCameraDistance * 0.1f;
    pos += mWheelPos;

    int speed = abs(mWheelPos - pos);

    if (mWheelPos >= pos) {
        mCurrentCameraState->mCameraDistance += float(speed) * zoomSpeed;
    } else {
        mCurrentCameraState->mCameraDistance -= float(speed) * zoomSpeed;
        mCurrentCameraState->mCameraDistance = std::max(0.001f, mCurrentCameraState->mCameraDistance);
    }

    mWheelPos = pos;
    mCurrentCameraState->mIsViewChanged = true;
    mIsFirstMouseMove = false;
}

void Viewer::onResize(int width, int height)
{
    resize(width, height);
}

void Viewer::drawHelpDialog()
{
#if defined(NANOVDB_USE_IMGUI)

    if (!mIsDrawingHelpDialog)
        return;

    ImGui::Begin("Help", &mIsDrawingHelpDialog, ImGuiWindowFlags_None);
    ImGui::Text("Coming soon...\n");
    ImGui::End();
#endif
}

void Viewer::drawRenderOptionsDialog()
{
#if defined(NANOVDB_USE_IMGUI)

    bool isChanged = false;

    if (!mIsDrawingRenderOptions)
        return;

    ImGui::Begin("Render Options", &mIsDrawingRenderOptions, ImGuiWindowFlags_AlwaysAutoResize);

    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("Render-options", tab_bar_flags)) {
        if (ImGui::BeginTabItem("Common")) {
            auto               group = mGridGroups[mRenderGroupIndex];
            static const char* methodNames = "Auto\0LevelSet\0FogVolume\0Grid\0Points\0\0";
            isChanged |= ImGui::Combo("Method", (int*)&group->mRenderMethod, methodNames);
            ImGui::Separator();

            drawRenderPlatformWidget("Platform");

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Output", ImGuiTreeNodeFlags_DefaultOpen)) {
                static char outputStr[128] = "";
                if (ImGui::InputTextWithHint("Output Prefix", mParams.mOutputPrefix.c_str(), outputStr, IM_ARRAYSIZE(outputStr))) {
                    mParams.mOutputPrefix.assign(outputStr);
                }

#if defined(NANOVDB_USE_NFD)
                ImGui::SameLine();
                if (ImGui::Button("Browse...")) {
                    std::string newFilePath;

                    auto currentFileName = mParams.mOutputPrefix;
                    auto currentFilePath = mParams.mOutputPrefix;
                    if (currentFilePath.find('/') != std::string::npos || currentFilePath.find('\\') != std::string::npos) {
                        currentFilePath = currentFilePath.substr(0, currentFilePath.find_last_of('/')).substr(0, currentFilePath.find_last_of('\\')) + '/';
                    } else {
                        currentFilePath = "./";
                    }
                    printf("currentFilePath: %s\n", currentFilePath.c_str());
                    if (currentFilePath != "./") {
                        currentFileName = currentFileName.substr(currentFilePath.length());
                    }

#if defined(_WIN32)
                    if (currentFilePath == "./") {
                        char buffer[MAX_PATH];
                        GetCurrentDirectoryA(MAX_PATH, buffer);
                        currentFilePath = buffer;
                    }
#endif
                    printf("currentFileName: %s\n", currentFileName.c_str());

                    if (openFolderDialog(newFilePath, currentFilePath)) {
                        newFilePath += '/';
                        printf("newFilePath: %s\n", newFilePath.c_str());
                        mParams.mOutputPrefix = newFilePath + currentFileName;
                        printf("mOutputPrefix: %s\n", mParams.mOutputPrefix.c_str());
                    }
                }
#endif

                isChanged |= ImGui::InputInt("Frame count", &mParams.mFrameCount, 1);

                if (mParams.mOutputPrefix.empty()) {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                }

                if (ImGui::Button("Render")) {
                    bool isSingleFrame = (mParams.mFrameCount <= 0);
                    int  count = (isSingleFrame) ? 1 : mParams.mFrameCount;
                    for (int i = 0; i < count; ++i) {
                        render(i);
                        saveFrameBuffer(!isSingleFrame, i);
                    }
                }

                if (mParams.mOutputPrefix.empty()) {
                    ImGui::PopItemFlag();
                    ImGui::PopStyleVar();
                }
            }
            ImGui::Separator();

            isChanged |= ImGui::Checkbox("Progressive", &mParams.mUseAccumulation);
            ImGui::Separator();

            ImGui::EndTabItem();
        }

        auto renderMethod = RenderMethod::AUTO;
        if (mRenderGroupIndex >= 0) {
            auto group = mGridGroups[mRenderGroupIndex];
            renderMethod = group->mRenderMethod;

            auto& gridHdl = group->mInstances[group->mCurrentGridIndex]->mGridHandle;
            if (renderMethod == RenderMethod::AUTO) {
                if (gridHdl.gridMetaData()->isFogVolume())
                    renderMethod = RenderMethod::FOG_VOLUME;
                else if (gridHdl.gridMetaData()->isLevelSet())
                    renderMethod = RenderMethod::LEVELSET;
                else if (gridHdl.gridMetaData()->isPointIndex())
                    renderMethod = RenderMethod::POINTS;
                else if (gridHdl.gridMetaData()->isPointData())
                    renderMethod = RenderMethod::POINTS;
                else
                    renderMethod = RenderMethod::GRID;
            }
        }

        if (renderMethod == RenderMethod::LEVELSET) {
            if (ImGui::BeginTabItem("LevelSet")) {
                if (ImGui::CollapsingHeader("Shading", ImGuiTreeNodeFlags_DefaultOpen)) {
                    isChanged |= ImGui::SliderFloat("Occlusion", &mParams.mOptions.useOcclusion, 0.0f, 1.0f);
                    isChanged |= ImGui::SliderFloat("Ground", &mParams.mOptions.useGround, 0.0f, 1.0f);
                    isChanged |= ImGui::SliderFloat("Lighting", &mParams.mOptions.useLighting, 0.0f, 1.0f);
                    isChanged |= ImGui::SliderFloat("Shadows", &mParams.mOptions.useShadows, 0.0f, 1.0f);
                    isChanged |= ImGui::InputInt("SamplesPerPixel", &mParams.mOptions.samplesPerPixel, 1);
                }
                ImGui::Separator();

                if (ImGui::CollapsingHeader("Tonemap", ImGuiTreeNodeFlags_DefaultOpen)) {
                    isChanged |= ImGui::SliderFloat("WhitePoint", &mParams.mOptions.tonemapWhitePoint, 1.0f, 20.0f);
                }

                ImGui::EndTabItem();
            }
        }

        if (renderMethod == RenderMethod::FOG_VOLUME) {
            if (ImGui::BeginTabItem("FogVolume")) {
                isChanged |= ImGui::SliderFloat("Volume density", &mParams.mOptions.volumeDensity, 0.0f, 1.0f);

                ImGui::Separator();

                if (ImGui::CollapsingHeader("Shading", ImGuiTreeNodeFlags_DefaultOpen)) {
                    isChanged |= ImGui::SliderFloat("Occlusion", &mParams.mOptions.useOcclusion, 0.0f, 1.0f);
                    isChanged |= ImGui::SliderFloat("Ground", &mParams.mOptions.useGround, 0.0f, 1.0f);
                    isChanged |= ImGui::SliderFloat("Lighting", &mParams.mOptions.useLighting, 0.0f, 1.0f);
                    isChanged |= ImGui::SliderFloat("Shadows", &mParams.mOptions.useShadows, 0.0f, 1.0f);
                    isChanged |= ImGui::InputInt("SamplesPerPixel", &mParams.mOptions.samplesPerPixel, 1);
                }
                ImGui::Separator();

                if (ImGui::CollapsingHeader("Tonemap", ImGuiTreeNodeFlags_DefaultOpen)) {
                    isChanged |= ImGui::SliderFloat("WhitePoint", &mParams.mOptions.tonemapWhitePoint, 1.0f, 20.0f);
                }

                ImGui::EndTabItem();
            }
        }

        if (renderMethod == RenderMethod::POINTS) {
            if (ImGui::BeginTabItem("Points")) {
                isChanged |= ImGui::SliderFloat("Volume density", &mParams.mOptions.volumeDensity, 0.0f, 1.0f);
                ImGui::Separator();
                if (ImGui::CollapsingHeader("Attributes", ImGuiTreeNodeFlags_DefaultOpen)) {
                    isChanged |= drawPointRenderOptionsWidget();
                }
                ImGui::EndTabItem();
            }
        }

        if (renderMethod == RenderMethod::GRID) {
            if (ImGui::BeginTabItem("Grid")) {
                ImGui::Text("No parameters.");
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }

    if (isChanged)
        resetAccumulationBuffer();

    ImGui::End();
#endif
}

bool Viewer::drawPointRenderOptionsWidget()
{
    bool isChanged = false;
#if defined(NANOVDB_USE_IMGUI)

    static std::vector<std::string> semanticNames;
    if (semanticNames.empty()) {
        for (int i = 1; i < (int)nanovdb::GridBlindDataSemantic::End; ++i)
            semanticNames.push_back(getStringForBlindDataSemantic(nanovdb::GridBlindDataSemantic(i)));
    }

    auto  group = mGridGroups[mRenderGroupIndex];
    auto& gridHdl = group->mInstances[group->mCurrentGridIndex]->mGridHandle;
    if (gridHdl.gridMetaData()->isPointData()) {
        auto grid = gridHdl.grid<uint32_t>();
        assert(grid);

        std::vector<std::string> attributeNames;
        int                      n = grid->blindDataCount();

        std::vector<nanovdb::GridBlindMetaData> attributeMeta;
        attributeMeta.push_back(nanovdb::GridBlindMetaData{});
        attributeNames.push_back("None");
        for (int i = 0; i < n; ++i) {
            auto meta = grid->blindMetaData(i);
            attributeMeta.push_back(meta);
            attributeNames.push_back(meta.mName + std::string(" (") + nanovdb::io::getStringForGridType(meta.mDataType) + std::string(")"));
        }

        static auto vector_getter = [](void* vec, int idx, const char** out_text) {
            auto& vector = *static_cast<std::vector<std::string>*>(vec);
            if (idx < 0 || idx >= static_cast<int>(vector.size())) {
                return false;
            }
            *out_text = vector.at(idx).c_str();
            return true;
        };

        int        attributeIndex = 0;
        static int semanticIndex = 0;
        // Left
        {
            ImGui::BeginChild("left pane", ImVec2(100, 200), true, ImGuiWindowFlags_AlwaysAutoResize);

            //ImGui::ListBox("", &semanticIndex, vector_getter, static_cast<void*>(&semanticNames), semanticNames.size(), semanticNames.size()-1);

            for (int i = 0; i < semanticNames.size(); i++) {
                if (ImGui::Selectable(semanticNames[i].c_str(), semanticIndex == i))
                    semanticIndex = i;
            }
            ImGui::EndChild();
        }
        ImGui::SameLine();

        // Right
        {
            //ImGui::BeginGroup();
            ImGui::BeginChild("item view", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), false, ImGuiWindowFlags_AlwaysAutoResize); // Leave room for 1 line below us
            attributeIndex = mParams.mOptions.attributeSemanticMap[semanticIndex + 1].attribute + 1; // add one as item[0] is "None"
            isChanged |= ImGui::Combo("Attribute", &attributeIndex, vector_getter, static_cast<void*>(&attributeNames), attributeNames.size());
            isChanged |= ImGui::DragFloat("Offset", &mParams.mOptions.attributeSemanticMap[semanticIndex + 1].offset, 0.01, 0.0, 1.0);
            isChanged |= ImGui::DragFloat("Gain", &mParams.mOptions.attributeSemanticMap[semanticIndex + 1].gain, 0.01, 0.0, 1.0);
            ImGui::EndChild();
            //ImGui::EndGroup();
        }

        if (isChanged) {
            mParams.mOptions.attributeSemanticMap[semanticIndex + 1].attribute = attributeIndex - 1; // minus one as item[0] is "None"
        }
    }
#endif
    return isChanged;
}

void Viewer::drawAboutDialog()
{
#if defined(NANOVDB_USE_IMGUI)

    if (!mIsDrawingAboutDialog)
        return;

    ImGuiIO& io = ImGui::GetIO();
    ImVec2   window_pos = ImVec2(io.DisplaySize.x / 2, io.DisplaySize.y / 2);
    ImVec2   window_pos_pivot = ImVec2(0.5f, 0.5f);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    //ImGui::SetNextWindowBgAlpha(0.35f);

    if (!ImGui::Begin("About", &mIsDrawingAboutDialog, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }

    ImGui::Text("NanoVDB Viewer\n");

    ImGui::Separator();

    ImGui::Text("Build options:\n");
    {
        ImGui::BeginChild("##build", ImVec2(0, 60), true);
#ifdef NANOVDB_USE_BLOSC
        ImGui::BulletText("BLOSC");
#endif
#ifdef NANOVDB_USE_OPENVDB
        ImGui::BulletText("OpenVDB");
#endif
#ifdef NANOVDB_USE_ZIP
        ImGui::BulletText("ZLIB");
#endif
#ifdef NANOVDB_USE_TBB
        ImGui::BulletText("Intel TBB");
#endif
#ifdef NANOVDB_USE_CUDA
        ImGui::BulletText("NVIDIA CUDA");
#endif
#ifdef NANOVDB_USE_OPENCL
        ImGui::BulletText("OpenCL");
#endif
        ImGui::EndChild();
    }

    ImGui::Separator();

    ImGui::Text("Supported render-platforms:\n");
    {
        ImGui::BeginChild("##platforms", ImVec2(0, 60), true);
        for (auto& it : mRenderLauncher.getPlatformNames())
            ImGui::BulletText("%s\n", it.c_str());
        ImGui::EndChild();
    }

    ImGui::End();
#endif
}

void Viewer::drawRenderPlatformWidget(const char* label)
{
#if defined(NANOVDB_USE_IMGUI)
    auto platforms = mRenderLauncher.getPlatformNames();

    static char comboStr[1024];
    char*       comboStrPtr = comboStr;
    for (auto& it : platforms) {
        strncpy(comboStrPtr, it.c_str(), it.length());
        comboStrPtr += it.length();
        *(comboStrPtr++) = 0;
    }
    *(comboStrPtr++) = 0;

    if (ImGui::Combo(label, &mParams.mRenderLauncherType, comboStr))
        setRenderPlatform(mParams.mRenderLauncherType);
#endif
}

void Viewer::drawGridTree()
{
#if defined(NANOVDB_USE_IMGUI)
    std::vector<int> deleteRequests;

    for (int i = 0; i < mGridGroups.size(); i++) {
        auto group = mGridGroups[i];

        if (ImGui::TreeNodeEx(group->mName.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            for (int ins = 0; ins < group->mInstances.size(); ins++) {
                const auto& instance = group->mInstances[ins];

                bool  isSelected = (group->mCurrentGridIndex == ins);
                auto& hdl = instance->mGridHandle;
                if (ImGui::Selectable(instance->mGridName.c_str(), isSelected)) {
                    setGridIndex(i, ins);
                }
            }
            ImGui::TreePop();
        }

        if (ImGui::BeginPopupContextItem((group->mName + "-context").c_str())) {
            if (ImGui::Button("Delete Grid"))
                deleteRequests.push_back(i);
            ImGui::EndPopup();
        }
    }

    removeGridIndices(deleteRequests);

#endif
}

void Viewer::drawGridOutliner()
{
    if (!mIsDrawingOutliner)
        return;

#if defined(NANOVDB_USE_IMGUI)

    const float DISTANCE = 10.0f;
    ImVec2      window_pos = ImVec2(DISTANCE, DISTANCE + 16);
    ImVec2      window_pos_pivot = ImVec2(0, 0);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.35f);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
    window_flags |= ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav; // | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;

    if (ImGui::Begin("Grids", &mIsDrawingOutliner, window_flags)) {
        drawGridTree();
    }
    ImGui::End();
#endif
}

void Viewer::drawGridStatsOverlay()
{
    if (!mIsDrawingGridStats)
        return;

#if defined(NANOVDB_USE_IMGUI)
    const float DISTANCE = 10.0f;
    ImGuiIO&    io = ImGui::GetIO();
    ImVec2      window_pos = ImVec2(DISTANCE, io.DisplaySize.y - DISTANCE);
    ImVec2      window_pos_pivot = ImVec2(0, 1);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.35f);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;

    if (ImGui::Begin("Grid:", &mIsDrawingGridStats, window_flags)) {
        if (mGridGroups.size() == 0) {
            ImGui::Text("Drag and drop a file onto the window.");
        } else {
            auto group = mGridGroups[mRenderGroupIndex];
            auto instance = group->mInstances[group->mCurrentGridIndex];
            auto meta = instance->mGridHandle.gridMetaData();
            auto bbox = meta->worldBBox();
            auto bboxMin = bbox.min();
            auto bboxMax = bbox.max();
            auto iBBox = meta->indexBBox();
            auto effectiveSize = iBBox.max() - iBBox.min() + nanovdb::Coord(1);

            ImGui::Text("File:");
            ImGui::SameLine(150);
            ImGui::Text("%s", instance->mFileName.c_str());
            ImGui::Text("Grid:");
            ImGui::SameLine(150);
            ImGui::Text("%s", meta->gridName());
            ImGui::Text("Effective res:");
            ImGui::SameLine(150);
            ImGui::Text("%dx%dx%d", effectiveSize[0], effectiveSize[1], effectiveSize[2]);
            ImGui::Text("BBox-min:");
            ImGui::SameLine(150);
            ImGui::Text("(%.2f,%.2f,%.2f)", bboxMin[0], bboxMin[1], bboxMin[2]);
            ImGui::Text("BBox-max:");
            ImGui::SameLine(150);
            ImGui::Text("(%.2f,%.2f,%.2f)", bboxMax[0], bboxMax[1], bboxMax[2]);
            ImGui::Text("Class(Type):");
            ImGui::SameLine(150);
            ImGui::Text("%s(%s)", nanovdb::io::getStringForGridClass(meta->gridClass()).c_str(), nanovdb::io::getStringForGridType(meta->gridType()).c_str());
            if (meta->gridClass() == nanovdb::GridClass::PointData || meta->gridClass() == nanovdb::GridClass::PointIndex) {
                ImGui::Text("Point count:");
                ImGui::SameLine(150);
                ImGui::Text("%" PRIu64, (meta->blindDataCount() > 0) ? meta->blindMetaData(0).mElementCount : 0);
            } else {
                ImGui::Text("Voxel count:");
                ImGui::SameLine(150);
                ImGui::Text("%" PRIu64, meta->activeVoxelCount());
            }
        }
    }
    ImGui::End();
#endif
}

void Viewer::drawRenderStatsOverlay()
{
    if (!mIsDrawingRenderStats)
        return;

#if defined(NANOVDB_USE_IMGUI)
    const float DISTANCE = 10.0f;
    ImGuiIO&    io = ImGui::GetIO();
    ImVec2      window_pos = ImVec2(io.DisplaySize.x - DISTANCE, DISTANCE + 16);
    ImVec2      window_pos_pivot = ImVec2(1.0f, 0.0f);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.35f);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;

    if (ImGui::Begin("Render Stats:", &mIsDrawingRenderStats, window_flags)) {
        ImGui::Text("FPS: %d", mFps);
        ImGui::Separator();
        drawRenderPlatformWidget("");
    }
    ImGui::End();
#endif
}

void Viewer::drawMenuBar()
{
#if defined(NANOVDB_USE_IMGUI)

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
#if defined(NANOVDB_USE_NFD)
            if (ImGui::MenuItem("Open...", "O")) {
                std::string filePath;
                if (openFileDialog(filePath)) {
                    try {
                        addGrid(filePath, filePath);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
                    }
                }
            }
#endif
            if (ImGui::MenuItem("Quit", "Q"))
                glfwSetWindowShouldClose((GLFWwindow*)mWindow, 1);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Show Outliner", NULL, mIsDrawingOutliner, true)) {
                mIsDrawingOutliner = !mIsDrawingOutliner;
            }
            if (ImGui::MenuItem("Show Grid Stats", NULL, mIsDrawingGridStats, true)) {
                mIsDrawingGridStats = !mIsDrawingGridStats;
            }
            if (ImGui::MenuItem("Show Render Stats", NULL, mIsDrawingRenderStats, true)) {
                mIsDrawingRenderStats = !mIsDrawingRenderStats;
            }
            if (ImGui::MenuItem("Turntable", "T", mParams.mUseTurntable, true)) {
                mParams.mUseTurntable = !mParams.mUseTurntable;
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Grids")) {
            drawGridTree();
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Render")) {
            std::string outputFilename = "no output prefix specified";
            if (!mParams.mOutputPrefix.empty()) {
                outputFilename = (mParams.mOutputPrefix + ".pfm");
            }
            if (ImGui::MenuItem("Save Screenshot", outputFilename.c_str(), false, !mParams.mOutputPrefix.empty())) {
                saveFrameBuffer(false);
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Options...")) {
                mIsDrawingRenderOptions = true;
            }

            if (ImGui::MenuItem("Play", nullptr, false, mPlaybackState == PlaybackState::STOP)) {
                mPlaybackState = PlaybackState::PLAY;
            }
            if (ImGui::MenuItem("Stop", nullptr, false, mPlaybackState == PlaybackState::PLAY)) {
                mPlaybackState = PlaybackState::STOP;
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("View Help"))
                mIsDrawingHelpDialog = true;
            ImGui::Separator();
            if (ImGui::MenuItem("About"))
                mIsDrawingAboutDialog = true;
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
#endif
}