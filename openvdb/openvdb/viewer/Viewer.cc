// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "Viewer.h"

#include "Camera.h"
#include "ClipBox.h"
#include "Font.h"
#include "RenderModules.h"
#include <openvdb/util/Formats.h> // for formattedInt()
#include <openvdb/util/logging.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/version.h> // for OPENVDB_LIBRARY_MAJOR_VERSION, etc.
#include <atomic>
#include <cmath> // for fabs()
#include <iomanip> // for std::setprecision()
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>
#include <limits>
#include <thread>
#include <chrono>

#if defined(_WIN32)
#include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>

namespace openvdb_viewer {

class ViewerImpl
{
public:
    using CameraPtr = std::shared_ptr<Camera>;
    using ClipBoxPtr = std::shared_ptr<ClipBox>;
    using RenderModulePtr = std::shared_ptr<RenderModule>;

    ViewerImpl();

    void init(const std::string& progName);

    std::string getVersionString() const;

    bool isOpen() const;
    bool open(int width = 900, int height = 800);
    void view(const openvdb::GridCPtrVec&);
    void handleEvents();
    void close();

    void resize(int width, int height);

    void showPrevGrid();
    void showNextGrid();

    bool needsDisplay();
    void setNeedsDisplay();

    void toggleRenderModule(size_t n);
    void toggleInfoText();

    // Internal
    void render();
    void interrupt();
    void setWindowTitle(double fps = 0.0);
    void showNthGrid(size_t n);
    void updateCutPlanes(int wheelPos);
    void swapBuffers();

    void keyCallback(int key, int action);
    void mouseButtonCallback(int button, int action);
    void mousePosCallback(int x, int y);
    void mouseWheelCallback(int pos);
    void windowSizeCallback(int width, int height);
    void windowRefreshCallback();

    static openvdb::BBoxd worldSpaceBBox(const openvdb::math::Transform&,
        const openvdb::CoordBBox&);
    static void sleep(double seconds);

private:
    bool mDidInit;
    CameraPtr mCamera;
    ClipBoxPtr mClipBox;
    RenderModulePtr mViewportModule;
    std::vector<RenderModulePtr> mRenderModules;
    openvdb::GridCPtrVec mGrids;
    size_t mGridIdx, mUpdates;
    std::string mGridName, mProgName, mGridInfo, mTransformInfo, mTreeInfo;
    int mWheelPos;
    bool mShiftIsDown, mCtrlIsDown, mShowInfo;
    bool mInterrupt;
    GLFWwindow* mWindow;
}; // class ViewerImpl


class ThreadManager
{
public:
    ThreadManager();

    void view(const openvdb::GridCPtrVec& gridList);
    void close();
    void resize(int width, int height);

private:
    void doView();
    static void* doViewTask(void* arg);

    std::atomic<bool> mRedisplay;
    bool mClose, mHasThread;
    std::thread mThread;
    openvdb::GridCPtrVec mGrids;
};


////////////////////////////////////////


namespace {

ViewerImpl* sViewer = nullptr;
ThreadManager* sThreadMgr = nullptr;
std::mutex sLock;


void
keyCB(GLFWwindow*, int key, int /*scancode*/, int action, int /*modifiers*/)
{
    if (sViewer) sViewer->keyCallback(key, action);
}


void
mouseButtonCB(GLFWwindow*, int button, int action, int /*modifiers*/)
{
    if (sViewer) sViewer->mouseButtonCallback(button, action);
}


void
mousePosCB(GLFWwindow*, double x, double y)
{
    if (sViewer) sViewer->mousePosCallback(int(x), int(y));
}


void
mouseWheelCB(GLFWwindow*, double /*xoffset*/, double yoffset)
{
    if (sViewer) sViewer->mouseWheelCallback(int(yoffset));
}


void
windowSizeCB(GLFWwindow*, int width, int height)
{
    if (sViewer) sViewer->windowSizeCallback(width, height);
}


void
windowRefreshCB(GLFWwindow*)
{
    if (sViewer) sViewer->windowRefreshCallback();
}

} // unnamed namespace


////////////////////////////////////////


Viewer
init(const std::string& progName, bool background)
{
    if (sViewer == nullptr) {
        std::lock_guard<std::mutex> lock(sLock);
        if (sViewer == nullptr) {
            OPENVDB_START_THREADSAFE_STATIC_WRITE
            sViewer = new ViewerImpl;
            OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
        }
    }
    sViewer->init(progName);

    if (background) {
        if (sThreadMgr == nullptr) {
            std::lock_guard<std::mutex> lock(sLock);
            if (sThreadMgr == nullptr) {
                OPENVDB_START_THREADSAFE_STATIC_WRITE
                sThreadMgr = new ThreadManager;
                OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
            }
        }
    } else {
        if (sThreadMgr != nullptr) {
            std::lock_guard<std::mutex> lock(sLock);
            delete sThreadMgr;
            OPENVDB_START_THREADSAFE_STATIC_WRITE
            sThreadMgr = nullptr;
            OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
        }
    }

    return Viewer();
}


void
exit()
{
    glfwTerminate();
}


////////////////////////////////////////


Viewer::Viewer()
{
    OPENVDB_LOG_DEBUG_RUNTIME("constructed Viewer from thread " << std::this_thread::get_id());
}


void
Viewer::open(int width, int height)
{
    if (sViewer) sViewer->open(width, height);
}


void
Viewer::view(const openvdb::GridCPtrVec& grids)
{
    if (sThreadMgr) {
        sThreadMgr->view(grids);
    } else if (sViewer) {
        sViewer->view(grids);
    }
}


void
Viewer::handleEvents()
{
    if (sViewer) sViewer->handleEvents();
}


void
Viewer::close()
{
    if (sThreadMgr) sThreadMgr->close();
    else if (sViewer) sViewer->close();
}


void
Viewer::resize(int width, int height)
{
    if (sViewer) sViewer->resize(width, height);
}


std::string
Viewer::getVersionString() const
{
    std::string version;
    if (sViewer) version = sViewer->getVersionString();
    return version;
}


////////////////////////////////////////


ThreadManager::ThreadManager()
    : mClose(false)
    , mHasThread(false)
{
    mRedisplay = false;
}


void
ThreadManager::view(const openvdb::GridCPtrVec& gridList)
{
    if (!sViewer) return;

    mGrids = gridList;
    mClose = false;
    mRedisplay = true;

    if (!mHasThread) {
        mThread = std::thread(doViewTask, this);
        mHasThread = true;
    }
}


void
ThreadManager::close()
{
    if (!sViewer) return;

    // Tell the viewer thread to exit.
    mRedisplay = false;
    mClose = true;
    // Tell the viewer to terminate its event loop.
    sViewer->interrupt();

    if (mHasThread) {
        mThread.join();
        mHasThread = false;
    }

    // Tell the viewer to close its window.
    sViewer->close();
}


void
ThreadManager::doView()
{
    // This function runs in its own thread.
    // The mClose and mRedisplay flags are set from the main thread.
    while (!mClose) {
        // If mRedisplay was true, then set it to false
        // and then, if sViewer, call view:
        bool expected = true;
        if (mRedisplay.compare_exchange_strong(expected, false)) {
            if (sViewer) sViewer->view(mGrids);
        }
        sViewer->sleep(0.5/*sec*/);
    }
}


//static
void*
ThreadManager::doViewTask(void* arg)
{
    if (ThreadManager* self = static_cast<ThreadManager*>(arg)) {
        self->doView();
    }
    return nullptr;
}


////////////////////////////////////////


ViewerImpl::ViewerImpl()
    : mDidInit(false)
    , mCamera(new Camera)
    , mClipBox(new ClipBox)
    , mGridIdx(0)
    , mUpdates(0)
    , mWheelPos(0)
    , mShiftIsDown(false)
    , mCtrlIsDown(false)
    , mShowInfo(true)
    , mInterrupt(false)
    , mWindow(nullptr)
{
}


void
ViewerImpl::init(const std::string& progName)
{
    mProgName = progName;

    if (!mDidInit) {
        struct Local {
            static void errorCB(int error, const char* descr) {
                OPENVDB_LOG_ERROR("GLFW Error " << error << ": " << descr);
            }
        };
        glfwSetErrorCallback(Local::errorCB);
        if (glfwInit() == GL_TRUE) {
            OPENVDB_LOG_DEBUG_RUNTIME("initialized GLFW from thread "
                << std::this_thread::get_id());
            mDidInit = true;
        } else {
            OPENVDB_LOG_ERROR("GLFW initialization failed");
        }
    }
    mViewportModule.reset(new ViewportModule);
}


std::string
ViewerImpl::getVersionString() const
{
    std::ostringstream ostr;

    ostr << "OpenVDB: " <<
        openvdb::OPENVDB_LIBRARY_MAJOR_VERSION << "." <<
        openvdb::OPENVDB_LIBRARY_MINOR_VERSION << "." <<
        openvdb::OPENVDB_LIBRARY_PATCH_VERSION;

    int major, minor, rev;
    glfwGetVersion(&major, &minor, &rev);
    ostr << ", " << "GLFW: " << major << "." << minor << "." << rev;

    if (mDidInit) {
        ostr << ", " << "OpenGL: ";
        std::shared_ptr<GLFWwindow> wPtr;
        GLFWwindow* w = mWindow;
        if (!w) {
            wPtr.reset(glfwCreateWindow(100, 100, "", nullptr, nullptr), &glfwDestroyWindow);
            w = wPtr.get();
        }
        if (w) {
            ostr << glfwGetWindowAttrib(w, GLFW_CONTEXT_VERSION_MAJOR) << "."
                << glfwGetWindowAttrib(w, GLFW_CONTEXT_VERSION_MINOR) << "."
                << glfwGetWindowAttrib(w, GLFW_CONTEXT_REVISION);
        }
    }
    return ostr.str();
}


bool
ViewerImpl::open(int width, int height)
{
    if (mWindow == nullptr) {
        glfwWindowHint(GLFW_RED_BITS, 8);
        glfwWindowHint(GLFW_GREEN_BITS, 8);
        glfwWindowHint(GLFW_BLUE_BITS, 8);
        glfwWindowHint(GLFW_ALPHA_BITS, 8);
        glfwWindowHint(GLFW_DEPTH_BITS, 32);
        glfwWindowHint(GLFW_STENCIL_BITS, 0);

        mWindow = glfwCreateWindow(
            width, height, mProgName.c_str(), /*monitor=*/nullptr, /*share=*/nullptr);

        OPENVDB_LOG_DEBUG_RUNTIME("created window " << std::hex << mWindow << std::dec
            << " from thread " << std::this_thread::get_id());

        if (mWindow != nullptr) {
            // Temporarily make the new window the current context, then create a font.
            std::shared_ptr<GLFWwindow> curWindow(
                glfwGetCurrentContext(), glfwMakeContextCurrent);
            glfwMakeContextCurrent(mWindow);
            BitmapFont13::initialize();
        }
    }
    mCamera->setWindow(mWindow);

    if (mWindow != nullptr) {
        glfwSetKeyCallback(mWindow, keyCB);
        glfwSetMouseButtonCallback(mWindow, mouseButtonCB);
        glfwSetCursorPosCallback(mWindow, mousePosCB);
        glfwSetScrollCallback(mWindow, mouseWheelCB);
        glfwSetWindowSizeCallback(mWindow, windowSizeCB);
        glfwSetWindowRefreshCallback(mWindow, windowRefreshCB);
    }
    return (mWindow != nullptr);
}


bool
ViewerImpl::isOpen() const
{
    return (mWindow != nullptr);
}


// Set a flag so as to break out of the event loop on the next iteration.
// (Useful only if the event loop is running in a separate thread.)
void
ViewerImpl::interrupt()
{
    mInterrupt = true;
    if (mWindow) glfwSetWindowShouldClose(mWindow, true);
}


void
ViewerImpl::handleEvents()
{
    glfwPollEvents();
}


void
ViewerImpl::close()
{
    OPENVDB_LOG_DEBUG_RUNTIME("about to close window " << std::hex << mWindow << std::dec
        << " from thread " << std::this_thread::get_id());

    mViewportModule.reset();
    mRenderModules.clear();
    mCamera->setWindow(nullptr);
    GLFWwindow* win = mWindow;
    mWindow = nullptr;
    glfwDestroyWindow(win);
    OPENVDB_LOG_DEBUG_RUNTIME("destroyed window " << std::hex << win << std::dec
        << " from thread " << std::this_thread::get_id());
}


////////////////////////////////////////


void
ViewerImpl::view(const openvdb::GridCPtrVec& gridList)
{
    if (!isOpen()) return;

    mGrids = gridList;
    mGridIdx = size_t(-1);
    mGridName.clear();

    // Compute the combined bounding box of all the grids.
    openvdb::BBoxd bbox(openvdb::Vec3d(0.0), openvdb::Vec3d(0.0));
    if (!gridList.empty()) {
        bbox = worldSpaceBBox(
            gridList[0]->transform(), gridList[0]->evalActiveVoxelBoundingBox());
        openvdb::Vec3d voxelSize = gridList[0]->voxelSize();

        for (size_t n = 1; n < gridList.size(); ++n) {
            bbox.expand(worldSpaceBBox(gridList[n]->transform(),
                gridList[n]->evalActiveVoxelBoundingBox()));

            voxelSize = minComponent(voxelSize, gridList[n]->voxelSize());
        }
        mClipBox->setStepSize(voxelSize);
    }
    mClipBox->setBBox(bbox);

    // Prepare window for rendering.
    glfwMakeContextCurrent(mWindow);

#if defined(_WIN32)
    // This must come after glfwMakeContextCurrent
    if (GLEW_OK != glewInit()) {
        OPENVDB_LOG_ERROR("GLEW initialization failed");
    }
#endif

    {
        // set up camera
        openvdb::Vec3d extents = bbox.extents();
        double maxExtent = std::max(extents[0], std::max(extents[1], extents[2]));
        mCamera->setTarget(bbox.getCenter(), maxExtent);
        mCamera->lookAtTarget();
        mCamera->setSpeed();
    }

    swapBuffers();
    setNeedsDisplay();


    //////////

    // Screen color
    glClearColor(0.85f, 0.85f, 0.85f, 0.0f);

    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    glPointSize(4);
    glLineWidth(2);
    //////////

    // construct render modules
    showNthGrid(/*n=*/0);


    // main loop

    size_t frame = 0;
    double time = glfwGetTime();

    glfwSwapInterval(1);

    OPENVDB_LOG_DEBUG_RUNTIME("starting to render in window " << std::hex << mWindow << std::dec
        << " from thread " << std::this_thread::get_id());

    mInterrupt = false;
    for (bool stop = false; !stop; ) {
        if (needsDisplay()) render();

        // eval fps
        ++frame;
        double elapsed = glfwGetTime() - time;
        if (elapsed > 1.0) {
            time = glfwGetTime();
            setWindowTitle(/*fps=*/double(frame) / elapsed);
            frame = 0;
        }

        // Swap front and back buffers
        swapBuffers();

        sleep(0.01/*sec*/);

        // Exit if the Esc key is pressed or the window is closed.
        handleEvents();
        stop = (mInterrupt || glfwWindowShouldClose(mWindow));
    }

    if (glfwGetCurrentContext() == mWindow) { ///< @todo not thread-safe
        // Detach this viewer's GL context.
        glfwMakeContextCurrent(nullptr);
        OPENVDB_LOG_DEBUG_RUNTIME("detached window " << std::hex << mWindow << std::dec
            << " from thread " << std::this_thread::get_id());
    }

    OPENVDB_LOG_DEBUG_RUNTIME("finished rendering in window " << std::hex << mWindow << std::dec
        << " from thread " << std::this_thread::get_id());
}


////////////////////////////////////////


void
ViewerImpl::resize(int width, int height)
{
    if (mWindow) glfwSetWindowSize(mWindow, width, height);
}


////////////////////////////////////////


void
ViewerImpl::render()
{
    if (mWindow == nullptr) return;

    // Prepare window for rendering.
    glfwMakeContextCurrent(mWindow);

    mCamera->aim();

    // draw scene
    mViewportModule->render(); // ground plane.

    mClipBox->render();
    mClipBox->enableClipping();

    for (size_t n = 0, N = mRenderModules.size(); n < N; ++n) {
        mRenderModules[n]->render();
    }

    mClipBox->disableClipping();

    // Render text

    if (mShowInfo) {
        BitmapFont13::enableFontRendering();

        glColor3d(0.2, 0.2, 0.2);

        int width, height;
        glfwGetFramebufferSize(mWindow, &width, &height);

        BitmapFont13::print(10, height - 13 - 10, mGridInfo);
        BitmapFont13::print(10, height - 13 - 30, mTransformInfo);
        BitmapFont13::print(10, height - 13 - 50, mTreeInfo);

        // Indicate via their hotkeys which render modules are enabled.
        std::string keys = "123";
        for (auto n: {0, 1, 2}) { if (!mRenderModules[n]->visible()) keys[n] = ' '; }
        BitmapFont13::print(width - 10 - 30, 10, keys);
        glColor3d(0.75, 0.75, 0.75);
        BitmapFont13::print(width - 10 - 30, 10, "123");

        BitmapFont13::disableFontRendering();
    }
}


////////////////////////////////////////


//static
void
ViewerImpl::sleep(double secs)
{
    secs = fabs(secs);
    int isecs = int(secs);
    std::this_thread::sleep_for(std::chrono::seconds(isecs));
}


////////////////////////////////////////


//static
openvdb::BBoxd
ViewerImpl::worldSpaceBBox(const openvdb::math::Transform& xform, const openvdb::CoordBBox& bbox)
{
    openvdb::Vec3d pMin = openvdb::Vec3d(std::numeric_limits<double>::max());
    openvdb::Vec3d pMax = -pMin;

    const openvdb::Coord& min = bbox.min();
    const openvdb::Coord& max = bbox.max();
    openvdb::Coord ijk;

    // corner 1
    openvdb::Vec3d ptn = xform.indexToWorld(min);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 2
    ijk[0] = min.x();
    ijk[1] = min.y();
    ijk[2] = max.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 3
    ijk[0] = max.x();
    ijk[1] = min.y();
    ijk[2] = max.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 4
    ijk[0] = max.x();
    ijk[1] = min.y();
    ijk[2] = min.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 5
    ijk[0] = min.x();
    ijk[1] = max.y();
    ijk[2] = min.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 6
    ijk[0] = min.x();
    ijk[1] = max.y();
    ijk[2] = max.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }


    // corner 7
    ptn = xform.indexToWorld(max);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    // corner 8
    ijk[0] = max.x();
    ijk[1] = max.y();
    ijk[2] = min.z();
    ptn = xform.indexToWorld(ijk);
    for (int i = 0; i < 3; ++i) {
        if (ptn[i] < pMin[i]) pMin[i] = ptn[i];
        if (ptn[i] > pMax[i]) pMax[i] = ptn[i];
    }

    return openvdb::BBoxd(pMin, pMax);
}


////////////////////////////////////////


void
ViewerImpl::updateCutPlanes(int wheelPos)
{
    double speed = std::abs(mWheelPos - wheelPos);
    if (mWheelPos < wheelPos) mClipBox->update(speed);
    else mClipBox->update(-speed);
    setNeedsDisplay();
}


////////////////////////////////////////


void
ViewerImpl::swapBuffers()
{
    glfwSwapBuffers(mWindow);
}


////////////////////////////////////////


void
ViewerImpl::setWindowTitle(double fps)
{
    std::ostringstream ss;
    ss  << mProgName << ": "
        << (mGridName.empty() ? std::string("OpenVDB") : mGridName)
        << " (" << (mGridIdx + 1) << " of " << mGrids.size() << ") @ "
        << std::setprecision(1) << std::fixed << fps << " fps";
    if (mWindow) glfwSetWindowTitle(mWindow, ss.str().c_str());
}


////////////////////////////////////////


void
ViewerImpl::showPrevGrid()
{
    if (const size_t numGrids = mGrids.size()) {
        size_t idx = ((numGrids + mGridIdx) - 1) % numGrids;
        showNthGrid(idx);
    }
}


void
ViewerImpl::showNextGrid()
{
    if (const size_t numGrids = mGrids.size()) {
        size_t idx = (mGridIdx + 1) % numGrids;
        showNthGrid(idx);
    }
}


void
ViewerImpl::showNthGrid(size_t n)
{
    if (mGrids.empty()) return;
    n = n % mGrids.size();
    if (n == mGridIdx) return;

    mGridName = mGrids[n]->getName();
    mGridIdx = n;

    // save render settings
    std::vector<bool> active(mRenderModules.size());
    for (size_t i = 0, I = active.size(); i < I; ++i) {
        active[i] = mRenderModules[i]->visible();
    }

    mRenderModules.clear();
    mRenderModules.push_back(RenderModulePtr(new TreeTopologyModule(mGrids[n])));
    mRenderModules.push_back(RenderModulePtr(new MeshModule(mGrids[n])));
    mRenderModules.push_back(RenderModulePtr(new VoxelModule(mGrids[n])));

    if (active.empty()) {
        for (size_t i = 1, I = mRenderModules.size(); i < I; ++i) {
            mRenderModules[i]->setVisible(false);
        }
    } else {
        for (size_t i = 0, I = active.size(); i < I; ++i) {
            mRenderModules[i]->setVisible(active[i]);
        }
    }

    // Collect info
    {
        std::ostringstream ostrm;
        std::string s = mGrids[n]->getName();
        const openvdb::GridClass cls = mGrids[n]->getGridClass();
        if (!s.empty()) ostrm << s << " / ";
        ostrm << mGrids[n]->valueType() << " / ";
        if (cls == openvdb::GRID_UNKNOWN) ostrm << " class unknown";
        else ostrm << " " << openvdb::GridBase::gridClassToString(cls);
        mGridInfo = ostrm.str();
    }
    {
        openvdb::Coord dim = mGrids[n]->evalActiveVoxelDim();
        std::ostringstream ostrm;
        ostrm << dim[0] << " x " << dim[1] << " x " << dim[2]
            << " / voxel size " << std::setprecision(4) << mGrids[n]->voxelSize()[0]
            << " (" << mGrids[n]->transform().mapType() << ")";
        mTransformInfo = ostrm.str();
    }
    {
        std::ostringstream ostrm;
        const openvdb::Index64 count = mGrids[n]->activeVoxelCount();
        ostrm << openvdb::util::formattedInt(count)
            << " active voxel" << (count == 1 ? "" : "s");
        mTreeInfo = ostrm.str();
    }
    {
        if (mGrids[n]->isType<openvdb::points::PointDataGrid>()) {
            const openvdb::points::PointDataGrid::ConstPtr points =
                openvdb::gridConstPtrCast<openvdb::points::PointDataGrid>(mGrids[n]);
            const openvdb::Index64 count = openvdb::points::pointCount(points->tree());
            std::ostringstream ostrm;
            ostrm << " / " << openvdb::util::formattedInt(count)
                 << " point" << (count == 1 ? "" : "s");
            mTreeInfo.append(ostrm.str());
        }
    }

    setWindowTitle();
}


////////////////////////////////////////


void
ViewerImpl::keyCallback(int key, int action)
{
    mCamera->keyCallback(key, action);

    if (mWindow == nullptr) return;
    const bool keyPress = (glfwGetKey(mWindow, key) == GLFW_PRESS);
    /// @todo Should use "modifiers" argument to keyCB().
    mShiftIsDown = glfwGetKey(mWindow, GLFW_KEY_LEFT_SHIFT);
    mCtrlIsDown = glfwGetKey(mWindow, GLFW_KEY_LEFT_CONTROL);

    if (keyPress) {
        switch (key) {
        case '1': case GLFW_KEY_KP_1:
            toggleRenderModule(0);
            break;
        case '2': case GLFW_KEY_KP_2:
            toggleRenderModule(1);
            break;
        case '3': case GLFW_KEY_KP_3:
            toggleRenderModule(2);
            break;
        case 'c': case 'C':
            mClipBox->reset();
            break;
        case 'h': case 'H': // center home
            mCamera->lookAt(openvdb::Vec3d(0.0), 10.0);
            break;
        case 'g': case 'G': // center geometry
            mCamera->lookAtTarget();
            break;
        case 'i': case 'I':
            toggleInfoText();
            break;
        case GLFW_KEY_LEFT:
            showPrevGrid();
            break;
        case GLFW_KEY_RIGHT:
            showNextGrid();
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(mWindow, true);
            break;
        }
    }

    switch (key) {
    case 'x': case 'X':
        mClipBox->activateXPlanes() = keyPress;
        break;
    case 'y': case 'Y':
        mClipBox->activateYPlanes() = keyPress;
        break;
    case 'z': case 'Z':
        mClipBox->activateZPlanes() = keyPress;
        break;
    }

    mClipBox->shiftIsDown() = mShiftIsDown;
    mClipBox->ctrlIsDown() = mCtrlIsDown;

    setNeedsDisplay();
}


void
ViewerImpl::mouseButtonCallback(int button, int action)
{
    mCamera->mouseButtonCallback(button, action);
    mClipBox->mouseButtonCallback(button, action);
    if (mCamera->needsDisplay()) setNeedsDisplay();
}


void
ViewerImpl::mousePosCallback(int x, int y)
{
    bool handled = mClipBox->mousePosCallback(x, y);
    if (!handled) mCamera->mousePosCallback(x, y);
    if (mCamera->needsDisplay()) setNeedsDisplay();
}


void
ViewerImpl::mouseWheelCallback(int pos)
{
    pos += mWheelPos;
    if (mClipBox->isActive()) {
        updateCutPlanes(pos);
    } else {
        mCamera->mouseWheelCallback(pos, mWheelPos);
        if (mCamera->needsDisplay()) setNeedsDisplay();
    }

    mWheelPos = pos;
}


void
ViewerImpl::windowSizeCallback(int, int)
{
    setNeedsDisplay();
}


void
ViewerImpl::windowRefreshCallback()
{
    setNeedsDisplay();
}


////////////////////////////////////////


bool
ViewerImpl::needsDisplay()
{
    if (mUpdates < 2) {
        mUpdates += 1;
        return true;
    }
    return false;
}


void
ViewerImpl::setNeedsDisplay()
{
    mUpdates = 0;
}


void
ViewerImpl::toggleRenderModule(size_t n)
{
    mRenderModules[n]->setVisible(!mRenderModules[n]->visible());
}


void
ViewerImpl::toggleInfoText()
{
    mShowInfo = !mShowInfo;
}

} // namespace openvdb_viewer
