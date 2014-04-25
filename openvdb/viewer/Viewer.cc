///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include "Viewer.h"

#include "Camera.h"
#include "ClipBox.h"
#include "Font.h"
#include "RenderModules.h"
#include <openvdb/util/Formats.h> // for formattedInt()
#include <tbb/mutex.h>
#include <iomanip> // for std::setprecision()
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <GLFW/glfw3.h>


namespace openvdb_viewer {

class ViewerImpl
{
public:
    typedef boost::shared_ptr<Camera> CameraPtr;
    typedef boost::shared_ptr<ClipBox> ClipBoxPtr;
    typedef boost::shared_ptr<RenderModule> RenderModulePtr;

    ViewerImpl();

    void init(const std::string& progName, bool verbose = false);

    void view(const openvdb::GridCPtrVec&, int width = 900, int height = 800);

    void showPrevGrid();
    void showNextGrid();

    bool needsDisplay();
    void setNeedsDisplay();

    void toggleRenderModule(size_t n);
    void toggleInfoText();

    void setWindowTitle(GLFWwindow* window, double fps = 0.0);
    void viewGrids(const openvdb::GridCPtrVec&, int width, int height);
    void render(int width, int height);
    void showNthGrid(size_t n);
    void updateCutPlanes(double wheelPos);

    void keyCallback(GLFWwindow* window, int key, int scancode, int action,
        int mods);
    void mouseButtonCallback(GLFWwindow* window, int button, int action,
            int mods);
    void mousePosCallback(GLFWwindow* window, double x, double y);
    void mouseWheelCallback(GLFWwindow* window, double x, double y);
    void windowSizeCallback(GLFWwindow* window, int width, int height);
    void windowRefreshCallback(GLFWwindow* window);

private:
    CameraPtr mCamera;
    ClipBoxPtr mClipBox;

    std::vector<RenderModulePtr> mRenderModules;
    openvdb::GridCPtrVec mGrids;

    size_t mGridIdx, mUpdates;
    std::string mGridName, mProgName, mGridInfo, mTransformInfo, mTreeInfo;
    bool mShiftIsDown, mCtrlIsDown, mShowInfo;
}; // class ViewerImpl


////////////////////////////////////////


namespace {

ViewerImpl* sViewer = NULL;
tbb::mutex sLock;


void
keyCB(GLFWwindow* window, int key, int scancode, int action,
        int mods)
{
    if (sViewer) sViewer->keyCallback(window, key, scancode, action, mods);
}


void
mouseButtonCB(GLFWwindow* window, int button, int action, int mods)
{
    if (sViewer) sViewer->mouseButtonCallback(window, button, action, mods);
}


void
mousePosCB(GLFWwindow* window, double x, double y)
{
    if (sViewer) sViewer->mousePosCallback(window, x, y);
}


void
mouseWheelCB(GLFWwindow* window, double x, double y)
{
    if (sViewer) sViewer->mouseWheelCallback(window, x, y);
}


void
windowSizeCB(GLFWwindow* window, int width, int height)
{
    if (sViewer) sViewer->windowSizeCallback(window, width, height);
}


void
windowRefreshCB(GLFWwindow* window)
{
    if (sViewer) sViewer->windowRefreshCallback(window);
}

} // unnamed namespace


////////////////////////////////////////


Viewer
init(const std::string& progName, bool verbose)
{
    if (sViewer == NULL) {
        tbb::mutex::scoped_lock(sLock);
        if (sViewer == NULL) {
            OPENVDB_START_THREADSAFE_STATIC_WRITE
            sViewer = new ViewerImpl;
            OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
        }
    }
    sViewer->init(progName, verbose);

    return Viewer();
}


////////////////////////////////////////


Viewer::Viewer()
{
}


void
Viewer::view(const openvdb::GridCPtrVec& grids, int width, int height)
{
    if (sViewer) sViewer->view(grids, width, height);
}


void
Viewer::showPrevGrid()
{
    if (sViewer) sViewer->showPrevGrid();
}


void
Viewer::showNextGrid()
{
    if (sViewer) sViewer->showNextGrid();
}


////////////////////////////////////////


ViewerImpl::ViewerImpl()
    : mCamera(new Camera)
    , mClipBox(new ClipBox)
    , mRenderModules(0)
    , mGridIdx(0)
    , mUpdates(0)
    , mShiftIsDown(false)
    , mCtrlIsDown(false)
    , mShowInfo(true)
{
}


void
ViewerImpl::init(const std::string& progName, bool verbose)
{
    mProgName = progName;

    if (glfwInit() != GL_TRUE) {
        OPENVDB_LOG_ERROR("GLFW Initialization Failed.");
    }

    if (verbose) {
        GLFWwindow* window;
        glfwWindowHint(GLFW_RED_BITS, 8);
        glfwWindowHint(GLFW_GREEN_BITS, 8);
        glfwWindowHint(GLFW_BLUE_BITS, 8);
        glfwWindowHint(GLFW_ALPHA_BITS, 8);
        glfwWindowHint(GLFW_DEPTH_BITS, 24);
        glfwWindowHint(GLFW_STENCIL_BITS, 0);
        if (window = glfwCreateWindow(100, 100, "", NULL, NULL)) {
            glfwMakeContextCurrent(window);
            int major, minor, rev;
            glfwGetVersion(&major, &minor, &rev);
            std::cout << "GLFW: " << major << "." << minor << "." << rev << "\n"
                << "OpenGL: " << glGetString(GL_VERSION) << std::endl;
            glfwDestroyWindow(window);
        }
    }
}


////////////////////////////////////////


void
ViewerImpl::setWindowTitle(GLFWwindow* window, double fps)
{
    std::ostringstream ss;
    ss  << mProgName << ": "
        << (mGridName.empty() ? std::string("OpenVDB") : mGridName)
        << " (" << (mGridIdx + 1) << " of " << mGrids.size() << ") @ "
        << std::setprecision(1) << std::fixed << fps << " fps";
    glfwSetWindowTitle(window, ss.str().c_str());
}


////////////////////////////////////////


void
ViewerImpl::render(int width, int height)
{
    mCamera->aim(width, height);

    // draw scene
    mRenderModules[0]->render(); // ground plane.

    mClipBox->render();
    mClipBox->enableClipping();

    for (size_t n = 1, N = mRenderModules.size(); n < N; ++n) {
        mRenderModules[n]->render();
    }

    mClipBox->disableClipping();

    // Render text

    if (mShowInfo) {
        BitmapFont13::enableFontRendering(width, height);

        glColor3f (0.2, 0.2, 0.2);

        BitmapFont13::print(10, height - 13 - 10, mGridInfo);
        BitmapFont13::print(10, height - 13 - 30, mTransformInfo);
        BitmapFont13::print(10, height - 13 - 50, mTreeInfo);

        BitmapFont13::disableFontRendering();
    }
}


////////////////////////////////////////


void
ViewerImpl::view(const openvdb::GridCPtrVec& gridList, int width, int height)
{
    viewGrids(gridList, width, height);
}


openvdb::BBoxd
worldSpaceBBox(const openvdb::math::Transform& xform, const openvdb::CoordBBox& bbox)
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


void
ViewerImpl::viewGrids(const openvdb::GridCPtrVec& gridList, int width, int height)
{
    mGrids = gridList;
    mGridIdx = size_t(-1);
    mGridName.clear();

    // Create window
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 32);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);
    boost::shared_ptr<GLFWwindow> window(glfwCreateWindow(
                width, height,      // Window size
                mProgName.c_str(),  // Window title
                NULL, NULL),
            glfwDestroyWindow);
    if (window.get() == NULL) {
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window.get());
    glfwSwapBuffers(window.get());

    BitmapFont13::initialize();

    //////////

    // Eval grid bbox

    openvdb::BBoxd bbox(openvdb::Vec3d(0.0), openvdb::Vec3d(0.0));

    if (!gridList.empty()) {
        bbox = worldSpaceBBox(gridList[0]->transform(), gridList[0]->evalActiveVoxelBoundingBox());
        openvdb::Vec3d voxelSize = gridList[0]->voxelSize();

        for (size_t n = 1; n < gridList.size(); ++n) {
            bbox.expand(worldSpaceBBox(gridList[n]->transform(),
                gridList[n]->evalActiveVoxelBoundingBox()));

            voxelSize = minComponent(voxelSize, gridList[n]->voxelSize());
        }

        mClipBox->setStepSize(voxelSize);
    }

    mClipBox->setBBox(bbox);


    // setup camera

    openvdb::Vec3d extents = bbox.extents();
    double max_extent = std::max(extents[0], std::max(extents[1], extents[2]));

    mCamera->setTarget(bbox.getCenter(), max_extent);
    mCamera->lookAtTarget();
    mCamera->setSpeed(/*zoom=*/0.1, /*strafe=*/0.002, /*tumbling=*/0.02);

    //////////

    // register callback functions

    glfwSetKeyCallback(window.get(), keyCB);
    glfwSetMouseButtonCallback(window.get(), mouseButtonCB);
    glfwSetCursorPosCallback(window.get(), mousePosCB);
    glfwSetScrollCallback(window.get(), mouseWheelCB);
    glfwSetWindowSizeCallback(window.get(), windowSizeCB);
    glfwSetWindowRefreshCallback(window.get(), windowRefreshCB);


    //////////

    // Screen color
    glClearColor(0.85, 0.85, 0.85, 0.0f);

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

    do {
        if (needsDisplay()) {
            glfwGetWindowSize(window.get(), &width, &height);
            render(width, height);
        }

        // eval fps
        ++frame;
        double elapsed = glfwGetTime() - time;
        if (elapsed > 1.0) {
            time = glfwGetTime();
            setWindowTitle(window.get(),
                    /*fps=*/double(frame) / elapsed);
            frame = 0;
        }

        // Swap front and back buffers
        glfwSwapBuffers(window.get());
        glfwPollEvents();

    // exit if the esc key is pressed or the window is closed.
    } while (!glfwWindowShouldClose(window.get()));

    glfwTerminate();
}


////////////////////////////////////////


void
ViewerImpl::updateCutPlanes(double wheelPosY)
{
    mClipBox->update(-wheelPosY);
    setNeedsDisplay();
}


////////////////////////////////////////


void
ViewerImpl::showPrevGrid()
{
    const size_t numGrids = mGrids.size();
    size_t idx = ((numGrids + mGridIdx) - 1) % numGrids;
    showNthGrid(idx);
}


void
ViewerImpl::showNextGrid()
{
    const size_t numGrids = mGrids.size();
    size_t idx = (mGridIdx + 1) % numGrids;
    showNthGrid(idx);
}


void
ViewerImpl::showNthGrid(size_t n)
{
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
    mRenderModules.push_back(RenderModulePtr(new ViewportModule));
    mRenderModules.push_back(RenderModulePtr(new TreeTopologyModule(mGrids[n])));
    mRenderModules.push_back(RenderModulePtr(new MeshModule(mGrids[n])));
    mRenderModules.push_back(RenderModulePtr(new ActiveValueModule(mGrids[n])));

    if (active.empty()) {
        for (size_t i = 2, I = mRenderModules.size(); i < I; ++i) {
            mRenderModules[i]->visible() = false;
        }
    } else {
        for (size_t i = 0, I = active.size(); i < I; ++i) {
            mRenderModules[i]->visible() = active[i];
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
}


////////////////////////////////////////


void
ViewerImpl::keyCallback(GLFWwindow* window, int key, int scancode, int action,
        int mods)
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE

    mCamera->keyCallback(window, key, scancode, action, mods);
    const bool keyPress = (action == GLFW_PRESS)
        || (action == GLFW_REPEAT);
    mShiftIsDown = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
    mCtrlIsDown = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;

    if (keyPress) {
        switch (key) {
        case GLFW_KEY_1:
            toggleRenderModule(1);
            break;
        case GLFW_KEY_2:
            toggleRenderModule(2);
            break;
        case GLFW_KEY_3:
            toggleRenderModule(3);
            break;
        case GLFW_KEY_C:
            mClipBox->reset();
            break;
        case GLFW_KEY_H: // center home
            mCamera->lookAt(openvdb::Vec3d(0.0), 10.0);
            break;
        case GLFW_KEY_G: // center geometry
            mCamera->lookAtTarget();
            break;
        case GLFW_KEY_I:
            toggleInfoText();
            break;
        case GLFW_KEY_LEFT:
            showPrevGrid();
            break;
        case GLFW_KEY_RIGHT:
            showNextGrid();
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        }

    }

    switch (key) {
        case GLFW_KEY_LEFT_SHIFT:
            mShiftIsDown = keyPress;
            break;
        case GLFW_KEY_LEFT_CONTROL:
            mCtrlIsDown = keyPress;
            break;
        case GLFW_KEY_X:
            mClipBox->activateXPlanes() = keyPress;
            break;
        case GLFW_KEY_Y:
            mClipBox->activateYPlanes() = keyPress;
            break;
        case GLFW_KEY_Z:
            mClipBox->activateZPlanes() = keyPress;
            break;
    }

    mClipBox->shiftIsDown() = mShiftIsDown;
    mClipBox->ctrlIsDown() = mCtrlIsDown;

    setNeedsDisplay();

    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
}


void
ViewerImpl::mouseButtonCallback(GLFWwindow* window, int button, int action,
        int mods)
{
    mCamera->mouseButtonCallback(window, button, action, mods);
    mClipBox->mouseButtonCallback(window, button, action, mods);
    if (mCamera->needsDisplay()) setNeedsDisplay();
}


void
ViewerImpl::mousePosCallback(GLFWwindow* window, double x, double y)
{
    bool handled = mClipBox->mousePosCallback(window, x, y);
    if (!handled) mCamera->mousePosCallback(window, x, y);
    if (mCamera->needsDisplay()) setNeedsDisplay();
}


void
ViewerImpl::mouseWheelCallback(GLFWwindow* window, double x, double y)
{
    if (mClipBox->isActive()) {
        updateCutPlanes(y);
    } else {
        mCamera->mouseWheelCallback(window, x, y);
        if (mCamera->needsDisplay()) setNeedsDisplay();
    }
}


void
ViewerImpl::windowSizeCallback(GLFWwindow*, int, int)
{
    setNeedsDisplay();
}


void
ViewerImpl::windowRefreshCallback(GLFWwindow* window)
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
    mRenderModules[n]->visible() = !mRenderModules[n]->visible();
}


void
ViewerImpl::toggleInfoText()
{
    mShowInfo = !mShowInfo;
}

} // namespace openvdb_viewer

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
