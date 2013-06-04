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

#include <GL/glfw.h>


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

    void setWindowTitle(double fps = 0.0);
    void viewGrids(const openvdb::GridCPtrVec&, int width, int height);
    void render();
    void showNthGrid(size_t n);
    void updateCutPlanes(int wheelPos);

    void keyCallback(int key, int action);
    void mouseButtonCallback(int button, int action);
    void mousePosCallback(int x, int y);
    void mouseWheelCallback(int pos);
    void windowSizeCallback(int width, int height);
    void windowRefreshCallback();

private:
    CameraPtr mCamera;
    ClipBoxPtr mClipBox;

    std::vector<RenderModulePtr> mRenderModules;
    openvdb::GridCPtrVec mGrids;

    size_t mGridIdx, mUpdates;
    std::string mGridName, mProgName, mGridInfo, mTransformInfo, mTreeInfo;
    int mWheelPos;
    bool mShiftIsDown, mCtrlIsDown, mShowInfo;
}; // class ViewerImpl


////////////////////////////////////////


namespace {

ViewerImpl* sViewer = NULL;
tbb::mutex sLock;


void
keyCB(int key, int action)
{
    if (sViewer) sViewer->keyCallback(key, action);
}


void
mouseButtonCB(int button, int action)
{
    if (sViewer) sViewer->mouseButtonCallback(button, action);
}


void
mousePosCB(int x, int y)
{
    if (sViewer) sViewer->mousePosCallback(x, y);
}


void
mouseWheelCB(int pos)
{
    if (sViewer) sViewer->mouseWheelCallback(pos);
}


void
windowSizeCB(int width, int height)
{
    if (sViewer) sViewer->windowSizeCallback(width, height);
}


void
windowRefreshCB()
{
    if (sViewer) sViewer->windowRefreshCallback();
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
    , mWheelPos(0)
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
        if (glfwOpenWindow(100, 100, 8, 8, 8, 8, 24, 0, GLFW_WINDOW)) {
            int major, minor, rev;
            glfwGetVersion(&major, &minor, &rev);
            std::cout << "GLFW: " << major << "." << minor << "." << rev << "\n"
                << "OpenGL: " << glGetString(GL_VERSION) << std::endl;
            glfwCloseWindow();
        }
    }
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
    glfwSetWindowTitle(ss.str().c_str());
}


////////////////////////////////////////


void
ViewerImpl::render()
{
    mCamera->aim();

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
        BitmapFont13::enableFontRendering();

        glColor3f (0.2, 0.2, 0.2);

        int width, height;
        glfwGetWindowSize(&width, &height);

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
    if (!glfwOpenWindow(width, height,  // Window size
                       8, 8, 8, 8,      // # of R,G,B, & A bits
                       32, 0,           // # of depth & stencil buffer bits
                       GLFW_WINDOW))    // Window mode
    {
        glfwTerminate();
        return;
    }

    glfwSetWindowTitle(mProgName.c_str());
    glfwSwapBuffers();

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

    glfwSetKeyCallback(keyCB);
    glfwSetMouseButtonCallback(mouseButtonCB);
    glfwSetMousePosCallback(mousePosCB);
    glfwSetMouseWheelCallback(mouseWheelCB);
    glfwSetWindowSizeCallback(windowSizeCB);
    glfwSetWindowRefreshCallback(windowRefreshCB);


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
        glfwSwapBuffers();

    // exit if the esc key is pressed or the window is closed.
    } while (!glfwGetKey(GLFW_KEY_ESC) && glfwGetWindowParam(GLFW_OPENED));

    glfwTerminate();
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

    setWindowTitle();
}


////////////////////////////////////////


void
ViewerImpl::keyCallback(int key, int action)
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE

    mCamera->keyCallback(key, action);
    const bool keyPress = glfwGetKey(key) == GLFW_PRESS;
    mShiftIsDown = glfwGetKey(GLFW_KEY_LSHIFT);
    mCtrlIsDown = glfwGetKey(GLFW_KEY_LCTRL);

    if (keyPress) {
        switch (key) {
        case '1':
            toggleRenderModule(1);
            break;
        case '2':
            toggleRenderModule(2);
            break;
        case '3':
            toggleRenderModule(3);
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

    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
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
