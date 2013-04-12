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
#include <iomanip> // for std::setprecision()
#include <iostream>
#include <sstream>
#include <math.h>
#include <limits>


////////////////////////////////////////

// Basic camera class

struct Viewer::Camera
{
    Camera();

    void aim();

    void lookAt(const openvdb::Vec3d& p, double dist = 1.0);
    void lookAtTarget();

    void setTarget(const openvdb::Vec3d& p, double dist = 1.0);

    void setNearFarPlanes(double n, double f) { mNearPlane = n; mFarPlane = f; }
    void setFieldOfView(double degrees) { mFov = degrees; }
    void setSpeed(double zoomSpeed, double strafeSpeed, double tumblingSpeed);

    void keyCallback(int key, int action);
    void mouseButtonCallback(int button, int action);
    void mousePosCallback(int x, int y);
    void mouseWheelCallback(int pos);

    bool needsDisplay() const { return mNeedsDisplay; }

private:
    // Camera parameters
    double mFov, mNearPlane, mFarPlane;
    openvdb::Vec3d mTarget, mLookAt, mUp, mForward, mRight, mEye;
    double mTumblingSpeed, mZoomSpeed, mStrafeSpeed;
    double mHead, mPitch, mTargetDistance, mDistance;

    // Input states
    bool mMouseDown, mStartTumbling, mZoomMode, mChanged, mNeedsDisplay;
    double mMouseXPos, mMouseYPos;
    int mWheelPos;

    static double sDeg2rad;
};

double Viewer::Camera::sDeg2rad = M_PI / 180.0;

Viewer::Camera::Camera()
    : mFov(65.0)
    , mNearPlane(0.1)
    , mFarPlane(10000.0)
    , mTarget(openvdb::Vec3d(0.0))
    , mLookAt(mTarget)
    , mUp(openvdb::Vec3d(0.0, 1.0, 0.0))
    , mForward(openvdb::Vec3d(0.0, 0.0, 1.0))
    , mRight(openvdb::Vec3d(1.0, 0.0, 0.0))
    , mEye(openvdb::Vec3d(0.0, 0.0, -1.0))
    , mTumblingSpeed(0.5)
    , mZoomSpeed(0.2)
    , mStrafeSpeed(0.05)
    , mHead(30.0)
    , mPitch(45.0)
    , mTargetDistance(25.0)
    , mDistance(mTargetDistance)
    , mMouseDown(false)
    , mStartTumbling(false)
    , mZoomMode(false)
    , mChanged(true)
    , mNeedsDisplay(true)
    , mMouseXPos(0.0)
    , mMouseYPos(0.0)
    , mWheelPos(0)
{
}

void
Viewer::Camera::lookAt(const openvdb::Vec3d& p, double dist)
{
    mLookAt = p;
    mDistance = dist;
    mNeedsDisplay = true;
}

void
Viewer::Camera::lookAtTarget()
{
    mLookAt = mTarget;
    mDistance = mTargetDistance;
    mNeedsDisplay = true;
}


void
Viewer::Camera::setSpeed(double zoomSpeed, double strafeSpeed, double tumblingSpeed)
{
    mZoomSpeed = std::max(0.0001, zoomSpeed);
    mStrafeSpeed = std::max(0.0001, strafeSpeed);
    mTumblingSpeed = std::max(0.2, tumblingSpeed);
    mTumblingSpeed = std::min(1.0, tumblingSpeed);
}

void
Viewer::Camera::setTarget(const openvdb::Vec3d& p, double dist)
{
    mTarget = p;
    mTargetDistance = dist;
}

void
Viewer::Camera::aim()
{
    // Get the window size
    int width, height;
    glfwGetWindowSize(&width, &height);

    // Make sure that height is non-zero to avoid division by zero
    height = height < 1 ? 1 : height;

    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Window aspect (assumes square pixels)
    double aspectRatio = (double)width / (double)height;

    // Set perspective view (fov is in degrees in the y direction.)
    gluPerspective(mFov, aspectRatio, mNearPlane, mFarPlane);

    if (mChanged) {

        mChanged = false;

        mEye[0] = mLookAt[0] + mDistance * std::cos(mHead * sDeg2rad) * std::cos(mPitch * sDeg2rad);
        mEye[1] = mLookAt[1] + mDistance * std::sin(mHead * sDeg2rad);
        mEye[2] = mLookAt[2] + mDistance * std::cos(mHead * sDeg2rad) * std::sin(mPitch * sDeg2rad);

        mForward = mLookAt - mEye;
        mForward.normalize();

        mUp[1] = std::cos(mHead * sDeg2rad) > 0 ? 1.0 : -1.0;
        mRight = mForward.cross(mUp);
    }

    // Set up modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(mEye[0], mEye[1], mEye[2],
              mLookAt[0], mLookAt[1], mLookAt[2],
              mUp[0], mUp[1], mUp[2]);

    mNeedsDisplay = false;
}


void
Viewer::Camera::keyCallback(int key, int )
{
    if (glfwGetKey(key) == GLFW_PRESS) {
        switch(key) {
            case GLFW_KEY_SPACE:
                mZoomMode = true;
                break;
        }
    } else if (glfwGetKey(key) == GLFW_RELEASE) {
        switch(key) {
            case GLFW_KEY_SPACE:
                mZoomMode = false;
                break;
        }
    }

    mChanged = true;
}


void
Viewer::Camera::mouseButtonCallback(int button, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) mMouseDown = true;
        else if (action == GLFW_RELEASE) mMouseDown = false;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            mMouseDown = true;
            mZoomMode = true;
        } else if (action == GLFW_RELEASE) {
            mMouseDown = false;
            mZoomMode = false;
        }
    }
    if (action == GLFW_RELEASE) mMouseDown = false;

    mStartTumbling = true;
    mChanged = true;
}


void
Viewer::Camera::mousePosCallback(int x, int y)
{
    if (mStartTumbling) {
        mMouseXPos = x;
        mMouseYPos = y;
        mStartTumbling = false;
    }

    double dx, dy;
    dx = x - mMouseXPos;
    dy = y - mMouseYPos;

    if (mMouseDown && !mZoomMode) {
        mNeedsDisplay = true;
        mHead += dy * mTumblingSpeed;
        mPitch += dx * mTumblingSpeed;
    } else if (mMouseDown && mZoomMode) {
        mNeedsDisplay = true;
        mLookAt += (dy * mUp - dx * mRight) * mStrafeSpeed;
    }

    mMouseXPos = x;
    mMouseYPos = y;
    mChanged = true;
}


void
Viewer::Camera::mouseWheelCallback(int pos)
{
    double speed = std::abs(mWheelPos - pos);

    if (mWheelPos < pos) {
        mDistance += speed * mZoomSpeed;
        setSpeed(mDistance * 0.1, mDistance * 0.002, mDistance * 0.02);
    } else {
        double temp = mDistance - speed * mZoomSpeed;
        mDistance = std::max(0.0, temp);
        setSpeed(mDistance * 0.1, mDistance * 0.002, mDistance * 0.02);
    }


    mWheelPos = pos;
    mChanged = true;
    mNeedsDisplay = true;
}



////////////////////////////////////////


Viewer::Viewer()
    : mCamera(new Camera)
    , mFirstRenderModule(1)
    , mSecondRenderModule(0)
    , mUpdates(0)
    , mGridIdx(0)
{
}


Viewer* Viewer::sViewer = NULL;
tbb::mutex sLock;


////////////////////////////////////////


Viewer&
Viewer::init(const std::string& progName, bool verbose)
{
    tbb::mutex::scoped_lock(sLock);

    if (sViewer == NULL) {
        OPENVDB_START_THREADSAFE_STATIC_WRITE
        sViewer = new Viewer;
        OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
    }

    sViewer->mProgName = progName;

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
    return *sViewer;
}


////////////////////////////////////////


void
Viewer::setWindowTitle(double fps)
{
    std::ostringstream ss;
    ss  << mProgName << ": "
        << (mGridName.empty() ? std::string("OpenVDB") : mGridName)
        << " @ " << std::setprecision(1) << std::fixed << fps << " fps";
    glfwSetWindowTitle(ss.str().c_str());
}


////////////////////////////////////////


void
Viewer::render()
{
    mCamera->aim();

    // draw scene
    mRenderModules[0]->render(); // ground plane.
    mRenderModules[mFirstRenderModule]->render();
    if (mSecondRenderModule != 0) mRenderModules[mSecondRenderModule]->render();
}


////////////////////////////////////////


void
Viewer::view(const openvdb::GridCPtrVec& gridList, int width, int height)
{
    sViewer->viewGrids(gridList, width, height);
}


void
Viewer::viewGrids(const openvdb::GridCPtrVec& gridList, int width, int height)
{
    mGrids = gridList;
    mGridIdx = size_t(-1);
    mGridName.clear();


    //////////

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


    //////////

    // eval scene bbox.

    openvdb::Vec3d min(std::numeric_limits<double>::max()), max(-min);
    for (size_t n = 0; n < gridList.size(); ++n) {
        openvdb::CoordBBox bbox = gridList[n]->evalActiveVoxelBoundingBox();
        min = openvdb::math::minComponent(min, gridList[n]->indexToWorld(bbox.min()));
        max = openvdb::math::maxComponent(max, gridList[n]->indexToWorld(bbox.max()));
    }

    double dim = std::abs(max[0] - min[0]);
    dim = std::max(dim, std::abs(max[1] - min[1]));
    dim = std::max(dim, std::abs(max[2] - min[2]));

    openvdb::Vec3d center(0.5 * (min + max));

    //////////

    // setup camera

    mCamera->setTarget(center, dim);
    mCamera->lookAtTarget();
    mCamera->setSpeed(/*zoom=*/0.1, /*strafe=*/0.002, /*tumbling=*/0.02);


    //////////

    // register callback functions

    glfwSetKeyCallback(Viewer::keyCallback);
    glfwSetMouseButtonCallback(Viewer::mouseButtonCallback);
    glfwSetMousePosCallback(Viewer::mousePosCallback);
    glfwSetMouseWheelCallback(Viewer::mouseWheelCallback);
    glfwSetWindowSizeCallback(Viewer::windowSizeCallback);


    //////////

    // Screen color
    glClearColor(0.85, 0.85, 0.85, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glShadeModel(GL_SMOOTH);


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
Viewer::showPrevGrid()
{
    const size_t numGrids = mGrids.size();
    size_t idx = ((numGrids + mGridIdx) - 1) % numGrids;
    showNthGrid(idx);
}


void
Viewer::showNextGrid()
{
    const size_t numGrids = mGrids.size();
    size_t idx = (mGridIdx + 1) % numGrids;
    showNthGrid(idx);
}


void
Viewer::showNthGrid(size_t n)
{
    n = n % mGrids.size();
    if (n == mGridIdx) return;

    mGridName = mGrids[n]->getName();
    mGridIdx = n;

    openvdb::GridCPtrVec curGrids;
    curGrids.push_back(mGrids[n]);

    mRenderModules.clear();
    mRenderModules.push_back(RenderModulePtr(new ViewportModule));
    mRenderModules.push_back(RenderModulePtr(new TreeTopologyModule(curGrids)));
    mRenderModules.push_back(RenderModulePtr(new MeshModule(curGrids)));

    setWindowTitle();
}


////////////////////////////////////////


void
Viewer::keyCallback(int key, int action)
{
    OPENVDB_START_THREADSAFE_STATIC_WRITE

    sViewer->mCamera->keyCallback(key, action);
    if (glfwGetKey(key) == GLFW_PRESS) {
        switch (key) {

            case '1':
                sViewer->mFirstRenderModule = 1;
                sViewer->mSecondRenderModule = 0;
                break;
            case '2':
                sViewer->mFirstRenderModule = 2;
                sViewer->mSecondRenderModule = 0;
                break;
            case '3':
                sViewer->mFirstRenderModule = 1;
                sViewer->mSecondRenderModule = 2;
                break;

            case 'g': case 'H': // center home
                sViewer->mCamera->lookAt(openvdb::Vec3d(0.0), 10.0);
                break;
            case 'h': case 'G': // center geometry
                sViewer->mCamera->lookAtTarget();
                break;

            case GLFW_KEY_LEFT:
                sViewer->showPrevGrid();
                break;
            case GLFW_KEY_RIGHT:
                sViewer->showNextGrid();
                break;
        }
    }

    sViewer->setNeedsDisplay();

    OPENVDB_FINISH_THREADSAFE_STATIC_WRITE
}

void
Viewer::mouseButtonCallback(int button, int action)
{
    sViewer->mCamera->mouseButtonCallback(button, action);
    if (sViewer->mCamera->needsDisplay()) sViewer->setNeedsDisplay();
}

void
Viewer::mousePosCallback(int x, int y)
{
    sViewer->mCamera->mousePosCallback(x, y);
    if (sViewer->mCamera->needsDisplay()) sViewer->setNeedsDisplay();
}

void
Viewer::mouseWheelCallback(int pos)
{
    sViewer->mCamera->mouseWheelCallback(pos);

    if (sViewer->mCamera->needsDisplay()) sViewer->setNeedsDisplay();
}

void
Viewer::windowSizeCallback(int, int)
{
    sViewer->setNeedsDisplay();
}


////////////////////////////////////////


bool
Viewer::needsDisplay()
{
    if (sViewer->mUpdates < 2) {
        sViewer->mUpdates += 1;
        return true;
    }
    return false;
}


void
Viewer::setNeedsDisplay()
{
    sViewer->mUpdates = 0;
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
