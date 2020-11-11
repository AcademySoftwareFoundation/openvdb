// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file Viewer.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Class definition for a minimal, render-agnostic nanovdb Grid viewer using GLFW for display.
*/

#pragma once
#include "Renderer.h"

class Viewer : public RendererBase
{
public:
    Viewer(const RendererParams& params);
    virtual ~Viewer();

    void   open() override;
    void   close() override;
    bool   render(int frame) override;
    void   resizeFrameBuffer(int width, int height) override;
    void   run() override;
    void   renderViewOverlay() override;
    double getTime() override;
    void   printHelp(std::ostream& s) const override;
    bool   updateCamera() override;
    void   updateAnimationControl();
    void   setSceneFrame(int frame);

    void onKey(int key, int action);
    void onMouseButton(int button, int action);
    void onMouseMove(int x, int y);
    void onMouseWheel(int pos);
    void onResize(int width, int height);
    void onRefresh() {}
    void onDrop(int numPaths, const char** paths);

    static void mainLoop(void* userData);
    bool        runLoop();

protected:
    void updateWindowTitle();

private:
    // gui.
    bool         mIsDrawingRenderStats = true;
    bool         mIsDrawingAboutDialog = false;
    bool         mIsDrawingHelpDialog = false;
    bool         mIsDrawingSceneGraph = false;
    bool         mIsDrawingAssets = false;
    bool         mIsDrawingRenderOptions = false;
    bool         mIsDrawingEventLog = false;
    bool         mLogAutoScroll = true;
    bool         mIsDrawingPendingGlyph = false;

    bool drawPointRenderOptionsWidget(SceneNode::Ptr node, int attachmentIndex);
    void drawRenderStatsOverlay();
    void drawAboutDialog();
    void drawRenderPlatformWidget(const char* label);
    void drawMenuBar();
    void drawSceneGraph();
    void drawAssets();
    void drawSceneGraphNodes();
    void drawHelpDialog();
    void drawRenderOptionsDialog();
    void drawEventLog();
    void drawGridInfo(const std::string& url, const std::string& gridName);
    bool drawMaterialParameters(SceneNode::Ptr node, MaterialClass mat);
    bool drawMaterialGridAttachment(SceneNode::Ptr node, int attachmentIndex);
    void drawPendingGlyph();

    std::string getScreenShotFilename(int iteration) const;

    void*  mWindow = nullptr;
    int    mWindowWidth = 0;
    int    mWindowHeight = 0;
    int    mFps = 0;
    size_t mFpsFrame = 0;
    double mTime = 0;

    enum class PlaybackState { STOP = 0,
                               PLAY = 1 };
    PlaybackState mPlaybackState = PlaybackState::STOP;
    float         mPlaybackTime = 0;
    float         mPlaybackLastTime = 0;
    float         mPlaybackRate = 30;

    // mouse state.
    bool  mMouseDown = false;
    bool  mIsFirstMouseMove = false;
    bool  mIsMouseRightDown = false;
    float mMouseX = 0;
    float mMouseY = 0;
    int   mWheelPos = 0;
};
