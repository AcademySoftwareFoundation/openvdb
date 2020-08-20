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
    void   render(int frame) override;
    void   resize(int width, int height) override;
    void   run() override;
    void   renderViewOverlay() override;
    double getTime() override;
    void   printHelp() const override;
    bool   updateCamera(int frame) override;

    void onKey(int key, int action);
    void onMouseButton(int button, int action);
    void onMouseMove(int x, int y);
    void onMouseWheel(int pos);
    void onResize(int width, int height);
    void onRefresh() {}
    void onDrop(int numPaths, const char** paths);

    static void mainLoop(void* userData);
    bool runLoop();

protected:
    void updateWindowTitle();


private:
    // gui.
    bool mIsDrawingGridStats = true;
    bool mIsDrawingRenderStats = true;
    bool mIsDrawingAboutDialog = false;
    bool mIsDrawingHelpDialog = false;
    bool mIsDrawingOutliner = false;
    bool mIsDrawingRenderOptions = false;

    bool drawPointRenderOptionsWidget();
    void drawRenderStatsOverlay();
    void drawGridStatsOverlay();
    void drawAboutDialog();
    void drawRenderPlatformWidget(const char* label);
    void drawMenuBar();
    void drawGridOutliner();
    void drawGridTree();
    void drawHelpDialog();
    void drawRenderOptionsDialog();

    void* mWindow = nullptr;
    int   mFps = 0;
    size_t mFpsFrame = 0;
    double mTime = 0;

    enum class PlaybackState { STOP=0,PLAY=1};
    PlaybackState mPlaybackState = PlaybackState::STOP;

    // mouse state.
    bool  mMouseDown = false;
    bool  mIsFirstMouseMove = false;
    bool  mIsMouseRightDown = false;
    float mMouseX = 0;
    float mMouseY = 0;
    int   mWheelPos = 0;
};
