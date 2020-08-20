// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file Renderer.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Class definition for a minimal, render-agnostic nanovdb Grid renderer.
*/

#pragma once

#include <string>
#include <vector>

#include "FrameBuffer.h"
#include <nanovdb/util/GridHandle.h>
#include "RenderLauncher.h"

inline std::string getStringForBlindDataSemantic(nanovdb::GridBlindDataSemantic semantic)
{
    switch (semantic) {
    case nanovdb::GridBlindDataSemantic::Unknown: return "Unknown";
    case nanovdb::GridBlindDataSemantic::PointPosition: return "PointPosition";
    case nanovdb::GridBlindDataSemantic::PointColor: return "PointColor";
    case nanovdb::GridBlindDataSemantic::PointNormal: return "PointNormal";
    case nanovdb::GridBlindDataSemantic::PointId: return "PointId";
    case nanovdb::GridBlindDataSemantic::PointRadius: return "PointRadius";
    case nanovdb::GridBlindDataSemantic::PointVelocity: return "PointVelocity";
    default: return "";
    }
}

struct RendererParams
{
    RendererParams();

    int         mWidth = 512;
    int         mHeight = 512;
    std::string mOutputPrefix;
    std::string mGoldPrefix;
    bool        mUseTurntable = false;
    bool        mUseAccumulation = true;
    int         mRenderLauncherType = 0;
    //RenderMethod    mRenderMethod = RenderMethod::AUTO;
    int             mFrameCount = 0;
    RenderConstants mOptions;
};

class RendererBase
{
public:
    RendererBase(const RendererParams& params);
    virtual ~RendererBase() {}

    virtual void run() = 0;

    virtual void   open() = 0;
    virtual void   close();
    virtual void   render(int frame);
    virtual void   resize(int width, int height);
    virtual void   renderViewOverlay();
    virtual double getTime();
    virtual bool   updateCamera(int frame);
    virtual void   printHelp() const;

    void  resetAccumulationBuffer();
    void  addGrid(std::string groupName, std::string fileName);
    void  addGrid(std::string groupName, std::string fileName, std::string gridName);
    bool  setRenderPlatformByName(std::string name);
    bool  saveFrameBuffer(bool useFrame, int frame = 0);
    float computePSNR(FrameBufferBase& other);

protected:
    void setRenderPlatform(int platform);
    void setGridIndex(int groupIndex, int gridIndex);
    void removeGridIndices(std::vector<int> indices);
    void resetCamera();

    std::unique_ptr<class FrameBufferBase> mFrameBuffer;
    int                                    mNumAccumulations = 0;
    RenderLauncher                         mRenderLauncher;
    RendererParams                         mParams;
    int                                    mRenderGroupIndex = -1;
    int                                    mFrame = 0;
    RenderStatistics                       mRenderStats;

    class CameraState
    {
    public:
        bool           mIsViewChanged = true;
        float          mCameraDistance = 1000.0f;
        nanovdb::Vec3f mCameraLookAt = nanovdb::Vec3f(0);
        nanovdb::Vec3f mCameraRotation = nanovdb::Vec3f(0, 3.142f / 2, 0);
        float          mFovY = 90.0f * 3.142f / 180.f;

        nanovdb::Vec3f U() const { return mCameraAxis[0]; }
        nanovdb::Vec3f V() const { return mCameraAxis[1]; }
        nanovdb::Vec3f W() const { return mCameraAxis[2]; }
        nanovdb::Vec3f eye() const { return mCameraPosition; }
        nanovdb::Vec3f target() const { return mCameraLookAt; }

        bool update();

    private:
        nanovdb::Vec3f mCameraPosition;
        nanovdb::Vec3f mCameraAxis[3];
    };

    struct GridInstance
    {
        nanovdb::GridHandle<> mGridHandle;
        std::string           mFileName;
        std::string           mFilePath;
        std::string           mGridName;
        nanovdb::GridClass    mGridClassOverride;
        RendererAttributeParams attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::End];
    };

    struct GridGroup
    {
        std::vector<std::shared_ptr<GridInstance>> mInstances;
        std::string                                mName;
        int                                        mCurrentGridIndex;
        nanovdb::BBox<nanovdb::Vec3d>              mBounds;
        RenderMethod                               mRenderMethod;
    };

    std::vector<std::shared_ptr<GridGroup>> mGridGroups;

    CameraState  mDefaultCameraState;
    CameraState* mCurrentCameraState = &mDefaultCameraState;
};
