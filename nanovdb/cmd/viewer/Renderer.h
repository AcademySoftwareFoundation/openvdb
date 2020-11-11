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
#include <queue>
#include <mutex>

#include "GridAssetUrl.h"
#include "FrameBuffer.h"
#include <nanovdb/util/GridHandle.h>
#include "RenderLauncher.h"
#include "GridManager.h"

// struct representing a scene graph node's grid attachment.
struct SceneNodeGridAttachment
{
    using Ptr = std::shared_ptr<SceneNodeGridAttachment>;

    int                          mIndex;
    GridManager::AssetGridStatus mStatus;
    std::string                  mFrameUrl;
    GridAssetUrl                 mAssetUrl;
    nanovdb::GridClass           mGridClassOverride;
    RendererAttributeParams      attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::End];
};

// struct representing a scene graph node.
struct SceneNode
{
    using Ptr = std::shared_ptr<SceneNode>;

    int                                       mIndex;
    std::string                               mName;
    std::vector<SceneNodeGridAttachment::Ptr> mAttachments;
    nanovdb::BBox<nanovdb::Vec3d>             mBounds;
    MaterialClass                             mMaterialClass = MaterialClass::kAuto;
    MaterialParameters                        mMaterialParameters;
};

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

    int                   mWidth = 512;
    int                   mHeight = 512;
    std::string           mOutputFilePath;
    std::string           mOutputExtension;
    int                   mOutputPadding = 4;
    std::string           mGoldPrefix;
    bool                  mUseTurntable = false;
    float                 mTurntableRate = 1;
    MaterialClass         mMaterialOverride = MaterialClass::kAuto;
    bool                  mUseAccumulation = true;
    int                   mRenderLauncherType = 0;
    int                   mFrameStart = 0;
    int                   mFrameEnd = 99;
    bool                  mFrameLoop = true;
    int                   mMaxProgressiveSamples = 1;
    float                 mMaterialBlackbodyTemperature = 1.0f;
    float                 mMaterialVolumeDensity = 1.0f;
    SceneRenderParameters mSceneParameters;
};

class RendererBase
{
public:
    RendererBase(const RendererParams& params);
    virtual ~RendererBase() {}

    virtual void run() = 0;

    virtual void   open() = 0;
    virtual void   close();
    virtual bool   render(int frame);
    virtual void   resizeFrameBuffer(int width, int height);
    virtual void   renderViewOverlay();
    virtual double getTime();
    bool           updateScene();
    void           setSceneFrame(int frame);
    int            getSceneFrame() const;
    virtual bool   updateCamera();
    virtual void   printHelp(std::ostream& s) const;
    void           renderSequence();
    bool           updateNodeAttachmentRequests(SceneNode::Ptr node, bool isSyncing, bool isPrinting, bool* isSelectedNodePending = nullptr);

    int            addGridAssetsAndNodes(const std::string& nodePrefix, std::vector<GridAssetUrl> urls);
    std::string    addSceneNode(const std::string& nodeName = "", bool makeUnique = true);
    void           setSceneNodeGridAttachment(const std::string& nodeName, int attachmentIndex, const GridAssetUrl& url);
    void           addGridAsset(const GridAssetUrl& url);
    std::string    updateFilePathWithFrame(const std::string& filePath, int frame) const;
    SceneNode::Ptr findNode(const std::string& name);
    SceneNode::Ptr findNodeByIndex(const int i);

    std::vector<std::string>     getGridNamesFromFile(const GridAssetUrl& url);
    GridManager::AssetGridStatus updateAttachmentState(const std::string& url, const GridManager::AssetStatusInfoType& residentAssetMap, SceneNodeGridAttachment::Ptr attachment);

    std::string nextUniqueNodeId(const std::string& prefix = "");

    void logError(const std::string& msg);
    void logInfo(const std::string& msg);
    void logDebug(const std::string& msg);

    void  resetAccumulationBuffer();
    bool  setRenderPlatformByName(std::string name);
    bool  saveFrameBuffer(int frame = 0, const std::string& filenameOverride = "", const std::string& formatOverride = "");
    float computePSNR(FrameBufferBase& other);

    void resetCamera(bool isFramingSceneNodeBounds = true);
    void setCamera(const nanovdb::Vec3f& rot);
    bool selectSceneNodeByIndex(int nodeIndex);

protected:
    void setRenderPlatform(int platform);

    void removeSceneNodes(std::vector<int> indices);
    void updateEventLog(bool isPrinting = false);

    GridManager                            mGridManager;
    bool                                   mIsDumpingLog = true;
    int                                    mLastEventIndex = 0;
    std::vector<GridManager::EventMessage> mEventMessages;
    std::unique_ptr<class FrameBufferBase> mFrameBuffer;
    int                                    mNumAccumulations = 0;
    RenderLauncher                         mRenderLauncher;
    RendererParams                         mParams;
    int                                    mSelectedSceneNodeIndex = -1;
    int                                    mPendingSceneFrame = 0;
    int                                    mLastSceneFrame = 0;
    RenderStatistics                       mRenderStats;
    uint32_t                               mNextUniqueId = 0;
    uint32_t                               mScreenShotIteration = 0;

    class CameraState
    {
    public:
        bool           mIsViewChanged = true;
        float          mCameraDistance = 1000.0f;
        nanovdb::Vec3f mCameraLookAt = nanovdb::Vec3f(0);
        nanovdb::Vec3f mCameraRotation = nanovdb::Vec3f(0, 3.142f / 2, 0);
        float          mFovY = 60.0f;
        float          mFrame = 0;

        nanovdb::Vec3f U() const { return mCameraAxis[0]; }
        nanovdb::Vec3f V() const { return mCameraAxis[1]; }
        nanovdb::Vec3f W() const { return mCameraAxis[2]; }
        nanovdb::Vec3f eye() const { return mCameraPosition; }
        nanovdb::Vec3f target() const { return mCameraLookAt; }

        bool update();

    private:
        nanovdb::Vec3f mCameraPosition = nanovdb::Vec3f{0, 0, 0};
        nanovdb::Vec3f mCameraAxis[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    };

    SceneNode::Ptr ensureSceneNode(const std::string& nodeName);
    void           updateAttachment(SceneNode::Ptr sceneNode, SceneNodeGridAttachment* attachment, const std::string& frameUrl, const std::string& gridName, GridManager::AssetGridStatus gridStatus);

    std::vector<SceneNode::Ptr> mSceneNodes;

    CameraState mDefaultCameraState;

public:
    CameraState* mCurrentCameraState = &mDefaultCameraState;
};
