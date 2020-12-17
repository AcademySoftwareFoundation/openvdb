// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file Renderer.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of RendererBase.
*/

#define _USE_MATH_DEFINES
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <unordered_set>

#include "StringUtils.h"
#include "Renderer.h"
#include "RenderLauncher.h"

RendererParams::RendererParams()
{
    mSceneParameters = makeSceneRenderParameters();
}

RendererBase::RendererBase(const RendererParams& params)
    : mParams(params)
{
    mGridManager.initialize();

    setRenderPlatform(0);

    mPendingSceneFrame = 0;
}

void RendererBase::close()
{
    mFrameBuffer.reset();
}

bool RendererBase::setRenderPlatformByName(std::string name)
{
    int found = mRenderLauncher.getPlatformIndexFromName(name);
    if (found < 0)
        return false;
    setRenderPlatform(found);
    return true;
}

void RendererBase::setRenderPlatform(int platform)
{
    mParams.mRenderLauncherType = platform;
    mRenderLauncher.setPlatform(mParams.mRenderLauncherType);
    resetAccumulationBuffer();
}

double RendererBase::getTime()
{
    auto t = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
    return duration / 1000000.0;
}

void RendererBase::renderViewOverlay()
{
}

void RendererBase::resizeFrameBuffer(int width, int height)
{
    mFrameBuffer->setup(width, height, FrameBufferBase::InternalFormat::RGBA32F);
    resetAccumulationBuffer();
}

SceneNode::Ptr RendererBase::ensureSceneNode(const std::string& nodeName)
{
    if (nodeName.empty())
        return nullptr;

    SceneNode::Ptr sceneNode = nullptr;

    // find existing sceneNode...
    for (auto& it : mSceneNodes) {
        if (it->mName == nodeName) {
            sceneNode = it;
        }
    }

    if (!sceneNode) {
        logInfo("Creating sceneNode[" + nodeName + "]");

        sceneNode = std::make_shared<SceneNode>();
        sceneNode->mMaterialClass = mParams.mMaterialOverride;
        sceneNode->mName = nodeName;
        sceneNode->mIndex = (int)mSceneNodes.size();
        sceneNode->mMaterialParameters = makeMaterialParameters();
        sceneNode->mMaterialParameters.volumeTemperatureScale = mParams.mMaterialBlackbodyTemperature;
        sceneNode->mMaterialParameters.volumeDensityScale = mParams.mMaterialVolumeDensity;
        mSceneNodes.push_back(sceneNode);

        // we currently only support 2 grid attachments.
        // if more are needed, then add them here.
        static constexpr int kMaxAttachments = 2;

        for (int i = 0; i < kMaxAttachments; ++i) {
            auto attachment = std::make_shared<SceneNodeGridAttachment>();
            attachment->mStatus = GridManager::AssetGridStatus::kUnknown;
            attachment->mIndex = i;
            sceneNode->mAttachments.push_back(attachment);
        }
    }

    // if no nodes are selected, then select this.
    // this will make the viewer to show the first created node by default.
    if (mSelectedSceneNodeIndex < 0)
        mSelectedSceneNodeIndex = (int)mSceneNodes.size() - 1;

    return sceneNode;
}

SceneNode::Ptr RendererBase::findNodeByIndex(const int i)
{
    if (i < 0 || i >= mSceneNodes.size())
        return nullptr;
    return mSceneNodes[i];
}

SceneNode::Ptr RendererBase::findNode(const std::string& name)
{
    for (size_t i = 0; i < mSceneNodes.size(); ++i) {
        if (mSceneNodes[i]->mName == name) {
            return mSceneNodes[i];
        }
    }
    return nullptr;
}

std::string RendererBase::nextUniqueNodeId(const std::string& name)
{
    // find name in the scene graph
    int  index = 0;
    auto prefix = name;
    if (prefix.empty())
        prefix = "default";
    std::ostringstream ss(prefix);
    while (findNode(ss.str())) {
        // add index to name...
        ss.str("");
        ss.clear();
        ss << prefix << index++;
    }
    return ss.str();
}

GridManager::AssetGridStatus RendererBase::updateAttachmentState(const std::string& url, const GridManager::AssetStatusInfoType& residentAssetMap, SceneNodeGridAttachment::Ptr attachment)
{
    auto assetUrl = attachment->mAssetUrl;
    if (!assetUrl)
        return GridManager::AssetGridStatus::kUnknown;

    auto assetIt = residentAssetMap.find(url);
    if (assetIt == residentAssetMap.end()) {
        return GridManager::AssetGridStatus::kUnknown;
    }

    // if we got this far then the asset exists.
    // we need to check the asset contains the grid-asset...

    auto assetInfo = assetIt->second;
    if (assetInfo.first) {
        // the asset had an error.
        return GridManager::AssetGridStatus::kError;
    }

    auto gridName = assetUrl.gridName();
    if (gridName.empty() && assetInfo.second.size() > 0)
        gridName = assetInfo.second.begin()->first;

    // if sceneNode gridname is specified, then find status of this asset & grid...
    GridManager::AssetGridStatus gridStatus = GridManager::AssetGridStatus::kUnknown;

    auto git = assetInfo.second.find(gridName);
    if (git != assetInfo.second.end())
        gridStatus = git->second;

    return gridStatus;
}

bool RendererBase::updateNodeAttachmentRequests(SceneNode::Ptr node, bool isSyncing, bool isPrinting, bool* isSelectedNodePending)
{
    if (!node)
        return false;

    // collect the latest resident assets and grids.
    auto residentAssetMap = mGridManager.getGridNameStatusInfo();

    bool hasErrors = false;

    // build request-list for selected nodes' attachments...
    std::vector<std::tuple<std::string, std::string>> urlRequests;

    for (auto& attachment : node->mAttachments) {
        auto assetUrl = attachment->mAssetUrl;
        if (!assetUrl)
            continue;

        auto url = attachment->mAssetUrl.updateUrlWithFrame(mPendingSceneFrame);
        auto gridAssetStatus = updateAttachmentState(url, residentAssetMap, attachment);
        updateAttachment(node, attachment.get(), url, assetUrl.gridName(), gridAssetStatus);

        // we ignore any assets that have errored or loaded, so we don't keep trying every frame.
        // NOTE: to reload an asset which has errored, we must remove the asset from the gridmanager.
        if (gridAssetStatus == GridManager::AssetGridStatus::kUnknown) {
            urlRequests.push_back({url, assetUrl.gridName()});
        }

        hasErrors |= (gridAssetStatus == GridManager::AssetGridStatus::kError);
    }

    if (isSelectedNodePending && urlRequests.size())
        *isSelectedNodePending = true;

    // submit selected nodes' attachment requests...
    for (auto& request : urlRequests) {
        mGridManager.addGrid(std::get<0>(request), std::get<1>(request));
    }

    if (isSyncing) {
        //logInfo("Waiting for assets...");
        do {
            updateEventLog(isPrinting);

            // we collect the latest resident assets and grids.
            auto residentAssetMap = mGridManager.getGridNameStatusInfo();

            // for each node, update the attachments...
            for (auto& attachment : node->mAttachments) {
                auto assetUrl = attachment->mAssetUrl;
                if (!assetUrl)
                    continue;

                auto url = assetUrl.updateUrlWithFrame(mPendingSceneFrame);
                auto gridAssetStatus = updateAttachmentState(url, residentAssetMap, attachment);
                hasErrors |= (gridAssetStatus == GridManager::AssetGridStatus::kError);

                updateAttachment(node, attachment.get(), url, assetUrl.gridName(), gridAssetStatus);
            }

        } while (isSyncing && mGridManager.poll()); // optionally loop while any asset requests are in flight...
    }

    updateEventLog(isPrinting);

    return !hasErrors;
}

void RendererBase::updateAttachment(SceneNode::Ptr sceneNode, SceneNodeGridAttachment* attachment, const std::string& frameUrl, const std::string& gridName, GridManager::AssetGridStatus gridStatus)
{
    attachment->mStatus = gridStatus;

    // update given the asset's status...

    auto gridHdlPtr = std::get<1>(mGridManager.getGrid(frameUrl, gridName));
    if (!gridHdlPtr || gridStatus != GridManager::AssetGridStatus::kLoaded) {
        return;
    }

    // the first time we find a loaded grid instance, we perform setup...
    if (attachment->mFrameUrl != frameUrl) {
        // the grid is now ready...
        // each new frame of this grid may require setup if it has changed.

        resetAccumulationBuffer();
        /*
        std::ostringstream ss;
        ss << "sceneNode[" << sceneNode->mName << "].attachment[" << frameUrl << "#" << gridName << "] is now ready.";
        logDebug(ss.str());
*/
        attachment->mFrameUrl = frameUrl;

        auto& gridHdl = *gridHdlPtr;
        auto* meta = gridHdl.gridMetaData();

        // update the grid instance's attribute map...
        if (gridHdl.gridMetaData()->isPointData()) {
            auto grid = gridHdl.grid<uint32_t>();
            assert(grid);

            // set defaults...
            for (int i = 0; i < (int)nanovdb::GridBlindDataSemantic::End; ++i) {
                attachment->attributeSemanticMap[i].attribute = -1;
                attachment->attributeSemanticMap[i].gain = 1.0f;
                attachment->attributeSemanticMap[i].offset = 0.0f;
            }
            attachment->attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius].offset = 0.1f;

            for (int i = 0; i < grid->blindDataCount(); ++i) {
                auto meta = grid->blindMetaData(i);
                attachment->attributeSemanticMap[(int)meta.mSemantic].attribute = i;
                attachment->attributeSemanticMap[(int)meta.mSemantic].gain = 1.0f;
                attachment->attributeSemanticMap[(int)meta.mSemantic].offset = 0.0f;
            }
        }

        attachment->mGridClassOverride = meta->gridClass();

        // this modifies the sceneNode's bounds.

        if (sceneNode->mBounds.empty()) {
            //std::cout << "initializing sceneNode bounds...\n";
            if (meta->activeVoxelCount() > 0) {
                sceneNode->mBounds = meta->worldBBox();
            }
        } else {
            //std::cout << "expanding sceneNode bounds...\n";
            if (meta->activeVoxelCount() > 0) {
                sceneNode->mBounds.expand(meta->worldBBox().max());
                sceneNode->mBounds.expand(meta->worldBBox().min());
            }
        }
    }
}

void RendererBase::updateEventLog(bool isPrinting)
{
    std::vector<GridManager::EventMessage> eventMessages;
    mLastEventIndex += mGridManager.getEventMessages(eventMessages, mLastEventIndex);
    for (size_t i = 0; i < eventMessages.size(); ++i) {
        if (isPrinting) {
            const auto& e = eventMessages[i];
            if (e.mType == GridManager::EventMessage::Type::kError)
                std::cout << "[ERR]: ";
            else if (e.mType == GridManager::EventMessage::Type::kDebug)
                std::cout << "[DBG]: ";
            else if (e.mType == GridManager::EventMessage::Type::kWarning)
                std::cout << "[WRN]: ";
            else if (e.mType == GridManager::EventMessage::Type::kInfo)
                std::cout << "[INF]: ";
            std::cout << e.mMessage << std::endl;
        }

        mEventMessages.emplace_back(std::move(eventMessages[i]));
    }
}

std::string RendererBase::updateFilePathWithFrame(const std::string& url, int frame) const
{
    if (url.find('%') != std::string::npos) {
        std::string tmp = url;
        char        fileNameBuf[FILENAME_MAX];
        while (1) {
            auto pos = tmp.find_last_of('%');
            if (pos == std::string::npos)
                break;
            auto segment = tmp.substr(pos);
            sprintf(fileNameBuf, segment.c_str(), frame);
            segment.assign(fileNameBuf);
            tmp = tmp.substr(0, pos) + segment;
        }
        return tmp;
    }
    return url;
}

void RendererBase::logDebug(const std::string& msg)
{
    mGridManager.addEventMessage(GridManager::EventMessage{GridManager::EventMessage::Type::kDebug, msg});
    updateEventLog(true);
}

void RendererBase::logError(const std::string& msg)
{
    mGridManager.addEventMessage(GridManager::EventMessage{GridManager::EventMessage::Type::kError, msg});
    updateEventLog(true);
}

void RendererBase::logInfo(const std::string& msg)
{
    mGridManager.addEventMessage(GridManager::EventMessage{GridManager::EventMessage::Type::kInfo, msg});
    updateEventLog(true);
}

std::string RendererBase::addSceneNode(const std::string& nodeName, bool makeUnique)
{
    auto newName = nodeName;
    if (makeUnique)
        newName = nextUniqueNodeId(nodeName);
    ensureSceneNode(newName);
    return newName;
}

void RendererBase::setSceneNodeGridAttachment(const std::string& nodeName, int attachmentIndex, const GridAssetUrl& url)
{
    auto node = ensureSceneNode(nodeName);
    if (node) {
        if (attachmentIndex >= node->mAttachments.size()) {
            logError("sceneNode[" + nodeName + "] ignoring grid attachment: " + url.fullname());
        } else {
            logInfo("sceneNode[" + nodeName + "] assigning grid attachment: " + url.fullname());
            if (node->mAttachments[attachmentIndex]->mAssetUrl != url) {
                node->mAttachments[attachmentIndex]->mAssetUrl = url;
                node->mAttachments[attachmentIndex]->mStatus = GridManager::AssetGridStatus::kUnknown;
                node->mAttachments[attachmentIndex]->mFrameUrl = "";
                node->mAttachments[attachmentIndex]->mGridClassOverride = nanovdb::GridClass::Unknown;
                // clear the bounds
                node->mBounds = nanovdb::BBoxR();
            }
        }
    }
}

int RendererBase::addGridAssetsAndNodes(const std::string& nodePrefix, std::vector<GridAssetUrl> urls)
{
    std::string nodeId;
    for (size_t i = 0; i < urls.size(); ++i) {
        auto& assetUrl = urls[i];
        if (assetUrl.scheme() == "file" && assetUrl.gridName().empty()) {
            auto gridNames = getGridNamesFromFile(assetUrl);
            for (auto& gridName : gridNames) {
                assetUrl.gridName() = gridName;
                addGridAsset(assetUrl);
                nodeId = addSceneNode(nodePrefix, true);
                setSceneNodeGridAttachment(nodeId, 0, assetUrl);
            }
        } else {
            addGridAsset(assetUrl);
            nodeId = addSceneNode(nodePrefix, true);
            setSceneNodeGridAttachment(nodeId, 0, assetUrl);
        }
    }
    updateEventLog(true);

    if (nodeId.length())
        return findNode(nodeId)->mIndex;
    return -1;
}

std::vector<std::string> RendererBase::getGridNamesFromFile(const GridAssetUrl& url)
{
    return mGridManager.getGridsNamesFromLocalFile(url.fullname(), url.getSequencePath(getSceneFrame()));
}

void RendererBase::addGridAsset(const GridAssetUrl& url)
{
    if (url.isSequence()) {
        // request the pending frame.
        auto frameUrl = url.updateUrlWithFrame(mPendingSceneFrame);
        mGridManager.addGrid(frameUrl, url.gridName());
    } else {
        mGridManager.addGrid(url.url(), url.gridName());
    }
}

void RendererBase::resetAccumulationBuffer()
{
    if (mNumAccumulations > 0) {
        mNumAccumulations = 0;
    }
}

bool RendererBase::CameraState::update()
{
    if (!mIsViewChanged)
        return false;

    nanovdb::Vec3<float> forward, right;

    right[0] = -cos(mCameraRotation[1]);
    right[1] = 0.0f;
    right[2] = sin(mCameraRotation[1]);

    forward[0] = -sin(mCameraRotation[1]) * cos(mCameraRotation[0]);
    forward[1] = -sin(mCameraRotation[0]);
    forward[2] = -cos(mCameraRotation[1]) * cos(mCameraRotation[0]);

    mCameraAxis[2] = forward;
    mCameraAxis[0] = right;
    mCameraAxis[1] = forward.cross(right).normalize();

    mCameraPosition[0] = mCameraLookAt[0] - forward[0] * mCameraDistance;
    mCameraPosition[1] = mCameraLookAt[1] - forward[1] * mCameraDistance;
    mCameraPosition[2] = mCameraLookAt[2] - forward[2] * mCameraDistance;

    mIsViewChanged = false;
    /*
	printf("Camera matrix:\n");
	printf("X: %f %f %f\n", mCameraAxis[0][0], mCameraAxis[0][1], mCameraAxis[0][2]);
	printf("Y: %f %f %f\n", mCameraAxis[1][0], mCameraAxis[1][1], mCameraAxis[1][2]);
	printf("Z: %f %f %f\n", mCameraAxis[2][0], mCameraAxis[2][1], mCameraAxis[2][2]);
	printf("T: %f %f %f\n", mCameraPosition[0], mCameraPosition[1], mCameraPosition[2]);
	*/
    return true;
}

bool RendererBase::render(int frame)
{
    if (mSceneNodes.size() == 0) {
        return false;
    }

    auto sceneNode = mSceneNodes[mSelectedSceneNodeIndex];
    assert(sceneNode->mAttachments.size() >= 0);

    bool hasCameraChanged = updateCamera();

    if (hasCameraChanged) {
        resetAccumulationBuffer();
    }

    auto attachment = sceneNode->mAttachments[0];

    // modify MaterialParameters...
    auto materialParameters = sceneNode->mMaterialParameters;
    std::memcpy(materialParameters.attributeSemanticMap, attachment->attributeSemanticMap, sizeof(RendererAttributeParams) * size_t(nanovdb::GridBlindDataSemantic::End));

    int w = mFrameBuffer->width();
    int h = mFrameBuffer->height();
    int numAccumulations = (mParams.mUseAccumulation) ? ++mNumAccumulations : 0;

    // build scene render parameters...
    auto wBbox = sceneNode->mBounds;
    if (wBbox.empty()) {
        // an invalid bounds will cause issues, so fixup.
        wBbox = nanovdb::BBoxR(nanovdb::Vec3R(0), nanovdb::Vec3R(1));
    }
    auto wBboxSize = wBbox.max() - wBbox.min();
    auto sceneParameters = mParams.mSceneParameters;
    sceneParameters.groundHeight = (float)wBbox.min()[1];
    sceneParameters.groundFalloff = (50.f * float(wBboxSize.length())) / tanf((3.142f / 180.f) * mCurrentCameraState->mFovY * 0.5f);
    sceneParameters.camera = Camera(mParams.mSceneParameters.camera.lensType(), mCurrentCameraState->eye(), mCurrentCameraState->target(), mCurrentCameraState->V(), mCurrentCameraState->mFovY, float(w) / h);
    sceneParameters.camera.ipd() = mParams.mSceneParameters.camera.ipd();

    bool renderRc = false;

    // update the material type based on the grid class and preset the values.
    auto materialClass = sceneNode->mMaterialClass;
    if (materialClass == MaterialClass::kAuto) {
        if (attachment->mGridClassOverride == nanovdb::GridClass::FogVolume)
            materialClass = MaterialClass::kFogVolumePathTracer;
        else if (attachment->mGridClassOverride == nanovdb::GridClass::LevelSet)
            materialClass = MaterialClass::kLevelSetFast;
        else if (attachment->mGridClassOverride == nanovdb::GridClass::PointData)
            materialClass = MaterialClass::kPointsFast;
        else if (attachment->mGridClassOverride == nanovdb::GridClass::PointIndex)
            materialClass = MaterialClass::kPointsFast;
        else if (attachment->mGridClassOverride == nanovdb::GridClass::VoxelVolume)
            materialClass = MaterialClass::kVoxels;
        else
            materialClass = MaterialClass::kGrid;
    }

    // collect the grid pointers...
    std::vector<GridRenderParameters> gridsAttachmentPtrs;
    for (size_t i = 0; i < sceneNode->mAttachments.size(); ++i) {
        auto attachment = sceneNode->mAttachments[i];
        auto url = attachment->mAssetUrl.updateUrlWithFrame(frame);
        auto gridName = attachment->mAssetUrl.gridName();
        auto gridAssetData = mGridManager.getGrid(url, gridName);
        gridsAttachmentPtrs.push_back(GridRenderParameters{std::get<0>(gridAssetData), std::get<1>(gridAssetData).get()});
    }

    renderRc = mRenderLauncher.render(materialClass, w, h, mFrameBuffer.get(), numAccumulations, (int)gridsAttachmentPtrs.size(), gridsAttachmentPtrs.data(), sceneParameters, materialParameters, &mRenderStats);
    return renderRc;
}

bool RendererBase::selectSceneNodeByIndex(int nodeIndex)
{
    bool hasChanged = false;

    if (mSceneNodes.size() == 0) {
        mSelectedSceneNodeIndex = -1;
        resetAccumulationBuffer();
        return true;
    }

    // clamp sceneNode index to valid range...
    if (mSceneNodes.size() > 0) {
        if (nodeIndex < 0)
            nodeIndex = int(mSceneNodes.size()) - 1;
        else if (nodeIndex > int(mSceneNodes.size()) - 1)
            nodeIndex = 0;

        if (mSelectedSceneNodeIndex != nodeIndex) {
            hasChanged = true;
            mSelectedSceneNodeIndex = nodeIndex;
        }
    }

    if (hasChanged) {
        resetAccumulationBuffer();
    }

    return hasChanged;
}

void RendererBase::removeSceneNodes(std::vector<int> indices)
{
    if (indices.size() == 0)
        return;

    std::sort(indices.begin(), indices.end());
    std::reverse(indices.begin(), indices.end());

    for (size_t i = 0; i < indices.size(); ++i) {
        mSceneNodes.erase(mSceneNodes.begin() + indices[i]);
    }

    int i = mSelectedSceneNodeIndex;
    if (i >= int(mSceneNodes.size()))
        i = int(mSceneNodes.size()) - 1;

    selectSceneNodeByIndex(i);
}

void RendererBase::printHelp(std::ostream& s) const
{
    s << "-------------------------------------\n";
    s << "- Renderer-platform     = (" << mRenderLauncher.getNameForPlatformIndex(mParams.mRenderLauncherType) << ")\n";
    s << "- Render Group          = (" << (mSelectedSceneNodeIndex) << ")\n";
    s << "- Render Progressive    = (" << (mParams.mUseAccumulation ? "ON" : "OFF") << ")\n";
    s << "-------------------------------------\n";
    s << "\n";
}

void RendererBase::setCamera(const nanovdb::Vec3f& rot)
{
    mCurrentCameraState->mCameraRotation = rot;
    mCurrentCameraState->mIsViewChanged = true;
    mCurrentCameraState->update();
}

void RendererBase::resetCamera(bool isFramingSceneNodeBounds)
{
    nanovdb::BBox<nanovdb::Vec3R> bbox(nanovdb::Vec3R(-100), nanovdb::Vec3R(100));

    if (mSelectedSceneNodeIndex >= 0) {
        auto sceneNode = mSceneNodes[mSelectedSceneNodeIndex];
        if (isFramingSceneNodeBounds) {
            if (!sceneNode->mBounds.empty()) {
                bbox = sceneNode->mBounds;
            }
        } else {
            auto attachment = sceneNode->mAttachments[0];
            auto url = attachment->mAssetUrl.updateUrlWithFrame(getSceneFrame());
            auto gridName = attachment->mAssetUrl.gridName();
            auto gridAssetData = mGridManager.getGrid(url, gridName);
            auto gridBounds = std::get<0>(gridAssetData);
            if (!gridBounds.empty()) {
                bbox = gridBounds;
            }
        }
    }

    // calculate camera target and distance...
    auto  bboxSize = (bbox.max() - bbox.min());
    float halfWidth = 0.5f * nanovdb::Max(1.0f, float(bboxSize.length()));

    if (mParams.mSceneParameters.camera.lensType() == Camera::LensType::kSpherical ||
        mParams.mSceneParameters.camera.lensType() == Camera::LensType::kODS) {
        mCurrentCameraState->mCameraLookAt = nanovdb::Vec3f(bbox.min() + bboxSize * 0.5);
        mCurrentCameraState->mCameraDistance = halfWidth;
        mCurrentCameraState->mCameraRotation = nanovdb::Vec3f(0, 0, 0);
        mCurrentCameraState->mCameraLookAt[1] = 0;
    } else {
        if (mParams.mCameraTarget[0] != std::numeric_limits<float>::max())
            mCurrentCameraState->mCameraLookAt = mParams.mCameraTarget;
        else
            mCurrentCameraState->mCameraLookAt = nanovdb::Vec3f(bbox.min() + bboxSize * 0.5);

        if (mParams.mCameraFov != std::numeric_limits<float>::max())
            mCurrentCameraState->mFovY = std::max(1.0f, mParams.mCameraFov);

        if (mParams.mCameraDistance != std::numeric_limits<float>::max())
            mCurrentCameraState->mCameraDistance = mParams.mCameraDistance;
        else
            mCurrentCameraState->mCameraDistance = halfWidth / tanf(mCurrentCameraState->mFovY * 0.5f * (3.142f / 180.f));

        if (mParams.mCameraRotation[0] != std::numeric_limits<float>::max())
            mCurrentCameraState->mCameraRotation = mParams.mCameraRotation * (3.142f / 180.f);
        else
            mCurrentCameraState->mCameraRotation = nanovdb::Vec3f(float(M_PI) / 8.f, float(M_PI) / 4.f, 0.f);
    }

    mCurrentCameraState->mIsViewChanged = true;
    mCurrentCameraState->update();
}

void RendererBase::setSceneFrame(int frame)
{
    // wrap requested frame into valid range...
    int frameCount = (mParams.mFrameEnd - mParams.mFrameStart + 1);
    if (frame > mParams.mFrameEnd) {
        if (mParams.mFrameLoop) {
            frame -= mParams.mFrameStart;
            frame = frame % frameCount;
            frame += mParams.mFrameStart;
        } else
            frame = mParams.mFrameEnd;
    } else if (frame < mParams.mFrameStart) {
        if (mParams.mFrameLoop) {
            frame -= mParams.mFrameStart;
            frame = (frame % frameCount + frameCount) % frameCount;
            frame += mParams.mFrameStart;
        } else
            frame = mParams.mFrameStart;
    }

    mPendingSceneFrame = frame;
    mLastSceneFrame = mPendingSceneFrame - 1;
}

int RendererBase::getSceneFrame() const
{
    return mLastSceneFrame;
}

bool RendererBase::updateScene()
{
    if (mPendingSceneFrame == mLastSceneFrame)
        return false;
    mLastSceneFrame = mPendingSceneFrame;
    return true;
}

void RendererBase::renderSequence()
{
    if (mSelectedSceneNodeIndex < 0)
        return;

    auto oldFrame = getSceneFrame();

    bool isSingleFrame = (mParams.mFrameEnd <= mParams.mFrameStart);

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

    bool hasError = false;

    for (int frame = mParams.mFrameStart; frame <= mParams.mFrameEnd; ++frame) {
        setSceneFrame(frame);

        // sync the grid manager.
        bool areAttachmentsReady = updateNodeAttachmentRequests(mSceneNodes[mSelectedSceneNodeIndex], true, mIsDumpingLog);
        if (areAttachmentsReady == false) {
            hasError = true;
            logError("Unable to render frame " + ss.str() + "; bad asset");
            break;
        }

        updateScene();

        for (int i = (mParams.mUseAccumulation) ? mParams.mMaxProgressiveSamples : 1; i > 0; --i) {
            render(frame);
        }

        if (mParams.mOutputFilePath.empty() == false) {
            hasError = (saveFrameBuffer(frame) == false);
            if (hasError) {
                break;
            }
        }
    }

    if (hasError == false) {
        logInfo("Rendering complete.");
    } else {
        logError("Rendering failed.");
    }

    setSceneFrame(oldFrame);
}

bool RendererBase::updateCamera()
{
    int  sceneFrame = getSceneFrame();
    bool isChanged = false;

    if (mCurrentCameraState->mFrame != (float)sceneFrame) {
        isChanged = true;
        mCurrentCameraState->mFrame = (float)sceneFrame;
    }

    if (mParams.mUseTurntable && isChanged) {
        int count = (mParams.mFrameEnd - mParams.mFrameStart + 1);
        mCurrentCameraState->mCameraRotation[1] = ((float(sceneFrame) * 2.0f * float(M_PI)) / count) / std::max(mParams.mTurntableRate, 1.0f);
        mCurrentCameraState->mIsViewChanged = true;
    }

    isChanged |= mCurrentCameraState->update();
    return isChanged;
}

bool RendererBase::saveFrameBuffer(int frame, const std::string& filenameOverride, const std::string& formatOverride)
{
    assert(mFrameBuffer);

    std::string filename = filenameOverride;
    if (filename.empty())
        filename = mParams.mOutputFilePath;

    if (filename.empty()) {
        logError("Output filename must be specified for framebuffer export.");
        return false;
    }

    std::string ext = formatOverride;
    if (ext.empty()) {
        ext = mParams.mOutputExtension;
        if (ext.empty()) {
            ext = urlGetPathExtension(filename);
        }
    }

    if (ext.empty()) {
        logError("File format can not be determined from output filename, \"" + filename + "\"");
        return false;
    }

    auto resolvedFilename = updateFilePathWithFrame(filename, frame);
    logDebug("Exporting framebuffer to " + resolvedFilename);

    if (mFrameBuffer->save(resolvedFilename.c_str(), ext.c_str(), 80) == false) {
        logError("Unable to export framebuffer to filename \"" + resolvedFilename + "\" as \"" + ext + "\"");
        return false;
    }

    // we maintain a counter so that screenshot names are always unique.
    mScreenShotIteration++;
    return true;
}

float RendererBase::computePSNR(FrameBufferBase& other)
{
    return mFrameBuffer->computePSNR(other);
}
