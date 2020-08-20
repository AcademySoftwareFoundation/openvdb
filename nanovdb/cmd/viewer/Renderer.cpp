// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RendererBase.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of RendererBase.
*/

#define _USE_MATH_DEFINES
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>

#include "Renderer.h"
#include "RenderLauncher.h"

#include <nanovdb/util/IO.h> // for NanoVDB file import
#include <nanovdb/util/GridBuilder.h>
#if defined(NANOVDB_USE_OPENVDB)
#include <nanovdb/util/OpenToNanoVDB.h>
#endif

static nanovdb::GridHandle<> createInternalGrid(std::string internalName)
{
    if (internalName == "ls_sphere_100") {
        return nanovdb::createLevelSetSphere(100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "ls_torus_100") {
        return nanovdb::createLevelSetTorus(100.0f, 50.f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "ls_box_100") {
        return nanovdb::createLevelSetBox(100.0f, 100.0f, 100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "fog_sphere_100") {
        return nanovdb::createFogVolumeSphere(100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "fog_torus_100") {
        return nanovdb::createFogVolumeTorus(100.0f, 50.f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "fog_box_100") {
        return nanovdb::createFogVolumeBox(100.0f, 100.0f, 100.0f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "points_sphere_100") {
        return nanovdb::createPointSphere(1, 100.0f, nanovdb::Vec3d(0), 1.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "points_torus_100") {
        return nanovdb::createPointTorus(1, 100.0f, 50.f, nanovdb::Vec3d(0), 1.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "points_box_100") {
        return nanovdb::createPointBox(1, 100.0f, 100.0f, 100.0f, nanovdb::Vec3d(0), 1.0f, nanovdb::Vec3R(0), internalName);
    } else if (internalName == "ls_bbox_100") {
        return nanovdb::createLevelSetBBox(100.0f, 100.0f, 100.0f, 10.f, nanovdb::Vec3d(0), 1.0f, 3.0f, nanovdb::Vec3R(0), internalName);
    } else {
        return nanovdb::GridHandle<>();
    }
}

RendererParams::RendererParams()
{
    mOptions.useLighting = 1;
    mOptions.useGround = 1;
    mOptions.useOcclusion = 0;
    mOptions.useShadows = 1;
    mOptions.useGroundReflections = 0;
    mOptions.samplesPerPixel = 1;
    mOptions.volumeDensity = 0.5f;
    mOptions.tonemapWhitePoint = 1.5f;
    mOptions.useTonemapping = true;
}

RendererBase::RendererBase(const RendererParams& params)
    : mParams(params)
{
#if defined(NANOVDB_USE_OPENVDB)
    openvdb::initialize();
#endif

    setRenderPlatform(0);

    mFrame = 0;
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

void RendererBase::resize(int width, int height)
{
    mFrameBuffer->setup(width, height, FrameBufferBase::InternalFormat::RGBA32F);
    resetAccumulationBuffer();
}

void RendererBase::addGrid(std::string groupName, std::string fileName)
{
    if (fileName.find("internal://", 0) != std::string::npos) {
        groupName = "__internal";
    }

    if (groupName == "__internal") {
        auto internalGridName = fileName;
        if (fileName.find("internal://", 0) != std::string::npos) {
            internalGridName = fileName.substr(11);
        }
        addGrid(groupName, groupName, internalGridName);
    } else if (fileName.substr(fileName.find_last_of(".") + 1) == "vdb") {
#if defined(NANOVDB_USE_OPENVDB)
        openvdb::io::File file(fileName);
        file.open(true);
        auto grids = file.getGrids();
        for (auto& grid : *grids) {
            addGrid(groupName, fileName, grid->getName());
        }
#else
        throw std::runtime_error("OpenVDB is not supported in this build. Please recompile with OpenVDB support.");
#endif
    } else {
        // load all the grids in the file...
        auto list = nanovdb::io::readGridMetaData(fileName);
        for (auto& m : list)
            addGrid(groupName, fileName, m.gridName);
    }
}

void RendererBase::addGrid(std::string groupName, std::string fileName, std::string gridName)
{
    // check it is not already resident!
    std::shared_ptr<GridGroup> group = nullptr;
    for (auto& it : mGridGroups) {
        if (it->mName == groupName) {
            group = it;
        }
    }

    if (!group) {
        group = std::make_shared<GridGroup>();
        group->mRenderMethod = RenderMethod::AUTO;
        group->mName = groupName;
        mGridGroups.push_back(group);
        //std::cout << "Creating group[" << groupName << "]" << std::endl;

    } else {
        for (auto& it : group->mInstances) {
            if (it->mFileName == fileName && it->mGridName == gridName) {
                throw std::runtime_error("Grid already loaded.");
            }
        }
    }

    nanovdb::GridHandle<> gridHdl;

    if (groupName == "__internal") {
        auto internalName = gridName;
        gridHdl = createInternalGrid(internalName);
    } else if (fileName.substr(fileName.find_last_of(".") + 1) == "vdb") {
#if defined(NANOVDB_USE_OPENVDB)
        openvdb::io::File file(fileName);
        file.open(false); //disable delayed loading
        auto grid = file.readGrid(gridName);
        std::cout << "Importing OpenVDB grid[" << grid->getName() << "]...\n";
        gridHdl = nanovdb::openToNanoVDB(grid);
#endif
    } else {
        std::cout << "Importing NanoVDB grid[" << gridName << "]...\n";
        if (gridName.length() > 0)
            gridHdl = nanovdb::io::readGrid<>(fileName, gridName);
        else
            gridHdl = nanovdb::io::readGrid<>(fileName);
    }

    if (!gridHdl) {
        std::stringstream ss;
        ss << "Unable to read " << gridName << " from " << fileName;
        throw std::runtime_error(ss.str());
    }

    auto* meta = gridHdl.gridMetaData();

    auto gridInstance = std::make_shared<GridInstance>();

    // update the grid instance's attribute map...
    if (gridHdl.gridMetaData()->isPointData()) {
        auto grid = gridHdl.grid<uint32_t>();
        assert(grid);

        char** names;
        int    n = grid->blindDataCount();
        names = new char*[n];

        // set defaults...
        for (int i = 0; i < (int)nanovdb::GridBlindDataSemantic::End; ++i) {
            gridInstance->attributeSemanticMap[i].attribute = -1;
            gridInstance->attributeSemanticMap[i].gain = 1.0f;
            gridInstance->attributeSemanticMap[i].offset = 0.0f;
        }
        gridInstance->attributeSemanticMap[(int)nanovdb::GridBlindDataSemantic::PointRadius].offset = 0.5f;

        for (int i = 0; i < n; ++i) {
            auto meta = grid->blindMetaData(i);
            gridInstance->attributeSemanticMap[(int)meta.mSemantic].attribute = i;
            gridInstance->attributeSemanticMap[(int)meta.mSemantic].gain = 1.0f;
            gridInstance->attributeSemanticMap[(int)meta.mSemantic].offset = 0.0f;
        }
    }

    gridInstance->mGridHandle = std::move(gridHdl);
    gridInstance->mFileName = fileName.substr(fileName.find_last_of('/') + 1).substr(fileName.find_last_of('\\') + 1);
    gridInstance->mFilePath = fileName;
    gridInstance->mGridName = meta->gridName();
    gridInstance->mGridClassOverride = meta->gridClass();

    if (meta->activeVoxelCount() > 0) {
        group->mBounds.expand(meta->worldBBox().max());
        group->mBounds.expand(meta->worldBBox().min());
    }

    group->mCurrentGridIndex = int(group->mInstances.size());

    group->mInstances.emplace_back(gridInstance);

    //std::cout << "Added instance[" << fileName << "] to group[" << group->mName << "]" << std::endl;

    setGridIndex(int(mGridGroups.size()) - 1, int(group->mInstances.size()) - 1);
    resetCamera();
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

void RendererBase::render(int frame)
{
    if (mGridGroups.size() == 0) {
        return;
    }

    bool hasCameraChanged = updateCamera(frame);

    if (hasCameraChanged) {
        resetAccumulationBuffer();
    }

    auto        group = mGridGroups[mRenderGroupIndex];
    auto        instance = group->mInstances[group->mCurrentGridIndex];
    const auto& gridHdl = instance->mGridHandle;

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

    renderRc = mRenderLauncher.render(renderMethod, w, h, mFrameBuffer.get(), camera, gridHdl, numAccumulations, renderConstants, &mRenderStats);
}

void RendererBase::setGridIndex(int groupIndex, int gridIndex)
{
    if (mGridGroups.size() > 0) {
        if (groupIndex < 0)
            groupIndex = int(mGridGroups.size()) - 1;
        else if (groupIndex > int(mGridGroups.size()) - 1)
            groupIndex = 0;
    }

    if (groupIndex < 0)
    {
        mRenderGroupIndex = groupIndex;
        resetAccumulationBuffer();   
        return;
    }

    auto group = mGridGroups[groupIndex];

    if (gridIndex < 0)
        gridIndex = int(group->mInstances.size()) - 1;
    else if (gridIndex > int(group->mInstances.size()) - 1)
        gridIndex = 0;

    if (group->mCurrentGridIndex == gridIndex && mRenderGroupIndex == groupIndex) {
        // no change!
        return;
    }

    mRenderGroupIndex = groupIndex;    
    group->mCurrentGridIndex = gridIndex;

    std::cout << "Selecting group["  << groupIndex << "].instance[" << gridIndex << "]\n";

    if (gridIndex < 0)
        return;

    // update the attribute map...
    auto instance = group->mInstances[gridIndex];
    memcpy(mParams.mOptions.attributeSemanticMap, instance->attributeSemanticMap, sizeof(RendererAttributeParams) * size_t(nanovdb::GridBlindDataSemantic::End));

    resetAccumulationBuffer();    
}

void RendererBase::removeGridIndices(std::vector<int> indices)
{
    if (indices.size() == 0)
        return;

    std::sort(indices.begin(), indices.end());
    std::reverse(indices.begin(), indices.end());

    for (int i = 0; i < indices.size(); ++i) {
        mGridGroups.erase(mGridGroups.begin() + indices[i]);
    }

    int i = mRenderGroupIndex;
    if (i >= int(mGridGroups.size()))
        i = int(mGridGroups.size()) - 1;

    setGridIndex(i, 0);
}

void RendererBase::printHelp() const
{
    std::cout << "-------------------------------------\n";
    std::cout << "- Renderer-platform     = (" << mRenderLauncher.getNameForPlatformIndex(mParams.mRenderLauncherType) << ")\n";
    std::cout << "- Render Group          = (" << (mRenderGroupIndex) << ")\n";
    std::cout << "- Render Progressive    = (" << (mParams.mUseAccumulation ? "ON" : "OFF") << ")\n";
    std::cout << "- Render Lighting       = (" << (mParams.mOptions.useLighting ? "ON" : "OFF") << ")\n";
    std::cout << "- Render Shadows        = (" << (mParams.mOptions.useShadows ? "ON" : "mFrameBufferOFF") << ")\n";
    std::cout << "- Render Ground-plane   = (" << (mParams.mOptions.useGround ? "ON" : "OFF") << ")\n";
    std::cout << "- Render Occlusion      = (" << (mParams.mOptions.useOcclusion ? "ON" : "OFF") << ")\n";
    std::cout << "-------------------------------------\n";
    std::cout << "\n";
}

void RendererBase::resetCamera()
{
    if (mRenderGroupIndex < 0) {
        return;
    }

    // calculate camera target and distance.

    auto bbox = mGridGroups[mRenderGroupIndex]->mBounds;

    auto bboxSize = (bbox.max() - bbox.min());
    mCurrentCameraState->mCameraDistance = nanovdb::Max(1.0f, float(bboxSize.length()) * 30.f);
    mCurrentCameraState->mCameraLookAt = nanovdb::Vec3f(bbox.min() + bboxSize * 0.5);
    mCurrentCameraState->mCameraRotation = nanovdb::Vec3f(M_PI / 8, (M_PI) / 4, 0);
    mCurrentCameraState->mIsViewChanged = true;
    mCurrentCameraState->update();
}

bool RendererBase::updateCamera(int frame)
{
    if (mParams.mUseTurntable) {
        int count = (mParams.mFrameCount == 0) ? 1 : mParams.mFrameCount;
        mCurrentCameraState->mCameraRotation[1] = (frame * 2.0f * M_PI) / count;
        mCurrentCameraState->mIsViewChanged = true;
    }
    return mCurrentCameraState->update();
}

bool RendererBase::saveFrameBuffer(bool useFrame, int frame)
{
    assert(mFrameBuffer);

    if (mParams.mOutputPrefix.empty()) {
        //std::cerr << "Output prefix must be specified on command-line to take screenshots." << std::endl;
        return false;
    }

    std::stringstream ss;
    if (useFrame) {
        ss << mParams.mOutputPrefix << '.' << std::setfill('0') << std::setw(4) << frame << std::setfill('\0') << ".pfm";
    } else {
        ss << mParams.mOutputPrefix << ".pfm";
    }
    return mFrameBuffer->save(ss.str().c_str());
}

float RendererBase::computePSNR(FrameBufferBase& other)
{
    return mFrameBuffer->computePSNR(other);
}
