// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncher.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Declaration of Grid Render-platform manager class.
*/

#pragma once
#include <nanovdb/util/Ray.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <memory>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

#include "RenderConstants.h"

class FrameBufferBase;

enum class MaterialClass { kAuto = 0,
                           kLevelSetFast,
                           kFogVolumePathTracer,
                           kGrid,
                           kPointsFast,
                           kBlackBodyVolumePathTracer,
                           kFogVolumeFast,
                           kCameraDiagnostic,
                           kNumTypes };

extern const char* kMaterialClassTypeStrings[(int)MaterialClass::kNumTypes];
extern const char* kCameraLensTypeStrings[(int)Camera::LensType::kNumTypes];

class RenderLauncherImplBase
{
public:
    virtual ~RenderLauncherImplBase() {}

    virtual std::string name() const = 0;

    virtual int getPriority() const { return 0; }

    virtual bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats = nullptr) = 0;
};

class RenderLauncher
{
public:
    RenderLauncher();

    std::string name() const;

    void setPlatform(int index) { mIndex = std::max(std::min(index, (int)mImpls.size() - 1), 0); }

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats);

    int size() const { return (int)mImpls.size(); }

    int         getPlatformIndexFromName(std::string name) const;
    std::string getNameForPlatformIndex(int i) const;

    std::vector<std::string> getPlatformNames() const;

private:
    int                                                  mIndex;
    std::vector<std::shared_ptr<RenderLauncherImplBase>> mImpls;
};

inline bool RenderLauncher::render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats)
{
    return mImpls[mIndex]->render(method, width, height, imgBuffer, numAccumulations, numGrids, grids, sceneParams, materialParams, stats);
}
