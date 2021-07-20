// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncher.cpp
	\brief Implementation of Grid Render-platform manager.
*/

#include "RenderLauncher.h"
#include "RenderFogVolumeUtils.h"
#include "RenderLevelSetUtils.h"
#include "RenderGridUtils.h"
#include "RenderPointsUtils.h"
#include "FrameBuffer.h"
#include "RenderLauncherImpl.h"

const char* kMaterialClassTypeStrings[(int)MaterialClass::kNumTypes] = {"Auto", "LevelSetFast", "FogVolumePathTracer", "Grid", "PointsFast", "BlackBodyVolumePathTracer", "FogVolumeFast", "Voxels", "CameraDiagnostic"};
const char* kCameraLensTypeStrings[(int)Camera::LensType::kNumTypes] = {"PinHole", "Spherical", "ODS"};

RenderLauncher::RenderLauncher()
    : mIndex(0)
{
    mImpls.push_back(std::make_shared<RenderLauncherCpu>());
    mImpls.push_back(std::make_shared<RenderLauncherCpuMT>());
    mImpls.push_back(std::make_shared<RenderLauncherC99>());

#if defined(NANOVDB_USE_OPENGL)
    mImpls.push_back(std::make_shared<RenderLauncherGL>());
#endif
#if defined(NANOVDB_USE_CUDA)
    mImpls.push_back(std::make_shared<RenderLauncherCUDA>());
#endif
#if defined(NANOVDB_USE_OPENCL)
    mImpls.push_back(std::make_shared<RenderLauncherCL>());
#endif
#if defined(NANOVDB_USE_OPTIX) && defined(NANOVDB_USE_CUDA)
    mImpls.push_back(std::make_shared<RenderLauncherOptix>());
#endif

    std::sort(mImpls.begin(), mImpls.end(), [](const std::shared_ptr<RenderLauncherImplBase>& a, const std::shared_ptr<RenderLauncherImplBase>& b) {
        return a->getPriority() > b->getPriority();
    });
}

std::string RenderLauncher::getNameForPlatformIndex(int i) const
{
    auto names = getPlatformNames();
    if (i < 0 || i >= (int)names.size())
        return "";
    return names[i];
}

int RenderLauncher::getPlatformIndexFromName(std::string name) const
{
    int  found = -1;
    auto names = getPlatformNames();
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == name) {
            found = (int)i;
            break;
        }
    }
    return found;
}

std::vector<std::string> RenderLauncher::getPlatformNames() const
{
    std::vector<std::string> names;
    for (auto& it : mImpls) {
        names.push_back(it->name());
    }
    return names;
}

std::string RenderLauncher::name() const
{
    return mImpls[mIndex]->name();
}

struct PlatformLauncherCpu
{
    template<typename ValueT>
    const nanovdb::NanoGrid<ValueT>* grid(const void* gridPtr)
    {
        auto gridHdl = reinterpret_cast<const nanovdb::GridHandle<>*>(gridPtr);
        if (!gridHdl)
            return nullptr;
        return gridHdl->grid<ValueT>();
    }

    template<typename FnT, typename... Args>
    bool render(int width, int height, const FnT& fn, Args... args) const
    {
        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                fn(ix, iy, width, height, args...);
            }
        }

        return true;
    }
};

bool RenderLauncherCpu::render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int /*numGrids*/, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats)
{
    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    float* imgPtr = (float*)imgBuffer->map((numAccumulations > 0) ? FrameBufferBase::AccessType::READ_WRITE : FrameBufferBase::AccessType::WRITE_ONLY);

    if (!imgPtr) {
        return false;
    }

    PlatformLauncherCpu methodLauncher;
    launchRender(methodLauncher, method, width, height, imgPtr, numAccumulations, grids, sceneParams, materialParams);

    imgBuffer->unmap();

    if (stats) {
        auto t1 = ClockT::now();
        stats->mDuration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    }

    return true;
}
