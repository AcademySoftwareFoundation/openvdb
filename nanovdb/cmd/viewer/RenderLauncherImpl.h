// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncherImpl.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Declaration of Grid Renderer implementations.
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
#include <chrono>
#include "RenderLauncher.h"
#include "RenderFogVolumeUtils.h"
#include "RenderLevelSetUtils.h"
#include "RenderGridUtils.h"
#include "RenderVoxelUtils.h"
#include "RenderPointsUtils.h"

class FrameBufferBase;

template<MaterialClass>
struct LauncherForType;

inline bool gridIsClass(const void* gridPtr, const nanovdb::GridClass gridclass)
{
    if (!gridPtr)
        return false;
    auto gridHdl = reinterpret_cast<const nanovdb::GridHandle<>*>(gridPtr);
    return gridHdl->gridMetaData()->gridClass() == gridclass;
}

inline bool gridIsType(const void* gridPtr, const nanovdb::GridType gridType)
{
    if (!gridPtr)
        return false;
    auto gridHdl = reinterpret_cast<const nanovdb::GridHandle<>*>(gridPtr);
    return gridHdl->gridMetaData()->gridType() == gridType;
}

template<>
struct LauncherForType<MaterialClass::kGrid>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters& gridParams, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        auto gridBounds = gridParams.bounds;

        if (gridBounds.empty()) {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
            return;
        }

        if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Float)) {
            auto grid = launcher.template grid<float>(gridParams.gridHandle);
            launcher.render(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Double)) {
            auto grid = launcher.template grid<double>(gridParams.gridHandle);
            launcher.render(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::UInt32)) {
            auto grid = launcher.template grid<uint32_t>(gridParams.gridHandle);
            launcher.render(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::PackedRGBA8)) {
            auto grid = launcher.template grid<nanovdb::PackedRGBA8>(gridParams.gridHandle);
            launcher.render(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Int64)) {
            auto grid = launcher.template grid<int64_t>(gridParams.gridHandle);
            launcher.render(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Vec3f)) {
            auto grid = launcher.template grid<nanovdb::Vec3f>(gridParams.gridHandle);
            launcher.render(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Vec3d)) {
            auto grid = launcher.template grid<nanovdb::Vec3d>(gridParams.gridHandle);
            launcher.render(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
        }
    }
};

template<>
struct LauncherForType<MaterialClass::kVoxels>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters& gridParams, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        auto gridBounds = gridParams.bounds;

        if (gridBounds.empty()) {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
            return;
        }

        if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Float)) {
            auto grid = launcher.template grid<float>(gridParams.gridHandle);
            launcher.render(width, height, render::voxel::RenderVoxelsRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Double)) {
            auto grid = launcher.template grid<double>(gridParams.gridHandle);
            launcher.render(width, height, render::voxel::RenderVoxelsRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::UInt32)) {
            auto grid = launcher.template grid<uint32_t>(gridParams.gridHandle);
            launcher.render(width, height, render::voxel::RenderVoxelsRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::PackedRGBA8)) {
            auto grid = launcher.template grid<nanovdb::PackedRGBA8>(gridParams.gridHandle);
            launcher.render(width, height, render::voxel::RenderVoxelsRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Int64)) {
            auto grid = launcher.template grid<int64_t>(gridParams.gridHandle);
            launcher.render(width, height, render::voxel::RenderVoxelsRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Vec3f)) {
            auto grid = launcher.template grid<nanovdb::Vec3f>(gridParams.gridHandle);
            launcher.render(width, height, render::voxel::RenderVoxelsRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Vec3d)) {
            auto grid = launcher.template grid<nanovdb::Vec3d>(gridParams.gridHandle);
            launcher.render(width, height, render::voxel::RenderVoxelsRgba32fFn(), imgPtr, numAccumulations, grid, sceneParams, materialParams);
        } else {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
        }
    }
};

template<>
struct LauncherForType<MaterialClass::kFogVolumePathTracer>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters& densityGridParams, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        auto densityBounds = densityGridParams.bounds;

        if (densityBounds.empty()) {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
            return;
        }

        if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Float)) {
            auto densityGrid = launcher.template grid<float>(densityGridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<float, 1>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<float, 0>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            }
        } else if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Double)) {
            auto densityGrid = launcher.template grid<double>(densityGridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<double, 1>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<double, 0>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            }
        } else if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Vec3f)) {
            auto densityGrid = launcher.template grid<nanovdb::Vec3f>(densityGridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<nanovdb::Vec3f, 1>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<nanovdb::Vec3f, 0>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            }
        } else if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Vec3d)) {
            auto densityGrid = launcher.template grid<nanovdb::Vec3d>(densityGridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<nanovdb::Vec3d, 1>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::fogvolume::RenderVolumeRgba32fFn<nanovdb::Vec3d, 0>(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams); break;
            }
        } else {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
        }
    }
};

template<>
struct LauncherForType<MaterialClass::kBlackBodyVolumePathTracer>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters& densityGridParams, const GridRenderParameters& temperatureGridParams, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        auto densityBounds = densityGridParams.bounds;

        if (densityBounds.empty()) {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
            return;
        }

        if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Float)) {
            auto densityGrid = launcher.template grid<float>(densityGridParams.gridHandle);
            auto temperatureGrid = launcher.template grid<float>(temperatureGridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::fogvolume::RenderBlackBodyVolumeRgba32fFn<float, 1>(), imgPtr, numAccumulations, densityBounds, densityGrid, temperatureGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::fogvolume::RenderBlackBodyVolumeRgba32fFn<float, 0>(), imgPtr, numAccumulations, densityBounds, densityGrid, temperatureGrid, sceneParams, materialParams); break;
            }
        } else if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Double)) {
            auto densityGrid = launcher.template grid<double>(densityGridParams.gridHandle);
            auto temperatureGrid = launcher.template grid<double>(temperatureGridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::fogvolume::RenderBlackBodyVolumeRgba32fFn<double, 1>(), imgPtr, numAccumulations, densityBounds, densityGrid, temperatureGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::fogvolume::RenderBlackBodyVolumeRgba32fFn<double, 0>(), imgPtr, numAccumulations, densityBounds, densityGrid, temperatureGrid, sceneParams, materialParams); break;
            }
        } else {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
        }
    }
};

template<>
struct LauncherForType<MaterialClass::kFogVolumeFast>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters& densityGridParams, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        auto densityBounds = densityGridParams.bounds;

        if (densityBounds.empty()) {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
            return;
        }

        if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Float)) {
            auto densityGrid = launcher.template grid<float>(densityGridParams.gridHandle);
            launcher.render(width, height, render::fogvolume::FogVolumeFastRenderFn(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams);
        } else if (gridIsType(densityGridParams.gridHandle, nanovdb::GridType::Double)) {
            auto densityGrid = launcher.template grid<double>(densityGridParams.gridHandle);
            launcher.render(width, height, render::fogvolume::FogVolumeFastRenderFn(), imgPtr, numAccumulations, densityBounds, densityGrid, sceneParams, materialParams);
        } else {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
        }
    }
};

template<>
struct LauncherForType<MaterialClass::kLevelSetFast>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters& gridParams, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        auto gridBounds = gridParams.bounds;

        if (gridBounds.empty()) {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
            return;
        }

        if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Float)) {
            auto lsGrid = launcher.template grid<float>(gridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::levelset::RenderLevelSetRgba32fFn<float, 1>(), imgPtr, numAccumulations, gridBounds, lsGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::levelset::RenderLevelSetRgba32fFn<float, 0>(), imgPtr, numAccumulations, gridBounds, lsGrid, sceneParams, materialParams); break;
            }
        } else if (gridIsType(gridParams.gridHandle, nanovdb::GridType::Double)) {
            auto lsGrid = launcher.template grid<double>(gridParams.gridHandle);
            switch (materialParams.interpolationOrder) {
            default:
            case 1: launcher.render(width, height, render::levelset::RenderLevelSetRgba32fFn<double, 1>(), imgPtr, numAccumulations, gridBounds, lsGrid, sceneParams, materialParams); break;
            case 0: launcher.render(width, height, render::levelset::RenderLevelSetRgba32fFn<double, 0>(), imgPtr, numAccumulations, gridBounds, lsGrid, sceneParams, materialParams); break;
            }
        } else {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
        }
    }
};

template<>
struct LauncherForType<MaterialClass::kPointsFast>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters& gridParams, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        auto gridBounds = gridParams.bounds;

        if (gridBounds.empty()) {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
            return;
        }

        if (gridIsType(gridParams.gridHandle, nanovdb::GridType::UInt32)) { // && gridIsClass(gridParams.gridHandle, nanovdb::GridClass::PointIndex)) {
            auto pointGrid = launcher.template grid<uint32_t>(gridParams.gridHandle);
            launcher.render(width, height, render::points::RenderPointsRgba32fFn(), imgPtr, numAccumulations, pointGrid, sceneParams, materialParams);
        } else {
            launcher.render(width, height, render::RenderEnvRgba32fFn(), imgPtr, numAccumulations, sceneParams, materialParams);
        }
    }
};

template<>
struct LauncherForType<MaterialClass::kCameraDiagnostic>
{
    template<typename LauncherT>
    void operator()(LauncherT launcher, int width, int height, float* imgPtr, int numAccumulations, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams) const
    {
        launcher.render(width, height, render::CameraDiagnosticRenderer(), imgPtr, numAccumulations, sceneParams, materialParams);
    }
};

template<typename LauncherT>
void launchRender(LauncherT& methodLauncher, MaterialClass method, int width, int height, float* imgPtr, int numAccumulations, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams)
{
    if (method == MaterialClass::kGrid) {
        LauncherForType<MaterialClass::kGrid>()(methodLauncher, width, height, imgPtr, numAccumulations, grids[0], sceneParams, materialParams);
    } else if (method == MaterialClass::kVoxels) {
        LauncherForType<MaterialClass::kVoxels>()(methodLauncher, width, height, imgPtr, numAccumulations, grids[0], sceneParams, materialParams);
    } else if (method == MaterialClass::kFogVolumePathTracer) {
        LauncherForType<MaterialClass::kFogVolumePathTracer>()(methodLauncher, width, height, imgPtr, numAccumulations, grids[0], sceneParams, materialParams);
    } else if (method == MaterialClass::kLevelSetFast) {
        LauncherForType<MaterialClass::kLevelSetFast>()(methodLauncher, width, height, imgPtr, numAccumulations, grids[0], sceneParams, materialParams);
    } else if (method == MaterialClass::kPointsFast) {
        LauncherForType<MaterialClass::kPointsFast>()(methodLauncher, width, height, imgPtr, numAccumulations, grids[0], sceneParams, materialParams);
    } else if (method == MaterialClass::kBlackBodyVolumePathTracer) {
        LauncherForType<MaterialClass::kBlackBodyVolumePathTracer>()(methodLauncher, width, height, imgPtr, numAccumulations, grids[0], grids[1], sceneParams, materialParams);
    } else if (method == MaterialClass::kFogVolumeFast) {
        LauncherForType<MaterialClass::kFogVolumeFast>()(methodLauncher, width, height, imgPtr, numAccumulations, grids[0], sceneParams, materialParams);
    } else if (method == MaterialClass::kCameraDiagnostic) {
        LauncherForType<MaterialClass::kCameraDiagnostic>()(methodLauncher, width, height, imgPtr, numAccumulations, sceneParams, materialParams);
    }
}

class RenderLauncherCpu : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats) override;

    std::string name() const override { return "host"; }

    int getPriority() const override { return 0; }
};

class RenderLauncherCpuMT : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats) override;

    std::string name() const override { return "host-mt"; }

    int getPriority() const override { return 10; }
};

#if defined(NANOVDB_USE_CUDA)
class RenderLauncherCUDA : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats) override;

    std::string name() const override { return "cuda"; }

    int getPriority() const override { return 20; }

    ~RenderLauncherCUDA() override;

public:
    struct GridResource
    {
        bool                                  mInitialized = false;
        std::chrono::steady_clock::time_point mLastUsedTime;
        void*                                 mDeviceGrid = nullptr;
    };

    struct ImageResource
    {
        bool   mInitialized = false;
        bool   mGlTextureResourceCUDAError = false;
        void*  mGlTextureResourceCUDA = nullptr;
        size_t mGlTextureResourceSize = 0;
        int    mGlTextureResourceId = 0;
    };

    std::shared_ptr<ImageResource> ensureImageResource();
    std::shared_ptr<GridResource>  ensureGridResource(const nanovdb::GridHandle<>* gridHdl);
    void*                          mapCUDA(int access, const std::shared_ptr<ImageResource>& resource, FrameBufferBase* imgBuffer, void* stream = 0);
    void                           unmapCUDA(const std::shared_ptr<ImageResource>& resource, FrameBufferBase* imgBuffer, void* stream = 0);

    std::shared_ptr<ImageResource>                       mImageResource;
    std::map<const void*, std::shared_ptr<GridResource>> mGridResources;
};
#endif

#if defined(NANOVDB_USE_OPTIX) && defined(NANOVDB_USE_CUDA)
class RenderLauncherOptix : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats) override;

    std::string name() const override { return "optix"; }

    int getPriority() const override { return 19; }

    ~RenderLauncherOptix() override;

private:
    struct Resource
    {
        ~Resource();

        bool          mInitialized = false;
        void*         mDeviceGrid = nullptr;
        void*         mGlTextureResourceCUDA = nullptr;
        size_t        mGlTextureResourceSize = 0;
        int           mGlTextureResourceId = 0;
        void*         mOptixRenderState = nullptr;
        MaterialClass mMaterialClass = MaterialClass::kAuto;
    };

    std::shared_ptr<Resource> ensureResource(const nanovdb::GridHandle<>& gridHdl, MaterialClass renderMethod);
    void*                     mapCUDA(int access, const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream = 0);
    void                      unmapCUDA(const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream = 0);

    std::map<const nanovdb::GridHandle<>*, std::shared_ptr<Resource>> mResources;
};
#endif

#if defined(NANOVDB_USE_OPENCL)
class RenderLauncherCL : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats) override;

    std::string name() const override { return "opencl"; }

    int getPriority() const override { return 9; }

    ~RenderLauncherCL() override;

private:
    struct Resource
    {
        bool   mInitialized = false;
        void*  mContextCl = nullptr;
        void*  mQueueCl = nullptr;
        void*  mDeviceCl = nullptr;
        void*  mProgramCl = nullptr;
        void*  mKernelLevelSetCl = nullptr;
        void*  mKernelFogVolumeCl = nullptr;
        void*  mGridBuffer = nullptr;
        void*  mNodeLevel0 = nullptr;
        void*  mNodeLevel1 = nullptr;
        void*  mNodeLevel2 = nullptr;
        void*  mRootData = nullptr;
        void*  mRootDataTiles = nullptr;
        void*  mGridData = nullptr;
        void*  mGlTextureResourceCL = nullptr;
        size_t mGlTextureResourceSize = 0;
    };

    std::shared_ptr<Resource> ensureResource(const nanovdb::GridHandle<>& gridHdl, void* glContext, void* glDisplay);
    void*                     mapCL(int access, const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer);
    void                      unmapCL(const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer);

    std::map<const nanovdb::GridHandle<>*, std::shared_ptr<Resource>> mResources;
};
#endif

#if defined(NANOVDB_USE_OPENGL)
class RenderLauncherGL : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats) override;

    std::string name() const override { return "glsl"; }

    int getPriority() const override { return 8; }

    ~RenderLauncherGL() override;

private:
    struct Resource
    {
        bool          mInitialized = false;
        MaterialClass mMethod = MaterialClass::kAuto;
        uint32_t      mBufferId = 0;
        uint32_t      mUniformBufferId = 0;
        uint32_t      mUniformBufferSize = 0;
        uint32_t      mProgramId = 0;
        uint32_t      mUniformBufferBindIndex = 0;
    };

    std::shared_ptr<Resource> ensureResource(const nanovdb::GridHandle<>& gridHdl, void* glContext, void* glDisplay, MaterialClass method);
    bool                      ensureProgramResource(const std::shared_ptr<Resource>& resource, std::string valueType, MaterialClass method);
    bool                      ensureGridResource(const std::shared_ptr<Resource>& resource, const nanovdb::NanoGrid<float>* grid, size_t gridByteSize);

    std::map<const nanovdb::GridHandle<>*, std::shared_ptr<Resource>> mResources;
};
#endif

class RenderLauncherC99 : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats) override;

    std::string name() const override { return "host-c99"; }

    int getPriority() const override { return 1; }
};
