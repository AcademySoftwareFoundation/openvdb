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

class FrameBufferBase;

class RenderLauncherCpu : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats) override;

    std::string name() const override { return "host"; }

    int getPriority() const override { return 0; }
};

class RenderLauncherCpuMT : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats) override;

    std::string name() const override { return "host-mt"; }

    int getPriority() const override { return 10; }
};

#if defined(NANOVDB_USE_CUDA)
class RenderLauncherCUDA : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats) override;

    std::string name() const override { return "cuda"; }

    int getPriority() const override { return 20; }

    ~RenderLauncherCUDA() override;

private:
    struct Resource
    {
        bool   mInitialized = false;
        void*  mDeviceGrid = nullptr;
        void*  mGlTextureResourceCUDA = nullptr;
        size_t mGlTextureResourceSize = 0;
        int    mGlTextureResourceId = 0;
    };

    std::shared_ptr<Resource> ensureResource(const nanovdb::GridHandle<>* gridHdl);
    void*                     mapCUDA(int access, const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream = 0);
    void                      unmapCUDA(const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream = 0);

    std::map<const void*, std::shared_ptr<Resource>> mResources;
};
#endif

#if defined(NANOVDB_USE_OPTIX) && defined(NANOVDB_USE_CUDA)
class RenderLauncherOptix : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats) override;

    std::string name() const override { return "optix"; }

    int getPriority() const override { return 19; }

    ~RenderLauncherOptix() override;

private:
    struct Resource
    {
        ~Resource();

        bool         mInitialized = false;
        void*        mDeviceGrid = nullptr;
        void*        mGlTextureResourceCUDA = nullptr;
        size_t       mGlTextureResourceSize = 0;
        int          mGlTextureResourceId = 0;
        void*        mOptixRenderState = nullptr;
        RenderMethod mRenderMethod = RenderMethod::AUTO;
    };

    std::shared_ptr<Resource> ensureResource(const nanovdb::GridHandle<>& gridHdl, RenderMethod renderMethod);
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

    bool render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats) override;

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

    bool render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats) override;

    std::string name() const override { return "glsl"; }

    int getPriority() const override { return 8; }

    ~RenderLauncherGL() override;

private:
    struct Resource
    {
        bool         mInitialized = false;
        RenderMethod mMethod = RenderMethod::AUTO;
        uint32_t     mBufferId = 0;
        uint32_t     mUniformBufferId = 0;
        uint32_t     mUniformBufferSize = 0;
        uint32_t     mProgramId = 0;
        uint32_t     mUniformBufferBindIndex = 0;
    };

    std::shared_ptr<Resource> ensureResource(const nanovdb::GridHandle<>& gridHdl, void* glContext, void* glDisplay, RenderMethod method);
    bool                      ensureProgramResource(const std::shared_ptr<Resource>& resource, std::string valueType, RenderMethod method);
    bool                      ensureGridResource(const std::shared_ptr<Resource>& resource, const nanovdb::NanoGrid<float>* grid, size_t gridByteSize);

    std::map<const nanovdb::GridHandle<>*, std::shared_ptr<Resource>> mResources;
};
#endif

class RenderLauncherC99 : public RenderLauncherImplBase
{
public:
    using RenderLauncherImplBase::render;

    bool render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats) override;

    std::string name() const override { return "host-c99"; }

    int getPriority() const override { return 1; }
};
