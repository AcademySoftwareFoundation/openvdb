// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncherCUDA.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of CUDA-platform Grid renderer.
*/

#ifdef NANOVDB_USE_CUDA

#include "RenderLauncherImpl.h"
#include "RenderFogVolumeUtils.h"
#include "RenderLevelSetUtils.h"
#include "RenderGridUtils.h"
#include "RenderPointsUtils.h"
#include "FrameBufferHost.h"
#if defined(NANOVDB_USE_OPENGL)
#include "FrameBufferGL.h"
#endif

#if defined(__CUDACC__)
#if defined(NANOVDB_USE_OPENGL)
#include <cuda_gl_interop.h>
#endif
#include <cuda_runtime_api.h>
#include <iostream>

#define NANOVDB_CUDA_SAFE_CALL(x) checkCUDA(x, __FILE__, __LINE__)

static bool checkCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        return false;
    }
    return true;
}

template<typename FnT, typename... Args>
__global__ void launchRenderKernel(int width, int height, FnT fn, Args... args)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) {
        return;
    }
    fn(ix, iy, width, height, args...);
}

template<typename FnT, typename... Args>
static void launchRender(int width, int height, FnT fn, Args... args)
{
    auto       divRoundUp = [](int a, int b) { return (a + b - 1) / b; };
    const dim3 threadsPerBlock(8, 8), numBlocks(divRoundUp(width, threadsPerBlock.x), divRoundUp(height, threadsPerBlock.y));
    launchRenderKernel<<<numBlocks, threadsPerBlock, 0, 0>>>(width, height, fn, args...);
    auto code = cudaGetLastError();
    if (code != cudaSuccess) {
        std::cerr << "launchRenderCuda CUDA error: " << cudaGetErrorString(code) << std::endl;
        exit(code);
    }
}

RenderLauncherCUDA::~RenderLauncherCUDA()
{
    for (auto& it : mResources) {
        cudaFree(it.second->mDeviceGrid);
    }
    mResources.clear();
}

std::shared_ptr<RenderLauncherCUDA::Resource> RenderLauncherCUDA::ensureResource(const nanovdb::GridHandle<>* gridHdl)
{
    std::shared_ptr<Resource> resource;
    auto                      it = mResources.find(gridHdl);
    if (it != mResources.end()) {
        resource = it->second;
    } else {
        std::cout << "Initializing CUDA renderer..." << std::endl;

        resource = std::make_shared<Resource>();
        mResources.insert(std::make_pair(gridHdl, resource));

        NANOVDB_CUDA_SAFE_CALL(cudaMalloc((void**)&resource->mDeviceGrid, gridHdl->size()));
        NANOVDB_CUDA_SAFE_CALL(cudaMemcpy(resource->mDeviceGrid, gridHdl->data(), gridHdl->size(), cudaMemcpyHostToDevice));

        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(code) << std::endl;
            return nullptr;
        }

        resource->mInitialized = true;
    }

    return resource;
}

void RenderLauncherCUDA::unmapCUDA(const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream)
{
#if defined(NANOVDB_USE_OPENGL)
    auto imgBufferGL = dynamic_cast<FrameBufferGL*>(imgBuffer);
    if (imgBufferGL) {
        NANOVDB_CUDA_SAFE_CALL(cudaGraphicsUnmapResources(
            1, (cudaGraphicsResource**)&resource->mGlTextureResourceCUDA, (cudaStream_t)stream));
        imgBuffer->invalidate();
        return;
    }
#endif

    imgBuffer->cudaUnmap(stream);
    imgBuffer->invalidate();
}

void* RenderLauncherCUDA::mapCUDA(int access, const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream)
{
    if (!imgBuffer->size()) {
        return nullptr;
    }

#if defined(NANOVDB_USE_OPENGL)
    auto imgBufferGL = dynamic_cast<FrameBufferGL*>(imgBuffer);
    if (imgBufferGL) {
        auto     accessGL = FrameBufferBase::AccessType(access);
        uint32_t accessCUDA = cudaGraphicsMapFlagsNone;
        if (accessGL == FrameBufferBase::AccessType::READ_ONLY) {
            accessCUDA = cudaGraphicsMapFlagsReadOnly;
        } else if (accessGL == FrameBufferBase::AccessType::WRITE_ONLY) {
            accessCUDA = cudaGraphicsMapFlagsWriteDiscard;
        }

        if (!resource->mGlTextureResourceCUDA || resource->mGlTextureResourceId != imgBufferGL->resourceId()) {
            std::cout << "registering GL resource(" << imgBufferGL->bufferGL() << ") [" << imgBufferGL->resourceId() << "] for CUDA. (" << imgBuffer->size() << "B)" << std::endl;

            if (stream) {
                NANOVDB_CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStream_t(stream)));
            } else {
                NANOVDB_CUDA_SAFE_CALL(cudaDeviceSynchronize());
            }

            NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
            NANOVDB_GL_SAFE_CALL(glFinish());

            if (resource->mGlTextureResourceCUDA) {
                std::cout << "unregistering GL resource [" << imgBufferGL->resourceId() << "] for CUDA." << std::endl;

                NANOVDB_CUDA_SAFE_CALL(cudaGraphicsUnregisterResource((cudaGraphicsResource*)resource->mGlTextureResourceCUDA));
                resource->mGlTextureResourceCUDA = nullptr;
                resource->mGlTextureResourceSize = 0;
                resource->mGlTextureResourceId = 0;
            }

            NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imgBufferGL->bufferGL()));

            cudaGraphicsResource* resCUDA = nullptr;
            bool                  rc =
                NANOVDB_CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&resCUDA,
                                                                    imgBufferGL->bufferGL(),
                                                                    accessCUDA));
            if (!rc) {
                std::cerr << "Can't register GL buffer (" << imgBufferGL->bufferGL() << ") with CUDA" << std::endl;
                exit(1);

                return nullptr;
            }

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            resource->mGlTextureResourceCUDA = resCUDA;
            resource->mGlTextureResourceId = imgBufferGL->resourceId();
            resource->mGlTextureResourceSize = imgBuffer->size();
        }

        cudaGraphicsResource* resCUDA = (cudaGraphicsResource*)resource->mGlTextureResourceCUDA;

        NANOVDB_CUDA_SAFE_CALL(cudaGraphicsResourceSetMapFlags(resCUDA, accessCUDA));

        void*  ptr = nullptr;
        size_t size = 0;
        NANOVDB_CUDA_SAFE_CALL(cudaGraphicsMapResources(
            1, (cudaGraphicsResource**)&resCUDA, (cudaStream_t)stream));
        NANOVDB_CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
            &ptr, &size, resCUDA));
        assert(size == imgBuffer->size());

        NANOVDB_GL_SAFE_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
        return ptr;
    }
#endif
    return imgBuffer->cudaMap(FrameBufferBase::AccessType(access));
}

bool RenderLauncherCUDA::render(RenderMethod method, int width, int height, FrameBufferBase* imgBuffer, Camera<float> camera, const nanovdb::GridHandle<>& gridHdl, int numAccumulations, const RenderConstants& params, RenderStatistics* stats)
{
    auto resource = ensureResource(&gridHdl);
    if (!resource || !resource->mInitialized) {
        return false;
    }

    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    float* imgPtr = (float*)mapCUDA(
        (int)((numAccumulations > 0) ? FrameBufferBase::AccessType::READ_WRITE : FrameBufferBase::AccessType::WRITE_ONLY),
        resource,
        imgBuffer);

    if (!imgPtr) {
        return false;
    }

    if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Float) {
        auto deviceGrid = reinterpret_cast<const nanovdb::NanoGrid<float>*>(resource->mDeviceGrid);

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        } else if (method == RenderMethod::LEVELSET) {
            launchRender(width, height, render::levelset::RenderLevelSetRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Double) {
        auto deviceGrid = reinterpret_cast<const nanovdb::NanoGrid<double>*>(resource->mDeviceGrid);

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        } else if (method == RenderMethod::LEVELSET) {
            launchRender(width, height, render::levelset::RenderLevelSetRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::UInt32) {
        auto deviceGrid = reinterpret_cast<const nanovdb::NanoGrid<uint32_t>*>(resource->mDeviceGrid);

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        } else if (method == RenderMethod::POINTS) {
            launchRender(width, height, render::points::RenderPointsRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Vec3f) {
        auto deviceGrid = reinterpret_cast<const nanovdb::NanoGrid<nanovdb::Vec3f>*>(resource->mDeviceGrid);

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Vec3d) {
        auto deviceGrid = reinterpret_cast<const nanovdb::NanoGrid<nanovdb::Vec3d>*>(resource->mDeviceGrid);

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        } else if (method == RenderMethod::FOG_VOLUME) {
            launchRender(width, height, render::fogvolume::RenderVolumeRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Int64) {
        auto deviceGrid = reinterpret_cast<const nanovdb::NanoGrid<int64_t>*>(resource->mDeviceGrid);

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        }
    } else if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Int32) {
        auto deviceGrid = reinterpret_cast<const nanovdb::NanoGrid<int32_t>*>(resource->mDeviceGrid);

        if (method == RenderMethod::GRID) {
            launchRender(width, height, render::grid::RenderGridRgba32fFn(), imgPtr, camera, deviceGrid, numAccumulations, params);
        }
    }

    unmapCUDA(resource, imgBuffer);

    if (stats) {
        cudaDeviceSynchronize();
        auto t1 = ClockT::now();
        stats->mDuration = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/1000.f;
    }
    
    return true;
}

#endif
#endif // NANOVDB_USE_CUDA
