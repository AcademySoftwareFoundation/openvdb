// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file FrameBufferHost.cpp
	\brief Implementation of FrameBufferHost.
*/

#include "FrameBufferHost.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#if defined(NANOVDB_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(NANOVDB_USE_OPENCL) && 0
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#if defined(NANOVDB_USE_CUDA)

#define NANOVDB_CUDA_SAFE_CALL(x) checkCUDA(x, __FILE__, __LINE__)

static bool checkCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        return false;
    }
    return true;
}

#endif

bool FrameBufferHost::cleanup()
{
    if (mBuffer) {
#if defined(NANOVDB_USE_CUDA)
        cudaFreeHost(mBuffer);
#else
        free(mBuffer);
#endif
    }
    mBuffer = nullptr;

    mWidth = 0;
    mHeight = 0;
    mElementSize = 0;
    mSize = 0;
    mIndex = 0;
    mInternalFormat = InternalFormat::UNKNOWN;
    mBufferUpdateId = 0;

    return true;
}

bool FrameBufferHost::setup(int w, int h, InternalFormat format)
{
    mElementSize = formatGetElementSize(format);
    mWidth = w;
    mHeight = h;
    mSize = w * h * mElementSize;
#if defined(NANOVDB_USE_CUDA)
    NANOVDB_CUDA_SAFE_CALL(cudaMallocHost(&mBuffer, mSize));
#else
    mBuffer = malloc(mSize);
#endif
    return true;
}

bool FrameBufferHost::render(int /*x*/, int /*y*/, int /*w*/, int /*h*/)
{
    return false;
}

void* FrameBufferHost::map(AccessType /*access*/)
{
    return mBuffer;
}

void FrameBufferHost::unmap()
{
}

void* FrameBufferHost::cudaMap(AccessType /*access*/, void* /*stream*/)
{
    //int writeIndex = (mIndex + 1) % BUFFER_COUNT;

    if (!mSize)
        return nullptr;

    void* ptr = mBuffer;
    return ptr;
}

void FrameBufferHost::cudaUnmap(void* stream)
{
    //int writeIndex = (mIndex + 1) % BUFFER_COUNT;
#if defined(NANOVDB_USE_CUDA)
    NANOVDB_CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStream_t(stream)));
#else
    (void)stream;
#endif
    invalidate();
}

void* FrameBufferHost::clMap(AccessType access, void* stream)
{
#if defined(NANOVDB_USE_OPENCL) && 0

    int writeIndex = (mIndex + 1) % BUFFER_COUNT;

    if (!mSize)
        return nullptr;

    cl_mem memCL = cl_mem(mBufferResourcesCL[writeIndex]);
    cl_int err = clEnqueueAcquireGLObjects(cl_command_queue(stream), 1, &memCL, 0, 0, NULL);

    if (err != CL_SUCCESS)
        return nullptr;

    return (void*)memCL;
#else
    (void)access;
    (void)stream;
    return nullptr;
#endif
}

void FrameBufferHost::clUnmap(void* stream)
{
#if defined(NANOVDB_USE_OPENCL) && 0
    int    writeIndex = (mIndex + 1) % BUFFER_COUNT;
    cl_mem bufferCL = cl_mem(mBufferResourcesCL[writeIndex]);
    cl_int err = clEnqueueReleaseGLObjects(cl_command_queue(stream), 1, &bufferCL, 0, 0, NULL);
    invalidate();
#else
    (void)stream;
#endif
}
