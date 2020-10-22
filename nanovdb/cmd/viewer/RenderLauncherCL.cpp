// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncherCL.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of OpenCL-platform Grid renderer.
*/

#ifdef NANOVDB_USE_OPENCL

#include "RenderLauncherImpl.h"
#include "FrameBufferHost.h"
#if defined(NANOVDB_USE_OPENGL)
#include "FrameBufferGL.h"
#endif
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenGL/gl.h>
#include <OpenGL/CGLDevice.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenCL/cl_gl_ext.h>
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

static const char* g_kKernelString_Platform_h = "code/CPlatform.h";
static const char* g_kKernelString_HDDA_h = "code/CHDDA.h";
static const char* g_kKernelString_NanoVDB_h = "code/CNanoVDB.h";
static const char* g_kKernelString_Platform_c = "code/CPlatform.c";
static const char* g_kKernelString_NanoVDB_c = "code/CNanoVDB.c";
static const char* g_kKernelString_HDDA_c = "code/CHDDA.c";
static const char* g_kKernelString_renderCommon_h = "code/renderCommon.h";
static const char* g_kKernelString_renderLevelSet_c = "code/renderLevelSet.c";
static const char* g_kKernelString_renderFogVolume_c = "code/renderFogVolume.c";

#define NANOVDB_CL_SAFE_CALL(x) checkCL(x, __FILE__, __LINE__)

#define _CL_ERROR_STR(code) \
    case code: return #code;

char const*
getErrorStringCL(cl_int const err)
{
    switch (err) {
        _CL_ERROR_STR(CL_SUCCESS);
        _CL_ERROR_STR(CL_DEVICE_NOT_FOUND);
        _CL_ERROR_STR(CL_DEVICE_NOT_AVAILABLE);
        _CL_ERROR_STR(CL_COMPILER_NOT_AVAILABLE);
        _CL_ERROR_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        _CL_ERROR_STR(CL_OUT_OF_RESOURCES);
        _CL_ERROR_STR(CL_OUT_OF_HOST_MEMORY);
        _CL_ERROR_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
        _CL_ERROR_STR(CL_MEM_COPY_OVERLAP);
        _CL_ERROR_STR(CL_IMAGE_FORMAT_MISMATCH);
        _CL_ERROR_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        _CL_ERROR_STR(CL_BUILD_PROGRAM_FAILURE);
        _CL_ERROR_STR(CL_MAP_FAILURE);
        _CL_ERROR_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        _CL_ERROR_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        _CL_ERROR_STR(CL_COMPILE_PROGRAM_FAILURE);
        _CL_ERROR_STR(CL_LINKER_NOT_AVAILABLE);
        _CL_ERROR_STR(CL_LINK_PROGRAM_FAILURE);
        _CL_ERROR_STR(CL_DEVICE_PARTITION_FAILED);
        _CL_ERROR_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        _CL_ERROR_STR(CL_INVALID_VALUE);
        _CL_ERROR_STR(CL_INVALID_DEVICE_TYPE);
        _CL_ERROR_STR(CL_INVALID_PLATFORM);
        _CL_ERROR_STR(CL_INVALID_DEVICE);
        _CL_ERROR_STR(CL_INVALID_CONTEXT);
        _CL_ERROR_STR(CL_INVALID_QUEUE_PROPERTIES);
        _CL_ERROR_STR(CL_INVALID_COMMAND_QUEUE);
        _CL_ERROR_STR(CL_INVALID_HOST_PTR);
        _CL_ERROR_STR(CL_INVALID_MEM_OBJECT);
        _CL_ERROR_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        _CL_ERROR_STR(CL_INVALID_IMAGE_SIZE);
        _CL_ERROR_STR(CL_INVALID_SAMPLER);
        _CL_ERROR_STR(CL_INVALID_BINARY);
        _CL_ERROR_STR(CL_INVALID_BUILD_OPTIONS);
        _CL_ERROR_STR(CL_INVALID_PROGRAM);
        _CL_ERROR_STR(CL_INVALID_PROGRAM_EXECUTABLE);
        _CL_ERROR_STR(CL_INVALID_KERNEL_NAME);
        _CL_ERROR_STR(CL_INVALID_KERNEL_DEFINITION);
        _CL_ERROR_STR(CL_INVALID_KERNEL);
        _CL_ERROR_STR(CL_INVALID_ARG_INDEX);
        _CL_ERROR_STR(CL_INVALID_ARG_VALUE);
        _CL_ERROR_STR(CL_INVALID_ARG_SIZE);
        _CL_ERROR_STR(CL_INVALID_KERNEL_ARGS);
        _CL_ERROR_STR(CL_INVALID_WORK_DIMENSION);
        _CL_ERROR_STR(CL_INVALID_WORK_GROUP_SIZE);
        _CL_ERROR_STR(CL_INVALID_WORK_ITEM_SIZE);
        _CL_ERROR_STR(CL_INVALID_GLOBAL_OFFSET);
        _CL_ERROR_STR(CL_INVALID_EVENT_WAIT_LIST);
        _CL_ERROR_STR(CL_INVALID_EVENT);
        _CL_ERROR_STR(CL_INVALID_OPERATION);
        _CL_ERROR_STR(CL_INVALID_GL_OBJECT);
        _CL_ERROR_STR(CL_INVALID_BUFFER_SIZE);
        _CL_ERROR_STR(CL_INVALID_MIP_LEVEL);
        _CL_ERROR_STR(CL_INVALID_GLOBAL_WORK_SIZE);
        _CL_ERROR_STR(CL_INVALID_PROPERTY);
        _CL_ERROR_STR(CL_INVALID_IMAGE_DESCRIPTOR);
        _CL_ERROR_STR(CL_INVALID_COMPILER_OPTIONS);
        _CL_ERROR_STR(CL_INVALID_LINKER_OPTIONS);
        _CL_ERROR_STR(CL_INVALID_DEVICE_PARTITION_COUNT);
    }
    return "Unknown error";
}

#undef _CL_ERROR_STR

bool checkCL(cl_int err, const char* file, const int line)
{
    while (err != CL_SUCCESS) {
        std::stringstream errStringStream;
        errStringStream << err << " (" << getErrorStringCL(err) << ")";
        std::cerr << errStringStream.str().c_str() << " in " << file << ":" << line << std::endl;
        return false;
    }

    return true;
}

RenderLauncherCL::~RenderLauncherCL()
{
    for (auto& it : mResources) {
        if (it.second->mGlTextureResourceCL) {
            NANOVDB_CL_SAFE_CALL(clReleaseMemObject(cl_mem(it.second->mGlTextureResourceCL)));
        }
        if (it.second->mGridBuffer) {
            NANOVDB_CL_SAFE_CALL(clReleaseMemObject(cl_mem(it.second->mGridBuffer)));
        }
        if (it.second->mProgramCl) {
            NANOVDB_CL_SAFE_CALL(clReleaseProgram(cl_program(it.second->mProgramCl)));
        }
        if (it.second->mKernelLevelSetCl) {
            NANOVDB_CL_SAFE_CALL(clReleaseKernel(cl_kernel(it.second->mKernelLevelSetCl)));
        }
        if (it.second->mKernelFogVolumeCl) {
            NANOVDB_CL_SAFE_CALL(clReleaseKernel(cl_kernel(it.second->mKernelFogVolumeCl)));
        }
        if (it.second->mQueueCl) {
            NANOVDB_CL_SAFE_CALL(clReleaseCommandQueue(cl_command_queue(it.second->mQueueCl)));
        }
        if (it.second->mContextCl) {
            NANOVDB_CL_SAFE_CALL(clReleaseContext(cl_context(it.second->mContextCl)));
        }
        if (it.second->mDeviceCl) {
            NANOVDB_CL_SAFE_CALL(clReleaseDevice(cl_device_id(it.second->mDeviceCl)));
        }
    }
    mResources.clear();
}

static std::string fileToString(std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ifstream::binary);
    if (!file) {
        std::stringstream msg;
        msg << "Unable to load code: " << filename;
        //throw std::runtime_error(msg.str().c_str());
        std::cerr << "Error: " << msg.str() << std::endl;
        return "";
    }

    std::string str;
    file.seekg(0, std::ios::end);
    str.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(file)),
               std::istreambuf_iterator<char>());

    return str;
}

std::shared_ptr<RenderLauncherCL::Resource> RenderLauncherCL::ensureResource(const nanovdb::GridHandle<>& gridHdl, void* glContext, void* glDisplay)
{
    std::shared_ptr<Resource> resource;
    auto                      it = mResources.find(&gridHdl);
    if (it != mResources.end()) {
        resource = it->second;
    } else {
        std::cout << "Initializing OpenCL renderer..." << std::endl;

        resource = std::make_shared<Resource>();
        mResources.insert(std::make_pair(&gridHdl, resource));

        cl_int         err;
        cl_device_id   deviceCL = 0;
        cl_platform_id platformCL = 0;
        cl_context     contextCL = 0;

        cl_uint numPlatformsCL = 0;
        clGetPlatformIDs(0, 0, &numPlatformsCL);
        if (numPlatformsCL == 0) {
            std::cerr << "Unable to find an OpenCL platform." << std::endl;
            return nullptr;
        }

        std::vector<cl_platform_id> platformIdsCL(numPlatformsCL);
        NANOVDB_CL_SAFE_CALL(clGetPlatformIDs(numPlatformsCL, platformIdsCL.data(), NULL));

#if defined(__APPLE__)
        CGLContextObj    glContextApple = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(glContextApple);
#endif

        bool useClGetGLContextInfo = false;
#if defined(__APPLE__)
        // only Apple seems to work here.
        // TODO: fix this!
        useClGetGLContextInfo = (glContext && glDisplay);
#endif

        if (useClGetGLContextInfo) {
            // try to find the display GPU by using clGetGLContextInfo...

            // create context from GL context.
            cl_context_properties contextPropsGL[] = {
#if defined(__APPLE__)
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                (cl_context_properties)kCGLShareGroup,
#elif defined(_WIN32)
                CL_GL_CONTEXT_KHR,
                (cl_context_properties)glContext,
                CL_WGL_HDC_KHR,
                (cl_context_properties)glDisplay,
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)platformCL,
#elif defined(__linux__)
                CL_GL_CONTEXT_KHR,
                (cl_context_properties)glContext,
                CL_GLX_DISPLAY_KHR,
                (cl_context_properties)glDisplay,
#endif
                0
            };

#if defined(__APPLE__)
            // create a context using the apple sharegroup.
            contextCL = clCreateContext(contextPropsGL, 0, NULL, NULL, NULL, &err);
            NANOVDB_CL_SAFE_CALL(err);

            if (contextCL) {
                // get the device connected to the display.
                clGetGLContextInfoAPPLE(contextCL, glContextApple, CL_CGL_DEVICE_FOR_CURRENT_VIRTUAL_SCREEN_APPLE, sizeof(cl_device_id), &deviceCL, NULL);
            }
#else
            contextCL = clCreateContext(contextPropsGL, 0, NULL, NULL, NULL, &err);
            NANOVDB_CL_SAFE_CALL(err);

            if (contextCL) {
                clGetGLContextInfoKHR_fn clGetGLContextInfo = NULL;
                if (!clGetGLContextInfo) {
                    clGetGLContextInfo = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
                    if (!clGetGLContextInfo) {
                        std::cerr << "Failed to query proc address for clGetGLContextInfoKHR." << std::endl;
                        return nullptr;
                    }
                }

                size_t numBytes = 0;
                NANOVDB_CL_SAFE_CALL(clGetGLContextInfo(contextPropsGL, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, 0, NULL, &numBytes));
                std::vector<cl_device_id> devices(numBytes / sizeof(cl_device_id));
                if (devices.size() > 0) {
                    NANOVDB_CL_SAFE_CALL(clGetGLContextInfo(contextPropsGL, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, numBytes, devices.data(), NULL));
                    deviceCL = devices[0];
                }
            }
#endif
        } else {
            // try to find a GPU with GL sharing by searching through the platforms...
#if defined(__APPLE__)
            const char* cl_gl_sharing_extension = "cl_APPLE_gl_sharing";
#else
            const char* cl_gl_sharing_extension = "cl_khr_gl_sharing";
#endif

            for (int i = 0; i < platformIdsCL.size(); ++i) {
                const cl_platform_id p = platformIdsCL[i];

                {
                    size_t infoSize = 0;
                    NANOVDB_CL_SAFE_CALL(clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, NULL, &infoSize));
                    std::vector<uint8_t> infoString(infoSize);
                    NANOVDB_CL_SAFE_CALL(clGetPlatformInfo(p, CL_PLATFORM_NAME, infoSize, infoString.data(), NULL));
                    std::cout << "Found OpenCL platform: " << (const char*)infoString.data() << std::endl;
                }

                // get all the GPU devices for this platform...
                cl_uint numGpuDevicesCL = 0;
                NANOVDB_CL_SAFE_CALL(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, 0, &numGpuDevicesCL));
                if (numGpuDevicesCL == 0)
                    continue;
                std::vector<cl_device_id> gpuDevicesCL(numGpuDevicesCL);
                NANOVDB_CL_SAFE_CALL(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, numGpuDevicesCL, gpuDevicesCL.data(), 0));

                for (unsigned int j = 0; j < numGpuDevicesCL; ++j) {
                    size_t extensionsSize = 0;
                    clGetDeviceInfo(gpuDevicesCL[j], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionsSize);
                    std::vector<uint8_t> deviceExtensions(extensionsSize / sizeof(uint8_t));
                    clGetDeviceInfo(gpuDevicesCL[j], CL_DEVICE_EXTENSIONS, extensionsSize, deviceExtensions.data(), NULL);
                    size_t deviceNameSize = 1024;
                    clGetDeviceInfo(gpuDevicesCL[j], CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
                    std::vector<uint8_t> deviceName(deviceNameSize / sizeof(uint8_t));
                    clGetDeviceInfo(gpuDevicesCL[j], CL_DEVICE_NAME, deviceNameSize, deviceName.data(), NULL);

                    if (std::strstr((const char*)deviceExtensions.data(), cl_gl_sharing_extension) != nullptr) {
                        std::cout << "Found OpenCL device: " << (const char*)deviceName.data() << std::endl;

                        if (deviceCL == 0) {
                            // just take the first device.
                            // TODO: better logic needed here!
                            // perhaps find the fastest?
                            deviceCL = gpuDevicesCL[j];
                            platformCL = p;
                        }
                    }
                }
            }

            if (!deviceCL) {
                std::cerr << "Unable to find an OpenCL device." << std::endl;
                return nullptr;
            }

            cl_context_properties* contextProps;

            cl_context_properties contextPropsDefault[] = {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)platformCL,
                0};

            // create context from GL context.
            cl_context_properties contextPropsGL[] = {
#if defined(__APPLE__)
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                (cl_context_properties)kCGLShareGroup,
#elif defined(_WIN32)
                CL_GL_CONTEXT_KHR,
                (cl_context_properties)glContext,
                CL_WGL_HDC_KHR,
                (cl_context_properties)glDisplay,
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)platformCL,
#elif defined(__linux__)
                CL_GL_CONTEXT_KHR,
                (cl_context_properties)glContext,
                CL_GLX_DISPLAY_KHR,
                (cl_context_properties)glDisplay,
#endif
                0
            };

            contextProps = contextPropsDefault;
            if (glContext && glDisplay) {
                contextProps = contextPropsGL;
            }

            contextCL = clCreateContext(contextProps, 1, &deviceCL, NULL, NULL, &err);
            //contextCL = clCreateContextFromType(contextProps, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
            NANOVDB_CL_SAFE_CALL(err);
        }

        if (!deviceCL) {
            std::cerr << "Unable to find an OpenCL device." << std::endl;
            return nullptr;
        } else {
            size_t deviceNameSize = 1024;
            clGetDeviceInfo(deviceCL, CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
            std::vector<uint8_t> deviceName(deviceNameSize / sizeof(uint8_t));
            clGetDeviceInfo(deviceCL, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), NULL);

            std::cout << "Using OpenCL device: " << (const char*)deviceName.data() << std::endl;
        }

        cl_command_queue queue = clCreateCommandQueue(contextCL, deviceCL, 0, &err);
        NANOVDB_CL_SAFE_CALL(err);

        std::string shaderCode;

        const char* valueTypeStr = "float";
        size_t      sizeofValueType = sizeof(float);

        std::stringstream ss;
        ss << std::string("#define CNANOVDB_COMPILER_OPENCL\n");
        ss << std::string("#define VALUETYPE ") << valueTypeStr << "\n";
        ss << std::string("#define SIZEOF_VALUETYPE ") << sizeofValueType << "\n";

        std::string includeFilePath = std::string(__FILE__);
        includeFilePath = includeFilePath.substr(0, includeFilePath.find_last_of('/')).substr(0, includeFilePath.find_last_of('\\')) + "/";

        ss << fileToString(includeFilePath + g_kKernelString_Platform_h);
        ss << fileToString(includeFilePath + g_kKernelString_HDDA_h);
        ss << fileToString(includeFilePath + g_kKernelString_NanoVDB_h);
        ss << fileToString(includeFilePath + g_kKernelString_Platform_c);
        ss << fileToString(includeFilePath + g_kKernelString_NanoVDB_c);
        ss << fileToString(includeFilePath + g_kKernelString_HDDA_c);
        ss << fileToString(includeFilePath + g_kKernelString_renderCommon_h);
        ss << fileToString(includeFilePath + g_kKernelString_renderLevelSet_c);
        ss << fileToString(includeFilePath + g_kKernelString_renderFogVolume_c);

        auto codeStr = new char[ss.str().length() + 1];
        std::memset(codeStr, 0, ss.str().length() + 1);
        std::memcpy(codeStr, ss.str().c_str(), ss.str().length());

        cl_program program = clCreateProgramWithSource(contextCL, 1, (const char**)&codeStr, NULL, &err);
        NANOVDB_CL_SAFE_CALL(err);

        delete[] codeStr;

        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::stringstream msg;
            msg << "CL compile error";

            if (err == CL_BUILD_PROGRAM_FAILURE) {
                size_t logBufferSize;
                clGetProgramBuildInfo(program, deviceCL, CL_PROGRAM_BUILD_LOG, 0, NULL, &logBufferSize);
                std::vector<uint8_t> logBuffer(logBufferSize);
                clGetProgramBuildInfo(program, deviceCL, CL_PROGRAM_BUILD_LOG, logBufferSize, logBuffer.data(), NULL);
                msg << ": " << logBuffer.data();
            } else {
                NANOVDB_CL_SAFE_CALL(err);
            }

            throw std::runtime_error(msg.str().c_str());
            std::cerr << "CL Error: " << msg.str() << std::endl;
            return nullptr;
        }

        cl_kernel kernelLevelSet = clCreateKernel(program, "renderLevelSet", &err);
        NANOVDB_CL_SAFE_CALL(err);

        cl_kernel kernelFogVolume = clCreateKernel(program, "renderFogVolume", &err);
        NANOVDB_CL_SAFE_CALL(err);

        cl_mem gridbuffer = clCreateBuffer(contextCL, CL_MEM_READ_ONLY, gridHdl.size(), NULL, &err);
        NANOVDB_CL_SAFE_CALL(err);

        NANOVDB_CL_SAFE_CALL(clEnqueueWriteBuffer(queue, gridbuffer, CL_TRUE, 0, gridHdl.size(), gridHdl.data(), 0, NULL, NULL));

        using GridT = nanovdb::NanoGrid<float>;
        using TreeT = GridT::TreeType;
        using RootT = TreeT::RootType;
        using Node2T = RootT::ChildNodeType;
        using Node1T = Node2T::ChildNodeType;
        using Node0T = Node1T::ChildNodeType;

        if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Float) {
            auto grid = gridHdl.grid<float>();

            uint32_t counts[] = {
                uint32_t(grid->tree().nodeCount<Node0T>() * Node0T::memUsage()),
                uint32_t(grid->tree().nodeCount<Node1T>() * Node1T::memUsage()),
                uint32_t(grid->tree().nodeCount<Node2T>() * Node2T::memUsage()),
                uint32_t(RootT::memUsage(grid->tree().root().tileCount()) - RootT::memUsage(0)),
                uint32_t(RootT::memUsage(0)),
                uint32_t(GridT::memUsage())};

            auto gridData = uintptr_t(grid);
            auto rootData = uintptr_t(&grid->tree().root());
            auto node2Level = (counts[2] > 0) ? uintptr_t(grid->tree().getNode<Node2T>(0)) : rootData + RootT::memUsage(0);
            auto node1Level = (counts[1] > 0) ? uintptr_t(grid->tree().getNode<Node1T>(0)) : node2Level;
            auto node0Level = (counts[0] > 0) ? uintptr_t(grid->tree().getNode<Node0T>(0)) : node1Level;

            uint32_t offsets[] = {
                uint32_t(node0Level - gridData),
                uint32_t(node1Level - gridData),
                uint32_t(node2Level - gridData),
                uint32_t(rootData - gridData + RootT::memUsage(0)),
                uint32_t(rootData - gridData),
                uint32_t(gridData - gridData)};

            cl_mem nodeLevelBuffers[6] = {nullptr};
            for (int i = 0; i < 6; ++i) {
                if (counts[i] > 0) {
                    cl_buffer_region region;
                    region.origin = offsets[i];
                    region.size = counts[i];
                    nodeLevelBuffers[i] = clCreateSubBuffer(gridbuffer, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
                    NANOVDB_CL_SAFE_CALL(err);
                }
            }

            resource->mQueueCl = queue;
            resource->mKernelLevelSetCl = kernelLevelSet;
            resource->mKernelFogVolumeCl = kernelFogVolume;
            resource->mProgramCl = program;
            resource->mDeviceCl = deviceCL;
            resource->mContextCl = contextCL;
            resource->mGridBuffer = gridbuffer;
            resource->mNodeLevel0 = nodeLevelBuffers[0];
            resource->mNodeLevel1 = nodeLevelBuffers[1];
            resource->mNodeLevel2 = nodeLevelBuffers[2];
            resource->mRootDataTiles = nodeLevelBuffers[3];
            resource->mRootData = nodeLevelBuffers[4];
            resource->mGridData = nodeLevelBuffers[5];
            resource->mInitialized = true;
        }
    }

    return resource;
}

void RenderLauncherCL::unmapCL(const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer)
{
    void* contextGL = nullptr;
    void* displayGL = nullptr;
#if defined(NANOVDB_USE_OPENGL)
    auto imgBufferGL = dynamic_cast<FrameBufferGL*>(imgBuffer);
    if (imgBufferGL) {
        contextGL = imgBufferGL->context();
        displayGL = imgBufferGL->display();
    }
#endif

    if (contextGL && displayGL) {
        cl_mem buffer = cl_mem(resource->mGlTextureResourceCL);
        NANOVDB_CL_SAFE_CALL(clEnqueueReleaseGLObjects(cl_command_queue(resource->mQueueCl), 1, &buffer, 0, 0, NULL));
    } else {
        imgBuffer->unmap();
        cl_mem buffer = cl_mem(resource->mGlTextureResourceCL);
        clReleaseMemObject(buffer);
        resource->mGlTextureResourceCL = nullptr;
    }
    imgBuffer->invalidate();
}

void* RenderLauncherCL::mapCL(int access, const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer)
{
    void* contextGL = nullptr;
    void* displayGL = nullptr;
#if defined(NANOVDB_USE_OPENGL)
    auto imgBufferGL = dynamic_cast<FrameBufferGL*>(imgBuffer);
    if (imgBufferGL) {
        contextGL = imgBufferGL->context();
        displayGL = imgBufferGL->display();
    }
#endif
    cl_mem buffer = nullptr;
    cl_int err;
    cl_int accessCL;

    if (access == (int)FrameBufferBase::AccessType::READ_ONLY) {
        accessCL = CL_MEM_READ_ONLY;
    } else if (access == (int)FrameBufferBase::AccessType::WRITE_ONLY) {
        accessCL = CL_MEM_WRITE_ONLY;
    } else {
        accessCL = CL_MEM_READ_WRITE;
    }

    if (contextGL && displayGL) {
#if defined(NANOVDB_USE_OPENGL)
        glFinish();

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imgBufferGL->bufferGL());

        if (!resource->mGlTextureResourceCL || resource->mGlTextureResourceSize != size_t(imgBuffer->size())) {
            std::cout << "Resizing OpenCL resources. (" << imgBuffer->size() << "B)" << std::endl;

            if (resource->mGlTextureResourceCL) {
                NANOVDB_CL_SAFE_CALL(clReleaseMemObject(cl_mem(resource->mGlTextureResourceCL)));
                resource->mGlTextureResourceSize = 0;
            }

#if 1
            // map the GL BufferObject
            buffer = clCreateFromGLBuffer(cl_context(resource->mContextCl),
                                          accessCL,
                                          uint32_t(imgBufferGL->bufferGL()),
                                          &err);
#else
            // map the GL Texture
            buffer = clCreateFromGLTexture(cl_context(resource->mContextCl),
                                           accessCL,
                                           GL_TEXTURE_2D,
                                           0,
                                           imgBufferGL->textureGL(),
                                           &err);
#endif
            NANOVDB_CL_SAFE_CALL(err);
            resource->mGlTextureResourceCL = buffer;
            resource->mGlTextureResourceSize = imgBuffer->size();
        }

        buffer = cl_mem(resource->mGlTextureResourceCL);
        if (buffer) {
            NANOVDB_CL_SAFE_CALL(clEnqueueAcquireGLObjects(cl_command_queue(resource->mQueueCl), 1, &buffer, 0, 0, NULL));
        }
#endif
        return buffer;
    } else {
        // we are unable to share the graphics resource->..
        // SLOW-PATH!

        float* imgPtr = (float*)imgBuffer->map(FrameBufferBase::AccessType(access));
        if (!imgPtr) {
            return nullptr;
        }

        buffer = clCreateBuffer(cl_context(resource->mContextCl),
                                accessCL | CL_MEM_USE_HOST_PTR,
                                imgBuffer->size(),
                                imgPtr,
                                &err);
        NANOVDB_CL_SAFE_CALL(err);
        resource->mGlTextureResourceCL = buffer;
    }
    return buffer;
}

bool RenderLauncherCL::render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int numGrids, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats)
{
    if (grids[0].gridHandle == nullptr)
        return false;

    auto& gridHdl = *reinterpret_cast<const nanovdb::GridHandle<>*>(grids[0].gridHandle);

    void* contextGL = nullptr;
    void* displayGL = nullptr;
#if defined(NANOVDB_USE_OPENGL)
    auto imgBufferGL = dynamic_cast<FrameBufferGL*>(imgBuffer);
    if (imgBufferGL) {
        contextGL = imgBufferGL->context();
        displayGL = imgBufferGL->display();
    }
#endif
    auto resource = ensureResource(gridHdl, contextGL, displayGL);
    if (!resource || !resource->mInitialized) {
        return false;
    }

    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    // map graphics image resource->..
    cl_mem imageBuffer = (cl_mem)mapCL(
        (int)((numAccumulations > 0) ? FrameBufferBase::AccessType::READ_WRITE : FrameBufferBase::AccessType::WRITE_ONLY),
        resource,
        imgBuffer);

    if (!imageBuffer) {
        return false;
    }

    // prepare data...

    struct Uniforms
    {
        cl_int         width;
        cl_int         height;
        cl_int         numAccumulations;
        cl_int         useBackground;
        cl_int         useGround;
        cl_int         useShadows;
        cl_int         useGroundReflections;
        cl_int         useLighting;
        float          useOcclusion;
        float          volumeDensityScale;
        float          volumeAlbedo;
        int            useTonemapping;
        float          tonemapWhitePoint;
        int            samplesPerPixel;
        float          groundHeight;
        float          groundFalloff;
        nanovdb::Vec3f cameraP;
        nanovdb::Vec3f cameraU;
        nanovdb::Vec3f cameraV;
        nanovdb::Vec3f cameraW;
        float          cameraAspect;
        float          cameraFovY;
    };

    Uniforms uniforms;
    uniforms.width = width;
    uniforms.height = height;
    uniforms.numAccumulations = numAccumulations;

    uniforms.useOcclusion = materialParams.useOcclusion;
    uniforms.volumeDensityScale = materialParams.volumeDensityScale;
    uniforms.volumeAlbedo = materialParams.volumeAlbedo;

    uniforms.samplesPerPixel = sceneParams.samplesPerPixel;
    uniforms.useGround = sceneParams.useGround;
    uniforms.useShadows = sceneParams.useShadows;
    uniforms.useGroundReflections = sceneParams.useGroundReflections;
    uniforms.useLighting = sceneParams.useLighting;
    uniforms.useBackground = sceneParams.useBackground;
    uniforms.useTonemapping = sceneParams.useTonemapping;
    uniforms.tonemapWhitePoint = sceneParams.tonemapWhitePoint;
    uniforms.groundHeight = sceneParams.groundHeight;
    uniforms.groundFalloff = sceneParams.groundFalloff;
    uniforms.cameraP = sceneParams.camera.P();
    uniforms.cameraU = sceneParams.camera.U();
    uniforms.cameraV = sceneParams.camera.V();
    uniforms.cameraW = sceneParams.camera.W();
    uniforms.cameraAspect = sceneParams.camera.aspect();
    uniforms.cameraFovY = sceneParams.camera.fov();

    // launch GL render...

    cl_kernel kernelCl = nullptr;
    if (method == MaterialClass::kLevelSetFast) {
        kernelCl = (cl_kernel)resource->mKernelLevelSetCl;
    } else if (method == MaterialClass::kFogVolumePathTracer) {
        kernelCl = (cl_kernel)resource->mKernelFogVolumeCl;
    } 

    if (!kernelCl)
        return false;

    int nArgs = 0;
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(cl_mem), &imageBuffer));
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(cl_mem), &resource->mNodeLevel0));
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(cl_mem), &resource->mNodeLevel1));
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(cl_mem), &resource->mNodeLevel2));
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(cl_mem), &resource->mRootData));
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(cl_mem), &resource->mRootDataTiles));
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(cl_mem), &resource->mGridData));
    NANOVDB_CL_SAFE_CALL(clSetKernelArg(kernelCl, nArgs++, sizeof(Uniforms), &uniforms));

    const size_t gridDim[] = {nanovdb::AlignUp<8>(width), nanovdb::AlignUp<8>(height), 1};
    const size_t blockDim[] = {8, 8, 1};
    NANOVDB_CL_SAFE_CALL(clEnqueueNDRangeKernel(
        cl_command_queue(resource->mQueueCl),
        kernelCl,
        2,
        NULL,
        gridDim,
        blockDim,
        0,
        NULL,
        NULL));

    NANOVDB_CL_SAFE_CALL(clFinish(cl_command_queue(resource->mQueueCl)));

    unmapCL(resource, imgBuffer);

    if (stats) {
        auto t1 = ClockT::now();
        stats->mDuration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    }

    return true;
}

#endif // NANOVDB_USE_OPENCL
