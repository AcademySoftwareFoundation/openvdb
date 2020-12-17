// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncherGL.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of GL-platform Grid renderer.
*/

#ifdef NANOVDB_USE_OPENGL

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include "RenderLauncherImpl.h"
#include "FrameBufferGL.h"

static const char* g_kKernelString_Platform_h = "code/CPlatform.h";
static const char* g_kKernelString_HDDA_h = "code/CHDDA.h";
static const char* g_kKernelString_NanoVDB_h = "code/CNanoVDB.h";
static const char* g_kKernelString_Platform_c = "code/CPlatform.c";
static const char* g_kKernelString_NanoVDB_c = "code/CNanoVDB.c";
static const char* g_kKernelString_HDDA_c = "code/CHDDA.c";
static const char* g_kKernelString_renderCommon_h = "code/renderCommon.h";
static const char* g_kKernelString_renderLevelSet_c = "code/renderLevelSet.c";
static const char* g_kKernelString_renderFogVolume_c = "code/renderFogVolume.c";
static const char* g_kKernelString_renderGrid_c = "code/renderGrid.c";

RenderLauncherGL::~RenderLauncherGL()
{
    for (auto& it : mResources) {
        if (it.second->mUniformBufferId)
            glDeleteBuffers(1, &it.second->mUniformBufferId);
        if (it.second->mBufferId)
            glDeleteBuffers(1, &it.second->mBufferId);
        if (it.second->mProgramId)
            glDeleteProgram(it.second->mProgramId);
    }
    mResources.clear();
}

bool RenderLauncherGL::ensureGridResource(const std::shared_ptr<Resource>& resource, const nanovdb::NanoGrid<float>* grid, size_t gridByteSize)
{
    GLint maxStorageBlockSize = 0;
    GLint maxStorageBlocks = 0;
    GLint storageOffsetAlignment = 0;
    GLint minMapBufferAlignment = 0;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxStorageBlockSize);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxStorageBlocks);
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &storageOffsetAlignment);
    glGetIntegerv(GL_MIN_MAP_BUFFER_ALIGNMENT, &minMapBufferAlignment);

    if (maxStorageBlockSize == 0) {
        std::cerr << "GL Error: "
                  << "GL_SHADER_STORAGE_BUFFER is not supported!" << std::endl;
        return false;
    }

    if (gridByteSize > (size_t)maxStorageBlockSize || maxStorageBlocks < 4) {
        std::stringstream msg;
        msg << "grid is too large. Requested size: " << gridByteSize << ", ";
        msg << "GL_MAX_SHADER_STORAGE_BLOCK_SIZE: " << maxStorageBlockSize << ", ";
        msg << "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: " << maxStorageBlocks;

        //throw std::runtime_error(msg.str().c_str());
        std::cerr << "GL Error: " << msg.str() << std::endl;
        return false;
    }

    glGenBuffers(1, &resource->mBufferId);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resource->mBufferId);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridByteSize, grid, GL_STATIC_READ);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    using GridT = nanovdb::NanoGrid<float>;
    using TreeT = GridT::TreeType;
    using RootT = TreeT::RootType;
    using Node2T = RootT::ChildNodeType;
    using Node1T = Node2T::ChildNodeType;
    using Node0T = Node1T::ChildNodeType;

    uint32_t counts[] = {
        uint32_t(grid->tree().nodeCount<Node0T>() * Node0T::memUsage()),
        uint32_t(grid->tree().nodeCount<Node1T>() * Node1T::memUsage()),
        uint32_t(grid->tree().nodeCount<Node2T>() * Node2T::memUsage()),
        uint32_t(RootT::memUsage(grid->tree().root().tileCount())),
        uint32_t(GridT::memUsage())};

    auto node0Level = grid->tree().getNode<Node0T>(0);
    auto node1Level = grid->tree().getNode<Node1T>(0);
    auto node2Level = grid->tree().getNode<Node2T>(0);
    auto rootData = &grid->tree().root();
    auto gridData = grid;

    uintptr_t gridBaseAddr = uintptr_t(grid);
    //uintptr_t treeBaseAddr = uintptr_t(&grid->tree());
    uint32_t  offsets[] = {
        uint32_t(uintptr_t(node0Level) - gridBaseAddr),
        uint32_t(uintptr_t(node1Level) - gridBaseAddr),
        uint32_t(uintptr_t(node2Level) - gridBaseAddr),
        uint32_t(uintptr_t(rootData) - gridBaseAddr),
        uint32_t(uintptr_t(gridData) - gridBaseAddr)};

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resource->mBufferId);

    // bind levels of the nanovdb tree...
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, resource->mBufferId, offsets[0], counts[0]);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, resource->mBufferId, offsets[1], counts[1]);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, resource->mBufferId, offsets[2], counts[2]);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, resource->mBufferId, offsets[3], counts[3]);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 4, resource->mBufferId, offsets[4], counts[4]);

    return true;
}

static std::string fileToString(std::string filename)
{
    std::ifstream file;
    file.open(filename);
    if (!file) {
        std::stringstream msg;
        msg << "Unable to load code: " << filename;
        throw std::runtime_error(msg.str().c_str());
    }

    std::string str;
    file.seekg(0, std::ios::end);
    str.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(file)),
               std::istreambuf_iterator<char>());

    return str;
}

bool RenderLauncherGL::ensureProgramResource(const std::shared_ptr<Resource>& resource, std::string /*valueType*/, MaterialClass method)
{
    if (resource->mUniformBufferId)
        glDeleteBuffers(1, &resource->mUniformBufferId);
    if (resource->mProgramId)
        glDeleteProgram(resource->mProgramId);

    resource->mMethod = method;
    resource->mProgramId = glCreateProgram();

    static const int LOG_MAX = 1024;
    static char      logBuffer[LOG_MAX];
    int              logLength;

    // Create and compile the compute shader.
    GLuint mComputeShader = glCreateShader(GL_COMPUTE_SHADER);

    const char* valueTypeStr = "float";
    size_t      sizeofValueType = sizeof(float);

    std::stringstream ss;
    ss << "#version 460\n#define CNANOVDB_COMPILER_GLSL\n#line -1\n";
    ss << std::string("#define VALUETYPE ") << valueTypeStr << "\n";
    ss << std::string("#define SIZEOF_VALUETYPE ") << sizeofValueType << "\n";

    std::string includeFilePath = std::string(__FILE__);
    includeFilePath = includeFilePath.substr(0, includeFilePath.find_last_of('/')).substr(0, includeFilePath.find_last_of('\\')) + "/";

    try {
        ss << fileToString(includeFilePath + g_kKernelString_Platform_h);
        ss << fileToString(includeFilePath + g_kKernelString_HDDA_h);
        ss << fileToString(includeFilePath + g_kKernelString_NanoVDB_h);
        ss << fileToString(includeFilePath + g_kKernelString_Platform_c);
        ss << fileToString(includeFilePath + g_kKernelString_NanoVDB_c);
        ss << fileToString(includeFilePath + g_kKernelString_HDDA_c);
        ss << fileToString(includeFilePath + g_kKernelString_renderCommon_h);
        if (method == MaterialClass::kLevelSetFast) {
            ss << fileToString(includeFilePath + g_kKernelString_renderLevelSet_c);
        } else if (method == MaterialClass::kFogVolumePathTracer) {
            ss << fileToString(includeFilePath + g_kKernelString_renderFogVolume_c);
        } else {
            ss << fileToString(includeFilePath + g_kKernelString_renderGrid_c);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "GLSL error: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        return false;
    }

    auto codeStr = new char[ss.str().length() + 1];
    std::memset(codeStr, 0, ss.str().length() + 1);
    std::memcpy(codeStr, ss.str().c_str(), ss.str().length());

    glShaderSource(mComputeShader, 1, &codeStr, NULL);

    delete[] codeStr;

    glCompileShader(mComputeShader);

    // Check if there were any issues compiling the shader.
    int rvalue;
    glGetShaderiv(mComputeShader, GL_COMPILE_STATUS, &rvalue);

    if (!rvalue) {
        glGetShaderInfoLog(mComputeShader, LOG_MAX, &logLength, logBuffer);

        std::stringstream msg;
        msg << "GLSL compile error: " << logBuffer;

        //throw std::runtime_error(msg.str().c_str());
        std::cerr << "GL Error: " << msg.str() << std::endl;
        return false;
    }

    // Attach and link the shader against the compute program.
    glAttachShader(resource->mProgramId, mComputeShader);
    glLinkProgram(resource->mProgramId);

    // Check if there were any issues linking the shader.
    glGetProgramiv(resource->mProgramId, GL_LINK_STATUS, &rvalue);

    if (!rvalue) {
        glGetProgramInfoLog(resource->mProgramId, LOG_MAX, &logLength, logBuffer);

        std::stringstream msg;
        msg << "GLSL link error: " << logBuffer;

        //throw std::runtime_error(msg.str().c_str());
        std::cerr << "GL Error: " << msg.str() << std::endl;
        return false;
    }

    resource->mUniformBufferBindIndex = glGetUniformBlockIndex(resource->mProgramId, "ArgUniforms");
    glGetActiveUniformBlockiv(resource->mProgramId, resource->mUniformBufferBindIndex, GL_UNIFORM_BLOCK_DATA_SIZE, (GLint*)&resource->mUniformBufferSize);

    glGenBuffers(1, &resource->mUniformBufferId);
    glBindBuffer(GL_UNIFORM_BUFFER, resource->mUniformBufferId);
    glBufferData(GL_UNIFORM_BUFFER, resource->mUniformBufferSize, nullptr, GL_DYNAMIC_DRAW);

    return true;
}

std::shared_ptr<RenderLauncherGL::Resource> RenderLauncherGL::ensureResource(const nanovdb::GridHandle<>& gridHdl, void* glContext, void* glDisplay, MaterialClass method)
{
    std::shared_ptr<Resource> resource;
    auto                      it = mResources.find(&gridHdl);
    if (it != mResources.end() && it->second->mMethod == method) {
        resource = it->second;
    } else {
        std::cout << "Initializing OpenGL renderer..." << std::endl;

        resource = std::make_shared<Resource>();
        mResources.insert(std::make_pair(&gridHdl, resource));
        resource->mMethod = method;

        if (glContext == nullptr || glDisplay == nullptr) {
            return resource;
        }

        if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Float) {
            auto grid = gridHdl.grid<float>();

            if (ensureGridResource(resource, grid, gridHdl.size()) &&
                ensureProgramResource(resource, "float", method))
                resource->mInitialized = true;
        }
    }

    return resource;
}

bool RenderLauncherGL::render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int /*numGrids*/, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats)
{
    if (grids[0].gridHandle == nullptr)
        return false;

    auto& gridHdl = *reinterpret_cast<const nanovdb::GridHandle<>*>(grids[0].gridHandle);

    void* contextGL = nullptr;
    void* displayGL = nullptr;
    auto  imgBufferGL = dynamic_cast<FrameBufferGL*>(imgBuffer);

    // currently the opengl renderer only works with the FrameBufferGL.
    if (!imgBufferGL) {
        std::cerr << "Error: frame-buffer does not support OpenGL" << std::endl;
        return false;
    }

    if (imgBufferGL) {
        contextGL = imgBufferGL->context();
        displayGL = imgBufferGL->display();
    }

    auto resource = ensureResource(gridHdl, contextGL, displayGL, method);
    if (!resource || !resource->mInitialized) {
        return false;
    }

    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    // prepare data...

    nanovdb::Vec3f cameraP = sceneParams.camera.P();
    nanovdb::Vec3f cameraU = sceneParams.camera.U();
    nanovdb::Vec3f cameraV = sceneParams.camera.V();
    nanovdb::Vec3f cameraW = sceneParams.camera.W();

    // launch GL render...

    glUseProgram(resource->mProgramId);

    glBindBufferBase(GL_UNIFORM_BUFFER, resource->mUniformBufferBindIndex, resource->mUniformBufferId);

    struct ArgUniforms
    {
        int   width;
        int   height;
        int   numAccumulations;
        int   useBackground;
        int   useGround;
        int   useShadows;
        int   useGroundReflections;
        int   useLighting;
        float useOcclusion;
        float volumeDensityScale;
        float volumeAlbedo;
        int   useTonemapping;
        float tonemapWhitePoint;
        int   samplesPerPixel;
        float groundHeight;
        float groundFalloff;
        float cameraPx, cameraPy, cameraPz;
        float cameraUx, cameraUy, cameraUz;
        float cameraVx, cameraVy, cameraVz;
        float cameraWx, cameraWy, cameraWz;
        float cameraAspect;
        float cameraFovY;
    };

    auto args = (ArgUniforms*)glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
    if (args) {
        auto& uniforms = *args;
        uniforms.width = width;
        uniforms.height = height;
        uniforms.numAccumulations = numAccumulations;

        uniforms.useShadows = sceneParams.useShadows;
        uniforms.useGroundReflections = sceneParams.useGroundReflections;
        uniforms.useLighting = sceneParams.useLighting;
        uniforms.useOcclusion = materialParams.useOcclusion;
        uniforms.volumeDensityScale = materialParams.volumeDensityScale;
        uniforms.volumeAlbedo = materialParams.volumeAlbedo;

        uniforms.samplesPerPixel = sceneParams.samplesPerPixel;
        uniforms.useBackground = sceneParams.useBackground;
        uniforms.useGround = sceneParams.useGround;
        uniforms.useTonemapping = sceneParams.useTonemapping;
        uniforms.tonemapWhitePoint = sceneParams.tonemapWhitePoint;
        uniforms.groundHeight = sceneParams.groundHeight;
        uniforms.groundFalloff = sceneParams.groundFalloff;
        uniforms.cameraPx = cameraP[0];
        uniforms.cameraPy = cameraP[1];
        uniforms.cameraPz = cameraP[2];
        uniforms.cameraUx = cameraU[0];
        uniforms.cameraUy = cameraU[1];
        uniforms.cameraUz = cameraU[2];
        uniforms.cameraVx = cameraV[0];
        uniforms.cameraVy = cameraV[1];
        uniforms.cameraVz = cameraV[2];
        uniforms.cameraWx = cameraW[0];
        uniforms.cameraWy = cameraW[1];
        uniforms.cameraWz = cameraW[2];
        uniforms.cameraAspect = sceneParams.camera.aspect();
        uniforms.cameraFovY = sceneParams.camera.fov();

        glUnmapBuffer(GL_UNIFORM_BUFFER);
    }

    NANOVDB_GL_CHECKERRORS();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resource->mBufferId);

    // bind image texture...
    GLenum imageAccess = GL_WRITE_ONLY;
    if (numAccumulations > 0) {
        imageAccess = GL_READ_WRITE;
    }

    glBindImageTexture(0, imgBufferGL->textureGL(), 0, GL_FALSE, 0, imageAccess, GL_RGBA32F);

    NANOVDB_GL_CHECKERRORS();

    glDispatchCompute((width + 7) / 8, (height + 7) / 8, 1);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    if (stats) {
        glFinish();
        auto t1 = ClockT::now();
        stats->mDuration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    }

    return true;
}

#endif // NANOVDB_USE_OPENGL