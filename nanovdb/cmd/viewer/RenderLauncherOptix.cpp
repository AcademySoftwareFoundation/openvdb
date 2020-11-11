// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file RenderLauncherOptix.cpp

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Implementation of Optix-platform Grid renderer.
*/

#if defined(NANOVDB_USE_OPTIX) && defined(NANOVDB_USE_CUDA)

#include "RenderLauncherImpl.h"
#include "FrameBufferHost.h"
#if defined(NANOVDB_USE_OPENGL)
#include "FrameBufferGL.h"
#endif
#if defined(NANOVDB_USE_OPENGL)
#include <cuda_gl_interop.h>
#endif
#include <cuda_runtime_api.h>
#include "optix/vec_math.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <nvrtc.h>

#include "optix/NanoVDB_optix.h"
#include <nanovdb/NanoVDB.h>

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

// Error check/report helper for users of the C API
#define NVRTC_CHECK_ERROR(func) \
    do { \
        nvrtcResult code = func; \
        if (code != NVRTC_SUCCESS) \
            throw std::runtime_error("ERROR: " __FILE__ "(" LINE_STR "): " + std::string(nvrtcGetErrorString(code))); \
    } while (0)

static const int g_sMaxTrace = 10;

template<typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<Camera>       RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

struct RenderState
{
    OptixDeviceContext          context = 0;
    OptixTraversableHandle      gas_handle = {};
    CUdeviceptr                 d_gas_output_buffer = {};
    OptixModule                 geometry_module = 0;
    OptixModule                 camera_module = 0;
    OptixModule                 shading_module = 0;
    OptixProgramGroup           raygen_prog_group = 0;
    OptixProgramGroup           miss_prog_group[RAY_TYPE_COUNT] = {0, 0};
    OptixProgramGroup           volume_prog_group[RAY_TYPE_COUNT] = {0, 0};
    OptixPipeline               pipeline = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    CUstream                    stream = 0;
    Params                      params;
    Params*                     d_params = nullptr;
    OptixShaderBindingTable     sbt = {};
};

#define NANOVDB_CUDA_SAFE_CALL(x) checkCUDA(x, __FILE__, __LINE__)

static bool checkCUDA(cudaError_t result, const char* file, const int line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime API error " << result << " in file " << file << ", line " << line << " : " << cudaGetErrorString(result) << ".\n";
        exit(1);
        return false;
    }
    return true;
}

class OptixException : public std::runtime_error
{
public:
    OptixException(const char* msg)
        : std::runtime_error(msg)
    {
    }

    OptixException(OptixResult res, const char* msg)
        : std::runtime_error(createMessage(res, msg).c_str())
    {
    }

private:
    std::string createMessage(OptixResult res, const char* msg)
    {
        std::ostringstream out;
        out << optixGetErrorName(res) << ": " << msg;
        return out.str();
    }
};

#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":" \
               << __LINE__ << ")\n"; \
            throw OptixException(res, ss.str().c_str()); \
        } \
    } while (0)

#define OPTIX_CHECK_LOG(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":" \
               << __LINE__ << ")\nLog:\n" \
               << log \
               << (sizeof_log > sizeof(log) ? "<TRUNCATED>" : "") \
               << "\n"; \
            throw OptixException(res, ss.str().c_str()); \
        } \
    } while (0)

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void createContext(RenderState& state)
{
    // Initialize CUDA
    NANOVDB_CUDA_SAFE_CALL(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    state.context = context;
}

static void buildGas(const RenderState&            state,
                     const OptixAccelBuildOptions& accel_options,
                     const OptixBuildInput&        build_input,
                     OptixTraversableHandle&       gas_handle,
                     CUdeviceptr&                  d_gas_output_buffer)
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr           d_temp_buffer_gas;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, &build_input, 1, &gas_buffer_sizes));

    NANOVDB_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = nanovdb::AlignUp<8>(gas_buffer_sizes.outputSizeInBytes);
    NANOVDB_CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(state.context,
                                0,
                                &accel_options,
                                &build_input,
                                1,
                                d_temp_buffer_gas,
                                gas_buffer_sizes.tempSizeInBytes,
                                d_buffer_temp_output_gas_and_compacted_size,
                                gas_buffer_sizes.outputSizeInBytes,
                                &gas_handle,
                                &emitProperty,
                                1));

    NANOVDB_CUDA_SAFE_CALL(cudaFree((void*)d_temp_buffer_gas));

    size_t compacted_gas_size;
    NANOVDB_CUDA_SAFE_CALL(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        NANOVDB_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(
            optixAccelCompact(state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

        NANOVDB_CUDA_SAFE_CALL(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    } else {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createGeometry(const std::vector<OptixAabb>& aabbs, CUdeviceptr& d_boundingBoxes, RenderState& state)
{
    NANOVDB_CUDA_SAFE_CALL(cudaFree((void*)d_boundingBoxes));

    uint32_t aabb_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

    NANOVDB_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&d_boundingBoxes), aabbs.size() * sizeof(OptixAabb)));
    NANOVDB_CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(d_boundingBoxes), aabbs.data(), aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#if (OPTIX_VERSION > 70000)
#define CUSTOM_PRIMITIVE_ARRAY customPrimitiveArray
#else
#define CUSTOM_PRIMITIVE_ARRAY aabbArray
#endif
    aabb_input.CUSTOM_PRIMITIVE_ARRAY.aabbBuffers = &d_boundingBoxes;
    aabb_input.CUSTOM_PRIMITIVE_ARRAY.flags = &aabb_input_flags;
    aabb_input.CUSTOM_PRIMITIVE_ARRAY.numSbtRecords = 1;
    aabb_input.CUSTOM_PRIMITIVE_ARRAY.numPrimitives = aabbs.size();
    aabb_input.CUSTOM_PRIMITIVE_ARRAY.sbtIndexOffsetBuffer = 0;
    aabb_input.CUSTOM_PRIMITIVE_ARRAY.sbtIndexOffsetSizeInBytes = 0;
    aabb_input.CUSTOM_PRIMITIVE_ARRAY.primitiveIndexOffset = 0;

    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION, // buildFlags
        OPTIX_BUILD_OPERATION_BUILD // operation
    };

    buildGas(state, accel_options, aabb_input, state.gas_handle, state.d_gas_output_buffer);
}

static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str());
    if (file.good()) {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        str = source_buffer.str();
        return true;
    }
    return false;
}

static void getCuStringFromFile(std::string& cu, std::string& location, const char* filename)
{
    std::vector<std::string> source_locations;

    std::string base_dir = std::string(__FILE__);
    base_dir = base_dir.substr(0, base_dir.find_last_of('/')).substr(0, base_dir.find_last_of('\\')) + "/";

    source_locations.push_back(base_dir + filename);

    for (const std::string& loc : source_locations) {
        // Try to get source code from file
        if (readSourceFile(cu, loc)) {
            location = loc;
            return;
        }
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error("Couldn't open source file " + std::string(filename));
}

static std::string g_nvrtcLog;

#define NANOVDB_OPTIX_RELATIVE_INCLUDE_DIRS \
    "optix", \
        "../../nanovdb", \
        "../..", \
        "..", \
        ".",

// These must be defined or NVRTC will fail to compile optix programs.
// CMake will define them automatically.
#define NANOVDB_OPTIX_ABSOLUTE_INCLUDE_DIRS \
    NANOVDB_OPTIX_RTC_OPTIX_DIR, \
        NANOVDB_OPTIX_RTC_CUDA_DIR

#define NANOVDB_CUDA_NVRTC_OPTIONS \
    "--std=c++11", \
        "-arch", \
        "compute_60", \
        "-use_fast_math", \
        "-lineinfo", \
        "-default-device", \
        "-rdc", \
        "true", \
        "-D__x86_64",

static void getPtxFromCuString(std::string& ptx, const char* cu_source, const char* name, const char** log_string)
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

    // Gather NVRTC options
    std::vector<const char*> options;

    std::string base_dir = std::string(__FILE__);
    base_dir = base_dir.substr(0, base_dir.find_last_of('/')).substr(0, base_dir.find_last_of('\\')) + "/";

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char*              abs_dirs[] = {NANOVDB_OPTIX_ABSOLUTE_INCLUDE_DIRS};
    const char*              rel_dirs[] = {NANOVDB_OPTIX_RELATIVE_INCLUDE_DIRS};

    for (const char* dir : abs_dirs) {
        include_dirs.push_back(std::string("--include-path=") + dir);
    }

    for (const char* dir : rel_dirs) {
        include_dirs.push_back("--include-path=" + base_dir + dir);
    }

    for (const std::string& dir : include_dirs) {
        options.push_back(dir.c_str());
    }

    // NVRTC options
#ifdef _WIN32
    const char* compiler_options[] = {NANOVDB_CUDA_NVRTC_OPTIONS "-DNANOVDB_OPTIX_RTC_WIN32"};
#else
    const char* compiler_options[] = {NANOVDB_CUDA_NVRTC_OPTIONS};
#endif
    std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    g_nvrtcLog.resize(log_size);
    if (log_size > 1) {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
        if (log_string)
            *log_string = g_nvrtcLog.c_str();
    }
    if (compileRes != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}

struct PtxSourceCache
{
    std::map<std::string, std::string*> map;
    ~PtxSourceCache()
    {
        for (std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it)
            delete it->second;
    }
};
static PtxSourceCache g_ptxSourceCache;

const char* getPtxString(const char* filename, const char** log = NULL)
{
    if (log)
        *log = NULL;

    std::string *                                 ptx, cu;
    std::string                                   key = std::string(filename);
    std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find(key);

    if (elem == g_ptxSourceCache.map.end()) {
        ptx = new std::string();
        std::string location;
        getCuStringFromFile(cu, location, filename);
        getPtxFromCuString(*ptx, cu.c_str(), location.c_str(), log);
        g_ptxSourceCache.map[key] = ptx;
    } else {
        ptx = elem->second;
    }

    return ptx->c_str();
}

void createModules(RenderState& state)
{
    OptixModuleCompileOptions module_compile_options;
    module_compile_options.maxRegisterCount = 100;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    //module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        const std::string ptx = getPtxString("optix/geometry.cu");
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(state.context,
                                                 &module_compile_options,
                                                 &state.pipeline_compile_options,
                                                 ptx.c_str(),
                                                 ptx.size(),
                                                 log,
                                                 &sizeof_log,
                                                 &state.geometry_module));
    }

    {
        const std::string ptx = getPtxString("optix/camera.cu");
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(state.context,
                                                 &module_compile_options,
                                                 &state.pipeline_compile_options,
                                                 ptx.c_str(),
                                                 ptx.size(),
                                                 log,
                                                 &sizeof_log,
                                                 &state.camera_module));
    }

    {
        const std::string ptx = getPtxString("optix/shading.cu");
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(state.context,
                                                 &module_compile_options,
                                                 &state.pipeline_compile_options,
                                                 ptx.c_str(),
                                                 ptx.size(),
                                                 log,
                                                 &sizeof_log,
                                                 &state.shading_module));
    }
}

static void createCameraProgram(RenderState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup        cam_prog_group;
    OptixProgramGroupOptions cam_prog_group_options = {};
    OptixProgramGroupDesc    cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__nanovdb_camera";

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &cam_prog_group_desc, 1, &cam_prog_group_options, log, &sizeof_log, &cam_prog_group));

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;
}

static void createVolumeProgram(MaterialClass renderMethod, RenderState& state, std::vector<OptixProgramGroup>& program_groups)
{
    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        OptixProgramGroup        radiance_prog_group;
        OptixProgramGroupOptions radiance_prog_group_options = {};
        OptixProgramGroupDesc    radiance_prog_group_desc = {};
        radiance_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        radiance_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
        radiance_prog_group_desc.hitgroup.moduleCH = state.shading_module;
        radiance_prog_group_desc.hitgroup.moduleAH = nullptr;

        if (renderMethod == MaterialClass::kLevelSetFast) {
            radiance_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_levelset";
            radiance_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_levelset_radiance";
            radiance_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        } else if (renderMethod == MaterialClass::kFogVolumePathTracer) {
            radiance_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
            radiance_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_radiance";
            radiance_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        } else {
            radiance_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_grid";
            radiance_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_grid_radiance";
            radiance_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        }

        OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context,
                                                &radiance_prog_group_desc,
                                                1,
                                                &radiance_prog_group_options,
                                                log,
                                                &sizeof_log,
                                                &radiance_prog_group));

        program_groups.push_back(radiance_prog_group);
        state.volume_prog_group[RAY_TYPE_RADIANCE] = radiance_prog_group;
    }

    {
        OptixProgramGroup        occlusion_prog_group;
        OptixProgramGroupOptions occlusion_prog_group_options = {};
        OptixProgramGroupDesc    occlusion_prog_group_desc = {};
        occlusion_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        occlusion_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
        occlusion_prog_group_desc.hitgroup.moduleCH = nullptr;
        occlusion_prog_group_desc.hitgroup.moduleAH = nullptr;
        occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        occlusion_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

        if (renderMethod == MaterialClass::kLevelSetFast) {
            occlusion_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_levelset";
        } else if (renderMethod == MaterialClass::kFogVolumePathTracer) {
            occlusion_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
            occlusion_prog_group_desc.hitgroup.moduleCH = state.shading_module;
            occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_occlusion";
        } else {
            occlusion_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_grid";
        }

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &occlusion_prog_group_desc, 1, &occlusion_prog_group_options, log, &sizeof_log, &occlusion_prog_group));

        program_groups.push_back(occlusion_prog_group);
        state.volume_prog_group[RAY_TYPE_OCCLUSION] = occlusion_prog_group;
    }
}

static void createMissProgram(MaterialClass renderMethod, RenderState& state, std::vector<OptixProgramGroup>& program_groups)
{
    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        OptixProgramGroup        miss_prog_group;
        OptixProgramGroupOptions miss_prog_group_options = {};
        OptixProgramGroupDesc    miss_prog_group_desc = {};

        miss_prog_group_desc.miss = {
            nullptr, // module
            nullptr // entryFunctionName
        };

        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.shading_module;
        if (renderMethod == MaterialClass::kLevelSetFast || renderMethod == MaterialClass::kGrid) {
            miss_prog_group_desc.miss.entryFunctionName = "__miss__levelset_radiance";
        } else if (renderMethod == MaterialClass::kFogVolumeFast || renderMethod == MaterialClass::kFogVolumePathTracer || renderMethod == MaterialClass::kBlackBodyVolumePathTracer) {
            miss_prog_group_desc.miss.entryFunctionName = "__miss__fogvolume_radiance";
        } else {
            miss_prog_group_desc.miss.entryFunctionName = "__miss__env_radiance";
        }

        OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context,
                                                &miss_prog_group_desc,
                                                1,
                                                &miss_prog_group_options,
                                                log,
                                                &sizeof_log,
                                                &state.miss_prog_group[RAY_TYPE_RADIANCE]));
    }

    {
        OptixProgramGroup        miss_prog_group;
        OptixProgramGroupOptions miss_prog_group_options = {};
        OptixProgramGroupDesc    miss_prog_group_desc = {};

        miss_prog_group_desc.miss = {
            nullptr, // module
            nullptr // entryFunctionName
        };

        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.shading_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context,
                                                &miss_prog_group_desc,
                                                1,
                                                &miss_prog_group_options,
                                                log,
                                                &sizeof_log,
                                                &state.miss_prog_group[RAY_TYPE_OCCLUSION]));
    }
}

void createPipeline(MaterialClass renderMethod, RenderState& state)
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {};
    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 3;
    state.pipeline_compile_options.numAttributeValues = 6;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "constantParams";

    // Prepare program groups
    createModules(state);
    createCameraProgram(state, program_groups);
    createVolumeProgram(renderMethod, state, program_groups);
    createMissProgram(renderMethod, state, program_groups);

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        g_sMaxTrace, // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, // debugLevel
    };
    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(state.context,
                                        &state.pipeline_compile_options,
                                        &pipeline_link_options,
                                        program_groups.data(),
                                        static_cast<unsigned int>(program_groups.size()),
                                        log,
                                        &sizeof_log,
                                        &state.pipeline));
}

void syncCameraDataToSbt(RenderState& state, const Camera& camera)
{
    RayGenRecord rg_sbt;

    optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt);
    rg_sbt.data = camera;

    NANOVDB_CUDA_SAFE_CALL(cudaMemcpy(
        reinterpret_cast<void*>(state.sbt.raygenRecord), &rg_sbt, sizeof(RayGenRecord), cudaMemcpyHostToDevice));
}

void createSBT(const VolumeGeometry& geometry, const VolumeMaterial& material, RenderState& state)
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        size_t      sizeof_raygen_record = sizeof(RayGenRecord);
        NANOVDB_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof_raygen_record));

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t      sizeof_miss_record = sizeof(MissRecord);
        NANOVDB_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof_miss_record * RAY_TYPE_COUNT));

        MissRecord ms_sbt[RAY_TYPE_COUNT];
        for (int i = 0; i < RAY_TYPE_COUNT; ++i) {
            optixSbtRecordPackHeader(state.miss_prog_group[i], &ms_sbt[i]);
            // data for miss program goes in here...
            //ms_sbt[i].data = {0.f, 0.f, 0.f};
        }

        NANOVDB_CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(d_miss_record),
                                          ms_sbt,
                                          sizeof_miss_record * RAY_TYPE_COUNT,
                                          cudaMemcpyHostToDevice));

        state.sbt.missRecordBase = d_miss_record;
        state.sbt.missRecordCount = RAY_TYPE_COUNT;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
    }

    // Hitgroup program record
    {
        const size_t                count_records = RAY_TYPE_COUNT;
        std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT);

        {
            int sbt_idx = 0;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.volume_prog_group[RAY_TYPE_RADIANCE], &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.geometry.volume = geometry;
            hitgroup_records[sbt_idx].data.shading.volume = material;
            sbt_idx++;

            OPTIX_CHECK(optixSbtRecordPackHeader(state.volume_prog_group[RAY_TYPE_OCCLUSION], &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.geometry.volume = geometry;
            hitgroup_records[sbt_idx].data.shading.volume = material;
            sbt_idx++;
        }

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
        NANOVDB_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), sizeof_hitgroup_record * count_records));

        NANOVDB_CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),
                                          hitgroup_records.data(),
                                          sizeof_hitgroup_record * count_records,
                                          cudaMemcpyHostToDevice));

        state.sbt.hitgroupRecordBase = d_hitgroup_records;
        state.sbt.hitgroupRecordCount = count_records;
        state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
    }
}

void initLaunchParams(RenderState& state)
{
    state.params.imgBuffer = nullptr; // Will be set when output buffer is mapped
    state.params.width = 0;
    state.params.height = 0;
    state.params.numAccumulations = 0u;
    state.params.maxDepth = g_sMaxTrace;
    state.params.sceneEpsilon = 1.e-4f;

    NANOVDB_CUDA_SAFE_CALL(cudaStreamCreate(&state.stream));
    NANOVDB_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

    state.params.handle = state.gas_handle;
}

RenderLauncherOptix::~RenderLauncherOptix()
{
    mResources.clear();
}

RenderLauncherOptix::Resource::~Resource()
{
    delete reinterpret_cast<RenderState*>(mOptixRenderState);
    cudaFree(mDeviceGrid);
}

std::shared_ptr<RenderLauncherOptix::Resource> RenderLauncherOptix::ensureResource(const nanovdb::GridHandle<>& gridHdl, MaterialClass renderMethod)
{
    assert(renderMethod != MaterialClass::kAuto);

    std::shared_ptr<Resource> resource;
    auto                      it = mResources.find(&gridHdl);
    if (it != mResources.end() && it->second->mMaterialClass == renderMethod) {
        resource = it->second;
    } else {
        std::cout << "Initializing OptiX renderer..." << std::endl;

        resource = std::make_shared<Resource>();
        resource->mMaterialClass = renderMethod;

        if (it != mResources.end())
            mResources.erase(it);

        mResources.insert(std::make_pair(&gridHdl, resource));

        NANOVDB_CUDA_SAFE_CALL(cudaMalloc((void**)&resource->mDeviceGrid, gridHdl.size()));
        NANOVDB_CUDA_SAFE_CALL(cudaMemcpy(resource->mDeviceGrid, gridHdl.data(), gridHdl.size(), cudaMemcpyHostToDevice));

        auto code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(code) << std::endl;
            return nullptr;
        }

        auto rs = new RenderState();

        RenderState& state = *rs;

        VolumeGeometry volume_geometry;
        volume_geometry.grid = resource->mDeviceGrid;

        VolumeMaterial volume_material;
        volume_material.importance_cutoff = 0.01f;
        volume_material.Ksigma = make_float3(1.0f, 1.0f, 1.0f);

        std::vector<OptixAabb> aabbs;
        CUdeviceptr            d_boundingBoxes = 0;
        static const int       NodeLevel = 0;

        if (NodeLevel >= 3) {
            auto      boundsMin = gridHdl.gridMetaData()->worldBBox().min();
            auto      boundsMax = gridHdl.gridMetaData()->worldBBox().max();
            OptixAabb aabb;

            aabb.minX = (float)boundsMin[0];
            aabb.minY = (float)boundsMin[1];
            aabb.minZ = (float)boundsMin[2];
            aabb.maxX = (float)boundsMax[0];
            aabb.maxY = (float)boundsMax[1];
            aabb.maxZ = (float)boundsMax[2];

            aabbs.push_back(aabb);
        } else {
            auto grid = gridHdl.grid<float>();

            int numNodes = grid->tree().nodeCount(NodeLevel);

            for (int i = 0; i < numNodes; ++i) {
                OptixAabb aabb;
                auto      node = grid->tree().getNode<NodeLevel>(i);
#if 1
                // use node's size.
                nanovdb::Coord boundsMin(node->origin());
                nanovdb::Coord boundsMax(node->origin() + nanovdb::Coord(node->dim()));
#else
                // use node's tight bounds.
                nanovdb::Coord boundsMin(node->bbox().min());
                nanovdb::Coord boundsMax(node->bbox().max() + nanovdb::Coord(1));
#endif
                nanovdb::Vec3f boundsMinf(boundsMin[0], boundsMin[1], boundsMin[2]);
                nanovdb::Vec3f boundsMaxf(boundsMax[0], boundsMax[1], boundsMax[2]);

                nanovdb::Vec3f worldBoundsMinf = grid->indexToWorldF(boundsMinf);
                nanovdb::Vec3f worldBoundsMaxf = grid->indexToWorldF(boundsMaxf);
                aabb.minX = (float)worldBoundsMinf[0];
                aabb.minY = (float)worldBoundsMinf[1];
                aabb.minZ = (float)worldBoundsMinf[2];
                aabb.maxX = (float)worldBoundsMaxf[0];
                aabb.maxY = (float)worldBoundsMaxf[1];
                aabb.maxZ = (float)worldBoundsMaxf[2];

                aabbs.push_back(aabb);
                //printf("%d. (%d %d %d) (%.2f %.2f %.2f)\n", i, boundsMin[0], boundsMin[1], boundsMin[2], volumeData.bmin.x, volumeData.bmin.y, volumeData.bmin.z);
            }
        }

        std::cout << "BVH contains " << aabbs.size() << " AABBs" << std::endl;

        createContext(state);

        if (aabbs.size() > 0) {
            createGeometry(aabbs, d_boundingBoxes, state);
        }

        createPipeline(renderMethod, state);
        createSBT(volume_geometry, volume_material, state);
        initLaunchParams(state);
        resource->mOptixRenderState = rs;
        resource->mInitialized = true;
    }

    return resource;
}

void RenderLauncherOptix::unmapCUDA(const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream)
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

void* RenderLauncherOptix::mapCUDA(int access, const std::shared_ptr<Resource>& resource, FrameBufferBase* imgBuffer, void* stream)
{
    if (!imgBuffer->size())
        return nullptr;

#if defined(NANOVDB_USE_OPENGL)
    auto imgBufferGL = dynamic_cast<FrameBufferGL*>(imgBuffer);
    if (imgBufferGL) {
        auto     accessGL = FrameBufferBase::AccessType(access);
        uint32_t accessCUDA = cudaGraphicsMapFlagsNone;
        if (accessGL == FrameBufferBase::AccessType::READ_ONLY)
            accessCUDA = cudaGraphicsMapFlagsReadOnly;
        else if (accessGL == FrameBufferBase::AccessType::WRITE_ONLY)
            accessCUDA = cudaGraphicsMapFlagsWriteDiscard;

        if (!resource->mGlTextureResourceCUDA || resource->mGlTextureResourceId != imgBufferGL->resourceId()) {
            std::cout << "registering GL resource(" << imgBufferGL->bufferGL() << ") [" << imgBufferGL->resourceId() << "] for CUDA. (" << imgBuffer->size() << "B)" << std::endl;

            if (stream)
                NANOVDB_CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStream_t(stream)));
            else
                NANOVDB_CUDA_SAFE_CALL(cudaDeviceSynchronize());

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

bool RenderLauncherOptix::render(MaterialClass method, int width, int height, FrameBufferBase* imgBuffer, int numAccumulations, int /*numGrids*/, const GridRenderParameters* grids, const SceneRenderParameters& sceneParams, const MaterialParameters& materialParams, RenderStatistics* stats)
{
    if (grids[0].gridHandle == nullptr)
        return false;

    auto& gridHdl = *reinterpret_cast<const nanovdb::GridHandle<>*>(grids[0].gridHandle);

    auto resource = ensureResource(gridHdl, method);
    if (!resource || !resource->mInitialized)
        return false;

    using ClockT = std::chrono::high_resolution_clock;
    auto t0 = ClockT::now();

    float* imgPtr = (float*)mapCUDA(
        (int)((numAccumulations > 0) ? FrameBufferBase::AccessType::READ_WRITE : FrameBufferBase::AccessType::WRITE_ONLY),
        resource,
        imgBuffer);

    if (!imgPtr)
        return false;

    auto& optixRenderState = *reinterpret_cast<RenderState*>(resource->mOptixRenderState);

    syncCameraDataToSbt(optixRenderState, sceneParams.camera);

    optixRenderState.params.width = width;
    optixRenderState.params.height = height;
    optixRenderState.params.imgBuffer = (float4*)imgPtr;
    optixRenderState.params.numAccumulations = numAccumulations;
    optixRenderState.params.materialConstants = materialParams;
    optixRenderState.params.sceneConstants = sceneParams;

    NANOVDB_CUDA_SAFE_CALL(cudaMemcpyAsync(
        reinterpret_cast<void*>(optixRenderState.d_params), &optixRenderState.params, sizeof(Params), cudaMemcpyHostToDevice, optixRenderState.stream));

    if (gridHdl.gridMetaData()->gridType() == nanovdb::GridType::Float) {
        OPTIX_CHECK(optixLaunch(optixRenderState.pipeline,
                                optixRenderState.stream,
                                reinterpret_cast<CUdeviceptr>(optixRenderState.d_params),
                                sizeof(Params),
                                &optixRenderState.sbt,
                                width,
                                height,
                                1));

        NANOVDB_CUDA_SAFE_CALL(cudaStreamSynchronize(optixRenderState.stream));
    }

    unmapCUDA(resource, imgBuffer);

    if (stats) {
        auto t1 = ClockT::now();
        stats->mDuration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.f;
    }

    return true;
}

#endif // NANOVDB_USE_OPTIX
