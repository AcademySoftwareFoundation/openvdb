// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compiler/SlangCompiler.cpp

    \author Petra Hapalova

    \brief
*/

#include "SlangCompiler.h"
#include <nanovdb_editor/putil/Shader.hpp>

#include <fstream>
#include <iomanip>
#include <cstring>
#include <filesystem>

// currently obsolete options
//#define SLANG_OBFUSCATE
//#define BINARY_OUTPUT

#if defined(USE_SLANG_DEBUG_OUTPUT)
    #define ASM_DEBUG_OUTPUT          // Compiles and saves the assembly code, won't be possible to create a compute pipeline
#endif

namespace pnanovdb_compiler
{
static const char* scalarTypeNames[] =
{
    "",             // SLANG_SCALAR_TYPE_NONE
    "void",         // SLANG_SCALAR_TYPE_VOID
    "bool",         // SLANG_SCALAR_TYPE_BOOL
    "int",          // SLANG_SCALAR_TYPE_INT32
    "uint",         // SLANG_SCALAR_TYPE_UINT32
    "int64",        // SLANG_SCALAR_TYPE_INT64
    "uint64",       // SLANG_SCALAR_TYPE_UINT64
    "float16",      // SLANG_SCALAR_TYPE_FLOAT16
    "float",        // SLANG_SCALAR_TYPE_FLOAT32
    "double"        // SLANG_SCALAR_TYPE_FLOAT64
};

static std::string getScalarTypeName(slang::TypeReflection::ScalarType type)
{
    return (type >= 0 && type < sizeof(scalarTypeNames)/sizeof(scalarTypeNames[0]))
        ? scalarTypeNames[type]
        : "float";
}

struct ResourceTypeSlangDesc
{
    pnanovdb_compute_descriptor_type_t type;
    SlangTypeKind typeKind;
    SlangResourceShape resourceShape;
    SlangResourceAccess access;
};

static const pnanovdb_uint32_t resourceTypeDescCount = 13u;

static ResourceTypeSlangDesc resourceTypeFromSlang[resourceTypeDescCount] = {
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_UNKNOWN, SLANG_TYPE_KIND_NONE, SLANG_RESOURCE_NONE, SLANG_RESOURCE_ACCESS_NONE},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_CONSTANT_BUFFER, SLANG_TYPE_KIND_CONSTANT_BUFFER, SLANG_RESOURCE_NONE, SLANG_RESOURCE_ACCESS_NONE},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_STRUCTURED_BUFFER, SLANG_TYPE_KIND_RESOURCE, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_STRUCTURED_BUFFER, SLANG_TYPE_KIND_RESOURCE, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER, SLANG_TYPE_KIND_RESOURCE, SLANG_TEXTURE_BUFFER, SLANG_RESOURCE_ACCESS_READ},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE, SLANG_TYPE_KIND_RESOURCE, SLANG_TEXTURE_1D, SLANG_RESOURCE_ACCESS_READ},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE, SLANG_TYPE_KIND_RESOURCE, SLANG_TEXTURE_2D, SLANG_RESOURCE_ACCESS_READ},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE, SLANG_TYPE_KIND_RESOURCE, SLANG_TEXTURE_3D, SLANG_RESOURCE_ACCESS_READ},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_SAMPLER, SLANG_TYPE_KIND_SAMPLER_STATE, SLANG_RESOURCE_NONE, SLANG_RESOURCE_ACCESS_NONE},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_BUFFER, SLANG_TYPE_KIND_RESOURCE, SLANG_RESOURCE_NONE, SLANG_RESOURCE_ACCESS_READ_WRITE},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE, SLANG_TYPE_KIND_RESOURCE, SLANG_TEXTURE_1D, SLANG_RESOURCE_ACCESS_READ_WRITE},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE, SLANG_TYPE_KIND_RESOURCE, SLANG_TEXTURE_2D, SLANG_RESOURCE_ACCESS_READ_WRITE},
    {PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE, SLANG_TYPE_KIND_RESOURCE, SLANG_TEXTURE_3D, SLANG_RESOURCE_ACCESS_READ_WRITE},
};

static pnanovdb_compute_descriptor_type_t resourceTypeFromSlangLookup(SlangTypeKind typeKind, SlangResourceShape resourceShape, SlangResourceAccess access)
{
    for (pnanovdb_uint32_t idx = 0u; idx < resourceTypeDescCount; idx++)
    {
        if (typeKind == resourceTypeFromSlang[idx].typeKind &&
            resourceShape == resourceTypeFromSlang[idx].resourceShape &&
            access == resourceTypeFromSlang[idx].access)
        {
            return resourceTypeFromSlang[idx].type;
        }
    }
    return resourceTypeFromSlang[0u].type;
}

SlangCompiler::SlangCompiler()
{
    slang::createGlobalSession(&globalSession_);

#ifdef _WIN32
    // On Windows, look for slang-llvm.dll in the same directory as slang.dll
    char slangLlvmPath[1024] = {0};
    HMODULE slangModule = GetModuleHandleA("slang.dll");
    if (slangModule)
    {
        GetModuleFileNameA(slangModule, slangLlvmPath, sizeof(slangLlvmPath));
        char* lastSlash = strrchr(slangLlvmPath, '\\');
        if (lastSlash)
        {
            *(lastSlash + 1) = '\0';
            strcat(slangLlvmPath, "slang-llvm");
            globalSession_->setDownstreamCompilerPath(SLANG_PASS_THROUGH_LLVM, slangLlvmPath);
        }
    }
#else
    globalSession_->setDownstreamCompilerPath(SLANG_PASS_THROUGH_LLVM, "slang-llvm");
#endif
    hasSlangLlvm_ = SLANG_SUCCEEDED(globalSession_->checkPassThroughSupport(SLANG_PASS_THROUGH_LLVM));
    if (!hasSlangLlvm_)
    {
        printf("Error: Slang LLVM not found\n");
    }
}

SlangCompiler::~SlangCompiler()
{
    for (auto& pair : sharedLibraries_)
    {
        if (pair.second)
        {
            pair.second->release();
        }
    }
    sharedLibraries_.clear();

    if (shader_)
    {
        shader_.reset();
    }
    if (globalSession_)
    {
        globalSession_->release();
    }
    //slang::shutdown();
}

// not used
bool SlangCompiler::compileFile(const char* sourceFile,
                                const char* destinationFile,
                                const char* variableName,
                                size_t numIncludePaths,
                                const char** includePaths,
                                const pnanovdb_compiler_settings_t* settings)
{
    std::ifstream inFile(sourceFile);
    if (!inFile)
    {
        return false;
    }

    std::string code((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    const char* out;
    std::filesystem::path fsPath(sourceFile);
    if (!compile(settings, fsPath.filename().string().c_str(), code.c_str(), numIncludePaths, includePaths) && !shader_)
    {
        printf("Slang shader compilation of '%s' failed\n", variableName);
        return false;
    }

#ifdef BINARY_OUTPUT
    std::ostringstream oss;
    oss << "const unsigned char " + std::string(variableName) + "[] = {\n  ";

    char* bytes = shader_->computeShader.byteCode.get();

    for (size_t i = 0; i < shader_->computeShader.byteCodeSize; ++i)
    {
        oss << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
        if (i != shader_->computeShader.byteCodeSize - 1)
        {
            oss << ", ";
        }
        if ((i + 1) % 12 == 0)
        {
            oss << "\n  ";
        }
    }
    oss << "\n};";
#endif

    std::ofstream outFile(destinationFile);
    if (outFile.is_open())
    {
#ifndef BINARY_OUTPUT
        outFile.write(static_cast<const char*>(shader_->computeShader.byteCode.get()), shader_->computeShader.byteCodeSize);
#else
        outFile << oss.str();
#endif
        outFile.close();
    }
    else
    {
        printf("Saving bytecode for '%s' failed\n", variableName);
        return false;
    }

    return true;
}

bool SlangCompiler::compile(const pnanovdb_compiler_settings_t* settings, const char* codeFileName, const char* codeString, size_t numIncludePaths, const char** includePaths)
{
    // https://shader-slang.com/slang/user-guide/compiling.html#using-the-compilation-api

    using namespace slang;

    // invalidate the previous shader
    if (shader_)
    {
        shader_.reset();
    }
    shader_ = std::make_shared<pnanovdb_shader::ShaderData>();
    shader_->setMetadata(codeString);

    SessionDesc sessionDesc;
    TargetDesc targetDesc;

    std::filesystem::path tempDir;
    std::filesystem::path originalPath;
    std::string dumpPrefix;

#ifdef ASM_DEBUG_OUTPUT
    targetDesc.format = SLANG_SPIRV_ASM;
#else
    if (settings->compile_target == PNANOVDB_COMPILE_TARGET_CPU)
    {
        targetDesc.format = SLANG_SHADER_HOST_CALLABLE;

        // Create a temporary directory for intermediates and make it the current working directory
        tempDir = createTempDirectory(shader_->computeShader.timestamp);
        dumpPrefix = std::string(codeFileName) + "_";
        originalPath = std::filesystem::current_path();
        std::filesystem::current_path(tempDir);
    }
    else if (settings->hlsl_output)
    {
        targetDesc.format = SLANG_HLSL;
    }
    else
    {
        targetDesc.format = SLANG_SPIRV;
    }
#endif

    // profiles: https://github.com/shader-slang/slang/blob/master/source/slang/slang-profile-defs.h
    if (settings->use_glslang)
    {
        targetDesc.profile = globalSession_->findProfile("glsl_450");
    }
    else if (settings->compile_target != PNANOVDB_COMPILE_TARGET_CPU)
    {
        targetDesc.profile = globalSession_->findProfile(PNANOVDB_REFLECT_XSTR(SLANG_PROFILE));
    }

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    sessionDesc.searchPaths = includePaths;
    sessionDesc.searchPathCount = numIncludePaths;
    sessionDesc.defaultMatrixLayoutMode = settings->is_row_major ? SLANG_MATRIX_LAYOUT_ROW_MAJOR : SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

#define ENTRY_CNT 4
    CompilerOptionEntry compilerOptionEntries[ENTRY_CNT];
    CompilerOptionValue entryValue;
    entryValue.kind = CompilerOptionValueKind::Int;
    entryValue.intValue0 = 0;

#if defined SLANG_OBFUSCATE
    entryValue.intValue0 = 1;
#endif
    compilerOptionEntries[0] = { CompilerOptionName::Obfuscate, entryValue };

    entryValue.intValue0 = 0;
    entryValue.intValue0 = settings->use_glslang ? 1 : 0;
    compilerOptionEntries[1] = { CompilerOptionName::EmitSpirvViaGLSL, entryValue };

    entryValue.intValue0 = SLANG_DEBUG_INFO_LEVEL_MINIMAL;
#ifdef _DEBUG
    entryValue.intValue0 = SLANG_DEBUG_INFO_LEVEL_MAXIMAL;
#endif
    compilerOptionEntries[2] = { CompilerOptionName::DebugInformation, entryValue };

    entryValue.intValue0 = (settings->compile_target == PNANOVDB_COMPILE_TARGET_CPU) ? 1 : 0;
    compilerOptionEntries[3] = { CompilerOptionName::DumpIntermediates, entryValue };

    // TODO: this is not working, string value is not set
    //entryValue.stringValue0 = dumpPrefix.c_str();
    //compilerOptionEntries[4] = { CompilerOptionName::DumpIntermediatePrefix, entryValue };

    sessionDesc.compilerOptionEntryCount = ENTRY_CNT;
    sessionDesc.compilerOptionEntries = compilerOptionEntries;

    ISession* slangSession = nullptr;
    globalSession_->createSession(sessionDesc, &slangSession);

    if (!slangSession)
    {
        printf("Error: Creating Slang session failed\n");
        return false;
    }

    ICompileRequest* request = nullptr;
    const SlangResult requestRes = slangSession->createCompileRequest(&request);
    if (SLANG_FAILED(requestRes) || !request)
    {
        return false;
    }

    int targetIndex = 0;

    if (settings->compile_target == PNANOVDB_COMPILE_TARGET_CPU)
    {
        if (!hasSlangLlvm_)
        {
            printf("Error: Slang LLVM not found\n");
            return false;
        }
        targetIndex = request->addCodeGenTarget(SLANG_SHADER_HOST_CALLABLE);
        request->setTargetFlags(targetIndex, SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM);
        request->setDumpIntermediatePrefix(dumpPrefix.c_str());
    }
    else if (settings->use_glslang)
    {
        request->setPassThrough(SLANG_PASS_THROUGH_GLSLANG);
    }

    const int translationUnitIndex = request->addTranslationUnit(SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
    request->addTranslationUnitSourceString(translationUnitIndex, codeFileName, codeString);

    std::string entryPointName = settings->entry_point_name;
    if (entryPointName.empty())
    {
        entryPointName = (settings->compile_target == PNANOVDB_COMPILE_TARGET_CPU ? "computeMain" : "main");
    }
    const int entryPointIndex = request->addEntryPoint(translationUnitIndex, entryPointName.c_str(), SLANG_STAGE_COMPUTE);

    const SlangResult compileRes = request->compile();

    if (auto diagnostics = request->getDiagnosticOutput())
    {
        if (diagnosticCallback_)
        {
            diagnosticCallback_(diagnostics);
        }
        else
        {
            printf("%s", diagnostics);
        }
    }

    if (SLANG_FAILED(compileRes))
    {
        request->Release();
        slangSession->Release();
        return false;
    }

    IComponentType* entryPoint = nullptr;
    request->getEntryPoint(entryPointIndex, &entryPoint);

    IComponentType* components[] = { entryPoint };
    IComponentType* composedProgram;
    SlangResult result = slangSession->createCompositeComponentType(components, 1, &composedProgram);
    if (SLANG_FAILED(result))
    {
        request->Release();
        slangSession->Release();
        return false;
    }

    // Use layout to get user parameters
    auto programLayout = composedProgram->getLayout();
    auto userParams = programLayout->findTypeByName("UserParams");
    if (userParams && userParams->getKind() == slang::TypeReflection::Kind::Struct)
    {
        for (uint32_t i = 0; i < userParams->getFieldCount(); i++)
        {
            auto member = userParams->getFieldByIndex(i);
            std::string name = member->getName();
            std::string typeName;
            size_t elementCount = 1u;

            auto type = member->getType();
            auto kind = type->getKind();
            if (kind == slang::TypeReflection::Kind::Scalar)
            {
                typeName = getScalarTypeName(type->getScalarType());
            }
            else if (kind == slang::TypeReflection::Kind::Vector)
            {
                typeName = type->getElementType()->getName();
                elementCount = type->getElementCount();
            }
            else if (kind == slang::TypeReflection::Kind::Matrix)
            {
                typeName = type->getElementType()->getName();
                elementCount = size_t(type->getRowCount() * type->getColumnCount());
            }
            shader_->userParameters[name] = { typeName, elementCount };
        }
    }

    // Use reflection to get shader parameters
    auto reflection = (ShaderReflection*)request->getReflection();
    uint32_t parameterCount = reflection->getParameterCount();
    shader_->parameters.clear();
    for (uint32_t i = 0; i != parameterCount; i++)
    {
        slang::VariableLayoutReflection* parameter = reflection->getParameterByIndex(i);
        const char* name = parameter->getName();
        const auto typeKind = (SlangTypeKind)parameter->getType()->getKind();
        const auto shape = parameter->getType()->getResourceShape();
        const auto access = parameter->getType()->getResourceAccess();

        shader_->parameters.push_back({ name, resourceTypeFromSlangLookup(typeKind, shape, access) });
    }

    shader_->computeShader.entryPointName = entryPointName;
    shader_->computeShader.compileTarget = settings->compile_target;

    if (settings->compile_target == PNANOVDB_COMPILE_TARGET_CPU)
    {
        Slang::ComPtr<ISlangSharedLibrary> sharedLibrary;
        const SlangResult sharedLibResult = request->getTargetHostCallable(targetIndex, sharedLibrary.writeRef());
        // TODO: above is deprecated but this one throws an internal error
        //const SlangResult sharedLibResult = composedProgram->getEntryPointHostCallable(entryPointIndex, targetIndex, sharedLibrary.writeRef());

        if (SLANG_FAILED(sharedLibResult) || !sharedLibrary)
        {
            printf("Error: Failed to get target host callable\n");
            request->Release();
            slangSession->Release();
            return false;
        }

        auto func = (CPPPrelude::ComputeFunc)sharedLibrary->findFuncByName(entryPointName.c_str());
        if (!func)
        {
            printf("Error: Failed to find entry point function '%s'\n", entryPointName.c_str());
            request->Release();
            slangSession->Release();
            return false;
        }

        sharedLibraries_[shader_->computeShader.hash] = std::move(sharedLibrary);

        readIntermediateFiles(tempDir);
        std::filesystem::current_path(originalPath);
        removeDirectory(tempDir);
    }
    else
    {
        // Get shader
        ISlangBlob* computeShaderBlob = nullptr;
        request->getEntryPointCodeBlob(entryPointIndex, 0, &computeShaderBlob);
        if (computeShaderBlob)
        {
            // Copy the bytecode so we take care of its lifetime
            shader_->computeShader.byteCodeSize = computeShaderBlob->getBufferSize();
            pnanovdb_shader::ByteCodePtr byteCode(new char[shader_->computeShader.byteCodeSize], std::default_delete<char[]>());
            std::memcpy(byteCode.get(), computeShaderBlob->getBufferPointer(), shader_->computeShader.byteCodeSize);
            shader_->computeShader.byteCode = std::move(byteCode);
            computeShaderBlob->Release();
        }

        shader_->computeShader.isHlsl = settings->hlsl_output;
    }

    request->Release();
    slangSession->Release();

    return true;
}

std::filesystem::path SlangCompiler::createTempDirectory(uint64_t id)
{
    std::filesystem::path tempDir = std::filesystem::temp_directory_path() / ("slang_" + std::to_string(id));
    std::filesystem::create_directories(tempDir);
    return tempDir;
}

void SlangCompiler::readIntermediateFiles(const std::filesystem::path& directory)
{
    shader_->computeShader.intermediateFileNames.clear();
    shader_->computeShader.intermediateFiles.clear();

    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file())
        {
            std::ifstream file(entry.path(), std::ios::binary);
            if (!file.is_open())
            {
                printf("Error: Failed to open intermediate file '%s'\n", entry.path().string().c_str());
                continue;
            }

            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            shader_->computeShader.intermediateFiles.push_back(content);

            std::string fileName = entry.path().stem().string();
            shader_->computeShader.intermediateFileNames.push_back(fileName);
        }
    }
}

void SlangCompiler::removeDirectory(const std::filesystem::path& directory)
{
    std::filesystem::remove_all(directory);
}

ShaderDataPtr compileShader(SlangCompiler& compiler, const pnanovdb_compute_shader_source_t* source, const pnanovdb_compiler_settings_t* settings)
{
    std::string code = source->source == nullptr ? "" : source->source;
    if (source->source_filename != nullptr)
    {
        std::string shaderPath = pnanovdb_shader::getShaderFilePath(source->source_filename);
        std::ifstream inFile(shaderPath);
        if (inFile)
        {
            code = std::string((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
            inFile.close();
        }
        else
        {
            printf("Error: Failed to open file with shader source: '%s'\n", shaderPath.c_str());
        }
    }

    std::string includesPath = pnanovdb_shader::getIncludePath((const char*)source->get_source_include_userdata);

    std::ifstream file(includesPath);
    std::string includesStr;
    if (file.is_open())
    {
        std::ostringstream sstream;
        sstream << file.rdbuf();
        file.close();

        includesStr = sstream.str();
    }
    else
    {
        printf("Error: Failed to open file with includes: '%s'\n", includesPath.c_str());
    }

    std::vector<const char*> includePaths;

    std::string path = pnanovdb_shader::getShaderDir();
    char* includes = (char*)includesStr.c_str();
    if (source->get_source_include)
    {
        while (includes)
        {
            // include path is concatenated and new char[] is returned, delete after compilation
            const char* includePath = source->get_source_include(&includes, path.c_str());
            includePaths.push_back(includePath);
        }
    }

    std::filesystem::path fsPath(source->source_filename);
    const bool result = compiler.compile(settings, fsPath.filename().string().c_str(), code.c_str(), includePaths.size(), includePaths.data());

    for (auto& includePath : includePaths)
    {
        delete[] includePath;
        includePath = nullptr;
    }

    if (!result)
    {
        printf("Error: Slang shader compilation failed\n");
        return nullptr;
    }

    // copy result of slang compilation
    auto shaderData = compiler.getShaderData();
    if (!shaderData)
    {
        printf("Error: Slang shader was not compiled\n");
        return nullptr;
    }

    printf("Slang shader '%s' was compiled\n", source->source_filename);

    return shaderData;
}

bool executeCpu(const SlangCompiler& compiler, const pnanovdb_shader::ShaderDesc& shader, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ, void* uniformParams, void* uniformState)
{
    CPPPrelude::ComputeFunc func = compiler.getComputeFunc(shader);
    if (!func)
    {
        printf("Error: Failed to get compute function for '%s'\n", shader.filePath.c_str());
        return false;
    }

    CPPPrelude::ComputeVaryingInput varyingInput;
    varyingInput.startGroupID = { 0, 0, 0 };
    varyingInput.endGroupID = { groupCountX, groupCountY, groupCountZ };

    func(
        &varyingInput,
        uniformParams,
        uniformState
    );

    return true;
}

void setDiagnosticCallback(SlangCompiler& compiler, pnanovdb_compiler_diagnostic_callback callback)
{
    compiler.setDiagnosticCallback(reinterpret_cast<SlangCompiler::DiagnosticCallback>(callback));
}
}
