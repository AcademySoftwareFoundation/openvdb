// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compiler/SlangCompiler.h

    \author Petra Hapalova

    \brief
*/

#pragma once

#include "CompiledShader.h"
#include "nanovdb_editor/putil/Compiler.h"

#define SLANG_PRELUDE_NAMESPACE CPPPrelude

#include <slang.h>

#include <slang-cpp-types.h>
#include <slang-com-helper.h>
#include <slang-com-ptr.h>

#include <string>
#include <vector>

namespace pnanovdb_compiler
{
    typedef std::shared_ptr<pnanovdb_shader::ShaderData> ShaderDataPtr;

    class SlangCompiler
    {
    public:
        using DiagnosticCallback = void(*)(const char* message);

        SlangCompiler();
        ~SlangCompiler();

        void setDiagnosticCallback(DiagnosticCallback callback)
        {
            diagnosticCallback_ = callback;
        }

        bool compileFile(const char* sourceFile,
                        const char* destinationFile,
                        const char* variableName,
                        size_t numIncludePaths,
                        const char** includePaths,
                        const pnanovdb_compiler_settings_t* settings);
        bool compile(const pnanovdb_compiler_settings_t* settings,
                    const char* codeFileName,
                    const char* codeString,
                    size_t numIncludePaths = 0u,
                    const char** includePaths = nullptr);

        ShaderDataPtr getShaderData() const
        {
            return shader_;
        }

        std::vector<pnanovdb_shader::ShaderParameter> getShaderParameters() const
        {
            if (shader_)
            {
                return shader_->parameters;
            }
            return {};
        }

        CPPPrelude::ComputeFunc getComputeFunc(const pnanovdb_shader::ShaderDesc& shader) const
        {
            auto it = sharedLibraries_.find(shader.hash);
            if (it != sharedLibraries_.end())
            {
                Slang::ComPtr<ISlangSharedLibrary> sharedLibrary = it->second;
                return (CPPPrelude::ComputeFunc)sharedLibrary->findFuncByName(shader.entryPointName.c_str());
            }
            return nullptr;
        }

    private:
        SlangCompiler(const SlangCompiler&) = delete;
        SlangCompiler& operator=(const SlangCompiler&) = delete;

        // Helper methods for intermediate file handling
        std::filesystem::path createTempDirectory(uint64_t id);
        void readIntermediateFiles(const std::filesystem::path& directory);
        void removeDirectory(const std::filesystem::path& directory);

        slang::IGlobalSession* globalSession_ = nullptr;
        ShaderDataPtr shader_ = nullptr;
        bool hasSlangLlvm_ = false;
        std::map<uint64_t, Slang::ComPtr<ISlangSharedLibrary>> sharedLibraries_;
        DiagnosticCallback diagnosticCallback_ = nullptr;
    };

    PNANOVDB_CAST_PAIR(pnanovdb_compiler_instance_t, SlangCompiler)

    ShaderDataPtr compileShader(SlangCompiler& compiler, const pnanovdb_compute_shader_source_t* source, const pnanovdb_compiler_settings_t* settings);
    bool executeCpu(const SlangCompiler& compiler, const pnanovdb_shader::ShaderDesc& shader, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ, void* entryPointParams, void* uniformState);
    void setDiagnosticCallback(SlangCompiler& compiler, pnanovdb_compiler_diagnostic_callback callback);
} // namespace pnanovdb_compiler
