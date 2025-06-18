// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compiler/CompiledShader.cpp

    \author Petra Hapalova

    \brief
*/

#include "CompiledShader.h"

namespace pnanovdb_shader
{
    void remove_shader(const char *filePath)
    {
        if (filePath == nullptr)
        {
            printf("Error: Could not remove cached shader file, path is empty\n");
            return;
        }
        const std::string shaderFilePath = getShaderCacheFilePath(filePath) + SHADER_EXT;
        std::remove(shaderFilePath.c_str());
        const std::string jsonFilePath = getShaderCacheFilePath(filePath) + JSON_EXT;
        std::remove(jsonFilePath.c_str());
    }

    bool save_shader(const char *filePath, ShaderData &shader)
    {
        if (filePath == nullptr)
        {
            printf("Error: Could not save shader to a file, path is empty\n");
            return false;
        }
        shader.computeShader.filePath = getShaderFilePath(filePath);

        if (shader.computeShader.isHlsl)
        {
            // only save the compiler output
            const std::string shaderFilePath = getShaderCacheFilePath(filePath) + SHADER_HLSL_EXT;
            std::ofstream outFile(shaderFilePath, std::ios::out);
            if (outFile)
            {
                outFile.write(shader.computeShader.byteCode.get(), shader.computeShader.byteCodeSize);
            }
            else
            {
                printf("Error: file '%s' could not be saved\n", shaderFilePath.c_str());
            }
            return true;
        }

        const std::string jsonFilePath = getShaderCacheFilePath(filePath) + JSON_EXT;
        nlohmann::ordered_json json = shader;
        std::ofstream outFile(jsonFilePath);
        if (outFile)
        {
            outFile << json.dump(4);
            outFile.close();
            return true;
        }
        return false;
    }

    bool get_shader(const char* relFilePath, ShaderDesc& shader)
    {
        nlohmann::json json = loadShaderJson(relFilePath);
        if (json.empty())
        {
            return false;
        }
        shader = json.get<ShaderDesc>();
        return true;
    }

    bool has_shader(const char* relFilePath, pnanovdb_compiler_settings_t** settings)
    {
        ShaderDesc shader;
        if (!get_shader(relFilePath, shader))
        {
            return false;
        }
        if (settings == nullptr || *settings == nullptr)
        {
            return false;
        }
        const std::string filePath = getShaderFilePath(relFilePath);
        std::ifstream sourceFile(filePath, std::ios::in);
        if (!sourceFile)
        {
            printf("Error: Shader source file '%s' could not be opened\n", filePath.c_str());
            return false;
        }
        std::string code((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
        sourceFile.close();

        std::size_t hash = getHash(code.c_str());
        if (shader.hash != hash)
        {
            // compiled source has changed
            return false;
        }

        for (const auto& includedFile : shader.includedFiles)
        {
            std::string includeFilePath = getShaderFilePath(std::get<0>(includedFile).c_str());
            std::ifstream includeFile(includeFilePath, std::ios::in);
            if (!includeFile)
            {
                printf("Error: Shader source file '%s' is missing included file '%s'\n", filePath.c_str(), includeFilePath.c_str());
                return false;
            }
            std::string includeCode((std::istreambuf_iterator<char>(includeFile)), std::istreambuf_iterator<char>());
            includeFile.close();
            if (getHash(includeCode.c_str()) != std::get<1>(includedFile))
            {
                // included file has changed
                return false;
            }
        }

        if ((*settings)->compile_target == PNANOVDB_COMPILE_TARGET_UNKNOWN)
        {
            // this was triggered with empty settings, check source changes and load settings from existing cache
            (*settings)->is_row_major = shader.isRowMajor;
            (*settings)->hlsl_output = shader.isHlsl;
            (*settings)->compile_target = shader.compileTarget;
            strcpy((*settings)->entry_point_name, shader.entryPointName.c_str());
            return true;
        }

        if (shader.isRowMajor != bool((*settings)->is_row_major))
        {
            // matrix layout has changed
            return false;
        }

        if (shader.compileTarget != (*settings)->compile_target)
        {
            // compiler target has changed
            return false;
        }

        if ((*settings)->hlsl_output)
        {
            std::string hlslFilePath = getShaderCacheFilePath(relFilePath) + SHADER_HLSL_EXT;
            std::ifstream hlslFile(hlslFilePath, std::ios::in);
            if (!hlslFile)
            {
                printf("Error: Shader source file '%s' is missing generated hlsl output\n", hlslFilePath.c_str());
                return false;
            }
            sourceFile.close();
        }

        return true;
    }
}
