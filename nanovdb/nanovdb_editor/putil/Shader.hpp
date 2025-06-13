// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Shader.hpp

    \author Petra Hapalova

    \brief
*/

#pragma once

#include "nanovdb_editor/putil/Compute.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pnanovdb_shader
{
#if defined(USE_SLANG_DEBUG_OUTPUT)
    static const char* SHADER_EXT = ".spv_asm";
#else
    static const char* SHADER_EXT = ".spv";
#endif
    static const char* SHADER_HLSL_EXT = ".hlsl";
    static const char* SHADER_CPP_EXT = ".cpp";
    static const char* JSON_EXT = ".json";
    // Version 0.3: Added UserParams
    static const char* SHADER_DESC_VERSION = "0.3";

    typedef std::shared_ptr<char[]> ByteCodePtr;
    typedef std::function<void(const char* filename, pnanovdb_uint32_t grid_dim_x, pnanovdb_uint32_t grid_dim_y, pnanovdb_uint32_t grid_dim_z)> run_shader_func_t;

    static uint64_t getTimestamp()
    {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    }

    static std::size_t getHash(const char* code)
    {
        std::hash<std::string> hasher;
        return hasher(code);
    }

    static bool isSymlink(const std::filesystem::path& path)
    {
    #ifdef _WIN32
        DWORD attributes = GetFileAttributes(path.string().c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES)
        {
            return false;
        }
        return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
    #else
        return std::filesystem::is_symlink(path);
    #endif
    }

    static std::filesystem::path resolveSymlink(const std::string& filePath)
    {
        std::filesystem::path path(filePath);
#ifdef _WIN32
        HANDLE hFile = CreateFile(path.string().c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
        if (hFile == INVALID_HANDLE_VALUE)
        {
            return path;
        }

        WCHAR targetPath[MAX_PATH];
        DWORD result = GetFinalPathNameByHandleW(hFile, targetPath, MAX_PATH, FILE_NAME_NORMALIZED);
        CloseHandle(hFile);
        if (result == 0)
        {
            return path;
        }
        return std::filesystem::canonical(targetPath);
    #else
        if (std::filesystem::exists(filePath))
        {
            return std::filesystem::canonical(path);
        }
    #endif
        return "";
    }

    static std::filesystem::path getCurrentDirectory()
    {
#ifdef _WIN32
        // Get module handle of the current DLL/executable
        HMODULE hModule = nullptr;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             (LPCSTR)&getCurrentDirectory,
                             &hModule))
        {
            WCHAR path[MAX_PATH];
            if (GetModuleFileNameW(hModule, path, MAX_PATH) != 0)
            {
                return std::filesystem::path(path).parent_path();
            }
        }
#else
        Dl_info dlInfo;
        if (dladdr((void*)&getCurrentDirectory, &dlInfo))
        {
            return std::filesystem::path(dlInfo.dli_fname).parent_path();
        }
#endif
        return "";
    }

    static std::filesystem::path getDirectoryNextToCurrent(const std::string& folderName)
    {
        return getCurrentDirectory().parent_path() / std::filesystem::path(folderName);
    }

    static std::string getShaderDir()
    {
        std::string shaderDir = getDirectoryNextToCurrent(NANOVDB_EDITOR_SHADER_DIR).string();
        return resolveSymlink(shaderDir).string();
    }

    // returns relative file path of the shader
    static std::string getShaderName(const char* filePath)
    {
        std::filesystem::path fsPath(filePath);
        std::filesystem::path shaderDir = getShaderDir();
        return fsPath.lexically_relative(shaderDir).generic_string();
    }

    // returns full file path of the shader
    static std::string getShaderFilePath(const char* relFilePath)
    {
        std::filesystem::path fsPath(relFilePath);
        if (std::filesystem::exists(fsPath))
        {
            return fsPath.string();
        }

        // resolve relative path
        std::string shaderPath = (std::filesystem::path(getShaderDir()) / fsPath).string();
        return resolveSymlink(shaderPath).string();
    }

    // returns file path without the extension
    static std::string getShaderCacheFilePath(const char* relFilePath)
    {
        std::filesystem::path fsPath(relFilePath);
        std::filesystem::path shaderCache = getDirectoryNextToCurrent(NANOVDB_EDITOR_SHADER_CACHE);

        // C/C++ application has shader cache next to the lib directory, python module uses temporary directory
        if (!std::filesystem::exists(shaderCache))
        {
            shaderCache = std::filesystem::temp_directory_path() / NANOVDB_EDITOR_TEMP_SHADER_CACHE;
            if (!std::filesystem::exists(shaderCache))
            {
                std::filesystem::create_directories(shaderCache);
            }
        }
        return (shaderCache / fsPath.filename()).string();
    }

    static std::string getUserParamsFilePath(const char* relFilePath)
    {
        return getShaderFilePath(relFilePath) + JSON_EXT;
    }

    static std::string getShaderParamsFilePath(const char* relFilePath)
    {
        return getShaderCacheFilePath(relFilePath) + JSON_EXT;
    }

    static std::string getIncludePath(const char* include)
    {
        std::filesystem::path fsPath(include);
        return (std::filesystem::path(getShaderDir()) / fsPath).string();
    }

    static std::string getGeneratedExtension(uint32_t compileTarget)
    {
        if (compileTarget == PNANOVDB_COMPILE_TARGET_CPU)
        {
            return SHADER_CPP_EXT;
        }
        else if (compileTarget == PNANOVDB_COMPILE_TARGET_VULKAN)
        {
            return SHADER_HLSL_EXT;
        }
        return "";
    }

    static nlohmann::json loadShaderJson(const char* relFilePath)
    {
        nlohmann::json shaderJson;
        if (relFilePath == nullptr)
        {
            printf("Error: Could not load shader from a file, path is empty\n");
            return shaderJson;
        }
        const std::string jsonFilePath = getShaderCacheFilePath(relFilePath) + JSON_EXT;
        std::ifstream inFile(jsonFilePath);
        if (inFile.is_open())
        {
            try
            {
                nlohmann::json json;
                inFile >> json;
                shaderJson = json.at("computeShader");
            }
            catch (const nlohmann::json::exception& e)
            {
                printf("Error: Failed to parse shader file: %s", e.what());
            }
            inFile.close();
        }
        return shaderJson;
    }

    static uint32_t getCompileTarget(const char* shaderName)
    {
        nlohmann::json json = loadShaderJson(shaderName);
        if (json.empty())
        {
            printf("Error: Shader not compiled yet\n");
            return PNANOVDB_COMPILE_TARGET_UNKNOWN;
        }
        return json.value("compileTarget", PNANOVDB_COMPILE_TARGET_UNKNOWN);
    }

    struct ShaderDesc
    {
        uint32_t compileTarget = PNANOVDB_COMPILE_TARGET_UNKNOWN;  // pnanovdb_compile_target_type_t
        std::string entryPointName = "main";
        ByteCodePtr byteCode = nullptr;
        uint64_t byteCodeSize = 0llu;
        uint64_t timestamp = 0llu;
        size_t hash = 0u;
        std::string filePath;            // path to the slang shader
        bool isRowMajor = false;
        bool isHlsl = false;
        std::vector<std::string> intermediateFiles;
        std::vector<std::string> intermediateFileNames;

        friend void to_json(nlohmann::ordered_json& json, const ShaderDesc& shaderDesc)
        {
            const std::string shaderFilePath = getShaderCacheFilePath(shaderDesc.filePath.c_str()) + SHADER_EXT;
            std::ofstream outFile(shaderFilePath, std::ios::out | std::ios::binary);
            if (outFile)
            {
                outFile.write(shaderDesc.byteCode.get(), shaderDesc.byteCodeSize);
                outFile.close();
            }
            else
            {
                printf("Error: file '%s' could not be saved\n", shaderFilePath.c_str());
            }
            for (size_t i = 0; i < shaderDesc.intermediateFileNames.size(); ++i)
            {
                std::string filePath = getShaderCacheFilePath(shaderDesc.intermediateFileNames[i].c_str()) + SHADER_CPP_EXT;
                std::ofstream outFile(filePath, std::ios::out | std::ios::binary);
                if (outFile)
                {
                    outFile.write(shaderDesc.intermediateFiles[i].c_str(), shaderDesc.intermediateFiles[i].size());
                    outFile.close();
                }
                else
                {
                    printf("Error: file '%s' could not be saved\n", filePath.c_str());
                }
            }
            if (shaderDesc.intermediateFiles.size() > 0)
            {
                // save the first intermediate file as the generated shader
                std::string filePath = getShaderCacheFilePath(shaderDesc.filePath.c_str()) + SHADER_CPP_EXT;
                std::ofstream outFile(filePath, std::ios::out | std::ios::binary);
                if (outFile)
                {
                    outFile.write(shaderDesc.intermediateFiles.front().c_str(), shaderDesc.intermediateFiles.front().size());
                    outFile.close();
                }
                else
                {
                    printf("Error: file '%s' could not be saved\n", filePath.c_str());
                }
            }
            json["compileTarget"] = shaderDesc.compileTarget;
            json["entryPointName"] = shaderDesc.entryPointName;
            json["byteCodeSize"] = shaderDesc.byteCodeSize;
            json["timestamp"] = shaderDesc.timestamp;
            json["hash"] = shaderDesc.hash;
            json["filePath"] = shaderDesc.filePath;
            json["isMatrixLayoutRowMajor"] = shaderDesc.isRowMajor;
            json["intermediateFileNames"] = shaderDesc.intermediateFileNames;
        }

        friend void from_json(const nlohmann::ordered_json& json, ShaderDesc& shaderDesc)
        {
            shaderDesc.compileTarget = json.value("compileTarget", shaderDesc.compileTarget);
            shaderDesc.entryPointName = json["entryPointName"].get<std::string>();
            shaderDesc.byteCodeSize = json["byteCodeSize"].get<uint64_t>();
            shaderDesc.timestamp = json["timestamp"].get<uint64_t>();
            shaderDesc.hash = json["hash"].get<size_t>();
            shaderDesc.filePath = json["filePath"].get<std::string>();
            shaderDesc.isRowMajor = json.value("isMatrixLayoutRowMajor", shaderDesc.isRowMajor);
            const std::string shaderFilePath = getShaderCacheFilePath(shaderDesc.filePath.c_str()) + SHADER_EXT;
            std::ifstream binFile(shaderFilePath, std::ios::in | std::ios::binary);
            if (binFile)
            {
                std::shared_ptr<char[]> buffer(new char[shaderDesc.byteCodeSize], std::default_delete<char[]>());
                binFile.read(buffer.get(), shaderDesc.byteCodeSize);
                binFile.close();
                shaderDesc.byteCode = buffer;
            }
            else
            {
                printf("Error: Shader binary file '%s' could not be opened\n", shaderFilePath.c_str());
            }
            shaderDesc.intermediateFileNames = json.value("intermediateFileNames", shaderDesc.intermediateFileNames);
        }
    };

    struct UserParameter
    {
        std::string type;
        size_t elementCount;

        friend void to_json(nlohmann::ordered_json& json, const UserParameter& userParameter)
        {
            json["type"] = userParameter.type;
            json["elementCount"] = userParameter.elementCount;
        }

        friend void from_json(const nlohmann::ordered_json& json, UserParameter& userParameter)
        {
            userParameter.type = json["type"].get<std::string>();
            userParameter.elementCount = json["elementCount"].get<size_t>();
        }
    };

    struct ShaderParameter
    {
        std::string name;
        pnanovdb_compute_descriptor_type_t type;

        friend void to_json(nlohmann::ordered_json& json, const ShaderParameter& shaderParameter)
        {
            json["name"] = shaderParameter.name;
            json["type"] = shaderParameter.type;
        }

        friend void from_json(const nlohmann::ordered_json& json, ShaderParameter& shaderParameter)
        {
            shaderParameter.name = json["name"].get<std::string>();
            shaderParameter.type = json["type"].get<pnanovdb_compute_descriptor_type_t>();
        }
    };

    // this struct is being serialized into json and a binary file
    struct ShaderData
    {
        std::unordered_map<std::string, UserParameter> userParameters;
        std::vector<ShaderParameter> parameters;
        ShaderDesc computeShader;

        void setMetadata(const char* code)
        {
            computeShader.timestamp = getTimestamp();
            computeShader.hash = getHash(code);
        }

        friend void to_json(nlohmann::ordered_json& json, const ShaderData& shader)
        {
            json["version"] = SHADER_DESC_VERSION;
            json["userParams"] = shader.userParameters;
            json["parameters"] = shader.parameters;
            json["computeShader"] = shader.computeShader;
        }

        friend void from_json(const nlohmann::ordered_json& json, ShaderData& shader)
        {
            shader.parameters = json["parameters"].get<std::vector<ShaderParameter>>();
            shader.computeShader = json["computeShader"].get<ShaderDesc>();

            std::string version = json["version"].get<std::string>();
            if (version > "0.2")
            {
                shader.userParameters = json["userParams"].get<std::unordered_map<std::string, UserParameter>>();
            }
        }
    };
}
