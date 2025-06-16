// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compiler/Compiler.cpp

    \author Petra Hapalova

    \brief
*/

#include "Compiler.h"
#include "SlangCompiler.h"

namespace pnanovdb_compiler
{
// userdata are include paths, function should return single include path and leave userdata with remaining include paths
// path is the base path prepended to the include path
pnanovdb_compute_shader_get_source_t get_source_include = [](void* userdata, const char* path) -> const char*
{
    const char* include = nullptr;
    char** includesPtr = (char**)userdata;
    char* includes = *includesPtr;
    char* newline = strchr(includes, '\n');
    if (!newline)
    {
        *includesPtr = nullptr;
        include = (const char*)includes;
    }
    else
    {
        *newline = '\0';
        include = includes;
        *includesPtr = newline + 1;
    }
    size_t length = strlen(path) + strlen(include) + 1;
    char* result = new char[length];
    strcpy(result, path);
    strcat(result, include);
    return result;
};

pnanovdb_compiler_instance_t* create_instance()
{
    SlangCompiler* ptr = new SlangCompiler();
    return cast(ptr);
}

void destroy_instance(pnanovdb_compiler_instance_t* instance)
{
    auto ptr = cast(instance);
    if (ptr)
    {
        delete ptr;
    }
}

pnanovdb_bool_t compile_file(pnanovdb_compiler_instance_t* instance, const char* filename, pnanovdb_compiler_settings_t* settings, pnanovdb_bool_t* shader_updated)
{
    if (shader_updated)
    {
        *shader_updated = PNANOVDB_FALSE;
    }

    bool result = pnanovdb_shader::has_shader(filename, &settings);
    if (result)
    {
        if (settings->compile_target == PNANOVDB_COMPILE_TARGET_CPU)
        {
            pnanovdb_shader::ShaderDesc shader;
            pnanovdb_shader::get_shader(filename, shader);
            auto compilerPtr = cast(instance);
            if (compilerPtr)
            {
                auto sharedLib = compilerPtr->getComputeFunc(shader);
                if (sharedLib)
                {
                    // shared library already exists and shader hasn't changed
                    return PNANOVDB_TRUE;
                }
            }
        }
        else
        {
            // cached shader already exists and hasn't changed
            return PNANOVDB_TRUE;
        }
    }

    auto compilerPtr = cast(instance ? instance : create_instance());

    pnanovdb_compute_shader_source_t source = {};
    source.source = nullptr;
    source.source_filename = filename;
    source.get_source_include_userdata = (void*)NANOVDB_EDITOR_INCLUDES_DIR;
    source.get_source_include = pnanovdb_compiler::get_source_include;

    auto shader = compileShader(*compilerPtr, &source, settings);

    if (!instance)
    {
        destroy_instance(cast(compilerPtr));
    }

    if (shader)
    {
        bool result = pnanovdb_shader::save_shader(source.source_filename, *shader);
        if (result)
        {
            printf("Compiled shader '%s' was updated\n", source.source_filename);

            if (shader_updated)
            {
                *shader_updated = PNANOVDB_TRUE;
            }
            return PNANOVDB_TRUE;
        }
    }
    pnanovdb_shader::remove_shader(source.source_filename);

    return PNANOVDB_FALSE;
}

pnanovdb_bool_t execute_cpu(pnanovdb_compiler_instance_t* instance, const char* filename, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ, void* uniformParams, void* uniformState)
{
    pnanovdb_shader::ShaderDesc shader;
    bool result = pnanovdb_shader::get_shader(filename, shader);
    if (!result)
    {
        printf("Error: Shader '%s' was not compiled\n", filename);
        return PNANOVDB_FALSE;
    }

    auto compilerPtr = cast(instance);

    return executeCpu(*compilerPtr, shader, groupCountX, groupCountY, groupCountZ, uniformParams, uniformState);
}

void set_diagnostic_callback(pnanovdb_compiler_instance_t* instance, pnanovdb_compiler_diagnostic_callback callback)
{
    auto compilerPtr = cast(instance);
    setDiagnosticCallback(*compilerPtr, callback);
}

PNANOVDB_API pnanovdb_compiler_t* pnanovdb_get_compiler()
{
    static pnanovdb_compiler_t compiler = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_compiler_t) };

    compiler.create_instance = create_instance;
    compiler.set_diagnostic_callback = set_diagnostic_callback;
    compiler.compile_shader_from_file = compile_file;
    compiler.execute_cpu = execute_cpu;
    compiler.destroy_instance = destroy_instance;
    return &compiler;
}
}
