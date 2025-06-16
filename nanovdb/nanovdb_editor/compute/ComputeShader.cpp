// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compute/ComputeShader.cpp

    \author Petra Hapalova

    \brief
*/

#include "ComputeShader.h"

namespace pnanovdb_compute
{
bool load_shader(const char *filePath, pnanovdb_shader::ShaderData *shader)
{
    using namespace pnanovdb_shader;
    if (filePath == nullptr)
    {
        printf("Error: Could not load shader from a file, path is empty\n");
        return false;
    }
    nlohmann::json json;
    const std::string jsonFilePath = getShaderCacheFilePath(filePath) + JSON_EXT;
    std::ifstream inFile(jsonFilePath);
    if (inFile.is_open())
    {
        try
        {
            inFile >> json;
        }
        catch (const nlohmann::json::exception &e)
        {
            printf("Error: Failed to parse JSON file: %s", e.what());
            return false;
        }
        inFile.close();
        if (shader)
        {
            *shader = json.get<ShaderData>();
        }
        return true;
    }

    return false;
}

pnanovdb_compute_shader_t* copy_shader(const pnanovdb_shader::ShaderData* shader)
{
    if (!shader)
    {
        return nullptr;
    }

    const auto& shaderParameters = shader->parameters;

    pnanovdb_compute_pipeline_desc_t pipelineDesc;
    pipelineDesc.bytecode = { shader->computeShader.byteCode.get(), shader->computeShader.byteCodeSize };
    pipelineDesc.binding_desc_count = shaderParameters.size();
    pipelineDesc.binding_descs = new pnanovdb_compute_binding_desc_t[pipelineDesc.binding_desc_count];

    pnanovdb_uint32_t descriptor_write_count = pipelineDesc.binding_desc_count;
    auto* descriptor_writes = new pnanovdb_compute_descriptor_write_t[descriptor_write_count];
    const char** resource_names = new const char*[descriptor_write_count];

    for (size_t i = 0u; i < shaderParameters.size(); ++i)
    {
        pnanovdb_compute_binding_desc_t bindingDesc;
        bindingDesc.type = shaderParameters[i].type;

        pnanovdb_compute_binding_desc_vulkan_t bindingVulkan;
        bindingVulkan.binding = i;
        bindingVulkan.descriptor_count = 1u;
        bindingVulkan.set = 0u;

        pnanovdb_compute_binding_desc_union_t bindingDescUnion;
        bindingDescUnion.vulkan = bindingVulkan;
        bindingDesc.binding_desc = bindingDescUnion;

        pipelineDesc.binding_descs[i] = bindingDesc;

        pnanovdb_compute_descriptor_write_t descriptorWrite;
        descriptorWrite.type = bindingDesc.type;

        pnanovdb_compute_descriptor_write_vulkan_t descriptorVulkan;
        descriptorVulkan.binding = i;
        descriptorVulkan.array_index = 0u;
        descriptorVulkan.set = 0u;

        pnanovdb_compute_descriptor_write_union_t descriptorWriteUnion;
        descriptorWriteUnion.vulkan = descriptorVulkan;
        descriptorWrite.write = descriptorWriteUnion;

        descriptor_writes[i] = descriptorWrite;

        char* name = new char[shaderParameters[i].name.size() + 1];
        strcpy(name, shaderParameters[i].name.c_str());
        resource_names[i] = name;
    }

    auto* computeShader = new compute_shader_t();
    computeShader->build = pnanovdb_compute_shader_build_t({ pipelineDesc, descriptor_writes, resource_names, descriptor_write_count, "" });
    computeShader->byteCodePtr = shader->computeShader.byteCode;

    return cast(computeShader);
}

pnanovdb_compute_shader_t* create_shader(const pnanovdb_compute_shader_source_t* source)
{
    pnanovdb_shader::ShaderData loadedShader;
    bool result = load_shader(source->source_filename, &loadedShader);
    if (result)
    {
        return copy_shader(&loadedShader);
    }

    return nullptr;
}

pnanovdb_bool_t map_shader_build(pnanovdb_compute_shader_t* shader, pnanovdb_compute_shader_build_t** out_build)
{
    auto compute_shader = cast(shader);
    if (!shader)
    {
        return PNANOVDB_FALSE;
    }

    *out_build = &compute_shader->build;

    return PNANOVDB_TRUE;
}

void destroy_shader(pnanovdb_compute_shader_t* shader)
{
    auto compute_shader = cast(shader);
    if (compute_shader->build.pipeline_desc.binding_descs)
    {
        delete[] compute_shader->build.pipeline_desc.binding_descs;
        compute_shader->build.pipeline_desc.binding_descs = nullptr;
    }

    if (compute_shader->build.descriptor_writes)
    {
        delete[] compute_shader->build.descriptor_writes;
        compute_shader->build.descriptor_writes = nullptr;
    }

    for (pnanovdb_uint32_t i = 0u; i < compute_shader->build.descriptor_write_count; ++i)
    {
        if (compute_shader->build.resource_names[i])
        {
            delete[] compute_shader->build.resource_names[i];
            compute_shader->build.resource_names[i] = nullptr;
        }
    }

    if (compute_shader->build.resource_names)
    {
        delete[] compute_shader->build.resource_names;
        compute_shader->build.resource_names = nullptr;
    }

    if (compute_shader)
    {
        delete compute_shader;
        compute_shader = nullptr;
    }
}
}

pnanovdb_compute_shader_interface_t* pnanovdb_get_compute_shader_interface()
{
    using namespace pnanovdb_compute;
    static pnanovdb_compute_shader_interface_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_compute_shader_interface_t) };

    iface.create_shader = create_shader;
    iface.map_shader_build = map_shader_build;
    iface.destroy_shader = destroy_shader;

    return &iface;
}
