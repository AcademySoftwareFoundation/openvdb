// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compiler/CompiledShader.h

    \author Petra Hapalova

    \brief
*/

#pragma once

#include <nanovdb_editor/putil/Shader.hpp>

namespace pnanovdb_shader
{
    void remove_shader(const char* filePath);
    bool save_shader(const char* filePath, pnanovdb_shader::ShaderData& shader);
    bool get_shader(const char* filePath, pnanovdb_shader::ShaderDesc& shader);
    bool has_shader(const char* filePath, pnanovdb_compiler_settings_t** settings);
}
