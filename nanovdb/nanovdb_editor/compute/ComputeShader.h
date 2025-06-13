// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compute/ComputeShader.h

    \author Petra Hapalova

    \brief
*/

#pragma once

#include "nanovdb_editor/putil/Shader.hpp"

namespace pnanovdb_compute
{
    struct compute_shader_t
    {
        pnanovdb_shader::ByteCodePtr byteCodePtr;        // we need to hold a reference count for the loaded shader
        pnanovdb_compute_shader_build_t build;
    };

    PNANOVDB_CAST_PAIR(pnanovdb_compute_shader_t, compute_shader_t)

    bool load_shader(const char* filePath, pnanovdb_shader::ShaderData* shader);
    pnanovdb_compute_shader_t* copy_shader(const pnanovdb_shader::ShaderData* shader);
}
