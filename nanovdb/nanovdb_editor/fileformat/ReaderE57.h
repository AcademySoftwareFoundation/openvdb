// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/fileformat/ReaderE57.h

    \author Petra Hapalova
    \brief
*/

#pragma once

#include <stddef.h>

namespace pnanovdb_fileformat
{
    void e57_to_float(const char* filename, size_t* array_size, float** positions_array, float** colors_array, float** normals_array = nullptr);
}
