// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/app/Node2Convert.h

    \author Andrew Reidmeyer

    \brief
*/

#define PNANOVDB_C
#define PNANOVDB_CMATH
#include "nanovdb/PNanoVDB2.h"

#include "nanovdb_editor/putil/Reflect.h"

namespace pnanovdb_editor
{
    void node2_convert(const char* nvdb_path, const char* dst_path);

    void node2_sphere(const char* dst_path);
}
