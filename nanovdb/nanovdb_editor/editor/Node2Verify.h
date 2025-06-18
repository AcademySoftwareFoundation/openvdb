// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/Node2Convert.h

    \author Andrew Reidmeyer

    \brief
*/

#define PNANOVDB_C
#include "nanovdb/PNanoVDB2.h"

#include "nanovdb_editor/putil/Reflect.h"

namespace pnanovdb_editor
{
    int node2_verify_gpu(const char* nvdb_ref_filepath, const char* nvdb_filepath);
}
