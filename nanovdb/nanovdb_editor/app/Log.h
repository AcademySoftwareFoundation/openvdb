// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/app/Log.h

    \author Petra Hapalova, Andrew Reidmeyer

    \brief
*/

#pragma once

#include <nanovdb_editor/putil/Compute.h>

#include <stdio.h>
#include <stdarg.h>

inline void pnanovdb_compute_log_print(pnanovdb_compute_log_level_t level, const char* format, ...)
{
    va_list args;
    va_start(args, format);

    const char* prefix = "Unknown";
    if (level == PNANOVDB_COMPUTE_LOG_LEVEL_ERROR)
    {
        prefix = "Error";
    }
    else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_WARNING)
    {
        prefix = "Warning";
    }
    else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_INFO)
    {
        prefix = "Info";
    }
    printf("%s: ", prefix);
    vprintf(format, args);
    printf("\n");

    va_end(args);
}
