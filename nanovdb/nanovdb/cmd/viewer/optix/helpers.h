
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <stdint.h>
#include <vector_types.h>
#include "vec_math.h"

#define float3_as_args(u) \
    reinterpret_cast<uint32_t&>((u).x), \
        reinterpret_cast<uint32_t&>((u).y), \
        reinterpret_cast<uint32_t&>((u).z)

#define array3_as_args(u) \
    reinterpret_cast<uint32_t&>((u)[0]), \
        reinterpret_cast<uint32_t&>((u)[1]), \
        reinterpret_cast<uint32_t&>((u)[2])