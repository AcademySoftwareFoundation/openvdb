// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>
#include <nanovdb/NanoVDB.h>
#include "ComputePrimitives.h"

// http://www.burtleburtle.net/bob/hash/doobs.html
inline __hostdev__ uint32_t hash(uint32_t x)
{
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

inline __hostdev__ float randomf(uint32_t s)
{
    return hash(s) / float(0xffffffffu);
}
