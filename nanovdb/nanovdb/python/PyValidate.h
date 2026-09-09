// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYVALIDATE_HAS_BEEN_INCLUDED
#define NANOVDB_PYVALIDATE_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

#include <cmath>
#include <string>

namespace pynanovdb {

/// @brief Raise a Python ValueError unless @a value is a finite, strictly
///        positive number. Used to validate geometric parameters (voxelSize,
///        narrow-band halfWidth) before they reach nanovdb::Map::set and the
///        grid builders, which only debug-assert positivity — release builds
///        would otherwise persist a singular / non-finite transform in the
///        grid header.
inline void requirePositiveFinite(double value, const char* fnName, const char* paramName)
{
    if (!(std::isfinite(value) && value > 0.0)) {
        std::string msg(fnName);
        msg += ": ";
        msg += paramName;
        msg += " must be a finite, strictly positive number; got ";
        msg += std::to_string(value);
        throw nanobind::value_error(msg.c_str());
    }
}

} // namespace pynanovdb

#endif
