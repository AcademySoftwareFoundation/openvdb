// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifndef NANOVDB_PYVALIDATE_HAS_BEEN_INCLUDED
#define NANOVDB_PYVALIDATE_HAS_BEEN_INCLUDED

#include <nanobind/nanobind.h>

#include <nanovdb/NanoVDB.h> // for NANOVDB_DATA_ALIGNMENT

#include <cmath>
#include <cstdint>
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

/// @brief Raise a Python ValueError unless @a ptr is non-null and aligned to
///        NANOVDB_DATA_ALIGNMENT (32 bytes), the alignment every NanoVDB
///        buffer must satisfy. The C++ buffer classes only assert this (and
///        the CUDA ones exit the process on failure), so wrapping
///        constructors must reject caller-provided memory up front. A plain
///        NumPy allocation is typically 16-byte aligned and fails this check.
inline void requireAlignedBuffer(const void* ptr, const char* fnName, const char* paramName)
{
    std::string msg(fnName);
    msg += ": ";
    msg += paramName;
    if (ptr == nullptr) {
        msg += " must be a non-null pointer";
        throw nanobind::value_error(msg.c_str());
    }
    const uint64_t misalignment = reinterpret_cast<uint64_t>(ptr) % NANOVDB_DATA_ALIGNMENT;
    if (misalignment) {
        msg += " must be aligned to ";
        msg += std::to_string(NANOVDB_DATA_ALIGNMENT);
        msg += " bytes (NANOVDB_DATA_ALIGNMENT); the given address is ";
        msg += std::to_string(misalignment);
        msg += " bytes past an aligned boundary. Allocate an oversized array "
               "and slice to an aligned offset, or use a NanoVDB-owned buffer.";
        throw nanobind::value_error(msg.c_str());
    }
}

/// @brief Raise a Python ValueError when an operation needs the host copy of
///        a dual host/device buffer and the handle has none. The C++ transfer
///        path only checkPtr-asserts this, which exits the process.
inline void requireHostCopy(const void* ptr, const char* fnName)
{
    if (ptr != nullptr) return;
    std::string msg(fnName);
    msg += ": the handle has no host copy (data() is null). Construct it from "
           "host data or call deviceDownload() on a handle that owns its "
           "buffer before using the host side.";
    throw nanobind::value_error(msg.c_str());
}

/// @brief Raise a Python ValueError when an operation needs the device copy
///        of a dual host/device buffer and the handle has none. Handles built
///        by the host-side tools (tools.cuda.createLevelSetSphere and
///        friends) start out host-only and need deviceUpload() first.
inline void requireDeviceCopy(const void* ptr, const char* fnName)
{
    if (ptr != nullptr) return;
    std::string msg(fnName);
    msg += ": the handle has no device copy (device_ptr() is 0). Call "
           "deviceUpload() first.";
    throw nanobind::value_error(msg.c_str());
}

/// @brief Raise a Python ValueError when a buffer or handle is empty and the
///        operation would hand a null pointer to the CUDA runtime, which the
///        C++ side only cudaCheck-asserts (and exits the process).
inline void requireNonEmptyBuffer(uint64_t size, const char* fnName)
{
    if (size != 0) return;
    std::string msg(fnName);
    msg += ": the buffer is empty (size() == 0); there is nothing to transfer.";
    throw nanobind::value_error(msg.c_str());
}

} // namespace pynanovdb

#endif
