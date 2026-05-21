// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyGridValidator.h"

#include <nanobind/operators.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/GridValidator.h>
#ifdef NANOVDB_USE_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#endif

#include <cstring>
#include <iostream>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;

namespace pynanovdb {

template<typename BufferT> void defineValidateGrids(nb::module_& m)
{
    m.def("validateGrids", &tools::validateGrids<GridHandle<BufferT>>,
        "handle"_a, "mode"_a, "verbose"_a);
}

template void defineValidateGrids<HostBuffer>(nb::module_&);
#ifdef NANOVDB_USE_CUDA
template void defineValidateGrids<cuda::DeviceBuffer>(nb::module_&);
#endif

namespace {

// callNanoGrid op for checkGrid: writes into a fixed-size buffer and returns
// it as a std::pair<bool, std::string> for nb::make_tuple consumption.
struct CheckGridOp
{
    template<typename BuildT>
    static std::pair<bool, std::string> known(const GridData* gridData,
                                              CheckMode        mode)
    {
        char buf[256];
        tools::checkGrid<BuildT>(
            static_cast<const NanoGrid<BuildT>*>(gridData), buf, mode);
        const bool ok = (buf[0] == '\0');
        return {ok, std::string(buf)};
    }
    static std::pair<bool, std::string> unknown(const GridData* gridData,
                                                CheckMode /*mode*/)
    {
        (void)gridData;
        return {false, "Unsupported GridType for checkGrid"};
    }
};

// callNanoGrid op for isValid — wraps tools::isValid<BuildT> for every
// switched-over BuildT.
struct IsValidOp
{
    template<typename BuildT>
    static bool known(const GridData* gridData, CheckMode mode, bool verbose)
    {
        return tools::isValid<BuildT>(
            static_cast<const NanoGrid<BuildT>*>(gridData), mode, verbose);
    }
    static bool unknown(const GridData* gridData, CheckMode /*mode*/, bool verbose)
    {
        if (verbose && gridData != nullptr) {
            char str[16];
            std::cerr << "Validation failed: Unsupported GridType: \""
                      << toStr(str, gridData->mGridType) << "\""
                      << std::endl;
        }
        return false;
    }
};

} // namespace

void defineGridValidatorModule(nb::module_& toolsModule)
{
    // Single-grid validate. Takes a handle, a grid index, and the
    // usual mode + verbose flags. Returns true iff the grid passes all
    // tests for the given mode. Bound for both host and device handles
    // (when CUDA is enabled), matching validateGrids' coverage —
    // tools::validateGrid does host-side dispatch via callNanoGrid on
    // the host-resident gridData() pointer that DeviceGridHandle also
    // exposes, so the same overload pair is appropriate.
    toolsModule.def("validateGrid",
        &tools::validateGrid<GridHandle<HostBuffer>>,
        "handle"_a, "gridID"_a,
        "mode"_a = CheckMode::Default, "verbose"_a = false,
        nb::call_guard<nb::gil_scoped_release>(),
        "Validate the gridID'th grid in the handle against the given "
        "CheckMode. Returns False (without raising) if gridID is out "
        "of range or the grid fails any check. CheckMode.Disable is a "
        "short-circuit that always returns True without inspecting "
        "the grid (even when gridID is out of range), matching the "
        "C++ behavior. Complements validateGrids() which checks the "
        "whole handle.");
#ifdef NANOVDB_USE_CUDA
    toolsModule.def("validateGrid",
        &tools::validateGrid<GridHandle<cuda::DeviceBuffer>>,
        "handle"_a, "gridID"_a,
        "mode"_a = CheckMode::Default, "verbose"_a = false,
        nb::call_guard<nb::gil_scoped_release>(),
        "Validate the gridID'th grid in the device handle (uses the "
        "host-resident copy of the grid metadata for the actual "
        "checks). Same semantics as the host-handle overload.");
#endif

    // Polymorphic checkGrid — returns (ok, error_message). Mirrors the C++
    // char-buffer-out signature, but the buffer is hidden inside the
    // binding so Python callers get a Python str.
    toolsModule.def("checkGrid",
        [](const GridData* gridData, CheckMode mode)
            -> std::pair<bool, std::string> {
            if (gridData == nullptr) {
                return {false, "Grid is None"};
            }
            return callNanoGrid<CheckGridOp>(gridData, mode);
        },
        "grid"_a, "mode"_a = CheckMode::Full,
        nb::call_guard<nb::gil_scoped_release>(),
        "Run structural validation checks on the grid for the given "
        "CheckMode. Returns a (ok, error_message) tuple — error_message "
        "is empty when ok is True.");

    // Polymorphic isValid — convenience wrapper. Same as checkGrid + a
    // checksum check, returning just the bool.
    toolsModule.def("isValid",
        [](const GridData* gridData, CheckMode mode, bool verbose) {
            if (gridData == nullptr) return false;
            return callNanoGrid<IsValidOp>(gridData, mode, verbose);
        },
        "grid"_a, "mode"_a = CheckMode::Default, "verbose"_a = false,
        nb::call_guard<nb::gil_scoped_release>(),
        "Return True iff the grid passes structural validation AND its "
        "stored checksum matches a freshly computed one for the given "
        "CheckMode.");
}

} // namespace pynanovdb
