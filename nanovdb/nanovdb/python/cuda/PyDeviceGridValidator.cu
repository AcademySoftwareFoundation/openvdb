// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyDeviceGridValidator.h"

#include <cstdint>

// Pull in cuda/GridHandle.cuh first: it includes GridChecksum.cuh (which
// GridValidator.cuh also pulls in) and then uses
// nanovdb::tools::cuda::updateChecksum, so it must be parsed before any
// translation unit sets GridChecksum.cuh's include guard. Matches the
// unittest's include ordering.
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/cuda/GridValidator.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

template<typename BuildT>
void defineDeviceIsValid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](const nanovdb::NanoGrid<BuildT>* d_grid, nanovdb::CheckMode mode,
           bool verbose, uintptr_t stream) -> bool {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // isValid runs structural checks in a device kernel plus a device
            // checksum validation; pure CUDA touching no Python objects (the
            // optional verbose diagnostic goes to std::cerr), so release the
            // GIL.
            nb::gil_scoped_release release;
            return nanovdb::tools::cuda::isValid<BuildT>(d_grid, mode, verbose, s);
        },
        "d_grid"_a,
        "mode"_a = nanovdb::CheckMode::Default,
        "verbose"_a = false,
        "stream"_a = 0,
        "Return True iff the device grid passes structural validation for the "
        "given CheckMode AND its stored checksum matches a freshly computed "
        "one. If verbose, the first failure is printed to stderr. stream is a "
        "raw CUDA stream handle (Python int; 0 = default stream).");
}

// Instantiate for the same BuildT set the host isValid covers via
// callNanoGrid (NanoVDB.h). tools::checkGrid compiles for all of these.
template void defineDeviceIsValid<float>(nb::module_&, const char*);
template void defineDeviceIsValid<double>(nb::module_&, const char*);
template void defineDeviceIsValid<int16_t>(nb::module_&, const char*);
template void defineDeviceIsValid<int32_t>(nb::module_&, const char*);
template void defineDeviceIsValid<int64_t>(nb::module_&, const char*);
template void defineDeviceIsValid<uint8_t>(nb::module_&, const char*);
template void defineDeviceIsValid<uint32_t>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::Vec3f>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::Vec3d>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::Vec4f>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::Vec4d>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::math::Rgba8>(nb::module_&, const char*);
template void defineDeviceIsValid<bool>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::Fp4>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::Fp8>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::Fp16>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::FpN>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::ValueIndex>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::ValueOnIndex>(nb::module_&, const char*);
template void defineDeviceIsValid<nanovdb::ValueMask>(nb::module_&, const char*);

} // namespace pynanovdb
