// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyDeviceGridStats.h"

#include <cstdint>

// GridStats.cuh uses NodeManager<BuildT> and (via GridHandle) the device
// checksum path but is not self-contained for either. Pull in
// cuda/GridHandle.cuh first: it includes cuda/NodeManager.cuh (for
// NodeManager) and GridChecksum.cuh, and must be parsed before any TU sets
// GridChecksum.cuh's include guard so tools::cuda::updateChecksum is declared
// when GridHandle.cuh is parsed. Matches the unittest's include ordering.
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/tools/cuda/GridStats.cuh>

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

template<typename BuildT>
void defineDeviceUpdateGridStats(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nanovdb::NanoGrid<BuildT>* d_grid, nanovdb::tools::StatsMode mode, uintptr_t stream) {
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            // updateGridStats launches kernels and synchronizes the stream;
            // pure CUDA touching no Python objects, so release the GIL. The
            // operation mutates the device grid in place.
            nb::gil_scoped_release release;
            nanovdb::tools::cuda::updateGridStats<BuildT>(d_grid, mode, s);
        },
        "d_grid"_a,
        "mode"_a = nanovdb::tools::StatsMode::Default,
        "stream"_a = 0,
        "Recompute and write per-node statistics into the given device grid "
        "in place (returns None). Does NOT recompute the grid checksum — call "
        "updateChecksum afterward if the checksum must stay valid. stream is a "
        "raw CUDA stream handle (Python int; 0 = default stream).");
}

// Scalar + vector + bool BuildTs. updateGridStats's MinMax / All branches
// instantiate Extrema<BuildT> / Stats<BuildT>, which are only meaningful for
// arithmetic value types, so the quantized / index / mask special BuildTs are
// intentionally NOT instantiated here (they would fail GridStats's ValueT
// static_assert / lack a usable Stats specialization). bool routes to the
// NoopStats path internally regardless of mode.
template void defineDeviceUpdateGridStats<float>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<double>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<int16_t>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<int32_t>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<int64_t>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<uint8_t>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<uint32_t>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<nanovdb::Vec3f>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<nanovdb::Vec3d>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<nanovdb::Vec4f>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<nanovdb::Vec4d>(nb::module_&, const char*);
template void defineDeviceUpdateGridStats<bool>(nb::module_&, const char*);

} // namespace pynanovdb
