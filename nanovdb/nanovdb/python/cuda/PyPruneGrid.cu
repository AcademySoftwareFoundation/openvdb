// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyPruneGrid.h"

#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>  // leaf count of a device grid

namespace nb = nanobind;
using namespace nb::literals;
// NOTE: deliberately NOT `using namespace nanovdb;`. These tools instantiate
// CUB DeviceScan, whose nvcc-generated host stub references unqualified
// `cuda::std::...`; with `nanovdb::cuda` in scope that becomes ambiguous and
// fails to compile. Fully qualify nanovdb:: instead (matches PyPointsToGrid.cu).

namespace pynanovdb {

template<typename BuildT> void definePruneGrid(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nanovdb::NanoGrid<BuildT>*                             dGrid,
           nb::ndarray<uint64_t, nb::c_contig, nb::device::cuda>  leafMask,
           uintptr_t                                              stream) {
            // The sidecar is a device array of nanovdb::Mask<3> (one 512-bit /
            // 8^3 leaf mask per leaf node, voxels to RETAIN), passed as raw
            // uint64 words. Mask<3> is exactly WORD_COUNT 64-bit words, so the
            // total word count must be a whole multiple of that, and should
            // equal (leaf count) * Mask<3>::WORD_COUNT for a well-formed call.
            constexpr size_t        wordsPerMask = nanovdb::Mask<3>::WORD_COUNT;
            const size_t            totalWords = leafMask.size();
            if (totalWords % wordsPerMask != 0)
                throw std::invalid_argument(
                    "pruneGrid: leafMask uint64 word count must be a multiple of "
                    "Mask<3>::WORD_COUNT (8); supply one 512-bit mask per leaf node");
            // PruneGrid indexes one Mask<3> per SOURCE leaf, so the sidecar
            // must cover every leaf; a shorter multiple of 8 would pass the
            // divisibility check but read out of bounds on the device.
            using Traits = nanovdb::util::cuda::DeviceGridTraits<BuildT>;
            const uint64_t leafCount = Traits::getTreeData(dGrid).mNodeCount[0];
            if (totalWords < leafCount * wordsPerMask)
                throw nb::value_error(
                    ("pruneGrid: leafMask holds " +
                     std::to_string(totalWords / wordsPerMask) +
                     " masks but the grid has " + std::to_string(leafCount) +
                     " leaf nodes; supply one Mask<3> (8 uint64) per leaf.")
                        .c_str());
            cudaStream_t            s = reinterpret_cast<cudaStream_t>(stream);
            const nanovdb::Mask<3>* d_mask =
                reinterpret_cast<const nanovdb::Mask<3>*>(leafMask.data());
            // PruneGrid::getHandle launches kernels and synchronizes the
            // stream; pure CUDA touching no Python objects, so release the GIL.
            nb::gil_scoped_release release;
            nanovdb::tools::cuda::PruneGrid<BuildT> pruner(dGrid, d_mask, s);
            return pruner.getHandle();
        },
        "dGrid"_a,
        "leafMask"_a,
        "stream"_a = 0,
        "Morphologically prune a device OnIndex grid against a per-leaf retain "
        "mask and return a fresh device GridHandle of the pruned grid. leafMask "
        "is a device uint64 array holding one nanovdb::Mask<3> (8 x uint64 = "
        "512 bits) per leaf node, in leaf order, marking voxels to RETAIN; its "
        "length must be (leaf count) * 8. stream is a raw CUDA stream handle "
        "(Python int; 0 = default stream).");
}

template void definePruneGrid<nanovdb::ValueOnIndex>(nb::module_&, const char*);

} // namespace pynanovdb
