// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include "PyInjectData.h"

#include <nanobind/ndarray.h>

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/cuda/Util.h>                // cudaCheck, operatorKernel
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>  // leaf count of a device grid
#include <nanovdb/util/cuda/Injection.cuh>         // Inject*Functor

namespace nb = nanobind;
using namespace nb::literals;
// Deliberately NOT `using namespace nanovdb;`: keep the device-grid type names
// fully qualified, matching the sibling tools.cuda bindings.

namespace pynanovdb {

namespace {

// Cast a Python device-grid object to NanoGrid<ValueOnIndex>*. The returned
// object's underlying address IS the device pointer; it must NOT be
// dereferenced on the host -- it is only passed to device kernels.
nanovdb::NanoGrid<nanovdb::ValueOnIndex>*
castOnIndexDeviceGrid(nb::handle py_grid, const char* fn_name)
{
    if (!nb::isinstance<nanovdb::NanoGrid<nanovdb::ValueOnIndex>>(py_grid)) {
        std::string msg(fn_name);
        msg += ": expected a NanoVDB device grid of build type ValueOnIndex "
               "(OnIndexGrid), obtained from DeviceGridHandle.deviceGrid(n)";
        throw nb::type_error(msg.c_str());
    }
    return &nb::cast<nanovdb::NanoGrid<nanovdb::ValueOnIndex>&>(py_grid);
}

// Leaf-node count read from device memory (one D2H copy of the tree header).
uint32_t leafCountOf(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* d_grid)
{
    using Traits = nanovdb::util::cuda::DeviceGridTraits<nanovdb::ValueOnIndex>;
    return Traits::getTreeData(d_grid).mNodeCount[0];
}

} // anonymous namespace

template<typename T> void defineInject(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::handle src_grid, nb::handle dst_grid,
           nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cuda> src_sidecar,
           nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cuda>       dst_sidecar,
           uintptr_t stream) {
            auto* src = castOnIndexDeviceGrid(src_grid, "inject");
            auto* dst = castOnIndexDeviceGrid(dst_grid, "inject");
            cudaStream_t   s    = reinterpret_cast<cudaStream_t>(stream);
            const T*       dSrc = src_sidecar.data();
            T*             dDst = dst_sidecar.data();
            const uint32_t srcLeafCount = leafCountOf(src);
            using Op = nanovdb::util::cuda::InjectGridDataFunctor<nanovdb::ValueOnIndex, T>;
            // operatorKernel launches one block per SOURCE leaf and copies the
            // src/dst intersection bit-parallel per warp; pure CUDA, no Python.
            nb::gil_scoped_release release;
            if (srcLeafCount)
                nanovdb::util::cuda::operatorKernel<Op>
                    <<<srcLeafCount, Op::MaxThreadsPerBlock, 0, s>>>(src, dst, dSrc, dDst);
            cudaCheck(cudaStreamSynchronize(s));
        },
        "src_grid"_a, "dst_grid"_a, "src_sidecar"_a, "dst_sidecar"_a, "stream"_a = 0,
        "Inject sidecar values from a source OnIndex device grid onto a "
        "destination OnIndex device grid (injectData; NanoVDB 2.0 paper, "
        "section 3.4). For every voxel present in BOTH grids the source sidecar "
        "value is copied to that voxel's slot in the destination sidecar; "
        "destination voxels with no source counterpart are left unchanged "
        "(so the source need not be a subset -- the copy is over the "
        "intersection). src_grid / dst_grid are device grids from "
        "DeviceGridHandle.deviceGrid(n); src_sidecar / dst_sidecar are 1-D "
        "device arrays indexed by each grid's value index (entry 0 is the "
        "background slot, untouched). Wraps "
        "nanovdb::util::cuda::InjectGridDataFunctor. stream is a raw CUDA "
        "stream handle (Python int; 0 = default stream).");
}

void defineInjectPredicateToMask(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::handle grid,
           nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cuda>  predicate,
           nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cuda>    leaf_masks,
           uintptr_t stream) {
            auto* d_grid = castOnIndexDeviceGrid(grid, "injectPredicateToMask");
            cudaStream_t   s         = reinterpret_cast<cudaStream_t>(stream);
            const uint32_t leafCount = leafCountOf(d_grid);
            constexpr size_t W = nanovdb::Mask<3>::WORD_COUNT;  // 8 uint64 / leaf
            if (leaf_masks.size() < static_cast<size_t>(leafCount) * W)
                throw nb::value_error(
                    "injectPredicateToMask: leaf_masks length must be at least "
                    "(leaf count) * 8 uint64 (one Mask<3> per leaf). A safe "
                    "upper bound is activeVoxelCount * 8, since every leaf "
                    "holds at least one active voxel.");
            const bool*       dPred = predicate.data();
            nanovdb::Mask<3>* dMask =
                reinterpret_cast<nanovdb::Mask<3>*>(leaf_masks.data());
            using Op = nanovdb::util::cuda::InjectPredicateToMaskFunctor<nanovdb::ValueOnIndex>;
            // One block per leaf; the functor zeroes each leaf mask, then sets
            // the bit of every active voxel whose predicate slot is true.
            nb::gil_scoped_release release;
            if (leafCount)
                nanovdb::util::cuda::operatorKernel<Op>
                    <<<leafCount, Op::MaxThreadsPerBlock, 0, s>>>(d_grid, dPred, dMask);
            cudaCheck(cudaStreamSynchronize(s));
        },
        "grid"_a, "predicate"_a, "leaf_masks"_a, "stream"_a = 0,
        "Build a per-leaf retain mask for pruneGrid from a boolean predicate "
        "over an OnIndex device grid's value indices. grid is a device grid "
        "from DeviceGridHandle.deviceGrid(n); predicate is a 1-D device bool "
        "array indexed by value index (entry n true => keep that voxel); "
        "leaf_masks is a 1-D device uint64 output of length at least "
        "(leaf count) * 8 (one nanovdb::Mask<3> per leaf, in leaf order), "
        "ready to pass straight to pruneGrid; activeVoxelCount * 8 is a safe "
        "size since every leaf holds at least one active voxel. Wraps "
        "nanovdb::util::cuda::InjectPredicateToMaskFunctor. stream is a raw "
        "CUDA stream handle (Python int; 0 = default stream).");
}

template void defineInject<float>(nb::module_&, const char*);
template void defineInject<double>(nb::module_&, const char*);

} // namespace pynanovdb

#endif
