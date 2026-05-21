// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include "PyVoxelBlockManager.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/VoxelBlockManager.h>

#include <cstring>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;
using nanovdb::tools::VoxelBlockManager;
using nanovdb::tools::VoxelBlockManagerBase;
using nanovdb::tools::VoxelBlockManagerHandle;
using nanovdb::tools::buildVoxelBlockManager;

namespace pynanovdb {

using VBMHandle = VoxelBlockManagerHandle<HostBuffer>;

// ----------------------- Log2BlockWidth dispatch --------------------------
//
// Log2BlockWidth is a compile-time template parameter on every VBM helper.
// We expose it to Python as a runtime int and dispatch via a switch that
// instantiates the four useful widths (BlockWidth = 64, 128, 256, 512).
// Larger widths are not bound by default; callers who need them can add a
// new case below.

template<typename F>
static auto dispatchLog2BlockWidth(int log2BlockWidth, F&& fn)
{
    switch (log2BlockWidth) {
        case 6: return fn(std::integral_constant<int, 6>{});
        case 7: return fn(std::integral_constant<int, 7>{});
        case 8: return fn(std::integral_constant<int, 8>{});
        case 9: return fn(std::integral_constant<int, 9>{});
        default:
            throw nb::value_error(
                "VoxelBlockManager: log2_block_width must be 6, 7, 8, or 9 "
                "(BlockWidth = 64, 128, 256, or 512). Larger widths are not "
                "bound in Python by default.");
    }
}

// ----------------------- decodeInverseMaps helper -------------------------
//
// Common implementation used by both the free function and the
// handle.decodeBlock(i) method. Allocates fresh leafIndex (uint32) and
// voxelOffset (uint16) NumPy arrays of length BlockWidth and fills them.
template<int Log2BlockWidth>
static nb::object pyDecodeInverseMapsImpl(const NanoGrid<ValueOnIndex>& grid,
                                          uint32_t firstLeafID,
                                          const uint64_t* jumpMap,
                                          uint64_t blockFirstOffset)
{
    constexpr int BlockWidth = 1 << Log2BlockWidth;

    // Allocate outputs as numpy-owned arrays. nb::ndarray with no parent
    // and the nb::numpy framework tag tells nanobind to allocate fresh
    // memory through numpy (so Python owns the result).
    auto* leafIndex = new uint32_t[BlockWidth];
    auto* voxelOffset = new uint16_t[BlockWidth];

    using VBM = VoxelBlockManager<Log2BlockWidth>;
    VBM::template decodeInverseMaps<ValueOnIndex>(
        &grid, firstLeafID, jumpMap, blockFirstOffset,
        leafIndex, voxelOffset);

    // Hand ownership to nanobind via the delete-on-destruction owner
    // capsule pattern. nb::capsule wraps a raw pointer + deleter and lives
    // for as long as the ndarray that takes it as parent.
    nb::capsule leafOwner(leafIndex,
        [](void* p) noexcept { delete[] static_cast<uint32_t*>(p); });
    nb::capsule offsetOwner(voxelOffset,
        [](void* p) noexcept { delete[] static_cast<uint16_t*>(p); });

    size_t shape[1] = {static_cast<size_t>(BlockWidth)};
    nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
        leafArr(leafIndex, size_t(1), shape, leafOwner);
    nb::ndarray<nb::numpy, uint16_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
        offsetArr(voxelOffset, size_t(1), shape, offsetOwner);
    return nb::make_tuple(
        nb::cast(leafArr,   nb::rv_policy::reference),
        nb::cast(offsetArr, nb::rv_policy::reference));
}

// ------------------- VoxelBlockManagerHandle binding ----------------------

static const NanoGrid<ValueOnIndex>* castOnIndexGrid(nb::handle py_grid,
                                                    const char* fn_name)
{
    if (!nb::isinstance<NanoGrid<ValueOnIndex>>(py_grid)) {
        std::string msg(fn_name);
        msg += ": grid must be a NanoVDB grid of build type ValueOnIndex (OnIndexGrid)";
        throw nb::type_error(msg.c_str());
    }
    return &nb::cast<const NanoGrid<ValueOnIndex>&>(py_grid);
}

static void defineHandle(nb::module_& toolsModule)
{
    nb::class_<VBMHandle>(toolsModule, "VoxelBlockManagerHandle",
        "Owns the firstLeafID / jumpMap metadata buffers backing a "
        "VoxelBlockManager. Constructed by nanovdb.tools.buildVoxelBlockManager.")
        .def(nb::init<>())
        .def("blockCount",  &VBMHandle::blockCount)
        .def("firstOffset", &VBMHandle::firstOffset)
        .def("lastOffset",  &VBMHandle::lastOffset)
        .def("reset",       &VBMHandle::reset)
        .def(
            "__bool__",
            [](const VBMHandle& h) { return h.blockCount() > 0; },
            nb::is_operator())
        // Zero-copy view of the (blockCount,) firstLeafID array.
        .def("firstLeafID",
            [](nb::handle py_self) -> nb::object {
                auto& h = nb::cast<VBMHandle&>(py_self);
                size_t shape[1] = {static_cast<size_t>(h.blockCount())};
                return nb::cast(
                    nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>,
                                nb::c_contig, nb::device::cpu>(
                        static_cast<void*>(h.hostFirstLeafID()),
                        size_t(1), shape, py_self),
                    nb::rv_policy::reference);
            },
            nb::keep_alive<0, 1>(),
            "Return a zero-copy (blockCount,) uint32 NumPy view of the "
            "firstLeafID array. The view keeps this handle alive.")
        // jumpMap is uint64_t[blockCount * JumpMapLength]. JumpMapLength
        // depends on log2_block_width (1 << (log2_block_width - 6)) and
        // isn't carried on the handle, so we take it as a method argument
        // and return the buffer reshaped to (blockCount, JumpMapLength).
        // Default is 1 (matching log2_block_width=6, the most common case).
        .def("jumpMap",
            [](nb::handle py_self, int jump_map_length) -> nb::object {
                if (jump_map_length < 1) {
                    throw nb::value_error(
                        "VoxelBlockManagerHandle.jumpMap: "
                        "jump_map_length must be >= 1.");
                }
                auto& h = nb::cast<VBMHandle&>(py_self);
                size_t shape[2] = {static_cast<size_t>(h.blockCount()),
                                   static_cast<size_t>(jump_map_length)};
                return nb::cast(
                    nb::ndarray<nb::numpy, uint64_t, nb::ndim<2>,
                                nb::c_contig, nb::device::cpu>(
                        static_cast<void*>(h.hostJumpMap()),
                        size_t(2), shape, py_self),
                    nb::rv_policy::reference);
            },
            nb::arg("jump_map_length") = 1,
            nb::keep_alive<0, 1>(),
            "Return a zero-copy (blockCount, jump_map_length) uint64 NumPy "
            "view of the jumpMap. jump_map_length = BlockWidth / 64 = "
            "1 << (log2_block_width - 6) — pass 1 (default) for "
            "log2_block_width=6, 2 for 7, 4 for 8, 8 for 9. The view keeps "
            "this handle alive.")
        // Decode the inverse maps for a single block of this VBM. The
        // log2_block_width passed must match the one the handle was built
        // with (the handle does not store it).
        .def("decodeBlock",
            [](VBMHandle& self,
               nb::handle py_grid,
               uint64_t block_index,
               int log2_block_width) -> nb::object {
                const auto* grid = castOnIndexGrid(py_grid,
                    "VoxelBlockManagerHandle.decodeBlock");
                if (block_index >= self.blockCount()) {
                    throw nb::index_error(
                        "VoxelBlockManagerHandle.decodeBlock(block_index): "
                        "block_index out of range [0, blockCount).");
                }
                // Defensive: NanoVDB's buildVoxelBlockManager doesn't always
                // initialize firstLeafID for blocks where no leaf starts at
                // a block boundary AND no leaf's iteration sweep reaches
                // them (e.g. when the source grid is tile-compressed, so
                // some sequential offsets correspond to tile values rather
                // than leaf voxels). The slot is then uninitialized memory;
                // passing it into decodeInverseMaps would lead to an OOB
                // read of tree.getFirstNode<0>()[garbage]. Catch the case
                // and raise rather than segfault.
                const uint32_t firstLeafID =
                    self.hostFirstLeafID()[block_index];
                const uint32_t nLeaves = grid->tree().nodeCount(0);
                if (firstLeafID >= nLeaves) {
                    throw nb::value_error(
                        "VoxelBlockManagerHandle.decodeBlock: the VBM's "
                        "firstLeafID for this block was not initialized by "
                        "buildVoxelBlockManager (the underlying algorithm "
                        "doesn't cover blocks that no leaf reaches via its "
                        "iteration). This typically happens on OnIndex "
                        "grids built from tile-compressed source grids; "
                        "until the issue is fixed upstream the workaround "
                        "is to build the source grid voxel-by-voxel with "
                        "build::Grid so it stays uncompressed.");
                }
                return dispatchLog2BlockWidth(log2_block_width, [&](auto W) {
                    constexpr int LBW = decltype(W)::value;
                    constexpr int BlockWidth = 1 << LBW;
                    constexpr int JumpMapLength =
                        VoxelBlockManagerBase<LBW>::JumpMapLength;
                    const uint64_t blockFirstOffset =
                        self.firstOffset() + block_index * BlockWidth;
                    return pyDecodeInverseMapsImpl<LBW>(
                        *grid,
                        firstLeafID,
                        self.hostJumpMap() + block_index * JumpMapLength,
                        blockFirstOffset);
                });
            },
            "grid"_a, "block_index"_a, "log2_block_width"_a = 6,
            "Decode the inverse maps for the block_index-th block of this "
            "VBM. Returns (leaf_index, voxel_offset) uint32 / uint16 NumPy "
            "arrays of length BlockWidth = 1<<log2_block_width. The "
            "log2_block_width must match the value the handle was built "
            "with.");
}

// ------------------- buildVoxelBlockManager binding -----------------------

static void defineBuild(nb::module_& toolsModule)
{
    toolsModule.def("buildVoxelBlockManager",
        [](nb::handle py_grid,
           int log2_block_width,
           uint64_t first_offset,
           uint64_t last_offset,
           uint64_t n_blocks) -> VBMHandle {
            const auto* grid = castOnIndexGrid(py_grid, "buildVoxelBlockManager");
            return dispatchLog2BlockWidth(log2_block_width, [&](auto W) {
                constexpr int LBW = decltype(W)::value;
                return buildVoxelBlockManager<LBW, HostBuffer>(
                    grid, first_offset, last_offset, n_blocks);
            });
        },
        "grid"_a,
        "log2_block_width"_a = 6,
        "first_offset"_a = 0,
        "last_offset"_a = 0,
        "n_blocks"_a = 0,
        "Build a host-side VoxelBlockManager from an OnIndexGrid. "
        "log2_block_width selects the per-block active-voxel count "
        "(6=64, 7=128, 8=256, 9=512). Pass 0 for first_offset / "
        "last_offset / n_blocks to use the full grid (first active "
        "voxel through grid.activeVoxelCount(), minimum block count).");
}

// --------------------- decodeInverseMaps binding --------------------------

static void defineDecode(nb::module_& toolsModule)
{
    toolsModule.def("decodeInverseMaps",
        [](nb::handle py_grid,
           uint32_t first_leaf_id,
           nb::ndarray<const uint64_t, nb::ndim<1>,
                       nb::c_contig, nb::device::cpu> jump_map,
           uint64_t block_first_offset,
           int log2_block_width) -> nb::object {
            const auto* grid = castOnIndexGrid(py_grid, "decodeInverseMaps");
            return dispatchLog2BlockWidth(log2_block_width, [&](auto W) {
                constexpr int LBW = decltype(W)::value;
                constexpr int JumpMapLength =
                    VoxelBlockManagerBase<LBW>::JumpMapLength;
                if (jump_map.shape(0) != JumpMapLength) {
                    std::string msg("decodeInverseMaps: jump_map must have "
                                    "length JumpMapLength = ");
                    msg += std::to_string(JumpMapLength);
                    msg += " for log2_block_width=";
                    msg += std::to_string(LBW);
                    throw nb::value_error(msg.c_str());
                }
                return pyDecodeInverseMapsImpl<LBW>(
                    *grid, first_leaf_id, jump_map.data(),
                    block_first_offset);
            });
        },
        "grid"_a,
        "first_leaf_id"_a,
        "jump_map"_a,
        "block_first_offset"_a,
        "log2_block_width"_a = 6,
        "Decode the inverse maps for a single voxel block of an OnIndexGrid. "
        "Returns a (leaf_index, voxel_offset) tuple of fresh NumPy arrays of "
        "length BlockWidth = 1<<log2_block_width. jump_map must have length "
        "BlockWidth/64.");
}

// ----- createOnIndexGrid test-scaffold factory (subset of Phase 5) --------

template<typename SrcBuildT>
static nb::object tryCreateOnIndexGrid(nb::handle py_grid,
                                       uint32_t channels,
                                       bool include_stats,
                                       bool include_tiles,
                                       int verbose)
{
    using SrcGridT = NanoGrid<SrcBuildT>;
    if (!nb::isinstance<SrcGridT>(py_grid)) {
        return nb::object();
    }
    const SrcGridT& src = nb::cast<const SrcGridT&>(py_grid);
    return nb::cast(
        tools::createNanoGrid<SrcGridT, ValueOnIndex, HostBuffer>(
            src, channels, include_stats, include_tiles, verbose));
}

static void defineCreateOnIndexGrid(nb::module_& toolsModule)
{
    toolsModule.def("createOnIndexGrid",
        [](nb::handle py_grid,
           uint32_t channels,
           bool include_stats,
           bool include_tiles,
           int verbose) -> nb::object {
            // Try every source BuildT we accept.
            if (auto r = tryCreateOnIndexGrid<float>(
                    py_grid, channels, include_stats, include_tiles, verbose);
                r.is_valid()) return r;
            if (auto r = tryCreateOnIndexGrid<double>(
                    py_grid, channels, include_stats, include_tiles, verbose);
                r.is_valid()) return r;
            if (auto r = tryCreateOnIndexGrid<int32_t>(
                    py_grid, channels, include_stats, include_tiles, verbose);
                r.is_valid()) return r;
            if (auto r = tryCreateOnIndexGrid<Vec3f>(
                    py_grid, channels, include_stats, include_tiles, verbose);
                r.is_valid()) return r;
            throw nb::type_error(
                "createOnIndexGrid: source grid must be a FloatGrid, "
                "DoubleGrid, Int32Grid, or Vec3fGrid (other source BuildTs "
                "are not yet bound).");
        },
        "src_grid"_a,
        "channels"_a = 0u,
        "include_stats"_a = true,
        "include_tiles"_a = true,
        "verbose"_a = 0,
        "Convert a source grid into a NanoGrid<ValueOnIndex> "
        "(OnIndexGrid). Accepts FloatGrid / DoubleGrid / Int32Grid / "
        "Vec3fGrid. Required for constructing inputs to "
        "buildVoxelBlockManager. The broader createNanoGrid<SrcGridT, "
        "DstBuildT> surface lands in a later phase.");
}

void defineVoxelBlockManagerModule(nb::module_& toolsModule)
{
    defineHandle(toolsModule);
    defineBuild(toolsModule);
    defineDecode(toolsModule);
    defineCreateOnIndexGrid(toolsModule);
}

} // namespace pynanovdb
