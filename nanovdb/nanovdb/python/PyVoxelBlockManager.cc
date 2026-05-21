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

// PyVBMHandle wraps the C++ VoxelBlockManagerHandle and carries the
// log2_block_width the handle was built with. The C++ handle does NOT store
// log2_block_width itself, so without this wrapper the Python binding would
// have to ask the caller every time — which the user can lie about and
// trigger out-of-bounds reads of the metadata buffers. Storing it once at
// build time and consulting it in every accessor closes that hole.
struct PyVBMHandle
{
    VoxelBlockManagerHandle<HostBuffer> handle;
    int                                 log2BlockWidth = 6;

    PyVBMHandle() = default;
    PyVBMHandle(VoxelBlockManagerHandle<HostBuffer>&& h, int lbw) noexcept
        : handle(std::move(h)), log2BlockWidth(lbw) {}

    PyVBMHandle(const PyVBMHandle&)            = delete;
    PyVBMHandle& operator=(const PyVBMHandle&) = delete;
    PyVBMHandle(PyVBMHandle&&)                 = default;
    PyVBMHandle& operator=(PyVBMHandle&&)      = default;

    uint64_t blockCount()  const { return handle.blockCount(); }
    uint64_t firstOffset() const { return handle.firstOffset(); }
    uint64_t lastOffset()  const { return handle.lastOffset(); }
    void     reset()             { handle.reset(); }
    int      blockWidth()    const { return 1 << log2BlockWidth; }
    int      jumpMapLength() const { return 1 << (log2BlockWidth - 6); }
};

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

    // Each call allocates fresh BlockWidth-sized output arrays for the
    // leaf-index and voxel-offset results. We use plain new[] (rather than
    // a numpy-allocated buffer) because the produced ndarrays are returned
    // by reference and Python owns them via the capsule deleters below —
    // when the ndarray is destroyed, the capsule's deleter runs delete[].
    auto* leafIndex = new uint32_t[BlockWidth];
    auto* voxelOffset = new uint16_t[BlockWidth];

    using VBM = VoxelBlockManager<Log2BlockWidth>;
    VBM::template decodeInverseMaps<ValueOnIndex>(
        &grid, firstLeafID, jumpMap, blockFirstOffset,
        leafIndex, voxelOffset);

    // nb::capsule wraps the raw pointer + matching delete[] so it can serve
    // as the ndarray's owner — the capsule lives as long as the ndarray and
    // its destruction runs the deleter.
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
    nb::class_<PyVBMHandle>(toolsModule, "VoxelBlockManagerHandle",
        "Owns the firstLeafID / jumpMap metadata buffers backing a "
        "VoxelBlockManager. Constructed by nanovdb.tools.buildVoxelBlockManager.")
        .def(nb::init<>())
        .def("blockCount",  &PyVBMHandle::blockCount)
        .def("firstOffset", &PyVBMHandle::firstOffset)
        .def("lastOffset",  &PyVBMHandle::lastOffset)
        .def("reset",       &PyVBMHandle::reset)
        .def_prop_ro("log2_block_width", [](const PyVBMHandle& h) { return h.log2BlockWidth; },
            "The log2_block_width this handle was built with. The jumpMap "
            "and decodeBlock outputs derive their shapes from this value.")
        .def_prop_ro("block_width", &PyVBMHandle::blockWidth,
            "BlockWidth = 1 << log2_block_width (64, 128, 256, or 512).")
        .def_prop_ro("jump_map_length", &PyVBMHandle::jumpMapLength,
            "JumpMapLength = BlockWidth / 64 (1, 2, 4, or 8).")
        .def(
            "__bool__",
            [](const PyVBMHandle& h) { return h.blockCount() > 0; },
            nb::is_operator())
        // Zero-copy view of the (blockCount,) firstLeafID array.
        .def("firstLeafID",
            [](nb::handle py_self) -> nb::object {
                auto& h = nb::cast<PyVBMHandle&>(py_self);
                size_t shape[1] = {static_cast<size_t>(h.blockCount())};
                // A default-constructed or reset() handle has a null
                // hostFirstLeafID(); we still return an empty (0,) ndarray
                // so callers don't have to branch on a None sentinel. The
                // dummy non-null pointer (the handle itself) keeps nanobind
                // happy; nothing is read since the leading shape is 0.
                uint32_t* raw = h.handle.hostFirstLeafID();
                void* data = (raw != nullptr) ? static_cast<void*>(raw)
                                              : static_cast<void*>(&h);
                return nb::cast(
                    nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>,
                                nb::c_contig, nb::device::cpu>(
                        data, size_t(1), shape, py_self),
                    nb::rv_policy::reference);
            },
            nb::keep_alive<0, 1>(),
            "Return a zero-copy (blockCount,) uint32 NumPy view of the "
            "firstLeafID array. Returns an empty (0,) array on a "
            "default-constructed or reset() handle. The view keeps this "
            "handle alive.")
        // jumpMap is uint64_t[blockCount * JumpMapLength]. JumpMapLength is
        // determined by the log2_block_width recorded on the handle, not by
        // the caller — that way the returned view always covers exactly the
        // allocated buffer, with no risk of OOB reads.
        .def("jumpMap",
            [](nb::handle py_self) -> nb::object {
                auto& h = nb::cast<PyVBMHandle&>(py_self);
                size_t shape[2] = {static_cast<size_t>(h.blockCount()),
                                   static_cast<size_t>(h.jumpMapLength())};
                // Same null-buffer guard as firstLeafID(): a
                // default-constructed / reset() handle has a null
                // hostJumpMap(); return an empty (0, jump_map_length)
                // ndarray rather than passing nullptr to nanobind.
                uint64_t* raw = h.handle.hostJumpMap();
                void* data = (raw != nullptr) ? static_cast<void*>(raw)
                                              : static_cast<void*>(&h);
                return nb::cast(
                    nb::ndarray<nb::numpy, uint64_t, nb::ndim<2>,
                                nb::c_contig, nb::device::cpu>(
                        data, size_t(2), shape, py_self),
                    nb::rv_policy::reference);
            },
            nb::keep_alive<0, 1>(),
            "Return a zero-copy (blockCount, jump_map_length) uint64 NumPy "
            "view of the jumpMap. The shape is determined by the "
            "log2_block_width the handle was built with. Returns an empty "
            "(0, jump_map_length) array on a default-constructed or reset() "
            "handle. The view keeps this handle alive.")
        // Decode the inverse maps for a single block of this VBM. The
        // log2_block_width is taken from the handle, so the caller cannot
        // request a width that doesn't match what was built.
        .def("decodeBlock",
            [](PyVBMHandle& self,
               nb::handle py_grid,
               uint64_t block_index) -> nb::object {
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
                    self.handle.hostFirstLeafID()[block_index];
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
                return dispatchLog2BlockWidth(self.log2BlockWidth, [&](auto W) {
                    constexpr int LBW = decltype(W)::value;
                    constexpr int BlockWidth = 1 << LBW;
                    constexpr int JumpMapLength =
                        VoxelBlockManagerBase<LBW>::JumpMapLength;
                    const uint64_t blockFirstOffset =
                        self.firstOffset() + block_index * BlockWidth;
                    return pyDecodeInverseMapsImpl<LBW>(
                        *grid,
                        firstLeafID,
                        self.handle.hostJumpMap() + block_index * JumpMapLength,
                        blockFirstOffset);
                });
            },
            "grid"_a, "block_index"_a,
            "Decode the inverse maps for the block_index-th block of this "
            "VBM. Returns (leaf_index, voxel_offset) uint32 / uint16 NumPy "
            "arrays of length BlockWidth = 1<<log2_block_width, using the "
            "log2_block_width the handle was built with.");
}

// ------------------- buildVoxelBlockManager binding -----------------------

static void defineBuild(nb::module_& toolsModule)
{
    toolsModule.def("buildVoxelBlockManager",
        [](nb::handle py_grid,
           int log2_block_width,
           uint64_t first_offset,
           uint64_t last_offset,
           uint64_t n_blocks) -> PyVBMHandle {
            const auto* grid = castOnIndexGrid(py_grid, "buildVoxelBlockManager");
            // The C++ implementation only NANOVDB_ASSERTs these preconditions,
            // which makes them no-ops in release builds. Validate them here
            // so Python callers get a clear error instead of UB / abort.
            if (!grid->isSequential()) {
                throw nb::value_error(
                    "buildVoxelBlockManager: grid must satisfy "
                    "grid.isSequential() (fixed-size, breadth-first node "
                    "layout). NanoVDB grids constructed via "
                    "tools.createOnIndexGrid satisfy this by default.");
            }
            return dispatchLog2BlockWidth(log2_block_width, [&](auto W) {
                constexpr int LBW = decltype(W)::value;
                constexpr uint64_t BlockWidth = uint64_t(1) << LBW;
                // first_offset must be 1 (mod BlockWidth). The C++ helper
                // normalizes a zero input to 1, so we only need to validate
                // the nonzero case ourselves.
                if (first_offset != 0 &&
                    ((first_offset - 1) & (BlockWidth - 1)) != 0) {
                    throw nb::value_error(
                        "buildVoxelBlockManager: first_offset must satisfy "
                        "first_offset == 1 (mod BlockWidth). Pass 0 (the "
                        "default) to let the implementation use 1.");
                }
                return PyVBMHandle(
                    buildVoxelBlockManager<LBW, HostBuffer>(
                        grid, first_offset, last_offset, n_blocks),
                    LBW);
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
        "voxel through grid.activeVoxelCount(), minimum block count). "
        "first_offset, if nonzero, must satisfy first_offset == 1 "
        "(mod BlockWidth).");
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
            // The C++ helper indexes tree.getFirstNode<0>()[first_leaf_id]
            // without a bounds check, so a stray first_leaf_id leads to an
            // OOB read. Validate up front. (We also require isSequential();
            // getFirstNode only makes sense on a sequential tree.)
            if (!grid->isSequential()) {
                throw nb::value_error(
                    "decodeInverseMaps: grid must satisfy "
                    "grid.isSequential().");
            }
            const uint32_t nLeaves = grid->tree().nodeCount(0);
            if (first_leaf_id >= nLeaves) {
                throw nb::index_error(
                    "decodeInverseMaps: first_leaf_id out of range "
                    "[0, grid.tree().nodeCount(0)).");
            }
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
        "BlockWidth/64. first_leaf_id must be in [0, grid.tree().nodeCount(0)).");
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
