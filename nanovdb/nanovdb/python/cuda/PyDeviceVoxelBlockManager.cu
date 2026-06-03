// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#ifdef NANOVDB_USE_CUDA

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

namespace nb = nanobind;
using namespace nb::literals;
using namespace nanovdb;
using nanovdb::tools::VoxelBlockManagerBase;
using nanovdb::tools::VoxelBlockManagerHandle;

namespace pynanovdb {

// ----------------------- Log2BlockWidth dispatch --------------------------
//
// Mirrors the host dispatchLog2BlockWidth (PyVoxelBlockManager.cc): turn the
// runtime log2_block_width into one of the four compile-time widths the
// device builder is instantiated for (BlockWidth = 64, 128, 256, 512).
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

// PyDeviceVBMHandle wraps the device VoxelBlockManagerHandle and carries the
// log2_block_width it was built with (parallel to the host PyVBMHandle). The
// C++ handle does NOT store log2_block_width itself, so recording it once at
// build time keeps the jumpMap view shape derivable from the handle rather
// than from a caller who could spoof it.
struct PyDeviceVBMHandle
{
    VoxelBlockManagerHandle<nanovdb::cuda::DeviceBuffer> handle;
    int                                                  log2BlockWidth = 6;

    PyDeviceVBMHandle() = default;
    PyDeviceVBMHandle(VoxelBlockManagerHandle<nanovdb::cuda::DeviceBuffer>&& h,
                      int lbw) noexcept
        : handle(std::move(h)), log2BlockWidth(lbw) {}

    PyDeviceVBMHandle(const PyDeviceVBMHandle&)            = delete;
    PyDeviceVBMHandle& operator=(const PyDeviceVBMHandle&) = delete;
    PyDeviceVBMHandle(PyDeviceVBMHandle&&)                 = default;
    PyDeviceVBMHandle& operator=(PyDeviceVBMHandle&&)      = default;

    uint64_t blockCount()  const { return handle.blockCount(); }
    uint64_t firstOffset() const { return handle.firstOffset(); }
    uint64_t lastOffset()  const { return handle.lastOffset(); }
    void     reset()             { handle.reset(); }
    int      blockWidth()    const { return 1 << log2BlockWidth; }
    int      jumpMapLength() const { return 1 << (log2BlockWidth - 6); }
};

// ------------------- OnIndex device grid cast helper ----------------------

static NanoGrid<ValueOnIndex>* castOnIndexDeviceGrid(nb::handle py_grid,
                                                     const char* fn_name)
{
    if (!nb::isinstance<NanoGrid<ValueOnIndex>>(py_grid)) {
        std::string msg(fn_name);
        msg += ": device_grid must be a NanoVDB device grid of build type "
               "ValueOnIndex (OnIndexGrid), obtained from "
               "DeviceGridHandle.deviceGrid(n)";
        throw nb::type_error(msg.c_str());
    }
    // The device builder takes a non-const NanoGrid<ValueOnIndex>*; the
    // returned object's underlying address IS the device pointer. It must NOT
    // be dereferenced on the host — it is only passed to device kernels.
    return &nb::cast<NanoGrid<ValueOnIndex>&>(py_grid);
}

// ------------------- DeviceVoxelBlockManagerHandle binding -----------------

static void defineHandle(nb::module_& m)
{
    nb::class_<PyDeviceVBMHandle>(m, "DeviceVoxelBlockManagerHandle",
        "Owns the device-resident firstLeafID / jumpMap metadata buffers "
        "backing a device VoxelBlockManager. Constructed by "
        "nanovdb.tools.cuda.buildVoxelBlockManager. The firstLeafID / jumpMap "
        "buffers are exposed zero-copy to CuPy / PyTorch / Numba via the CUDA "
        "Array Interface and DLPack.")
        .def("blockCount",  &PyDeviceVBMHandle::blockCount,
             "Number of voxel blocks managed by this handle.")
        .def("firstOffset", &PyDeviceVBMHandle::firstOffset,
             "Sequential voxel index of the first active voxel covered "
             "by this handle (1 by default when the handle covers the "
             "full grid).")
        .def("lastOffset",  &PyDeviceVBMHandle::lastOffset,
             "Sequential voxel index of the last active voxel covered "
             "by this handle.")
        .def("reset",       &PyDeviceVBMHandle::reset,
             "Release this handle's device buffers and reset it to the empty state.")
        .def_prop_ro("log2_block_width",
             [](const PyDeviceVBMHandle& h) { return h.log2BlockWidth; },
            "The log2_block_width this handle was built with. The jumpMap "
            "view derives its shape from this value.")
        .def_prop_ro("block_width", &PyDeviceVBMHandle::blockWidth,
            "BlockWidth = 1 << log2_block_width (64, 128, 256, or 512).")
        .def_prop_ro("jump_map_length", &PyDeviceVBMHandle::jumpMapLength,
            "JumpMapLength = BlockWidth / 64 (1, 2, 4, or 8).")
        .def(
            "__bool__",
            [](const PyDeviceVBMHandle& h) { return h.blockCount() > 0; },
            nb::is_operator())
        // ------------------- raw device pointers -------------------
        .def(
            "first_leaf_id_ptr",
            [](PyDeviceVBMHandle& h) {
                return reinterpret_cast<uintptr_t>(h.handle.deviceFirstLeafID());
            },
            "Raw device pointer to the firstLeafID array (uint32 x blockCount) "
            "as a Python int (0 if the handle is empty).")
        .def(
            "jump_map_ptr",
            [](PyDeviceVBMHandle& h) {
                return reinterpret_cast<uintptr_t>(h.handle.deviceJumpMap());
            },
            "Raw device pointer to the jumpMap array "
            "(uint64 x blockCount x jump_map_length) as a Python int (0 if the "
            "handle is empty).")
        // ------------------- firstLeafID device view -------------------
        // Zero-copy DEVICE view of the (blockCount,) uint32 firstLeafID array,
        // mirroring the host firstLeafID() but using the Phase-B device
        // interop (nb::device::cuda ndarray) instead of a host numpy view.
        .def(
            "firstLeafID",
            [](nb::handle py_self) -> nb::object {
                auto& h = nb::cast<PyDeviceVBMHandle&>(py_self);
                size_t shape[1] = {static_cast<size_t>(h.blockCount())};
                // A default-constructed / reset() handle has a null
                // deviceFirstLeafID(); still return an empty (0,) array so
                // callers don't branch on a None sentinel. The dummy non-null
                // pointer (the handle itself) keeps nanobind happy; nothing is
                // read since the leading shape is 0.
                uint32_t* raw = h.handle.deviceFirstLeafID();
                void* data = (raw != nullptr) ? static_cast<void*>(raw)
                                              : static_cast<void*>(&h);
                nb::ndarray<nb::device::cuda, uint32_t, nb::ndim<1>> arr(
                    data, size_t(1), shape, py_self);
                return nb::cast(arr, nb::rv_policy::reference);
            },
            // No nb::keep_alive<0,1> here: the returned no-framework device
            // ndarray is exported as a DLPack capsule (not weak-referenceable),
            // so keep_alive would throw "could not create a weak reference".
            // Lifetime is already anchored by the ndarray's py_self owner arg.
            "Return a zero-copy (blockCount,) uint32 DEVICE array view of the "
            "firstLeafID array, consumable by CuPy / PyTorch / Numba. Returns "
            "an empty (0,) array on a default-constructed or reset() handle. "
            "The view keeps this handle alive.")
        // ------------------- jumpMap device view -------------------
        // Zero-copy DEVICE view of the (blockCount, jumpMapLength) uint64
        // jumpMap. jumpMapLength derives from the handle's recorded
        // log2_block_width, never the caller, so the view exactly covers the
        // allocated buffer.
        .def(
            "jumpMap",
            [](nb::handle py_self) -> nb::object {
                auto& h = nb::cast<PyDeviceVBMHandle&>(py_self);
                size_t shape[2] = {static_cast<size_t>(h.blockCount()),
                                   static_cast<size_t>(h.jumpMapLength())};
                uint64_t* raw = h.handle.deviceJumpMap();
                void* data = (raw != nullptr) ? static_cast<void*>(raw)
                                              : static_cast<void*>(&h);
                nb::ndarray<nb::device::cuda, uint64_t, nb::ndim<2>> arr(
                    data, size_t(2), shape, py_self);
                return nb::cast(arr, nb::rv_policy::reference);
            },
            // No keep_alive (see firstLeafID): the device-ndarray capsule is
            // not weak-referenceable; the py_self owner arg anchors lifetime.
            "Return a zero-copy (blockCount, jump_map_length) uint64 DEVICE "
            "array view of the jumpMap, consumable by CuPy / PyTorch / Numba. "
            "The shape is determined by the log2_block_width the handle was "
            "built with. Returns an empty (0, jump_map_length) array on a "
            "default-constructed or reset() handle. The view keeps this handle "
            "alive.")
        // ------------------- CUDA Array Interface (v3) -------------------
        // CAI describing the firstLeafID array as 1-D uint32. (CuPy / Numba
        // consume the __cuda_array_interface__ attribute directly; the jumpMap
        // is reachable as a 2-D device ndarray via jumpMap() above.)
        .def_prop_ro(
            "__cuda_array_interface__",
            [](PyDeviceVBMHandle& h) {
                nb::dict iface;
                iface["shape"] = nb::make_tuple(h.blockCount());
                iface["typestr"] = "<u4";
                iface["data"] = nb::make_tuple(
                    reinterpret_cast<uintptr_t>(h.handle.deviceFirstLeafID()),
                    false);
                iface["version"] = 3;
                iface["strides"] = nb::none();
                iface["stream"] = 1;
                return iface;
            },
            "CUDA Array Interface (v3) view of the firstLeafID array as 1-D "
            "uint32 — lets CuPy / Numba / PyTorch consume it zero-copy. The "
            "jumpMap is available as a 2-D device array via jumpMap().")
        .def(
            "__dlpack_device__",
            [](const PyDeviceVBMHandle&) {
                int device = 0;
                cudaGetDevice(&device);
                return nb::make_tuple(2, device);  // 2 == kDLCUDA
            },
            "DLPack device tuple (kDLCUDA, device_id) for the firstLeafID "
            "device buffer.")
        .def(
            "__dlpack__",
            [](nb::handle self, nb::handle /*stream*/) {
                auto& h = nb::cast<PyDeviceVBMHandle&>(self);
                size_t shape[1] = {static_cast<size_t>(h.blockCount())};
                uint32_t* raw = h.handle.deviceFirstLeafID();
                void* data = (raw != nullptr) ? static_cast<void*>(raw)
                                              : static_cast<void*>(&h);
                // nb::cast of a no-framework device ndarray IS the "dltensor"
                // capsule (what __dlpack__ must return); return it directly.
                nb::ndarray<nb::device::cuda, uint32_t, nb::ndim<1>> arr(
                    data, size_t(1), shape, self);
                return nb::cast(arr, nb::rv_policy::reference);
            },
            "stream"_a = nb::none(),
            "DLPack capsule exporting the firstLeafID array as 1-D uint32, "
            "parented to this handle. The jumpMap is available as a 2-D device "
            "array via jumpMap().");
}

// ------------------- buildVoxelBlockManager (device) binding ---------------

static void defineBuild(nb::module_& m)
{
    m.def("buildVoxelBlockManager",
        [](nb::handle py_grid,
           int log2_block_width,
           uint64_t first_offset,
           uint64_t last_offset,
           uint64_t n_blocks,
           uintptr_t stream) -> PyDeviceVBMHandle {
            auto* d_grid = castOnIndexDeviceGrid(py_grid, "buildVoxelBlockManager");
            cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
            return dispatchLog2BlockWidth(log2_block_width, [&](auto W) {
                constexpr int LBW = decltype(W)::value;
                using Base = VoxelBlockManagerBase<LBW>;
                constexpr uint64_t BlockWidth = Base::BlockWidth;
                // first_offset, if nonzero, must satisfy first_offset == 1
                // (mod BlockWidth); the C++ builder only NANOVDB_ASSERTs this
                // (a no-op in release), so validate it here for a clear error.
                if (first_offset != 0 &&
                    ((first_offset - 1) & (BlockWidth - 1)) != 0) {
                    throw nb::value_error(
                        "buildVoxelBlockManager: first_offset must satisfy "
                        "first_offset == 1 (mod BlockWidth). Pass 0 (the "
                        "default) to let the implementation use 1.");
                }
                VoxelBlockManagerHandle<nanovdb::cuda::DeviceBuffer> handle;
                {
                    // The device builder reads activeVoxelCount / lowerCount
                    // from device memory and launches kernels; pure C++/CUDA
                    // touching no Python objects, so release the GIL.
                    nb::gil_scoped_release release;
                    handle = nanovdb::tools::cuda::buildVoxelBlockManager<
                        LBW, nanovdb::cuda::DeviceBuffer>(
                            d_grid, first_offset, last_offset, n_blocks, s);
                }
                return PyDeviceVBMHandle(std::move(handle), LBW);
            });
        },
        "device_grid"_a,
        "log2_block_width"_a = 6,
        "first_offset"_a = 0,
        "last_offset"_a = 0,
        "n_blocks"_a = 0,
        "stream"_a = 0,
        // The device builder reads the grid from device memory; keep the
        // device grid (and its owning DeviceGridHandle) alive for the duration
        // of the build. The returned handle owns its own device metadata
        // buffers, so it does NOT need to keep the grid alive afterwards.
        "Build a device-side VoxelBlockManager from an OnIndex DEVICE grid. "
        "device_grid MUST be a device grid (from "
        "DeviceGridHandle.deviceGrid(n)); passing a host grid is a usage "
        "error. log2_block_width selects the per-block active-voxel count "
        "(6=64, 7=128, 8=256, 9=512). Pass 0 for first_offset / last_offset / "
        "n_blocks to use the full grid (first active voxel through "
        "activeVoxelCount, minimum block count); these are read from device "
        "memory. first_offset, if nonzero, must satisfy first_offset == 1 "
        "(mod BlockWidth). stream is a raw CUDA stream handle (Python int; 0 = "
        "default stream).");
}

// NOTE: decodeInverseMaps is intentionally NOT bound on the device. The device
// decode (VoxelBlockManager<Log2BlockWidth>::decodeInverseMaps) is a __device__
// function: it uses threadIdx / __syncthreads / shared-memory output arrays and
// is callable only from within a CUDA kernel, never from host code. Users who
// want device-side decode should call it from their own kernels via the shipped
// header <nanovdb/tools/cuda/VoxelBlockManager.cuh>, feeding it the firstLeafID
// / jumpMap device pointers (first_leaf_id_ptr / jump_map_ptr above) and the
// device grid pointer.

// ------------------- gatherBoxStencil (VBM box-stencil gather) -------------
//
// Materialise, for every active voxel, the values of its 3x3x3 neighbourhood
// into a dense (valueCount, 27) array -- the "dense-ise the sparse stencil"
// bridge that lets tile / array frameworks (CuPy, cuTile, ...) run VDB stencils
// without pointer-chasing the tree. Column j is the 3x3x3 spoke
// (di+1)*9 + (dj+1)*3 + (dk+1): the centre is column 13 and the six faces are
// columns 4, 10, 12, 14, 16, 22. Inactive neighbours read the sidecar's
// background slot (value index 0).
template<int Log2BlockWidth, typename T>
__global__ void gatherBoxStencilKernel(
    const NanoGrid<ValueOnIndex>* grid,
    const uint32_t* firstLeafID, const uint64_t* jumpMap, uint64_t firstOffset,
    const T* values, T* out)
{
    constexpr int BW  = 1 << Log2BlockWidth;
    constexpr int JML = BW / 64;
    using VBM = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;
    __shared__ uint32_t smem_leafIndex[BW];
    __shared__ uint16_t smem_voxelOffset[BW];
    const uint64_t blockFirstOffset = firstOffset + uint64_t(blockIdx.x) * BW;
    VBM::template decodeInverseMaps<ValueOnIndex>(
        grid, firstLeafID[blockIdx.x], &jumpMap[uint64_t(blockIdx.x) * JML],
        blockFirstOffset, smem_leafIndex, smem_voxelOffset);
    const int tID = threadIdx.x;
    if (smem_leafIndex[tID] == VBM::UnusedLeafIndex) return;
    uint64_t st[27];
    VBM::template computeBoxStencil<ValueOnIndex>(
        grid, smem_leafIndex, smem_voxelOffset, st);
    const uint64_t c = st[13];                              // centre value index
    #pragma unroll
    for (int j = 0; j < 27; ++j) out[c * 27 + j] = values[st[j]];  // 0 -> background
}

template<typename T> void defineGatherBoxStencil(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::handle py_grid,
           nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cuda> values,
           nb::ndarray<T, nb::shape<-1, 27>, nb::c_contig, nb::device::cuda>  out,
           int log2_block_width, uintptr_t stream) {
            auto* d_grid = castOnIndexDeviceGrid(py_grid, "gatherBoxStencil");
            cudaStream_t s     = reinterpret_cast<cudaStream_t>(stream);
            const T*     dVals = values.data();
            T*           dOut  = out.data();
            // Build a transient VBM, then one block per VBM block decodes and
            // gathers the 27 neighbour values; pure CUDA, so release the GIL.
            nb::gil_scoped_release release;
            dispatchLog2BlockWidth(log2_block_width, [&](auto W) {
                constexpr int LBW = decltype(W)::value;
                auto handle = nanovdb::tools::cuda::buildVoxelBlockManager<
                    LBW, nanovdb::cuda::DeviceBuffer>(d_grid, 0, 0, 0, s);
                const uint32_t bc = static_cast<uint32_t>(handle.blockCount());
                if (bc)
                    gatherBoxStencilKernel<LBW, T><<<bc, (1 << LBW), 0, s>>>(
                        d_grid, handle.deviceFirstLeafID(), handle.deviceJumpMap(),
                        handle.firstOffset(), dVals, dOut);
                cudaCheck(cudaStreamSynchronize(s));
                return 0;
            });
        },
        "device_grid"_a, "values"_a, "out"_a, "log2_block_width"_a = 9, "stream"_a = 0,
        "Gather the 3x3x3 box-stencil neighbour values of every active voxel "
        "into a dense (valueCount, 27) array -- the bridge that lets tile / "
        "array frameworks (CuPy, cuTile, ...) run VDB stencils without pointer-"
        "chasing the tree. device_grid is an OnIndex device grid from "
        "DeviceGridHandle.deviceGrid(n); values is a 1-D device array indexed by "
        "value index (the per-voxel sidecar; entry 0 is the background slot); out "
        "is a 2-D device array of shape (rows, 27) with rows >= valueCount, "
        "filled so out[k, j] is voxel k's neighbour value at 3x3x3 spoke "
        "j = (di+1)*9+(dj+1)*3+(dk+1) (centre j=13; the six faces are "
        "j = 4, 10, 12, 14, 16, 22). Inactive neighbours read values[0]. A "
        "transient VoxelBlockManager is built internally at log2_block_width "
        "(6/7/8/9). stream is a raw CUDA stream handle (Python int; 0 = default "
        "stream).");
}

// ------------------- activeVoxelCoords (VBM coordinate decode) -------------
//
// Write each active voxel's index-space coordinate into a dense (valueCount, 3)
// int32 array, keyed by value index -- the decode companion to
// gatherBoxStencil. Lets callers recover "where is value index k" (e.g. to bake
// a result back into a grid, or to scatter to a dense field) without a
// hand-written decode kernel.
template<int Log2BlockWidth>
__global__ void activeVoxelCoordsKernel(
    const NanoGrid<ValueOnIndex>* grid,
    const uint32_t* firstLeafID, const uint64_t* jumpMap, uint64_t firstOffset,
    int32_t* out)
{
    constexpr int BW  = 1 << Log2BlockWidth;
    constexpr int JML = BW / 64;
    using VBM = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;
    __shared__ uint32_t smem_leafIndex[BW];
    __shared__ uint16_t smem_voxelOffset[BW];
    const uint64_t blockFirstOffset = firstOffset + uint64_t(blockIdx.x) * BW;
    VBM::template decodeInverseMaps<ValueOnIndex>(
        grid, firstLeafID[blockIdx.x], &jumpMap[uint64_t(blockIdx.x) * JML],
        blockFirstOffset, smem_leafIndex, smem_voxelOffset);
    const int tID = threadIdx.x;
    if (smem_leafIndex[tID] == VBM::UnusedLeafIndex) return;
    const auto&    leaf = grid->tree().getFirstNode<0>()[smem_leafIndex[tID]];
    const Coord    c    = leaf.offsetToGlobalCoord(smem_voxelOffset[tID]);
    const uint64_t idx  = leaf.getValue(smem_voxelOffset[tID]);
    out[idx * 3 + 0] = c[0];
    out[idx * 3 + 1] = c[1];
    out[idx * 3 + 2] = c[2];
}

void defineActiveVoxelCoords(nb::module_& m, const char* name)
{
    m.def(
        name,
        [](nb::handle py_grid,
           nb::ndarray<int32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> out,
           int log2_block_width, uintptr_t stream) {
            auto* d_grid = castOnIndexDeviceGrid(py_grid, "activeVoxelCoords");
            cudaStream_t s    = reinterpret_cast<cudaStream_t>(stream);
            int32_t*     dOut = out.data();
            nb::gil_scoped_release release;
            dispatchLog2BlockWidth(log2_block_width, [&](auto W) {
                constexpr int LBW = decltype(W)::value;
                auto handle = nanovdb::tools::cuda::buildVoxelBlockManager<
                    LBW, nanovdb::cuda::DeviceBuffer>(d_grid, 0, 0, 0, s);
                const uint32_t bc = static_cast<uint32_t>(handle.blockCount());
                if (bc)
                    activeVoxelCoordsKernel<LBW><<<bc, (1 << LBW), 0, s>>>(
                        d_grid, handle.deviceFirstLeafID(), handle.deviceJumpMap(),
                        handle.firstOffset(), dOut);
                cudaCheck(cudaStreamSynchronize(s));
                return 0;
            });
        },
        "device_grid"_a, "out"_a, "log2_block_width"_a = 9, "stream"_a = 0,
        "Write each active voxel's index-space coordinate into a dense "
        "(rows, 3) int32 device array keyed by value index (rows >= valueCount); "
        "out[k] is the (i, j, k) coordinate of value index k (row 0, the "
        "background slot, is left untouched). The decode companion to "
        "gatherBoxStencil -- recovers per-voxel coordinates without a "
        "hand-written decode kernel (e.g. to bake a sidecar result back into a "
        "grid). device_grid is an OnIndex device grid from "
        "DeviceGridHandle.deviceGrid(n); a transient VoxelBlockManager is built "
        "internally at log2_block_width (6/7/8/9). stream is a raw CUDA stream "
        "handle (Python int; 0 = default stream).");
}

void defineDeviceVoxelBlockManager(nb::module_& m)
{
    defineHandle(m);
    defineBuild(m);
    defineGatherBoxStencil<float>(m, "gatherBoxStencil");
    defineGatherBoxStencil<double>(m, "gatherBoxStencil");
    defineActiveVoxelCoords(m, "activeVoxelCoords");
}

} // namespace pynanovdb

#endif
