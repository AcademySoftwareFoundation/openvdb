#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""GPU interop unit tests for the NanoVDB Python bindings.

These exercise the ``nanovdb.cuda`` and ``nanovdb.tools.cuda`` surface:
the CUDA-Array-Interface (CAI v3) and DLPack zero-copy bridges on
``DeviceBuffer`` / ``DeviceGridHandle`` / ``UnifiedBuffer``, the
``grid.data_ptr()`` host/device pointer ABI, raw-stream arguments, the
device-grid morphology / index / QC ops, the device NodeManager and
VoxelBlockManager, and the multi-GPU ``DistributedPointsToGrid``
pipeline over managed (unified) memory.

The whole module self-skips when the extension was built without CUDA
or when no CUDA-capable GPU is present, mirroring the device-test
gating in TestNanoVDB.py. Individual tests additionally skip when CuPy
(the only GPU-array framework assumed present) or PyTorch is missing.

Run directly with: python TestGpuInterop.py -v
"""

import os
import subprocess
import sys
import tempfile
import unittest

# If on Windows, add required dll directories from our binary build tree
# (mirrors the bootstrap in TestNanoVDB.py).
if 'add_dll_directory' in dir(os):
    for p in os.environ.get('PATH', '').split(os.pathsep):
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except OSError:
                pass

import nanovdb


def _require_cupy(test):
    """Skip ``test`` (returns the cupy module) unless CuPy is importable."""
    try:
        import cupy as cp
    except ImportError:
        test.skipTest("CuPy not installed")
    return cp


def _build_device_onindex_grid(radius=20.0):
    """Build, write, read-back, and upload a device OnIndex grid.

    Returns (handle, deviceGrid) where handle is a
    nanovdb.cuda.DeviceGridHandle that has been deviceUpload()ed and
    deviceGrid is the DEVICE OnIndexGrid (data_ptr is a device pointer).
    The handle must be kept alive for as long as deviceGrid is used.
    """
    h = nanovdb.tools.createLevelSetSphere(radius=radius, voxelSize=1.0)
    fg = h.grid(0)
    onh = nanovdb.tools.createOnIndexGrid(fg)
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False)
    tmp.close()
    nanovdb.io.writeGrid(tmp.name, onh)
    try:
        dh = nanovdb.io.deviceReadGrid(tmp.name)
    finally:
        os.unlink(tmp.name)
    dh.deviceUpload(0, True)
    return dh, dh.deviceGrid(0)


def _device_onindex_from_coords(cp, coords_np):
    """Build a device OnIndex grid directly from (N,3) int32 index coords (no
    host round-trip; voxelsToOnIndexGrid returns a device grid). Returns
    (handle, deviceGrid, activeVoxelCount, coordsByValueIndex) where the last is
    a CuPy (count+1, 3) int32 array mapping value index -> coord (row 0 unused)."""
    import numpy as np
    coords = cp.asarray(np.ascontiguousarray(coords_np, dtype=np.int32))
    dh = nanovdb.tools.cuda.voxelsToOnIndexGrid(coords, 1.0)
    dg = dh.deviceGrid(0)
    n = int(nanovdb.tools.cuda.buildVoxelBlockManager(dg, 9, 0, 0, 0, 0).lastOffset())
    by_index = cp.empty((n + 1, 3), dtype=cp.int32)
    nanovdb.tools.cuda.activeVoxelCoords(dg, by_index)
    return dh, dg, n, by_index


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestCompileOptions(unittest.TestCase):
    """nanovdb.cuda.compile_options for feeding a runtime CUDA compiler."""

    def test_include_flag_first(self):
        opts = nanovdb.cuda.compile_options()
        self.assertIsInstance(opts, tuple)
        self.assertEqual(len(opts), 1)
        self.assertTrue(opts[0].startswith("-I"))
        # The include flag points at the bundled NanoVDB header dir.
        self.assertTrue(opts[0].endswith(os.path.join("nanovdb", "include")))

    def test_extra_flags_appended_in_order(self):
        opts = nanovdb.cuda.compile_options("-std=c++17", "-O3")
        self.assertEqual(opts[0][:2], "-I")
        self.assertEqual(opts[1:], ("-std=c++17", "-O3"))

    def test_top_level_alias_matches(self):
        # nanovdb._cuda_compile_options is the underlying closure.
        self.assertEqual(
            nanovdb.cuda.compile_options(), nanovdb._cuda_compile_options())


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceBufferInterop(unittest.TestCase):
    """DeviceBuffer CAI v3 / DLPack and from_external wrapping."""

    def test_from_external_wraps_managed_memory(self):
        cp = _require_cupy(self)
        prev = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        try:
            buf = cp.zeros(256, dtype=cp.uint8)
            gpu_ptr = int(buf.data.ptr)
            # Managed memory: a single pointer is valid on both host and
            # device, so host_ptr == device_ptr is legal here.
            ext = nanovdb.cuda.DeviceBuffer.from_external(256, gpu_ptr, gpu_ptr)
            self.assertEqual(ext.size(), 256)
            self.assertEqual(ext.device_ptr(), gpu_ptr)
            self.assertEqual(ext.host_ptr(), gpu_ptr)
        finally:
            cp.cuda.set_allocator(prev)

    def test_from_external_rejects_null_device_pointer(self):
        # gpu_ptr == 0 is a usage error: there is no device memory to wrap.
        with self.assertRaises(ValueError):
            nanovdb.cuda.DeviceBuffer.from_external(256, 0, 0)
        with self.assertRaises(ValueError):
            nanovdb.cuda.DeviceBuffer.from_external(256, 0, 12345)

    def test_cuda_array_interface_v3(self):
        cp = _require_cupy(self)
        prev = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        try:
            buf = cp.zeros(256, dtype=cp.uint8)
            gpu_ptr = int(buf.data.ptr)
            ext = nanovdb.cuda.DeviceBuffer.from_external(256, gpu_ptr, gpu_ptr)
            cai = ext.__cuda_array_interface__
            self.assertEqual(cai["version"], 3)
            self.assertEqual(cai["typestr"], "|u1")
            self.assertEqual(cai["shape"], (256,))
            self.assertEqual(cai["data"][0], gpu_ptr)
            # cupy zero-copy view aliases the same device pointer.
            arr = cp.asarray(ext)
            self.assertEqual(arr.nbytes, ext.size())
            self.assertEqual(int(arr.data.ptr), ext.device_ptr())
        finally:
            cp.cuda.set_allocator(prev)

    def test_dlpack_round_trip(self):
        cp = _require_cupy(self)
        prev = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        try:
            buf = cp.zeros(256, dtype=cp.uint8)
            gpu_ptr = int(buf.data.ptr)
            ext = nanovdb.cuda.DeviceBuffer.from_external(256, gpu_ptr, gpu_ptr)
            # kDLCUDA == 2
            self.assertEqual(ext.__dlpack_device__()[0], 2)
            arr = cp.from_dlpack(ext)
            self.assertEqual(arr.nbytes, ext.size())
            self.assertEqual(int(arr.data.ptr), ext.device_ptr())
        finally:
            cp.cuda.set_allocator(prev)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceGridHandleInterop(unittest.TestCase):
    """DeviceGridHandle CAI/DLPack, the host/device data_ptr ABI, and
    from_buffer zero-copy adoption."""

    def test_cai_exposes_device_buffer_after_upload(self):
        dh, dg = _build_device_onindex_grid(20.0)
        # After upload the whole device buffer is exposed as 1-D uint8.
        cai = dh.__cuda_array_interface__
        self.assertEqual(cai["version"], 3)
        self.assertEqual(cai["typestr"], "|u1")
        self.assertEqual(cai["shape"], (dh.size(),))
        self.assertNotEqual(dh.device_ptr(), 0)
        self.assertEqual(cai["data"][0], dh.device_ptr())

    def test_cupy_zero_copy_aliases_device_ptr(self):
        cp = _require_cupy(self)
        dh, dg = _build_device_onindex_grid(20.0)
        arr = cp.asarray(dh)
        self.assertEqual(arr.nbytes, dh.size())
        self.assertEqual(int(arr.data.ptr), dh.device_ptr())
        # The typed device grid's data_ptr() points at the same buffer base.
        self.assertEqual(dg.data_ptr(), dh.device_ptr())

    def test_dlpack_zero_copy(self):
        cp = _require_cupy(self)
        dh, dg = _build_device_onindex_grid(20.0)
        arr = cp.from_dlpack(dh)
        self.assertEqual(arr.nbytes, dh.size())
        self.assertEqual(int(arr.data.ptr), dh.device_ptr())

    def test_host_vs_device_data_ptr_ambiguity(self):
        # A single handle exposes BOTH a host grid() and a device
        # deviceGrid() after upload; the data_ptr() values differ and the
        # grid object cannot tell host from device — provenance is the
        # caller's responsibility (see the docstring on data_ptr()).
        dh, dg = _build_device_onindex_grid(20.0)
        hg = dh.grid(0)
        self.assertIsNotNone(hg)
        self.assertIsNotNone(dg)
        self.assertNotEqual(hg.data_ptr(), dg.data_ptr())
        self.assertEqual(dg.data_ptr(), dh.device_ptr())

    def test_host_accessor_on_device_grid_segfaults_in_subprocess(self):
        # Calling a HOST-side accessor on a grid from deviceGrid(n)
        # dereferences device memory on the CPU and crashes the process.
        # Run it in a subprocess so the SIGSEGV does not abort the test
        # session, and assert the documented crash contract.
        code = (
            "import nanovdb, tempfile, os\n"
            "h = nanovdb.tools.createLevelSetSphere(radius=20.0, voxelSize=1.0)\n"
            "fg = h.grid(0)\n"
            "onh = nanovdb.tools.createOnIndexGrid(fg)\n"
            "t = tempfile.NamedTemporaryFile(suffix='.nvdb', delete=False)\n"
            "t.close()\n"
            "nanovdb.io.writeGrid(t.name, onh)\n"
            "dh = nanovdb.io.deviceReadGrid(t.name)\n"
            "os.unlink(t.name)\n"
            "dh.deviceUpload(0, True)\n"
            "dg = dh.deviceGrid(0)\n"
            "acc = dg.getAccessor()\n"
            "acc.getValue(nanovdb.math.Coord(0, 0, 0))\n"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True)
        # Killed by a signal => negative return code; -11 is SIGSEGV. Any
        # crash (negative) satisfies the "this is illegal" contract.
        self.assertLess(
            r.returncode, 0,
            "host accessor on a device grid was expected to crash, "
            f"got returncode {r.returncode}")

    def test_from_buffer_adopts_managed_buffer(self):
        cp = _require_cupy(self)
        dh, dg = _build_device_onindex_grid(20.0)
        src = cp.asarray(dh)  # zero-copy device view of a valid grid
        prev = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        try:
            # Managed buffer holding a copy of the valid grid bytes; a
            # single pointer serves as both host and device pointer.
            mbuf = cp.empty(int(dh.size()), dtype=cp.uint8)
            mbuf[:] = src
            cp.cuda.runtime.deviceSynchronize()
            ptr = int(mbuf.data.ptr)
            ext = nanovdb.cuda.DeviceBuffer.from_external(
                int(dh.size()), ptr, ptr)
            adopted = nanovdb.cuda.DeviceGridHandle.from_buffer(ext)
            self.assertEqual(adopted.gridType(0), nanovdb.GridType.OnIndex)
            self.assertEqual(adopted.size(), dh.size())
            # from_buffer MOVES the buffer; the source is left empty.
            self.assertEqual(ext.size(), 0)
        finally:
            cp.cuda.set_allocator(prev)

    def test_from_buffer_rejects_garbage(self):
        cp = _require_cupy(self)
        prev = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        try:
            mbuf = cp.zeros(512, dtype=cp.uint8)  # not a NanoVDB grid header
            ptr = int(mbuf.data.ptr)
            ext = nanovdb.cuda.DeviceBuffer.from_external(512, ptr, ptr)
            with self.assertRaises(RuntimeError):
                nanovdb.cuda.DeviceGridHandle.from_buffer(ext)
        finally:
            cp.cuda.set_allocator(prev)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestUnifiedBufferInterop(unittest.TestCase):
    """UnifiedBuffer create / capacity / zero-copy."""

    def test_create_size_equals_capacity(self):
        ub = nanovdb.cuda.UnifiedBuffer.create(1024)
        self.assertEqual(ub.size(), 1024)
        self.assertEqual(ub.capacity(), 1024)
        self.assertFalse(ub.isEmpty())

    def test_create_with_reserved_capacity(self):
        ub = nanovdb.cuda.UnifiedBuffer.create(1024, 4096)
        self.assertEqual(ub.size(), 1024)
        self.assertEqual(ub.capacity(), 4096)

    def test_resize_within_capacity_keeps_page_table(self):
        ub = nanovdb.cuda.UnifiedBuffer.create(1024, 4096)
        ub.resize(2048)
        self.assertEqual(ub.size(), 2048)
        self.assertEqual(ub.capacity(), 4096)

    def test_cupy_zero_copy(self):
        cp = _require_cupy(self)
        ub = nanovdb.cuda.UnifiedBuffer.create(2048)
        cai = ub.__cuda_array_interface__
        self.assertEqual(cai["version"], 3)
        self.assertEqual(cai["shape"], (2048,))
        arr = cp.asarray(ub)
        self.assertEqual(arr.nbytes, ub.size())
        arr2 = cp.from_dlpack(ub)
        self.assertEqual(arr2.nbytes, ub.size())


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestPointAndVoxelRasterizers(unittest.TestCase):
    """tools.cuda point / voxel rasterizers, including raw-stream args."""

    def test_points_to_grid_world_float(self):
        cp = _require_cupy(self)
        import numpy as np
        pts = cp.asarray(np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]],
                                  dtype=np.float32))
        h = nanovdb.tools.cuda.pointsToGrid(pts, 1.0, 0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.PointIndex)

    def test_points_to_grid_world_double(self):
        cp = _require_cupy(self)
        import numpy as np
        pts = cp.asarray(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64))
        h = nanovdb.tools.cuda.pointsToGrid(pts, 1.0, 0)
        self.assertEqual(h.gridType(0), nanovdb.GridType.PointIndex)

    def test_voxels_to_index_family(self):
        cp = _require_cupy(self)
        import numpy as np
        vox = cp.asarray(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                  dtype=np.int32))
        self.assertEqual(
            nanovdb.tools.cuda.voxelsToOnIndexGrid(vox, 1.0, 0).gridType(0),
            nanovdb.GridType.OnIndex)
        self.assertEqual(
            nanovdb.tools.cuda.voxelsToIndexGrid(vox, 1.0, 0).gridType(0),
            nanovdb.GridType.Index)
        self.assertEqual(
            nanovdb.tools.cuda.voxelsToRGBA8Grid(vox, 1.0, 0).gridType(0),
            nanovdb.GridType.RGBA8)
        self.assertEqual(
            nanovdb.tools.cuda.pointsToRGBA8Grid(vox, 1.0, 0).gridType(0),
            nanovdb.GridType.RGBA8)

    def test_non_default_stream_accepted(self):
        cp = _require_cupy(self)
        import numpy as np
        pts = cp.asarray(np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32))
        stream = cp.cuda.Stream(non_blocking=True)
        # Stream args everywhere are raw CUDA stream handles as Python ints.
        h = nanovdb.tools.cuda.pointsToGrid(pts, 1.0, stream.ptr)
        self.assertEqual(h.gridType(0), nanovdb.GridType.PointIndex)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceGridOps(unittest.TestCase):
    """tools.cuda morphology / QC / sampling on device grids."""

    def test_morphology_returns_onindex(self):
        dh, dg = _build_device_onindex_grid(20.0)
        # op 26 == NN_FACE_EDGE_VERTEX (verified). 18 is unimplemented.
        self.assertEqual(
            nanovdb.tools.cuda.dilateGrid(dg, 26, 0).gridType(0),
            nanovdb.GridType.OnIndex)
        self.assertEqual(
            nanovdb.tools.cuda.coarsenGrid(dg, 0).gridType(0),
            nanovdb.GridType.OnIndex)
        self.assertEqual(
            nanovdb.tools.cuda.refineGrid(dg, 0).gridType(0),
            nanovdb.GridType.OnIndex)

    def test_is_valid_on_device_grid(self):
        dh, dg = _build_device_onindex_grid(20.0)
        self.assertTrue(nanovdb.tools.cuda.isValid(dg))

    def test_signed_flood_fill_in_place(self):
        h = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 20)
        h.deviceUpload(0, True)
        dg = h.deviceGrid(0)
        self.assertIsNotNone(dg)
        # In place on the device grid; no return value.
        self.assertIsNone(nanovdb.tools.cuda.signedFloodFill(dg))

    def test_sample_from_voxels_float(self):
        cp = _require_cupy(self)
        import numpy as np
        h = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 20)
        h.deviceUpload(0, True)
        dg = h.deviceGrid(0)
        pts = cp.asarray(np.array([[0, 0, 0], [20, 0, 0], [10, 0, 0]],
                                  dtype=np.float32))
        vals = cp.empty(3, dtype=cp.float32)
        nanovdb.tools.cuda.sampleFromVoxels(pts, dg, vals, 0)
        out = cp.asnumpy(vals)
        self.assertEqual(out.shape, (3,))

    def test_sample_from_voxels_with_gradients(self):
        cp = _require_cupy(self)
        import numpy as np
        h = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 20)
        h.deviceUpload(0, True)
        dg = h.deviceGrid(0)
        pts = cp.asarray(np.array([[10, 0, 0]], dtype=np.float32))
        vals = cp.empty(1, dtype=cp.float32)
        grads = cp.empty((1, 3), dtype=cp.float32)
        nanovdb.tools.cuda.sampleFromVoxels(pts, dg, vals, grads, 0)
        self.assertEqual(cp.asnumpy(grads).shape, (1, 3))


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceNodeManager(unittest.TestCase):
    """createDeviceNodeManager over a device OnIndex grid."""

    def test_node_manager_handle(self):
        dh, dg = _build_device_onindex_grid(20.0)
        nmh = nanovdb.cuda.createDeviceNodeManager(dg, 0)
        self.assertIsNotNone(nmh)
        self.assertGreater(nmh.size(), 0)
        mgr = nmh.mgr()
        # mgr() returns the TYPED device NodeManager, not a raw int.
        self.assertIsNotNone(mgr)
        self.assertNotIsInstance(mgr, int)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceVoxelBlockManager(unittest.TestCase):
    """buildVoxelBlockManager on a device OnIndex grid + zero-copy buffers."""

    def test_block_manager_properties(self):
        dh, dg = _build_device_onindex_grid(100.0)
        vbm = nanovdb.tools.cuda.buildVoxelBlockManager(dg, 6, 0, 0, 0, 0)
        self.assertGreater(vbm.blockCount(), 0)
        # block_width / log2_block_width / jump_map_length are PROPERTIES.
        self.assertEqual(vbm.block_width, 64)
        self.assertEqual(vbm.log2_block_width, 6)
        # firstOffset == 1 (mod BlockWidth) for a full grid.
        self.assertEqual(vbm.firstOffset(), 1)
        self.assertGreater(vbm.lastOffset(), 0)

    def test_first_leaf_id_zero_copy(self):
        cp = _require_cupy(self)
        dh, dg = _build_device_onindex_grid(100.0)
        vbm = nanovdb.tools.cuda.buildVoxelBlockManager(dg, 6, 0, 0, 0, 0)
        # firstLeafID() / jumpMap() return DLPack capsules directly.
        fl = cp.from_dlpack(vbm.firstLeafID())
        self.assertEqual(fl.dtype, cp.uint32)
        self.assertEqual(fl.shape[0], vbm.blockCount())
        self.assertEqual(int(fl.data.ptr), vbm.first_leaf_id_ptr())

    def test_jump_map_zero_copy(self):
        cp = _require_cupy(self)
        dh, dg = _build_device_onindex_grid(100.0)
        vbm = nanovdb.tools.cuda.buildVoxelBlockManager(dg, 6, 0, 0, 0, 0)
        jm = cp.from_dlpack(vbm.jumpMap())
        self.assertEqual(jm.dtype, cp.uint64)
        self.assertEqual(jm.shape[0], vbm.blockCount())
        self.assertEqual(int(jm.data.ptr), vbm.jump_map_ptr())

    def test_gather_box_stencil_dtypes(self):
        """gatherBoxStencil works for float32 AND the integer payload dtypes
        (int32/uint32). The centre spoke (column 13) is each voxel's own value,
        inactive neighbours read values[0] (=0 here), and gathering arange yields
        the SAME neighbour-index pattern in every dtype -- the property that lets
        callers gather a neighbour-INDEX table directly in int32.

        Built from explicit coords (voxelsToOnIndexGrid) so active-voxel indexing
        is contiguous -- the invariant the VoxelBlockManager gather relies on."""
        cp = _require_cupy(self)
        import numpy as np
        block = np.array([(i, j, k) for i in range(5) for j in range(5) for k in range(5)],
                         dtype=np.int32)
        _h, dg, n, _co = _device_onindex_from_coords(cp, block)
        N = n + 1
        ref = None
        for dt in (cp.float32, cp.int32, cp.uint32):
            idx = cp.arange(N, dtype=dt)
            out = cp.empty((N, 27), dtype=dt)
            nanovdb.tools.cuda.gatherBoxStencil(dg, idx, out)
            self.assertEqual(out.dtype, cp.dtype(dt))
            self.assertTrue(bool((out[1:N, 13] == idx[1:N]).all()))   # centre == self
            self.assertTrue(bool((out[1:N] == 0).any()))              # inactive -> values[0]
            # row 0 (background slot) is never written by the kernel, so compare
            # only the written rows across dtypes.
            as_i64 = out[1:N].astype(cp.int64)
            if ref is None:
                ref = as_i64
            else:
                self.assertTrue(bool((as_i64 == ref).all()))          # dtype-independent

    def test_gather_rejects_noncontiguous_grid(self):
        """gatherBoxStencil / activeVoxelCoords raise a clear ValueError (rather
        than an illegal memory access) on a grid whose active-voxel indexing is
        NOT contiguous -- e.g. createOnIndexGrid's default include_stats /
        include_tiles, which add value slots past activeVoxelCount."""
        cp = _require_cupy(self)
        dh, dg = _build_device_onindex_grid(20.0)   # createOnIndexGrid defaults: stats+tiles
        N = int(nanovdb.tools.cuda.buildVoxelBlockManager(dg, 9, 0, 0, 0, 0).lastOffset()) + 1
        with self.assertRaises(ValueError):
            nanovdb.tools.cuda.gatherBoxStencil(dg, cp.arange(N, dtype=cp.int32),
                                                cp.empty((N, 27), dtype=cp.int32))
        with self.assertRaises(ValueError):
            nanovdb.tools.cuda.activeVoxelCoords(dg, cp.empty((N, 3), dtype=cp.int32))

    def test_gather_box_stencil_columns(self):
        """gatherBoxStencilColumns gathers a CHOSEN subset of the 27 spokes into
        an (N,K) table -- equal to the matching columns of the full gather -- and
        validates its (host) spoke list + the out shape."""
        cp = _require_cupy(self)
        import numpy as np
        block = np.array([(i, j, k) for i in range(5) for j in range(5) for k in range(5)],
                         dtype=np.int32)
        _h, dg, n, _co = _device_onindex_from_coords(cp, block)
        N = n + 1
        idx = cp.arange(N, dtype=cp.int32)
        full = cp.empty((N, 27), dtype=cp.int32)
        nanovdb.tools.cuda.gatherBoxStencil(dg, idx, full)
        spokes = np.array([13, 22, 16, 14, 4, 10, 12], dtype=np.int32)   # centre + 6 faces
        sub = cp.empty((N, spokes.shape[0]), dtype=cp.int32)
        nanovdb.tools.cuda.gatherBoxStencilColumns(dg, idx, sub, spokes)
        self.assertTrue(bool((sub[1:N] == full[1:N][:, cp.asarray(spokes)]).all()))
        with self.assertRaises(ValueError):          # spoke out of [0, 27)
            nanovdb.tools.cuda.gatherBoxStencilColumns(
                dg, idx, cp.empty((N, 2), dtype=cp.int32), np.array([13, 99], np.int32))
        with self.assertRaises(ValueError):          # out.shape[1] != len(spokes)
            nanovdb.tools.cuda.gatherBoxStencilColumns(dg, idx, cp.empty((N, 3), dtype=cp.int32), spokes)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceInject(unittest.TestCase):
    """inject / injectFeatures copy a sidecar over the src/dst voxel
    intersection, for float AND the integer payloads (int32/uint32) -- the copy
    is via the assignment operator, so it is type-generic."""

    # src = a 4^3 block; dst = a 6^3 block (a superset), so the intersection is
    # exactly src. enc(coord) gives each voxel a distinct in-range value.
    SRC = [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]
    DST = [(i, j, k) for i in range(6) for j in range(6) for k in range(6)]
    SENT = 255  # distinct from every enc value; exact in f32/i32/u32

    @staticmethod
    def _enc(c):                                   # (...,3) int -> (...) int
        return (c[..., 0] * 6 + c[..., 1]) * 6 + c[..., 2]

    def test_inject_scalar_dtypes(self):
        cp = _require_cupy(self)
        import numpy as np
        _sh, sdg, ns, sco = _device_onindex_from_coords(cp, np.array(self.SRC, np.int32))
        _dh, ddg, nd, dco = _device_onindex_from_coords(cp, np.array(self.DST, np.int32))
        sco_h = cp.asnumpy(sco[1:]).astype(np.int64)
        dco_h = cp.asnumpy(dco[1:]).astype(np.int64)
        in_src = (dco_h < 4).all(axis=1)
        expect = np.where(in_src, self._enc(dco_h), self.SENT)
        for dt in (cp.float32, cp.int32, cp.uint32):
            with self.subTest(dtype=np.dtype(dt).name):
                src_vals = cp.zeros(ns + 1, dtype=dt)
                src_vals[1:] = cp.asarray(self._enc(sco_h), dtype=dt)
                dst_vals = cp.full(nd + 1, self.SENT, dtype=dt)
                nanovdb.tools.cuda.inject(sdg, ddg, src_vals, dst_vals)
                self.assertEqual(dst_vals.dtype, cp.dtype(dt))
                got = cp.asnumpy(dst_vals[1:]).astype(np.int64)
                self.assertTrue(bool((got == expect).all()))

    def test_inject_features_dtypes(self):
        """injectFeatures: the 2-D (value count, dim) form, integer + float."""
        cp = _require_cupy(self)
        import numpy as np
        _sh, sdg, ns, sco = _device_onindex_from_coords(cp, np.array(self.SRC, np.int32))
        _dh, ddg, nd, dco = _device_onindex_from_coords(cp, np.array(self.DST, np.int32))
        sco_h = cp.asnumpy(sco[1:]).astype(np.int64)
        dco_h = cp.asnumpy(dco[1:]).astype(np.int64)
        in_src = (dco_h < 4).all(axis=1)
        enc_d = self._enc(dco_h)
        for dt in (cp.float32, cp.int32):
            with self.subTest(dtype=np.dtype(dt).name):
                src_vals = cp.zeros((ns + 1, 2), dtype=dt)
                e = self._enc(sco_h)
                src_vals[1:, 0] = cp.asarray(e, dtype=dt)
                src_vals[1:, 1] = cp.asarray(e + 1000, dtype=dt)   # second channel
                dst_vals = cp.full((nd + 1, 2), self.SENT, dtype=dt)
                nanovdb.tools.cuda.inject(sdg, ddg, src_vals, dst_vals)
                got = cp.asnumpy(dst_vals[1:]).astype(np.int64)
                exp0 = np.where(in_src, enc_d, self.SENT)
                exp1 = np.where(in_src, enc_d + 1000, self.SENT)
                self.assertTrue(bool((got[:, 0] == exp0).all()))
                self.assertTrue(bool((got[:, 1] == exp1).all()))


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceTypedSidecars(unittest.TestCase):
    """Integer payloads / build types: addBlindData with int32/int64 blind data,
    and indexToGrid materialising an Int32 destination grid."""

    BLOCK = [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]

    def test_add_blind_data_int_dtypes(self):
        cp = _require_cupy(self)
        import numpy as np
        _h, dg, n, _co = _device_onindex_from_coords(cp, np.array(self.BLOCK, np.int32))
        for dt in (cp.int32, cp.int64, cp.float32):     # float32 = regression
            with self.subTest(dtype=np.dtype(dt).name):
                blind = (cp.arange(n + 1, dtype=dt) * 3)
                out_dh = nanovdb.tools.cuda.addBlindData(
                    dg, blind, nanovdb.GridBlindDataClass.ChannelArray,
                    nanovdb.GridBlindDataSemantic.Unknown, "labels")
                out_dh.deviceDownload(0, True)
                g = out_dh.grid(0)
                self.assertEqual(g.blindDataCount(), 1)
                bd = np.asarray(g.getBlindData(0))
                self.assertEqual(bd.dtype, np.dtype(dt))
                self.assertTrue(bool((bd == cp.asnumpy(blind)).all()))

    def test_index_to_grid_int32(self):
        cp = _require_cupy(self)
        import numpy as np
        _h, dg, n, _co = _device_onindex_from_coords(cp, np.array(self.BLOCK, np.int32))
        vals = cp.arange(n + 1, dtype=cp.int32) * 7        # int32 sidecar -> Int32Grid
        out_dh = nanovdb.tools.cuda.indexToGrid(dg, vals)
        self.assertEqual(out_dh.gridType(0), nanovdb.GridType.Int32)
        out_dh.deviceDownload(0, True)
        self.assertEqual(out_dh.grid(0).activeVoxelCount(), n)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDeviceInfra(unittest.TestCase):
    """DeviceMesh / DeviceStreamMap / DeviceResource / TempDevicePool."""

    def test_device_mesh(self):
        mesh = nanovdb.cuda.DeviceMesh()
        self.assertGreaterEqual(mesh.deviceCount(), 1)
        # canAccessPeer is well-defined (self-peer is typically False).
        self.assertIsInstance(mesh.canAccessPeer(0, 0), bool)

    def test_device_stream_map(self):
        DSM = nanovdb.cuda.DeviceStreamMap
        sm = DSM(DSM.DeviceType.Unified, [], 0)
        self.assertGreaterEqual(sm.deviceCount(), 1)
        # stream(deviceId) is a raw CUDA stream handle as a Python int.
        self.assertIsInstance(sm.stream(0), int)

    def test_device_resource_alloc_dealloc(self):
        DR = nanovdb.cuda.DeviceResource
        self.assertEqual(DR.DEFAULT_ALIGNMENT, 256)
        ptr = DR.allocateAsync(1024, 256, 0)
        self.assertIsInstance(ptr, int)
        self.assertNotEqual(ptr, 0)
        DR.deallocateAsync(ptr, 1024, 256, 0)

    def test_temp_device_pool(self):
        tp = nanovdb.cuda.TempDevicePool()
        tp.setRequestedSize(2048)
        tp.reallocate(0)
        self.assertEqual(tp.requestedSize(), 2048)
        self.assertGreaterEqual(tp.size(), 2048)
        self.assertNotEqual(tp.data(), 0)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestDistributedPointsToGrid(unittest.TestCase):
    """Multi-GPU DistributedPointsToGrid over MANAGED coordinate arrays.

    With a single device this exercises the trivial single-GPU path, but
    still validates the cuda_managed memory constraint and the
    UnifiedGridHandle result type.
    """

    def test_managed_coords_produce_unified_handle(self):
        cp = _require_cupy(self)
        import numpy as np
        prev = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.cuda.malloc_managed)
        try:
            pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]],
                           dtype=np.int32)
            voxels = cp.asarray(pts)  # MANAGED (unified) memory
            mesh = nanovdb.cuda.DeviceMesh()
            # The mesh must outlive the converter (held by reference).
            conv = nanovdb.tools.cuda.DistributedPointsToGrid(
                mesh, 1.0, (0.0, 0.0, 0.0))
            uh = conv.getHandle(voxels)
            self.assertEqual(uh.gridType(0), nanovdb.GridType.OnIndex)
            # UnifiedGridHandle intentionally does NOT expose the
            # CAI / DLPack bridges that DeviceGridHandle does.
            self.assertFalse(hasattr(uh, "__cuda_array_interface__"))
        finally:
            cp.cuda.set_allocator(prev)


@unittest.skipIf(
    not nanovdb.isCudaAvailable(), "nanovdb module was compiled without CUDA support"
)
@unittest.skipIf(
    not nanovdb.isGpuAvailable(), "No CUDA-capable GPU available"
)
class TestTorchInterop(unittest.TestCase):
    """Optional PyTorch interop (skips when torch is not installed)."""

    def test_torch_cuda_array_interface(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        if not torch.cuda.is_available():
            self.skipTest("PyTorch built without CUDA / no GPU")
        dh, dg = _build_device_onindex_grid(20.0)
        # torch can adopt the DeviceGridHandle device buffer via DLPack.
        t = torch.from_dlpack(dh)
        self.assertEqual(t.numel(), dh.size())
        self.assertTrue(t.is_cuda)


if __name__ == "__main__":
    unittest.main()
