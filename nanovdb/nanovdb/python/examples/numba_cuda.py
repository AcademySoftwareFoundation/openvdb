# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Consume a NanoVDB device grid from a Numba CUDA kernel.

Numba does not parse the C++ NanoVDB header, so it cannot use the
``NanoGrid`` accessor types directly the way the CuPy RawKernel example
(``cupy_rawkernel.py``) does. The realistic Numba pattern is therefore to
operate on the zero-copy *device arrays* NanoVDB hands back — the whole
grid buffer (CAI / DLPack on the handle) or the typed per-leaf / blind-data
buffers — rather than to dereference the ``NanoGrid`` ABI from JIT'd code.

Device-pointer ABI recap (see cupy_rawkernel.py for the C++ ABI route):

* ``handle.deviceGrid(n).data_ptr()`` is a raw DEVICE pointer (Python int)
  to a ``nanovdb::NanoGrid<BuildT>``. Decoding the NanoGrid layout from a
  Numba kernel means re-implementing the offset math by hand — out of
  scope for this example.
* The supported, zero-copy Numba path: wrap a NanoVDB buffer that already
  exposes ``__cuda_array_interface__`` (the DeviceGridHandle, a
  UnifiedBuffer, or the VoxelBlockManager firstLeafID / jumpMap DLPack
  capsules) with ``numba.cuda.as_cuda_array`` and process the raw bytes /
  indices in a kernel.
* Provenance is the caller's responsibility: only feed device pointers /
  device CAI buffers to device kernels. A host pointer from ``grid(n)``
  would fault on the GPU.

This example wraps the whole device grid buffer as a uint8 CUDA array and
runs a trivial reduction over it to show the interop plumbing. Requires
Numba (with CUDA support) and a CUDA-capable GPU; it skips with a message
when Numba is not installed.

Run with: python numba_cuda.py
"""
import nanovdb


def main():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This example requires a CUDA build of nanovdb and a GPU. "
              "Skipping.")
        return
    try:
        from numba import cuda
    except ImportError:
        print("This example requires Numba (with CUDA support). Install it "
              "with: pip install numba")
        return
    if not cuda.is_available():
        print("Numba reports no CUDA device available. Skipping.")
        return

    # Build a float level-set sphere on the device.
    handle = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 20)
    handle.deviceUpload(0, True)
    print(f"Device grid buffer: {handle.size()} bytes at "
          f"{hex(handle.device_ptr())}")

    # Zero-copy: Numba adopts the handle's __cuda_array_interface__ (the
    # whole device buffer as 1-D uint8) with no copy.
    dev_bytes = cuda.as_cuda_array(handle)
    print(f"  numba CUDA array: shape={dev_bytes.shape}, "
          f"dtype={dev_bytes.dtype}")

    # Trivial per-element kernel over the raw buffer (XOR-fold into a
    # device scalar) just to demonstrate launching JIT'd Numba code on a
    # NanoVDB-owned device buffer.
    @cuda.jit
    def fold_xor(buf, out):
        i = cuda.grid(1)
        if i < buf.size:
            cuda.atomic.xor(out, 0, buf[i])

    import numpy as np
    out = cuda.to_device(np.zeros(1, dtype=np.uint8))
    threads = 256
    blocks = (dev_bytes.size + threads - 1) // threads
    fold_xor[blocks, threads](dev_bytes, out)
    cuda.synchronize()
    print(f"  XOR-fold of device buffer = {int(out.copy_to_host()[0])}")
    print("  (Decoding the NanoGrid ABI itself from Numba requires hand-"
          "rolled offset math; use the CuPy RawKernel route for C++ "
          "accessor access — see cupy_rawkernel.py.)")


if __name__ == "__main__":
    main()
