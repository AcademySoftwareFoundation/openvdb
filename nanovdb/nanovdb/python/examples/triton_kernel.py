# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Consume a NanoVDB device buffer from a Triton kernel.

Like Numba (``numba_cuda.py``), Triton does not parse the C++ NanoVDB
header and has no notion of the ``NanoGrid`` ABI. Triton kernels operate
on flat tensors addressed by pointer + offset, so the natural NanoVDB
interop is over the zero-copy *device buffers* NanoVDB exposes, not the
structured grid.

Device-pointer ABI recap (see cupy_rawkernel.py for the C++ ABI route):

* ``handle.deviceGrid(n).data_ptr()`` is a raw DEVICE pointer (Python int)
  to a ``nanovdb::NanoGrid<BuildT>``. Triton cannot decode that layout, so
  this example does not pass the grid pointer to the kernel directly.
* Triton works through a framework tensor: take the handle's zero-copy
  ``__cuda_array_interface__`` / ``__dlpack__`` view (e.g.
  ``torch.from_dlpack(handle)``) and hand the resulting tensor to the
  kernel. Triton then sees a normal contiguous device tensor.
* Provenance is the caller's responsibility: only device buffers go to
  device kernels.

This example wraps the device grid buffer as a Torch CUDA tensor and runs
a trivial Triton element-wise kernel over it. Requires Triton, PyTorch
(with CUDA), and a CUDA-capable GPU; it skips with a message if any are
missing.

Run with: python triton_kernel.py
"""
import nanovdb


def main():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This example requires a CUDA build of nanovdb and a GPU. "
              "Skipping.")
        return
    try:
        import triton
        import triton.language as tl
    except ImportError:
        print("This example requires Triton. Install it with: "
              "pip install triton")
        return
    try:
        import torch
    except ImportError:
        print("This example requires PyTorch (Triton kernels are launched "
              "over Torch tensors). Install it with: pip install torch")
        return
    if not torch.cuda.is_available():
        print("PyTorch reports no CUDA device available. Skipping.")
        return

    # Build a float level-set sphere on the device.
    handle = nanovdb.tools.cuda.createLevelSetSphere(nanovdb.GridType.Float, 20)
    handle.deviceUpload(0, True)
    print(f"Device grid buffer: {handle.size()} bytes at "
          f"{hex(handle.device_ptr())}")

    # Zero-copy: Torch adopts the handle's DLPack device buffer (1-D uint8).
    buf = torch.from_dlpack(handle)
    print(f"  torch tensor: shape={tuple(buf.shape)}, dtype={buf.dtype}, "
          f"is_cuda={buf.is_cuda}")

    @triton.jit
    def count_nonzero(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        tl.atomic_add(out_ptr, tl.sum((x != 0).to(tl.int32)))

    n = buf.numel()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    count_nonzero[grid](buf, out, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    print(f"  Triton counted {int(out.item())} non-zero bytes in the device "
          "grid buffer")
    print("  (Decoding the NanoGrid ABI itself is out of scope for Triton; "
          "use the CuPy RawKernel route for C++ accessor access — see "
          "cupy_rawkernel.py.)")


if __name__ == "__main__":
    main()
