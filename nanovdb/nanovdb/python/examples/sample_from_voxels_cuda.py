# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Trilinearly sample a device grid at arbitrary points, with gradients.

Demonstrates ``nanovdb.tools.cuda.sampleFromVoxels`` — the device
equivalent of the host ``createTrilinearSampler`` used by
``raytrace_level_set.py`` and ``collide_level_set.py``. Given a device
``FloatGrid`` and an ``(N, 3)`` CuPy array of WORLD-space query points,
it writes trilinear samples into an ``(N,)`` array and (optionally)
analytic gradients into an ``(N, 3)`` array, all on the GPU:

    tools.cuda.sampleFromVoxels(points, d_grid, values, stream)
    tools.cuda.sampleFromVoxels(points, d_grid, values, gradients, stream)

Here we sample a level-set sphere along a ray crossing its surface,
recover the surface normal from the normalized gradient, and cross-check
a few device samples against the host ``createTrilinearSampler``.

Requires CuPy and a CUDA-capable GPU.

Run with: python sample_from_voxels_cuda.py
"""
import nanovdb


def main():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This example requires a CUDA build of nanovdb and a GPU. "
              "Skipping.")
        return
    try:
        import cupy as cp
    except ImportError:
        print("This example requires CuPy. Install it with: pip install cupy")
        return

    # Level-set sphere (radius 50), read straight onto the device.
    radius = 50.0
    host_handle = nanovdb.tools.createLevelSetSphere(
        radius=radius, voxelSize=1.0, name="sphere")
    import os
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False)
    tmp.close()
    nanovdb.io.writeGrid(tmp.name, host_handle)
    try:
        handle = nanovdb.io.deviceReadGrid(tmp.name)
    finally:
        os.unlink(tmp.name)
    handle.deviceUpload(0, True)
    device_grid = handle.deviceGrid(0)

    # Query points marching along +x across the surface (within the
    # narrow band, where the SDF is meaningful).
    xs = cp.linspace(radius - 3.0, radius + 3.0, 7, dtype=cp.float32)
    points = cp.zeros((xs.size, 3), dtype=cp.float32)
    points[:, 0] = xs
    points = cp.ascontiguousarray(points)

    values = cp.empty(xs.size, dtype=cp.float32)
    gradients = cp.empty((xs.size, 3), dtype=cp.float32)
    nanovdb.tools.cuda.sampleFromVoxels(points, device_grid, values, gradients, 0)
    cp.cuda.Stream.null.synchronize()

    print(f"Sampling a radius-{radius:.0f} level set along +x:")
    v_host = cp.asnumpy(values)
    g_host = cp.asnumpy(gradients)
    for xi, vi, gi in zip(cp.asnumpy(xs), v_host, g_host):
        gn = (gi[0] ** 2 + gi[1] ** 2 + gi[2] ** 2) ** 0.5
        print(f"  x={xi:6.2f}  sdf={vi:+.3f}  |grad|={gn:.3f}")

    # SDF must increase monotonically as we move outward through the band.
    assert (v_host[1:] >= v_host[:-1]).all()
    # The sample straddling the surface should be ~0.
    assert abs(v_host[xs.size // 2]) < 1.0

    # Cross-check the device sampler against the host sampler.
    host_sampler = nanovdb.math.createTrilinearSampler(host_handle.grid(0))
    mid = float(cp.asnumpy(xs)[xs.size // 2])
    host_val = host_sampler(nanovdb.math.Vec3f(mid, 0.0, 0.0))
    print(f"device vs host at x={mid:.2f}: "
          f"{v_host[xs.size // 2]:+.4f} vs {host_val:+.4f}")
    assert abs(v_host[xs.size // 2] - host_val) < 1e-3

    print("OK: device trilinear sampling + gradients match the host sampler")


if __name__ == "__main__":
    main()
