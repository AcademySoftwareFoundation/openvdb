# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Collide falling particles against a level set on the GPU.

GPU counterpart to ``collide_level_set.py`` and a Python port of the
device path of ``ex_collide_level_set``. The CPU version loops over
particles one at a time, querying the SDF and its gradient per particle;
here the entire particle set is a pair of ``(N, 3)`` CuPy arrays and a
single ``nanovdb.tools.cuda.sampleFromVoxels`` call returns the signed
distance AND the analytic gradient for every particle at once — exactly
what a collision response needs (distance to push out along, normal to
reflect about). The per-step reflection is then plain vectorized CuPy.

Requires CuPy and a CUDA-capable GPU.

Run with: python collide_level_set_cuda.py
"""
import os
import tempfile

import nanovdb

NUM_PARTICLES = 4000
NUM_STEPS = 60
DT = 0.1
GRAVITY = -9.8
RADIUS = 100.0
BAND = 3.0  # narrow-band half-width in world units (default 3 voxels)


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

    host_handle = nanovdb.tools.createLevelSetSphere(
        radius=RADIUS, voxelSize=1.0, name="sphere")
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False)
    tmp.close()
    nanovdb.io.writeGrid(tmp.name, host_handle)
    try:
        handle = nanovdb.io.deviceReadGrid(tmp.name)
    finally:
        os.unlink(tmp.name)
    handle.deviceUpload(0, True)
    device_grid = handle.deviceGrid(0)

    # Seed particles above the north pole of the sphere, falling down.
    rng = cp.random.RandomState(42)
    p = cp.empty((NUM_PARTICLES, 3), dtype=cp.float32)
    p[:, 0] = rng.uniform(-30.0, 30.0, NUM_PARTICLES)
    p[:, 1] = rng.uniform(RADIUS + 15.0, RADIUS + 40.0, NUM_PARTICLES)
    p[:, 2] = rng.uniform(-30.0, 30.0, NUM_PARTICLES)
    v = cp.zeros((NUM_PARTICLES, 3), dtype=cp.float32)
    v[:, 1] = -20.0

    values = cp.empty(NUM_PARTICLES, dtype=cp.float32)
    grads = cp.empty((NUM_PARTICLES, 3), dtype=cp.float32)
    total_collisions = 0

    for step in range(NUM_STEPS):
        v[:, 1] += GRAVITY * DT
        next_p = cp.ascontiguousarray(p + v * DT)

        nanovdb.tools.cuda.sampleFromVoxels(next_p, device_grid, values, grads, 0)
        cp.cuda.Stream.null.synchronize()

        # A collision is a point inside the surface but still within the
        # meaningful narrow band (deep interior clamps to -background).
        hit = (values <= 0.0) & (values > -BAND)
        collisions = int(hit.sum())
        total_collisions += collisions

        # Normalized surface normals from the sampled gradients.
        norm = cp.linalg.norm(grads, axis=1, keepdims=True)
        n = grads / cp.maximum(norm, 1e-8)
        mask = hit[:, None]
        # Project penetrating particles back onto the surface...
        next_p = cp.where(mask, next_p - values[:, None] * n, next_p)
        # ...and reflect their velocity about the surface normal.
        v_dot_n = (v * n).sum(axis=1, keepdims=True)
        v = cp.where(mask, v - 2.0 * v_dot_n * n, v)
        p = next_p

        if collisions:
            print(f"step {step:2d}: {collisions} collisions")

    print(f"total collisions over {NUM_STEPS} steps: {total_collisions}")
    assert total_collisions > 0

    # No particle should end up deep inside the surface.
    nanovdb.tools.cuda.sampleFromVoxels(cp.ascontiguousarray(p), device_grid, values, 0)
    cp.cuda.Stream.null.synchronize()
    in_band = (values > -BAND) & (values < BAND)
    if bool(in_band.any()):
        worst = float(values[in_band].min())
        print(f"min final SDF among in-band particles: {worst:.3f} world units "
              f"(> 0 means all bounced clear of the surface)")
        assert worst > -2.0


if __name__ == "__main__":
    main()
