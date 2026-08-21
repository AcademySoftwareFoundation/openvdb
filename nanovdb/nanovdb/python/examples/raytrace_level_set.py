# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Ray-trace a narrow-band level set on the CPU and write a PGM image.

Python port of the host path of ex_raytrace_level_set. The C++ example
finds surface hits with math::Ray + HDDA ZeroCrossing, which are not
bound in Python; this port re-expresses the search as sphere tracing —
the clamped SDF value itself bounds the safe step size — using the
trilinear sampler, with sampler.zeroCrossing() confirming the interval
and sampler.gradient() shading the hit. Every sample is one
Python-to-C++ call, so the default RES is modest; raise it for a nicer
image.

Run with: python raytrace_level_set.py
"""
import math
import os
import tempfile

import nanovdb

RES = 64          # image is RES x RES pixels
FOV_DEG = 45.0
LIGHT = (0.577, 0.577, 0.577)  # unit vector toward the light


def clip_to_bbox(eye, direction, bbox):
    """Slab-clip a ray against a CoordBBox; returns (t0, t1) or None."""
    t0, t1 = 0.0, math.inf
    for axis in range(3):
        lo = bbox.min[axis] - eye[axis]
        hi = bbox.max[axis] + 1.0 - eye[axis]
        d = direction[axis]
        if abs(d) < 1e-12:
            if lo > 0.0 or hi < 0.0:
                return None
            continue
        near, far = lo / d, hi / d
        if near > far:
            near, far = far, near
        t0, t1 = max(t0, near), min(t1, far)
        if t0 > t1:
            return None
    return t0, t1


def trace(sampler, eye, direction, t0, t1, voxel_size):
    """Sphere-trace from t0 to t1; returns (hit t, crossing seen) or None."""
    t = t0
    while t < t1:
        pos = nanovdb.math.Vec3d(eye[0] + t * direction[0],
                                 eye[1] + t * direction[1],
                                 eye[2] + t * direction[2])
        d = sampler(pos)  # world-unit SDF, clamped to the narrow band
        if d <= 0.0:
            # zeroCrossing() reports whether the reconstruction stencil
            # here straddles the iso-surface — the sampler-level analog
            # of the HDDA ZeroCrossing test in the C++ example.
            return t, sampler.zeroCrossing(pos)
        # The clamped SDF bounds the distance to the surface, so it is
        # a safe (index-space) step size; never step below half a voxel.
        t += max(d / voxel_size, 0.5)
    return None


def main():
    handle = nanovdb.tools.createLevelSetSphere(radius=100.0, name="sphere")
    grid = handle.grid()
    bbox = grid.indexBBox()
    voxel_size = grid.voxelSize()[0]
    sampler = nanovdb.math.createTrilinearSampler(grid)

    # Perspective camera looking down -z, as in the C++ RayGenOp,
    # working directly in index space (the grid transform is uniform).
    dim = [bbox.max[i] + 1 - bbox.min[i] for i in range(3)]
    center = [bbox.min[i] + 0.5 * dim[i] for i in range(3)]
    eye = nanovdb.math.Vec3d(center[0], center[1], center[2] + 2.0 * dim[2])
    tan_fov = math.tan(math.radians(FOV_DEG) * 0.5)

    pixels = bytearray(RES * RES)
    crossings = 0
    for y in range(RES):
        for x in range(RES):
            px = (2.0 * (x + 0.5) / RES - 1.0) * tan_fov
            py = (2.0 * (y + 0.5) / RES - 1.0) * tan_fov
            norm = math.sqrt(px * px + py * py + 1.0)
            direction = (px / norm, py / norm, -1.0 / norm)

            span = clip_to_bbox(eye, direction, bbox)
            if span is None:
                continue
            hit = trace(sampler, eye, direction, span[0], span[1],
                        voxel_size)
            if hit is None:
                continue
            t_hit, crossed = hit
            crossings += crossed
            pos = nanovdb.math.Vec3d(eye[0] + t_hit * direction[0],
                                     eye[1] + t_hit * direction[1],
                                     eye[2] + t_hit * direction[2])
            # gradient() is in index-space units — normalize before use.
            n = sampler.gradient(pos)
            n.normalize()
            shade = max(0.0, n[0] * LIGHT[0] + n[1] * LIGHT[1]
                        + n[2] * LIGHT[2])
            pixels[y * RES + x] = int(255 * shade)

    out_dir = tempfile.mkdtemp(prefix="nanovdb_")
    path = os.path.join(out_dir, "raytrace_level_set.pgm")
    with open(path, "wb") as f:
        f.write(f"P5\n{RES} {RES}\n255\n".encode("ascii"))
        f.write(bytes(pixels))
    lit = sum(1 for p in pixels if p > 0)
    print(f"Rendered {RES}x{RES} image, {lit} lit pixels "
          f"({crossings} confirmed zero-crossings) -> {path}")
    assert lit > 0 and crossings > 0


if __name__ == "__main__":
    main()
