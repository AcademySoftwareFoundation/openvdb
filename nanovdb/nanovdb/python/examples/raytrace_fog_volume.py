# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Ray-march a fog volume on the CPU and write a PGM image.

Python port of the host path of ex_raytrace_fog_volume: each ray is
clipped to the grid's index bounding box, then integrated with fixed
steps, accumulating transmittance from the density sampled through a
ReadAccessor at Coord.Floor of the march position — the accessor-based
idiom, in contrast to the sampler-based raytrace_level_set.py. Every
step is one Python-to-C++ call, so the default RES is modest; raise it
for a nicer image.

Run with: python raytrace_fog_volume.py
"""
import math
import os
import tempfile

import nanovdb

RES = 64        # image is RES x RES pixels
FOV_DEG = 45.0
DT = 1.0        # march step, in voxels
SIGMA = 0.2     # extinction scale applied to the sampled density


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


def main():
    handle = nanovdb.tools.createFogVolumeSphere(radius=50.0, name="fog")
    grid = handle.grid()
    bbox = grid.indexBBox()
    acc = grid.getAccessor()

    # Perspective camera looking down -z, as in the C++ RayGenOp,
    # working directly in index space (the grid transform is uniform).
    dim = [bbox.max[i] + 1 - bbox.min[i] for i in range(3)]
    center = [bbox.min[i] + 0.5 * dim[i] for i in range(3)]
    eye = (center[0], center[1], center[2] + 2.0 * dim[2])
    tan_fov = math.tan(math.radians(FOV_DEG) * 0.5)

    pixels = bytearray(RES * RES)
    for y in range(RES):
        for x in range(RES):
            px = (2.0 * (x + 0.5) / RES - 1.0) * tan_fov
            py = (2.0 * (y + 0.5) / RES - 1.0) * tan_fov
            norm = math.sqrt(px * px + py * py + 1.0)
            direction = (px / norm, py / norm, -1.0 / norm)

            span = clip_to_bbox(eye, direction, bbox)
            if span is None:
                continue
            transmittance = 1.0
            t = span[0]
            while t < span[1]:
                pos = nanovdb.math.Vec3d(eye[0] + t * direction[0],
                                         eye[1] + t * direction[1],
                                         eye[2] + t * direction[2])
                density = acc.getValue(nanovdb.math.Coord.Floor(pos))
                transmittance *= 1.0 - density * SIGMA * DT
                if transmittance < 0.005:
                    break
                t += DT
            pixels[y * RES + x] = int(255 * (1.0 - transmittance))

    out_dir = tempfile.mkdtemp(prefix="nanovdb_")
    path = os.path.join(out_dir, "raytrace_fog_volume.pgm")
    with open(path, "wb") as f:
        f.write(f"P5\n{RES} {RES}\n255\n".encode("ascii"))
        f.write(bytes(pixels))
    lit = sum(1 for p in pixels if p > 0)
    print(f"Rendered {RES}x{RES} image, {lit} foggy pixels -> {path}")
    assert lit > 0


if __name__ == "__main__":
    main()
