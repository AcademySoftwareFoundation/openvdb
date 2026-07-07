# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Ray-trace a level set on the GPU with a custom CUDA kernel.

GPU counterpart to ``raytrace_level_set.py`` and a Python port of the
device path of ``ex_raytrace_level_set``. The host port loops over
pixels in Python and had to approximate the C++ HDDA surface search with
sphere tracing (``math::Ray`` / ``ZeroCrossing`` are not bound in
Python). Inside a ``cupy.RawKernel`` we have the FULL C++ NanoVDB API,
so this version is the faithful original: one CUDA thread per pixel,
``nanovdb::math::Ray`` clipped to the grid, ``nanovdb::math::ZeroCrossing``
(HDDA) for the exact surface hit, and a central-difference gradient for
Lambert shading. The whole image renders in one launch.

The kernel is compiled against the bundled NanoVDB headers via
``nanovdb.cuda.compile_options()`` (with the ``NANOVDB_INCLUDE`` dev-tree
fallback from ``cupy_rawkernel.py``) and reads the device grid straight
from ``deviceGrid(0).data_ptr()``.

Requires CuPy and a CUDA-capable GPU.

Run with: python raytrace_level_set_cuda.py
"""
import os
import tempfile

import nanovdb

RES = 256
FOV_DEG = 45.0
LIGHT = (0.577, 0.577, 0.577)

KERNEL_SRC = r"""
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>

extern "C" __global__
void render_level_set(const nanovdb::NanoGrid<float>* d_grid,
                      unsigned char* image, int res, float tan_fov,
                      float eye_x, float eye_y, float eye_z,
                      float lx, float ly, float lz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= res || y >= res) return;

    using Vec3T = nanovdb::math::Vec3f;
    using RayT = nanovdb::math::Ray<float>;

    auto acc = d_grid->tree().getAccessor();
    const float px = (2.0f * (x + 0.5f) / res - 1.0f) * tan_fov;
    const float py = (2.0f * (y + 0.5f) / res - 1.0f) * tan_fov;
    const float inv = 1.0f / sqrtf(px * px + py * py + 1.0f);

    RayT ray(Vec3T(eye_x, eye_y, eye_z),
             Vec3T(px * inv, py * inv, -inv));
    unsigned char shade = 0;
    if (ray.clip(d_grid->indexBBox())) {
        nanovdb::Coord ijk;
        float v = 0.0f, t = 0.0f;
        if (nanovdb::math::ZeroCrossing(ray, acc, ijk, v, t)) {
            float gx = acc.getValue(ijk.offsetBy(1, 0, 0))
                     - acc.getValue(ijk.offsetBy(-1, 0, 0));
            float gy = acc.getValue(ijk.offsetBy(0, 1, 0))
                     - acc.getValue(ijk.offsetBy(0, -1, 0));
            float gz = acc.getValue(ijk.offsetBy(0, 0, 1))
                     - acc.getValue(ijk.offsetBy(0, 0, -1));
            const float gl = sqrtf(gx * gx + gy * gy + gz * gz);
            if (gl > 0.0f) { gx /= gl; gy /= gl; gz /= gl; }
            float s = gx * lx + gy * ly + gz * lz;
            if (s < 0.0f) s = 0.0f;
            shade = (unsigned char)(255.0f * s);
        }
    }
    image[y * res + x] = shade;
}
"""


def _include_options():
    """compile_options(), falling back to $NANOVDB_INCLUDE in a dev tree."""
    opts = list(nanovdb.cuda.compile_options("-std=c++17"))
    inc_dir = opts[0][2:]
    if not os.path.isdir(inc_dir):
        env_inc = os.environ.get("NANOVDB_INCLUDE")
        if env_inc and os.path.isdir(env_inc):
            opts[0] = f"-I{env_inc}"
        else:
            print(f"NanoVDB headers not found at {inc_dir!r} (expected in an "
                  "installed wheel). Set NANOVDB_INCLUDE to the dir that "
                  "contains nanovdb/NanoVDB.h to run from a source tree.")
            return None
    return tuple(opts)


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

    options = _include_options()
    if options is None:
        return

    # Level-set sphere built on the device; download once for the host-side
    # camera setup (index bounding box).
    handle = nanovdb.tools.cuda.createLevelSetSphere(
        nanovdb.GridType.Float, radius=100.0)
    handle.deviceUpload(0, True)
    handle.deviceDownload(0, True)
    device_grid = handle.deviceGrid(0)
    bbox = handle.grid(0).indexBBox()

    import math
    dim = [bbox.max[i] + 1 - bbox.min[i] for i in range(3)]
    center = [bbox.min[i] + 0.5 * dim[i] for i in range(3)]
    eye = (center[0], center[1], center[2] + 2.0 * dim[2])
    tan_fov = math.tan(math.radians(FOV_DEG) * 0.5)

    kernel = cp.RawKernel(
        KERNEL_SRC, "render_level_set", options=options, backend="nvrtc")
    image = cp.zeros(RES * RES, dtype=cp.uint8)
    block = (16, 16)
    grid = ((RES + 15) // 16, (RES + 15) // 16)
    kernel(grid, block,
           (device_grid.data_ptr(), image, RES, cp.float32(tan_fov),
            cp.float32(eye[0]), cp.float32(eye[1]), cp.float32(eye[2]),
            cp.float32(LIGHT[0]), cp.float32(LIGHT[1]), cp.float32(LIGHT[2])))
    cp.cuda.runtime.deviceSynchronize()

    host_image = cp.asnumpy(image)
    lit = int((host_image > 0).sum())
    out_dir = tempfile.mkdtemp(prefix="nanovdb_")
    path = os.path.join(out_dir, "raytrace_level_set_cuda.pgm")
    with open(path, "wb") as f:
        f.write(f"P5\n{RES} {RES}\n255\n".encode("ascii"))
        f.write(host_image.tobytes())
    print(f"Rendered {RES}x{RES} level set on the GPU in one launch, "
          f"{lit} lit pixels -> {path}")
    assert lit > 0


if __name__ == "__main__":
    main()
