# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""levelset_filter NVIDIA cuTile backend -- per-voxel stencils as tile kernels.

The cuTile backend for levelset_filter.py: this is levelset_filter_cupy.py with
the per-voxel stencil math (deform / renorm / extrapolate) moved from CuPy array
ops into NVIDIA cuTile (`cuda.tile`) kernels that `ct.load`/`ct.store` over
`(TILE,)` tiles. The SPARSE half (neighbour gather, coord decode, topology)
stays in bound NanoVDB ops; only the DENSE per-voxel compute is cuTile. Run the
full file->file filter through the driver:

    python levelset_filter.py cutile  input.nvdb output.nvdb [outer_iterations]

Running this file directly executes a stencil-only smoke test (the deform +
renorm kernels on a sphere, no file I/O and no narrow-band retrack):

    python levelset_filter_cutile.py

The kernels load six 1-D face arrays per tile (cuTile tile dims must be
compile-time powers of two, so a `(TILE, 6)` tile of the gather columns isn't
allowed); the sign-clamped background BC is applied in CuPy before launch
(`lsf.gather_faces`), and the extrapolation BC is done inside its kernel.
Validated against levelset_filter_cupy.py (same spheres shrink the same amount).
Requires CuPy + cuda-tile (`cuda.tile`) and a CUDA-capable GPU.
"""
import nanovdb

import levelset_filter as lsf

try:
    import cupy as cp
    import cuda.tile as ct
    HAVE_CUTILE = True
except ImportError:
    HAVE_CUTILE = False


def _kernel(fn):
    """Apply @cuda.tile.kernel when available; else a no-op so the module still
    imports (the kernels are never called when cuTile / a GPU is absent)."""
    return ct.kernel(fn) if HAVE_CUTILE else fn


def make_backend():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This backend requires a CUDA build of nanovdb and a GPU. Skipping.")
        return None
    if not HAVE_CUTILE:
        print("This backend requires CuPy + cuda-tile (cuda.tile). Skipping.")
        return None
    return Backend(cp)


TILE = 256
SENTINEL = 1.0e30          # must match levelset_filter.SENTINEL (read inside a kernel)


# ----------------------------- cuTile kernels -----------------------------
@_kernel
def laplacian_kernel(c, xm, xp, ym, yp, zm, zp, out):
    b = ct.bid(0)
    cc = ct.load(c, index=(b,), shape=(TILE,))
    s6 = (ct.load(xm, index=(b,), shape=(TILE,)) + ct.load(xp, index=(b,), shape=(TILE,))
          + ct.load(ym, index=(b,), shape=(TILE,)) + ct.load(yp, index=(b,), shape=(TILE,))
          + ct.load(zm, index=(b,), shape=(TILE,)) + ct.load(zp, index=(b,), shape=(TILE,)))
    ct.store(out, index=(b,), tile=cc + (s6 - 6.0 * cc) / 6.0)


@_kernel
def godunov_kernel(c, xm, xp, ym, yp, zm, zp, dx, dt, out):
    b = ct.bid(0)
    cc = ct.load(c, index=(b,), shape=(TILE,))
    xm_ = ct.load(xm, index=(b,), shape=(TILE,)); xp_ = ct.load(xp, index=(b,), shape=(TILE,))
    ym_ = ct.load(ym, index=(b,), shape=(TILE,)); yp_ = ct.load(yp, index=(b,), shape=(TILE,))
    zm_ = ct.load(zm, index=(b,), shape=(TILE,)); zp_ = ct.load(zp, index=(b,), shape=(TILE,))
    s = cc / ct.sqrt(cc * cc + dx * dx)
    pos = s > 0.0

    def axis(dm, dp):
        return ct.where(pos,
                        ct.maximum(ct.maximum(dm, 0.0) * ct.maximum(dm, 0.0),
                                   ct.minimum(dp, 0.0) * ct.minimum(dp, 0.0)),
                        ct.maximum(ct.minimum(dm, 0.0) * ct.minimum(dm, 0.0),
                                   ct.maximum(dp, 0.0) * ct.maximum(dp, 0.0)))

    grad = ct.sqrt(axis((cc - xm_) / dx, (xp_ - cc) / dx)
                   + axis((cc - ym_) / dx, (yp_ - cc) / dx)
                   + axis((cc - zm_) / dx, (zp_ - cc) / dx))
    ct.store(out, index=(b,), tile=cc - dt * s * (grad - 1.0))


@_kernel
def extrapolate_kernel(c, xm, xp, ym, yp, zm, zp, dx, out):
    b = ct.bid(0)
    cc = ct.load(c, index=(b,), shape=(TILE,))
    f0 = ct.load(xm, index=(b,), shape=(TILE,)); f1 = ct.load(xp, index=(b,), shape=(TILE,))
    f2 = ct.load(ym, index=(b,), shape=(TILE,)); f3 = ct.load(yp, index=(b,), shape=(TILE,))
    f4 = ct.load(zm, index=(b,), shape=(TILE,)); f5 = ct.load(zp, index=(b,), shape=(TILE,))
    best = ct.full((TILE,), SENTINEL, ct.float32)      # min-|value| active neighbour
    best = ct.where(ct.abs(f0) < ct.abs(best), f0, best)
    best = ct.where(ct.abs(f1) < ct.abs(best), f1, best)
    best = ct.where(ct.abs(f2) < ct.abs(best), f2, best)
    best = ct.where(ct.abs(f3) < ct.abs(best), f3, best)
    best = ct.where(ct.abs(f4) < ct.abs(best), f4, best)
    best = ct.where(ct.abs(f5) < ct.abs(best), f5, best)
    filled = best + ct.where(best >= 0.0, dx, -dx)     # phi_nbr + sign*dx
    is_new = cc == SENTINEL
    has = ct.abs(best) < SENTINEL
    ct.store(out, index=(b,), tile=ct.where(is_new, ct.where(has, filled, cc), cc))


def _pad(cp, a, m):
    out = cp.zeros(m, dtype=a.dtype)
    out[:a.shape[0]] = a
    return out


class Backend(lsf.GatherBackend):
    """Per-voxel stencils as cuTile kernels; `setup`/`active_coords` come from the
    shared GatherBackend (gatherBoxStencil / activeVoxelCoords)."""

    NAME = "cutile"

    def _apply(self, kernel, g, phi, clamp, background, extra):
        """gather (driver) -> contiguous columns -> pad -> cuTile launch -> new
        value-indexed sidecar."""
        cp = self.cp
        n = g["n"]
        c, f = lsf.gather_faces(cp, g, phi, background, clamp=clamp)
        cols = [cp.ascontiguousarray(f[:, i]) for i in range(6)]
        m = ct.cdiv(n, TILE) * TILE
        args = [_pad(cp, a, m) for a in (cp.ascontiguousarray(c), *cols)]
        out = cp.zeros(m, dtype=cp.float32)
        ct.launch(cp.cuda.get_current_stream(), (ct.cdiv(m, TILE), 1, 1),
                  kernel, (*args, *extra, out))
        cp.cuda.runtime.deviceSynchronize()
        new = cp.full(n + 1, SENTINEL, dtype=cp.float32)
        new[1:n + 1] = out[:n]
        return new

    def laplacian(self, g, phi, half_width):
        return self._apply(laplacian_kernel, g, phi, True, half_width, ())

    def godunov(self, g, phi, vx, half_width):
        return self._apply(godunov_kernel, g, phi, True, half_width,
                           (float(vx), float(0.3 * vx)))

    def extrapolate(self, g, phi, vx):
        return self._apply(extrapolate_kernel, g, phi, False, 0.0, (float(vx),))


if __name__ == "__main__":
    backend = make_backend()
    if backend is not None:
        lsf.stencil_demo(backend)
