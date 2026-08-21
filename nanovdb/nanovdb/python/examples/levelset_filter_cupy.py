# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""levelset_filter pure-CuPy backend -- per-voxel stencils as plain CuPy arrays.

The kernel-free backend for levelset_filter.py: every stage runs as plain
**CuPy** array math on the dense `(n, 6)` face values returned by the bound
`gatherBoxStencil` -- no `cupy.RawModule`, no CUDA C++, no nvcc. Run the full
file->file filter through the driver:

    python levelset_filter.py cupy  input.nvdb output.nvdb [outer_iterations]

Running this file directly executes a stencil-only smoke test (the deform +
renorm steps on a sphere, no file I/O and no narrow-band retrack):

    python levelset_filter_cupy.py

The dense `(n, 6)` arrays this operates on are exactly the shape a tile
framework consumes -- see levelset_filter_cutile.py for the same math as cuTile
kernels, and levelset_filter_rawkernel.py for a fused hand-written CUDA kernel.
Requires only CuPy and a CUDA-capable GPU (no nvcc / NanoVDB headers).
"""
import nanovdb

import levelset_filter as lsf


def make_backend():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This backend requires a CUDA build of nanovdb and a GPU. Skipping.")
        return None
    try:
        import cupy as cp
    except ImportError:
        print("This backend requires CuPy. Skipping.")
        return None
    return Backend(cp)


class Backend(lsf.GatherBackend):
    """Per-voxel stencils as CuPy array ops; `setup`/`active_coords` come from the
    shared GatherBackend (gatherBoxStencil / activeVoxelCoords)."""

    NAME = "cupy"

    def laplacian(self, g, phi, half_width):
        """phi += (sum6 - 6 phi)/6."""
        cp = self.cp
        c, f = lsf.gather_faces(cp, g, phi, half_width, clamp=True)
        out = phi.copy()
        out[1:g["n"] + 1] = c + (f.sum(axis=1) - 6.0 * c) / 6.0
        out[0] = lsf.SENTINEL
        return out

    def godunov(self, g, phi, vx, half_width):
        """phi -= dt*S(phi)*(|grad phi| - 1)  -- first-order Godunov reinit."""
        cp = self.cp
        c, f = lsf.gather_faces(cp, g, phi, half_width, clamp=True)
        xm, xp, ym, yp, zm, zp = (f[:, i] for i in range(6))
        dt = 0.3 * vx
        s = c / cp.sqrt(c * c + vx * vx)

        def gd(dm, dp):
            return cp.where(s > 0,
                            cp.maximum(cp.maximum(dm, 0.0) ** 2, cp.minimum(dp, 0.0) ** 2),
                            cp.maximum(cp.minimum(dm, 0.0) ** 2, cp.maximum(dp, 0.0) ** 2))

        grad = cp.sqrt(gd((c - xm) / vx, (xp - c) / vx)
                       + gd((c - ym) / vx, (yp - c) / vx)
                       + gd((c - zm) / vx, (zp - c) / vx))
        out = phi.copy()
        out[1:g["n"] + 1] = c - dt * s * (grad - 1.0)
        out[0] = lsf.SENTINEL
        return out

    def extrapolate(self, g, phi, vx):
        """Fill freshly-dilated (sentinel) voxels from the nearest in-band face
        neighbour: phi = phi_nbr + sign(phi_nbr)*dx."""
        cp = self.cp
        c, f = lsf.gather_faces(cp, g, phi, 0.0, clamp=False)   # raw spokes
        known = f != lsf.SENTINEL
        best = cp.take_along_axis(
            f, cp.argmin(cp.where(known, cp.abs(f), cp.inf), axis=1)[:, None], axis=1)[:, 0]
        out = phi.copy()
        fill = (c == lsf.SENTINEL) & known.any(axis=1)
        out[1:g["n"] + 1] = cp.where(fill, best + cp.copysign(cp.float32(vx), best), c)
        out[0] = lsf.SENTINEL
        return out


if __name__ == "__main__":
    backend = make_backend()
    if backend is not None:
        lsf.stencil_demo(backend)
