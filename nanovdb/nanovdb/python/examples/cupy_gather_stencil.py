# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Kernel-free VBM stencils in CuPy via nanovdb.tools.cuda.gatherBoxStencil.

`cupy_levelset_filter.py` runs the per-voxel level-set stencils (Laplacian
deform, Godunov reinitialisation) as hand-written `cupy.RawModule` kernels that
call the device `VoxelBlockManager` decode + `computeBoxStencil`. This example
shows the *kernel-free* alternative: `gatherBoxStencil` materialises every
active voxel's 3x3x3 neighbourhood into a dense `(valueCount, 27)` array, and
the stencil arithmetic then runs as plain **CuPy** array math — no CUDA kernel,
no thread indexing, no coordinates. The same dense `(N, 27)` array is exactly
what a cuTile `@ct.kernel` would `ct.load` as tiles.

Data model: an OnIndexGrid plus a value-indexed float sidecar `phi` (entry 0 is
the background slot). `gatherBoxStencil(grid, phi, out)` fills `out[k, j]` with
the value of voxel `k`'s neighbour at 3x3x3 spoke `j` (centre `j=13`; the six
faces `j = 4, 22 (-/+x), 10, 16 (-/+y), 12, 14 (-/+z)`); inactive spokes read
`phi[0]`. We set `phi[0]` to a sentinel so inactive neighbours are detectable in
the gathered array — that lets us apply the same sign-consistent clamped
boundary condition the kernel version uses (`copysign(background, phi_centre)`),
in pure CuPy.

Scope: fixed-topology stencils only (deform + reinit). The narrow-band retrack
(`dilateGrid`/`inject`/`pruneGrid`) and file I/O live in `cupy_levelset_filter.py`.
Verified by a physical invariant: after a few Laplacian steps degrade the
signed-distance property, the Godunov reinitialisation drives |grad phi| back
toward 1 — all computed kernel-free from the gathered neighbourhood.

Requires CuPy and a CUDA-capable GPU; self-skips otherwise.
"""
import os
import tempfile

import nanovdb


SENTINEL = 1.0e30   # phi[0]: marks inactive neighbours in the gathered array
# 3x3x3 spoke columns for the six faces, in -/+ x, y, z order.
FACES = [4, 22, 10, 16, 12, 14]


def _gpu_or_skip():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This example requires a CUDA build of nanovdb and a GPU. Skipping.")
        return None
    try:
        import cupy as cp
    except ImportError:
        print("This example requires CuPy. Skipping.")
        return None
    return cp


def load_sphere(cp, radius=20.0, vx=1.0):
    """Build a sphere level set as an OnIndex device grid + value-indexed phi (CuPy)."""
    import numpy as np
    io, T = nanovdb.io, nanovdb.tools
    fg = T.createLevelSetSphere(radius=radius, voxelSize=vx, name="sphere")
    onh = T.createOnIndexGrid(fg.grid(0), channels=1,
                              include_stats=False, include_tiles=False)
    # The SDF is baked into blind channel 0 in value-index order (entry 0 is the
    # background slot) -- read it on the host with grid.getBlindData (no kernel).
    phi = cp.asarray(np.array(onh.grid(0).getBlindData(0), dtype=np.float32))
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False); tmp.close()
    io.writeGrid(tmp.name, onh)
    dh = io.deviceReadGrid(tmp.name); dh.deviceUpload(0, True)
    os.unlink(tmp.name)
    phi[0] = SENTINEL                       # inactive-neighbour marker
    return dh, dh.deviceGrid(0), int(phi.shape[0]) - 1, phi


def gather_faces(cp, grid, phi, n, background):
    """gatherBoxStencil -> the 6 face-neighbour values with the clamped-background BC."""
    nbrs = cp.empty((n + 1, 27), dtype=cp.float32)
    nanovdb.tools.cuda.gatherBoxStencil(grid, phi, nbrs)
    c = phi[1:n + 1]
    f = nbrs[1:n + 1][:, FACES]             # (n, 6): -x,+x,-y,+y,-z,+z
    # Inactive spokes came back as SENTINEL; substitute the sign-consistent
    # clamped background, exactly as the kernel's readNbr does.
    f = cp.where(f == SENTINEL, cp.copysign(cp.float32(background), c)[:, None], f)
    return c, f


def laplacian_step(cp, grid, phi, n, background):
    """phi += (sum6 - 6 phi) / 6   -- pure CuPy on the gathered neighbourhood."""
    c, f = gather_faces(cp, grid, phi, n, background)
    out = phi.copy()
    out[1:n + 1] = c + (f.sum(axis=1) - 6.0 * c) / 6.0
    out[0] = SENTINEL
    return out


def godunov_step(cp, grid, phi, n, dx, dt, background):
    """phi -= dt*S(phi)*(|grad phi| - 1)  with a first-order Godunov upwind gradient."""
    c, f = gather_faces(cp, grid, phi, n, background)
    xm, xp, ym, yp, zm, zp = (f[:, i] for i in range(6))
    s = c / cp.sqrt(c * c + dx * dx)

    def g(dm, dp):  # Rouy-Tourin upwind selection
        return cp.where(s > 0,
                        cp.maximum(cp.maximum(dm, 0.0) ** 2, cp.minimum(dp, 0.0) ** 2),
                        cp.maximum(cp.minimum(dm, 0.0) ** 2, cp.maximum(dp, 0.0) ** 2))

    grad = cp.sqrt(g((c - xm) / dx, (xp - c) / dx)
                   + g((c - ym) / dx, (yp - c) / dx)
                   + g((c - zm) / dx, (zp - c) / dx))
    out = phi.copy()
    out[1:n + 1] = c - dt * s * (grad - 1.0)
    out[0] = SENTINEL
    return out


def grad_error(cp, grid, phi, n, dx):
    """mean ||grad phi| - 1| over band-interior voxels (all six faces active)."""
    nbrs = cp.empty((n + 1, 27), dtype=cp.float32)
    nanovdb.tools.cuda.gatherBoxStencil(grid, phi, nbrs)
    f = nbrs[1:n + 1][:, FACES]
    interior = (f != SENTINEL).all(axis=1)
    xm, xp, ym, yp, zm, zp = (f[:, i] for i in range(6))
    mag = cp.sqrt(((xp - xm) / (2 * dx)) ** 2
                  + ((yp - ym) / (2 * dx)) ** 2
                  + ((zp - zm) / (2 * dx)) ** 2)
    return float(cp.mean(cp.abs(mag - 1.0)[interior])) if int(interior.sum()) else float("nan")


def main():
    cp = _gpu_or_skip()
    if cp is None:
        return
    vx, background = 1.0, 3.0
    dh, grid, n, phi = load_sphere(cp, radius=20.0, vx=vx)   # keep dh alive (owns the grid)
    print(f"sphere OnIndex level set: {n} active voxels")
    print(f"  initial      mean||grad|-1| = {grad_error(cp, grid, phi, n, vx):.4f}")

    # DEFORM: a few kernel-free Laplacian steps degrade the signed-distance property.
    for _ in range(4):
        phi = laplacian_step(cp, grid, phi, n, background)
    err_deformed = grad_error(cp, grid, phi, n, vx)
    print(f"  after deform mean||grad|-1| = {err_deformed:.4f}")

    # RENORMALISE: kernel-free Godunov reinit drives |grad phi| back toward 1.
    for _ in range(8):
        phi = godunov_step(cp, grid, phi, n, vx, 0.3 * vx, background)
    err_reinit = grad_error(cp, grid, phi, n, vx)
    print(f"  after reinit mean||grad|-1| = {err_reinit:.4f}")

    assert err_reinit < err_deformed, "kernel-free Godunov reinit did not restore |grad phi|"
    print("OK: deform + reinit ran as pure CuPy on gatherBoxStencil (no CUDA kernel); "
          "reinitialisation restored |grad phi| -> 1.")


if __name__ == "__main__":
    main()
