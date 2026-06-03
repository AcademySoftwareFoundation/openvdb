# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""GPU LevelSetFilter on NanoVDB .nvdb files -- pure-CuPy (kernel-free) backend.

One of three sibling examples applying the SAME GPU level-set filter (Laplacian
deform + Godunov reinit + narrow-band retrack, driven by the VoxelBlockManager);
they differ only in how the per-voxel stencils are computed:
  * levelset_filter_rawkernel.py  -- a hand-written CUDA kernel
  * levelset_filter_cupy.py        -- pure CuPy array ops (no kernel)  (this file)
  * levelset_filter_cutile.py      -- NVIDIA cuTile tile kernels

Where levelset_filter_rawkernel.py runs the per-voxel stencils as `cupy.RawModule`
kernels calling the device VoxelBlockManager decode + `computeBoxStencil`, here
every stage runs as plain **CuPy** array math on top of bound NanoVDB ops -- no
`RawModule`, no CUDA C++, no `nvcc`:

    python levelset_filter_cupy.py  input.nvdb  output.nvdb  [outer_iterations]

How each stage stays kernel-free:
  * READ      `createOnIndexGrid` + `grid.getBlindData(0)` -> value-indexed SDF.
  * DEFORM    `gatherBoxStencil` -> dense (N, 27) neighbour values; the Laplacian
              `phi += (sum6 - 6 phi)/6` is then a CuPy expression.
  * RENORM    same gather; a first-order Godunov reinit in CuPy.
  * RETRACK   `dilateGrid` -> `inject` -> extrapolate new voxels (CuPy on the
              gather) -> `|phi|<=halfWidth` predicate (CuPy) -> `injectPredicate-
              ToMask` -> `pruneGrid` -> `inject`.
  * WRITE     `activeVoxelCoords` -> per-voxel coords, baked into a grid with
              `tools.build.FloatGrid`; written back in the input's style
              (`FloatGrid`, or `OnIndexGrid` with the SDF in a blind channel).

The whole per-voxel surface lives in dense arrays (`gatherBoxStencil` for the
neighbourhood, `activeVoxelCoords` for positions), which is exactly the shape a
tile framework such as cuTile consumes -- the CuPy math here would map onto a
`@ct.kernel` operating on the same `(N, 27)` / `(N, 3)` arrays.

The sidecar's background slot `phi[0]` is set to a sentinel so inactive
neighbours are detectable in the gathered array; that lets the CuPy code apply
the same sign-consistent clamped boundary condition (`copysign(background,
phi_centre)`) the kernel version uses.

Scope: first-order Godunov reinitialisation, no advection or alpha mask; the
output's inactive interior carries +background (no signed flood-fill). Run with
no arguments for a self-test over both input styles. Requires only CuPy and a
CUDA-capable GPU (no nvcc / NanoVDB headers); self-skips otherwise.
"""
import os
import sys
import tempfile

import numpy as np

import nanovdb


LOG2_BLOCK_WIDTH = 9
SENTINEL = 1.0e30          # phi[0]: marks inactive neighbours in the gathered array
NN_FACE = 6               # nanovdb::tools::morphology::NN_FACE (6-face dilation)
# 3x3x3 spoke columns for the six faces, in -/+ x, y, z order (centre is col 13).
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


def _setup(cp, handle):
    """DeviceGridHandle -> {handle, grid, n}; n = active-voxel count (= valueCount-1)."""
    grid = handle.deviceGrid(0)
    if grid is None or grid.data_ptr() == 0:
        handle.deviceUpload(0, True)
        grid = handle.deviceGrid(0)
    n = int(nanovdb.tools.cuda.buildVoxelBlockManager(
        grid, log2_block_width=LOG2_BLOCK_WIDTH).lastOffset())
    return {"handle": handle, "grid": grid, "n": n}


def read_to_device(cp, path, band):
    """Read a .nvdb (FloatGrid OR OnIndex+SDF) -> (g, phi, vx, half_width, style)."""
    io, T = nanovdb.io, nanovdb.tools
    host = io.readGrid(path)
    gtype = host.gridType(0)
    vx = float(host.grid(0).voxelSize()[0])
    tmp = None
    if gtype == nanovdb.GridType.Float:
        onh = T.createOnIndexGrid(host.grid(0), channels=1,
                                  include_stats=False, include_tiles=False)
        sdf = np.array(onh.grid(0).getBlindData(0), dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False); tmp.close()
        io.writeGrid(tmp.name, onh)
        dh = io.deviceReadGrid(tmp.name)
    elif gtype == nanovdb.GridType.OnIndex:
        if host.grid(0).blindDataCount() == 0:
            raise SystemExit(f"{path}: OnIndex grid has no blind-data SDF channel.")
        sdf = np.array(host.grid(0).getBlindData(0), dtype=np.float32)
        dh = io.deviceReadGrid(path)
    else:
        raise SystemExit(f"{path}: unsupported grid type {gtype} (expected Float or OnIndex).")
    g = _setup(cp, dh)
    if sdf.shape[0] != g["n"] + 1:
        raise SystemExit(f"{path}: SDF channel length {sdf.shape[0]} != activeVoxelCount+1 "
                         f"({g['n'] + 1}); the OnIndex grid must use contiguous voxel indexing "
                         "(built with include_stats=False, include_tiles=False).")
    phi = cp.asarray(sdf)
    phi[0] = SENTINEL                              # inactive-neighbour marker
    if tmp is not None:
        os.unlink(tmp.name)
    return g, phi, vx, band * vx, gtype


def _gather_faces(cp, g, phi, background):
    """gatherBoxStencil -> the 6 face values with the clamped-background BC."""
    n = g["n"]
    nbrs = cp.empty((n + 1, 27), dtype=cp.float32)
    nanovdb.tools.cuda.gatherBoxStencil(g["grid"], phi, nbrs)
    c = phi[1:n + 1]
    f = nbrs[1:n + 1][:, FACES]
    f = cp.where(f == SENTINEL, cp.copysign(cp.float32(background), c)[:, None], f)
    return c, f


def laplacian_step(cp, g, phi, background):
    """phi += (sum6 - 6 phi)/6  -- pure CuPy on the gathered neighbourhood."""
    c, f = _gather_faces(cp, g, phi, background)
    out = phi.copy()
    out[1:g["n"] + 1] = c + (f.sum(axis=1) - 6.0 * c) / 6.0
    out[0] = SENTINEL
    return out


def godunov_step(cp, g, phi, dx, dt, background):
    """phi -= dt*S(phi)*(|grad phi| - 1)  -- first-order Godunov, pure CuPy."""
    c, f = _gather_faces(cp, g, phi, background)
    xm, xp, ym, yp, zm, zp = (f[:, i] for i in range(6))
    s = c / cp.sqrt(c * c + dx * dx)

    def gd(dm, dp):
        return cp.where(s > 0,
                        cp.maximum(cp.maximum(dm, 0.0) ** 2, cp.minimum(dp, 0.0) ** 2),
                        cp.maximum(cp.minimum(dm, 0.0) ** 2, cp.maximum(dp, 0.0) ** 2))

    grad = cp.sqrt(gd((c - xm) / dx, (xp - c) / dx)
                   + gd((c - ym) / dx, (yp - c) / dx)
                   + gd((c - zm) / dx, (zp - c) / dx))
    out = phi.copy()
    out[1:g["n"] + 1] = c - dt * s * (grad - 1.0)
    out[0] = SENTINEL
    return out


def _extrapolate(cp, g, phi, dx):
    """Fill freshly-dilated (sentinel) voxels from the nearest in-band face
    neighbour: phi = phi_nbr + sign(phi_nbr)*dx -- pure CuPy on the gather."""
    n = g["n"]
    nbrs = cp.empty((n + 1, 27), dtype=cp.float32)
    nanovdb.tools.cuda.gatherBoxStencil(g["grid"], phi, nbrs)
    c = phi[1:n + 1]
    f = nbrs[1:n + 1][:, FACES]                    # raw: inactive spokes are SENTINEL
    known = f != SENTINEL
    best = cp.take_along_axis(
        f, cp.argmin(cp.where(known, cp.abs(f), cp.inf), axis=1)[:, None], axis=1)[:, 0]
    out = phi.copy()
    fill = (c == SENTINEL) & known.any(axis=1)
    out[1:n + 1] = cp.where(fill, best + cp.copysign(cp.float32(dx), best), c)
    out[0] = SENTINEL
    return out


def rebuild(cp, g, phi, vx, half_width):
    """Narrow-band retrack: dilate -> inject -> extrapolate -> prune -> inject."""
    TC = nanovdb.tools.cuda
    gd = _setup(cp, TC.dilateGrid(g["grid"], op=NN_FACE))
    phi_d = cp.full(gd["n"] + 1, SENTINEL, dtype=cp.float32)
    TC.inject(g["grid"], gd["grid"], phi, phi_d)   # carry old phi (intersection)
    phi_d = _extrapolate(cp, gd, phi_d, vx)        # fill the new ring (CuPy)
    predicate = cp.abs(phi_d) <= half_width        # phi_d[0]=SENTINEL -> False
    leaf_masks = cp.zeros(gd["n"] * 8, dtype=cp.uint64)
    TC.injectPredicateToMask(gd["grid"], predicate, leaf_masks)
    gp = _setup(cp, TC.pruneGrid(gd["grid"], leaf_masks))
    phi_p = cp.full(gp["n"] + 1, SENTINEL, dtype=cp.float32)
    TC.inject(gd["grid"], gp["grid"], phi_d, phi_p)
    return gp, phi_p


def surface_radius(cp, g, phi, vx):
    """Mean world radius of zero-crossing voxels (|phi| < dx/2), coords via activeVoxelCoords."""
    coords = cp.empty((g["n"] + 1, 3), dtype=cp.int32)
    nanovdb.tools.cuda.activeVoxelCoords(g["grid"], coords)
    v = phi[1:g["n"] + 1]
    near = cp.abs(v) < 0.5 * vx
    c = coords[1:g["n"] + 1][near].astype(cp.float64) * vx
    return float(cp.mean(cp.linalg.norm(c, axis=1))) if int(near.sum()) else float("nan")


def write_output(cp, g, phi, vx, path, style, band, name="filtered"):
    """Bake (coords, phi) into a host FloatGrid; write in `style` (Float or OnIndex+SDF)."""
    T, io = nanovdb.tools, nanovdb.io
    coords = cp.empty((g["n"] + 1, 3), dtype=cp.int32)
    nanovdb.tools.cuda.activeVoxelCoords(g["grid"], coords)   # kernel-free coord decode
    coords = cp.asnumpy(coords)
    v = cp.asnumpy(phi)
    builder = T.build.FloatGrid(float(band * vx))
    builder.setName(name)
    builder.setTransform(vx)
    acc = builder.getAccessor()
    Coord = nanovdb.math.Coord
    for k in range(1, g["n"] + 1):
        i, j, kk = coords[k]
        acc.setValue(Coord(int(i), int(j), int(kk)), float(v[k]))
    fh = builder.to_nanovdb()
    if style == nanovdb.GridType.OnIndex:
        io.writeGrid(path, T.createOnIndexGrid(fh.grid(0), channels=1,
                                               include_stats=False, include_tiles=False))
    else:
        io.writeGrid(path, fh)


def filter_file(in_path, out_path, outer_iters=6, band=3, deform_iters=4, normalize_iters=5):
    cp = _gpu_or_skip()
    if cp is None:
        return None
    g, phi, vx, half_width, gtype = read_to_device(cp, in_path, band)
    print(f"read {in_path}: {gtype}, {g['n']} active voxels, voxelSize {vx:g}")
    r0 = surface_radius(cp, g, phi, vx)
    for it in range(outer_iters):
        for _ in range(deform_iters):
            phi = laplacian_step(cp, g, phi, half_width)
        for _ in range(normalize_iters):
            phi = godunov_step(cp, g, phi, vx, 0.3 * vx, half_width)
        g, phi = rebuild(cp, g, phi, vx, half_width)
        r = surface_radius(cp, g, phi, vx)
        print(f"  iter {it + 1}: {g['n']:7d} active, surface radius = {r:.4f}")
    write_output(cp, g, phi, vx, out_path, gtype, band)
    style = "OnIndex+SDF" if gtype == nanovdb.GridType.OnIndex else "FloatGrid"
    print(f"wrote {out_path}: {style} (same style as input), {g['n']} active voxels "
          f"(surface radius {r0:.4f} -> {r:.4f}); no CUDA kernel used")
    return r0, r, gtype


def self_test():
    cp = _gpu_or_skip()
    if cp is None:
        return
    io, T, GT = nanovdb.io, nanovdb.tools, nanovdb.GridType
    tmps = [tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False) for _ in range(4)]
    for t in tmps:
        t.close()
    f_in, f_out, o_in, o_out = (t.name for t in tmps)

    io.writeGrid(f_in, T.createLevelSetSphere(radius=20.0, voxelSize=1.0, name="sphere"))
    print("self-test 1: FloatGrid sphere -> kernel-free filter -> FloatGrid")
    r0, r, style = filter_file(f_in, f_out, outer_iters=6)
    rb = io.readGrid(f_out)
    assert style == GT.Float and rb.gridType(0) == GT.Float, "output style is not FloatGrid"
    assert r < r0 - 0.05, "sphere did not shrink under curvature flow"

    sphere = T.createLevelSetSphere(radius=18.0, voxelSize=1.0, name="sphere")
    io.writeGrid(o_in, T.createOnIndexGrid(sphere.grid(0), channels=1,
                                           include_stats=False, include_tiles=False))
    print("self-test 2: OnIndex+SDF sphere -> kernel-free filter -> OnIndex+SDF")
    r0b, rb2, style2 = filter_file(o_in, o_out, outer_iters=4)
    ro = io.readGrid(o_out)
    assert style2 == GT.OnIndex and ro.gridType(0) == GT.OnIndex, "output style is not OnIndex"
    assert ro.grid(0).blindDataCount() >= 1, "OnIndex output has no SDF blind channel"
    assert rb2 < r0b - 0.05, "sphere did not shrink under curvature flow"

    print("OK: full file->file LevelSetFilter ran entirely in CuPy on bound ops "
          "(gatherBoxStencil / activeVoxelCoords / inject / dilateGrid / pruneGrid) -- "
          "no CUDA kernel; output style matches input.")
    for n in (f_in, f_out, o_in, o_out):
        os.unlink(n)


def main(argv):
    if len(argv) >= 3:
        outer = int(argv[3]) if len(argv) >= 4 else 6
        filter_file(argv[1], argv[2], outer)
    elif len(argv) == 1:
        self_test()
    else:
        print(__doc__)
        raise SystemExit("usage: levelset_filter_cupy.py input.nvdb output.nvdb "
                         "[outer_iterations]   (no args = self-test)")


if __name__ == "__main__":
    main(sys.argv)
