# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""GPU LevelSetFilter on NanoVDB .nvdb files -- NVIDIA cuTile backend.

One of three sibling examples applying the SAME GPU level-set filter (Laplacian
deform + Godunov reinit + narrow-band retrack, driven by the VoxelBlockManager);
they differ only in how the per-voxel stencils are computed:
  * levelset_filter_rawkernel.py  -- a hand-written CUDA kernel
  * levelset_filter_cupy.py        -- pure CuPy array ops (no kernel)
  * levelset_filter_cutile.py      -- NVIDIA cuTile tile kernels  (this file)

This is levelset_filter_cupy.py with the per-voxel stencil math moved from CuPy
array ops into NVIDIA cuTile (`cuda.tile`) kernels. The whole filter runs in one
process:

  read    : nanovdb createOnIndexGrid + grid.getBlindData  (value-indexed SDF)
  deform  : gatherBoxStencil -> dense (N,27); a cuTile @_kernel does the
            Laplacian  phi += (sum6 - 6 phi)/6  over (TILE,) tiles
  renorm  : same gather; a cuTile kernel does the first-order Godunov reinit
  retrack : dilateGrid -> inject -> cuTile extrapolate kernel (fills the new
            ring) -> |phi|<=halfWidth predicate -> injectPredicateToMask ->
            pruneGrid -> inject
  write   : activeVoxelCoords -> tools.build.FloatGrid, in the input's style

So the SPARSE work (neighbour gather, coord decode, topology) stays in bound
NanoVDB ops, and the DENSE per-voxel compute is cuTile. The cuTile kernels load
six 1-D face arrays per tile (cuTile tile dims must be compile-time powers of
two, so a (TILE,6) tile of the gather columns isn't allowed); the sign-clamped
background BC is applied in CuPy before launch (inactive spokes come back as the
sentinel phi[0]); the extrapolation BC is done inside its kernel.

Validated against levelset_filter_cupy.py's results (same spheres shrink the
same amount). Run with no arguments for a self-test over both input styles.
Requires CuPy + cuda-tile (`cuda.tile`) and a CUDA-capable GPU; self-skips
otherwise.
    python levelset_filter_cutile.py [in.nvdb out.nvdb [iters]]
"""
import os
import sys
import tempfile

import numpy as np

import nanovdb

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


def _gpu_or_skip():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This example requires a CUDA build of nanovdb and a GPU. Skipping.")
        return False
    if not HAVE_CUTILE:
        print("This example requires CuPy + cuda-tile (cuda.tile). Skipping.")
        return False
    return True


TILE = 256
SENTINEL = 1.0e30
NN_FACE = 6
FACES = [4, 22, 10, 16, 12, 14]   # 3x3x3 spokes: -x,+x,-y,+y,-z,+z


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


# ----------------------------- helpers -----------------------------
def _setup(handle):
    grid = handle.deviceGrid(0)
    if grid is None or grid.data_ptr() == 0:
        handle.deviceUpload(0, True)
        grid = handle.deviceGrid(0)
    n = int(nanovdb.tools.cuda.buildVoxelBlockManager(grid, log2_block_width=9).lastOffset())
    return {"handle": handle, "grid": grid, "n": n}


def read_to_device(path, band):
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
        raise SystemExit(f"{path}: unsupported grid type {gtype}.")
    g = _setup(dh)
    if sdf.shape[0] != g["n"] + 1:
        raise SystemExit(f"{path}: SDF channel length {sdf.shape[0]} != activeVoxelCount+1 "
                         f"({g['n'] + 1}).")
    phi = cp.asarray(sdf); phi[0] = SENTINEL
    if tmp is not None:
        os.unlink(tmp.name)
    return g, phi, vx, band * vx, gtype


def _gather_faces(grid, phi, n, clamp, background):
    nbrs = cp.empty((n + 1, 27), dtype=cp.float32)
    nanovdb.tools.cuda.gatherBoxStencil(grid, phi, nbrs)
    c = phi[1:n + 1]
    cols = []
    for col in FACES:
        f = nbrs[1:n + 1, col]
        if clamp:
            f = cp.where(f == SENTINEL, cp.copysign(cp.float32(background), c), f)
        cols.append(cp.ascontiguousarray(f))
    return cp.ascontiguousarray(c), cols


def _pad(a, m):
    out = cp.zeros(m, dtype=a.dtype)
    out[:a.shape[0]] = a
    return out


def _apply(kernel, g, phi, clamp, background, extra):
    """gather -> pad -> cuTile launch -> new value-indexed sidecar."""
    n = g["n"]
    c, faces = _gather_faces(g["grid"], phi, n, clamp, background)
    m = ct.cdiv(n, TILE) * TILE
    args = [_pad(a, m) for a in (c, *faces)]
    out = cp.zeros(m, dtype=cp.float32)
    ct.launch(cp.cuda.get_current_stream(), (ct.cdiv(m, TILE), 1, 1),
              kernel, (*args, *extra, out))
    cp.cuda.runtime.deviceSynchronize()
    new = cp.full(n + 1, SENTINEL, dtype=cp.float32)
    new[1:n + 1] = out[:n]
    return new


def laplacian(g, phi, half_width):
    return _apply(laplacian_kernel, g, phi, True, half_width, ())


def godunov(g, phi, vx, half_width):
    return _apply(godunov_kernel, g, phi, True, half_width, (float(vx), float(0.3 * vx)))


def extrapolate(g, phi, vx):
    return _apply(extrapolate_kernel, g, phi, False, 0.0, (float(vx),))


def rebuild(g, phi, vx, half_width):
    TC = nanovdb.tools.cuda
    gd = _setup(TC.dilateGrid(g["grid"], op=NN_FACE))
    phi_d = cp.full(gd["n"] + 1, SENTINEL, dtype=cp.float32)
    TC.inject(g["grid"], gd["grid"], phi, phi_d)
    phi_d = extrapolate(gd, phi_d, vx)                 # cuTile fills the new ring
    predicate = cp.abs(phi_d) <= half_width
    leaf_masks = cp.zeros(gd["n"] * 8, dtype=cp.uint64)
    TC.injectPredicateToMask(gd["grid"], predicate, leaf_masks)
    gp = _setup(TC.pruneGrid(gd["grid"], leaf_masks))
    phi_p = cp.full(gp["n"] + 1, SENTINEL, dtype=cp.float32)
    TC.inject(gd["grid"], gp["grid"], phi_d, phi_p)
    return gp, phi_p


def surface_radius(g, phi, vx):
    coords = cp.empty((g["n"] + 1, 3), dtype=cp.int32)
    nanovdb.tools.cuda.activeVoxelCoords(g["grid"], coords)
    v = phi[1:g["n"] + 1]
    near = cp.abs(v) < 0.5 * vx
    c = coords[1:g["n"] + 1][near].astype(cp.float64) * vx
    return float(cp.mean(cp.linalg.norm(c, axis=1))) if int(near.sum()) else float("nan")


def write_output(g, phi, vx, path, style, band, name="filtered"):
    T, io = nanovdb.tools, nanovdb.io
    coords = cp.empty((g["n"] + 1, 3), dtype=cp.int32)
    nanovdb.tools.cuda.activeVoxelCoords(g["grid"], coords)
    coords = cp.asnumpy(coords); v = cp.asnumpy(phi)
    builder = T.build.FloatGrid(float(band * vx)); builder.setName(name); builder.setTransform(vx)
    acc = builder.getAccessor(); Coord = nanovdb.math.Coord
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
    if not _gpu_or_skip():
        return None
    g, phi, vx, half_width, gtype = read_to_device(in_path, band)
    print(f"read {in_path}: {gtype}, {g['n']} active voxels, voxelSize {vx:g}")
    r0 = surface_radius(g, phi, vx)
    for it in range(outer_iters):
        for _ in range(deform_iters):
            phi = laplacian(g, phi, half_width)
        for _ in range(normalize_iters):
            phi = godunov(g, phi, vx, half_width)
        g, phi = rebuild(g, phi, vx, half_width)
        r = surface_radius(g, phi, vx)
        print(f"  iter {it + 1}: {g['n']:7d} active, surface radius = {r:.4f}")
    write_output(g, phi, vx, out_path, gtype, band)
    style = "OnIndex+SDF" if gtype == nanovdb.GridType.OnIndex else "FloatGrid"
    print(f"wrote {out_path}: {style} (same style as input), {g['n']} active voxels "
          f"(surface radius {r0:.4f} -> {r:.4f}); per-voxel stencils ran as cuTile kernels")
    return r0, r, gtype


def self_test():
    if not _gpu_or_skip():
        return
    io, T, GT = nanovdb.io, nanovdb.tools, nanovdb.GridType
    tmps = [tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False) for _ in range(4)]
    for t in tmps:
        t.close()
    f_in, f_out, o_in, o_out = (t.name for t in tmps)
    io.writeGrid(f_in, T.createLevelSetSphere(radius=20.0, voxelSize=1.0, name="sphere"))
    print("self-test 1: FloatGrid sphere -> cuTile filter -> FloatGrid")
    r0, r, style = filter_file(f_in, f_out, outer_iters=6)
    assert style == GT.Float and io.readGrid(f_out).gridType(0) == GT.Float
    assert r < r0 - 0.05, "sphere did not shrink"
    sphere = T.createLevelSetSphere(radius=18.0, voxelSize=1.0, name="sphere")
    io.writeGrid(o_in, T.createOnIndexGrid(sphere.grid(0), channels=1,
                                           include_stats=False, include_tiles=False))
    print("self-test 2: OnIndex+SDF sphere -> cuTile filter -> OnIndex+SDF")
    r0b, rb2, style2 = filter_file(o_in, o_out, outer_iters=4)
    ro = io.readGrid(o_out)
    assert style2 == GT.OnIndex and ro.gridType(0) == GT.OnIndex and ro.grid(0).blindDataCount() >= 1
    assert rb2 < r0b - 0.05, "sphere did not shrink"
    print("OK: full file->file LevelSetFilter ran with cuTile per-voxel kernels "
          "(deform/renorm/extrapolate) + bound NanoVDB ops; output style preserved.")
    for nm in (f_in, f_out, o_in, o_out):
        os.unlink(nm)


def main(argv):
    if len(argv) >= 3:
        filter_file(argv[1], argv[2], int(argv[3]) if len(argv) >= 4 else 6)
    elif len(argv) == 1:
        self_test()
    else:
        raise SystemExit("usage: levelset_filter_cutile.py [in.nvdb out.nvdb [iters]]")


if __name__ == "__main__":
    main(sys.argv)
