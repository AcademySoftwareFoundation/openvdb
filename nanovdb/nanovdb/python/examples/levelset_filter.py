# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""GPU LevelSetFilter on NanoVDB .nvdb files -- driver for three compute backends.

This is the runnable driver for ONE GPU level-set filter -- a
tools::LevelSetFilter-style Laplacian deform + first-order Godunov reinit +
narrow-band retrack, driven by the device VoxelBlockManager -- with the
per-voxel stencil math supplied by one of three interchangeable backends:

    python levelset_filter.py rawkernel  in.nvdb out.nvdb [outer_iters]
    python levelset_filter.py cupy       in.nvdb out.nvdb [outer_iters]
    python levelset_filter.py cutile     in.nvdb out.nvdb [outer_iters]

(no input/output files => a self-test for that backend over both input styles)

The backends live in sibling files and differ ONLY in how the dense per-voxel
stencils are computed:
  * levelset_filter_rawkernel.py -- a hand-written CUDA kernel (cupy.RawModule)
                                     that decodes the VBM + gathers in-kernel
  * levelset_filter_cupy.py       -- pure CuPy array ops over a dense gather
  * levelset_filter_cutile.py     -- NVIDIA cuTile tile kernels over the gather
Each also runs standalone: `python levelset_filter_<backend>.py` executes a
stencil-only smoke test (deform + renorm on a sphere, no file I/O / no retrack)
of just that backend's math.

Everything that is NOT backend-specific lives here and is shared by all three:
the .nvdb read (FloatGrid OR OnIndexGrid+blind-SDF, auto-detected), the
style-preserving write, the narrow-band retrack (dilateGrid -> inject ->
extrapolate -> injectPredicateToMask -> pruneGrid -> inject), the surface-radius
probe, and the outer filter loop. A backend is a small object implementing:

    class Backend:
        NAME                                    # "rawkernel" | "cupy" | "cutile"
        cp                                      # the cupy module
        setup(handle)        -> g               # {"grid", "n", ...}; per topology change
        active_coords(g)     -> (n+1, 3) int32  # value-indexed voxel coords
        laplacian(g, phi, half_width)  -> phi'
        godunov(g, phi, vx, half_width) -> phi'
        extrapolate(g, phi, vx)        -> phi'

and a module-level `make_backend()` that returns a `Backend` (or None, with a
printed reason, if that backend's requirements -- a CUDA build, a GPU, CuPy,
nvcc, or cuda.tile -- are absent). The opaque `g` context is produced by the
backend and handed back to it, so each backend stashes whatever bookkeeping it
needs (the rawkernel backend caches VBM pointers + decoded coords; the gather
backends keep it minimal). `g["grid"]` (the device OnIndex grid) and `g["n"]`
(the active-voxel count, = valueCount - 1) are the only fields the driver reads.

The SDF is carried as a value-indexed device array `phi` of length n+1: slot 0
is the background, slots 1..n the active voxels (the order the VBM decode and
`getBlindData` use). `phi[0]` is held at a sentinel so inactive neighbours are
detectable in a dense gather. Scope: first-order Godunov reinit, no advection or
alpha mask; the output's inactive interior carries +background (no signed
flood-fill).
"""
import importlib
import os
import sys
import tempfile

import numpy as np

import nanovdb


BACKENDS = ("rawkernel", "cupy", "cutile")

LOG2_BLOCK_WIDTH = 9
SENTINEL = 1.0e30          # phi[0]: marks inactive neighbours in a dense gather
NN_FACE = 6                # nanovdb::tools::morphology::NN_FACE (6-face dilation)
# 3x3x3 box-stencil spoke columns for the six faces, in -/+ x, y, z order
# (spoke = (di+1)*9 + (dj+1)*3 + (dk+1); centre = 13).
FACES = [4, 22, 10, 16, 12, 14]
BAND = 3                   # narrow-band half width, in voxels
DEFORM_ITERS = 4           # Laplacian deform sub-iterations per outer iteration
NORMALIZE_ITERS = 5        # Godunov reinit sub-iterations per outer iteration


class GatherBackend:
    """Base for backends that read each voxel's 3x3x3 neighbourhood as a dense
    (n+1, 27) array via the bound `gatherBoxStencil` -- the CuPy and cuTile
    backends. Subclasses implement only the per-voxel stencils
    (`laplacian`/`godunov`/`extrapolate`); `setup`/`active_coords` are generic."""

    def __init__(self, cp):
        self.cp = cp

    def setup(self, handle):
        cp = self.cp
        grid = handle.deviceGrid(0)
        if grid is None or grid.data_ptr() == 0:
            handle.deviceUpload(0, True)
            grid = handle.deviceGrid(0)
        n = int(nanovdb.tools.cuda.buildVoxelBlockManager(
            grid, log2_block_width=LOG2_BLOCK_WIDTH).lastOffset())
        return {"handle": handle, "grid": grid, "n": n}

    def active_coords(self, g):
        cp = self.cp
        coords = cp.empty((g["n"] + 1, 3), dtype=cp.int32)
        nanovdb.tools.cuda.activeVoxelCoords(g["grid"], coords)
        return coords


def load_backend(name):
    """Import levelset_filter_<name> and construct its Backend, or return None
    (the backend prints why) if its requirements are unavailable."""
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; choose one of {', '.join(BACKENDS)}")
    return importlib.import_module(f"levelset_filter_{name}").make_backend()


def gather_faces(cp, g, phi, background, clamp=True):
    """`gatherBoxStencil` -> (centre values c, (n, 6) face values f).

    With clamp=True the sign-clamped background BC (`copysign(background, c)`)
    replaces inactive (sentinel) spokes -- the boundary condition the deform and
    reinit stencils use. With clamp=False the raw sentinel is preserved, so the
    extrapolation stencil can tell which face neighbours are actually active."""
    n = g["n"]
    nbrs = cp.empty((n + 1, 27), dtype=cp.float32)
    nanovdb.tools.cuda.gatherBoxStencil(g["grid"], phi, nbrs)
    c = phi[1:n + 1]
    f = nbrs[1:n + 1][:, FACES]
    if clamp:
        f = cp.where(f == SENTINEL, cp.copysign(cp.float32(background), c)[:, None], f)
    return c, f


def read_to_device(backend, path, band):
    """Read a .nvdb (FloatGrid OR OnIndex+SDF) -> (g, phi, vx, half_width, style).

    A FloatGrid is baked to OnIndex+blind-SDF first (the representation the VBM
    operates on); an OnIndexGrid is read directly. `style` (the input grid type)
    is returned so the output can be written back in the same form."""
    cp = backend.cp
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
    g = backend.setup(dh)
    if sdf.shape[0] != g["n"] + 1:
        raise SystemExit(f"{path}: SDF channel length {sdf.shape[0]} != activeVoxelCount+1 "
                         f"({g['n'] + 1}); the OnIndex grid must use contiguous voxel indexing "
                         "(built with include_stats=False, include_tiles=False).")
    phi = cp.asarray(sdf)
    phi[0] = SENTINEL                              # inactive-neighbour marker
    if tmp is not None:
        os.unlink(tmp.name)
    return g, phi, vx, band * vx, gtype


def sphere_on_device(backend, radius, voxel_size=1.0, band=BAND, name="sphere"):
    """Build a level-set sphere as OnIndex+SDF on the device -> (g, phi, vx,
    half_width). Used by the backends' standalone stencil smoke tests."""
    cp = backend.cp
    io, T = nanovdb.io, nanovdb.tools
    fg = T.createLevelSetSphere(radius=radius, voxelSize=voxel_size, name=name)
    onh = T.createOnIndexGrid(fg.grid(0), channels=1,
                              include_stats=False, include_tiles=False)
    sdf = np.array(onh.grid(0).getBlindData(0), dtype=np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False); tmp.close()
    io.writeGrid(tmp.name, onh)
    dh = io.deviceReadGrid(tmp.name)
    os.unlink(tmp.name)
    g = backend.setup(dh)
    phi = cp.asarray(sdf)
    phi[0] = SENTINEL
    return g, phi, voxel_size, band * voxel_size


def rebuild(backend, g, phi, vx, half_width):
    """Narrow-band retrack: dilate -> inject old phi -> extrapolate the new ring
    (backend) -> |phi| <= halfWidth predicate -> injectPredicateToMask -> prune
    -> inject. Returns the pruned grid context + its value-indexed sidecar."""
    cp, TC = backend.cp, nanovdb.tools.cuda
    gd = backend.setup(TC.dilateGrid(g["grid"], op=NN_FACE))
    phi_d = cp.full(gd["n"] + 1, SENTINEL, dtype=cp.float32)
    TC.inject(g["grid"], gd["grid"], phi, phi_d)   # carry old phi (intersection)
    phi_d = backend.extrapolate(gd, phi_d, vx)     # fill the freshly-dilated ring
    predicate = cp.abs(phi_d) <= half_width        # phi_d[0]=SENTINEL -> False
    leaf_masks = cp.zeros(gd["n"] * 8, dtype=cp.uint64)
    TC.injectPredicateToMask(gd["grid"], predicate, leaf_masks)
    gp = backend.setup(TC.pruneGrid(gd["grid"], leaf_masks))
    phi_p = cp.full(gp["n"] + 1, SENTINEL, dtype=cp.float32)
    TC.inject(gd["grid"], gp["grid"], phi_d, phi_p)
    return gp, phi_p


def surface_radius(backend, g, phi, vx):
    """Mean world radius of the zero-crossing voxels (|phi| < dx/2)."""
    cp = backend.cp
    coords = backend.active_coords(g)
    v = phi[1:g["n"] + 1]
    near = cp.abs(v) < 0.5 * vx
    c = coords[1:g["n"] + 1][near].astype(cp.float64) * vx
    return float(cp.mean(cp.linalg.norm(c, axis=1))) if int(near.sum()) else float("nan")


def write_output(backend, g, phi, vx, path, style, band, name="filtered"):
    """Bake (coords, phi) into a host FloatGrid and write it in `style`
    (GridType.Float -> a FloatGrid; GridType.OnIndex -> an OnIndexGrid with the
    SDF in blind channel 0)."""
    cp = backend.cp
    T, io = nanovdb.tools, nanovdb.io
    coords = cp.asnumpy(backend.active_coords(g))
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


def run_filter(backend, in_path, out_path, outer_iters=6, band=BAND,
               deform_iters=DEFORM_ITERS, normalize_iters=NORMALIZE_ITERS):
    """Read -> N outer iterations (deform x k, renorm x k, retrack) -> write."""
    g, phi, vx, half_width, gtype = read_to_device(backend, in_path, band)
    print(f"read {in_path}: {gtype}, {g['n']} active voxels, voxelSize {vx:g} "
          f"[backend: {backend.NAME}]")
    r0 = r = surface_radius(backend, g, phi, vx)
    for it in range(outer_iters):
        for _ in range(deform_iters):
            phi = backend.laplacian(g, phi, half_width)
        for _ in range(normalize_iters):
            phi = backend.godunov(g, phi, vx, half_width)
        g, phi = rebuild(backend, g, phi, vx, half_width)
        r = surface_radius(backend, g, phi, vx)
        print(f"  iter {it + 1}: {g['n']:7d} active, surface radius = {r:.4f}")
    write_output(backend, g, phi, vx, out_path, gtype, band)
    style = "OnIndex+SDF" if gtype == nanovdb.GridType.OnIndex else "FloatGrid"
    print(f"wrote {out_path}: {style} (same style as input), {g['n']} active voxels "
          f"(surface radius {r0:.4f} -> {r:.4f}) [backend: {backend.NAME}]")
    return r0, r, gtype


def stencil_demo(backend, radius=20.0, deform_iters=DEFORM_ITERS,
                 normalize_iters=NORMALIZE_ITERS):
    """Run ONLY the per-voxel stencils (deform + renorm) on a sphere -- no file
    I/O and no narrow-band retrack -- as a focused check of one backend's dense
    math. Invoked by each backend file's `__main__`."""
    g, phi, vx, half_width = sphere_on_device(backend, radius)
    r0 = surface_radius(backend, g, phi, vx)
    for _ in range(deform_iters):
        phi = backend.laplacian(g, phi, half_width)
    for _ in range(normalize_iters):
        phi = backend.godunov(g, phi, vx, half_width)
    r = surface_radius(backend, g, phi, vx)
    print(f"[{backend.NAME}] stencil-only smoke test on a sphere (radius {radius:g}, "
          f"{g['n']} voxels, no retrack / no I/O): surface radius {r0:.4f} -> {r:.4f}")
    assert r < r0, "Laplacian deform should shrink the sphere"
    print(f"OK [{backend.NAME}]: deform + renorm stencils run and move the surface.")


def self_test(backend):
    """Filter sphere .nvdb files of both styles; assert the style round-trips and
    the sphere shrinks under curvature flow."""
    io, T, GT = nanovdb.io, nanovdb.tools, nanovdb.GridType
    tmps = [tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False) for _ in range(4)]
    for t in tmps:
        t.close()
    f_in, f_out, o_in, o_out = (t.name for t in tmps)

    io.writeGrid(f_in, T.createLevelSetSphere(radius=20.0, voxelSize=1.0, name="sphere"))
    print(f"self-test 1 [{backend.NAME}]: FloatGrid sphere -> filter -> FloatGrid")
    r0, r, style = run_filter(backend, f_in, f_out, outer_iters=6)
    assert style == GT.Float and io.readGrid(f_out).gridType(0) == GT.Float, \
        "output style is not FloatGrid"
    assert r < r0 - 0.05, "sphere did not shrink under curvature flow"

    sphere = T.createLevelSetSphere(radius=18.0, voxelSize=1.0, name="sphere")
    io.writeGrid(o_in, T.createOnIndexGrid(sphere.grid(0), channels=1,
                                           include_stats=False, include_tiles=False))
    print(f"self-test 2 [{backend.NAME}]: OnIndex+SDF sphere -> filter -> OnIndex+SDF")
    r0b, rb, style2 = run_filter(backend, o_in, o_out, outer_iters=4)
    ro = io.readGrid(o_out)
    assert style2 == GT.OnIndex and ro.gridType(0) == GT.OnIndex, "output style is not OnIndex"
    assert ro.grid(0).blindDataCount() >= 1, "OnIndex output has no SDF blind channel"
    assert rb < r0b - 0.05, "sphere did not shrink under curvature flow"

    print(f"OK [{backend.NAME}]: both input styles filter correctly and the output "
          "style matches the input.")
    for n in (f_in, f_out, o_in, o_out):
        os.unlink(n)


def main(argv):
    if len(argv) < 2 or argv[1] not in BACKENDS:
        print(__doc__)
        raise SystemExit(f"usage: levelset_filter.py {{{'|'.join(BACKENDS)}}} "
                         "[input.nvdb output.nvdb [outer_iterations]]   (no files = self-test)")
    backend = load_backend(argv[1])
    if backend is None:
        return
    rest = argv[2:]
    if len(rest) >= 2:
        outer = int(rest[2]) if len(rest) >= 3 else 6
        run_filter(backend, rest[0], rest[1], outer)
    elif len(rest) == 0:
        self_test(backend)
    else:
        raise SystemExit("provide BOTH input.nvdb and output.nvdb, or neither (self-test).")


if __name__ == "__main__":
    main(sys.argv)
