# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""GPU LevelSetFilter on NanoVDB .nvdb files, driven by the VoxelBlockManager.

Reads a .nvdb file, runs N iterations of the GPU LevelSetFilter loop (the
diffusion + renormalisation + narrow-band retrack that OpenVDB's
tools::LevelSetFilter + LevelSetTracker perform), and writes the result to
another .nvdb file:

    python cupy_levelset_filter.py  input.nvdb  output.nvdb  [outer_iterations]

The input grid may be EITHER form (the script detects which):
  * a FloatGrid level set (per-voxel float SDF), or
  * an OnIndexGrid whose float SDF is stored in blind-data channel 0.

Both are normalised to "(OnIndex topology on the device) + (float SDF sidecar)",
which is the representation the VoxelBlockManager operates on:
  * FloatGrid  -> tools.createOnIndexGrid(fg, channels=1) bakes the SDF into a
                  blind channel; write to a temp .nvdb; io.deviceReadGrid it.
  * OnIndexGrid -> io.deviceReadGrid directly.
The SDF sidecar is read on the host with grid.getBlindData(0) (value-index
order: [0] is the background slot, 1..N the active voxels -- the same order the
VBM decode uses), then uploaded to the device.

The OUTPUT is written in the SAME style as the input: a FloatGrid input yields a
FloatGrid .nvdb, an OnIndex+SDF input yields an OnIndexGrid .nvdb with the SDF in
blind channel 0. The result is baked on the host (tools.build.FloatGrid ->
to_nanovdb), optionally converted back to OnIndex via createOnIndexGrid, then
io.writeGrid. (Writing the result as a device-built grid via indexToGrid is
avoided: in testing it dropped the high-value-index voxels for these
stats/tiles-free index grids.)

Each filter iteration runs three stages:
  1. DEFORM      Laplacian flow            phi += (sum6 - 6 phi)/6      (VBM stencil)
  2. RENORMALIZE Godunov reinit            phi -= dt*S(phi)*(|grad|-1)  (VBM stencil)
  3. REBUILD     dilateGrid -> inject -> extrapolate -> injectPredicateToMask
                 -> pruneGrid -> inject   (native bound ops + one extrapolate kernel)

Scope / limitations: first-order Godunov reinitialisation (not higher-order
WENO), no advection and no alpha mask. The prune keeps |phi| <= band*voxelSize,
so the active band tracks the surface, but the output's inactive interior
carries +background (there is no signed flood-fill).

Run without arguments for a self-test: it builds sphere .nvdb files in both
forms (FloatGrid and OnIndex+SDF), filters each, and asserts that the output
style matches the input and the sphere shrinks under curvature flow.

Requires CuPy, a CUDA-capable GPU, and nvcc (CuPy honours the NVCC env var). In a
dev/source tree, set NANOVDB_INCLUDE to the dir containing nanovdb/NanoVDB.h.
"""
import os
import sys
import tempfile

import numpy as np

import nanovdb


LOG2_BLOCK_WIDTH = 9
BLOCK_WIDTH = 1 << LOG2_BLOCK_WIDTH
NN_FACE = 6           # nanovdb::tools::morphology::NN_FACE (6-face dilation)
SENTINEL = 1.0e30     # "value not yet known" marker for freshly-dilated voxels
SDF_BLIND_CHANNEL = 0  # blind-data channel holding the float SDF on OnIndex input

# computeBoxStencil spoke ids: spoke = (di+1)*9 + (dj+1)*3 + (dk+1).
#   centre=13;  +x=22 -x=4;  +y=16 -y=10;  +z=14 -z=12
KERNEL_SRC = r"""
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>

using namespace nanovdb;
using VBM = nanovdb::tools::cuda::VoxelBlockManager<9>;   // BlockWidth = 512
static constexpr int BLOCK_WIDTH     = 512;
static constexpr int JUMP_MAP_LENGTH = BLOCK_WIDTH / 64;  // = 8
static constexpr float SENTINEL      = 1.0e30f;

__device__ static unsigned long long
decodeBlock(const NanoGrid<ValueOnIndex>* grid,
            const uint32_t* firstLeafID, const uint64_t* jumpMap,
            uint64_t firstOffset,
            uint32_t* smem_leafIndex, uint16_t* smem_voxelOffset)
{
    const int blockID = blockIdx.x;
    const uint64_t blockFirstOffset =
        firstOffset + (uint64_t)blockID * BLOCK_WIDTH;
    VBM::decodeInverseMaps<ValueOnIndex>(
        grid, firstLeafID[blockID],
        &jumpMap[(uint64_t)blockID * JUMP_MAP_LENGTH],
        blockFirstOffset, smem_leafIndex, smem_voxelOffset);
    return blockFirstOffset;   // decodeInverseMaps ends in __syncthreads()
}

__device__ static inline float
readNbr(const float* v, uint64_t spoke, float vc, float background)
{
    return (spoke == 0) ? copysignf(background, vc) : v[spoke];
}

extern "C" __global__
void decode_coords(const NanoGrid<ValueOnIndex>* grid,
                   const uint32_t* firstLeafID, const uint64_t* jumpMap,
                   uint64_t firstOffset, int* coords)
{
    __shared__ uint32_t smem_leafIndex[BLOCK_WIDTH];
    __shared__ uint16_t smem_voxelOffset[BLOCK_WIDTH];
    decodeBlock(grid, firstLeafID, jumpMap, firstOffset,
                smem_leafIndex, smem_voxelOffset);
    const int tID = threadIdx.x;
    if (smem_leafIndex[tID] == VBM::UnusedLeafIndex) return;
    const auto& leaf = grid->tree().getFirstNode<0>()[smem_leafIndex[tID]];
    const uint16_t off = smem_voxelOffset[tID];
    const Coord c = leaf.offsetToGlobalCoord(off);
    const uint64_t idx = leaf.getValue(off);
    coords[idx * 3 + 0] = c[0];
    coords[idx * 3 + 1] = c[1];
    coords[idx * 3 + 2] = c[2];
}

// One Laplacian-flow iteration (OpenVDB LevelSetFilter::laplacianImpl).
extern "C" __global__
void laplacian_step(const NanoGrid<ValueOnIndex>* grid,
                    const uint32_t* firstLeafID, const uint64_t* jumpMap,
                    uint64_t firstOffset, float background,
                    const float* vin, float* vout)
{
    __shared__ uint32_t smem_leafIndex[BLOCK_WIDTH];
    __shared__ uint16_t smem_voxelOffset[BLOCK_WIDTH];
    decodeBlock(grid, firstLeafID, jumpMap, firstOffset,
                smem_leafIndex, smem_voxelOffset);
    const int tID = threadIdx.x;
    if (smem_leafIndex[tID] == VBM::UnusedLeafIndex) return;

    uint64_t st[27];
    VBM::computeBoxStencil<ValueOnIndex>(grid, smem_leafIndex, smem_voxelOffset, st);

    const uint64_t c = st[13];
    const float vc = vin[c];
    const float sum6 = readNbr(vin, st[22], vc, background)
                     + readNbr(vin, st[4],  vc, background)
                     + readNbr(vin, st[16], vc, background)
                     + readNbr(vin, st[10], vc, background)
                     + readNbr(vin, st[14], vc, background)
                     + readNbr(vin, st[12], vc, background);
    vout[c] = vc + (sum6 - 6.0f * vc) / 6.0f;
}

// One Godunov reinitialisation iteration: phi -= dt*S(phi)*(|grad phi| - 1).
extern "C" __global__
void godunov_step(const NanoGrid<ValueOnIndex>* grid,
                  const uint32_t* firstLeafID, const uint64_t* jumpMap,
                  uint64_t firstOffset, float dx, float dt, float background,
                  const float* vin, float* vout)
{
    __shared__ uint32_t smem_leafIndex[BLOCK_WIDTH];
    __shared__ uint16_t smem_voxelOffset[BLOCK_WIDTH];
    decodeBlock(grid, firstLeafID, jumpMap, firstOffset,
                smem_leafIndex, smem_voxelOffset);
    const int tID = threadIdx.x;
    if (smem_leafIndex[tID] == VBM::UnusedLeafIndex) return;

    uint64_t st[27];
    VBM::computeBoxStencil<ValueOnIndex>(grid, smem_leafIndex, smem_voxelOffset, st);

    const uint64_t c = st[13];
    const float vc = vin[c];
    const float xm = readNbr(vin, st[4],  vc, background);
    const float xp = readNbr(vin, st[22], vc, background);
    const float ym = readNbr(vin, st[10], vc, background);
    const float yp = readNbr(vin, st[16], vc, background);
    const float zm = readNbr(vin, st[12], vc, background);
    const float zp = readNbr(vin, st[14], vc, background);

    const float Dxm = (vc - xm) / dx, Dxp = (xp - vc) / dx;
    const float Dym = (vc - ym) / dx, Dyp = (yp - vc) / dx;
    const float Dzm = (vc - zm) / dx, Dzp = (zp - vc) / dx;

    const float s = vc / sqrtf(vc * vc + dx * dx);   // smoothed sign
    float gx, gy, gz;
    if (s > 0.0f) {
        gx = fmaxf(powf(fmaxf(Dxm, 0.0f), 2), powf(fminf(Dxp, 0.0f), 2));
        gy = fmaxf(powf(fmaxf(Dym, 0.0f), 2), powf(fminf(Dyp, 0.0f), 2));
        gz = fmaxf(powf(fmaxf(Dzm, 0.0f), 2), powf(fminf(Dzp, 0.0f), 2));
    } else {
        gx = fmaxf(powf(fminf(Dxm, 0.0f), 2), powf(fmaxf(Dxp, 0.0f), 2));
        gy = fmaxf(powf(fminf(Dym, 0.0f), 2), powf(fmaxf(Dyp, 0.0f), 2));
        gz = fmaxf(powf(fminf(Dzm, 0.0f), 2), powf(fmaxf(Dzp, 0.0f), 2));
    }
    const float grad = sqrtf(gx + gy + gz);
    vout[c] = vc - dt * s * (grad - 1.0f);
}

// Fill freshly-dilated (sentinel) voxels by SDF extrapolation from their
// nearest in-band face neighbour: phi = phi_nbr + sign(phi_nbr)*dx.
extern "C" __global__
void extrapolate(const NanoGrid<ValueOnIndex>* grid,
                 const uint32_t* firstLeafID, const uint64_t* jumpMap,
                 uint64_t firstOffset, float dx, const float* vin, float* vout)
{
    __shared__ uint32_t smem_leafIndex[BLOCK_WIDTH];
    __shared__ uint16_t smem_voxelOffset[BLOCK_WIDTH];
    decodeBlock(grid, firstLeafID, jumpMap, firstOffset,
                smem_leafIndex, smem_voxelOffset);
    const int tID = threadIdx.x;
    if (smem_leafIndex[tID] == VBM::UnusedLeafIndex) return;

    uint64_t st[27];
    VBM::computeBoxStencil<ValueOnIndex>(grid, smem_leafIndex, smem_voxelOffset, st);

    const uint64_t c = st[13];
    const float vc = vin[c];
    if (vc != SENTINEL) { vout[c] = vc; return; }      // already known
    const int faces[6] = {22, 4, 16, 10, 14, 12};
    float best = SENTINEL;
    for (int f = 0; f < 6; f++) {
        const uint64_t ni = st[faces[f]];
        if (ni != 0) {                                 // active neighbour
            const float v = vin[ni];
            if (v != SENTINEL && fabsf(v) < fabsf(best)) best = v;
        }
    }
    vout[c] = (best != SENTINEL) ? best + copysignf(dx, best) : vc;
}
"""


def _include_options():
    opts = list(nanovdb.cuda.compile_options("-std=c++17"))
    inc = opts[0][2:]
    if not os.path.isdir(inc):
        env = os.environ.get("NANOVDB_INCLUDE")
        if env and os.path.isdir(os.path.join(env, "nanovdb")):
            opts[0] = f"-I{env}"
        else:
            print(f"NanoVDB headers not found at {inc!r}. In a source tree set "
                  "NANOVDB_INCLUDE to the dir containing nanovdb/NanoVDB.h.")
            return None
    return tuple(opts)


class Filter:
    """Holds the compiled kernels + bound ops and runs the level-set filter."""

    def __init__(self, cp, band=3, deform_iters=4, normalize_iters=5):
        self.cp = cp
        self.tc = nanovdb.tools.cuda
        self.band = band
        self.deform_iters = deform_iters
        self.normalize_iters = normalize_iters
        options = _include_options()
        if options is None:
            raise SystemExit(1)
        m = cp.RawModule(code=KERNEL_SRC, backend="nvcc", options=options)
        self.k_decode = m.get_function("decode_coords")
        self.k_laplacian = m.get_function("laplacian_step")
        self.k_godunov = m.get_function("godunov_step")
        self.k_extrapolate = m.get_function("extrapolate")

    # ---- device OnIndex grid + VBM bookkeeping -------------------------------
    def setup(self, handle):
        cp = self.cp
        grid = handle.deviceGrid(0)
        if grid is None or grid.data_ptr() == 0:
            handle.deviceUpload(0, True)
            grid = handle.deviceGrid(0)
        vbm = self.tc.buildVoxelBlockManager(grid, log2_block_width=LOG2_BLOCK_WIDTH)
        n, bc = int(vbm.lastOffset()), int(vbm.blockCount())
        coords = cp.zeros((n + 1, 3), dtype=cp.int32)
        self.k_decode((bc,), (BLOCK_WIDTH,),
                      (grid.data_ptr(), vbm.first_leaf_id_ptr(), vbm.jump_map_ptr(),
                       np.uint64(vbm.firstOffset()), coords))
        cp.cuda.runtime.deviceSynchronize()
        return dict(handle=handle, grid=grid, vbm=vbm, n=n, bc=bc,
                    fo=np.uint64(vbm.firstOffset()),
                    fid=vbm.first_leaf_id_ptr(), jmp=vbm.jump_map_ptr(),
                    gptr=grid.data_ptr(), coords=coords)

    def _vbm(self, g):
        return (g["gptr"], g["fid"], g["jmp"], g["fo"])

    # ---- one full LevelSetFilter iteration -----------------------------------
    def step(self, g, vals, vx, half_width):
        cp = self.cp
        bg = np.float32(half_width)
        buf = cp.empty_like(vals)
        # 1. DEFORM: Laplacian flow.
        for _ in range(self.deform_iters):
            self.k_laplacian((g["bc"],), (BLOCK_WIDTH,), (*self._vbm(g), bg, vals, buf))
            vals, buf = buf, vals
        # 2. RENORMALIZE: Godunov reinitialisation.
        for _ in range(self.normalize_iters):
            self.k_godunov((g["bc"],), (BLOCK_WIDTH,),
                           (*self._vbm(g), np.float32(vx), np.float32(0.3 * vx), bg, vals, buf))
            vals, buf = buf, vals
        cp.cuda.runtime.deviceSynchronize()
        # 3. REBUILD BAND: dilateGrid -> inject -> extrapolate -> prune -> inject.
        gd = self.setup(self.tc.dilateGrid(g["grid"], op=NN_FACE))
        vals_d = cp.full(gd["n"] + 1, SENTINEL, dtype=cp.float32)
        self.tc.inject(g["grid"], gd["grid"], vals, vals_d)
        ebuf = cp.empty_like(vals_d)
        self.k_extrapolate((gd["bc"],), (BLOCK_WIDTH,), (*self._vbm(gd), np.float32(vx), vals_d, ebuf))
        vals_d = ebuf
        predicate = cp.abs(vals_d) <= half_width
        leaf_masks = cp.zeros(gd["n"] * 8, dtype=cp.uint64)   # activeVoxelCount*8
        self.tc.injectPredicateToMask(gd["grid"], predicate, leaf_masks)
        gp = self.setup(self.tc.pruneGrid(gd["grid"], leaf_masks))
        vals_p = cp.full(gp["n"] + 1, half_width, dtype=cp.float32)
        self.tc.inject(gd["grid"], gp["grid"], vals_d, vals_p)
        return gp, vals_p

    def surface_radius(self, g, vals, vx):
        cp = self.cp
        v = vals[1:g["n"] + 1]
        near = cp.abs(v) < 0.5 * vx
        c = g["coords"][1:g["n"] + 1][near].astype(cp.float64) * vx
        return float(cp.mean(cp.linalg.norm(c, axis=1))) if int(near.sum()) else float("nan")


def read_to_device(flt, path):
    """Read a .nvdb (FloatGrid OR OnIndex+SDF) -> (device-grid dict, sidecar, vx, half_width, style)."""
    cp = flt.cp
    io, T = nanovdb.io, nanovdb.tools
    host = io.readGrid(path)
    gtype = host.gridType(0)
    vx = float(host.grid(0).voxelSize()[0])
    half_width = flt.band * vx
    tmp = None
    if gtype == nanovdb.GridType.Float:
        # Bake the per-voxel SDF into an OnIndex blind channel; read it on the
        # host via grid.getBlindData (value-index order: [0]=background, 1..N).
        idx_host = T.createOnIndexGrid(host.grid(0), channels=1,
                                       include_stats=False, include_tiles=False)
        sdf = np.array(idx_host.grid(0).getBlindData(SDF_BLIND_CHANNEL), dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False); tmp.close()
        io.writeGrid(tmp.name, idx_host)
        dev = io.deviceReadGrid(tmp.name)
    elif gtype == nanovdb.GridType.OnIndex:
        if host.grid(0).blindDataCount() == 0:
            raise SystemExit(f"{path}: OnIndex grid has no blind-data SDF channel.")
        sdf = np.array(host.grid(0).getBlindData(SDF_BLIND_CHANNEL), dtype=np.float32)
        dev = io.deviceReadGrid(path)
    else:
        raise SystemExit(f"{path}: unsupported grid type {gtype} "
                         "(expected Float or OnIndex).")
    dev.deviceUpload(0, True)
    g = flt.setup(dev)
    if sdf.shape[0] != g["n"] + 1:
        raise SystemExit(f"{path}: SDF channel length {sdf.shape[0]} != activeVoxelCount+1 "
                         f"({g['n'] + 1}); the OnIndex grid must use contiguous voxel "
                         "indexing (built with include_stats=False, include_tiles=False).")
    vals = cp.asarray(sdf)   # value-indexed sidecar; vals[0] is the background slot
    if tmp is not None:
        os.unlink(tmp.name)
    return g, vals, vx, half_width, gtype


def write_output(flt, g, vals, vx, path, style, name="filtered"):
    """Bake the final (coords, SDF) into a host FloatGrid; write it in `style`.

    style == GridType.Float    -> a FloatGrid .nvdb
    style == GridType.OnIndex  -> an OnIndexGrid .nvdb with the SDF in blind channel 0
    """
    cp = flt.cp
    T, io = nanovdb.tools, nanovdb.io
    coords = cp.asnumpy(g["coords"])
    v = cp.asnumpy(vals)
    builder = T.build.FloatGrid(float(flt.band * vx))
    builder.setName(name)
    builder.setTransform(vx)
    acc = builder.getAccessor()
    Coord = nanovdb.math.Coord
    for k in range(1, g["n"] + 1):
        i, j, kk = coords[k]
        acc.setValue(Coord(int(i), int(j), int(kk)), float(v[k]))
    fh = builder.to_nanovdb()
    if style == nanovdb.GridType.OnIndex:
        idx_out = T.createOnIndexGrid(fh.grid(0), channels=1,
                                      include_stats=False, include_tiles=False)
        io.writeGrid(path, idx_out)
    else:
        io.writeGrid(path, fh)


def _gpu_or_skip():
    """Return the cupy module, or None (with a printed reason) if GPU filtering
    is unavailable -- lets the example self-skip cleanly like the others."""
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This example requires a CUDA build of nanovdb and a GPU. Skipping.")
        return None
    try:
        import cupy as cp
    except ImportError:
        print("This example requires CuPy (plus nvcc on PATH or $NVCC). Skipping.")
        return None
    return cp


def filter_file(in_path, out_path, outer_iters=6):
    cp = _gpu_or_skip()
    if cp is None:
        return None
    flt = Filter(cp)
    g, vals, vx, half_width, gtype = read_to_device(flt, in_path)
    print(f"read {in_path}: {gtype}, {g['n']} active voxels, voxelSize {vx:g}")
    r0 = flt.surface_radius(g, vals, vx)
    for it in range(outer_iters):
        g, vals = flt.step(g, vals, vx, half_width)
        r = flt.surface_radius(g, vals, vx)
        print(f"  iter {it + 1}: {g['n']:7d} active, surface radius = {r:.4f}")
    write_output(flt, g, vals, vx, out_path, gtype)
    style = "OnIndex+SDF" if gtype == nanovdb.GridType.OnIndex else "FloatGrid"
    print(f"wrote {out_path}: {style} (same style as input), {g['n']} active voxels "
          f"(surface radius {r0:.4f} -> {r:.4f})")
    return r0, r, gtype


def self_test():
    """No-args run: build sphere .nvdb files of both styles, filter, assert invariants."""
    if _gpu_or_skip() is None:
        return
    io, T, GT = nanovdb.io, nanovdb.tools, nanovdb.GridType
    tmps = [tempfile.NamedTemporaryFile(suffix=".nvdb", delete=False) for _ in range(4)]
    for t in tmps:
        t.close()
    f_in, f_out, o_in, o_out = (t.name for t in tmps)

    # 1. FloatGrid in -> FloatGrid out.
    io.writeGrid(f_in, T.createLevelSetSphere(radius=20.0, voxelSize=1.0, name="sphere"))
    print("self-test 1: FloatGrid sphere -> filter -> FloatGrid")
    r0, r, style = filter_file(f_in, f_out, outer_iters=6)
    rb = io.readGrid(f_out)
    assert style == GT.Float and rb.gridType(0) == GT.Float, "output style is not FloatGrid"
    assert r < r0 - 0.05, "sphere did not shrink under curvature flow"

    # 2. OnIndex+SDF in -> OnIndex+SDF out (style preserved).
    sphere = T.createLevelSetSphere(radius=18.0, voxelSize=1.0, name="sphere")
    io.writeGrid(o_in, T.createOnIndexGrid(sphere.grid(0), channels=1,
                                           include_stats=False, include_tiles=False))
    print("self-test 2: OnIndex+SDF sphere -> filter -> OnIndex+SDF")
    r0b, rb2, style2 = filter_file(o_in, o_out, outer_iters=4)
    ro = io.readGrid(o_out)
    assert style2 == GT.OnIndex and ro.gridType(0) == GT.OnIndex, "output style is not OnIndex"
    assert ro.grid(0).blindDataCount() >= 1, "OnIndex output has no SDF blind channel"
    assert rb2 < r0b - 0.05, "sphere did not shrink under curvature flow"

    print("OK: both styles filter correctly and the output style matches the input.")
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
        raise SystemExit("usage: cupy_levelset_filter.py input.nvdb output.nvdb "
                         "[outer_iterations]   (no args = self-test)")


if __name__ == "__main__":
    main(sys.argv)
