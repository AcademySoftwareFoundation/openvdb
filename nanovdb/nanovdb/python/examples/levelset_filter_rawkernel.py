# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""levelset_filter compiled-CUDA-kernel backend -- one fused cupy.RawModule.

The hand-written-CUDA-kernel backend for levelset_filter.py: each stencil is a
`cupy.RawModule` (nvcc) kernel that fuses the VoxelBlockManager decode
(`decodeInverseMaps`) + 3x3x3 neighbour gather (`computeBoxStencil`) + the
per-voxel update into a single launch -- no dense `gatherBoxStencil` array. Run
the full file->file filter through the driver:

    python levelset_filter.py rawkernel  input.nvdb output.nvdb [outer_iterations]

Running this file directly executes a stencil-only smoke test (the deform +
renorm kernels on a sphere, no file I/O and no narrow-band retrack):

    python levelset_filter_rawkernel.py

This is the fastest of the three backends (the gather is fused into the compute,
no `(n, 27)` array is materialised) and the most code -- compare the kernel-free
levelset_filter_cupy.py / cuTile levelset_filter_cutile.py for the same math.
Requires CuPy, a CUDA-capable GPU, and nvcc (CuPy honours the NVCC env var). In
a dev/source tree, set NANOVDB_INCLUDE to the dir containing nanovdb/NanoVDB.h.
"""
import os

import numpy as np

import nanovdb

import levelset_filter as lsf


LOG2_BLOCK_WIDTH = lsf.LOG2_BLOCK_WIDTH       # 9
BLOCK_WIDTH = 1 << LOG2_BLOCK_WIDTH           # 512
NN_FACE = lsf.NN_FACE                         # 6-face dilation
SENTINEL = lsf.SENTINEL                       # "value not yet known" marker

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


def make_backend():
    if not (nanovdb.isCudaAvailable() and nanovdb.isGpuAvailable()):
        print("This backend requires a CUDA build of nanovdb and a GPU. Skipping.")
        return None
    try:
        import cupy as cp
    except ImportError:
        print("This backend requires CuPy (plus nvcc on PATH or $NVCC). Skipping.")
        return None
    options = _include_options()
    if options is None:
        return None
    return Backend(cp, options)


class Backend:
    """Per-voxel stencils as fused cupy.RawModule CUDA kernels that decode the VBM
    and gather the 3x3x3 neighbourhood in-kernel (no dense gatherBoxStencil), so
    this backend keeps its own VBM bookkeeping in the `g` context."""

    NAME = "rawkernel"

    def __init__(self, cp, options):
        self.cp = cp
        self.tc = nanovdb.tools.cuda
        m = cp.RawModule(code=KERNEL_SRC, backend="nvcc", options=options)
        self.k_decode = m.get_function("decode_coords")
        self.k_laplacian = m.get_function("laplacian_step")
        self.k_godunov = m.get_function("godunov_step")
        self.k_extrapolate = m.get_function("extrapolate")

    def setup(self, handle):
        """Device OnIndex grid + VBM pointers; decode value-indexed coords once."""
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

    def active_coords(self, g):
        return g["coords"]                            # decoded in setup()

    def _vbm(self, g):
        return (g["gptr"], g["fid"], g["jmp"], g["fo"])

    def laplacian(self, g, phi, half_width):
        cp = self.cp
        out = cp.empty_like(phi)
        self.k_laplacian((g["bc"],), (BLOCK_WIDTH,),
                         (*self._vbm(g), np.float32(half_width), phi, out))
        out[0] = SENTINEL
        return out

    def godunov(self, g, phi, vx, half_width):
        cp = self.cp
        out = cp.empty_like(phi)
        self.k_godunov((g["bc"],), (BLOCK_WIDTH,),
                       (*self._vbm(g), np.float32(vx), np.float32(0.3 * vx),
                        np.float32(half_width), phi, out))
        out[0] = SENTINEL
        return out

    def extrapolate(self, g, phi, vx):
        cp = self.cp
        out = cp.empty_like(phi)
        self.k_extrapolate((g["bc"],), (BLOCK_WIDTH,),
                           (*self._vbm(g), np.float32(vx), phi, out))
        out[0] = SENTINEL
        return out


if __name__ == "__main__":
    backend = make_backend()
    if backend is not None:
        lsf.stencil_demo(backend)
