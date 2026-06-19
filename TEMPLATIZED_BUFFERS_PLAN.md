# Templatized Scratch Buffers — Implementation Plan

> Branch: `templatized-buffers` (forked from ASF `upstream/master`).
> Working note — **delete before any upstream PR**.

## 1. Goal & context

fvdb-core [PR #655](https://github.com/openvdb/fvdb-core/pull/655) works around a real
problem: nanoVDB's CUDA scratch allocations go through `cudaMallocAsync` /
`cudaFreeAsync`, which live in a **separate pool** from PyTorch's
`c10::cuda::CUDACachingAllocator`. The two pools fragment each other, and large
workloads (multi-frame TSDF integration) hit OOM with memory still free in
aggregate.

That PR's fix is a workaround: it **forks** `nanovdb/cuda/DeviceBuffer.h` and
`nanovdb/cuda/DeviceResource.h` into fvdb's tree and swaps the allocator *inside*
`nanovdb::cuda::DeviceBuffer` to PyTorch's caching allocator. Fork-of-headers is
fragile (their own PR text says CPM patching silently breaks on upstream drift).

**This branch provides the upstream-side fix instead: a buffer *type* seam.** The
transient scratch buffers in the topology operators become a defaulted template
parameter, so a downstream like fvdb can inject its own
PyTorch-allocator-backed buffer type (e.g. `TorchDeviceBuffer`) **without forking
any headers** — and drop the `DeviceBuffer.h`/`DeviceResource.h` forks entirely.

## 2. Locked design decisions

1. **Type seam, not allocator seam.** Keep upstream `DeviceBuffer` /
   `DeviceResource` untouched (still `cudaMallocAsync`). Inject behavior by
   swapping the *buffer type*, matching nanoVDB's existing `BufferT` idiom
   (`GridHandle<BufferT>`, `TopologyBuilder::getBuffer<BufferT>`).

2. **`ScratchBufferT` is its own class-level, defaulted template parameter** —
   decoupled from the *output* buffer type. This supports cases like
   `ScratchBufferT = DeviceBuffer` while output `BufferT = UnifiedBuffer`, and
   fvdb's `ScratchBufferT = TorchDeviceBuffer` with any output buffer.

   ```cpp
   template <typename BuildT, typename ScratchBufferT = nanovdb::cuda::DeviceBuffer>
   class TopologyBuilder { ... };
   ```

   The **output** buffer type stays exactly where it is: the method-level
   `getBuffer<BufferT>` / `getHandle<BufferT>` parameter. Untouched by this work.

3. **`ScratchBufferT` concept** — the contract the scratch sites rely on:
   - `static ScratchBufferT create(size_t bytes, const ScratchBufferT* pool, int device, cudaStream_t stream)`
   - `<ptr> deviceData()`  (device pointer; `void*` for `DeviceBuffer`, `uint8_t*` for `TorchDeviceBuffer`)
   - `void clear(cudaStream_t stream)`  ← stream-accepting overload **to be added to `TorchDeviceBuffer`**
   - default-constructible + move-assignable (members are default-constructed, then assigned)

4. **Dual-use buffers stay `DeviceBuffer`** (not templatized in this branch):
   - `TopologyBuilder::mProcessedRoot` — host-populated, then `deviceUpload`'d
   - `TopologyBuilder::mData` — `sizeof(Data)`, host + device
   These are the only two buffers with host+device dual use (verified: the only
   `deviceUpload` sites; there are **no** `deviceDownload` calls anywhere).
   *Future step (out of scope here):* split each into a host buffer + a device
   `ScratchBufferT` half with an explicit `cudaMemcpyAsync`. The device half then
   reuses the same `ScratchBufferT`; nothing escapes the downstream pool. The
   concept above is already sufficient for that future device-half — no rework.

5. **Cast fix.** `static_cast<T*>(void*)` is legal but `static_cast<T*>(uint8_t*)`
   is not; `reinterpret_cast<T*>` is legal from both. So switch **direct**
   `static_cast` on a templatized buffer's `.deviceData()`/`.data()` to
   `reinterpret_cast` (same runtime behavior, no `if constexpr`). Surgical, **not**
   a blanket replace — see §5.

6. **Lift `Data` out of `TopologyBuilder`** → `topology::detail::TopologyBuilderData<BuildT>`
   (parameterized on `BuildT` only), with `using Data = …;` alias in the class.
   **DONE** (pulled forward, right after M1) — it's cleaner and unblocks making the
   `TopologyBuilder` param non-defaulted. The 10 detail functors now reference
   `TopologyBuilderData<BuildT>` directly (no `TopologyBuilder` mention).
   *Why it's needed:* `Data` is a nested type, so `TopologyBuilder<BuildT, A>::Data`
   and `…<BuildT, B>::Data` are **distinct** types per instantiation. The 10 detail
   functors hardcode `typename TopologyBuilder<BuildT>::Data` (= the *default*
   `…, DeviceBuffer>::Data`); a non-default `ScratchBufferT` instantiation's
   `deviceData()` would return a mismatched `Data*` → compile error.
   *Why it can be deferred:* with the default `DeviceBuffer` everything resolves to
   one `Data` type, so M1–M3 compile and stay byte-identical without it. `Data` is
   the **only** structural blocker for non-default instantiation (casts + concept are
   handled in the M1–M3 pass), so lifting it + the canary in M5 validates M1–M3 at
   once. **Corollary:** do **not** add a non-default instantiation until M5.

## 3. Files in scope

| File | Work |
| :-- | :-- |
| `nanovdb/nanovdb/tools/cuda/TopologyBuilder.cuh` | **Core.** Add `ScratchBufferT` param; templatize 8 device-only scratch members + transient locals; cast fixes. |
| `nanovdb/nanovdb/tools/cuda/DilateGrid.cuh` | Add `ScratchBufferT` param; `mBuilder` → `TopologyBuilder<BuildT, ScratchBufferT>`. |
| `nanovdb/nanovdb/tools/cuda/PruneGrid.cuh`   | same |
| `nanovdb/nanovdb/tools/cuda/MergeGrids.cuh`  | same |
| `nanovdb/nanovdb/tools/cuda/CoarsenGrid.cuh` | same |
| `nanovdb/nanovdb/tools/cuda/RefineGrid.cuh`  | same |
| `nanovdb/nanovdb/tools/cuda/MeshToGrid.cuh`  | Add `ScratchBufferT` param; templatize its 3 device members + transient locals; cast fixes. |

## 4. Buffer inventory (per file)

### TopologyBuilder — templatize → `ScratchBufferT`
Members: `mUpperMasks`, `mLowerMasks`, `mUpperOffsets`, `mLowerOffsets`,
`mLeafOffsets`, `mVoxelOffsets`, `mLowerParents`, `mLeafParents`.
Transient locals: `upperCountsBuffer`, `lowerCountsBuffer`, `leafCountsBuffer`.

**Stay `DeviceBuffer`:** `mProcessedRoot`, `mData` (dual-use, §2.4).

### MeshToGrid — templatize → `ScratchBufferT`
Members: `mXformedTriangles`, `mBoxTrianglePairsBuffer`, `mUniqueRootOriginsBuffer`.
Transient locals: `rootBoxCounts`, `rootBoxOffsets`, `keysBuffer`,
`sortedKeysBuffer`, `uniqueKeysBuffer`, `numSelectedBuffer`, `retainMaskBuffer`,
`countsBuffer`, `offsetsBuffer`, `newPairsBuffer`, `sidecarBuffer`.
(`mBuilder.mProcessedRoot` here is TopologyBuilder's dual-use buffer → unchanged.)

### Morphology clients — no buffers of their own to templatize
Only thread the param into `mBuilder`. Their `srcRoot*Buffer` locals are
`nanovdb::HostBuffer` (host memory) → **out of scope**.

## 5. Cast strategy (surgical)

**Change** (direct cast on a templatized buffer):
- `static_cast<uint32_t*>(mUpperOffsets.deviceData())` → `reinterpret_cast<...>`
- ~22 sites in TopologyBuilder (the `*Offsets`/`*Parents`/`*CountsBuffer` ones),
  plus the device-scratch sites in MeshToGrid.

**Leave as `static_cast` (out of scope):**
- `mProcessedRoot.deviceData()` / `mData.deviceData()` — dual-use, return `void*`.
- `void* deviceUpperMasks(){ return mUpperMasks.deviceData(); }` accessors — the
  accessor return type is `void*` (implicit `uint8_t*`→`void*` is fine), so call
  sites like `static_cast<Mask<5>*>(mBuilder.deviceUpperMasks())` stay valid.
- `static_cast<GridT*>(mBuilder.data()->d_bufferPtr)` — raw `void*` member.
- `static_cast<...>(handle.deviceData())` — that's the **output** `BufferT`.

## 6. ABI / compatibility

Header-only templates; default `ScratchBufferT = nanovdb::cuda::DeviceBuffer`.
All existing instantiations (the `ex_*_nanovdb_cuda` examples and
`unittest/TestNanoVDB.cu`) compile unchanged and produce byte-identical behavior.
No ABI concern.

## 7. Known residuals outside a downstream pool (documented, not fixed here)

- `mProcessedRoot`, `mData` — dual-use, stay `DeviceBuffer` (`cudaMallocAsync`).
  `mData` is trivial; `mProcessedRoot` is the one real residual. Resolved later
  by the host/device split (§2.4).
- `mTempDevicePool` (`nanovdb::cuda::TempDevicePool`, via `DeviceResource`) — cub
  temp storage, still `cudaMallocAsync`. Not templatized here; flagged for a
  follow-up if its footprint matters to fvdb.

## 8. Milestones

> M1–M3 keep `Data` nested and use only the **default** `DeviceBuffer`; run the
> topology unittests after each (`DilateInjectPrune_*`, `RefineCoarsen_*`,
> `MergeGrids_*`, `MeshToGrid_*`, `mergeSplit*`) to confirm byte-identical behavior.

- **M1 — TopologyBuilder.cuh.** Add `ScratchBufferT`; templatize 8 scratch members
  + 3 transient locals; `static_cast`→`reinterpret_cast` on those. Verify green.
- **M2 — Morphology clients.** Thread `ScratchBufferT` through Dilate/Prune/Merge/
  Coarsen/Refine (class param + `mBuilder` type). Verify green.
- **M3 — MeshToGrid.cuh.** Param + members + locals + cast fixes. Verify green.
- **M4 — Build verification.** Build nanoVDB CUDA examples + full `TestNanoVDB.cu`
  with default `ScratchBufferT` → no regression.
- **`Data` lift — DONE** (between M1 and M2; see §2.6). Build + topology tests green.
- **M5 — Make `TopologyBuilder` `ScratchBufferT` non-defaulted + prove the seam. DONE.**
  - Dropped the default on `TopologyBuilder` (internal helper; default lives only on
    the public clients) → a client that forgets to forward fails to compile.
  - Lifted `MeshToGrid::BoxTrianglePair` to `nanovdb::tools::cuda` scope (+ public
    alias), mirroring the `Data` lift, so it's one type across instantiations.
  - Added `UnifiedBuffer::clear(cudaStream_t)` so `UnifiedBuffer` satisfies the
    scratch concept (parallels the `TorchDeviceBuffer` addition fvdb will make).
  - Fixed `getHandleAndUDF` to actually honor `SidecarBufferT` (`SidecarBufferT::create`
    + `reinterpret_cast`); previously it hardcoded `DeviceBuffer` for the sidecar.
  - Forwarded `ScratchBufferT` into MeshToGrid's internal `PruneGrid` so the all-Unified
    case is genuinely all-Unified scratch.
  - Canary: refactored the 4 topology unittests into helpers templated on every buffer
    axis (scratch / grid-output / sidecar-output), instantiated as all-`DeviceBuffer`
    and all-`UnifiedBuffer`. All 8 pass; full CUDA suite 55/56 (only the pre-existing
    `UnifiedBuffer_IO` data-file failure). First runtime exercise of a non-`DeviceBuffer`
    type — the seam is real.
- **M6 — follow-ups (separate branches/PRs):** dual-use host/device split;
  `mTempDevicePool`; fvdb-side `TorchDeviceBuffer` + stream-accepting `clear()`,
  then retire the `DeviceBuffer.h`/`DeviceResource.h` forks.

## 9. Build / verify commands

```bash
# Configure with nanoVDB + CUDA + examples + unit tests
cmake -S . -B build \
  -DOPENVDB_BUILD_NANOVDB=ON -DNANOVDB_USE_CUDA=ON \
  -DNANOVDB_BUILD_EXAMPLES=ON -DNANOVDB_BUILD_UNITTESTS=ON
cmake --build build -j
# (CUDA device required to *run* TestNanoVDB; compilation is the primary check here.)
```
