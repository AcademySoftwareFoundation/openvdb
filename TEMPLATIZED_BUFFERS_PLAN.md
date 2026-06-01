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

- **M1 — TopologyBuilder.cuh.** Add `ScratchBufferT`; templatize members + locals;
  cast fixes. Compile examples + unittest with default param → no regression.
- **M2 — Morphology clients.** Thread `ScratchBufferT` through Dilate/Prune/Merge/
  Coarsen/Refine (class param + `mBuilder` type).
- **M3 — MeshToGrid.cuh.** Param + members + locals + cast fixes.
- **M4 — Build verification.** Build nanoVDB CUDA examples + `TestNanoVDB.cu` with
  default `ScratchBufferT`. Optional canary: instantiate a client with an
  alternate `ScratchBufferT` (e.g. `UnifiedBuffer`) to prove the seam is real.
- **M5 — follow-ups (separate branches/PRs):** dual-use host/device split;
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
