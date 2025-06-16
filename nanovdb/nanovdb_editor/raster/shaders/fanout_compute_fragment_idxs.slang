// fanout_compute_fragment_idxs.slang

struct constants_t
{
    uint prim_batch_size;
    uint ijk_batch_size;
    uint64_t fragment_window_min;
    uint64_t fragment_window_max;
    uint global_prim_begin_idx;
    uint global_prim_count;
    float voxel_size;
    float voxel_size_inv;
};

ConstantBuffer<constants_t> constants;

StructuredBuffer<uint64_t> fragment_counts_in;
StructuredBuffer<uint64_t> scan_fragment_counts_in;
StructuredBuffer<uint> prim_idxs_in;

RWStructuredBuffer<uint> prim_raster_idxs_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint ijk_idx = group_idx.x * 256u + thread_idx.x;

    if (ijk_idx >= constants.ijk_batch_size)
    {
        return;
    }

    uint prim_idx = prim_idxs_in[ijk_idx];

    uint64_t fragment_count = fragment_counts_in[prim_idx];
    uint64_t scan_fragment_count = scan_fragment_counts_in[prim_idx];

    uint64_t fragment_range_min = scan_fragment_count - fragment_count;
    uint64_t fragment_range_max = scan_fragment_count;

    uint64_t global_ijk_idx = uint64_t(ijk_idx) + constants.fragment_window_min;

    uint64_t global_idx_max = scan_fragment_counts_in[constants.prim_batch_size - 1u];

    uint prim_raster_idx = ~0u;
    if (global_ijk_idx >= fragment_range_min &&
        global_ijk_idx < fragment_range_max &&
        global_ijk_idx < global_idx_max)
    {
        prim_raster_idx = uint(global_ijk_idx - fragment_range_min);
    }

    prim_raster_idxs_out[ijk_idx] = prim_raster_idx;
}
