// fanout_scatter_prim.slang

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

RWStructuredBuffer<uint> prim_idxs_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint prim_idx = group_idx.x * 256u + thread_idx.x;

    if (prim_idx >= constants.prim_batch_size)
    {
        return;
    }

    uint64_t fragment_count = fragment_counts_in[prim_idx];
    uint64_t scan_fragment_count = scan_fragment_counts_in[prim_idx];

    uint64_t fragment_range_min = scan_fragment_count - fragment_count;
    uint64_t fragment_range_max = scan_fragment_count;

    // clip to window
    if (fragment_range_min < constants.fragment_window_min &&
        fragment_range_max > constants.fragment_window_min)
    {
        fragment_range_min = constants.fragment_window_min;
    }
    // scatter
    if (fragment_range_min >= constants.fragment_window_min &&
        fragment_range_min < constants.fragment_window_max)
    {
        uint scatter_idx = uint(fragment_range_min - constants.fragment_window_min);

        prim_idxs_out[scatter_idx] = prim_idx;
    }
}
