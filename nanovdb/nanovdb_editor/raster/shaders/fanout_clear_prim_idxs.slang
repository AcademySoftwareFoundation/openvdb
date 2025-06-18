// fanout_clear_prim_idxs.slang

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

RWStructuredBuffer<uint> prim_idxs_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint ijk_idx = group_idx.x * 256u + thread_idx.x;

    if (ijk_idx >= constants.ijk_batch_size)
    {
        return;
    }

    prim_idxs_out[ijk_idx] = 0u;
}
