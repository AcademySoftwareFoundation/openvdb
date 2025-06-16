// grid_build_clear.slang

struct constants_t
{
    uint workgroup_count;
    uint max_node_count;
    uint buf_word_count;
    uint point_count;
    uint empty_grid_word_count;
    uint active_node_type;
    uint child_node_type;
};

ConstantBuffer<constants_t> constants;

StructuredBuffer<uint> upload;

RWStructuredBuffer<uint> buf;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint group_begin_idx = group_idx.x;
    uint group_count = (constants.buf_word_count + 255u) / 256u;
    for (uint local_group_idx = group_begin_idx; local_group_idx < group_count; local_group_idx += constants.workgroup_count)
    {
        uint tidx = local_group_idx * 256u + thread_idx.x;
        if (tidx < constants.buf_word_count)
        {
            uint val = 0u;
            if (tidx < constants.empty_grid_word_count)
            {
                val = upload[tidx];
            }
            buf[tidx] = val;
        }
    }
}
