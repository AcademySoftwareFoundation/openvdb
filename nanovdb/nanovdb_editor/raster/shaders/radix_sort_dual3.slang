// radix_sort_dual3.slang

struct constants_t
{
    uint workgroup_count;
    uint key_count;
};

StructuredBuffer<uint> val_tmp_in;
StructuredBuffer<uint> key_low_in;
StructuredBuffer<uint> key_high_in;
StructuredBuffer<uint> val_in;

ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint> key_low_copy_out;
RWStructuredBuffer<uint> key_high_copy_out;
RWStructuredBuffer<uint> val_copy_out;

[shader("compute")]
[numthreads(1024u, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint idx = 1024u * group_idx.x + thread_idx.x;

    if (idx >= constants.key_count)
    {
        return;
    }

    uint src_idx = val_tmp_in[idx];

    key_low_copy_out[idx] = key_low_in[src_idx];
    key_high_copy_out[idx] = key_high_in[src_idx];
    val_copy_out[idx] = val_in[src_idx];
}
