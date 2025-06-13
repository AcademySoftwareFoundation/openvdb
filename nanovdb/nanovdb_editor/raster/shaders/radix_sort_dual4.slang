// radix_sort_dual4.slang

struct constants_t
{
    uint workgroup_count;
    uint key_count;
};

StructuredBuffer<uint> key_low_copy_in;
StructuredBuffer<uint> key_high_copy_in;
StructuredBuffer<uint> val_copy_in;

ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint> key_low_out;
RWStructuredBuffer<uint> key_high_out;
RWStructuredBuffer<uint> val_out;

[shader("compute")]
[numthreads(1024u, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint idx = 1024u * group_idx.x + thread_idx.x;

    if (idx >= constants.key_count)
    {
        return;
    }

    key_low_out[idx] = key_low_copy_in[idx];
    key_high_out[idx] = key_high_copy_in[idx];
    val_out[idx] = val_copy_in[idx];
}
