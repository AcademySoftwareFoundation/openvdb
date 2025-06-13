// scan1.slang

struct constants_t
{
    uint val_count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint> val_in;
ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint> reduce_out;

#include <workgroup_scan.slang>

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint val4_idx = group_idx.x * 256u + thread_idx.x;

    uint4 val4;
    val4.x = (4u * val4_idx + 0u < constants.val_count) ? val_in[4u * val4_idx + 0u] : 0u;
    val4.y = (4u * val4_idx + 1u < constants.val_count) ? val_in[4u * val4_idx + 1u] : 0u;
    val4.z = (4u * val4_idx + 2u < constants.val_count) ? val_in[4u * val4_idx + 2u] : 0u;
    val4.w = (4u * val4_idx + 3u < constants.val_count) ? val_in[4u * val4_idx + 3u] : 0u;

    uint total_count = 0u;
    workgroup_reduce(thread_idx.x, val4, total_count);

    if (thread_idx.x == 0)
    {
        reduce_out[group_idx.x] = total_count;
    }
}
