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
StructuredBuffer<uint> reduce_scan_in;

RWStructuredBuffer<uint> val_out;

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
    uint4 scan_val;
    workgroup_scan(thread_idx.x, val4, scan_val, total_count);

    uint global_offset = 0u;
    if (group_idx.x > 0u)
    {
        global_offset = reduce_scan_in[group_idx.x - 1u];
    }

    scan_val += global_offset;

    if (4u * val4_idx + 0u < constants.val_count) {val_out[4u * val4_idx + 0u] = scan_val.x;}
    if (4u * val4_idx + 1u < constants.val_count) {val_out[4u * val4_idx + 1u] = scan_val.y;}
    if (4u * val4_idx + 2u < constants.val_count) {val_out[4u * val4_idx + 2u] = scan_val.z;}
    if (4u * val4_idx + 3u < constants.val_count) {val_out[4u * val4_idx + 3u] = scan_val.w;}
}
