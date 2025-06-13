// scan2.slang

struct constants_t
{
    uint val_count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint> reduce_in;
ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint> reduce_scan_out;

#include <workgroup_scan.slang>

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint workgroup_count = (constants.val_count + 1023u) / 1024u;
    uint scan_pass_count = (workgroup_count + 1023u) / 1024u;
    uint global_offset = 0u;
    for (uint scan_pass_idx = 0u; scan_pass_idx < scan_pass_count; scan_pass_idx++)
    {
        uint reduce4_idx = scan_pass_idx * 256u + thread_idx.x;

        uint4 reduce4;
        reduce4.x = (4u * reduce4_idx + 0u < workgroup_count) ? reduce_in[4u * reduce4_idx + 0u] : 0u;
        reduce4.y = (4u * reduce4_idx + 1u < workgroup_count) ? reduce_in[4u * reduce4_idx + 1u] : 0u;
        reduce4.z = (4u * reduce4_idx + 2u < workgroup_count) ? reduce_in[4u * reduce4_idx + 2u] : 0u;
        reduce4.w = (4u * reduce4_idx + 3u < workgroup_count) ? reduce_in[4u * reduce4_idx + 3u] : 0u;

        uint total_count = 0u;
        uint4 reduce_scan;
        workgroup_scan_max(thread_idx.x, reduce4, reduce_scan, total_count);

        reduce_scan.x = workgroup_uint_max(reduce_scan.x, global_offset);
        reduce_scan.y = workgroup_uint_max(reduce_scan.y, global_offset);
        reduce_scan.z = workgroup_uint_max(reduce_scan.z, global_offset);
        reduce_scan.w = workgroup_uint_max(reduce_scan.w, global_offset);
        if (4u * reduce4_idx + 0u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 0u] = reduce_scan.x;}
        if (4u * reduce4_idx + 1u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 1u] = reduce_scan.y;}
        if (4u * reduce4_idx + 2u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 2u] = reduce_scan.z;}
        if (4u * reduce4_idx + 3u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 3u] = reduce_scan.w;}

        global_offset = workgroup_uint_max(global_offset, total_count);
    }
}
