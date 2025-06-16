// scan2.slang

struct constants_t
{
    uint val_count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint64_t> reduce_in;
ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint64_t> reduce_scan_out;

#include <workgroup_scan_uint64.slang>

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint workgroup_count = (constants.val_count + 1023u) / 1024u;
    uint scan_pass_count = (workgroup_count + 1023u) / 1024u;
    uint64_t global_offset = 0llu;
    for (uint scan_pass_idx = 0u; scan_pass_idx < scan_pass_count; scan_pass_idx++)
    {
        uint reduce4_idx = scan_pass_idx * 256u + thread_idx.x;

        vector<uint64_t, 4> reduce4;
        reduce4.x = (4u * reduce4_idx + 0u < workgroup_count) ? reduce_in[4u * reduce4_idx + 0u] : 0llu;
        reduce4.y = (4u * reduce4_idx + 1u < workgroup_count) ? reduce_in[4u * reduce4_idx + 1u] : 0llu;
        reduce4.z = (4u * reduce4_idx + 2u < workgroup_count) ? reduce_in[4u * reduce4_idx + 2u] : 0llu;
        reduce4.w = (4u * reduce4_idx + 3u < workgroup_count) ? reduce_in[4u * reduce4_idx + 3u] : 0llu;

        uint64_t total_count = 0llu;
        vector<uint64_t, 4> reduce_scan;
        workgroup_scan_uint64(thread_idx.x, reduce4, reduce_scan, total_count);

        reduce_scan += global_offset;
        if (4u * reduce4_idx + 0u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 0u] = reduce_scan.x;}
        if (4u * reduce4_idx + 1u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 1u] = reduce_scan.y;}
        if (4u * reduce4_idx + 2u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 2u] = reduce_scan.z;}
        if (4u * reduce4_idx + 3u < workgroup_count) {reduce_scan_out[4u * reduce4_idx + 3u] = reduce_scan.w;}

        global_offset += total_count;
    }
}
