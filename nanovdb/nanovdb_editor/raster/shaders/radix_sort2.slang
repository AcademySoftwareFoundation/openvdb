// radix_sort2.slang

struct constants_t
{
    uint workgroup_count;
    uint pass_start;
    uint pass_mask;
    uint pass_bit_count;
    uint counter_count;
    uint key_bits_count;
    uint key_count;
    uint pad1;
};

StructuredBuffer<uint4> counters_in;
ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint4> counters_out;

#include <workgroup_scan.slang>

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    if (group_idx.x == 0u)
    {
        uint global_offset = 0u;

        uint pass_count = (constants.counter_count / 4 + 256u - 1) / (256u);
        uint idx = thread_idx.x;
        for (uint pass_id = 0; pass_id < pass_count; pass_id++)
        {
            uint block_offset = 0u;

            uint4 count_local = (idx < constants.counter_count / 4) ? counters_in[idx] : uint4(0, 0, 0, 0);

            uint4 count_global;
            workgroup_scan(thread_idx.x, count_local, count_global, block_offset);

            // make inclusive
            count_global.x -= count_local.x;
            count_global.y -= count_local.y;
            count_global.z -= count_local.z;
            count_global.w -= count_local.w;

            count_global.x += global_offset;
            count_global.y += global_offset;
            count_global.z += global_offset;
            count_global.w += global_offset;

            if (idx < constants.counter_count / 4)
            {
                counters_out[idx] = count_global;
            }

            global_offset += block_offset;

            idx += 256u;
        }
    }
    else
    {
        uint pass_count = (constants.counter_count / 4 + 256u - 1) / (256u);
        uint idx = thread_idx.x;
        for (uint pass_id = 0; pass_id < pass_count; pass_id++)
        {
            uint4 count_local = (idx < constants.counter_count / 4) ? counters_in[idx + constants.counter_count / 4] : uint4(0, 0, 0, 0);

            smem[thread_idx.x] = count_local.x + count_local.y + count_local.z + count_local.w;

            GroupMemoryBarrierWithGroupSync();

            uint scan_total = 0;
            if ((thread_idx.x & 3) >= 1) scan_total += smem[4 * (thread_idx.x / 4) + 0];
            if ((thread_idx.x & 3) >= 2) scan_total += smem[4 * (thread_idx.x / 4) + 1];
            if ((thread_idx.x & 3) >= 3) scan_total += smem[4 * (thread_idx.x / 4) + 2];

            // make final scan exclusive
            count_local.w = count_local.z + count_local.y + count_local.x + scan_total;
            count_local.z = count_local.y + count_local.x + scan_total;
            count_local.y = count_local.x + scan_total;
            count_local.x = scan_total;

            if (idx < constants.counter_count / 4)
            {
                counters_out[idx + constants.counter_count / 4] = count_local;
            }

            idx += 256u;
        }
    }
}
