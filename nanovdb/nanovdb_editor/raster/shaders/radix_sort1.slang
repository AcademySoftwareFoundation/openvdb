// radix_sort1.slang

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

StructuredBuffer<uint4> key_in;
ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint> counters_out;

#include <workgroup_scan.slang>

groupshared uint4 smem_uint4[320u];
uint4 read_smem_uint4_idx(uint addr, uint idx)
{
    return smem_uint4[addr + idx];
}
void write_smem_uint4_idx(uint addr, uint idx, uint4 val)
{
    smem_uint4[addr + idx] = val;
}

void count(inout uint4 counter, uint bucket)
{
    if (bucket == 0) counter.x += (1 << 0);
    if (bucket == 1) counter.x += (1 << 16);
    if (bucket == 2) counter.y += (1 << 0);
    if (bucket == 3) counter.y += (1 << 16);
    if (bucket == 4) counter.z += (1 << 0);
    if (bucket == 5) counter.z += (1 << 16);
    if (bucket == 6) counter.w += (1 << 0);
    if (bucket == 7) counter.w += (1 << 16);
    if (bucket == 8) counter.x += (1 << 8);
    if (bucket == 9) counter.x += (1 << 24);
    if (bucket == 10) counter.y += (1 << 8);
    if (bucket == 11) counter.y += (1 << 24);
    if (bucket == 12) counter.z += (1 << 8);
    if (bucket == 13) counter.z += (1 << 24);
    if (bucket == 14) counter.w += (1 << 8);
    if (bucket == 15) counter.w += (1 << 24);
}

uint4 expand8to16l(uint4 counter)
{
    uint4 counterL;
    counterL.x = counter.x & 0x00FF00FF;
    counterL.y = counter.y & 0x00FF00FF;
    counterL.z = counter.z & 0x00FF00FF;
    counterL.w = counter.w & 0x00FF00FF;
    return counterL;
}

uint4 expand8to16h(uint4 counter)
{
    uint4 counterH;
    counterH.x = (counter.x & 0xFF00FF00) >> 8;
    counterH.y = (counter.y & 0xFF00FF00) >> 8;
    counterH.z = (counter.z & 0xFF00FF00) >> 8;
    counterH.w = (counter.w & 0xFF00FF00) >> 8;
    return counterH;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint idx = 256u * group_idx.x + thread_idx.x;

    uint scount0_addr = 0;
    uint scount1_addr = 256u;

    uint4 local_count = uint4(0, 0, 0, 0);

    uint4 key_local = uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    if (4u * idx + 3u < constants.key_count)
    {
        key_local = key_in[idx];
    }
    else
    {
        if (4u * idx + 0u < constants.key_count) { key_local.x = key_in[idx].x;}
        if (4u * idx + 1u < constants.key_count) { key_local.y = key_in[idx].y;}
        if (4u * idx + 2u < constants.key_count) { key_local.z = key_in[idx].z;}
    }

    // masking
    key_local = (key_local >> constants.pass_start) & constants.pass_mask;

    count(local_count, key_local.x);
    count(local_count, key_local.y);
    count(local_count, key_local.z);
    count(local_count, key_local.w);

    write_smem_uint4_idx(scount0_addr, thread_idx.x, local_count);
    GroupMemoryBarrierWithGroupSync();

    if (thread_idx.x < 256u / 4)
    {
        local_count = read_smem_uint4_idx(scount0_addr, thread_idx.x);
        local_count = local_count + read_smem_uint4_idx(scount0_addr, thread_idx.x + 1 * 256u / 4);
        local_count = local_count + read_smem_uint4_idx(scount0_addr, thread_idx.x + 2 * 256u / 4);
        local_count = local_count + read_smem_uint4_idx(scount0_addr, thread_idx.x + 3 * 256u / 4);
        write_smem_uint4_idx(scount1_addr, thread_idx.x, local_count);
    }
    GroupMemoryBarrierWithGroupSync();

    // expand to 16-bit from 8-bit
    if (thread_idx.x < 256u / 16)
    {
        uint4 local_count_h;
        local_count = expand8to16l(read_smem_uint4_idx(scount1_addr, thread_idx.x));
        local_count_h = expand8to16h(read_smem_uint4_idx(scount1_addr, thread_idx.x));
        local_count = local_count + expand8to16l(read_smem_uint4_idx(scount1_addr, thread_idx.x + 1 * 256u / 16));
        local_count_h = local_count_h + expand8to16h(read_smem_uint4_idx(scount1_addr, thread_idx.x + 1 * 256u / 16));
        local_count = local_count + expand8to16l(read_smem_uint4_idx(scount1_addr, thread_idx.x + 2 * 256u / 16));
        local_count_h = local_count_h + expand8to16h(read_smem_uint4_idx(scount1_addr, thread_idx.x + 2 * 256u / 16));
        local_count = local_count + expand8to16l(read_smem_uint4_idx(scount1_addr, thread_idx.x + 3 * 256u / 16));
        local_count_h = local_count_h + expand8to16h(read_smem_uint4_idx(scount1_addr, thread_idx.x + 3 * 256u / 16));
        write_smem_uint4_idx(scount0_addr, thread_idx.x, local_count);
        write_smem_uint4_idx(scount0_addr, thread_idx.x + 256u / 16, local_count_h);
    }
    GroupMemoryBarrierWithGroupSync();

    // two sets of 16 uint4 left to be reduced
    uint set_id = thread_idx.x / (256u / 2);
    uint set_lane_id = thread_idx.x & (256u / 2 - 1);
    if (set_lane_id < 256u / 64)
    {
        uint offset = set_id * 256u / 16;
        local_count = read_smem_uint4_idx(scount0_addr, set_lane_id + offset);
        local_count = local_count + read_smem_uint4_idx(scount0_addr, set_lane_id + 1 * 256u / 64 + offset);
        local_count = local_count + read_smem_uint4_idx(scount0_addr, set_lane_id + 2 * 256u / 64 + offset);
        local_count = local_count + read_smem_uint4_idx(scount0_addr, set_lane_id + 3 * 256u / 64 + offset);
        write_smem_uint4_idx(scount1_addr, set_lane_id + set_id * 256u / 64, local_count);
    }
    GroupMemoryBarrierWithGroupSync();

    // two sets of 4 uint4 left to be reduced
    if (set_lane_id == 0)
    {
        uint offset = set_id * 256u / 64;
        local_count = read_smem_uint4_idx(scount1_addr, 0 + offset);
        local_count = local_count + read_smem_uint4_idx(scount1_addr, 1 + offset);
        local_count = local_count + read_smem_uint4_idx(scount1_addr, 2 + offset);
        local_count = local_count + read_smem_uint4_idx(scount1_addr, 3 + offset);

        uint bucketOffset = 8 * set_id;

        // output counter values for global scan
        counters_out[(bucketOffset + 0) * constants.workgroup_count + group_idx.x] = (local_count.x & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 1) * constants.workgroup_count + group_idx.x] = (local_count.x & 0xFFFF0000) >> 16;
        counters_out[(bucketOffset + 2) * constants.workgroup_count + group_idx.x] = (local_count.y & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 3) * constants.workgroup_count + group_idx.x] = (local_count.y & 0xFFFF0000) >> 16;
        counters_out[(bucketOffset + 4) * constants.workgroup_count + group_idx.x] = (local_count.z & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 5) * constants.workgroup_count + group_idx.x] = (local_count.z & 0xFFFF0000) >> 16;
        counters_out[(bucketOffset + 6) * constants.workgroup_count + group_idx.x] = (local_count.w & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 7) * constants.workgroup_count + group_idx.x] = (local_count.w & 0xFFFF0000) >> 16;

        // output counter values for local scan
        counters_out[(bucketOffset + 0) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.x & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 1) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.x & 0xFFFF0000) >> 16;
        counters_out[(bucketOffset + 2) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.y & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 3) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.y & 0xFFFF0000) >> 16;
        counters_out[(bucketOffset + 4) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.z & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 5) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.z & 0xFFFF0000) >> 16;
        counters_out[(bucketOffset + 6) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.w & 0x0000FFFF) >> 0;
        counters_out[(bucketOffset + 7) + 16 * (constants.workgroup_count + group_idx.x)] = (local_count.w & 0xFFFF0000) >> 16;
    }
}
