// radix_sort3.slang

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
StructuredBuffer<uint4> val_in;
StructuredBuffer<uint> counters_in;
ConstantBuffer<constants_t> constants;

RWStructuredBuffer<uint> key_out;
RWStructuredBuffer<uint> val_out;

#define WORKGROUP_SCAN_SMEM_WORD_COUNT (512u + 1024u + 1024u + 16u + 16u)
#include <workgroup_scan.slang>

// where pred==1 indicates a zero allocation, pred==0 indicates a one allocation
uint4 split4(uint thread_idx, uint4 pred)
{
    uint total_count;
    uint4 scan_val;
    workgroup_scan(thread_idx, pred, scan_val, total_count);

    uint4 rank;
    rank.x = bool(pred.x) ? scan_val.x - 1 : 4 * thread_idx + 0 - scan_val.x + total_count;
    rank.y = bool(pred.y) ? scan_val.y - 1 : 4 * thread_idx + 1 - scan_val.y + total_count;
    rank.z = bool(pred.z) ? scan_val.z - 1 : 4 * thread_idx + 2 - scan_val.z + total_count;
    rank.w = bool(pred.w) ? scan_val.w - 1 : 4 * thread_idx + 3 - scan_val.w + total_count;

    return rank;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_idx : SV_GroupID, uint3 thread_idx : SV_GroupThreadID)
{
    uint idx = 256u * group_idx.x + thread_idx.x;

    uint skey_addr = 512u;
    uint sval_addr = 512u + 1024u;
    uint global_counters_addr = 512u + 1024u + 1024u;
    uint local_counters_addr = 512u + 1024u + 1024u + 16u;

    // preload counter values
    if (thread_idx.x < 16)
    {
        write_smem_idx(global_counters_addr, thread_idx.x, counters_in[thread_idx.x * constants.workgroup_count + group_idx.x]);
        write_smem_idx(local_counters_addr, thread_idx.x, counters_in[thread_idx.x + 16 * (constants.workgroup_count + group_idx.x)]);
    }

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

    uint4 val_local = uint4(0u, 0u, 0u, 0u);
    if (4u * idx + 3u < constants.key_count)
    {
        val_local = val_in[idx];
    }
    else
    {
        if (4u * idx + 0u < constants.key_count) { val_local.x = val_in[idx].x;}
        if (4u * idx + 1u < constants.key_count) { val_local.y = val_in[idx].y;}
        if (4u * idx + 2u < constants.key_count) { val_local.z = val_in[idx].z;}
    }

    for (uint pass_id = 0u; pass_id < constants.pass_bit_count; pass_id++)
    {
        uint4 alloc_val;
        uint4 key_local_masked = (key_local >> constants.pass_start) & constants.pass_mask;
        alloc_val.x = ((key_local_masked.x >> pass_id) & 1) ^ 1u;
        alloc_val.y = ((key_local_masked.y >> pass_id) & 1) ^ 1u;
        alloc_val.z = ((key_local_masked.z >> pass_id) & 1) ^ 1u;
        alloc_val.w = ((key_local_masked.w >> pass_id) & 1) ^ 1u;

        uint4 allocIdx = split4(thread_idx.x, alloc_val);

        write_smem_idx(skey_addr, allocIdx.x, key_local.x);
        write_smem_idx(skey_addr, allocIdx.y, key_local.y);
        write_smem_idx(skey_addr, allocIdx.z, key_local.z);
        write_smem_idx(skey_addr, allocIdx.w, key_local.w);
        write_smem_idx(sval_addr, allocIdx.x, val_local.x);
        write_smem_idx(sval_addr, allocIdx.y, val_local.y);
        write_smem_idx(sval_addr, allocIdx.z, val_local.z);
        write_smem_idx(sval_addr, allocIdx.w, val_local.w);

        GroupMemoryBarrierWithGroupSync();

        key_local.x = read_smem_idx(skey_addr, 4 * thread_idx.x + 0);
        key_local.y = read_smem_idx(skey_addr, 4 * thread_idx.x + 1);
        key_local.z = read_smem_idx(skey_addr, 4 * thread_idx.x + 2);
        key_local.w = read_smem_idx(skey_addr, 4 * thread_idx.x + 3);
        val_local.x = read_smem_idx(sval_addr, 4 * thread_idx.x + 0);
        val_local.y = read_smem_idx(sval_addr, 4 * thread_idx.x + 1);
        val_local.z = read_smem_idx(sval_addr, 4 * thread_idx.x + 2);
        val_local.w = read_smem_idx(sval_addr, 4 * thread_idx.x + 3);
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint shared_idx = thread_idx.x; shared_idx < 4 * 256u; shared_idx += 256u)
    {
        uint bucketIdx = (read_smem_idx(skey_addr, shared_idx) >> constants.pass_start) & constants.pass_mask;
        uint dst_idx = shared_idx - read_smem_idx(local_counters_addr, bucketIdx) + read_smem_idx(global_counters_addr, bucketIdx);
        if (dst_idx < constants.key_count)
        {
            key_out[dst_idx] = read_smem_idx(skey_addr, shared_idx);
            val_out[dst_idx] = read_smem_idx(sval_addr, shared_idx);
        }
    }
}
