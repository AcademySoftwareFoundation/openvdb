// workgroup_scan.slang
#ifndef WORKGROUP_SCAN_SMEM_WORD_COUNT
#define WORKGROUP_SCAN_SMEM_WORD_COUNT 512u
#endif
groupshared uint smem[WORKGROUP_SCAN_SMEM_WORD_COUNT];
uint read_smem_idx(uint addr, uint idx)
{
    return smem[addr + idx];
}
void write_smem_idx(uint addr, uint idx, uint val)
{
    smem[addr + idx] = val;
}

// ----------------------------- workgroup uint4 add ---------------------------------------

void workgroup_reduce(uint thread_idx, uint4 val, inout uint total_count)
{
    uint sdata0_addr = 0;
    uint sdata1_addr = 256u;

    uint local_val = val.x + val.y + val.z + val.w;
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 64u)
    {
        local_val += read_smem_idx(sdata0_addr, thread_idx + 64u);
        local_val += read_smem_idx(sdata0_addr, thread_idx + 128u);
        local_val += read_smem_idx(sdata0_addr, thread_idx + 192u);
        write_smem_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 16u)
    {
        local_val += read_smem_idx(sdata1_addr, thread_idx + 16u);
        local_val += read_smem_idx(sdata1_addr, thread_idx + 32u);
        local_val += read_smem_idx(sdata1_addr, thread_idx + 48u);
        write_smem_idx(sdata0_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 4u)
    {
        local_val += read_smem_idx(sdata0_addr, thread_idx + 4u);
        local_val += read_smem_idx(sdata0_addr, thread_idx + 8u);
        local_val += read_smem_idx(sdata0_addr, thread_idx + 12u);
        write_smem_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    // compute total_count
    total_count = read_smem_idx(sdata1_addr, 0) +
        read_smem_idx(sdata1_addr, 1) +
        read_smem_idx(sdata1_addr, 2) +
        read_smem_idx(sdata1_addr, 3);
}

void workgroup_scan(uint thread_idx, uint4 val, inout uint4 result, inout uint total_count)
{
    uint sdata0_addr = 0;
    uint sdata1_addr = 256u;

    uint local_val = val.x + val.y + val.z + val.w;
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 1) local_val += read_smem_idx(sdata0_addr, thread_idx - 1);
    if (thread_idx >= 2) local_val += read_smem_idx(sdata0_addr, thread_idx - 2);
    if (thread_idx >= 3) local_val += read_smem_idx(sdata0_addr, thread_idx - 3);
    write_smem_idx(sdata1_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 4) local_val += read_smem_idx(sdata1_addr, thread_idx - 4);
    if (thread_idx >= 8) local_val += read_smem_idx(sdata1_addr, thread_idx - 8);
    if (thread_idx >= 12) local_val += read_smem_idx(sdata1_addr, thread_idx - 12);
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 16) local_val += read_smem_idx(sdata0_addr, thread_idx - 16);
    if (thread_idx >= 32) local_val += read_smem_idx(sdata0_addr, thread_idx - 32);
    if (thread_idx >= 48) local_val += read_smem_idx(sdata0_addr, thread_idx - 48);
    write_smem_idx(sdata1_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 64) local_val += read_smem_idx(sdata1_addr, thread_idx - 64);
    if (thread_idx >= 128) local_val += read_smem_idx(sdata1_addr, thread_idx - 128);
    if (thread_idx >= 192) local_val += read_smem_idx(sdata1_addr, thread_idx - 192);

    result.w = local_val;
    result.z = result.w - val.w;
    result.y = result.z - val.z;
    result.x = result.y - val.y;

    // compute total_count
    total_count = read_smem_idx(sdata1_addr, 63) +
        read_smem_idx(sdata1_addr, 127) +
        read_smem_idx(sdata1_addr, 191) +
        read_smem_idx(sdata1_addr, 255);
}

// ----------------------------- workgroup uint4 max ---------------------------------------

uint workgroup_uint_max(uint a, uint b)
{
    return a > b ? a : b;
}

void workgroup_reduce_max(uint thread_idx, uint4 val, inout uint total_count)
{
    uint sdata0_addr = 0;
    uint sdata1_addr = 256u;

    uint local_val = workgroup_uint_max(
        workgroup_uint_max(val.x, val.y),
        workgroup_uint_max(val.z, val.w));
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 64u)
    {
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 64u));
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 128u));
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 192u));
        write_smem_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 16u)
    {
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx + 16u));
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx + 32u));
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx + 48u));
        write_smem_idx(sdata0_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 4u)
    {
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 4u));
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 8u));
        local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 12u));
        write_smem_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    // compute total_count
    total_count = workgroup_uint_max(
        workgroup_uint_max(read_smem_idx(sdata1_addr, 0), read_smem_idx(sdata1_addr, 1)),
        workgroup_uint_max(read_smem_idx(sdata1_addr, 2), read_smem_idx(sdata1_addr, 3)));
}

void workgroup_scan_max(uint thread_idx, uint4 val, inout uint4 result, inout uint total_count)
{
    uint sdata0_addr = 0;
    uint sdata1_addr = 256u;

    uint local_val = workgroup_uint_max(
        workgroup_uint_max(val.x, val.y),
        workgroup_uint_max(val.z, val.w));
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 1) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx - 1));
    if (thread_idx >= 2) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx - 2));
    if (thread_idx >= 3) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx - 3));
    write_smem_idx(sdata1_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 4) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx - 4));
    if (thread_idx >= 8) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx - 8));
    if (thread_idx >= 12) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx - 12));
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 16) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx - 16));
    if (thread_idx >= 32) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx - 32));
    if (thread_idx >= 48) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata0_addr, thread_idx - 48));
    write_smem_idx(sdata1_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 64) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx - 64));
    if (thread_idx >= 128) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx - 128));
    if (thread_idx >= 192) local_val = workgroup_uint_max(local_val, read_smem_idx(sdata1_addr, thread_idx - 192));
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    uint scan_val = 0u;
    if (thread_idx > 0u)
    {
        scan_val = read_smem_idx(sdata0_addr, thread_idx - 1);
    }
    result.x = workgroup_uint_max(scan_val, val.x);
    result.y = workgroup_uint_max(result.x, val.y);
    result.z = workgroup_uint_max(result.y, val.z);
    result.w = workgroup_uint_max(result.z, val.w);

    // compute total_count
    total_count = read_smem_idx(sdata0_addr, 255);

    GroupMemoryBarrierWithGroupSync();
}

// ----------------------------- workgroup int max ---------------------------------------

#define WORKGROUP_INT_MIN (-2147483647 - 1)
#define WORKGROUP_INT_MAX 2147483647

int workgroup_int_max(int a, int b)
{
    return a > b ? a : b;
}

int workgroup_int_min(int a, int b)
{
    return a < b ? a : b;
}

void workgroup_reduce_max_int(uint thread_idx, int val, inout int val_max)
{
    uint sdata0_addr = 0;
    uint sdata1_addr = 256u;

    int local_val = val;
    write_smem_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 64u)
    {
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 64u));
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 128u));
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 192u));
        write_smem_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 16u)
    {
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata1_addr, thread_idx + 16u));
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata1_addr, thread_idx + 32u));
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata1_addr, thread_idx + 48u));
        write_smem_idx(sdata0_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 4u)
    {
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 4u));
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 8u));
        local_val = workgroup_int_max(local_val, read_smem_idx(sdata0_addr, thread_idx + 12u));
        write_smem_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    // compute val_max
    val_max = workgroup_int_max(
        workgroup_int_max(read_smem_idx(sdata1_addr, 0), read_smem_idx(sdata1_addr, 1)),
        workgroup_int_max(read_smem_idx(sdata1_addr, 2), read_smem_idx(sdata1_addr, 3)));
}

void workgroup_reduce_min_int(uint thread_idx, int val, inout int val_min)
{
    workgroup_reduce_max_int(thread_idx, -val, val_min);
    val_min = -val_min;
}