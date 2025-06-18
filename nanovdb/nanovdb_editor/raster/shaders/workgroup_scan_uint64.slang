// workgroup_scan_uint64.slang
groupshared uint64_t smem_64[512u];
uint64_t read_smem_64_idx(uint addr, uint idx)
{
    return smem_64[addr + idx];
}
void write_smem_64_idx(uint addr, uint idx, uint64_t val)
{
    smem_64[addr + idx] = val;
}

// ----------------------------- workgroup uint4 add ---------------------------------------

void workgroup_reduce_uint64(uint thread_idx, vector<uint64_t, 4> val, inout uint64_t total_count)
{
    uint sdata0_addr = 0;
    uint sdata1_addr = 256u;

    uint64_t local_val = val.x + val.y + val.z + val.w;
    write_smem_64_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 64u)
    {
        local_val += read_smem_64_idx(sdata0_addr, thread_idx + 64u);
        local_val += read_smem_64_idx(sdata0_addr, thread_idx + 128u);
        local_val += read_smem_64_idx(sdata0_addr, thread_idx + 192u);
        write_smem_64_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 16u)
    {
        local_val += read_smem_64_idx(sdata1_addr, thread_idx + 16u);
        local_val += read_smem_64_idx(sdata1_addr, thread_idx + 32u);
        local_val += read_smem_64_idx(sdata1_addr, thread_idx + 48u);
        write_smem_64_idx(sdata0_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx < 4u)
    {
        local_val += read_smem_64_idx(sdata0_addr, thread_idx + 4u);
        local_val += read_smem_64_idx(sdata0_addr, thread_idx + 8u);
        local_val += read_smem_64_idx(sdata0_addr, thread_idx + 12u);
        write_smem_64_idx(sdata1_addr, thread_idx, local_val);
    }

    GroupMemoryBarrierWithGroupSync();

    // compute total_count
    total_count = read_smem_64_idx(sdata1_addr, 0) +
        read_smem_64_idx(sdata1_addr, 1) +
        read_smem_64_idx(sdata1_addr, 2) +
        read_smem_64_idx(sdata1_addr, 3);
}

void workgroup_scan_uint64(uint thread_idx, vector<uint64_t, 4> val, inout vector<uint64_t, 4> result, inout uint64_t total_count)
{
    uint sdata0_addr = 0;
    uint sdata1_addr = 256u;

    uint64_t local_val = val.x + val.y + val.z + val.w;
    write_smem_64_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 1) local_val += read_smem_64_idx(sdata0_addr, thread_idx - 1);
    if (thread_idx >= 2) local_val += read_smem_64_idx(sdata0_addr, thread_idx - 2);
    if (thread_idx >= 3) local_val += read_smem_64_idx(sdata0_addr, thread_idx - 3);
    write_smem_64_idx(sdata1_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 4) local_val += read_smem_64_idx(sdata1_addr, thread_idx - 4);
    if (thread_idx >= 8) local_val += read_smem_64_idx(sdata1_addr, thread_idx - 8);
    if (thread_idx >= 12) local_val += read_smem_64_idx(sdata1_addr, thread_idx - 12);
    write_smem_64_idx(sdata0_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 16) local_val += read_smem_64_idx(sdata0_addr, thread_idx - 16);
    if (thread_idx >= 32) local_val += read_smem_64_idx(sdata0_addr, thread_idx - 32);
    if (thread_idx >= 48) local_val += read_smem_64_idx(sdata0_addr, thread_idx - 48);
    write_smem_64_idx(sdata1_addr, thread_idx, local_val);

    GroupMemoryBarrierWithGroupSync();

    if (thread_idx >= 64) local_val += read_smem_64_idx(sdata1_addr, thread_idx - 64);
    if (thread_idx >= 128) local_val += read_smem_64_idx(sdata1_addr, thread_idx - 128);
    if (thread_idx >= 192) local_val += read_smem_64_idx(sdata1_addr, thread_idx - 192);

    result.w = local_val;
    result.z = result.w - val.w;
    result.y = result.z - val.z;
    result.x = result.y - val.y;

    // compute total_count
    total_count = read_smem_64_idx(sdata1_addr, 63) +
        read_smem_64_idx(sdata1_addr, 127) +
        read_smem_64_idx(sdata1_addr, 191) +
        read_smem_64_idx(sdata1_addr, 255);
}
