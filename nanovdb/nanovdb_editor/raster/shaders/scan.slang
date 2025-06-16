// scan.slang
#define PNANOVDB_HLSL
#define PNANOVDB_ADDRESS_32
#define PNANOVDB_BUF_HLSL_RW
#define PNANOVDB_GPU_THREADS
#include "putil/Scan.h"

struct constants_t
{
    pnanovdb_uint32_t val_count;
    pnanovdb_uint32_t pad0;
    pnanovdb_uint32_t pad1;
    pnanovdb_uint32_t pad2;
};

RWStructuredBuffer<uint> data_in;
ConstantBuffer<constants_t> constants;
RWStructuredBuffer<uint> data_out;
RWStructuredBuffer<uint> scratch;

// makes pnanovdb code more copy paste friendly
#define val_buf data_in
#define result_buf data_out
#define tmp_buf scratch
#define smem_buf scratch
#define status_buf scratch

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
#if 0
    uint tidx = dispatchThreadID.x;

    pnanovdb_uvec4_t val[1] = { uint4(
        data_in[4u * tidx + 0u],
        data_in[4u * tidx + 1u],
        data_in[4u * tidx + 2u],
        data_in[4u * tidx + 3u])
    };
    pnanovdb_uvec4_t result[1];

    pnanovdb_uint32_t total_count = 0u;
    pnanovdb_workgroup_scan(
        groupThreadID.x,
        scratch, pnanovdb_address_null(),
        val,
        result,
        total_count
    );

    data_out[4u * tidx + 0u] = result[0].x;
    data_out[4u * tidx + 1u] = result[0].y;
    data_out[4u * tidx + 2u] = result[0].z;
    data_out[4u * tidx + 3u] = result[0].w;
#endif

#if 1
    pnanovdb_address_t tmp_addr = pnanovdb_address_null();

    const uint workgroup_count = 64u;

    // allocate global memory for global sync
    pnanovdb_address_t status_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), 4096u * 4u);

    // global smem size
    pnanovdb_address_t smem_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * pnanovdb_scan_smem_size_in_words * 4u);

    // per workgroup address
    smem_addr = pnanovdb_address_offset_product(smem_addr, groupID.x, pnanovdb_scan_smem_size_in_words * 4u);

    // isolate sync vs scan smem
    pnanovdb_address_t smem_sync_addr = smem_addr;
    smem_addr = pnanovdb_address_offset(smem_addr, 8u);

    // allocate global tmp buffers for scan itself
    tmp_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(constants.val_count));

    pnanovdb_sync_state_t sync_state;
    pnanovdb_sync_init(sync_state, groupThreadID.x, 1u, status_buf, status_addr);

    pnanovdb_uint32_t total_count = 0u;
    pnanovdb_scan(
        groupThreadID.x,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        status_buf, status_addr,
        constants.val_count,
        val_buf, pnanovdb_address_null(),
        tmp_buf, tmp_addr,
        result_buf, pnanovdb_address_null(),
        total_count
    );
#endif
}
