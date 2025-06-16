// test.slang
struct constants_t
{
    int magic_number;
    int pad1;
    int pad2;
    int pad3;
};

StructuredBuffer<int> data_in;
ConstantBuffer<constants_t> constants;
RWStructuredBuffer<int> data_out;
RWStructuredBuffer<uint> scratch;

[shader("compute")]
[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;

    data_out[idx] = data_in[idx] + constants.magic_number;
}
