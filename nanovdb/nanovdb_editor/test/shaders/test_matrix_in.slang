// test_matrix.slang
struct constants_t
{
    float4x4 matrix;
};

StructuredBuffer<int> data_in;
ConstantBuffer<constants_t> constants;
RWStructuredBuffer<int> data_out;
RWStructuredBuffer<uint> scratch;

[shader("compute")]
[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 tidx = int2(dispatchThreadID.xy);

    // Store the matrix in the output buffer in row-major order
    uint idx = 4 * tidx.x + tidx.y;
    data_out[idx] = int(constants.matrix[tidx.x][tidx.y]);
}
