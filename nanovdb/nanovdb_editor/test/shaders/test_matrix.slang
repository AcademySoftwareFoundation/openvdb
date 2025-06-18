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
    float4x4 matrix = float4x4(
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0
    );

    // Store the matrix in the output buffer in row-major order
    uint idx = 4 * tidx.x + tidx.y;
    data_out[idx] = int(matrix[tidx.x][tidx.y]);
}
