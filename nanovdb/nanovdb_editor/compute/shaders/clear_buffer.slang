// clear_buffer.slang
RWStructuredBuffer<uint> data_out;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    data_out[dispatchThreadID.x] = 0u;
}
