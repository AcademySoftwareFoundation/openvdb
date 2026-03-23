// step_timing_bench.cpp — isolate cycle cost per step of Plan #1

#include <cstdint>
#include <cstdio>
#include <x86intrin.h>

union qword { uint64_t ui64; uint8_t ui8[8]; };
static constexpr uint64_t kSpread = UINT64_C(0x0101010101010101);

static inline uint64_t scatterLSB(uint64_t src)
{
    uint64_t x = src & 0xFFu;
    x = (x | (x << 14) | (x << 28) | (x << 42)) & UINT64_C(0x0003000300030003);
    x = (x | (x <<  7))                          & UINT64_C(0x0101010101010101);
    return x;
}
static inline uint64_t lcg64(uint64_t& s)
{
    s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return s;
}

int main()
{
    const int NITER = 100000;
    uint64_t seed = UINT64_C(0xdeadbeefcafe);

    const int NINPUTS = 256;
    uint64_t maskBuf[NINPUTS][8];
    uint64_t mPSBuf[NINPUTS];
    for (int i = 0; i < NINPUTS; i++) {
        mPSBuf[i] = 0; uint64_t cum = 0;
        for (int x = 0; x < 8; x++) {
            maskBuf[i][x] = lcg64(seed);
            if (x < 7) {
                // simple popcount for setup
                uint64_t v = maskBuf[i][x];
                v = v - ((v>>1) & UINT64_C(0x5555555555555555));
                v = (v & UINT64_C(0x3333333333333333)) + ((v>>2) & UINT64_C(0x3333333333333333));
                v = (v + (v>>4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
                cum += (v * UINT64_C(0x0101010101010101)) >> 56;
                mPSBuf[i] |= (cum & 0x1FFu) << (9*x);
            }
        }
    }

    alignas(64) qword data[8][8], shifts[8][8], rowOffset[8][8];
    alignas(64) uint16_t prefixSum[512];
    volatile uint64_t sink = 0;

    uint64_t cyc[6] = {};

    for (int iter = 0; iter < NITER; iter++) {
        const uint64_t* mw = maskBuf[iter & (NINPUTS-1)];
        uint64_t mps = mPSBuf[iter & (NINPUTS-1)];

        uint64_t t0 = __rdtsc();

        // Step 1+2: indicator fill + Z-pass
        for (int x = 0; x < 8; x++) {
            #pragma omp simd
            for (int y = 0; y < 8; y++)
                data[x][y].ui64 = scatterLSB(mw[x] >> (y * 8));
        }
        for (int x = 0; x < 8; x++) {
            #pragma omp simd
            for (int y = 0; y < 8; y++) {
                data[x][y].ui64 += data[x][y].ui64 << 8;
                data[x][y].ui64 += data[x][y].ui64 << 16;
                data[x][y].ui64 += data[x][y].ui64 << 32;
            }
        }

        uint64_t t1 = __rdtsc();

        // Step 3a: extract shifts
        for (int x = 0; x < 8; x++) {
            #pragma omp simd
            for (int y = 0; y < 8; y++)
                shifts[x][y].ui64 = data[x][y].ui64 >> 56;
        }

        uint64_t t2 = __rdtsc();

        // Step 3b: exclusive y-prefix scan
        for (int x = 0; x < 8; x++) rowOffset[x][0].ui64 = 0;
        for (int y = 1; y < 8; y++)
            for (int x = 0; x < 8; x++)
                rowOffset[x][y].ui64 = rowOffset[x][y-1].ui64 + shifts[x][y-1].ui64;

        uint64_t t3 = __rdtsc();

        // Step 3c+3d: broadcast + add
        for (int x = 0; x < 8; x++) {
            #pragma omp simd
            for (int y = 0; y < 8; y++)
                data[x][y].ui64 += rowOffset[x][y].ui64 * kSpread;
        }

        uint64_t t4 = __rdtsc();

        // Step 4: zero-extend to uint16
        for (int x = 0; x < 8; x++)
            for (int y = 0; y < 8; y++)
                for (int z = 0; z < 8; z++)
                    prefixSum[x*64 + y*8 + z] = data[x][y].ui8[z];

        // Step 5: cross-slice add
        uint16_t xOffset[8]; xOffset[0] = 0;
        for (int x = 1; x < 8; x++)
            xOffset[x] = (uint16_t)((mps >> (9*(x-1))) & 0x1FFu);
        for (int x = 0; x < 8; x++) {
            uint16_t* p = prefixSum + x * 64;
            for (int i = 0; i < 64; i++) p[i] += xOffset[x];
        }

        uint64_t t5 = __rdtsc();

        cyc[0] += t1 - t0;
        cyc[1] += t2 - t1;
        cyc[2] += t3 - t2;
        cyc[3] += t4 - t3;
        cyc[4] += t5 - t4;
        cyc[5] += t5 - t0;
        sink ^= prefixSum[iter & 511];
    }

    const double N = NITER;
    std::printf("Step breakdown (cycles/call, avg over %d iters):\n", NITER);
    std::printf("  Step 1+2  indicator fill + Z-pass : %6.1f\n", cyc[0]/N);
    std::printf("  Step 3a   extract shifts           : %6.1f\n", cyc[1]/N);
    std::printf("  Step 3b   exclusive y-prefix scan  : %6.1f\n", cyc[2]/N);
    std::printf("  Step 3c+d broadcast + add          : %6.1f\n", cyc[3]/N);
    std::printf("  Step 4+5  zero-extend + xOffset    : %6.1f\n", cyc[4]/N);
    std::printf("  Total                              : %6.1f\n", cyc[5]/N);
    std::printf("  sink: %llu\n", (unsigned long long)sink);
    return 0;
}
