// plan1_prefix_bench.cpp
//
// Cycle-accurate benchmark for computeLinearPrefixPlan1.
//
// Method:
//   - Outer timing loop: NWARM warm-up iterations (not timed), then NITER
//     timed iterations.  Per-call cycle count is derived from rdtsc.
//   - A sink XOR of the output prevents dead-code elimination.
//   - perf stat can be layered on top for hardware counter verification.
//
// Compile:
//   g++ -O3 -march=core-avx2 -fopenmp -std=c++17 \
//       -o plan1_prefix_bench plan1_prefix_bench.cpp && ./plan1_prefix_bench
//
// Verify with perf:
//   perf stat -e cycles,instructions,r0100,r0200 ./plan1_prefix_bench

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <x86intrin.h>   // __rdtsc

union qword { uint64_t ui64; uint8_t ui8[8]; };
static constexpr uint64_t kSpread = UINT64_C(0x0101010101010101);

static inline uint64_t countOn64(uint64_t x)
{
    x =  x - ((x >> 1)  & UINT64_C(0x5555555555555555));
    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4))  & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return (x * UINT64_C(0x0101010101010101)) >> 56;
}

static inline uint64_t scatterLSB(uint64_t src)
{
    uint64_t x = src & 0xFFu;
    x = (x | (x << 14) | (x << 28) | (x << 42)) & UINT64_C(0x0003000300030003);
    x = (x | (x <<  7))                          & UINT64_C(0x0101010101010101);
    return x;
}

__attribute__((noinline))
static void computeLinearPrefixPlan1(
    const uint64_t maskWords[8],
    uint64_t       mPrefixSum,
    uint16_t       prefixSum[512])
{
    alignas(64) qword data[8][8];

    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            data[x][y].ui64 = scatterLSB(maskWords[x] >> (y * 8));
    }

    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++) {
            data[x][y].ui64 += data[x][y].ui64 << 8;
            data[x][y].ui64 += data[x][y].ui64 << 16;
            data[x][y].ui64 += data[x][y].ui64 << 32;
        }
    }

    alignas(64) qword shifts[8][8];
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            shifts[x][y].ui64 = data[x][y].ui64 >> 56;
    }

    alignas(64) qword rowOffset[8][8];
    for (int x = 0; x < 8; x++)
        rowOffset[x][0].ui64 = 0;
    for (int y = 1; y < 8; y++)
        for (int x = 0; x < 8; x++)
            rowOffset[x][y].ui64 = rowOffset[x][y-1].ui64 + shifts[x][y-1].ui64;

    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            data[x][y].ui64 += rowOffset[x][y].ui64 * kSpread;
    }

    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++)
            for (int z = 0; z < 8; z++)
                prefixSum[x*64 + y*8 + z] = data[x][y].ui8[z];

    uint16_t xOffset[8];
    xOffset[0] = 0;
    for (int x = 1; x < 8; x++)
        xOffset[x] = (uint16_t)((mPrefixSum >> (9*(x-1))) & 0x1FFu);

    for (int x = 0; x < 8; x++) {
        uint16_t* p = prefixSum + x * 64;
        for (int i = 0; i < 64; i++)
            p[i] += xOffset[x];
    }
}

static inline uint64_t lcg64(uint64_t& s)
{
    s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return s;
}

int main()
{
    const int NWARM =  1000;
    const int NITER = 50000;

    uint64_t seed = UINT64_C(0xdeadbeefcafe);

    // Pre-generate inputs so the benchmark loop itself does no PRNG work.
    const int NINPUTS = 256;
    uint64_t  maskBuf[NINPUTS][8];
    uint64_t  mPSBuf[NINPUTS];

    for (int i = 0; i < NINPUTS; i++) {
        uint64_t cumulative = 0;
        mPSBuf[i] = 0;
        for (int x = 0; x < 8; x++) {
            maskBuf[i][x] = lcg64(seed);
            if (x < 7) {
                cumulative += countOn64(maskBuf[i][x]);
                mPSBuf[i] |= (cumulative & 0x1FFu) << (9*x);
            }
        }
    }

    alignas(64) uint16_t prefixSum[512];
    volatile uint64_t sink = 0;   // prevent dead-code elimination

    // Warm-up
    for (int i = 0; i < NWARM; i++) {
        computeLinearPrefixPlan1(maskBuf[i & (NINPUTS-1)],
                                 mPSBuf[i & (NINPUTS-1)],
                                 prefixSum);
        sink ^= prefixSum[0];
    }

    // Timed run
    const uint64_t t0 = __rdtsc();
    for (int i = 0; i < NITER; i++) {
        computeLinearPrefixPlan1(maskBuf[i & (NINPUTS-1)],
                                 mPSBuf[i & (NINPUTS-1)],
                                 prefixSum);
        sink ^= prefixSum[i & 511];
    }
    const uint64_t t1 = __rdtsc();

    const double cycles_per_call = (double)(t1 - t0) / NITER;
    std::printf("computeLinearPrefixPlan1 (AVX2)\n");
    std::printf("  iterations  : %d\n", NITER);
    std::printf("  total ticks : %llu\n", (unsigned long long)(t1 - t0));
    std::printf("  cycles/call : %.1f\n", cycles_per_call);
    std::printf("  sink        : %llu\n", (unsigned long long)sink);

    return 0;
}
