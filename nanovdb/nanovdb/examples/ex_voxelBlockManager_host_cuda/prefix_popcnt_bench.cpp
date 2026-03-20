// prefix_popcnt_bench.cpp
//
// Isolated benchmark for the prefixCountRealigned[32][16] computation.
// Input:  uint32_t maskWords[16]  (the 512-bit valueMask of one leaf, as 16 x 32-bit words)
// Output: uint32_t prefixCountRealigned[32][16]
//   [step][lane] = inclusive prefix popcount of bits 0..step in maskWords[lane],
//                  then shifted by baseOffset[lane] to give a *global* voxel index prefix.
//
// Two implementations:
//   computePrefixPopcnt      — auto-vectorised (relies on software Hamming weight + #pragma omp simd)
//   computePrefixPopcntAVX2  — explicit AVX2 intrinsics (vpshufb nibble-table popcount)
//
// Compile:
//   g++ -O3 -mavx2 -fopenmp -o prefix_popcnt_bench prefix_popcnt_bench.cpp
//
// Inspect assembly:
//   g++ -O3 -mavx2 -fopenmp -S -o prefix_popcnt_bench.s prefix_popcnt_bench.cpp
//   objdump -d prefix_popcnt_bench | less
//
// Profile:
//   perf stat ./prefix_popcnt_bench
//   perf record -g ./prefix_popcnt_bench && perf report

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>

// -------------------------------------------------------------------------
// Software popcount (Hamming weight) for uint32_t.
// Uses only integer arithmetic — promotes auto-vectorisation to SIMD
// rather than scalar POPCNT, enabling "vertical SIMD" over lanes.
// Note: with -mavx2 (and without -mno-popcnt) the compiler will replace
// this with the scalar hardware popcntl instruction, defeating SIMD.
// -------------------------------------------------------------------------
static inline uint32_t popcount32(uint32_t x)
{
    x =  x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return (x * 0x01010101u) >> 24;
}

// -------------------------------------------------------------------------
// AUTO-VECTORISED version — requires -mno-popcnt to avoid scalar fallback.
// -------------------------------------------------------------------------
__attribute__((noinline))
static uint32_t computePrefixPopcnt(const uint32_t maskWords[16],
                                    uint32_t prefixCountRealigned[32][16])
{
    // Phase 1: inclusive prefix popcount within each 32-bit word.
    //
    // Safe inclusive-prefix mask: (uint32_t(2) << step) - 1u
    //   At step=31 unsigned overflow gives 0xFFFFFFFF (avoids UB of 1u<<32).
    //   At step=31 the result equals the full per-word popcount (used by Phase 2).
    for (int step = 0; step < 32; step++) {
        const uint32_t mask = (uint32_t(2) << step) - 1u;
        #pragma omp simd
        for (int lane = 0; lane < 16; lane++)
            prefixCountRealigned[step][lane] = popcount32(maskWords[lane] & mask);
    }

    // Phase 2: convert per-word prefix counts to global prefix counts.
    //
    // baseOffset[lane] = exclusive prefix scan of row[31] (= per-word total popcnts).
    // Adding baseOffset[lane] to every row gives global sequential voxel offsets.
    uint32_t baseOffset[16];
    baseOffset[0] = 0;
    for (int lane = 1; lane < 16; lane++)
        baseOffset[lane] = baseOffset[lane - 1] + prefixCountRealigned[31][lane - 1];

    for (int step = 0; step < 32; step++) {
        #pragma omp simd
        for (int lane = 0; lane < 16; lane++)
            prefixCountRealigned[step][lane] += baseOffset[lane];
    }

    return prefixCountRealigned[31][15]; // prevent DCE
}

// -------------------------------------------------------------------------
// AVX2 explicit-intrinsics version.
//
// Popcount strategy: vpshufb nibble-table lookup.
//   For each 32-bit element:
//     lo  = v & 0x0f                    (low  nibble per byte)
//     hi  = (v >> 4) & 0x0f             (high nibble per byte)
//     cnt = shuffle(lut, lo) + shuffle(lut, hi)   (byte popcounts via LUT)
//     sum = madd(maddubs(cnt, 1), 1)    (sum 4 bytes → 32-bit lane popcount)
//
//   vpshufb: 1-cycle throughput, 3-cycle latency (vs ~10 instructions for Hamming).
//   Processes 8 uint32 lanes per __m256i; two __m256i cover all 16 lanes.
//
// Phase 1: maskWords loaded once outside the step loop (constant across steps).
// Phase 2: scalar prefix scan for baseOffset (16 adds, negligible), then
//          explicit vpaddd to add baseOffset to every row.
// -------------------------------------------------------------------------
__attribute__((noinline))
static uint32_t computePrefixPopcntAVX2(const uint32_t maskWords[16],
                                         uint32_t prefixCountRealigned[32][16])
{
    // Nibble popcount LUT — same 16-entry table replicated in both 128-bit lanes.
    // lut[i] = popcount(i) for i in 0..15.
    const __m256i lut = _mm256_set_epi8(
        4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
        4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0);
    const __m256i low4  = _mm256_set1_epi8(0x0f);  // nibble mask
    const __m256i ones8 = _mm256_set1_epi8(1);      // for maddubs: sum byte pairs -> 16-bit
    const __m256i ones16= _mm256_set1_epi16(1);     // for madd:    sum 16-bit pairs -> 32-bit

    // Load all 16 mask words once — they are constant across all 32 steps.
    const __m256i w0 = _mm256_loadu_si256((const __m256i*)(maskWords));     // lanes  0..7
    const __m256i w1 = _mm256_loadu_si256((const __m256i*)(maskWords + 8)); // lanes 8..15

    // -----------------------------------------------------------------------
    // Phase 1: for each bitstep, AND with inclusive-prefix mask, then popcount.
    // -----------------------------------------------------------------------
    for (int step = 0; step < 32; step++) {
        // Broadcast the step mask across all 8 lanes of each YMM.
        // Cast to int: up to 0xFFFFFFFF is fine for bitwise ops.
        const __m256i mask = _mm256_set1_epi32((int)((uint32_t(2) << step) - 1u));

        // AND with prefix mask
        __m256i a0 = _mm256_and_si256(w0, mask);
        __m256i a1 = _mm256_and_si256(w1, mask);

        // vpshufb popcount for a0 (lanes 0..7)
        __m256i lo0  = _mm256_and_si256(a0, low4);
        __m256i hi0  = _mm256_and_si256(_mm256_srli_epi16(a0, 4), low4);
        __m256i cnt0 = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo0),
                                        _mm256_shuffle_epi8(lut, hi0));
        __m256i s0   = _mm256_madd_epi16(_mm256_maddubs_epi16(cnt0, ones8), ones16);

        // vpshufb popcount for a1 (lanes 8..15)
        __m256i lo1  = _mm256_and_si256(a1, low4);
        __m256i hi1  = _mm256_and_si256(_mm256_srli_epi16(a1, 4), low4);
        __m256i cnt1 = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo1),
                                        _mm256_shuffle_epi8(lut, hi1));
        __m256i s1   = _mm256_madd_epi16(_mm256_maddubs_epi16(cnt1, ones8), ones16);

        _mm256_store_si256((__m256i*)(prefixCountRealigned[step]),     s0);
        _mm256_store_si256((__m256i*)(prefixCountRealigned[step] + 8), s1);
    }

    // -----------------------------------------------------------------------
    // Phase 2: convert per-word prefix counts to global prefix counts.
    //
    // Scalar prefix scan for baseOffset (16 elements — effectively free).
    // Then two vpaddd per row to add baseOffset to all 16 lanes at once.
    // -----------------------------------------------------------------------
    uint32_t baseOffset[16];
    baseOffset[0] = 0;
    for (int lane = 1; lane < 16; lane++)
        baseOffset[lane] = baseOffset[lane - 1] + prefixCountRealigned[31][lane - 1];

    const __m256i base0 = _mm256_loadu_si256((const __m256i*)(baseOffset));
    const __m256i base1 = _mm256_loadu_si256((const __m256i*)(baseOffset + 8));

    for (int step = 0; step < 32; step++) {
        __m256i r0 = _mm256_load_si256((const __m256i*)(prefixCountRealigned[step]));
        __m256i r1 = _mm256_load_si256((const __m256i*)(prefixCountRealigned[step] + 8));
        _mm256_store_si256((__m256i*)(prefixCountRealigned[step]),
                           _mm256_add_epi32(r0, base0));
        _mm256_store_si256((__m256i*)(prefixCountRealigned[step] + 8),
                           _mm256_add_epi32(r1, base1));
    }

    return prefixCountRealigned[31][15]; // prevent DCE
}

// -------------------------------------------------------------------------
// Minimal LCG PRNG — runtime-unknown inputs prevent the compiler from
// constant-folding the kernel across loop iterations.
// -------------------------------------------------------------------------
static inline uint32_t lcg(uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

static double bench(uint32_t& seed, int N_BLOCKS, int N_RUNS,
                    uint32_t prefixCountRealigned[32][16],
                    volatile uint32_t& dummy,
                    bool useAVX2)
{
    struct timespec t0, t1;
    double minMs = 1e30;
    for (int r = 0; r < N_RUNS; r++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int b = 0; b < N_BLOCKS; b++) {
            uint32_t maskWords[16];
            for (int lane = 0; lane < 16; lane++) maskWords[lane] = lcg(seed);
            if (useAVX2)
                dummy += computePrefixPopcntAVX2(maskWords, prefixCountRealigned);
            else
                dummy += computePrefixPopcnt(maskWords, prefixCountRealigned);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = (t1.tv_sec - t0.tv_sec) * 1e3
                  + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
        if (ms < minMs) minMs = ms;
        std::printf("  run %2d: %.3f ms\n", r, ms);
    }
    return minMs;
}

int main(int argc, char** argv)
{
    uint32_t seed = (argc > 1) ? (uint32_t)std::atoi(argv[1]) : 0xdeadbeef;

    const int N_BLOCKS = 1 << 20; // 1 M blocks  (= 128 M voxels at full occupancy)
    const int N_WARMUP = 4;
    const int N_RUNS   = 10;

    alignas(64) uint32_t prefixCountRealigned[32][16];
    volatile uint32_t dummy = 0;

    // --- auto-vectorised ---
    std::printf("=== auto-vectorised (popcount32 + #pragma omp simd) ===\n");
    for (int i = 0; i < N_WARMUP; i++) {
        uint32_t maskWords[16];
        for (int lane = 0; lane < 16; lane++) maskWords[lane] = lcg(seed);
        dummy += computePrefixPopcnt(maskWords, prefixCountRealigned);
    }
    double msAuto = bench(seed, N_BLOCKS, N_RUNS, prefixCountRealigned, dummy, false);
    std::printf("min: %.3f ms  (%.1f ns/block)\n\n", msAuto, msAuto * 1e6 / N_BLOCKS);

    // --- AVX2 explicit intrinsics ---
    std::printf("=== AVX2 explicit intrinsics (vpshufb nibble-table) ===\n");
    for (int i = 0; i < N_WARMUP; i++) {
        uint32_t maskWords[16];
        for (int lane = 0; lane < 16; lane++) maskWords[lane] = lcg(seed);
        dummy += computePrefixPopcntAVX2(maskWords, prefixCountRealigned);
    }
    double msAVX2 = bench(seed, N_BLOCKS, N_RUNS, prefixCountRealigned, dummy, true);
    std::printf("min: %.3f ms  (%.1f ns/block)\n\n", msAVX2, msAVX2 * 1e6 / N_BLOCKS);

    std::printf("speedup: %.2fx  (dummy=%u)\n", msAuto / msAVX2, (unsigned)dummy);
    return 0;
}
