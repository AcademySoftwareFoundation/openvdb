// within_word_prefix_test.cpp
//
// Correctness test for the bit-parallel z+y prefix sum algorithm.
//
// Algorithm summary:
//   Given maskWords[8] (the 8 x 64-bit words of a NanoVDB Mask<3>),
//   compute data[z][x].ui8[y] = inclusive prefix popcount of the
//   2D rectangle [0..y][0..z] within word x.
//
//   Step 1 (indicator fill):
//     data[z][x].ui64 = (maskWords[x] >> z) & kSpread    (bit(x, y, z) into byte y)
//   Step 2 (z-pass, inclusive prefix sum over z):
//     data[z][x].ui64 += data[z-1][x].ui64
//   Step 3 (y-pass, Hillis-Steele shift-and-add within each uint64):
//     data[z][x].ui64 += data[z][x].ui64 << 8
//     data[z][x].ui64 += data[z][x].ui64 << 16
//     data[z][x].ui64 += data[z][x].ui64 << 32
//
// After these steps:
//   data[z][x].ui8[y] = Σ_{y'≤y, z'≤z} bit(x, y', z')
//                     = 2D rectangle inclusive sum (NOT the linear prefix sum).
//
// Reference (linear exclusive prefix within word x):
//   countOn(maskWords[x] & ((1ULL << (y*8 + z)) - 1))
//
// This test verifies both:
//   1. That data[z][x].ui8[y] matches the 2D rectangle sum reference (expected: PASS)
//   2. Whether data[z][x].ui8[y] matches the linear exclusive prefix (expected: FAIL
//      for most voxels — the algorithm computes a different quantity)
//
// Compile:
//   g++ -O3 -march=core-avx2 -fopenmp -std=c++17 -o within_word_prefix_test within_word_prefix_test.cpp
//
// Run:
//   ./within_word_prefix_test

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// -------------------------------------------------------------------------
// qword: union allowing byte-level access to a 64-bit integer.
// ui8[y] = byte at index y (y = 0..7 = y-coordinate of the voxel within leaf).
// -------------------------------------------------------------------------
union qword {
    uint64_t ui64;
    uint8_t  ui8[8];
};

// Spread mask: bit 0 of each byte = 1, all other bits = 0.
static constexpr uint64_t kSpread = 0x0101010101010101ULL;

// -------------------------------------------------------------------------
// Software popcount (avoids hardware POPCNT under -mavx2 defeating SIMD).
// -------------------------------------------------------------------------
static inline uint64_t countOn64(uint64_t x)
{
    x =  x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (x * 0x0101010101010101ULL) >> 56;
}

// -------------------------------------------------------------------------
// Algorithm: z-pass then y-pass.
// data[z][x] where x=word index (0..7), z=bit-within-word z-coordinate (0..7).
// data[z][x].ui8[y] after completion = 2D rectangle inclusive sum over [0..y][0..z].
// -------------------------------------------------------------------------
__attribute__((noinline))
static void computeZYPrefix(const uint64_t maskWords[8], qword data[8][8])
{
    // Step 1: indicator fill — data[z][x].ui8[y] = bit(x, y, z) ∈ {0, 1}.
    // Extracts bit z from each byte of maskWords[x] via kSpread isolation.
    for (int z = 0; z < 8; z++) {
        #pragma omp simd
        for (int x = 0; x < 8; x++)
            data[z][x].ui64 = (maskWords[x] >> z) & kSpread;
    }

    // Step 2: z-pass — inclusive prefix sum over z for each (x, y).
    // data[z][x].ui8[y] after this step = Σ_{z'=0..z} bit(x, y, z')
    //   = count of active voxels in z-column (x, y) up to depth z.
    for (int z = 1; z < 8; z++) {
        #pragma omp simd
        for (int x = 0; x < 8; x++)
            data[z][x].ui64 += data[z-1][x].ui64;
    }

    // Step 3: y-pass — Hillis-Steele prefix scan within each uint64.
    // Shift-and-add within uint64: no inter-byte carry because per-byte max is 8
    // (at most 8 active voxels per z-column after the z-pass).
    // After this step: data[z][x].ui8[y] = Σ_{y'=0..y, z'=0..z} bit(x, y', z')
    for (int z = 0; z < 8; z++) {
        #pragma omp simd
        for (int x = 0; x < 8; x++) {
            data[z][x].ui64 += data[z][x].ui64 << 8;
            data[z][x].ui64 += data[z][x].ui64 << 16;
            data[z][x].ui64 += data[z][x].ui64 << 32;
        }
    }
}

// -------------------------------------------------------------------------
// Reference 1: 2D rectangle inclusive sum for (x, y, z).
// Expected to match data[z][x].ui8[y] exactly.
// -------------------------------------------------------------------------
static uint32_t ref2DRect(const uint64_t maskWords[8], int x, int y, int z)
{
    uint32_t count = 0;
    for (int yp = 0; yp <= y; yp++)
        for (int zp = 0; zp <= z; zp++)
            count += (maskWords[x] >> (yp * 8 + zp)) & 1u;
    return count;
}

// -------------------------------------------------------------------------
// Reference 2: linear exclusive prefix popcount at position i = y*8+z within word x.
// This is what NanoVDB's getValue() computes (countOn(w & (mask-1u))):
//   exclusive count of set bits before position i in maskWords[x].
// -------------------------------------------------------------------------
static uint32_t refLinearExclusive(const uint64_t maskWords[8], int x, int y, int z)
{
    const uint64_t w = maskWords[x];
    const int i = y * 8 + z;
    if (i == 0) return 0;
    const uint64_t mask = (uint64_t(1) << i) - 1ULL;
    return static_cast<uint32_t>(countOn64(w & mask));
}

// -------------------------------------------------------------------------
// Minimal LCG PRNG for random inputs.
// -------------------------------------------------------------------------
static inline uint64_t lcg64(uint64_t& state)
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
}

int main()
{
    const int N_TESTS = 1000;
    uint64_t seed = 0xdeadbeefcafeULL;

    int pass2D   = 0;
    int fail2D   = 0;
    int passLin  = 0;
    int failLin  = 0;

    // Track first discrepancy vs linear reference for reporting
    bool firstLinearMismatch = false;
    int  firstMismatchX = -1, firstMismatchY = -1, firstMismatchZ = -1;
    uint32_t firstAlgo = 0, firstRef = 0;
    uint64_t firstMaskWords[8] = {};

    alignas(64) qword data[8][8];

    for (int t = 0; t < N_TESTS; t++) {
        uint64_t maskWords[8];
        for (int x = 0; x < 8; x++)
            maskWords[x] = lcg64(seed);

        computeZYPrefix(maskWords, data);

        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                for (int y = 0; y < 8; y++) {
                    const uint32_t algo = data[z][x].ui8[y];

                    // Test 1: vs 2D rectangle inclusive sum
                    const uint32_t r2D = ref2DRect(maskWords, x, y, z);
                    if (algo == r2D) pass2D++;
                    else             fail2D++;

                    // Test 2: vs linear exclusive prefix (inclusive = exclusive + active_bit)
                    // Our algo is inclusive (2D rect), ref is linear exclusive — they differ.
                    // To compare apples-to-apples we compare algo against linear inclusive:
                    //   linearInclusive = linearExclusive + bit(x, y, z)
                    const uint32_t linExcl = refLinearExclusive(maskWords, x, y, z);
                    const uint32_t linIncl = linExcl + ((maskWords[x] >> (y*8+z)) & 1u);
                    if (algo == linIncl) passLin++;
                    else {
                        failLin++;
                        if (!firstLinearMismatch) {
                            firstLinearMismatch = true;
                            firstMismatchX = x; firstMismatchY = y; firstMismatchZ = z;
                            firstAlgo = algo; firstRef = linIncl;
                            for (int i = 0; i < 8; i++) firstMaskWords[i] = maskWords[i];
                        }
                    }
                }
            }
        }
    }

    const int total = N_TESTS * 8 * 8 * 8;

    std::printf("=== 2D rectangle inclusive sum (expected: all PASS) ===\n");
    std::printf("  PASS: %d / %d\n", pass2D, total);
    std::printf("  FAIL: %d / %d\n", fail2D, total);
    std::printf("\n");

    std::printf("=== Linear inclusive prefix (algo vs linear: expected FAIL if algo != linear prefix) ===\n");
    std::printf("  PASS: %d / %d\n", passLin, total);
    std::printf("  FAIL: %d / %d\n", failLin, total);
    if (firstLinearMismatch) {
        std::printf("\n  First mismatch at (x=%d, y=%d, z=%d):\n", firstMismatchX, firstMismatchY, firstMismatchZ);
        std::printf("    algo (2D rect incl) = %u\n", firstAlgo);
        std::printf("    linear incl ref     = %u\n", firstRef);
        std::printf("    maskWords[%d] = 0x%016llx\n", firstMismatchX, (unsigned long long)firstMaskWords[firstMismatchX]);
    }
    std::printf("\n");

    // Summary
    const bool ok = (fail2D == 0);
    std::printf("=== Summary ===\n");
    std::printf("  2D rectangle test: %s\n", ok ? "PASS" : "FAIL");
    std::printf("  Algorithm computes 2D rectangle inclusive sum, NOT linear prefix sum.\n");
    if (failLin > 0)
        std::printf("  %d positions differ from linear inclusive prefix — further steps needed.\n", failLin);
    else
        std::printf("  Unexpectedly: algorithm matches linear inclusive for all tested positions.\n");

    return ok ? 0 : 1;
}
