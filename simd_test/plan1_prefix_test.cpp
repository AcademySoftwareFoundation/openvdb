// plan1_prefix_test.cpp
//
// Correctness test for the Plan #1 linear inclusive prefix sum algorithm.
//
// Layout: data[x][y].ui8[z]
//   x = word index       (0..7) — which of the 8 x 64-bit mask words
//   y = byte-within-word (0..7) — slow index within the word
//   z = bit-within-byte  (0..7) — fast index (byte axis inside the uint64)
//
// NanoVDB linear voxel index within a leaf: i = x*64 + y*8 + z
//
// Output: uint16_t prefixSum[512]
//   prefixSum[i] = linear inclusive prefix popcount at position i
//                = number of active voxels at positions 0..i (inclusive, within the leaf)
//   To obtain the exclusive prefix (matching getValue() - mOffset), subtract the active bit.
//
// Algorithm steps:
//   1. Indicator fill:  data[x][y].ui8[z] = I[x][y][z]            (scalar triple loop)
//   2. Z-pass:          Hillis-Steele within-uint64                -> z-inclusive prefix
//   3. Y-pass:          exclusive row-prefix scan + broadcast+add  -> linear prefix within word
//      3a. Extract row popcounts: shifts[x][y].ui64 = data[x][y].ui64 >> 56
//      3b. Exclusive y-prefix scan of shifts (sequential over y, independent over x)
//      3c. Broadcast byte 0 of each rowOffset to all 8 bytes (multiply by kSpread)
//      3d. Add to data[x][y].ui64
//   4. Zero-extend:     data[x][y].ui8[z] -> uint16_t prefixSum[x*64 + y*8 + z]
//   5. Cross-slice add: add xOffset[x] (decoded from mPrefixSum) to each 64-entry slice
//
// Key property of Plan #1 vs the original data[z][x] algorithm:
//   The Z-pass IS the fast within-uint64 Hillis-Steele (since z is the byte index).
//   The Y-pass adds the complete preceding-row counts directly as a scalar broadcast —
//   no 2D rectangle detour and no rectangle→linear fixup needed.
//
// Reference:
//   refLinearInclusive(x, y, z) = xOffset[x]
//                                + countOn64(maskWords[x] & ((2ULL << (y*8+z)) - 1))
//   Safe mask form: (2ULL << 63) wraps to 0 via unsigned overflow -> -1u = 0xFFFFFFFFFFFFFFFF.
//
// Compile:
//   g++ -O3 -march=core-avx2 -fopenmp -std=c++17 \
//       -o plan1_prefix_test plan1_prefix_test.cpp && ./plan1_prefix_test

#include <cstdint>
#include <cstdio>
#include <cstdlib>

union qword { uint64_t ui64; uint8_t ui8[8]; };
static constexpr uint64_t kSpread = UINT64_C(0x0101010101010101);

// Software popcount — avoids hardware popcntl defeating SIMD under -mavx2.
static inline uint64_t countOn64(uint64_t x)
{
    x =  x - ((x >> 1)  & UINT64_C(0x5555555555555555));
    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4))  & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return (x * UINT64_C(0x0101010101010101)) >> 56;
}

// -------------------------------------------------------------------------
// Main algorithm — Plan #1
// -------------------------------------------------------------------------
__attribute__((noinline))
static void computeLinearPrefixPlan1(
    const uint64_t maskWords[8],  // 8 x 64-bit valueMask words
    uint64_t       mPrefixSum,    // packed 7 x 9-bit exclusive cross-slice popcounts
    uint16_t       prefixSum[512])
{
    alignas(64) qword data[8][8];

    // ------------------------------------------------------------------
    // Step 1: indicator fill
    // data[x][y].ui8[z] = bit (y*8+z) of maskWords[x] = I[x][y][z] ∈ {0,1}
    //
    // For each word x: extract byte y, then unpack its 8 bits into 8 bytes.
    // TODO: replace with a vectorised bit-unpack in an optimisation pass.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) {
            const uint8_t b = (uint8_t)(maskWords[x] >> (y * 8));
            for (int z = 0; z < 8; z++)
                data[x][y].ui8[z] = (b >> z) & 1u;
        }

    // ------------------------------------------------------------------
    // Step 2: Z-pass — Hillis-Steele inclusive prefix sum over z.
    //
    // z is the byte index within each uint64, so the scan is performed
    // by three shift-and-add operations inside each uint64.
    // No inter-byte carry: values are ≤ 1 entering, ≤ 8 after the pass.
    //
    // After: data[x][y].ui8[z] = Σ_{z'=0..z} I[x][y][z']
    // Bonus: data[x][y].ui8[7] = full row y popcount — available for free.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++) {
            data[x][y].ui64 += data[x][y].ui64 << 8;
            data[x][y].ui64 += data[x][y].ui64 << 16;
            data[x][y].ui64 += data[x][y].ui64 << 32;
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Y-pass — produce the linear prefix sum within each x-slice.
    //
    // After the Z-pass, data[x][y].ui8[z] holds only the within-row-y
    // contribution.  To get the full linear prefix we must add the total
    // popcount of all preceding rows y' < y.  This is done without a
    // 2D-rectangle detour by computing the exclusive row-prefix scan
    // directly and broadcasting it into all z bytes at once.
    //
    // Sub-steps:
    //   3a. Extract: shifts[x][y].ui64 = data[x][y].ui64 >> 56
    //       byte 0 of each shifts entry = full row y popcount (≤ 8 ≤ 255).
    //       All other bytes are zero.
    //
    //   3b. Exclusive y-prefix scan of shifts:
    //       rowOffset[x][y].ui64 = Σ_{y'<y} shifts[x][y'].ui64
    //       Sequential over y (loop-carried dependency in y);
    //       independent across x — with a transposed [y][x] layout the
    //       x-loop would be unit-stride and vectorisable via AVX2
    //       (4 × uint64 per register) or AVX-512 (8 × uint64 per register).
    //
    //   3c+3d. Broadcast byte 0 to all 8 bytes (multiply by kSpread) and add:
    //       data[x][y].ui64 += rowOffset[x][y].ui64 * kSpread
    //
    // After: data[x][y].ui8[z] = Σ_{y'<y} Σ_{z'=0..7} I[x][y'][z']
    //                           + Σ_{z'=0..z} I[x][y][z']
    //      = linear inclusive prefix sum within word x at position (y, z).
    // ------------------------------------------------------------------

    // 3a: extract row popcounts into byte 0 of shifts
    alignas(64) qword shifts[8][8];
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            shifts[x][y].ui64 = data[x][y].ui64 >> 56;
    }

    // 3b: exclusive y-prefix scan
    alignas(64) qword rowOffset[8][8];
    for (int x = 0; x < 8; x++)
        rowOffset[x][0].ui64 = 0;
    for (int y = 1; y < 8; y++)
        for (int x = 0; x < 8; x++)   // independent over x at each y-step
            rowOffset[x][y].ui64 = rowOffset[x][y-1].ui64 + shifts[x][y-1].ui64;

    // 3c+3d: broadcast and add
    for (int x = 0; x < 8; x++) {
        #pragma omp simd
        for (int y = 0; y < 8; y++)
            data[x][y].ui64 += rowOffset[x][y].ui64 * kSpread;
    }

    // ------------------------------------------------------------------
    // Step 4: zero-extend bytes to uint16_t in linear index order.
    // data[x][y].ui8[z] -> prefixSum[x*64 + y*8 + z]
    // Values ≤ 64 (at most 64 active voxels per 64-bit word): safe to
    // zero-extend.  The output is already in linear order — no reordering.
    // Vectorisable as vpmovzxbw over 64 contiguous bytes per x-slice.
    // ------------------------------------------------------------------
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++)
            for (int z = 0; z < 8; z++)
                prefixSum[x*64 + y*8 + z] = data[x][y].ui8[z];

    // ------------------------------------------------------------------
    // Step 5: add cross-slice offsets decoded from mPrefixSum.
    //
    // mPrefixSum encodes 7 exclusive cumulative popcounts as 9-bit fields:
    //   xOffset[0] = 0
    //   xOffset[x] = (mPrefixSum >> 9*(x-1)) & 0x1FF  for x = 1..7
    //
    // Add xOffset[x] to all 64 uint16_t entries of slice x.
    // Each slice is 128 contiguous bytes → 4 AVX2 vpbroadcastw+vpaddw ops.
    // ------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// Reference: linear inclusive prefix popcount for voxel (x, y, z).
//
// Counts all active voxels at positions 0 .. y*8+z within word x,
// plus all active voxels in words 0 .. x-1 (via xOffset).
//
// Safe mask: (2ULL << bitPos) - 1u  includes bits 0..bitPos.
// At bitPos=63: (2ULL<<63) wraps to 0 (unsigned), -1u = 0xFFFFFFFFFFFFFFFF. ✓
// -------------------------------------------------------------------------
static uint16_t refLinearInclusive(const uint64_t maskWords[8],
                                    const uint16_t xOffset[8],
                                    int x, int y, int z)
{
    const int      bitPos = y * 8 + z;
    const uint64_t mask   = (UINT64_C(2) << bitPos) - 1u;
    return (uint16_t)(xOffset[x] + countOn64(maskWords[x] & mask));
}

// -------------------------------------------------------------------------
// LCG PRNG
// -------------------------------------------------------------------------
static inline uint64_t lcg64(uint64_t& s)
{
    s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return s;
}

int main()
{
    const int N_TESTS = 1000;
    uint64_t  seed    = UINT64_C(0xdeadbeefcafe);
    int pass = 0, fail = 0;

    // Track first mismatch for diagnostics
    bool     firstFail = false;
    int      failX = 0, failY = 0, failZ = 0;
    uint16_t failGot = 0, failRef = 0;

    uint16_t prefixSum[512];

    for (int t = 0; t < N_TESTS; t++) {
        uint64_t maskWords[8];
        for (int x = 0; x < 8; x++)
            maskWords[x] = lcg64(seed);

        // Build mPrefixSum and xOffset from the random masks.
        // mPrefixSum encodes 7 × 9-bit exclusive prefix popcounts at x=1..7.
        uint64_t mPrefixSum = 0;
        uint16_t xOffset[8];
        xOffset[0] = 0;
        uint64_t cumulative = 0;
        for (int x = 0; x < 7; x++) {
            cumulative += countOn64(maskWords[x]);
            xOffset[x+1] = (uint16_t)cumulative;
            mPrefixSum |= (cumulative & 0x1FFu) << (9*x);
        }

        computeLinearPrefixPlan1(maskWords, mPrefixSum, prefixSum);

        for (int x = 0; x < 8; x++)
            for (int y = 0; y < 8; y++)
                for (int z = 0; z < 8; z++) {
                    const uint16_t got = prefixSum[x*64 + y*8 + z];
                    const uint16_t ref = refLinearInclusive(maskWords, xOffset, x, y, z);
                    if (got == ref) {
                        pass++;
                    } else {
                        fail++;
                        if (!firstFail) {
                            firstFail = true;
                            failX = x; failY = y; failZ = z;
                            failGot = got; failRef = ref;
                        }
                    }
                }
    }

    const int total = N_TESTS * 512;
    std::printf("Plan #1 linear inclusive prefix: PASS=%d FAIL=%d / %d\n",
                pass, fail, total);
    if (fail > 0) {
        std::printf("  First mismatch at (x=%d, y=%d, z=%d): got=%u ref=%u\n",
                    failX, failY, failZ, failGot, failRef);
    }
    std::printf("%s\n", fail == 0 ? "ALL PASS" : "*** FAILURES ***");
    return fail == 0 ? 0 : 1;
}
