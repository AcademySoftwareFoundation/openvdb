// transpose_test.cpp
//
// Correctness test for nanovdb::util::transposeBytes8x8 and
// nanovdb::util::transposeBits8x8 declared in nanovdb/util/Transpose.h.
//
// Compile (SIMD path):
//   g++ -O3 -march=core-avx2 -DNANOVDB_USE_INTRINSICS -std=c++17 \
//       -I ../nanovdb -o transpose_test transpose_test.cpp && ./transpose_test
//
// Compile (scalar path):
//   g++ -O3 -march=core-avx2               -std=c++17 \
//       -I ../nanovdb -o transpose_test transpose_test.cpp && ./transpose_test

#include <nanovdb/util/Transpose.h>

#include <cstdio>
#include <cstdlib>

// -------------------------------------------------------------------------
// Simple LCG PRNG
// -------------------------------------------------------------------------
static inline uint64_t lcg64(uint64_t& s)
{
    s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return s;
}

// -------------------------------------------------------------------------
// Reference: scalar byte-matrix transpose
// -------------------------------------------------------------------------
static void refTransposeBytes(const uint64_t src[8], uint64_t dst[8])
{
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    uint8_t*       d = reinterpret_cast<uint8_t*>(dst);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            d[i * 8 + j] = s[j * 8 + i];
}

// -------------------------------------------------------------------------
// Reference: scalar 8×8 bit-matrix transpose (extract-bit loop)
// -------------------------------------------------------------------------
static uint64_t refTransposeBits(uint64_t v)
{
    uint64_t out = 0;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) {
            const int srcBit = r * 8 + c;
            const int dstBit = c * 8 + r;
            if ((v >> srcBit) & 1u)
                out |= UINT64_C(1) << dstBit;
        }
    return out;
}

// -------------------------------------------------------------------------
// Roundtrip property: transpose(transpose(x)) == x
// -------------------------------------------------------------------------
static bool bytesRoundtrip(const uint64_t src[8])
{
    alignas(16) uint64_t tmp[8], back[8];
    nanovdb::util::transposeBytes8x8(src, tmp);
    nanovdb::util::transposeBytes8x8(tmp, back);
    for (int i = 0; i < 8; ++i)
        if (back[i] != src[i]) return false;
    return true;
}

int main()
{
    const int N = 10000;
    uint64_t  seed = UINT64_C(0xdeadbeefcafe);

    int passBytes = 0, failBytes = 0;
    int passBits  = 0, failBits  = 0;
    int passRound = 0, failRound = 0;

    alignas(16) uint64_t src[8], dstSIMD[8], refB[8];

    for (int t = 0; t < N; ++t) {
        for (int x = 0; x < 8; ++x)
            src[x] = lcg64(seed);

        // --- transposeBytes8x8 vs reference ---
        nanovdb::util::transposeBytes8x8(src, dstSIMD);
        refTransposeBytes(src, refB);
        bool okBytes = true;
        for (int i = 0; i < 8; ++i)
            if (dstSIMD[i] != refB[i]) { okBytes = false; break; }
        if (okBytes) ++passBytes; else ++failBytes;

        // --- transposeBytes8x8 roundtrip ---
        if (bytesRoundtrip(src)) ++passRound; else ++failRound;

        // --- transposeBits8x8 vs reference, per-word ---
        bool okBits = true;
        for (int x = 0; x < 8; ++x) {
            if (nanovdb::util::transposeBits8x8(src[x]) != refTransposeBits(src[x])) {
                okBits = false; break;
            }
        }
        if (okBits) ++passBits; else ++failBits;
    }

    std::printf("transposeBytes8x8  vs reference : PASS=%d FAIL=%d / %d\n", passBytes, failBytes, N);
    std::printf("transposeBytes8x8  roundtrip    : PASS=%d FAIL=%d / %d\n", passRound, failRound, N);
    std::printf("transposeBits8x8   vs reference : PASS=%d FAIL=%d / %d\n", passBits,  failBits,  N);

    const bool ok = (failBytes == 0 && failRound == 0 && failBits == 0);
    std::printf("%s\n", ok ? "ALL PASS" : "*** FAILURES ***");
    return ok ? 0 : 1;
}
