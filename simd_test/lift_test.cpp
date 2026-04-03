#include "StencilKernel.h"
#include <cmath>
#include <cstdio>

using namespace nanovdb::util;
using namespace nanovdb::math;

// ---------------------------------------------------------------------------
// SIMD wrapper — noinline to prevent constant-folding in the test
// ---------------------------------------------------------------------------
constexpr int W = 16;
using FloatSimd = Simd<float, W>;

__attribute__((noinline))
FloatSimd runSimdNormSqGrad(const FloatSimd sv[19], float dx, float isoValue)
{
    WenoStencilKernel<FloatSimd> sk(dx);
    for (int n = 0; n < 19; n++) sk[n] = sv[n];
    return sk.normSqGrad(isoValue);
}

// ---------------------------------------------------------------------------
// Reference: scalar path — same kernel class instantiated with T=float
// ---------------------------------------------------------------------------
float refNormSqGrad(const float v[19], float dx, float isoValue = 0.f)
{
    WenoStencilKernel<float> sk(dx);
    for (int n = 0; n < 19; n++) sk[n] = v[n];
    return sk.normSqGrad(isoValue);
}

// ---------------------------------------------------------------------------
int main()
{
    const float dx = 0.1f;

    // Storage: SoA layout — inData[n] holds W lane values for stencil position n
    float inData[19][W]{};
    float refValues[W][19];

    for (int i = 0; i < W; i++)
        for (int n = 0; n < 19; n++) {
            refValues[i][n] = std::sin(n * 0.3f + i * 0.5f);
            inData[n][i]    = refValues[i][n];
        }

    // Load into Simd — each FloatSimd holds one stencil position across all W lanes
    FloatSimd sv[19];
    for (int n = 0; n < 19; n++)
        sv[n] = FloatSimd(inData[n], element_aligned);

    FloatSimd result = runSimdNormSqGrad(sv, dx, 0.f);

    printf("WenoNormSqGrad full 3-axis (W=%d, dx=%.2f):\n", W, dx);
    bool allOk = true;
    for (int i = 0; i < W; i++) {
        float ref = refNormSqGrad(refValues[i], dx, 0.f);
        float got = result[i];
        bool  ok  = std::abs(got - ref) < 1e-5f * std::abs(ref) + 1e-10f;
        printf("  lane %2d: %12.6f  ref: %12.6f  %s\n", i, got, ref, ok ? "OK" : "FAIL");
        allOk &= ok;
    }
    printf("\nOverall: %s\n", allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}
