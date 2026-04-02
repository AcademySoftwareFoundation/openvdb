#include <tuple>
#include <array>
#include <utility>
#include <cmath>
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Type transformation: replace each T in a tuple with std::array<T, W>
// ---------------------------------------------------------------------------
template<typename TupleT, int W> struct ToSimdTuple;
template<int W, typename... Ts>
struct ToSimdTuple<std::tuple<Ts...>, W> {
    using type = std::tuple<std::array<Ts, W>...>;
};

// ---------------------------------------------------------------------------
// extractSlice: given a tuple of arrays, return a tuple of the i-th elements
// ---------------------------------------------------------------------------
template<typename SimdTupleT, std::size_t... Is>
auto extractSlice(const SimdTupleT& t, int i, std::index_sequence<Is...>) {
    return std::make_tuple(std::get<Is>(t)[i]...);
}

// ---------------------------------------------------------------------------
// storeSlice: write a scalar tuple into the i-th slot of a SIMD tuple
// ---------------------------------------------------------------------------
template<typename SimdTupleT, typename ScalarTupleT, std::size_t... Is>
void storeSlice(SimdTupleT& t, int i, const ScalarTupleT& s, std::index_sequence<Is...>) {
    ((std::get<Is>(t)[i] = std::get<Is>(s)), ...);
}

// ---------------------------------------------------------------------------
// liftToSimd: lift a scalar tuple->tuple function to operate on W-wide arrays
// ---------------------------------------------------------------------------
template<int W, typename ScalarFn>
auto liftToSimd(ScalarFn f) {
    return [f](const auto& simdIn, auto& simdOut) {
        constexpr auto inSize  = std::tuple_size_v<std::decay_t<decltype(simdIn)>>;
        constexpr auto outSize = std::tuple_size_v<std::decay_t<decltype(simdOut)>>;
        for (int i = 0; i < W; i++) {
            auto scalarIn  = extractSlice(simdIn,  i, std::make_index_sequence<inSize>{});
            auto scalarOut = f(scalarIn);
            storeSlice(simdOut, i, scalarOut, std::make_index_sequence<outSize>{});
        }
    };
}

// ---------------------------------------------------------------------------
// WENO5 upwind interpolation (from Stencils.h)
// ---------------------------------------------------------------------------
inline float weno5(float v1, float v2, float v3, float v4, float v5, float dx2 = 1.f)
{
    static constexpr float C = 13.f / 12.f;
    const float eps = 1.0e-6f * dx2;
    const float A1 = 0.1f / ((C*(v1-2*v2+v3)*(v1-2*v2+v3) + 0.25f*(v1-4*v2+3*v3)*(v1-4*v2+3*v3) + eps) *
                              (C*(v1-2*v2+v3)*(v1-2*v2+v3) + 0.25f*(v1-4*v2+3*v3)*(v1-4*v2+3*v3) + eps));
    const float A2 = 0.6f / ((C*(v2-2*v3+v4)*(v2-2*v3+v4) + 0.25f*(v2-v4)*(v2-v4) + eps) *
                              (C*(v2-2*v3+v4)*(v2-2*v3+v4) + 0.25f*(v2-v4)*(v2-v4) + eps));
    const float A3 = 0.3f / ((C*(v3-2*v4+v5)*(v3-2*v4+v5) + 0.25f*(3*v3-4*v4+v5)*(3*v3-4*v4+v5) + eps) *
                              (C*(v3-2*v4+v5)*(v3-2*v4+v5) + 0.25f*(3*v3-4*v4+v5)*(3*v3-4*v4+v5) + eps));
    return (A1*(2*v1 - 7*v2 + 11*v3) + A2*(5*v3 - v2 + 2*v4) + A3*(2*v3 + 5*v4 - v5)) / (6*(A1+A2+A3));
}

// ---------------------------------------------------------------------------
// GodunovsNormSqrd — blend formulation
//
// Computes both the outside and inside squared terms for each axis via
// ternary blend on isOutside.  The intent is that each ternary compiles to
// vcmpps + vblendvps rather than a branch, but GCC's vectorizer currently
// still reports "control flow in loop" even when isOutside is a compile-time
// constant.  See INVESTIGATION.md for the full vectorization story.
// ---------------------------------------------------------------------------
inline float godunovsNormSqrd(bool isOutside,
    float dP_xm, float dP_xp,
    float dP_ym, float dP_yp,
    float dP_zm, float dP_zp)
{
    float xm = isOutside ? std::max( dP_xm, 0.f) * std::max( dP_xm, 0.f)
                         : std::max(-dP_xm, 0.f) * std::max(-dP_xm, 0.f);
    float xp = isOutside ? std::max(-dP_xp, 0.f) * std::max(-dP_xp, 0.f)
                         : std::max( dP_xp, 0.f) * std::max( dP_xp, 0.f);
    float ym = isOutside ? std::max( dP_ym, 0.f) * std::max( dP_ym, 0.f)
                         : std::max(-dP_ym, 0.f) * std::max(-dP_ym, 0.f);
    float yp = isOutside ? std::max(-dP_yp, 0.f) * std::max(-dP_yp, 0.f)
                         : std::max( dP_yp, 0.f) * std::max( dP_yp, 0.f);
    float zm = isOutside ? std::max( dP_zm, 0.f) * std::max( dP_zm, 0.f)
                         : std::max(-dP_zm, 0.f) * std::max(-dP_zm, 0.f);
    float zp = isOutside ? std::max(-dP_zp, 0.f) * std::max(-dP_zp, 0.f)
                         : std::max( dP_zp, 0.f) * std::max( dP_zp, 0.f);
    return std::max(xm, xp) + std::max(ym, yp) + std::max(zm, zp);
}

// ---------------------------------------------------------------------------
// WenoNormSqGrad scalar lambda
//
// Input tuple indices follow WenoPt<i,j,k>::idx:
//   0        = center  (0, 0, 0)
//   1, 2, 3  = x-axis  (-3,-2,-1)
//   4, 5, 6  = x-axis  ( 1, 2, 3)
//   7, 8, 9  = y-axis  (-3,-2,-1)
//  10,11,12  = y-axis  ( 1, 2, 3)
//  13,14,15  = z-axis  (-3,-2,-1)
//  16,17,18  = z-axis  ( 1, 2, 3)
// ---------------------------------------------------------------------------
using WenoIn  = std::tuple<float,float,float,float,float,float,float,
                           float,float,float,float,float,float,
                           float,float,float,float,float,float>;
using WenoOut = std::tuple<float>;

// dx2 = dx^2 (scale for WENO eps), invDx2 = 1/dx^2, isoValue = level set iso
auto makeNormSqGrad(float dx2, float invDx2, float isoValue = 0.f) {
    return [=](WenoIn in) -> WenoOut {
        const float
            v0  = std::get< 0>(in),
            v1  = std::get< 1>(in), v2  = std::get< 2>(in), v3  = std::get< 3>(in),
            v4  = std::get< 4>(in), v5  = std::get< 5>(in), v6  = std::get< 6>(in),
            v7  = std::get< 7>(in), v8  = std::get< 8>(in), v9  = std::get< 9>(in),
            v10 = std::get<10>(in), v11 = std::get<11>(in), v12 = std::get<12>(in),
            v13 = std::get<13>(in), v14 = std::get<14>(in), v15 = std::get<15>(in),
            v16 = std::get<16>(in), v17 = std::get<17>(in), v18 = std::get<18>(in);

        const float
            dP_xm = weno5(v2-v1,   v3-v2,   v0-v3,   v4-v0,   v5-v4,   dx2),
            dP_xp = weno5(v6-v5,   v5-v4,   v4-v0,   v0-v3,   v3-v2,   dx2),
            dP_ym = weno5(v8-v7,   v9-v8,   v0-v9,   v10-v0,  v11-v10, dx2),
            dP_yp = weno5(v12-v11, v11-v10, v10-v0,  v0-v9,   v9-v8,   dx2),
            dP_zm = weno5(v14-v13, v15-v14, v0-v15,  v16-v0,  v17-v16, dx2),
            dP_zp = weno5(v18-v17, v17-v16, v16-v0,  v0-v15,  v15-v14, dx2);

        return { invDx2 * godunovsNormSqrd(v0 > isoValue,
                     dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp) };
    };
}

// ---------------------------------------------------------------------------
// SIMD wrapper
// ---------------------------------------------------------------------------
constexpr int W = 16;
using WenoSimdIn  = typename ToSimdTuple<WenoIn,  W>::type;
using WenoSimdOut = typename ToSimdTuple<WenoOut, W>::type;

__attribute__((noinline))
void runSimdNormSqGrad(const WenoSimdIn& simdIn, WenoSimdOut& simdOut,
                       float dx2, float invDx2, float isoValue)
{
    auto kernel     = makeNormSqGrad(dx2, invDx2, isoValue);
    auto simdKernel = liftToSimd<W>(kernel);
    simdKernel(simdIn, simdOut);
}

// ---------------------------------------------------------------------------
// Reference: scalar normSqGrad directly on a float[19] array
// ---------------------------------------------------------------------------
float refNormSqGrad(const float* v, float dx2, float invDx2, float isoValue = 0.f)
{
    const float
        dP_xm = weno5(v[2]-v[1], v[3]-v[2], v[0]-v[3], v[4]-v[0], v[5]-v[4], dx2),
        dP_xp = weno5(v[6]-v[5], v[5]-v[4], v[4]-v[0], v[0]-v[3], v[3]-v[2], dx2),
        dP_ym = weno5(v[8]-v[7], v[9]-v[8], v[0]-v[9], v[10]-v[0], v[11]-v[10], dx2),
        dP_yp = weno5(v[12]-v[11], v[11]-v[10], v[10]-v[0], v[0]-v[9], v[9]-v[8], dx2),
        dP_zm = weno5(v[14]-v[13], v[15]-v[14], v[0]-v[15], v[16]-v[0], v[17]-v[16], dx2),
        dP_zp = weno5(v[18]-v[17], v[17]-v[16], v[16]-v[0], v[0]-v[15], v[15]-v[14], dx2);
    return invDx2 * godunovsNormSqrd(v[0] > isoValue,
                        dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp);
}

// ---------------------------------------------------------------------------
int main()
{
    const float dx = 0.1f, dx2 = dx*dx, invDx2 = 1.f/(dx*dx);

    // Fill 16 lanes with distinct synthetic level-set-like values
    WenoSimdIn simdIn{};
    float refValues[W][19];

    for (int i = 0; i < W; i++) {
        // Smooth profile: v[n] ~ sin(n * 0.3 + i * 0.5)
        for (int n = 0; n < 19; n++) {
            float val = std::sin(n * 0.3f + i * 0.5f);
            refValues[i][n] = val;
        }
        std::get< 0>(simdIn)[i] = refValues[i][ 0];
        std::get< 1>(simdIn)[i] = refValues[i][ 1];
        std::get< 2>(simdIn)[i] = refValues[i][ 2];
        std::get< 3>(simdIn)[i] = refValues[i][ 3];
        std::get< 4>(simdIn)[i] = refValues[i][ 4];
        std::get< 5>(simdIn)[i] = refValues[i][ 5];
        std::get< 6>(simdIn)[i] = refValues[i][ 6];
        std::get< 7>(simdIn)[i] = refValues[i][ 7];
        std::get< 8>(simdIn)[i] = refValues[i][ 8];
        std::get< 9>(simdIn)[i] = refValues[i][ 9];
        std::get<10>(simdIn)[i] = refValues[i][10];
        std::get<11>(simdIn)[i] = refValues[i][11];
        std::get<12>(simdIn)[i] = refValues[i][12];
        std::get<13>(simdIn)[i] = refValues[i][13];
        std::get<14>(simdIn)[i] = refValues[i][14];
        std::get<15>(simdIn)[i] = refValues[i][15];
        std::get<16>(simdIn)[i] = refValues[i][16];
        std::get<17>(simdIn)[i] = refValues[i][17];
        std::get<18>(simdIn)[i] = refValues[i][18];
    }

    WenoSimdOut simdOut{};
    runSimdNormSqGrad(simdIn, simdOut, dx2, invDx2, 0.f);

    printf("WenoNormSqGrad (W=%d, dx=%.2f):\n", W, dx);
    bool allOk = true;
    for (int i = 0; i < W; i++) {
        float ref  = refNormSqGrad(refValues[i], dx2, invDx2, 0.f);
        float got  = std::get<0>(simdOut)[i];
        bool  ok   = std::abs(got - ref) < 1e-5f * std::abs(ref) + 1e-10f;
        printf("  lane %2d: %12.6f  ref: %12.6f  %s\n", i, got, ref, ok ? "OK" : "FAIL");
        allOk &= ok;
    }
    printf("\nOverall: %s\n", allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}
