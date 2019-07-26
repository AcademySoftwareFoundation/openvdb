// This header implements performance-critical bitwise routines with gcc and clang compiler intrinsics, which compile into BSR, BSF, POPCNT instructions.

/// Return the number of on bits in the given 8-bit value.
inline Index32 CountOn( Byte v )
{
    return __builtin_popcount( v );
}

/// Return the number of on bits in the given 32-bit value.
inline Index32 CountOn( Index32 v )
{
    return __builtin_popcount( v );
}

inline Index32 CountOn( Index64 v )
{
    return static_cast<Index32>( __builtin_popcountll( v ) );
}

/// Return the least significant on bit of the given 32-bit value.
inline Index32 FindLowestOn( Index32 v )
{
    assert( v );
    return __builtin_ctz( v );
}

/// Return the least significant on bit of the given 8-bit value.
inline Index32 FindLowestOn( Byte v )
{
    assert( v );
    return __builtin_ctz( v );
}

/// Return the most significant on bit of the given 32-bit value.
inline Index32 FindHighestOn( Index32 v )
{
    // BSR is very fast, computing result anyway so the compiler can emit CMOV for `operator ?`
    const Index32 result = 31 - __builtin_clz( v );
    return ( 0 != v ) ? result : 0;
}