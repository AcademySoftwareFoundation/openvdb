// This header implements performance-critical bitwise routines with vc++ compiler intrinsics, which compile into BSR, BSF, POPCNT instructions.

/// Return the number of on bits in the given 8-bit value.
inline Index32 CountOn( Byte v )
{
    return __popcnt16( v );
}

/// Return the number of on bits in the given 32-bit value.
inline Index32 CountOn( Index32 v )
{
    return __popcnt( v );
}

/// Return the number of on bits in the given 64-bit value.
inline Index32 CountOn( Index64 v )
{
#ifdef _M_X64
    return static_cast<Index32>( __popcnt64( v ) );
#else
    return __popcnt( static_cast<Index32>( v ) ) + __popcnt( static_cast<Index32>( v >> 32 ) );
#endif
}

/// Return the least significant on bit of the given 32-bit value.
inline Index32 FindLowestOn( Index32 v )
{
    assert( v );
    unsigned long index;
    _BitScanForward( &index, v );
    return index;
}

/// Return the least significant on bit of the given 8-bit value.
inline Index32 FindLowestOn( Byte v )
{
    return FindLowestOn( static_cast<Index32>( v ) );
}

/// Return the most significant on bit of the given 32-bit value.
inline Index32 FindHighestOn( Index32 v )
{
    unsigned long index;
    const uint8_t anyBit = _BitScanReverse( &index, v );
    return anyBit ? index : 0;
}