// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file LegacyStencilAccessor.h

    \brief Scalar stencil-index accessor using a NanoVDB ReadAccessor.

    LegacyStencilAccessor resolves each stencil tap via a path-cached
    NanoVDB ReadAccessor, one voxel at a time.  It is templatized on the
    same StencilT policy class used by StencilAccessor, so the tap-offset
    table is shared at compile time.

    This mirrors the approach of OpenVDB's math/Stencils.h: the accessor
    caches the last-visited tree path so that consecutive taps within the
    same leaf are cheap, but distant taps (e.g. WENO5 radius-3 offsets)
    can evict the center-leaf path.  That cache-pressure problem is the
    motivation for the BatchAccessor / StencilAccessor design.

    Intended uses
    -------------
      - Correctness oracle for StencilAccessor: sharing StencilT guarantees
        identical tap offsets, so a mismatch is a genuine bug.
      - Benchmark baseline: measures the cost of the accessor path-eviction
        problem that StencilAccessor is designed to eliminate.

    Thread safety
    -------------
    Each instance owns its ReadAccessor.  Construct one per thread.

    Template parameters
    -------------------
    BuildT    NanoVDB build type (e.g. ValueOnIndex).
    StencilT  Policy class describing the stencil.  Must expose:
                using Taps = std::tuple<StencilPoint<di,dj,dk>...>;
              Same type as passed to StencilAccessor.
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/StencilAccessor.h>  // StencilPoint, detail::findIndex

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>      // std::index_sequence, std::make_index_sequence

namespace nanovdb {

template<typename BuildT, typename StencilT>
class LegacyStencilAccessor
{
    using GridT = NanoGrid<BuildT>;

    static constexpr int SIZE = int(std::tuple_size_v<typename StencilT::Taps>);

public:
    explicit LegacyStencilAccessor(const GridT& grid)
        : mAcc(grid.getAccessor()) {}

    // -------------------------------------------------------------------------
    // moveTo -- resolve all SIZE tap indices for the voxel at @a center.
    //
    // Calls ReadAccessor::getValue(center + offset) for each tap in StencilT::Taps.
    // The path cache inside mAcc amortizes tree-traversal cost for nearby taps,
    // but distant taps (e.g. WENO5 ±3) may evict the center-leaf path.
    //
    // Results are valid until the next moveTo() call.
    // -------------------------------------------------------------------------
    void moveTo(const Coord& center)
    {
        fillTaps(center, std::make_index_sequence<SIZE>{});
    }

    // -------------------------------------------------------------------------
    // operator[] -- indexed tap access.  i must be in [0, SIZE).
    // -------------------------------------------------------------------------
    uint64_t operator[](int i) const { return mStencil[i]; }

    // -------------------------------------------------------------------------
    // getValue<DI,DJ,DK> -- compile-time named tap access.
    //
    // Same interface as StencilAccessor::getValue; resolved entirely at
    // compile time via detail::findIndex.
    // -------------------------------------------------------------------------
    template<int DI, int DJ, int DK>
    uint64_t getValue() const
    {
        constexpr int I = detail::findIndex<typename StencilT::Taps, DI, DJ, DK>(
            std::make_index_sequence<SIZE>{});
        static_assert(I >= 0, "LegacyStencilAccessor::getValue: tap not in stencil");
        return mStencil[I];
    }

    static constexpr int size() { return SIZE; }

private:
    template<size_t... Is>
    void fillTaps(const Coord& center, std::index_sequence<Is...>)
    {
        using Taps = typename StencilT::Taps;
        ((mStencil[Is] = static_cast<uint64_t>(
            mAcc.getValue(center + Coord(
                std::tuple_element_t<Is, Taps>::di,
                std::tuple_element_t<Is, Taps>::dj,
                std::tuple_element_t<Is, Taps>::dk)))), ...);
    }

    DefaultReadAccessor<BuildT> mAcc;
    uint64_t                    mStencil[SIZE];
};

} // namespace nanovdb
