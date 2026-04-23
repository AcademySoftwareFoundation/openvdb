// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file LegacyStencilAccessor.h

    \brief Scalar stencil-index accessor using a NanoVDB ReadAccessor.

    LegacyStencilAccessor resolves each stencil tap via a path-cached
    NanoVDB ReadAccessor, one voxel at a time.  It is templatized on a
    StencilT policy class whose Taps tuple defines the tap offsets.

    This mirrors the approach of OpenVDB's math/Stencils.h: the accessor
    caches the last-visited tree path so that consecutive taps within the
    same leaf are cheap, but distant taps (e.g. WENO5 radius-3 offsets)
    can evict the center-leaf path.

    Thread safety
    -------------
    Each instance owns its ReadAccessor.  Construct one per thread.

    Template parameters
    -------------------
    BuildT    NanoVDB build type (e.g. ValueOnIndex).
    StencilT  Policy class describing the stencil.  Must expose:
                using Taps = std::tuple<S<di,dj,dk>...>;
              where each S is any type with static int members di, dj, dk
              (e.g. WenoStencil<>::TapPoint).
*/

#pragma once

#include <nanovdb/NanoVDB.h>

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

    // Compile-time inverse map: (DI,DJ,DK) → slot index in StencilT::Taps.
    // Returns -1 if no matching tap exists; getValue() turns that into a
    // static_assert.  Same shape as WenoStencil<W>::findTap (kept local here
    // to avoid a cross-header dependency).
    template<int DI, int DJ, int DK, size_t... Is>
    static constexpr int findTap(std::index_sequence<Is...>)
    {
        using Taps = typename StencilT::Taps;
        int result = -1;
        ((std::tuple_element_t<Is, Taps>::di == DI &&
          std::tuple_element_t<Is, Taps>::dj == DJ &&
          std::tuple_element_t<Is, Taps>::dk == DK &&
          result < 0 ? (result = int(Is)) : 0), ...);
        return result;
    }

public:
    // Leaf-only ReadAccessor (cache level 0 only).  The DefaultReadAccessor
    // (levels 0/1/2) caches upper and lower nodes too, but those slots are
    // never consulted during a GetValue cache-miss resolution -- the fallback
    // goes straight to mRoot->getAndCache(...).  Using a 1-level accessor
    // removes passive bookkeeping of the upper/lower slots on every miss and
    // keeps the benchmark honest about what's being measured.
    using AccessorT = ReadAccessor<BuildT, 0, -1, -1>;

    explicit LegacyStencilAccessor(const GridT& grid)
        : mAcc(grid.tree().root()) {}

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
    // -------------------------------------------------------------------------
    template<int DI, int DJ, int DK>
    uint64_t getValue() const
    {
        constexpr int I = findTap<DI, DJ, DK>(std::make_index_sequence<SIZE>{});
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

    AccessorT mAcc;
    uint64_t  mStencil[SIZE];
};

} // namespace nanovdb
