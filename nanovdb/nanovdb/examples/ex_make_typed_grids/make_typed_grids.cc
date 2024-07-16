// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>

// Helper struct to create a default value for the type.
// We use a helper struct so we can specialize it for different types.
template<typename ValueT, bool isFloatingPoint = !std::is_integral<ValueT>::value>
struct makeDefaultValue
{
    inline ValueT operator()(const nanovdb::Coord& /*ijk*/, int /*radius*/) const
    {
        return ValueT(1);
    }
};

template<typename ValueT>
struct makeDefaultValue<ValueT, false>
{
    inline ValueT operator()(const nanovdb::Coord& /*ijk*/, int /*radius*/) const
    {
        return ValueT(1);
    }
};

void buildGridForType(std::vector<nanovdb::GridHandle<>>&)
{
}

template<typename T, typename... Ts>
void buildGridForType(std::vector<nanovdb::GridHandle<>>& gridHandles, T const& bgValue, Ts const&... rest)
{
    using ValueT = T;
    std::string typeNameStr = typeid(T).name();

    try {

        nanovdb::tools::build::Grid<ValueT> grid(bgValue, typeNameStr);
        auto acc = grid.getAccessor();
        const int radius = 16;
        for (int z = -radius; z <= radius; ++z) {
            for (int y = -radius; y <= radius; ++y) {
                for (int x = -radius; x <= radius; ++x) {
                    const auto ijk = nanovdb::Coord(x, y, z);
                    if (nanovdb::Vec3f(ijk).length() <= radius)
                        acc.setValue(ijk, makeDefaultValue<ValueT>()(ijk, radius));
                }
            }
        }
        gridHandles.push_back(nanovdb::tools::createNanoGrid(grid));
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    buildGridForType(gridHandles, rest...);
}

/// @brief Creates a NanoVDB grids for each GridType.
///
/// @note This example only depends on NanoVDB.
int main()
{
    std::vector<nanovdb::GridHandle<>> gridHandles;
    try {
        /*
        GridType : uint32_t { Unknown = 0,
                                 Float = 1,
                                 Double = 2,
                                 Int16 = 3,
                                 Int32 = 4,
                                 Int64 = 5,
                                 Vec3f = 6,
                                 Vec3d = 7,
                                 Mask = 8,
                                 FP16 = 9,
                                 UInt32 = 10,
                                 End = 11 };
                                 */

        buildGridForType(gridHandles, float(0), double(0), int16_t(0), int32_t(0), int64_t(0), uint32_t(0), nanovdb::Vec3f(0) /*, nanovdb::Vec3d(0)*/ /*, bool(false)*/ /*, uint16_t(0)*/);
#if 0
        nanovdb::io::writeGrids("data/custom_types.nvdb", gridHandles);
#else
        nanovdb::io::writeUncompressedGrids("data/custom_types.nvdb", gridHandles);
#endif
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}