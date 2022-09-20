#include <openvdb/tools/LevelSetFilter.h>


/* OpenVDBGrid public member function list

void resizeBandwidth(double width)

void offsetLevelSet(double r)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
void
OpenVDBGrid<V>::resizeBandwidth(double width)
{
    scalar_type_assert<V>();

    using MaskT = typename wlGridType::template ValueConverter<float>::Type;
    using InterrupterT = mma::interrupt::LLInterrupter;

    InterrupterT interrupt;

    openvdb::tools::LevelSetFilter<wlGridType, MaskT, InterrupterT> filter(*grid(), &interrupt);
    filter.resize(width);

    setLastModified();
}

template<typename V>
void
OpenVDBGrid<V>::offsetLevelSet(double r)
{
    scalar_type_assert<V>();

    using MaskT = typename wlGridType::template ValueConverter<float>::Type;
    using InterrupterT = mma::interrupt::LLInterrupter;

    InterrupterT interrupt;

    openvdb::tools::LevelSetFilter<wlGridType, MaskT, InterrupterT> filter(*grid(), &interrupt);
    filter.offset(r);

    setLastModified();
}
