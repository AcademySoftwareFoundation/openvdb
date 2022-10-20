// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_TRANSFORM_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_TRANSFORM_HAS_BEEN_INCLUDED

#include <openvdb/math/Transform.h>


/* openvdbmma::pngio members

 class GridAdjustment

 public members are

 scalarMultiply
 gammaAdjust

*/


namespace openvdbmma {
namespace transform {

//////////// grid adjustment class

template<typename GridT, typename T>
class GridAdjustment
{
public:

    using ValueT = typename GridT::ValueType;
    using GridPtr = typename GridT::Ptr;

    GridAdjustment(GridPtr grid) : mGrid(grid)
    {
    }

    ~GridAdjustment() {}

    void scalarMultiply(T fac);

    void gammaAdjust(T gamma);

private:

    //////////// transformation functors

    struct ScalarTimes {
        T mFac;
        explicit ScalarTimes(const T fac): mFac(fac) {}
        inline ValueT operator()(const ValueT& x) const {
            return mFac * x;
        }
    };

    struct ClampedScalarTimes {
        T mFac;
        explicit ClampedScalarTimes(const T fac): mFac(fac) {}
        inline ValueT operator()(const ValueT& x) const {
            return math::Clamp01(mFac * x);
        }
    };

    struct GammaPow {
        T mGamma;
        explicit GammaPow(const T gamma): mGamma(gamma) {}
        inline ValueT operator()(const ValueT& x) const {
            return math::Pow(x, mGamma);
        }
    };

    inline bool validGammaAdjustInput(const T gamma) const
    {
        return gamma > 0.0 && mGrid->getGridClass() == GRID_FOG_VOLUME;
    }

    void multiplyMetaValue(std::string key, T fac)
    {
        using MetaIter = openvdb::MetaMap::MetaIterator;

        float facold = 1.0;

        for (MetaIter iter = mGrid->beginMeta(); iter != mGrid->endMeta(); ++iter)
            if (iter->first == key) {
                openvdb::Metadata::Ptr metadata = iter->second;
                facold = static_cast<openvdb::FloatMetadata&>(*metadata).value();
                break;
            }

        mGrid->insertMeta(key, openvdb::FloatMetadata(facold * (float)fac));
    }

    //////////// private members

    GridPtr mGrid;

}; // end of GridAdjustment class


//////////// GridAdjustment public member function definitions

template<typename GridT, typename T>
inline void
GridAdjustment<GridT, T>::scalarMultiply(T fac)
{
    using TreeT = typename GridT::TreeType;

    multiplyMetaValue(META_SCALING_FACTOR, fac);

    if (mGrid->getGridClass() == GRID_FOG_VOLUME) {
        transformActiveLeafValues<TreeT, ClampedScalarTimes>(mGrid->tree(),
            ClampedScalarTimes(fac));
    } else {
        transformActiveLeafValues<TreeT, ScalarTimes>(mGrid->tree(), ScalarTimes(fac));
    }
}

template<typename GridT, typename T>
inline void
GridAdjustment<GridT, T>::gammaAdjust(T gamma)
{
    using TreeT = typename GridT::TreeType;

    if (!validGammaAdjustInput(gamma))
        throw mma::LibraryError(LIBRARY_NUMERICAL_ERROR);

    multiplyMetaValue(META_GAMMA_ADJUSTMENT, gamma);

    transformActiveLeafValues<TreeT, GammaPow>(mGrid->tree(), GammaPow(gamma));
}

} // namespace transform
} // namespace openvdbmma

#endif // OPENVDBLINK_UTILITIES_TRANSFORM_HAS_BEEN_INCLUDED
