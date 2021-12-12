// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey
///
/// @file points/PointAdvect.h
///
/// @brief Ability to advect VDB Points through a velocity field.

#ifndef OPENVDB_POINTS_POINT_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_ADVECT_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/VelocityFields.h>

#include <openvdb/points/AttributeGroup.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointMove.h>

#include <memory>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


/// @brief Advect points in a PointDataGrid through a velocity grid
/// @param points               the PointDataGrid containing the points to be advected.
/// @param velocity             a velocity grid to be sampled.
/// @param integrationOrder     the integration scheme to use (1 is forward euler, 4 is runge-kutta 4th)
/// @param dt                   delta time.
/// @param timeSteps            number of advection steps to perform.
/// @param advectFilter         an optional advection index filter (moves a subset of the points)
/// @param filter               an optional index filter (deletes a subset of the points)
template <typename PointDataGridT, typename VelGridT,
    typename AdvectFilterT = NullFilter, typename FilterT = NullFilter>
inline void advectPoints(PointDataGridT& points, const VelGridT& velocity,
                         const Index integrationOrder, const double dt, const Index timeSteps,
                         const AdvectFilterT& advectFilter = NullFilter(),
                         const FilterT& filter = NullFilter());


////////////////////////////////////////

/// @cond OPENVDB_DOCS_INTERNAL

namespace point_advect_internal {

enum IntegrationOrder {
    INTEGRATION_ORDER_FWD_EULER = 1,
    INTEGRATION_ORDER_RK_2ND,
    INTEGRATION_ORDER_RK_3RD,
    INTEGRATION_ORDER_RK_4TH
};

template <typename VelGridT, Index IntegrationOrder, bool Staggered, typename FilterT>
class AdvectionDeformer
{
public:
    using IntegratorT = openvdb::tools::VelocityIntegrator<VelGridT, Staggered>;

    AdvectionDeformer(const VelGridT& velocityGrid, const double timeStep, const int steps,
                      const FilterT& filter)
        : mIntegrator(velocityGrid)
        , mTimeStep(timeStep)
        , mSteps(steps)
        , mFilter(filter) { }

    template <typename LeafT>
    void reset(const LeafT& leaf, size_t /*idx*/)
    {
        mFilter.reset(leaf);
    }

    template <typename IndexIterT>
    void apply(Vec3d& position, const IndexIterT& iter) const
    {
        if (mFilter.valid(iter)) {
            for (int n = 0; n < mSteps; ++n) {
                mIntegrator.template rungeKutta<IntegrationOrder, openvdb::Vec3d>(
                    static_cast<typename IntegratorT::ElementType>(mTimeStep), position);
            }
        }
    }

private:
    IntegratorT mIntegrator;
    double mTimeStep;
    const int mSteps;
    FilterT mFilter;
}; // class AdvectionDeformer


template <typename PointDataGridT, typename VelGridT, typename AdvectFilterT, typename FilterT>
struct AdvectionOp
{
    AdvectionOp(PointDataGridT& points, const VelGridT& velocity,
                const Index integrationOrder, const double timeStep, const Index steps,
                const AdvectFilterT& advectFilter,
                const FilterT& filter)
        : mPoints(points)
        , mVelocity(velocity)
        , mIntegrationOrder(integrationOrder)
        , mTimeStep(timeStep)
        , mSteps(steps)
        , mAdvectFilter(advectFilter)
        , mFilter(filter) { }

    void advect()
    {
        // early-exit if no leafs
        if (mPoints.constTree().leafCount() == 0)            return;

        if (mVelocity.getGridClass() == openvdb::GRID_STAGGERED) {
            resolveStaggered<true>();
        } else {
            resolveStaggered<false>();
        }
    }

private:
    template <int IntegrationOrder, bool Staggered>
    void resolveIntegrationOrder()
    {
        const auto leaf = mPoints.constTree().cbeginLeaf();
        if (!leaf)  return;

        NullFilter nullFilter;

        // revert to NullFilter if all points are being evaluated
        if (mAdvectFilter.state() == index::ALL) {
            AdvectionDeformer<VelGridT, IntegrationOrder, Staggered, NullFilter> deformer(
                mVelocity, mTimeStep, mSteps, nullFilter);
            movePoints(mPoints, deformer, mFilter);
        }
        else {
            AdvectionDeformer<VelGridT, IntegrationOrder, Staggered, AdvectFilterT> deformer(
                mVelocity, mTimeStep, mSteps, mAdvectFilter);
            movePoints(mPoints, deformer, mFilter);
        }
    }

    template <bool Staggered>
    void resolveStaggered()
    {
        if (mIntegrationOrder == INTEGRATION_ORDER_FWD_EULER) {
            resolveIntegrationOrder<1, Staggered>();
        } else if (mIntegrationOrder == INTEGRATION_ORDER_RK_2ND) {
            resolveIntegrationOrder<2, Staggered>();
        } else if (mIntegrationOrder == INTEGRATION_ORDER_RK_3RD) {
            resolveIntegrationOrder<3, Staggered>();
        } else if (mIntegrationOrder == INTEGRATION_ORDER_RK_4TH) {
            resolveIntegrationOrder<4, Staggered>();
        }
    }

    PointDataGridT& mPoints;
    const VelGridT& mVelocity;
    const Index mIntegrationOrder;
    const double mTimeStep;
    const Index mSteps;
    const AdvectFilterT& mAdvectFilter;
    const FilterT& mFilter;
}; // struct AdvectionOp

} // namespace point_advect_internal

/// @endcond

////////////////////////////////////////


template <typename PointDataGridT, typename VelGridT, typename AdvectFilterT, typename FilterT>
inline void advectPoints(PointDataGridT& points, const VelGridT& velocity,
                         const Index integrationOrder, const double timeStep, const Index steps,
                         const AdvectFilterT& advectFilter,
                         const FilterT& filter)
{
    using namespace point_advect_internal;

    if (steps == 0)     return;

    if (integrationOrder > 4) {
        throw ValueError{"Unknown integration order for advecting points."};
    }

    AdvectionOp<PointDataGridT, VelGridT, AdvectFilterT, FilterT> op(
        points, velocity, integrationOrder, timeStep, steps,
        advectFilter, filter);

    // advect the points
    op.advect();
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_ADVECT_HAS_BEEN_INCLUDED
