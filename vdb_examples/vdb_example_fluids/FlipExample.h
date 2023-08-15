// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

# pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointAdvect.h> // for advectPoints
#include <openvdb/points/PointAttribute.h> // for appendAttribute
#include <openvdb/points/PointDataGrid.h> // for PointDataGrid
#include <openvdb/points/PointRasterizeTrilinear.h> // for rasterizing to the grid
#include <openvdb/points/PointSample.h> // for PointSample
#include <openvdb/points/PointScatter.h> // for point sampling
#include <openvdb/tools/Composite.h> // for tools::compMax
#include <openvdb/tools/GridOperators.h> // for divergence and gradient
#include <openvdb/tools/MeshToVolume.h> // for createLevelSetBox
#include <openvdb/tools/Morphology.h> // for erodeActiveValues
#include <openvdb/tools/PoissonSolver.h> // for poisson solve
#include <openvdb/tools/VolumeAdvect.h> // for tools::VolumeAdvection
#include <openvdb/tree/LeafManager.h> // for LeafManager

using namespace openvdb;

namespace example {

class FlipSolver {

public:

    FlipSolver(float const voxelSize);

    void render();

private:

    void initializeFreeFall();
    void initializePool();
    void initializeDamBreak();

    void substep(float const dt);

    // Rasterize particle velocity to the grid void particlesToGrid();

    // FLIP update: Interpolate the delta of velocity update (v_np1 - v_n)
    // back to the particle
    void gridToParticles();
    void updateParticles(float const dt);
    void updateParticlesVelocity();

    // Update particle position based on velocity on the grid
    void advectParticles(float const dt);

    // Make the velocity on the grid to be divergence free
    void pressureProjection(bool print);

    void gridVelocityUpdate(float const dt);

    // Apply velocity on Neumann-pressure faces
    void velocityBCCorrection(Vec3SGrid::Ptr vecGrid);
    void extrapolateToCollider(Vec3SGrid& vecGrid);

    float computeDivergence(FloatGrid::Ptr& divGrid, const Vec3SGrid::Ptr vecGrid, const std::string& suffix);
    float computeLInfinity(const FloatGrid& grid);

    void addGravity(float const dt);
    void computeFlipVelocity(float const dt);

    void writeVDBs(int const frame);
    void writeVDBsVerbose(int const frame);

    struct BoundaryOp {
        BoundaryOp(float const voxelSize,
                   FloatGrid::Ptr collider,
                   Vec3SGrid::Ptr vCurr) :
                   voxelSize(voxelSize),
                   collider(collider),
                   vCurr(vCurr) {}

        void operator()(const openvdb::Coord& ijk,
                        const openvdb::Coord& neighbor,
                        double& source,
                        double& diagonal) const
        {
            float const dirichletBC = 0.f;
            bool isInsideCollider = collider->tree().isValueOn(neighbor);
            auto vNgbr = Vec3s(0.f, 0.f, 0.f); // static collider
            if (isInsideCollider) {
                double delta = 0.0;
                // Neumann pressure from bbox
                if (neighbor.x() + 1 == ijk.x() /* left x-face */) {
                    delta += vNgbr[0];
                }
                if (neighbor.x() - 1 == ijk.x() /* right x-face */) {
                    delta -= vNgbr[0];
                }
                if (neighbor.y() + 1 == ijk.y() /* bottom y-face */) {
                    delta += vNgbr[1];
                }
                if (neighbor.y() - 1 == ijk.y() /* top y-face */) {
                    delta -= vNgbr[1];
                }
                if (neighbor.z() + 1 == ijk.z() /* back z-face */) {
                    delta += vNgbr[2];
                }
                if (neighbor.z() - 1 == ijk.z() /* front z-face */) {
                    delta -= vNgbr[2];
                }
                // Note: in the SOP_OpenVDB_Remove_Divergence, we need to multiply
                // this by 0.5, because the gradient that's used is using
                // central-differences in a collocated grid, instead of the staggered one.
                source += delta / voxelSize;
            } else /* Dirichlet */ {
                diagonal -= 1.0;
                source -= dirichletBC;
            }
        }

        float voxelSize;
        FloatGrid::Ptr collider;
        Vec3SGrid::Ptr vCurr;
    };


    struct FlipUpdateOp
    {
        explicit FlipUpdateOp(Index64 const velAtrIdx,
                              Index64 const vPicAtrIdx,
                              Index64 const vFlipAtrIdx,
                              float const alpha)
                              : velAtrIdx(velAtrIdx),
                                vPicAtrIdx(vPicAtrIdx),
                                vFlipAtrIdx(vFlipAtrIdx),
                                alpha(alpha) { }

        void operator()(const tree::LeafManager<points::PointDataTree>::LeafRange& range) const {
            for (auto leafIter = range.begin(); leafIter; ++leafIter) {
                points::AttributeArray& velArray = leafIter->attributeArray(velAtrIdx);
                points::AttributeArray const& vPicArray = leafIter->constAttributeArray(vPicAtrIdx);
                points::AttributeArray const& vFlipArray = leafIter->constAttributeArray(vFlipAtrIdx);
                points::AttributeWriteHandle<Vec3s> velHandle(velArray);
                points::AttributeHandle<Vec3s> vPicHandle(vPicArray);
                points::AttributeHandle<Vec3s> vFlipHandle(vFlipArray);
                // Iterate over active indices in the leaf.
                for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
                    auto vPic = vPicHandle.get(*indexIter);
                    auto vFlip = vFlipHandle.get(*indexIter);
                    auto newVel = alpha * (vPic + vFlip) + (1 - alpha) * vPic;
                    velHandle.set(*indexIter, newVel);
                }
            }
        }

        Index64 velAtrIdx;
        Index64 vPicAtrIdx;
        Index64 vFlipAtrIdx;
        float alpha;
    };


    // Apply Gravity Functor. Meant to be used with
    // foreach in LeafManager
    struct ApplyGravityOp
    {
        ApplyGravityOp(float const dt, Vec3s const gravity) : dt(dt), gravity(gravity) {}

        template <typename T>
        void operator()(T &leaf, size_t) const
        {
            for (typename T::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
                Vec3s newVal = *iter  + dt * gravity;
                iter.setValue(newVal);
            }
        }

        Vec3s const gravity;
        float const dt;
    };// ApplyGravityOp


    // Functor to fix the normal velocity on a collider.
    struct VelocityBCCorrectionOp
    {
        VelocityBCCorrectionOp(Vec3SGrid::Ptr vNext, FloatGrid::Ptr collider, BoolGrid::Ptr interiorPressure) :
            vNext(vNext), collider(collider), interiorPressure(interiorPressure) {}

        template <typename T>
        void operator()(T &leaf, size_t) const
        {
            auto vNextAcc = vNext->getAccessor();
            auto intrAcc = interiorPressure->getConstAccessor();
            for (typename T::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
                auto ijk = iter.getCoord();
                auto im1jk = ijk.offsetBy(-1, 0, 0);
                auto ijm1k = ijk.offsetBy(0, -1, 0);
                auto ijkm1 = ijk.offsetBy(0, 0, -1);
                auto ip1jk = ijk.offsetBy(1, 0, 0);
                auto ijp1k = ijk.offsetBy(0, 1, 0);
                auto ijkp1 = ijk.offsetBy(0, 0, 1);

                if (intrAcc.isValueOn(im1jk))
                {
                    auto val = vNextAcc.getValue(ijk);
                    vNextAcc.setValue(ijk, Vec3s(0.f, val[1], val[2]));
                }

                if (intrAcc.isValueOn(ijm1k))
                {
                    auto val = vNextAcc.getValue(ijk);
                    vNextAcc.setValue(ijk, Vec3s(val[0], 0.f, val[2]));
                }

                if (intrAcc.isValueOn(ijkm1))
                {
                    auto val = vNextAcc.getValue(ijk);
                    vNextAcc.setValue(ijk, Vec3s(val[0], val[1], 0.f));
                }

                if (intrAcc.isValueOn(ip1jk))
                {
                    auto val = vNextAcc.getValue(ip1jk);
                    vNextAcc.setValue(ip1jk, Vec3s(0.f, val[1], val[2]));
                }

                if (intrAcc.isValueOn(ijp1k))
                {
                    auto val = vNextAcc.getValue(ijp1k);
                    vNextAcc.setValue(ijp1k, Vec3s(val[0], 0.f, val[2]));
                }

                if (intrAcc.isValueOn(ijkp1))
                {
                    auto val = vNextAcc.getValue(ijkp1);
                    vNextAcc.setValue(ijkp1, Vec3s(val[0], val[1], 0.f));
                }
            }
        }

        Vec3SGrid::Ptr vNext;
        FloatGrid::Ptr collider;
        BoolGrid::Ptr interiorPressure;
    };// VelocityBCCorrectionOp


    // Compute the difference between vNext and the original rasterized
    // vCurr (before the addition of gravity). To be used with foreach in LeafManager.
    struct ComputeFlipVelocityOp
    {
        ComputeFlipVelocityOp(Vec3SGrid::Ptr vCurr,
                              Vec3SGrid::Ptr vNext,
                              float const dt,
                              Vec3s const gravity) :
                              vCurr(vCurr),
                              vNext(vNext),
                              dt(dt),
                              gravity(gravity) {}

        template <typename T>
        void operator()(T &leaf, size_t) const
        {
            auto vCurrAcc = vCurr->getAccessor();
            auto vNextAcc = vNext->getAccessor();
            for (typename T::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
                auto ijk = iter.getCoord();
                Vec3s val = vNextAcc.getValue(ijk) - vCurrAcc.getValue(ijk) - dt * gravity;
                iter.setValue(val);
            }
        }

        Vec3SGrid::Ptr vCurr;
        Vec3SGrid::Ptr vNext;
        Vec3s const gravity;
        float const dt;
    };// ComputeFlipVelocityOp

    // Pressure projection: mVNext = mVCurr - grad pressure.
    // Assumes that vNext at ijk already has the value of vCurr.
    struct SubtractPressureGradientOp
    {
        SubtractPressureGradientOp(BoolGrid::Ptr interiorPressure,
                                   FloatGrid::Ptr pressure,
                                   Vec3SGrid::Ptr vCurr,
                                   float const voxelSize) :
                                   interiorPressure(interiorPressure),
                                   pressure(pressure),
                                   vCurr(vCurr),
                                   voxelSize(voxelSize)
                                   {}

        template <typename T>
        void operator()(T &leaf, size_t) const
        {
            auto interiorAcc = interiorPressure->getConstAccessor();
            auto pressureAcc = pressure->getConstAccessor();
            auto vCurrAcc = vCurr->getConstAccessor();
            for (typename T::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) {
                math::Coord ijk = iter.getCoord();
                math::Coord im1jk = ijk.offsetBy(-1, 0, 0);
                math::Coord ijm1k = ijk.offsetBy(0, -1, 0);
                math::Coord ijkm1 = ijk.offsetBy(0, 0, -1);

                // Only updates velocity if it is a face of fluid cell
                if (interiorAcc.isValueOn(ijk) ||
                    interiorAcc.isValueOn(im1jk) ||
                    interiorAcc.isValueOn(ijm1k) ||
                    interiorAcc.isValueOn(ijkm1)) {
                    Vec3s gradijk;
                    gradijk[0] = pressureAcc.getValue(ijk) - pressureAcc.getValue(ijk.offsetBy(-1, 0, 0));
                    gradijk[1] = pressureAcc.getValue(ijk) - pressureAcc.getValue(ijk.offsetBy(0, -1, 0));
                    gradijk[2] = pressureAcc.getValue(ijk) - pressureAcc.getValue(ijk.offsetBy(0, 0, -1));
                    auto val = vCurrAcc.getValue(ijk) - gradijk  *  voxelSize;
                    iter.setValue(val);
                }
            }
        }

        BoolGrid::Ptr interiorPressure;
        FloatGrid::Ptr pressure;
        Vec3SGrid::Ptr vCurr;
        float voxelSize;
    };// SubtractPressureGradientOp


    float mVoxelSize = 0.1f;
    Vec3s mGravity = Vec3s(0.f, -9.8f/2, 0.f);
    int mPointsPerVoxel = 8;
    math::Transform::Ptr mXform;

    points::PointDataGrid::Ptr mPoints;
    FloatGrid::Ptr mCollider;
    Vec3SGrid::Ptr mVCurr;
    Vec3SGrid::Ptr mVNext;
    Vec3SGrid::Ptr mVDiff; // For FlIP (Fluid Implicit Particle)

    // For verbose/debugging output
    FloatGrid::Ptr mPressure;
    FloatGrid::Ptr mDivBefore;
    FloatGrid::Ptr mDivAfter;
    BoolGrid::Ptr mInteriorPressure;
};


FlipSolver::FlipSolver(float const voxelSize) : mVoxelSize(voxelSize)
{
    initializeDamBreak();
}


void
FlipSolver::initializeFreeFall() {
    using BBox = math::BBox<Vec3s>;

    mXform = math::Transform::createLinearTransform(mVoxelSize);

    auto wsFluidInit = BBox(Vec3s(3.f, 3.f, 3.f) /* min */, Vec3s(4.f, 4.f, 4.f) /* max */);
    FloatGrid::Ptr fluidLSInit = tools::createLevelSetBox<FloatGrid>(wsFluidInit, *mXform);

    auto wsDomain = BBox(Vec3s(0.f, 0.f, 0.f) /* min */, Vec3s(14.f, 0.5f, 14.f) /* max */); // world space domain
    mCollider = tools::createLevelSetBox<FloatGrid>(wsDomain, *mXform);
    mCollider->setGridClass(GRID_LEVEL_SET);
    mCollider->setName("collider");

    mPoints = points::denseUniformPointScatter(*fluidLSInit, mPointsPerVoxel);
    mPoints->setName("Points");
    points::appendAttribute<Vec3s>(mPoints->tree(),
                                   "velocity" /* attribute name */,
                                   Vec3s(0.f, 0.f, 0.f) /* uniform value */,
                                   1 /* stride or total count */,
                                   true /* constant stride */,
                                   nullptr /* default value */,
                                   false /* hidden */,
                                   false /* transient */);
    points::appendAttribute<Vec3s>(mPoints->tree(),
                                   "v_pic" /* attribute name */,
                                   Vec3s(0.f, 0.f, 0.f) /* uniform value */,
                                   1 /* stride or total count */,
                                   true /* constant stride */,
                                   nullptr /* default value */,
                                   false /* hidden */,
                                   false /* transient */);
    points::appendAttribute<Vec3s>(mPoints->tree(),
                                   "v_flip" /* attribute name */,
                                   Vec3s(0.f, 0.f, 0.f) /* uniform value */,
                                   1 /* stride or total count */,
                                   true /* constant stride */,
                                   nullptr /* default value */,
                                   false /* hidden */,
                                   false /* transient */);

    openvdb::Index64 count = openvdb::points::pointCount(mPoints->tree());
    std::cout << "PointCount=" << count << std::endl;
}


void
FlipSolver::initializePool() {
    using BBox = math::BBox<Vec3s>;

    mXform = math::Transform::createLinearTransform(mVoxelSize);

    Vec3s minFI = Vec3s(2.f, 2.f, 2.f);
    Vec3s maxFI = Vec3s(3.f, 2.5f, 3.f);
    Vec3s maxFI2 = Vec3s(3.f, 5.1f, 3.f);
    Coord minFIcoord = mXform->worldToIndexNodeCentered(minFI);
    Coord maxFIcoord = mXform->worldToIndexNodeCentered(maxFI);
    Coord maxFIcoord2 = mXform->worldToIndexNodeCentered(maxFI2);
    Vec3s minBBoxvec = Vec3s(1.8f, 1.8f, 1.8f);
    Vec3s maxBBoxvec = Vec3s(3.2f, 3.2f, 3.2f);
    Coord minBBoxcoord = mXform->worldToIndexNodeCentered(minBBoxvec);
    Coord maxBBoxcoord = mXform->worldToIndexNodeCentered(maxBBoxvec);
    auto wsFluidInit = BBox( minFI/* min */,  maxFI/* max */);
    FloatGrid::Ptr fluidLSInit = FloatGrid::create(/*bg = */0.f);
    fluidLSInit->denseFill(CoordBBox(minFIcoord, maxFIcoord), /*value = */ 1.0, /*active = */ true);
    fluidLSInit->setTransform(mXform);
    FloatGrid::Ptr fluidLSInit2 = FloatGrid::create(/*bg = */0.f);
    fluidLSInit2->denseFill(CoordBBox(minFIcoord, maxFIcoord2), /*value = */ 1.0, /*active = */ true);
    fluidLSInit2->setTransform(mXform);

    mCollider = FloatGrid::create(/*bg = */0.f);
    mCollider->denseFill(CoordBBox(minBBoxcoord, maxBBoxcoord), /*value = */ 1.0, /*active = */ true);
    mCollider->setTransform(mXform);
    mCollider->topologyDifference(*fluidLSInit2);
    mCollider->setName("collider");
    openvdb::tools::pruneInactive(mCollider->tree());

    mPoints = points::denseUniformPointScatter(*fluidLSInit, mPointsPerVoxel);
    mPoints->setName("Points");
    points::appendAttribute<Vec3s>(mPoints->tree(),
                                   "velocity" /* attribute name */,
                                   Vec3s(0.f, 0.f, 0.f) /* uniform value */,
                                   1 /* stride or total count */,
                                   true /* constant stride */,
                                   nullptr /* default value */,
                                   false /* hidden */,
                                   false /* transient */);

    openvdb::Index64 count = openvdb::points::pointCount(mPoints->tree());
    std::cout << "PointCount=" << count << std::endl;
}


void
FlipSolver::initializeDamBreak() {
    using BBox = math::BBox<Vec3s>;

    mXform = math::Transform::createLinearTransform(mVoxelSize);
    float const padding = 2 * mVoxelSize;

    Vec3s minFI = Vec3s(0.f, 0.f, 0.f);
    Vec3s maxFI = Vec3s(2.f + mVoxelSize, 4.f + mVoxelSize, 5.f + mVoxelSize);
    Coord minFICoord = mXform->worldToIndexNodeCentered(minFI);
    Coord maxFICoord = mXform->worldToIndexNodeCentered(maxFI);
    FloatGrid::Ptr fluidLSInit = FloatGrid::create(/*bg = */0.f);
    fluidLSInit->denseFill(CoordBBox(minFICoord, maxFICoord), /*value = */ 1.0, /*active = */ true);
    fluidLSInit->setTransform(mXform);

    Vec3s maxIntr = Vec3s(14.f + mVoxelSize, 5.f + mVoxelSize, 5.f + mVoxelSize);
    Coord maxFIIntrCoord = mXform->worldToIndexNodeCentered(maxIntr);
    FloatGrid::Ptr negativeSpace = FloatGrid::create(/*bg = */0.f);
    negativeSpace->denseFill(CoordBBox(minFICoord, maxFIIntrCoord), /*value = */ 1.0, /*active = */ true);
    negativeSpace->setTransform(mXform);

    Vec3s minBBoxvec = Vec3s(-padding, -padding, -padding);
    Vec3s maxBBoxvec = Vec3s(14.f + padding + mVoxelSize, 5.f + padding + mVoxelSize, 5.f + padding + mVoxelSize);
    Coord minBBoxcoord = mXform->worldToIndexNodeCentered(minBBoxvec);
    Coord maxBBoxcoord = mXform->worldToIndexNodeCentered(maxBBoxvec);
    mCollider = FloatGrid::create(/*bg = */0.f);
    mCollider->denseFill(CoordBBox(minBBoxcoord, maxBBoxcoord), /*value = */ 1.0, /*active = */ true);
    mCollider->setTransform(mXform);
    mCollider->topologyDifference(*negativeSpace);
    mCollider->topologyDifference(*fluidLSInit);
    mCollider->setName("collider");
    openvdb::tools::pruneInactive(mCollider->tree());

    mPoints = points::denseUniformPointScatter(*fluidLSInit, mPointsPerVoxel);
    mPoints->setName("Points");
    points::appendAttribute<Vec3s>(mPoints->tree(),
                                   "velocity" /* attribute name */,
                                   Vec3s(0.f, 0.f, 0.f) /* uniform value */,
                                   1 /* stride or total count */,
                                   true /* constant stride */,
                                   nullptr /* default value */,
                                   false /* hidden */,
                                   false /* transient */);
    points::appendAttribute<Vec3s>(mPoints->tree(),
                                   "v_pic" /* attribute name */,
                                   Vec3s(0.f, 0.f, 0.f) /* uniform value */,
                                   1 /* stride or total count */,
                                   true /* constant stride */,
                                   nullptr /* default value */,
                                   false /* hidden */,
                                   false /* transient */);
    points::appendAttribute<Vec3s>(mPoints->tree(),
                                   "v_flip" /* attribute name */,
                                   Vec3s(0.f, 0.f, 0.f) /* uniform value */,
                                   1 /* stride or total count */,
                                   true /* constant stride */,
                                   nullptr /* default value */,
                                   false /* hidden */,
                                   false /* transient */);

    openvdb::Index64 count = openvdb::points::pointCount(mPoints->tree());
    std::cout << "PointCount=" << count << std::endl;
}


void
FlipSolver::particlesToGrid(){
    TreeBase::Ptr baseVTree = points::rasterizeTrilinear<true /* staggered */, Vec3s>(mPoints->tree(), "velocity");

    Vec3STree::Ptr velTree = DynamicPtrCast<Vec3STree>(baseVTree);
    mVCurr = Vec3SGrid::create(velTree);
    mVCurr->setGridClass(GRID_STAGGERED);
    mVCurr->setTransform(mXform);
    mVCurr->setName("v_curr");

    mVNext = mVCurr->deepCopy();
    mVNext->setName("v_next");

    // Determine the fluid domain
    mInteriorPressure = BoolGrid::create(false);
    mInteriorPressure->tree().topologyUnion(mPoints->tree());
    mInteriorPressure->tree().topologyDifference(mCollider->tree());
    mInteriorPressure->tree().voxelizeActiveTiles();
    mInteriorPressure->setTransform(mXform);
    mInteriorPressure->setName("interior_pressure");
}


void
FlipSolver::addGravity(float const dt) {
    tree::LeafManager<Vec3STree> lm(mVCurr->tree());
    FlipSolver::ApplyGravityOp op(dt, mGravity);
    lm.foreach(op);
}


void
FlipSolver::computeFlipVelocity(float const dt) {
    mVDiff = Vec3SGrid::create(Vec3s(0.f, 0.f, 0.f));
    (mVDiff->tree()).topologyUnion(mVNext->tree());
    mVDiff->setGridClass(GRID_STAGGERED);
    mVDiff->setTransform(mXform);

    tree::LeafManager<Vec3STree> lm(mVDiff->tree());
    FlipSolver::ComputeFlipVelocityOp op(mVCurr, mVNext, dt, mGravity);
    lm.foreach(op);
}


void
FlipSolver::extrapolateToCollider(Vec3SGrid& vecGrid) {
    BoolGrid::Ptr narrowBand = BoolGrid::create(false);
    (narrowBand->tree()).topologyUnion(vecGrid.tree());
    tools::dilateActiveValues(narrowBand->tree(), /*iterations=*/1, tools::NN_FACE, tools::IGNORE_TILES);
    narrowBand->topologyDifference(vecGrid);
    Vec3SGrid::Ptr copyVel = vecGrid.deepCopy();

    auto velAcc = vecGrid.getAccessor();
    auto copyAcc = copyVel->getAccessor();
    auto cldrAcc = mCollider->getConstAccessor();
    auto nbAcc = narrowBand->getConstAccessor();
    auto intAcc = mInteriorPressure->getConstAccessor();

    for (auto iter = mCollider->beginValueOn(); iter; ++iter) {
        math::Coord ijk = iter.getCoord();
        math::Coord im1jk = ijk.offsetBy(-1, 0, 0);
        math::Coord ijm1k = ijk.offsetBy(0, -1, 0);
        math::Coord ijkm1 = ijk.offsetBy(0, 0, -1);
        math::Coord ip1jk = ijk.offsetBy(1, 0, 0);
        math::Coord ijp1k = ijk.offsetBy(0, 1, 0);
        math::Coord ijkp1 = ijk.offsetBy(0, 0, 1);

        auto val = velAcc.getValue(ijk);

        float wx[4] = {static_cast<float>(intAcc.isValueOn(ijm1k)),
                       static_cast<float>(intAcc.isValueOn(ijp1k)),
                       static_cast<float>(intAcc.isValueOn(ijkm1)),
                       static_cast<float>(intAcc.isValueOn(ijkp1))};
        float wxsum = wx[0] + wx[1] + wx[2] + wx[3];

        if (wxsum > 0.f) {
            float v[4] = {copyAcc.getValue(ijm1k)[0],
                          copyAcc.getValue(ijp1k)[0],
                          copyAcc.getValue(ijkm1)[0],
                          copyAcc.getValue(ijkp1)[0]};
        float w[4] = {static_cast<float>(intAcc.isValueOn(ijm1k)),
                       static_cast<float>(intAcc.isValueOn(ijp1k)),
                       static_cast<float>(intAcc.isValueOn(ijkm1)),
                       static_cast<float>(intAcc.isValueOn(ijkp1))};
            float normalize = 1.f / (w[0] + w[1] + w[2] + w[3]);
            float extrapolate = 1 * (w[0] * v[0] + w[1] * v[1] + w[2] * v[2] + w[3] * v[3]);
            val[0] = extrapolate;
        }

        float wy[4] = {static_cast<float>(intAcc.isValueOn(im1jk)),
                       static_cast<float>(intAcc.isValueOn(ip1jk)),
                       static_cast<float>(intAcc.isValueOn(ijkm1)),
                       static_cast<float>(intAcc.isValueOn(ijkp1))};
        float wysum = (wy[0] + wy[1] + wy[2] + wy[3]);

        if (wysum >0.f ) {
        //if (nbAcc.isValueOn(ijm1k)) {
            float v[4] = {copyAcc.getValue(im1jk)[1],
                          copyAcc.getValue(ip1jk)[1],
                          copyAcc.getValue(ijkm1)[1],
                          copyAcc.getValue(ijkp1)[1]};
            float w[4] = {static_cast<float>(intAcc.isValueOn(im1jk)),
                          static_cast<float>(intAcc.isValueOn(ip1jk)),
                          static_cast<float>(intAcc.isValueOn(ijkm1)),
                          static_cast<float>(intAcc.isValueOn(ijkp1))};
            float normalize = 1.f / (w[0] + w[1] + w[2] + w[3]);
            float extrapolate = normalize * (w[0] * v[0] + w[1] * v[1] + w[2] * v[2] + w[3] * v[3]);
            val[1] = extrapolate;
            // if (normalize < 1.f) {
            //     std::cout << "warning! ijk = " << ijk << "1./normalize = " << 1.f/normalize << std::endl;
            // }
        }

        float wz[4] = {static_cast<float>(intAcc.isValueOn(im1jk)),
                       static_cast<float>(intAcc.isValueOn(ip1jk)),
                       static_cast<float>(intAcc.isValueOn(ijm1k)),
                       static_cast<float>(intAcc.isValueOn(ijp1k))};
        float wzsum = wz[0] + wz[1] + wz[2] + wz[3];
        if (wzsum > 0.f ) {
            float v[4] = {copyAcc.getValue(im1jk)[2],
                          copyAcc.getValue(ip1jk)[2],
                          copyAcc.getValue(ijm1k)[2],
                          copyAcc.getValue(ijp1k)[2]};
            float w[4] = {static_cast<float>(intAcc.isValueOn(im1jk)),
                          static_cast<float>(intAcc.isValueOn(ip1jk)),
                          static_cast<float>(intAcc.isValueOn(ijm1k)),
                          static_cast<float>(intAcc.isValueOn(ijp1k))};
            float normalize = 1.f / (w[0] + w[1] + w[2] + w[3]);
            float extrapolate = 1 * (w[0] * v[0] + w[1] * v[1] + w[2] * v[2] + w[3] * v[3]);
            val[2] = extrapolate;
        }
        velAcc.setValue(ijk, val);
    }
}


void
FlipSolver::velocityBCCorrection(Vec3SGrid::Ptr vecGrid) {
    tree::LeafManager<FloatTree> lmc(mCollider->tree());
    FlipSolver::VelocityBCCorrectionOp opVelCorrection(vecGrid, mCollider, mInteriorPressure);
    lmc.foreach(opVelCorrection);
}


float
FlipSolver::computeDivergence(FloatGrid::Ptr& divGrid, const Vec3SGrid::Ptr vecGrid, const std::string& suffix) {
    std::string name = "div_";
    name += suffix.c_str();

    divGrid = tools::divergence(*vecGrid);
    divGrid->tree().topologyIntersection(mInteriorPressure->tree());
    divGrid->setName(name.c_str());
    float div = computeLInfinity(*divGrid);
    std::cout << "Divergence " << suffix.c_str() << " = " << div << std::endl;
    return div;
}


float
FlipSolver::computeLInfinity(const FloatGrid& grid) {
    float ret = 0.f;
    auto acc = grid.getConstAccessor();
    for (auto iter = grid.beginValueOn(); iter; ++iter) {
        math::Coord ijk = iter.getCoord();
        auto val = acc.getValue(ijk);
        if (std::abs(val) > std::abs(ret)) {
            ret = val;
        }
    }
    return ret;
}

void
FlipSolver::pressureProjection(bool print) {
    using TreeType = FloatTree;
    using ValueType = TreeType::ValueType;
    using MaskGridType = BoolGrid;
    using PCT = openvdb::math::pcg::JacobiPreconditioner<openvdb::tools::poisson::LaplacianMatrix>;
    const double epsilon = math::Delta<ValueType>::value();

    math::pcg::State state = math::pcg::terminationDefaults<ValueType>();
    state.iterations = 100000;
    state.relativeError = state.absoluteError = epsilon;
    FlipSolver::BoundaryOp bop(mVoxelSize, mCollider, mVCurr);
    util::NullInterrupter interrupter;

    // Compute the right hand size
    computeDivergence(mDivBefore, mVCurr, "before");

    // Solve the Poisson equation
    FloatTree::Ptr fluidPressure = tools::poisson::solveWithBoundaryConditionsAndPreconditioner<PCT>(
        mDivBefore->tree(), mInteriorPressure->tree(), bop, state, interrupter, /*staggered=*/true);

    FloatGrid::Ptr fluidPressureGrid = FloatGrid::create(fluidPressure);
    fluidPressureGrid->setTransform(mXform);
    mPressure = fluidPressureGrid->copy();
    mPressure->setName("pressure");

    // Pressure projection: subtract grad p from current velocity
    tree::LeafManager<Vec3STree> lmv(mVNext->tree());
    FlipSolver::SubtractPressureGradientOp opSubtractGradP(mInteriorPressure, fluidPressureGrid, mVCurr, mVoxelSize);
    lmv.foreach(opSubtractGradP);

    if (print) {
        std::cout << "Projection success: " << state.success << std::endl;
        std::cout << "Projection iterations: " << state.iterations << std::endl;
        std::cout << "Projection relative error: " << state.relativeError << std::endl;
        std::cout << "Projection absolute error: " << state.absoluteError << std::endl;
        std::cout << "Pressure->activeVoxelCount() =  " << fluidPressure->activeVoxelCount() << std::endl;
    }
}

void
FlipSolver::gridVelocityUpdate(float const dt) {
    addGravity(dt);
    velocityBCCorrection(mVCurr);
    pressureProjection(true /* print */);
    velocityBCCorrection(mVNext);
    computeDivergence(mDivAfter, mVNext, "after");

    extrapolateToCollider(*mVCurr);
    extrapolateToCollider(*mVNext);
    computeFlipVelocity(dt);
}


void
FlipSolver::substep(float const dt) {
    particlesToGrid();
    gridVelocityUpdate(dt);
    gridToParticles();
    updateParticles(dt);
}


void
FlipSolver::updateParticlesVelocity() {
    // Create a leaf iterator for the PointDataTree.
    auto leafIter = (mPoints->tree()).beginLeaf();

    // Retrieve the index from the descriptor.
    // Used to get the array attribute in the functor.
    auto descriptor = leafIter->attributeSet().descriptor();
    Index64 velIdx = descriptor.find("velocity");
    Index64 vPicIdx = descriptor.find("v_pic");
    Index64 vFlipIdx = descriptor.find("v_flip");

    // PIC/FLIP update
    tree::LeafManager<points::PointDataTree> leafManager(mPoints->tree());
    FlipSolver::FlipUpdateOp op(velIdx, vPicIdx, vFlipIdx, 0.01f /* alpha in PIC/FlIP update */);
    tbb::parallel_for(leafManager.leafRange(), op);
}


void
FlipSolver::updateParticles(float const dt) {
    updateParticlesVelocity();
    advectParticles(dt);
}


void
FlipSolver::render() {
    float const dt = 1.f/24.f;
    for (int frame = 0; frame < 200; ++frame) {
        std::cout << "\nframe = " << frame << "\n";
        float numSubStep = 10.f;
        for (int i = 0; i < static_cast<int>(numSubStep); ++i) {
            std::cout << "\n==frame = " << frame << "==\n";
            substep(dt/numSubStep);
        }
        writeVDBs(frame);
        writeVDBsVerbose(frame);
    }
}


void
FlipSolver::gridToParticles() {
    // Interpolate PIC velocity
    points::boxSample(*mPoints, *mVNext, "v_pic");

    // Interpolate FLIP velocity
    points::boxSample(*mPoints, *mVDiff, "v_flip");
}


void
FlipSolver::advectParticles(float const dt) {
    Index const integrationOrder = 4;
    int const steps = 1;

    points::advectPoints(*mPoints, *mVNext, integrationOrder, dt, steps);
}


void
FlipSolver::writeVDBs(int const frame) {
    std::ostringstream ss;
    ss << "water_particles_" << std::setw(3) << std::setfill('0') << frame << ".vdb";
    std::string fileName(ss.str());
    io::File file(fileName.c_str());
    file.write({mPoints});
    file.close();
}


void
FlipSolver::writeVDBsVerbose(int const frame) {
    std::ostringstream ss;
    ss << "water_volume_" << std::setw(3) << std::setfill('0') << frame << ".vdb";
    std::string fileName(ss.str());
    openvdb::io::File file(fileName.c_str());

    openvdb::GridPtrVec grids;
    grids.push_back(mCollider);
    grids.push_back(mVCurr);
    grids.push_back(mVNext);
    grids.push_back(mDivBefore);
    grids.push_back(mDivAfter);
    grids.push_back(mPressure);
    grids.push_back(mInteriorPressure);
    file.write(grids);
    file.close();
}
}
