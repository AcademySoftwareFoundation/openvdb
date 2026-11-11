// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file unittest/LevelSetMultiObjectFixtures.h
///
/// @brief Test-only fixture builders shared by TestLevelSetMultiObject.cc and
/// TestLevelSetMultiObjectMesh.cc: two spheres, the seeded triple-junction
/// set and the eight-object vortex.

#ifndef OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_FIXTURES_HAS_BEEN_INCLUDED
#define OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_FIXTURES_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTubes.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace lsmo_test {

using MLS = openvdb::tools::MultiLevelSet<>;
/// Label tree of the default MultiLevelSet (16-bit labels).
using LabelTree = MLS::LabelGridT::TreeType;
using Forces = openvdb::tools::MultiObjectForces<MLS>;


////////////////////////////////////////
// Two spheres

/// Spheres of radius 10*dx at (-+15*dx, 0, 0), labels 1 and 2, via
/// fromSignedGrids, as a @c MultiLevelSetT (the default MultiLevelSet unless
/// given).
template<typename MultiLevelSetT = MLS>
inline typename MultiLevelSetT::Ptr
makeTwoSpheres(double dx)
{
    using namespace openvdb;
    const double r = 10.0 * dx, cx = 15.0 * dx;
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(float(-cx), 0, 0), float(dx));
    in[2] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(float(cx), 0, 0), float(dx));
    return MultiLevelSetT::fromSignedGrids(in);
}


////////////////////////////////////////
// Triple junction

/// Three labels meeting at 120 degrees along a straight junction line through
/// index point c with unit direction a. In the plane frame (u, v) normal to a
/// (u = x projected off a, v = a x u, so u = x and v = y for a z-parallel
/// line), label l (1..3) owns the polar angles
/// [theta0 + (l-1)*120deg, theta0 + l*120deg) about the line.
struct TripleJunction
{
    using Vec2d = openvdb::Vec2d;
    using Vec3d = openvdb::Vec3d;

    double dx = 0.5;
    double theta0;
    Vec3d c, a, u, v;  // index space
    int H;             // active box [-H, H-1]^3

    TripleJunction(double th, const Vec3d& center, const Vec3d& dir, int h = 12)
        : theta0(th), c(center), a(dir.unitSafe()), H(h)
    {
        const Vec3d ref = std::abs(a[0]) < 0.9 ? Vec3d(1, 0, 0) : Vec3d(0, 1, 0);
        u = (ref - a * ref.dot(a)).unitSafe();
        v = a.cross(u);
    }

    Vec2d local(const Vec3d& w) const
    {
        const Vec3d d = w / dx - c;
        return Vec2d(d.dot(u), d.dot(v));
    }
    Vec2d ray(int k) const
    {
        const double t = theta0 + k * 2.0 * openvdb::math::pi<double>() / 3.0;
        return Vec2d(std::cos(t), std::sin(t));
    }
    int sector(const Vec3d& w) const
    {
        const Vec2d p = local(w);
        double t = std::atan2(p[1], p[0]) - theta0;
        while (t < 0.0) t += 2.0 * openvdb::math::pi<double>();
        while (t >= 2.0 * openvdb::math::pi<double>()) t -= 2.0 * openvdb::math::pi<double>();
        return std::min(3, int(t / (2.0 * openvdb::math::pi<double>() / 3.0)) + 1);
    }
    /// Distance from p to the ray {t*r, t >= 0} (index units).
    static double rayDist(const Vec2d& p, const Vec2d& r)
    {
        const double t = p.dot(r);
        return t >= 0.0 ? (p - r * t).length() : p.length();
    }
    /// Analytic signed distance to sector @a l, world units.
    double signedDist(const Vec3d& w, int l) const
    {
        const Vec2d p = local(w);
        const double d = std::min(rayDist(p, ray(l - 1)), rayDist(p, ray(l)));
        return dx * (sector(w) == l ? -d : d);
    }
    double lineDist(const Vec3d& w) const { return local(w).length(); } // index units
    /// Chebyshev distance (index units) from the voxel center to the box faces.
    double boxDepth(const openvdb::Coord& p) const
    {
        return H - std::max({std::abs(p[0] + 0.5), std::abs(p[1] + 0.5), std::abs(p[2] + 0.5)});
    }

    /// The pair built directly from analytic data: phi = distance to the
    /// voxel's own sector boundary, active where phi < 3*dx inside the box,
    /// and a one-voxel inactive shell holding the labels.
    MLS::Ptr build() const
    {
        using namespace openvdb;
        const int halfWidth = 3;
        auto xform = math::Transform::createLinearTransform(dx);
        FloatGrid::Ptr phi = FloatGrid::create(float(halfWidth * dx));
        MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
        phi->setTransform(xform);
        lab->setTransform(xform->copy());
        auto pa = phi->getAccessor();
        auto la = lab->getAccessor();
        for (int i = -H - 1; i <= H; ++i)
            for (int j = -H - 1; j <= H; ++j)
                for (int k = -H - 1; k <= H; ++k) {
                    const Coord q(i, j, k);
                    const Vec3d w = xform->indexToWorld(q);
                    const int l = sector(w);
                    const double d = std::abs(signedDist(w, l));
                    if (boxDepth(q) > 0.0 && d < halfWidth * dx) {
                        pa.setValue(q, float(d));
                        la.setValue(q, static_cast<MLS::LabelT>(l));
                    } else {
                        pa.touchLeaf(q);
                        la.setValueOff(q, static_cast<MLS::LabelT>(l));
                    }
                }
        return MLS::Ptr(new MLS(phi, lab, halfWidth));
    }
};

/// Seeded junction set: the original fixture (theta0 = 10deg,
/// z-parallel through (0.37, -0.21)), seven z-parallel lines with random
/// theta0 and sub-voxel offset, and four tilted lines. Uses its own
/// generator arithmetic so the set is the same on every platform; random
/// numbers are drawn in a fixed order (theta, center xyz, direction xy).
inline std::vector<TripleJunction>
junctionSet()
{
    using openvdb::Vec3d;
    std::mt19937 rng(20261108u);
    auto u01 = [&rng]() { return double(rng()) / 4294967296.0; };
    std::vector<TripleJunction> set;
    set.emplace_back(10.0 * openvdb::math::pi<double>() / 180.0, Vec3d(0.37, -0.21, 0.0), Vec3d(0, 0, 1));
    // Draw into named locals: the order of evaluation of constructor
    // arguments is unspecified, the order of these statements is not.
    for (int n = 0; n < 7; ++n) {
        const double th = u01() * 2.0 * openvdb::math::pi<double>();
        const double cx = u01() - 0.5, cy = u01() - 0.5, cz = u01() - 0.5;
        set.emplace_back(th, Vec3d(cx, cy, cz), Vec3d(0, 0, 1));
    }
    for (int n = 0; n < 4; ++n) {
        const double th = u01() * 2.0 * openvdb::math::pi<double>();
        const double cx = u01() - 0.5, cy = u01() - 0.5, cz = u01() - 0.5;
        const double dxx = u01() - 0.5, dyy = u01() - 0.5;
        set.emplace_back(th, Vec3d(cx, cy, cz), Vec3d(dxx, dyy, 1.0));
    }
    return set;
}


////////////////////////////////////////
// Velocity fields and the vortex

/// Dense Vec3SGrid over the index box @a box (voxel size @a dx), each voxel
/// set to f(its world position); background zero.
inline openvdb::Vec3SGrid::Ptr
makeVelocity(const openvdb::CoordBBox& box, double dx,
    const std::function<openvdb::Vec3s(const openvdb::Vec3d&)>& f)
{
    using namespace openvdb;
    Vec3SGrid::Ptr vel = Vec3SGrid::create(Vec3s(0.0f));
    vel->setTransform(math::Transform::createLinearTransform(dx));
    vel->denseFill(box, Vec3s(0.0f), /*active=*/true);
    tree::LeafManager<Vec3STree> leaves(vel->tree());
    leaves.foreach([&](Vec3STree::LeafNodeType& leaf, size_t) {
        for (auto it = leaf.beginValueOn(); it; ++it) {
            it.setValue(f(vel->indexToWorld(it.getCoord())));
        }
    });
    return vel;
}

/// Voxel count per label over the whole label tree (active and inactive
/// voxels, tiles counted by their voxel count; label 0 excluded).
template<typename MultiLevelSetT>
inline std::map<int32_t, openvdb::Index64>
labelVoxelCounts(const MultiLevelSetT& m)
{
    std::map<int32_t, openvdb::Index64> n;
    for (auto it = m.labels()->tree().cbeginValueAll(); it; ++it) {
        if (*it == 0) continue;
        n[int32_t(*it)] += it.getVoxelCount();
    }
    return n;
}

/// Distinct unordered label pairs (a, b), a != b, both non-zero, with
/// face-adjacent voxels (one of them active).
template<typename MultiLevelSetT>
inline std::set<std::pair<int32_t, int32_t>>
contactPairs(const MultiLevelSetT& m)
{
    using namespace openvdb;
    using LabelTreeT = typename MultiLevelSetT::LabelGridT::TreeType;
    std::set<std::pair<int32_t, int32_t>> pairs;
    tree::ValueAccessor<const LabelTreeT> lab(m.labels()->tree());
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    for (auto it = m.labels()->tree().cbeginValueOn(); it; ++it) {
        const int32_t a = *it;
        if (a == 0) continue;
        for (int f = 0; f < 6; ++f) {
            const int32_t b = lab.getValue(it.getCoord() + offs[f]);
            if (b != 0 && b != a) pairs.insert({std::min(a, b), std::max(a, b)});
        }
    }
    return pairs;
}

/// Vortex field: a divergence-free inward spiral towards the origin (the
/// center of the velocity grid), in cylindrical coordinates (r, theta, z)
/// about the z axis:
/// - radial inflow u_r = -inflow*tanh(r/core): ~inflow far from the axis,
///   vanishing linearly at it;
/// - solid-body swirl u_theta = angularVelocity*r (rigid rotation, so
///   neighbors on one circle do not shear past each other);
/// - axial outflow u_z = z*D(r) with D = -(1/r) d(r u_r)/dr, which cancels
///   the horizontal convergence: the flow preserves volume, so objects
///   spiral in while stretching along z about the plane z = 0.
/// An exact flow never brings disjoint objects into contact (gaps between
/// neighbors on one circle shrink in proportion to r); contact comes from
/// those gaps closing below a voxel as the objects are drawn inwards.
struct Vortex
{
    /// Schedule: chunk length, cap on the time to first contact, end time.
    static constexpr double kChunk = 2.0, kContactCap = 70.0, kEndTime = 60.0;

    /// Parameters (see runVortexEightObjects in TestLevelSetMultiObject.cc for
    /// why they were chosen).
    static constexpr double kShapeScale = 1.4;      ///< factor on every shape dimension
    static constexpr double kRingPolygonRadius = 5.0; ///< label 8 polygon radius before scaling
    static constexpr double kStartRadius = 34.0;    ///< initial distance of centers from the axis
    static constexpr double kInflow = 0.4;          ///< far-field radial speed towards the axis
    static constexpr double kCoreRadius = 8.0;      ///< inflow vanishes linearly inside ~this
    static constexpr double kAngularVelocity = 0.012; ///< swirl, radians per unit time
    static constexpr double kCurvatureWeight = 0.003;
    static constexpr int kBoxHalfWidth = 80;        ///< velocity grid is [-80, 80]^3
    static constexpr int kNumObjects = 8;
    /// Below this r/core, convergence() uses its limit at the axis.
    static constexpr double kAxisTolerance = 1e-6;
    double inflow = kInflow, core = kCoreRadius, angularVelocity = kAngularVelocity,
        r0 = kStartRadius, ringRadius = kRingPolygonRadius;
    double scale = kShapeScale, curvature = kCurvatureWeight;
    int box = kBoxHalfWidth;

    /// D(r) = -(1/r) d(r u_r)/dr, the horizontal convergence rate.
    double convergence(double r) const
    {
        const double x = r / core;
        if (x < kAxisTolerance) return 2.0 * inflow / core;
        const double sech = 1.0 / std::cosh(x);
        return inflow * (std::tanh(x) + x * sech * sech) / r;
    }
    openvdb::Vec3s operator()(const openvdb::Vec3d& p) const
    {
        using openvdb::Vec3d;
        const double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
        const double ur = -inflow * std::tanh(r / core);
        Vec3d v = Vec3d(-p[1], p[0], 0.0) * angularVelocity
            + Vec3d(0.0, 0.0, p[2] * convergence(r));
        if (r > 0.0) v += Vec3d(p[0], p[1], 0.0) * (ur / r);
        return openvdb::Vec3s(v);
    }

    /// The vortex velocity grid: dense over [-80, 80]^3, dx = 1.
    openvdb::Vec3SGrid::Ptr velocity() const
    {
        return makeVelocity(openvdb::CoordBBox(openvdb::Coord(-box), openvdb::Coord(box)), 1.0,
            *this);
    }

    /// Shape of label @a l (1..8) centered at @a c, dx = 1, half-width 3.
    /// Base (pre-scale) sizes: sphere r = 10; cube side 12; tetrahedron,
    /// octahedron, dodecahedron, icosahedron with factory scale (circumradius)
    /// 10; capsule along z with radius 5 and axis endpoints +-5; closed ring
    /// = 8-segment polygon of radius kRingPolygonRadius in the xy plane with
    /// tube radius 3.
    /// Every length is multiplied by kShapeScale. No rotations.
    openvdb::FloatGrid::Ptr shape(int32_t l, const openvdb::Vec3f& c) const
    {
        using namespace openvdb;
        const float S = float(scale);
        const float dx = 1.0f, hw = 3.0f;
        switch (l) {
        case 1: return tools::createLevelSetSphere<FloatGrid>(10.0f * S, c, dx, hw);
        case 2: return tools::createLevelSetCube<FloatGrid>(12.0f * S, c, dx, hw);
        case 3: return tools::createLevelSetTetrahedron<FloatGrid>(10.0f * S, c, dx, hw);
        case 4: return tools::createLevelSetOctahedron<FloatGrid>(10.0f * S, c, dx, hw);
        case 5: return tools::createLevelSetDodecahedron<FloatGrid>(10.0f * S, c, dx, hw);
        case 6: return tools::createLevelSetIcosahedron<FloatGrid>(10.0f * S, c, dx, hw);
        case 7:
            return tools::createLevelSetCapsule<FloatGrid>(
                c - Vec3f(0.0f, 0.0f, 5.0f * S), c + Vec3f(0.0f, 0.0f, 5.0f * S), 5.0f * S, dx, hw);
        default: {
            std::vector<Vec3f> verts;
            std::vector<Vec2I> segs;
            for (int i = 0; i < 8; ++i) {
                const double a = i * math::pi<double>() / 4.0;
                verts.push_back(c + Vec3f(float(ringRadius * S * std::cos(a)),
                    float(ringRadius * S * std::sin(a)), 0.0f));
                segs.push_back(Vec2I(i, (i + 1) % 8));
            }
            return tools::createLevelSetTubeComplex<FloatGrid>(verts, segs, 3.0f * S, dx, hw);
        }
        }
    }

    /// Initial signed grids, labels 1-8: label l centered at angle
    /// (l-1)*45 degrees on the circle r = r0 in the plane z = 0, i.e. equally
    /// spaced around the periphery of the vortex.
    std::map<int32_t, openvdb::FloatGrid::ConstPtr> inputs() const
    {
        using namespace openvdb;
        std::map<int32_t, FloatGrid::ConstPtr> in;
        for (int32_t l = 1; l <= kNumObjects; ++l) {
            const double a = (l - 1) * 2.0 * math::pi<double>() / kNumObjects;
            in[l] = shape(l, Vec3f(Vec3d(std::cos(a), std::sin(a), 0.0) * r0));
        }
        return in;
    }

    /// @brief Run the vortex schedule on @a m: advection weight 1, curvature
    /// weight kCurvatureWeight, chunks of kChunk until >= 4 contact pairs (capped at
    /// t = kContactCap), then on in chunks to the end time kEndTime.
    /// @details perChunk(t, steps, pairs, afterContact) runs after every
    /// chunk (afterContact: chunks completed after the first chunk with
    /// >= 4 pairs, -1 before); returning false stops the run.
    /// @return the final time; @a tContact receives the time of the first
    /// chunk with >= 4 pairs (-1 if none).
    /// @a configure is applied to the evolver before the run (e.g. to
    /// select its schemes; the defaults otherwise).
    using Evolver = openvdb::tools::LevelSetMultiObject<MLS, Forces>;
    template<typename PerChunk>
    double run(MLS& m, const openvdb::Vec3SGrid::Ptr& vel, size_t& steps, double& tContact,
        PerChunk&& perChunk, const std::function<void(Evolver&)>& configure = {}) const
    {
        Forces forces;
        forces.setAdvection(vel, 1.0f);
        forces.setCurvatureWeight(float(curvature));
        Evolver lsmo(m, forces);
        if (configure) configure(lsmo);
        double t = 0.0;
        steps = 0;
        tContact = -1.0;
        int after = -1;
        while (after < 0 || t < kEndTime - 1e-9) {
            if (after < 0 && t >= kContactCap - 1e-9) break;
            const double t1 = after >= 0 ? std::min(t + kChunk, kEndTime) : t + kChunk;
            steps += lsmo.advect(t, t1);
            t = t1;
            const auto pairs = contactPairs(m);
            if (after >= 0) {
                ++after;
            } else if (pairs.size() >= 4) {
                after = 0;
                tContact = t;
            }
            if (!perChunk(t, steps, pairs, after)) break;
        }
        return t;
    }

    /// The vortex end state (labels 1-8 after run()).
    MLS::Ptr endState() const
    {
        MLS::Ptr m = MLS::fromSignedGrids(inputs(), 3);
        size_t steps = 0;
        double tContact = -1.0;
        run(*m, velocity(), steps, tContact,
            [](double, size_t, const std::set<std::pair<int32_t, int32_t>>&, int) { return true; });
        return m;
    }
};

} // namespace lsmo_test

#endif // OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_FIXTURES_HAS_BEEN_INCLUDED
