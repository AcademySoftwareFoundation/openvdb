// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetMultiObject.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTubes.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VelocityFields.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "LevelSetMultiObjectFixtures.h"
#include "LevelSetMultiObjectTopology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class TestLevelSetMultiObject: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};

namespace {

using namespace openvdb;
using MLS = lsmo_test::MLS;
using LabelTree = lsmo_test::LabelTree;
using lsmo_test::contactPairs;
using lsmo_test::junctionSet;
using lsmo_test::labelVoxelCounts;
using lsmo_test::makeTwoSpheres;
using lsmo_test::makeVelocity;
using lsmo_test::Vortex;
using lsmo_test::TripleJunction;

// 16^3 block: label 1 for x<8, 2 for x>=8, phi = |x - 7.5|*dx clamped to 3*dx.
template<typename MultiLevelSetT = MLS>
typename MultiLevelSetT::Ptr
makePair(double dx = 1.0)
{
    using LabelGridT = typename MultiLevelSetT::LabelGridT;
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(3.0 * dx));
    typename LabelGridT::Ptr lab = LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 16; ++k) {
                const Coord c(i, j, k);
                const float d = std::min(float(std::abs(i - 7.5) * dx), float(3.0 * dx));
                pa.setValue(c, d);
                la.setValue(c, i < 8 ? 1 : 2);
            }
    return typename MultiLevelSetT::Ptr(new MultiLevelSetT(phi, lab, 3));
}

} // anonymous namespace

namespace {
const char* const kBackgroundMessage =
    "phi background must equal halfWidth * voxel size and the label background must be 0";

template<typename T, typename = void>
struct HasSetClampSpeed : std::false_type {};
template<typename T>
struct HasSetClampSpeed<T, decltype(void(std::declval<T&>().setClampSpeed(true)))>
    : std::true_type {};
template<typename T, typename = void>
struct HasClampSpeed : std::false_type {};
template<typename T>
struct HasClampSpeed<T, decltype(void(std::declval<const T&>().clampSpeed()))>
    : std::true_type {};

} // anonymous namespace


TEST_F(TestLevelSetMultiObject, HolderValidate)
{
    auto m = makePair();
    std::string msg;
    EXPECT_TRUE(m->validate(&msg));
    EXPECT_EQ(m->phi()->metaValue<std::string>("multi_level_set_role"), "phi");
    EXPECT_EQ(m->labels()->metaValue<std::string>("multi_level_set_role"), "labels");
    EXPECT_EQ(m->phi()->getGridClass(), GRID_UNKNOWN);

    m->phi()->tree().setValueOn(Coord(2, 2, 2), -1.0f);
    msg.clear();
    EXPECT_FALSE(m->validate(&msg));
    EXPECT_EQ(msg, "phi must be non-negative and finite at every active voxel");

    auto m2 = makePair();
    m2->labels()->tree().setValueOn(Coord(40, 0, 0), 1);
    msg.clear();
    EXPECT_FALSE(m2->validate(&msg));
    EXPECT_EQ(msg, "phi and label trees have different active topology");
}

TEST_F(TestLevelSetMultiObject, HolderThrows)
{
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(math::Transform::createLinearTransform(1.0));
    lab->setTransform(math::Transform::createLinearTransform(0.5));
    EXPECT_THROW(MLS(phi, lab, 3), ValueError);

    lab->setTransform(math::Transform::createLinearTransform(1.0));
    EXPECT_THROW(MLS(phi, lab, 1), ValueError);
    // the phi background must be halfWidth*dx
    FloatGrid::Ptr phiHw2 = FloatGrid::create(2.0f);
    phiHw2->setTransform(math::Transform::createLinearTransform(1.0));
    EXPECT_NO_THROW(MLS(phiHw2, lab, 2));

    auto nonUniform = math::Transform::Ptr(new math::Transform(math::MapBase::Ptr(new math::ScaleMap(Vec3d(1.0, 2.0, 1.0)))));
    FloatGrid::Ptr phi2 = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab2 = MLS::LabelGridT::create(0);
    phi2->setTransform(nonUniform);
    lab2->setTransform(nonUniform->copy());
    EXPECT_THROW(MLS(phi2, lab2, 3), ValueError);
}


namespace {

// Bitwise equality of two trees: active topology, leaf layout and every
// value (active or not).
template<typename TreeT>
Index64
treeDifferences(const TreeT& a, const TreeT& b)
{
    if (!a.hasSameTopology(b) || a.leafCount() != b.leafCount()) return Index64(1) << 40;
    Index64 diff = 0;
    tree::ValueAccessor<const TreeT> acc(b);
    for (auto it = a.cbeginValueAll(); it; ++it) {
        const auto va = *it;
        const auto vb = acc.getValue(it.getCoord());
        if (std::memcmp(&va, &vb, sizeof(va)) != 0) ++diff;
    }
    return diff;
}

// Labels 1..3 in 8^3 blocks over [0, 23]^3, label 0 outside.
template<typename GridT>
typename GridT::Ptr
makeBlockLabels()
{
    typename GridT::Ptr labels = GridT::create(0);
    auto acc = labels->getAccessor();
    for (int i = 0; i < 24; ++i)
        for (int j = 0; j < 24; ++j)
            for (int k = 0; k < 24; ++k) {
                acc.setValue(Coord(i, j, k),
                    typename GridT::ValueType(1 + ((i / 8 + j / 8 + k / 8) % 3)));
            }
    return labels;
}

// fromLabelGrid, a few evolver steps (1.5 x the time of five full steps) and
// extractSignedDistance with label tree type LabelTreeT; returns the final phi
// tree and label-1 distance.
template<typename LabelTreeT>
std::pair<FloatTree::Ptr, FloatGrid::Ptr>
runWithLabelTree()
{
    using MLST = tools::MultiLevelSet<FloatTree, LabelTreeT>;
    using ForcesT = tools::MultiObjectForces<MLST>;
    auto labels = makeBlockLabels<Grid<LabelTreeT>>();
    auto m = MLST::fromLabelGrid(*labels);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3}));
    ForcesT forces;
    forces.setPressure(FloatGrid::create(1.0f), 1.0f);
    forces.setCurvatureWeight(0.1f);
    forces.setFixedLabels({3});
    tools::LevelSetMultiObject<MLST, ForcesT> lsmo(*m, forces);
    // 1.5 x the time of five full steps
    double t5 = 0.0;
    for (int n = 0; n < 5; ++n) t5 += lsmo.step(1.0f);
    for (double t = t5; t < 1.5 * t5 - 1e-9;) t += lsmo.step(float(1.5 * t5 - t));
    EXPECT_TRUE(m->validate(&msg)) << msg;
    tools::MultiLevelSetRedistance<MLST>(*m).redistance();
    return std::make_pair(FloatTree::Ptr(new FloatTree(m->phi()->tree())),
        m->extractSignedDistance(1));
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, OtherLabelTreeTypes)
{
    // The distance tree must hold floats (a static_assert); the label tree
    // may hold any integer type. Int16 (the default), Int64 and UInt32 labels
    // give results identical to Int32 labels.
    const auto ref = runWithLabelTree<Int32Tree>();
    const auto i16 = runWithLabelTree<tools::Int16LabelTree>();
    const auto i64 = runWithLabelTree<Int64Tree>();
    const auto u32 = runWithLabelTree<UInt32Tree>();
    ASSERT_GT(ref.second->activeVoxelCount(), Index64(0));
    EXPECT_EQ(treeDifferences(*ref.first, *i16.first), Index64(0));
    EXPECT_EQ(treeDifferences(ref.second->tree(), i16.second->tree()), Index64(0));
    EXPECT_EQ(treeDifferences(*ref.first, *i64.first), Index64(0));
    EXPECT_EQ(treeDifferences(*ref.first, *u32.first), Index64(0));
    EXPECT_EQ(treeDifferences(ref.second->tree(), i64.second->tree()), Index64(0));
    EXPECT_EQ(treeDifferences(ref.second->tree(), u32.second->tree()), Index64(0));
}

TEST_F(TestLevelSetMultiObject, Int16LabelsDefault)
{
    // The default label tree holds 16-bit labels: half the label bytes of
    // the Int32 variant for the same pair.
    static_assert(std::is_same<MLS::LabelGridT::TreeType, tools::Int16LabelTree>::value,
        "the default label tree is Int16LabelTree");
    static_assert(std::is_same<tools::Int16LabelGrid, Grid<tools::Int16LabelTree>>::value,
        "Int16LabelGrid is the grid of Int16LabelTree");
    using MLS32 = tools::MultiLevelSet<FloatTree, Int32Tree>;
    auto m16 = makeTwoSpheres(0.1);
    auto m32 = makeTwoSpheres<MLS32>(0.1);
    std::cout << "Int16LabelsDefault: two-sphere labels memUsage int16 "
              << m16->labels()->memUsage() << " B, int32 " << m32->labels()->memUsage()
              << " B" << std::endl;
    EXPECT_LT(m16->labels()->memUsage(), m32->labels()->memUsage());
    // same pair, same labels
    EXPECT_EQ(treeDifferences(m16->phi()->tree(), m32->phi()->tree()), Index64(0));
    EXPECT_TRUE(m16->labels()->tree().hasSameTopology(m32->labels()->tree()));
    tree::ValueAccessor<const Int32Tree> acc32(m32->labels()->tree());
    for (auto it = m16->labels()->tree().cbeginValueAll(); it; ++it) {
        ASSERT_EQ(int32_t(*it), acc32.getValue(it.getCoord())) << it.getCoord();
    }
}

TEST_F(TestLevelSetMultiObject, Int16RangeThrows)
{
    // Labels are int32_t in the API; storing one outside the range of the
    // 16-bit label tree throws instead of truncating.
    constexpr int32_t kTooLarge = 40000, kTooSmall = -32769;
    constexpr int32_t kMinInt16 = std::numeric_limits<int16_t>::min();
    constexpr int32_t kMaxInt16 = std::numeric_limits<int16_t>::max();
    auto blockLabels = [](int32_t a, int32_t b) {
        Int32Grid::Ptr g = Int32Grid::create(0);
        auto acc = g->getAccessor();
        for (auto p = CoordBBox(Coord(0), Coord(7)).begin(); p; ++p) {
            acc.setValue(*p, (*p)[0] < 4 ? a : b);
        }
        return g;
    };
    // fromLabelGrid, active voxel out of range (both directions)
    EXPECT_THROW(MLS::fromLabelGrid(*blockLabels(1, kTooLarge)), ValueError);
    EXPECT_THROW(MLS::fromLabelGrid(*blockLabels(kTooSmall, 1)), ValueError);
    // ... an inactive voxel and an inactive tile out of range
    {
        Int32Grid::Ptr g = blockLabels(1, 2);
        g->tree().setValueOff(Coord(100, 0, 0), kTooLarge);
        EXPECT_THROW(MLS::fromLabelGrid(*g), ValueError);
        Int32Grid::Ptr t = blockLabels(1, 2);
        t->tree().addTile(1, Coord(256, 0, 0), kTooLarge, false);
        EXPECT_THROW(MLS::fromLabelGrid(*t), ValueError);
    }
    // the extremes of the range are accepted and kept exactly
    {
        auto m = MLS::fromLabelGrid(*blockLabels(kMinInt16, kMaxInt16));
        std::string msg;
        EXPECT_TRUE(m->validate(&msg)) << msg;
        EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{kMinInt16, kMaxInt16}));
    }
    // the Int32 variant keeps large labels
    {
        using MLS32 = tools::MultiLevelSet<FloatTree, Int32Tree>;
        auto m = MLS32::fromLabelGrid(*blockLabels(1, kTooLarge));
        EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, kTooLarge}));
    }
    // fromSignedGrids, a key out of range
    {
        std::map<int32_t, FloatGrid::ConstPtr> in;
        in[1] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(-6, 0, 0), 1.0f);
        in[kTooLarge] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(6, 0, 0), 1.0f);
        EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    }
    // the evolver's kinematic label is stored into the label tree
    {
        auto m = makeTwoSpheres(1.0);
        lsmo_test::Forces forces;
        tools::LevelSetMultiObject<MLS, lsmo_test::Forces> lsmo(*m, forces);
        FloatGrid::Ptr sdf = tools::createLevelSetSphere<FloatGrid>(3.0f, Vec3f(0), 1.0f);
        auto still = [](double) { return math::Mat4d::identity(); };
        EXPECT_THROW(lsmo.setKinematicLabel(kTooLarge, sdf, still), ValueError);
        EXPECT_NO_THROW(lsmo.setKinematicLabel(kMaxInt16, sdf, still));
    }
    // reading a label the tree cannot hold finds nothing (no aliasing:
    // 65537 truncates to 1 in 16 bits)
    {
        auto m = makeTwoSpheres(1.0);
        constexpr int32_t kAliasOfOne = 65537;
        EXPECT_EQ(m->extractSignedDistance(kAliasOfOne)->activeVoxelCount(), Index64(0));
        EXPECT_GT(m->extractSignedDistance(1)->activeVoxelCount(), Index64(0));
    }
}

TEST_F(TestLevelSetMultiObject, Int16GridIO)
{
    // OpenVDB does not register int16 grids; registerMultiLevelSetGridTypes()
    // does (idempotent), so the pair round-trips through a .vdb file.
    tools::registerMultiLevelSetGridTypes();
    tools::registerMultiLevelSetGridTypes();
    EXPECT_TRUE(tools::Int16LabelGrid::isRegistered());
    auto m = makeTwoSpheres(0.25);
    m->phi()->setName("phi");
    m->labels()->setName("labels");
    lsmo_test::ScopedDebugDir dir;
    const std::string path = (dir.path() / "int16_pair.vdb").string();
    {
        io::File file(path);
        file.write(GridPtrVec{m->phi(), m->labels()});
        file.close();
    }
    io::File file(path);
    file.open();
    FloatGrid::Ptr phi = gridPtrCast<FloatGrid>(file.readGrid("phi"));
    tools::Int16LabelGrid::Ptr labels =
        gridPtrCast<tools::Int16LabelGrid>(file.readGrid("labels"));
    file.close();
    ASSERT_TRUE(phi);
    ASSERT_TRUE(labels);
    MLS back(phi, labels, m->halfWidth());
    std::string msg;
    EXPECT_TRUE(back.validate(&msg)) << msg;
    EXPECT_EQ(treeDifferences(m->phi()->tree(), phi->tree()), Index64(0));
    EXPECT_TRUE(labels->tree().hasSameTopology(m->labels()->tree()));
    tree::ValueAccessor<const tools::Int16LabelTree> acc(m->labels()->tree());
    for (auto it = labels->tree().cbeginValueAll(); it; ++it) {
        ASSERT_EQ(*it, acc.getValue(it.getCoord())) << it.getCoord();
    }
    EXPECT_EQ(back.labelSet(), m->labelSet());
}


TEST_F(TestLevelSetMultiObject, ExtractInterrupt)
{
    struct Interrupter: public util::NullInterrupter
    {
        int calls = 0, starts = 0, ends = 0, fireAt = 3;
        void start(const char* = nullptr) override { ++starts; }
        void end() override { ++ends; }
        bool wasInterrupted(int = -1) override { return ++calls >= fireAt; }
    };
    auto m = makeTwoSpheres(1.0);
    FloatGrid::Ptr full = m->extractSignedDistance(1);

    // never fires: identical result
    Interrupter quiet;
    quiet.fireAt = std::numeric_limits<int>::max();
    FloatGrid::Ptr same = m->extractSignedDistance(1, -1, 1, &quiet);
    EXPECT_GT(quiet.calls, 0);
    EXPECT_EQ(quiet.starts, 1);
    EXPECT_EQ(quiet.ends, 1);
    EXPECT_EQ(treeDifferences(full->tree(), same->tree()), Index64(0));

    // fires after two Jacobi iterations: still a valid level set grid
    Interrupter early;
    FloatGrid::Ptr part = m->extractSignedDistance(1, -1, 1, &early);
    EXPECT_EQ(early.calls, early.fireAt);
    EXPECT_EQ(early.ends, 1);
    EXPECT_EQ(part->getGridClass(), GRID_LEVEL_SET);
    EXPECT_FLOAT_EQ(part->background(), full->background());
    const float bg = part->background();
    for (auto it = part->tree().cbeginValueAll(); it; ++it) {
        ASSERT_TRUE(std::isfinite(*it));
        if (it.isValueOn()) ASSERT_LT(std::abs(*it), bg);
        else ASSERT_EQ(std::abs(*it), bg);
    }
    EXPECT_GT(part->activeVoxelCount(), Index64(0));
    EXPECT_LT(part->activeVoxelCount(), full->activeVoxelCount());
}

TEST_F(TestLevelSetMultiObject, HolderSignedDistance)
{
    auto m = makePair(0.5);
    const Coord c(3, 0, 0);
    const float d = m->phi()->tree().getValue(c);
    EXPECT_GT(d, 0.0f);
    EXPECT_FLOAT_EQ(m->signedDistance(c, 1), -d);
    EXPECT_FLOAT_EQ(m->signedDistance(c, 2), d);
    EXPECT_DOUBLE_EQ(m->voxelSize(), 0.5);
    EXPECT_EQ(m->halfWidth(), 3);
}

TEST_F(TestLevelSetMultiObject, HolderLabelSet)
{
    // labels beyond 16 bits: the Int32 variant
    auto m = makePair<tools::MultiLevelSet<FloatTree, Int32Tree>>();
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2}));
    m->labels()->tree().setValueOn(Coord(0, 0, 0), -7);
    m->labels()->tree().setValueOn(Coord(1, 0, 0), 1 << 30);
    m->labels()->tree().setValueOn(Coord(2, 0, 0), 0);
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{-7, 1, 2, 1 << 30}));
}

TEST_F(TestLevelSetMultiObject, HolderDilateErode)
{
    auto m = makePair();
    const Index64 n0 = m->phi()->activeVoxelCount();
    m->dilate(2);
    EXPECT_GT(m->phi()->activeVoxelCount(), n0);
    EXPECT_EQ(m->phi()->activeVoxelCount(), m->labels()->activeVoxelCount());
    EXPECT_TRUE(m->validate());
    // new voxels outside x<0 copy label 1 from their neighbors
    EXPECT_EQ(m->labels()->tree().getValue(Coord(-1, 5, 5)), 1);
    EXPECT_EQ(m->labels()->tree().getValue(Coord(16, 5, 5)), 2);
    EXPECT_FLOAT_EQ(m->phi()->tree().getValue(Coord(-1, 5, 5)), 3.0f);

    m->erode(2);
    EXPECT_EQ(m->phi()->tree().leafCount(), m->labels()->tree().leafCount());
    EXPECT_TRUE(m->phi()->tree().hasSameTopology(m->labels()->tree()));
    EXPECT_EQ(m->phi()->activeVoxelCount(), n0);
    EXPECT_EQ(m->phi()->activeVoxelCount(), m->labels()->activeVoxelCount());
    EXPECT_TRUE(m->validate());
}


namespace {

// Planar interface at x = 7.3 (label 1 for x <= 7, label 2 for x >= 8),
// active band x in [4, 11] (four voxels on each side), y, z in [0, 15].
// Layers x in [7 - halo, 8 + halo] hold the exact distances (halo 0: the two
// interface layers, enough for redistance(); halo 1: also the outer
// neighbors read by the subcell reseed); if @a exact is false the remaining
// band voxels hold garbage (alternating 0.5*dx and halfWidth*dx).
MLS::Ptr
makePlanar(double dx, bool exact, int halo = 1)
{
    const int halfWidth = 3;
    auto xform = math::Transform::createLinearTransform(dx);
    FloatGrid::Ptr phi = FloatGrid::create(float(halfWidth * dx));
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    phi->setTransform(xform);
    lab->setTransform(xform->copy());
    auto pa = phi->getAccessor();
    auto la = lab->getAccessor();
    for (int i = 4; i <= 11; ++i)
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 16; ++k) {
                const Coord c(i, j, k);
                float d = float(std::abs(i - 7.3) * dx);
                if (!exact && (i < 7 - halo || i > 8 + halo)) {
                    d = ((i + j + k) % 2) ? float(0.5 * dx) : float(halfWidth * dx);
                }
                pa.setValue(c, d);
                la.setValue(c, i <= 7 ? 1 : 2);
            }
    // One inactive shell around the band holding the deep-interior labels
    // (inactive labels keep their region's label), with phi leaves kept in lockstep at the background.
    for (int i = 3; i <= 12; ++i)
        for (int j = -1; j <= 16; ++j)
            for (int k = -1; k <= 16; ++k) {
                const Coord c(i, j, k);
                if (pa.isValueOn(c)) continue;
                pa.touchLeaf(c); // inactive phi already holds the background
                la.setValueOff(c, i <= 7 ? 1 : 2);
            }
    return MLS::Ptr(new MLS(phi, lab, halfWidth));
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, RedistancePlanar)
{
    const double dx = 0.5;
    for (int reseed = 0; reseed < 2; ++reseed) {
        SCOPED_TRACE(reseed ? "reinitialize" : "redistance");
        auto m = makePlanar(dx, /*exact=*/false, /*halo=*/reseed);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        const Index64 nActive = m->phi()->activeVoxelCount();
        const Index64 nLeaves = m->phi()->tree().leafCount();

        tools::MultiLevelSetRedistance<MLS> redist(*m);
        const int iterations = reseed ? redist.reinitialize() : redist.redistance();
        EXPECT_GE(iterations, 1);
        EXPECT_LE(iterations, int(std::ceil(3.0 * std::sqrt(3.0))) + 2);

        EXPECT_EQ(m->phi()->activeVoxelCount(), nActive);
        EXPECT_EQ(m->phi()->tree().leafCount(), nLeaves);
        EXPECT_TRUE(m->validate(&msg)) << msg;
        for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
            const Coord c = it.getCoord();
            EXPECT_NEAR(*it, std::abs(c[0] - 7.3) * dx, 1e-3 * dx) << c;
            EXPECT_EQ(m->labels()->tree().getValue(c), c[0] <= 7 ? 1 : 2);
        }
        // inactive phi stays at the background
        for (auto leaf = m->phi()->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto v = leaf->cbeginValueOff(); v; ++v) EXPECT_FLOAT_EQ(*v, float(3.0 * dx));
        }
    }
}


namespace {

// Interface layer: active voxels with an active face neighbor of opposite
// sign. Errors are |sdf - analytic| in units of dx; analytic takes world
// coordinates.
double
maxInterfaceError(const FloatGrid& sdf, std::function<double(Vec3d)> analytic)
{
    const double dx = sdf.voxelSize()[0];
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    tree::ValueAccessor<const FloatTree> acc(sdf.tree());
    double maxErr = 0.0;
    Index64 n = 0;
    for (auto it = sdf.tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool iface = false;
        for (int f = 0; f < 6 && !iface; ++f) {
            float v;
            if (acc.probeValue(p + offs[f], v)) iface = (v < 0.0f) != (*it < 0.0f);
        }
        if (!iface) continue;
        ++n;
        maxErr = std::max(maxErr, std::abs(*it - analytic(sdf.indexToWorld(p))) / dx);
    }
    EXPECT_GT(n, Index64(0));
    return maxErr;
}

// RMS of |sdf - analytic| / dx over all active voxels.
double
rmsBandError(const FloatGrid& sdf, std::function<double(Vec3d)> analytic)
{
    const double dx = sdf.voxelSize()[0];
    double sum = 0.0;
    Index64 n = 0;
    for (auto it = sdf.tree().cbeginValueOn(); it; ++it, ++n) {
        const double e = (*it - analytic(sdf.indexToWorld(it.getCoord()))) / dx;
        sum += e * e;
    }
    EXPECT_GT(n, Index64(0));
    return n ? std::sqrt(sum / double(n)) : 0.0;
}

// Raw signedDistance(ijk, label) over the active voxels of @a m whose world
// position satisfies @a keep.
FloatGrid::Ptr
rawSignedGrid(const MLS& m, int32_t label, std::function<bool(Vec3d)> keep)
{
    FloatGrid::Ptr g = FloatGrid::create(float(m.halfWidth() * m.voxelSize()));
    g->setTransform(m.phi()->transform().copy());
    auto acc = g->getAccessor();
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        if (keep(m.phi()->indexToWorld(p))) acc.setValue(p, m.signedDistance(p, label));
    }
    return g;
}

// Every seed (an active voxel with a face neighbor, active or
// not, of a different label, as the redistancer sees it) lies within one
// voxel of the true interface; distToInterface takes world coordinates.
void
expectNoSpuriousSeeds(const MLS& m, std::function<double(Vec3d)> distToInterface)
{
    const double dx = m.voxelSize();
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    tree::ValueAccessor<const LabelTree> lab(m.labels()->tree());
    Index64 nSeeds = 0, nBad = 0;
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        bool seed = false;
        for (int f = 0; f < 6; ++f) seed |= lab.getValue(p + offs[f]) != lab.getValue(p);
        if (!seed) continue;
        ++nSeeds;
        const double d = distToInterface(m.phi()->indexToWorld(p));
        if (d > dx * (1.0 + 1e-6)) {
            if (++nBad <= 5) ADD_FAILURE() << "spurious seed at " << p << ", interface " << d / dx << " dx away";
        }
    }
    EXPECT_GT(nSeeds, Index64(0));
    EXPECT_EQ(nBad, Index64(0));
}

struct TwoSphereErrors { double maxIface[2], rms[2]; };


// Raw signedDistance errors near each sphere of makeTwoSpheres, in dx.
TwoSphereErrors
twoSphereRoundTrip(double dx)
{
    const double r = 10.0 * dx, cx = 15.0 * dx;
    auto m = makeTwoSpheres(dx);
    EXPECT_TRUE(m);
    if (!m) return TwoSphereErrors{};
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(m->halfWidth(), 3);
    EXPECT_DOUBLE_EQ(m->voxelSize(), dx);
    const auto sphere = [r](double c) {
        return [r, c](Vec3d w) { return (w - Vec3d(c, 0, 0)).length() - r; };
    };
    expectNoSpuriousSeeds(*m, [&](Vec3d w) {
        return std::min(std::abs(sphere(-cx)(w)), std::abs(sphere(cx)(w)));
    });
    TwoSphereErrors e;
    for (int s = 0; s < 2; ++s) {
        const double c = s ? cx : -cx;
        auto g = rawSignedGrid(*m, s + 1, [s](Vec3d w) { return s ? w[0] > 0.0 : w[0] < 0.0; });
        e.maxIface[s] = maxInterfaceError(*g, sphere(c));
        e.rms[s] = rmsBandError(*g, sphere(c));
    }
    return e;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, RoundTripTwoSpheres)
{
    const TwoSphereErrors e = twoSphereRoundTrip(1.0);
    for (int s = 0; s < 2; ++s) {
        std::cout << "RoundTripTwoSpheres label " << s + 1 << ": max interface error "
            << e.maxIface[s] << " dx, RMS band error " << e.rms[s] << " dx" << std::endl;
        EXPECT_LT(e.maxIface[s], 0.05);
        EXPECT_LT(e.rms[s], 0.1);
    }
}


TEST_F(TestLevelSetMultiObject, FromLabelGrid)
{
    // Label 1 on [5, 24]^3 inside label 2 on [0, 29]^3; label 0 elsewhere.
    const double dx = 0.5;
    Int32Grid::Ptr in = Int32Grid::create(0);
    in->setTransform(math::Transform::createLinearTransform(dx));
    in->tree().fill(CoordBBox(Coord(0), Coord(29)), 2, /*active=*/true);
    in->tree().fill(CoordBBox(Coord(5), Coord(24)), 1, /*active=*/true);
    in->tree().voxelizeActiveTiles();

    auto m = MLS::fromLabelGrid(*in);
    ASSERT_TRUE(m);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(m->halfWidth(), 3);
    EXPECT_DOUBLE_EQ(m->voxelSize(), dx);

    // Index-space distance from a voxel center to the face planes of a box
    // whose voxels span [lo, hi] (faces at lo - 0.5 and hi + 0.5).
    const auto boxDist = [](const Vec3d& p, double lo, double hi) {
        const double l = lo - 0.5, h = hi + 0.5;
        double inner = std::numeric_limits<double>::max();
        bool inside = true;
        Vec3d q(0.0);
        for (int a = 0; a < 3; ++a) {
            q[a] = std::max(0.0, std::max(l - p[a], p[a] - h));
            if (p[a] < l || p[a] > h) inside = false;
            inner = std::min(inner, std::min(p[a] - l, h - p[a]));
        }
        return inside ? inner : q.length();
    };
    expectNoSpuriousSeeds(*m, [&](Vec3d w) {
        const Vec3d p = w / dx;
        return dx * std::min(boxDist(p, 5, 24), boxDist(p, 0, 29));
    });

    // Every interface voxel keeps its frozen face-midpoint seed, 0.5*dx
    // (fromLabelGrid redistances without reseeding).
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    const auto& lab = m->labels()->tree();
    Index64 nFace = 0;
    for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        EXPECT_EQ(lab.getValue(p), in->tree().getValue(p)) << p;
        bool iface = false;
        for (int f = 0; f < 6; ++f) iface |= lab.getValue(p + offs[f]) != lab.getValue(p);
        if (!iface) continue;
        ++nFace;
        EXPECT_NEAR(*it, 0.5 * dx, 1e-3 * dx) << p;
    }
    EXPECT_GT(nFace, Index64(6 * 18 * 18));
    // band: every active voxel within halfWidth*dx (plus solver tolerance)
    for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) EXPECT_LE(*it, float(3.0 * dx));
}

TEST_F(TestLevelSetMultiObject, FromSignedGridsThrows)
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    FloatGrid::Ptr fog = tools::createLevelSetSphere<FloatGrid>(5.0f, Vec3f(0), 1.0f);
    fog->setGridClass(GRID_FOG_VOLUME);
    in[1] = fog;
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);

    FloatGrid::Ptr a = tools::createLevelSetSphere<FloatGrid>(5.0f, Vec3f(-8, 0, 0), 1.0f);
    FloatGrid::Ptr b = tools::createLevelSetSphere<FloatGrid>(5.0f, Vec3f(8, 0, 0), 1.0f);
    FloatGrid::Ptr c = tools::createLevelSetSphere<FloatGrid>(2.5f, Vec3f(4, 0, 0), 0.5f);

    in.clear(); // empty map
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in[1] = a; in[2] = FloatGrid::ConstPtr(); // null grid
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in.clear(); in[0] = a; in[2] = b; // key 0 is the background
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in.clear(); in[1] = a; in[2] = c; // mismatched transforms
    EXPECT_THROW(MLS::fromSignedGrids(in), ValueError);
    in.clear(); in[1] = a; in[2] = b;
    EXPECT_THROW(MLS::fromSignedGrids(in, 1), ValueError); // halfWidth < 2
    EXPECT_NO_THROW(MLS::fromSignedGrids(in, 2));
}

namespace {

void
expectLevelSetGrid(const FloatGrid& g, const MLS& m, int halfWidth)
{
    EXPECT_EQ(g.getGridClass(), GRID_LEVEL_SET);
    EXPECT_FLOAT_EQ(g.background(), float(halfWidth * m.voxelSize()));
    EXPECT_EQ(g.transform(), m.phi()->transform());
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, TripleJunction)
{
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    const std::vector<TripleJunction> set = junctionSet();
    double worstFar = 0.0, worstNear = 0.0, worstRms = 0.0;
    for (size_t n = 0; n < set.size(); ++n) {
        const TripleJunction& tj = set[n];
        const double dx = tj.dx;
        auto m = tj.build();
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3}));

        for (int l = 1; l <= 3; ++l) {
            SCOPED_TRACE("junction " + std::to_string(n) + " label " + std::to_string(l));
            const auto analytic = [&tj, l](Vec3d w) { return tj.signedDist(w, l); };

            FloatGrid::Ptr g = m->extractSignedDistance(l);
            ASSERT_TRUE(g);
            expectLevelSetGrid(*g, *m, 3);
            const double rms = rmsBandError(*g, analytic);

            // Interface-layer error binned by distance to the junction line.
            const double edges[4] = {1.0, 2.0, 4.0, 1e9};
            double binMax[4] = {0, 0, 0, 0};
            Index64 binN[4] = {0, 0, 0, 0};
            tree::ValueAccessor<const FloatTree> acc(g->tree());
            for (auto it = g->tree().cbeginValueOn(); it; ++it) {
                const Coord p = it.getCoord();
                bool iface = false;
                for (int f = 0; f < 6 && !iface; ++f) {
                    float v;
                    if (acc.probeValue(p + offs[f], v)) iface = (v < 0.0f) != (*it < 0.0f);
                }
                if (!iface) continue;
                const Vec3d w = g->indexToWorld(p);
                int b = 0;
                while (tj.lineDist(w) >= edges[b]) ++b;
                ++binN[b];
                binMax[b] = std::max(binMax[b], std::abs(*it - analytic(w)) / dx);
            }
            // Gates (first order near the line: psi_l is not smooth there):
            // < 0.05*dx at least 2 voxels from the line, < 0.5*dx within 2
            // voxels, RMS over the whole band < 0.1*dx.
            const double farMax = std::max(binMax[2], binMax[3]);
            const double nearMax = std::max(binMax[0], binMax[1]);
            std::cout << "TripleJunction " << n << " label " << l << ": interface max >=2 vox "
                << farMax << " dx, <2 vox " << nearMax << " dx, RMS " << rms
                << " dx; by line distance [<1: " << binMax[0] << " (" << binN[0] << ")"
                << ", 1-2: " << binMax[1] << " (" << binN[1] << ")"
                << ", 2-4: " << binMax[2] << " (" << binN[2] << ")"
                << ", >4: " << binMax[3] << " (" << binN[3] << ")]" << std::endl;
            EXPECT_LT(farMax, 0.05);
            EXPECT_LT(nearMax, 0.5);
            EXPECT_LT(rms, 0.1);
            worstFar = std::max(worstFar, farMax);
            worstNear = std::max(worstNear, nearMax);
            worstRms = std::max(worstRms, rms);

            // Every voxel within halfWidth of the label boundary is in the
            // band (checked 3+ voxels inside the fixture box, beyond which
            // the boundary's nearest point may lie outside the truncated band).
            for (auto it = g->tree().cbeginValueOn(); it; ++it) EXPECT_LT(std::abs(*it), 3.0 * dx);
            Index64 missing = 0;
            for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
                const Coord c = it.getCoord();
                if (tj.boxDepth(c) < 3.0) continue;
                const Vec3d w = m->phi()->indexToWorld(c);
                if (std::abs(analytic(w)) < 2.5 * dx && !g->tree().isValueOn(c)) ++missing;
            }
            EXPECT_EQ(missing, Index64(0));

            // Regression guard: raw +-phi underestimates psi_l
            // outside l near the junction.
            auto raw = rawSignedGrid(*m, l, [&tj, l](Vec3d w) { return tj.sector(w) != l; });
            double maxUnder = 0.0;
            for (auto it = raw->tree().cbeginValueOn(); it; ++it) {
                const Vec3d w = raw->indexToWorld(it.getCoord());
                maxUnder = std::max(maxUnder, (analytic(w) - *it) / dx);
            }
            EXPECT_GT(maxUnder, 0.2);
        }
    }
    std::cout << "TripleJunction over " << set.size() << " junctions: worst interface max >=2 vox "
        << worstFar << " dx, <2 vox " << worstNear << " dx, worst RMS " << worstRms << " dx" << std::endl;
}

namespace {


/// Three labels around a z-parallel line through index point (c, 0) with
/// arbitrary angular spans (degrees, summing to 360): label l owns the
/// polar angles [b_{l-1}, b_l) about the line, b_0 = theta0. Same analytic
/// distance and pair construction as lsmo_test::TripleJunction.
struct SectorJunction
{
    double dx = 0.5;
    double b[3];      // start angle of each sector, radians
    double span01[2]; // cumulative end angles of sectors 1 and 2 past b[0]
    Vec2d c;
    int H = 12;

    SectorJunction(double theta0Deg, const double spanDeg[3], const Vec2d& center) : c(center)
    {
        const double rad = math::pi<double>() / 180.0;
        b[0] = theta0Deg * rad;
        b[1] = b[0] + spanDeg[0] * rad;
        b[2] = b[1] + spanDeg[1] * rad;
        span01[0] = spanDeg[0] * rad;
        span01[1] = (spanDeg[0] + spanDeg[1]) * rad;
    }
    Vec2d local(const Vec3d& w) const { return Vec2d(w[0] / dx - c[0], w[1] / dx - c[1]); }
    Vec2d ray(int k) const { return Vec2d(std::cos(b[k % 3]), std::sin(b[k % 3])); }
    int sector(const Vec3d& w) const
    {
        const Vec2d p = local(w);
        const double twoPi = 2.0 * math::pi<double>();
        double t = std::atan2(p[1], p[0]) - b[0];
        while (t < 0.0) t += twoPi;
        while (t >= twoPi) t -= twoPi;
        return t < span01[0] ? 1 : (t < span01[1] ? 2 : 3);
    }
    double signedDist(const Vec3d& w, int l) const
    {
        const Vec2d p = local(w);
        const double d = std::min(TripleJunction::rayDist(p, ray(l - 1)),
            TripleJunction::rayDist(p, ray(l)));
        return dx * (sector(w) == l ? -d : d);
    }
    double lineDist(const Vec3d& w) const { return local(w).length(); }
    double boxDepth(const Coord& p) const
    {
        return H - std::max({std::abs(p[0] + 0.5), std::abs(p[1] + 0.5), std::abs(p[2] + 0.5)});
    }
    MLS::Ptr build() const
    {
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


} // anonymous namespace


TEST_F(TestLevelSetMultiObject, ExtractAbsentLabel)
{
    auto m = makeTwoSpheres(0.5);
    ASSERT_TRUE(m);
    FloatGrid::Ptr g = m->extractSignedDistance(99);
    ASSERT_TRUE(g);
    expectLevelSetGrid(*g, *m, 3);
    EXPECT_EQ(g->activeVoxelCount(), Index64(0));
    EXPECT_GT(g->background(), 0.0f);
    EXPECT_GT(g->tree().getValue(Coord(-30, 0, 0)), 0.0f);
}

// ---------------------------------------------------------------------------
// LabelAwareStencil and MultiObjectForces
// ---------------------------------------------------------------------------

namespace {

using Stencil = tools::LabelAwareStencil<MLS>;
using Forces = tools::MultiObjectForces<MLS>;

// First active voxel of label 1 whose +x neighbor is label 2 (interface).
Coord
sphereInterfaceVoxel(const MLS& m)
{
    // Sphere 1 is centered at (-15,0,0) dx with radius 10 dx: its surface
    // point towards -y is at (-15, -10, 0).
    const Coord c(-15, -10, 0);
    for (int dj = -1; dj <= 1; ++dj) {
        const Coord p = c.offsetBy(0, dj, 0);
        if (m.labels()->tree().getValue(p) == 1
            && m.labels()->tree().getValue(p.offsetBy(0, -1, 0)) != 1) return p;
    }
    return c;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, StencilSphere)
{
    const double dx = 1.0, r = 10.0;
    auto m = makeTwoSpheres(dx);
    const Coord ijk = sphereInterfaceVoxel(*m);
    ASSERT_EQ(m->labels()->tree().getValue(ijk), 1);
    Stencil s(*m);
    s.moveTo(ijk, 1);
    const Vec3f g = s.gradient();
    EXPECT_NEAR(g.length(), 1.0f, 0.05f);
    // Outward normal at the -y pole of sphere 1 is -y.
    EXPECT_LT(g[1], -0.9f);
    // Single-voxel curvature carries second-difference noise from the
    // redistanced phi, so check the mean over the interface layer of the
    // sphere (voxels of label 1 with a face neighbor of another label).
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    double sum = 0.0;
    int n = 0;
    for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
        if (*it != 1) continue;
        const Coord p = it.getCoord();
        bool iface = false;
        for (int f = 0; f < 6 && !iface; ++f) iface = m->labels()->tree().getValue(p + offs[f]) != 1;
        if (!iface || p[0] > -10) continue; // sphere-1 side away from the junction with sphere 2
        s.moveTo(p, 1);
        sum += s.meanCurvatureNormGrad();
        ++n;
    }
    ASSERT_GT(n, 100);
    EXPECT_NEAR(sum / n, 1.0 / r, 0.1 / r);
}

TEST_F(TestLevelSetMultiObject, StencilLabelSign)
{
    auto m = makeTwoSpheres(1.0);
    const Coord ijk(15, 0, 0); // center of sphere 2
    ASSERT_EQ(m->labels()->tree().getValue(ijk), 2);
    const float phi = m->phi()->tree().getValue(ijk);
    ASSERT_GT(phi, 0.0f);
    Stencil s(*m);
    s.moveTo(ijk, 1);
    EXPECT_EQ((s.getValue<0, 0, 0>()), phi);
    s.moveTo(ijk, 2);
    EXPECT_EQ((s.getValue<0, 0, 0>()), -phi);
}

TEST_F(TestLevelSetMultiObject, ForcesPressureRescale)
{
    FloatGrid::Ptr p = FloatGrid::create(0.0f);
    auto a = p->getAccessor();
    a.setValue(Coord(0, 0, 0), -2.0f);
    a.setValue(Coord(1, 0, 0), 0.0f);
    a.setValue(Coord(2, 0, 0), 4.0f);
    Forces f;
    f.setPressure(p, 1.0f, 0.0f);
    ASSERT_TRUE(f.pressure());
    EXPECT_FLOAT_EQ(f.pressure()->tree().getValue(Coord(0, 0, 0)), -1.0f);
    EXPECT_FLOAT_EQ(f.pressure()->tree().getValue(Coord(1, 0, 0)), 0.0f);
    EXPECT_FLOAT_EQ(f.pressure()->tree().getValue(Coord(2, 0, 0)), 1.0f);
    // Input is untouched.
    EXPECT_FLOAT_EQ(p->tree().getValue(Coord(2, 0, 0)), 4.0f);
    // NaN target: no rescale.
    Forces g;
    g.setPressure(p, 1.0f);
    EXPECT_FLOAT_EQ(g.pressure()->tree().getValue(Coord(2, 0, 0)), 4.0f);
}


TEST_F(TestLevelSetMultiObject, ForcesFixedAndInactive)
{
    Forces f;
    // Label 0 is passive in the evolver, not fixed in the functor.
    EXPECT_FALSE(f.isFixed(0));
    EXPECT_FALSE(f.isFixed(3));
    EXPECT_TRUE(f.isActive(3));
    f.setFixedLabels({4});
    f.setInactiveLabels({5});
    EXPECT_TRUE(f.isFixed(4));
    EXPECT_FALSE(f.isFixed(3));
    EXPECT_FALSE(f.isActive(5));
    EXPECT_TRUE(f.isActive(4));

    auto m = makePair();
    Stencil s(*m);
    s.moveTo(Coord(7, 8, 8), 1);
    EXPECT_FLOAT_EQ(f(Coord(7, 8, 8), 5, s), 0.999f);
    EXPECT_FLOAT_EQ(f(Coord(7, 8, 8), 4, s), 0.0f);

    EXPECT_TRUE(std::isinf(f.stableDt(1.0f)));
    f.setCurvatureWeight(0.5f);
    EXPECT_FLOAT_EQ(f.stableDt(2.0f), 4.0f / 3.0f);
}

TEST_F(TestLevelSetMultiObject, ForcesMissingGrid)
{
    Forces f;
    EXPECT_THROW(f.setPressure(nullptr, 1.0f), ValueError);
    EXPECT_THROW(f.setAdvection(nullptr, 1.0f), ValueError);
}


////////////////////////////////////////
// Evolver

namespace {

// dpsi/dt = s for every non-fixed label (unit normal speed where |grad psi| = 1).
struct ConstantSpeed
{
    float s = 0.0f;
    std::set<int32_t> fixed;
    float operator()(const Coord&, int32_t, const Stencil&) const { return s; }
    bool isFixed(int32_t l) const { return l != 0 && fixed.count(l) != 0; }
    bool isActive(int32_t) const { return true; }
    float stableDt(float) const { return std::numeric_limits<float>::infinity(); }
};

// Per-label constant dpsi/dt drawn once from a seeded generator in [-1, 1].
struct RandomSpeed
{
    std::map<int32_t, float> speeds;
    RandomSpeed(const std::vector<int32_t>& labels, unsigned seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int32_t l : labels) speeds[l] = dist(gen);
    }
    float operator()(const Coord&, int32_t l, const Stencil&) const
    {
        auto it = speeds.find(l);
        return it == speeds.end() ? 0.0f : it->second;
    }
    bool isFixed(int32_t) const { return false; }
    bool isActive(int32_t) const { return true; }
    float stableDt(float) const { return std::numeric_limits<float>::infinity(); }
};

// wasInterrupted() returns true from its third call on.
struct CountingInterrupter: public util::NullInterrupter
{
    int calls = 0;
    bool wasInterrupted(int = -1) override { return ++calls >= 3; }
};

// Single sphere of label 1 via fromSignedGrids.
MLS::Ptr
makeSphere(double r, const Vec3d& c = Vec3d(0.0), double dx = 1.0)
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(c), float(dx));
    return MLS::fromSignedGrids(in);
}

// Zero crossings of a signed grid along active +x/+y/+z voxel edges (world).
std::vector<Vec3d>
zeroCrossings(const FloatGrid& sdf)
{
    std::vector<Vec3d> pts;
    tree::ValueAccessor<const FloatTree> acc(sdf.tree());
    for (auto it = sdf.tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        const float a = *it;
        for (int axis = 0; axis < 3; ++axis) {
            Coord q = p;
            q[axis] += 1;
            float b;
            if (!acc.probeValue(q, b) || ((a < 0.0f) == (b < 0.0f))) continue;
            const double t = double(a) / (double(a) - double(b));
            pts.push_back(sdf.indexToWorld(p.asVec3d() + (q.asVec3d() - p.asVec3d()) * t));
        }
    }
    return pts;
}

double
meanRadius(const std::vector<Vec3d>& pts, const Vec3d& c)
{
    double sum = 0.0;
    for (const Vec3d& p : pts) sum += (p - c).length();
    return pts.empty() ? 0.0 : sum / double(pts.size());
}

Vec3d
centroid(const std::vector<Vec3d>& pts)
{
    Vec3d sum(0.0);
    for (const Vec3d& p : pts) sum += p;
    return pts.empty() ? sum : sum / double(pts.size());
}


// The interiors {psi_l < 0} of the per-label
// extractSignedDistance grids are pairwise disjoint (up to 1e-3*dx;
// @a overlaps counts violations) and, with the background region, cover the
// band consistently with the labels (@a badCover counts violations).
void
partitionErrors(const MLS& m, Index64& overlaps, Index64& badCover)
{
    const double dx = m.voxelSize();
    const std::vector<int32_t> ls = m.labelSet();
    std::map<int32_t, FloatGrid::Ptr> sdf;
    for (int32_t l : ls) sdf[l] = m.extractSignedDistance(l);
    const float eps = float(-1e-3 * dx);
    overlaps = badCover = 0;
    tree::ValueAccessor<const LabelTree> lab(m.labels()->tree());
    for (auto it = m.phi()->tree().cbeginValueOn(); it; ++it) {
        const Coord p = it.getCoord();
        int inside = 0, deep = 0;
        int32_t which = 0;
        for (int32_t l : ls) {
            const float v = sdf[l]->tree().getValue(p);
            if (v < 0.0f) { ++inside; which = l; }
            if (v < eps) ++deep;
        }
        if (deep > 1) ++overlaps;
        if (inside > 1 || (inside == 1 && lab.getValue(p) != which)
            || (inside == 0 && lab.getValue(p) != 0)) ++badCover;
    }
}

} // anonymous namespace


namespace {

// Max interface motion argument of the scenario helpers below meaning "keep
// the evolver's default".
constexpr double kDefaultMotion = 0.0;

// Failed assertions recorded so far by the current test.
int
failureCount()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->result()
        ->total_part_count();
}

template<typename EvolverT>
void
applyMotion(EvolverT& e, double motion)
{
    if (motion > kDefaultMotion) e.setMaxInterfaceMotion(motion);
}

// Sphere of radius 8 grown at unit speed to t = 6 (gate: mean radius
// 14 +- 0.5*dx). @return the number of steps.
size_t
runPressureGrowth(double motion)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    ConstantSpeed speed;
    speed.s = -1.0f; // dpsi/dt = -1: the object grows at unit speed
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(*m, speed);
    applyMotion(lsmo, motion);
    const auto start = std::chrono::steady_clock::now();
    const size_t steps = lsmo.advect(0.0, 6.0);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    EXPECT_GE(steps, size_t(6));
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
    std::cout << "PressureGrowth: motion " << lsmo.maxInterfaceMotion() << " steps " << steps
              << " (" << 1e3 * secs / double(std::max(steps, size_t(1))) << " ms/step)"
              << " mean radius " << r << " (expected 14)" << std::endl;
    EXPECT_NEAR(r, 8.0 + 6.0, 0.5 * dx);
    return steps;
}

// Sphere of radius 10 under unit curvature flow to t = 30 (gate: r^2 =
// 40 +- 2*sqrt(40)*0.5*dx).
void
runCurvatureFlow(double motion)
{
    auto m = makeSphere(10.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    applyMotion(lsmo, motion);
    const size_t steps = lsmo.advect(0.0, 30.0);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    const double r = meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0));
    // r^2 = r0^2 - 2t = 100 - 60
    std::cout << "CurvatureFlow: motion " << lsmo.maxInterfaceMotion() << " steps " << steps
              << " r^2 " << r * r << " (expected 40)" << std::endl;
    EXPECT_NEAR(r * r, 40.0, 2.0 * std::sqrt(40.0) * 0.5 * dx);
}

// Sphere of radius 8 advected by (1, 0, 0) to t = 7.5 (gate: centroid shift
// (7.5, 0, 0) +- 0.5*dx per axis).
void
runUniformAdvection(double motion)
{
    auto m = makeSphere(8.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    applyMotion(lsmo, motion);
    const Vec3d c0 = centroid(zeroCrossings(*m->extractSignedDistance(1)));
    const size_t steps = lsmo.advect(0.0, 7.5);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
    const auto pts = zeroCrossings(*m->extractSignedDistance(1));
    const Vec3d c1 = centroid(pts);
    std::cout << "UniformAdvection: motion " << lsmo.maxInterfaceMotion() << " steps " << steps
              << " shift " << (c1 - c0) << " mean radius " << meanRadius(pts, c1) << std::endl;
    EXPECT_NEAR(c1[0] - c0[0], 7.5, 0.5 * dx);
    EXPECT_NEAR(c1[1] - c0[1], 0.0, 0.5 * dx);
    EXPECT_NEAR(c1[2] - c0[2], 0.0, 0.5 * dx);
}

} // anonymous namespace


TEST_F(TestLevelSetMultiObject, PerLabelAdvection)
{
    // Label 1 uses the shared field (1,0,0), label 2 its own (0,1,0);
    // clearing the override returns label 2 to the shared field.
    auto m = makeTwoSpheres(1.0);
    const double dx = m->voxelSize();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    forces.setAdvection(2, Vec3SGrid::create(Vec3s(0.0f, 1.0f, 0.0f)), 1.0f);
    EXPECT_THROW(forces.setAdvection(3, Vec3SGrid::ConstPtr(), 1.0f), ValueError);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    auto center = [&](int32_t l) { return centroid(zeroCrossings(*m->extractSignedDistance(l))); };
    const Vec3d a0 = center(1), b0 = center(2);
    lsmo.advect(0.0, 5.0);
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const Vec3d a1 = center(1), b1 = center(2);
    std::cout << "PerLabelAdvection: label 1 shift " << (a1 - a0) << ", label 2 shift "
              << (b1 - b0) << std::endl;
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(a1[k] - a0[k], k == 0 ? 5.0 : 0.0, 0.5 * dx) << "axis " << k;
        EXPECT_NEAR(b1[k] - b0[k], k == 1 ? 5.0 : 0.0, 0.5 * dx) << "axis " << k;
    }
    forces.clearLabelAdvection(2);
    lsmo.advect(5.0, 7.0);
    const Vec3d b2 = center(2);
    EXPECT_NEAR(b2[0] - b1[0], 2.0, 0.5 * dx);
    EXPECT_NEAR(b2[1] - b1[1], 0.0, 0.5 * dx);
}

TEST_F(TestLevelSetMultiObject, KinematicLabelBasics)
{
    // A kinematic sphere (label 2, radius 4) translating along +x at unit
    // speed through label 0 and through a non-protected block (label 1):
    // its region is exactly the moved ball at every chunk (it never loses
    // voxels), the block is pushed aside (none of its voxels inside the
    // ball, the block persists), validate and the partition hold.
    const double r = 4.0;
    const Vec3d c0(-10.0, 15.0, 15.0);
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (auto p = CoordBBox(Coord(6, 9, 9), Coord(24, 21, 21)).begin(); p; ++p) acc.setValue(*p, 1);
        for (auto p = CoordBBox(Coord::round(c0) - Coord(5), Coord::round(c0) + Coord(5)).begin(); p; ++p) {
            if (((*p).asVec3d() - c0).length() < r) acc.setValue(*p, 2);
        }
    }
    auto m = MLS::fromLabelGrid(*labels);
    Forces forces; // no speeds: the block moves only where the sphere pushes it
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    EXPECT_THROW(lsmo.setKinematicLabel(0, FloatGrid::create(), [](double) {
        return math::Mat4d::identity(); }), ValueError);
    EXPECT_THROW(lsmo.setKinematicLabel(2, FloatGrid::ConstPtr(), [](double) {
        return math::Mat4d::identity(); }), ValueError);
    EXPECT_THROW(lsmo.setKinematicLabel(2, FloatGrid::create(), nullptr), ValueError);
    auto motion = [](double t) {
        math::Mat4d mm = math::Mat4d::identity();
        mm.setTranslation(Vec3d(t, 0.0, 0.0));
        return mm;
    };
    lsmo.setKinematicLabel(2, tools::createLevelSetSphere<FloatGrid>(float(r), Vec3f(c0), 1.0f),
        motion);
    const CoordBBox domain(Coord(-20, 0, 0), Coord(40, 30, 30));
    auto count = [&](int32_t l) {
        Index64 n = 0;
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) n += acc.getValue(*p) == l;
        return n;
    };
    const Index64 block0 = count(1);
    Index64 minBlock = block0;
    for (double t = 0.0; t < 44.0 - 1e-9; t += 2.0) {
        lsmo.advect(t, t + 2.0);
        const Vec3d c = c0 + Vec3d(t + 2.0, 0.0, 0.0);
        Index64 missing = 0, extra = 0, blockInside = 0;
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) {
            const double d = ((*p).asVec3d() - c).length() - r;
            const int32_t l = acc.getValue(*p);
            if (d < -0.1 && l != 2) ++missing;
            if (d > 0.1 && l == 2) ++extra;
            if (d < 0.0 && l == 1) ++blockInside;
        }
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "t " << t + 2.0 << ": " << msg;
        Index64 overlaps = 0, badCover = 0;
        partitionErrors(*m, overlaps, badCover);
        EXPECT_EQ(missing, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(extra, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(blockInside, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(overlaps, Index64(0)) << "t " << t + 2.0;
        EXPECT_EQ(badCover, Index64(0)) << "t " << t + 2.0;
        minBlock = std::min(minBlock, count(1));
    }
    const Index64 block1 = count(1);
    std::cout << "KinematicLabelBasics: block voxels " << block0 << " -> min " << minBlock
              << " -> " << block1 << ", sphere voxels " << count(2) << std::endl;
    EXPECT_LT(minBlock, block0);  // the sphere displaced block voxels
    EXPECT_GT(block1, block0 / 2); // the block persists
    EXPECT_NEAR(lsmo.time(), 44.0, 1e-6);
}


namespace {

// Sphere 1 grows into the fixed sphere 2 over 30 steps of at most 0.75
// (gate: no voxel of label 2 changes; the labels end in contact).
void
runFixedLabels(double motion)
{
    auto m = makeTwoSpheres(1.0);
    std::map<Coord, float> before;
    for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
        if (*it == 2) before[it.getCoord()] = m->phi()->tree().getValue(it.getCoord());
    }
    ASSERT_FALSE(before.empty());
    ConstantSpeed speed;
    speed.s = -1.0f;
    speed.fixed = {2};
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(*m, speed);
    applyMotion(lsmo, motion);
    for (int n = 0; n < 30; ++n) {
        // dt = min(0.75, motion*cfl): at the default motion 0.5 sphere 1
        // grows ~15 voxels, reaching the wall after ~20 steps
        lsmo.step(0.75f);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << "step " << n << ": " << msg;
    }
    Index64 changed = 0;
    for (const auto& kv : before) {
        const float phi = m->phi()->tree().getValue(kv.first);
        if (!m->labels()->tree().isValueOn(kv.first)
            || m->labels()->tree().getValue(kv.first) != 2
            || std::memcmp(&phi, &kv.second, sizeof(float)) != 0) ++changed;
    }
    EXPECT_EQ(changed, Index64(0));
    // the moving label is in contact with the fixed one
    const Coord* offs = tools::lsmo_internal::faceOffsets();
    Index64 contact = 0;
    for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) {
        if (*it != 1) continue;
        for (int f = 0; f < 6; ++f) contact += m->labels()->tree().getValue(it.getCoord() + offs[f]) == 2;
    }
    EXPECT_GT(contact, Index64(0));
}

} // anonymous namespace


TEST_F(TestLevelSetMultiObject, EmptyStep)
{
    FloatGrid::Ptr phi = FloatGrid::create(3.0f);
    MLS::LabelGridT::Ptr lab = MLS::LabelGridT::create(0);
    {
        auto pa = phi->getAccessor();
        auto la = lab->getAccessor();
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                for (int k = 0; k < 8; ++k) {
                    pa.setValue(Coord(i, j, k), 1.0f);
                    la.setValue(Coord(i, j, k), 0);
                }
    }
    MLS m(phi, lab, 3);
    const FloatTree phiBefore(m.phi()->tree());
    const LabelTree labBefore(m.labels()->tree());
    ConstantSpeed speed;
    speed.s = -1.0f;
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(m, speed);
    EXPECT_EQ(lsmo.step(0.5f), 0.5f);
    EXPECT_TRUE(m.phi()->tree().hasSameTopology(phiBefore));
    EXPECT_TRUE(m.labels()->tree().hasSameTopology(labBefore));
    for (auto it = phiBefore.cbeginValueOn(); it; ++it) {
        EXPECT_EQ(m.phi()->tree().getValue(it.getCoord()), *it);
        EXPECT_EQ(m.labels()->tree().getValue(it.getCoord()), 0);
    }
}

TEST_F(TestLevelSetMultiObject, StepPeakMemory)
{
    // Two spheres of radius 15 at (+-16, 0, 0) in a 64^3 box (dx = 1, halfWidth 3),
    // both growing into contact. The evolver's own working memory during a step
    // (auxiliary leaf buffers plus pooled per-leaf scratch) stays within 1.3x the
    // memory of the pair (a deep copy of both trees alone would be about 1x, on
    // top of the candidate lists), also once the pooled scratch has been reused
    // over many steps of a growing band. Pair bytes are taken before each step
    // (tree.memUsage, an approximation of the allocated bytes). TVD-RK2 keeps
    // the time-n state for its second stage, so it may also use the phi
    // tree's bytes on top.
    for (math::TemporalIntegrationScheme scheme : {math::TVD_RK1, math::TVD_RK2}) {
        const std::string name = scheme == math::TVD_RK1 ? "TVD_RK1" : "TVD_RK2";
        std::map<int32_t, FloatGrid::ConstPtr> in;
        in[1] = tools::createLevelSetSphere<FloatGrid>(15.0f, Vec3f(-16.0f, 0.0f, 0.0f), 1.0f);
        in[2] = tools::createLevelSetSphere<FloatGrid>(15.0f, Vec3f(16.0f, 0.0f, 0.0f), 1.0f);
        auto m = MLS::fromSignedGrids(in, 3);
        ConstantSpeed speed;
        speed.s = -1.0f;
        tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(*m, speed);
        lsmo.setTemporalScheme(scheme);
        EXPECT_EQ(lsmo.scratchMemUsage(), size_t(0));
        for (int n = 0; n < 16; ++n) {
            const size_t phiBytes = size_t(m->phi()->memUsage());
            const size_t pairBytes = phiBytes + size_t(m->labels()->memUsage());
            ASSERT_GT(lsmo.step(1.0f), 0.0f);
            const size_t scratch = lsmo.scratchMemUsage();
            const double bound = 1.3 * double(pairBytes)
                + (scheme == math::TVD_RK2 ? double(phiBytes) : 0.0);
            std::cout << "StepPeakMemory " << name << ": step " << n << " pair " << pairBytes
                      << " B, phi " << phiBytes << " B, scratch " << scratch << " B ("
                      << double(scratch) / double(pairBytes) << "x pair, bound "
                      << bound / double(pairBytes) << "x)" << std::endl;
            EXPECT_GT(scratch, size_t(0)) << name << " step " << n;
            EXPECT_LE(double(scratch), bound) << name << " step " << n;
        }
        std::string msg;
        EXPECT_TRUE(m->validate(&msg)) << msg;
    }
}

TEST_F(TestLevelSetMultiObject, InterruptMidAdvect)
{
    auto m = makeSphere(8.0);
    ConstantSpeed speed;
    speed.s = -1.0f;
    CountingInterrupter interrupter;
    tools::LevelSetMultiObject<MLS, ConstantSpeed, CountingInterrupter> lsmo(*m, speed, &interrupter);
    // uninterrupted, t = 15 needs at least 15 steps (dt <= cfl*dx)
    const size_t steps = lsmo.advect(0.0, 15.0);
    EXPECT_LT(steps, size_t(15));
    EXPECT_EQ(steps, size_t(2));
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}

TEST_F(TestLevelSetMultiObject, SplitStepInterrupt)
{
    // Pure curvature flow: one split step would cover t = 30 (no hyperbolic
    // speed), in parabolic substeps of dx^2/6. The interrupter fires on its
    // third check: advect's check before the step, then after each of the
    // first two substeps.
    auto m = makeSphere(10.0);
    Forces forces;
    forces.setCurvatureWeight(1.0f);
    CountingInterrupter interrupter;
    tools::LevelSetMultiObject<MLS, Forces, CountingInterrupter> lsmo(*m, forces, &interrupter);
    const size_t steps = lsmo.advect(0.0, 30.0);
    EXPECT_EQ(steps, size_t(1));
    EXPECT_EQ(interrupter.calls, 3);
    EXPECT_GT(lsmo.time(), 0.0);
    EXPECT_LT(lsmo.time(), 30.0);
    EXPECT_NEAR(lsmo.time(), 2.0 / 6.0, 1e-5); // two substeps of dx^2/6
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}


namespace {

// Per-label constant dpsi/dt.
struct LabelSpeed
{
    std::map<int32_t, float> speeds;
    std::set<int32_t> fixed;
    float operator()(const Coord&, int32_t l, const Stencil&) const
    {
        auto it = speeds.find(l);
        return it == speeds.end() ? 0.0f : it->second;
    }
    bool isFixed(int32_t l) const { return l != 0 && fixed.count(l) != 0; }
    bool isActive(int32_t) const { return true; }
    float stableDt(float) const { return std::numeric_limits<float>::infinity(); }
};

// Review probe of ContactNoCascade; @a motion as for runPressureGrowth.
void
runContactNoCascade(double motion)
{
    // Review probe: label 1 slab x < 10, label 2 block x in [10,13), y < 3,
    // in contact; label 2 is faster. Along the contact shell (10, y, 12)
    // label 2 may advance at most one voxel per step (the evolver caps dt so
    // interfaces move at most maxInterfaceMotion()*cfl*dx per step: half a
    // voxel by default).
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (int i = 0; i < 13; ++i)
            for (int j = 0; j < 24; ++j)
                for (int k = 0; k < 24; ++k) {
                    if (i < 10) acc.setValue(Coord(i, j, k), 1);
                    else if (j < 3) acc.setValue(Coord(i, j, k), 2);
                }
    }
    auto m = MLS::fromLabelGrid(*labels);
    LabelSpeed speed;
    speed.speeds = {{1, -0.5f}, {2, -1.0f}};
    tools::LevelSetMultiObject<MLS, LabelSpeed> lsmo(*m, speed);
    applyMotion(lsmo, motion);
    const double cap = lsmo.maxInterfaceMotion();
    auto extent = [&]() {
        int maxj = -1;
        for (int j = 0; j < 24; ++j) {
            const Coord c(10, j, 12);
            if (m->labels()->tree().isValueOn(c) && m->labels()->tree().getValue(c) == 2) maxj = j;
        }
        return maxj;
    };
    auto count = [&]() {
        Index64 n = 0;
        for (auto it = m->labels()->tree().cbeginValueOn(); it; ++it) n += *it == 2;
        return n;
    };
    const int y0 = extent();
    ASSERT_EQ(y0, 2);
    std::vector<Index64> counts{count()};
    std::vector<int> extents{y0};
    const double cfl = 0.999;
    // 1.5 x the time of three full steps: four full steps and a half step
    const double tEnd = 1.5 * 3.0 * cap * cfl;
    double t = 0.0;
    int nSteps = 0;
    while (t < tEnd - 1e-6) {
        const double expected = std::min(cap * cfl, tEnd - t);
        const float dt = lsmo.step(float(tEnd - t));
        // max|F| = 1: at most the motion cap (R24)
        EXPECT_FLOAT_EQ(dt, float(expected)) << "step " << nSteps + 1;
        t += dt;
        ++nSteps;
        counts.push_back(count());
        extents.push_back(extent());
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
    }
    ASSERT_EQ(nSteps, 5);
    std::cout << "ContactNoCascade: motion " << cap << " t " << t << ", extent";
    for (int e : extents) std::cout << " " << e;
    std::cout << ", label-2 voxels";
    for (Index64 c : counts) std::cout << " " << c;
    std::cout << std::endl;
    for (int s = 1; s <= nSteps; ++s) {
        EXPECT_LE(extents[s] - extents[s - 1], int(std::floor(cfl + 0.5))) << "step " << s;
    }
    // cumulative: the front (speed 1) has moved t; one voxel of slack at the
    // junction with label 1, where psi_2 is only the pair lower bound
    EXPECT_LE(extents[nSteps] - extents[0], int(std::floor(t + 1.0 + 1e-6)));
    // No cascade anywhere: after the run every label-2 voxel (active or not)
    // is within t + dx of a voxel of the initial block
    // x in [10,13), y in [0,3), z in [0,24).
    Index64 far = 0;
    const double bound = t + 1.0;
    for (auto it = m->labels()->tree().cbeginValueAll(); it; ++it) {
        if (*it != 2) continue;
        CoordBBox box;
        it.getBoundingBox(box);
        for (auto p = box.begin(); p; ++p) {
            const Coord c = *p;
            const double dxv = std::max({10 - c[0], c[0] - 12, 0});
            const double dyv = std::max({0 - c[1], c[1] - 2, 0});
            const double dzv = std::max({0 - c[2], c[2] - 23, 0});
            if (std::sqrt(dxv * dxv + dyv * dyv + dzv * dzv) > bound + 1e-6) ++far;
        }
    }
    EXPECT_EQ(far, Index64(0));
}

} // anonymous namespace


////////////////////////////////////////
// Single-object equivalence with OpenVDB's single
// level set path (LevelSetAdvection / LevelSetFilter, FIRST_BIAS + TVD_RK1).

namespace {


// Fixture: sphere r = 12 dx, centered off the rotation axis so the rotational
// field moves it.
const double kEqRadius = 12.0;
const Vec3d kEqCenter(16.0, 0.0, 0.0);


void
referenceAdvect(FloatGrid& ref, const Vec3SGrid& vel, double t0, double t1,
    math::BiasedGradientScheme spatial = math::FIRST_BIAS,
    math::TemporalIntegrationScheme temporal = math::TVD_RK1)
{
    using Field = tools::DiscreteField<Vec3SGrid>;
    Field field(vel);
    tools::LevelSetAdvection<FloatGrid, Field> adv(ref, field);
    adv.setSpatialScheme(spatial);
    adv.setTemporalScheme(temporal);
    adv.advect(float(t0), float(t1));
}


} // anonymous namespace


TEST_F(TestLevelSetMultiObject, SchemeDefaultsAndSetters)
{
    auto m = makeSphere(8.0);
    Forces forces;
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    EXPECT_EQ(lsmo.spatialScheme(), math::HJWENO5_BIAS);
    EXPECT_EQ(lsmo.temporalScheme(), math::TVD_RK2);
    for (math::BiasedGradientScheme s : {math::SECOND_BIAS, math::THIRD_BIAS, math::WENO5_BIAS}) {
        EXPECT_THROW(lsmo.setSpatialScheme(s), ValueError) << s;
    }
    EXPECT_THROW(lsmo.setTemporalScheme(math::TVD_RK3), ValueError);
    lsmo.setSpatialScheme(math::FIRST_BIAS);
    lsmo.setTemporalScheme(math::TVD_RK1);
    EXPECT_EQ(lsmo.spatialScheme(), math::FIRST_BIAS);
    EXPECT_EQ(lsmo.temporalScheme(), math::TVD_RK1);
    // a stencil made without a scheme is first order
    tools::LabelAwareStencil<MLS> s(*m);
    EXPECT_EQ(s.spatialScheme(), math::FIRST_BIAS);
    EXPECT_THROW(tools::LabelAwareStencil<MLS>(*m, math::WENO5_BIAS), ValueError);
}

TEST_F(TestLevelSetMultiObject, Weno5NeedsHalfWidth3)
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(8.0f, Vec3f(0.0f), 1.0f, 2.0f);
    auto m = MLS::fromSignedGrids(in, 2);
    ASSERT_EQ(m->halfWidth(), 2);
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    using Evolver = tools::LevelSetMultiObject<MLS, Forces>;
    // the default HJWENO5 needs a band of 3 voxels: construction throws
    EXPECT_THROW(Evolver(*m, forces), ValueError);
    EXPECT_THROW(Evolver(*m, forces, math::HJWENO5_BIAS), ValueError);
    EXPECT_THROW(tools::LabelAwareStencil<MLS>(*m, math::HJWENO5_BIAS), ValueError);
    Evolver lsmo(*m, forces, math::FIRST_BIAS);
    EXPECT_EQ(lsmo.spatialScheme(), math::FIRST_BIAS);
    EXPECT_THROW(lsmo.setSpatialScheme(math::HJWENO5_BIAS), ValueError);
    EXPECT_EQ(lsmo.spatialScheme(), math::FIRST_BIAS); // unchanged by the throw
    EXPECT_GT(lsmo.step(1.0f), 0.0f);
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}


////////////////////////////////////////
// Eight objects in a vortex.

namespace {

Index64
countNaN(const FloatGrid& g)
{
    Index64 n = 0;
    for (auto it = g.tree().cbeginValueAll(); it; ++it) n += std::isnan(*it) ? 1 : 0;
    return n;
}

// Voxels with psi < 0 of a level set grid (active and inactive, tiles by
// voxel count).
Index64
insideVoxelCount(const FloatGrid& g)
{
    Index64 n = 0;
    for (auto it = g.tree().cbeginValueAll(); it; ++it) n += *it < 0.0f ? it.getVoxelCount() : 0;
    return n;
}

} // anonymous namespace

namespace {

// The VortexEightObjects scenario and gates; @a motion as for
// runPressureGrowth.
void
runVortexEightObjects(double motion)
{
    // Scene (lsmo_test::Vortex):
    // - eight shapes, labels 1-8, every length times Vortex::kShapeScale =
    //   1.4 (sphere r = 14; cube side 16.8; tetrahedron, octahedron,
    //   dodecahedron, icosahedron with circumradius 14; capsule along z with
    //   radius 7 and axis endpoints +-7; closed ring = 8-segment polygon of
    //   radius 7 in the xy plane with tube radius 4.2). No rotations.
    // - equally spaced on a circle near the periphery of the vortex: label l
    //   centered at angle (l-1)*45 degrees, r = 34, z = 0; dx = 1,
    //   halfWidth 3.
    // - Vortex field (see Vortex): a volume-preserving inward spiral towards
    //   the center of the [-80, 80]^3 velocity grid: inflow 0.4 (vanishing
    //   inside r ~ 8), solid-body swirl 0.012 rad per unit time, axial
    //   outflow cancelling the convergence; advection weight 1, curvature
    //   weight 0.003.
    // - Why these values: an axisymmetric inflow keeps the angular gaps of
    //   objects on one circle, so neighbors touch only once their gaps
    //   (proportional to r) close below a voxel. r = 34 leaves five
    //   neighbor pairs 1.1 to 1.7 voxels apart (the ring's polygon radius
    //   was reduced from 7 to 5 so that it does not touch the sphere at the
    //   start), so contacts form in the outer part of the spiral, before the
    //   objects are strongly stretched near the axis. Rigid rotation (rather
    //   than a Rankine swirl) keeps neighbors from sliding apart once in
    //   contact. A stronger inflow brings the objects deep into the core,
    //   where the stretching costs every object a third or more of its
    //   volume; a weaker one gives contact late. With the default schemes
    //   contact (>= 4 pairs) comes at t = 34 and the smallest label keeps
    //   ~950 voxels at t = 60 (tetrahedron, label 3).
    // - Advect in chunks of 2 until >= 4 contact pairs (capped at t = 70),
    //   then on to t = 60 (Vortex::kEndTime); contacts must stay >= 4 at
    //   every chunk after first contact (>= 3 at full-voxel motion, whose
    //   coarser steps erode the one-voxel contacts of sharp corners).
    // Volume gate (two-sided): with refLoss the loss of the same
    // shape advected alone by LevelSetAdvection with the evolver's schemes
    // (the defaults, HJWENO5 / TVD-RK2) in the
    // same field over the same time, each label's voxel loss lies in
    // [1.1*min(refLoss, 0) - slack, 1.1*max(refLoss, 0) + slack], slack 2%
    // (5% at full-voxel motion, where the ring squeezed against the sphere
    // loses ~6 points more than alone). The exact flow preserves volume;
    // both runs lose volume numerically, mostly where objects stretch.
    // (Shapes, placement, field and schedule: lsmo_test::Vortex.)
    // Contact pairs required after first contact, and the volume gate's
    // slack (fraction), at half-voxel and at full-voxel motion.
    constexpr size_t kMinPairsAfterContact = 4, kMinPairsAfterContactFullVoxel = 3;
    constexpr double kVolumeSlack = 0.02, kVolumeSlackFullVoxel = 0.05;
    const bool fullVoxel = motion > tools::lsmo_internal::kHalfVoxelMotion;
    const size_t minPairsAfter = fullVoxel ? kMinPairsAfterContactFullVoxel
                                           : kMinPairsAfterContact;
    const double volumeSlack = fullVoxel ? kVolumeSlackFullVoxel : kVolumeSlack;
    const Vortex vortex;
    const std::map<int32_t, FloatGrid::ConstPtr> in = vortex.inputs();
    auto m = MLS::fromSignedGrids(in, 3);
    ASSERT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}));
    const std::map<int32_t, Index64> vol0 = labelVoxelCounts(*m);
    ASSERT_TRUE(contactPairs(*m).empty());

    Vec3SGrid::Ptr vel = vortex.velocity();
    size_t steps = 0;
    double tContact = -1.0;
    std::set<std::pair<int32_t, int32_t>> pairs;
    // failures of this run only (FullVoxelMotion runs several scenarios)
    const int failures0 = failureCount();
    const auto start = std::chrono::steady_clock::now();
    math::BiasedGradientScheme spatial = math::UNKNOWN_BIAS;
    math::TemporalIntegrationScheme temporal = math::UNKNOWN_TIS;
    double usedMotion = 0.0;
    // Keep going to the end time; contacts must persist after first contact.
    const double t = vortex.run(*m, vel, steps, tContact,
        [&](double tc, size_t n, const std::set<std::pair<int32_t, int32_t>>& p, int after) {
            std::string msg;
            if (!m->validate(&msg)) { ADD_FAILURE() << "t " << tc << ": " << msg; return false; }
            Index64 overlaps = 0, badCover = 0;
            partitionErrors(*m, overlaps, badCover);
            EXPECT_EQ(overlaps, Index64(0)) << "t " << tc;
            EXPECT_EQ(badCover, Index64(0)) << "t " << tc;
            EXPECT_EQ(countNaN(*m->phi()), Index64(0)) << "t " << tc;
            // phi = 0 belongs to no label: the evolver never writes it
            Index64 zeros = 0;
            for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) zeros += *it == 0.0f;
            EXPECT_EQ(zeros, Index64(0)) << "t " << tc;
            pairs = p;
            std::cout << "Vortex: t " << tc << " steps " << n << " contacts " << p.size();
            for (auto& lv : labelVoxelCounts(*m)) std::cout << " " << lv.first << ":" << lv.second;
            std::cout << std::endl;
            if (after >= 1) { EXPECT_GE(p.size(), minPairsAfter) << "t " << tc; }
            return failureCount() == failures0;
        },
        [&](Vortex::Evolver& e) {
            applyMotion(e, motion);
            usedMotion = e.maxInterfaceMotion();
            spatial = e.spatialScheme();
            temporal = e.temporalScheme();
        });
    ASSERT_EQ(failureCount(), failures0);
    std::cout << "Vortex: >= 4 contact pairs first at t " << tContact << std::endl;
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "Vortex: motion " << usedMotion << ", t " << t
              << ", " << steps << " steps, " << secs << " s ("
              << 1e3 * secs / double(std::max(steps, size_t(1))) << " ms/step), contact pairs";
    for (const auto& pr : pairs) std::cout << " " << pr.first << "-" << pr.second;
    std::cout << std::endl;
    EXPECT_GE(pairs.size(), size_t(4));

    const std::map<int32_t, Index64> vol1 = labelVoxelCounts(*m);
    for (int32_t l = 1; l <= 8; ++l) {
        // reference: the same shape alone, LevelSetAdvection with the
        // evolver's schemes to the same t
        FloatGrid::Ptr ref = in.at(l)->deepCopy();
        const double r0 = double(insideVoxelCount(*ref));
        referenceAdvect(*ref, *vel, 0.0, t, spatial, temporal);
        const double refLoss = 1.0 - double(insideVoxelCount(*ref)) / r0;
        const double v0 = double(vol0.at(l));
        const double v1 = vol1.count(l) ? double(vol1.at(l)) : 0.0;
        const double loss = 1.0 - v1 / v0;
        std::cout << "Vortex: label " << l << " voxels " << v0 << " -> " << v1 << " (loss "
                  << 100.0 * loss << "%), reference loss " << 100.0 * refLoss << "%" << std::endl;
        EXPECT_GE(v1, 1.0) << "label " << l;
        // two-sided (a reference gain bounds MOGAC's gain too)
        EXPECT_LE(loss, 1.1 * std::max(refLoss, 0.0) + volumeSlack) << "label " << l;
        EXPECT_GE(loss, 1.1 * std::min(refLoss, 0.0) - volumeSlack) << "label " << l;
    }
}

} // anonymous namespace


////////////////////////////////////////
// Volume constraints


TEST_F(TestLevelSetMultiObject, VolumeConstraintApi)
{
    using Evolver = tools::LevelSetMultiObject<MLS, LabelSpeed>;
    // labelVolume: a sphere of radius 8 (dx 1 and 0.5) within 1% of 4/3 pi r^3;
    // an absent label has volume 0
    const double r = 8.0;
    const double exact = 4.0 / 3.0 * math::pi<double>() * r * r * r;
    for (const double dx : {1.0, 0.5}) {
        auto m = makeSphere(r, Vec3d(0.0), dx);
        LabelSpeed speed;
        Evolver e(*m, speed);
        const double v = e.labelVolume(1);
        std::cout << "VolumeConstraintApi: dx " << dx << " sphere volume " << v << " (exact "
                  << exact << ")" << std::endl;
        EXPECT_NEAR(v, exact, 0.01 * exact) << "dx " << dx;
        EXPECT_EQ(e.labelVolume(7), 0.0);
    }

    auto m = makeSphere(r);
    LabelSpeed speed;
    speed.fixed = {2};
    Evolver e(*m, speed);
    EXPECT_DOUBLE_EQ(e.volumeRelaxation(), 0.5);
    EXPECT_THROW(e.setVolumeConstraint(0, 1.0), ValueError);  // passive label 0
    EXPECT_THROW(e.preserveInitialVolume(0), ValueError);
    EXPECT_THROW(e.setVolumeConstraint(2, 1.0), ValueError);  // fixed
    EXPECT_THROW(e.setVolumeConstraint(1, -1.0), ValueError); // negative target
    EXPECT_THROW(e.setVolumeConstraint(1, std::numeric_limits<double>::quiet_NaN()), ValueError);
    EXPECT_THROW(e.setVolumeRelaxation(0.0), ValueError);
    EXPECT_THROW(e.setVolumeRelaxation(1.5), ValueError);
    auto still = [](double) { return math::Mat4d::identity(); };
    e.setKinematicLabel(3, tools::createLevelSetSphere<FloatGrid>(2.0f, Vec3f(20.0f), 1.0f), still);
    EXPECT_THROW(e.setVolumeConstraint(3, 1.0), ValueError);  // kinematic
    EXPECT_THROW(e.preserveInitialVolume(3), ValueError);
    e.clearKinematicLabel(3);
    e.preserveInitialVolume(1);
    EXPECT_THROW(e.setKinematicLabel(1, tools::createLevelSetSphere<FloatGrid>(2.0f,
        Vec3f(20.0f), 1.0f), still), ValueError); // constrained labels cannot become kinematic
    EXPECT_EQ(e.volumeConstraints().size(), size_t(1));
    EXPECT_DOUBLE_EQ(e.volumeConstraints().at(1), e.labelVolume(1));

    // Growth to a target with zero speed: the constraint alone moves the
    // interface, to within 1% of the target of a radius-9 sphere's volume.
    const double target = 4.0 / 3.0 * math::pi<double>() * 729.0;
    e.setVolumeConstraint(1, target);
    for (int n = 0; n < 20; ++n) {
        const float dt = e.step(1.0f);
        ASSERT_TRUE(std::isfinite(dt) && dt > 0.0f);
    }
    std::string msg;
    ASSERT_TRUE(m->validate(&msg)) << msg;
    const double v = e.labelVolume(1);
    std::cout << "VolumeConstraintApi: grown to " << v << " (target " << target << ")" << std::endl;
    EXPECT_NEAR(v, target, 0.01 * target);
    EXPECT_NEAR(meanRadius(zeroCrossings(*m->extractSignedDistance(1)), Vec3d(0.0)), 9.0, 0.25);

    // Cleared: zero speed leaves the volume where it is.
    e.clearVolumeConstraints();
    EXPECT_TRUE(e.volumeConstraints().empty());
    for (int n = 0; n < 5; ++n) e.step(1.0f);
    EXPECT_NEAR(e.labelVolume(1), v, 1e-3 * v);
}

TEST_F(TestLevelSetMultiObject, VolumeConstraintCancelsDrift)
{
    // A sphere of radius 8 under a uniform shrinking speed (dpsi/dt = 0.5)
    // for t = 10: unconstrained it loses most of its volume; constrained to
    // its initial volume the mean-speed term cancels the drift, so it ends
    // within 2% (a proportional correction alone would lag by about one
    // step's drift divided by the relaxation).
    double change[2] = {0.0, 0.0};
    for (int k = 0; k < 2; ++k) {
        auto m = makeSphere(8.0);
        LabelSpeed speed;
        speed.speeds[1] = 0.5f;
        tools::LevelSetMultiObject<MLS, LabelSpeed> e(*m, speed);
        const double v0 = e.labelVolume(1);
        if (k == 1) e.preserveInitialVolume(1);
        e.advect(0.0, 10.0);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        change[k] = e.labelVolume(1) / v0 - 1.0;
        std::cout << "VolumeConstraintCancelsDrift: " << (k ? "constrained" : "unconstrained")
                  << " volume change " << 100.0 * change[k] << "%" << std::endl;
    }
    EXPECT_LT(change[0], -0.5);
    EXPECT_LE(std::abs(change[1]), 0.02);
}


////////////////////////////////////////
// Topology-preserving evolution

namespace {

using Conn = tools::MultiObjectConnectivity;
const Conn kAllConn[4] = {Conn::FG26_BG6, Conn::FG6_BG26, Conn::FG18_BG6, Conn::FG6_BG18};


} // anonymous namespace


namespace {

// The three spheres of the cube test: radius 4, centers at x = 48.
const Coord kOrbitSpheres[3] = {Coord(48, 10, 12), Coord(48, 20, 28), Coord(48, 29, 15)};
// Angular speed about x (rad per unit time): one revolution over the run.
const double kOrbitOmega = 2.0 * math::pi<double>() / 84.0;

// Common initial centroid of the spheres: the point the orbit axis
// (parallel to x) passes through.
Vec3d
orbitCenter()
{
    Vec3d c(0.0);
    for (const Coord& p : kOrbitSpheres) c += p.asVec3d();
    return c / 3.0;
}

// @a p rotated about the orbit axis by kOrbitOmega * @a t.
Vec3d
orbitPosition(const Vec3d& p, double t)
{
    const Vec3d c = orbitCenter();
    const double a = kOrbitOmega * t, dy = p[1] - c[1], dz = p[2] - c[2];
    return Vec3d(p[0], c[1] + dy * std::cos(a) - dz * std::sin(a),
        c[2] + dy * std::sin(a) + dz * std::cos(a));
}

// The same rigid rotation as a matrix in the row-vector convention of
// Mat4d::transform: x' = (x - c) R + c.
math::Mat4d
orbitMotion(double t)
{
    const Vec3d c = orbitCenter();
    const double a = kOrbitOmega * t, ca = std::cos(a), sa = std::sin(a);
    math::Mat4d m = math::Mat4d::identity();
    m(1, 1) = ca; m(1, 2) = sa;
    m(2, 1) = -sa; m(2, 2) = ca;
    for (int j = 0; j < 3; ++j) {
        double v = c[j];
        for (int i = 0; i < 3; ++i) v -= c[i] * m(i, j);
        m(3, j) = v;
    }
    return m;
}

// Volume centroid of a signed grid with the sub-voxel inside fraction
// H(psi) = clamp(0.5 - psi/dx, 0, 1), over @a box (index space).
Vec3d
insideCentroid(const FloatGrid& sdf, const CoordBBox& box)
{
    const double dx = sdf.voxelSize()[0];
    tree::ValueAccessor<const FloatTree> acc(sdf.tree());
    Vec3d sum(0.0);
    double w = 0.0;
    for (auto p = box.begin(); p; ++p) {
        const double h = std::min(1.0, std::max(0.0, 0.5 - double(acc.getValue(*p)) / dx));
        sum += sdf.indexToWorld(*p) * h;
        w += h;
    }
    return w > 0.0 ? sum / w : sum;
}

constexpr double kCubeDx = 1.0; // voxel size of the cube scenario

struct CubeRunResult
{
    std::vector<lsmo_test::DigitalTopology> topo; // cube, per chunk
    double shift = 0.0;      // centroid x shift of label 1 (diagnostic)
    double frontShift = 0.0; // leading-face shift along the centerline column (sub-voxel)
    double maxXShift = 0.0;  // change of max x over all label-1 voxels (diagnostic)
    bool changed = false;    // cube topology differed from (1, 0, 1) at some chunk
    double firstChange = -1.0;                // end time of the first changed chunk
    lsmo_test::DigitalTopology firstChangeTopo; // cube topology at that chunk
    double maxCentroidError[3] = {0, 0, 0}; // sphere centroid vs rotated center, max over chunks (dx)
    double minVolumeFraction[3] = {1e9, 1e9, 1e9}; // sphere volume / (4/3) pi 4^3, min over chunks
    int maxComponents[3] = {0, 0, 0}; // sphere components, max over chunks (diagnostic)
    Index64 belowFloor = 0; // active voxels with phi < minimumPhi*dx, summed over chunks
    int maxCavities[3] = {0, 0, 0};   // sphere cavities, max over chunks (diagnostic)
    MLS::Ptr state;          // final state
};

// Cube (label 1, side 40 voxels, [0,39]^3) advected along +x by
// 1.5 x (side + sphere diameter + 8) = 84 voxels through three balls of
// radius 4 (labels 2-4, kinematic) centered at x = 48 at distinct (y, z),
// which orbit together rigidly about the axis parallel to x through their
// common centroid at kOrbitOmega (one revolution over the run), in the
// plane orthogonal to the cube's motion. The evolver protects topology
// under @a rule; object topology is verified with that same rule
// (connectivityOf(evolver)). With @a protect, label 1 is
// topology-preserving and the per-chunk gates are asserted (cube (1,0,1),
// consistent meshes, validate, partition, sphere centroids within 1 dx of
// the rotated centers; the obstacles' claims on cube voxels pass the
// simple-point test and trapped cube voxels are wrapped outward); the worst
// centroid error, smallest volume and largest component and cavity counts
// per sphere are returned (the last three as diagnostics: a sphere can be
// dented where it exits the protected cube). Without
// @a protect (the control) the run stops at the first chunk where the
// cube's topology changed. @a tEnd < 0 runs the whole distance.
CubeRunResult
runCubeThroughSpheres(bool protect, Conn rule, int grainSize = 1, double tEnd = -1.0,
    double motion = kDefaultMotion)
{
    const int side = 40;
    const bool fullRun = !(tEnd > 0.0);
    const double distance = fullRun ? 1.5 * (side + 8.0 + 8.0) : tEnd;
    const double fullDistance = 1.5 * (side + 8.0 + 8.0);
    const CoordBBox domain(Coord(-8, -8, -8),
        Coord(side - 1 + int(fullDistance) + 8, side + 7, side + 7));
    Int32Grid::Ptr labels = Int32Grid::create(0);
    {
        auto acc = labels->getAccessor();
        for (auto p = CoordBBox(Coord(0), Coord(side - 1)).begin(); p; ++p) acc.setValue(*p, 1);
        for (int s = 0; s < 3; ++s) {
            const Coord& c = kOrbitSpheres[s];
            for (auto p = CoordBBox(c - Coord(4), c + Coord(4)).begin(); p; ++p) {
                if ((*p - c).asVec3i().lengthSqr() <= 16) acc.setValue(*p, 2 + s);
            }
        }
    }
    auto m = MLS::fromLabelGrid(*labels);
    EXPECT_EQ(m->labelSet(), (std::vector<int32_t>{1, 2, 3, 4}));
    const double dx = m->voxelSize();
    auto labelCentroidX = [&]() {
        double sum = 0.0, n = 0.0;
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) {
            if (acc.getValue(*p) == 1) { sum += (*p)[0]; n += 1.0; }
        }
        return n > 0.0 ? sum / n : 0.0;
    };
    const double x0 = labelCentroidX();
    auto labelMaxX = [&]() {
        int mx = std::numeric_limits<int>::min();
        tree::ValueAccessor<const LabelTree> acc(m->labels()->tree());
        for (auto p = domain.begin(); p; ++p) {
            if (acc.getValue(*p) == 1) mx = std::max(mx, (*p)[0]);
        }
        return mx;
    };
    const int front0 = labelMaxX();
    // Leading face along the cube's centerline column (y, z) = (20, 20),
    // next to its center (19.5, 19.5): the last sub-voxel zero crossing
    // (- to +) of the extracted psi_1 along x. The orbits stay clear of it:
    // the sphere centers circle the orbit axis through (19.67, 18.33) at
    // radii 9.7-11.6, so a sphere of radius 4 comes no closer than 5.7
    // voxels to the axis, and this column is 1.7 voxels from it.
    const Coord column(0, 20, 20);
    auto centerFront = [&]() {
        FloatGrid::Ptr sdf = m->extractSignedDistance(1);
        tree::ValueAccessor<const FloatTree> acc(sdf->tree());
        double front = std::numeric_limits<double>::quiet_NaN();
        for (int x = domain.min()[0]; x < domain.max()[0]; ++x) {
            const float a = acc.getValue(Coord(x, column[1], column[2]));
            const float b = acc.getValue(Coord(x + 1, column[1], column[2]));
            if (a < 0.0f && !(b < 0.0f)) front = (x + double(a) / double(a - b)) * dx;
        }
        return front;
    };
    {
        const Vec3d c = orbitCenter();
        for (const Coord& sc : kOrbitSpheres) {
            const double r = std::hypot(sc[1] - c[1], sc[2] - c[2]);
            const double dc = std::hypot(column[1] - c[1], column[2] - c[2]);
            EXPECT_GT(r - 4.0 - 1.0, dc) << "an orbit crosses the centerline column";
        }
    }
    const double centerFront0 = centerFront();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(1.0f, 0.0f, 0.0f)), 1.0f);
    forces.setCurvatureWeight(0.0f);
    tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
    lsmo.setGrainSize(grainSize);
    applyMotion(lsmo, motion);
    for (int s = 0; s < 3; ++s) {
        lsmo.setKinematicLabel(2 + s, tools::createLevelSetSphere<FloatGrid>(4.0f,
            Vec3f(kOrbitSpheres[s].asVec3d()), float(dx)), orbitMotion);
    }
    lsmo.setConnectivity(rule);
    if (protect) lsmo.setTopologyPreservingLabels({1});
    // the matrix motion is the analytic rotation
    for (double tt : {0.0, 10.0, 31.0}) {
        const Vec3d p = kOrbitSpheres[0].asVec3d();
        EXPECT_LT((orbitMotion(tt).transform(p) - orbitPosition(p, tt)).length(), 1e-9);
    }
    const Conn conn = lsmo_test::connectivityOf(lsmo);
    const std::string ruleName = lsmo_test::connectivityName(conn);

    CubeRunResult res;
    const lsmo_test::DigitalTopology ball{1, 0, 1};
    const std::string tag = std::string(protect ? "protected" : "control") + "/" + ruleName;
    const std::string log = "TopologyCube(" + tag + ")";
    auto chunkTag = [&](double t, const std::string& what) {
        std::ostringstream os;
        os << tag << "/t" << std::setw(3) << std::setfill('0') << int(std::lround(t)) << "/" << what;
        return os.str();
    };
    EXPECT_TOPOLOGY_CONSISTENT(lsmo_test::verifyLabelTopology(*m, 1, domain, conn,
        chunkTag(0.0, "cube"), &ball, lsmo_test::phiFloorOf(lsmo)));
    auto volume = [](const FloatGrid& g) {
        return g.tree().activeVoxelCount() > 0 ? tools::levelSetVolume(g) : 0.0;
    };
    const double exactVolume = 4.0 / 3.0 * math::pi<double>() * 64.0 * dx * dx * dx;
    const double chunk = 2.0;
    double t = 0.0;
    size_t steps = 0;
    const auto start = std::chrono::steady_clock::now();
    while (t < distance - 1e-9) {
        const double t1 = std::min(distance, t + chunk);
        steps += lsmo.advect(t, t1);
        t = t1;
        const double floor = lsmo_test::phiFloorOf(lsmo);
        const auto cube = lsmo_test::verifyLabelTopology(*m, 1, domain, conn, chunkTag(t, "cube"),
            protect ? &ball : nullptr, floor);
        EXPECT_TOPOLOGY_CONSISTENT(cube);
        // the phi floor holds on protected interfaces (protected runs); without
        // protection there is no floor and smaller phi values occur
        EXPECT_TRUE(lsmo.validateTopologyFloor()) << "t " << t;
        for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
            if (*it < float(lsmo.minimumPhi() * dx)) ++res.belowFloor;
        }
        res.topo.push_back(cube.region);
        std::ostringstream row;
        row << log << ": t " << t << " steps " << steps << " cube " << cube.region;
        if (cube.region != ball && !res.changed) {
            res.changed = true;
            res.firstChange = t;
            res.firstChangeTopo = cube.region;
        }
        if (!protect) {
            std::cout << row.str() << std::endl;
            if (res.changed) {
                std::cout << log << ": first topology change at t " << t << ": " << cube.message
                          << std::endl;
                break;
            }
            continue;
        }
        bool chunkOk = cube.ok;
        for (int s = 0; s < 3; ++s) {
            const int32_t l = 2 + s;
            // Spheres: consistent meshes and centroids are gated; their
            // components and volume are diagnostics only, since cube voxels
            // the protected cube cannot give up can dent a sphere where it
            // exits the cube (see setKinematicLabel).
            const std::string stag = chunkTag(t, "sphere" + std::to_string(l));
            const auto sphere = lsmo_test::verifyLabelTopology(*m, l, domain, conn, stag, nullptr,
                floor);
            EXPECT_TOPOLOGY_CONSISTENT(sphere);
            chunkOk = chunkOk && sphere.ok;
            res.maxComponents[s] = std::max(res.maxComponents[s], sphere.region.components);
            res.maxCavities[s] = std::max(res.maxCavities[s], sphere.region.cavities);
            FloatGrid::Ptr sdf = m->extractSignedDistance(l);
            const Vec3d expected = orbitPosition(kOrbitSpheres[s].asVec3d(), t);
            const CoordBBox box(Coord::round(expected) - Coord(12), Coord::round(expected) + Coord(12));
            const double err = (insideCentroid(*sdf, box) - expected).length() / dx;
            const double frac = volume(*sdf) / exactVolume;
            res.maxCentroidError[s] = std::max(res.maxCentroidError[s], err);
            res.minVolumeFraction[s] = std::min(res.minVolumeFraction[s], frac);
            EXPECT_LE(err, 1.0) << stag;
            chunkOk = chunkOk && err <= 1.0;
            row << " L" << l << " " << sphere.region << " err " << err << " vol " << frac;
        }
        std::cout << row.str() << std::endl;
        std::string msg;
        const bool valid = m->validate(&msg);
        EXPECT_TRUE(valid) << "t " << t << ": " << msg;
        Index64 overlaps = 0, badCover = 0;
        partitionErrors(*m, overlaps, badCover);
        EXPECT_EQ(overlaps, Index64(0)) << "t " << t;
        EXPECT_EQ(badCover, Index64(0)) << "t " << t;
        // stop at this run's first failed chunk (not at failures of earlier runs)
        if (!(chunkOk && valid && overlaps == 0 && badCover == 0)) break;
    }
    res.shift = labelCentroidX() - x0;
    res.frontShift = centerFront() - centerFront0;
    res.maxXShift = double(labelMaxX() - front0);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << log << ": motion " << lsmo.maxInterfaceMotion() << ", " << steps << " steps, "
              << secs << " s (" << 1e3 * secs / double(std::max(steps, size_t(1)))
              << " ms/step), t " << t
              << ", active phi below minimumPhi*dx (summed over chunks) " << res.belowFloor
              << ", leading-face shift (centerline) " << res.frontShift
              << ", max-x shift (diagnostic) " << res.maxXShift << ", centroid shift " << res.shift
              << std::endl;
    if (protect) {
        for (int s = 0; s < 3; ++s) {
            std::cout << log << ": sphere " << 2 + s << " max centroid error "
                      << res.maxCentroidError[s] << " dx, min volume "
                      << 100.0 * res.minVolumeFraction[s] << "% of (4/3) pi 4^3, max components "
                      << res.maxComponents[s] << ", max cavities " << res.maxCavities[s]
                      << " (diagnostics)" << std::endl;
        }
    }
    res.state = m;
    return res;
}

} // anonymous namespace

namespace {

// The TopologyCubeThroughSpheres gates under both rules; @a motion as for
// runPressureGrowth.
void
runTopologyCubeGates(double motion)
{
    const double distance = 84.0;
    for (Conn rule : {Conn::FG26_BG6, Conn::FG6_BG26}) {
        SCOPED_TRACE(lsmo_test::connectivityName(rule));
        const CubeRunResult prot = runCubeThroughSpheres(true, rule, 1, -1.0, motion);
        EXPECT_FALSE(prot.changed);
        // Leading face along the centerline column, not the centroid:
        // topology protection leaves label-1 material tethered around the
        // spheres the cube has passed, so the centroid lags (printed only);
        // the max x over all label-1 voxels also sees material the wrap
        // left ahead of the face in a sphere's column (printed only).
        EXPECT_NEAR(prot.frontShift, distance, 2.0 * kCubeDx);
        for (int s = 0; s < 3; ++s) {
            // also asserted at every chunk; volume and components are printed only
            EXPECT_LE(prot.maxCentroidError[s], 1.0) << "sphere " << 2 + s;
        }
        // control: without protection the spheres change the cube's topology
        const CubeRunResult ctrl = runCubeThroughSpheres(false, rule, 1, -1.0, motion);
        EXPECT_TRUE(ctrl.changed);
        // no protected label: no floor, and phi values below it occur
        EXPECT_GT(ctrl.belowFloor, Index64(0));
        std::cout << "TopologyCube(control/" << lsmo_test::connectivityName(rule)
                  << "): first change at t " << ctrl.firstChange << " " << ctrl.firstChangeTopo
                  << std::endl;
    }
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// Accuracy of extractSignedDistance after evolution, and world units
// ---------------------------------------------------------------------------

namespace {

// Self-consistency of extractSignedDistance(@a label): the extracted grid's
// zero isosurface is meshed (volumeToMesh) and converted back to an exact
// distance to that mesh (meshToLevelSet); @a ifaceMax is the max error over
// the voxels within one voxel of the surface, @a rms the RMS error over the
// extracted band, both in voxels. The metric's own floor (an analytic sphere
// against its marching-cubes mesh) is about 0.055*dx near the surface.
void
extractSelfConsistency(const MLS& m, int32_t label, double& ifaceMax, double& rms)
{
    FloatGrid::Ptr g = m.extractSignedDistance(label);
    std::vector<Vec3s> points;
    std::vector<Vec3I> tris;
    std::vector<Vec4I> quads;
    tools::volumeToMesh(*g, points, tris, quads, 0.0);
    FloatGrid::Ptr ref = tools::meshToLevelSet<FloatGrid>(g->transform(), points, tris, quads,
        4.0f);
    const double dx = m.voxelSize();
    tree::ValueAccessor<const FloatTree> acc(ref->tree());
    ifaceMax = 0.0;
    double sum2 = 0.0;
    size_t n = 0;
    for (auto it = g->tree().cbeginValueOn(); it; ++it) {
        float r;
        if (!acc.probeValue(it.getCoord(), r)) continue;
        const double err = std::abs(double(*it) - double(r)) / dx;
        if (std::abs(r) < dx) ifaceMax = std::max(ifaceMax, err);
        sum2 += err * err;
        ++n;
    }
    rms = n ? std::sqrt(sum2 / double(n)) : 0.0;
}

} // anonymous namespace


namespace {

// Shear rate of the velocity v = (kShearRate*z, 0, 0) of SpeedExtensionKeepsDistance.
constexpr float kShearRate = 0.05f;
// Gates of SpeedExtensionKeepsDistance: the extension-on interface max error
// is at most this fraction of the extension-off error, and at most this many
// voxels.
constexpr double kSpeedExtensionErrorRatio = 0.6;
constexpr double kSpeedExtensionMaxIfaceError = 0.15;

// The shear field on @a xf, dense over the region the band of the sheared
// sphere of SpeedExtensionKeepsDistance sweeps (zero outside).
Vec3SGrid::Ptr
makeShearField(const math::Transform& xf)
{
    Vec3SGrid::Ptr g = Vec3SGrid::create(Vec3f(0.0f));
    g->setTransform(xf.copy());
    auto acc = g->getAccessor();
    for (auto p = CoordBBox(Coord(-36, -20, -20), Coord(36, 20, 20)).begin(); p; ++p) {
        const Vec3d w = xf.indexToWorld(*p);
        acc.setValue(*p, Vec3f(kShearRate * float(w.z()), 0.0f, 0.0f));
    }
    return g;
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, SpeedExtensionKeepsDistance)
{
    // ExtractAccuracyAfterEvolution's advection case in a shear field: the
    // closest-interface-point sampling of the velocity (the default) keeps
    // the distance property within kSpeedExtensionMaxIfaceError*dx at the
    // interface and clearly better than voxel-center sampling (at most
    // kSpeedExtensionErrorRatio of its error). The voxel-center error itself
    // is only reported (about 0.21*dx: the shear changes |grad psi| at the
    // interface voxels, which the redistancing keeps as seeds).
    {
        Forces f;
        EXPECT_TRUE(f.speedExtension()); // on by default
        f.setSpeedExtension(false);
        EXPECT_FALSE(f.speedExtension());
    }

    // The closest point of an interface voxel lies on the sphere, and is the
    // same for both labels of the interface (one sample serves both).
    {
        const double r = 12.3;
        const Vec3d c(0.3, 0.1, 0.2);
        auto m = makeSphere(r, c);
        Stencil s(*m);
        int n = 0;
        double worst = 0.0;
        for (auto it = m->phi()->tree().cbeginValueOn(); it; ++it) {
            if (std::abs(*it) > 1.0f) continue;
            const Coord p = it.getCoord();
            s.moveTo(p, 1);
            const Vec3d q1 = s.closestInterfacePoint();
            s.moveTo(p, 0);
            const Vec3d q0 = s.closestInterfacePoint();
            EXPECT_EQ(q0, q1) << p;
            worst = std::max(worst, std::abs((q1 - c).length() - r));
            ++n;
        }
        ASSERT_GT(n, 1000);
        std::cout << "SpeedExtensionKeepsDistance: closest points of " << n
                  << " interface voxels within " << worst << " dx of the sphere" << std::endl;
        EXPECT_LT(worst, 0.1);
    }

    constexpr double kT1 = 15.0;
    double iface[2], rms[2];
    for (int on = 0; on < 2; ++on) {
        auto m = makeSphere(12.3, Vec3d(0.3, 0.1, 0.2));
        Forces forces;
        forces.setSpeedExtension(on == 1);
        forces.setAdvection(makeShearField(m->phi()->transform()), 1.0f);
        tools::LevelSetMultiObject<MLS, Forces> lsmo(*m, forces);
        const size_t steps = lsmo.advect(0.0, kT1);
        std::string msg;
        ASSERT_TRUE(m->validate(&msg)) << msg;
        extractSelfConsistency(*m, 1, iface[on], rms[on]);
        std::cout << "SpeedExtensionKeepsDistance shear (t = " << kT1 << ", " << steps
                  << " steps, extension " << (on ? "on" : "off") << "): interface max error "
                  << iface[on] << " dx, band RMS " << rms[on] << " dx" << std::endl;
    }
    EXPECT_LE(iface[1], kSpeedExtensionErrorRatio * iface[0]);
    EXPECT_LE(iface[1], kSpeedExtensionMaxIfaceError);
}


////////////////////////////////////////
// Incremental (dirty-region) redistancing


TEST_F(TestLevelSetMultiObject, MaxInterfaceMotionApi)
{
    auto m = makeSphere(8.0);
    ConstantSpeed speed;
    tools::LevelSetMultiObject<MLS, ConstantSpeed> lsmo(*m, speed);
    EXPECT_EQ(lsmo.maxInterfaceMotion(), tools::lsmo_internal::kDefaultMaxInterfaceMotionVoxels);
    lsmo.setMaxInterfaceMotion(tools::lsmo_internal::kFullVoxelMotion);
    EXPECT_EQ(lsmo.maxInterfaceMotion(), 1.0);
    lsmo.setMaxInterfaceMotion(tools::lsmo_internal::kHalfVoxelMotion);
    EXPECT_EQ(lsmo.maxInterfaceMotion(), 0.5);
    for (double bad : {0.0, -0.5, 0.75, 1.5, 2.0, std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(lsmo.setMaxInterfaceMotion(bad), ValueError) << bad;
        EXPECT_EQ(lsmo.maxInterfaceMotion(), 0.5) << bad;
    }
    // the cap of an unconstrained step: motion * cfl * dx / max|F|
    speed.s = -1.0f;
    EXPECT_FLOAT_EQ(lsmo.step(10.0f), 0.5f * lsmo.cfl());
    lsmo.setMaxInterfaceMotion(1.0);
    EXPECT_FLOAT_EQ(lsmo.step(10.0f), lsmo.cfl());
    std::string msg;
    EXPECT_TRUE(m->validate(&msg)) << msg;
}

TEST_F(TestLevelSetMultiObject, FullVoxelMotion)
{
    // The gates of these scenarios (the same as at half-voxel motion), at
    // full-voxel motion (setMaxInterfaceMotion(1.0)): ContactNoCascade (label-2 extent
    // growth at most one voxel per step, no cascade), PressureGrowth,
    // UniformAdvection, CurvatureFlow, VortexEightObjects (t = 60),
    // TopologyCubeThroughSpheres (both rules, protected and control) and
    // FixedLabels; and PressureGrowth takes at most 0.6x the steps of the
    // half-voxel run.
    constexpr double kFull = tools::lsmo_internal::kFullVoxelMotion;
    constexpr double kHalf = tools::lsmo_internal::kHalfVoxelMotion;
    constexpr double kMaxStepRatio = 0.6;
    {
        SCOPED_TRACE("ContactNoCascade");
        runContactNoCascade(kFull);
    }
    {
        SCOPED_TRACE("PressureGrowth");
        const size_t half = runPressureGrowth(kHalf);
        const size_t full = runPressureGrowth(kFull);
        std::cout << "FullVoxelMotion: PressureGrowth steps " << full << " vs " << half
                  << " at half-voxel motion" << std::endl;
        EXPECT_LE(double(full), kMaxStepRatio * double(half));
    }
    {
        SCOPED_TRACE("UniformAdvection");
        runUniformAdvection(kFull);
    }
    {
        SCOPED_TRACE("CurvatureFlow");
        runCurvatureFlow(kFull);
    }
    {
        SCOPED_TRACE("FixedLabels");
        runFixedLabels(kFull);
    }
    {
        SCOPED_TRACE("VortexEightObjects");
        runVortexEightObjects(kFull);
    }
    {
        SCOPED_TRACE("TopologyCubeThroughSpheres");
        runTopologyCubeGates(kFull);
    }
}


namespace {

// Two spheres side by side (labels 1 and 2) and a small one (label 3)
// above them, half width 3.
MLS::Ptr
makeEditScene()
{
    std::map<int32_t, FloatGrid::ConstPtr> in;
    in[1] = tools::createLevelSetSphere<FloatGrid>(7.0f, Vec3f(-6.0f, 0.0f, 0.0f), 1.0f, 3.0f);
    in[2] = tools::createLevelSetSphere<FloatGrid>(7.0f, Vec3f(6.0f, 0.0f, 0.0f), 1.0f, 3.0f);
    in[3] = tools::createLevelSetSphere<FloatGrid>(4.0f, Vec3f(0.0f, 10.0f, 0.0f), 1.0f, 3.0f);
    return MLS::fromSignedGrids(in, 3);
}

// External edits of @a m: dilate the band by one voxel and rename label
// @a from to @a to (every voxel and tile, active or not).
void
editExternally(MLS& m, int32_t from, int32_t to)
{
    m.dilate(1);
    using LabelT = LabelTree::ValueType;
    for (auto it = m.labels()->tree().beginValueAll(); it; ++it) {
        if (*it == LabelT(from)) it.setValue(LabelT(to));
    }
}

} // anonymous namespace

TEST_F(TestLevelSetMultiObject, ExternalEditsMarked)
{
    // Edits of the pair between advect() calls (MultiLevelSet::dilate() and
    // a relabel) announced with markExternallyModified(): the next step
    // re-solves the whole band, so builds with OPENVDB_ENABLE_ASSERTS do not
    // abort in the dirty-region check, and the incremental run keeps
    // matching a full-redistance run (labels identical, phi within the
    // redistance tolerance).
    using Evolver = tools::LevelSetMultiObject<MLS, Forces>;
    MLS::Ptr incr = makeEditScene(), full = makeEditScene();
    const double dx = incr->voxelSize();
    Forces forces;
    forces.setAdvection(Vec3SGrid::create(Vec3s(0.5f, 0.25f, 0.0f)), 1.0f);
    forces.setCurvatureWeight(0.2f);
    Evolver ei(*incr, forces), ef(*full, forces);
    ef.setFullRedistance(true);
    constexpr double kChunk = 1.0;
    constexpr int kChunks = 6;
    for (int c = 0; c < kChunks; ++c) {
        SCOPED_TRACE("chunk " + std::to_string(c));
        if (c == 2 || c == 4) {
            // a relabel after the first edit, back again after the second
            const int32_t from = c == 2 ? 3 : 5, to = c == 2 ? 5 : 3;
            editExternally(*incr, from, to);
            editExternally(*full, from, to);
            ei.markExternallyModified();
            ef.markExternallyModified();
        }
        const double t0 = c * kChunk;
        EXPECT_GT(ei.advect(t0, t0 + kChunk), size_t(0));
        ef.advect(t0, t0 + kChunk);
        std::string msg;
        EXPECT_TRUE(incr->validate(&msg)) << msg;
        Index64 overlaps = 0, badCover = 0;
        partitionErrors(*incr, overlaps, badCover);
        EXPECT_EQ(overlaps, Index64(0));
        EXPECT_EQ(badCover, Index64(0));
        EXPECT_EQ(treeDifferences(full->labels()->tree(), incr->labels()->tree()), Index64(0));
        ASSERT_TRUE(full->phi()->tree().hasSameTopology(incr->phi()->tree()));
        double maxDiff = 0.0;
        tree::ValueAccessor<const FloatTree> acc(incr->phi()->tree());
        for (auto it = full->phi()->tree().cbeginValueOn(); it; ++it) {
            maxDiff = std::max(maxDiff, std::abs(double(*it) - double(acc.getValue(it.getCoord()))));
        }
        EXPECT_LE(maxDiff, 1e-4 * dx);
        const std::vector<int32_t> expected = (c >= 2 && c < 4)
            ? std::vector<int32_t>{1, 2, 5} : std::vector<int32_t>{1, 2, 3};
        EXPECT_EQ(incr->labelSet(), expected);
    }
}
