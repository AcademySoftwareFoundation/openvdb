// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tools/LevelSetUtil.h> // for sdfInteriorMask()
#include <openvdb/tools/ParticlesToLevelSet.h>

#include <gtest/gtest.h>

#include <vector>


#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/0.0);


class TestParticlesToLevelSet: public ::testing::Test
{
public:
    void SetUp() override {openvdb::initialize();}
    void TearDown() override {openvdb::uninitialize();}

    void writeGrid(openvdb::GridBase::Ptr grid, std::string fileName) const
    {
        std::cout << "\nWriting \""<<fileName<<"\" to file\n";
        grid->setName("TestParticlesToLevelSet");
        openvdb::GridPtrVec grids;
        grids.push_back(grid);
        openvdb::io::File file(fileName + ".vdb");
        file.write(grids);
        file.close();
    }
};


class MyParticleList
{
protected:
    struct MyParticle {
        openvdb::Vec3R p, v;
        openvdb::Real  r;
    };
    openvdb::Real           mRadiusScale;
    openvdb::Real           mVelocityScale;
    std::vector<MyParticle> mParticleList;
public:

    typedef openvdb::Vec3R  PosType;

    MyParticleList(openvdb::Real rScale=1, openvdb::Real vScale=1)
        : mRadiusScale(rScale), mVelocityScale(vScale) {}
    void add(const openvdb::Vec3R &p, const openvdb::Real &r,
             const openvdb::Vec3R &v=openvdb::Vec3R(0,0,0))
    {
        MyParticle pa;
        pa.p = p;
        pa.r = r;
        pa.v = v;
        mParticleList.push_back(pa);
    }
    /// @return coordinate bbox in the space of the specified transfrom
    openvdb::CoordBBox getBBox(const openvdb::GridBase& grid) {
        openvdb::CoordBBox bbox;
        openvdb::Coord &min= bbox.min(), &max = bbox.max();
        openvdb::Vec3R pos;
        openvdb::Real rad, invDx = 1/grid.voxelSize()[0];
        for (size_t n=0, e=this->size(); n<e; ++n) {
            this->getPosRad(n, pos, rad);
            const openvdb::Vec3d xyz = grid.worldToIndex(pos);
            const openvdb::Real   r  = rad * invDx;
            for (int i=0; i<3; ++i) {
                min[i] = openvdb::math::Min(min[i], openvdb::math::Floor(xyz[i] - r));
                max[i] = openvdb::math::Max(max[i], openvdb::math::Ceil( xyz[i] + r));
            }
        }
        return bbox;
    }
    //typedef int AttributeType;
    // The methods below are only required for the unit-tests
    openvdb::Vec3R pos(int n)   const {return mParticleList[n].p;}
    openvdb::Vec3R vel(int n)   const {return mVelocityScale*mParticleList[n].v;}
    openvdb::Real radius(int n) const {return mRadiusScale*mParticleList[n].r;}

    //////////////////////////////////////////////////////////////////////////////
    /// The methods below are the only ones required by tools::ParticleToLevelSet
    /// @note We return by value since the radius and velocities are modified
    /// by the scaling factors! Also these methods are all assumed to
    /// be thread-safe.

    /// Return the total number of particles in list.
    ///  Always required!
    size_t size() const { return mParticleList.size(); }

    /// Get the world space position of n'th particle.
    /// Required by ParticledToLevelSet::rasterizeSphere(*this,radius).
    void getPos(size_t n,  openvdb::Vec3R&pos) const { pos = mParticleList[n].p; }


    void getPosRad(size_t n,  openvdb::Vec3R& pos, openvdb::Real& rad) const {
        pos = mParticleList[n].p;
        rad = mRadiusScale*mParticleList[n].r;
    }
    void getPosRadVel(size_t n,  openvdb::Vec3R& pos, openvdb::Real& rad, openvdb::Vec3R& vel) const {
        pos = mParticleList[n].p;
        rad = mRadiusScale*mParticleList[n].r;
        vel = mVelocityScale*mParticleList[n].v;
    }
    // The method below is only required for attribute transfer
    void getAtt(size_t n, openvdb::Index32& att) const { att = openvdb::Index32(n); }
};


TEST_F(TestParticlesToLevelSet, testBlindData)
{
    using BlindTypeIF = openvdb::tools::p2ls_internal::BlindData<openvdb::Index, float>;

    BlindTypeIF value(openvdb::Index(8), 5.2f);
    EXPECT_EQ(openvdb::Index(8), value.visible());
    ASSERT_DOUBLES_EXACTLY_EQUAL(5.2f, value.blind());

    BlindTypeIF value2(openvdb::Index(13), 1.6f);

    { // test equality
        // only visible portion needs to be equal
        BlindTypeIF blind(openvdb::Index(13), 6.7f);
        EXPECT_TRUE(value2 == blind);
    }

    { // test addition of two blind types
        BlindTypeIF blind = value + value2;
        EXPECT_EQ(openvdb::Index(8+13), blind.visible());
        EXPECT_EQ(0.0f, blind.blind()); // blind values are both dropped
    }

    { // test addition of blind type with visible type
        BlindTypeIF blind = value + 3;
        EXPECT_EQ(openvdb::Index(8+3), blind.visible());
        EXPECT_EQ(5.2f, blind.blind());
    }

    { // test addition of blind type with type that requires casting
        // note that this will generate conversion warnings if not handled properly
        BlindTypeIF blind = value + 3.7;
        EXPECT_EQ(openvdb::Index(8+3), blind.visible());
        EXPECT_EQ(5.2f, blind.blind());
    }
}


TEST_F(TestParticlesToLevelSet, testMyParticleList)
{
    MyParticleList pa;
    EXPECT_EQ(0, int(pa.size()));
    pa.add(openvdb::Vec3R(10,10,10), 2, openvdb::Vec3R(1,0,0));
    EXPECT_EQ(1, int(pa.size()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, pa.pos(0)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, pa.pos(0)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(10, pa.pos(0)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(1 , pa.vel(0)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(0)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(0)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(2 , pa.radius(0));
    pa.add(openvdb::Vec3R(20,20,20), 3);
    EXPECT_EQ(2, int(pa.size()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(20, pa.pos(1)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(20, pa.pos(1)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(20, pa.pos(1)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(1)[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(1)[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(0 , pa.vel(1)[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL(3 , pa.radius(1));

    const float voxelSize = 0.5f, halfWidth = 4.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
    openvdb::CoordBBox bbox = pa.getBBox(*ls);
    ASSERT_DOUBLES_EXACTLY_EQUAL((10-2)/voxelSize, bbox.min()[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((10-2)/voxelSize, bbox.min()[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((10-2)/voxelSize, bbox.min()[2]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((20+3)/voxelSize, bbox.max()[0]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((20+3)/voxelSize, bbox.max()[1]);
    ASSERT_DOUBLES_EXACTLY_EQUAL((20+3)/voxelSize, bbox.max()[2]);
}


TEST_F(TestParticlesToLevelSet, testRasterizeSpheres)
{
    MyParticleList pa;
    pa.add(openvdb::Vec3R(10,10,10), 2);
    pa.add(openvdb::Vec3R(20,20,20), 2);
    // testing CSG
    pa.add(openvdb::Vec3R(31.0,31,31), 5);
    pa.add(openvdb::Vec3R(31.5,31,31), 5);
    pa.add(openvdb::Vec3R(32.0,31,31), 5);
    pa.add(openvdb::Vec3R(32.5,31,31), 5);
    pa.add(openvdb::Vec3R(33.0,31,31), 5);
    pa.add(openvdb::Vec3R(33.5,31,31), 5);
    pa.add(openvdb::Vec3R(34.0,31,31), 5);
    pa.add(openvdb::Vec3R(34.5,31,31), 5);
    pa.add(openvdb::Vec3R(35.0,31,31), 5);
    pa.add(openvdb::Vec3R(35.5,31,31), 5);
    pa.add(openvdb::Vec3R(36.0,31,31), 5);
    EXPECT_EQ(13, int(pa.size()));

    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster(*ls);

    raster.setGrainSize(1);//a value of zero disables threading
    raster.rasterizeSpheres(pa);
    raster.finalize();
    //openvdb::FloatGrid::Ptr ls = raster.getSdfGrid();

    //ls->tree().print(std::cout,4);
    //this->writeGrid(ls, "testRasterizeSpheres");

    ASSERT_DOUBLES_EXACTLY_EQUAL(halfWidth * voxelSize,
        ls->tree().getValue(openvdb::Coord( 0, 0, 0)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord( 6,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord( 7,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord( 8,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord( 9,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(10,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(11,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(12,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(13,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(14,10,10)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,16,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,17,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,18,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,19,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(20,20,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,21,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,22,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,23,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,24,20)));
    {// full but slow test of all voxels
        openvdb::CoordBBox bbox = pa.getBBox(*ls);
        bbox.expand(static_cast<int>(halfWidth)+1);
        openvdb::Index64 count=0;
        const float outside = ls->background(), inside = -outside;
        const openvdb::Coord &min=bbox.min(), &max=bbox.max();
        for (openvdb::Coord ijk=min; ijk[0]<max[0]; ++ijk[0]) {
            for (ijk[1]=min[1]; ijk[1]<max[1]; ++ijk[1]) {
                for (ijk[2]=min[2]; ijk[2]<max[2]; ++ijk[2]) {
                    const openvdb::Vec3d xyz = ls->indexToWorld(ijk.asVec3d());
                    double dist = (xyz-pa.pos(0)).length()-pa.radius(0);
                    for (int i = 1, s = int(pa.size()); i < s; ++i) {
                        dist=openvdb::math::Min(dist,(xyz-pa.pos(i)).length()-pa.radius(i));
                    }
                    const float val = ls->tree().getValue(ijk);
                    if (dist >= outside) {
                        EXPECT_NEAR(outside, val, 0.0001);
                        EXPECT_TRUE(ls->tree().isValueOff(ijk));
                    } else if( dist <= inside ) {
                        EXPECT_NEAR(inside, val, 0.0001);
                        EXPECT_TRUE(ls->tree().isValueOff(ijk));
                    } else {
                        EXPECT_NEAR(  dist, val, 0.0001);
                        EXPECT_TRUE(ls->tree().isValueOn(ijk));
                        ++count;
                    }
                }
            }
        }
        //std::cerr << "\nExpected active voxel count = " << count
        //    << ", actual active voxle count = "
        //    << ls->activeVoxelCount() << std::endl;
        EXPECT_EQ(count, ls->activeVoxelCount());
    }
}


TEST_F(TestParticlesToLevelSet, testRasterizeSpheresAndId)
{
    MyParticleList pa(0.5f);
    pa.add(openvdb::Vec3R(10,10,10), 4);
    pa.add(openvdb::Vec3R(20,20,20), 4);
    // testing CSG
    pa.add(openvdb::Vec3R(31.0,31,31),10);
    pa.add(openvdb::Vec3R(31.5,31,31),10);
    pa.add(openvdb::Vec3R(32.0,31,31),10);
    pa.add(openvdb::Vec3R(32.5,31,31),10);
    pa.add(openvdb::Vec3R(33.0,31,31),10);
    pa.add(openvdb::Vec3R(33.5,31,31),10);
    pa.add(openvdb::Vec3R(34.0,31,31),10);
    pa.add(openvdb::Vec3R(34.5,31,31),10);
    pa.add(openvdb::Vec3R(35.0,31,31),10);
    pa.add(openvdb::Vec3R(35.5,31,31),10);
    pa.add(openvdb::Vec3R(36.0,31,31),10);
    EXPECT_EQ(13, int(pa.size()));

    typedef openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32> RasterT;
    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);

    RasterT raster(*ls);
    raster.setGrainSize(1);//a value of zero disables threading
    raster.rasterizeSpheres(pa);
    raster.finalize();
    const RasterT::AttGridType::Ptr id = raster.attributeGrid();

    int minVal = std::numeric_limits<int>::max(), maxVal = -minVal;
    for (RasterT::AttGridType::ValueOnCIter i=id->cbeginValueOn(); i; ++i) {
        minVal = openvdb::math::Min(minVal, int(*i));
        maxVal = openvdb::math::Max(maxVal, int(*i));
    }
    EXPECT_EQ(0 , minVal);
    EXPECT_EQ(12, maxVal);

    //grid.tree().print(std::cout,4);
    //id->print(std::cout,4);
    //this->writeGrid(ls, "testRasterizeSpheres");

    ASSERT_DOUBLES_EXACTLY_EQUAL(halfWidth * voxelSize,
                                 ls->tree().getValue(openvdb::Coord( 0, 0, 0)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord( 6,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord( 7,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord( 8,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord( 9,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(10,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(11,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(12,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(13,10,10)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(14,10,10)));

    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,16,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,17,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,18,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,19,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-2, ls->tree().getValue(openvdb::Coord(20,20,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(-1, ls->tree().getValue(openvdb::Coord(20,21,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 0, ls->tree().getValue(openvdb::Coord(20,22,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 1, ls->tree().getValue(openvdb::Coord(20,23,20)));
    ASSERT_DOUBLES_EXACTLY_EQUAL( 2, ls->tree().getValue(openvdb::Coord(20,24,20)));

    {// full but slow test of all voxels
        openvdb::CoordBBox bbox = pa.getBBox(*ls);
        bbox.expand(static_cast<int>(halfWidth)+1);
        openvdb::Index64 count = 0;
        const float outside = ls->background(), inside = -outside;
        const openvdb::Coord &min=bbox.min(), &max=bbox.max();
        for (openvdb::Coord ijk=min; ijk[0]<max[0]; ++ijk[0]) {
            for (ijk[1]=min[1]; ijk[1]<max[1]; ++ijk[1]) {
                for (ijk[2]=min[2]; ijk[2]<max[2]; ++ijk[2]) {
                    const openvdb::Vec3d xyz = ls->indexToWorld(ijk.asVec3d());
                    double dist = (xyz-pa.pos(0)).length()-pa.radius(0);
                    openvdb::Index32 k =0;
                    for (int i = 1, s = int(pa.size()); i < s; ++i) {
                        double d = (xyz-pa.pos(i)).length()-pa.radius(i);
                        if (d<dist) {
                            k = openvdb::Index32(i);
                            dist = d;
                        }
                    }//loop over particles
                    const float val = ls->tree().getValue(ijk);
                    openvdb::Index32 m = id->tree().getValue(ijk);
                    if (dist >= outside) {
                        EXPECT_NEAR(outside, val, 0.0001);
                        EXPECT_TRUE(ls->tree().isValueOff(ijk));
                        //EXPECT_EQ(openvdb::util::INVALID_IDX, m);
                        EXPECT_TRUE(id->tree().isValueOff(ijk));
                    } else if( dist <= inside ) {
                        EXPECT_NEAR(inside, val, 0.0001);
                        EXPECT_TRUE(ls->tree().isValueOff(ijk));
                        //EXPECT_EQ(openvdb::util::INVALID_IDX, m);
                        EXPECT_TRUE(id->tree().isValueOff(ijk));
                    } else {
                        EXPECT_NEAR(  dist, val, 0.0001);
                        EXPECT_TRUE(ls->tree().isValueOn(ijk));
                        EXPECT_EQ(k, m);
                        EXPECT_TRUE(id->tree().isValueOn(ijk));
                        ++count;
                    }
                }
            }
        }
        //std::cerr << "\nExpected active voxel count = " << count
        //    << ", actual active voxle count = "
        //    << ls->activeVoxelCount() << std::endl;
        EXPECT_EQ(count, ls->activeVoxelCount());
    }
}


/// This is not really a conventional unit-test since the result of
/// the tests are written to a file and need to be visually verified!
TEST_F(TestParticlesToLevelSet, testRasterizeTrails)
{
    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);

    MyParticleList pa(1,5);

    // This particle radius = 1 < 1.5 i.e. it's below the Nyquist frequency and hence ignored
    pa.add(openvdb::Vec3R(  0,  0,  0), 1, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(-10,-10,-10), 2, openvdb::Vec3R( 2, 0, 0));
    pa.add(openvdb::Vec3R( 10, 10, 10), 3, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(  0,  0,  0), 6, openvdb::Vec3R( 0, 0,-5));
    pa.add(openvdb::Vec3R( 20,  0,  0), 2, openvdb::Vec3R( 0, 0, 0));

    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster(*ls);
    raster.rasterizeTrails(pa, 0.75);//scale offset between two instances

    //ls->tree().print(std::cout, 4);
    //this->writeGrid(ls, "testRasterizeTrails");
}


TEST_F(TestParticlesToLevelSet, testRasterizeTrailsAndId)
{
    MyParticleList pa(1,5);

    // This particle radius = 1 < 1.5 i.e. it's below the Nyquist frequency and hence ignored
    pa.add(openvdb::Vec3R(  0,  0,  0), 1, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(-10,-10,-10), 2, openvdb::Vec3R( 2, 0, 0));
    pa.add(openvdb::Vec3R( 10, 10, 10), 3, openvdb::Vec3R( 0, 1, 0));
    pa.add(openvdb::Vec3R(  0,  0,  0), 6, openvdb::Vec3R( 0, 0,-5));

    typedef openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index> RasterT;
    const float voxelSize = 1.0f, halfWidth = 2.0f;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
    RasterT raster(*ls);
    raster.rasterizeTrails(pa, 0.75);//scale offset between two instances
    raster.finalize();
    const RasterT::AttGridType::Ptr id = raster.attributeGrid();
    EXPECT_TRUE(!ls->empty());
    EXPECT_TRUE(!id->empty());
    EXPECT_EQ(ls->activeVoxelCount(),id->activeVoxelCount());

    int min = std::numeric_limits<int>::max(), max = -min;
    for (RasterT::AttGridType::ValueOnCIter i=id->cbeginValueOn(); i; ++i) {
        min = openvdb::math::Min(min, int(*i));
        max = openvdb::math::Max(max, int(*i));
    }
    EXPECT_EQ(1, min);//first particle is ignored because of its small rdadius!
    EXPECT_EQ(3, max);

    //ls->tree().print(std::cout, 4);
    //this->writeGrid(ls, "testRasterizeTrails");
}


TEST_F(TestParticlesToLevelSet, testMaskOutput)
{
    using namespace openvdb;

    using SdfGridType = FloatGrid;
    using MaskGridType = MaskGrid;

    MyParticleList pa;
    const Vec3R vel(10, 5, 1);
    pa.add(Vec3R(84.7252, 85.7946, 84.4266), 11.8569, vel);
    pa.add(Vec3R(47.9977, 81.2169, 47.7665), 5.45313, vel);
    pa.add(Vec3R(87.0087, 14.0351, 95.7155), 7.36483, vel);
    pa.add(Vec3R(75.8616, 53.7373, 58.202),  14.4127, vel);
    pa.add(Vec3R(14.9675, 32.4141, 13.5218), 4.33101, vel);
    pa.add(Vec3R(96.9809, 9.92804, 90.2349), 12.2613, vel);
    pa.add(Vec3R(63.4274, 3.84254, 32.5047), 12.1566, vel);
    pa.add(Vec3R(62.351,  47.4698, 41.4369), 11.637,  vel);
    pa.add(Vec3R(62.2846, 1.35716, 66.2527), 18.9914, vel);
    pa.add(Vec3R(44.1711, 1.99877, 45.1159), 1.11429, vel);

    {
        // Test variable-radius particles.

        // Rasterize into an SDF.
        auto sdf = createLevelSet<SdfGridType>();
        tools::particlesToSdf(pa, *sdf);

        // Rasterize into a boolean mask.
        auto mask = MaskGridType::create();
        tools::particlesToMask(pa, *mask);

        // Verify that the rasterized mask matches the interior of the SDF.
        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        EXPECT_TRUE(interior);
        interior->tree().voxelizeActiveTiles();
        EXPECT_EQ(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        EXPECT_EQ(0, int(interior->activeVoxelCount()));
    }
    {
        // Test fixed-radius particles.

        auto sdf = createLevelSet<SdfGridType>();
        tools::particlesToSdf(pa, *sdf, /*radius=*/10.0);

        auto mask = MaskGridType::create();
        tools::particlesToMask(pa, *mask, /*radius=*/10.0);

        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        EXPECT_TRUE(interior);
        interior->tree().voxelizeActiveTiles();
        EXPECT_EQ(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        EXPECT_EQ(0, int(interior->activeVoxelCount()));
    }
    {
        // Test particle trails.

        auto sdf = createLevelSet<SdfGridType>();
        tools::particleTrailsToSdf(pa, *sdf);

        auto mask = MaskGridType::create();
        tools::particleTrailsToMask(pa, *mask);

        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        EXPECT_TRUE(interior);
        interior->tree().voxelizeActiveTiles();
        EXPECT_EQ(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        EXPECT_EQ(0, int(interior->activeVoxelCount()));
    }
    {
        // Test attribute transfer.

        auto sdf = createLevelSet<SdfGridType>();
        tools::ParticlesToLevelSet<SdfGridType, Index32> p2sdf(*sdf);
        p2sdf.rasterizeSpheres(pa);
        p2sdf.finalize(/*prune=*/true);
        const auto sdfAttr = p2sdf.attributeGrid();
        EXPECT_TRUE(sdfAttr);

        auto mask = MaskGridType::create();
        tools::ParticlesToLevelSet<MaskGridType, Index32> p2mask(*mask);
        p2mask.rasterizeSpheres(pa);
        p2mask.finalize(/*prune=*/true);
        const auto maskAttr = p2mask.attributeGrid();
        EXPECT_TRUE(maskAttr);

        mask->tree().voxelizeActiveTiles();
        auto interior = tools::sdfInteriorMask(*sdf);
        EXPECT_TRUE(interior);
        interior->tree().voxelizeActiveTiles();
        EXPECT_EQ(interior->activeVoxelCount(), mask->activeVoxelCount());
        interior->topologyDifference(*mask);
        EXPECT_EQ(0, int(interior->activeVoxelCount()));

        // Verify that the mask- and SDF-generated attribute grids match.
        auto sdfAcc = sdfAttr->getConstAccessor();
        auto maskAcc = maskAttr->getConstAccessor();
        for (auto it = interior->cbeginValueOn(); it; ++it) {
            const auto& c = it.getCoord();
            EXPECT_EQ(sdfAcc.getValue(c), maskAcc.getValue(c));
        }
    }
}
