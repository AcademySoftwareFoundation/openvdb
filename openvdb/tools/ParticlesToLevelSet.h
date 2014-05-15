///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @author Ken Museth
///
/// @file ParticlesToLevelSet.h
///
/// @brief This tool converts particles (with position, radius and
/// velocity) into a singed distance field encoded as a narrow band
/// level set. Optionally arbitrary attributes on the particles can
/// be transferred resulting in an additional attribute grid with the
/// same topology as the level set grid.
///
/// @note This fast particle to level set converter is always intended
/// to be combined with some kind of surface post processing,
/// i.e. tools::Filter. Without such post processing the generated
/// surface is typically too noisy and blooby. However it serves as a
/// great and fast starting point for subsequent level set surface
/// processing and convolution.
///
/// The @c ParticleListT template argument below refers to any class
/// with the following interface (see unittest/TestParticlesToLevelSet.cc
/// and SOP_DW_OpenVDBParticleVoxelizer for practical examples):
/// @code
///
/// class ParticleList {
///   ...
/// public:
///
///   // Return the total number of particles in list.
///   // Always required!
///   size_t         size()          const;
///
///   // Get the world space position of n'th particle.
///   // Required by ParticledToLevelSet::rasterizeSphere(*this,radius).
///   void getPos(size_t n, Vec3R& xyz) const;
///
///   // Get the world space position and radius of n'th particle.
///   // Required by ParticledToLevelSet::rasterizeSphere(*this).
///   void getPosRad(size_t n, Vec3R& xyz, Real& rad) const;
///
///   // Get the world space position, radius and velocity of n'th particle.
///   // Required by ParticledToLevelSet::rasterizeSphere(*this,radius).
///   void getPosRadVel(size_t n, Vec3R& xyz, Real& rad, Vec3R& vel) const;
///
///   // Get the attribute of the n'th particle. AttributeType is user-defined!
///   // Only required if attribute transfer is enabled in ParticledToLevelSet.
///   void getAtt(AttributeType& att) const;
/// };
/// @endcode
///
/// @note See unittest/TestParticlesToLevelSet.cc for an example.
///
/// The @c InterruptT template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = NULL)// called when computations begin
///   void end()                         // called when computations end
///   bool wasInterrupted(int percent=-1)// return true to break computation
/// };
/// @endcode
///
/// @note If no template argument is provided for this InterruptT
/// the util::NullInterrupter is used which implies that all
/// interrupter calls are no-ops (i.e. incurs no computational overhead).

#ifndef OPENVDB_TOOLS_PARTICLES_TO_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_PARTICLES_TO_LEVELSET_HAS_BEEN_INCLUDED

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/if.hpp>
#include <openvdb/util/Util.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Transform.h>
#include <openvdb/util/NullInterrupter.h>
#include "Composite.h" // for csgUnion()

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


// This is a simple type that combines a distance value and a particle
// attribute. It's required for attribute transfer which is performed
// in the ParticlesToLevelSet::Raster memberclass defined below.
namespace local {template <typename VisibleT, typename BlindT> class BlindData;}

template<typename SdfGridT,
         typename AttributeT = void,
         typename InterrupterT = util::NullInterrupter>
class ParticlesToLevelSet
{
public:

    typedef typename boost::is_void<AttributeT>::type DisableT;
    typedef InterrupterT                          InterrupterType;

    typedef SdfGridT                              SdfGridType;
    typedef typename SdfGridT::ValueType          SdfType;

    typedef typename boost::mpl::if_<DisableT, size_t, AttributeT>::type  AttType;
    typedef typename SdfGridT::template ValueConverter<AttType>::Type AttGridType;

    BOOST_STATIC_ASSERT(boost::is_floating_point<SdfType>::value);

    /// @brief Constructor using an exiting signed distance,
    /// i.e. narrow band level set, grid.
    ///
    /// @param grid      Level set grid in which particles are rasterized
    /// @param interrupt Callback to interrupt a long-running process
    ///
    /// @note The input grid is assumed to be a valid level set and if
    /// it already contains voxels (with SDF values) partices are unioned
    /// onto the exisinting level set surface. However, if attribute tranfer
    /// is enabled, i.e. AttributeT != void, attributes are only
    /// generated for voxels that overlap with particles, not the existing
    /// voxels in the input grid (for which no attributes exist!).
    ///
    /// @details The width in voxel units of the generated narrow band level set is
    /// given by 2*background/dx, where background is the background value
    /// stored in the grid, and dx is the voxel size derived from the
    /// transform also stored in the grid. Also note that -background
    /// corresponds to the constant value inside the generated narrow
    /// band level sets. Finally the default NullInterrupter should
    /// compile out interruption checks during optimization, thus
    /// incurring no run-time overhead.
    explicit ParticlesToLevelSet(SdfGridT& grid, InterrupterT* interrupt = NULL);

    /// Destructor
    ~ParticlesToLevelSet() { delete mBlindGrid; }

    /// @brief This methods syncs up the level set and attribute grids
    /// and therefore needs to be called before any of these grids are
    /// used and after the last call to any of the rasterizer methods.
    ///
    /// @note Avoid calling this method more then once and only after
    /// all the particles have been rasterized. It has no effect if
    /// attribute transfer is disabled, i.e. AttributeT = void.
    void finalize();

    /// @brief Return a shared pointer to the grid containing the
    /// (optional) attribute.
    ///
    /// @warning If attribute transfer was disabled, i.e. AttributeT =
    /// void, or finalize() was not called the pointer is NULL!
    typename AttGridType::Ptr attributeGrid() { return mAttGrid; }

    /// @brief Return the size of a voxel in world units
    Real getVoxelSize() const { return mDx; }

    /// @brief Return the half-width of the narrow band in voxel units
    Real getHalfWidth() const { return mHalfWidth; }

    /// @brief Return the smallest radius allowed in voxel units
    Real getRmin() const { return mRmin; }
    /// @brief Return the largest radius allowed in voxel units
    Real getRmax() const { return mRmax; }

    /// @brief Return true if any particles were ignored due to their size
    bool ignoredParticles() const { return mMinCount>0 || mMaxCount>0; }
    /// @brief Return number of small particles that were ignore due to Rmin
    size_t getMinCount() const { return mMinCount; }
    /// @brief Return number of large particles that were ignore due to Rmax
    size_t getMaxCount() const { return mMaxCount; }

    /// @brief set the smallest radius allowed in voxel units
    void setRmin(Real Rmin) { mRmin = math::Max(Real(0),Rmin); }
    /// @brief set the largest radius allowed in voxel units
    void setRmax(Real Rmax) { mRmax = math::Max(mRmin,Rmax); }

    /// @brief Rreturn the grain-size used for multi-threading
    int  getGrainSize() const { return mGrainSize; }
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(int grainSize) { mGrainSize = grainSize; }

    /// @brief Rasterize a sphere per particle derived from their
    /// position and radius. All spheres are CSG unioned.
    ///
    /// @param pa Particles with position and radius.
    template <typename ParticleListT>
    void rasterizeSpheres(const ParticleListT& pa);

    /// @brief Rasterize a sphere per particle derived from their
    /// position and constant radius. All spheres are CSG unioned.
    ///
    /// @param pa Particles with position.
    /// @param radius Constant particle radius in world units.
    template <typename ParticleListT>
    void rasterizeSpheres(const ParticleListT& pa, Real radius);

    /// @brief Rasterize a trail per particle derived from their
    /// position, radius and velocity. Each trail is generated
    /// as CSG unions of sphere instances with decreasing radius.
    ///
    /// @param pa particles with position, radius and velocity.
    /// @param delta controls distance between sphere instances
    /// (default=1). Be careful not to use too small values since this
    /// can lead to excessive computation per trail (which the
    /// interrupter can't stop).
    ///
    /// @note The direction of a trail is inverse to the direction of
    /// the velocity vector, and the length is given by |V|. The radius
    /// at the head of the trail is given by the radius of the particle
    /// and the radius at the tail of the trail is Rmin voxel units which
    /// has a default value of 1.5 corresponding to the Nyquist
    /// frequency!
    template <typename ParticleListT>
    void rasterizeTrails(const ParticleListT& pa, Real delta=1.0);

private:

    typedef local::BlindData<SdfType, AttType>    BlindType;
    typedef typename SdfGridT::template ValueConverter<BlindType>::Type BlindGridType;

    /// Class with multi-threaded implementation of particle rasterization
    template<typename ParticleListT, typename GridT> struct Raster;

    SdfGridType*   mSdfGrid;
    typename AttGridType::Ptr   mAttGrid;
    BlindGridType* mBlindGrid;
    InterrupterT*  mInterrupter;
    Real           mDx, mHalfWidth;
    Real           mRmin, mRmax;//ignore particles outside this range of radii in voxel
    size_t         mMinCount, mMaxCount;//counters for ignored particles!
    int            mGrainSize;

};//end of ParticlesToLevelSet class

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
inline ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
ParticlesToLevelSet(SdfGridT& grid, InterrupterT* interrupter) :
    mSdfGrid(&grid),
    mBlindGrid(NULL),
    mInterrupter(interrupter),
    mDx(grid.voxelSize()[0]),
    mHalfWidth(grid.background()/mDx),
    mRmin(1.5),// corresponds to the Nyquist grid sampling frequency
    mRmax(100.0),// corresponds to a huge particle (probably too large!)
    mMinCount(0),
    mMaxCount(0),
    mGrainSize(1)
{
    if (!mSdfGrid->hasUniformVoxels() ) {
        OPENVDB_THROW(RuntimeError,
                      "ParticlesToLevelSet only supports uniform voxels!");
    }
    if (mSdfGrid->getGridClass() != GRID_LEVEL_SET) {
        OPENVDB_THROW(RuntimeError,
                      "ParticlesToLevelSet only supports level sets!"
                      "\nUse Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
    }

    if (!DisableT::value) {
        mBlindGrid = new BlindGridType(BlindType(grid.background()));
        mBlindGrid->setTransform(mSdfGrid->transform().copy());
    }
}

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template <typename ParticleListT>
inline void ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
rasterizeSpheres(const ParticleListT& pa)
{
    if (DisableT::value) {
        Raster<ParticleListT, SdfGridT> r(*this, mSdfGrid, pa);
        r.rasterizeSpheres();
    } else {
        Raster<ParticleListT, BlindGridType> r(*this, mBlindGrid, pa);
        r.rasterizeSpheres();
    }
}

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template <typename ParticleListT>
inline void ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
rasterizeSpheres(const ParticleListT& pa, Real radius)
{
    if (DisableT::value) {
        Raster<ParticleListT, SdfGridT> r(*this, mSdfGrid, pa);
        r.rasterizeSpheres(radius/mDx);
    } else {
        Raster<ParticleListT, BlindGridType> r(*this, mBlindGrid, pa);
        r.rasterizeSpheres(radius/mDx);
    }
}

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template <typename ParticleListT>
inline void ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::
rasterizeTrails(const ParticleListT& pa, Real delta)
{
    if (DisableT::value) {
        Raster<ParticleListT, SdfGridT> r(*this, mSdfGrid, pa);
        r.rasterizeTrails(delta);
    } else {
        Raster<ParticleListT, BlindGridType> r(*this, mBlindGrid, pa);
        r.rasterizeTrails(delta);
    }
}

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
inline void
ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::finalize()
{
    if (mBlindGrid==NULL) return;

    typedef typename SdfGridType::TreeType   SdfTreeT;
    typedef typename AttGridType::TreeType   AttTreeT;
    typedef typename BlindGridType::TreeType BlindTreeT;
    // Use topology copy constructors since output grids have the same topology as mBlindDataGrid
    const BlindTreeT& tree = mBlindGrid->tree();

    // New level set tree
    typename SdfTreeT::Ptr sdfTree(new SdfTreeT(
        tree, tree.background().visible(), openvdb::TopologyCopy()));

    // Note this overwrites any existing attribute grids!
    typename AttTreeT::Ptr attTree(new AttTreeT(
        tree, tree.background().blind(), openvdb::TopologyCopy()));
    mAttGrid = typename AttGridType::Ptr(new AttGridType(attTree));
    mAttGrid->setTransform(mBlindGrid->transform().copy());

    // Extract the level set and IDs from mBlindDataGrid. We will
    // explore the fact that by design active values always live
    // at the leaf node level, i.e. no active tiles exist in level sets
    typedef typename BlindTreeT::LeafCIter    LeafIterT;
    typedef typename BlindTreeT::LeafNodeType LeafT;
    typedef typename SdfTreeT::LeafNodeType   SdfLeafT;
    typedef typename AttTreeT::LeafNodeType   AttLeafT;
    for (LeafIterT n = tree.cbeginLeaf(); n; ++n) {
        const LeafT& leaf = *n;
        const openvdb::Coord xyz = leaf.origin();
        // Get leafnodes that were allocated during topology contruction!
        SdfLeafT* sdfLeaf = sdfTree->probeLeaf(xyz);
        AttLeafT* attLeaf = attTree->probeLeaf(xyz);
        for (typename LeafT::ValueOnCIter m=leaf.cbeginValueOn(); m; ++m) {
            // Use linear offset (vs coordinate) access for better performance!
            const openvdb::Index k = m.pos();
            const BlindType& v = *m;
            sdfLeaf->setValueOnly(k, v.visible());
            attLeaf->setValueOnly(k, v.blind());
        }
    }
    sdfTree->signedFloodFill();//required since we only transferred active voxels!

    if (mSdfGrid->empty()) {
        mSdfGrid->setTree(sdfTree);
    } else {
        tools::csgUnion(mSdfGrid->tree(), *sdfTree, /*prune=*/true);
    }
}

///////////////////////////////////////////////////////////

template<typename SdfGridT, typename AttributeT, typename InterrupterT>
template<typename ParticleListT, typename GridT>
struct ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT>::Raster
{
    typedef typename boost::is_void<AttributeT>::type DisableT;
    typedef ParticlesToLevelSet<SdfGridT, AttributeT, InterrupterT> ParticlesToLevelSetT;
    typedef typename ParticlesToLevelSetT::SdfType   SdfT;//type of signed distance values
    typedef typename ParticlesToLevelSetT::AttType   AttT;//type of particle attribute
    typedef typename GridT::ValueType                ValueT;
    typedef typename GridT::Accessor                 AccessorT;

    /// @brief Main constructor
    Raster(ParticlesToLevelSetT& parent, GridT* grid, const ParticleListT& particles)
        : mParent(parent),
          mParticles(particles),
          mGrid(grid),
          mMap(*(mGrid->transform().baseMap())),
          mMinCount(0),
          mMaxCount(0),
          mOwnsGrid(false)
    {
    }

    /// @brief Copy constructor called by tbb threads
    Raster(Raster& other, tbb::split)
        : mParent(other.mParent),
          mParticles(other.mParticles),
          mGrid(new GridT(*other.mGrid, openvdb::ShallowCopy())),
          mMap(other.mMap),
          mMinCount(0),
          mMaxCount(0),
          mTask(other.mTask),
          mOwnsGrid(true)
    {
        mGrid->newTree();
    }

    virtual ~Raster() { if (mOwnsGrid) delete mGrid; }

    /// @brief Rasterize a sphere per particle derived from their
    /// position and radius. All spheres are CSG unioned.
    void rasterizeSpheres()
    {
        mMinCount = mMaxCount = 0;
        if (mParent.mInterrupter) {
            mParent.mInterrupter->start("Rasterizing particles to level set using spheres");
        }
        mTask = boost::bind(&Raster::rasterSpheres, _1, _2);
        this->cook();
        if (mParent.mInterrupter) mParent.mInterrupter->end();
    }
    /// @brief Rasterize a sphere per particle derived from their
    /// position and constant radius. All spheres are CSG unioned.
    /// @param radius constant radius of all particles in voxel units.
    void rasterizeSpheres(Real radius)
    {
        mMinCount = radius < mParent.mRmin ? mParticles.size() : 0;
        mMaxCount = radius > mParent.mRmax ? mParticles.size() : 0;
        if (mMinCount>0 || mMaxCount>0) {//skipping all particles!
            mParent.mMinCount = mMinCount;
            mParent.mMaxCount = mMaxCount;
        } else {
            if (mParent.mInterrupter) {
                mParent.mInterrupter->start(
                    "Rasterizing particles to level set using const spheres");
            }
            mTask = boost::bind(&Raster::rasterFixedSpheres, _1, _2, SdfT(radius));
            this->cook();
            if (mParent.mInterrupter) mParent.mInterrupter->end();
        }
    }
    /// @brief Rasterize a trail per particle derived from their
    /// position, radius and velocity. Each trail is generated
    /// as CSG unions of sphere instances with decreasing radius.
    ///
    /// @param delta controls distance between sphere instances
    /// (default=1). Be careful not to use too small values since this
    /// can lead to excessive computation per trail (which the
    /// interrupter can't stop).
    ///
    /// @note The direction of a trail is inverse to the direction of
    /// the velocity vector, and the length is given by |V|. The radius
    /// at the head of the trail is given by the radius of the particle
    /// and the radius at the tail of the trail is Rmin voxel units which
    /// has a default value of 1.5 corresponding to the Nyquist frequency!
    void rasterizeTrails(Real delta=1.0)
    {
        mMinCount = mMaxCount = 0;
        if (mParent.mInterrupter) {
            mParent.mInterrupter->start("Rasterizing particles to level set using trails");
        }
        mTask = boost::bind(&Raster::rasterTrails, _1, _2, SdfT(delta));
        this->cook();
        if (mParent.mInterrupter) mParent.mInterrupter->end();
    }

    /// @brief Kicks off the optionally multithreaded computation
    void operator()(const tbb::blocked_range<size_t>& r)
    {
        assert(mTask);
        mTask(this, r);
        mParent.mMinCount = mMinCount;
        mParent.mMaxCount = mMaxCount;
    }

    /// @brief Reguired by tbb::parallel_reduce
    void join(Raster& other)
    {
        tools::csgUnion(*mGrid, *other.mGrid, /*prune=*/true);
        mMinCount += other.mMinCount;
        mMaxCount += other.mMaxCount;
    }
private:
    /// Disallow assignment since some of the members are references
    Raster& operator=(const Raster& other) { return *this; }

    /// @return true if the particle is too small or too large
    bool ignoreParticle(SdfT R)
    {
        if (R < mParent.mRmin) {// below the cutoff radius
            ++mMinCount;
            return true;
        }
        if (R > mParent.mRmax) {// above the cutoff radius
            ++mMaxCount;
            return true;
        }
        return false;
    }
    /// @brief Reguired by tbb::parallel_reduce to multithreaded
    /// rasterization of particles as spheres with variable radius
    ///
    /// @param r tbb's default range referring to the list of particles
    void rasterSpheres(const tbb::blocked_range<size_t>& r)
    {
        AccessorT acc = mGrid->getAccessor(); // local accessor
        bool run = true;
        const SdfT invDx = 1/mParent.mDx;
        AttT att;
        Vec3R pos;
        Real rad;
        for (Index32 id = r.begin(), e=r.end(); run && id != e; ++id) {
            mParticles.getPosRad(id, pos, rad);
            const SdfT R = invDx * rad;// in voxel units
            if (this->ignoreParticle(R)) continue;
            const Vec3R P = mMap.applyInverseMap(pos);
            this->getAtt<DisableT>(id, att);
            run = this->makeSphere(P, R, att, acc);
        }//end loop over particles
    }
    /// @brief Reguired by tbb::parallel_reduce to multithreaded
    /// rasterization of particles as spheres with a fixed radius
    ///
    /// @param r tbb's default range referring to the list of particles
    void rasterFixedSpheres(const tbb::blocked_range<size_t>& r, SdfT R)
    {
        const SdfT dx = mParent.mDx, w = mParent.mHalfWidth;// in voxel units
        AccessorT acc = mGrid->getAccessor(); // local accessor
        const ValueT inside = -mGrid->background();
        const SdfT max = R + w;// maximum distance in voxel units
        const SdfT max2 = math::Pow2(max);//square of maximum distance in voxel units
        const SdfT min2 = math::Pow2(math::Max(SdfT(0), R - w));//square of minimum distance
        ValueT v;
        size_t count = 0;
        AttT att;
        Vec3R pos;
        for (size_t id = r.begin(), e=r.end(); id != e; ++id) {
            this->getAtt<DisableT>(id, att);
            mParticles.getPos(id, pos);
            const Vec3R P = mMap.applyInverseMap(pos);
            const Coord a(math::Floor(P[0]-max),math::Floor(P[1]-max),math::Floor(P[2]-max));
            const Coord b(math::Ceil( P[0]+max),math::Ceil( P[1]+max),math::Ceil( P[2]+max));
            for ( Coord c = a; c.x() <= b.x(); ++c.x() ) {
                //only check interrupter every 32'th scan in x
                if (!(count++ & (1<<5)-1) && util::wasInterrupted(mParent.mInterrupter)) {
                    tbb::task::self().cancel_group_execution();
                    return;
                }
                SdfT x2 = math::Pow2( c.x() - P[0] );
                for ( c.y() = a.y(); c.y() <= b.y(); ++c.y() ) {
                    SdfT x2y2 = x2 + math::Pow2( c.y() - P[1] );
                    for ( c.z() = a.z(); c.z() <= b.z(); ++c.z() ) {
                        SdfT x2y2z2 = x2y2 + math::Pow2(c.z()- P[2]);//square distance from c to P
                        if ( x2y2z2 >= max2 || (!acc.probeValue(c,v) && v<ValueT(0)) )
                            continue;//outside narrow band of particle or inside existing level set
                        if ( x2y2z2 <= min2 ) {//inside narrow band of the particle.
                            acc.setValueOff(c, inside);
                            continue;
                        }
                        // convert signed distance from voxel units to world units
                        const ValueT d=Merge(dx*(math::Sqrt(x2y2z2) - R), att);
                        if (d < v) acc.setValue(c, d);//CSG union
                    }//end loop over z
                }//end loop over y
            }//end loop over x
        }//end loop over particles
    }
    /// @brief Reguired by tbb::parallel_reduce to multithreaded
    /// rasterization of particles as spheres with velocity blurring
    ///
    /// @param r tbb's default range referring to the list of particles
    void rasterTrails(const tbb::blocked_range<size_t>& r, SdfT delta)
    {
        AccessorT acc = mGrid->getAccessor(); // local accessor
        bool run = true;
        AttT att;
        Vec3R pos, vel;
        Real rad;
        const Vec3R origin = mMap.applyInverseMap(Vec3R(0,0,0));
        const SdfT Rmin = mParent.mRmin, invDx = 1/mParent.mDx;
        for (size_t id = r.begin(), e=r.end(); run && id != e; ++id) {
            mParticles.getPosRadVel(id, pos, rad, vel);
            const SdfT R0 = invDx*rad;
            if (this->ignoreParticle(R0)) continue;
            this->getAtt<DisableT>(id, att);
            const Vec3R P0 = mMap.applyInverseMap(pos);
            const Vec3R V  = mMap.applyInverseMap(vel) - origin;//exclude translation
            const SdfT speed = V.length(), inv_speed=1.0/speed;
            const Vec3R N = -V*inv_speed;// inverse normalized direction
            Vec3R P = P0;// local position of instance
            SdfT R = R0, d=0;// local radius and length of trail
            for (size_t m=0; run && d <= speed ; ++m) {
                run = this->makeSphere(P, R, att, acc);
                P += 0.5*delta*R*N;// adaptive offset along inverse velocity direction
                d  = (P-P0).length();// current length of trail
                R  = R0-(R0-Rmin)*d*inv_speed;// R = R0 -> mRmin(e.g. 1.5)
            }//end loop over sphere instances
        }//end loop over particles
    }

    void cook()
    {
        if (mParent.mGrainSize>0) {
            tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0,mParticles.size(),mParent.mGrainSize), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, mParticles.size()));
        }
    }

    /// @brief Rasterize sphere at position P and radius R into a
    /// narrow-band level set with half-width, mHalfWidth.
    /// @return false if it was interrupted
    ///
    /// @param P coordinates of the particle position in voxel units
    /// @param R radius of particle in voxel units
    /// @param id
    /// @param accessor grid accessor with a private copy of the grid
    ///
    /// @note For best performance all computations are performed in
    /// voxel-space with the important exception of the final level set
    /// value that is converted to world units (e.g. the grid stores
    /// the closest Euclidian signed distances measured in world
    /// units). Also note we use the convention of positive distances
    /// outside the surface an negative distances inside the surface.
    bool makeSphere(const Vec3R &P, SdfT R, const AttT& att, AccessorT& acc)
    {
        const ValueT inside = -mGrid->background();
        const SdfT dx = mParent.mDx, w = mParent.mHalfWidth;
        const SdfT max = R + w;// maximum distance in voxel units
        const Coord a(math::Floor(P[0]-max),math::Floor(P[1]-max),math::Floor(P[2]-max));
        const Coord b(math::Ceil( P[0]+max),math::Ceil( P[1]+max),math::Ceil( P[2]+max));
        const SdfT max2 = math::Pow2(max);//square of maximum distance in voxel units
        const SdfT min2 = math::Pow2(math::Max(SdfT(0), R - w));//square of minimum distance
        ValueT v;
        size_t count = 0;
        for ( Coord c = a; c.x() <= b.x(); ++c.x() ) {
            //only check interrupter every 32'th scan in x
            if (!(count++ & (1<<5)-1) && util::wasInterrupted(mParent.mInterrupter)) {
                tbb::task::self().cancel_group_execution();
                return false;
            }
            SdfT x2 = math::Pow2( c.x() - P[0] );
            for ( c.y() = a.y(); c.y() <= b.y(); ++c.y() ) {
                SdfT x2y2 = x2 + math::Pow2( c.y() - P[1] );
                for ( c.z() = a.z(); c.z() <= b.z(); ++c.z() ) {
                    SdfT x2y2z2 = x2y2 + math::Pow2( c.z() - P[2] );//square distance from c to P
                    if ( x2y2z2 >= max2 || (!acc.probeValue(c,v) && v<ValueT(0)) )
                        continue;//outside narrow band of the particle or inside existing level set
                    if ( x2y2z2 <= min2 ) {//inside narrow band of the particle.
                        acc.setValueOff(c, inside);
                        continue;
                    }
                    // convert signed distance from voxel units to world units
                    //const ValueT d=dx*(math::Sqrt(x2y2z2) - R);
                    const ValueT d=Merge(dx*(math::Sqrt(x2y2z2) - R), att);
                    if (d < v) acc.setValue(c, d);//CSG union
                }//end loop over z
            }//end loop over y
        }//end loop over x
        return true;
    }
    typedef typename boost::function<void (Raster*, const tbb::blocked_range<size_t>&)> FuncType;

    template <typename DisableType>
    typename boost::enable_if<DisableType>::type
    getAtt(size_t, AttT&) const {;}

    template <typename DisableType>
    typename boost::disable_if<DisableType>::type
    getAtt(size_t n, AttT& a) const {mParticles.getAtt(n, a);}

    template <typename T>
    typename boost::enable_if<boost::is_same<T,ValueT>, ValueT>::type
    Merge(T s, const AttT&) const { return s; }

    template <typename T>
    typename boost::disable_if<boost::is_same<T,ValueT>, ValueT>::type
    Merge(T s, const AttT& a) const { return ValueT(s,a); }

    ParticlesToLevelSetT& mParent;
    const ParticleListT&  mParticles;//list of particles
    GridT*                mGrid;
    const math::MapBase&  mMap;
    size_t                mMinCount, mMaxCount;//counters for ignored particles!
    FuncType              mTask;
    const bool            mOwnsGrid;
};//end of Raster struct


///////////////////// YOU CAN SAFELY IGNORE THIS SECTION /////////////////////

namespace local {
// This is a simple type that combines a distance value and a particle
// attribute. It's required for attribute transfer which is defined in the
// Raster class above.
template <typename VisibleT, typename BlindT>
class BlindData
{
  public:
    typedef VisibleT type;
    typedef VisibleT VisibleType;
    typedef BlindT   BlindType;
    explicit BlindData() {}
    explicit BlindData(VisibleT v) : mVisible(v) {}
    BlindData(VisibleT v, BlindT b) : mVisible(v), mBlind(b) {}
    BlindData& operator=(const BlindData& rhs)
    {
        mVisible = rhs.mVisible;
        mBlind = rhs.mBlind;
        return *this;
    }
    const VisibleT& visible() const { return mVisible; }
    const BlindT&   blind()   const { return mBlind; }
    bool operator==(const BlindData& rhs)     const { return mVisible == rhs.mVisible; }
    bool operator< (const BlindData& rhs)     const { return mVisible <  rhs.mVisible; };
    bool operator> (const BlindData& rhs)     const { return mVisible >  rhs.mVisible; };
    BlindData operator+(const BlindData& rhs) const { return BlindData(mVisible + rhs.mVisible); };
    BlindData operator+(const VisibleT&  rhs) const { return BlindData(mVisible + rhs); };
    BlindData operator-(const BlindData& rhs) const { return BlindData(mVisible - rhs.mVisible); };
    BlindData operator-() const { return BlindData(-mVisible, mBlind); }
protected:
    VisibleT mVisible;
    BlindT   mBlind;
};
// Required by several of the tree nodes
template <typename VisibleT, typename BlindT>
inline std::ostream& operator<<(std::ostream& ostr, const BlindData<VisibleT, BlindT>& rhs)
{
    ostr << rhs.visible();
    return ostr;
}
// Required by math::Abs
template <typename VisibleT, typename BlindT>
inline BlindData<VisibleT, BlindT> Abs(const BlindData<VisibleT, BlindT>& x)
{
    return BlindData<VisibleT, BlindT>(math::Abs(x.visible()), x.blind());
}
}// local namespace

//////////////////////////////////////////////////////////////////////////////

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_PARTICLES_TO_LEVELSET_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
