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
/// @brief This tool rasterizes particles (with position, radius and velocity)
/// into a narrow-band level set.
///
/// @note This fast particle to level set converter is always intended
/// to be combined with some kind of surface post processing,
/// i.e. tools::Filter. Without such post processing the generated
/// surface is typically too noisy and blooby. However it serves as a
/// great and fast starting point for subsequent level set surface
/// processing and convolution. In the near future we will add support
/// for anisotropic particle kernels.
///
/// The @c ParticleListT template argument below refers to any class
/// with the following interface (see unittest/TestParticlesToLevelSet.cc
/// and SOP_DW_OpenVDBParticleVoxelizer for practical examples):
/// @code
/// class ParticleList {
///   ...
/// public:
///   openvdb::Index size()       const;// number of particles in list
///   openvdb::Vec3R pos(int n)   const;// world space position of n'th particle
///   openvdb::Vec3R vel(int n)   const;// world space velocity of n'th particle
///   openvdb::Real radius(int n) const;// world space radius of n'th particle
/// };
/// @endcode
///
/// @note All methods are assumed to be thread-safe.
/// Also note all access methods return by value
/// since this allows for especailly the radius and velocities to be
/// scaled (i.e. modified) relative to the internal representations
/// (see  unittest/TestParticlesToLevelSet.cc for an example).
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
namespace local {
/// Trait class needed to merge and split a distance and a particle id required
/// during attribute transfer. The default implementation of merge simply
/// ignores the particle id. A specialized implementation is given
/// below for a Dual type that holds both a distance and a particle id.
template <typename T>
struct DualTrait
{
    static T merge(T dist, Index32) { return dist; }
    static T split(T dist) { return dist; }
};
}// namespace local

template<typename GridT,
         typename ParticleListT,
         typename InterruptT=util::NullInterrupter,
         typename RealT = typename GridT::ValueType>
class ParticlesToLevelSet
{
public:
    /// @brief Main constructor using a default interrupter
    ///
    /// @param grid       contains the grid in which particles are rasterized
    /// @param interrupt  callback to interrupt a long-running process
    ///
    /// @note The width in voxel units of the generated narrow band level set is
    /// given by 2*background/dx, where background is the background value
    /// stored in the grid, and dx is the voxel size derived from the
    /// transform stored in the grid. Also note that -background
    /// corresponds to the constant value inside the generated narrow
    /// band level sets. Finally the default NullInterrupter should
    /// compile out interruption checks during optimization, thus
    /// incurring no run-time overhead.
    ///
    ParticlesToLevelSet(GridT& grid, InterruptT* interrupt = NULL) :
        mGrid(&grid),
        mPa(NULL),
        mDx(grid.transform().voxelSize()[0]),
        mHalfWidth(local::DualTrait<ValueT>::split(grid.background()) / mDx),
        mRmin(1.5),// corresponds to the Nyquist grid sampling frequency
        mRmax(100.0f),// corresponds to a huge particle (probably too large!)
        mGrainSize(1),
        mInterrupter(interrupt),
        mMinCount(0),
        mMaxCount(0),
        mIsSlave(false)
    {
        if ( !mGrid->hasUniformVoxels() ) {
            OPENVDB_THROW(RuntimeError,
                "The transform must have uniform scale for ParticlesToLevelSet to function!");
        }
        if (mGrid->getGridClass() != GRID_LEVEL_SET) {
            OPENVDB_THROW(RuntimeError,
                "ParticlesToLevelSet only supports level sets!"
                "\nUse Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
        }
        /// @todo create a new tree rather than CSG into an existing tree
        //mGrid->newTree();
    }
    /// @brief Copy constructor called by tbb
    ParticlesToLevelSet(ParticlesToLevelSet& other, tbb::split) :
        mGrid(new GridT(*other.mGrid, openvdb::ShallowCopy())),
        mPa(other.mPa),
        mDx(other.mDx),
        mHalfWidth(other.mHalfWidth),
        mRmin(other.mRmin),
        mRmax(other.mRmax),
        mGrainSize(other.mGrainSize),
        mTask(other.mTask),
        mInterrupter(other.mInterrupter),
        mMinCount(0),
        mMaxCount(0),
        mIsSlave(true)
    {
        mGrid->newTree();
    }
    virtual ~ParticlesToLevelSet() { if (mIsSlave) delete mGrid; }

    /// @return Half-width (in voxle units) of the narrow band level set
    RealT getHalfWidth() const { return mHalfWidth; }

    /// @return Voxel size in world units
    RealT getVoxelSize() const { return mDx; }

    /// @return the smallest radius allowed in voxel units
    RealT getRmin() const { return mRmin; }
    /// @return the largest radius allowed in voxel units
    RealT getRmax() const { return mRmax; }

    /// @return true if any particles were ignored due to their size
    bool ignoredParticles() const { return mMinCount>0 || mMaxCount>0; }
    /// @return number of small particles that were ignore due to Rmin
    size_t getMinCount() const { return mMinCount; }
    /// @return number of large particles that were ignore due to Rmax
    size_t getMaxCount() const { return mMaxCount; }

    /// set the smallest radius allowed in voxel units
    void setRmin(RealT Rmin) { mRmin = math::Max(RealT(0),Rmin); }
    /// set the largest radius allowed in voxel units
    void setRmax(RealT Rmax) { mRmax = math::Max(mRmin,Rmax); }

    /// @return the grain-size used for multi-threading
    int  getGrainSize() const { return mGrainSize; }
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(int grainSize) { mGrainSize = grainSize; }

    /// @brief Rasterize a sphere per particle derived from their
    /// position and radius. All spheres are CSG unioned.
    /// @param pa particles with position, radius and velocity.
    void rasterizeSpheres(const ParticleListT& pa)
    {
        mPa = &pa;
        if (mInterrupter) mInterrupter->start("Rasterizing particles to level set using spheres");
        mTask = boost::bind(&ParticlesToLevelSet::rasterSpheres, _1, _2);
        this->cook();
        if (mInterrupter) mInterrupter->end();
    }

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
    /// has a default value of 1.5 corresponding to the Nyquist frequency!
    void rasterizeTrails(const ParticleListT& pa, Real delta=1.0)
    {
        mPa = &pa;
        if (mInterrupter) mInterrupter->start("Rasterizing particles to level set using trails");
        mTask = boost::bind(&ParticlesToLevelSet::rasterTrails, _1, _2, RealT(delta));
        this->cook();
        if (mInterrupter) mInterrupter->end();
    }

    //  ========> DO NOT CALL ANY OF THE PUBLIC METHODS BELOW! <=============

    /// @brief Non-const functor called by tbb::parallel_reduce threads
    ///
    /// @note Do not call this method directly!
    void operator()(const tbb::blocked_range<size_t>& r)
    {
        if (!mTask) {
            OPENVDB_THROW(ValueError, "mTask is undefined - don't call operator() directly!");
        }
        mTask(this, r);
    }

    /// @brief Method called by tbb::parallel_reduce threads
    ///
    /// @note Do not call this method directly!
    void join(ParticlesToLevelSet& other)
    {
        tools::csgUnion(*mGrid, *other.mGrid, /*prune=*/true);
        mMinCount += other.mMinCount;
        mMaxCount += other.mMaxCount;
    }

private:
    typedef typename GridT::ValueType ValueT;
    typedef typename GridT::Accessor Accessor;
    typedef typename boost::function<void (ParticlesToLevelSet*,
                                           const tbb::blocked_range<size_t>&)> FuncType;
    GridT*                mGrid;
    const ParticleListT*  mPa;//list of particles
    const RealT           mDx;//size of voxel in world units
    const RealT           mHalfWidth;//half width of narrow band LS in voxels
    RealT                 mRmin;//ignore particles smaller than this radius in voxels
    RealT                 mRmax;//ignore particles larger than this radius in voxels
    int                   mGrainSize;
    FuncType              mTask;
    InterruptT*           mInterrupter;
    size_t                mMinCount, mMaxCount;//counters for ignored particles!
    const bool            mIsSlave;

    void cook()
    {
        if (mGrainSize>0) {
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0,mPa->size(),mGrainSize), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, mPa->size()));
        }
    }
    /// @return true if the particle is too small or too large
    inline bool ignoreParticle(RealT R)
    {
        if (R<mRmin) {// below the cutoff radius
            ++mMinCount;
            return true;
        }
        if (R>mRmax) {// above the cutoff radius
            ++mMaxCount;
            return true;
        }
        return false;
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
    inline bool rasterSphere(const Vec3R &P, RealT R, Index32 id, Accessor& accessor)
    {
        const ValueT inside = -mGrid->background();
        const RealT dx = mDx;
        const RealT max = R + mHalfWidth;// maximum distance in voxel units
        const Coord a(math::Floor(P[0]-max),math::Floor(P[1]-max),math::Floor(P[2]-max));
        const Coord b(math::Ceil( P[0]+max),math::Ceil( P[1]+max),math::Ceil( P[2]+max));
        const RealT max2 = math::Pow2(max);//square of maximum distance in voxel units
        const RealT min2 = math::Pow2(math::Max(RealT(0), R - mHalfWidth));//square of minimum distance
        ValueT v;
        size_t count = 0;
        for ( Coord c = a; c.x() <= b.x(); ++c.x() ) {
            //only check interrupter every 32'th scan in x
            if (!(count++ & (1<<5)-1) && util::wasInterrupted(mInterrupter)) {
                tbb::task::self().cancel_group_execution();
                return false;
            }
            RealT x2 = math::Pow2( c.x() - P[0] );
            for ( c.y() = a.y(); c.y() <= b.y(); ++c.y() ) {
                RealT x2y2 = x2 + math::Pow2( c.y() - P[1] );
                for ( c.z() = a.z(); c.z() <= b.z(); ++c.z() ) {
                    RealT x2y2z2 = x2y2 + math::Pow2( c.z() - P[2] );//square distance from c to P
                    if ( x2y2z2 >= max2 || (!accessor.probeValue(c,v) && v<ValueT(0)) )
                        continue;//outside narrow band of the particle or inside existing ls
                    if ( x2y2z2 <= min2 ) {//inside narrowband of the particle.
                        accessor.setValueOff(c, inside);
                        continue;
                    }
                    // distance in world units
                    const ValueT d = local::DualTrait<ValueT>::merge(dx*(math::Sqrt(x2y2z2) - R), id);
                    if (d < v) accessor.setValue(c, d);//CSG union
                }//end loop over z
            }//end loop over y
        }//end loop over x
        return true;
    }

    /// @brief Rasterize particles as spheres with variable radius
    ///
    /// @param r tbb's default range referring to the list of particles
    void rasterSpheres(const tbb::blocked_range<size_t> &r)
    {
        Accessor accessor = mGrid->getAccessor(); // local accessor
        const RealT inv_dx = RealT(1)/mDx;
        bool run = true;
        for (Index32 id = r.begin(), e=r.end(); run && id != e; ++id) {
            const RealT R = inv_dx*mPa->radius(id);// in voxel units
            if (this->ignoreParticle(R)) continue;
            const Vec3R P = mGrid->transform().worldToIndex(mPa->pos(id));
            run = this->rasterSphere(P, R, id, accessor);
        }//end loop over particles
    }

    /// @brief Rasterize particles as trails with length = |V|
    ///
    /// @param r tbb's default range referring to the list of particles
    /// @param delta scale distance between the velocity blurring of
    /// particles. Increasing it (above 1) typically results in aliasing!
    ///
    /// @note All computations are performed in voxle units. Also
    /// for very small values of delta the number of instances will
    /// increase resulting in very slow rasterization that cannot be
    /// interrupted!
    void rasterTrails(const tbb::blocked_range<size_t> &r,
                      RealT delta)//scale distance between instances (eg 1.0)
    {
        Accessor accessor = mGrid->getAccessor(); // local accessor
        const RealT inv_dx = RealT(1)/mDx, Rmin = mRmin;
        bool run = true;
        for (Index32 id = r.begin(), e=r.end(); run && id != e; ++id) {
            const RealT  R0 = inv_dx*mPa->radius(id);
            if (this->ignoreParticle(R0)) continue;
            const Vec3R P0 = mGrid->transform().worldToIndex(mPa->pos(id)),
                         V = inv_dx*mPa->vel(id);
            const RealT speed = V.length(), inv_speed=1.0/speed;
            const Vec3R N = -V*inv_speed;// inverse normalized direction
            Vec3R  P = P0;// local position of instance
            RealT R = R0, d=0;// local radius and length of trail
            for (size_t m=0; run && d < speed ; ++m) {
                run = this->rasterSphere(P, R, id, accessor);
                P += 0.5*delta*R*N;// adaptive offset along inverse velocity direction
                d  = (P-P0).length();// current length of trail
                R  = R0-(R0-Rmin)*d*inv_speed;// R = R0 -> mRmin(e.g. 1.5)
            }//loop over sphere instances
        }//end loop over particles
    }
};//end of ParticlesToLevelSet class

///////////////////// YOU CAN SAFELY IGNORE THIS SECTION /////////////////////
namespace local {
// This is a simple type that combines a distance value and a particle
// id. It's required for attribute transfer which is defined in the
// ParticlesToLevelSetAndId class below.
template <typename RealT>
class Dual
{
public:
    explicit Dual() : mId(util::INVALID_IDX) {}
    explicit Dual(RealT d) : mDist(d), mId(util::INVALID_IDX) {}
    explicit Dual(RealT d, Index32 id) : mDist(d), mId(id) {}
    Dual& operator=(const Dual& rhs) { mDist = rhs.mDist; mId = rhs.mId; return *this;}
    bool isIdValid() const { return mId != util::INVALID_IDX; }
    Index32 id()   const { return mId; }
    RealT dist() const { return mDist; }
    bool operator!=(const Dual& rhs) const  { return mDist != rhs.mDist; }
    bool operator==(const Dual& rhs) const  { return mDist == rhs.mDist; }
    bool operator< (const Dual& rhs) const  { return mDist <  rhs.mDist; };
    bool operator<=(const Dual& rhs) const  { return mDist <= rhs.mDist; };
    bool operator> (const Dual& rhs) const  { return mDist >  rhs.mDist; };
    Dual operator+ (const Dual& rhs) const  { return Dual(mDist+rhs.mDist); };
    Dual operator+ (const RealT& rhs) const { return Dual(mDist+rhs); };
    Dual operator- (const Dual& rhs) const  { return Dual(mDist-rhs.mDist); };
    Dual operator-() const { return Dual(-mDist); }
protected:
    RealT   mDist;
    Index32 mId;
};
// Required by several of the tree nodes
template <typename RealT>
inline std::ostream& operator<<(std::ostream& ostr, const Dual<RealT>& rhs)
{
    ostr << rhs.dist();
    return ostr;
}
// Required by math::Abs
template <typename RealT>
inline Dual<RealT> Abs(const Dual<RealT>& x)
{
    return Dual<RealT>(math::Abs(x.dist()),x.id());
}
// Specialization of trait class used to merge and split a distance and a particle id
template <typename T>
struct DualTrait<Dual<T> >
{
    static Dual<T> merge(T dist, Index32 id) { return Dual<T>(dist, id); }
    static T split(Dual<T> dual) { return dual.dist(); }
};
}// local namespace
//////////////////////////////////////////////////////////////////////////////

/// @brief Use this wrapper class to convert particles into a level set and a
/// separate index grid of closest-point particle id. The latter can
/// be used to subsequently transfer particles attributes into
/// separate grids.
/// @note This class has the same API as ParticlesToLevelSet - the
/// only exception being the raster methods that return the index grid!
template<typename LevelSetGridT,
         typename ParticleListT,
         typename InterruptT = util::NullInterrupter>
class ParticlesToLevelSetAndId
{
public:
    typedef typename LevelSetGridT::ValueType  RealT;
    typedef typename local::Dual<RealT>        DualT;
    typedef typename LevelSetGridT::TreeType   RealTreeT;
    typedef typename RealTreeT::template ValueConverter<DualT>::Type DualTreeT;
    typedef Int32Tree                          IndxTreeT;
    typedef Grid<DualTreeT>                    DualGridT;
    typedef Int32Grid                          IndxGridT;

    ParticlesToLevelSetAndId(LevelSetGridT& ls, InterruptT* interrupter = NULL) :
        mRealGrid(ls),
        mDualGrid(DualT(ls.background()))
    {
        mDualGrid.setGridClass(ls.getGridClass());
        mDualGrid.setTransform(ls.transformPtr());
        mRaster = new RasterT(mDualGrid, interrupter);
    }

    virtual ~ParticlesToLevelSetAndId() { delete mRaster; }

    /// @return Half-width (in voxle units) of the narrow band level set
    RealT getHalfWidth() const { return mRaster->getHalfWidth(); }

    /// @return Voxel size in world units
    RealT getVoxelSize() const { return mRaster->getVoxelSize(); }

     /// @return true if any particles were ignored due to their size
    bool ignoredParticles() const { return mRaster->ignoredParticles(); }
    /// @return number of small particles that were ignore due to Rmin
    size_t getMinCount() const { return mRaster->getMinCount(); }
    /// @return number of large particles that were ignore due to Rmax
    size_t getMaxCount() const { return mRaster->getMaxCount(); }

    /// @return the smallest radius allowed in voxel units
    RealT getRmin() const { return mRaster->getRmin(); }
     /// @return the largest radius allowed in voxel units
    RealT getRmax() const { return mRaster->getRmax(); }

    /// set the smallest radius allowed in voxel units
    void setRmin(RealT Rmin) { mRaster->setRmin(Rmin); }
    /// set the largest radius allowed in voxel units
    void setRmax(RealT Rmax) { mRaster->setRmax(Rmax); }

    /// @return the grain-size used for multi-threading
    int  getGrainSize() const { return mRaster->getGrainSize(); }
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(int grainSize) { mRaster->setGrainSize(grainSize); }

    /// @brief Rasterize a sphere per particle derived from their
    /// position and radius. All spheres are CSG unioned.
    /// @return An index grid storing the id of the closest particle.
    /// @param pa particles with position, radius and velocity.
    typename IndxGridT::Ptr rasterizeSpheres(const ParticleListT& pa)
    {
        mRaster->rasterizeSpheres(pa);
        return this->extract();
    }
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
    /// has a default value of 1.5 corresponding to the Nyquist frequency!
    typename IndxGridT::Ptr rasterizeTrails(const ParticleListT& pa, Real delta=1.0)
    {
        mRaster->rasterizeTrails(pa, delta);
        return this->extract();
    }

private:

    /// disallow copy construction
    ParticlesToLevelSetAndId(const ParticlesToLevelSetAndId& other) {}
    /// disallow copy assignment
    ParticlesToLevelSetAndId& operator=(const ParticlesToLevelSetAndId& rhs)
    {
        return *this;
    }
    /// @brief Private method to extract the level set and index grid.
    typename IndxGridT::Ptr extract()
    {
        // Use topology copy constructors since output grids have the
        // same topology as mDualGrid
        const DualTreeT& dualTree = mDualGrid.tree();
        typename IndxTreeT::Ptr indxTree(new IndxTreeT(dualTree,util::INVALID_IDX,TopologyCopy()));
        typename IndxGridT::Ptr indxGrid = typename IndxGridT::Ptr(new IndxGridT(indxTree));
        indxGrid->setTransform(mDualGrid.transformPtr());
        typename RealTreeT::Ptr realTree(new RealTreeT(dualTree,mRealGrid.background(),TopologyCopy()));
        mRealGrid.setTree(realTree);

        // Extract the level set and IDs from mDualGrid. We will
        // explore the fact that by design active values always live
        // at the leaf node level, i.e. no active tiles exist in level sets
        typedef typename DualGridT::TreeType::LeafCIter        LeafIterT;
        typedef typename DualGridT::TreeType::LeafNodeType     LeafT;
        typedef typename LevelSetGridT::TreeType::LeafNodeType RealLeafT;
        typedef typename IndxGridT::TreeType::LeafNodeType     IndxLeafT;
        RealTreeT& realTreeRef = *realTree;
        IndxTreeT& indxTreeRef = *indxTree;
        for (LeafIterT n = mDualGrid.tree().cbeginLeaf(); n; ++n) {
            const LeafT& leaf = *n;
            const Coord xyz = leaf.getOrigin();
            // Get leafnodes that were allocated during topology contruction!
            RealLeafT& i = *realTreeRef.probeLeaf(xyz);
            IndxLeafT& j = *indxTreeRef.probeLeaf(xyz);
            for (typename LeafT::ValueOnCIter m=leaf.cbeginValueOn(); m; ++m) {
                // Use linear offset (vs coordinate) access for better performance!
                const Index k = m.pos();
                const DualT& v = *m;
                i.setValueOnly(k, v.dist());
                j.setValueOnly(k, v.id());
            }
        }
        mRealGrid.signedFloodFill();//required since we only transferred active voxels!
        return indxGrid;
    }

    typedef ParticlesToLevelSet<DualGridT, ParticleListT, InterruptT, RealT> RasterT;
    LevelSetGridT& mRealGrid;// input level set grid
    DualGridT      mDualGrid;// grid encoding both the level set and the point id
    RasterT*       mRaster;
};//end of ParticlesToLevelSetAndId

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_PARTICLES_TO_LEVELSET_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
