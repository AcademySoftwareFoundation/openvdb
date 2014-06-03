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
/// @file PointScatter.h
///
/// @brief We offer three differet algorithms (each in its own class)
///        for scattering of point in active voxels:
///
/// 1) UniformPointScatter. Has two modes: Either randomly distributes
///    a fixed number of points in the active voxels, or the user can
///    specify a fixed probability of having a points per unit of volume.
///
/// 2) DenseUniformPointScatter. Randomly distributes points in active
///    voxels using a fixed number of points per voxel.
///
/// 3) NonIniformPointScatter. Define the local probability of having
///    a point in a voxel as the product of a global density and the
///    value of the voxel itself.

#ifndef OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/util/NullInterrupter.h>
#include <boost/random/uniform_01.hpp>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>
#include <boost/scoped_array.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// Forward declaration of base class
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = openvdb::util::NullInterrupter>
class BasePointScatter;

/// @brief The two point scatters UniformPointScatter and
/// NonUniformPointScatter depend on the following two classes:
///
/// The @c PointAccessorType template argument below refers to any class
/// with the following interface:
/// @code
/// class PointAccessor {
///   ...
/// public:
///   void add(const openvdb::Vec3R &pos);// appends point with world positions pos
/// };
/// @endcode
///
///
/// The @c InterruptType template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = NULL)// called when computations begin
///   void end()                         // called when computations end
///   bool wasInterrupted(int percent=-1)// return true to break computation
///};
/// @endcode
///
/// @note If no template argument is provided for this InterruptType
/// the util::NullInterrupter is used which implies that all
/// interrupter calls are no-ops (i.e. incurs no computational overhead).


/// @brief Uniform scatters of point in the active voxels.
/// The point count is either explicitly defined or implicitly
/// through the specification of a global density (=points-per-volume)
///
/// @note This uniform scattering technique assumes that the number of
/// points is generally smaller than the number of active voxels
/// (including virtual active voxels in active tiles).
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = openvdb::util::NullInterrupter>
class UniformPointScatter : public BasePointScatter<PointAccessorType,
                                                    RandomGenerator,
                                                    InterruptType>
{
public:
    typedef BasePointScatter<PointAccessorType, RandomGenerator, InterruptType> BaseT;

    UniformPointScatter(PointAccessorType& points,
                        Index64 pointCount,
                        RandomGenerator& randGen,
                        InterruptType* interrupt = NULL)
        : BaseT(points, randGen, interrupt)
        , mTargetPointCount(pointCount)
        , mPointsPerVolume(0.0f)
    {
    }
    UniformPointScatter(PointAccessorType& points,
                        float pointsPerVolume,
                        RandomGenerator& randGen,
                        InterruptType* interrupt = NULL)
        : BaseT(points, randGen, interrupt)
        , mTargetPointCount(0)
        , mPointsPerVolume(pointsPerVolume)
    {
    }

    /// @brief This is the main functor method implementing the actual
    /// scattering of points.
    template<typename GridT>
    bool operator()(const GridT& grid, bool threaded = true)
    {
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) return false;
        const openvdb::Vec3d dim = grid.voxelSize();
        if (mPointsPerVolume>0) {
            BaseT::start("Uniform scattering with fixed point density");
            mTargetPointCount = Index64(mPointsPerVolume*dim[0]*dim[1]*dim[2])*mVoxelCount;
        } else if (mTargetPointCount>0) {
            BaseT::start("Uniform scattering with fixed point count");
            mPointsPerVolume = mTargetPointCount/float(dim[0]*dim[1]*dim[2] * mVoxelCount);
        } else {
            return false;
        }

        BuildList b(mVoxelCount, mTargetPointCount);
        b.build(threaded);

        openvdb::CoordBBox bbox;
        typename GridT::ValueOnCIter valueIter = grid.cbeginValueOn();
        for (Index64 i=0, n=valueIter.getVoxelCount() ; i != mTargetPointCount; ++i) {
            if (BaseT::interrupt()) return false;
            const Index64 voxelId = b.mList[i];//list[i];
            while ( n <= voxelId ) {
                ++valueIter;
                n += valueIter.getVoxelCount();
            }
            if (valueIter.isVoxelValue()) {// a majorty is expected to be voxels
                const openvdb::Coord min = valueIter.getCoord();
                const openvdb::Vec3R dmin(min.x()-0.5, min.y()-0.5, min.z()-0.5);
                BaseT::addPoint(grid, dmin);
            } else {// tiles contain multiple (virtual) voxels
                valueIter.getBoundingBox(bbox);
                const openvdb::Coord size(bbox.extents());
                const openvdb::Vec3R dmin(bbox.min().x()-0.5,
                                          bbox.min().y()-0.5,
                                          bbox.min().z()-0.5);
                BaseT::addPoint(grid, dmin, size);
            }
        }
        BaseT::end();
        return true;
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Uniformely scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\" corresponding to "
           << mPointsPerVolume << " points per volume." << std::endl;
    }

    float getPointsPerVolume() const { return mPointsPerVolume; }
    Index64 getTargetPointCount() const { return mTargetPointCount; }

private:

    // Builds list of voxels to contain a point. It is a little
    // complicated because the resulting list must be deterministic
    // despite the fact that it combines multi-threading and random
    // number generation.
    struct BuildList
    {
        static const size_t LOG2SIZE = 13, SIZE = 1 << LOG2SIZE;
        BuildList(Index64 voxelCount, Index64 pointCount, size_t seed = 0)
            : mOwnsList(true)
            , mList(new Index64[pointCount])
            , mPointCount(pointCount)
            , mVoxelCount(voxelCount)
            , mSeed(seed)
        {
        }
        BuildList(const BuildList& other)
            : mOwnsList(false)
            , mList(other.mList)
            , mPointCount(other.mPointCount)
            , mVoxelCount(other.mVoxelCount)
            , mSeed(other.mSeed)
        {
        }
        ~BuildList() { if (mOwnsList) delete [] mList; }
        void build(bool threaded = true)
        {
            const size_t n = mPointCount >> LOG2SIZE;
            if ( threaded && n > 0 ) {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, n, 1), *this);
                const size_t m = n << LOG2SIZE;
                if ( m < mPointCount ) (*this)(m, mPointCount);
            } else {
                (*this)(0, mPointCount);
            }
            tbb::parallel_sort(mList, mList + mPointCount);
        }
        void operator()(const tbb::blocked_range<size_t>& r) const
        {
            for (size_t n = r.begin()<<LOG2SIZE, m = r.end()<<LOG2SIZE; n != m; n += SIZE) {
                (*this)(n, n + SIZE);
            }
        }
        inline void operator()(size_t begin, size_t end) const
        {
            mRandomGen.seed(begin + mSeed);
            const double maxId = static_cast<double>(mVoxelCount-1);
            for (size_t i = begin; i != end; ++i) {
                mList[i] = static_cast<Index64>(math::Round(maxId*mRandom(mRandomGen)));
            }
        }
        const bool                        mOwnsList;
        Index64*                          mList;
        const Index64                     mPointCount;
        const Index64                     mVoxelCount;
        const size_t                      mSeed;
        mutable RandomGenerator           mRandomGen;
        mutable boost::uniform_01<double> mRandom;
    };// BuildList

    using BaseT::mPointCount;
    using BaseT::mVoxelCount;
    using BaseT::mRandomGen;
    Index64 mTargetPointCount;
    float mPointsPerVolume;

}; // class UniformPointScatter

/// @brief Scatters a fixed (and integer) number of points in all
/// active voxels and tiles.
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = openvdb::util::NullInterrupter>
class DenseUniformPointScatter : public BasePointScatter<PointAccessorType,
                                                         RandomGenerator,
                                                         InterruptType>
{
public:
    typedef BasePointScatter<PointAccessorType, RandomGenerator, InterruptType> BaseT;

    DenseUniformPointScatter(PointAccessorType& points,
                             size_t pointsPerVoxel,
                             RandomGenerator& randGen,
                             InterruptType* interrupt = NULL)
        : BaseT(points, randGen, interrupt)
        , mPointsPerVoxel(pointsPerVoxel)
    {
    }

    /// This is the main functor method implementing the actual scattering of points.
    template<typename GridT>
    bool operator()(const GridT& grid)
    {
        typedef typename GridT::ValueOnCIter ValueIter;
        if (mPointsPerVoxel == 0) return false;
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) return false;
        BaseT::start("Dense uniform scattering with fixed point count");
        openvdb::CoordBBox bbox;
        for (ValueIter iter = grid.cbeginValueOn(); iter; ++iter) {
            if (BaseT::interrupt()) return false;
            if (iter.isVoxelValue()) {// a majorty is expected to be voxels
                const openvdb::Coord min = iter.getCoord();
                const openvdb::Vec3R dmin(min.x()-0.5, min.y()-0.5, min.z()-0.5);
                for (size_t n = 0, m = mPointsPerVoxel; n != m; ++n) {
                    BaseT::addPoint(grid, dmin);
                }
            } else {// tiles contain multiple (virtual) voxels
                iter.getBoundingBox(bbox);
                const openvdb::Coord size(bbox.extents());
                const openvdb::Vec3R dmin(bbox.min().x()-0.5,
                                          bbox.min().y()-0.5,
                                          bbox.min().z()-0.5);
                for (size_t n = 0, m = mPointsPerVoxel * iter.getVoxelCount(); n != m; ++n) {
                    BaseT::addPoint(grid, dmin, size);
                }
            }
        }
        BaseT::end();
        return true;
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Dense uniformely scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\" corresponding to "
           << mPointsPerVoxel << " points per voxel." << std::endl;
    }

    size_t getPointsPerVoxel() const { return mPointsPerVoxel; }

private:
    using BaseT::mPointCount;
    using BaseT::mVoxelCount;
    size_t mPointsPerVoxel;
}; // class DenseUniformPointScatter

/// @brief Non-uniform scatters of point in the active voxels.
/// The local point count is implicitly defined as a product of
/// of a global density (called pointsPerVolume) and the local voxel
/// (or tile) value.
///
/// @note This scattering technique can be significantly slower
/// than a uniform scattering since its computational complexity
/// is proportional to the active voxel (and tile) count.
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = openvdb::util::NullInterrupter>
class NonUniformPointScatter : public BasePointScatter<PointAccessorType,
                                                       RandomGenerator,
                                                       InterruptType>
{
public:
    typedef BasePointScatter<PointAccessorType, RandomGenerator, InterruptType> BaseT;

    NonUniformPointScatter(PointAccessorType& points,
                           float pointsPerVolume,
                           RandomGenerator& randGen,
                           InterruptType* interrupt = NULL)
        : BaseT(points, randGen, interrupt)
        , mPointsPerVolume(pointsPerVolume)//note this is merely a
                                           //multiplyer for the local point density
    {
    }

    /// This is the main functor method implementing the actual scattering of points.
    template<typename GridT>
    bool operator()(const GridT& grid)
    {
        if (mPointsPerVolume <= 0.0f) return false;
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) return false;
        BaseT::start("Non-uniform scattering with local point density");
        const openvdb::Vec3d dim = grid.voxelSize();
        const double volumePerVoxel = dim[0]*dim[1]*dim[2],
                     pointsPerVoxel = mPointsPerVolume * volumePerVoxel;
        openvdb::CoordBBox bbox;
        for (typename GridT::ValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
            if (BaseT::interrupt()) return false;
            const double d = (*iter) * pointsPerVoxel * iter.getVoxelCount();
            const int n = int(d);
            if (iter.isVoxelValue()) { // a majorty is expected to be voxels
                const openvdb::Coord min = iter.getCoord();
                const openvdb::Vec3R dmin(min.x()-0.5, min.y()-0.5, min.z()-0.5);
                for (int i = 0; i < n; ++i) BaseT::addPoint(grid, dmin);
                if (BaseT::getRand() < (d - n)) BaseT::addPoint(grid, dmin);
            } else { // tiles contain multiple (virtual) voxels
                iter.getBoundingBox(bbox);
                const openvdb::Coord size(bbox.extents());
                const openvdb::Vec3R dmin(bbox.min().x()-0.5,
                                          bbox.min().y()-0.5,
                                          bbox.min().z()-0.5);
                for (int i = 0; i < n; ++i) BaseT::addPoint(grid, dmin, size);
                if (BaseT::getRand() < (d - n)) BaseT::addPoint(grid, dmin, size);
            }
        }//loop over the active values
        BaseT::end();
        return true;
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Non-uniformely scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\"." << std::endl;
    }

    float getPointPerVolume() const { return mPointsPerVolume; }

private:
    using BaseT::mPointCount;
    using BaseT::mVoxelCount;
    float mPointsPerVolume;

}; // class NonUniformPointScatter

/// Base class of all the point scattering classes defined above
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType>
class BasePointScatter
{
public:

    Index64 getPointCount() const { return mPointCount; }
    Index64 getVoxelCount() const { return mVoxelCount; }

protected:

    /// This is a base class so the constructor is protected
    BasePointScatter(PointAccessorType& points,
                     RandomGenerator& randGen,
                     InterruptType* interrupt = NULL)
        : mPoints(points)
        , mInterrupter(interrupt)
        , mPointCount(0)
        , mVoxelCount(0)
        , mInterruptCount(0)
        , mRandomGen(randGen)
    {
    }

    PointAccessorType&        mPoints;
    InterruptType*            mInterrupter;
    Index64                   mPointCount;
    Index64                   mVoxelCount;
    Index64                   mInterruptCount;
    RandomGenerator&          mRandomGen;
    boost::uniform_01<double> mRandom;

    inline void start(const char* name)
    {
        if (mInterrupter) mInterrupter->start(name);
    }

    inline void end()
    {
        if (mInterrupter) mInterrupter->end();
    }

    inline bool interrupt()
    {
        //only check interrupter for every 32'th call
        return !(mInterruptCount++ & ((1<<5)-1)) && util::wasInterrupted(mInterrupter);
    }

    inline double getRand() { return mRandom(mRandomGen); }

    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &pos, const openvdb::Vec3R &delta)
    {
        mPoints.add(grid.indexToWorld(pos + delta));
        ++mPointCount;
    }
    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &dmin)
    {
        const openvdb::Vec3R p(this->getRand(), this->getRand(), this->getRand());
        this->addPoint(grid, dmin, p);
    }
    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &dmin, const openvdb::Coord &size)
    {
        const openvdb::Vec3R p(size.x()*this->getRand(),
                               size.y()*this->getRand(),
                               size.z()*this->getRand());
        this->addPoint(grid, dmin, p);
    }
};// class BasePointScatter

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
