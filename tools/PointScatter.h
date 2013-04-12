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

#ifndef OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/util/NullInterrupter.h>
#include <boost/random/uniform_01.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

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
class UniformPointScatter
{
public:
    UniformPointScatter(PointAccessorType& points,
                        int pointCount,
                        RandomGenerator& randGen,
                        InterruptType* interrupt = NULL):
        mPoints(points),
        mInterrupter(interrupt),
        mPointCount(pointCount),
        mPointsPerVolume(0.0f),
        mVoxelCount(0),
        mRandomGen(randGen)
    {
    }
    UniformPointScatter(PointAccessorType& points,
                        float pointsPerVolume,
                        RandomGenerator& randGen,
                        InterruptType* interrupt = NULL):
        mPoints(points),
        mInterrupter(interrupt),
        mPointCount(0),
        mPointsPerVolume(pointsPerVolume),
        mVoxelCount(0),
        mRandomGen(randGen)
    {
    }

    /// This is the main functor method implementing the actual scattering of points.
    template<typename GridT>
    void operator()(const GridT& grid)
    {
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) return;//throw std::runtime_error("No voxels in which to scatter points!");
        const openvdb::Index64 voxelId = mVoxelCount - 1;
        const openvdb::Vec3d dim = grid.voxelSize();
        if (mPointsPerVolume>0) {
            if (mInterrupter) mInterrupter->start("Uniform scattering with fixed point density");
            mPointCount = int(mPointsPerVolume * dim[0]*dim[1]*dim[2] * mVoxelCount);
        } else if (mPointCount>0) {
            if (mInterrupter) mInterrupter->start("Uniform scattering with fixed point count");
            mPointsPerVolume = mPointCount/float(dim[0]*dim[1]*dim[2] * mVoxelCount);
        } else {
            return;
            //throw std::runtime_error("Invalid point count and point density");
        }
        openvdb::CoordBBox bbox;
        /// build sorted multi-map of random voxel-ids to contain a point
        std::multiset<openvdb::Index64> mVoxelSet;
        for (int i=0, chunks=100000; i<mPointCount; i += chunks) {
            if (util::wasInterrupted(mInterrupter)) return;
            //    throw std::runtime_error("processing was interrupted");
            //}
            /// @todo Multi-thread the generation of mVoxelSet
            for (int j=i, end=std::min(i+chunks, mPointCount); j<end; ++j) {
                mVoxelSet.insert(openvdb::Index64(voxelId*getRand()));
            }
        }
        std::multiset<openvdb::Index64>::iterator voxelIter =
            mVoxelSet.begin(), voxelEnd = mVoxelSet.end();
        typename GridT::ValueOnCIter valueIter = grid.cbeginValueOn();
        mPointCount = 0;
        size_t interruptCount = 0;
        for (openvdb::Index64 i=valueIter.getVoxelCount(); voxelIter != voxelEnd; ++voxelIter) {
            //only check interrupter for every 32'th particle
            if (!(interruptCount++ & (1<<5)-1) && util::wasInterrupted(mInterrupter)) return;
            while ( i <= *voxelIter ) {
                ++valueIter;
                i += valueIter.getVoxelCount();
            }
            if (valueIter.isVoxelValue()) {// a majorty is expected to be voxels
                const openvdb::Coord min = valueIter.getCoord();
                const openvdb::Vec3R dmin(min.x()-0.5, min.y()-0.5, min.z()-0.5);
                this->addPoint(grid, dmin);
            } else {// tiles contain multiple (virtual) voxels
                valueIter.getBoundingBox(bbox);
                const openvdb::Coord size(bbox.extents());
                const openvdb::Vec3R dmin(bbox.min().x()-0.5,
                                          bbox.min().y()-0.5,
                                          bbox.min().z()-0.5);
                this->addPoint(grid, dmin, size);
            }
        }
        if (mInterrupter) mInterrupter->end();
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Uniformely scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\" corresponding to "
           << mPointsPerVolume << " points per volume." << std::endl;
    }

    int getPointCount() const { return mPointCount; }
    float getPointsPerVolume() const { return mPointsPerVolume; }
    openvdb::Index64 getVoxelCount() const { return mVoxelCount; }

private:
    PointAccessorType&        mPoints;
    InterruptType*            mInterrupter;
    int                       mPointCount;
    float                     mPointsPerVolume;
    openvdb::Index64          mVoxelCount;
    RandomGenerator&          mRandomGen;
    boost::uniform_01<double> mRandom;

    double getRand() { return mRandom(mRandomGen); }

    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &pos, const openvdb::Vec3R &delta)
    {
        mPoints.add(grid.indexToWorld(pos + delta));
        ++mPointCount;
    }
    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &dmin)
    {
        this->addPoint(grid, dmin, openvdb::Vec3R(getRand(),getRand(),getRand()));
    }
    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &dmin, const openvdb::Coord &size)
    {
        const openvdb::Vec3R d(size.x()*getRand(),size.y()*getRand(),size.z()*getRand());
        this->addPoint(grid, dmin, d);
    }
}; // class UniformPointScatter


/// @brief Non-uniform scatters of point in the active voxels.
/// The local point count is implicitly defined as a product of
/// of a global density and the local voxel (or tile) value.
///
/// @note This scattering technique can be significantly slower
/// than a uniform scattering since its computational complexity
/// is proportional to the active voxel (and tile) count.
template<typename PointAccessorType,
         typename RandomGenerator,
         typename InterruptType = openvdb::util::NullInterrupter>
class NonUniformPointScatter
{
public:
    NonUniformPointScatter(PointAccessorType& points,
                           float pointsPerVolume,
                           RandomGenerator& randGen,
                           InterruptType* interrupt = NULL):
        mPoints(points),
        mInterrupter(interrupt),
        mPointCount(0),
        mPointsPerVolume(pointsPerVolume),//note this is NOT the local point density
        mVoxelCount(0),
        mRandomGen(randGen)
    {
    }

    /// This is the main functor method implementing the actual scattering of points.
    template<typename GridT>
    void operator()(const GridT& grid)
    {
        mVoxelCount = grid.activeVoxelCount();
        if (mVoxelCount == 0) throw std::runtime_error("No voxels in which to scatter points!");
        if (mInterrupter) mInterrupter->start("Non-uniform scattering with local point density");
        const openvdb::Vec3d dim = grid.voxelSize();
        const double volumePerVoxel = dim[0]*dim[1]*dim[2],
                     pointsPerVoxel = mPointsPerVolume * volumePerVoxel;
        openvdb::CoordBBox bbox;
        size_t interruptCount = 0;
        for (typename GridT::ValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
            //only check interrupter for every 32'th active value
            if (!(interruptCount++ & (1<<5)-1) && util::wasInterrupted(mInterrupter)) return;
            const double d = (*iter) * pointsPerVoxel * iter.getVoxelCount();
            const int n = int(d);
            if (iter.isVoxelValue()) { // a majorty is expected to be voxels
                const openvdb::Coord min = iter.getCoord();
                const openvdb::Vec3R dmin(min.x()-0.5, min.y()-0.5, min.z()-0.5);
                for (int i = 0; i < n; ++i) this->addPoint(grid, dmin);
                if (getRand() < (d - n)) this->addPoint(grid, dmin);
            } else { // tiles contain multiple (virtual) voxels
                iter.getBoundingBox(bbox);
                const openvdb::Coord size(bbox.extents());
                const openvdb::Vec3R dmin(bbox.min().x()-0.5,
                                          bbox.min().y()-0.5,
                                          bbox.min().z()-0.5);
                for (int i = 0; i < n; ++i) this->addPoint(grid, dmin, size);
                if (getRand() < (d - n)) this->addPoint(grid, dmin, size);
            }
        }//loop over the active values
        if (mInterrupter) mInterrupter->end();
    }

    // The following methods should only be called after the
    // the operator() method was called
    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Non-uniformely scattered " << mPointCount << " points into " << mVoxelCount
           << " active voxels in \"" << name << "\"." << std::endl;
    }

    int   getPointCount() const { return mPointCount; }
    openvdb::Index64  getVoxelCount() const { return mVoxelCount; }
    
private:
    PointAccessorType&        mPoints;
    InterruptType*            mInterrupter;
    int                       mPointCount;
    float                     mPointsPerVolume;
    openvdb::Index64          mVoxelCount;
    RandomGenerator&          mRandomGen;
    boost::uniform_01<double> mRandom;

    double getRand() { return mRandom(mRandomGen); }

    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &pos, const openvdb::Vec3R &delta)
    {
        mPoints.add(grid.indexToWorld(pos + delta));
        ++mPointCount;
    }
    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &dmin)
    {
        this->addPoint(grid, dmin, openvdb::Vec3R(getRand(),getRand(),getRand()));
    }
    template <typename GridT>
    inline void addPoint(const GridT &grid, const openvdb::Vec3R &dmin, const openvdb::Coord &size)
    {
        const openvdb::Vec3R d(size.x()*getRand(),size.y()*getRand(),size.z()*getRand());
        this->addPoint(grid, dmin, d);
    }
    
}; // class NonUniformPointScatter

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POINT_SCATTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
