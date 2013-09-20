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
///
/// @file RayIntersector.h
///
/// @author Ken Museth
///
/// @brief Accelerated intersection of a ray with a narrow-band level set.
/// @todo Add acceleration ray-marching for volume rendering.
///
/// @details This file defines three classes that together perform accelerated
/// intersection of a ray with a narrow-band level set.  The main class is
/// LevelSetRayIntersector, which is templated on the LinearIntersector class
/// and calls instances of the LevelSetHDDA class.  The reason to split ray intersection
/// into three classes is twofold. First LevelSetRayIntersector defined the public 
/// API for client code and LinearIntersector defines the algorithm used for the 
/// ray-level-set intersection. In other words this design will allow
/// for the public API to be fixed while the intersection algorithm
/// can change without resolving to (slow) virtual methods. Second, LevelSetHDDA, 
/// which implements a hierarchical Differential Digital Analyzer,
/// relies on partial template specialization, so it has to be a standalone class
/// (as opposed to a member class of LevelSetRayIntersector).
///
/// @warning Make sure to assign a local copy of the
/// LevelSetRayIntersector to each computational thread. This is
/// important becauce it contains an instance of the LinearIntersector
/// which in turn is not thread-safe (due to a ValueAccessor among
/// other things). However copying is very efficient. 
///
/// @see tools/RayTracer.h for examples of intended usage.
///
/// @todo Add TrilinearIntersector, as an alternative to LinearIntersector,
/// that performs analytical 3D trilinear intersection tests, i.e., solves
/// cubic equations. This is slower than but more accurate than the 1D
/// linear interpolation in LinearIntersector.

#ifndef OPENVDB_TOOLS_LEVELSETRAYINTERSECTOR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETRAYINTERSECTOR_HAS_BEEN_INCLUDED

#include <openvdb/math/Ray.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <boost/utility.hpp>
#include <boost/type_traits/is_floating_point.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

// Helper class that implements hierarchical Digital Differential Analyzers
// specialized for ray intersections with level lets
template <typename GridT, int NodeLevel> struct LevelSetHDDA;

// Helper class that implements the actual search of the zero-crossing
// of the level set along the direction of a ray. This particular
// implementation uses iterative linear search.   
template<typename GridT, int Iterations = 0, typename RealT = double>
class LinearSearchImpl;


////////////////////////////////////////


/// @brief This class provides the public API for intersecting a ray
/// with a narrow-band level set.
/// @details It wraps an SearchImplT with a simple public API and
/// performs the actual hierarchical tree node and voxel traversal.
/// @warning Use the (default) copy-constructor to make sure each
/// computational thread has their own instance of this class. This is
/// important since the SearchImplT contains a ValueAccessor that is
/// not thread-safe. However copying is very efficient.     
template<typename GridT,
         typename SearchImplT = LinearSearchImpl<GridT>,
         int NodeLevel = GridT::TreeType::RootNodeType::ChildNodeType::LEVEL,
         typename RayT = math::Ray<Real> >
class LevelSetRayIntersector
{
public:
    typedef GridT                         GridType;
    typedef RayT                          RayType;
    typedef typename RayT::Vec3T          Vec3Type;
    typedef typename GridT::ValueType     ValueT;
    typedef typename GridT::TreeType      TreeT;

    BOOST_STATIC_ASSERT( NodeLevel >= -1 && NodeLevel < int(TreeT::DEPTH)-1);
    BOOST_STATIC_ASSERT(boost::is_floating_point<ValueT>::value);

    /// @brief Constructor
    /// @param grid level set grid to intersect rays against
    LevelSetRayIntersector(const GridT& grid): mTester(grid)
    {
        if (!grid.hasUniformVoxels() ) {
            OPENVDB_THROW(RuntimeError,
                          "LevelSetRayIntersector only supports uniform voxels!");
        }
        if (grid.getGridClass() != GRID_LEVEL_SET) {
            OPENVDB_THROW(RuntimeError,
                          "LevelSetRayIntersector only supports level sets!"
                          "\nUse Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
        }
    }
    
    /// @brief Return @c true if the index-space ray intersects the level set
    /// @param iRay ray represented in index space
    bool intersectsIS(const RayType& iRay) const
    {
        if (!mTester.setIndexRay(iRay)) return false;//missed bbox
        return LevelSetHDDA<TreeT, NodeLevel>::test(mTester);
    }
    
    /// @brief Return @c true if the index-space ray intersects the level set.
    /// @param iRay ray represented in index space.
    /// @param xyz  if an intersection was found it is assigned the
    ///             intersection point in index space, otherwise it is unchanged.
    bool intersectsIS(const RayType& iRay, Vec3Type& xyz) const
    {
        if (!mTester.setIndexRay(iRay)) return false;//missed bbox
        if (!LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getIndexPos(xyz);
        return true;
    }
    
    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    bool intersectsWS(const RayType& wRay) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        return LevelSetHDDA<TreeT, NodeLevel>::test(mTester);
    }
    
    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    /// @param world  if an intersection was found it is assigned the
    ///               intersection point in world space, otherwise it is unchanged
    bool intersectsWS(const RayType& wRay, Vec3Type& world) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        if (!LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getWorldPos(world);
        return true;
    }
   
    /// @brief Return @c true if the world-space ray intersects the level set.
    /// @param wRay   ray represented in world space.
    /// @param world  if an intersection was found it is assigned the
    ///               intersection point in world space, otherwise it is unchanged.
    /// @param normal if an intersection was found it is assigned the normal
    ///               of the level set surface in world space, otherwise it is unchanged.
    bool intersectsWS(const RayType& wRay, Vec3Type& world, Vec3Type& normal) const
    {
        if (!mTester.setWorldRay(wRay)) return false;//missed bbox
        if (!LevelSetHDDA<TreeT, NodeLevel>::test(mTester)) return false;//missed level set
        mTester.getWorldPosAndNml(world, normal);
        return true;
    }
    
private:

    mutable SearchImplT mTester;
    
};// LevelSetRayIntersector


////////////////////////////////////////

    
/// @brief Implements linear iterative search for a zero-crossing of
/// the level set along along the direction of the ray.
///
/// @note Since this class is used internally in
/// LevelSetRayIntersector (define above) and LevelSetHDDA (defined below) 
/// client code will never interact directly with its API. This also
/// explains why we are not concerned with the fact that several of
/// its methods are unsafe to call unless zero-crossings were
/// already detected. 
///    
/// @details It is approximate due to the limited number of iterations
/// which can can be defined with a template parameter. However the default value
/// has proven surprisingly accurate and fast. In fact more iterations
/// are not guaranteed to give significantly better results.
///
/// @warning Since this class internally stores a ValueAccessor it is NOT thread-safe,
/// so make sure to give each thread its own instance.  This of course also means that
/// the cost of allocating an instance should (if possible) be amortized over
/// as many ray intersections as possible.
template<typename GridT, int Iterations, typename RealT>
class LinearSearchImpl
{
public:
    typedef math::Ray<RealT>              RayT;
    typedef typename GridT::ValueType     ValueT;
    typedef typename GridT::ConstAccessor AccessorT;
    typedef math::BoxStencil<GridT>       StencilT;
    typedef typename StencilT::Vec3Type   Vec3T;

    /// @brief Constructor from a grid.
    LinearSearchImpl(const GridT& grid)
        : mRay(Vec3T(0,0,0), Vec3T(1,0,0)),//dummy ray
          mStencil(grid), mThreshold(2*grid.voxelSize()[0])
    {
        // Computing a BBOX of the leaf nodes is extremely fast, e.g. 15ms for the "crawler"
        if (!grid.tree().evalLeafBoundingBox(mBBox)) {
            OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
        }
    }

    /// @brief Return @c true if an intersection is detected.
    /// @param ijk Grid coordinate of the node origin or voxel being tested.
    /// @param time Time along the index ray being tested.
    /// @warning Only if and intersection is detected is it safe to
    /// call getIndexPos, getWorldPos and getWorldPosAndNml!
    inline bool operator()(const Coord& ijk, RealT time)
    {
        ValueT V;
        if (mStencil.accessor().probeValue(ijk, V) &&//inside narrow band?
            math::Abs(V)<mThreshold) {// close to zero-crossing?
            mT[1] = time;
            mV[1] = this->interpValue(time);
            if (math::ZeroCrossing(mV[0], mV[1])) {
                mTime = this->interpTime();
                OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
                for (int n=0; Iterations>0 && n<Iterations; ++n) {//resolved at compile-time
                    V = this->interpValue(mTime);
                    const int m = math::ZeroCrossing(mV[0], V) ? 1 : 0;
                    mV[m] = V;
                    mT[m] = mTime;
                    mTime = this->interpTime();
                }
                OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
                return true;
            }
            mT[0] = mT[1];
            mV[0] = mV[1];
        }
        return false;
    }

    /// @brief Initiate the local voxel intersection test.
    /// @warning Make sure to call this method before the local voxel intersection test. 
    inline void init(RealT t0)
    {
        mT[0] = t0;
        mV[0] = this->interpValue(t0);
    }
   
    /// @brief Return the time of intersection.
    /// @warning Only trust the return value AFTER an intersection was detected.
    //inline RealT time() const { return mTime; }

    inline void setRange(RealT t0, RealT t1) { mRay.setTimes(t0, t1); }

    /// @brief Return a const reference to the ray.
    inline const RayT& ray() const { return mRay; }

    /// @brief Return true if a node of the the specified type exists at ijk.
    template <typename NodeT>
    inline bool hasNode(const Coord& ijk)
    {
        return mStencil.accessor().template probeConstNode<NodeT>(ijk) != NULL;
    }

    /// @brief Return @c false the ray misses the bbox of the grid.
    /// @param iRay Ray represented in index space.
    /// @warning Call this method before the ray traversal starts.
    inline bool setIndexRay(const RayT& iRay)
    {
        mRay = iRay;
        return mRay.clip(mBBox);//did it hit the bbox
    }

    /// @brief Return @c false the ray misses the bbox of the grid.
    /// @param wRay Ray represented in world space.
    /// @warning Call this method before the ray traversal starts.
    inline bool setWorldRay(const RayT& wRay)
    {
        mRay = wRay.applyInverseMap(*(mStencil.grid().transform().baseMap()));
        return mRay.clip(mBBox);//did it hit the bbox
    }
    
    /// @brief Get the intersection point in index space.
    /// @param xyz The position in index space of the intersection.
    inline void getIndexPos(Vec3d& xyz) const { xyz = mRay(mTime); }

    /// @brief Get the intersection point in world space.
    /// @param xyz The position in world space of the intersection.
    inline void getWorldPos(Vec3d& xyz) const { xyz = mStencil.grid().indexToWorld(mRay(mTime)); }

    /// @brief Get the intersection point and normal in world space
    /// @param xyz The position in world space of the intersection.
    /// @param nml The surface normal in world space of the intersection.
    inline void getWorldPosAndNml(Vec3d& xyz, Vec3d& nml)
    {
        this->getIndexPos(xyz);
        mStencil.moveTo(xyz);
        nml = mStencil.gradient(xyz);
        nml.normalize();
        xyz = mStencil.grid().indexToWorld(xyz);
    }

private:
    
    inline RealT interpTime()
    {
        assert(math::isApproxLarger(mT[1], mT[0], 1e-6));
        return mT[0]+(mT[1]-mT[0])*mV[0]/(mV[0]-mV[1]);
    }

    inline RealT interpValue(RealT time)
    {
        const Vec3R pos = mRay(time);
        mStencil.moveTo(pos);
        return mStencil.interpolation(pos);
    }
    
    RayT            mRay;
    StencilT        mStencil;
    RealT           mTime;
    ValueT          mV[2];
    RealT           mT[2];
    ValueT          mThreshold;
    math::CoordBBox mBBox;
};// LinearSearchImpl

    
////////////////////////////////////////


/// @brief Helper class that implements Hierarchical Digital Differential Analyzers
/// and is specialized for ray intersections with level sets
template<typename TreeT, int NodeLevel>
struct LevelSetHDDA
{
    typedef typename TreeT::RootNodeType::NodeChainType ChainT;
    typedef typename boost::mpl::at<ChainT, boost::mpl::int_<NodeLevel> >::type NodeT;

    template <typename TesterT>
    static bool test(TesterT& tester)
    {
        math::DDA<typename TesterT::RayT, NodeT::TOTAL> dda(tester.ray());
        do {
            if (tester.template hasNode<NodeT>(dda.voxel())) {
                tester.setRange(dda.time(), dda.next());
                if (LevelSetHDDA<TreeT, NodeLevel-1>::test(tester)) return true;
            }
        } while(dda.step());
        return false;
    }
};

/// @brief Specialization of Hierarchical Digital Differential Analyzer
/// class that intersects a ray against the voxels of a level set
template<typename TreeT>
struct LevelSetHDDA<TreeT, -1>
{
    template <typename TesterT>
    static bool test(TesterT& tester)
    {
        math::DDA<typename TesterT::RayT, 0> dda(tester.ray());
        tester.init(dda.time());
        do { if (tester(dda.voxel(), dda.next())) return true; } while(dda.step());
        return false;
    }
};

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETRAYINTERSECTOR_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
