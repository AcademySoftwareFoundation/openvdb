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
/// @brief Accelerated intersection of a ray with a narrow-band level
/// set or a generic (e.g. density) volume. This will of course be
/// useful for respectively surface and volume rendering.
///
/// @details This file defines two main classes,
/// LevelSetRayIntersector and VolumeRayIntersector, as well as the
/// three support classes LevelSetHDDA, VolumeHDDA and LinearSearchImpl.
/// The LevelSetRayIntersector is templated on the LinearSearchImpl class
/// and calls instances of the LevelSetHDDA class. The reason to split
/// level set ray intersection into three classes is twofold. First
/// LevelSetRayIntersector defines the public API for client code and
/// LinearSearchImpl defines the actual algorithm used for the 
/// ray level-set intersection. In other words this design will allow
/// for the public API to be fixed while the intersection algorithm
/// can change without resolving to (slow) virtual methods. Second,
/// LevelSetHDDA, which implements a hierarchical Differential Digital
/// Analyzer, relies on partial template specialization, so it has to
/// be a standalone class (as opposed to a member class of
/// LevelSetRayIntersector). The VolumeRayIntersector is conceptually
/// much simpler then the LevelSetRayIntersector, and hence it only
/// depends on VolumeHDDA that implements the hierarchical
/// Differential Digital Analyzer.


#ifndef OPENVDB_TOOLS_RAYINTERSECTOR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_RAYINTERSECTOR_HAS_BEEN_INCLUDED

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
// specialized for ray intersections with level sets
template <typename GridT, int NodeLevel> struct LevelSetHDDA;


/// Helper class that implements hierarchical Digital Differential Analysers
/// specialized for ray intersections with density (vs level set surfaces)    
template <typename GridT, int NodeLevel> struct VolumeHDDA;

// Helper class that implements the actual search of the zero-crossing
// of the level set along the direction of a ray. This particular
// implementation uses iterative linear search.   
template<typename GridT, int Iterations = 0, typename RealT = double>
class LinearSearchImpl;


//////////////////////////////////////// LevelSetRayIntersector ////////////////////////////////////////


/// @brief This class provides the public API for intersecting a ray
/// with a narrow-band level set.
///
/// @details It wraps an SearchImplT with a simple public API and
/// performs the actual hierarchical tree node and voxel traversal.
///
/// @warning Use the (default) copy-constructor to make sure each
/// computational thread has their own instance of this class. This is
/// important since the SearchImplT contains a ValueAccessor that is
/// not thread-safe. However copying is very efficient. 
///
/// @see tools/RayTracer.h for examples of intended usage.
///
/// @todo Add TrilinearSearchImpl, as an alternative to LinearSearchImpl,
/// that performs analytical 3D trilinear intersection tests, i.e., solves
/// cubic equations. This is slower but also more accurate than the 1D
/// linear interpolation in LinearSearchImpl.
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


//////////////////////////////////////// VolumeRayIntersector ////////////////////////////////////////


/// @brief This class provides the public API for intersecting a ray
/// with a generic (e.g. density) volume.
/// @details Internally it performs the actual hierarchical tree node traversal.
/// @warning Use the (default) copy-constructor to make sure each
/// computational thread has their own instance of this class. This is
/// important since it contains a ValueAccessor that is
/// not thread-safe and a CoordBBox of the active voxels that should
/// not be re-computed for each thread. However copying is very efficient.
/// @par Example:
/// @code
/// // Create an instance for the master thread
/// VolumeRayIntersector inter(grid);
/// // For each additional thread use the copy contructor. This
/// // amortizes the overhead of computing the bbox of the active voxels!
/// VolumeRayIntersector inter2(inter);
/// // Before each ray-traversal set the index ray.      
/// iter.setIndexRay(ray);
/// // or world ray
/// iter.setWorldRay(ray);    
/// // Now you can begin the ray-marching using consecutive calls to VolumeRayIntersector::march
/// double t0=0, t1=0;// note the entry and exit times are with respect to the INDEX ray    
/// while ( int n = inter.march(t0, t1) ) {// perform line-integration between t0 and t1
///   if (n == 1) {//hit a tile so the value between t0 and t1 is constant
///   } else {//n == 2 so we hit a leaf node and the value between t0 and t1 is unknown
/// }}
/// @endcode
template<typename GridT,
         int NodeLevel = GridT::TreeType::RootNodeType::ChildNodeType::LEVEL,
         typename RayT = math::Ray<Real> >
class VolumeRayIntersector
{
public:
    typedef GridT                         GridType;
    typedef RayT                          RayType;
    typedef typename GridT::TreeType      TreeT;

    BOOST_STATIC_ASSERT( NodeLevel >= 0 && NodeLevel < int(TreeT::DEPTH)-1);

    /// @brief Constructor
    /// @param grid Generic grid to intersect rays against.
    /// @warning In the near future we will add support for grids with frustrum transforms. 
    VolumeRayIntersector(const GridT& grid): mGrid(&grid), mAccessor(grid.tree())
    {
        if (!grid.hasUniformVoxels() ) {
            OPENVDB_THROW(RuntimeError,
                          "VolumeRayIntersector only supports uniform voxels!");
        }
        if ( grid.empty() ) {
            OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
        }
        grid.tree().root().evalActiveBoundingBox(mBBox, /*visit individual voxels*/false); 
       
        mBBox.max().offset(1);//padding so the bbox of a node becomes (origin,origin + node_dim)
    }
    
    /// @brief Return @c false if the index ray misses the bbox of the grid.
    /// @param iRay Ray represented in index space.
    /// @warning Call this method (or setWorldRay) before the ray
    /// traversal starts and use the return value to decide if further
    /// marching is required. 
    inline bool setIndexRay(const RayT& iRay)
    {
        mRay = iRay;
        const bool hit = mRay.clip(mBBox);
        if (hit) mTmax = mRay.t1();
        return hit;
    }

    /// @brief Return @c false if the world ray misses the bbox of the grid.
    /// @param wRay Ray represented in world space.
    /// @warning Call this method (or setIndexRay) before the ray
    /// traversal starts and use the return value to decide if further
    /// marching is required. 
    /// @details Since hit times are computed with repect to the ray
    /// represented in index space of the current grid, it is
    /// recommended that either the client code uses getIndexPos to
    /// compute index position from hit times or alternatively keeps
    /// an instance of the index ray and instead uses setIndexRay to
    /// initialize the ray. 
    inline bool setWorldRay(const RayT& wRay)
    {
        return this->setIndexRay(wRay.worldToIndex(*mGrid));
    }

    /// @brief Return 0 if not hit was detected. A return value of 1
    /// means it hit an active tile, and a return value of 2 means it
    /// hit a LeafNode. Only when a hit is detected are t0 and t1
    /// updated with the corresponding entry and exit times along the
    /// INDEX ray!
    /// @param t0 If the return value > 0 this is the time of the first hit.
    /// @param t1 If the return value > 0 this is the time of the second hit.
    /// @warning t0 and t1 are computed with repect to the ray represented in
    /// index space of the current grid, not world space!
    inline int march(Real& t0, Real& t1)
    {
        const int n = mRay.test() ? VolumeHDDA<TreeT, NodeLevel>::test(*this) : 0;
        if (n>0) {
            t0 = mRay.t0();
            t1 = mRay.t1();
        }
        mRay.setTimes(mRay.t1() + 1e-9, mTmax);
        return n;
    }

    /// @brief Return the floating-point index position along the
    /// current index ray at the specified time.
    inline Vec3R getIndexPos(Real time) const { return mRay(time); }

    /// @brief Return the floating-point world position along the
    /// current index ray at the specified time.
    inline Vec3R getWorldPos(Real time) const { return mGrid->indexToWorld(mRay(time)); }
    
private:
    
    inline void setRange(Real t0, Real t1) { mRay.setTimes(t0, t1); }

    /// @brief Return a const reference to the ray.
    inline const RayT& ray() const { return mRay; }

    /// @brief Return true if a node of the the specified type exists at ijk.
    template <typename NodeT>
    inline bool hasNode(const Coord& ijk)
    {
        return mAccessor.template probeConstNode<NodeT>(ijk) != NULL;
    }

    bool isValueOn(const Coord& ijk) { return mAccessor.isValueOn(ijk); }

    template<typename, int> friend struct VolumeHDDA;

    typedef typename GridT::ConstAccessor AccessorT;
    
    const GridT*    mGrid;
    AccessorT       mAccessor;
    RayT            mRay;
    Real            mTmax;
    math::CoordBBox mBBox;
    
};// VolumeRayIntersector
    

//////////////////////////////////////// LinearSearchImpl ////////////////////////////////////////

    
/// @brief Implements linear iterative search for a zero-crossing of
/// the level set along along the direction of the ray.
///
/// @note Since this class is used internally in
/// LevelSetRayIntersector (define above) and LevelSetHDDA (defined below) 
/// client code should never interact directly with its API. This also
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
        : mStencil(grid), mThreshold(2*grid.voxelSize()[0])
      {
          if ( grid.empty() ) {
              OPENVDB_THROW(RuntimeError, "LinearSearchImpl does not supports empty grids");
          }
          grid.tree().root().evalActiveBoundingBox(mBBox, /*visit individual voxels*/false);
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
        mRay = wRay.worldToIndex(mStencil.grid());
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

    /// @brief Initiate the local voxel intersection test.
    /// @warning Make sure to call this method before the local voxel intersection test. 
    inline void init(RealT t0)
    {
        mT[0] = t0;
        mV[0] = this->interpValue(t0);
    }

    inline void setRange(RealT t0, RealT t1) { mRay.setTimes(t0, t1); }

    /// @brief Return a const reference to the ray.
    inline const RayT& ray() const { return mRay; }

    /// @brief Return true if a node of the the specified type exists at ijk.
    template <typename NodeT>
    inline bool hasNode(const Coord& ijk)
    {
        return mStencil.accessor().template probeConstNode<NodeT>(ijk) != NULL;
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

    template<typename, int> friend struct LevelSetHDDA;
    
    RayT            mRay;
    StencilT        mStencil;
    RealT           mTime;
    ValueT          mV[2];
    RealT           mT[2];
    ValueT          mThreshold;
    math::CoordBBox mBBox;
};// LinearSearchImpl

//////////////////////////////////////// LevelSetHDDA ////////////////////////////////////////


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


//////////////////////////////////////// VolumeHDDA ////////////////////////////////////////

    
/// Helper class that implements Hierarchical Digital Differential Analyzers
/// for ray intersections against a generic volume.
template <typename TreeT, int NodeLevel>
struct VolumeHDDA
{
    typedef typename TreeT::RootNodeType::NodeChainType ChainT;
    typedef typename boost::mpl::at<ChainT, boost::mpl::int_<NodeLevel> >::type NodeT;
    
    template <typename TesterT>
    static int test(TesterT& tester)
    {
        math::DDA<typename TesterT::RayType, NodeT::TOTAL> dda(tester.ray());
        do {
            if (tester.template hasNode<NodeT>(dda.voxel())) {//child node
                tester.setRange(dda.time(), dda.next());
                int hit = VolumeHDDA<TreeT, NodeLevel-1>::test(tester);
                if (hit > 0) return hit;
            } else if (tester.isValueOn(dda.voxel())) {//active tile
                tester.setRange(dda.time(), dda.next());
                return 1;//active tile
            }
        } while (dda.step());
        return 0;//no hits
    }
};

/// @brief Specialization of Hierarchical Digital Differential Analyzer
/// class that intersects against the voxels of a generic volume.    
template <typename TreeT>
struct VolumeHDDA<TreeT, -1>     
{
    template <typename TesterT>
    static int test(TesterT&) { return 2; }//hit leaf so don't traverse voxels.
};    

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_RAYINTERSECTOR_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
