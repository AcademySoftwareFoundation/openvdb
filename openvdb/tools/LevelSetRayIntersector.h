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
/// @file LevelSetRayIntersector.h
///
/// @brief Intersects a ray with a narrow-band level set.
///
/// @author Ken Museth
///
/// @todo Add analytical intersection test, i.e. solve cubic equation.
///
/// @note See the TestLevelSetRayIntersector unit-test for applications.

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
    
/// Helper class that implements hierarchical Digital Differential Analysers   
template <typename GridT, int NodeLevel> struct HDDA;

/// @brief This a level-set-ray intersector base on iterative linear
/// interpolation along the direction of the ray. Thus it is
/// approximate in the sense that it only detects an intersection when
/// the voxelized ray encounters a zero-crossing. The number of linear
/// iterations can be defined with a template parameter, but the
/// default value has proven fairly accurate and fast.
///
/// @warning Since this class internally stores a ValueAccessor for
/// improved random access it is NOT thread-safe so make sure to
/// declare an instance per thread. This of course also means the
/// overhead of allocating an instance should (if possible) be
/// amortized by as many ray-intersections as possible.
///
/// @note More iterations are not guaranteed to give better results.    
template<typename GridT, int Iterations = 0, typename RealT = double>    
class LinearIntersector
{
public:
    typedef typename GridT::ValueType     ValueT;
    typedef typename GridT::ConstAccessor AccessorT;
    typedef math::BoxStencil<GridT>       StencilT;
    typedef typename StencilT::Vec3Type   Vec3T;

    /// @brief Constructor from a grid.
    LinearIntersector(const GridT& grid) : mStencil(grid), mThreshold(2*grid.voxelSize()[0])
    {
    }

    /// @brief Call this method before the ray traversal starts
    template <typename RayT>
    void init(const RayT& ray)
    {
        mTime = mT[0] = mT[1] = ray.t0();
        const Vec3T pos = ray.start();
        const Coord ijk = Coord::floor(pos);
        ValueT V;
        if (mStencil.accessor().probeValue(ijk, V)) {//inside narrow band?
            mStencil.moveTo(ijk,  V);
            const Vec3T uvw = pos - Vec3T(ijk[0], ijk[1], ijk[2]);
            V = mStencil.interpolation(uvw);
        }
        mV[0] = mV[1] = V;
    }

    /// @brief Return @c true if an intersection is detected. Only then
    /// can the return value from time() be trusted.
    template <typename RayT>
    bool operator()(const Coord& ijk, RealT time, const RayT& ray)
    {
        ValueT V;
        if (mStencil.accessor().probeValue(ijk, V) &&//inside narrow band?
            math::Abs(V)<mThreshold) {// close to zero-crossing?
            const Vec3T xyz(ijk[0], ijk[1], ijk[2]);
            mT[1] = time;
            mStencil.moveTo(ijk,  V);//update 7 remaining stencil points
            Vec3T uvw = ray(mT[1]) - xyz;//normalized voxel coordinates
            mV[1] = mStencil.interpolation(uvw);
            if (math::ZeroCrossing(mV[0], mV[1])) {
                mTime = this->linearInterp();
                OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
                if (Iterations > 0) {//resolved at compile-time
                    for (int n=0; n<Iterations; ++n) {//resolved at compile-time
                        uvw = ray(mTime) - xyz;//normalized
                        V = mStencil.interpolation(uvw);
                        if (math::ZeroCrossing(mV[0], V)) {
                            mV[1] = V;
                            mT[1] = mTime;
                        } else {
                            mV[0] = V;
                            mT[0] = mTime;
                        }
                        mTime = this->linearInterp();
                    }
                }
                OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
                return true;
            }
            
            mT[0] = mT[1];
            mV[0] = mV[1];
        } 
        return false;
    }
   
    /// @brief Return a const reference to a ValueAccessor
    const AccessorT& accessor() const { return mStencil.accessor(); }

    /// @brief Return the time of intersection.
    /// @warning Only trust the return value AFTER an intersection was detected.
    RealT time() const { return mTime; }

private:

    inline RealT linearInterp()
    {
        assert(math::isApproxLarger(mT[1], mT[0], 1e-6));
        return mT[0]+(mT[1]-mT[0])*mV[0]/(mV[0]-mV[1]);
    }
    
    StencilT mStencil;
    RealT    mTime;
    ValueT   mV[2];
    RealT    mT[2];
    ValueT   mThreshold;
};    
    

//===============================================================   

/// @brief Intersect a ray with a narrow-band level set.
///
/// @note Performs hierarchical tree node and voxel traversal
template<typename GridT,
         int NodeLevel = GridT::TreeType::RootNodeType::ChildNodeType::LEVEL,
         typename RayT = math::Ray<double> >
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
    ///
    /// @param grid Level set grid to intersect rays against
    LevelSetRayIntersector(const GridT& grid): mGrid(&grid), mVoxelSize(grid.voxelSize()[0])
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
        if (!grid.tree().evalLeafBoundingBox(mBBox)) {
            OPENVDB_THROW(RuntimeError, "LevelSetRayIntersector does not supports empty grids");
        }
    }
    /// @brief Return @c true if the index space ray intersects the level set.
    /// @param iRay Ray represented in index space.
    /// @param ijk  If an intersection was found it is assigned the
    ///             intersection point in index space. Otherwise it is unchanged.
    /// @param tester The tester for the actual ray-level-set intersection
    /// @note The index space Ray is passed by value since it's internally
    /// clipped against the index-space bounding-box of the grid.
    template <typename TesterT>
    bool intersectsIS(RayType iRay, TesterT& tester, Vec3Type& ijk) const
    {
        if (!iRay.clip(mBBox)) return false;//missed bbox
        tester.init(iRay);
        if (!HDDA<TreeT, NodeLevel>::test(iRay, tester)) return false;//missed level set
        ijk = iRay(tester.time());
        return true;
    }
    /// @brief Return @c true if the world space ray intersects the level set.
    /// @param wRay   Ray represented in world space.
    /// @param tester The tester for the actual ray-level-set intersection
    /// @param xyz    If an intersection was found it is assigned the
    ///               intersection point in world space. Else it is unchanged.
    template <typename TesterT>
    bool intersectsWS(const RayType& wRay, TesterT& tester, Vec3Type& xyz) const
    {
        RayType iRay = wRay.applyInverseMap(*(mGrid->transform().baseMap()));
        Vec3Type ijk;
        if (!this->intersectsIS(iRay, tester, ijk)) return false;
        xyz = mGrid->indexToWorld(ijk);
        return true;
    }

    const GridT& grid() const { return *mGrid; }

private:

    const GridT*    mGrid;
    const ValueT    mVoxelSize;
    math::CoordBBox mBBox;
    
};// LevelSetRayIntersector

/// Helper class that implements Hierarchical Digital Differential Analyzers.    
template <typename TreeT, int NodeLevel>
struct HDDA
{
    typedef typename TreeT::RootNodeType::NodeChainType ChainT;
    typedef typename boost::mpl::at<ChainT, boost::mpl::int_<NodeLevel> >::type NodeT;
    
    template <typename RayT, typename TesterT>
    static bool test(const RayT& ray, TesterT& tester)
    {
        math::DDA<RayT, NodeT::TOTAL> dda(ray);
        do {
            if (const NodeT* node =
                tester.accessor().template probeConstNode<NodeT>(dda.voxel())) {
                RayT subRay = ray;
                subRay.clip(node->getNodeBoundingBox());
                if (HDDA<TreeT, NodeLevel-1>::test(subRay, tester)) return true;
            }
            dda.step();
        } while (ray.test(dda.time()));
        return false;
    }
};

/// @brief Specialization of Hierarchical Digital Differential Analyzer
/// class that intersects against the voxels.    
template <typename TreeT>
struct HDDA<TreeT, -1>     
{
    template <typename RayT, typename TesterT>
    static bool test(const RayT& ray, TesterT& tester)
    {
        math::DDA<RayT, 0> dda(ray);
        do {
            if (tester(dda.voxel(), dda.time(), ray)) return true;
            dda.step();
        } while(ray.test(dda.time()));
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
