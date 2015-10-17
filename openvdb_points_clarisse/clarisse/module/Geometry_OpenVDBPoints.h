///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
/// @file Geometry_OpenVDBPoints.h
///
/// @author Dan Bailey
///
/// @brief  OpenVDB Points Geometry Primitive
///


#ifndef OPENVDB_CLARISSE_GEOMETRY_OPENVDBPOINTS_HAS_BEEN_INCLUDED
#define OPENVDB_CLARISSE_GEOMETRY_OPENVDBPOINTS_HAS_BEEN_INCLUDED

#include <boost/ptr_container/ptr_vector.hpp>

#include <app_base.h>
#include <geometry_object.h>
#include <geometry_raytrace_ctx.h>

#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>

#include <ctx_eval.h>


////////////////////////////////////////


namespace internal {

// interpolate between two bounding boxes using weight w
// weight is expected to be between 0 and 1
void interpolate(   openvdb::BBoxd& bbox, const double& w,
                    const openvdb::BBoxd& bbox0, const openvdb::BBoxd& bbox1);

}


////////////////////////////////////////


class Geometry_OpenVDBPoints : public GeometryObject
{
public:
    // BVHTree3 is a hard-coded three-level BVH tree
    //     level 1 is 8x8x8 (node is equal to a leaf node)
    //     level 2 is 4x4x4 (each node is 1/8 of a leaf)
    //     level 3 is 2x2x2 (each node is 1/64 of a leaf)
    template <typename T>
    struct BVHTree3
    {
        static const openvdb::Index DIM = openvdb::tools::PointDataTree::LeafNodeType::DIM;
        static const openvdb::Index DIMSQ = DIM*DIM;

        T node1;
        T node2[DIM];
        T node3[DIMSQ];
    };


    ////////////////////////////////////////


    // convert a coord into a BVH index (depending on level)
    template <unsigned level>
    static unsigned coordToIndex(const openvdb::Coord& ijk);

    // LeafBVHTree stores two BVH trees, one for each time sample (t=0, t=1)
    // @todo: extend to support more than two samples
    class LeafBVHTree
    {
    public:
        static const openvdb::Index DIM = openvdb::tools::PointDataTree::LeafNodeType::DIM;

        // expand the bbox for a specific coord and time based on position and radius
        template <unsigned isTo>
        void expand(const openvdb::Coord& ijk, const openvdb::Vec3f& position, const float radius);

        // propogate bboxes from level 3 up to levels 1 and 2
        void propagate(const openvdb::Coord& ijk);

        // retrieve the bounding box for a specific node based on level and index
        template <unsigned level>
        void interpolate(openvdb::BBoxd& bbox, const double& t, const unsigned index = 0) const;

        // determine if the ray hits the bbox of a specific node based on level and index
        template <unsigned level>
        bool hit(const openvdb::math::Ray<double>& ray, const double& t, const unsigned index = 0) const;

        // convenience function to retrieve bbox at level 1
        const openvdb::BBoxd& bbox1(const bool isT0) const;

    private:
        BVHTree3<openvdb::BBoxd> mTreeT0;
        BVHTree3<openvdb::BBoxd> mTreeT1;
    };


    ////////////////////////////////////////


    // Stores the LeafBVHTree and an intersection cache
    // Based on profiling, this caching scheme was generally found to be fastest:
    //     level 2 of the cache is pre-computed
    //     level 3 of the cache is cached on-demand
    class CachedBVHTree
    {
    public:
        static const openvdb::Index SIZE = BVHTree3<bool>::DIMSQ;

        explicit CachedBVHTree(const LeafBVHTree& tree):
            mTree(tree) { reset(); }

        // cache all ray hits for nodes at a specific level
        template <unsigned level>
        void computeHits(const openvdb::math::Ray<double>& ray, const double& t);

        // test whether the ray has hit the node at a specific coord and level
        template <unsigned level>
        bool hit(const openvdb::math::Ray<double>& ray, const double& t, const openvdb::Coord& ijk = openvdb::Coord());

    private:
        // cache a hit
        template <unsigned level>
        void storeHit(const unsigned index = 0);

        // zero the cache
        void reset();

        const LeafBVHTree& mTree;
        BVHTree3<bool> mIntersections;
        bool mCache[SIZE];
    };


    ////////////////////////////////////////


    // common Point Data typedefs

    typedef openvdb::tools::PointDataTree::LeafNodeType PointDataLeaf;
    typedef openvdb::tools::PointDataAccessor<openvdb::tools::PointDataTree> PointDataAccessor;

    // acceleration typedefs

    typedef boost::ptr_vector<openvdb::BBoxd> BBoxContainer;
    typedef boost::ptr_vector<LeafBVHTree> LeafBVHTreeContainer;


    ////////////////////////////////////////

    // Custom Methods

    ////////////////////////////////////////


    static Geometry_OpenVDBPoints* create(  const openvdb::tools::PointDataGrid::Ptr& grid,
                                            const bool override_radius,
                                            const double& radius,
                                            const double& fps,
                                            const double& motionBlurLength,
                                            const AppBase::MotionBlurDirectionMode& motionBlurDirection);

    Geometry_OpenVDBPoints( const openvdb::tools::PointDataGrid::Ptr& grid,
                            const openvdb::tools::AttributeSet::Descriptor& descriptor,
                            const std::string& velocityType,
                            const std::string& radiusType,
                            const bool overrideRadius,
                            const double& radius,
                            const double& timeSegment,
                            const double& timeOffset);

    void computeAccelerationStructures();


    ////////////////////////////////////////

    // Clarisse Geometry Methods
    // (note these methods that do not adhere to the OpenVDB Coding Style)

    ////////////////////////////////////////


    Geometry_OpenVDBPoints* get_copy() const;

    size_t get_memory_size() const;

    GMathBbox3d get_bbox() const;

    GMathBbox3d get_bbox_at(const CtxEval& evalContext,
                            const double& time) const;

    bool is_animated() const;

    const CoreBasicArray<CoreString>& get_shading_group_names() const;

    unsigned int get_primitive_count() const;
    unsigned int get_primitive_edge_count(const unsigned int& id) const;

    void compute_primitive_bbox(const CtxEval& evalContext,
                                const unsigned int& id,
                                GMathBbox3d& bbox) const;

    void compute_primitive_bbox_at( const CtxEval& evalContext,
                                    const unsigned int& id,
                                    const double& time,
                                    GMathBbox3d& bbox) const;

    unsigned int get_primitive_shading_group_index(const unsigned int& id) const { return 0; }

    void compute_fragment_sample (  const CtxEval &evalContext,
                                    const GeometryFragment &fragment,
                                    GeometrySample &sample) const;

    void intersect_primitive(   const CtxEval& evalContext,
                                const unsigned int& id,
                                GeometryRaytraceCtx& raytraceContext) const;

private:
    template <typename VelocityType, typename RadiusType>
    void intersect_typed_primitive( const CtxEval& eval_ctx,
                                    const unsigned int& id,
                                    GeometryRaytraceCtx& raytrace_ctx) const;

    inline void compute_uv( const GMathVec3d& pos,
                            double& u, double& v,
                            double& radius) const;

    // VDB grid (shared ptr is only used to hold onto ownership)
    openvdb::tools::PointDataGrid::Ptr m_grid;

    // cache leaves, descriptor, transform and base map
    std::vector<PointDataLeaf*> m_leaves;
    const openvdb::tools::AttributeSet::Descriptor& m_descriptor;
    const openvdb::math::Transform& m_transform;
    const openvdb::math::MapBase& m_baseMap;

    // attribute types
    const std::string m_velocityType;
    const std::string m_radiusType;

    // radius parameters
    const bool m_overrideRadius;
    const double m_radius;
    double m_radiusIndexSpace;

    // motion blur parameters
    const bool m_enableMotionBlur;
    const double m_timeSegment;
    const double m_overTimeSegment;
    const double m_timeOffset;

    // shading group names
    CoreArray<CoreString> m_shadingGroupNames;

    // acceleration data structures

    LeafBVHTreeContainer mBVHTrees;

    openvdb::BBoxd m_bboxT0;
    openvdb::BBoxd m_bboxT1;

    // count
    unsigned int m_primitiveCount;
}; // Geometry_OpenVDBPoints

#endif // OPENVDB_CLARISSE_GEOMETRY_OPENVDBPOINTS_HAS_BEEN_INCLUDED
