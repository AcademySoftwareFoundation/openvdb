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
/// @file Geometry_OpenVDBPoints.cc
///
/// @author Dan Bailey

#include <app_object.h>

#include <openvdb_points/tools/PointCount.h>

#include "Geometry_OpenVDBPoints.h"

using namespace openvdb;
using namespace openvdb::tools;


////////////////////////////////////////


namespace internal {

void interpolate(BBoxd& bbox, const double& w, const BBoxd& bbox0, const BBoxd& bbox1)
{
    assert(w >= 0.0);
    assert(w <= 1.0);

    const float w1 = w;
    const float w0 = 1.0f - w;

    bbox.min() = bbox0.min() * w0 + bbox1.min() * w1;
    bbox.max() = bbox0.max() * w0 + bbox1.max() * w1;
}

} // namespace internal


////////////////////////////////////////


template <unsigned level>
unsigned Geometry_OpenVDBPoints::coordToIndex(const Coord& ijk)
{
    static const Index DIM = PointDataTree::LeafNodeType::DIM;

    const unsigned split = (level == 3 ? 4 : level);

    const unsigned i = ijk.x() & (DIM-1u);
    const unsigned j = ijk.y() & (DIM-1u);
    const unsigned k = ijk.z() & (DIM-1u);

    const unsigned index = (i < (DIM/split)) * split * split + (j < (DIM/split)) * split + (k < (DIM/split));

    return index;
}


template <unsigned isT0>
void Geometry_OpenVDBPoints::LeafBVHTree::expand(   const Coord& ijk,
                                                    const Vec3f& position, const float radius)
{
    const unsigned index = coordToIndex<3>(ijk);

    BBoxd& bbox = isT0 ? mTreeT0.node3[index] : mTreeT1.node3[index];

    bbox.expand(position - radius);
    bbox.expand(position + radius);
}


void Geometry_OpenVDBPoints::LeafBVHTree::propagate(const Coord& ijk)
{
    const unsigned index3 = coordToIndex<3>(ijk);

    const BBoxd& bboxT0 = mTreeT0.node3[index3];
    const BBoxd& bboxT1 = mTreeT1.node3[index3];

    const unsigned index2 = coordToIndex<2>(ijk);

    mTreeT0.node2[index2].expand(bboxT0);
    mTreeT1.node2[index2].expand(bboxT1);

    mTreeT0.node1.expand(bboxT0);
    mTreeT1.node1.expand(bboxT1);
}


template <unsigned level>
void Geometry_OpenVDBPoints::LeafBVHTree::interpolate(BBoxd& bbox, const double& t, const unsigned index) const
{
    if (level == 1)     internal::interpolate(bbox, t, mTreeT0.node1, mTreeT1.node1);
    if (level == 2)     internal::interpolate(bbox, t, mTreeT0.node2[index], mTreeT1.node2[index]);
    if (level == 3)     internal::interpolate(bbox, t, mTreeT0.node3[index], mTreeT1.node3[index]);
}


template <unsigned level>
bool Geometry_OpenVDBPoints::LeafBVHTree::hit(const math::Ray<double>& ray,
                                    const double& t, const unsigned index) const
{
    openvdb::BBoxd bbox;

    interpolate<level>(bbox, t, index);

    return ray.intersects(bbox);
}


const BBoxd& Geometry_OpenVDBPoints::LeafBVHTree::bbox1(const bool isT0) const
{
    return isT0 ? mTreeT0.node1 : mTreeT1.node1;
}


////////////////////////////////////////


template <unsigned level>
void Geometry_OpenVDBPoints::CachedBVHTree::storeHit(const unsigned index)
{
    if (level == 1)     mIntersections.node1 = true;
    if (level == 2)     mIntersections.node2[index] = true;
    if (level == 3)     mIntersections.node3[index] = true;
}


template <unsigned level>
void Geometry_OpenVDBPoints::CachedBVHTree::computeHits(const math::Ray<double>& ray, const double& t)
{
    const unsigned nodes =  (level == 3 ? BVHTree3<bool>::DIMSQ :
                            (level == 2 ? BVHTree3<bool>::DIM : 1));

    for (unsigned index = 0; index < nodes; index++) {
        if (mTree.hit<level>(ray, t, index))    storeHit<level>(index);
    }
}


template <unsigned level>
bool Geometry_OpenVDBPoints::CachedBVHTree::hit(const math::Ray<double>& ray, const double& t, const Coord& ijk)
{
    const unsigned index = coordToIndex<level>(ijk);

    // cache on-demand for level 3

    if (level == 3)
    {
        if (!mCache[index])
        {
            if (mTree.hit<level>(ray, t, index))    storeHit<level>(index);
            mCache[index] = true;
        }
    }

    if (level == 1)     return mIntersections.node1;
    if (level == 2)     return mIntersections.node2[index];
    if (level == 3)     return mIntersections.node3[index];

    return false;
}


void Geometry_OpenVDBPoints::CachedBVHTree::reset()
{
    for (unsigned i = 0; i < SIZE; i++)  mCache[i] = false;
}


////////////////////////////////////////


struct ComputeBBoxPerPrimitiveOp {

    typedef tree::LeafManager<PointDataTree> LeafManagerT;
    typedef LeafManagerT::LeafRange LeafRangeT;

    typedef PointDataTree::LeafNodeType PointDataLeafNode;

    typedef boost::ptr_vector<Geometry_OpenVDBPoints::LeafBVHTree> LeafBVHTreeContainer;
    typedef std::map<Coord, unsigned> OriginToIndexMap;

    ComputeBBoxPerPrimitiveOp(  LeafBVHTreeContainer& bvhTrees,
                                const OriginToIndexMap& originToIndex,
                                const PointDataTree& tree,
                                const bool overrideRadius,
                                const float radius,
                                const float backwardOffset,
                                const float forwardOffset)
        : mBVHTrees(bvhTrees)
        , mOriginToIndex(originToIndex)
        , mOverrideRadius(overrideRadius)
        , mRadius(radius)
        , mBackwardOffset(backwardOffset)
        , mForwardOffset(forwardOffset) { }

    void operator()(const LeafManagerT::LeafRange& range) const {

        for (LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            LeafBVHTreeContainer& bvhTrees = const_cast<LeafBVHTreeContainer&>(mBVHTrees);

            const unsigned id = mOriginToIndex.at(leaf->origin());

            const AttributeHandle<Vec3f>::Ptr positionHandle = AttributeHandle<Vec3f>::create(leaf->attributeArray("P"));

            AttributeHandle<Vec3f>::Ptr velocityHandle;
            AttributeHandle<float>::Ptr radiusHandle;

            if (leaf->hasAttribute("v"))        velocityHandle = AttributeHandle<Vec3f>::create(leaf->attributeArray("v"));
            if (leaf->hasAttribute("pscale"))   radiusHandle = AttributeHandle<float>::create(leaf->attributeArray("pscale"));

            Geometry_OpenVDBPoints::LeafBVHTree& bvhTree = bvhTrees[id];

            for (PointDataLeafNode::ValueOnCIter value = leaf->cbeginValueOn(); value; ++value)
            {
                Coord ijk = value.getCoord();

                const Vec3i gridIndexSpace = ijk.asVec3i();

                for (IndexIter iter = leaf->beginIndex(ijk); iter; ++iter) {

                    Vec3f positionValue;
                    Vec3f velocityValue;
                    float radiusValue;

                    positionValue = positionHandle->get(*iter);

                    if (velocityHandle)  velocityValue = velocityHandle->get(*iter);
                    else                 velocityValue = Vec3f(0, 0, 0);

                    if (radiusHandle && !mOverrideRadius)
                    {
                        radiusValue = radiusHandle->get(*iter);

                        // scale and convert to index space

                        radiusValue *= mRadius;
                    }
                    else
                    {
                        radiusValue = mRadius;
                    }

                    const Vec3f positionVoxelSpace(positionValue[0], positionValue[1], positionValue[2]);
                    const Vec3f positionIndexSpace = positionVoxelSpace + gridIndexSpace;

                    if (velocityHandle)
                    {
                        // offset position for beginning and end of velocity vector

                        bvhTree.expand</*isT0=*/false>(ijk, positionIndexSpace - velocityValue * mBackwardOffset, radiusValue);
                        bvhTree.expand</*isT0=*/true>(ijk, positionIndexSpace + velocityValue * mForwardOffset, radiusValue);
                    }
                    else
                    {
                        // offset position for radius

                        bvhTree.expand</*isT0=*/false>(ijk, positionIndexSpace, radiusValue);
                        bvhTree.expand</*isT0=*/true>(ijk, positionIndexSpace, radiusValue);
                    }

                }
            }

            for (PointDataLeafNode::ValueOnCIter value = leaf->cbeginValueOn(); value; ++value)
            {
                Coord ijk = value.getCoord();

                bvhTree.propagate(ijk);
            }
        }
    }

    //////////

    LeafBVHTreeContainer&               mBVHTrees;
    const OriginToIndexMap&             mOriginToIndex;

    const bool                          mOverrideRadius;
    const float                         mRadius;
    const float                         mBackwardOffset;
    const float                         mForwardOffset;
};

Geometry_OpenVDBPoints*
Geometry_OpenVDBPoints::create( const PointDataGrid::Ptr& grid,
                                const bool overrideRadius,
                                const double& radius,
                                const double& fps,
                                const double& motionBlurLength,
                                const AppBase::MotionBlurDirectionMode& motionBlurDirection)
{
    if (!grid)  return 0;

    PointDataTree& tree = grid->tree();

    // ensure voxels are uniform

    if (!grid->hasUniformVoxels())  return 0;

    // ensure position attribute exists

    PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if (!iter)  return 0;

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    if (descriptor.find("P") == AttributeSet::INVALID_POS)    return 0;

    const bool enableMotionBlur = iter->hasAttribute("v");
    const bool enableRadius = iter->hasAttribute("pscale");

    const size_t velocityIndex = descriptor.find("v");
    const size_t radiusIndex = descriptor.find("pscale");

    // retrieve the velocity and radius types

    std::string velocityType = "vec3s";
    std::string radiusType = "float";

    if (enableMotionBlur) {
        const NamePair& type = descriptor.type(velocityIndex);
        velocityType = type.first;
    }

    if (enableRadius) {
        const NamePair& type = descriptor.type(radiusIndex);
        radiusType = type.first;
    }

    // check attribute types are valid

    if (velocityType != "vec3s" && velocityType != "vec3h") {
        std::cerr << "Invalid Velocity Type: " << velocityType << std::endl;
        return 0;
    }

    if (radiusType != "float" && radiusType != "half") {
        std::cerr << "Invalid Radius Type: " << radiusType << std::endl;
        return 0;
    }

    // compute the desired time segment and time offset for motion blur

    double timeSegment = 0.0;
    double timeOffset = 0.0;

    if (enableMotionBlur)
    {
        timeSegment = motionBlurLength / fps;

        if (motionBlurDirection == AppBase::MOTION_BLUR_DIRECTION_MODE_BACKWARD) {
            timeOffset = timeSegment;
        }
        else if (motionBlurDirection == AppBase::MOTION_BLUR_DIRECTION_MODE_CENTERED) {
            timeOffset = timeSegment / 2.0;
        }
    }

    // check there are points

    if (pointCount(tree) == 0)    return 0;

    return new Geometry_OpenVDBPoints(  grid, descriptor, velocityType, radiusType,
                                        overrideRadius, radius, timeSegment, timeOffset);
}

Geometry_OpenVDBPoints::Geometry_OpenVDBPoints( const PointDataGrid::Ptr& grid,
                                                const AttributeSet::Descriptor& descriptor,
                                                const std::string& velocityType,
                                                const std::string& radiusType,
                                                const bool overrideRadius,
                                                const double& radius,
                                                const double& timeSegment,
                                                const double& timeOffset)
    : GeometryObject()
    , m_grid(grid)
    , m_descriptor(descriptor)
    , m_transform(grid->transform())
    , m_baseMap(*m_transform.baseMap())
    , m_velocityType(velocityType)
    , m_radiusType(radiusType)
    , m_overrideRadius(overrideRadius)
    , m_radius(radius)
    , m_radiusIndexSpace(0.0)
    , m_enableMotionBlur(timeSegment != 0.0)
    , m_timeSegment(timeSegment)
    , m_overTimeSegment(timeSegment == 0.0 ? 0.0 : (1.0 / timeSegment))
    , m_timeOffset(timeOffset)
    , m_primitiveCount(0)
{
    m_shadingGroupNames.resize(1);
    m_shadingGroupNames[0] = "points";

    // convert world space radius to index space

    m_radiusIndexSpace = m_radius / m_baseMap.voxelSize()[0];
}

void Geometry_OpenVDBPoints::computeAccelerationStructures()
{
    assert(m_grid);

    PointDataTree& tree = m_grid->tree();

    // compute an accurate bounding box per leaf

    typedef std::map<Coord, unsigned> LeafOriginToIndex;
    LeafOriginToIndex originToIndex;

    PointDataTree::LeafIter iter = tree.beginLeaf();

    unsigned i = 0;

    for (; iter; ++iter)
    {
        PointDataLeaf& leaf = *iter;

        originToIndex[leaf.origin()] = i++;

        m_leaves.push_back(&leaf);

        mBVHTrees.push_back(new LeafBVHTree);
    }

    m_primitiveCount = i;

    // compute backwards and forwards offsets for motion blur

    const float backwardOffset = -m_timeOffset;
    const float forwardOffset = m_timeSegment - m_timeOffset;

    const float radius = m_overrideRadius ? m_radiusIndexSpace : m_radius;

    ComputeBBoxPerPrimitiveOp compute(mBVHTrees, originToIndex, tree, m_overrideRadius, radius, backwardOffset, forwardOffset);
    tbb::parallel_for(tree::LeafManager<PointDataTree>(tree).leafRange(), compute);

    for (LeafBVHTreeContainer::const_iterator it = mBVHTrees.begin(), itEnd = mBVHTrees.end(); it != itEnd; ++it)
    {
        const LeafBVHTree& bvhTree = *it;

        m_bboxT0.expand(bvhTree.bbox1(/*isT0=*/true));
        m_bboxT1.expand(bvhTree.bbox1(/*isT0=*/false));
    }
}

Geometry_OpenVDBPoints* Geometry_OpenVDBPoints::get_copy() const
{
    return new Geometry_OpenVDBPoints(  m_grid, m_descriptor, m_velocityType, m_radiusType,
                                        m_overrideRadius, m_radius, m_timeSegment, m_timeOffset);
}

size_t Geometry_OpenVDBPoints::get_memory_size() const
{
    return  sizeof(*this) +
            sizeof(m_leaves) +
            sizeof(mBVHTrees);
}

GMathBbox3d Geometry_OpenVDBPoints::get_bbox() const
{
    GMathBbox3d clarisseBbox;

    if (!m_grid)    return clarisseBbox;

    BBoxd bbox;

    bbox.expand(m_bboxT0);
    bbox.expand(m_bboxT1);

    const BBoxd& bboxWorld = m_transform.indexToWorld(bbox);

    clarisseBbox[0][0] = bboxWorld.min()[0];
    clarisseBbox[0][1] = bboxWorld.min()[1];
    clarisseBbox[0][2] = bboxWorld.min()[2];

    clarisseBbox[1][0] = bboxWorld.max()[0];
    clarisseBbox[1][1] = bboxWorld.max()[1];
    clarisseBbox[1][2] = bboxWorld.max()[2];

    return clarisseBbox;
}

GMathBbox3d Geometry_OpenVDBPoints::get_bbox_at(const CtxEval& eval_ctx,
                        const double& time) const
{
    GMathBbox3d clarisseBbox;

    if (!m_grid)    return clarisseBbox;

    const double adjusted_time = (time + m_timeOffset) * m_overTimeSegment;

    BBoxd bbox;

    internal::interpolate(bbox, adjusted_time, m_bboxT0, m_bboxT1);

    const BBoxd& bboxWorld = m_transform.indexToWorld(bbox);

    clarisseBbox[0][0] = bboxWorld.min()[0];
    clarisseBbox[0][1] = bboxWorld.min()[1];
    clarisseBbox[0][2] = bboxWorld.min()[2];

    clarisseBbox[1][0] = bboxWorld.max()[0];
    clarisseBbox[1][1] = bboxWorld.max()[1];
    clarisseBbox[1][2] = bboxWorld.max()[2];

    return clarisseBbox;
}

bool Geometry_OpenVDBPoints::is_animated() const
{
    // this is mandatory to tell the acceleration structure to compute motion blur
    return m_enableMotionBlur;
}

const CoreBasicArray<CoreString>& Geometry_OpenVDBPoints::get_shading_group_names() const
{
    return m_shadingGroupNames;
}

unsigned int Geometry_OpenVDBPoints::get_primitive_count() const
{
    return m_primitiveCount;
}

unsigned int Geometry_OpenVDBPoints::get_primitive_edge_count(const unsigned int& id) const
{
    return 4;
}

void Geometry_OpenVDBPoints::compute_primitive_bbox(const CtxEval& eval_ctx,
                            const unsigned int& id,
                            GMathBbox3d& clarisseBbox) const
{
    if (!m_grid)  return;

    const LeafBVHTree& bvhTree = mBVHTrees[id];

    BBoxd bbox;

    bbox.expand(bvhTree.bbox1(/*isT0=*/true));
    bbox.expand(bvhTree.bbox1(/*isT0=*/false));

    const BBoxd& bboxWorld = m_transform.indexToWorld(bbox);

    clarisseBbox[0][0] = bboxWorld.min()[0];
    clarisseBbox[0][1] = bboxWorld.min()[1];
    clarisseBbox[0][2] = bboxWorld.min()[2];

    clarisseBbox[1][0] = bboxWorld.max()[0];
    clarisseBbox[1][1] = bboxWorld.max()[1];
    clarisseBbox[1][2] = bboxWorld.max()[2];
}

void Geometry_OpenVDBPoints::compute_primitive_bbox_at( const CtxEval& eval_ctx,
                                const unsigned int& id,
                                const double& time,
                                GMathBbox3d& clarisseBbox) const
{
    const double adjusted_time = (time + m_timeOffset) * m_overTimeSegment;

    BBoxd bbox;

    internal::interpolate(bbox, adjusted_time,
            mBVHTrees[id].bbox1(/*isT0=*/true), mBVHTrees[id].bbox1(/*isT0=*/false));

    const BBoxd& bboxWorld = m_transform.indexToWorld(bbox);

    clarisseBbox[0][0] = bboxWorld.min()[0];
    clarisseBbox[0][1] = bboxWorld.min()[1];
    clarisseBbox[0][2] = bboxWorld.min()[2];

    clarisseBbox[1][0] = bboxWorld.max()[0];
    clarisseBbox[1][1] = bboxWorld.max()[1];
    clarisseBbox[1][2] = bboxWorld.max()[2];
}

void Geometry_OpenVDBPoints::compute_fragment_sample (
    const CtxEval &eval_ctx,
    const GeometryFragment &fragment,
    GeometrySample &sample) const
{
    const double radius = fragment.get_medium_density();

    const GMathVec3d gmath_origin_position(fragment.get_medium_density_diff());

    GMathVec3d position, dpdu, dpdv;
    const double phi = fragment.get_u() * gmath_two_pi;
    const double cosphi = cos(phi);
    const double sinphi = sin(phi);
    const double theta = (fragment.get_v() - 0.5) * gmath_pi;
    const double costheta = cos(theta);
    const double sintheta = sin(theta);
    const double zrad = costheta * radius;
    position[0] = -sinphi * zrad;
    position[1] = sintheta * radius;
    position[2] = -cosphi * zrad;
    if (zrad > gmath_epsilon)
    {
        dpdu[2] = -position[0] * gmath_two_pi;
        dpdu[0] = position[2] * gmath_two_pi;
    }
    else
    {
        dpdu[2] = sinphi * gmath_two_pi;
        dpdu[0] = -cosphi * gmath_two_pi;
    }

    dpdu[1] = 0;
    dpdv[2] = position[1] * cosphi * gmath_pi;
    dpdv[0] = position[1] * sinphi * gmath_pi;
    dpdv[1] = costheta * radius * gmath_pi;

    const Vec3d sample_position(   position[0] + gmath_origin_position[0],
                                            position[1] + gmath_origin_position[1],
                                            position[2] + gmath_origin_position[2]);

    const Vec3d world_sample_position(m_transform.indexToWorld(sample_position));

    const GMathVec3d gmath_world_position(world_sample_position[0], world_sample_position[1], world_sample_position[2]);

    sample.init_surface(gmath_world_position, dpdu, dpdv);
}

void Geometry_OpenVDBPoints::intersect_primitive(
    const CtxEval& eval_ctx,
    const unsigned int& id,
    GeometryRaytraceCtx& raytrace_ctx) const
{
    if (m_velocityType == "vec3s" && m_radiusType == "float") {
        intersect_typed_primitive<Vec3f, float>(eval_ctx, id, raytrace_ctx);
    }
    else if (m_velocityType == "vec3s" && m_radiusType == "half") {
        intersect_typed_primitive<Vec3f, half>(eval_ctx, id, raytrace_ctx);
    }
    else if (m_velocityType == "vec3h" && m_radiusType == "float") {
        intersect_typed_primitive<math::Vec3<half>, float>(eval_ctx, id, raytrace_ctx);
    }
    else if (m_velocityType == "vec3h" && m_radiusType == "half") {
        intersect_typed_primitive<math::Vec3<half>, half>(eval_ctx, id, raytrace_ctx);
    }
}

template <typename VelocityType, typename RadiusType>
void Geometry_OpenVDBPoints::intersect_typed_primitive(
    const CtxEval& eval_ctx,
    const unsigned int& id,
    GeometryRaytraceCtx& raytrace_ctx) const
{
    typedef math::Ray<double> RayT;
    typedef RayT::Vec3Type Vec3T;

    if (raytrace_ctx.get_ray_count() == 0)     return;

    assert(m_leaves[id]);

    const PointDataLeaf& leaf = *m_leaves[id];

    assert(leaf.attributeSet().find("P") != AttributeSet::INVALID_POS);

    // obtain the attribute handles

    const AttributeHandle<Vec3f>::Ptr positionHandle = AttributeHandle<Vec3f>::create(leaf.attributeArray("P"));

    typename AttributeHandle<VelocityType>::Ptr velocityHandle;
    typename AttributeHandle<RadiusType>::Ptr radiusHandle;

    if (leaf.hasAttribute("v"))                             velocityHandle = AttributeHandle<VelocityType>::create(leaf.attributeArray("v"));
    if (leaf.hasAttribute("pscale") && !m_overrideRadius)   radiusHandle = AttributeHandle<RadiusType>::create(leaf.attributeArray("pscale"));

    // three-level leaf BVH tree:
    //     level1: 8x8x8 voxels
    //     level2: 4x4x4 voxels
    //     level3: 2x2x2 voxels

    const LeafBVHTree& bvhTree = mBVHTrees[id];

    // cache wraps the BVH tree for lowering the number of expensive intersection tests:
    //     level1: ignored (as this is leaf-level)
    //     level2: pre-computed
    //     level3: cached on-demand

    CachedBVHTree cachedTree(bvhTree);

    for (int ray_index  = raytrace_ctx.get_first_index();
             ray_index <= raytrace_ctx.get_last_index(); ++ray_index)
    {
        // initialise ray

        const GMathRay& ray = raytrace_ctx.get_world_ray(ray_index);

        const double tnear = raytrace_ctx.get_tnear(ray_index);
        const double tfar = raytrace_ctx.get_tfar(ray_index);

        const GMathVec3d& dir = ray.get_direction();
        const GMathVec3d& orig = ray.get_origin();

        const double time = ray.get_time();

        // only enable velocity if time is non-zero

        const bool enable_velocity = (velocityHandle && time != 0.0);

        // convert GMathRay into openvdb::Ray

        const RayT vdb_ray( Vec3T(orig[0], orig[1], orig[2]),
                            Vec3T(dir[0],  dir[1],  dir[2]),
                            tnear, tfar);

        // as t is in world-space, the length of the inverse jacobian of the direction
        // is stored to convert the index-space t back into world space

        const double inv_length = 1.0 / m_baseMap.applyInverseJacobian(vdb_ray.dir()).length();

        // convert world-space ray into index-space

        const RayT vdb_index_ray(vdb_ray.worldToIndex(*m_grid));

        // compute interpolated bounding boxes

        const double adjusted_time = (time + m_timeOffset) * m_overTimeSegment;

        // pre-compute level 2 BVH (level 3 is cached on-demand)

        cachedTree.computeHits</*level=*/2>(vdb_index_ray, adjusted_time);

        double u, v, w;
        w = 0.0f;

        // initialise times

        double t = std::numeric_limits<double>::max();
        double t0 = 0.0f;
        double t1 = 0.0f;

        for (PointDataLeaf::ValueOnCIter value = leaf.cbeginValueOn(); value; ++value)
        {
            const Coord ijk = value.getCoord();

            if (!cachedTree.hit</*level=*/2>(vdb_index_ray, adjusted_time, ijk))   continue;

            if (!cachedTree.hit</*level=*/3>(vdb_index_ray, adjusted_time, ijk))   continue;

            bool hit = false;

            unsigned int sub_primitive_id(0);

            Vec3f position(0.0f);
            double radius(1.0);

            for (IndexIter iter = leaf.beginIndex(ijk); iter; ++iter) {

                Vec3f positionValue;
                RadiusType radiusValue;

                positionValue = positionHandle->get(*iter);

                // compute radius in index space (apply scaling if required)

                if (radiusHandle && !m_overrideRadius) {
                    radiusValue = radiusHandle->get(*iter) * m_radius;
                }
                else {
                    radiusValue = m_radiusIndexSpace;
                }

                // retrieve index position

                Vec3d position_index_space = positionValue + ijk.asVec3d();

                if (enable_velocity)
                {
                    position_index_space += velocityHandle->get(*iter) * time;
                }

                const double radius0 = radiusValue;

                // perform ray intersection

                if (vdb_index_ray.intersects(position_index_space, radius0, t0, t1))
                {
                    if (t0 < t)
                    {
                        t = t0;
                        position = position_index_space;
                        radius = radius0;
                        sub_primitive_id = (unsigned int)*iter;
                    }

                    hit = true;
                }
            }

            if (hit)
            {
                const Vec3d hit_point = vdb_index_ray(t) - position;

                GMathVec3d near_hp(hit_point.x(), hit_point.y(), hit_point.z());

                compute_uv(near_hp, u, v, radius);

                GeometryMediumDescriptor medium;
                medium.density = radius;
                medium.density_diff = GMathVec3d(position.x(), position.y(), position.z());

                raytrace_ctx.push_intersection(
                    eval_ctx,
                    ray_index,
                    id,
                    u, v, w,
                    sub_primitive_id,
                    u, v, w,
                    t * inv_length,
                    /*bias = */ radius * 1.0e-6,
                    medium);
            }
        }
    }
}

void Geometry_OpenVDBPoints::compute_uv(
    const GMathVec3d& pos,
    double& u,
    double& v,
    double& radius) const
{
    const double theta = acos(gmath_mind(gmath_maxd(pos[1] * (1.0 / radius), -1.0), 1.0));

    const double phi = atan2(-pos[0], -pos[2]);
    u = (phi < 0.0 ? phi + gmath_two_pi : phi) * gmath_inv_two_pi;
    v = 1.0 - theta * gmath_inv_pi;
}
