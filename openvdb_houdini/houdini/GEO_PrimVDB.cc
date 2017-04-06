///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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

/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *	Jeff Lait
 *	Side Effects Software Inc
 *	477 Richmond Street West
 *	Toronto, Ontario
 *	Canada   M5V 3E7
 *	416-504-9876
 *
 * NAME:	GEO_PrimVDB.C ( GEO Library, C++)
 *
 * COMMENTS:	Base class for vdbs.
 */

#include <UT/UT_Version.h>
#if (UT_VERSION_INT < 0x0c050157) // earlier than 12.5.343

#include "GEO_PrimVDB.h"

#include <SYS/SYS_AtomicPtr.h>
#include <SYS/SYS_Math.h>

#include <UT/UT_Debug.h>
#include <UT/UT_Defines.h>

#if (UT_VERSION_INT >= 0x0d00023d) // 13.0.573 or later
#include <UT/UT_EnvControl.h>
#include <UT/UT_FSATable.h>
#else
#include <UT/UT_FSA.h>
#endif

#include <UT/UT_IStream.h>
#include <UT/UT_JSONParser.h>
#include <UT/UT_JSONValue.h>
#include <UT/UT_JSONWriter.h>
#include <UT/UT_Matrix.h>
#include <UT/UT_MatrixSolver.h>
#include <UT/UT_ScopedPtr.h>
#include <UT/UT_SparseArray.h>
#include <UT/UT_StackTrace.h>
#include <UT/UT_SysClone.h>
#include "UT_VDBUtils.h"
#include <UT/UT_Vector.h>
#include <UT/UT_Version.h>
#include <UT/UT_XformOrder.h>

#include <GA/GA_AttributeRefMap.h>
#include <GA/GA_AttributeRefMapDestHandle.h>
#include <GA/GA_Defragment.h>
#include <GA/GA_ElementWrangler.h>
#if (UT_VERSION_INT >= 0x0c010048) // 12.1.72 or later
#include <GA/GA_IntrinsicMacros.h>
#endif
#include <GA/GA_MergeMap.h>
#include <GA/GA_PrimitiveJSON.h>
#include <GA/GA_RangeMemberQuery.h>
#include <GA/GA_SaveMap.h>
#include <GA/GA_WeightedSum.h>
#include <GA/GA_WorkVertexBuffer.h>

#include <GEO/GEO_Detail.h>
#include <GEO/GEO_PrimType.h>
#include <GEO/GEO_PrimVolume.h>
#include <GEO/GEO_AttributeHandleList.h>
#include <GEO/GEO_WorkVertexBuffer.h>

#include <openvdb/io/Stream.h>
#include <openvdb/math/Maps.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <openvdb/tools/VectorTransformer.h>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>


#if (UT_VERSION_INT < 0x0c0100B6) // earlier than 12.1.182
static bool
geo_JVDBError(UT_JSONParser &p, const GA_Primitive *prim, const char *m)
{
    p.addFatal("Error loading %s: %s", prim->getTypeName(), m);
    return false;
}
#endif

#if (UT_VERSION_INT < 0x0c010048) // earlier than 12.1.72
GA_IntrinsicManager::Registrar
GEO_PrimVDB::registerIntrinsics(GA_PrimitiveDefinition &defn)
{
    ///defn.setMergeConstructor(&gaPrimitiveMergeConstructor);
    return GEO_Primitive::registerIntrinsics(defn);
}
#endif


GEO_PrimVDB::UniqueId
GEO_PrimVDB::nextUniqueId()
{
    static AtomicUniqueId	theUniqueId;
    return static_cast<UniqueId>(theUniqueId.add(1));
}


GEO_PrimVDB::GEO_PrimVDB(GEO_Detail *d, GA_Offset offset)
    : GEO_Primitive(d, offset)
    , myGridAccessor()
    , myVis(GEO_VOLUMEVIS_SMOKE, /*iso*/0.0, /*density*/1.0)
    , myUniqueId(GEO_PrimVDB::nextUniqueId())
    , myTreeUniqueId(GEO_PrimVDB::nextUniqueId())
    , myMetadataUniqueId(GEO_PrimVDB::nextUniqueId())
    , myTransformUniqueId(GEO_PrimVDB::nextUniqueId())
{
    myVertex = allocateVertex();
#if (UT_VERSION_INT < 0x0c050000) // earlier than 12.5.0
    myStashedState = false;
    if (d) d->addVolumeRef();
#endif
}

GEO_PrimVDB::GEO_PrimVDB(const GA_MergeMap &map, GA_Detail &detail,
                         GA_Offset offset, const GEO_PrimVDB &src_prim)
    : GEO_Primitive(static_cast<GEO_Detail *>(&detail), offset)
    , myVis(src_prim.myVis)
{
    myUniqueId.exchange(src_prim.getUniqueId());

    if (map.isIdentityMap(GA_ATTRIB_VERTEX))
    {
    	myVertex = src_prim.myVertex;
    }
    else
    {
	GA_Offset sidx = src_prim.myVertex; // Get source index
	myVertex = map.mapDestFromSource(GA_ATTRIB_VERTEX, sidx);
    }
#if (UT_VERSION_INT < 0x0c050000) // earlier than 12.5.0
    myStashedState = false;
    static_cast<GEO_Detail &>(detail).addVolumeRef();
#endif

    copyGridFrom(src_prim); // makes a shallow copy
}

GEO_PrimVDB::~GEO_PrimVDB()
{
    if (GAisValid(myVertex))
	destroyVertex(myVertex);
#if (UT_VERSION_INT < 0x0c050000) // earlier than 12.5.0
    if (!myStashedState && getParent())
	getParent()->delVolumeRef();
#endif
}

void
GEO_PrimVDB::clearForDeletion()
{
    myVertex = GA_INVALID_OFFSET;
    GEO_Primitive::clearForDeletion();
}

#if (UT_VERSION_INT >= 0x0d000000) // 13.0 or later
void
GEO_PrimVDB::stashed(bool beingstashed, GA_Offset offset)
{
    // NB: Base class must be unstashed before we can call allocateVertex().
    GEO_Primitive::stashed(beingstashed, offset);
    myVertex = beingstashed ? GA_INVALID_OFFSET : allocateVertex();
#else
void
GEO_PrimVDB::stashed(int onoff, GA_Offset offset)
{
#if (UT_VERSION_INT < 0x0c050000) // earlier than 12.5.0
    if (getParent())
    {
	if (onoff)
	    getParent()->delVolumeRef();
	else
	    getParent()->addVolumeRef();
    }
    myStashedState = (onoff != 0);
#endif
    // NB: Base class must be unstashed before we can call allocateVertex().
    GEO_Primitive::stashed(onoff, offset);
    myVertex = onoff ? GA_INVALID_OFFSET : allocateVertex();
#endif
    // Set our internal state to default
    myVis = GEO_VolumeOptions(GEO_VOLUMEVIS_SMOKE, /*iso*/0.0, /*density*/1.0);
}

bool
GEO_PrimVDB::evaluatePointRefMap(GA_Offset result_vtx,
    GA_AttributeRefMap &map,
    fpreal /*u*/, fpreal /*v*/,
    uint /*du*/, uint /*dv*/) const
{
    map.copyValue(GA_ATTRIB_VERTEX, result_vtx,
		  GA_ATTRIB_VERTEX, getVertexOffset(0));
    return true;
}

// Houdini assumes that the depth scaling of the frustum is being done in the
// linear part of the NonlinearFrustumMap. This method ensures that if the
// grid has a frustum depth not equal to 1, then it returns an equivalent map
// which does.
static openvdb::math::NonlinearFrustumMap::ConstPtr
geoStandardFrustumMapPtr(const GEO_PrimVDB &vdb)
{
    using namespace openvdb::math;
    using openvdb::Vec3d;

    const Transform &transform = vdb.getGrid().transform();
    UT_ASSERT(transform.baseMap()->isType<NonlinearFrustumMap>());
    NonlinearFrustumMap::ConstPtr
	frustum_map = transform.constMap<NonlinearFrustumMap>();

    // If the depth is already 1, then just return the original
    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    if (frustum_map->getDepth() == 1.0)
	return frustum_map;
    OPENVDB_NO_FP_EQUALITY_WARNING_END

    AffineMap secondMap = frustum_map->secondMap();
    secondMap.accumPreScale(Vec3d(1, 1, frustum_map->getDepth()));
    return NonlinearFrustumMap::ConstPtr(
	    new NonlinearFrustumMap(frustum_map->getBBox(),
				    frustum_map->getTaper(),
				    /*depth*/1.0, secondMap.copy()));
}

// The returned space's fromVoxelSpace() method will convert 0-1
// coordinates over the bbox extents to world space (and vice versa for
// toVoxelSpace()).
GEO_PrimVolumeXform
GEO_PrimVDB::getSpaceTransform(const UT_BoundingBoxD &bbox) const
{
    using namespace openvdb;
    using namespace openvdb::math;
    using openvdb::Vec3d;
    using openvdb::Mat4d;

    MapBase::ConstPtr	base_map = getGrid().transform().baseMap();
    BBoxd		active_bbox(UTvdbConvert(bbox.minvec()),
				    UTvdbConvert(bbox.maxvec()));
    UT_Matrix4D		transform(1.0); // identity
    fpreal		new_taper(1.0); // no taper default

    // If the active_bbox is empty(), we do not want to produce a singular
    // matrix.
    if (active_bbox.empty())
    {
	if (base_map->isType<NonlinearFrustumMap>())
	{
	    // Ideally, we use getFrustumBounds() here but it returns the
	    // wrong type.
	    const NonlinearFrustumMap& frustum_map =
		*getGrid().transform().constMap<NonlinearFrustumMap>();
	    active_bbox = frustum_map.getBBox();
	    active_bbox.translate(Vec3d(+0.5));
	}
	else
	{
	    // Use a single voxel from index origin to act as a pass-through
	    active_bbox = BBoxd(Vec3d(0.0), 1.0);
	}
    }

    // Shift the active_bbox by half a voxel to account for the fact that
    // UT_VoxelArray's index coordinates are cell-centred while grid indices
    // are cell edge-aligned.
    active_bbox.translate(Vec3d(-0.5));

    if (base_map->isType<NonlinearFrustumMap>())
    {
	// NOTES:
	// - VDB's nonlinear frustum goes from index-space to world-space in
	//   two steps:
	//	1. From index-space to NDC space where we have [-0.5,+0.5] XY
	//	   on the Z=0 plane, tapered outwards to to the Z=1 plane
	//	   (where depth=1).
	//	2. Then the base_map->secondMap() is applied to convert it
	//	   into world-space.
	// - Our goal is to come up with an equivalent transform which goes
	//   from [-1,+1] unit-space to world-space, matching GEO_PrimVDB's
	//   node-centred samples with GEO_PrimVolume's cell-centered samples.

	NonlinearFrustumMap::ConstPtr map_ptr = geoStandardFrustumMapPtr(*this);
	const NonlinearFrustumMap& frustum_map = *map_ptr;

	// Math below only handles NonlinearFrustumMap's with a depth of 1.
	UT_ASSERT(frustum_map.getDepth() == 1);

	BBoxd	    frustum_bbox = frustum_map.getBBox();
	UT_Vector3D frustum_size(UTvdbConvert(frustum_bbox.extents()));
	UT_Vector3D inv_frustum_size = 1.0 / frustum_size;

	// Find active_bbox in the 1 by 1 -> taper by taper frustum
	UT_Vector3D active_size(UTvdbConvert(active_bbox.extents()));
	UT_Vector3D offset_uniform =
	    UTvdbConvert(active_bbox.min() - frustum_bbox.min())
		* inv_frustum_size;
	UT_Vector3  scale = active_size * inv_frustum_size;

	// Compute the z coordinates of 'active_bbox' in the
	// 0-1 space (normed to the frustum size)
	fpreal z_min = offset_uniform.z();
	fpreal z_max = offset_uniform.z() + scale.z();

	// We need a new taper since active_bbox might have a different
	// near/far plane ratio. The mag values are calculated using VDB's
	// taper function but then we divide min by max because Houdini's taper
	// ratio is inversed.
	fpreal frustum_taper = frustum_map.getTaper();
	fpreal gamma = 1.0/frustum_taper - 1.0;
	fpreal mag_min = 1 + gamma * z_min;
	fpreal mag_max = 1 + gamma * z_max;
	new_taper = mag_min / mag_max;

	// xform will go from 0-1 voxel space to the tapered, near=1x1 frustum
	UT_Matrix4D xform(1);
	xform.scale(mag_min, mag_min, 1.0);

	xform.scale(0.5, 0.5, 0.5);
	xform.scale(scale.x(), scale.y(), scale.z());

	// Scale our correctly tapered box
	// Offset the correctly scaled and tapered box into the right xy
	// position.
	OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
	if (gamma != 0.0)
	OPENVDB_NO_FP_EQUALITY_WARNING_END
	{
	    // Scale by the inverse of the taper since NonlinearFrustumMap
	    // creates tapers in the -z direction (a positive taper will
	    // increase the ratio of the near / far) but only scales the far
	    // face (effectively, the +z face is scaled by 1 / taper and
	    // the -z face is kept at 1.0)
	    xform.scale(1.0 / new_taper, 1.0 / new_taper, 1.0);

	    // The distance from the near plane that the tapered frustum sides
	    // meet ie. the position of the z-plane that gets mapped to 0 in
	    // the taper map
	    fpreal z_i = 1.0 / gamma;
	    xform.translate(0, 0, z_i + z_min + 0.5 * scale.z());

	    // Compute the shear: it is the offset on the near plane of the
	    // current frustum to where we want it to be
	    UT_Vector3D frustum_center(0.5*frustum_size);
	    UT_Vector3D active_center(0.5*active_size);
	    // The current active_bbox position
	    UT_Vector3D bbox_offset = frustum_center - active_center;
	    // Compute the offset to the real position. We add back half-voxel
	    // so that the reference base is at the old min
	    UT_Vector3D shear = UTvdbConvert(active_bbox.min() + Vec3d(0.5))
				    - bbox_offset;
	    shear *= inv_frustum_size;
	    shear /= z_i;
	    xform.shear(0, shear.x(), shear.y());

	    // Translate ourselves back so that the tapered plane of
	    // frustum_bbox is at 0
	    xform.translate(0, 0, -z_i);
	}
	else
	{
	    // Translate bottom corner of box to (0,0,0)
	    xform.translate(-0.5, -0.5, 0.0);
	    xform.translate(0.5*scale.x(), 0.5*scale.y(), 0.5*scale.z());
	    xform.translate(offset_uniform.x(),
			    offset_uniform.y(),
			    offset_uniform.z());
	}

	// `transform` now brings us from a tapered (1*y/x) box to world space,
	// We want to go from a tapered, near=1x1 frustum to world space, so
	// prescale by the aspect
	fpreal aspect = frustum_bbox.extents().y() / frustum_bbox.extents().x();
	xform.scale(1.0, aspect, 1.0);

	Mat4d mat4 = frustum_map.secondMap().getMat4();
	transform = xform * UTvdbConvert(mat4);
    }
    else
    {
	// NOTES:
	// - VDB's grid transform goes from index-space to world-space.
	// - Our goal is to come up with an equivalent transform which goes
	//   from [-1,+1] unit-space to world-space, matching GEO_PrimVDB's
	//   node-centred samples with GEO_PrimVolume's cell-centered samples.
	//   (NOTE: fromVoxelSpace() converts from [0,+1] to [-1,+1])

	// Create transform which converts [-1,+1] unit-space to [0,+1]
	transform.translate(1.0, 1.0, 1.0);
	transform.scale(0.5, 0.5, 0.5);

	// Go from the [0,1] volume space, to [min,max] where
	// min and max are in the vdb index space. Note that UT_VoxelArray
	// doesn't support negative coordinates which is why we need to shift
	// the index origin to the bbox min.
	transform.scale(active_bbox.extents().x(),
			active_bbox.extents().y(),
			active_bbox.extents().z());
	transform.translate(active_bbox.min().x(),
			    active_bbox.min().y(),
			    active_bbox.min().z());

	// Post-multiply by the affine map which converts index to world space
	transform = transform * UTvdbConvert(base_map->getAffineMap()->getMat4());
    }

    UT_Vector3 translate;
    transform.getTranslates(translate);

    GEO_PrimVolumeXform result;
    result.myXform = transform;
    result.myCenter = translate;

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    result.myHasTaper = (new_taper != 1.0);
    OPENVDB_NO_FP_EQUALITY_WARNING_END

    transform.invert();
    result.myInverseXform = transform;

    result.myTaperX = new_taper;
    result.myTaperY = new_taper;

    return result;
}

// Return a GEO_PrimVolumeXform which maps [-0.5,+0.5] Houdini voxel space
// coordinates over the VDB's active voxel bbox into world space.
GEO_PrimVolumeXform
GEO_PrimVDB::getSpaceTransform() const
{
    const openvdb::CoordBBox bbox = getGrid().evalActiveVoxelBoundingBox();
    return getSpaceTransform(UTvdbConvert(bbox));
}

bool
GEO_PrimVDB::conditionMatrix(UT_Matrix4D &mat4)
{
    // This tolerance is just one factor larger than what
    // AffineMap::updateAcceleration() uses to ensure that we never trigger the
    // exception.
    const double tol = 4.0 * openvdb::math::Tolerance<double>::value();
    const double min_diag = SYScbrt(tol);
    if (!SYSequalZero(mat4.determinant3(), tol))
	return false;

    UT_MatrixSolverD solver;
    UT_Matrix3D mat3(mat4);
    const int N = 3;
    UT_MatrixD m(1,N, 1,N), v(1,N, 1,N), diag(1,N, 1,N), tmp(1,N, 1,N);
    UT_VectorD w(1,N);

    m.setSubmatrix3(1, 1, mat3);
    if (!solver.SVDDecomp(m, w, v))
    {
	// Give up and return a scale matrix as small as possible
	mat4.identity();
	mat4.scale(min_diag, min_diag, min_diag);
    }
    else
    {
	v.transpose(tmp);
	v = tmp;
	diag.makeIdentity();
	for (int i = 1; i <= N; i++)
	    diag(i,i) = SYSmax(min_diag, w(i));
	m.postMult(diag, tmp);
	tmp.postMult(v, m);
	m.getSubmatrix3(mat3, 1, 1);
	mat4 = mat3;
    }
    return true;
}

// All AffineMap creation must to through this to avoid crashes when passing
// singular matrices into OpenVDB
template<typename T>
static boost::shared_ptr<T>
geoCreateAffineMap(const UT_Matrix4D& mat4)
{
    using namespace openvdb::math;

    boost::shared_ptr<T> transform;
    UT_Matrix4D new_mat4(mat4);
    (void) GEO_PrimVDB::conditionMatrix(new_mat4);
    try
    {
	transform.reset(new AffineMap(UTvdbConvert(new_mat4)));
    }
    catch (openvdb::ArithmeticError &)
    {
	UT_ASSERT(!"Failed to create affine map");
	transform.reset(new AffineMap());
    }
    return transform;
}

// All calls to createLinearTransform with a matrix4 must to through this to
// avoid crashes when passing singular matrices into OpenVDB
static openvdb::math::Transform::Ptr
geoCreateLinearTransform(const UT_Matrix4D& mat4)
{
    using namespace openvdb::math;
    return Transform::Ptr(new Transform(geoCreateAffineMap<MapBase>(mat4)));
}

void
GEO_PrimVDB::setSpaceTransform(
	const GEO_PrimVolumeXform &space,
	const UT_Vector3R &resolution,
	bool force_taper)
{
    using namespace openvdb;
    using namespace openvdb::math;
    using openvdb::Vec3d;

    // VDB's nonlinear frustum goes from index-space to world-space in
    // two steps:
    //	1. From index-space to NDC space where we have [-0.5,+0.5] XY
    //	   on the Z=0 plane, tapered outwards to to the Z=1 plane
    //	   (where depth=1).
    //	2. Then the base_map->secondMap() is applied to convert it
    //	   into world-space.
    // On the other hand, 'space' converts from [-1,+1] space over the given
    // resolution into world-space.
    //
    // Our goal is to come up with an equivalent NonlinearTransformMap of
    // 'space' which converts from index space to world-space, matching
    // GEO_PrimVDB's node-centred samples with GEO_PrimVolume's cell-centered
    // samples.

    Transform::Ptr xform;

    if (force_taper || space.myHasTaper)
    {
	// VDB only supports a single taper value so use average of the two
	fpreal taper = 0.5*(space.myTaperX + space.myTaperY);

	// Create a matrix which goes from vdb's post-taper XY(-0.5,+0.5) space
	// to [-1,1] space so that we can post-multiply by space's transform to
	// get into world-space.
	//
	// NonlinearFrustumMap use's 1/taper as its taper value, going from
	// Z=0 to Z=1. So we first scale it by the taper to undo this.
	UT_Matrix4D transform(1.0);
	transform.scale(taper, taper, 1.0);
	// Account for aspect ratio
	transform.scale(1.0, resolution.x() / resolution.y(), 1.0);
	// We now go from XY(-0.5,+0.5)/Z(0,1) to XY(-1,+1)/Z(2)
	transform.scale(2.0, 2.0, 2.0);
	// Translate into [-1,+1] on all axes by centering the Z axis
	transform.translate(0.0, 0.0, -1.0);

	// Now apply the space's linear transform
	UT_Matrix4D linear;
	linear = space.myXform; // convert UT_Matrix3 to UT_Matrix4
	transform *= linear;
	transform.translate(
		    space.myCenter.x(), space.myCenter.y(), space.myCenter.z());

	// In order to get VDB to match Houdini, we create a frustum map using
	// Houdini's bbox, so we offset by -0.5 voxel in order taper the
	// Houdini bbox in VDB index space.  This allows us to align Houdini's
	// cell-centered samples with VDB node-centered ones.
	BBoxd bbox(Vec3d(0.0), UTvdbConvert(resolution));
	bbox.translate(Vec3d(-0.5));

	// Build a NonlinearFrustumMap from this
	MapBase::Ptr affine_map(geoCreateAffineMap<MapBase>(transform));
	xform.reset(new Transform(MapBase::Ptr(
	    new NonlinearFrustumMap(bbox, taper, /*depth*/1.0, affine_map))));
    }
    else // optimize for a linear transform if we have no taper
    {
	// NOTES:
	// - Houdini's transform goes from [-1,+1] unit-space to world-space.
	// - VDB's grid transform goes from index-space to world-space.

	UT_Matrix4D matx(/*identity*/1.0);
	UT_Matrix4D mult;

	// UT_VoxelArray's index coordinates are cell-centred while grid
	// indices are cell edge-aligned. So first offset the VDB indices by
	// +0.5 voxel to convert them into Houdini indices.
	matx.translate(0.5, 0.5, 0.5);

	// Now convert the indices from [0,dim] to [-1,+1]
	matx.scale(2.0/resolution(0), 2.0/resolution(1), 2.0/resolution(2));
	matx.translate(-1.0, -1.0, -1.0);

	// Post-multiply with Houdini transform to get result that converts
	// into world-space.
	mult = space.myXform;
	matx *= mult;
	matx.translate(space.myCenter(0), space.myCenter(1), space.myCenter(2));

	// Create a linear transform using the matrix
	xform = geoCreateLinearTransform(matx);
    }

    myGridAccessor.setTransform(*xform, *this);
}

// This will give you the a GEO_PrimVolumeXform. Given an index, it will
// compute the world space position of that index.
GEO_PrimVolumeXform
GEO_PrimVDB::getIndexSpaceTransform() const
{
    using namespace openvdb;
    using namespace openvdb::math;
    using openvdb::Vec3d;
    using openvdb::Mat4d;

    // This taper function follows from the conversion code in
    // GEO_PrimVolume::fromVoxelSpace() until until myXform/myCenter is
    // applied. It has been simplified somewhat, and uses the definition that
    // gamma = taper - 1.
    struct Local
    {
	static fpreal taper(fpreal x, fpreal z, fpreal gamma)
	{
	    return 2 * (x - 0.5) * (1 + gamma * (1 - z));
	}
    };

    fpreal	new_taper = 1.0;
    UT_Matrix4D	transform(1.0); // identity

    MapBase::ConstPtr base_map = getGrid().transform().baseMap();
    if (base_map->isType<NonlinearFrustumMap>())
    {
	// NOTES:
	// - VDB's nonlinear frustum goes from index-space to world-space in
	//   two steps:
	//	1. From index-space to NDC space where we have [-0.5,+0.5] XY
	//	   on the Z=0 plane, tapered outwards to to the Z=1 plane
	//	   (where depth=1).
	//	2. Then the base_map->secondMap() is applied to convert it
	//	   into world-space.
	// - Our goal is to come up with an equivalent GEO_PrimVolumeXform
	//   which goes from index-space to world-space, matching GEO_PrimVDB's
	//   node-centred samples with GEO_PrimVolume's cell-centered samples.
	// - The gotcha here is that callers use fromVoxelSpace() which will
	//   first do a mapping of [0,1] to [-1,+1] which we need to undo.

	NonlinearFrustumMap::ConstPtr map_ptr = geoStandardFrustumMapPtr(*this);
	const NonlinearFrustumMap& frustum_map = *map_ptr;

	// Math below only handles NonlinearFrustumMap's with a depth of 1.
	UT_ASSERT(frustum_map.getDepth() == 1);
	fpreal taper = frustum_map.getTaper();

	// We need to create a taper value for fromVoxelSpace()'s incoming
	// Houdini index space coordinate, so the bbox we want to taper with is
	// actually the Houdini index bbox.
	UT_BoundingBox bbox;
	getFrustumBounds(bbox);

	fpreal x = bbox.xsize();
	fpreal y = bbox.ysize();
	fpreal z = bbox.zsize();

	Vec3d real_min(bbox.xmin(), bbox.ymin(), bbox.zmin());
	Vec3d real_max(bbox.xmax(), bbox.ymax(), bbox.zmax());

	// Compute a new taper based on the expected ratio of these two
	// z positions
	fpreal z_min = real_min.z();
	fpreal z_max = real_max.z();

	//
	// If t = (1+g(1-a))/(1+g(1-b)) then g = (1-t)/(t(1-b) - (1-a))
	//    where t = taper; g = new_taper - 1, a = z_min, b = z_max;
	//
	fpreal new_gamma = (1 - taper) / (taper * (1 - z_max) - (1 - z_min));
	new_taper = new_gamma + 1;

	// Since we are tapering the index space, the taper map adds a
	// scale and a shear so we find these and invert them

	fpreal x_max_pos = Local::taper(real_max.x(), z_max, new_gamma);
	fpreal x_min_pos = Local::taper(real_min.x(), z_max, new_gamma);
	// Now, move x_max_pos = -x_min_pos with a shear
	fpreal x_scale = x_max_pos - x_min_pos;
	fpreal shear_x = 0.5 * x_scale - x_max_pos;

	// Do the same for y
	fpreal y_max_pos = Local::taper(real_max.y(), z_max, new_gamma);
	fpreal y_min_pos = Local::taper(real_min.y(), z_max, new_gamma);
	fpreal y_scale = y_max_pos - y_min_pos;
	fpreal shear_y = 0.5 * y_scale - y_max_pos;

	transform.translate(0, 0, -2*(z_min - 0.5));

	// Scale z so that our frustum depth range is 0-1
	transform.scale(1, 1, 0.5/z);

	// Apply the shear
	OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
	if (taper != 1.0)
	OPENVDB_NO_FP_EQUALITY_WARNING_END
	{
	    fpreal z_i = 1.0 / (taper - 1);
	    transform.translate(0, 0, -z_i-1.0);
	    transform.shear(0, -shear_x / z_i, -shear_y / z_i);
	    transform.translate(0, 0, z_i+1.0);
	}
	else
	{
	    transform.translate(shear_x, shear_y, 0.0);
	}

	transform.scale(1.0/x_scale, 1.0/y_scale, 1.0);

	// Scale by 1/taper to convert taper definitions
	transform.scale(1.0 / taper, 1.0 / taper, 1.0);

	// Account for aspect ratio
	transform.scale(1, y/x, 1);

	Mat4d mat4 = frustum_map.secondMap().getMat4();
	transform *= UTvdbConvert(mat4);
    }
    else
    {
	// We only deal with nonlinear maps that are frustum maps
	UT_ASSERT(base_map->isLinear()
		  && "Found unexpected nonlinear MapBase.");

	// Since VDB's transform is already from index-space to world-space, we
	// just need to undo the [0,1] -> [-1,+1] mapping that fromVoxelSpace()
	// does before transforming by myXform/myCenter. The math is thus:
	//     scale(1/2)*translate(0.5)
	// But we also want to shift VDB's node-centred samples to match
	// GEO_PrimVolume's cell-centered ones so we want:
	//     scale(1/2)*translate(0.5)*translate(-0.5)
	// This reduces down to just scale(1/2)
	//
	transform.scale(0.5, 0.5, 0.5);

	transform *= UTvdbConvert(base_map->getAffineMap()->getMat4());
    }

    GEO_PrimVolumeXform result;
    result.myXform = transform;
    transform.getTranslates(result.myCenter);

    OPENVDB_NO_FP_EQUALITY_WARNING_BEGIN
    result.myHasTaper = (new_taper != 1.0);
    OPENVDB_NO_FP_EQUALITY_WARNING_END

    transform.invert();
    result.myInverseXform = transform;

    result.myTaperX = new_taper;
    result.myTaperY = new_taper;

    return result;
}

bool
GEO_PrimVDB::isSDF() const
{
    if (getGrid().getGridClass() == openvdb::GRID_LEVEL_SET)
	return true;

    return false;
}

fpreal
GEO_PrimVDB::getTaper() const
{
    return getSpaceTransform().myTaperX;
}

void
GEO_PrimVDB::reverse()
{
}

UT_Vector3
GEO_PrimVDB::computeNormal() const
{
    return UT_Vector3(0, 0, 0);
}



template <typename GridType>
static void
geo_calcVolume(const GridType &grid, fpreal &volume)
{
	bool calculated = false;
    if (grid.getGridClass() == openvdb::GRID_LEVEL_SET) {
		try {
 	    	volume = openvdb::tools::levelSetVolume(grid);
	    	calculated = true;
		} catch (std::exception& /*e*/) { } // ignore
	}

	// Simply account for the total number of active voxels
    if (!calculated) {
        const openvdb::Vec3d size = grid.voxelSize();
        volume = size[0] * size[1] * size[2] * grid.activeVoxelCount();
    }
}

fpreal
GEO_PrimVDB::calcVolume(const UT_Vector3 &) const
{
    fpreal volume = 0;
    UTvdbCallAllType(getStorageType(), geo_calcVolume, getGrid(), volume);
    return volume;
}

template <typename GridType>
static void
geo_calcArea(const GridType &grid, fpreal &area)
{
	bool calculated = false;
    if (grid.getGridClass() == openvdb::GRID_LEVEL_SET) {
    	try {
			area = openvdb::tools::levelSetArea(grid);
		} catch (std::exception& /*e*/) { } // ignore
    }

    if (!calculated) {
        typedef typename GridType::TreeType::LeafCIter LeafIter;
        typedef typename GridType::TreeType::LeafNodeType::ValueOnCIter VoxelIter;
        using openvdb::Coord;
        const Coord normals[] = {Coord(0,0,-1), Coord(0,0,1), Coord(-1,0,0),
                                 Coord(1,0,0), Coord(0,-1,0), Coord(0,1,0)};
        // NOTE: we assume rectangular prism voxels
        openvdb::Vec3d voxel_size = grid.voxelSize();
        const fpreal areas[] = {fpreal(voxel_size.x() * voxel_size.y()),
                                fpreal(voxel_size.x() * voxel_size.y()),
                                fpreal(voxel_size.y() * voxel_size.z()),
                                fpreal(voxel_size.y() * voxel_size.z()),
                                fpreal(voxel_size.z() * voxel_size.x()),
                                fpreal(voxel_size.z() * voxel_size.x())};
        area = 0;
        for (LeafIter leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
            // Visit all the active voxels in this leaf node.
            for (VoxelIter iter = leaf->cbeginValueOn(); iter; ++iter) {
                // Iterate through all the neighboring voxels
                for (int i=0; i<6; i++)
                    if (!grid.tree().isValueOn(iter.getCoord() + normals[i])) area += areas[i];
            }
        }
    }
}

fpreal
GEO_PrimVDB::calcArea() const
{
    // Calculate the surface area of all the exterior voxels.
    fpreal area = 0;
    UTvdbCallAllType(getStorageType(), geo_calcArea, getGrid(), area);
    return area;
}

void
GEO_PrimVDB::enlargePointBounds(UT_BoundingBox &box) const
{
    UT_BoundingBox qbox;
    if (getBBox(&qbox))
	box.enlargeBounds(qbox);
}

bool
GEO_PrimVDB::enlargeBoundingBox(UT_BoundingRect &box,
				const GA_Attribute *P) const
{
    const GA_Detail	&gdp = getDetail();

    if (!P)
	P = gdp.getP();
    else if (P != gdp.getP())
	return GEO_Primitive::enlargeBoundingBox(box, P);

    UT_BoundingBox	my_bbox;
    if (getBBox(&my_bbox))
    {
	box.enlargeBounds(my_bbox.xmin(), my_bbox.ymin());
	box.enlargeBounds(my_bbox.xmax(), my_bbox.ymax());
	return true;
    }
    return false;
}

bool
GEO_PrimVDB::enlargeBoundingBox(UT_BoundingBox &box,
				const GA_Attribute *P) const
{
    const GA_Detail	&gdp = getDetail();

    if (!P)
	P = gdp.getP();
    else if (P != gdp.getP())
	return GEO_Primitive::enlargeBoundingBox(box, P);

    UT_BoundingBox	my_bbox;
    if (getBBox(&my_bbox))
    {
	box.enlargeBounds(my_bbox);
	return true;
    }
    return false;
}

bool
GEO_PrimVDB::enlargeBoundingSphere(UT_BoundingSphere &sphere,
				   const GA_Attribute *P) const
{
    const GA_Detail	&gdp = getDetail();

    if (!P)
	P = gdp.getP();
    else if (P != gdp.getP())
	return GEO_Primitive::enlargeBoundingSphere(sphere, P);

    addToBSphere(&sphere);
    return true;
}

int64
GEO_PrimVDB::getBaseMemoryUsage() const
{
    exint mem = 0;
    if (hasGrid())
        mem += getGrid().memUsage();
    return mem;
}

GA_Size
GEO_PrimVDB::getVertexCount(void) const
{
    return 1;
}


template <typename GridType>
static inline typename GridType::ValueType
geo_doubleToGridValue(double val)
{
    typedef typename GridType::ValueType ValueT;
    // This ugly construction avoids compiler warnings when,
    // for example, initializing an openvdb::Vec3i with a double.
    return ValueT(openvdb::zeroVal<ValueT>() + val);
}


template <typename GridType>
static fpreal
geo_sampleGrid(const GridType &grid, const UT_Vector3 &pos)
{
    const openvdb::math::Transform &	xform = grid.transform();
    openvdb::math::Vec3d		vpos;
    typename GridType::ValueType	value;

    vpos = openvdb::math::Vec3d(pos.x(), pos.y(), pos.z());
    vpos = xform.worldToIndex(vpos);

    openvdb::tools::BoxSampler::sample(grid.tree(), vpos, value);

    fpreal result = value;

    return result;
}

template <typename GridType>
static UT_Vector3D
geo_sampleGridV3(const GridType &grid, const UT_Vector3 &pos)
{
    const openvdb::math::Transform &	xform = grid.transform();
    openvdb::math::Vec3d		vpos;
    typename GridType::ValueType	value;

    vpos = openvdb::math::Vec3d(pos.x(), pos.y(), pos.z());
    vpos = xform.worldToIndex(vpos);

    openvdb::tools::BoxSampler::sample(grid.tree(), vpos, value);

    UT_Vector3D result;
    result.x() = double(value[0]);
    result.y() = double(value[1]);
    result.z() = double(value[2]);

    return result;
}

template <typename GridType, typename T>
static void
geo_sampleGridMany(const GridType &grid,
		T *f, int stride,
		const UT_Vector3 *pos,
		int num)
{
    typename GridType::ConstAccessor accessor = grid.getAccessor();

    const openvdb::math::Transform &	xform = grid.transform();
    openvdb::math::Vec3d		vpos;
    typename GridType::ValueType	value;


    for (int i = 0; i < num; i++)
    {
	vpos = openvdb::math::Vec3d(pos[i].x(), pos[i].y(), pos[i].z());
	vpos = xform.worldToIndex(vpos);

	openvdb::tools::BoxSampler::sample(accessor, vpos, value);

	*f = T(value);
	f += stride;
    }
}

template <typename GridType, typename T>
static void
geo_sampleVecGridMany(const GridType &grid,
		T *f, int stride,
		const UT_Vector3 *pos,
		int num)
{
    typename GridType::ConstAccessor accessor = grid.getAccessor();

    const openvdb::math::Transform &	xform = grid.transform();
    openvdb::math::Vec3d		vpos;
    typename GridType::ValueType	value;


    for (int i = 0; i < num; i++)
    {
	vpos = openvdb::math::Vec3d(pos[i].x(), pos[i].y(), pos[i].z());
	vpos = xform.worldToIndex(vpos);

	openvdb::tools::BoxSampler::sample(accessor, vpos, value);

	f->x() = value[0];
	f->y() = value[1];
	f->z() = value[2];
	f += stride;
    }
}

static fpreal
geoEvaluateVDB(const GEO_PrimVDB *vdb, const UT_Vector3 &pos)
{
    UTvdbReturnScalarType(vdb->getStorageType(), geo_sampleGrid, vdb->getGrid(), pos);
    if (vdb->getStorageType() == UT_VDB_BOOL) {
        return geo_sampleGrid<openvdb::BoolGrid>(
            UTvdbGridCast<openvdb::BoolGrid>(vdb->getGrid()), pos);
    }
    return 0;
}

static UT_Vector3D
geoEvaluateVDB_V3(const GEO_PrimVDB *vdb, const UT_Vector3 &pos)
{
    UTvdbReturnVec3Type(vdb->getStorageType(),
			  geo_sampleGridV3, vdb->getGrid(), pos);
    return UT_Vector3D(0, 0, 0);
}

static void
geoEvaluateVDBMany(const GEO_PrimVDB *vdb, float *f, int stride, const UT_Vector3 *pos, int num)
{
    UTvdbReturnScalarType(vdb->getStorageType(),
	    geo_sampleGridMany, vdb->getGrid(), f, stride, pos, num);
    for (int i = 0; i < num; i++)
    {
	*f = 0;
	f += stride;
    }
}

static void
geoEvaluateVDBMany(const GEO_PrimVDB *vdb, int *f, int stride, const UT_Vector3 *pos, int num)
{
    UTvdbReturnScalarType(vdb->getStorageType(),
	    geo_sampleGridMany, vdb->getGrid(), f, stride, pos, num);
    for (int i = 0; i < num; i++)
    {
	*f = 0;
	f += stride;
    }
}

static void
geoEvaluateVDBMany(const GEO_PrimVDB *vdb, UT_Vector3 *f, int stride, const UT_Vector3 *pos, int num)
{
    UTvdbReturnVec3Type(vdb->getStorageType(),
	    geo_sampleVecGridMany, vdb->getGrid(), f, stride, pos, num);
    for (int i = 0; i < num; i++)
    {
	*f = 0;
	f += stride;
    }
}

fpreal
GEO_PrimVDB::getValueF(const UT_Vector3 &pos) const
{
    return geoEvaluateVDB(this, pos);
}

UT_Vector3D
GEO_PrimVDB::getValueV3(const UT_Vector3 &pos) const
{
    return geoEvaluateVDB_V3(this, pos);
}

void
GEO_PrimVDB::getValues(float *f, int stride, const UT_Vector3 *pos, int num) const
{
    return geoEvaluateVDBMany(this, f, stride, pos, num);
}

void
GEO_PrimVDB::getValues(int *f, int stride, const UT_Vector3 *pos, int num) const
{
    return geoEvaluateVDBMany(this, f, stride, pos, num);
}

void
GEO_PrimVDB::getValues(UT_Vector3 *f, int stride, const UT_Vector3 *pos, int num) const
{
    return geoEvaluateVDBMany(this, f, stride, pos, num);
}

namespace // anonymous
{

template <bool NORMALIZE>
class geo_EvalGradients
{
public:
    geo_EvalGradients(
	    UT_Vector3 *gradients,
	    int stride,
	    const UT_Vector3 *positions,
	    int num_positions)
	: myGradients(gradients)
	, myStride(stride)
	, myPos(positions)
	, myNumPos(num_positions)
    {
    }

    template<typename GridT>
    void operator()(const GridT &grid)
    {
	using namespace openvdb;
	typedef typename GridT::ConstAccessor	AccessorT;
	typedef typename GridT::ValueType	ValueT;

	const math::Transform &	    xform = grid.transform();
	const math::Vec3d	    dim = grid.voxelSize();
	const double		    vox_size = SYSmin(dim[0], dim[1], dim[2]);
	const double		    h = 0.5 * vox_size;
	const math::Vec3d	    mask[] =
	    { math::Vec3d(h, 0, 0)
	    , math::Vec3d(0, h, 0)
	    , math::Vec3d(0, 0, h)
	    };
	AccessorT		    accessor = grid.getConstAccessor();
	UT_Vector3 *		    gradient = myGradients;

	for (int i = 0; i < myNumPos; i++, gradient += myStride)
	{
	    const math::Vec3d	pos(myPos[i].x(), myPos[i].y(), myPos[i].z());

	    for (int j = 0; j < 3; j++)
	    {
		const math::Vec3d   vpos0 = xform.worldToIndex(pos - mask[j]);
		const math::Vec3d   vpos1 = xform.worldToIndex(pos + mask[j]);
		ValueT		    v0, v1;

		tools::BoxSampler::sample<AccessorT>(accessor, vpos0, v0);
		tools::BoxSampler::sample<AccessorT>(accessor, vpos1, v1);
		if (NORMALIZE)
		    (*gradient)(j) = (v1 - v0);
		else
		    (*gradient)(j) = (v1 - v0) / vox_size;
	    }
	    if (NORMALIZE)
		gradient->normalize();
	}
    }

private:
    UT_Vector3 *	myGradients;
    int			myStride;
    const UT_Vector3 *	myPos;
    int			myNumPos;
};

} // namespace anonymous

bool
GEO_PrimVDB::evalGradients(
	UT_Vector3 *gradients,
	int stride,
	const UT_Vector3 *pos,
	int num_pos,
	bool normalize) const
{
    if (normalize)
    {
	geo_EvalGradients<true> eval(gradients, stride, pos, num_pos);
	return UTvdbProcessTypedGridScalar(getStorageType(), getGrid(), eval);
    }
    else
    {
	geo_EvalGradients<false> eval(gradients, stride, pos, num_pos);
	return UTvdbProcessTypedGridScalar(getStorageType(), getGrid(), eval);
    }
}

bool
GEO_PrimVDB::isAligned(const GEO_PrimVDB *vdb) const
{
    if (getGrid().transform() == vdb->getGrid().transform())
	return true;
    return false;
}

bool
GEO_PrimVDB::isActiveRegionMatched(const GEO_PrimVDB *vdb) const
{
    if (!isAligned(vdb))
	return false;
    // Ideally we'd invoke hasSameTopology?
    return vdb->getGrid().baseTreePtr() == getGrid().baseTreePtr();
}

void
GEO_PrimVDB::indexToPos(int x, int y, int z, UT_Vector3 &pos) const
{
    openvdb::math::Vec3d		vpos;

    vpos = openvdb::math::Vec3d(x, y, z);
    vpos = getGrid().indexToWorld(vpos);
    pos = UT_Vector3(vpos[0], vpos[1], vpos[2]);
}

void
GEO_PrimVDB::findexToPos(UT_Vector3 idx, UT_Vector3 &pos) const
{
    openvdb::math::Vec3d		vpos;

    vpos = openvdb::math::Vec3d(idx.x(), idx.y(), idx.z());
    vpos = getGrid().indexToWorld(vpos);
    pos = UT_Vector3(vpos[0], vpos[1], vpos[2]);
}

void
GEO_PrimVDB::posToIndex(UT_Vector3 pos, int &x, int &y, int &z) const
{
    openvdb::math::Vec3d vpos(pos.data());
    openvdb::math::Coord
	coord = getGrid().transform().worldToIndexCellCentered(vpos);
    x = coord.x();
    y = coord.y();
    z = coord.z();
}

void
GEO_PrimVDB::posToIndex(UT_Vector3 pos, UT_Vector3 &index) const
{
    openvdb::math::Vec3d		vpos;

    vpos = openvdb::math::Vec3d(pos.x(), pos.y(), pos.z());
    vpos = getGrid().worldToIndex(vpos);

    index = UTvdbConvert(vpos);
}

template <typename GridType>
static fpreal
geo_sampleIndex(const GridType &grid, int ix, int iy, int iz)
{
    openvdb::math::Coord		xyz;
    typename GridType::ValueType	value;

    xyz = openvdb::math::Coord(ix, iy, iz);

    value = grid.tree().getValue(xyz);

    fpreal result = value;

    return result;
}

template <typename GridType>
static UT_Vector3D
geo_sampleIndexV3(const GridType &grid, int ix, int iy, int iz)
{
    openvdb::math::Coord		xyz;
    typename GridType::ValueType	value;

    xyz = openvdb::math::Coord(ix, iy, iz);

    value = grid.tree().getValue(xyz);

    UT_Vector3D result;

    result.x() = double(value[0]);
    result.y() = double(value[1]);
    result.z() = double(value[2]);

    return result;
}

template <typename GridType, typename T>
static void
geo_sampleIndexMany(const GridType &grid,
		T *f, int stride,
		const int *ix, const int *iy, const int *iz,
		int num)
{
    typename GridType::ConstAccessor accessor = grid.getAccessor();

    openvdb::math::Coord		xyz;
    typename GridType::ValueType	value;

    for (int i = 0; i < num; i++)
    {
	xyz = openvdb::math::Coord(ix[i], iy[i], iz[i]);

	value = accessor.getValue(xyz);

	*f = T(value);
	f += stride;
    }
}

template <typename GridType, typename T>
static void
geo_sampleVecIndexMany(const GridType &grid,
		T *f, int stride,
		const int *ix, const int *iy, const int *iz,
		int num)
{
    typename GridType::ConstAccessor accessor = grid.getAccessor();

    openvdb::math::Coord		xyz;
    typename GridType::ValueType	value;

    for (int i = 0; i < num; i++)
    {
	xyz = openvdb::math::Coord(ix[i], iy[i], iz[i]);

	value = accessor.getValue(xyz);

	f->x() = value[0];
	f->y() = value[1];
	f->z() = value[2];
	f += stride;
    }
}

static fpreal
geoEvaluateIndexVDB(const GEO_PrimVDB *vdb,
		int ix, int iy, int iz)
{
    UTvdbReturnScalarType(vdb->getStorageType(),
			  geo_sampleIndex, vdb->getGrid(), ix, iy, iz);

    return 0.0;
}

static UT_Vector3D
geoEvaluateIndexVDB_V3(const GEO_PrimVDB *vdb,
		int ix, int iy, int iz)
{
    UTvdbReturnVec3Type(vdb->getStorageType(),
			  geo_sampleIndexV3, vdb->getGrid(), ix, iy, iz);

    return UT_Vector3D(0.0, 0, 0);
}

static void
geoEvaluateIndexVDBMany(const GEO_PrimVDB *vdb,
		float *f, int stride,
		const int *ix, const int *iy, const int *iz,
		int num)
{
    UTvdbReturnScalarType(vdb->getStorageType(),
	    geo_sampleIndexMany, vdb->getGrid(), f, stride, ix, iy, iz, num);
    for (int i = 0; i < num; i++)
    {
	*f = 0;
	f += stride;
    }
}

static void
geoEvaluateIndexVDBMany(const GEO_PrimVDB *vdb,
		int *f, int stride,
		const int *ix, const int *iy, const int *iz,
		int num)
{
    UTvdbReturnScalarType(vdb->getStorageType(),
	    geo_sampleIndexMany, vdb->getGrid(), f, stride, ix, iy, iz, num);
    for (int i = 0; i < num; i++)
    {
	*f = 0;
	f += stride;
    }
}

static void
geoEvaluateIndexVDBMany(const GEO_PrimVDB *vdb,
		UT_Vector3 *f, int stride,
		const int *ix, const int *iy, const int *iz,
		int num)
{
    UTvdbReturnVec3Type(vdb->getStorageType(),
	    geo_sampleVecIndexMany, vdb->getGrid(), f, stride, ix, iy, iz, num);
    for (int i = 0; i < num; i++)
    {
	*f = 0;
	f += stride;
    }
}

fpreal
GEO_PrimVDB::getValueAtIndexF(int ix, int iy, int iz) const
{
    return geoEvaluateIndexVDB(this, ix, iy, iz);
}

UT_Vector3D
GEO_PrimVDB::getValueAtIndexV3(int ix, int iy, int iz) const
{
    return geoEvaluateIndexVDB_V3(this, ix, iy, iz);
}

void
GEO_PrimVDB::getValuesAtIndices(float *f, int stride, const int *ix, const int *iy, const int *iz, int num) const
{
    geoEvaluateIndexVDBMany(this, f, stride, ix, iy, iz, num);
}

void
GEO_PrimVDB::getValuesAtIndices(int *f, int stride, const int *ix, const int *iy, const int *iz, int num) const
{
    geoEvaluateIndexVDBMany(this, f, stride, ix, iy, iz, num);
}

void
GEO_PrimVDB::getValuesAtIndices(UT_Vector3 *f, int stride, const int *ix, const int *iy, const int *iz, int num) const
{
    geoEvaluateIndexVDBMany(this, f, stride, ix, iy, iz, num);
}

UT_Vector3
GEO_PrimVDB::getGradient(const UT_Vector3 &pos) const
{
    UT_Vector3		grad;

    grad = 0;

    evalGradients(&grad, 1, &pos, 1, false);

    return grad;
}


////////////////////////////////////////


namespace {

// Functor for use with UTvdbProcessTypedGridVec3() to apply a transform
// to the voxel values of vector-valued grids
struct gu_VecXformOp
{
    openvdb::Mat4d mat;
    gu_VecXformOp(const openvdb::Mat4d& _mat): mat(_mat) {}
    template<typename GridT> void operator()(GridT& grid) const
    {
        openvdb::tools::transformVectors(grid, mat);
    }
};

} // unnamed namespace


void
GEO_PrimVDB::transform(const UT_Matrix4 &mat)
{
    if (!hasGrid()) return;

    try {
        using openvdb::GridBase;
        using namespace openvdb::math;

        // Get the transform
        const GridBase&	  const_grid = getConstGrid();
        MapBase::ConstPtr base_map = const_grid.transform().baseMap();
        Mat4d base_mat4 = base_map->getAffineMap()->getMat4();

        // Get the 3x3 subcomponent of the matrix
        Vec3d translation = base_mat4.getTranslation();
        Mat3d vdbmatrix = base_mat4.getMat3();

        // Multiply our mat with the mat3
        UT_Matrix3D transformed(mat);
        transformed = UTvdbConvert(vdbmatrix) * transformed;

        // Put it into a mat4 and translate it
        UT_Matrix4D final;
        final = transformed;
        final.setTranslates(UTvdbConvert(translation));

        // Make an affine matrix out of it
        AffineMap::Ptr map(geoCreateAffineMap<AffineMap>(final));

        // Set the affine matrix from our base_map into this map
        MapBase::Ptr result = simplify(map);
        if (base_map->isType<NonlinearFrustumMap>()) {
            const NonlinearFrustumMap& frustum_map =
                *const_grid.transform().constMap<NonlinearFrustumMap>();

            MapBase::Ptr new_frustum_map (new NonlinearFrustumMap(
                frustum_map.getBBox(),
                frustum_map.getTaper(),
                frustum_map.getDepth(),
                result));

            result = new_frustum_map;
        }

        // This sets the vertex position to `translation` as well
        myGridAccessor.setTransform(Transform(result), *this);

        // If (and only if) the grid is vector-valued, apply the transform to
        // each voxel's value.
        if (const_grid.getVectorType() != openvdb::VEC_INVARIANT) {
            gu_VecXformOp op(UTvdbConvert(UT_Matrix4D(mat)));
            GEOvdbProcessTypedGridVec3(*this, op, /*make_unique*/true);
        }

    } catch (std::exception& /*e*/) {
        UT_ASSERT(!"Failed to apply transform");
    }
}


void
GEO_PrimVDB::copyGridFrom(const GEO_PrimVDB& src_prim)
{
    setGrid(src_prim.getGrid()); // makes a shallow copy

    // Copy the source primitive's grid serial numbers.
    myTreeUniqueId.exchange(src_prim.getTreeUniqueId());
    myMetadataUniqueId.exchange(src_prim.getMetadataUniqueId());
    myTransformUniqueId.exchange(src_prim.getTransformUniqueId());
}


// If myGrid's tree is shared, replace the tree with a deep copy of itself.
// Note: myGrid's metadata and transform are assumed to never be shared
// (see setGrid()).
void
GEO_PrimVDB::GridAccessor::makeGridUnique()
{
    if (myGrid) {
	UT_ASSERT(myGrid.unique());
	openvdb::TreeBase::Ptr localTreePtr = myGrid->baseTreePtr();
	if (localTreePtr.use_count() > 2) { // myGrid + localTreePtr = 2
	    myGrid->setTree(myGrid->constBaseTree().copy());
	}
    }
}


void
GEO_PrimVDB::setTransform4(const UT_Matrix4 &xform4)
{
    setTransform4(static_cast<UT_DMatrix4>(xform4));
}

void
GEO_PrimVDB::setTransform4(const UT_DMatrix4 &xform4)
{
    using namespace openvdb::math;
    myGridAccessor.setTransform(*geoCreateLinearTransform(xform4), *this);
}

void
GEO_PrimVDB::getRes(int &rx, int &ry, int &rz) const
{
    using namespace openvdb;

    const GridBase &	grid = getGrid();
    const math::Vec3d	dim = grid.evalActiveVoxelDim().asVec3d();

    rx = static_cast<int>(dim[0]);
    ry = static_cast<int>(dim[1]);
    rz = static_cast<int>(dim[2]);
}

fpreal
GEO_PrimVDB::getVoxelDiameter() const
{
    UT_Vector3		p1, p2;

    indexToPos(0, 0, 0, p1);
    indexToPos(1, 1, 1, p2);

    p2 -= p1;

    return p2.length();
}

UT_Vector3
GEO_PrimVDB::getVoxelSize() const
{
    UT_Vector3		p1, p2;
    UT_Vector3		vsize;

    indexToPos(0, 0, 0, p1);

    indexToPos(1, 0, 0, p2);
    p2 -= p1;
    vsize.x() = p2.length();

    indexToPos(0, 1, 0, p2);
    p2 -= p1;
    vsize.y() = p2.length();

    indexToPos(0, 0, 1, p2);
    p2 -= p1;
    vsize.z() = p2.length();

    return vsize;
}


bool
GEO_PrimVDB::getFrustumBounds(UT_BoundingBox &idxbox) const
{
    using namespace openvdb;
    using namespace openvdb::math;
    using openvdb::CoordBBox;
    using openvdb::Vec3d;

    idxbox.makeInvalid();

    // See if we have a non-linear map, this is the sign
    // we want to be bounded.
    MapBase::ConstPtr base_map = getGrid().transform().baseMap();
    if (base_map->isType<NonlinearFrustumMap>())
    {
	const NonlinearFrustumMap& frustum_map =
	    *getGrid().transform().constMap<NonlinearFrustumMap>();

	// The returned idxbox is intended to be used with
	// getIndexSpaceTransform() which will shift it by -0.5 voxel. So we
	// need to add +0.5 to compensate.
	BBoxd bbox = frustum_map.getBBox();
	bbox.translate(Vec3d(+0.5));

	idxbox.initBounds( UTvdbConvert(bbox.min()) );
	idxbox.enlargeBounds( UTvdbConvert(bbox.max()) );

	return true;
    }

    return false;
}

static bool
geoGetFrustumBoundsFromVDB(const GEO_PrimVDB *vdb, openvdb::CoordBBox &box)
{
    using namespace openvdb;

    UT_BoundingBox		clip;
    bool			doclip;

    doclip = vdb->getFrustumBounds(clip);
    if (doclip)
    {
	box = CoordBBox( Coord( (int)SYSrint(clip.xmin()), (int)SYSrint(clip.ymin()), (int)SYSrint(clip.zmin()) ),
			 Coord( (int)SYSrint(clip.xmax()), (int)SYSrint(clip.ymax()), (int)SYSrint(clip.zmax()) ) );
    }
    return doclip;
}

// The result of the intersection of active regions goes into grid_a
template <typename GridTypeA, typename GridTypeB>
static void
geoIntersect(GridTypeA& grid_a, const GridTypeB &grid_b)
{
    typename GridTypeA::Accessor 	access_a = grid_a.getAccessor();
    typename GridTypeB::ConstAccessor 	access_b = grid_b.getAccessor();

    // For each on value in a, set it off if b is also off
    for (typename GridTypeA::ValueOnCIter
	 iter = grid_a.cbeginValueOn(); iter; ++iter)
    {
	openvdb::CoordBBox bbox = iter.getBoundingBox();
	for (int k=bbox.min().z(); k<=bbox.max().z(); k++)
	{
	    for (int j=bbox.min().y(); j<=bbox.max().y(); j++)
	    {
		for (int i=bbox.min().x(); i<=bbox.max().x(); i++)
		{
		    openvdb::Coord coord(i, j, k);
		    if (!access_b.isValueOn(coord))
		    {
			access_a.setValue(coord, grid_a.background());
			access_a.setValueOff(coord);
		    }
		}
	    }
	}
    }
}

template <typename GridType>
static void
geoActivateBBox(GridType& grid,
		const openvdb::CoordBBox &bbox,
		bool setvalue,
		double value,
		GEO_PrimVDB::ActivateOperation operation,
		bool doclip,
		const openvdb::CoordBBox &clipbox)
{
    typename GridType::Accessor 	access = grid.getAccessor();

    switch (operation)
    {
	case GEO_PrimVDB::ACTIVATE_UNION: // Union
	    if (doclip)
	    {
		openvdb::CoordBBox	clipped = bbox;
		clipped = bbox;
		clipped.min().maxComponent(clipbox.min());
		clipped.max().minComponent(clipbox.max());

		geoActivateBBox(grid, clipped, setvalue, value,
				operation,
				false,
				clipped);
		break;
	    }
	    if (setvalue)
	    {
		grid.fill(bbox, geo_doubleToGridValue<GridType>(value), /*active*/true);
	    }
	    else
	    {
		// Just activate
		for (int k=bbox.min().z(); k<=bbox.max().z(); k++)
		{
		    for (int j=bbox.min().y(); j<=bbox.max().y(); j++)
		    {
			for (int i=bbox.min().x(); i<=bbox.max().x(); i++)
			{
			    openvdb::Coord coord(i, j, k);
			    access.setValueOn(coord);
			}
		    }
		}
	    }
	    break;
	case GEO_PrimVDB::ACTIVATE_INTERSECT: // Intersect
	    for (typename GridType::ValueOnCIter
		 iter = grid.cbeginValueOn(); iter; ++iter)
	    {
		openvdb::CoordBBox nodebbox = iter.getBoundingBox();

		// If there is no overlap, we set to off.
		if (!bbox.hasOverlap(nodebbox))
		{
		    grid.fill(nodebbox, grid.background(), /*active*/false);
		}
		else
		{
		    // Check each voxel.
		    for (int k=nodebbox.min().z(); k<=nodebbox.max().z(); k++)
		    {
			for (int j=nodebbox.min().y(); j<=nodebbox.max().y(); j++)
			{
			    for (int i=nodebbox.min().x(); i<=nodebbox.max().x(); i++)
			    {
				openvdb::Coord coord(i,j,k);

				if (!bbox.isInside(coord))
				{
				    // Always set background.
				    access.setValue(coord, grid.background());
				    access.setValueOff(coord);
				}
			    }
			}
		    }
		}
	    }
	    break;
	case GEO_PrimVDB::ACTIVATE_SUBTRACT: // Difference
	    // No matter what, we clear the background colour
	    // for inactive.
	    grid.fill(bbox, grid.background(), /*active*/false);
	    break;
	case GEO_PrimVDB::ACTIVATE_COPY:		// Copy
	    // intersect
	    geoActivateBBox(grid, bbox, setvalue, value, GEO_PrimVDB::ACTIVATE_INTERSECT, doclip, clipbox);
	    // and union
	    geoActivateBBox(grid, bbox, setvalue, value, GEO_PrimVDB::ACTIVATE_UNION, doclip, clipbox);
	    break;
    }
}

void
GEO_PrimVDB::activateIndexBBoxAdapter(const void* bboxPtr,
				      ActivateOperation operation,
				      bool setvalue,
				      fpreal value)
{
    using namespace openvdb;

    // bboxPtr is assumed to point to an openvdb::vX_Y_Z::CoordBBox, for some
    // version X.Y.Z of OpenVDB that may be newer than the one with which
    // libHoudiniGEO.so was built.  This is safe provided that CoordBBox and
    // its member objects are ABI-compatible between the two OpenVDB versions.
    const CoordBBox& bbox = *static_cast<const CoordBBox*>(bboxPtr);

    bool			doclip;
    CoordBBox			clipbox;
    doclip = geoGetFrustumBoundsFromVDB(this, clipbox);

    // Activate based on the parameters and inputs
    UTvdbCallAllType(this->getStorageType(), geoActivateBBox,
		     this->getGrid(),
		     bbox,
		     setvalue,
		     value,
		     operation,
		     doclip, clipbox);
}

// Gets a conservative bounding box that maps to a coordinate
// in index space.
openvdb::CoordBBox
geoMapCoord(const openvdb::CoordBBox& bbox_b,
	    GEO_PrimVolumeXform xform_a,
	    GEO_PrimVolumeXform xform_b)
{
    using openvdb::Coord;
    using openvdb::CoordBBox;
    // Get the eight corners of the voxel
    Coord x = Coord(bbox_b.extents().x(), 0, 0);
    Coord y = Coord(0, bbox_b.extents().y(), 0);
    Coord z = Coord(0, 0, bbox_b.extents().z());
    Coord m = bbox_b.min();

    const Coord corners[] = {
	m, m+z, m+y, m+y+z, m+x, m+x+z, m+x+y, m+x+y+z,
    };

    CoordBBox index_bbox;
    for (int i=0; i<8; i++)
    {
	UT_Vector3 corner = UT_Vector3(corners[i].x(), corners[i].y(), corners[i].z());
	UT_Vector3 index = xform_a.toVoxelSpace(xform_b.fromVoxelSpace(corner));
	Coord coord(int32(index.x()), int32(index.y()), int32(index.z()));
	if (i == 0)
	    index_bbox = CoordBBox(coord, coord);
	else
	    index_bbox.expand(coord);
    }
    return index_bbox;
}

openvdb::CoordBBox
geoMapCoord(const openvdb::Coord& coord_b,
	    GEO_PrimVolumeXform xform_a,
	    GEO_PrimVolumeXform xform_b)
{
    const openvdb::CoordBBox bbox_b(coord_b, coord_b + openvdb::Coord(1,1,1));
    return geoMapCoord(bbox_b, xform_a, xform_b);
}

template <typename GridTypeA, typename GridTypeB>
void
geoUnalignedUnion(GridTypeA &grid_a,
		  const GridTypeB &grid_b,
		  GEO_PrimVolumeXform xform_a,
		  GEO_PrimVolumeXform xform_b,
		  bool setvalue, double value,
		  bool doclip, const openvdb::CoordBBox &clipbox)
{
    typename GridTypeA::Accessor 	access_a = grid_a.getAccessor();
    typename GridTypeB::ConstAccessor 	access_b = grid_b.getAccessor();

    for (typename GridTypeB::ValueOnCIter
	 iter = grid_b.cbeginValueOn(); iter; ++iter)
    {
	openvdb::CoordBBox bbox_b = iter.getBoundingBox();
	openvdb::CoordBBox bbox_a = geoMapCoord(bbox_b, xform_a, xform_b);

	// Set the whole bbox to on
	geoActivateBBox(grid_a, bbox_a, setvalue, value, GEO_PrimVDB::ACTIVATE_UNION,
			    doclip, clipbox);
    }
}

template <typename GridTypeA, typename GridTypeB>
void
geoUnalignedDifference(GridTypeA &grid_a,
		       const GridTypeB &grid_b,
		       GEO_PrimVolumeXform xform_a,
		       GEO_PrimVolumeXform xform_b)
{
    typename GridTypeA::Accessor 	access_a = grid_a.getAccessor();
    typename GridTypeB::ConstAccessor 	access_b = grid_b.getAccessor();

    for (typename GridTypeB::ValueOnCIter
	 iter = grid_b.cbeginValueOn(); iter; ++iter)
    {
	// TODO: is the whole bounding box on here, or not?
	//       do some testing.
	openvdb::CoordBBox bbox_b = iter.getBoundingBox();
	openvdb::CoordBBox bbox_a = geoMapCoord(bbox_b, xform_a, xform_b);

	// Set the whole bbox to off
	grid_a.fill(bbox_a, grid_a.background(), false);
    }
}

template <typename GridType>
static bool
geoContainsActiveVoxels(const openvdb::CoordBBox& bbox,
			GridType&,
			typename GridType::ConstAccessor& access)
{
    for (int k=bbox.min().z(); k<=bbox.max().z(); k++)
    {
	for (int j=bbox.min().y(); j<=bbox.max().y(); j++)
	{
	    for (int i=bbox.min().x(); i<=bbox.max().x(); i++)
	    {
		if (access.isValueOn(openvdb::Coord(i,j,k)))
		{
		    return true;
		}
	    }
	}
    }
    return false;
}

template <typename GridTypeA, typename GridTypeB>
static void
geoUnalignedIntersect(GridTypeA &grid_a,
		      const GridTypeB &grid_b,
		      GEO_PrimVolumeXform xform_a,
		      GEO_PrimVolumeXform xform_b)
{
    typename GridTypeA::Accessor 	access_a = grid_a.getAccessor();
    typename GridTypeB::ConstAccessor 	access_b = grid_b.getAccessor();

    for (typename GridTypeA::ValueOnCIter
	 iter = grid_a.cbeginValueOn(); iter; ++iter)
    {
	openvdb::CoordBBox bbox = iter.getBoundingBox();
	for (int k=bbox.min().z(); k<=bbox.max().z(); k++)
	{
	    for (int j=bbox.min().y(); j<=bbox.max().y(); j++)
	    {
		for (int i=bbox.min().x(); i<=bbox.max().x(); i++)
		{
		    openvdb::Coord coord(i,j,k);
		    openvdb::CoordBBox bbox_b =
			geoMapCoord(coord, xform_b, xform_a);
		    if (!geoContainsActiveVoxels(bbox_b, grid_b, access_b))
		    {
			access_a.setValue(coord, grid_a.background());
			access_a.setValueOff(coord);
		    }
		}
	    }
	}

    }
}

// The result of the union of active regions goes into grid_a
template <typename GridTypeA, typename GridTypeB>
static void
geoUnion(GridTypeA& grid_a, const GridTypeB &grid_b, bool setvalue, double value, bool doclip, const openvdb::CoordBBox &clipbox)
{
    typename GridTypeA::Accessor        access_a = grid_a.getAccessor();
    typename GridTypeB::ConstAccessor   access_b = grid_b.getAccessor();

    if (!doclip && !setvalue) {
        grid_a.tree().topologyUnion(grid_b.tree());
        return;
    }

    // For each on value in b, set a on
    for (typename GridTypeB::ValueOnCIter iter = grid_b.cbeginValueOn(); iter; ++iter) {
        openvdb::CoordBBox bbox = iter.getBoundingBox();
        // Intersect with our destination
        if (doclip) {
            bbox.min().maxComponent(clipbox.min());
            bbox.max().minComponent(clipbox.max());
        }

        for (int k=bbox.min().z(); k<=bbox.max().z(); k++) {
            for (int j=bbox.min().y(); j<=bbox.max().y(); j++) {
                for (int i=bbox.min().x(); i<=bbox.max().x(); i++) {
                    openvdb::Coord coord(i, j, k);
                    if (setvalue) {
                        access_a.setValue(coord, geo_doubleToGridValue<GridTypeA>(value));
                    } else {
                        access_a.setValueOn(coord);
                    }
                }
            }
        }
    }
}

// The result of the union of active regions goes into grid_a
template <typename GridTypeA, typename GridTypeB>
static void
geoDifference(GridTypeA& grid_a, const GridTypeB &grid_b)
{
    typename GridTypeA::Accessor 	access_a = grid_a.getAccessor();
    typename GridTypeB::ConstAccessor 	access_b = grid_b.getAccessor();

    // For each on value in a, set it off if b is on
    for (typename GridTypeA::ValueOnCIter
	 iter = grid_a.cbeginValueOn(); iter; ++iter)
    {
	openvdb::CoordBBox bbox = iter.getBoundingBox();
	for (int k=bbox.min().z(); k<=bbox.max().z(); k++)
	{
	    for (int j=bbox.min().y(); j<=bbox.max().y(); j++)
	    {
		for (int i=bbox.min().x(); i<=bbox.max().x(); i++)
		{
		    openvdb::Coord coord(i, j, k);
		    // TODO: conditional needed? Profile please.
		    if (access_b.isValueOn(coord))
		    {
			access_a.setValue(coord, grid_a.background());
			access_a.setValueOff(coord);
		    }
		}
	    }
	}
    }
}

template <typename GridTypeB>
static void
geoDoUnion(const GridTypeB &grid_b, GEO_PrimVolumeXform xform_b, GEO_PrimVDB &vdb_a, bool setvalue, double value, bool doclip, const openvdb::CoordBBox &clipbox)
{
    // If the transforms are equal, we can do an aligned union
    if (grid_b.transform() == vdb_a.getGrid().transform())
    {
	UTvdbCallAllType(vdb_a.getStorageType(), geoUnion,
			 vdb_a.getGrid(), grid_b, setvalue, value,
			 doclip, clipbox);
    }
    else
    {
	UTvdbCallAllType(vdb_a.getStorageType(), geoUnalignedUnion,
			 vdb_a.getGrid(), grid_b,
			 vdb_a.getIndexSpaceTransform(),
			 xform_b, setvalue, value,
			 doclip, clipbox);
    }
}

template <typename GridTypeB>
static void
geoDoIntersect(const GridTypeB &grid_b, GEO_PrimVolumeXform xform_b,
	       GEO_PrimVDB &vdb_a)
{
    if (grid_b.transform() == vdb_a.getGrid().transform())
    {
	UTvdbCallAllType(vdb_a.getStorageType(),
			 geoIntersect, vdb_a.getGrid(), grid_b);
    }
    else
    {
	UTvdbCallAllType(vdb_a.getStorageType(),
			 geoUnalignedIntersect, vdb_a.getGrid(),
			 grid_b, vdb_a.getIndexSpaceTransform(),
			 xform_b);
    }
}

template <typename GridTypeB>
static void
geoDoDifference(const GridTypeB &grid_b,
		GEO_PrimVolumeXform xform_b,
		GEO_PrimVDB &vdb_a)
{
    if (grid_b.transform() == vdb_a.getGrid().transform())
    {
	UTvdbCallAllType(vdb_a.getStorageType(), geoDifference,
			 vdb_a.getGrid(), grid_b);
    }
    else
    {
	UTvdbCallAllType(vdb_a.getStorageType(), geoUnalignedDifference,
			 vdb_a.getGrid(), grid_b,
			 vdb_a.getIndexSpaceTransform(),
			 xform_b);
    }
}


void
GEO_PrimVDB::activateByVDB(const GEO_PrimVDB *input_vdb,
		ActivateOperation operation,
		bool setvalue, fpreal value)
{
    const openvdb::GridBase& input_grid = input_vdb->getGrid();

    bool				doclip;
    openvdb::CoordBBox			clipbox;

    doclip = geoGetFrustumBoundsFromVDB(this, clipbox);

    switch (operation)
    {
	case GEO_PrimVDB::ACTIVATE_UNION: // Union
	    UTvdbCallAllType(input_vdb->getStorageType(),
			     geoDoUnion, input_grid,
			     input_vdb->getIndexSpaceTransform(),
			     *this,
			     setvalue,
			     value,
			     doclip, clipbox);
	    break;
	case GEO_PrimVDB::ACTIVATE_INTERSECT: // Intersect
	    UTvdbCallAllType(input_vdb->getStorageType(),
			     geoDoIntersect, input_grid,
			     input_vdb->getIndexSpaceTransform(),
			     *this);
	    break;
	case GEO_PrimVDB::ACTIVATE_SUBTRACT: // Difference
	    UTvdbCallAllType(input_vdb->getStorageType(),
			     geoDoDifference, input_grid,
			     input_vdb->getIndexSpaceTransform(),
			     *this);
	    break;
	case GEO_PrimVDB::ACTIVATE_COPY: // Copy
	    UTvdbCallAllType(input_vdb->getStorageType(),
			     geoDoIntersect, input_grid,
			     input_vdb->getIndexSpaceTransform(),
			     *this);
	    UTvdbCallAllType(input_vdb->getStorageType(),
			     geoDoUnion, input_grid,
			     input_vdb->getIndexSpaceTransform(),
			     *this,
			     setvalue,
			     value,
			     doclip, clipbox);
	    break;
    }
}

UT_Matrix4D
GEO_PrimVDB::getTransform4() const
{
    using namespace openvdb;
    using namespace openvdb::math;

    UT_Matrix4D mat4;
    const Transform &gxform = getGrid().transform();
    NonlinearFrustumMap::ConstPtr fmap = gxform.map<NonlinearFrustumMap>();
    if (fmap)
    {
	const openvdb::BBoxd &bbox = fmap->getBBox();
	const openvdb::Vec3d center = bbox.getCenter();
	const openvdb::Vec3d size = bbox.extents();

	// TODO: Use fmap->linearMap() once that actually works
	mat4.identity();
	mat4.translate(-center.x(), -center.y(), -bbox.min().z());
	// NOTE: We scale both XY axes by size.x() because the secondMap()
	//       has the aspect ratio baked in
	mat4.scale(1.0/size.x(), 1.0/size.x(), 1.0/size.z());
	mat4 *= UTvdbConvert(fmap->secondMap().getMat4());
    }
    else
    {
	mat4 = UTvdbConvert(gxform.baseMap()->getAffineMap()->getMat4());
    }
    return mat4;
}

void
GEO_PrimVDB::getLocalTransform(UT_Matrix3D &result) const
{
    result = getTransform4();
}

void
GEO_PrimVDB::setLocalTransform(const UT_Matrix3D &new_mat3)
{
    using namespace openvdb;
    using namespace openvdb::math;

    Transform::Ptr xform;
    UT_Matrix4D new_mat4;
    new_mat4 = new_mat3;
    new_mat4.setTranslates(getDetail().getPos3(vertexPoint(0)));

    const Transform & gxform = getGrid().transform();
    NonlinearFrustumMap::ConstPtr fmap = gxform.map<NonlinearFrustumMap>();
    if (fmap)
    {
	fmap = geoStandardFrustumMapPtr(*this);
	const openvdb::BBoxd &bbox = fmap->getBBox();
	const openvdb::Vec3d center = bbox.getCenter();
	const openvdb::Vec3d size = bbox.extents();

	// TODO: Use fmap->linearMap() once that actually works
	UT_Matrix4D second;
	second.identity();
	second.translate(-0.5, -0.5, 0.0); // adjust for frustum map center
	// NOTE: We scale both XY axes by size.x() because the secondMap()
	//       has the aspect ratio baked in
	second.scale(size.x(), size.x(), size.z());
	second.translate(center.x(), center.y(), bbox.min().z());
	second *= new_mat4;
	xform.reset(new Transform(MapBase::Ptr(
	    new NonlinearFrustumMap(fmap->getBBox(), fmap->getTaper(),
				    /*depth*/1.0,
				    geoCreateAffineMap<MapBase>(second)))));
    }
    else
    {
	xform = geoCreateLinearTransform(new_mat4);
    }
    myGridAccessor.setTransform(*xform, *this);
}

int
GEO_PrimVDB::detachPoints(GA_PointGroup &grp)
{
    int		count = 0;

    if (grp.containsOffset(vertexPoint(0)))
	count++;

    if (count == 0)
	return 0;

    if (count == 1)
	return -2;

    return -1;
}

GA_Primitive::GA_DereferenceStatus
GEO_PrimVDB::dereferencePoint(GA_Offset point, bool)
{
    return vertexPoint(0) == point
		? GA_DEREFERENCE_DESTROY
		: GA_DEREFERENCE_OK;
}

GA_Primitive::GA_DereferenceStatus
GEO_PrimVDB::dereferencePoints(const GA_RangeMemberQuery &point_query, bool)
{
    return point_query.contains(vertexPoint(0))
		? GA_DEREFERENCE_DESTROY
		: GA_DEREFERENCE_OK;
}

///
/// JSON methods
///

namespace { // unnamed

class geo_PrimVDBJSON : public GA_PrimitiveJSON
{
public:
    geo_PrimVDBJSON() {}
    virtual ~geo_PrimVDBJSON() {}

    enum
    {
	geo_TBJ_VERTEX,
	geo_TBJ_VDB,
	geo_TBJ_VDB_VISUALIZATION,
	geo_TBJ_ENTRIES
    };

    const GEO_PrimVDB	*vdb(const GA_Primitive *p) const
			{ return static_cast<const GEO_PrimVDB *>(p); }
    GEO_PrimVDB		*vdb(GA_Primitive *p) const
			{ return static_cast<GEO_PrimVDB *>(p); }

    virtual int		getEntries() const
			{ return geo_TBJ_ENTRIES; }

    virtual const char *
    getKeyword(int i) const
    {
	switch (i)
	{
	    case geo_TBJ_VERTEX:	    return "vertex";
	    case geo_TBJ_VDB:		    return "vdb";
	    case geo_TBJ_VDB_VISUALIZATION: return "vdbvis";
	    case geo_TBJ_ENTRIES:	    break;
	}
	UT_ASSERT(0);
	return NULL;
    }

    virtual bool
    shouldSaveField(const GA_Primitive*, int i, const GA_SaveMap&) const
    {
	switch (i)
	{
	    case geo_TBJ_VERTEX:	    return true;
	    case geo_TBJ_VDB:		    return true;
	    case geo_TBJ_VDB_VISUALIZATION: return true;
	    case geo_TBJ_ENTRIES:	    break;
	}
	UT_ASSERT(0);
	return false;
    }

    virtual bool
    saveField(const GA_Primitive *pr, int i, UT_JSONWriter &w,
	      const GA_SaveMap &map) const
    {
	switch (i)
	{
	    case geo_TBJ_VERTEX:
	    {
		GA_Offset vtx = vdb(pr)->getVertexOffset(0);
		return w.jsonInt(int64(map.getVertexIndex(vtx)));
	    }
	    case geo_TBJ_VDB:
		return vdb(pr)->saveVDB(w);
	    case geo_TBJ_VDB_VISUALIZATION:
		return vdb(pr)->saveVisualization(w, map);

	    case geo_TBJ_ENTRIES:
		break;
	}
	return false;
    }
    virtual bool
    loadField(GA_Primitive *pr, int i, UT_JSONParser &p,
	      const GA_LoadMap &map) const
    {
	switch (i)
	{
	    case geo_TBJ_VERTEX:
	    {
		int64 vidx;
		if (!p.parseInt(vidx))
		    return false;
		GA_Offset voff = map.getVertexOffset(GA_Index(vidx));
		// Assign the preallocated vertex, but
		// do not bother updating the topology,
		// which will be done at the end of the
		// load anyway.
		vdb(pr)->assignVertex(voff, false);
		return true;
	    }
	    case geo_TBJ_VDB:
		return vdb(pr)->loadVDB(p);
	    case geo_TBJ_VDB_VISUALIZATION:
		return vdb(pr)->loadVisualization(p, map);

	    case geo_TBJ_ENTRIES:
		break;
	}
	UT_ASSERT(0);
	return false;
    }

    // Implement these methods to be the same as the H12.5 base class version.
    // In H12.1, these methods were pure virtual.
#if 1
    virtual bool
    saveField(const GA_Primitive *pr, int i, UT_JSONValue &val,
	      const GA_SaveMap &map) const
    {
#if (UT_VERSION_INT >= 0x0c0100B6) // 12.1.182 or later
	UT_AutoJSONWriter w(val);
	return saveField(pr, i, *w, map);
#else
	GA_Offset		vtx;
	switch (i)
	{
	    case geo_TBJ_VERTEX:
	        vtx = vdb(pr)->getVertexOffset(0);
	        val.setInt(map.getVertexIndex(vtx));
	        return true;
	    case geo_TBJ_VDB:
	    case geo_TBJ_VDB_VISUALIZATION:
	        return false;

	    case geo_TBJ_ENTRIES:
	        break;
	}
	UT_ASSERT(0);
	return false;
#endif
    }
    // Re-implement the H12.5 base class version, note that this was pure
    // virtual in H12.1.
    virtual bool
    loadField(GA_Primitive *pr, int i, UT_JSONParser &p,
	      const UT_JSONValue &jval, const GA_LoadMap &map) const
    {
#if (UT_VERSION_INT >= 0x0c0100B6) // 12.1.182 or later
	UT_AutoJSONParser parser(jval);
	bool ok = loadField(pr, i, *parser, map);
	p.stealErrors(*parser);
	return ok;
#else
	int64		voff;

	switch (i)
	{
	    case geo_TBJ_VERTEX:
		if (!jval.import(voff))
		    return geo_JVDBError(p, pr, "Unable to import vertex");;
		voff = map.getVertexOffset(voff);
		// Assign the preallocated vertex, but
		// don't bother updating the topology,
		// which will be done at the end of the
		// load anyway.
		vdb(pr)->assignVertex(GA_Offset(voff), false);
		return true;
	    case geo_TBJ_VDB:
		return vdb(pr)->loadVDB(p);
	    case geo_TBJ_VDB_VISUALIZATION:
		return false;

	    case geo_TBJ_ENTRIES:
		break;
	}
	UT_ASSERT(0);
	return false;
#endif
    }
#endif

    virtual bool
    isEqual(int i, const GA_Primitive *p0, const GA_Primitive *p1) const
    {
	switch (i)
	{
	    case geo_TBJ_VERTEX:
		return (p0->getVertexOffset(0) == p1->getVertexOffset(0));
	    case geo_TBJ_VDB:
		return false; // never save these tags as uniform
	    case geo_TBJ_VDB_VISUALIZATION:
		return (vdb(p0)->getVisOptions() == vdb(p1)->getVisOptions());
	    case geo_TBJ_ENTRIES:
		break;
	}
	UT_ASSERT(0);
	return false;
    }

private:
};

} // namespace unnamed


static const GA_PrimitiveJSON *
vdbJSON()
{
    static SYS_AtomicPtr<GA_PrimitiveJSON> theJSON;

    if (!theJSON) {
        GA_PrimitiveJSON* json = new geo_PrimVDBJSON;
        if (NULL != theJSON.compare_swap(NULL, json)) {
            delete json;
            json = NULL;
        }
    }
    return theJSON;
}

const GA_PrimitiveJSON *
GEO_PrimVDB::getJSON() const
{
    return vdbJSON();
}


// This method is called by multiple places internally in Houdini.
static void
geoSetVDBStreamCompression(openvdb::io::Stream& vos, bool backwards_compatible)
{
    // Always enable full compression, since it is fast and compresses level
    // sets and fog volumes well.
    uint32_t compression = openvdb::io::COMPRESS_ACTIVE_MASK;
    // Enable blosc compression unless we want it to be backwards compatible.
    if (vos.hasBloscCompression() && !backwards_compatible) {
        compression |= openvdb::io::COMPRESS_BLOSC;
    }
    vos.setCompression(compression);
}


bool
GEO_PrimVDB::saveVDB(UT_JSONWriter &w) const
{
#if (UT_VERSION_INT >= 0x0c0002bf) // 12.0.703 or later
    bool	ok = true;

    try
    {
	openvdb::GridCPtrVec grids;
	grids.push_back(getConstGridPtr());

	UT_JSONWriter::TiledStream os(w);

	openvdb::io::Stream			vos(os);
	openvdb::MetaMap			meta;

#if (UT_VERSION_INT >= 0x0d00023d) // 13.0.573 or later
    geoSetVDBStreamCompression(vos,
        UT_EnvControl::getInt(ENV_HOUDINI13_VOLUME_COMPATIBILITY));
#else
    geoSetVDBStreamCompression(vos, /*backwards_compatible*/true);
#endif

	// Visual C++ requires a default meta object declared on the stack
	vos.write(grids, meta);
    }
    catch (std::exception &e)
    {
	std::cerr << "Save failure: " << e.what() << "\n";
	ok = false;
    }
    return ok;
#else
    #warning OpenVDB .bgeo I/O is not available prior to 12.0.703
    return false;
#endif
}

bool
GEO_PrimVDB::loadVDB(UT_JSONParser &p)
{
#if (UT_VERSION_INT >= 0x0c0002bf) // 12.0.703 or later
    try
    {
	UT_JSONParser::TiledStream	is(p);

	openvdb::io::Stream		vis(is);

	openvdb::GridPtrVecPtr	grids = vis.getGrids();

	int count = (grids ? grids->size() : 0);
	if (count != 1)
	{
	    UT_String mesg;
	    mesg.sprintf("expected to read 1 grid, got %d grid%s",
		count, count == 1 ? "" : "s");
	    throw std::runtime_error(mesg.nonNullBuffer());
	}

	openvdb::GridBase::Ptr grid = (*grids)[0];
	UT_ASSERT(grid);
	if (grid) setGrid(*grid);
    }
    catch (std::exception &e)
    {
	std::cerr << "Load failure: " << e.what() << "\n";
	return false;
    }
    return true;
#else
    #warning OpenVDB .bgeo I/O is not available prior to 12.0.703
    return false;
#endif
}

namespace // anonymous
{

enum
{
    geo_JVOL_VISMODE,
    geo_JVOL_VISISO,
    geo_JVOL_VISDENSITY,
};
UT_FSATable	theJVolumeViz(
    geo_JVOL_VISMODE,		"mode",
    geo_JVOL_VISISO,		"iso",
    geo_JVOL_VISDENSITY,	"density",
    -1,				NULL
);

} // namespace anonymous

bool
GEO_PrimVDB::saveVisualization(UT_JSONWriter &w, const GA_SaveMap &) const
{
#if (UT_VERSION_INT < 0x0c010072) // earlier than 12.1.114
    #warning OpenVDB visualization options do not persist prior to 12.0.114
    return false;
#else
    bool	ok = true;
    ok = ok && w.jsonBeginMap();

    ok = ok && w.jsonKeyToken(theJVolumeViz.getToken(geo_JVOL_VISMODE));
    ok = ok && w.jsonString(GEOgetVolumeVisToken(myVis.myMode));

    ok = ok && w.jsonKeyToken(theJVolumeViz.getToken(geo_JVOL_VISISO));
    ok = ok && w.jsonReal(myVis.myIso);

    ok = ok && w.jsonKeyToken(theJVolumeViz.getToken(geo_JVOL_VISDENSITY));
    ok = ok && w.jsonReal(myVis.myDensity);

    return ok && w.jsonEndMap();
#endif
}

bool
GEO_PrimVDB::loadVisualization(UT_JSONParser &p, const GA_LoadMap &)
{
#if (UT_VERSION_INT < 0x0c010072) // earlier than 12.1.114
    #warning OpenVDB visualization options do not persist prior to 12.0.114
    return false;
#else
    UT_JSONParser::traverser	it;
    GEO_VolumeVis		mode = myVis.myMode;
    fpreal			iso = myVis.myIso;
    fpreal			density = myVis.myDensity;
    UT_WorkBuffer		key;
    fpreal64			fval;
    bool			foundmap=false, ok = true;

    for (it = p.beginMap(); ok && !it.atEnd(); ++it)
    {
	foundmap = true;
	if (!it.getLowerKey(key))
	{
	    ok = false;
	    break;
	}
	switch (theJVolumeViz.findSymbol(key.buffer()))
	{
	    case geo_JVOL_VISMODE:
		if (ok = p.parseString(key))
		    mode = GEOgetVolumeVisEnum(
				key.buffer(), GEO_VOLUMEVIS_SMOKE);
		break;
	    case geo_JVOL_VISISO:
		if (ok = p.parseReal(fval))
		    iso = fval;
		break;
	    case geo_JVOL_VISDENSITY:
		if (ok = p.parseReal(fval))
		    density = fval;
		break;
	    default:
		p.addWarning("Unexpected key for volume visualization: %s",
			key.buffer());
		ok = p.skipNextObject();
		break;
	}
    }
    if (!foundmap)
    {
	p.addFatal("Expected a JSON map for volume visualization data");
	ok = false;
    }
    if (ok)
	setVisualization(mode, iso, density);
    return ok;
#endif
}

template <typename GridType>
static void
geo_sumPosDensity(const GridType &grid, fpreal64 &sum)
{
    sum = 0;
    for (typename GridType::ValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
        fpreal value = *iter;
        if (value > 0) {
            if (iter.isTileValue()) sum += value * iter.getVoxelCount();
            else sum += value;
        }
    }
}

fpreal
GEO_PrimVDB::calcPositiveDensity() const
{
    fpreal64 density = 0;

    UT_IF_ASSERT(UT_VDBType type = getStorageType();)
    UT_ASSERT(type == UT_VDB_FLOAT || type == UT_VDB_DOUBLE);

    UTvdbCallRealType(getStorageType(), geo_sumPosDensity, getGrid(), density);

    int numvoxel = getGrid().activeVoxelCount();
    if (numvoxel)
	density /= numvoxel;

    UT_Vector3 zero(0, 0, 0);
    density *= calcVolume(zero);

    return density;
}

int
GEO_PrimVDB::getBBox(UT_BoundingBox *bbox) const
{
    if (hasGrid())
    {
	using namespace openvdb;

	CoordBBox vbox;

	const openvdb::GridBase &grid = getGrid();
	// NOTE: We use evalActiveVoxelBoundingBox() so that it matches
	//       getRes() which calls evalActiveVoxelDim().
	if (!grid.baseTree().evalActiveVoxelBoundingBox(vbox))
	{
	    bbox->makeInvalid();
	    return false;
	}

	const math::Transform &xform = grid.transform();

	for (int i = 0; i < 8; i++)
	{
	    math::Vec3d vpos(
		(i&1) ? vbox.min()[0] - 0.5 : vbox.max()[0] + 0.5,
		(i&2) ? vbox.min()[1] - 0.5 : vbox.max()[1] + 0.5,
		(i&4) ? vbox.min()[2] - 0.5 : vbox.max()[2] + 0.5);
	    vpos = xform.indexToWorld(vpos);

	    UT_Vector3 worldpos(vpos.x(), vpos.y(), vpos.z());
	    if (i == 0)
		bbox->initBounds(worldpos);
	    else
		bbox->enlargeBounds(worldpos);
	}
	return true;
    }

    bbox->initBounds(getDetail().getPos3(vertexPoint(0)));

    return true;
}

UT_Vector3
GEO_PrimVDB::baryCenter() const
{
    // Return the center of the index space
    if (!hasGrid())
	return UT_Vector3(0, 0, 0);

    const openvdb::GridBase &grid = getGrid();
    openvdb::CoordBBox bbox = grid.evalActiveVoxelBoundingBox();
    UT_Vector3 pos;
    findexToPos(UTvdbConvert(bbox.getCenter()), pos);
    return pos;
}

bool
GEO_PrimVDB::isDegenerate() const
{
    return false;
}

//
// Methods to handle vertex attributes for the attribute dictionary
//
void
#if (UT_VERSION_INT >= 0x0d000000)
GEO_PrimVDB::copyPrimitive(const GEO_Primitive *psrc)
#else
GEO_PrimVDB::copyPrimitive(const GEO_Primitive *psrc, GEO_Point **ptredirect)
#endif
{
    if (psrc == this) return;

    const GEO_PrimVDB *src = (const GEO_PrimVDB *)psrc;
    const GA_IndexMap &src_points = src->getParent()->getPointMap();

    copyGridFrom(*src); // makes a shallow copy

    // TODO: Well and good to reuse the attribute handle for all our
    //       vertices, but we should do so across primitives as well.
    GA_VertexWrangler vertex_wrangler(*getParent(), *src->getParent());

    GA_Offset v = myVertex;
    GA_Index ptind = src_points.indexFromOffset(src->vertexPoint(0));
#if (UT_VERSION_INT >= 0x0d000000)
    GA_Offset ptoff = getParent()->pointOffset(ptind);
    wireVertex(v, ptoff);
#else
    GEO_Point *ppt = ptredirect[ptind];
    wireVertex(v, ppt ? ppt->getMapOffset() : GA_INVALID_OFFSET);
#endif
    vertex_wrangler.copyAttributeValues(v, src->fastVertexOffset(0));
    myVis = src->myVis;
}

#if (UT_VERSION_INT < 0x0d000000) // Deleted in 13.0
#if (UT_VERSION_INT >= 0x0c050132) // 12.5.306 or later
void
GEO_PrimVDB::copyOffsetPrimitive(const GEO_Primitive *psrc, GA_Index basept)
#else
void
GEO_PrimVDB::copyOffsetPrimitive(const GEO_Primitive *psrc, int basept)
#endif
{
    if (psrc == this) return;

    const GEO_PrimVDB	*src = (const GEO_PrimVDB *)psrc;
    const GA_IndexMap	&points = getParent()->getPointMap();
    const GA_IndexMap	&src_points = src->getParent()->getPointMap();
    GA_Offset		 ppt;

    copyGridFrom(*src); // makes a shallow copy

    // TODO: Well and good to reuse the attribute handle for all our
    //       points/vertices, but we should do so across primitives
    //       as well.
    GA_VertexWrangler		 vertex_wrangler(*getParent(),
						 *src->getParent());

    GA_Offset	v = fastVertexOffset(0);
    ppt = points.offsetFromIndex(
	    src_points.indexFromOffset(src->vertexPoint(0)) + basept);
    wireVertex(v, ppt);
    vertex_wrangler.copyAttributeValues(v, src->fastVertexOffset(0));
    myVis = src->myVis;
}
#endif

static inline
openvdb::math::Vec3d
vdbTranslation(const openvdb::math::Transform &xform)
{
    return xform.baseMap()->getAffineMap()->getMat4().getTranslation();
}

// Replace the grid's translation with the prim's vertex position
void
GEO_PrimVDB::GridAccessor::updateGridTranslates(const GEO_PrimVDB &prim) const
{
    using namespace	openvdb::math;
    const GA_Detail &	geo = prim.getDetail();

    // It is possible our vertex offset is invalid, such as us
    // being a stashed primitive.
    if (!GAisValid(prim.getVertexOffset(0)))
	return;

    GA_Offset		ptoff = prim.vertexPoint(0);
    Vec3d		newpos = UTvdbConvert(geo.getPos3(ptoff));
    Vec3d		oldpos = vdbTranslation(myGrid->transform());
    MapBase::ConstPtr	map = myGrid->transform().baseMap();

    if (isApproxEqual(oldpos, newpos))
	return;

    const_cast<GEO_PrimVDB&>(prim).incrTransformUniqueId();
    Vec3d delta = newpos - oldpos;
    const_cast<GEO_PrimVDB::GridAccessor *>(this)->makeGridUnique();
    myGrid->setTransform(
	    boost::make_shared<Transform>(map->postTranslate(delta)));
}

// Copy the translation from xform and set into our vertex position
void
GEO_PrimVDB::GridAccessor::setVertexPositionAdapter(
	const void* xformPtr,
	GEO_PrimVDB &prim)
{
    // xformPtr is assumed to point to an openvdb::vX_Y_Z::math::Transform,
    // for some version X.Y.Z of OpenVDB that may be newer than the one
    // with which libHoudiniGEO.so was built.  This is safe provided that
    // math::Transform and its member objects are ABI-compatible between
    // the two OpenVDB versions.
    const openvdb::math::Transform& xform =
	*static_cast<const openvdb::math::Transform*>(xformPtr);
    if (myGrid && &myGrid->transform() == &xform)
	return;
    prim.incrTransformUniqueId();
    prim.getDetail().setPos3(
	    prim.vertexPoint(0), UTvdbConvert(vdbTranslation(xform)));
}

void
GEO_PrimVDB::GridAccessor::setTransformAdapter(
	const void* xformPtr,
	GEO_PrimVDB &prim)
{
    if (!myGrid)
	return;
    // xformPtr is assumed to point to an openvdb::vX_Y_Z::math::Transform,
    // for some version X.Y.Z of OpenVDB that may be newer than the one
    // with which libHoudiniGEO.so was built.  This is safe provided that
    // math::Transform and its member objects are ABI-compatible between
    // the two OpenVDB versions.
    const openvdb::math::Transform& xform =
	*static_cast<const openvdb::math::Transform*>(xformPtr);
    setVertexPosition(xform, prim);
    myGrid->setTransform(xform.copy());
}


void
GEO_PrimVDB::GridAccessor::setGridAdapter(
	const void* gridPtr,
	GEO_PrimVDB &prim)
{
    // gridPtr is assumed to point to an openvdb::vX_Y_Z::GridBase, for some
    // version X.Y.Z of OpenVDB that may be newer than the one with which
    // libHoudiniGEO.so was built.  This is safe provided that GridBase and
    // its member objects are ABI-compatible between the two OpenVDB versions.
    const openvdb::GridBase& grid =
	*static_cast<const openvdb::GridBase*>(gridPtr);
    if (myGrid.get() == &grid)
	return;
    setVertexPosition(grid.transform(), prim);
#ifdef OPENVDB_3_ABI_COMPATIBLE
    myGrid = grid.copyGrid(); // always shallow-copy the source grid
#else
    myGrid = openvdb::ConstPtrCast<openvdb::GridBase>(
	grid.copyGrid()); // always shallow-copy the source grid
#endif
    myStorageType = UTvdbGetGridType(*myGrid);
}


GEO_Primitive *
GEO_PrimVDB::copy(int preserve_shared_pts) const
{
    GEO_Primitive *clone = GEO_Primitive::copy(preserve_shared_pts);

    if (clone)
    {
	GEO_PrimVDB*	 vdb = static_cast<GEO_PrimVDB*>(clone);
	GA_Offset	 ppt;

	// Give the clone the same serial number as this primitive.
	vdb->myUniqueId.exchange(this->getUniqueId());

	// Give the clone a shallow copy of this primitive's grid.
	vdb->copyGridFrom(*this);

	// TODO: Well and good to reuse the attribute handle for all our
	//       points/vertices, but we should do so across primitives
	//       as well.
	GA_ElementWranglerCache	 wranglers(*getParent(),
					   GA_PointWrangler::INCLUDE_P);

	int nvtx = getVertexCount();

	if (preserve_shared_pts)
	{
	    UT_SparseArray<GA_Offset *>	addedpoints;
	    GA_Offset			*ppt_ptr;

	    for (int i = 0; i < nvtx; i++)
	    {
		GA_Offset		 src_ppt = vertexPoint(i);
		GA_Offset		 v  = vdb->fastVertexOffset(i);
		GA_Offset		 sv = fastVertexOffset(i);

		if (!(ppt_ptr = addedpoints(src_ppt)))
		{
		    ppt = getParent()->appendPointOffset();
		    wranglers.getPoint().copyAttributeValues(ppt, src_ppt);
		    addedpoints.append(src_ppt, new GA_Offset(ppt));
		}
		else
		    ppt = *ppt_ptr;
		vdb->wireVertex(v, ppt);
		wranglers.getVertex().copyAttributeValues(v, sv);
	    }

	    int dummy_index;
	    for (int i = 0; i < addedpoints.entries(); i++)
		delete (GA_Offset *)addedpoints.getRawEntry(i, dummy_index);
	}
	else
	{
	    for (int i = 0; i < nvtx; i++)
	    {
		GA_Offset	v = vdb->fastVertexOffset(i);
		ppt = getParent()->appendPointOffset();
		vdb->wireVertex(v, ppt);
		wranglers.getPoint().copyAttributeValues(ppt, vertexPoint(i));
		wranglers.getVertex().copyAttributeValues(v, fastVertexOffset(i));
	    }
	}
        vdb->myVis = myVis;
    }
    return clone;
}

void
GEO_PrimVDB::copyUnwiredForMerge(const GA_Primitive *prim_src, const GA_MergeMap &map)
{
    UT_ASSERT(prim_src != this);

    const GEO_PrimVDB* src = static_cast<const GEO_PrimVDB*>(prim_src);

    if (GAisValid(myVertex))
	destroyVertex(myVertex);

    if (map.isIdentityMap(GA_ATTRIB_VERTEX))
    {
    	myVertex = src->myVertex;
    }
    else
    {
	GA_Offset sidx = src->myVertex; // Get source index
	// Map to dest
	myVertex = map.mapDestFromSource(GA_ATTRIB_VERTEX, sidx);
    }

    copyGridFrom(*src); // makes a shallow copy

    myVis = src->myVis;
}

void
GEO_PrimVDB::assignVertex(GA_Offset new_vtx, bool update_topology)
{
    if (myVertex != new_vtx)
    {
	destroyVertex(myVertex);
	myVertex = new_vtx;
	if (update_topology)
	    registerVertex(myVertex);
    }
}

void
GEO_PrimVDB::swapVertexOffsets(const GA_Defragment &defrag)
{
    GA_Offset	v = myVertex;
    if (defrag.hasOffsetChanged(v))
    {
	myVertex = defrag.mapOffset(v);
    }
}

const char *
GEO_PrimVDB::getGridName() const
{
    GA_ROHandleS nameAttr(getParent(), GA_ATTRIB_PRIMITIVE, "name");
    return nameAttr.isValid() ? nameAttr.get(getMapOffset()) : "";
}


#if (UT_VERSION_INT >= 0x0c010048) // 12.1.72 or later

namespace // anonymous
{

#if (UT_VERSION_INT < 0x0c050000) // earlier than 12.5.0
    typedef int	    geo_Size;
#else
    typedef GA_Size geo_Size;
#endif

    // Intrinsic attributes
    enum geo_Intrinsic
    {
	geo_INTRINSIC_BACKGROUND,
	geo_INTRINSIC_VOXELSIZE,
	geo_INTRINSIC_ACTIVEVOXELDIM,
	geo_INTRINSIC_ACTIVEVOXELCOUNT,
	geo_INTRINSIC_TRANSFORM,
	geo_INTRINSIC_VOLUMEVISUALMODE,
	geo_INTRINSIC_VOLUMEVISUALDENSITY,
	geo_INTRINSIC_VOLUMEVISUALISO,

	geo_INTRINSIC_META_GRID_CLASS,
	geo_INTRINSIC_META_GRID_CREATOR,
	geo_INTRINSIC_META_IS_LOCAL_SPACE,
	geo_INTRINSIC_META_SAVE_HALF_FLOAT,
	geo_INTRINSIC_META_VALUE_TYPE,
	geo_INTRINSIC_META_VECTOR_TYPE,

	geo_NUM_INTRINSICS
    };

    const UT_FSATable theMetaNames(
	geo_INTRINSIC_META_GRID_CLASS,	    "vdb_class",
	geo_INTRINSIC_META_GRID_CREATOR,    "vdb_creator",
	geo_INTRINSIC_META_IS_LOCAL_SPACE,  "vdb_is_local_space",
	geo_INTRINSIC_META_SAVE_HALF_FLOAT, "vdb_is_saved_as_half_float",
	geo_INTRINSIC_META_VALUE_TYPE,	    "vdb_value_type",
	geo_INTRINSIC_META_VECTOR_TYPE,	    "vdb_vector_type",
	-1,				    NULL
    );

    geo_Size
    intrinsicBackgroundTupleSize(const GEO_PrimVDB *p)
    {
	return UTvdbGetGridTupleSize(p->getStorageType());
    }
    template <typename GridT> void
    intrinsicBackgroundV(const GridT &grid, fpreal64 *v, GA_Size n)
    {
	typename GridT::ValueType background = grid.background();
	for (GA_Size i = 0; i < n; i++)
	    v[i] = background[i];
    }
    template <typename GridT> void
    intrinsicBackgroundS(const GridT &grid, fpreal64 *v)
    {
	v[0] = (fpreal64)grid.background();
    }
    geo_Size
    intrinsicBackground(const GEO_PrimVDB *p, fpreal64 *v, GA_Size size)
    {
	UT_VDBType  grid_type = p->getStorageType();
	GA_Size	    n = SYSmin(UTvdbGetGridTupleSize(grid_type), size);

	UT_ASSERT(n > 0);
	UTvdbCallScalarType(grid_type, intrinsicBackgroundS, p->getGrid(), v)
	else UTvdbCallVec3Type(grid_type, intrinsicBackgroundV,
			       p->getGrid(), v, n)
	else n = 0;

	return n;
    }

    geo_Size
    intrinsicVoxelSize(const GEO_PrimVDB *prim, fpreal64 *v, GA_Size size)
    {
	openvdb::Vec3d voxel_size = prim->getGrid().voxelSize();
	GA_Size n = SYSmin(3, size);
	for (GA_Size i = 0; i < n; i++)
	    v[i] = voxel_size[i];
	return n;
    }

    geo_Size
    intrinsicActiveVoxelDim(const GEO_PrimVDB *prim, int64 *v, GA_Size size)
    {
	using namespace openvdb;
	Coord   dim = prim->getGrid().evalActiveVoxelDim();
	GA_Size	n = SYSmin(3, size);
	for (GA_Size i = 0; i < n; i++)
	    v[i] = dim[i];
	return n;
    }
    int64
    intrinsicActiveVoxelCount(const GEO_PrimVDB *prim)
    {
	return prim->getGrid().activeVoxelCount();
    }

    geo_Size
    intrinsicTransform(const GEO_PrimVDB *prim, fpreal64 *v, GA_Size size)
    {
	using namespace openvdb;
	const GridBase &	    grid = prim->getGrid();
	const math::Transform &	    xform = grid.transform();
	math::MapBase::ConstPtr	    bmap = xform.baseMap();
	math::AffineMap::Ptr	    amap = bmap->getAffineMap();
	math::Mat4d		    m4 = amap->getMat4();
	const double *		    data = m4.asPointer();

	size = SYSmin(size, 16);
	for (int i = 0; i < size; ++i)
	    v[i] = data[i];
	    return geo_Size(size);
    }
    geo_Size
    intrinsicSetTransform(GEO_PrimVDB *q, const fpreal64 *v, GA_Size size)
    {
	if (size < 16)
	    return 0;
	UT_DMatrix4	m(v[0], v[1], v[2], v[3],
			  v[4], v[5], v[6], v[7],
			  v[8], v[9], v[10], v[11],
			  v[12], v[13], v[14], v[15]);
	q->setTransform4(m);
	return 16;
    }
    const char *
    intrinsicVisualMode(const GEO_PrimVDB *p)
    {
#if (UT_VERSION_INT < 0x0c010072) // earlier than 12.1.114
	return "";
#else
	return GEOgetVolumeVisToken(p->getVisualization());
#endif
    }

    openvdb::Metadata::ConstPtr
    intrinsicGetMeta(const GEO_PrimVDB *p, geo_Intrinsic id)
    {
	using namespace openvdb;
	return p->getGrid()[theMetaNames.getToken(id) + 4];
    }
    void
    intrinsicSetMeta(
	    GEO_PrimVDB *p,
	    geo_Intrinsic id,
	    const openvdb::Metadata &meta)
    {
	using namespace openvdb;

	MetaMap &meta_map = p->getMetadata();
	const char *name = theMetaNames.getToken(id) + 4;
	meta_map.removeMeta(name);
	meta_map.insertMeta(name, meta);
    }

    void
    intrinsicGetMetaString(
	    const GEO_PrimVDB *p,
	    geo_Intrinsic id,
	    UT_String &v)
    {
	using namespace openvdb;
	Metadata::ConstPtr meta = intrinsicGetMeta(p, id);
	if (meta)
	    v = meta->str();
	else
	    v = "";
    }
    void
    intrinsicSetMetaString(
	    GEO_PrimVDB *p,
	    geo_Intrinsic id,
	    const char *v)
    {
	intrinsicSetMeta(p, id, openvdb::StringMetadata(v));
    }

    bool
    intrinsicGetMetaBool(const GEO_PrimVDB *p, geo_Intrinsic id)
    {
	using namespace openvdb;
	Metadata::ConstPtr meta = intrinsicGetMeta(p, id);
	if (meta)
	    return meta->asBool();
	else
	    return false;
    }
    void
    intrinsicSetMetaBool(GEO_PrimVDB *p, geo_Intrinsic id, int64 v)
    {
	intrinsicSetMeta(p, id, openvdb::BoolMetadata(v != 0));
    }

} // namespace anonymous

#define VDB_INTRINSIC_META_STR(CLASS, ID) { \
	struct callbacks { \
	    static geo_Size evalS(const CLASS *o, UT_String &v) \
	    { intrinsicGetMetaString(o, ID, v); return 1; } \
	    static geo_Size evalSA(const CLASS *o, UT_StringArray &v) \
	    { \
		UT_String     str; \
		intrinsicGetMetaString(o, ID, str); \
        v.append(str); \
		return 1; \
	    } \
	    static geo_Size setSS(CLASS *o, const char **v, GA_Size) \
	    { intrinsicSetMetaString(o, ID, v[0]); return 1; } \
	    static geo_Size setSA(CLASS *o, const UT_StringArray &a) \
	    { intrinsicSetMetaString(o, ID, a(0)); return 1; } \
	}; \
	GA_INTRINSIC_DEF_S(ID, theMetaNames.getToken(ID), 1) \
	myEval[ID].myS  = callbacks::evalS; \
	myEval[ID].mySA = callbacks::evalSA; \
	myEval[ID].mySetSS = callbacks::setSS; \
	myEval[ID].mySetSA = callbacks::setSA; \
	myEval[ID].myReadOnly = false; \
    }
#define VDB_INTRINSIC_META_BOOL(CLASS, ID) { \
	struct callbacks { \
	    static geo_Size eval(const CLASS *o, int64 *v, GA_Size) \
	    { v[0] = intrinsicGetMetaBool(o, ID); return 1; } \
	    static geo_Size setFunc(CLASS *o, const int64 *v, GA_Size) \
	    { intrinsicSetMetaBool(o, ID, v[0]); return 1; } \
	}; \
	GA_INTRINSIC_DEF_I(ID, theMetaNames.getToken(ID), 1) \
	myEval[ID].myI = callbacks::eval; \
	myEval[ID].mySetI = callbacks::setFunc; \
	myEval[ID].myReadOnly = false; \
    }

GA_START_INTRINSIC_DEF(GEO_PrimVDB, geo_NUM_INTRINSICS)

    GA_INTRINSIC_VARYING_F(GEO_PrimVDB, geo_INTRINSIC_BACKGROUND,
	    "background", intrinsicBackgroundTupleSize, intrinsicBackground);
    GA_INTRINSIC_TUPLE_F(GEO_PrimVDB, geo_INTRINSIC_VOXELSIZE,
	    "voxelsize", 3, intrinsicVoxelSize);

    GA_INTRINSIC_TUPLE_I(GEO_PrimVDB, geo_INTRINSIC_ACTIVEVOXELDIM,
	    "activevoxeldimensions", 3, intrinsicActiveVoxelDim);
    GA_INTRINSIC_I(GEO_PrimVDB, geo_INTRINSIC_ACTIVEVOXELCOUNT,
	    "activevoxelcount", intrinsicActiveVoxelCount);

    GA_INTRINSIC_TUPLE_F(GEO_PrimVDB, geo_INTRINSIC_TRANSFORM,
	    "transform", 16, intrinsicTransform);
    GA_INTRINSIC_SET_TUPLE_F(GEO_PrimVDB, geo_INTRINSIC_TRANSFORM,
	    intrinsicSetTransform);

    GA_INTRINSIC_S(GEO_PrimVDB, geo_INTRINSIC_VOLUMEVISUALMODE,
	    "volumevisualmode", intrinsicVisualMode)
    GA_INTRINSIC_METHOD_F(GEO_PrimVDB, geo_INTRINSIC_VOLUMEVISUALDENSITY,
	    "volumevisualdensity", getVisDensity)
    GA_INTRINSIC_METHOD_F(GEO_PrimVDB, geo_INTRINSIC_VOLUMEVISUALISO,
	    "volumevisualiso", getVisIso)

    VDB_INTRINSIC_META_STR(GEO_PrimVDB, geo_INTRINSIC_META_GRID_CLASS)
    VDB_INTRINSIC_META_STR(GEO_PrimVDB, geo_INTRINSIC_META_GRID_CREATOR)
    VDB_INTRINSIC_META_BOOL(GEO_PrimVDB, geo_INTRINSIC_META_IS_LOCAL_SPACE)
    VDB_INTRINSIC_META_BOOL(GEO_PrimVDB, geo_INTRINSIC_META_SAVE_HALF_FLOAT)
    VDB_INTRINSIC_META_STR(GEO_PrimVDB, geo_INTRINSIC_META_VALUE_TYPE)
    VDB_INTRINSIC_META_STR(GEO_PrimVDB, geo_INTRINSIC_META_VECTOR_TYPE)

GA_END_INTRINSIC_DEF(GEO_PrimVDB, GEO_Primitive)

/*static*/ bool
GEO_PrimVDB::isIntrinsicMetadata(const char *name)
{
#if (UT_VERSION_INT < 0x0c010072) // earlier than 12.1.114
    return (-1 != theMetaNames.findSymbol(name));
#else
    return theMetaNames.contains(name);
#endif
}

#endif // (UT_VERSION_INT >= 0x0c010048) // 12.1.72 or later

#endif // UT_VERSION_INT < 0x0c050157 // earlier than 12.5.343

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
