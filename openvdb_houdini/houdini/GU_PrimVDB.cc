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
 * NAME:	GU_PrimVDB.C ( GU Library, C++)
 *
 * COMMENTS:	Definitions for utility functions of vdb.
 */

#include <UT/UT_Version.h>
#if (UT_VERSION_INT < 0x0c050157) // earlier than 12.5.343

#include "GU_PrimVDB.h"

#include "GT_GEOPrimCollectVDB.h"
#include <GU/GU_ConvertParms.h>
#include <GU/GU_PrimPoly.h>
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
#include <GU/GU_PrimPolySoup.h>
#endif
#include <GU/GU_PrimVolume.h>
#include <GU/GU_RayIntersect.h>
#include <GU/GU_Surfacer.h>

#include <GEO/GEO_AttributeHandleList.h>
#include <GEO/GEO_Closure.h>
#include <GEO/GEO_WorkVertexBuffer.h>

#include <GA/GA_AttributeFilter.h>
#include <GA/GA_ElementWrangler.h>
#include <GA/GA_GBAttributeMath.h>
#include <GA/GA_PageHandle.h>
#include <GA/GA_PageIterator.h>
#include <GA/GA_SplittableRange.h>

#include <UT/UT_Debug.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Lock.h>
#include <UT/UT_ParallelUtil.h>
#include <UT/UT_ScopedPtr.h>
#include <UT/UT_StopWatch.h>
#include <UT/UT_Version.h>

#include <SYS/SYS_Types.h>

#include <openvdb/tools/VolumeToMesh.h>

#include <boost/function.hpp>
#include <boost/scope_exit.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/utility/enable_if.hpp>
#include <algorithm>


#define TIMING_DEF \
    UT_StopWatch timer; \
    if (verbose) timer.start();
#define TIMING_LOG(msg) \
    if (verbose) { \
	printf(msg ": %f ms\n", 1000*timer.stop()); \
	fflush(stdout); \
	timer.start(); \
    }


#if (UT_VERSION_INT < 0x0c0500F5) // Prior to 12.5.245

// These methods come from the H12.5 version of GU_ConvertParms.h for compiling
// in H12.1.
namespace GU_Convert_H12_5 {

class GU_ConvertMarker
{
public:
    GU_ConvertMarker(const GA_Detail &geo)
	: myGeo(geo)
	, myPrimBegin(primOff())
	, myPtBegin(ptOff())
    {
    }

    GA_Range getPrimitives() const
    {
	return GA_Range(myGeo.getPrimitiveMap(), myPrimBegin, primOff());
    }
    GA_Range getPoints() const
    {
	return GA_Range(myGeo.getPointMap(), myPtBegin, ptOff() + 1);
    }

    GA_Offset	primitiveBegin() const	{ return myPrimBegin; }
    GA_Offset	pointBegin() const	{ return myPtBegin; }

    GA_Size numPrimitives() const	{ return primOff() - myPrimBegin; }
    GA_Size numPoints() const		{ return ptOff() - myPtBegin; }

private:
    GA_Offset	primOff() const
		{ return myGeo.getPrimitiveMap().lastOffset() + 1; }
    GA_Offset	ptOff() const	{ return myGeo.getPointMap().lastOffset() + 1; }

private:
    const GA_Detail &	myGeo;
    GA_Offset		myPrimBegin;
    GA_Offset		myPtBegin;
};

// Implementation which uses un-cached wranglers for H12.1, in H12.5 these
// wranglers are cached across all primitives with GU_ConvertParms itself.
static void
GUconvertCopySingleVertexPrimAttribsAndGroups(
	GU_ConvertParms &parms,
	const GA_Detail &src,
	GA_Offset src_primoff,
	GA_Detail &dst,
	const GA_Range &dst_prims,
	const GA_Range &dst_points)
{
    UT_ScopedPtr<GA_ElementWranglerCache> cache;
#ifdef SESI_OPENVDB
    if (parms.preserveGroups)
	cache.reset(new GA_ElementWranglerCache(dst, src,
				    GA_AttributeFilter::selectGroup()));
    else
#endif
	cache.reset(new GA_ElementWranglerCache(dst, src,
				    GA_PointWrangler::EXCLUDE_P));

    const GA_Primitive &	src_prim = *(src.getPrimitiveList().get(
								src_primoff));
    GA_Offset			src_vtxoff = src_prim.getVertexOffset(0);
    GA_Offset			src_ptoff = src_prim.getPointOffset(0);
    GA_ElementWranglerCache &	wranglers = *cache;
    GA_PrimitiveWrangler &	prim_wrangler = wranglers.getPrimitive();
    GA_VertexWrangler &		vtx_wrangler = wranglers.getVertex();
    GA_PointWrangler &		pt_wrangler = wranglers.getPoint();
    GA_PrimitiveWrangler *	prim_group_wrangler = NULL;
    GA_PointWrangler *		pt_group_wrangler = NULL;

#ifndef SESI_OPENVDB
    bool have_vtx_attribs = true;
    bool have_pt_attribs = true;
#else
    // This is only optimization available in H12.5
    bool have_vtx_attribs = (vtx_wrangler.getNumAttributes() > 0);
    bool have_pt_attribs = (pt_wrangler.getNumAttributes() > 0);

    if (parms.preserveGroups)
    {
	prim_group_wrangler = &parms.getGroupWranglers(dst,&src).getPrimitive();
	if (prim_group_wrangler->getNumAttributes() <= 0)
	    prim_group_wrangler = NULL;
	pt_group_wrangler = &parms.getGroupWranglers(dst,&src).getPoint();
	if (pt_group_wrangler->getNumAttributes() <= 0)
	    pt_group_wrangler = NULL;
    }
#endif

    UT_ASSERT(src_prim.getVertexCount() == 1);
    for (GA_Iterator i(dst_prims); !i.atEnd(); ++i)
    {
	if (prim_group_wrangler)
	    prim_group_wrangler->copyAttributeValues(*i, src_primoff);
	if (have_pt_attribs)
	    prim_wrangler.copyAttributeValues(*i, src_primoff);
	if (have_vtx_attribs)
	{
	    GA_Primitive &dst_prim = *(dst.getPrimitiveList().get(*i));
	    GA_Size nvtx = dst_prim.getVertexCount();
	    for (GA_Size j = 0; j < nvtx; ++j)
	    {
		vtx_wrangler.copyAttributeValues(
			dst_prim.getVertexOffset(j),
			src_vtxoff);
	    }
	}
    }

    for (GA_Iterator i(dst_points); !i.atEnd(); ++i)
    {
	if (pt_group_wrangler)
	    pt_group_wrangler->copyAttributeValues(*i, src_ptoff);
	if (have_pt_attribs)
	    pt_wrangler.copyAttributeValues(*i, src_ptoff);
    }
}

} // namespace GU_Convert_H12_5

using GU_Convert_H12_5::GU_ConvertMarker;
using GU_Convert_H12_5::GUconvertCopySingleVertexPrimAttribsAndGroups;
#endif // Prior to 12.5.245


GA_PrimitiveDefinition *GU_PrimVDB::theDefinition = 0;

GU_PrimVDB*
GU_PrimVDB::build(GU_Detail *gdp, bool append_points)
{
#ifndef SESI_OPENVDB
    // This is only necessary as a stop gap measure until we have the
    // registration code split out properly.
    if (!GU_PrimVDB::theDefinition)
	GU_PrimVDB::registerMyself(&GUgetFactory());

    GU_PrimVDB* primVdb = (GU_PrimVDB *)gdp->appendPrimitive(GU_PrimVDB::theTypeId());

#else

    GU_PrimVDB* primVdb = (GU_PrimVDB *)gdp->appendPrimitive(GEO_PRIMVDB);

#endif

    if (append_points) {

    	GEO_Primitive *prim = primVdb;
    	const int npts = primVdb->getVertexCount();
    	for (int i = 0; i < npts; i++) {
    	    prim->getVertexElement(i).setPointOffset(gdp->appendPointOffset());
    	}
    }
    return primVdb;
}

GU_PrimVDB*
GU_PrimVDB::buildFromGridAdapter(GU_Detail& gdp, void* gridPtr,
    const GEO_PrimVDB* src, const char* name)
{
    // gridPtr is assumed to point to an openvdb::vX_Y_Z::GridBase::Ptr, for
    // some version X.Y.Z of OpenVDB that may be newer than the one with which
    // libHoudiniGEO.so was built.  This is safe provided that GridBase and
    // its member objects are ABI-compatible between the two OpenVDB versions.
    openvdb::GridBase::Ptr grid =
	*static_cast<openvdb::GridBase::Ptr*>(gridPtr);
    if (!grid)
	return NULL;

    GU_PrimVDB* vdb = GU_PrimVDB::build(&gdp);
    if (vdb != NULL) {
        if (src != NULL) {
            // Copy the source primitive's attributes to this primitive,
            // then transfer those attributes to this grid's metadata.
            vdb->copyAttributesAndGroups(*src, /*copyGroups=*/true);
            GU_PrimVDB::createMetadataFromGridAttrs(*grid, *vdb, gdp);

	    // Copy the source's visualization options.
	    GEO_VolumeOptions	visopt = src->getVisOptions();
	    vdb->setVisualization(visopt.myMode, visopt.myIso, visopt.myDensity);
        }

	// Ensure that certain metadata exists (grid name, grid class, etc.).
	if (name != NULL) grid->setName(name);
	grid->removeMeta("value_type");
	grid->insertMeta("value_type", openvdb::StringMetadata(grid->valueType()));
	// For each of the following, force any existing metadata's value
	// to be one of the supported values.
	grid->setGridClass(grid->getGridClass());
	grid->setVectorType(grid->getVectorType());
	grid->setIsInWorldSpace(grid->isInWorldSpace());
	grid->setSaveFloatAsHalf(grid->saveFloatAsHalf());

        // Transfer the grid's metadata to primitive attributes.
        GU_PrimVDB::createGridAttrsFromMetadata(*vdb, *grid, gdp);

	vdb->setGrid(*grid);

	// If we had no source, have to set options to reasonable
	// defaults.
	if (src == NULL)
	{
	    if (grid->getGridClass() == openvdb::GRID_LEVEL_SET) 
	    {
		vdb->setVisualization(GEO_VOLUMEVIS_ISO,
				      vdb->getVisIso(), vdb->getVisDensity());
	    } 
	    else 
	    {
		vdb->setVisualization(GEO_VOLUMEVIS_SMOKE,
				      vdb->getVisIso(), vdb->getVisDensity());
	    }
	}
    }
    return vdb;
}

int64
GU_PrimVDB::getMemoryUsage() const
{
    int64 mem = sizeof(*this);
    mem += GEO_PrimVDB::getBaseMemoryUsage();
    return mem;
}

namespace // anonymous
{

class gu_VolumeMax
{
public:
    gu_VolumeMax(
	    const UT_VoxelArrayReadHandleF &vox,
	    UT_AutoInterrupt &progress)
	: myVox(vox)
	, myProgress(progress)
	, myMax(std::numeric_limits<float>::min())
    {
    }
    gu_VolumeMax(const gu_VolumeMax &other, UT_Split)
	: myVox(other.myVox)
	, myProgress(other.myProgress)
	// NOTE: other.myMax could be half written-to while this
        //       constructor is being called, so don't use its
        //       value here.  Initialize myMax as in the main
        //       constructor.
	, myMax(std::numeric_limits<float>::min())
    {
    }

    void operator()(const UT_BlockedRange<int> &range)
    {
	uint8 bcnt = 0;

	for (int i = range.begin(); i != range.end(); ++i) {
	    float   min_value;
	    float   max_value;

	    myVox->getLinearTile(i)->findMinMax(min_value, max_value);
	    myMax = SYSmax(myMax, max_value);

	    if (!bcnt++ && myProgress.wasInterrupted())
		break;
	}
    }

    void join(const gu_VolumeMax &other)
    {
	myMax = std::max(myMax, other.myMax);
    }

    float findMax()
    {
	UTparallelReduce(UT_BlockedRange<int>(0, myVox->numTiles()), *this);
	return myMax;
    }

private:
    const UT_VoxelArrayReadHandleF &	myVox;
    UT_AutoInterrupt &			myProgress;
    float				myMax;
};

class gu_ConvertToVDB
{
public:
    gu_ConvertToVDB(
	    const UT_VoxelArrayReadHandleF &vox,
	    float background,
	    UT_AutoInterrupt &progress)
	: myVox(vox)
	, myGrid(openvdb::FloatGrid::create(background))
	, myProgress(progress)
    {
    }
    gu_ConvertToVDB(const gu_ConvertToVDB &other, UT_Split)
	: myVox(other.myVox)
	, myGrid(openvdb::FloatGrid::create(other.myGrid->background()))
	, myProgress(other.myProgress)
    {
    }

    openvdb::FloatGrid::Ptr run()
    {
	using namespace openvdb;

	// Grid::merge() is currently broken
#if 1
	UTparallelReduce(UT_BlockedRange<int>(0, myVox->numTiles()), *this);
#else
	(*this)(UT_BlockedRange<int>(0, myVox->numTiles()));
#endif
	// This commented out code tests whether Grid::merge() works for
	// tile values, which currently doesn't as of v0.96.0
#if 0 //def UT_DEBUG
{
    openvdb::FloatGrid::Ptr a = openvdb::FloatGrid::create(/*background*/0.0);
    openvdb::FloatGrid::Ptr b = openvdb::FloatGrid::create(/*background*/0.0);
    a->fill(CoordBBox(Coord(16,16,16), Coord(31,31,31)), /*value*/1.0);
    b->fill(CoordBBox(Coord(0,0,0),    Coord(15,15,15)), /*value*/1.0);
    int a_count_old = a->activeVoxelCount();
    int b_count_old = b->activeVoxelCount();
    a->merge(*b);
    int a_count_new = a->activeVoxelCount();
    int b_count_new = b->activeVoxelCount();
    CoordBBox bbox;
    cerr << "a_count_old=" << a_count_old << ", b_count_old=" << b_count_old << endl;
    cerr << "a_count_new=" << a_count_new << ", b_count_new=" << b_count_new << endl;
    cerr << "bbox=" << a->evalActiveVoxelBoundingBox() << endl;
}
#endif

	// Check if the VDB grid can be made empty
	openvdb::Coord dim = myGrid->evalActiveVoxelDim();
	if (dim[0] == 1 && dim[1] == 1 && dim[2] == 1) {
	    openvdb::Coord ijk = myGrid->evalActiveVoxelBoundingBox().min();
	    float value = myGrid->tree().getValue(ijk);
	    if (openvdb::math::isApproxEqual<float>(value, myGrid->background())) {
		    myGrid->clear();
	    }
	}
	
	return myGrid;
    }

    void operator()(const UT_BlockedRange<int> &range)
    {
	using namespace openvdb;

	FloatGrid &		grid = *myGrid.get();
	const float		background = grid.background();
	const UT_VoxelArrayF &	vox = *myVox;
	uint8			bcnt = 0;

	FloatGrid::Accessor acc = grid.getAccessor();

	for (int i = range.begin(); i != range.end(); ++i) {

	    const UT_VoxelTile<float> &	tile = *vox.getLinearTile(i);
	    Coord			org;
	    Coord			dim;

	    vox.linearTileToXYZ(i, org[0], org[1], org[2]);
	    org[0] *= TILESIZE; org[1] *= TILESIZE; org[2] *= TILESIZE;
	    dim[0] = tile.xres(); dim[1] = tile.yres(); dim[2] = tile.zres();

	    if (tile.isConstant()) {
		CoordBBox   bbox(org, org + dim.offsetBy(-1));
		float	    value = tile(0, 0, 0);

		if (!SYSisEqual(value, background)) {
		    grid.fill(bbox, value);
		}
	    } else {
		openvdb::Coord ijk;
		for (ijk[2] = 0; ijk[2] < dim[2]; ++ijk[2]) {
		    for (ijk[1] = 0; ijk[1] < dim[1]; ++ijk[1]) {
			for (ijk[0] = 0; ijk[0] < dim[0]; ++ijk[0]) {
			    float value = tile(ijk[0], ijk[1], ijk[2]);
			    if (!SYSisEqual(value, background)) {
				Coord pos = ijk.offsetBy(org[0], org[1], org[2]);
				acc.setValue(pos, value);
			    }
			}
		    }
		}
	    }

	    if (!bcnt++ && myProgress.wasInterrupted())
		break;
	}
    }

    void join(const gu_ConvertToVDB &other)
    {
	if (myProgress.wasInterrupted())
	    return;
	UT_IF_ASSERT(int old_count = myGrid->activeVoxelCount();)
	UT_IF_ASSERT(int other_count = other.myGrid->activeVoxelCount();)
	myGrid->merge(*other.myGrid);
	UT_ASSERT(myGrid->activeVoxelCount() == old_count + other_count);
    }

private:
    const UT_VoxelArrayReadHandleF &	myVox;
    openvdb::FloatGrid::Ptr		myGrid;
    UT_AutoInterrupt &			myProgress;

}; // class gu_ConvertToVDB

} // namespace anonymous

GU_PrimVDB *
GU_PrimVDB::buildFromPrimVolume(
	GU_Detail &geo,
	const GEO_PrimVolume &vol,
	const char *name,
	const bool flood_sdf,
	const bool prune,
	const float tolerance)
{
    using namespace openvdb;

    UT_AutoInterrupt		progress("Converting to VDB");
    UT_VoxelArrayReadHandleF	vox = vol.getVoxelHandle();

    float background;
    if (vol.isSDF())
    {
	gu_VolumeMax max_op(vox, progress);
	background = max_op.findMax();
	if (progress.wasInterrupted())
	    return NULL;
    }
    else
    {
	if (vol.getBorder() == GEO_VOLUMEBORDER_CONSTANT)
	    background = vol.getBorderValue();
	else
	    background = 0.0;
    }

    gu_ConvertToVDB converter(vox, background, progress);
    FloatGrid::Ptr grid = converter.run();
    if (progress.wasInterrupted())
	return NULL;

    if (vol.isSDF())
	grid->setGridClass(GridClass(GRID_LEVEL_SET));
    else
	grid->setGridClass(GridClass(GRID_FOG_VOLUME));
    
    if (prune) {
        grid->pruneGrid(tolerance);
    }
    
    if (flood_sdf && vol.isSDF()) {
        // only call signed flood fill on SDFs
        grid->signedFloodFill();
    }

    GU_PrimVDB *prim_vdb = buildFromGrid(geo, grid, NULL, name);
    if (!prim_vdb)
	return NULL;
    int rx, ry, rz;
    vol.getRes(rx, ry, rz);
    prim_vdb->setSpaceTransform(vol.getSpaceTransform(), UT_Vector3R(rx,ry,rz));
    prim_vdb->setVisualization(
		vol.getVisualization(), vol.getVisIso(), vol.getVisDensity());
    return prim_vdb;
}

// Copy the exclusive bbox [start,end) from myVox into acc
static void
guCopyVoxelBBox(
	const UT_VoxelArrayReadHandleF &vox,
	openvdb::FloatGrid::Accessor &acc,
	openvdb::Coord start, openvdb::Coord end)
{
    openvdb::Coord c;
    for (c[0] = start[0] ; c[0] < end[0]; c[0]++) {
	for (c[1] = start[1] ; c[1] < end[1]; c[1]++) {
	    for (c[2] = start[2] ; c[2] < end[2]; c[2]++) {
		float value = vox->getValue(c[0], c[1], c[2]);
		acc.setValueOnly(c, value);
	    }
	}
    }
}

void
GU_PrimVDB::expandBorderFromPrimVolume(const GEO_PrimVolume &vol, int pad)
{
    using namespace openvdb;

    UT_AutoInterrupt		    progress("Add inactive VDB border");
    const UT_VoxelArrayReadHandleF  vox(vol.getVoxelHandle());
    const Coord			    res(vox->getXRes(),
					vox->getYRes(),
					vox->getZRes());
    GridBase &			    base = getGrid();
    FloatGrid &			    grid = UTvdbGridCast<FloatGrid>(base);
    FloatGrid::Accessor		    acc = grid.getAccessor();

    // For simplicity, we overdraw the edges and corners
    for (int axis = 0; axis < 3; axis++) {

	if (progress.wasInterrupted())
	    return;

	openvdb::Coord beg(-pad, -pad, -pad);
	openvdb::Coord end = res.offsetBy(+pad);

	beg[axis] = -pad;
	end[axis] = 0;
	guCopyVoxelBBox(vox, acc, beg, end);

	beg[axis] = res[axis];
	end[axis] = res[axis] + pad;
	guCopyVoxelBBox(vox, acc, beg, end);
    }
}

// The following code is for HDK only
#ifndef SESI_OPENVDB
// Static callback for our factory.
static GA_Primitive*
gu_newPrimVDB(GA_Detail &detail, GA_Offset offset)
{
    return new GU_PrimVDB(static_cast<GU_Detail *>(&detail), offset);
}

static GA_Primitive*
gaPrimitiveMergeConstructor(const GA_MergeMap &map,
                            GA_Detail &dest_detail,
                            GA_Offset dest_offset,
                            const GA_Primitive &src_prim)
{
    return new GU_PrimVDB(map, dest_detail, dest_offset, static_cast<const GU_PrimVDB &>(src_prim));
}

static UT_Lock theInitPrimDefLock;

void
GU_PrimVDB::registerMyself(GA_PrimitiveFactory *factory)
{
    // Ignore double registration
    if (theDefinition) return;

    UT_Lock::Scope lock(theInitPrimDefLock);

    if (theDefinition) return;

#if defined(__ICC)
    // Disable ICC "assignment to static variable" warning,
    // since the assignment to theDefinition is mutex-protected.
    __pragma(warning(disable:1711));
#endif

    theDefinition = factory->registerDefinition("VDB",
        gu_newPrimVDB, GA_FAMILY_NONE);

#if defined(__ICC)
    __pragma(warning(default:1711));
#endif

    if (!theDefinition) {
        if (!factory->lookupDefinition("VDB")) {
            //std::cerr << "WARNING: failed to register GU_PrimVDB\n";
        }
        return;
    }

    theDefinition->setLabel("Sparse Volumes (VDBs)");
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later 
    theDefinition->setHasLocalTransform(true);
#endif
    theDefinition->setMergeConstructor(&gaPrimitiveMergeConstructor);
    registerIntrinsics(*theDefinition);

    // Register the GT tesselation too (now we know what type id we have)
    openvdb_houdini::GT_GEOPrimCollectVDB::registerPrimitive(theDefinition->getId());
}
#endif

GEO_Primitive *
GU_PrimVDB::convertToNewPrim(
	GEO_Detail &dst_geo,
	GU_ConvertParms &parms,
	fpreal adaptivity,
	bool &success) const
{
    GEO_Primitive *	prim = NULL; 

    success = false;
    if (parms.toType == GEO_PrimTypeCompat::GEOPRIMPOLY)
    {
	prim = convertToPoly(dst_geo, parms, adaptivity,
			     /*polysoup*/false, success);
    }
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later 
    else if (parms.toType == GEO_PrimTypeCompat::GEOPRIMPOLYSOUP)
    {
	prim = convertToPoly(dst_geo, parms, adaptivity,
			     /*polysoup*/true, success);
    }
#endif
    else if (parms.toType == GEO_PrimTypeCompat::GEOPRIMVOLUME)
    {
	prim = convertToPrimVolume(dst_geo, parms);
	if (prim)
	    success = true;
    }

    return prim;
}

GEO_Primitive *
GU_PrimVDB::convertNew(GU_ConvertParms &parms)
{
    bool success = false;
    return convertToNewPrim(*getParent(), parms, /*adaptivity*/0, success);
}

static void
guCopyMesh(
	GEO_Detail& detail,
	openvdb::tools::VolumeToMesh& mesher,
	bool buildpolysoup,
	bool verbose)
{
    TIMING_DEF;

    const openvdb::tools::PointList& points = mesher.pointList();
    openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

#if (UT_VERSION_INT < 0x0c050000) // earlier than 12.5.0
    const GA_Offset lastIdx(detail.getPointMap().lastOffset()+1);

    for (size_t n = 0, N = mesher.pointListSize(); n < N; ++n) {
        GA_Offset ptoff = detail.appendPointOffset();
        detail.setPos3(ptoff, points[n].x(), points[n].y(), points[n].z());
    }

    TIMING_LOG("Copy Points");

    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {

        const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        // Copy quads
        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {

            const openvdb::Vec4I& quad = polygons.quad(i);
            GEO_PrimPoly& prim = *GU_PrimPoly::build(
                static_cast<GU_Detail*>(&detail), 4, GU_POLY_CLOSED, 0);

            for (int v = 0; v < 4; ++v) {
                prim(v).setPointOffset(lastIdx + quad[v]);
            }
        }


        // Copy triangles (adaptive mesh)
        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {

            const openvdb::Vec3I& triangle = polygons.triangle(i);
            GEO_PrimPoly& prim = *GU_PrimPoly::build(
                static_cast<GU_Detail*>(&detail), 3, GU_POLY_CLOSED, 0);

            for (int v = 0; v < 3; ++v) {
                prim(v).setPointOffset(lastIdx + triangle[v]);
            }
        }
    }

    TIMING_LOG("Build Polys");

#else // 12.5.0 or later

    // NOTE: Adaptive meshes consist of tringles and quads.

    // Construct the points
    GA_Size npoints = mesher.pointListSize();
    GA_Offset startpt = detail.appendPointBlock(npoints);
    UT_ASSERT_COMPILETIME(sizeof(openvdb::tools::PointList::element_type) == sizeof(UT_Vector3));
    GA_RWHandleV3 pthandle(detail.getP());
    pthandle.setBlock(startpt, npoints, (UT_Vector3 *)points.get());

    TIMING_LOG("Copy Points");

    // Construct the array of polygon point numbers
    // NOTE: For quad meshes, the number of quads is about the number of points,
    //       so the number of vertices is about 4*npoints

    GA_Size nquads = 0, ntris = 0;
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
        nquads += polygons.numQuads();
        ntris += polygons.numTriangles();
    }

    TIMING_LOG("Count Quads and Tris");

    // Don't create anything if nothing to create
    if (!ntris && !nquads)
        return;

    GA_Size nverts = nquads*4 + ntris*3;
    UT_IntArray verts(nverts, nverts);
    GA_Size iquad = 0;
    GA_Size itri = nquads*4;
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        const openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        // Copy quads
        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
            const openvdb::Vec4I& quad = polygons.quad(i);
            verts(iquad++) = quad[0];
            verts(iquad++) = quad[1];
            verts(iquad++) = quad[2];
            verts(iquad++) = quad[3];
        }

        // Copy triangles (adaptive mesh)
        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
            const openvdb::Vec3I& triangle = polygons.triangle(i);
            verts(itri++) = triangle[0];
            verts(itri++) = triangle[1];
            verts(itri++) = triangle[2];
        }
    }

    TIMING_LOG("Get Quad and Tri Verts");

    GEO_PolyCounts sizelist;
    if (nquads)
        sizelist.append(4, nquads);
    if (ntris)
        sizelist.append(3, ntris);
    if (buildpolysoup)
        GU_PrimPolySoup::build(&detail, startpt, npoints, sizelist, verts.array());
    else
        GU_PrimPoly::buildBlock(&detail, startpt, npoints, sizelist, verts.array());

    TIMING_LOG("Build Polys");

#endif
}

namespace {
class gu_VDBNormalsParallel
{
public:
    gu_VDBNormalsParallel(GA_Attribute *p, GA_Attribute *n, const GU_PrimVDB &vdb)
        : myP(p)
        , myN(n)
        , myVDB(vdb)
    {}

    void operator()(const GA_SplittableRange &r) const
    {
        UT_Interrupt *boss = UTgetInterrupt();
        GA_ROPageHandleV3 positions(myP);
        GA_RWPageHandleV3 normals(myN);

        for (GA_PageIterator pit = r.beginPages(); !pit.atEnd(); ++pit)
        {
            if (boss->opInterrupt())
                break;

            const GA_Offset pagefirstoff = pit.getFirstOffsetInPage();
            positions.setPage(pagefirstoff);
            normals.setPage(pagefirstoff);
            GA_Offset start;
            GA_Offset end;
            for (GA_Iterator it = pit.begin(); it.blockAdvance(start, end); )
            {
                myVDB.evalGradients(&normals.value(start), 1,
                                    &positions.value(start), end - start,
                                    /*normalize*/true);
            }
        }
    }
private:
    GA_Attribute *const myP;
    GA_Attribute *const myN;
    const GU_PrimVDB &myVDB;
};
}

GEO_Primitive *
GU_PrimVDB::convertToPoly(
	GEO_Detail &dst_geo,
	GU_ConvertParms &parms,
	fpreal adaptivity,
	bool polysoup,
	bool &success) const
{
    using namespace openvdb;

    UT_AutoInterrupt    progress("Convert VDB to Polygons");
    GU_ConvertMarker	marker(dst_geo);
    bool		verbose = false;

    success = false;

    try
    {
	tools::VolumeToMesh mesher(parms.myOffset, adaptivity);
	UTvdbProcessTypedGridScalar(getStorageType(), getGrid(), mesher);
	if (progress.wasInterrupted())
	    return NULL;
	guCopyMesh(dst_geo, mesher, polysoup, verbose);
	if (progress.wasInterrupted())
	    return NULL;
    }
    catch (std::exception& /*e*/)
    {
        return NULL;
    }
    GA_Range points(marker.getPoints());
    GUconvertCopySingleVertexPrimAttribsAndGroups(
	    parms, *getParent(), getMapOffset(), dst_geo,
	    marker.getPrimitives(), points);
    if (progress.wasInterrupted())
	return NULL;

    // If there was already a point normal attribute, we should compute normals
    // to avoid getting zero default values for the new polygons.
    GA_RWAttributeRef normal_ref = dst_geo.findNormalAttribute(GA_ATTRIB_POINT);
    if (normal_ref.isValid() && marker.numPoints() > 0)
    {
        UTparallelFor(GA_SplittableRange(points),
                      gu_VDBNormalsParallel(dst_geo.getP(), normal_ref.getAttribute(), *this));
	if (progress.wasInterrupted())
	    return NULL;
    }

    // At this point, we have succeeded, marker.numPrimitives() might be 0 if
    // we had an empty VDB.
    success = true;
    if (marker.numPrimitives() <= 0)
	return NULL;

    return dst_geo.getGEOPrimitive(marker.primitiveBegin());
}

/*static*/ void
GU_PrimVDB::convertPrimVolumeToPolySoup(
	GU_Detail &dst_geo,
	const GEO_PrimVolume &src_vol)
{
    using namespace openvdb;
    UT_AutoInterrupt progress("Convert to Polygons");

    GU_PrimVDB &vdb = *buildFromPrimVolume(
			    dst_geo, src_vol, NULL,
			    /*flood*/false, /*prune*/true, /*tol*/0);
    // NOTE: This syntax is for Boost 1.46 used by H12.1. It is different in
    // Boost 1.51 used by H12.5.
    BOOST_SCOPE_EXIT( (&vdb) (&dst_geo) )
    {
	dst_geo.destroyPrimitive(vdb, /*and_points*/true);
    } BOOST_SCOPE_EXIT_END

    if (progress.wasInterrupted())
	return;

    try
    {
	BoolGrid::Ptr mask;
	if (src_vol.getBorder() != GEO_VOLUMEBORDER_CONSTANT)
	{
	    Coord res;
	    src_vol.getRes(res[0], res[1], res[2]);
	    CoordBBox bbox(Coord(0, 0, 0), res.offsetBy(-1)); // inclusive
	    if (bbox.hasVolume())
	    {
		vdb.expandBorderFromPrimVolume(src_vol, 4);
		if (progress.wasInterrupted())
		    return;
		mask = BoolGrid::create(/*background*/false);
		mask->setTransform(vdb.getGrid().transform().copy());
		mask->fill(bbox, /*foreground*/true);
	    }
	}

	tools::VolumeToMesh mesher(src_vol.getVisIso());
	mesher.setSurfaceMask(mask);
	GEOvdbProcessTypedGridScalar(vdb, mesher);
	if (progress.wasInterrupted())
	    return;
	guCopyMesh(dst_geo, mesher, /*polysoup*/true, /*verbose*/false);
	if (progress.wasInterrupted())
	    return;
    }
    catch (std::exception& /*e*/)
    {
    }
}

namespace // anonymous
{

#define SCALAR_RET(T) \
	typename boost::enable_if< boost::is_arithmetic< T >, T >::type

#define NON_SCALAR_RET(T) \
	typename boost::disable_if< boost::is_arithmetic< T >, T >::type

/// Houdini Volume wrapper to abstract multiple volumes with a consistent API.
template <int TUPLE_SIZE>
class VoxelArrayVolume
{
public:
    VoxelArrayVolume(GU_Detail& geo, const openvdb::Coord& dim): mGeo(geo)
    {
	for (int i = 0; i < TUPLE_SIZE; i++) {
	    mVol[i] = (GU_PrimVolume *)GU_PrimVolume::build(&mGeo);
	    mHandle[i] = mVol[i]->getVoxelWriteHandle();
	    mHandle[i]->size(dim[0], dim[1], dim[2]);
	}
    }

    template <class ValueT>
    void
    setValue(const openvdb::Coord& ijk, const ValueT& v,
	     SCALAR_RET(ValueT) *dummy = 0)
    {
	mHandle[0]->setValue(ijk[0], ijk[1], ijk[2], float(v));
    }
    template <class ValueT>
    void
    setValue(const openvdb::Coord& ijk, const ValueT& v,
	     NON_SCALAR_RET(ValueT) *dummy = 0)
    {
	for (int i = 0; i < TUPLE_SIZE; i++)
	    mHandle[i]->setValue(ijk[0], ijk[1], ijk[2], float(v[i]));
    }

    template <class ValueT>
    void
    setVolumeOptions(
	    bool is_sdf, const ValueT& background,
	    GEO_VolumeVis vismode, fpreal iso, fpreal density,
	    SCALAR_RET(ValueT) *dummy = 0)
    {
	if (is_sdf) {
	    mVol[0]->setBorder(GEO_VOLUMEBORDER_SDF, background);
	    mVol[0]->setVisualization(vismode, iso, density);
	} else {
	    mVol[0]->setBorder(GEO_VOLUMEBORDER_CONSTANT, background);
	    mVol[0]->setVisualization(vismode, iso, density);
	}
    }
    template <class ValueT>
    void
    setVolumeOptions(
	    bool is_sdf, const ValueT& background,
	    GEO_VolumeVis vismode, fpreal iso, fpreal density,
	    NON_SCALAR_RET(ValueT) *dummy = 0)
    {
	if (is_sdf) {
	    for (int i = 0; i < TUPLE_SIZE; i++) {
		mVol[i]->setBorder(GEO_VOLUMEBORDER_SDF, background[i]);
		mVol[i]->setVisualization(vismode, iso, density);
	    }
	} else {
	    for (int i = 0; i < TUPLE_SIZE; i++) {
		mVol[i]->setBorder(GEO_VOLUMEBORDER_CONSTANT, background[i]);
		mVol[i]->setVisualization(vismode, iso, density);
	    }
	}
    }

    void setSpaceTransform(const GEO_PrimVolumeXform& s)
    {
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or newer
	for (int i = 0; i < TUPLE_SIZE; i++)
	    mVol[i]->setSpaceTransform(s);
#else
	for (int i = 0; i < TUPLE_SIZE; i++) {
	    GEO_Detail *gdp = mVol[i]->getParent();
	    gdp->setPos3(mVol[i]->getPointOffset(0), s.myCenter);
	    mVol[i]->setTransform(s.myXform);
	    mVol[i]->setTaperX(s.myTaperX);
	    mVol[i]->setTaperY(s.myTaperY);
	}
#endif
    }

private:
    GU_Detail& mGeo;
    GU_PrimVolume *mVol[TUPLE_SIZE];
    UT_VoxelArrayWriteHandleF mHandle[TUPLE_SIZE];
};

/// @brief Converts an OpenVDB grid into one/three Houdini Volume.
/// @note Vector grids are converted into three Houdini Volumes.
template <class VolumeT>
class gu_ConvertFromVDB
{
public:

    gu_ConvertFromVDB(
	    GEO_Detail& dst_geo,
	    const GU_PrimVDB& src_vdb,
	    UT_AutoInterrupt& progress)
	: mDstGeo(static_cast<GU_Detail&>(dst_geo))
	, mSrcVDB(src_vdb)
	, mProgress(progress)
    {
    }

    template<typename GridT>
    void operator()(const GridT &grid)
    {
	using namespace openvdb;

	uint8	bcnt = 0;

        tree::ValueAccessor<const typename GridT::TreeType> acc(grid.tree());
        CoordBBox bbox = grid.evalActiveVoxelBoundingBox();
        Coord dim = grid.evalActiveVoxelDim();
                
	// Creating a Houdini volume with a zero bbox seems to break the transform.
	// (probably related to the bbox derived 'local space')
	Coord res = dim;
	if (res[0] == 0) res[0] = 1;
	if (res[1] == 0) res[1] = 1;
	if (res[2] == 0) res[2] = 1;  

	bool is_sdf = (grid.getGridClass() == GRID_LEVEL_SET);

	VolumeT vol(mDstGeo, res);

	vol.setVolumeOptions(is_sdf, grid.background(),
			     mSrcVDB.getVisualization(), mSrcVDB.getVisIso(),
			     mSrcVDB.getVisDensity());
	vol.setSpaceTransform(mSrcVDB.getSpaceTransform());

        Coord ijk, xyz;
        for (xyz[0] = 0; xyz[0] < dim[0]; ++xyz[0]) {
            ijk[0] = bbox.min()[0] + xyz[0];

            for (xyz[1] = 0; xyz[1] < dim[1]; ++xyz[1]) {
                ijk[1] = bbox.min()[1] + xyz[1];

                for (xyz[2] = 0; xyz[2] < dim[2]; ++xyz[2]) {

                    ijk[2] = bbox.min()[2] + xyz[2];
                    vol.setValue(xyz, acc.getValue(ijk));
                }

            }

	    if (!bcnt++ && mProgress.wasInterrupted())
		return;
        }
    }

private:
    GU_Detail&		mDstGeo;
    const GU_PrimVDB&	mSrcVDB;
    UT_AutoInterrupt&	mProgress;
};

} // namespace anonymous

GEO_Primitive *
GU_PrimVDB::convertToPrimVolume(GEO_Detail &dst_geo, GU_ConvertParms &parms) const
{
    using namespace openvdb;

    UT_AutoInterrupt    progress("Convert VDB to Volume");
    GU_ConvertMarker	marker(dst_geo);

    bool processed = false;
    { // Try to convert scalar grid
	gu_ConvertFromVDB< VoxelArrayVolume<1> >
	    converter(dst_geo, *this, progress);
	processed = GEOvdbProcessTypedGridScalar(*this, converter);
    }
    if (!processed) {  // Try to convert vector grid
	gu_ConvertFromVDB< VoxelArrayVolume<3> >
	    converter(dst_geo, *this, progress);
	processed = GEOvdbProcessTypedGridVec3(*this, converter);
    }

    if (!processed || marker.numPrimitives() <= 0 || progress.wasInterrupted())
	return NULL;

    // Copy attributes from source to dest primitives
    GUconvertCopySingleVertexPrimAttribsAndGroups(
	    parms, *getParent(), getMapOffset(),
	    dst_geo, marker.getPrimitives(), marker.getPoints());

    // Handle the name attribute if needed
    if (marker.numPrimitives() > 1)
    {
	GA_ROHandleS src_name(getParent(), GA_ATTRIB_PRIMITIVE, "name");
	GA_RWHandleS dst_name(&dst_geo, GA_ATTRIB_PRIMITIVE, "name");

	if (src_name.isValid() && dst_name.isValid())
	{
	    UT_String name(src_name.get(getMapOffset()));
	    if (name.isstring())
	    {
		UT_String   full_name(name);
		int	    last = name.length() + 1;
		const char  component[] = { 'x', 'y', 'z', 'w' };

		full_name += ".x";
		for (int i = 0; i < marker.numPrimitives(); i++)
		{
		    if (i < 4)
			full_name(last) = component[i];
		    else
			full_name.sprintf("%s%d", (const char *)name, i);
		    dst_name.set(marker.primitiveBegin() + GA_Offset(i),
				 full_name);
		}
	    }
	}
    }

    return dst_geo.getGEOPrimitive(marker.primitiveBegin());
}

GEO_Primitive *
GU_PrimVDB::convert(GU_ConvertParms &parms, GA_PointGroup *usedpts)
{
    bool	    success = false;
    GEO_Primitive * prim;

    prim = convertToNewPrim(*getParent(), parms, /*adaptivity*/0, success);
    if (success)
    {
	if (usedpts)
	    addPointRefToGroup(*usedpts);

	GA_PrimitiveGroup *group = parms.getDeletePrimitives();
	if (group)
	    group->add(this);
	else
	    getParent()->deletePrimitive(*this, !usedpts);
    }
    return prim;
}

/*static*/ void
GU_PrimVDB::convertVolumesToVDBs(
	GU_Detail &dst_geo,
	const GU_Detail &src_geo,
	GU_ConvertParms &parms,
	bool flood_sdf,
	bool prune,
	fpreal tolerance,
	bool keep_original)
{
    UT_AutoInterrupt progress("Convert");

    GEO_Primitive *prim;
    GEO_Primitive *next;
    GA_FOR_SAFE_GROUP_PRIMITIVES(&src_geo, parms.primGroup, prim, next)
    {
        if (progress.wasInterrupted())
	    break;
	if (prim->getTypeId() != GEO_PRIMVOLUME)
	    continue;

	GEO_PrimVolume *vol = UTverify_cast<GEO_PrimVolume*>(prim);
	GA_Offset voloff = vol->getMapOffset();
	GU_ConvertMarker marker(dst_geo);

	GU_PrimVDB *new_prim;
	new_prim = GU_PrimVDB::buildFromPrimVolume(
			dst_geo, *vol, NULL, flood_sdf, prune, tolerance);
	if (!new_prim || progress.wasInterrupted())
	    break;

	GUconvertCopySingleVertexPrimAttribsAndGroups(
		parms, src_geo, voloff, dst_geo,
		marker.getPrimitives(), marker.getPoints());

	if (!keep_original && (&dst_geo == &src_geo))
	    dst_geo.deletePrimitive(*vol, /*and points*/true);
    }
}

/*static*/ void
GU_PrimVDB::convertVDBs(
	GU_Detail &dst_geo,
	const GU_Detail &src_geo,
	GU_ConvertParms &parms,
	fpreal adaptivity,
	bool keep_original)
{
    UT_AutoInterrupt	progress("Convert");

    GEO_Primitive *prim;
    GEO_Primitive *next;
    GA_FOR_SAFE_GROUP_PRIMITIVES(&src_geo, parms.primGroup, prim, next)
    {
        if (progress.wasInterrupted())
	    break;

	GU_PrimVDB *vdb = dynamic_cast<GU_PrimVDB*>(
				static_cast<GU_Primitive*>(prim->castTo()));
	if (vdb == NULL)
	    continue;

	bool success = false;
	(void) vdb->convertToNewPrim(dst_geo, parms, adaptivity, success);
	if (success && !keep_original && (&dst_geo == &src_geo))
	    dst_geo.deletePrimitive(*vdb, /*and points*/true);
    }
}

void
GU_PrimVDB::normal(NormalComp& /*output*/) const
{
    // No need here.
}

void *
GU_PrimVDB::castTo() const
{
    return (GU_Primitive *)this;
}

const GEO_Primitive *
GU_PrimVDB::castToGeo(void) const
{
    return this;
}

GU_RayIntersect *
GU_PrimVDB::createRayCache(int &callermustdelete)
{
    GU_Detail		*gdp	= (GU_Detail *)getParent();
    GU_RayIntersect	*intersect;

    callermustdelete = 0;
    if (gdp->cacheable())
	buildRayCache();

    intersect = getRayCache();
    if (!intersect)
    {
	intersect = new GU_RayIntersect(gdp, this);
	callermustdelete = 1;
    }

    return intersect;
}

int
GU_PrimVDB::intersectRay(const UT_Vector3 &org, const UT_Vector3 &dir,
		float tmax, float , float *distance,
		UT_Vector3 *pos, UT_Vector3 *nml,
		int, float *, float *, int) const
{
    int			result;
    float		dist;
    UT_BoundingBox	bbox;

    // TODO: Build ray cache and intsrect properly.
    getBBox(&bbox);
    result =  bbox.intersectRay(org, dir, tmax, &dist, nml);
    if (result)
    {
	if (distance) *distance = dist;
	if (pos) *pos = org + dist * dir;
    }
    return result;
}


////////////////////////////////////////


namespace {

using namespace openvdb;

typedef boost::function<void (GEO_Detail&, const GEO_PrimVDB&,
    const char* /*name*/, const openvdb::Metadata&)> AttrCreator;
typedef std::map<std::string, AttrCreator> MetaToAttrMap;


const MetaToAttrMap* sMetaToAttrMap = NULL;
UT_Lock sLock;


// For OpenVDB metadata of specific types, create primitive attributes
// by calling GEO_Detail::addTuple() with appropriate arguments.
template<typename MetadataT> void addPrimAttr(GEO_Detail&, const char* /*name*/) {}

#define ADD_PRIM_ATTR(MetaT, StoreT, tupleSize) \
    template<> void addPrimAttr<MetaT>(GEO_Detail& gdp, const char* name) { \
        gdp.addTuple(StoreT, GA_ATTRIB_PRIMITIVE, name, tupleSize); \
    }

ADD_PRIM_ATTR(BoolMetadata,   GA_STORE_INT8,   1)
ADD_PRIM_ATTR(FloatMetadata,  GA_STORE_REAL32, 1)
ADD_PRIM_ATTR(DoubleMetadata, GA_STORE_REAL64, 1)
ADD_PRIM_ATTR(Int32Metadata,  GA_STORE_INT32,  1)
ADD_PRIM_ATTR(Int64Metadata,  GA_STORE_INT64,  1)
ADD_PRIM_ATTR(StringMetadata, GA_STORE_STRING, 1)
ADD_PRIM_ATTR(Vec2IMetadata,  GA_STORE_INT32,  2)
ADD_PRIM_ATTR(Vec2SMetadata,  GA_STORE_REAL32, 2)
ADD_PRIM_ATTR(Vec2DMetadata,  GA_STORE_REAL64, 2)
ADD_PRIM_ATTR(Vec3IMetadata,  GA_STORE_INT32,  3)
ADD_PRIM_ATTR(Vec3SMetadata,  GA_STORE_REAL32, 3)
ADD_PRIM_ATTR(Vec3DMetadata,  GA_STORE_REAL64, 3)

#undef ADD_PRIM_ATTR


// For OpenVDB metadata of specific types, set the value of
// a primitive attribute of a corresponding type.
template<typename MetadataT> void setPrimAttr(GEO_AttributeHandle&, const MetadataT&) {}

#define SET_PRIM_ATTR(MetaT) \
    template<> void setPrimAttr<MetaT>(GEO_AttributeHandle& attr, const MetaT& meta)

SET_PRIM_ATTR(BoolMetadata) { attr.setI(meta.value()); }
SET_PRIM_ATTR(FloatMetadata) { attr.setF(meta.value()); }
SET_PRIM_ATTR(DoubleMetadata) { attr.setF(meta.value()); }
SET_PRIM_ATTR(Int32Metadata) { attr.setI(meta.value()); }
SET_PRIM_ATTR(Int64Metadata) { attr.setI(meta.value()); } ///< @todo can we set a 64-bit value?
SET_PRIM_ATTR(StringMetadata) { attr.setString(UT_String(meta.value())); }
SET_PRIM_ATTR(Vec2IMetadata) {
    const Vec2i& v = meta.value(); attr.setI(v[0], 0); attr.setI(v[1], 1);
}
SET_PRIM_ATTR(Vec2SMetadata) {
    const Vec2s& v = meta.value(); attr.setF(v[0], 0); attr.setF(v[1], 1);
}
SET_PRIM_ATTR(Vec2DMetadata) {
    const Vec2d& v = meta.value(); attr.setF(v[0], 0); attr.setF(v[1], 1);
}
SET_PRIM_ATTR(Vec3IMetadata) {
    const Vec3i& v = meta.value(); attr.setI(v[0], 0); attr.setI(v[1], 1); attr.setI(v[2], 2);
}
SET_PRIM_ATTR(Vec3SMetadata) {
    const Vec3s& v = meta.value(); attr.setF(v[0], 0); attr.setF(v[1], 1); attr.setF(v[2], 2);
}
SET_PRIM_ATTR(Vec3DMetadata) {
    const Vec3d& v = meta.value(); attr.setF(v[0], 0); attr.setF(v[1], 1); attr.setF(v[2], 2);
}

#undef SET_PRIM_ATTR


template<typename MetadataT>
void
addAndSetPrimAttr(GEO_Detail& gdp, const GEO_PrimVDB& prim,
    const char* name, const Metadata& meta)
{
    /// @todo If there is an existing attribute with the given name but
    /// a different type, this will replace the old attribute with a new one.
    /// See GA_ReuseStrategy for alternative behaviors.
    addPrimAttr<MetadataT>(gdp, name);
    GEO_AttributeHandle attr = gdp.getPrimAttribute(name);
    if (attr.isAttributeValid()) {
        attr.setElement(&prim);
        setPrimAttr<MetadataT>(attr, static_cast<const MetadataT&>(meta));
    }
}


// Construct a mapping from OpenVDB metadata types to functions
// that create primitive attributes of corresponding types.
void
initMetaToAttrMap()
{
    using namespace openvdb;

    if (sMetaToAttrMap != NULL) return;

    UT_Lock::Scope lock(sLock);
    if (sMetaToAttrMap != NULL) return;

    MetaToAttrMap* newMap = new MetaToAttrMap;
    (*newMap)[BoolMetadata::staticTypeName()] =   addAndSetPrimAttr<BoolMetadata>;
    (*newMap)[FloatMetadata::staticTypeName()] =  addAndSetPrimAttr<FloatMetadata>;
    (*newMap)[DoubleMetadata::staticTypeName()] = addAndSetPrimAttr<DoubleMetadata>;
    (*newMap)[Int32Metadata::staticTypeName()] =  addAndSetPrimAttr<Int32Metadata>;
    (*newMap)[Int64Metadata::staticTypeName()] =  addAndSetPrimAttr<Int64Metadata>;
    (*newMap)[StringMetadata::staticTypeName()] = addAndSetPrimAttr<StringMetadata>;
    (*newMap)[Vec2IMetadata::staticTypeName()] =  addAndSetPrimAttr<Vec2IMetadata>;
    (*newMap)[Vec2SMetadata::staticTypeName()] =  addAndSetPrimAttr<Vec2SMetadata>;
    (*newMap)[Vec2DMetadata::staticTypeName()] =  addAndSetPrimAttr<Vec2DMetadata>;
    (*newMap)[Vec3IMetadata::staticTypeName()] =  addAndSetPrimAttr<Vec3IMetadata>;
    (*newMap)[Vec3SMetadata::staticTypeName()] =  addAndSetPrimAttr<Vec3SMetadata>;
    (*newMap)[Vec3DMetadata::staticTypeName()] =  addAndSetPrimAttr<Vec3DMetadata>;

#if defined(__ICC)
    // Disable ICC "assignment to static variable" warning, since
    // the assignment to sMetaToAttrMap is mutex-protected.
    __pragma(warning(disable:1711));
#endif

    sMetaToAttrMap = newMap;

#if defined(__ICC)
    __pragma(warning(default:1711));
#endif
}

} // unnamed namespace


void
GU_PrimVDB::syncAttrsFromMetadata()
{
    if (GEO_Detail* detail = this->getParent()) {
        createGridAttrsFromMetadata(*this, this->getConstGrid(), *detail);
    }
}


void
GU_PrimVDB::createGridAttrsFromMetadataAdapter(
	const GEO_PrimVDB& prim,
	const void* gridPtr,
	GEO_Detail& aGdp)
{
    // gridPtr is assumed to point to an openvdb::vX_Y_Z::GridBase, for some
    // version X.Y.Z of OpenVDB that may be newer than the one with which
    // libHoudiniGEO.so was built.  This is safe provided that GridBase and
    // its member objects are ABI-compatible between the two OpenVDB versions.
    const openvdb::GridBase& grid =
	*static_cast<const openvdb::GridBase*>(gridPtr);

    initMetaToAttrMap();

    for (openvdb::MetaMap::ConstMetaIterator metaIt = grid.beginMeta(),
        metaEnd = grid.endMeta(); metaIt != metaEnd; ++metaIt)
    {
        if (openvdb::Metadata::Ptr meta = metaIt->second) {
            std::string name = metaIt->first;

            // Prefix attribute names (except for the "name" attribute)
            // with "vdb_" to avoid conflicts with existing attributes.
            UT_String str(name);
            str.toLower();
            str.forceValidVariableName();
            if (str != "name" && !str.startsWith("vdb_")) {
                name = "vdb_" + name;
            }
            if (isIntrinsicMetadata(name.c_str()))
                continue;

            // If this grid's name is empty and a "name" attribute
            // doesn't already exist, don't create one.
            if (str == "name"
                && meta->typeName() == openvdb::StringMetadata::staticTypeName()
                && meta->str().empty())
            {
                GEO_AttributeHandle attr = aGdp.getPrimAttribute(name.c_str());
                if (!attr.isAttributeValid())
                    continue;
            }

            MetaToAttrMap::const_iterator creatorIt =
                sMetaToAttrMap->find(meta->typeName());
            if (creatorIt != sMetaToAttrMap->end()) {
                creatorIt->second(aGdp, prim, name.c_str(), *meta);
            } else {
                /// @todo Add warning:
                // std::string("discarded metadata \"") + name
                //    + "\" of unsupported type " + meta->typeName()
            }
        }
    }
}


void
GU_PrimVDB::createMetadataFromGridAttrsAdapter(
	void* gridPtr,
	const GEO_PrimVDB& prim,
	const GEO_Detail& aGdp)
{
    using namespace openvdb;

    // gridPtr is assumed to point to an openvdb::vX_Y_Z::GridBase, for some
    // version X.Y.Z of OpenVDB that may be newer than the one with which
    // libHoudiniGEO.so was built.  This is safe provided that GridBase and
    // its member objects are ABI-compatible between the two OpenVDB versions.
    openvdb::GridBase& grid = *static_cast<openvdb::GridBase*>(gridPtr);

    const GA_AttributeSet& attrs = aGdp.getAttributes();
    for (GA_AttributeDict::iterator it = attrs.begin(GA_ATTRIB_PRIMITIVE); !it.atEnd(); ++it) {
        if (!it.name()) continue;

        GEO_AttributeHandle attr = aGdp.getPrimAttribute(it.name());
        if (!attr.isAttributeValid()) continue;

        std::string name = it.name();
        {
            // Strip off the "vdb_" prefix for export.
            UT_String str = it.name();
            str.toLower();
            if (str.startsWith("vdb_")) name = name.substr(4);
        }

        attr.setElement(&prim);

        const int entries = it.attrib()->getTupleSize();

        switch (it.attrib()->getStorageClass()) { ///< @todo can we get the specific GA_Storage?

        case GA_STORECLASS_INT:
            switch (entries) {
            case 1:
                grid.removeMeta(name);
                if (name.substr(0, 3) == "is_") {
                    // Scalar integer attributes whose names begin with "is_"
                    // are mapped to boolean metadata.
                    grid.insertMeta(name, BoolMetadata(attr.getI()));
                } else {
                    grid.insertMeta(name, Int32Metadata(attr.getI()));
                }
                break;
            case 2:
                grid.removeMeta(name);
                grid.insertMeta(name,
                    Vec2IMetadata(Vec2i(attr.getI(0), attr.getI(1))));
                break;
            case 3:
                grid.removeMeta(name);
                grid.insertMeta(name,
                    Vec3IMetadata(Vec3i(attr.getI(0), attr.getI(1), attr.getI(2))));
                break;
            default:
                {
                    /// @todo Add warning:
                    //std::ostringstream ostr;
                    //ostr << "Skipped int[" << entries << "] metadata attribute \""
                    //    << it.name() << "\" (int tuples of size > 3 are not supported)";
                }
                break;
            }
            break;

        case GA_STORECLASS_FLOAT:
            switch (entries) {
            case 1:
                grid.removeMeta(name);
                grid.insertMeta(name, DoubleMetadata(attr.getF()));
                break;
            case 2:
                grid.removeMeta(name);
                grid.insertMeta(name,
                    Vec2DMetadata(Vec2d(attr.getF(0), attr.getF(1))));
                break;
            case 3:
                grid.removeMeta(name);
                grid.insertMeta(name,
                    Vec3DMetadata(Vec3d(attr.getF(0), attr.getF(1), attr.getF(2))));
                break;
            default:
                {
                    /// @todo Add warning:
                    //std::ostringstream ostr;
                    //ostr << "Skipped float[" << entries << "] metadata attribute \""
                    //    << it.name() << "\" (float tuples of size > 3 are not supported)";
                }
                break;
            }
            break;

        case GA_STORECLASS_STRING:
            if (entries == 1) {
                UT_String s;
                attr.getString(s);
                grid.removeMeta(name);
                grid.insertMeta(name, StringMetadata(s.toStdString()));
            } else {
                /// @todo Add warning:
                //std::ostringstream ostr;
                //ostr << "Skipped string[" << entries << "] metadata attribute \""
                //    << it.name() << "\" (string tuples are not supported)";
            }
            break;

        case GA_STORECLASS_INVALID: break;
        case GA_STORECLASS_OTHER: break;
        }
    }
}


////////////////////////////////////////

// Following code is for HDK only
#ifndef SESI_OPENVDB
// This is the usual DSO hook.
extern "C" {
void
newGeometryPrim(GA_PrimitiveFactory *factory)
{
    GU_PrimVDB::registerMyself(factory);
}

} // extern "C"
#endif

#endif // UT_VERSION_INT < 0x0c050157 // earlier than 12.5.343

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
