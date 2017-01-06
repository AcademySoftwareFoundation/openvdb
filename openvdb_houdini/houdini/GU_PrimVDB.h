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
 * NAME:	GU_PrimVDB.h ( GU Library, C++)
 *
 * COMMENTS: Custom VDB primitive.
 */

#include <UT/UT_Version.h>
#if !defined(SESI_OPENVDB) && (UT_VERSION_INT >= 0x0c050157) // 12.5.343 or later

#include <GU/GU_PrimVDB.h>

namespace openvdb_houdini {
using ::GU_PrimVDB;
}

#else // earlier than 12.5.343

#ifndef __HDK_GU_PrimVDB__
#define __HDK_GU_PrimVDB__

//#include "GU_API.h"
#include <GA/GA_PrimitiveDefinition.h>
#include "GEO_PrimVDB.h"
#include <GU/GU_Detail.h>
#include <GU/GU_Prim.h>
#include <UT/UT_Matrix4.h>
#include <UT/UT_VoxelArray.h>
#include <openvdb/Platform.h>
#include <stddef.h>


class GA_Attribute;
class GEO_PrimVolume;
class UT_MemoryCounter;


class OPENVDB_HOUDINI_API GU_PrimVDB : public GEO_PrimVDB, public GU_Primitive
{
protected:
    /// NOTE: Primitives should not be deleted directly.  They are managed
    ///       by the GA_PrimitiveList and the stash.
    virtual ~GU_PrimVDB() {}

public:
    /// NOTE: This constructor should only be called via GU_PrimitiveFactory.
    GU_PrimVDB(GU_Detail *gdp, GA_Offset offset=GA_INVALID_OFFSET)
        : GEO_PrimVDB(gdp, offset)
        , GU_Primitive()
    {}

    /// NOTE: This constructor should only be called via GU_PrimitiveFactory.
    GU_PrimVDB(const GA_MergeMap &map, GA_Detail &detail,
               GA_Offset offset, const GU_PrimVDB &src_prim)
        : GEO_PrimVDB(map, detail, offset, src_prim)
        , GU_Primitive()
    {}

    /// Report approximate memory usage.
    virtual int64 getMemoryUsage() const;

#ifndef SESI_OPENVDB
    /// Allows you to find out what this primitive type was named.
    static GA_PrimitiveTypeId	 theTypeId() { return theDefinition->getId(); }

    /// Must be invoked during the factory callback to add us to the
    /// list of primitives
    static void		registerMyself(GA_PrimitiveFactory *factory);
#endif

    virtual const GA_PrimitiveDefinition &getTypeDef() const
    {
        UT_ASSERT(theDefinition);
        return *theDefinition;
    }

    // Conversion Methods

    virtual GEO_Primitive	*convert(GU_ConvertParms &parms,
					 GA_PointGroup *usedpts = 0);
    virtual GEO_Primitive	*convertNew(GU_ConvertParms &parms);

    /// Convert all GEO_PrimVolume primitives in geometry to
    /// GEO_PrimVDB, preserving prim/vertex/point attributes (and prim/point
    /// groups if requested).
    static void			convertVolumesToVDBs(
					GU_Detail &dst_geo,
					const GU_Detail &src_geo,
					GU_ConvertParms &parms,
					bool flood_sdf,
					bool prune,
					fpreal tolerance,
					bool keep_original);

    /// Convert all GEO_PrimVDB primitives in geometry to parms.toType,
    /// preserving prim/vertex/point attributes (and prim/point groups if
    /// requested).
    /// @{
    static void			convertVDBs(
					GU_Detail &dst_geo,
					const GU_Detail &src_geo,
					GU_ConvertParms &parms,
					fpreal adaptivity,
					bool keep_original);
    static void			convertVDBs(
					GU_Detail &dst_geo,
					const GU_Detail &src_geo,
					GU_ConvertParms &parms,
					fpreal adaptivity,
					bool keep_original,
					bool split_disjoint_volumes);
    /// @}

#if (UT_VERSION_INT < 0x0d050000) // Earlier than 13.5
    virtual void		*castTo (void) const;
    virtual const GEO_Primitive	*castToGeo(void) const;
#endif

    // NOTE:  For static member functions please call in the following
    //        manner.  <ptrvalue> = GU_PrimVDB::<functname>
    //        i.e.        partptr = GU_PrimVDB::build(params...);

    // Optional Build Method

    static GU_PrimVDB *	build(GU_Detail *gdp, bool append_points = true);

    /// Store a VDB grid in a new VDB primitive and add the primitive
    /// to a geometry detail.
    /// @param gdp   the detail to which to add the new primitive
    /// @param grid  a grid to be associated with the new primitive
    /// @param src   if non-null, copy attributes and groups from this primitive
    /// @param name  if non-null, set the new primitive's @c name attribute to
    ///     this string; otherwise, if @a src is non-null, use its name
    static GU_PrimVDB* buildFromGrid(GU_Detail& gdp, openvdb::GridBase::Ptr grid,
	const GEO_PrimVDB* src = NULL, const char* name = NULL)
    {
	return GU_PrimVDB::buildFromGridAdapter(gdp, &grid, src, name);
    }

    /// Create new VDB primitive from the given native volume primitive
    static GU_PrimVDB *	buildFromPrimVolume(
			    GU_Detail &geo,
			    const GEO_PrimVolume &vol,
			    const char *name,
			    const bool flood_sdf = false,
			    const bool prune = false,
			    const float tolerance = 0.0);

    /// A fast method for converting a primitive volume to a polysoup via VDB
    /// into the given gdp. It will _not_ copy attributes because this is a
    /// special case used for display purposes only.
    static void		convertPrimVolumeToPolySoup(
			    GU_Detail &dst_geo,
			    const GEO_PrimVolume &src_vol);

    virtual void	normal(NormalComp &output) const;

#if (UT_VERSION_INT < 0x0d050000) // Earlier than 13.5
    virtual int		intersectRay(const UT_Vector3 &o, const UT_Vector3 &d,
				float tmax = 1E17F, float tol = 1E-12F,
				float *distance = 0, UT_Vector3 *pos = 0,
				UT_Vector3 *nml = 0, int accurate = 0,
				float *u = 0, float *v = 0,
				int ignoretrim = 1) const;
#endif

    // callermustdelete is true if the returned cache is to be deleted by
    // the caller.
#if (UT_VERSION_INT < 0x0d050000) // Earlier than 13.5

#if (UT_VERSION_INT >= 0x0d000000) // 13.0 or later
    SYS_DEPRECATED_HDK(13.0)
#endif
    virtual GU_RayIntersect	*createRayCache(int &callermustdelete);
#endif

    /// @brief Transfer any metadata associated with this primitive's
    /// VDB grid to primitive attributes.
    void syncAttrsFromMetadata();

    /// @brief Transfer any metadata associated with a VDB grid
    /// to primitive attributes on a VDB primitive.
    /// @param prim  the primitive to be populated with attributes
    /// @param grid  the grid whose metadata should be transferred
    /// @param gdp   the detail to which to transfer attributes
    static void createGridAttrsFromMetadata(
        const GEO_PrimVDB& prim,
        const openvdb::GridBase& grid,
        GEO_Detail& gdp)
    {
        GU_PrimVDB::createGridAttrsFromMetadataAdapter(prim, &grid, gdp);
    }

    /// @brief Transfer any metadata associated with the given MetaMap
    /// to attributes on the given element specified by owner.
    /// @param owner    the type of element
    /// @param element  the offset of the element
    /// @param meta_map the metadata that should be transferred
    /// @param gdp      the detail to which to transfer attributes
    static void createAttrsFromMetadata(
        GA_AttributeOwner owner,
        GA_Offset element,
        const openvdb::MetaMap& meta_map,
        GEO_Detail& gdp)
    {
        GU_PrimVDB::createAttrsFromMetadataAdapter(owner, element, &meta_map, gdp);
    }

    /// @brief Transfer a VDB primitive's attributes to a VDB grid as metadata.
    /// @param grid  the grid to be populated with metadata
    /// @param prim  the primitive whose attributes should be transferred
    /// @param gdp   the detail from which to retrieve primitive attributes
    static void createMetadataFromGridAttrs(
        openvdb::GridBase& grid,
        const GEO_PrimVDB& prim,
        const GEO_Detail& gdp)
    {
        GU_PrimVDB::createMetadataFromGridAttrsAdapter(&grid, prim, gdp);
    }

    /// @brief Transfer attributes to VDB metadata.
    /// @param meta_map  the output metadata
    /// @param owner     the type of element
    /// @param element   the offset of the element
    /// @param geo       the detail from which to retrieve primitive attributes
    static void createMetadataFromAttrs(
        openvdb::MetaMap& meta_map,
        GA_AttributeOwner owner,
        GA_Offset element,
        const GEO_Detail& geo)
    {
        GU_PrimVDB::createMetadataFromAttrsAdapter(&meta_map, owner, element, geo);
    }

private: // METHODS

    /// Add a border of the given radius by evaluating from the given volume.
    /// It assumes that the VDB is a float grid and that the voxel array has
    /// the same index space, so this can really only be safely called after
    /// buildFromPrimVolume(). This is used to ensure that non-constant borders
    /// can be converted at the expense of some extra memory.
    void		expandBorderFromPrimVolume(
			    const GEO_PrimVolume &vol,
			    int border_radius);

    GEO_Primitive *	convertToNewPrim(
			    GEO_Detail &dst_geo,
			    GU_ConvertParms &parms,
			    fpreal adaptivity,
			    bool split_disjoint_volumes,
			    bool &success) const;
    GEO_Primitive *	convertToPrimVolume(
			    GEO_Detail &dst_geo,
			    GU_ConvertParms &parms,
			    bool split_disjoint_volumes) const;
    GEO_Primitive *	convertToPoly(
			    GEO_Detail &dst_geo,
			    GU_ConvertParms &parms,
			    fpreal adaptivity,
			    bool buildpolysoup,
			    bool &success) const;

    static GU_PrimVDB*	buildFromGridAdapter(
			    GU_Detail& gdp,
			    void* grid,
			    const GEO_PrimVDB*,
			    const char* name);
    static void		createGridAttrsFromMetadataAdapter(
			    const GEO_PrimVDB& prim,
			    const void* grid,
			    GEO_Detail& gdp);
    static void		createMetadataFromGridAttrsAdapter(
			    void* grid,
			    const GEO_PrimVDB&,
			    const GEO_Detail&);

    static void createAttrsFromMetadataAdapter(
        GA_AttributeOwner owner,
        GA_Offset element,
        const void* meta_map_ptr,
        GEO_Detail& geo);

    static void createMetadataFromAttrsAdapter(
        void* meta_map_ptr,
        GA_AttributeOwner owner,
        GA_Offset element,
        const GEO_Detail& geo);

private: // DATA

    static GA_PrimitiveDefinition	*theDefinition;
    friend class			 GU_PrimitiveFactory;
#if (UT_VERSION_INT >= 0x0d000000) // 13.0 or later
    SYS_DEPRECATED_PUSH_DISABLE()
#endif
};
#if (UT_VERSION_INT >= 0x0d000000) // 13.0 or later
    SYS_DEPRECATED_POP_DISABLE()
#endif


#ifndef SESI_OPENVDB
namespace openvdb_houdini {
using ::GU_PrimVDB;
} // namespace openvdb_houdini
#endif

#endif // __HDK_GU_PrimVDB__

#endif // UT_VERSION_INT < 0x0c050157 // earlier than 12.5.343

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
