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
/// @file Utils.cc
/// @author FX R&D Simulation team
/// @brief Utility classes and functions for OpenVDB plugins

#include "Utils.h"

#include <houdini_utils/ParmFactory.h>
#include "GEO_PrimVDB.h"
#include <GU/GU_Detail.h>
#include <UT/UT_String.h>
#include <UT/UT_Version.h>

namespace openvdb_houdini {

VdbPrimCIterator::VdbPrimCIterator(const GEO_Detail* gdp, const GA_PrimitiveGroup* group,
    FilterFunc filter):
    mIter(gdp ? new GA_GBPrimitiveIterator(*gdp, group) : NULL),
    mFilter(filter)
{
    // Ensure that, after construction, this iterator points to
    // a valid VDB primitive (if there is one).
    if (NULL == getPrimitive()) advance();
}


VdbPrimCIterator::VdbPrimCIterator(const GEO_Detail* gdp, GA_Range::safedeletions,
    const GA_PrimitiveGroup* group, FilterFunc filter):
    mIter(gdp ? new GA_GBPrimitiveIterator(*gdp, group, GA_Range::safedeletions()) : NULL),
    mFilter(filter)
{
    // Ensure that, after construction, this iterator points to
    // a valid VDB primitive (if there is one).
    if (NULL == getPrimitive()) advance();
}


VdbPrimCIterator::VdbPrimCIterator(const VdbPrimCIterator& other):
    mIter(other.mIter ? new GA_GBPrimitiveIterator(*other.mIter) : NULL),
    mFilter(other.mFilter)
{
}


VdbPrimCIterator&
VdbPrimCIterator::operator=(const VdbPrimCIterator& other)
{
    if (&other != this) {
        mIter.reset(other.mIter ? new GA_GBPrimitiveIterator(*other.mIter) : NULL);
        mFilter = other.mFilter;
    }
    return *this;
}


void
VdbPrimCIterator::advance()
{
    if (mIter) {
        GA_GBPrimitiveIterator& iter = *mIter;
        for (++iter; iter.getPrimitive() != NULL && getPrimitive() == NULL; ++iter) {}
    }
}


const GU_PrimVDB*
VdbPrimCIterator::getPrimitive() const
{
    if (mIter) {
        if (GA_Primitive* prim = mIter->getPrimitive()) {
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
            const GA_PrimitiveTypeId primVdbTypeId = GA_PRIMVDB;
#else
            const GA_PrimitiveTypeId primVdbTypeId = GU_PrimVDB::theTypeId();
#endif
            if (prim->getTypeId() == primVdbTypeId) {
                GU_PrimVDB* vdb = UTverify_cast<GU_PrimVDB*>(prim);
                if (mFilter && !mFilter(*vdb)) return NULL;
                return vdb;
            }
        }
    }
    return NULL;
}


UT_String
VdbPrimCIterator::getPrimitiveName(const UT_String& defaultName) const
{
    // We must have always deep enabled on returned UT_String objects to avoid
    // having it deleted before the caller has a chance to use it.
    UT_String name(UT_String::ALWAYS_DEEP);

    if (const GU_PrimVDB* vdb = getPrimitive()) {
        name = vdb->getGridName();
        if (!name.isstring()) name = defaultName;
    }
    return name;
}


UT_String
VdbPrimCIterator::getPrimitiveNameOrIndex() const
{
    UT_String name;
    name.itoa(this->getIndex());
    return this->getPrimitiveName(/*defaultName=*/name);
}


////////////////////////////////////////


VdbPrimIterator::VdbPrimIterator(const VdbPrimIterator& other): VdbPrimCIterator(other)
{
}


VdbPrimIterator&
VdbPrimIterator::operator=(const VdbPrimIterator& other)
{
    if (&other != this) VdbPrimCIterator::operator=(other);
    return *this;
}


////////////////////////////////////////


GU_PrimVDB*
createVdbPrimitive(GU_Detail& gdp, GridPtr grid, const char* name)
{
    return (!grid ? NULL : GU_PrimVDB::buildFromGrid(gdp, grid, /*src=*/NULL, name));
}


GU_PrimVDB*
replaceVdbPrimitive(GU_Detail& gdp, GridPtr grid, GEO_PrimVDB& src,
    const bool copyAttrs, const char* name)
{
    GU_PrimVDB* vdb = NULL;
    if (grid) {
        vdb = GU_PrimVDB::buildFromGrid(gdp, grid, (copyAttrs ? &src : NULL), name);
        gdp.destroyPrimitive(src, /*andPoints=*/true);
    }
    return vdb;
}


////////////////////////////////////////


bool
evalGridBBox(GridCRef grid, UT_Vector3 corners[8], bool expandHalfVoxel)
{
    if (grid.activeVoxelCount() == 0) return false;

    openvdb::CoordBBox activeBBox = grid.evalActiveVoxelBoundingBox();
    if (!activeBBox) return false;

    openvdb::BBoxd voxelBBox(activeBBox.min().asVec3d(), activeBBox.max().asVec3d());
    if (expandHalfVoxel) {
        voxelBBox.min() -= openvdb::Vec3d(0.5);
        voxelBBox.max() += openvdb::Vec3d(0.5);
    }

    openvdb::Vec3R bbox[8];
    bbox[0] = voxelBBox.min();
    bbox[1].init(voxelBBox.min()[0], voxelBBox.min()[1], voxelBBox.max()[2]);
    bbox[2].init(voxelBBox.max()[0], voxelBBox.min()[1], voxelBBox.max()[2]);
    bbox[3].init(voxelBBox.max()[0], voxelBBox.min()[1], voxelBBox.min()[2]);
    bbox[4].init(voxelBBox.min()[0], voxelBBox.max()[1], voxelBBox.min()[2]);
    bbox[5].init(voxelBBox.min()[0], voxelBBox.max()[1], voxelBBox.max()[2]);
    bbox[6] = voxelBBox.max();
    bbox[7].init(voxelBBox.max()[0], voxelBBox.max()[1], voxelBBox.min()[2]);

    const openvdb::math::Transform& xform = grid.transform();
    bbox[0] = xform.indexToWorld(bbox[0]);
    bbox[1] = xform.indexToWorld(bbox[1]);
    bbox[2] = xform.indexToWorld(bbox[2]);
    bbox[3] = xform.indexToWorld(bbox[3]);
    bbox[4] = xform.indexToWorld(bbox[4]);
    bbox[5] = xform.indexToWorld(bbox[5]);
    bbox[6] = xform.indexToWorld(bbox[6]);
    bbox[7] = xform.indexToWorld(bbox[7]);

    for (size_t i = 0; i < 8; ++i) {
        corners[i].assign(bbox[i][0], bbox[i][1], bbox[i][2]);
    }

    return true;
}


////////////////////////////////////////


openvdb::CoordBBox
makeCoordBBox(const UT_BoundingBox& b, const openvdb::math::Transform& t)
{
    openvdb::Vec3d minWS, maxWS, minIS, maxIS;

    minWS[0] = double(b.xmin());
    minWS[1] = double(b.ymin());
    minWS[2] = double(b.zmin());

    maxWS[0] = double(b.xmax());
    maxWS[1] = double(b.ymax());
    maxWS[2] = double(b.zmax());

    openvdb::math::calculateBounds(t, minWS, maxWS, minIS, maxIS);

    openvdb::CoordBBox box;
    box.min() = openvdb::Coord::floor(minIS);
    box.max() = openvdb::Coord::ceil(maxIS);

    return box;
}


} // namespace openvdb_houdini

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
