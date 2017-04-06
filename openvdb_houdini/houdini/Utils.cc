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
#ifdef OPENVDB_USE_LOG4CPLUS
#include <openvdb/util/logging.h>
#include <UT/UT_ErrorManager.h>
#include <CHOP/CHOP_Error.h> // for CHOP_ERROR_MESSAGE
#include <DOP/DOP_Error.h> // for DOP_MESSAGE
#include <POP/POP_Error.h> // for POP_MESSAGE
#include <ROP/ROP_Error.h> // for ROP_MESSAGE
#include <VOP/VOP_Error.h> // for VOP_MESSAGE
#include <VOPNET/VOPNET_Error.h> // for VOPNET_MESSAGE
#include <string>
#endif


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
    // We must have ALWAYS_DEEP enabled on returned UT_String objects to avoid
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


UT_String
VdbPrimCIterator::getPrimitiveIndexAndName(bool keepEmptyName) const
{
    // We must have ALWAYS_DEEP enabled on returned UT_String objects to avoid
    // having it deleted before the caller has a chance to use it.
    UT_String result(UT_String::ALWAYS_DEEP);

    if (const GU_PrimVDB* vdb = getPrimitive()) {
        result.itoa(this->getIndex());
        UT_String name = vdb->getGridName();
        if (keepEmptyName || name.isstring()) {
            result += (" (" + name.toStdString() + ")").c_str();
        }
    }
    return result;
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
        corners[i].assign(float(bbox[i][0]), float(bbox[i][1]), float(bbox[i][2]));
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


////////////////////////////////////////


#ifndef OPENVDB_USE_LOG4CPLUS

void startLogForwarding(OP_OpTypeId) {}
void stopLogForwarding(OP_OpTypeId) {}
bool isLogForwarding(OP_OpTypeId) { return false; }

#else

namespace {

namespace l4c = log4cplus;

/// @brief log4cplus appender that directs log messages to UT_ErrorManager
class HoudiniAppender: public l4c::Appender
{
public:
    /// @param opType  SOP_OPTYPE_NAME, POP_OPTYPE_NAME, etc. (see OP_Node.h)
    /// @param code    SOP_MESSAGE, SOP_VEX_ERROR, POP_MESSAGE, etc.
    ///                (see SOP_Error.h, POP_Error.h, etc.)
    HoudiniAppender(const char* opType, int code): mOpType(opType), mCode(code) {}

    ~HoudiniAppender() override
    {
        close();
        destructorImpl(); // must be called by Appender subclasses
    }

    void append(const l4c::spi::InternalLoggingEvent& event) override
    {
        if (mClosed) return;

        auto* errMgr = UTgetErrorManager();
        if (!errMgr || errMgr->isDisabled()) return;

        const l4c::LogLevel level = event.getLogLevel();
        const std::string& msg = event.getMessage();
        const std::string& file = event.getFile();
        const int line = event.getLine();

        const UT_SourceLocation
            loc{file.c_str(), line},
            *locPtr = (file.empty() ? nullptr : &loc);

        UT_ErrorSeverity severity = UT_ERROR_NONE;
        switch (level) {
            case l4c::DEBUG_LOG_LEVEL: severity = UT_ERROR_MESSAGE; break;
            case l4c::INFO_LOG_LEVEL: severity = UT_ERROR_MESSAGE; break;
            case l4c::WARN_LOG_LEVEL: severity = UT_ERROR_WARNING; break;
            case l4c::ERROR_LOG_LEVEL: severity = UT_ERROR_ABORT; break;
            case l4c::FATAL_LOG_LEVEL: severity = UT_ERROR_FATAL; break;
        }
        errMgr->addGeneric(mOpType.c_str(), mCode, msg.c_str(), severity, locPtr);
    }

    void close() override { mClosed = true; }

private:
    std::string mOpType = INVALID_OPTYPE_NAME;
    int mCode = 0;
    bool mClosed = false;
};


inline l4c::tstring
getAppenderName(const OP_TypeInfo& opInfo)
{
    return LOG4CPLUS_STRING_TO_TSTRING(
        std::string{"HOUDINI_"} + static_cast<const char*>(opInfo.myOptypeName));
}


/// @brief Return the error code for user-supplied messages in operators of the given type.
inline int
getGenericMessageCode(OP_OpTypeId opId)
{
    switch (opId) {
        case CHOP_OPTYPE_ID:   return CHOP_ERROR_MESSAGE;
        case DOP_OPTYPE_ID:    return DOP_MESSAGE;
        case POP_OPTYPE_ID:    return POP_MESSAGE;
        case ROP_OPTYPE_ID:    return ROP_MESSAGE;
        case SOP_OPTYPE_ID:    return SOP_MESSAGE;
        case VOP_OPTYPE_ID:    return VOP_MESSAGE;
        case VOPNET_OPTYPE_ID: return VOPNET_MESSAGE;
        default: break;
    }
    return 0;
}


inline void
setLogForwarding(OP_OpTypeId opId, bool enable)
{
    const auto* opInfo = OP_Node::getOpInfoFromOpTypeID(opId);
    if (!opInfo) return;

    const auto appenderName = getAppenderName(*opInfo);

    auto logger = openvdb::logging::internal::getLogger();
    auto appender = logger.getAppender(appenderName);

    if (appender && !enable) {
        // If an appender for the given operator type exists, remove it.
        logger.removeAppender(appender);
    } else if (!appender && enable) {
        // If an appender for the given operator type doesn't already exist, create one.
        // Otherwise, do nothing: operators of the same type can share a single appender.
        appender = log4cplus::SharedAppenderPtr{
            new HoudiniAppender{opInfo->myOptypeName, getGenericMessageCode(opId)}};
        appender->setName(appenderName);
        logger.addAppender(appender);
    }
}

} // anonymous namespace


void
startLogForwarding(OP_OpTypeId opId)
{
    setLogForwarding(opId, true);
}


void
stopLogForwarding(OP_OpTypeId opId)
{
    setLogForwarding(opId, false);
}


bool
isLogForwarding(OP_OpTypeId opId)
{
    if (const auto* opInfo = OP_Node::getOpInfoFromOpTypeID(opId)) {
        return openvdb::logging::internal::getLogger().getAppender(
            getAppenderName(*opInfo));
    }
    return false;
}

#endif // OPENVDB_USE_LOG4CPLUS

} // namespace openvdb_houdini

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
