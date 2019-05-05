///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
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
/// @file SOP_NodeVDB.cc
/// @author FX R&D OpenVDB team

#include "SOP_NodeVDB.h"

#include <houdini_utils/geometry.h>
#include <openvdb/points/PointDataGrid.h>
#include "PointUtils.h"
#include "Utils.h"
#include "GEO_PrimVDB.h"
#include "GU_PrimVDB.h"
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <OP/OP_NodeInfoParms.h>
#include <PRM/PRM_Parm.h>
#include <PRM/PRM_Type.h>
#include <SOP/SOP_Cache.h> // for stealable
#include <UT/UT_InfoTree.h>
#include <UT/UT_SharedPtr.h>
#include <tbb/mutex.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

/// Enables custom UT_InfoTree data from SOP_NodeVDB::fillInfoTreeNodeSpecific()
/// which is used to populate the mako templates in Houdini 16 and greater.
/// The templates are used to provide MMB information on Houdini primitives and
/// are installed as part of the Houdini toolkit $HH/config/NodeInfoTemplates.
/// This code has since been absorbed by SideFX, but we continue to keep
/// it around to demonstrate how to extend the templates in Houdini. Note
/// that the current implementation is a close duplicate of the data populated
/// by Houdini, so this will clash with native Houdini names. The templates
/// may also change in future Houdini versions, so do not expect this to
/// produce valid results out the box.
///
/// For users wishing to customize the .mako files, you can use python to
/// inspect the current mako structure.
///
/// @code
/// infoTree = hou.node('/obj/geo1/vdbfrompolygons1').infoTree()
/// sopInfo  = infoTree.branches()['SOP Info']
/// sparseInfo = sopInfo.branches()['Sparse Volumes']
/// @endcode
///
/// These mako branches are the paths that are populated by UT_InfoTree. The
/// mako files responsible for producing VDB specific data are geometry.mako,
/// called by sop.mako.
///
//#define OPENVDB_CUSTOM_MAKO


namespace openvdb_houdini {

namespace node_info_text {

using Mutex = tbb::mutex;
using Lock = Mutex::scoped_lock;
// map of function callbacks to grid types
using ApplyGridSpecificInfoTextMap = std::map<openvdb::Name, ApplyGridSpecificInfoText>;

struct LockedInfoTextRegistry
{
    LockedInfoTextRegistry() {}
    ~LockedInfoTextRegistry() {}

    Mutex mMutex;
    ApplyGridSpecificInfoTextMap mApplyGridSpecificInfoTextMap;
};

// Declare this at file scope to ensure thread-safe initialization
static Mutex theInitInfoTextRegistryMutex;

// Global function for accessing the regsitry
static LockedInfoTextRegistry*
getInfoTextRegistry()
{
    Lock lock(theInitInfoTextRegistryMutex);

    static LockedInfoTextRegistry *registry = nullptr;

    if (registry == nullptr) {
#if defined(__ICC)
__pragma(warning(disable:1711)) // disable ICC "assignment to static variable" warnings
#endif
        registry = new LockedInfoTextRegistry();
#if defined(__ICC)
__pragma(warning(default:1711))
#endif
    }

    return registry;
}


void registerGridSpecificInfoText(const std::string&, ApplyGridSpecificInfoText);
ApplyGridSpecificInfoText getGridSpecificInfoText(const std::string&);


void
registerGridSpecificInfoText(const std::string& gridType, ApplyGridSpecificInfoText callback)
{
    LockedInfoTextRegistry *registry = getInfoTextRegistry();
    Lock lock(registry->mMutex);

    if(registry->mApplyGridSpecificInfoTextMap.find(gridType) !=
       registry->mApplyGridSpecificInfoTextMap.end()) return;

    registry->mApplyGridSpecificInfoTextMap[gridType] = callback;
}

/// @brief Return a pointer to a grid information function, or @c nullptr
///        if no specific function has been registered for the given grid type.
/// @note The defaultNodeSpecificInfoText() method is always returned prior to Houdini 14.
ApplyGridSpecificInfoText
getGridSpecificInfoText(const std::string& gridType)
{
    LockedInfoTextRegistry *registry = getInfoTextRegistry();
    Lock lock(registry->mMutex);

    const ApplyGridSpecificInfoTextMap::const_iterator iter =
        registry->mApplyGridSpecificInfoTextMap.find(gridType);

    if (iter == registry->mApplyGridSpecificInfoTextMap.end() || iter->second == nullptr) {
        return nullptr; // Native prim info is sufficient
    }

    return iter->second;
}

} // namespace node_info_text


////////////////////////////////////////


SOP_NodeVDB::SOP_NodeVDB(OP_Network* net, const char* name, OP_Operator* op):
    SOP_Node(net, name, op)
{
#ifndef SESI_OPENVDB
    // Initialize the OpenVDB library
    openvdb::initialize();
    // Forward OpenVDB log messages to the UT_ErrorManager (for all SOPs).
    startLogForwarding(SOP_OPTYPE_ID);
#endif

    // Register grid-specific info text for Point Data Grids
    node_info_text::registerGridSpecificInfoText<openvdb::points::PointDataGrid>(
        &pointDataGridSpecificInfoText);

    // Set the flag to draw guide geometry
    mySopFlags.setNeedGuide1(true);
}


////////////////////////////////////////


const GA_PrimitiveGroup*
SOP_NodeVDB::matchGroup(GU_Detail& aGdp, const std::string& pattern)
{
    /// @internal Presumably, when a group name pattern matches multiple groups,
    /// a new group must be created that is the union of the matching groups,
    /// and therefore the detail must be non-const.  Since inputGeo() returns
    /// a const detail, we can't match groups in input details; however,
    /// we usually copy input 0 to the output detail, so we can in effect
    /// match groups from input 0 by matching them in the output instead.

    const GA_PrimitiveGroup* group = nullptr;
    if (!pattern.empty()) {
        // If a pattern was provided, try to match it.
        group = parsePrimitiveGroups(pattern.c_str(), GroupCreator(&aGdp, false));
        if (!group) {
            // Report an error if the pattern didn't match.
            throw std::runtime_error(("Invalid group (" + pattern + ")").c_str());
        }
    }
    return group;
}

const GA_PrimitiveGroup*
SOP_NodeVDB::matchGroup(const GU_Detail& aGdp, const std::string& pattern)
{
    const GA_PrimitiveGroup* group = nullptr;
    if (!pattern.empty()) {
        // If a pattern was provided, try to match it.
        group = parsePrimitiveGroups(pattern.c_str(), GroupCreator(&aGdp));
        if (!group) {
            // Report an error if the pattern didn't match.
            throw std::runtime_error(("Invalid group (" + pattern + ")").c_str());
        }
    }
    return group;
}


////////////////////////////////////////


void
SOP_NodeVDB::fillInfoTreeNodeSpecific(UT_InfoTree& tree, const OP_NodeInfoTreeParms& parms)
{
    SOP_Node::fillInfoTreeNodeSpecific(tree, parms);

    // Add the OpenVDB library version number to this node's
    // extended operator information.
    if (UT_InfoTree* child = tree.addChildMap("OpenVDB")) {
        child->addProperties("OpenVDB Version", openvdb::getLibraryAbiVersionString());
    }

#ifdef OPENVDB_CUSTOM_MAKO
    UT_StringArray sparseVolumeTreePath({"SOP Info", "Sparse Volumes"});
    if (UT_InfoTree* sparseVolumes = tree.getDescendentPtr(sparseVolumeTreePath)) {
        if (UT_InfoTree* info = sparseVolumes->addChildBranch("OpenVDB Points")) {

            OP_Context context(parms.getTime());
            GU_DetailHandle gdHandle = getCookedGeoHandle(context);
            if (gdHandle.isNull()) return;

            GU_DetailHandleAutoReadLock gdLock(gdHandle);
            const GU_Detail* tmpGdp = gdLock.getGdp();
            if (!tmpGdp) return;

            info->addColumnHeading("Point Count");
            info->addColumnHeading("Point Groups");
            info->addColumnHeading("Point Attributes");

            for (VdbPrimCIterator it(tmpGdp); it; ++it) {
                const openvdb::GridBase::ConstPtr grid = it->getConstGridPtr();
                if (!grid) continue;
                if (!grid->isType<openvdb::points::PointDataGrid>()) continue;

                const openvdb::points::PointDataGrid& points =
                    *openvdb::gridConstPtrCast<openvdb::points::PointDataGrid>(grid);

                std::string countStr, groupStr, attributeStr;
                collectPointInfo(points, countStr, groupStr, attributeStr);

                ut_PropertyRow* row = info->addProperties();
                row->append(countStr);
                row->append(groupStr);
                row->append(attributeStr);
            }
        }
    }
#endif
}


void
SOP_NodeVDB::getNodeSpecificInfoText(OP_Context &context, OP_NodeInfoParms &parms)
{
    SOP_Node::getNodeSpecificInfoText(context, parms);

#ifdef SESI_OPENVDB
    // Nothing needed since we will report it as part of native prim info
#else
    // Get a handle to the geometry.
    GU_DetailHandle gd_handle = getCookedGeoHandle(context);

   // Check if we have a valid detail handle.
    if(gd_handle.isNull()) return;

    // Lock it for reading.
    GU_DetailHandleAutoReadLock gd_lock(gd_handle);
    // Finally, get at the actual GU_Detail.
    const GU_Detail* tmp_gdp = gd_lock.getGdp();

    std::ostringstream infoStr;

    unsigned gridn = 0;

    for (VdbPrimCIterator it(tmp_gdp); it; ++it) {

        const openvdb::GridBase& grid = it->getGrid();

        node_info_text::ApplyGridSpecificInfoText callback =
            node_info_text::getGridSpecificInfoText(grid.type());
        if (callback) {
            // Note, the output string stream for every new grid is initialized with
            // its index and houdini primitive name prior to executing the callback
            const UT_String gridName = it.getPrimitiveName();

            infoStr << "  (" << it.getIndex() << ")";
            if(gridName.isstring()) infoStr << " name: '" << gridName << "',";


            (*callback)(infoStr, grid);

            infoStr<<"\n";

            ++gridn;
        }
    }

    if (gridn > 0) {
        std::ostringstream headStr;
        headStr << gridn << " Custom VDB grid" << (gridn == 1 ? "" : "s") << "\n";

        parms.append(headStr.str().c_str());
        parms.append(infoStr.str().c_str());
    }
#endif
}


////////////////////////////////////////


OP_ERROR
SOP_NodeVDB::duplicateSourceStealable(const unsigned index,
    OP_Context& context, GU_Detail **pgdp, GU_DetailHandle& gdh, bool clean)
{
    OPENVDB_NO_DEPRECATION_WARNING_BEGIN
    // traverse upstream nodes, if unload is not possible, duplicate the source
    if (!isSourceStealable(index, context)) {
        duplicateSource(index, context, *pgdp, clean);
        unlockInput(index);
        return error();
    }
    OPENVDB_NO_DEPRECATION_WARNING_END

    // get the input GU_Detail handle and unlock the inputs
    GU_DetailHandle inputgdh = inputGeoHandle(index);

    unlockInput(index);
    SOP_Node *input = CAST_SOPNODE(getInput(index));

    if (!input) {
        addError(SOP_MESSAGE, "Invalid input SOP Node when attempting to unload.");
        return error();
    }

    // explicitly unload the data from the input SOP
    const bool unloadSuccessful = input->unloadData();

    // check if we only have one reference
    const bool soleReference = (inputgdh.getRefCount() == 1);

    // if the unload was unsuccessful or the reference count is not one, we fall back to
    // explicitly copying the input onto the gdp
    if (!(unloadSuccessful && soleReference)) {
        const GU_Detail *src = inputgdh.readLock();
        UT_ASSERT(src);
        if (src)  (*pgdp)->copy(*src);
        inputgdh.unlock(src);
        return error();
    }

    // release our old write lock on gdp (setup by cookMe())
    gdh.unlock(*pgdp);
    // point to the input's old gdp and setup a write lock
    gdh = inputgdh;
    *pgdp = gdh.writeLock();

    return error();
}


bool
SOP_NodeVDB::isSourceStealable(const unsigned index, OP_Context& context) const
{
    struct Local {
        static inline OP_Node* nextStealableInput(
            const unsigned idx, const fpreal now, const OP_Node* node)
        {
            OP_Node* input = node->getInput(idx);
            while (input) {
                OP_Node* passThrough = input->getPassThroughNode(now);
                if (!passThrough) break;
                input = passThrough;
            }
            return input;
        }
    }; // struct Local

    const fpreal now = context.getTime();

    for (OP_Node* node = Local::nextStealableInput(index, now, this); node != nullptr;
        node = Local::nextStealableInput(index, now, node))
    {
        // cont'd if it is a SOP_NULL.
        std::string opname = node->getName().toStdString().substr(0, 4);
        if (opname == "null") continue;

        // if the SOP is a cache SOP we don't want to try and alter its data without a deep copy
        if (dynamic_cast<SOP_Cache*>(node))  return false;

        if (node->getUnload() != 0)
            return true;
        else
            return false;
    }
    return false;
}


OP_ERROR
SOP_NodeVDB::duplicateSourceStealable(const unsigned index, OP_Context& context)
{
    OPENVDB_NO_DEPRECATION_WARNING_BEGIN
    auto error = this->duplicateSourceStealable(index, context, &gdp, myGdpHandle, true);
    OPENVDB_NO_DEPRECATION_WARNING_END
    return error;
}


////////////////////////////////////////


const SOP_NodeVerb*
SOP_NodeVDB::cookVerb() const
{
    if (const auto* verb = SOP_NodeVerb::lookupVerb(getOperator()->getName())) {
        return verb; ///< @todo consider caching this
    }
    return SOP_Node::cookVerb();
}


OP_ERROR
SOP_NodeVDB::cookMySop(OP_Context& context)
{
    if (cookVerb()) {
        return cookMyselfAsVerb(context);
    }
    return cookVDBSop(context);
}


////////////////////////////////////////


namespace {

void
createEmptyGridGlyph(GU_Detail& gdp, GridCRef grid)
{
    openvdb::Vec3R lines[6];

    lines[0].init(-0.5, 0.0, 0.0);
    lines[1].init( 0.5, 0.0, 0.0);
    lines[2].init( 0.0,-0.5, 0.0);
    lines[3].init( 0.0, 0.5, 0.0);
    lines[4].init( 0.0, 0.0,-0.5);
    lines[5].init( 0.0, 0.0, 0.5);

    const openvdb::math::Transform &xform = grid.transform();
    lines[0] = xform.indexToWorld(lines[0]);
    lines[1] = xform.indexToWorld(lines[1]);
    lines[2] = xform.indexToWorld(lines[2]);
    lines[3] = xform.indexToWorld(lines[3]);
    lines[4] = xform.indexToWorld(lines[4]);
    lines[5] = xform.indexToWorld(lines[5]);

    UT_SharedPtr<GU_Detail> tmpGDP(new GU_Detail);

    UT_Vector3 color(0.1f, 1.0f, 0.1f);
    tmpGDP->addFloatTuple(GA_ATTRIB_POINT, "Cd", 3, GA_Defaults(color.data(), 3));

    GU_PrimPoly *poly;

    for (int i = 0; i < 6; i += 2) {
        poly = GU_PrimPoly::build(&*tmpGDP, 2, GU_POLY_OPEN);

        tmpGDP->setPos3(poly->getPointOffset(i % 2),
            UT_Vector3(float(lines[i][0]), float(lines[i][1]), float(lines[i][2])));

        tmpGDP->setPos3(poly->getPointOffset(i % 2 + 1),
            UT_Vector3(float(lines[i + 1][0]), float(lines[i + 1][1]), float(lines[i + 1][2])));
    }

    gdp.merge(*tmpGDP);
}

} // unnamed namespace


OP_ERROR
SOP_NodeVDB::cookMyGuide1(OP_Context& context)
{
#ifndef SESI_OPENVDB
    myGuide1->clearAndDestroy();
    UT_Vector3 color(0.1f, 0.1f, 1.0f);
    UT_Vector3 corners[8];

    // For each VDB primitive (with a non-null grid pointer) in the group...
    for (VdbPrimIterator it(gdp); it; ++it) {
        if (evalGridBBox(it->getGrid(), corners, /*expandHalfVoxel=*/true)) {
            houdini_utils::createBox(*myGuide1, corners, &color);
        } else {
            createEmptyGridGlyph(*myGuide1, it->getGrid());
        }
    }
#endif
    return SOP_Node::cookMyGuide1(context);
}


////////////////////////////////////////


openvdb::Vec3f
SOP_NodeVDB::evalVec3f(const char *name, fpreal time) const
{
    return openvdb::Vec3f(float(evalFloat(name, 0, time)),
                          float(evalFloat(name, 1, time)),
                          float(evalFloat(name, 2, time)));
}

openvdb::Vec3R
SOP_NodeVDB::evalVec3R(const char *name, fpreal time) const
{
    return openvdb::Vec3R(evalFloat(name, 0, time),
                          evalFloat(name, 1, time),
                          evalFloat(name, 2, time));
}

openvdb::Vec3i
SOP_NodeVDB::evalVec3i(const char *name, fpreal time) const
{
    using ValueT = openvdb::Vec3i::value_type;
    return openvdb::Vec3i(static_cast<ValueT>(evalInt(name, 0, time)),
                          static_cast<ValueT>(evalInt(name, 1, time)),
                          static_cast<ValueT>(evalInt(name, 2, time)));
}

openvdb::Vec2R
SOP_NodeVDB::evalVec2R(const char *name, fpreal time) const
{
    return openvdb::Vec2R(evalFloat(name, 0, time),
                          evalFloat(name, 1, time));
}

openvdb::Vec2i
SOP_NodeVDB::evalVec2i(const char *name, fpreal time) const
{
    using ValueT = openvdb::Vec2i::value_type;
    return openvdb::Vec2i(static_cast<ValueT>(evalInt(name, 0, time)),
                          static_cast<ValueT>(evalInt(name, 1, time)));
}


std::string
SOP_NodeVDB::evalStdString(const char* name, fpreal time, int index) const
{
    UT_String str;
    evalString(str, name, index, time);
    return str.toStdString();
}


////////////////////////////////////////


void
SOP_NodeVDB::resolveRenamedParm(PRM_ParmList& obsoleteParms,
    const char* oldName, const char* newName)
{
    PRM_Parm* parm = obsoleteParms.getParmPtr(oldName);
    if (parm && !parm->isFactoryDefault()) {
        if (this->hasParm(newName)) {
            this->getParm(newName).copyParm(*parm);
        }
    }
}


////////////////////////////////////////


namespace {


/// @brief Default OpPolicy for OpenVDB operator types
class DefaultOpenVDBOpPolicy: public houdini_utils::OpPolicy
{
public:
    std::string getValidName(const std::string& english)
    {
        UT_String s(english);
        // Remove non-alphanumeric characters from the name.
        s.forceValidVariableName();
        std::string name = s.toStdString();
        // Remove spaces and underscores.
        name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
        name.erase(std::remove(name.begin(), name.end(), '_'), name.end());
        return name;
    }

    std::string getLowercaseName(const std::string& english)
    {
        UT_String s(english);
        // Lowercase
        s.toLower();
        return s.toStdString();
    }

    /// @brief OpenVDB operators of each flavor (SOP, POP, etc.) share
    /// an icon named "SOP_OpenVDB", "POP_OpenVDB", etc.
    std::string getIconName(const houdini_utils::OpFactory& factory) override
    {
        return factory.flavorString() + "_OpenVDB";
    }
};


/// @brief SideFX OpPolicy for OpenVDB operator types
class SESIOpenVDBOpPolicy: public DefaultOpenVDBOpPolicy
{
public:
    std::string getName(const houdini_utils::OpFactory&, const std::string& english) override
    {
        return this->getLowercaseName(this->getValidName(english));
    }
};


/// @brief ASWF OpPolicy for OpenVDB operator types
class ASWFOpenVDBOpPolicy: public DefaultOpenVDBOpPolicy
{
public:
    std::string getName(const houdini_utils::OpFactory&, const std::string& english) override
    {
        return "DW_Open" + this->getValidName(english);
    }

    std::string getLabelName(const houdini_utils::OpFactory& factory) override
    {
        return "Open" + factory.english();
    }

    std::string getFirstName(const houdini_utils::OpFactory& factory) override
    {
        return this->getLowercaseName(this->getValidName(this->getLabelName(factory)));
    }
};


#ifdef SESI_OPENVDB
using OpenVDBOpPolicy = SESIOpenVDBOpPolicy;
#else
using OpenVDBOpPolicy = ASWFOpenVDBOpPolicy;
#endif // SESI_OPENVDB

} // unnamed namespace


OpenVDBOpFactory::OpenVDBOpFactory(
    const std::string& english,
    OP_Constructor ctor,
    houdini_utils::ParmList& parms,
    OP_OperatorTable& table,
    houdini_utils::OpFactory::OpFlavor flavor):
    houdini_utils::OpFactory(OpenVDBOpPolicy(), english, ctor, parms, table, flavor)
{
}

} // namespace openvdb_houdini

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
