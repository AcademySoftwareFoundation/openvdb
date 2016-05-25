///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_From_Particles.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Converts points into signed distance / level set volumes.
///
/// @note The world space narrow band half-width is encoded in the
///       background value of a level set grid.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/AttributeTransferUtil.h>
#include <openvdb_houdini/GU_VDBPointTools.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>

#include <openvdb/Grid.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/TopologyToLevelSet.h>

#include <CH/CH_Manager.h>
#include <GA/GA_Types.h> // for GA_ATTRIB_POINT
#include <PRM/PRM_Parm.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


namespace {

// This wrapper class is required by openvdb::tools::ParticlesToLeveSet
class ParticleList
{
public:
    // Required by @c openvdb::tools::PointPartitioner
    typedef openvdb::Vec3R  PosType;

    ParticleList(const GU_Detail* gdp,
                 openvdb::Real radiusMult = 1,
                 openvdb::Real velocityMult = 1) :
        mGdp(gdp),
        mScaleHandle(gdp, GA_ATTRIB_POINT, GEO_STD_ATTRIB_PSCALE),
        mVelHandle(gdp, GA_ATTRIB_POINT, GEO_STD_ATTRIB_VELOCITY),
        mHasRadius(mScaleHandle.isValid()),
        mHasVelocity(mVelHandle.isValid()),
        mRadiusMult(radiusMult),
        mVelocityMult(velocityMult)
    {
    }

    // Do the particles have non-constant radius
    bool hasRadius()   const { return mHasRadius;}

    // Do the particles have velocity
    bool hasVelocity() const { return mHasVelocity;}

    // Multiplier for the radius
    openvdb::Real& radiusMult() { return mRadiusMult; }
    const openvdb::Real& radiusMult() const { return mRadiusMult; }

    // The public methods below are the only ones required
    // by tools::ParticlesToLevelSet
    size_t size() const { return mGdp->getNumPoints(); }

    // Position of particle in world space
    // This is required by ParticlesToLevelSet::rasterizeSpheres(*this,radius)
    void getPos(size_t n, openvdb::Vec3R& xyz) const
    {
        const UT_Vector3 p = mGdp->getPos3(mGdp->pointOffset(n));
        xyz[0] = p[0], xyz[1] = p[1], xyz[2] = p[2];
    }
    // Position and radius of particle in world space
    // This is required by ParticlesToLevelSet::rasterizeSpheres(*this)
    void getPosRad(size_t n, openvdb::Vec3R& xyz, openvdb::Real& rad) const
    {
        assert(mHasRadius);
        const GA_Offset m = mGdp->pointOffset(n);
        const UT_Vector3 p = mGdp->getPos3(m);
        xyz[0] = p[0], xyz[1] = p[1], xyz[2] = p[2];
        rad = mRadiusMult*mScaleHandle.get(m);
    }
    // Position, radius and velocity of particle in world space
    // This is required by ParticlesToLevelSet::rasterizeTrails
    void getPosRadVel(size_t n, openvdb::Vec3R& xyz,
                      openvdb::Real& rad, openvdb::Vec3R& vel) const
    {
        assert(mHasVelocity);
        const GA_Offset m = mGdp->pointOffset(n);
        const UT_Vector3 p = mGdp->getPos3(m);
        xyz[0] = p[0], xyz[1] = p[1], xyz[2] = p[2];
        rad = mHasRadius ? mRadiusMult*mScaleHandle.get(m) : mRadiusMult;
        const UT_Vector3 v = mVelHandle.get(m);
        vel[0] = mVelocityMult*v[0], vel[1] = mVelocityMult*v[1], vel[2] = mVelocityMult*v[2];
    }
    // Required for attribute transfer
    void getAtt(size_t n, openvdb::Int32& att) const { att = openvdb::Int32(n); }

protected:

    const GU_Detail*    mGdp;
    GA_ROHandleF        mScaleHandle;
    GA_ROHandleV3       mVelHandle;
    const bool          mHasRadius, mHasVelocity;
    openvdb::Real       mRadiusMult; // multiplier for radius
    const openvdb::Real mVelocityMult; // multiplier for velocity
};// ParticleList


////////////////////////////////////////


inline int
lookupAttrInput(const PRM_SpareData* spare)
{
    const char  *istring;
    if (!spare) return 0;
    istring = spare->getValue("sop_input");
    return istring ? atoi(istring) : 0;
}

inline void
sopBuildAttrMenu(void* data, PRM_Name* menuEntries, int themenusize,
    const PRM_SpareData* spare, const PRM_Parm*)
{
    if (data == NULL || menuEntries == NULL || spare == NULL) return;

    SOP_Node* sop = CAST_SOPNODE((OP_Node *)data);

    if (sop == NULL) {
        // terminate and quit
        menuEntries[0].setToken(0);
        menuEntries[0].setLabel(0);
        return;
    }


    int inputIndex = lookupAttrInput(spare);
    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

    size_t menuIdx = 0, menuEnd(themenusize - 2);


    // null object
    menuEntries[menuIdx].setToken("0");
    menuEntries[menuIdx++].setLabel("- no attribute selected -");

    if (gdp) {

        // point attribute names
        GA_AttributeDict::iterator iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);

        if(!iter.atEnd() && menuIdx != menuEnd) {

            if (menuIdx > 0) {
                menuEntries[menuIdx].setToken(PRM_Name::mySeparator);
                menuEntries[menuIdx++].setLabel(PRM_Name::mySeparator);
            }

            for (; !iter.atEnd() && menuIdx != menuEnd; ++iter) {

                std::ostringstream token;
                token << (*iter)->getName();

                menuEntries[menuIdx].setToken(token.str().c_str());
                menuEntries[menuIdx++].setLabel(token.str().c_str());
            }

            // Special case
            menuEntries[menuIdx].setToken("point_list_index");
            menuEntries[menuIdx++].setLabel("point_list_index");
        }
    }

    // terminator
    menuEntries[menuIdx].setToken(0);
    menuEntries[menuIdx].setLabel(0);
}

const PRM_ChoiceList PrimAttrMenu(
    PRM_ChoiceListType(PRM_CHOICELIST_EXCLUSIVE | PRM_CHOICELIST_REPLACE), sopBuildAttrMenu);

} // unnamed namespace


////////////////////////////////////////


class SOP_OpenVDB_From_Particles: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_From_Particles(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_From_Particles() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i > 0); }

    int convertUnits();

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

private:
    virtual void resolveObsoleteParms(PRM_ParmList*);
    void convert(openvdb::FloatGrid::Ptr, ParticleList&, hvdb::Interrupter&);
    void convertWithAttributes(
        openvdb::FloatGrid::Ptr, ParticleList&, const GU_Detail&, hvdb::Interrupter&);

    int constructGenericAtttributeList(
        hvdb::AttributeDetailList &pointAttributes,
        const GU_Detail&,
        const openvdb::Int32Grid& closestPtnIdxGrid);

    template <class ValueType>
    void addAttributeDetails(
        hvdb::AttributeDetailList &attributeList,
        const GA_Attribute *attribute,
        const GA_AIFTuple *tupleAIF,
        const int attrTupleSize,
        const openvdb::Int32Grid& closestPtnIdxGrid,
        std::string& customName,
        int vecType = -1);

    void transferAttributes(
        hvdb::AttributeDetailList &pointAttributes,
        const openvdb::Int32Grid&,
        openvdb::math::Transform::Ptr& transform,
        const GU_Detail&);

    float mVoxelSize;
    fpreal mTime;
};


namespace {

// Callback to convert from voxel to world space units
int
convertUnitsCB(void* data, int /*idx*/, float /*time*/, const PRM_Template*)
{
    SOP_OpenVDB_From_Particles* sop = static_cast<SOP_OpenVDB_From_Particles*>(data);
    if (sop == NULL) return 0;
    return sop->convertUnits();
}

} // unnamed namespace


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "distancevdb", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Compute a narrow-band signed distance / level set grid "
            "from the input points."));

    parms.add(hutil::ParmFactory(PRM_STRING, "distancevdbname", "Distance VDB")
        .setDefault("surface")
        .setHelpText("Distance grid name"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "fogvdb", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Compute a fog volume grid by remapping the level set "
            "volume to [0, 1] range.  The interior region is marked active "
            "and set to one, the interior portion of the active narrow-band "
            "is remapped to (0, 1] range to produce a smooth gradient and "
            "all exterior regions are set to zero, marked inactive and pruned."));

    parms.add(hutil::ParmFactory(PRM_STRING, "fogvdbname", "Fog VDB")
        .setDefault("density")
        .setHelpText("Fog volume grid name"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "maskvdb", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Compute an alpha mask grid that can be used in subsequent "
            "filtering nodes to constrain smoothing operations and preserve "
            "surface features."));

    parms.add(hutil::ParmFactory(PRM_STRING, "maskvdbname", "Mask VDB")
        .setDefault("boundingvolume")
        .setHelpText("Mask grid name"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "boundinglimit", "Bounding Limit")
              .setDefault(0.25)
              .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_RESTRICTED, 1)
              .setHelpText("Percentage to increase and decrease the "
                  "particle radius.  Used to define the maximum and minimum "
                  "limit surfaces for the alpha mask construction."));

    parms.add(hutil::ParmFactory(PRM_STRING, "referencevdb", "Reference VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setHelpText("VDB grid that defines the output transform.  "
            "The half-band width is matched if the input grid is a level set."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "merge", "Merge With Reference VDB"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5)
        .setHelpText("Uniform voxel edge length in world units.  "
            "Decrease the voxel size to increase the volume resolution."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "useworldspace", "Use World Space for Band")
        .setCallbackFunc(&convertUnitsCB)
        .setHelpText("Switch between voxel and world space units for "
            "the half-band width."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "halfbandvoxels", "Half-Band Voxels")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText("Half the width of the narrow band in voxel units.  "
            "The default value 3 is recommended for level set volumes."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "halfband", "Half-Band")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 10)
        .setHelpText("Half the width of the narrow band in world space units."));


    parms.beginExclusiveSwitcher("conversion", "Conversion");
    parms.addFolder("Spheres");

    parms.add(hutil::ParmFactory(PRM_FLT_J, "particlescale", "Particle Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0)
        .setHelpText("The pscale point attribute, which defines the world space "
            "particle radius, will be scaled by this.  A value of one is assumed "
            "if the pscale attribute is missing."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "minradius", "Minimum Radius")
        .setDefault(1.5)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0)
        .setHelpText("Minimum radius in voxel units after scaling.  "
            "Particles smaller than this limit are ignored."));

     parms.add(hutil::ParmFactory(PRM_TOGGLE, "velocitytrails", "Velocity Trails")
        .setHelpText("Velocity trail splatting toggle.  Note this feature "
            "requires a velocity point attribute named 'v' of 3fv type."));

     parms.add(hutil::ParmFactory(PRM_FLT_J, "velocityscale", "Velocity Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_UI, 0.0, PRM_RANGE_UI, 1.0)
        .setHelpText("Scales the velocity point attribute 'v'.  Use "
            "this parameter to control the length of the velocity trails."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "trailresolution", "Trail Resolution")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.2, PRM_RANGE_UI, 2.0)
        .setHelpText("Defines the distance between particle instances.  Use this "
            "parameter to control aliasing and number of particle instances."));

    hutil::ParmList transferParms;

    transferParms.add(hutil::ParmFactory(PRM_STRING, "attribute#", "Attribute")
        .setChoiceList(&PrimAttrMenu)
        .setSpareData(&SOP_Node::theFirstInput)
        .setHelpText("Select a point attribute to transfer.  Supports integer "
            "and floating point attributes of arbitrary precisions and tuple sizes."));

    transferParms.add(hutil::ParmFactory(PRM_STRING, "attributeGridName#", "VDB Name")
        .setHelpText("The attribute name is used as the output grid name by default.  "
            "A different grid name can be specified in this field if desired."));

    {
        std::vector<std::string> items;
        for (int i = 0; i < openvdb::NUM_VEC_TYPES ; ++i) {
            items.push_back(openvdb::GridBase::vecTypeToString(openvdb::VecType(i)));
            items.push_back(openvdb::GridBase::vecTypeExamples(openvdb::VecType(i)));
        }

        transferParms.add(hutil::ParmFactory(PRM_ORD, "vecType#", "Vector Type")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "attrList", "Attributes")
        .setHelpText("Transfer point attributes to each voxel in the level set's narrow band")
        .setMultiparms(transferParms)
        .setDefault(PRMzeroDefaults));


    parms.addFolder("Points");

    parms.add(hutil::ParmFactory(PRM_INT_J, "dilation", "Dilation")
              .setDefault(PRMoneDefaults)
              .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
              .setHelpText("Number of morphological dilation iterations "
                  "used to expand the active voxel region."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "closing", "Closing")
              .setDefault(PRMoneDefaults)
              .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
              .setHelpText("Number of morphological closing iterations "
                  "used to fill gaps in the active voxel region."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "smoothing", "Smoothing")
              .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
              .setHelpText("Number of surface smoothing interations."));

    parms.endSwitcher();


    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "optionsHeading", "Options"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "gradientWidth",  "Gradient width"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "customGradientWidth", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "ptnIndexGridName", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "ptnIndexGrid", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "transformHeading", "Transform"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "outputHeading", "Output grids"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "threading", "Threading"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "matchlevelset", "Match reference level set band"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "levelSet", "").setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "gridName", "").setDefault("surface"));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "fogVolumeGridName", "").setDefault("density"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "fogVolume", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "maskVolume", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "maskVolumeGridName", "").setDefault("mask"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "maskWidth", "").setDefault(0.25));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "group", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "writeintoref", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "bandWidth", "").setDefault(PRMthreeDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "bandWidthWS", "").setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "worldSpaceUnits", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "maxWidth", "Max Half-width"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "Rmax", "Max Radius In Voxels"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "voxelSize", "").setDefault(PRMpointOneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "dR", "").setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "Rmin", "").setDefault(1.5));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "dV", "").setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "dX", "").setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "transferHeading", "Attribute transfer"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "particleHeading", "Conversion settings"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "Prune Level Set"));
    {
        const char* items[] = { "sphere", "Spherical", "trail",  "Velocity Trail", NULL };
        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "footprint", "Particle Footprint")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    // temporary parameters used in the 3.2 beta
    obsoleteParms.add(hutil::ParmFactory(PRM_INT_J, "dilate", "").setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_INT_J, "erode", "").setDefault(PRMoneDefaults));


    hvdb::OpenVDBOpFactory("OpenVDB From Particles",
        SOP_OpenVDB_From_Particles::factory, parms, *table)
        .addAlias("OpenVDB Particle Voxelizer")
        .setObsoleteParms(obsoleteParms)
        .addInput("Points to convert")
        .addOptionalInput("Optional VDB grid that defines the output transform. "
            "The half-band width is matched if the input grid is a level set.");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_From_Particles::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_From_Particles(net, name, op);
}


SOP_OpenVDB_From_Particles::SOP_OpenVDB_From_Particles(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
    , mVoxelSize(0.1f)
    , mTime(0.0)
{
}


////////////////////////////////////////


int
SOP_OpenVDB_From_Particles::convertUnits()
{
    const fpreal time = CHgetEvalTime();

    if (evalInt("useworldspace", 0, time) != 0) {
        setFloat("halfband", 0, time,
            evalFloat("halfbandvoxels", 0, time) * mVoxelSize);
    } else {
        setFloat("halfbandvoxels", 0, time,
            evalFloat("halfband", 0, time) / mVoxelSize);
    }

    return 1;
}


void
SOP_OpenVDB_From_Particles::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms || obsoleteParms->allDefaults() != 0) return;

    typedef std::pair<std::string, std::string> StringPair;

    const fpreal time = CHgetEvalTime();

    PRM_Parm* parm = obsoleteParms->getParmPtr("footprint");
    if (parm && !parm->isFactoryDefault()) {
        setInt("velocitytrails", 0, time, 1);
    }

    // Note this function is only evaluated once.

    std::vector<StringPair> floatAttributes;
    floatAttributes.push_back(StringPair("dR", "particlescale"));
    floatAttributes.push_back(StringPair("Rmin", "minradius"));
    floatAttributes.push_back(StringPair("dV", "velocityscale"));
    floatAttributes.push_back(StringPair("dX", "trailresolution"));
    floatAttributes.push_back(StringPair("voxelSize", "voxelsize"));
    floatAttributes.push_back(StringPair("maskWidth", "boundinglimit"));
    floatAttributes.push_back(StringPair("bandWidth", "halfbandvoxels"));
    floatAttributes.push_back(StringPair("bandWidthWS", "halfband"));

    for (size_t n = 0, N = floatAttributes.size(); n < N; ++n) {

        const char* oldName = floatAttributes[n].first.c_str();
        const char* newName = floatAttributes[n].second.c_str();

        parm = obsoleteParms->getParmPtr(oldName);
        if (parm && !parm->isFactoryDefault()) {
            setFloat(newName, 0, time, obsoleteParms->evalFloat(oldName, 0, time));
        }
    }


    std::vector<StringPair> intAttributes;
    intAttributes.push_back(StringPair("levelSet", "distancevdb"));
    intAttributes.push_back(StringPair("fogVolume", "fogvdb"));
    intAttributes.push_back(StringPair("maskVolume", "maskvdb"));
    intAttributes.push_back(StringPair("writeintoref", "merge"));
    intAttributes.push_back(StringPair("worldSpaceUnits", "useworldspace"));

    for (size_t n = 0, N = intAttributes.size(); n < N; ++n) {

        const char* oldName = intAttributes[n].first.c_str();
        const char* newName = intAttributes[n].second.c_str();

        parm = obsoleteParms->getParmPtr(oldName);
        if (parm && !parm->isFactoryDefault()) {
            setInt(newName, 0, time, obsoleteParms->evalInt(oldName, 0, time));
        }
    }


    std::vector<StringPair> strAttributes;
    strAttributes.push_back(StringPair("gridName", "distancevdbname"));
    strAttributes.push_back(StringPair("fogVolumeGridName", "fogvdbname"));
    strAttributes.push_back(StringPair("maskVolumeGridName", "maskvdbname"));
    strAttributes.push_back(StringPair("group", "referencevdb"));

    for (size_t n = 0, N = strAttributes.size(); n < N; ++n) {

        const char* oldName = strAttributes[n].first.c_str();
        const char* newName = strAttributes[n].second.c_str();

        parm = obsoleteParms->getParmPtr(oldName);
        if (parm && !parm->isFactoryDefault()) {
            UT_String str;
            obsoleteParms->evalString(str, oldName, 0, time);
            if (str.length() > 0) {
                setString(str, CH_STRING_LITERAL, newName, 0, time);
            }
        }
    }

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_From_Particles::updateParmsFlags()
{
    bool changed = false;

    changed |= enableParm("distancevdbname", evalInt("distancevdb", 0, 0));
    changed |= enableParm("fogvdbname", evalInt("fogvdb", 0, 0));

    bool useMask = evalInt("maskvdb", 0, 0) == 1;
    changed |= enableParm("boundinglimit", useMask);
    changed |= enableParm("maskvdbname", useMask);

    bool hasRefInput = this->nInputs() == 2;
    changed |= enableParm("referencevdb", hasRefInput);
    changed |= enableParm("merge", hasRefInput);
    changed |= enableParm("voxelsize", !hasRefInput);

    bool useWSUnits = bool(evalInt("useworldspace", 0, 0));
    changed |= setVisibleState("halfbandvoxels", !useWSUnits);
    changed |= setVisibleState("halfband", useWSUnits);

    bool useTrails =  evalInt("velocitytrails", 0, 0) == 1;
    changed |= enableParm("trailresolution", useTrails);
    changed |= enableParm("velocityscale", useTrails);

    // enable / disable vector type menu
    UT_String attrName;
    GA_ROAttributeRef attrRef;
    const GU_Detail* ptGeo = this->getInputLastGeo(0, CHgetEvalTime());
    if (ptGeo) {
        for (int i = 1, N = evalInt("attrList", 0, 0); i <= N; ++i) {

            evalStringInst("attribute#", &i, attrName, 0, 0);
            bool isVector = false;

            if (attrName.length() != 0) {

                attrRef = ptGeo->findPointAttribute(attrName);
                if (attrRef.isValid()) {

                    const GA_Attribute *attr = attrRef.getAttribute();
                    if (attr) {
                        const GA_TypeInfo typeInfo = attr->getTypeInfo();
                        isVector = (typeInfo == GA_TYPE_HPOINT || typeInfo == GA_TYPE_POINT
                            || typeInfo == GA_TYPE_VECTOR || typeInfo == GA_TYPE_NORMAL);

                        if (!isVector) {
                            const GA_AIFTuple *tupleAIF = attr->getAIFTuple();
                            if (tupleAIF) isVector = tupleAIF->getTupleSize(attr) == 3;
                        }
                    }
                }
            }
            changed |= enableParmInst("vecType#", &i, isVector);
            changed |= setVisibleStateInst("vecType#", &i, isVector);
        }
    }

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_From_Particles::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();

        hvdb::Interrupter boss("From Particles");


        const GU_Detail* ptGeo = inputGeo(0, context);
        const GU_Detail* refGeo = inputGeo(1, context);
        bool refexists = refGeo != NULL;

        mTime = context.getTime();
        mVoxelSize = float(evalFloat("voxelsize", 0, mTime));

        if (mVoxelSize < 1e-5) {
            std::ostringstream ostr;
            ostr << "The voxel size ("<< mVoxelSize << ") is too small.";
            addError(SOP_MESSAGE, ostr.str().c_str());
            return error();
        }

        const bool outputLevelSetGrid   = bool(evalInt("distancevdb",   0, mTime));
        const bool outputFogVolumeGrid  = bool(evalInt("fogvdb",  0, mTime));
        const bool outputMaskVolumeGrid = bool(evalInt("maskvdb", 0, mTime));
        const bool outputAttributeGrid  = bool(evalInt("attrList",   0, mTime) > 0);

        if (!outputFogVolumeGrid && !outputLevelSetGrid && !outputAttributeGrid) {
             addWarning(SOP_MESSAGE, "No output selected");
             return error();
        }

        ParticleList paList(ptGeo,
            evalFloat("particlescale", 0, mTime), evalFloat("velocityscale", 0, mTime));

        float background = 0.0;

        if (evalInt("useworldspace", 0, mTime) != 0) {
            background = float(evalFloat("halfband", 0, mTime));
        } else {
            background = mVoxelSize * float(evalFloat("halfbandvoxels", 0, mTime));
        }

        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(mVoxelSize);

        openvdb::FloatGrid::Ptr outputGrid;

        // Optionally copy reference grid and/or transform.
        if (refexists) {

            UT_String groupStr;
            evalString(groupStr, "referencevdb", 0, mTime);

            const GA_PrimitiveGroup *group =
                matchGroup(const_cast<GU_Detail&>(*refGeo), groupStr.toStdString());

            hvdb::VdbPrimCIterator it(refGeo, group);
            const hvdb::GU_PrimVDB* refPrim = *it;

            if (refPrim) {

                transform = refPrim->getGrid().transform().copy();
                mVoxelSize = static_cast<float>(transform->voxelSize()[0]);

                bool isLevelSet = refPrim->getGrid().getGridClass() == openvdb::GRID_LEVEL_SET;

                // match the narrow band width
                if (isLevelSet && refPrim->getGrid().type() == openvdb::FloatGrid::gridType()) {
                    background =
                        openvdb::gridConstPtrCast<openvdb::FloatGrid>(refPrim->getGridPtr())->background();
                    addMessage(SOP_MESSAGE, "Note: Matching reference level set half-band width "
                        " and background value.  (UI half-band parameter is ignored.)");
                }

                if (evalInt("merge", 0, mTime) != 0) {

                    if (isLevelSet) {
                        outputGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(
                            refPrim->getGrid().deepCopyGrid());

                        if (!outputGrid) {
                            addWarning(SOP_MESSAGE, "Cannot write into the selected"
                                       " reference grid because it is not a float grid.");
                        }

                    } else {
                        addWarning(SOP_MESSAGE, "Can only write directly into a level set grid.");
                    }
                }

            } else {
                addError(SOP_MESSAGE, "Second input has no VDB primitives.");
                return error();
            }
        }


        // Perform the particle conversion.
        if (!boss.wasInterrupted()) {

            if (!outputGrid) {
                outputGrid = openvdb::FloatGrid::create(background);
            }

            outputGrid->setGridClass(openvdb::GRID_LEVEL_SET);
            outputGrid->setTransform(transform);

            const bool doSphereConversion = evalInt("conversion",  0, mTime) == 0;

            // Point topology conversion settings
            int dilation = evalInt("dilation", 0, mTime);
            int closing = evalInt("closing", 0, mTime);
            int smoothing = evalInt("smoothing", 0, mTime);
            int bandWidth = int(std::ceil(background / mVoxelSize));
            openvdb::MaskGrid::Ptr pointMaskGrid;

            if (doSphereConversion) {

                if (evalInt("velocitytrails", 0, mTime) != 0 && !paList.hasVelocity()) {
                    addWarning(SOP_MESSAGE, "Velocity trails require a velocity point attribute named 'v' of 3fv type.");
                }

                if (outputAttributeGrid) {
                    this->convertWithAttributes(outputGrid, paList, *ptGeo, boss);
                } else {
                    this->convert(outputGrid, paList, boss);
                }

            } else {

                pointMaskGrid = GUvdbCreatePointMaskGrid(*transform, *ptGeo);

                openvdb::FloatGrid::Ptr sdfGrid = openvdb::tools::topologyToLevelSet(
                    *pointMaskGrid, bandWidth, closing, dilation, smoothing, &boss);

                openvdb::tools::csgUnion(*outputGrid, *sdfGrid);
            }

            if (outputMaskVolumeGrid) {

                openvdb::Real radiusScale = paList.radiusMult();
                openvdb::Real offset = openvdb::Real(evalFloat("boundinglimit", 0,  mTime));
                offset = std::min(std::max(offset, 0.0), 1.0); // clamp to zero-one range.

                openvdb::FloatGrid::Ptr maxGrid = openvdb::FloatGrid::create(background);
                maxGrid->setGridClass(openvdb::GRID_LEVEL_SET);
                maxGrid->setTransform(transform->copy());

                openvdb::FloatGrid::Ptr minGrid = openvdb::FloatGrid::create(background);
                minGrid->setGridClass(openvdb::GRID_LEVEL_SET);
                minGrid->setTransform(transform->copy());

                if (offset > 0.0f) {

                    if (doSphereConversion) {

                        paList.radiusMult() = radiusScale * (1.0 + offset);
                        this->convert(maxGrid, paList, boss);

                        paList.radiusMult() = radiusScale * (1.0 - offset);
                        this->convert(minGrid, paList, boss);

                    } else {

                        if (!pointMaskGrid) {
                            pointMaskGrid = GUvdbCreatePointMaskGrid(*transform, *ptGeo);
                        }

                        openvdb::Real dx = openvdb::Real(std::min(dilation, 1));
                        int increase = int(std::ceil(dx * (1.0 + offset)));
                        int decrease = int(dx * (1.0 - offset));

                        maxGrid = openvdb::tools::topologyToLevelSet(
                            *pointMaskGrid, bandWidth, closing, increase, smoothing, &boss);

                        minGrid = openvdb::tools::topologyToLevelSet(
                            *pointMaskGrid, bandWidth, closing, decrease, smoothing, &boss);
                    }
                }

                openvdb::tools::csgDifference(*maxGrid, *minGrid);
                openvdb::tools::sdfToFogVolume(*maxGrid);

                UT_String name = "";
                evalString(name, "maskvdbname", 0, mTime);
                maxGrid->setName(name.toStdString());
                hvdb::createVdbPrimitive(*gdp, maxGrid);
            }

            if (outputLevelSetGrid) {
                UT_String name = "";
                evalString(name, "distancevdbname", 0, mTime);
                outputGrid->setName(name.toStdString());
                hvdb::createVdbPrimitive(*gdp, outputGrid);
            }

            if (outputFogVolumeGrid) {

                // Only duplicate the output grid if both distance
                // and fog volume grids are exported.
                if (outputLevelSetGrid) {
                    outputGrid = outputGrid->deepCopy();
                }

                openvdb::tools::sdfToFogVolume(*outputGrid);

                UT_String name = "";
                evalString(name, "fogvdbname", 0, mTime);
                outputGrid->setName(name.toStdString());
                hvdb::createVdbPrimitive(*gdp, outputGrid);
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}


////////////////////////////////////////

void
SOP_OpenVDB_From_Particles::convert(
    openvdb::FloatGrid::Ptr outputGrid, ParticleList& paList, hvdb::Interrupter& boss)
{
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, void, hvdb::Interrupter>
        raster(*outputGrid, &boss);

    raster.setRmin(evalFloat("minradius", 0,  mTime));
    raster.setRmax(1e15f);

    bool velocityTrails = evalInt("velocitytrails", 0, mTime) == 1;

    if (velocityTrails && paList.hasVelocity()) {
        raster.rasterizeTrails(paList, float(evalFloat("trailresolution", 0,  mTime)));
    } else if (paList.hasRadius()){
        raster.rasterizeSpheres(paList);
    } else {
        raster.rasterizeSpheres(paList, paList.radiusMult());
    }

    // always prune to produce a valid narrow-band level set.
    raster.finalize(/*prune=*/true);

    if (raster.ignoredParticles()) {
        std::ostringstream ostr;
        ostr << "Ignored " << raster.getMinCount() << " small and " << raster.getMaxCount()
             << " large particles (hint: change Minimum Radius in Voxels)";
        addWarning(SOP_MESSAGE, ostr.str().c_str());
    }
}


void
SOP_OpenVDB_From_Particles::convertWithAttributes(
    openvdb::FloatGrid::Ptr outputGrid, ParticleList& paList, const GU_Detail& ptGeo, hvdb::Interrupter& boss)
{
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Int32, hvdb::Interrupter>
        raster(*outputGrid, &boss);

    raster.setRmin(evalFloat("minradius", 0,  mTime));
    raster.setRmax(1e15f);

    bool velocityTrails = evalInt("velocitytrails", 0, mTime) == 1;

    if (velocityTrails && paList.hasVelocity()) {
        raster.rasterizeTrails(paList, float(evalFloat("trailresolution", 0,  mTime)));
    } else if (paList.hasRadius()){
        raster.rasterizeSpheres(paList);
    } else {
        raster.rasterizeSpheres(paList, paList.radiusMult());
    }

    // always prune to produce a valid narrow-band level set.
    raster.finalize(/*prune=*/true);

    openvdb::Int32Grid::Ptr closestPtnIdxGrid = raster.attributeGrid();

    if (raster.ignoredParticles()) {
        std::ostringstream ostr;
        ostr << "Ignored " << raster.getMinCount() << " small and " << raster.getMaxCount()
             << " large particles (hint: change Minimum Radius in Voxels)";
        addWarning(SOP_MESSAGE, ostr.str().c_str());
    }

    if (boss.wasInterrupted()) return;

    // Transfer point attributes.
    if (evalInt("attrList", 0, mTime) > 0) {
        hvdb::AttributeDetailList pointAttributes;

        int closestPointIndexInstance =
            constructGenericAtttributeList(pointAttributes, ptGeo, *closestPtnIdxGrid);

        openvdb::math::Transform::Ptr transform = outputGrid->transformPtr();
        transferAttributes(pointAttributes, *closestPtnIdxGrid, transform, ptGeo);

        if (closestPointIndexInstance > -1) { // Export the closest point idx grid.
            UT_String gridNameStr;
            evalStringInst("attributeGridName#", &closestPointIndexInstance, gridNameStr, 0, mTime);
            if (gridNameStr.length() == 0) gridNameStr = "point_list_index";
            hvdb::createVdbPrimitive(*gdp, closestPtnIdxGrid, gridNameStr.toStdString().c_str());
        }
    }
}


// Helper methods for point attribute transfer

int
SOP_OpenVDB_From_Particles::constructGenericAtttributeList(
    hvdb::AttributeDetailList &pointAttributes,
    const GU_Detail& ptGeo,
    const openvdb::Int32Grid& closestPtnIdxGrid)
{
    UT_String attrName;
    GA_ROAttributeRef attrRef;
    GA_Range range;
    int closestPointIndexInstance = -1;

    // for each selected attribute
    for (int i = 1, N = evalInt("attrList", 0, mTime); i <= N; ++i) {

        evalStringInst("attribute#", &i, attrName, 0, mTime);

        if (attrName.length() == 0) continue;

        if (attrName == "point_list_index") {
            // The closest point index grid is a special case,
            // the converter has already generated it for us.
            closestPointIndexInstance = i;
            continue;
        }

        attrRef = ptGeo.findPointAttribute(attrName);

        if (!attrRef.isValid()) {
            std::ostringstream ostr;
            ostr << "Skipped unrecognized attribute: '"<< attrName << "'";
            addWarning(SOP_MESSAGE, ostr.str().c_str());
            continue;
        }

        evalStringInst("attributeGridName#", &i, attrName, 0, mTime);
        std::string customName = attrName.toStdString();

        int vecType = evalIntInst("vecType#", &i, 0, mTime);

        const GA_Attribute *attr = attrRef.getAttribute();
        if (!attr) {
            addWarning(SOP_MESSAGE, "Skipped unrecognized attribute type");
            continue;
        }

        const GA_AIFTuple *tupleAIF = attr->getAIFTuple();
        if (!tupleAIF) {
            addWarning(SOP_MESSAGE, "Skipped unrecognized attribute type");
            continue;
        }

        const GA_Storage attrStorage = tupleAIF->getStorage(attr);
        const int attrTupleSize = tupleAIF->getTupleSize(attr);
        const GA_TypeInfo typeInfo = attr->getTypeInfo();

        const bool interpertAsVector = (typeInfo == GA_TYPE_HPOINT || typeInfo == GA_TYPE_POINT
            || typeInfo == GA_TYPE_VECTOR || typeInfo == GA_TYPE_NORMAL);

        switch (attrStorage)
        {
            case GA_STORE_INT16:
            case GA_STORE_INT32:
                if (interpertAsVector || attrTupleSize == 3) {
                    addAttributeDetails<openvdb::Vec3i>(pointAttributes, attr, tupleAIF,
                        attrTupleSize, closestPtnIdxGrid, customName, vecType);
                } else {
                    addAttributeDetails<openvdb::Int32>(pointAttributes, attr, tupleAIF,
                        attrTupleSize, closestPtnIdxGrid, customName);
                }

                break;
            case GA_STORE_INT64:
                addAttributeDetails<openvdb::Int64>(pointAttributes, attr, tupleAIF,
                    attrTupleSize, closestPtnIdxGrid, customName);
                break;
            case GA_STORE_REAL16:
            case GA_STORE_REAL32:
                if (interpertAsVector || attrTupleSize == 3) {
                    addAttributeDetails<openvdb::Vec3s>(pointAttributes, attr, tupleAIF,
                        attrTupleSize, closestPtnIdxGrid, customName, vecType);
                } else {
                    addAttributeDetails<float>(pointAttributes, attr, tupleAIF,
                        attrTupleSize, closestPtnIdxGrid, customName);
                }

                break;
            case GA_STORE_REAL64:
                if (interpertAsVector || attrTupleSize == 3) {
                    addAttributeDetails<openvdb::Vec3d>(pointAttributes, attr, tupleAIF,
                        attrTupleSize, closestPtnIdxGrid, customName, vecType);
                } else {
                    addAttributeDetails<double>(pointAttributes, attr, tupleAIF,
                        attrTupleSize, closestPtnIdxGrid, customName);
                }

                break;
            default:
                addWarning(SOP_MESSAGE, "Skipped unrecognized attribute type");
                break;
        }
    }
    return closestPointIndexInstance;
}


template <class ValueType>
void
SOP_OpenVDB_From_Particles::addAttributeDetails(
    hvdb::AttributeDetailList &attributeList,
    const GA_Attribute *attribute,
    const GA_AIFTuple *tupleAIF,
    const int attrTupleSize,
    const openvdb::Int32Grid& closestPtnIdxGrid,
    std::string& customName,
    int vecType)
{

    // Defines a new type of a tree having the same hierarchy as the incoming
    // Int32Grid's tree but potentially a different value type.
    typedef typename openvdb::Int32Grid::TreeType::ValueConverter<ValueType>::Type TreeType;
    typedef typename openvdb::Grid<TreeType> GridType;


    if (vecType != -1) { // Vector grid
         // Get the attribute's default value.
         ValueType defValue =
             hvdb::evalAttrDefault<ValueType>(tupleAIF->getDefaults(attribute), 0);

        // Construct a new tree that matches the closestPtnIdxGrid's active voxel topology.
        typename TreeType::Ptr tree(
            new TreeType(closestPtnIdxGrid.tree(), defValue, openvdb::TopologyCopy()));
        typename GridType::Ptr grid(GridType::create(tree));

        grid->setVectorType(openvdb::VecType(vecType));

        attributeList.push_back(hvdb::AttributeDetailBase::Ptr(
            new hvdb::AttributeDetail<GridType>(grid, attribute, tupleAIF, 0, true)));

        if (customName.size() > 0) {
            attributeList[attributeList.size()-1]->name() = customName;
        }

    } else {
        for (int c = 0; c < attrTupleSize; ++c) {
            // Get the attribute's default value.
            ValueType defValue =
                hvdb::evalAttrDefault<ValueType>(tupleAIF->getDefaults(attribute), c);

            // Construct a new tree that matches the closestPtnIdxGrid's active voxel topology.
            typename TreeType::Ptr tree(
                new TreeType(closestPtnIdxGrid.tree(), defValue, openvdb::TopologyCopy()));
            typename GridType::Ptr grid(GridType::create(tree));

            attributeList.push_back(hvdb::AttributeDetailBase::Ptr(
                new hvdb::AttributeDetail<GridType>(grid, attribute, tupleAIF, c)));

            if (customName.size() > 0) {
                std::ostringstream name;
                name << customName;
                if(attrTupleSize != 1) name << "_" << c;

                attributeList[attributeList.size()-1]->name() = name.str();
            }
        }
    }
}


void
SOP_OpenVDB_From_Particles::transferAttributes(
    hvdb::AttributeDetailList &pointAttributes,
    const openvdb::Int32Grid& closestPtnIdxGrid,
    openvdb::math::Transform::Ptr& transform,
    const GU_Detail& ptGeo)
{
    // Threaded attribute transfer.
    hvdb::PointAttrTransfer transferOp(pointAttributes, closestPtnIdxGrid, ptGeo);
    transferOp.runParallel();

    // Construct and add VDB primitives to the gdp
    for (size_t i = 0, N = pointAttributes.size(); i < N; ++i) {
        hvdb::AttributeDetailBase::Ptr& attrDetail = pointAttributes[i];
        std::ostringstream gridName;
        gridName << attrDetail->name();
        attrDetail->grid()->setTransform(transform);
        hvdb::createVdbPrimitive(*gdp, attrDetail->grid(), gridName.str().c_str());
    }
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )

