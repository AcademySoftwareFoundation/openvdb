///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @brief Converts particles (w. radius and velocity) to narrow-band
///        level sets stored in an OpenVDB grid.
/// @note The width of the narrow band level set is defined from the
///       background value and voxel size of the input grid:
///       voxel width = 2*background/dx

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/AttributeTransferUtil.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <CH/CH_Manager.h>
#include <GA/GA_Types.h> // for GA_ATTRIB_POINT


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


namespace {

// This wrapper class is required by openvdb::tools::ParticlesToLeveSet
class ParticleList
{
public:
    // required typedef for bucketing
    typedef openvdb::Vec3R    value_type;

    ParticleList(const GU_Detail* gdp,
                 openvdb::Real radiusMult = 1,
                 openvdb::Real velocityMult = 1) :
        mGdp(gdp),
        mScaleHandle(gdp, GA_ATTRIB_POINT, "pscale"),
        mVelHandle(gdp, GA_ATTRIB_POINT, "v"),
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


// Convenient settings struct
struct Settings
{
    Settings(): mRasterizeTrails(false), mDx(1.0) {}
    bool mRasterizeTrails;
    float mDx;
};


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

    void convert(openvdb::FloatGrid::Ptr, ParticleList&, const Settings&);
    void convertWithAttributes(openvdb::FloatGrid::Ptr, ParticleList&,
        const Settings&, const GU_Detail&);

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

    float mVoxelSize, mTime;
    hvdb::Interrupter mBoss;
};


////////////////////////////////////////


namespace {
// Callback to convert from voxel to world space units
int
convertUnitsCB(void* data, int /*idx*/, float /*time*/, const PRM_Template*)
{
    SOP_OpenVDB_From_Particles* sop = static_cast<SOP_OpenVDB_From_Particles*>(data);
    if (sop == NULL) return 0;
    return sop->convertUnits();
}
}


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    //////////
    // Output grids

    // Level set
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "levelSet", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Enable / disable the level set output."));

    parms.add(hutil::ParmFactory(PRM_STRING, "gridName", "Distance VDB")
        .setDefault("surface")
        .setHelpText("Output the level set grid.\nVoxels in the "
            "narrow band are made active. (A grid name can optionally be "
            "specified in the text field.)"));

    // Fog volume
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "fogVolume", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Enable / disable the fog volume output."));

    parms.add(hutil::ParmFactory(PRM_STRING, "fogVolumeGridName", "Fog VDB")
        .setDefault("density")
        .setHelpText("Output the fog volume grid.\nGenerated from the "
            "level set, the interior narrow band is transformed into a "
            "0 to 1 gradient and the remaining interior values are set "
            "to 1. Exterior values and the background are set to 0. The "
            "interior is still a sparse representation, but "
                     "the values are active."));

    // Mask volume
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "maskVolume", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Enable / disable the mask volume output."));

    parms.add(hutil::ParmFactory(PRM_STRING, "maskVolumeGridName", "Mask VDB")
        .setDefault("mask")
        .setHelpText("Output the mask volume grid.\nGenerate an alpha mask "
              "that is very useful for subsequent constrained level set smoothing "
              "of the level set surface from the particles. This alpha mask is defined "
              "as the fog volume derived from the CSG difference between a level set "
              "surface with a maximum radius of the particles and a level set surface "
              "with a minimum radius of the particles. This mask will guarentee that "
              "subsequent level set smoothing is constrained between the min/max surfaces, "
              " thus avoiding that surface details are completely smoothed away!"));

    //////////
    // Conversion settings

    parms.add(hutil::ParmFactory(PRM_HEADING, "particleHeading", "Conversion settings"));

    // Group name (Transform reference)
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Reference VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setHelpText("References the first/selected grid's transform."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "writeintoref", "Merge With Reference VDB"));


    // Voxel size
    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelSize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5));

    // Narrow-band {
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldSpaceUnits", "Use World Space for Band")
              .setCallbackFunc(&convertUnitsCB));

    // Voxel-space width
    parms.add(hutil::ParmFactory(PRM_FLT_J, "bandWidth", "Half-Band Voxels")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText("Half the width of the narrow band in voxel units. "
                     "For level set operations use a value of 3, or for better "
                     "performance a value of 1 followed by the application "
                     "of the \"Trim Narrow Band\" SOP."));

    // World-space width
    parms.add(hutil::ParmFactory(PRM_FLT_J, "bandWidthWS", "Half-Band")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 10));
    // }

    // dR (radius scale)
    parms.add(hutil::ParmFactory(PRM_FLT_J, "dR", "Particle Radius Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0)
        .setHelpText("Scaling multiplier for the radius. Use this parameter "
                     "to control the size of the particle footprints. Note "
                     "particles smaller than \"Min voxel radius\" (e.g. 1.5 "
                     " which is the Nyquist limit) are NOT rasterized!"));

    // Rmin (minimum radius)
    parms.add(hutil::ParmFactory(PRM_FLT_J, "Rmin", "Minimum Radius in Voxels")
        .setDefault(1.5)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0)
        .setHelpText("Smallest allowed radius in voxels. Use this parameter "
                     "to ignore small particles. Note the default value of 1.5 "
                     "voxels correspond to the Nyquist grid sampling frequency, "
                     "the smallest size the grid can safely represent. Thus, "
                     "values smaller than 1.5 will likely result in aliasing!"));

     // Prune level set
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "Prune Level Set")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Enable / disable the pruning of nodes, i.e. leafs filled with "
                     "the inside values are compactly represented by a tile value. "
                     "This option only has an effect if the particles are larger then "
                     "the leaf nodes so it is normally recommended to leave it disabled."));

    // Width of the mask for constraining subsequent deformations
    parms.add(hutil::ParmFactory(PRM_FLT_J, "maskWidth", "Mask Width Scale")
              .setDefault(0.25)
              .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_RESTRICTED, 1)
              .setHelpText("Use this scaling paramter to adjust the width of "
                           "alpha mask that is useful for subsequent level set"
                           "filtering!"));

    // Particle footprints
    {
        const char* items[] = {
            "sphere", "Spherical",
            "trail",  "Velocity Trail",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "footprint", "Particle Footprint")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setHelpText("Use this parameter to select the "
                "type of particle footprints (spherical or conical)."));
    }

    // dV (velocity scale)
    parms.add(hutil::ParmFactory(PRM_FLT_J, "dV", "Velocity Multiplier")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_UI, 0.0, PRM_RANGE_UI, 2.0)
        .setHelpText("Scaling multiplier for the velocity. Use this parameter to control "
            "the length of velocity trails."));

    // dX (scale of offset between sphere instances)
    parms.add(hutil::ParmFactory(PRM_FLT_J, "dX", "Velocity Spacing")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.2, PRM_RANGE_UI, 2.0)
        .setHelpText("Scale of distance between sphere instances. Use this parameter "
            "to control the aliasing and the number of instances."));

    //////////
    // Point attribute transfer
    parms.add(hutil::ParmFactory(PRM_HEADING, "transferHeading", "Attribute transfer"));

    hutil::ParmList attrParms;

    // Attribute name
    attrParms.add(hutil::ParmFactory(PRM_STRING, "attribute#", "Attribute")
        .setChoiceList(&PrimAttrMenu)
        .setSpareData(&SOP_Node::theFirstInput)
        .setHelpText("Select a point attribute to transfer. "
            "Supports integer and floating point attributes of "
            "arbitrary precisions and tuple sizes."));

    attrParms.add(hutil::ParmFactory(PRM_STRING, "attributeGridName#", "VDB Name")
        .setHelpText("The attribute name is used as the output grid name by default."
            " A different grid name can be specified in this field if desired."));

    // Vec type menu
    {
        std::vector<std::string> items;
        for (int i = 0; i < openvdb::NUM_VEC_TYPES ; ++i) {
            items.push_back(openvdb::GridBase::vecTypeToString(openvdb::VecType(i)));
            items.push_back(openvdb::GridBase::vecTypeExamples(openvdb::VecType(i)));
        }

        attrParms.add(hutil::ParmFactory(PRM_ORD, "vecType#", "Vector Type")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    // Add multi parm
    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "attrList", "Point Attributes")
        .setHelpText("Transfer point attributes to each voxel in the level set's narrow band")
        .setMultiparms(attrParms)
        .setDefault(PRMzeroDefaults));

    //////////
    // Obsolete parameters

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
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "maxWidth", "Max Half-width"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "Rmax", "Max Radius In Voxels"));


    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory("OpenVDB From Particles",
        SOP_OpenVDB_From_Particles::factory, parms, *table)
        .addAlias("OpenVDB Particle Voxelizer")
        .setObsoleteParms(obsoleteParms)
        .addInput("Particles to Convert")
        .addOptionalInput("Optional Reference VDB "
            "(for transform and narrow-band width matching)");
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
    , mTime(0.0f)
    , mBoss("Converting particles to level set")
{
}


////////////////////////////////////////


int
SOP_OpenVDB_From_Particles::convertUnits()
{
    const bool toWSUnits = static_cast<bool>(evalInt("worldSpaceUnits", 0, 0));

    if (toWSUnits) {
        setFloat("bandWidthWS", 0, 0, evalFloat("bandWidth", 0, 0) * mVoxelSize);
        return 1;
    }

    setFloat("bandWidth", 0, 0, evalFloat("bandWidthWS", 0, 0) / mVoxelSize);

    return 1;
}


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_From_Particles::updateParmsFlags()
{
    bool changed = false;
    int refexists = (this->nInputs() == 2);

    changed |= enableParm("group", refexists);
    changed |= enableParm("writeintoref", refexists);

    changed |= enableParm("voxelSize", !refexists);

    const bool wsUnits = bool(evalInt("worldSpaceUnits", 0, 0));

    changed |= setVisibleState("bandWidth", !wsUnits);
    changed |= setVisibleState("bandWidthWS", wsUnits);


    // Particle conversion
    const bool useTrails =  evalInt("footprint", 0, 0) == 1;
    changed |= enableParm("dX", useTrails);
    changed |= enableParm("dV", useTrails);

    changed |= setVisibleState("dX", useTrails);
    changed |= setVisibleState("dV", useTrails);

    changed |= setVisibleState("maskWidth", evalInt("maskVolume", 0, 0) == 1);

    // Output
    changed |= enableParm("gridName", evalInt("levelSet", 0, 0));
    changed |= enableParm("fogVolumeGridName", evalInt("fogVolume", 0, 0));

    bool build_mask = evalInt("maskVolume", 0, 0) != 0;
    changed |= enableParm("maskVolumeGridName", build_mask);
    changed |= enableParm("maskWidth", build_mask);

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

        // Check for particles in the primary (left) input port
        const GU_Detail* ptGeo = inputGeo(0, context);
        const GU_Detail* refGeo = inputGeo(1, context);
        bool refexists = refGeo != NULL;

        // Set member data
        mTime = static_cast<float>(context.getTime());
        mVoxelSize = static_cast<float>(evalFloat("voxelSize", 0, mTime));

        const bool outputLevelSetGrid   = bool(evalInt("levelSet",   0, mTime));
        const bool outputFogVolumeGrid  = bool(evalInt("fogVolume",  0, mTime));
        const bool outputMaskVolumeGrid = bool(evalInt("maskVolume", 0, mTime));
        const bool outputAttributeGrid  = bool(evalInt("attrList",   0, mTime) > 0);

        if (!outputFogVolumeGrid && !outputLevelSetGrid && !outputAttributeGrid) {
             addWarning(SOP_MESSAGE, "No output selected");
             return error();
        }

        ParticleList paList(ptGeo, evalFloat("dR", 0,  mTime), evalFloat("dV", 0,  mTime));

        // Get the conversion settings from the UI parameters.
        Settings settings;
        settings.mRasterizeTrails =  evalInt("footprint", 0, mTime) == 1;
        settings.mDx = static_cast<float>(evalFloat("dX", 0,  mTime));

        const bool wsUnits = bool(evalInt("worldSpaceUnits", 0, mTime));
        float background = 0.0;

        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(mVoxelSize);

        openvdb::FloatGrid::Ptr outputGrid, minGrid, maxGrid;

        // Optionally copy reference grid and/or transform.
        if (refexists) {

            UT_String groupStr;
            evalString(groupStr, "group", 0, mTime);

            const GA_PrimitiveGroup *group =
                matchGroup(const_cast<GU_Detail&>(*refGeo), groupStr.toStdString());

            hvdb::VdbPrimCIterator it(refGeo, group);
            const hvdb::GU_PrimVDB* refPrim = *it;

            if (refPrim) {
                transform = refPrim->getGrid().transform().copy();
                mVoxelSize = static_cast<float>(transform->voxelSize()[0]);

                if (bool(evalInt("writeintoref", 0, mTime))) {
                    if (refPrim->getGrid().getGridClass() == openvdb::GRID_LEVEL_SET) {
                        outputGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(
                            refPrim->getGrid().deepCopyGrid());

                        if (!outputGrid) {
                            addWarning(SOP_MESSAGE, "Cannot write into the selected"
                                       " reference grid because it is not a float grid.");
                        }

                        if (outputMaskVolumeGrid) {
                            minGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(
                                refPrim->getGrid().deepCopyGrid());
                            maxGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(
                                refPrim->getGrid().deepCopyGrid());
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

        // Set narrow band half-width
        if (wsUnits) background = static_cast<float>(evalFloat("bandWidthWS", 0, mTime));
        else background = static_cast<float>(mVoxelSize * evalFloat("bandWidth", 0, mTime));


        // Perform the particle conversion.
        if (mBoss.wasInterrupted()) {
                addWarning(SOP_MESSAGE, "Process was interrupted");
        } else {

            // Construct a new level set grid.
            if (!outputGrid) outputGrid = openvdb::FloatGrid::create(background);

            outputGrid->setGridClass(openvdb::GRID_LEVEL_SET);
            outputGrid->setTransform(transform);

            if (outputAttributeGrid) {
                // Converts and outputs point attribute grids
                this->convertWithAttributes(outputGrid, paList, settings, *ptGeo);
            } else {
                this->convert(outputGrid, paList, settings);
            }

            if (outputMaskVolumeGrid) {
                const float maskWidth = static_cast<float>(evalFloat("maskWidth", 0,  mTime));
                // Max grid
                if (!maxGrid) maxGrid = openvdb::FloatGrid::create(background);
                if (maskWidth > 0.0f) {
                    maxGrid->setGridClass(openvdb::GRID_LEVEL_SET);
                    maxGrid->setTransform(transform->copy());
                    paList.radiusMult() *= (1.0f + maskWidth);
                    this->convert(maxGrid, paList, settings);

                    // Min grid
                    if ( maskWidth < 1.0f) {
                        if (!minGrid) minGrid = openvdb::FloatGrid::create(background);
                        minGrid->setGridClass(openvdb::GRID_LEVEL_SET);
                        minGrid->setTransform(transform->copy());
                        paList.radiusMult() *= (1.0f  - maskWidth)/(1.0f + maskWidth);
                        this->convert(minGrid, paList, settings);

                        // CSG difference
                        openvdb::tools::csgDifference(*maxGrid, *minGrid);
                    }
                }

                // Convert to fog volume
                openvdb::tools::sdfToFogVolume(*maxGrid);

                // Add fog volume to gdp
                UT_String gridNameStr = "";
                evalString(gridNameStr, "maskVolumeGridName", 0, mTime);
                outputGrid->setName(gridNameStr.toStdString());
                hvdb::createVdbPrimitive(*gdp, maxGrid, gridNameStr.toStdString().c_str());
            }// masked volume grid

            // Output level set grid
            if ( outputLevelSetGrid) {
                UT_String gridNameStr = "";
                evalString(gridNameStr, "gridName", 0, mTime);
                outputGrid->setName(gridNameStr.toStdString());
                hvdb::createVdbPrimitive(*gdp, outputGrid, gridNameStr.toStdString().c_str());
            }

            // Output fog volume grid
            if ( outputFogVolumeGrid) {

                // If no level set grid is exported the original level set
                // grid is modified in place.

                if (outputLevelSetGrid) outputGrid = outputGrid->deepCopy();

                openvdb::tools::sdfToFogVolume(*outputGrid);

                UT_String gridNameStr = "";
                evalString(gridNameStr, "fogVolumeGridName", 0, mTime);
                outputGrid->setName(gridNameStr.toStdString());
                hvdb::createVdbPrimitive(*gdp, outputGrid, gridNameStr.toStdString().c_str());
            }
        }
        mBoss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}


////////////////////////////////////////

void
SOP_OpenVDB_From_Particles::convert(openvdb::FloatGrid::Ptr outputGrid,
                                    ParticleList& paList,
                                    const Settings& settings)
{
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, void, hvdb::Interrupter>
        raster(*outputGrid, &mBoss);

    raster.setRmin(evalFloat("Rmin", 0,  mTime));
    raster.setRmax(1e15f);

    if (settings.mRasterizeTrails && paList.hasVelocity()) {
        raster.rasterizeTrails(paList, settings.mDx);
    } else if (paList.hasRadius()){
        raster.rasterizeSpheres(paList);
    } else {
        raster.rasterizeSpheres(paList, paList.radiusMult());
    }

    raster.finalize(evalInt("prune", 0, 0));

    if (raster.ignoredParticles()) {
        std::ostringstream ostr;
        ostr << "Ignored " << raster.getMinCount() << " small and " << raster.getMaxCount()
             << " large particles (hint: change Minimum Radius in Voxels)";
        addWarning(SOP_MESSAGE, ostr.str().c_str());
    }
}


void
SOP_OpenVDB_From_Particles::convertWithAttributes(openvdb::FloatGrid::Ptr outputGrid,
                                                  ParticleList& paList,
                                                  const Settings& settings,
                                                  const GU_Detail& ptGeo)
{
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Int32, hvdb::Interrupter>
        raster(*outputGrid, &mBoss);

    raster.setRmin(evalFloat("Rmin", 0,  mTime));
    raster.setRmax(1e15f);

    if (settings.mRasterizeTrails && paList.hasVelocity()) {
        raster.rasterizeTrails(paList, settings.mDx);
    } else if (paList.hasRadius()){
        raster.rasterizeSpheres(paList);
    } else {
        raster.rasterizeSpheres(paList, paList.radiusMult());
    }

    raster.finalize(evalInt("prune", 0, 0));

    openvdb::Int32Grid::Ptr closestPtnIdxGrid = raster.attributeGrid();

    if (raster.ignoredParticles()) {
        std::ostringstream ostr;
        ostr << "Ignored " << raster.getMinCount() << " small and " << raster.getMaxCount()
             << " large particles (hint: change Minimum Radius in Voxels)";
        addWarning(SOP_MESSAGE, ostr.str().c_str());
    }

    if (mBoss.wasInterrupted()) return;

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


////////////////////////////////////////

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


////////////////////////////////////////


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


////////////////////////////////////////


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

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
