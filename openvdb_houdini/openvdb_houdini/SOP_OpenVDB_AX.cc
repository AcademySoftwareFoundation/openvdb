// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file SOP_OpenVDB_AX.cc
///
/// @authors  Nick Avramoussis, Richard Jones, Francisco Gochez, Matt Warner
///
/// @brief AX SOP for OpenVDB Points and Volumes
///

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif

#include "AXUtils.h"

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/compiler/Compiler.h>
#include <openvdb_ax/compiler/Logger.h>
#include <openvdb_ax/compiler/CustomData.h>
#include <openvdb_ax/compiler/PointExecutable.h>
#include <openvdb_ax/compiler/VolumeExecutable.h>

#include <houdini_utils/ParmFactory.h>
#include <houdini_utils/geometry.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/PointUtils.h>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointDelete.h>
#include <openvdb/points/IndexIterator.h>

#include <CH/CH_Channel.h>
#include <CH/CH_Manager.h>
#include <CH/CH_LocalVariable.h>
#include <CMD/CMD_Manager.h>
#include <CMD/CMD_Variable.h>
#include <OP/OP_CommandManager.h>
#include <OP/OP_Director.h>
#include <OP/OP_Expression.h>
#include <OP/OP_Channels.h>
#include <PRM/PRM_Parm.h>
#include <UT/UT_Ramp.h>
#include <UT/UT_Version.h>

#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

namespace hvdb = openvdb_houdini;
namespace hax =  openvdb_ax_houdini;
namespace hutil = houdini_utils;

using namespace openvdb;

struct CompilerCache
{
    ax::Compiler::Ptr mCompiler = nullptr;
    ax::Logger::Ptr mLogger = nullptr;
    ax::ast::Tree::Ptr mSyntaxTree = nullptr;
    ax::CustomData::Ptr mCustomData = nullptr;
    ax::PointExecutable::Ptr mPointExecutable = nullptr;
    ax::VolumeExecutable::Ptr mVolumeExecutable = nullptr;
    ax::AttributeRegistry::Ptr mAttributeRegistry = nullptr;

    // point variables

    bool mRequiresDeletion = false;
};

/// @brief  A cached set of parameters, usually evaluated from the Houdini
///         UI which, on change, requires a re-compilation of AX. Does not
///         include the code snippet itself as this is handled separately.
/// @note   Should generally not be used to query the current state of the
///         parameters on the UI as this is only guaranteed to be up to date
///         on successfully compilations.
struct ParameterCache
{
    inline bool operator==(const ParameterCache& other) const
    {
        return mHScriptSupport == other.mHScriptSupport &&
               mVEXSupport == other.mVEXSupport &&
               mTargetType == other.mTargetType;
    }

    inline bool operator!=(const ParameterCache& other) const
    {
        return !(other == *this);
    }

    bool mHScriptSupport = true;
    bool mVEXSupport = true;
    hax::TargetType mTargetType = hax::TargetType::LOCAL;
};

/// @brief  Initialize the compiler function registry with a list of
///         available function calls. Optionally include houdini VEX hooks.
///
/// @param  compiler  The compiler object to set the function registry on
/// @param  allowVex  Whether to include support for available houdini functions
void initializeFunctionRegistry(ax::Compiler& compiler, const bool allowVex)
{
    ax::codegen::FunctionRegistry::UniquePtr functionRegistry =
        ax::codegen::createDefaultRegistry();

    if (allowVex) {
        hax::registerCustomHoudiniFunctions(*functionRegistry);
    }

    compiler.setFunctionRegistry(std::move(functionRegistry));
}

void checkAttributesAgainstList(const std::string& list,
                                const std::vector<UT_String>& newAttributes)
{
    if (newAttributes.empty()) return;

    UT_String msg;
    // attributes are in reverse order as they appear in snippet
    for (auto iter = newAttributes.rbegin(); iter != newAttributes.rend(); ++iter) {
        if (!iter->multiMatch(list.c_str())) {
            msg += " @";
            msg += *iter;
        }
    }

    if (msg.length() != 0) {
        msg.prepend("The following attributes are missing and do not appear in the \"Attributes To Create\" list:");
        throw std::runtime_error(msg.c_str());
    }
}

/// @brief  If the provided logger has warnings or errors, add an explicit
///   warning or error message to the SOP. Returns true if ERRORS existed.
bool appendLoggerMessage(SOP_VDBCacheOptions& sop,
    const ax::Logger& logger,
    const char* type)
{
    const size_t errs = logger.errors();
    const size_t warns = logger.warnings();

    if (warns) {
        std::stringstream os;
        const bool multi = warns > 1;
        if (multi) os << warns << ' ';
        os <<"AX " << type << " warning";
        if (multi) os << 's';
        os <<"!\n";
        sop.addWarning(SOP_MESSAGE, os.str().c_str());
    }

    if (errs) {
        std::stringstream os;
        const bool multi = errs > 1;
        if (multi) os << errs << ' ';
        os <<"AX " << type << " error";
        if (multi) os << 's';
        os <<"!\n";
        sop.addError(SOP_MESSAGE, os.str().c_str());
    }

    return errs;
}

////////////////////////////////////////


class SOP_OpenVDB_AX: public hvdb::SOP_NodeVDB
{
public:

    SOP_OpenVDB_AX(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_AX() override = default;

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    class Cache : public SOP_VDBCacheOptions
    {
    public:
        Cache();
        ~Cache() override = default;

        OP_ERROR cookVDBSop(OP_Context&) override;
        /// @brief  See SOP_OpenVDB_AX::evaluateExternalExpressions
        void evaluateExternalExpressions(const double time,
                        const hax::ChannelExpressionSet& set,
                        const bool hvars,
                        OP_Node* evaluationNode);
        /// @brief  See SOP_OpenVDB_AX::evalInsertHScriptVariable
        bool evalInsertHScriptVariable(const std::string& name,
                        const std::string& accessedType,
                        ax::CustomData& data);

    private:
        unsigned mHash;

        ParameterCache mParameterCache;
        CompilerCache mCompilerCache;

        // The current set of channel and $ expressions.

        hax::ChannelExpressionSet mChExpressionSet;
        hax::ChannelExpressionSet mDollarExpressionSet;
    };

protected:
    void resolveObsoleteParms(PRM_ParmList*) override;
    bool updateParmsFlags() override;
    void syncNodeVersion(const char*, const char*, bool*) override;
}; // class SOP_OpenVDB_AX


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "vdbgroup", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenu));

    {
        const char* items[] = {
            "points",   "Points",
            "volumes",  "Volumes",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "runover", "Run Over")
            .setDefault("points")
            .setHelpText("Whether to run this snippet over OpenVDB Points or OpenVDB Volumes.")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

#ifdef DNEG_OPENVDB_AX
    {
        const char* items[] = {
            "active",    "Active Voxels",
            "inactive",  "Inactive Voxels",
            "all",       "All Voxels",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "activity", "")
            .setDefault("active")
            .setHelpText("Whether to run this snippet over Active, Inactive or All voxels.")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
#endif

    parms.add(hutil::ParmFactory(PRM_STRING, "vdbpointsgroup", "VDB Points Group")
        .setHelpText("Specify a point group name to perform the execution on. If no name is "
                     "given, the AX snippet is applied to all points.")
        .setChoiceList(&hvdb::VDBPointsGroupMenuInput1));

    parms.beginSwitcher("tabMenu1");
    parms.addFolder("Code");

    static PRM_SpareData theEditor(PRM_SpareArgs()
            << PRM_SpareToken(PRM_SpareData::getEditorToken(), "1")
            << PRM_SpareToken(PRM_SpareData::getEditorLanguageToken(), "ax")
            << PRM_SpareToken(PRM_SpareData::getEditorLinesRangeToken(), "8-40")
    );

    parms.add(hutil::ParmFactory(PRM_STRING, "snippet", "AX Expression")
        .setHelpText("A snippet of AX code that will manipulate the attributes on the VDB Points or "
                     "the VDB voxel values.")
        .setSpareData(&theEditor));

    parms.add(hutil::ParmFactory(PRM_STRING, "attributestocreate", "Attributes To Create")
        .setHelpText("Specify the attributes allowed to be created if they are not present on the input. "
                     "Use * to allow all attributes, or specify them by name in a space separated list. ")
        .setDefault("*"));

    // language/script modifiers

    parms.addFolder("Options");

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "allowvex", "Allow VEX")
        .setDefault(PRMoneDefaults)
        .setHelpText("Whether to enable support for various VEX functionality. When disabled, only AX "
                     "syntax is supported."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "hscriptvars", "Allow HScript Variables")
        .setDefault(PRMoneDefaults)
        .setHelpText("Whether to enable support for various $ variables available in the current node's "
                     "context. As $ is used for custom parameters in AX, a warning will be generated "
                     "if a Houdini parameter also exists of the same name as the given $ variable."));

    parms.add(hutil::ParmFactory(PRM_STRING, "cwdpath", "Evaluation Node Path")
        .setTypeExtended(PRM_TYPE_DYNAMIC_PATH)
        .setDefault(".")
        .setHelpText("Functions like ch() and $ syntax usually evaluate with respect to this node. "
            "Enter a node path here to override where the path search starts from. This is useful for "
            "embedding in a digital asset, where you want searches to start from the asset root. "
            "Note that HScript variables (if enabled) always refer to the AX node and ignore "
            "the evaluation path."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "ignoretiles", "Ignore Active Tiles")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "Whether to ignore active tiles in the input volumes, otherwise active tiles are converted "
            "into dense voxels (only where necessary). Only applies to volumes that are written to."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "Prune")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("Collapse regions of constant value in output grids. "
            "Voxel values are considered equal if they differ "
            "by less than the specified threshold.")
        .setDocumentation(nullptr));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "tolerance", "Prune Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1)
        .setTooltip(
            "When pruning is enabled, voxel values are considered equal"
            " if they differ by less than the specified tolerance."
            " Only applies to volumes that are written to.")
        .setDocumentation(
            "If enabled, reduce the memory footprint of output grids that have"
            " (sufficiently large) regions of voxels with the same value,"
            " where values are considered equal if they differ by less than"
            " the specified threshold."
            " Only applies to volumes that are written to.\n\n"
            ":note:\n"
            "    Pruning affects only the memory usage of a grid.\n"
            "    It does not remove voxels, apart from inactive voxels\n"
            "    whose value is equal to the background."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "compact", "Compact Attributes")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Whether to try to compact VDB Point Attributes after execution.")
        .setDocumentation(
            "Whether to try to compact VDB Point Attributes after execution\n\n"
            ":note:\n"
            "    Compacting uniform values affects only the memory usage of the attributes.\n"));

    hutil::ParmList bindings;
    bindings.add(hutil::ParmFactory(PRM_STRING, "attrname#", "Attribute Name")
        .setHelpText("Attribute name on geometry."));
    bindings.add(hutil::ParmFactory(PRM_STRING, "binding#", "AX Binding")
        .setHelpText("Attribute binding in AX code snippet."));

    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "bindings", "Attribute Bindings")
        .setHelpText("The number of attribute bindings. These are used to remove the need to update the code snippet "
                     "when an attribute name changes on the inputs, the binding in the code will instead refer to this attribute."
                     " If not included in this list, attributes will automatically bind by name.")
        .setMultiparms(bindings)
        .setDefault(PRMzeroDefaults));

    parms.endSwitcher();

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    {
        const char* items[] = {
            "points",   "Points",
            "volumes",  "Volumes",
            nullptr
        };

        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "targettype", "Target Type")
            .setDefault("points")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "pointsgroup", "VDB Points Group"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "createmissing", "Create Missing")
        .setDefault(PRMoneDefaults));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "createattributes", "Create New Attributes")
        .setDefault(PRMoneDefaults));

    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory factory("VDB AX", SOP_OpenVDB_AX::factory, parms, *table);

    factory.setObsoleteParms(obsoleteParms);
    factory.addInput("VDBs to manipulate");
    factory.addAliasVerbatim("DW_OpenVDBAX");
    factory.addAliasVerbatim("DN_OpenVDBAX");
    factory.setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_OpenVDB_AX::Cache; });
    factory.setDocumentation(R"(
#icon: COMMON/openvdb
#tags: vdb

"""Runs an AX snippet to modify point and volume values in VDBs"""

@overview

This is a very powerful, low-level node that lets those who are familiar with the AX language manipulate attributes on points and voxel values in VDBs.

AX is a language created by the DNEG FX R&D team that closely matches VEX but operates natively on VDB point and volume grids.
Note that the language is not yet as extensive as Houdini VEX and only supports a subset of similar functionality.

@examples

{{{
#!vex
@density = 1.0f; // Set the float attribute density to 1.0f
}}}
{{{
#!vex
i@id = 5; // Set the integer attribute id to 5
}}}
{{{
#!vex
vec3f@v1 = { 5.0f, 5.0f, 10.3f }; // Create a new float vector attribute
vector@v2 = { 5.0f, 5.0f, 10.3f }; // Create a new float vector attribute using VEX syntax
}}}
{{{
#!vex
vec3i@vid = { 3, -1, 10 }; // Create a new integer vector attribute
}}}

See the [ASWF OpenVDB AX documentation|https://www.openvdb.org/documentation/openvdbax.html] for source code
and usage examples.

@vexsyntax VEX Hooks
The OpenVDB AX Houdini SOP supports all features of AX and a variety of Houdini VEX syntax/features to help users transition into writing AX code.
The VEX feature set also gives users the ability to write typical Houdini specific functions within AX. The table below lists all VEX
features which can be used, as well as equivalent AX functionality. If no AX function is shown, the VEX function can still be used but will not be
available outside of a Houdini context.
:note: Allow Vex Symbols must be enabled to access these features.
:note: `$` AX syntax should always be used over the AX external() functions unless attempting to query unknown strings.

VEX Syntax/Function ||
    AX Syntax/Function ||
        Description ||

`ch(string_path)` |
    `$string_path, external(string_path)` |
        Finds a float channel value.

`chv(string_path)` |
    `v$string_path, externalv(string_path)` |
        Finds a string channel value.

`chs(string_path)` |
     `s$string_path` |
         Finds a string channel value.

`chramp(string_path)` |
    |
        Provides access to the chramp VEX function.

`vector2` |
    `vec2f` |
        Syntax for creating a vector 2 of floats.

`vector` |
    `vec3f` |
        Syntax for creating a vector 3 of floats.

`vector4` |
    `vec4f` |
        Syntax for creating a vector 4 of floats.

`matrix3` |
    `mat3f` |
        Syntax for creating a 3x3 matrix of floats.

`matrix` |
    `mat4f` |
        Syntax for creating a 4x4 matrix of floats.

`@ix, @iy, @iz` |
    `getcoordx(), getcoordy(), getcoordz()` |
        When executing over volumes, returns the index X, Y, Z coordinate of the current voxel.

@hscriptsyntax HScript Variables
HScript $ variables can also be accessed within AX. Note that the $ syntax in AX is equivalent to a Houdini channel function and is used to look-up
custom variables within AX. A different set of HScript variables will be available depending on the current Houdini Context. For a complete
list, [see here|/network/expressions#globals]
:note: Allow HScript Variables must be enabled to access HScript variables.
:tip: `@Frame` and `@Time` can be accessed with `$F` and `$T` respectively.

@axverb AX as a Python Verb
The AX SOP can be used within compiled blocks and as a verb through Houdini's python interface. The latter however introduces some restrictions to
the code which can be used due to the lack of a connected Houdini network. Through Python, the following restriction are imposed:
* $ Syntax for paths cannot be used. `ch` and `external` should be used instead.

* Relative channel paths with `ch` and `external` functions will produce an error. These must be converted to absolute paths.

For more information on Compiled Blocks and Python verbs [see here|/model/compile].

@functions Supported Functions
#display: collapsible collapsed
:note: For an up-to-date list of available functions, see the online AX documentation or run `vdb_ax functions --list` from the command line.

[Include:/ax/functions]
)");

    // Add backward compatible support if building against VDB 6.2
    // copy the implementation in vdb in regards to the vdb and houdini
    // version string, but also append the ax version (which as of merger
    // into VDB is the same as the VDB version)

#if (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER > 6 || \
    (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER >= 6 && OPENVDB_LIBRARY_MINOR_VERSION_NUMBER >= 2))
    std::stringstream ss;
    ss << "vdb" << OPENVDB_LIBRARY_VERSION_STRING << " ";
    ss << "houdini" << SYS_Version::full() << " ";
    ss << "vdb_ax" << OPENVDB_LIBRARY_VERSION_STRING;
    factory.addSpareData({{"operatorversion", ss.str()}});
#endif

}

////////////////////////////////////////


OP_Node*
SOP_OpenVDB_AX::factory(OP_Network* net,
                                    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_AX(net, name, op);
}

SOP_OpenVDB_AX::SOP_OpenVDB_AX(OP_Network* net,
        const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
{
    ax::initialize();
}

SOP_OpenVDB_AX::Cache::Cache()
    : mHash(0)
    , mParameterCache()
    , mCompilerCache()
    , mChExpressionSet()
    , mDollarExpressionSet()
{
    mCompilerCache.mCompiler = ax::Compiler::create();
    mCompilerCache.mCustomData.reset(new ax::CustomData);

    static auto locFromStr = [&] (const std::string& str) -> UT_SourceLocation {
        // find error location at end of message
        size_t locColon = str.rfind(":");
        size_t locLine = str.rfind(" ", locColon);
        int line = std::atoi(str.substr(locLine + 1, locColon - locLine - 1).c_str());
        int col = std::atoi(str.substr(locColon + 1, str.size()).c_str());
        // currently only does one character, as we don't know the offending code's length
        return UT_SourceLocation(nullptr, line, col, col+1);
    };

    mCompilerCache.mLogger.reset(new ax::Logger(
        [this](const std::string& str) {
            UT_SourceLocation loc = locFromStr(str);
            this->cookparms()->sopAddError(SOP_MESSAGE, str.c_str(), &loc);
        },
        [this](const std::string& str) {
            UT_SourceLocation loc = locFromStr(str);
            this->cookparms()->sopAddWarning(SOP_MESSAGE, str.c_str(), &loc);
        })
    );

    mCompilerCache.mLogger->setErrorPrefix("");
    mCompilerCache.mLogger->setWarningPrefix("");

    // initialize the function registry with VEX support as default
    initializeFunctionRegistry(*mCompilerCache.mCompiler, /*allow vex*/true);
}

void
SOP_OpenVDB_AX::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    resolveRenamedParm(*obsoleteParms, "targettype", "runover");
    resolveRenamedParm(*obsoleteParms, "pointsgroup", "vdbpointsgroup");

    // sync createattributes or createmissing to attributestocreate. if either
    // are not default, then they existed previously with old settings
    // (i.e. they were off) and attributestocreate needs to be cleared

    PRM_Parm* parm = obsoleteParms->getParmPtr("createattributes");
    bool update = parm && !parm->isFactoryDefault();
    parm = obsoleteParms->getParmPtr("createmissing");
    update |= parm && !parm->isFactoryDefault();

    if (update && this->hasParm("attributestocreate")) {
        this->getParm("attributestocreate").setValue(/*time*/0, "", CH_STRING_LITERAL);
    }

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}

bool
SOP_OpenVDB_AX::updateParmsFlags()
{
    bool changed = false;
    const bool points = evalInt("runover", 0, 0) == 0;
    changed |= enableParm("vdbpointsgroup", points);
    changed |= setVisibleState("vdbpointsgroup", points);

    changed |= enableParm("prune", !points);
    const bool prune = static_cast<bool>(evalInt("prune", 0, 0));
    changed |= enableParm("tolerance", prune && !points );
    changed |= setVisibleState("prune", !points);
    changed |= setVisibleState("tolerance", !points);
    changed |= enableParm("ignoretiles", !points);
    changed |= setVisibleState("ignoretiles", !points);

#ifdef DNEG_OPENVDB_AX
    changed |= enableParm("activity", !points);
    changed |= setVisibleState("activity", !points);
#endif

    changed |= enableParm("compact", points);
    changed |= setVisibleState("compact", points);

    return changed;
}

void SOP_OpenVDB_AX::syncNodeVersion(const char* old_version,
    const char* cur_version, bool* node_deleted)
{
    // A set of callbacks which are run each time a parameter is synchronized
    using ParamFunctionCallback = std::function<std::string(const SOP_OpenVDB_AX&)>;
    using ParamMap = std::map<std::string, ParamFunctionCallback>;

    // This map contains sync version parameters which instances of this
    // node are expected to apply sequentially in ascending order from their
    // old version (exclusive) to their current version (inclusive).
    static const std::unordered_map<std::string, ParamMap> versions = {
        {
        "0.1.0", {
            // We can't just return 0 as the expected behaviour here is for points to always
            // create attribute and for volumes to error. This preserves that behaviour.
            { "attributestocreate",
                [](const SOP_OpenVDB_AX& node) -> std::string {
                    const int targetType = static_cast<int>(node.evalInt("runover", 0, 0));
                    if (targetType == 0) return "*"; // points, keep default (on)
                    else return ""; // volumes, turn off
                }
            }
        }},
        {
        "8.0.2", {
            { "ignoretiles",
                [](const SOP_OpenVDB_AX& node) -> std::string {
                    return "1";
                }
            },
        }}
    };

    auto axVersion = [](const UT_String& version) -> std::string {
        if (!version.startsWith("vdb")) return "";
        std::string axversion(version.c_str());
        const size_t pos = axversion.find("vdb_ax");
        if (pos == std::string::npos) return "";
        axversion = axversion.substr(pos + 6);
        return axversion;
    };

    const UT_String old(old_version);
    const UT_String current(cur_version);

    const std::string currentAx = axVersion(current);
    if (currentAx.empty()) {
        // unable to parse current version
        SOP_Node::syncNodeVersion(old_version, cur_version, node_deleted);
        return;
    }

    std::string oldAx = axVersion(old);
    if (oldAx.empty()) {
        // if can't parse, old version was created before the spare data vdb_ax
        // version name was added .i.e. version 0.0.0
        oldAx = "0.0.0";
    }

    // @note  UT_String::compareVersionString(A, B) returns the following:
    //   -1 if A < B  i.e. ("0.0.0", "1.0.0")
    //    0 if A == B i.e. ("1.0.0", "1.0.0")
    //    1 if A > B  i.e. ("1.0.0", "0.0.0")

    // if current <= old, return
    if (UT_String::compareVersionString(currentAx.c_str(), oldAx.c_str()) == -1) {
        SOP_Node::syncNodeVersion(old_version, cur_version, node_deleted);
        return;
    }

    // for each set of version keys in the version map that lie in-between (old, current],
    // apply the parameter updates
    for (const auto& versionData : versions) {
        const std::string& version = versionData.first;
        // only apply param updates if the version key is greater than the old version
        if (UT_String::compareVersionString(version.c_str(), oldAx.c_str()) != 1) {
            // oldVersion is less or equal to the current version key, continue
            continue;
        }
        // exit if the current version is greater than the version key
        if (UT_String::compareVersionString(currentAx.c_str(), version.c_str()) == 1) {
            break;
        }

        // apply this set of updates
        for (auto& data : versionData.second) {
            const std::string& name = data.first;
            const ParamFunctionCallback& callback = data.second;
            PRM_Parm* parm = this->getParmPtr(name);
            if (!parm) continue;
            const std::string valuestr = callback(*this);
            parm->setValue(/*time*/0, valuestr.c_str(), CH_STRING_LITERAL);
         }
    }

    SOP_Node::syncNodeVersion(old_version, cur_version, node_deleted);
}


////////////////////////////////////////

namespace {
struct PruneOp {
    PruneOp(const fpreal tol)
        : mTol(tol) {}

    template<typename GridT>
    void operator()(GridT& grid) const {
        tools::prune(grid.tree(), typename GridT::TreeType::ValueType(mTol));
    }
    const fpreal mTol;
};
}

OP_ERROR
SOP_OpenVDB_AX::Cache::cookVDBSop(OP_Context& context)
{
    try {
        // may be null if cooking as a verb i.e. through python
        SOP_OpenVDB_AX* self =
            static_cast<SOP_OpenVDB_AX*>(this->cookparms()->getSrcNode());

        hvdb::HoudiniInterrupter boss("Executing OpenVDB AX");

        const fpreal time = context.getTime();

        // Get ui params, including grids to process
        UT_String groupStr;
        evalString(groupStr, "vdbgroup", 0, time);
        const GA_PrimitiveGroup *group =
            matchGroup(const_cast<GU_Detail&>(*gdp), groupStr.toStdString());
        groupStr.clear();

        hvdb::VdbPrimIterator vdbIt(gdp, group);
        if (!vdbIt) return error();

        // Evaluate the code snippet field

        // @note We generally want to evaluate the raw string to optimise channel
        // links and stop $ variables from auto expanding. If $ variables are
        // Houdini variables, such as $F and $T then it's fine if they're expanded,
        // but we won't be able to optimise using AX's custom data (the string will
        // require re-compilation on every cook). However, if $ variables are paths
        // using AX's $ syntax then we don't want to auto expand these to null
        // values. Unfortunately, if we've built with compiled SOP support, there's
        // no way to evaluate the raw string unless a source node instance exists.
        // This won't exist if it's being cooked as a verb through python - we
        // fall back to calling evalString in this case, but this will cause errors
        // if $ is being used for path syntax.

        UT_String snippet;
        if (self) self->evalStringRaw(snippet, "snippet", 0, time);
        else      this->evalString(snippet, "snippet", 0, time);
        if (snippet.length() == 0) return error();

        const int targetType = static_cast<int>(evalInt("runover", 0, time));

        // get the node which is set as the current evaluation path. If we can't find the
        // node, all channel links are zero valued. This matches VEX behaviour.
        UT_String path;
        this->evalString(path, "cwdpath", 0, time);
        OP_Node* evaluationNode = this->cookparms()->getCwd()->findNode(path);
        if (!evaluationNode) {
            const std::string message = "The node \"" + path.toStdString() + "\" was "
                "not found or was the wrong type for this operation. All channel "
                "references and $ parameters will evaluate to 0.";
            addWarning(SOP_MESSAGE, message.c_str());
        }

        ParameterCache parmCache;
        parmCache.mTargetType = static_cast<hax::TargetType>(targetType);
        parmCache.mVEXSupport = evalInt("allowvex", 0, time);
        parmCache.mHScriptSupport = evalInt("hscriptvars", 0, time);

        // @TODO use parameter update notifications to query if the snippet
        // has changed rather than hashing the code

        const unsigned hash = snippet.hash();
        const bool recompile =
            (hash != mHash || parmCache != mParameterCache);

        openvdb::ax::AttributeBindings bindings;
        const int numBindings = static_cast<int>(evalInt("bindings", 0, time));
        UT_String name, binding;
        std::unordered_set<std::string> unbindable = {"P", "ix", "iy", "iz"};
        bool bindError = false;
        for (int i = 1; i <= numBindings; ++i) {
            evalStringInst("attrname#", &i, name, 0, time);
            evalStringInst("binding#", &i, binding, 0, time);
            if (name.equal("") || binding.equal("")) continue;
            const std::string bindingStr = binding.toStdString();
            const std::string nameStr = name.toStdString();
            if (mParameterCache.mTargetType == hax::TargetType::VOLUMES) {
                // check nothing tries to bind to P, ix, iy, iz
                // as these have been replaced by function calls
                for (const auto& unbind : unbindable) {
                    if (bindingStr == unbind || nameStr == unbind) {
                        addError(SOP_MESSAGE,
                                 std::string("Cannot bind to \"@" + unbind +
                                  "\" as it is a protected name.").c_str());
                        bindError = true;
                    }
                }
            }
            bindings.set(bindingStr, nameStr);
        }
        if (bindError) return error();

        if (recompile) {

            // Empty the hash - if there are any compiler failures, the hash won't be
            // initialized but the engine data maybe modified. If the code is then changed
            // back to the previous hash, this path will not be correctly executed
            // without this explicit change

            mHash = 0;

            mCompilerCache.mLogger->clear();
            mChExpressionSet.clear();
            mDollarExpressionSet.clear();

            // if VEX support flag has changes, re-initialize the available functions

            if (mParameterCache.mVEXSupport != parmCache.mVEXSupport) {
                initializeFunctionRegistry(*mCompilerCache.mCompiler, parmCache.mVEXSupport);
            }

            // build the AST from the provided snippet

            openvdb::ax::ast::Tree::ConstPtr tree =
                ax::ast::parse(snippet.nonNullBuffer(), *mCompilerCache.mLogger);

            // currently only catches single syntax error but could be updated
            // to catch multiple. Further still can be updated to encounter
            // syntax errors AND output a valid tree
            // @todo: update to catch multiple errors and output tree when possible

            if (!tree) {
                appendLoggerMessage(*this, *mCompilerCache.mLogger, "syntax");
                return error();
            }

            // store a copy of the AST to modify, the logger will store the original for error printing

            mCompilerCache.mSyntaxTree.reset(tree->copy());

            // find all externally accessed data - do this before conversion from VEX
            // so identify HScript tokens which have been explicitly requested with $
            // (as otherwise we'll pick up optimised or user $ paths)

            hax::findChannelExpressions(*mCompilerCache.mSyntaxTree, mChExpressionSet);
            hax::findDollarExpressions(*mCompilerCache.mSyntaxTree, mDollarExpressionSet);

            // begin preprocessors

            if (parmCache.mVEXSupport) {

                // if we're allowing VEX syntax, convert any supported VEX functions and
                // accesses to AX syntax. Note that there may be some functions, such as
                // chramp, that are not reliant on VEX as we re-implement them in the AX
                // Houdini plugin but not yet in the AX Core library

                hax::convertASTFromVEX(*mCompilerCache.mSyntaxTree, parmCache.mTargetType);
            }

            // optimise external lookup function calls into $ calls if the argument is a string literal

            hax::convertASTKnownLookups(*mCompilerCache.mSyntaxTree);

            // end preprocessors

            // reset any custom data

            mCompilerCache.mCustomData->reset();

            // initialize local variables - do this outside of evaluateExternalExpressions
            // so it's not called for every cook if nothing has changed

            if (!mDollarExpressionSet.empty() && parmCache.mHScriptSupport) {
                this->cookparms()->setupLocalVars();
            }

            evaluateExternalExpressions(time, mChExpressionSet, /*no $ support*/false, evaluationNode);
            evaluateExternalExpressions(time, mDollarExpressionSet, parmCache.mHScriptSupport, evaluationNode);

            if (parmCache.mTargetType == hax::TargetType::POINTS) {
                mCompilerCache.mPointExecutable =
                    mCompilerCache.mCompiler->compile<ax::PointExecutable>
                        (*mCompilerCache.mSyntaxTree, *mCompilerCache.mLogger, mCompilerCache.mCustomData);
            }
            else if (parmCache.mTargetType == hax::TargetType::VOLUMES) {
                mCompilerCache.mVolumeExecutable =
                    mCompilerCache.mCompiler->compile<ax::VolumeExecutable>
                        (*mCompilerCache.mSyntaxTree, *mCompilerCache.mLogger, mCompilerCache.mCustomData);
            }

            // update the parameter cache

            mParameterCache = parmCache;

            // add compilation warnings/errors

            if (appendLoggerMessage(*this, *mCompilerCache.mLogger, "compiler")) {
                return error();
            }

            // if successful, also create the attribute registry to check against
            mCompilerCache.mAttributeRegistry = openvdb::ax::AttributeRegistry::create(*mCompilerCache.mSyntaxTree);

            // set the hash only if compilation was successful - Houdini sops tend to cook
            // multiple times, especially on fail. If we assign the hash prior to this it will
            // be incorrectly cached

            mHash = hash;
        }
        else {
            evaluateExternalExpressions(time, mChExpressionSet, /*no $ support*/false, evaluationNode);
            evaluateExternalExpressions(time, mDollarExpressionSet, parmCache.mHScriptSupport, evaluationNode);
        }

        snippet.clear();

        const std::string attribList = evalStdString("attributestocreate", time);

        if (mParameterCache.mTargetType == hax::TargetType::POINTS) {

            UT_String pointsStr;
            evalString(pointsStr, "vdbpointsgroup", 0, time);
            const std::string pointsGroup = pointsStr.toStdString();

            for (; vdbIt; ++vdbIt) {
                if (boss.wasInterrupted()) {
                    throw std::runtime_error("processing was interrupted");
                }

                GU_PrimVDB* vdbPrim = *vdbIt;

                if (!(vdbPrim->getConstGridPtr()->isType<points::PointDataGrid>())) continue;
                vdbPrim->makeGridUnique();

                points::PointDataGrid::Ptr points =
                    gridPtrCast<points::PointDataGrid>(vdbPrim->getGridPtr());

                if (!mCompilerCache.mPointExecutable) {
                    throw std::runtime_error("No point executable has been built");
                }

                const auto& leafIter = points->tree().cbeginLeaf();
                if (!leafIter) continue; // empty

                // check the attributes that are not being created already exist

                std::vector<UT_String> missingAttributes;
                const auto& desc = leafIter->attributeSet().descriptor();

                for (const auto& attribute : mCompilerCache.mAttributeRegistry->data()) {
                    const std::string* nameptr = &attribute.name();
                    const std::string* binding = bindings.dataNameBoundTo(*nameptr);
                    if (binding) nameptr = binding;
                    assert(nameptr);
                    const std::string& name = *nameptr;
                    if (desc.find(name) == openvdb::points::AttributeSet::INVALID_POS) {
                        missingAttributes.emplace_back(name);
                    }
                }
                checkAttributesAgainstList(attribList, missingAttributes);

                mCompilerCache.mPointExecutable->setGroupExecution(pointsGroup);
                mCompilerCache.mPointExecutable->setCreateMissing(true);
                mCompilerCache.mPointExecutable->setAttributeBindings(bindings);
                mCompilerCache.mPointExecutable->execute(*points);

                if (evalInt("compact", 0, time)) {
                    openvdb::points::compactAttributes(points->tree());
                }
            }
        }
        else if (mParameterCache.mTargetType == hax::TargetType::VOLUMES) {

            GridPtrVec grids;
            std::vector<GU_PrimVDB*> guPrims;
            std::set<std::string> names;

            for (; vdbIt; ++vdbIt) {
                if (boss.wasInterrupted()) {
                    throw std::runtime_error("processing was interrupted");
                }

                GU_PrimVDB* vdbPrim = *vdbIt;
                if (vdbPrim->getConstGridPtr()->isType<points::PointDataGrid>()) continue;
                vdbPrim->makeGridUnique();

                const std::string name = vdbPrim->getGridName();
                if (names.count(name)) {
                    addWarning(SOP_MESSAGE,
                        std::string("Multiple VDBs \"" + name + "\" encountered. "
                        "Only the first grid will be processed.").c_str());
                }

                // AX determines the grid access from the grid name. Houdini only
                // updates the VDB Grid name on de-serialization, so ensure the
                // grid's metadata name is up-to-date

                const openvdb::GridBase::Ptr grid = vdbPrim->getGridPtr();
                if (name != grid->getName()) grid->setName(name);

                names.insert(name);
                grids.emplace_back(grid);
                guPrims.emplace_back(vdbPrim);
            }

            if (!mCompilerCache.mVolumeExecutable) {
                throw std::runtime_error("No volume executable has been built");
            }

#ifdef DNEG_OPENVDB_AX
            const ax::VolumeExecutable::IterType
                iterType = static_cast<ax::VolumeExecutable::IterType>(evalInt("activity", 0, time));
#endif

            const size_t size = grids.size();

            // check the attributes that are not being created already exist
            std::vector<UT_String> missingAttributes;
            const auto& attribRegistry = mCompilerCache.mAttributeRegistry;
            for (const auto& attribute : attribRegistry->data()) {
                    const std::string* nameptr = &attribute.name();
                    const std::string* binding = bindings.dataNameBoundTo(*nameptr);
                    if (binding) nameptr = binding;
                    assert(nameptr);
                    const std::string& name = *nameptr;
                    if (names.find(name) == names.cend()) missingAttributes.emplace_back(name);
            }
            checkAttributesAgainstList(attribList, missingAttributes);

            auto applyOpToWriteGrids = [&](const auto& op) {
                for (auto& vdbPrim : guPrims) {
                    if (attribRegistry->isWritable(vdbPrim->getGridName(),
                            openvdb::ax::ast::tokens::UNKNOWN)) {
                        if (boss.wasInterrupted()) {
                            throw std::runtime_error("processing was interrupted");
                        }
                        hvdb::GEOvdbApply<hvdb::VolumeGridTypes>(*vdbPrim, op);
                    }
                }
            };

            // if not processing tiles, only run this executable on the lowest,
            // (leaf) level, otherwise process the entire tree.
            const openvdb::Index maxLevel =
                evalInt("ignoretiles", 0, time) ?
                    0 : openvdb::FloatTree::DEPTH-1; // 0=leaf

            mCompilerCache.mVolumeExecutable->setTreeExecutionLevel(/*leaf=*/0, maxLevel);

#ifdef DNEG_OPENVDB_AX
            mCompilerCache.mVolumeExecutable->setValueIterator(iterType);
#endif
            mCompilerCache.mVolumeExecutable->setCreateMissing(true);
            mCompilerCache.mVolumeExecutable->setAttributeBindings(bindings);
            mCompilerCache.mVolumeExecutable->execute(grids);

            if (evalInt("prune", 0, time)) {
                const fpreal tol = evalFloat("tolerance", 0, time);
                const PruneOp op(tol);
                applyOpToWriteGrids(op);
            }

            std::vector<openvdb::GridBase::Ptr> invalid;
            for (size_t pos = size; pos < grids.size(); ++pos) {
                auto& grid = grids[pos];
                // Call apply with a noop as createVdbPrimitive requires a grid ptr.
                // apply will return false if the grid is not one of the supported types
                if (!grid->apply<hvdb::AllGridTypes>([](auto&){})) {
                    invalid.emplace_back(grid);
                }
                else {
                    hvdb::createVdbPrimitive(*gdp, grid);
                }
            }

            if (!invalid.empty()) {
                std::ostringstream os;
                os << "Unable to create the following grid types as these are not supported by Houdini:\n";
                for (auto& grid : invalid) {
                    os << "Grid Name: " << grid->getName() << ", Type: " << grid->valueType() << '\n';
                }
                addWarning(SOP_MESSAGE, os.str().c_str());
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

bool
SOP_OpenVDB_AX::Cache::evalInsertHScriptVariable(const std::string& name,
                                                 const std::string& accessedType,
                                                 ax::CustomData& data)
{
    OP_Director* const director = OPgetDirector();
    OP_CommandManager* const manager = director ? director->getCommandManager() : nullptr;
    CMD_VariableTable* const table = manager ? manager->getGlobalVariables() : nullptr;

    bool isVariable = false;

    std::unique_ptr<UT_String> valueStrPtr;
    std::unique_ptr<fpreal32> valueFloatPtr;
    std::string expectedType;

    if (table && table->hasVariable(name.c_str())) {

        isVariable = true;

        // the accessed variable is a valid hscript global var - attempt to evaluate it as
        // a float or a string. If the float evaluation fails or returns 0.0f, assume the
        // variable is a string

        UT_String valueStr;
        table->getVariable(name.c_str(), valueStr);

        if (valueStr.length() > 0) {
            const std::string str = valueStr.toStdString();
            try {
                const fpreal32 valueFloat = static_cast<fpreal32>(std::stod(str));
                valueFloatPtr.reset(new fpreal32(valueFloat));
                expectedType = openvdb::typeNameAsString<float>();
            }
            catch (...) {}

            if (!valueFloatPtr) {
                valueStrPtr.reset(new UT_String(valueStr));
                expectedType = openvdb::typeNameAsString<std::string>();
            }
        }
    }
    else {

        // not a global variable, attempt to evaluate as a local

        OP_Node* self = this->cookparms()->getCwd();

        OP_Channels* channels = self->getChannels();
        if (!channels) return false;

        int index = -1;
        const CH_LocalVariable* const var =
            channels->resolveVariable(name.c_str(), index);
        if (!var) return false;

        isVariable = true;
        UT_ASSERT(index >= 0);

        expectedType = var->flag & CH_VARIABLE_STRVAL ?
            openvdb::typeNameAsString<std::string>() :
            openvdb::typeNameAsString<float>();

        if (var->flag & CH_VARIABLE_STRVAL) {
            UT_String value;
            if (channels->getVariableValue(value, index, var->id, /*thread*/0)) {
                valueStrPtr.reset(new UT_String(value));
            }
        }
        else {
            fpreal value;
            if (channels->getVariableValue(value, index, var->id, /*thread*/0)) {
                valueFloatPtr.reset(new fpreal32(static_cast<fpreal32>(value)));
            }
        }

        // If the channel is time dependent, ensure it's propagated to this node

        if (valueFloatPtr || valueStrPtr) {
            DEP_MicroNode* dep = this->cookparms()->depnode();
            if (dep && !dep->isTimeDependent() && var->isTimeDependent()) {
                dep->setTimeDependent(true);
            }
        }
    }

    if (!isVariable) return false;

    if (valueFloatPtr || valueStrPtr) {

        // if here, we've evaluated the variable successfully as either a float or string

        if (accessedType != expectedType) {
            // If the types differ, differ to the compiler to insert the correct zero val
            const std::string message = "HScript variable \"" + name + "\" has been accessed"
                " with an incompatible type. Expected to be \"" + expectedType + "\". Accessed "
                " with \"" + accessedType + "\".";
            addWarning(SOP_MESSAGE, message.c_str());
        }
        else if (valueStrPtr) {
            typename TypedMetadata<std::string>::Ptr meta(new TypedMetadata<std::string>(valueStrPtr->toStdString()));
            data.insertData<TypedMetadata<std::string>>(name, meta);
        }
        else {
            UT_ASSERT(valueFloatPtr);
            typename TypedMetadata<float>::Ptr meta(new TypedMetadata<float>(*valueFloatPtr));
            data.insertData<TypedMetadata<float>>(name, meta);
        }

        return true;
    }

    // we've been unable to insert a valid variable due to some internal Houdini
    // type evaluation error. The compiler will ensure that it's initialized to a
    // valid zero val.

    const std::string message = "Unable to evaluate accessed HScript Variable \"" + name + "\".";
    addWarning(SOP_MESSAGE, message.c_str());
    return false;
}

void
SOP_OpenVDB_AX::Cache::evaluateExternalExpressions(const double time,
                                                   const hax::ChannelExpressionSet& set,
                                                   const bool hvars,
                                                   OP_Node* evaluationNode)
{
    using VectorData = TypedMetadata<math::Vec3<float>>;
    using FloatData = TypedMetadata<float>;
    using StringData = TypedMetadata<openvdb::ax::codegen::String>;

    ax::CustomData& data = *(mCompilerCache.mCustomData);

    // For compilable SOPs, see if we can connect this cache back to a SOP instance by
    // querying the source node. If this doesn't exist, then we're most likely being
    // cooked as a verb through python and we'll be unable to evaluate relative
    // references.

    OP_Node* self = this->cookparms()->getCwd();
    const bool hasSrcNode = this->cookparms()->getSrcNode() != nullptr;
    DEP_MicroNode* dep = this->cookparms()->depnode();

    for (const hax::ChannelExpressionPair& expresionPair : set) {

        // get the type that was requested and the name of the item. The name is
        // either a channel path or a Houdini HScript Variable

        const std::string& type = expresionPair.first;
        const std::string& nameOrPath = expresionPair.second;
        if (nameOrPath.empty()) continue;

        // Process the item as if it were a hscript token first if hscript support
        // is enabled.

        // Note that hscript local variables are evaluated local to *this, not local
        // to the evaluationNode.

        if (hvars) {

            // try and add this item with the requested type and name. If the type
            // doesnt match it's actual type, defer to the compiler to initialise a zero val
            // item and continue with a warning. If the name isn't a hscript token, it's
            // most likely a channel path

            // @note that for compiled SOPs being cooked as verbs this will always return false
            // as we evaluate the expanded string .i.e. if hasSrcNode is false, nameOrPath will
            // never be a $ variable. Execute this branch anyway to support this in the future

            if (this->evalInsertHScriptVariable(nameOrPath, type, data)) {

                // see if the current SOP instance has a parm

                if (hasSrcNode && self->hasParm(nameOrPath.c_str())) {
                    const std::string message = "Initialized HScript Token \"" + nameOrPath +
                        "\" is also a valid channel path. Consider renaming this parameter.";
                    addWarning(SOP_MESSAGE, message.c_str());
                }
                continue;
            }
        }

        // If running in python, we can't process relative channel links as we don't know
        // the source location

        const bool isAbsolutePath = nameOrPath[0] == '/';
        if (!hasSrcNode && !isAbsolutePath) {
            throw std::runtime_error("Unable to process relative channel link \"" + nameOrPath
                + "\" when cooking as a verb.");
        }

        // if we're here, process the item as a channel

        // @note this currently matches houdini vex behaviour
        //
        // - 1) ch(p) with p = parm - return parm evalauted as float
        // - 2) ch(p) with p = channel - return channel evalauted as float
        //        in both cases, if the parm is not a single value, return 0
        //
        // - 3) chv(p) with p = parm - return parm evalauted as vec
        //        if p is not a vec3, fill as many elements as possible.
        //        for example, if p is a scalar value of 1, return (1,0,0)
        //        if p is a vec2 (2,3), return (2,3,0)
        // - 4) chv(p) with p = channel - return channel evalauted as vec
        //        as it's a channel it's always going to be a single value.
        //        in this case return a vector filled with the single value.
        //        for example, if p = 1, return (1,1,1)
        //
        //  -5) chramp(p) - as ramps are alwyas multi parms, we don't  have
        //        to consider the case where it could be a channel

        const bool isCHRampLookup(type == "ramp");
        const bool isCHLookup(!isCHRampLookup &&
            type == openvdb::typeNameAsString<float>());
        const bool isCHVLookup(!isCHRampLookup && !isCHLookup &&
            type == openvdb::typeNameAsString<openvdb::Vec3f>());
        const bool isCHSLookup(!isCHRampLookup && !isCHLookup && !isCHVLookup &&
            type == openvdb::typeNameAsString<std::string>());

        const bool lookForChannel = !isCHRampLookup;

        // findParmRelativeTo finds the node and parameter index on the node which is
        // related to the nameOrPath relative to this node
        // @note Do NOT use OPgetParameter() directly as this seems to cause crashes
        // when used with various DOP networks

        int index(0), subIndex(0);
        OP_Node* node(nullptr);
        bool validPath = false;

        if (evaluationNode) {
            // @todo: cache channelFinder?
            OP_ExprFindCh channelFinder;
            validPath = channelFinder.findParmRelativeTo(*evaluationNode,
                                                 nameOrPath.c_str(),
                                                 time,
                                                 node,            /*the node holding the param*/
                                                 index,           /*parm index on the node*/
                                                 subIndex,        /*sub index of parm if not channel*/
                                                 lookForChannel); /*is_for_channel_name*/

            // if no channel found and we're using CHV, try looking for the parm directly

            if (!validPath && isCHVLookup) {
                validPath =
                    channelFinder.findParmRelativeTo(*evaluationNode,
                                                     nameOrPath.c_str(),
                                                     time,
                                                     node,       /*the node holding the param*/
                                                     index,      /*parm index on the node*/
                                                     subIndex,   /*sub index of parm if not channel*/
                                                     false);     /*is_for_channel_name*/
            }
        }

        if (validPath) {

            assert(node);

            if (isCHVLookup) {

                Vec3f value;
                if (subIndex != -1) {
                    // parm was a channel
                    value = openvdb::Vec3f(node->evalFloat(index, subIndex, time));
                }
                else {
                    // parm was a direct parm
                    value[0] = static_cast<float>(node->evalFloat(index, 0, time));
                    value[1] = static_cast<float>(node->evalFloat(index, 1, time));
                    value[2] = static_cast<float>(node->evalFloat(index, 2, time));
                }

                VectorData::Ptr vecData(new VectorData(value));
                data.insertData<VectorData>(nameOrPath, vecData);

                // add an extra input to all the relevant channels of this
                // parameter if this dep micronode exists

                if (dep) {
                    PRM_Parm& parm = node->getParm(index);

                    // micro node is guaranteed to exist as we've evaluated the param
                    if (subIndex == -1) {
                        dep->addExplicitInput(parm.microNode(0));
                        dep->addExplicitInput(parm.microNode(1));
                        dep->addExplicitInput(parm.microNode(2));
                    }
                    else {
                        dep->addExplicitInput(parm.microNode(subIndex));
                    }
                }
            }
            else if (isCHLookup) {

                assert(subIndex != -1);

                // use evalFloat rather than parm->getValue() to wrap the conversion to a float
                const float value = static_cast<float>(node->evalFloat(index, subIndex, time));

                FloatData::Ptr floatData(new FloatData(value));
                data.insertData(nameOrPath, floatData);

                // add a dependency to this micronode if it exists

                if (dep) {
                    // micro node is guaranteed to exist as we've evaluated the param
                    PRM_Parm& parm = node->getParm(index);
                    dep->addExplicitInput(parm.microNode(subIndex));
                }
            }
            else if (isCHSLookup) {

                assert(subIndex != -1);

                UT_String string;
                node->evalString(string, index, subIndex, time);

                StringData::Ptr stringData(new StringData(string.toStdString()));
                data.insertData(nameOrPath, stringData);

                // add a dependency to this micronode if it exists

                if (dep) {
                    // micro node is guaranteed to exist as we've evaluated the param
                    PRM_Parm& parm = node->getParm(index);
                    dep->addExplicitInput(parm.microNode(subIndex));
                }
            }
            else if (isCHRampLookup) {

                PRM_Parm& parm = node->getParm(index);
                const bool isRamp = parm.isRampType();
                hax::RampDataCache::Ptr ramp(new hax::RampDataCache());

                if (!isRamp) {
                    const std::string message =
                        "Invalid channel reference: " + nameOrPath + ". Parameter is not a ramp.";
                    addWarning(SOP_MESSAGE, message.c_str());
                    data.insertData(nameOrPath, ramp);
                    continue;
                }

                node->updateRampFromMultiParm(time, parm, ramp->value());
                data.insertData(nameOrPath, ramp);

                // add all parms of this ramps multi parm as a dependency to this
                // micronode if it exists

                if (dep) {
                    OP_Node::addMultiparmInterests(*dep, node, parm);
                }
            }
        }
        else {

            if (isCHVLookup) {
                VectorData::Ptr vecData(new VectorData(openvdb::Vec3f::zero()));
                data.insertData<VectorData>(nameOrPath, vecData);
            }
            else if (isCHLookup) {
                FloatData::Ptr floatData(new FloatData(0.0f));
                data.insertData<FloatData>(nameOrPath, floatData);
            }
            else if (isCHSLookup) {
                StringData::Ptr stringData(new StringData(""));
                data.insertData<StringData>(nameOrPath, stringData);
            }
            else if (isCHRampLookup) {
                hax::RampDataCache::Ptr ramp(new hax::RampDataCache());
                data.insertData<hax::RampDataCache>(nameOrPath, ramp);
            }

            // Only warn if we can't find the channel reference on a valid
            // evaluation node - a global warning is applied for this case
            if (evaluationNode) {
                const std::string message = "Invalid channel reference: " + nameOrPath;
                addWarning(SOP_MESSAGE, message.c_str());
            }
        }
    }
}
