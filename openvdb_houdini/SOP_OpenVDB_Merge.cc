// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file SOP_OpenVDB_Merge.cc
///
/// @author Dan Bailey
///
/// @brief Merge OpenVDB grids.

#include "SOP_OpenVDB_Merge.proto.h"

#include <PRM/PRM_TemplateBuilder.h>
#include <UT/UT_DSOVersion.h>


class SOP_OpenVDB_Merge : public SOP_Node
{
public:
    static PRM_Template *buildTemplates();
    static OP_Node *myConstructor(OP_Network *net, const char *name, OP_Operator *op)
    {
        return new SOP_OpenVDB_Merge(net, name, op);
    }

    static const UT_StringHolder theSOPTypeName;

    virtual const SOP_NodeVerb *cookVerb() const final;

protected:
    SOP_OpenVDB_Merge(OP_Network *net, const char *name, OP_Operator *op)
        : SOP_Node(net, name, op) { }

    virtual ~SOP_OpenVDB_Merge() {}

    virtual OP_ERROR cookMySop(OP_Context &context) final
    {
        return cookMyselfAsVerb(context);
    }
};


const UT_StringHolder SOP_OpenVDB_Merge::theSOPTypeName("DW_OpenVDBMerge");


void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(new OP_Operator(
        SOP_OpenVDB_Merge::theSOPTypeName,
        "VDB Merge",
        SOP_OpenVDB_Merge::myConstructor,
        SOP_OpenVDB_Merge::buildTemplates(),
        1,
        OP_MULTI_INPUT_MAX,
        nullptr,
        0));
}


static const char *theDsFile = R"THEDSFILE(
{
    name        parameters

    inputlabel  1   "VDBs To Merge"

    parm {
        name    "group"
        label   "Group"
        type    string
        default { "" }
    }
    parm {
        name    "vdbpointsgroup"
        label   "VDB Points Group"
        type    string
        default { "" }
    }
    parm {
        name    "collation"
        label   "Collation"
        type    ordinal
        default { "nameclassandtype" }
        menu {
            "nameclassandtype"  "Name, VDB Class and Value Type"
            "nameandclass"      "Name and VDB Class"
            "classandtype"      "VDB Class and Value Type"
        }
    }
    parm {
        name    "firstvdbisreference"
        label   "First VDB Is Only for Reference"
        type    toggle
        default { "0" }
    }
    parm {
        name    "resample"
        label   "Resample VDBs"
        type    ordinal
        default { "first" }
        menu {
            "first"         "To Match First VDB"
            "highestres"    "To Match Highest-Res VDB"
            "lowestres"     "To Match Lowest-Res VDB"
        }
        joinnext
    }
    parm {
        name    "resampleinterp"
        label   "Interpolation"
        type    ordinal
        default { "linear" }
        menu {
            "point"     "Nearest"
            "linear"    "Linear"
            "quadratic" "Quadratic"
        }
    }
    groupsimple {
        name    "mergeoperationgroup"
        label   "Merge Operation"

        parm {
            name    "op_fog"
            label   "Fog VDBs"
            type    ordinal
            default { "add" }
            menu {
                "none"              "None"
                "add"               "Add"
                "multiply"          "Multiply"
                "maximum"           "Maximum"
                "minimum"           "Minimum"
                "topounion"         "Activity Union"
                "topointersect"     "Activity Intersection"
                "topounionall"      "Activity Union All VDBs"
                "topointersectall"  "Activity Intersection All VDBs"
            }
            joinnext
        }
        parm {
            name    "op_scalar"
            label   "Scalar VDBs"
            type    ordinal
            default { "add" }
            menu {
                "none"              "None"
                "add"               "Add"
                "multiply"          "Multiply"
                "maximum"           "Maximum"
                "minimum"           "Minimum"
                "topounion"         "Activity Union"
                "topointersect"     "Activity Intersection"
                "topounionall"      "Activity Union All VDBs"
                "topointersectall"  "Activity Intersection All VDBs"
            }
        }
        parm {
            name    "op_sdf"
            label   "SDF VDBs"
            type    ordinal
            default { "sdfunion" }
            menu {
                "none"              "None"
                "sdfunion"          "SDF Union"
                "sdfintersect"      "SDF Intersect"
                "topounion"         "Activity Union"
                "topointersect"     "Activity Intersection"
                "topounionall"      "Activity Union All VDBs"
                "topointersectall"  "Activity Intersection All VDBs"
            }
            joinnext
        }
        parm {
            name    "op_vector"
            label   "Vector VDBs"
            type    ordinal
            default { "add" }
            menu {
                "none"              "None"
                "add"               "Add"
                "multiply"          "Multiply"
                "topounion"         "Activity Union"
                "topointersect"     "Activity Intersection"
                "topounionall"      "Activity Union All VDBs"
                "topointersectall"  "Activity Intersection All VDBs"
            }
        }
        parm {
            name    "op_boolean"
            label   "Boolean VDBs"
            type    ordinal
            default { "topounion" }
            menu {
                "none"              "None"
                "topounion"         "Activity Union"
                "topointersect"     "Activity Intersection"
                "topounionall"      "Activity Union All VDBs"
                "topointersectall"  "Activity Intersection All VDBs"
            }
            joinnext
        }
        parm {
            name    "op_points"
            label   "VDB Points"
            type    ordinal
            default { "pointmerge" }
            menu {
                "none"              "None"
                "pointmerge"        "Merge"
                "topounion"         "Activity Union"
                "topointersect"     "Activity Intersection"
                "topounionall"      "Activity Union All VDBs"
                "topointersectall"  "Activity Intersection All VDBs"
            }
        }
    }
    groupsimple {
        name    "postprocessgroup"
        label   "Post-Process"

        parm {
            name    "flood"
            label   "Signed-Flood-Fill Output SDFs"
            type    toggle
            default { "0" }
        }
        parm {
            name    "deactivate"
            type    toggle
            nolabel
            default { "0" }
            joinnext
        }
        parm {
            name        "bgtolerance"
            label       "Deactivate Tolerance"
            type        float
            default     { "0" }
            range       { 0! 1 }
            disablewhen "{ deactivate == 0 }"
        }
        parm {
            name    "prune"
            type    toggle
            nolabel
            default { "0" }
            joinnext
        }
        parm {
            name        "tolerance"
            label       "Prune Tolerance"
            type        float
            default     { "0" }
            range       { 0! 1 }
            disablewhen "{ prune == 0 }"
        }
    }
}
)THEDSFILE";


PRM_Template*
SOP_OpenVDB_Merge::buildTemplates()
{
    static PRM_TemplateBuilder templ("SOP_OpenVDB_Merge.cc"_sh, theDsFile);
    return templ.templates();
}


class SOP_OpenVDB_Merge_Verb : public SOP_NodeVerb
{
public:
    SOP_OpenVDB_Merge_Verb() {}
    virtual ~SOP_OpenVDB_Merge_Verb() {}

    virtual SOP_NodeParms *allocParms() const { return new SOP_OpenVDB_MergeParms(); }
    virtual UT_StringHolder name() const { return SOP_OpenVDB_Merge::theSOPTypeName; }

    virtual CookMode cookMode(const SOP_NodeParms* parms) const { return COOK_GENERIC; }

    virtual void cook(const CookParms& cookparms) const;

    static const SOP_NodeVerb::Register<SOP_OpenVDB_Merge_Verb> theVerb;
};


const SOP_NodeVerb::Register<SOP_OpenVDB_Merge_Verb> SOP_OpenVDB_Merge_Verb::theVerb;


const SOP_NodeVerb *
SOP_OpenVDB_Merge::cookVerb() const
{
    return SOP_OpenVDB_Merge_Verb::theVerb.get();
}


void
SOP_OpenVDB_Merge_Verb::cook(const SOP_NodeVerb::CookParms&) const
{
    // do merge
}
