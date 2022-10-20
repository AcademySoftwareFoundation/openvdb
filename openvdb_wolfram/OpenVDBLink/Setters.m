(* ::Package:: *)

(* ::Title:: *)
(*Setters*)


(* ::Subtitle:: *)
(*Set various properties about a grid.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBSetProperty"]


OpenVDBSetProperty::usage = "OpenVDBSetProperty[expr, \"prop\", val] sets the value of property \"prop\" to val for the given OpenVDB grid.";


(* ::Section:: *)
(*OpenVDBSetProperty*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBSetProperty[args___] /; !CheckArgs[OpenVDBSetProperty[args], {2, 3}] = $Failed;


OpenVDBSetProperty[args___] :=
    With[{res = iOpenVDBSetProperty[args]},
        res /; res =!= $Failed
    ]


OpenVDBSetProperty[args___] := mOpenVDBSetProperty[args]


(* ::Subsection::Closed:: *)
(*iOpenVDBSetProperty*)


iOpenVDBSetProperty[vdb_?OpenVDBGridQ, args__] :=
    Block[{parsedargs, res},
        parsedargs = setterArguments[args];
        (
            res = MapThread[setVDBProperty[vdb, ##]&, parsedargs];

            If[MatchQ[{args}, {_String, _} | {_String -> _}],
                res[[1]],
                res
            ]

        ) /; parsedargs =!= $Failed
    ]


iOpenVDBSetProperty[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBSetProperty] = {"ArgumentsPattern" -> {_, _, _.}};


(* ::Subsection::Closed:: *)
(*Messages*)


mOpenVDBSetProperty[expr_, ___] /; messageGridQ[expr, OpenVDBSetProperty, False] = $Failed;


mOpenVDBSetProperty[_, args__] /; setterArguments[args] === $Failed :=
    (
        Message[OpenVDBSetProperty::spec];
        $Failed
    )


mOpenVDBSetProperty[_, args__] :=
    Block[{parsed, props},
        parsed = setterArguments[args][[1]];
        props = Pick[parsed, Lookup[$setterPropertyAssoc, parsed, $Failed], $Failed];
        (
            Message[OpenVDBSetProperty::prop, First[props]];
            $Failed
        ) /; Length[props] > 0
    ]


mOpenVDBSetProperty[___] = $Failed;


OpenVDBSetProperty::spec = "Invalid property\[Hyphen]value specification.";


OpenVDBSetProperty::prop = "`1` is not a valid property to set.";


(* ::Section:: *)
(*Setters*)


(* ::Subsection::Closed:: *)
(*openVDBSetBackgroundValue*)


openVDBSetBackgroundValue[vdb_?OpenVDBBooleanGridQ, bool_?BooleanQ] := openVDBSetBackgroundValue[vdb, Boole[bool]]


openVDBSetBackgroundValue[vdb_?nonMaskGridQ, bg_] :=
    With[{res = Quiet[Check[vdb["setBackgroundValue"[bg]];True, False]]},
        Which[
            OpenVDBBooleanGridQ[vdb],
                Unitize[bg],
            Precision[vdb["getBackgroundValue"[]]] =!= \[Infinity],
                N[bg],
            True,
                bg
        ] /; res
    ]


openVDBSetBackgroundValue[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetCreator*)


openVDBSetCreator[vdb_, creator_String] :=
    (
        vdb["setCreator"[creator]];
        creator
    )


openVDBSetCreator[vdb_, None] := openVDBSetCreator[vdb, ""]


openVDBSetCreator[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetDescription*)


openVDBSetDescription[vdb_, description_String] :=
    (
        vdb["setStringMetadata"["description", description]];
        description
    )


openVDBSetDescription[vdb_, None] := openVDBSetDescription[vdb, ""]


openVDBSetDescription[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetGridClass*)


openVDBSetGridClass[vdb_?OpenVDBScalarGridQ, gc_] :=
    With[{gcid = gridClassID[gc]},
        vdb["setGridClass"[gcid]];
        gridClassName[gcid]
    ]


openVDBSetGridClass[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetName*)


openVDBSetName[vdb_, name_String] :=
    (
        vdb["setName"[name]];
        name
    )


openVDBSetName[vdb_, None] := openVDBSetName[vdb, ""]


openVDBSetName[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBSetVoxelSize*)


openVDBSetVoxelSize[vdb_, spacing_?Positive] :=
    (
        vdb["setVoxelSize"[spacing]];
        N[spacing]
    )


openVDBSetVoxelSize[___] = $Failed;


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*setterArguments*)


setterArguments[prop_String, val_] := {{prop}, {val}}
setterArguments[props_List, vals_List] /; Length[props] == Length[vals] := {props, vals}
setterArguments[prop_String -> val_] := {{prop}, {val}}
setterArguments[props_List -> vals_List] /; Length[props] == Length[vals] := {props, vals}
setterArguments[data:{(_ -> _)..}] := {data[[All, 1]], data[[All, 2]]}
setterArguments[___] = $Failed;


(* ::Subsection::Closed:: *)
(*setterPropertyFunctions*)


$setterPropertyAssoc = <|
    "BackgroundValue" -> openVDBSetBackgroundValue,
    "Creator" -> openVDBSetCreator,
    "Description" -> openVDBSetDescription,
    "GridClass" -> openVDBSetGridClass,
    "Name" -> openVDBSetName,
    "VoxelSize" -> openVDBSetVoxelSize
|>;


setVDBProperty[id_, prop_, val_] :=
    With[{setter = Lookup[$setterPropertyAssoc, prop, $Failed]},
        setter[id, val] /; setter =!= $Failed
    ]


setVDBProperty[__] = $Failed;
