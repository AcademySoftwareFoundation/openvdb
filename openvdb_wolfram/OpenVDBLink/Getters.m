(* ::Package:: *)

(* ::Title:: *)
(*Getters*)


(* ::Subtitle:: *)
(*Get various properties about a grid.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBProperty"]


OpenVDBProperty::usage = "OpenVDBProperty[expr, \"prop\"] returns the value of property \"prop\" for the given OpenVDB grid.";


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*getterPropertyFunctions*)


$getterPropertyAssoc = <|
    "ActiveLeafVoxelCount" -> openVDBGetActiveLeafVoxelCount,
    "ActiveTileCount" -> openVDBGetActiveTileCount,
    "ActiveVoxelCount" -> openVDBGetActiveVoxelCount,
    "BackgroundValue" -> openVDBGetBackgroundValue,
    "BoundingGridVoxelCount" -> openVDBGetBoundingGridVoxelCount,
    "CreationDate" -> openVDBGetCreationDate,
    "Creator" -> openVDBGetCreator,
    "Description" -> openVDBGetDescription,
    "Empty" -> openVDBGetEmpty,
    "ExpressionID" -> openVDBGetExpressionID,
    "GammaAdjustment" -> openVDBGetGammaAdjustment,
    "GrayscaleWidth" -> openVDBGetGrayscaleWidth,
    "GridClass" -> openVDBGetGridClass,
    "GridType" -> openVDBGetGridType,
    "HalfWidth" -> openVDBGetHalfwidth,
    "IndexBoundingBox" -> openVDBGetGridBoundingBox,
    "IndexDimensions" -> openVDBGetGridDimensions,
    "LastModifiedDate" -> openVDBGetLastModifiedDate,
    "MaxValue" -> openVDBGetMaxValue,
    "MemoryUsage" -> openVDBGetMemoryUsage,
    "MinValue" -> openVDBGetMinValue,
    "MinMaxValues" -> openVDBGetMinMaxValues,
    "Name" -> openVDBGetName,
    "Properties" -> openVDBGetProperties,
    "PropertyValueGrid" -> openVDBGetPropertyValueGrid,
    "UniformVoxels" -> openVDBGetUniformVoxels,
    "VoxelSize" -> openVDBGetVoxelSize,
    "WorldBoundingBox" -> openVDBGetBoundingBox,
    "WorldDimensions" -> openVDBGetDimensions
|>;


$allProperties = DeleteCases[Keys[$getterPropertyAssoc], "Properties" | "PropertyValueGrid" | "MinValue" | "MaxValue"];


getterPropertyFunctions[All] := getterPropertyFunctions[$allProperties]


getterPropertyFunctions[props_] :=
    With[{funcs = Lookup[$getterPropertyAssoc, props, $Failed]},
        funcs /; FreeQ[funcs, $Failed, {1}]
    ]


getterPropertyFunctions[___] = $Failed;


(* ::Subsection::Closed:: *)
(*formatProperties*)


formatProperties[Automatic, prop_String, measurement_] := measurement
formatProperties[Automatic, props_List, measurements_] := formatProperties["List", props, measurements]
formatProperties[Automatic, All, measurements_] := formatProperties["Association", $allProperties, measurements]


formatProperties[format_, prop_String, measurement_] := formatProperties[format, {prop}, {measurement}]
formatProperties[format_, All, measurements_] := formatProperties[format, $allProperties, measurements]


formatProperties["Association", props_List, measurements_] := AssociationThread[props, measurements]


formatProperties["Dataset", props_List, measurements_] := Dataset[formatProperties["Association", props, measurements]]


formatProperties["List", props_List, measurements_] := measurements


formatProperties["RuleList", props_List, measurements_] := Thread[props -> measurements]


formatProperties[___] = $Failed


validReturnFormatQ["Association"] = True;
validReturnFormatQ["Dataset"] = True;
validReturnFormatQ["List"] = True;
validReturnFormatQ["RuleList"] = True;
validReturnFormatQ[Automatic] = True;
validReturnFormatQ[___] = False;


(* ::Subsection::Closed:: *)
(*OpenVDBGrid overload*)


(vdb_OpenVDBGrid)[key:(_List | _String)] :=
    Block[{lookup, res},
        lookup = Lookup[$getterPropertyAssoc, key];
        (
            res = If[ListQ[lookup],
                Through[lookup[vdb]],
                lookup[vdb]
            ];

            res /; res =!= $Failed

        ) /; !MissingQ[lookup]
    ]


(vdb_OpenVDBGrid)[key:(_List | _String), format_] :=
    Block[{res},
        res = OpenVDBProperty[vdb, key, format];

        res /; res =!= $Failed
    ]


(vdb_OpenVDBGrid)[args___] := mOpenVDBProperty[vdb, args]


(* ::Section:: *)
(*OpenVDBProperty*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBProperty[args___] /; !CheckArgs[OpenVDBProperty[args], {2, 3}] = $Failed;


OpenVDBProperty[args___] :=
    With[{res = iOpenVDBProperty[args]},
        res /; res =!= $Failed
    ]


OpenVDBProperty[args___] := mOpenVDBProperty[args]


(* ::Subsection::Closed:: *)
(*iOpenVDBProperty*)


iOpenVDBProperty[vdb_?OpenVDBGridQ, props_, format_:Automatic] :=
    Block[{propfuncs, measurements, res},
        propfuncs = getterPropertyFunctions[props];
        (
            measurements = queryVDBProperty[vdb, propfuncs];
            (
                res = formatProperties[format, props, measurements];

                res /; res =!= $Failed

            ) /; measurements =!= $Failed

        ) /; propfuncs =!= $Failed
    ]


iOpenVDBProperty[___] = $Failed;


queryVDBProperty[vdb_, pfunc_List] := Through[pfunc[vdb]]


queryVDBProperty[vdb_, pfunc_] := pfunc[vdb]


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBProperty, 1];


SyntaxInformation[OpenVDBProperty] = {"ArgumentsPattern" -> {_, _, _.}};


addCodeCompletion[OpenVDBProperty][None, Keys[$getterPropertyAssoc], None];


(* ::Subsection::Closed:: *)
(*Messages*)


mOpenVDBProperty[expr_, ___] /; messageGridQ[expr, OpenVDBProperty] = $Failed;


mOpenVDBProperty[_, props_, ___] /; getterPropertyFunctions[props] === $Failed :=
    (
        If[ListQ[props],
            Message[OpenVDBProperty::props, props],
            Message[OpenVDBProperty::prop, props]
        ];
        $Failed
    )


mOpenVDBProperty[_, _, format_] /; !validReturnFormatQ[format] :=
    (
        Message[OpenVDBProperty::frmt, format, 3];
        $Failed
    )


mOpenVDBProperty[___] = $Failed;


OpenVDBProperty::prop = "`1` is not a valid property. Use \"Properties\" to see the list of available properties.";
OpenVDBProperty::props = "`1` is not a valid list of properties. Use \"Properties\" to see the list of available properties.";
OpenVDBProperty::frmt = "`1` at position `2` is not one of \"Association\", \"Dataset\", \"List\", \"RuleList\", or Automatic."


(* ::Section:: *)
(*Getters*)


(* ::Subsection::Closed:: *)
(*openVDBGetActiveLeafVoxelCount*)


openVDBGetActiveLeafVoxelCount[vdb_] :=
    With[{cnt = vdb["getActiveLeafVoxelCount"[]]},
        cnt /; IntegerQ[cnt] && NonNegative[cnt]
    ]


openVDBGetActiveLeafVoxelCount[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetActiveTileCount*)


openVDBGetActiveTileCount[vdb_] :=
    With[{cnt = vdb["getActiveTileCount"[]]},
        cnt /; IntegerQ[cnt] && NonNegative[cnt]
    ]


openVDBGetActiveTileCount[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetActiveVoxelCount*)


openVDBGetActiveVoxelCount[vdb_] :=
    With[{cnt = vdb["getActiveVoxelCount"[]]},
        cnt /; IntegerQ[cnt] && NonNegative[cnt]
    ]


openVDBGetActiveVoxelCount[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetBackgroundValue*)


openVDBGetBackgroundValue[vdb_?nonMaskGridQ] :=
    With[{bg = vdb["getBackgroundValue"[]]},
        bg /; NumericQ[bg] || ListQ[bg]
    ]


openVDBGetBackgroundValue[_?OpenVDBMaskGridQ] = Missing["NotApplicable"];


openVDBGetBackgroundValue[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetBoundingBox*)


openVDBGetBoundingBox[vdb_] :=
    Block[{griddims, voxsize},
        griddims = openVDBGetGridBoundingBox[vdb];
        (
            voxsize = openVDBGetVoxelSize[vdb];

            voxsize * griddims /; voxsize =!= $Failed

        ) /; griddims =!= $Failed
    ]


openVDBGetBoundingBox[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetBoundingGridVoxelCount*)


openVDBGetBoundingGridVoxelCount[vdb_] :=
    With[{dims = openVDBGetGridDimensions[vdb]},
        Times @@ dims /; dims =!= $Failed
    ]


openVDBGetBoundingGridVoxelCount[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetCreationDate*)


openVDBGetCreationDate[vdb_] :=
    Block[{res},
        res = Quiet @ vdb["getIntegerMetadata"["creation_date"]];
        If[IntegerQ[res],
            FromUnixTime[res],
            Missing["NotAvailable"]
        ]
    ]


(* ::Subsection::Closed:: *)
(*openVDBGetCreator*)


openVDBGetCreator[vdb_] :=
    With[{name = vdb["getCreator"[]]},
        If[StringQ[name] && StringLength[name] > 0,
            name,
            Missing["NotAvailable"]
        ]
    ]


openVDBGetCreator[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetDescription*)


openVDBGetDescription[vdb_] :=
    Block[{res},
        res = Quiet @ vdb["getStringMetadata"["description"]];
        If[StringQ[res] && StringLength[res] > 0,
            res,
            Missing["NotAvailable"]
        ]
    ]


(* ::Subsection::Closed:: *)
(*openVDBGetDimensions*)


openVDBGetDimensions[vdb_] :=
    Block[{griddims, voxsize},
        griddims = openVDBGetGridDimensions[vdb];
        (
            voxsize = openVDBGetVoxelSize[vdb];

            voxsize * griddims /; voxsize =!= $Failed

        ) /; griddims =!= $Failed
    ]


openVDBGetDimensions[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetEmpty*)


openVDBGetEmpty[vdb_] := emptyVDBQ[vdb]


(* ::Subsection::Closed:: *)
(*openVDBGetExpressionID*)


openVDBGetExpressionID[_[id_, ___]] := id


(* ::Subsection::Closed:: *)
(*openVDBGetGammaAdjustment*)


openVDBGetGammaAdjustment[vdb_?fogVolumeQ] :=
    Block[{res},
        res = Quiet @ vdb["getRealMetadata"["gamma_adjustment"]];
        If[NumberQ[res],
            res,
            Missing["NotAvailable"]
        ]
    ]


openVDBGetGammaAdjustment[___] = Missing["NotApplicable"];


(* ::Subsection::Closed:: *)
(*openVDBGetGrayscaleWidth*)


openVDBGetGrayscaleWidth[vdb_?fogVolumeQ] :=
    Block[{res},
        res = Quiet[Divide[vdb["getRealMetadata"["cutoff_distance"]], vdb["getRealMetadata"["scaling_factor"]]]];
        If[NumberQ[res],
            res,
            Missing["NotAvailable"]
        ]
    ]


openVDBGetGrayscaleWidth[___] = Missing["NotApplicable"];


(* ::Subsection::Closed:: *)
(*openVDBGetGridBoundingBox*)


openVDBGetGridBoundingBox[vdb_] :=
    With[{bbox = vdb["getGridBoundingBox"[]]},
        bbox /; MatrixQ[bbox, IntegerQ] && Dimensions[bbox] === {3, 2}
    ]


openVDBGetGridBoundingBox[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetGridClass*)


openVDBGetGridClass[vdb_ /; !OpenVDBScalarGridQ[vdb]] = Missing["NotApplicable"];


openVDBGetGridClass[vdb_] :=
    With[{gc = vdb["getGridClass"[]]},
        gridClassName[gc] /; IntegerQ[gc]
    ]


openVDBGetGridClass[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetGridDimensions*)


openVDBGetGridDimensions[vdb_] :=
    With[{bbox = vdb["getGridDimensions"[]]},
        bbox /; VectorQ[bbox, IntegerQ] && Length[bbox] === 3
    ]


openVDBGetGridDimensions[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetGridType*)


openVDBGetGridType[vdb_] :=
    With[{type = vdb["getGridType"[]]},
        If[StringQ[type],
            type,
            Missing["NotAvailable"]
        ]
    ]


openVDBGetGridType[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetHalfwidth*)


openVDBGetHalfwidth[vdb_?levelSetQ] :=
    With[{hw = halfWidth[vdb]},
        If[TrueQ[Positive[hw]],
            hw,
            Missing["NotAvailable"]
        ]
    ]


openVDBGetHalfwidth[___] = Missing["NotApplicable"];


(* ::Subsection::Closed:: *)
(*openVDBGetLastModifiedDate*)


openVDBGetLastModifiedDate[vdb_] :=
    Block[{res},
        res = Quiet @ vdb["getIntegerMetadata"["last_modified_date"]];
        If[IntegerQ[res],
            FromUnixTime[res],
            Missing["NotAvailable"]
        ]
    ]


(* ::Subsection::Closed:: *)
(*openVDBGetMaxValue*)


openVDBGetMaxValue[vdb_?nonMaskGridQ] :=
    With[{minmax = openVDBGetMinMaxValues[vdb]},
        minmax[[2]] /; ListQ[minmax]
    ]


openVDBGetMaxValue[_?OpenVDBMaskGridQ] = Missing["NotApplicable"];


openVDBGetMaxValue[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetMemoryUsage*)


openVDBGetMemoryUsage[vdb_] :=
    With[{cnt = vdb["getMemoryUsage"[]]},
        cnt /; IntegerQ[cnt] && NonNegative[cnt]
    ]


openVDBGetMemoryUsage[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetMinValue*)


openVDBGetMinValue[vdb_?nonMaskGridQ] :=
    With[{minmax = openVDBGetMinMaxValues[vdb]},
        minmax[[1]] /; ListQ[minmax]
    ]


openVDBGetMinValue[_?OpenVDBMaskGridQ] = Missing["NotApplicable"];


openVDBGetMinValue[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetMinMaxValues*)


openVDBGetMinMaxValues[vdb_?nonMaskGridQ] :=
    With[{minmax = vdb["getMinMaxValues"[]]},
        minmax /; ArrayQ[minmax, _, NumericQ] && Length[minmax] === 2
    ]


openVDBGetMinMaxValues[_?OpenVDBMaskGridQ] = Missing["NotApplicable"];


openVDBGetMinMaxValues[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetName*)


openVDBGetName[vdb_] :=
    With[{name = vdb["getName"[]]},
        If[StringQ[name] && StringLength[name] > 0,
            name,
            Missing["NotAvailable"]
        ]
    ]


openVDBGetName[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetProperties*)


openVDBGetProperties[_] := Keys[$getterPropertyAssoc]


(* ::Subsection::Closed:: *)
(*openVDBGetPropertyValueGrid*)


openVDBGetPropertyValueGrid[vdb_] :=
    Grid[
        DeleteCases[List @@@ OpenVDBProperty[vdb, All, "RuleList"], {_, Missing["NotApplicable"]}],
        Alignment -> {{Right, Left}},
        Frame -> All,
        Spacings -> {1, 0.75}
    ]


(* ::Subsection::Closed:: *)
(*openVDBGetUniformVoxels*)


openVDBGetUniformVoxels[vdb_] :=
    With[{uniform = vdb["getHasUniformVoxels"[]]},
        uniform /; BooleanQ[uniform]
    ]


openVDBGetUniformVoxels[___] = $Failed;


(* ::Subsection::Closed:: *)
(*openVDBGetVoxelSize*)


openVDBGetVoxelSize[vdb_] :=
    With[{vx = voxelSize[vdb]},
        vx /; NumericQ[vx]
    ]


openVDBGetVoxelSize[___] = $Failed;
