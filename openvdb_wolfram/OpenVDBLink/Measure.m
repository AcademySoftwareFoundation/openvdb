(* ::Package:: *)

(* ::Title:: *)
(*Measure*)


(* ::Subtitle:: *)
(*Compute area, volume, genus, Euler characteristic, ...*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBArea"]
PackageExport["OpenVDBEulerCharacteristic"]
PackageExport["OpenVDBGenus"]
PackageExport["OpenVDBVolume"]


OpenVDBArea::usage = "OpenVDBArea[expr] finds the surface area of an OpenVDB level set.";
OpenVDBEulerCharacteristic::usage = "OpenVDBEulerCharacteristic[expr] finds the Euler characteristic of an OpenVDB level set.";
OpenVDBGenus::usage = "OpenVDBGenus[expr] finds the genus of an OpenVDB level set.";
OpenVDBVolume::usage = "OpenVDBVolume[expr] finds the volume of an OpenVDB level set.";


(* ::Section:: *)
(*OpenVDBArea*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBArea[args___] /; !CheckArgs[OpenVDBArea[args], {1, 2}] = $Failed;


OpenVDBArea[args___] :=
    With[{res = iOpenVDBArea[args]},
        res /; res =!= $Failed
    ]


OpenVDBArea[args___] := mOpenVDBArea[args]


(* ::Subsection::Closed:: *)
(*iOpenVDBArea*)


iOpenVDBArea[vdb_?OpenVDBScalarGridQ, regime_?regimeQ] :=
    Block[{area, scale},
        area = scalarArea[vdb];
        (
            area = regimeConvert[vdb, area, $worldregime -> regime, 2];

            area

        ) /; area =!= $Failed
    ]


iOpenVDBArea[vdb_] := iOpenVDBArea[vdb, $worldregime]


iOpenVDBArea[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBArea, 1];


SyntaxInformation[OpenVDBArea] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBArea] = $worldregime;


(* ::Subsection::Closed:: *)
(*Utilities*)


scalarArea[vdb_?emptyVDBQ] = 0.


scalarArea[vdb_?levelSetQ] :=
    With[{vol = vdb["levelSetGridArea"[]]},
        vol /; NumericQ[vol]
    ]


scalarArea[vdb_] /; !levelSetQ[vdb] = Indeterminate;


scalarArea[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Messages*)


mOpenVDBArea[expr_, ___] /; messageScalarGridQ[expr, OpenVDBArea] = $Failed;


mOpenVDBArea[_, regime_] /; messageRegimeSpecQ[regime, OpenVDBArea] = $Failed;


mOpenVDBArea[___] = $Failed;


(* ::Section:: *)
(*OpenVDBEulerCharacteristic*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBEulerCharacteristic[args___] /; !CheckArgs[OpenVDBEulerCharacteristic[args], 1] = $Failed;


OpenVDBEulerCharacteristic[args___] :=
    With[{res = iOpenVDBEulerCharacteristic[args]},
        res /; res =!= $Failed
    ]


OpenVDBEulerCharacteristic[args___] := mOpenVDBEulerCharacteristic[args]


(* ::Subsection::Closed:: *)
(*iOpenVDBEulerCharacteristic*)


iOpenVDBEulerCharacteristic[vdb_?OpenVDBScalarGridQ] :=
    Block[{char},
        char = Which[
            !levelSetQ[vdb], Indeterminate,
            emptyVDBQ[vdb], Undefined,
            True, Replace[vdb["levelSetGridEulerCharacteristic"[]], Except[_Integer] -> $Failed, {0}]
        ];

        char /; char =!= $Failed
    ]


iOpenVDBEulerCharacteristic[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBEulerCharacteristic, 1];


SyntaxInformation[OpenVDBEulerCharacteristic] = {"ArgumentsPattern" -> {_}};


(* ::Subsection::Closed:: *)
(*Messages*)


mOpenVDBEulerCharacteristic[expr_] /; messageScalarGridQ[expr, OpenVDBEulerCharacteristic] = $Failed;


mOpenVDBEulerCharacteristic[___] = $Failed;


(* ::Section:: *)
(*OpenVDBGenus*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBGenus[args___] /; !CheckArgs[OpenVDBGenus[args], 1] = $Failed;


OpenVDBGenus[args___] :=
    With[{res = iOpenVDBGenus[args]},
        res /; res =!= $Failed
    ]


OpenVDBGenus[args___] := mOpenVDBGenus[args]


(* ::Subsection::Closed:: *)
(*iOpenVDBGenus*)


iOpenVDBGenus[vdb_?OpenVDBScalarGridQ] :=
    Block[{genus},
        genus = Which[
            !levelSetQ[vdb], Indeterminate,
            emptyVDBQ[vdb], Undefined,
            True, Replace[vdb["levelSetGridGenus"[]], Except[_Integer] -> $Failed, {0}]
        ];

        genus /; genus =!= $Failed
    ]


iOpenVDBGenus[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBGenus, 1];


SyntaxInformation[OpenVDBGenus] = {"ArgumentsPattern" -> {_}};


(* ::Subsection::Closed:: *)
(*Messages*)


mOpenVDBGenus[expr_, ___] /; messageScalarGridQ[expr, OpenVDBGenus] = $Failed;


mOpenVDBGenus[___] = $Failed;


(* ::Section:: *)
(*OpenVDBVolume*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBVolume[args___] /; !CheckArgs[OpenVDBVolume[args], {1, 2}] = $Failed;


OpenVDBVolume[args___] :=
    With[{res = iOpenVDBVolume[args]},
        res /; res =!= $Failed
    ]


OpenVDBVolume[args___] := mOpenVDBVolume[args]


(* ::Subsection::Closed:: *)
(*iOpenVDBVolume*)


iOpenVDBVolume[vdb_?OpenVDBScalarGridQ, regime_?regimeQ] :=
    Block[{volume, scale},
        volume = scalarVolume[vdb];
        (
            volume = regimeConvert[vdb, volume, $worldregime -> regime, 3];

            volume

        ) /; volume =!= $Failed
    ]


iOpenVDBVolume[vdb_] := iOpenVDBVolume[vdb, $worldregime]


iOpenVDBVolume[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBVolume, 1];


SyntaxInformation[OpenVDBVolume] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBVolume] = $worldregime;


(* ::Subsection::Closed:: *)
(*Utilities*)


scalarVolume[vdb_?emptyVDBQ] := If[TrueQ[Negative[vdb["getBackgroundValue"[]]]], \[Infinity], 0.]


scalarVolume[vdb_?levelSetQ] :=
    With[{vol = vdb["levelSetGridVolume"[]]},
        vol /; NumericQ[vol]
    ]


scalarVolume[vdb_?fogVolumeQ] :=
    With[{tots = OpenVDBActiveVoxelSliceTotals[vdb]},
        (
            Total[tots] * vdb["VoxelSize"]^3

        ) /; VectorQ[tots, NumericQ]
    ]


scalarVolume[vdb_] /; !levelSetQ[vdb] && !fogVolumeQ[vdb] = Indeterminate;


scalarVolume[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Messages*)


mOpenVDBVolume[expr_, ___] /; messageScalarGridQ[expr, OpenVDBVolume] = $Failed;


mOpenVDBVolume[_, regime_] /; messageRegimeSpecQ[regime, OpenVDBVolume] = $Failed;


mOpenVDBVolume[___] = $Failed;
