(* ::Package:: *)

(* ::Title:: *)
(*FogVolume*)


(* ::Subtitle:: *)
(*Convert a level set into a fog volume.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBToFogVolume"]
PackageExport["OpenVDBFogVolume"]


OpenVDBToFogVolume::usage = "OpenVDBToFogVolume[expr] modifies the scalar level set expr by converting it to a fog volume representation.";
OpenVDBFogVolume::usage = "OpenVDBFogVolume[expr] creates a fog volume representation of expr and stores the result to a new OpenVDB grid.";


(* ::Section:: *)
(*FogVolume*)


(* ::Subsection::Closed:: *)
(*OpenVDBToFogVolume*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBToFogVolume[args___] /; !CheckArgs[OpenVDBToFogVolume[args], {1, 2}] = $Failed;


OpenVDBToFogVolume[args___] :=
    With[{res = iOpenVDBToFogVolume[args]},
        res /; res =!= $Failed
    ]


OpenVDBToFogVolume[args___] := mOpenVDBToFogVolume[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBToFogVolume*)


iOpenVDBToFogVolume[vdb_?OpenVDBScalarGridQ, cutoff_ -> regime_?regimeQ] /; levelSetQ[vdb] :=
    Block[{wcutoff},
        wcutoff = If[TrueQ[Positive[cutoff]],
            cutoff,
            halfWidth[vdb]
        ];

        wcutoff = regimeConvert[vdb, wcutoff, regime -> $worldregime];

        vdb["levelSetToFogVolume"[wcutoff]];

        vdb
    ]


iOpenVDBToFogVolume[vdb_] := iOpenVDBToFogVolume[vdb, Automatic -> $indexregime]


iOpenVDBToFogVolume[vdb_, cutoff_] /; NumericQ[cutoff] || cutoff === Automatic := iOpenVDBToFogVolume[vdb, cutoff -> $indexregime]


iOpenVDBToFogVolume[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBToFogVolume, 1];


SyntaxInformation[OpenVDBToFogVolume] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBToFogVolume] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBToFogVolume[expr_, ___] /; messageScalarGridQ[expr, OpenVDBToFogVolume] = $Failed;


mOpenVDBToFogVolume[expr_, ___] /; messageLevelSetGridQ[expr, OpenVDBToFogVolume] = $Failed;


mOpenVDBToFogVolume[___] = $Failed;


(* ::Subsection::Closed:: *)
(*OpenVDBFogVolume*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBFogVolume[args___] /; !CheckArgs[OpenVDBFogVolume[args], {1, 2}] = $Failed;


OpenVDBFogVolume[args___] :=
    With[{res = iOpenVDBFogVolume[args]},
        res /; res =!= $Failed
    ]


OpenVDBFogVolume[args___] := mOpenVDBFogVolume[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBFogVolume*)


iOpenVDBFogVolume[vdb_?OpenVDBScalarGridQ, cutoff_ -> regime_?regimeQ] /; levelSetQ[vdb] :=
    Block[{vdbcopy},
        vdbcopy = OpenVDBCopyGrid[vdb];

        iOpenVDBToFogVolume[vdbcopy, cutoff -> regime];

        vdbcopy
    ]


iOpenVDBFogVolume[vdb_] := iOpenVDBFogVolume[vdb, Automatic -> $indexregime]


iOpenVDBFogVolume[vdb_, cutoff_] /; NumericQ[cutoff] || cutoff === Automatic := iOpenVDBFogVolume[vdb, cutoff -> $indexregime]


iOpenVDBFogVolume[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBFogVolume, 1];


SyntaxInformation[OpenVDBFogVolume] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBFogVolume] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBFogVolume[expr_, ___] /; messageScalarGridQ[expr, OpenVDBFogVolume] = $Failed;


mOpenVDBFogVolume[expr_, ___] /; messageLevelSetGridQ[expr, OpenVDBFogVolume] = $Failed;


mOpenVDBFogVolume[___] = $Failed;
