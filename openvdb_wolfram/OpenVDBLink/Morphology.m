(* ::Package:: *)

(* ::Title:: *)
(*Morphology*)


(* ::Subtitle:: *)
(*Resize bandwidth, erode, dilate, ...*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBResizeBandwidth"]
PackageExport["OpenVDBDilation"]
PackageExport["OpenVDBErosion"]
PackageExport["OpenVDBClosing"]
PackageExport["OpenVDBOpening"]


OpenVDBResizeBandwidth::usage = "OpenVDBResizeBandwidth[expr, width] resizes the width of the narrow band of a level set.";
OpenVDBDilation::usage = "OpenVDBDilation[expr, r] gives the morphological dilation of a level set.";
OpenVDBErosion::usage = "OpenVDBErosion[expr, r] gives the morphological erosion of a level set.";
OpenVDBClosing::usage = "OpenVDBClosing[expr, r] gives the morphological closing of a level set.";
OpenVDBOpening::usage = "OpenVDBOpening[expr, r] gives the morphological opening of a level set.";


(* ::Section:: *)
(*Bandwidth*)


(* ::Subsection::Closed:: *)
(*OpenVDBResizeBandwidth*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBResizeBandwidth[args___] /; !CheckArgs[OpenVDBResizeBandwidth[args], {1, 2}] = $Failed;


OpenVDBResizeBandwidth[args___] :=
    With[{res = iOpenVDBResizeBandwidth[args]},
        res /; res =!= $Failed
    ]


OpenVDBResizeBandwidth[args___] := mOpenVDBResizeBandwidth[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBResizeBandwidth*)


iOpenVDBResizeBandwidth[vdb_?OpenVDBScalarGridQ, width_?Positive -> regime_?regimeQ] /; levelSetQ[vdb] :=
    Block[{w},
        w = regimeConvert[vdb, width, regime -> $indexregime];
        (
            vdb["resizeBandwidth"[w]];
            vdb
        ) /; w > 0
    ]


iOpenVDBResizeBandwidth[vdb_, width_?Positive] := iOpenVDBResizeBandwidth[vdb, width -> $indexregime]


iOpenVDBResizeBandwidth[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBResizeBandwidth, 1];


SyntaxInformation[OpenVDBResizeBandwidth] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBResizeBandwidth] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBResizeBandwidth[args___] := messageMorphologyFunction[OpenVDBResizeBandwidth, args]


(* ::Section:: *)
(*Morphology*)


(* ::Subsection::Closed:: *)
(*OpenVDBDilation*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBDilation[args___] /; !CheckArgs[OpenVDBDilation[args], {1, 2}] = $Failed;


OpenVDBDilation[args___] :=
    With[{res = iOpenVDBDilation[args]},
        res /; res =!= $Failed
    ]


OpenVDBDilation[args___] := mOpenVDBDilation[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBDilation*)


iOpenVDBDilation[vdb_?OpenVDBScalarGridQ, r_?realQ -> regime_?regimeQ] /; levelSetQ[vdb] :=
    Block[{offset},
        offset = regimeConvert[vdb, -r, regime -> $worldregime];

        vdb["offsetLevelSet"[offset]];

        vdb
    ]


iOpenVDBDilation[vdb_, r_?realQ] := iOpenVDBDilation[vdb, r -> $worldregime]


iOpenVDBDilation[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBDilation, 1];


SyntaxInformation[OpenVDBDilation] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBDilation] = $worldregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBDilation[args___] := messageMorphologyFunction[OpenVDBDilation, args]


(* ::Subsection::Closed:: *)
(*OpenVDBErosion*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBErosion[args___] /; !CheckArgs[OpenVDBErosion[args], {1, 2}] = $Failed;


OpenVDBErosion[args___] :=
    With[{res = iOpenVDBErosion[args]},
        res /; res =!= $Failed
    ]


OpenVDBErosion[args___] := mOpenVDBErosion[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBErosion*)


iOpenVDBErosion[vdb_?OpenVDBScalarGridQ, r_?realQ -> regime_?regimeQ] /; levelSetQ[vdb] :=
    Block[{offset},
        offset = regimeConvert[vdb, r, regime -> $worldregime];

        vdb["offsetLevelSet"[offset]];

        vdb
    ]


iOpenVDBErosion[vdb_, r_?realQ] := iOpenVDBErosion[vdb, r -> $worldregime]


iOpenVDBErosion[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBErosion, 1];


SyntaxInformation[OpenVDBErosion] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBErosion] = $worldregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBErosion[args___] := messageMorphologyFunction[OpenVDBErosion, args]


(* ::Subsection::Closed:: *)
(*OpenVDBClosing*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBClosing[args___] /; !CheckArgs[OpenVDBClosing[args], {1, 2}] = $Failed;


OpenVDBClosing[args___] :=
    With[{res = iOpenVDBClosing[args]},
        res /; res =!= $Failed
    ]


OpenVDBClosing[args___] := mOpenVDBClosing[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBClosing*)


iOpenVDBClosing[vdb_?OpenVDBScalarGridQ, r_?NonNegative -> regime_?regimeQ] /; levelSetQ[vdb] :=
    iOpenVDBErosion[iOpenVDBDilation[vdb, r -> regime], r -> regime]


iOpenVDBClosing[vdb_, r_?NonNegative] := iOpenVDBClosing[vdb, r -> $worldregime]


iOpenVDBClosing[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBClosing, 1];


SyntaxInformation[OpenVDBClosing] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBClosing] = $worldregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBClosing[args___] := messageMorphologyFunction[OpenVDBClosing, args]


(* ::Subsection::Closed:: *)
(*OpenVDBOpening*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBOpening[args___] /; !CheckArgs[OpenVDBOpening[args], {1, 2}] = $Failed;


OpenVDBOpening[args___] :=
    With[{res = iOpenVDBOpening[args]},
        res /; res =!= $Failed
    ]


OpenVDBOpening[args___] := mOpenVDBOpening[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBOpening*)


iOpenVDBOpening[vdb_?OpenVDBScalarGridQ, r_?NonNegative -> regime_?regimeQ] /; levelSetQ[vdb] :=
    iOpenVDBDilation[iOpenVDBErosion[vdb, r -> regime], r -> regime]


iOpenVDBOpening[vdb_, r_?NonNegative] := iOpenVDBOpening[vdb, r -> $worldregime]


iOpenVDBOpening[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBOpening, 1];


SyntaxInformation[OpenVDBOpening] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBOpening] = $worldregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBOpening[args___] := messageMorphologyFunction[OpenVDBOpening, args]


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*messageMorphologyFunction*)


messageMorphologyFunction[head_, expr_, ___] /; messageScalarGridQ[expr, head] = $Failed;


messageMorphologyFunction[head_, expr_, ___] /; messageLevelSetGridQ[expr, head] = $Failed;


messageMorphologyFunction[head_, _, w_] /; messagePositiveDistanceQ[w, 2, head] = $Failed;


messageMorphologyFunction[___] = $Failed;


messagePositiveDistanceQ[w_ -> regime_, pos_, head_] := messageRegimeSpecQ[regime, head] || messagePositiveDistanceQ[w, pos, head]


messagePositiveDistanceQ[w_, pos_, head_] /; !TrueQ[NonNegative[w]] :=
    (
        Message[head::nonneg, w, pos];
        True
    )


messagePositiveDistanceQ[___] = False;


General::nonneg = "`1` at position `2` should be a non\[Hyphen]negative number.";
