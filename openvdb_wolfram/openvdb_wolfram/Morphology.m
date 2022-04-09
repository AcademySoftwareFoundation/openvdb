(* ::Package:: *)

(* ::Title:: *)
(*Morphology*)


(* ::Subtitle:: *)
(*Resize bandwidth, erode, dilate, ...*)


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


OpenVDBResizeBandwidth[vdb_?OpenVDBScalarGridQ, width_?Positive -> regime_?regimeQ] /; levelSetQ[vdb] :=
	Block[{w},
		w = regimeConvert[vdb, width, regime -> $indexregime];
		(
			vdb["resizeBandwidth"[w]];
			vdb
		) /; w > 0
	]


OpenVDBResizeBandwidth[vdb_, width_?Positive] := OpenVDBResizeBandwidth[vdb, width -> $indexregime]


OpenVDBResizeBandwidth[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBResizeBandwidth, 1];


SyntaxInformation[OpenVDBResizeBandwidth] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBResizeBandwidth] = $indexregime;


(* ::Section:: *)
(*Morphology*)


(* ::Subsection::Closed:: *)
(*OpenVDBDilation*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBDilation[vdb_?OpenVDBScalarGridQ, r_?realQ -> regime_?regimeQ] /; levelSetQ[vdb] :=
	Block[{offset},
		offset = regimeConvert[vdb, -r, regime -> $worldregime];
		
		vdb["offsetLevelSet"[offset]];
		
		vdb
	]


OpenVDBDilation[vdb_, r_?realQ] := OpenVDBDilation[vdb, r -> $worldregime]


OpenVDBDilation[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBDilation, 1];


SyntaxInformation[OpenVDBDilation] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBDilation] = $worldregime;


(* ::Subsection::Closed:: *)
(*OpenVDBErosion*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBErosion[vdb_?OpenVDBScalarGridQ, r_?realQ -> regime_?regimeQ] /; levelSetQ[vdb] :=
	Block[{offset},
		offset = regimeConvert[vdb, r, regime -> $worldregime];
		
		vdb["offsetLevelSet"[offset]];
		
		vdb
	]


OpenVDBErosion[vdb_, r_?realQ] := OpenVDBErosion[vdb, r -> $worldregime]


OpenVDBErosion[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBErosion, 1];


SyntaxInformation[OpenVDBErosion] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBErosion] = $worldregime;


(* ::Subsection::Closed:: *)
(*OpenVDBClosing*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBClosing[vdb_?OpenVDBScalarGridQ, r_?NonNegative -> regime_?regimeQ] /; levelSetQ[vdb] := 
	OpenVDBErosion[OpenVDBDilation[vdb, r -> regime], r -> regime]


OpenVDBClosing[vdb_, r_?NonNegative] := OpenVDBClosing[vdb, r -> $worldregime]


OpenVDBClosing[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBClosing, 1];


SyntaxInformation[OpenVDBClosing] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBClosing] = $worldregime;


(* ::Subsection::Closed:: *)
(*OpenVDBOpening*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBOpening[vdb_?OpenVDBScalarGridQ, r_?NonNegative -> regime_?regimeQ] /; levelSetQ[vdb] := 
	OpenVDBDilation[OpenVDBErosion[vdb, r -> regime], r -> regime]


OpenVDBOpening[vdb_, r_?NonNegative] := OpenVDBOpening[vdb, r -> $worldregime]


OpenVDBOpening[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBOpening, 1];


SyntaxInformation[OpenVDBOpening] = {"ArgumentsPattern" -> {_, _}};


OpenVDBDefaultSpace[OpenVDBOpening] = $worldregime;
