(* ::Package:: *)

(* ::Title:: *)
(*Measure*)


(* ::Subtitle:: *)
(*Compute area, volume, genus, Euler characteristic, ...*)


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


OpenVDBArea[vdb_?OpenVDBScalarGridQ, regime_:$worldregime] /; regimeQ[regime] := iOpenVDBArea[vdb, regime]


OpenVDBArea[___] = $Failed;


(* ::Subsection::Closed:: *)
(*iOpenVDBArea*)


iOpenVDBArea[vdb_, ___] /; !levelSetQ[vdb] = Indeterminate;


iOpenVDBArea[_?emptyVDBQ, ___] = 0.;


iOpenVDBArea[vdb_, regime_] :=
	Block[{area, scale},
		area = vdb["levelSetGridArea"[]];
		(
			area = regimeConvert[vdb, area, $worldregime -> regime, 2];
			
			area
			
		) /; NumericQ[area]
	]


iOpenVDBArea[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBArea, 1];


SyntaxInformation[OpenVDBArea] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBArea] = $worldregime;


(* ::Section:: *)
(*OpenVDBEulerCharacteristic*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBEulerCharacteristic[vdb_?OpenVDBScalarGridQ] := iOpenVDBEulerCharacteristic[vdb]


OpenVDBEulerCharacteristic[___] = $Failed;


(* ::Subsection::Closed:: *)
(*iOpenVDBEulerCharacteristic*)


iOpenVDBEulerCharacteristic[vdb_] /; !levelSetQ[vdb] = Indeterminate;


iOpenVDBEulerCharacteristic[_?emptyVDBQ] = Undefined;


iOpenVDBEulerCharacteristic[vdb_] :=
	With[{char = vdb["levelSetGridEulerCharacteristic"[]]},
		char /; IntegerQ[char]
	]


iOpenVDBEulerCharacteristic[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBEulerCharacteristic, 1];


SyntaxInformation[OpenVDBEulerCharacteristic] = {"ArgumentsPattern" -> {_}};


(* ::Section:: *)
(*OpenVDBGenus*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBGenus[vdb_?OpenVDBScalarGridQ] := iOpenVDBGenus[vdb]


OpenVDBGenus[___] = $Failed;


(* ::Subsection::Closed:: *)
(*iOpenVDBGenus*)


iOpenVDBGenus[vdb_] /; !levelSetQ[vdb] = Indeterminate;


iOpenVDBGenus[_?emptyVDBQ] = Undefined;


iOpenVDBGenus[vdb_] :=
	With[{genus = vdb["levelSetGridGenus"[]]},
		genus /; IntegerQ[genus]
	]


iOpenVDBGenus[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBGenus, 1];


SyntaxInformation[OpenVDBGenus] = {"ArgumentsPattern" -> {_}};


(* ::Section:: *)
(*OpenVDBVolume*)


(* ::Subsection::Closed:: *)
(*Main*)


OpenVDBVolume[vdb_?OpenVDBScalarGridQ, regime_:$worldregime] /; regimeQ[regime] := iOpenVDBVolume[vdb, regime]


OpenVDBVolume[___] = $Failed;


(* ::Subsection::Closed:: *)
(*iOpenVDBVolume*)


iOpenVDBVolume[vdb_, ___] /; !levelSetQ[vdb] = Indeterminate;


iOpenVDBVolume[vdb_?emptyVDBQ, ___] := If[TrueQ[Negative[vdb["getBackgroundValue"[]]]], \[Infinity], 0.]


iOpenVDBVolume[vdb_, regime_] :=
	Block[{volume, scale},
		volume = vdb["levelSetGridVolume"[]];
		(
			volume = regimeConvert[vdb, volume, $worldregime -> regime, 3];
			
			volume
			
		) /; NumericQ[volume]
	]


iOpenVDBVolume[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBVolume, 1];


SyntaxInformation[OpenVDBVolume] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBVolume] = $worldregime;
