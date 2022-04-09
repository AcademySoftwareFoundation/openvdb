(* ::Package:: *)

(* ::Title:: *)
(*FogVolume*)


(* ::Subtitle:: *)
(*Convert a level set into a fog volume.*)


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


OpenVDBToFogVolume[vdb_?OpenVDBScalarGridQ, cutoff_ -> regime_?regimeQ] /; levelSetQ[vdb] := 
	Block[{wcutoff},
		wcutoff = If[TrueQ[Positive[cutoff]],
			cutoff,
			halfWidth[vdb]
		];
		
		wcutoff = regimeConvert[vdb, wcutoff, regime -> $worldregime];
		
		vdb["levelSetToFogVolume"[wcutoff]];
		
		vdb
	]


OpenVDBToFogVolume[vdb_] := OpenVDBToFogVolume[vdb, Automatic -> $indexregime]


OpenVDBToFogVolume[vdb_, cutoff_] /; NumericQ[cutoff] || cutoff === Automatic := OpenVDBToFogVolume[vdb, cutoff -> $indexregime]


OpenVDBToFogVolume[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBToFogVolume, 1];


SyntaxInformation[OpenVDBToFogVolume] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBToFogVolume] = $indexregime;


(* ::Subsection::Closed:: *)
(*OpenVDBFogVolume*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBFogVolume[vdb_?OpenVDBScalarGridQ, cutoff_ -> regime_?regimeQ] /; levelSetQ[vdb] := 
	Block[{vdbcopy},
		vdbcopy = OpenVDBCopyGrid[vdb];
		
		OpenVDBToFogVolume[vdbcopy, cutoff -> regime];
		
		vdbcopy
	]


OpenVDBFogVolume[vdb_] := OpenVDBFogVolume[vdb, Automatic -> $indexregime]


OpenVDBFogVolume[vdb_, cutoff_] /; NumericQ[cutoff] || cutoff === Automatic := OpenVDBFogVolume[vdb, cutoff -> $indexregime]


OpenVDBFogVolume[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBFogVolume, 1];


SyntaxInformation[OpenVDBFogVolume] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBFogVolume] = $indexregime;
