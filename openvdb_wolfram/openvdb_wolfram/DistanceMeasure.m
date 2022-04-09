(* ::Package:: *)

(* ::Title:: *)
(*DistanceMeasure*)


(* ::Subtitle:: *)
(*Perform membership, nearest, and distance queries on a grid.*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBMember"]
PackageExport["OpenVDBNearest"]
PackageExport["OpenVDBDistance"]
PackageExport["OpenVDBSignedDistance"]
PackageExport["OpenVDBFillWithBalls"]


OpenVDBMember::usage = "OpenVDBMember[expr, pt] determines if the point pt lies within the region given by a scalar grid.";
OpenVDBNearest::usage = "OpenVDBNearest[expr, pt] finds closest point to pt on the iso surface of a scalar grid.";
OpenVDBDistance::usage = "OpenVDBDistance[expr, pt] finds minimum distance from pt to the iso surface of a scalar grid.";
OpenVDBSignedDistance::usage = "OpenVDBSignedDistance[expr, pt] finds minimum distance from pt to the iso surface of a scalar grid and returns a negative value if pt lies within the region given by the grid.";
OpenVDBFillWithBalls::usage = "OpenVDBFillWithBalls[expr, n, {rmin, rmax}] fills a closed scalar grid with up to n adaptively-sized balls with radii between rmin and rmax.";


(* ::Section:: *)
(*Membership*)


(* ::Subsection::Closed:: *)
(*OpenVDBMember*)


(* ::Subsubsection::Closed:: *)
(*Main*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[OpenVDBMember] = {"IsoValue" -> Automatic};


OpenVDBMember[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, OptionsPattern[]] := 
	Block[{isovalue, mems},
		isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
		(
			mems = vdb["gridMember"[pts, isovalue]];
			
			If[fogVolumeQ[vdb], Subtract[1, mems], mems] /; VectorQ[mems, IntegerQ]
			
		) /; realQ[isovalue]
	]


OpenVDBMember[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, opts___] := 
	OpenVDBMember[vdb, regimeConvert[vdb, pts, $worldregime -> $indexregime] -> $indexregime, opts]


OpenVDBMember[vdb_, pts_?coordinateQ -> regime_, opts___] := 
	With[{res = OpenVDBMember[vdb, {pts} -> regime, opts]},
		res[[1]] /; VectorQ[res, IntegerQ] && Length[res] === 1
	]


OpenVDBMember[vdb_, pts_List, opts___] := OpenVDBMember[vdb, pts -> $worldregime, opts]


OpenVDBMember[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBMember, 1];


SyntaxInformation[OpenVDBMember] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBMember] = $worldregime;


(* ::Section:: *)
(*Nearest*)


(* ::Subsection::Closed:: *)
(*OpenVDBNearest*)


(* ::Subsubsection::Closed:: *)
(*Main*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[OpenVDBNearest] = {"IsoValue" -> Automatic};


OpenVDBNearest[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, OptionsPattern[]] := 
	Block[{isovalue, nearest},
		isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
		(
			nearest = vdb["gridNearest"[pts, isovalue]];
			
			nearest /; MatrixQ[nearest, NumericQ]
			
		) /; realQ[isovalue]
	]


OpenVDBNearest[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, opts___] := 
	OpenVDBNearest[vdb, regimeConvert[vdb, pts, $indexregime -> $worldregime] -> $worldregime, opts]


OpenVDBNearest[vdb_, pts_?coordinateQ -> regime_, opts___] := 
	With[{res = OpenVDBNearest[vdb, {pts} -> regime, opts]},
		res[[1]] /; MatrixQ[res] && Dimensions[res] === {1, 3}
	]


OpenVDBNearest[vdb_, pts_List, opts___] := OpenVDBNearest[vdb, pts -> $worldregime, opts]


OpenVDBNearest[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBNearest, 1];


SyntaxInformation[OpenVDBNearest] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBNearest] = $worldregime;


(* ::Section:: *)
(*Distance*)


(* ::Subsection::Closed:: *)
(*OpenVDBDistance*)


(* ::Subsubsection::Closed:: *)
(*Main*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[OpenVDBDistance] = {"IsoValue" -> Automatic};


OpenVDBDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, OptionsPattern[]] := 
	Block[{isovalue, dists},
		isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
		(
			dists = vdb["gridDistance"[pts, isovalue]];
			
			dists /; VectorQ[dists, NumericQ]
			
		) /; realQ[isovalue]
	]


OpenVDBDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, opts___] := 
	OpenVDBDistance[vdb, regimeConvert[vdb, pts, $indexregime -> $worldregime] -> $worldregime, opts]


OpenVDBDistance[vdb_, pts_?coordinateQ -> regime_, opts___] := 
	With[{res = OpenVDBDistance[vdb, {pts} -> regime, opts]},
		res[[1]] /; VectorQ[res, NumericQ] && Length[res] === 1
	]


OpenVDBDistance[vdb_, pts_List, opts___] := OpenVDBDistance[vdb, pts -> $worldregime, opts]


OpenVDBDistance[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBDistance, 1];


SyntaxInformation[OpenVDBDistance] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBDistance] = $worldregime;


(* ::Subsection::Closed:: *)
(*OpenVDBSignedDistance*)


(* ::Subsubsection::Closed:: *)
(*Main*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[OpenVDBSignedDistance] = {"IsoValue" -> Automatic};


OpenVDBSignedDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, OptionsPattern[]] := 
	Block[{isovalue, dists},
		isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
		(
			dists = vdb["gridSignedDistance"[pts, isovalue]];
			
			If[fogVolumeQ[vdb], Minus[dists], dists] /; VectorQ[dists, NumericQ]
			
		) /; realQ[isovalue]
	]


OpenVDBSignedDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, opts___] := 
	OpenVDBSignedDistance[vdb, regimeConvert[vdb, pts, $indexregime -> $worldregime] -> $worldregime, opts]


OpenVDBSignedDistance[vdb_, pts_?coordinateQ -> regime_, opts___] := 
	With[{res = OpenVDBSignedDistance[vdb, {pts} -> regime, opts]},
		res[[1]] /; VectorQ[res, NumericQ] && Length[res] === 1
	]


OpenVDBSignedDistance[vdb_, pts_List, opts___] := OpenVDBSignedDistance[vdb, pts -> $worldregime, opts]


OpenVDBSignedDistance[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBSignedDistance, 1];


SyntaxInformation[OpenVDBSignedDistance] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBSignedDistance] = $worldregime;


(* ::Section:: *)
(*Balls*)


(* ::Subsection::Closed:: *)
(*OpenVDBFillWithBalls*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBFillWithBalls] = {"IsoValue" -> Automatic, "Overlapping" -> False, "ReturnType" -> Automatic, "SeedCount" -> Automatic};


OpenVDBFillWithBalls[vdb_?OpenVDBScalarGridQ, n_Integer?Positive, rspec_, OptionsPattern[]] := 
	Block[{parsedrspec, isovalue, overlappingQ, rettype, seedcnt, rmin, rmax, res},
		parsedrspec = parseRadiiSpec[vdb, rspec];
		isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
		overlappingQ = TrueQ[OptionValue["Overlapping"]];
		rettype = parseBallReturnType[OptionValue["ReturnType"]];
		seedcnt = parseBallSeedCount[OptionValue["SeedCount"]];
		(
			{rmin, rmax} = parsedrspec;

			res = vdb["fillWithBalls"[1, n, overlappingQ, rmin, rmax, isovalue, seedcnt]];

			returnBalls[res, rettype] /; MatrixQ[res]

		) /; parsedrspec =!= $Failed && realQ[isovalue] && rettype =!= $Failed && seedcnt > 0
	]


OpenVDBFillWithBalls[vdb_?OpenVDBScalarGridQ, n_, opts:OptionsPattern[]] := 
	OpenVDBFillWithBalls[vdb, n, {0.5vdb["VoxelSize"], \[Infinity]} -> $worldregime, opts]


OpenVDBFillWithBalls[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBFillWithBalls, 1];


SyntaxInformation[OpenVDBFillWithBalls] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBFillWithBalls] = $worldregime;


(* ::Subsubsection::Closed:: *)
(*Utilities*)


parseRadiiSpec[vdb_, rspec:Except[_Rule]] := parseRadiiSpec[vdb, rspec -> $worldregime]


parseRadiiSpec[vdb_, (r_?NumericQ) -> regime_] := parseRadiiSpec[vdb, {0.5vdb["VoxelSize"], r} -> regime]


parseRadiiSpec[vdb_, {rmin_, rmax_} -> regime_?regimeQ] /; rmin <= rmax := 
	Block[{bds, \[Delta], rcap},
		bds = vdb["IndexDimensions"];
		\[Delta] = vdb["VoxelSize"];
		(
			rcap = Round[0.5*Last[bds] + 3];
			Clip[regimeConvert[vdb, {rmin, rmax}, regime -> $indexregime], {0.0, rcap}]

		) /; ListQ[bds] && Positive[\[Delta]]
	]


parseRadiiSpec[___] = $Failed;


(* default on the openvdb value side *)
parseBallSeedCount[Automatic] = 10000;
parseBallSeedCount[n_Integer?Positive] := n
parseBallSeedCount[___] = $Failed;


parseBallReturnType[ret:("Regions" | "Balls")] = "Regions";
parseBallReturnType[ret:("PackedArray" | "Packed")] = "PackedArray";
parseBallReturnType[Automatic] = parseBallReturnType["Regions"];


returnBalls[balls_, ret_] :=
	Block[{balls2},
		balls2 = If[balls[[-1, 4]] == 0.0,
			DeleteDuplicates[balls],
			balls
		];

		If[ret === "PackedArray",
			balls2,
			If[#4 == 0.0, 
				Point[{#1, #2, #3}],
				Ball[{#1, #2, #3}, #4]
			]& @@@ balls2
		]
	]
