(* ::Package:: *)

(* ::Title:: *)
(*CSG*)


(* ::Subtitle:: *)
(*Union, intersect, difference, and clip grids.*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBUnionTo"]
PackageExport["OpenVDBIntersectWith"]
PackageExport["OpenVDBDifferenceFrom"]
PackageExport["OpenVDBUnion"]
PackageExport["OpenVDBIntersection"]
PackageExport["OpenVDBDifference"]
PackageExport["OpenVDBClip"]


OpenVDBUnionTo::usage = "OpenVDBUnionTo[expr1, expr2, \[Ellipsis]] performs the union of OpenVDB grids and stores the result to expr1, deleting all other expri.";
OpenVDBIntersectWith::usage = "OpenVDBIntersectWith[expr1, expr2, \[Ellipsis]] performs the intersection of OpenVDB grids and stores the result to expr1, deleting all other expri.";
OpenVDBDifferenceFrom::usage = "OpenVDBDifferenceFrom[expr1, expr2, \[Ellipsis]] subtracts OpenVDB grids expr2, \[Ellipsis] from expr1 and stores the result to expr1, deleting all other expri.";


OpenVDBUnion::usage = "OpenVDBUnion[expr1, expr2, \[Ellipsis]] performs the union of OpenVDB grids and stores the result to a new OpenVDB grid.";
OpenVDBIntersection::usage = "OpenVDBIntersection[expr1, expr2, \[Ellipsis]] performs the intersection of OpenVDB grids and stores the result to a new OpenVDB grid.";
OpenVDBDifference::usage = "OpenVDBDifference[expr1, expr2, \[Ellipsis]] subtracts OpenVDB grids expr2, \[Ellipsis] from expr1 and stores the result to a new OpenVDB grid.";


OpenVDBClip::usage = "OpenVDBClip[expr, bds] clips an OpenVDB grid over bounds bds.";


(* ::Section:: *)
(*Boolean operations*)


(* ::Subsection::Closed:: *)
(*OpenVDBUnion*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBUnion] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBUnion[OptionsPattern[]] := 
	Block[{vdb},
		vdb = OpenVDBCreateGrid["GridClass" -> "LevelSet"];
		(
			OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
			
			vdb	
			
		) /; OpenVDBGridQ[vdb]
	]


OpenVDBUnion[vdb_?OpenVDBScalarGridQ, vdbs___, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] := 
	Block[{ivdb},
		ivdb = OpenVDBCreateGrid[vdb];
		ivdb["gridUnionCopy"[{vdb, vdbs}[[All, 1]]]];
		
		OpenVDBSetProperty[ivdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		ivdb
	]


OpenVDBUnion[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBUnion];


SyntaxInformation[OpenVDBUnion] = {"ArgumentsPattern" -> {___, OptionsPattern[]}};


(* ::Subsection::Closed:: *)
(*OpenVDBIntersection*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBIntersection] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBIntersection[OptionsPattern[]] := 
	Block[{vdb},
		vdb = OpenVDBCreateGrid["GridClass" -> "LevelSet"];
		(
			OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
			
			vdb	
			
		) /; OpenVDBGridQ[vdb]
	]


OpenVDBIntersection[vdb_?OpenVDBScalarGridQ, vdbs___, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] := 
	Block[{ivdb},
		ivdb = OpenVDBCreateGrid[{vdb}[[1]]];
		ivdb["gridIntersectionCopy"[{vdb, vdbs}[[All, 1]]]];
		
		OpenVDBSetProperty[ivdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		ivdb
	]


OpenVDBIntersection[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBIntersection];


SyntaxInformation[OpenVDBIntersection] = {"ArgumentsPattern" -> {___, OptionsPattern[]}};


(* ::Subsection::Closed:: *)
(*OpenVDBDifference*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBDifference] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBDifference[OptionsPattern[]] := 
	Block[{vdb},
		vdb = OpenVDBCreateGrid["GridClass" -> "LevelSet"];
		(
			OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
			
			vdb
				
		) /; OpenVDBGridQ[vdb]
	]


OpenVDBDifference[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] := 
	(
		OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		vdb
	)


OpenVDBDifference[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] := 
	Block[{union, vdbdiff},
		union = iOpenVDBUnion[vdbs];
		(
			vdbdiff = OpenVDBCreateGrid[vdb];
			vdbdiff["gridDifferenceCopy"[vdb[[1]], union[[1]]]];
		
			OpenVDBSetProperty[vdbdiff, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
			
			vdbdiff
			
		) /; OpenVDBGridQ[union]
	]


OpenVDBDifference[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBDifference];


SyntaxInformation[OpenVDBDifference] = {"ArgumentsPattern" -> {___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Utilities*)


iOpenVDBUnion[vdb_] := vdb
iOpenVDBUnion[vdbs__] := OpenVDBUnion[vdbs]


(* ::Section:: *)
(*In place Boolean operations*)


(* ::Subsection::Closed:: *)
(*OpenVDBUnionTo*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBUnionTo] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBUnionTo[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] := 
	(
		OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		vdb
	)


OpenVDBUnionTo[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] := 
	(
		Scan[vdb["gridUnion"[#]]&, {vdbs}[[All, 1]]];
		
		OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		vdb
	)


OpenVDBUnionTo[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBUnionTo];


SyntaxInformation[OpenVDBUnionTo] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsection::Closed:: *)
(*OpenVDBIntersectWith*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBIntersectWith] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBIntersectWith[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] := 
	(
		OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		vdb
	)


OpenVDBIntersectWith[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] := 
	(
		Scan[vdb["gridIntersection"[#]]&, {vdbs}[[All, 1]]];
		
		OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		vdb
	)


OpenVDBIntersectWith[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBIntersectWith];


SyntaxInformation[OpenVDBIntersectWith] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsection::Closed:: *)
(*OpenVDBDifferenceFrom*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBDifferenceFrom] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBDifferenceFrom[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] := 
	(
		OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
		
		vdb
	)


OpenVDBDifferenceFrom[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] := 
	Block[{vdbunion},
		vdbunion = OpenVDBUnionTo[vdbs];
		(
			vdb["gridDifference"[vdbunion[[1]]]];
			
			OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
			
			vdb
		) /; OpenVDBGridQ[vdbunion]
	]


OpenVDBDifferenceFrom[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBDifferenceFrom];


SyntaxInformation[OpenVDBDifferenceFrom] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Section:: *)
(*Clipping*)


(* ::Subsection::Closed:: *)
(*OpenVDBClip*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBClip] = {"CloseBoundary" -> True, "Creator" -> Inherited, "Name" -> Inherited};


OpenVDBClip[vdb_?OpenVDBScalarGridQ, bspec_List -> regime_?regimeQ, opts:OptionsPattern[]] :=
	Block[{bds, closeQ, clip},
		bds = parseClipBounds[vdb, bspec, regime];
		closeQ = TrueQ[OptionValue["CloseBoundary"]];
		(
			clip = iOpenVDBClip[vdb, bds, closeQ];
			(
				OpenVDBSetProperty[clip, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];
				
				clip
				
			) /; OpenVDBGridQ[clip]
			
		) /; bds =!= $Failed
	]


OpenVDBClip[vdb_, bds_List, opts:OptionsPattern[]] := OpenVDBClip[vdb, bds -> $worldregime, opts]


OpenVDBClip[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBClip, 1];


SyntaxInformation[OpenVDBClip] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBClip] = $worldregime;


(* ::Subsubsection::Closed:: *)
(*iOpenVDBClip*)


iOpenVDBClip[vdb_?emptyVDBQ, __] := vdb


iOpenVDBClip[vdb_?levelSetQ, bds_, True] :=
	Block[{voxsize, halfwidth, cube, clipvdb},
		voxsize = voxelSize[vdb];
		halfwidth = halfWidth[vdb];
		(
			cube = OpenVDBLevelSet[Cuboid @@ Transpose[bds], voxsize, halfwidth, "ScalarType" -> vdb[[2]]];
			clipvdb = OpenVDBIntersection[vdb, cube];
			
			clipvdb /; OpenVDBGridQ[clipvdb]
			
		) /; voxsize > 0 && halfwidth > 0
	]


iOpenVDBClip[vdb_, bds_, closeQ_] /; closeQ =!= True || !levelSetQ[vdb] :=
	Block[{inst, bdata, clipvdb},
		clipvdb = OpenVDBCreateGrid[vdb];
		
		clipvdb["clipGrid"[vdb[[1]], bds]];
		
		clipvdb
	]


iOpenVDBClip[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Utilities*)


parseClipBounds[vdb_, bspec_, regime_] := 
	Block[{bds, voxsize, vdbbbox},
		bds = iParseClipBounds[bspec];
		voxsize = voxelSize[vdb];
		(
			bds = regimeConvert[vdb, bds, regime -> $worldregime];
			
			vdbbbox = voxsize*(vdb["getGridBoundingBox"[]] + {{-2, 2}, {-2, 2}, {-2, 2}});
			
			boundingBoxIntersection[{vdbbbox, bds}]
			
		) /; MatrixQ[bds] && voxsize > 0
	]

parseClipBounds[___] = $Failed;


iParseClipBounds[bds_?bounds3DQ] := bds


iParseClipBounds[l_List] := 
	With[{res = iParseClipBounds /@ l},
		boundingBoxIntersection[res] /; FreeQ[res, $Failed, {1}]
	]


iParseClipBounds[(dir_Integer) -> {v1_, v2_}] /; 1 <= dir <= 3 && -\[Infinity] <= v1 < v2 <= \[Infinity] := Insert[{{-\[Infinity], \[Infinity]}, {-\[Infinity], \[Infinity]}}, {v1, v2}, dir]


iParseClipBounds[(dir_Integer) -> v_?NumericQ] /; 1 <= Abs[dir] <= 3 := 
	If[dir < 0,
		iParseClipBounds[Minus[dir] -> {v, \[Infinity]}],
		iParseClipBounds[dir -> {-\[Infinity], v}]
	]


iParseClipBounds[Cuboid[lo_]?ConstantRegionQ] := Transpose[{lo, lo+1}]


iParseClipBounds[Cuboid[lo_, hi_]?ConstantRegionQ] := Transpose[{lo, hi}]


iParseClipBounds[___] = $Failed;


boundingBoxIntersection[bds_] := 
	Developer`ToPackedArray @ {
		{Max[bds[[All, 1, 1]]], Min[bds[[All, 1, 2]]]},
		{Max[bds[[All, 2, 1]]], Min[bds[[All, 2, 2]]]},
		{Max[bds[[All, 3, 1]]], Min[bds[[All, 3, 2]]]}
	}
