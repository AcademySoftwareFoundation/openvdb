(* ::Package:: *)

(* ::Title:: *)
(*AggregateData*)


(* ::Subtitle:: *)
(*Data such as slice totals, slices, or dense grids, etc.*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBActiveVoxelSliceTotals"]
PackageExport["OpenVDBSlice"]
PackageExport["OpenVDBData"]
PackageExport["OpenVDBActiveTiles"]
PackageExport["OpenVDBActiveVoxels"]


OpenVDBActiveVoxelSliceTotals::usage = "OpenVDBActiveVoxelSliceTotals[expr] computes active voxel totals for each horizontal slice of an OpenVDB grid.";


OpenVDBSlice::usage = "OpenVDBSlice[expr, z] returns a slice of an OpenVDB grid at height z.";
OpenVDBData::usage = "OpenVDBData[expr] returns a dense array representation of an OpenVDB grid.";


OpenVDBActiveTiles::usage = "OpenVDBActiveTiles[expr] returns all active tiles of an OpenVDB grid.";
OpenVDBActiveVoxels::usage = "OpenVDBActiveVoxels[expr] returns all active voxels of an OpenVDB grid as a SparseArray.";


(* ::Section:: *)
(*Totals*)


(* ::Subsection::Closed:: *)
(*OpenVDBActiveVoxelSliceTotals*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Clear[OpenVDBActiveVoxelSliceTotals]


OpenVDBActiveVoxelSliceTotals[vdb_?carefulNonMaskGridQ, {z1_, z2_} -> regime_?regimeQ, cntfunc_] /; z1 <= z2 :=
	Block[{zindex, counter, counts},
		zindex = regimeConvert[vdb, {z1, z2}, regime -> $indexregime];
		counter = Replace[cntfunc, {"Value"|Automatic -> "sliceVoxelValueTotals", "Count" -> "sliceVoxelCounts", _ -> $Failed}, {0}];
		(
			counts = vdb[counter[##]]& @@ zindex;
			
			counts /; ArrayQ[counts, _, NumericQ]
			
		) /; StringQ[counter] && (cntfunc === "Count" || !OpenVDBBooleanGridQ[vdb])
	]


OpenVDBActiveVoxelSliceTotals[vdb_, {z1_?NumericQ, z2_?NumericQ}, args___] := OpenVDBActiveVoxelSliceTotals[vdb, {z1, z2} -> $indexregime, args]


OpenVDBActiveVoxelSliceTotals[vdb_?OpenVDBGridQ, Automatic, args___] := OpenVDBActiveVoxelSliceTotals[vdb, vdb["IndexBoundingBox"][[-1]], args]


OpenVDBActiveVoxelSliceTotals[vdb_] := OpenVDBActiveVoxelSliceTotals[vdb, Automatic, Automatic]


OpenVDBActiveVoxelSliceTotals[vdb_, zs_] := OpenVDBActiveVoxelSliceTotals[vdb, zs, Automatic]


OpenVDBActiveVoxelSliceTotals[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBActiveVoxelSliceTotals, 1];


SyntaxInformation[OpenVDBActiveVoxelSliceTotals] = {"ArgumentsPattern" -> {_, _., _.}};


addCodeCompletion[OpenVDBActiveVoxelSliceTotals][None, None, {"Value", "Count"}];


OpenVDBDefaultSpace[OpenVDBActiveVoxelSliceTotals] = $indexregime;


(* ::Section:: *)
(*Slice*)


(* ::Subsection::Closed:: *)
(*OpenVDBSlice*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBSlice] = {"MirrorSlice" -> False};


OpenVDBSlice[vdb_?carefulNonMaskGridQ, z_?NumericQ -> regime_?regimeQ, bds_?bounds2DQ, OptionsPattern[]] :=
	Block[{mirrorQ, threadedQ, zindex, bdsindex, data},
		mirrorQ = TrueQ[OptionValue["MirrorSlice"]];
		threadedQ = True;
		
		zindex = regimeConvert[vdb, z, regime -> $indexregime];
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		
		data = vdb["gridSlice"[zindex, bdsindex, mirrorQ, threadedQ]];
		
		data /; ArrayQ[data]
	]


OpenVDBSlice[vdb_, z_?NumericQ, args___] := OpenVDBSlice[vdb, z -> $indexregime, args]


OpenVDBSlice[vdb_?carefulNonMaskGridQ, z_, Automatic, opts___] := OpenVDBSlice[vdb, z, Most[vdb["IndexBoundingBox"]], opts]


OpenVDBSlice[vdb_, z_, opts:OptionsPattern[]] := OpenVDBSlice[vdb, z, Automatic, opts]


OpenVDBSlice[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBSlice, 1];


SyntaxInformation[OpenVDBSlice] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBSlice] = $indexregime;


(* ::Section:: *)
(*Data*)


(* ::Subsection::Closed:: *)
(*OpenVDBData*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBData[vdb_?carefulNonMaskGridQ, bds_?bounds3DQ -> regime_?regimeQ] :=
	Block[{bdsindex, data},
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		data = vdb["gridData"[bdsindex]];
			
		data /; ArrayQ[data]
	]


OpenVDBData[vdb_?carefulNonMaskGridQ] := OpenVDBData[vdb, vdb["IndexBoundingBox"] -> $indexregime]


OpenVDBData[vdb_, Automatic] := OpenVDBData[vdb, vdb["IndexBoundingBox"] -> $indexregime]


OpenVDBData[vdb_, bspec_List] /; bounds3DQ[bspec] || intervalQ[bspec] := OpenVDBData[vdb, bspec -> $indexregime]


OpenVDBData[vdb_?carefulNonMaskGridQ, int_?intervalQ -> regime_?regimeQ] := 
	Block[{bds2d},
		bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];
		
		OpenVDBData[vdb, Append[bds2d, int] -> regime] /; bounds2DQ[bds2d]
	]


OpenVDBData[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBData, 1];


SyntaxInformation[OpenVDBData] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBData] = $indexregime;


(* ::Section:: *)
(*Active regions*)


(* ::Subsection::Closed:: *)
(*OpenVDBActiveTiles*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBActiveTiles] = {"PartialOverlap" -> True};


OpenVDBActiveTiles[vdb_?OpenVDBGridQ, bds_?bounds3DQ -> regime_?regimeQ, OptionsPattern[]] :=
	Block[{bdsindex, partialoverlap, tiles},
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		partialoverlap = TrueQ[OptionValue["PartialOverlap"]];
		
		tiles = vdb["activeTiles"[bdsindex, partialoverlap]];
		
		tiles /; ListQ[tiles]
	]


OpenVDBActiveTiles[vdb_?OpenVDBGridQ, opts:OptionsPattern[]] := OpenVDBActiveTiles[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


OpenVDBActiveTiles[vdb_?OpenVDBGridQ, Automatic, opts:OptionsPattern[]] := OpenVDBActiveTiles[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


OpenVDBActiveTiles[vdb_, bds_?bounds3DQ, opts:OptionsPattern[]] := OpenVDBActiveTiles[vdb, bds -> $indexregime, opts]


OpenVDBActiveTiles[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBActiveTiles, 1];


SyntaxInformation[OpenVDBActiveTiles] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBActiveTiles] = $indexregime;


(* ::Subsection::Closed:: *)
(*OpenVDBActiveVoxels*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBActiveVoxels[vdb_?OpenVDBGridQ, bds_?bounds3DQ -> regime_?regimeQ, ret_] :=
	Block[{bdsindex, res},
		bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
		
		res = iOpenVDBActiveVoxels[vdb, bdsindex, ret];
		
		res /; res =!= $Failed
	]


OpenVDBActiveVoxels[vdb_?OpenVDBGridQ] := OpenVDBActiveVoxels[vdb, vdb["IndexBoundingBox"] -> $indexregime]


OpenVDBActiveVoxels[vdb_?OpenVDBGridQ, Automatic, args___] := OpenVDBActiveVoxels[vdb, vdb["IndexBoundingBox"] -> $indexregime, args]


OpenVDBActiveVoxels[vdb_, bds_List, args___] := OpenVDBActiveVoxels[vdb, bds -> $indexregime, args]


OpenVDBActiveVoxels[vdb_, bds_] := OpenVDBActiveVoxels[vdb, bds, Automatic]


OpenVDBActiveVoxels[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBActiveVoxels, 1];


SyntaxInformation[OpenVDBActiveVoxels] = {"ArgumentsPattern" -> {_, _., _.}};


OpenVDBDefaultSpace[OpenVDBActiveVoxels] = $indexregime;


addCodeCompletion[OpenVDBActiveVoxels][None, None, {"SparseArray", "Positions", "Values", "SparseArrayList"}];


(* ::Subsubsection::Closed:: *)
(*iOpenVDBActiveVoxels*)


iOpenVDBActiveVoxels[vdb_, bds_, "SparseArray"|Automatic] /; AnyTrue[{OpenVDBScalarGridQ, OpenVDBIntegerGridQ, OpenVDBBooleanGridQ}, #[vdb]&] :=
	Block[{res},
		res = vdb["activeVoxels"[bds]];
		
		res /; ArrayQ[res, 3]
	]


iOpenVDBActiveVoxels[vdb_?OpenVDBVectorGridQ, bds_, "SparseArrayList"|Automatic] :=
	Block[{pos, vals, dims, offset, res},
		pos = iOpenVDBActiveVoxels[vdb, bds, "Positions"];
		vals = iOpenVDBActiveVoxels[vdb, bds, "Values"];
		(
			dims = Abs[Subtract @@@ bds] + 1;
			offset = 1 - bds[[All, 1]];
			
			Statistics`Library`MatrixRowTranslate[pos, offset];
			
			res = SparseArray[pos -> #, dims]& /@ Transpose[vals];
			
			res /; VectorQ[res, ArrayQ[#, 3]&]
			
		) /; pos =!= $Failed && vals =!= $Failed
	]


iOpenVDBActiveVoxels[vdb_?OpenVDBMaskGridQ, bds_, "SparseArray"|Automatic] :=
	Block[{pos, dims, offset},
		pos = vdb["activeVoxelPositions"[bds]];
		(
			dims = 1 + Abs[Subtract @@@ bds];
			offset = 1 - bds[[All, 1]];
			
			Statistics`Library`MatrixRowTranslate[pos, offset];
			(
				SparseArray[pos -> _, dims]
				
			) /; Min[dims] > 0
			
		) /; MatrixQ[pos, IntegerQ]
	]


iOpenVDBActiveVoxels[vdb_, bds_, "Positions"] :=
	Block[{res},
		res = vdb["activeVoxelPositions"[bds]];
		
		res /; MatrixQ[res, IntegerQ]
	]


iOpenVDBActiveVoxels[vdb_?nonMaskGridQ, bds_, "Values"] :=
	Block[{res},
		res = vdb["activeVoxelValues"[bds]];
		
		res /; ArrayQ[res, _, NumericQ]
	]


iOpenVDBActiveVoxels[___] = $Failed;
