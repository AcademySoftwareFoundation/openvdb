(* ::Package:: *)

(* ::Title:: *)
(*AggregateData*)


(* ::Subtitle:: *)
(*Data such as slice totals, slices, or dense grids, etc.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


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


OpenVDBActiveVoxelSliceTotals[args___] /; !CheckArgs[OpenVDBActiveVoxelSliceTotals[args], {1, 3}] = $Failed;


OpenVDBActiveVoxelSliceTotals[args___] :=
    With[{res = iOpenVDBActiveVoxelSliceTotals[args]},
        res /; res =!= $Failed
    ]


OpenVDBActiveVoxelSliceTotals[args___] := mOpenVDBActiveVoxelSliceTotals[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBActiveVoxelSliceTotals*)


iOpenVDBActiveVoxelSliceTotals[vdb_?carefulNonMaskGridQ, {z1_, z2_} -> regime_?regimeQ, cntfunc_] /; z1 <= z2 :=
    Block[{zindex, counter, counts},
        zindex = regimeConvert[vdb, {z1, z2}, regime -> $indexregime];
        counter = parseSliceTotalCounter[cntfunc];
        (
            counts = vdb[counter[##]]& @@ zindex;

            counts /; ArrayQ[counts, _, NumericQ]

        ) /; StringQ[counter] && (cntfunc === "Count" || !OpenVDBBooleanGridQ[vdb])
    ]


iOpenVDBActiveVoxelSliceTotals[vdb_, {z1_?NumericQ, z2_?NumericQ}, args___] := iOpenVDBActiveVoxelSliceTotals[vdb, {z1, z2} -> $indexregime, args]


iOpenVDBActiveVoxelSliceTotals[vdb_?OpenVDBGridQ, Automatic, args___] := iOpenVDBActiveVoxelSliceTotals[vdb, vdb["IndexBoundingBox"][[-1]], args]


iOpenVDBActiveVoxelSliceTotals[vdb_] := iOpenVDBActiveVoxelSliceTotals[vdb, Automatic, Automatic]


iOpenVDBActiveVoxelSliceTotals[vdb_, zs_] := iOpenVDBActiveVoxelSliceTotals[vdb, zs, Automatic]


iOpenVDBActiveVoxelSliceTotals[___] = $Failed;


parseSliceTotalCounter = Replace[#, {"Value"|Automatic -> "sliceVoxelValueTotals", "Count" -> "sliceVoxelCounts", _ -> $Failed}, {0}]&;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBActiveVoxelSliceTotals, 1];


SyntaxInformation[OpenVDBActiveVoxelSliceTotals] = {"ArgumentsPattern" -> {_, _., _.}};


addCodeCompletion[OpenVDBActiveVoxelSliceTotals][None, None, {"Value", "Count"}];


OpenVDBDefaultSpace[OpenVDBActiveVoxelSliceTotals] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBActiveVoxelSliceTotals[expr_, ___] /; messageGridQ[expr, OpenVDBActiveVoxelSliceTotals] = $Failed;


mOpenVDBActiveVoxelSliceTotals[expr_, ___] /; messageNonMaskGridQ[expr, OpenVDBActiveVoxelSliceTotals] = $Failed;


mOpenVDBActiveVoxelSliceTotals[_, zspec_, ___] /; messageZSpecQ[zspec, OpenVDBActiveVoxelSliceTotals] = $Failed;


mOpenVDBActiveVoxelSliceTotals[_, _, cntfunc_] :=
    (
        If[parseSliceTotalCounter[cntfunc] === $Failed,
            Message[OpenVDBActiveVoxelSliceTotals::cntr, cntfunc, 3]
        ];
        $Failed
    );


mOpenVDBActiveVoxelSliceTotals[___] = $Failed;


OpenVDBActiveVoxelSliceTotals::cntr = "`1` at position `2` is not one of \"Value\", \"Count\", or Automatic.";


(* ::Section:: *)
(*Slice*)


(* ::Subsection::Closed:: *)
(*OpenVDBSlice*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBSlice] = {"MirrorSlice" -> False};


OpenVDBSlice[args___] /; !CheckArgs[OpenVDBSlice[args], {1, 3}] = $Failed;


OpenVDBSlice[args___] :=
    With[{res = iOpenVDBSlice[args]},
        res /; res =!= $Failed
    ]


OpenVDBSlice[args___] := mOpenVDBSlice[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBSlice*)


Options[iOpenVDBSlice] = Options[OpenVDBSlice];


iOpenVDBSlice[vdb_?carefulNonMaskGridQ, z_?NumericQ -> regime_?regimeQ, bds_?bounds2DQ, OptionsPattern[]] :=
    Block[{mirrorQ, threadedQ, zindex, bdsindex, data},
        mirrorQ = TrueQ[OptionValue["MirrorSlice"]];
        threadedQ = True;

        zindex = regimeConvert[vdb, z, regime -> $indexregime];
        bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];

        data = vdb["gridSlice"[zindex, bdsindex, mirrorQ, threadedQ]];

        data /; ArrayQ[data]
    ]


iOpenVDBSlice[vdb_, z_?NumericQ, args___] := iOpenVDBSlice[vdb, z -> $indexregime, args]


iOpenVDBSlice[vdb_?carefulNonMaskGridQ, z_, Automatic, opts___] := iOpenVDBSlice[vdb, z, Most[vdb["IndexBoundingBox"]], opts]


iOpenVDBSlice[vdb_, z_, opts:OptionsPattern[]] := iOpenVDBSlice[vdb, z, Automatic, opts]


iOpenVDBSlice[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBSlice, 1];


SyntaxInformation[OpenVDBSlice] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBSlice] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBSlice[expr_, ___] /; messageGridQ[expr, OpenVDBSlice] = $Failed;


mOpenVDBSlice[expr_, ___] /; messageNonMaskGridQ[expr, OpenVDBSlice] = $Failed;


mOpenVDBSlice[_, z_, ___] /; messageZSliceQ[z, OpenVDBSlice] = $Failed;


mOpenVDBSlice[_, _, bbox_, ___] /; message2DBBoxQ[bbox, OpenVDBSlice] = $Failed;


mOpenVDBSlice[___] = $Failed;


(* ::Section:: *)
(*Data*)


(* ::Subsection::Closed:: *)
(*OpenVDBData*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBData[args___] /; !CheckArgs[OpenVDBData[args], {1, 2}] = $Failed;


OpenVDBData[args___] :=
    With[{res = iOpenVDBData[args]},
        res /; res =!= $Failed
    ]


OpenVDBData[args___] := mOpenVDBData[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBData*)


iOpenVDBData[vdb_?carefulNonMaskGridQ, bds_?bounds3DQ -> regime_?regimeQ] :=
    Block[{bdsindex, data},
        bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
        data = vdb["gridData"[bdsindex]];

        data /; ArrayQ[data]
    ]


iOpenVDBData[vdb_?carefulNonMaskGridQ] := iOpenVDBData[vdb, vdb["IndexBoundingBox"] -> $indexregime]


iOpenVDBData[vdb_, Automatic] := iOpenVDBData[vdb, vdb["IndexBoundingBox"] -> $indexregime]


iOpenVDBData[vdb_, bspec_List] /; bounds3DQ[bspec] || intervalQ[bspec] := iOpenVDBData[vdb, bspec -> $indexregime]


iOpenVDBData[vdb_?carefulNonMaskGridQ, int_?intervalQ -> regime_?regimeQ] :=
    Block[{bds2d},
        bds2d = regimeConvert[vdb, Most[vdb["IndexBoundingBox"]], $indexregime -> regime];

        iOpenVDBData[vdb, Append[bds2d, int] -> regime] /; bounds2DQ[bds2d]
    ]


iOpenVDBData[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBData, 1];


SyntaxInformation[OpenVDBData] = {"ArgumentsPattern" -> {_, _.}};


OpenVDBDefaultSpace[OpenVDBData] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBData[expr_, ___] /; messageGridQ[expr, OpenVDBData] = $Failed;


mOpenVDBData[expr_, ___] /; messageNonMaskGridQ[expr, OpenVDBData] = $Failed;


mOpenVDBData[_, bbox_, ___] /; message3DBBoxQ[bbox, OpenVDBData] = $Failed;


mOpenVDBData[___] = $Failed;


(* ::Section:: *)
(*Active regions*)


(* ::Subsection::Closed:: *)
(*OpenVDBActiveTiles*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBActiveTiles] = {"PartialOverlap" -> True};


OpenVDBActiveTiles[args___] /; !CheckArgs[OpenVDBActiveTiles[args], {1, 2}] = $Failed;


OpenVDBActiveTiles[args___] :=
    With[{res = iOpenVDBActiveTiles[args]},
        res /; res =!= $Failed
    ]


OpenVDBActiveTiles[args___] := mOpenVDBActiveTiles[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBActiveTiles*)


Options[iOpenVDBActiveTiles] = Options[OpenVDBActiveTiles];


iOpenVDBActiveTiles[vdb_?OpenVDBGridQ, bds_?bounds3DQ -> regime_?regimeQ, OptionsPattern[]] :=
    Block[{bdsindex, partialoverlap, tiles},
        bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];
        partialoverlap = TrueQ[OptionValue["PartialOverlap"]];

        tiles = vdb["activeTiles"[bdsindex, partialoverlap]];

        tiles /; ListQ[tiles]
    ]


iOpenVDBActiveTiles[vdb_?OpenVDBGridQ, opts:OptionsPattern[]] := iOpenVDBActiveTiles[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


iOpenVDBActiveTiles[vdb_?OpenVDBGridQ, Automatic, opts:OptionsPattern[]] := iOpenVDBActiveTiles[vdb, vdb["IndexBoundingBox"] -> $indexregime, opts]


iOpenVDBActiveTiles[vdb_, bds_?bounds3DQ, opts:OptionsPattern[]] := iOpenVDBActiveTiles[vdb, bds -> $indexregime, opts]


iOpenVDBActiveTiles[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBActiveTiles, 1];


SyntaxInformation[OpenVDBActiveTiles] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBActiveTiles] = $indexregime;


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBActiveTiles[expr_, ___] /; messageGridQ[expr, OpenVDBActiveTiles] = $Failed;


mOpenVDBActiveTiles[_, bbox_, ___] /; message3DBBoxQ[bbox, OpenVDBActiveTiles] = $Failed;


mOpenVDBActiveTiles[___] = $Failed;


(* ::Subsection::Closed:: *)
(*OpenVDBActiveVoxels*)


(* ::Subsubsection::Closed:: *)
(*Main*)


OpenVDBActiveVoxels[args___] /; !CheckArgs[OpenVDBActiveVoxels[args], {1, 3}] = $Failed;


OpenVDBActiveVoxels[args___] :=
    With[{res = pOpenVDBActiveVoxels[args]},
        res /; res =!= $Failed
    ]


OpenVDBActiveVoxels[args___] := mOpenVDBActiveVoxels[args]


(* ::Subsubsection::Closed:: *)
(*pOpenVDBActiveVoxels*)


pOpenVDBActiveVoxels[vdb_?OpenVDBGridQ, bds_?bounds3DQ -> regime_?regimeQ, ret_] :=
    Block[{bdsindex, res},
        bdsindex = regimeConvert[vdb, bds, regime -> $indexregime];

        res = iOpenVDBActiveVoxels[vdb, bdsindex, ret];

        res /; res =!= $Failed
    ]


pOpenVDBActiveVoxels[vdb_?OpenVDBGridQ] := pOpenVDBActiveVoxels[vdb, vdb["IndexBoundingBox"] -> $indexregime]


pOpenVDBActiveVoxels[vdb_?OpenVDBGridQ, Automatic, args___] := pOpenVDBActiveVoxels[vdb, vdb["IndexBoundingBox"] -> $indexregime, args]


pOpenVDBActiveVoxels[vdb_, bds_List, args___] := pOpenVDBActiveVoxels[vdb, bds -> $indexregime, args]


pOpenVDBActiveVoxels[vdb_, bds_] := pOpenVDBActiveVoxels[vdb, bds, Automatic]


pOpenVDBActiveVoxels[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[pOpenVDBActiveVoxels, 1];


SyntaxInformation[OpenVDBActiveVoxels] = {"ArgumentsPattern" -> {_, _., _.}};


OpenVDBDefaultSpace[OpenVDBActiveVoxels] = $indexregime;


addCodeCompletion[OpenVDBActiveVoxels][None, None, {"SparseArray", "Positions", "Values"}];


(* ::Subsubsection::Closed:: *)
(*iOpenVDBActiveVoxels*)


iOpenVDBActiveVoxels[vdb_, bds_, "SparseArray"|Automatic] /; AnyTrue[{OpenVDBScalarGridQ, OpenVDBIntegerGridQ, OpenVDBBooleanGridQ}, #[vdb]&] :=
    Block[{res},
        res = vdb["activeVoxels"[bds]];

        res /; ArrayQ[res, 3]
    ]


iOpenVDBActiveVoxels[vdb_?OpenVDBVectorGridQ, bds_, "SparseArray"|Automatic] :=
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


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBActiveVoxels[expr_, ___] /; messageGridQ[expr, OpenVDBActiveVoxels] = $Failed;


mOpenVDBActiveVoxels[_, bbox_, ___] /; message3DBBoxQ[bbox, OpenVDBActiveVoxels] = $Failed;


mOpenVDBActiveVoxels[_, _, ret_, ___] /; !MatchQ[ret, "SparseArray"|"Positions"|"Values"|Automatic] :=
    (
        Message[OpenVDBActiveVoxels::rettype, 3];
        $Failed
    )


mOpenVDBActiveVoxels[_?OpenVDBMaskGridQ, _, ret_, ___] /; !MatchQ[ret, "SparseArray"|"Values"] :=
    (
        Message[OpenVDBActiveVoxels::mask];
        $Failed
    )


mOpenVDBActiveVoxels[___] = $Failed;


OpenVDBActiveVoxels::rettype = "The value at position `1` should be one of \"SparseArray\", \"Positions\", \"Values\", or Automatic.";
OpenVDBActiveVoxels::mask = "The return types \"SparseArray\", and \"Values\" are not supported for mask grids.";
