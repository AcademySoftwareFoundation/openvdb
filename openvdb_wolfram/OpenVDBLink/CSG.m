(* ::Package:: *)

(* ::Title:: *)
(*CSG*)


(* ::Subtitle:: *)
(*Union, intersect, difference, and clip grids.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBUnionTo"]
PackageExport["OpenVDBIntersectWith"]
PackageExport["OpenVDBDifferenceFrom"]
PackageExport["OpenVDBUnion"]
PackageExport["OpenVDBIntersection"]
PackageExport["OpenVDBDifference"]
PackageExport["OpenVDBMaxOf"]
PackageExport["OpenVDBMinOf"]
PackageExport["OpenVDBMax"]
PackageExport["OpenVDBMin"]
PackageExport["OpenVDBClip"]


OpenVDBUnionTo::usage = "OpenVDBUnionTo[expr1, expr2, \[Ellipsis]] performs the union of OpenVDB grids and stores the result to expr1, deleting all other expri.";
OpenVDBIntersectWith::usage = "OpenVDBIntersectWith[expr1, expr2, \[Ellipsis]] performs the intersection of OpenVDB grids and stores the result to expr1, deleting all other expri.";
OpenVDBDifferenceFrom::usage = "OpenVDBDifferenceFrom[expr1, expr2, \[Ellipsis]] subtracts OpenVDB grids expr2, \[Ellipsis] from expr1 and stores the result to expr1, deleting all other expri.";


OpenVDBUnion::usage = "OpenVDBUnion[expr1, expr2, \[Ellipsis]] performs the union of OpenVDB grids and stores the result to a new OpenVDB grid.";
OpenVDBIntersection::usage = "OpenVDBIntersection[expr1, expr2, \[Ellipsis]] performs the intersection of OpenVDB grids and stores the result to a new OpenVDB grid.";
OpenVDBDifference::usage = "OpenVDBDifference[expr1, expr2, \[Ellipsis]] subtracts OpenVDB grids expr2, \[Ellipsis] from expr1 and stores the result to a new OpenVDB grid.";


OpenVDBMaxOf::usage = "OpenVDBMaxOf[expr1, expr2, \[Ellipsis]] performs the voxelwise maximum of OpenVDB grids and stores the result to expr1, deleting all other expri.";
OpenVDBMinOf::usage = "OpenVDBMinOf[expr1, expr2, \[Ellipsis]] performs the voxelwise minimum of OpenVDB grids and stores the result to expr1, deleting all other expri.";


OpenVDBMax::usage = "OpenVDBMax[expr1, expr2, \[Ellipsis]] performs the voxelwise maximum of OpenVDB grids and stores the result to a new OpenVDB grid.";
OpenVDBMin::usage = "OpenVDBMin[expr1, expr2, \[Ellipsis]] performs the voxelwise minimum of OpenVDB grids and stores the result to a new OpenVDB grid.";


OpenVDBClip::usage = "OpenVDBClip[expr, bds] clips an OpenVDB grid over bounds bds.";


(* ::Section:: *)
(*Boolean operations*)


(* ::Subsection::Closed:: *)
(*OpenVDBUnion*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBUnion] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBUnion[args___] /; !CheckArgs[OpenVDBUnion[args], {0, \[Infinity]}] = $Failed;


OpenVDBUnion[args___] :=
    With[{res = iOpenVDBUnion[args]},
        res /; res =!= $Failed
    ]


OpenVDBUnion[args___] := mOpenVDBUnion[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBUnion*)


Options[iOpenVDBUnion] = Options[OpenVDBUnion];


iOpenVDBUnion[OptionsPattern[]] :=
    Block[{vdb},
        vdb = OpenVDBCreateGrid["GridClass" -> "LevelSet"];
        (
            OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

            vdb

        ) /; OpenVDBGridQ[vdb]
    ]


iOpenVDBUnion[vdb_?OpenVDBScalarGridQ, vdbs___, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    Block[{ivdb},
        ivdb = OpenVDBCreateGrid[vdb];
        ivdb["gridUnionCopy"[{vdb, vdbs}[[All, 1]]]];

        OpenVDBSetProperty[ivdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        ivdb
    ]


iOpenVDBUnion[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBUnion];


SyntaxInformation[OpenVDBUnion] = {"ArgumentsPattern" -> {___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBUnion[args___] := messageBooleanFunction[OpenVDBUnion, args]


(* ::Subsection::Closed:: *)
(*OpenVDBIntersection*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBIntersection] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBIntersection[args___] /; !CheckArgs[OpenVDBIntersection[args], {0, \[Infinity]}] = $Failed;


OpenVDBIntersection[args___] :=
    With[{res = iOpenVDBIntersection[args]},
        res /; res =!= $Failed
    ]


OpenVDBIntersection[args___] := mOpenVDBIntersection[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBIntersection*)


Options[iOpenVDBIntersection] = Options[OpenVDBIntersection];


iOpenVDBIntersection[OptionsPattern[]] :=
    Block[{vdb},
        vdb = OpenVDBCreateGrid["GridClass" -> "LevelSet"];
        (
            OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

            vdb

        ) /; OpenVDBGridQ[vdb]
    ]


iOpenVDBIntersection[vdb_?OpenVDBScalarGridQ, vdbs___, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    Block[{ivdb},
        ivdb = OpenVDBCreateGrid[{vdb}[[1]]];
        ivdb["gridIntersectionCopy"[{vdb, vdbs}[[All, 1]]]];

        OpenVDBSetProperty[ivdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        ivdb
    ]


iOpenVDBIntersection[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBIntersection];


SyntaxInformation[OpenVDBIntersection] = {"ArgumentsPattern" -> {___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBIntersection[args___] := messageBooleanFunction[OpenVDBIntersection, args]


(* ::Subsection::Closed:: *)
(*OpenVDBDifference*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBDifference] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBDifference[args___] /; !CheckArgs[OpenVDBDifference[args], {0, \[Infinity]}] = $Failed;


OpenVDBDifference[args___] :=
    With[{res = iOpenVDBDifference[args]},
        res /; res =!= $Failed
    ]


OpenVDBDifference[args___] := mOpenVDBDifference[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBDifference*)


Options[iOpenVDBDifference] = Options[OpenVDBDifference];


iOpenVDBDifference[OptionsPattern[]] :=
    Block[{vdb},
        vdb = OpenVDBCreateGrid["GridClass" -> "LevelSet"];
        (
            OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

            vdb

        ) /; OpenVDBGridQ[vdb]
    ]


iOpenVDBDifference[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] :=
    (
        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBDifference[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    Block[{union, vdbdiff},
        union = vdbUnion[vdbs];
        (
            vdbdiff = OpenVDBCreateGrid[vdb];
            vdbdiff["gridDifferenceCopy"[vdb[[1]], union[[1]]]];

            OpenVDBSetProperty[vdbdiff, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

            vdbdiff

        ) /; OpenVDBGridQ[union]
    ]


iOpenVDBDifference[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBDifference];


SyntaxInformation[OpenVDBDifference] = {"ArgumentsPattern" -> {___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Utilities*)


vdbUnion[vdb_] := vdb
vdbUnion[vdbs__] := iOpenVDBUnion[vdbs]


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBDifference[args___] := messageBooleanFunction[OpenVDBDifference, args]


(* ::Section:: *)
(*In place Boolean operations*)


(* ::Subsection::Closed:: *)
(*OpenVDBUnionTo*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBUnionTo] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBUnionTo[args___] /; !CheckArgs[OpenVDBUnionTo[args], {1, \[Infinity]}] = $Failed;


OpenVDBUnionTo[args___] :=
    With[{res = iOpenVDBUnionTo[args]},
        res /; res =!= $Failed
    ]


OpenVDBUnionTo[args___] := mOpenVDBUnionTo[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBUnionTo*)


Options[iOpenVDBUnionTo] = Options[OpenVDBUnionTo];


iOpenVDBUnionTo[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] :=
    (
        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBUnionTo[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    (
        Scan[vdb["gridUnion"[#]]&, {vdbs}[[All, 1]]];

        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBUnionTo[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBUnionTo];


SyntaxInformation[OpenVDBUnionTo] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBUnionTo[args___] := messageBooleanFunction[OpenVDBUnionTo, args]


(* ::Subsection::Closed:: *)
(*OpenVDBIntersectWith*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBIntersectWith] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBIntersectWith[args___] /; !CheckArgs[OpenVDBIntersectWith[args], {1, \[Infinity]}] = $Failed;


OpenVDBIntersectWith[args___] :=
    With[{res = iOpenVDBIntersectWith[args]},
        res /; res =!= $Failed
    ]


OpenVDBIntersectWith[args___] := mOpenVDBIntersectWith[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBIntersectWith*)


Options[iOpenVDBIntersectWith] = Options[OpenVDBIntersectWith];


iOpenVDBIntersectWith[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] :=
    (
        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBIntersectWith[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    (
        Scan[vdb["gridIntersection"[#]]&, {vdbs}[[All, 1]]];

        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBIntersectWith[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBIntersectWith];


SyntaxInformation[OpenVDBIntersectWith] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBIntersectWith[args___] := messageBooleanFunction[OpenVDBIntersectWith, args]


(* ::Subsection::Closed:: *)
(*OpenVDBDifferenceFrom*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBDifferenceFrom] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBDifferenceFrom[args___] /; !CheckArgs[OpenVDBDifferenceFrom[args], {1, \[Infinity]}] = $Failed;


OpenVDBDifferenceFrom[args___] :=
    With[{res = iOpenVDBDifferenceFrom[args]},
        res /; res =!= $Failed
    ]


OpenVDBDifferenceFrom[args___] := mOpenVDBDifferenceFrom[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBDifferenceFrom*)


Options[iOpenVDBDifferenceFrom] = Options[OpenVDBDifferenceFrom];


iOpenVDBDifferenceFrom[vdb_?OpenVDBScalarGridQ, OptionsPattern[]] :=
    (
        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBDifferenceFrom[vdb_?OpenVDBScalarGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    Block[{vdbunion},
        vdbunion = OpenVDBUnionTo[vdbs];
        (
            vdb["gridDifference"[vdbunion[[1]]]];

            OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

            vdb
        ) /; OpenVDBGridQ[vdbunion]
    ]


iOpenVDBDifferenceFrom[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBDifferenceFrom];


SyntaxInformation[OpenVDBDifferenceFrom] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBDifferenceFrom[args___] := messageBooleanFunction[OpenVDBUnion, args]


(* ::Section:: *)
(*Composite operations*)


(* ::Subsection::Closed:: *)
(*OpenVDBMax*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBMax] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBMax[args___] /; !CheckArguments[OpenVDBMax[args], {1, \[Infinity]}] = $Failed;


OpenVDBMax[args___] :=
    With[{res = iOpenVDBMax[args]},
        res /; res =!= $Failed
    ]


OpenVDBMax[args___] := mOpenVDBMax[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBMax*)


Options[iOpenVDBMax] = Options[OpenVDBMax];


iOpenVDBMax[vdb_?OpenVDBGridQ, OptionsPattern[]] :=
    Block[{vdbcopy},
        vdbcopy = OpenVDBCopyGrid[vdb];

        OpenVDBSetProperty[vdbcopy, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdbcopy
    ]


iOpenVDBMax[vdb_?OpenVDBGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    Block[{vdbcopy},
        vdbcopy = OpenVDBCopyGrid[vdb];

        (* don't copy all grids at once, instead pass one at a time and destroy the copy *)
        Do[iOpenVDBMaxOf[vdbcopy, OpenVDBCopyGrid[v]], {v, {vdbs}}];

        OpenVDBSetProperty[vdbcopy, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdbcopy
    ]


iOpenVDBMax[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBMax];


SyntaxInformation[OpenVDBMax] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBMax[args___] = $Failed;


(* ::Subsection::Closed:: *)
(*OpenVDBMin*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBMin] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBMin[args___] /; !CheckArguments[OpenVDBMin[args], {1, \[Infinity]}] = $Failed;


OpenVDBMin[args___] :=
    With[{res = iOpenVDBMin[args]},
        res /; res =!= $Failed
    ]


OpenVDBMin[args___] := mOpenVDBMin[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBMin*)


Options[iOpenVDBMin] = Options[OpenVDBMin];


iOpenVDBMin[vdb_?OpenVDBGridQ, OptionsPattern[]] :=
    Block[{vdbcopy},
        vdbcopy = OpenVDBCopyGrid[vdb];

        OpenVDBSetProperty[vdbcopy, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdbcopy
    ]


iOpenVDBMin[vdb_?OpenVDBGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    Block[{vdbcopy},
        vdbcopy = OpenVDBCopyGrid[vdb];

        (* don't copy all grids at once, instead pass one at a time and destroy the copy *)
        Do[iOpenVDBMinOf[vdbcopy, OpenVDBCopyGrid[v]], {v, {vdbs}}];

        OpenVDBSetProperty[vdbcopy, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdbcopy
    ]


iOpenVDBMin[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBMin];


SyntaxInformation[OpenVDBMin] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBMin[args___] = $Failed;


(* ::Section:: *)
(*In place Composite operations*)


(* ::Subsection::Closed:: *)
(*OpenVDBMaxOf*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBMaxOf] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBMaxOf[args___] /; !CheckArguments[OpenVDBMaxOf[args], {1, \[Infinity]}] = $Failed;


OpenVDBMaxOf[args___] :=
    With[{res = iOpenVDBMaxOf[args]},
        res /; res =!= $Failed
    ]


OpenVDBMaxOf[args___] := mOpenVDBMaxOf[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBMaxOf*)


Options[iOpenVDBMaxOf] = Options[OpenVDBMaxOf];


iOpenVDBMaxOf[vdb_?OpenVDBGridQ, OptionsPattern[]] :=
    (
        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBMaxOf[vdb_?OpenVDBGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    (
        Scan[vdb["gridMax"[#]]&, {vdbs}[[All, 1]]];

        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBMaxOf[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBMaxOf];


SyntaxInformation[OpenVDBMaxOf] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBMaxOf[args___] = $Failed;


(* ::Subsection::Closed:: *)
(*OpenVDBMinOf*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBMinOf] = {"Creator" -> Inherited, "Name" -> Inherited};


OpenVDBMinOf[args___] /; !CheckArguments[OpenVDBMinOf[args], {1, \[Infinity]}] = $Failed;


OpenVDBMinOf[args___] :=
    With[{res = iOpenVDBMinOf[args]},
        res /; res =!= $Failed
    ]


OpenVDBMinOf[args___] := mOpenVDBMinOf[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBMinOf*)


Options[iOpenVDBMinOf] = Options[OpenVDBMinOf];


iOpenVDBMinOf[vdb_?OpenVDBGridQ, OptionsPattern[]] :=
    (
        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBMinOf[vdb_?OpenVDBGridQ, vdbs__, OptionsPattern[]] /; sameGridTypeQ[vdb, vdbs] :=
    (
        Scan[vdb["gridMin"[#]]&, {vdbs}[[All, 1]]];

        OpenVDBSetProperty[vdb, {"Creator", "Name"}, OptionValue[{"Creator", "Name"}]];

        vdb
    )


iOpenVDBMinOf[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBMinOf];


SyntaxInformation[OpenVDBMinOf] = {"ArgumentsPattern" -> {_, ___, OptionsPattern[]}};


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBMinOf[args___] = $Failed;


(* ::Section:: *)
(*Clipping*)


(* ::Subsection::Closed:: *)
(*OpenVDBClip*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBClip] = {"CloseBoundary" -> True, "Creator" -> Inherited, "Name" -> Inherited};


OpenVDBClip[args___] /; !CheckArgs[OpenVDBClip[args], 2] = $Failed;


OpenVDBClip[args___] :=
    With[{res = pOpenVDBClip[args]},
        res /; res =!= $Failed
    ]


OpenVDBClip[args___] := mOpenVDBClip[args]


(* ::Subsubsection::Closed:: *)
(*pOpenVDBClip*)


Options[pOpenVDBClip] = Options[OpenVDBClip];


pOpenVDBClip[vdb_?OpenVDBScalarGridQ, bspec_List -> regime_?regimeQ, opts:OptionsPattern[]] :=
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


pOpenVDBClip[vdb_, bds_List, opts:OptionsPattern[]] := pOpenVDBClip[vdb, bds -> $worldregime, opts]


pOpenVDBClip[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[pOpenVDBClip, 1];


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


(* ::Subsubsection::Closed:: *)
(*Messages*)


mOpenVDBClip[expr_, ___] /; messageScalarGridQ[expr, OpenVDBClip] = $Failed;


mOpenVDBClip[_, bbox_, ___] /; message3DBBoxQ[bbox, OpenVDBClip] = $Failed;


mOpenVDBClip[___] = $Failed;


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*messageBooleanFunction*)


Options[messageBooleanFunction] = Options[OpenVDBUnion];


messageBooleanFunction[head_, vdbs___, OptionsPattern[]] :=
    Catch[
        Do[
            If[messageScalarGridQ[vdb, head], Throw[$Failed]],
            {vdb, {vdbs}}
        ];

        messageSameGridTypeQ[vdbs, head];

        $Failed
    ]


messageBooleanFunction[___] = $Failed;
