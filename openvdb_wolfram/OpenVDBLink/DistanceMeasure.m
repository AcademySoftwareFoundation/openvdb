(* ::Package:: *)

(* ::Title:: *)
(*DistanceMeasure*)


(* ::Subtitle:: *)
(*Perform membership, nearest, and distance queries on a grid.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


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


Options[OpenVDBMember] = {"IsoValue" -> Automatic};


OpenVDBMember[args___] /; !CheckArgs[OpenVDBMember[args], 2] = $Failed;


OpenVDBMember[args___] :=
    With[{res = iOpenVDBMember[args]},
        res /; res =!= $Failed
    ]


OpenVDBMember[args___] := messageCPTFunction[OpenVDBMember, args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBMember*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[iOpenVDBMember] = Options[OpenVDBMember];


iOpenVDBMember[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, OptionsPattern[]] :=
    Block[{isovalue, mems},
        isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
        (
            mems = vdb["gridMember"[pts, isovalue]];

            If[fogVolumeQ[vdb], Subtract[1, mems], mems] /; VectorQ[mems, IntegerQ]

        ) /; realQ[isovalue]
    ]


iOpenVDBMember[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, opts___] :=
    iOpenVDBMember[vdb, regimeConvert[vdb, pts, $worldregime -> $indexregime] -> $indexregime, opts]


iOpenVDBMember[vdb_, pts_?coordinateQ -> regime_, opts___] :=
    With[{res = iOpenVDBMember[vdb, {pts} -> regime, opts]},
        res[[1]] /; VectorQ[res, IntegerQ] && Length[res] === 1
    ]


iOpenVDBMember[vdb_, pts_List, opts___] := iOpenVDBMember[vdb, pts -> $worldregime, opts]


iOpenVDBMember[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBMember, 1];


SyntaxInformation[OpenVDBMember] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBMember] = $worldregime;


(* ::Section:: *)
(*Nearest*)


(* ::Subsection::Closed:: *)
(*OpenVDBNearest*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBNearest] = {"IsoValue" -> Automatic};


OpenVDBNearest[args___] /; !CheckArgs[OpenVDBNearest[args], 2] = $Failed;


OpenVDBNearest[args___] :=
    With[{res = iOpenVDBNearest[args]},
        res /; res =!= $Failed
    ]


OpenVDBNearest[args___] := messageCPTFunction[OpenVDBNearest, args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBNearest*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[iOpenVDBNearest] = Options[OpenVDBNearest];


iOpenVDBNearest[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, OptionsPattern[]] :=
    Block[{isovalue, nearest},
        isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
        (
            nearest = vdb["gridNearest"[pts, isovalue]];

            nearest /; MatrixQ[nearest, NumericQ]

        ) /; realQ[isovalue]
    ]


iOpenVDBNearest[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, opts___] :=
    iOpenVDBNearest[vdb, regimeConvert[vdb, pts, $indexregime -> $worldregime] -> $worldregime, opts]


iOpenVDBNearest[vdb_, pts_?coordinateQ -> regime_, opts___] :=
    With[{res = iOpenVDBNearest[vdb, {pts} -> regime, opts]},
        res[[1]] /; MatrixQ[res] && Dimensions[res] === {1, 3}
    ]


iOpenVDBNearest[vdb_, pts_List, opts___] := iOpenVDBNearest[vdb, pts -> $worldregime, opts]


iOpenVDBNearest[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBNearest, 1];


SyntaxInformation[OpenVDBNearest] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBNearest] = $worldregime;


(* ::Section:: *)
(*Distance*)


(* ::Subsection::Closed:: *)
(*OpenVDBDistance*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBDistance] = {"IsoValue" -> Automatic};


OpenVDBDistance[args___] /; !CheckArgs[OpenVDBDistance[args], 2] = $Failed;


OpenVDBDistance[args___] :=
    With[{res = iOpenVDBDistance[args]},
        res /; res =!= $Failed
    ]


OpenVDBDistance[args___] := messageCPTFunction[OpenVDBDistance, args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBDistance*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[iOpenVDBDistance] = Options[OpenVDBDistance];


iOpenVDBDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, OptionsPattern[]] :=
    Block[{isovalue, dists},
        isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
        (
            dists = vdb["gridDistance"[pts, isovalue]];

            dists /; VectorQ[dists, NumericQ]

        ) /; realQ[isovalue]
    ]


iOpenVDBDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, opts___] :=
    iOpenVDBDistance[vdb, regimeConvert[vdb, pts, $indexregime -> $worldregime] -> $worldregime, opts]


iOpenVDBDistance[vdb_, pts_?coordinateQ -> regime_, opts___] :=
    With[{res = iOpenVDBDistance[vdb, {pts} -> regime, opts]},
        res[[1]] /; VectorQ[res, NumericQ] && Length[res] === 1
    ]


iOpenVDBDistance[vdb_, pts_List, opts___] := iOpenVDBDistance[vdb, pts -> $worldregime, opts]


iOpenVDBDistance[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBDistance, 1];


SyntaxInformation[OpenVDBDistance] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBDistance] = $worldregime;


(* ::Subsection::Closed:: *)
(*OpenVDBSignedDistance*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBSignedDistance] = {"IsoValue" -> Automatic};


OpenVDBSignedDistance[args___] /; !CheckArgs[OpenVDBSignedDistance[args], 2] = $Failed;


OpenVDBSignedDistance[args___] :=
    With[{res = iOpenVDBSignedDistance[args]},
        res /; res =!= $Failed
    ]


OpenVDBSignedDistance[args___] := messageCPTFunction[OpenVDBSignedDistance, args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBSignedDistance*)


(* ::Text:: *)
(*Skipping call to regimeConvert to avoid duplicating points when unnecessary.*)


Options[iOpenVDBSignedDistance] = Options[OpenVDBSignedDistance];


iOpenVDBSignedDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $worldregime, OptionsPattern[]] :=
    Block[{isovalue, dists},
        isovalue = gridIsoValue[OptionValue["IsoValue"], vdb];
        (
            dists = vdb["gridSignedDistance"[pts, isovalue]];

            If[fogVolumeQ[vdb], Minus[dists], dists] /; VectorQ[dists, NumericQ]

        ) /; realQ[isovalue]
    ]


iOpenVDBSignedDistance[vdb_?OpenVDBScalarGridQ, pts_?coordinatesQ -> $indexregime, opts___] :=
    iOpenVDBSignedDistance[vdb, regimeConvert[vdb, pts, $indexregime -> $worldregime] -> $worldregime, opts]


iOpenVDBSignedDistance[vdb_, pts_?coordinateQ -> regime_, opts___] :=
    With[{res = iOpenVDBSignedDistance[vdb, {pts} -> regime, opts]},
        res[[1]] /; VectorQ[res, NumericQ] && Length[res] === 1
    ]


iOpenVDBSignedDistance[vdb_, pts_List, opts___] := iOpenVDBSignedDistance[vdb, pts -> $worldregime, opts]


iOpenVDBSignedDistance[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBSignedDistance, 1];


SyntaxInformation[OpenVDBSignedDistance] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};


OpenVDBDefaultSpace[OpenVDBSignedDistance] = $worldregime;


(* ::Section:: *)
(*Balls*)


(* ::Subsection::Closed:: *)
(*OpenVDBFillWithBalls*)


(* ::Subsubsection::Closed:: *)
(*Main*)


Options[OpenVDBFillWithBalls] = {"IsoValue" -> Automatic, "Overlapping" -> False, "ReturnType" -> Automatic, "SeedCount" -> Automatic};


OpenVDBFillWithBalls[args___] /; !CheckArgs[OpenVDBFillWithBalls[args], {2, 3}] = $Failed;


OpenVDBFillWithBalls[args___] :=
    With[{res = iOpenVDBFillWithBalls[args]},
        res /; res =!= $Failed
    ]


OpenVDBFillWithBalls[args___] := mOpenVDBFillWithBalls[args]


(* ::Subsubsection::Closed:: *)
(*iOpenVDBFillWithBalls*)


Options[iOpenVDBFillWithBalls] = Options[OpenVDBFillWithBalls];


iOpenVDBFillWithBalls[vdb_?OpenVDBScalarGridQ, n_Integer?Positive, rspec_, OptionsPattern[]] :=
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


iOpenVDBFillWithBalls[vdb_?OpenVDBScalarGridQ, n_, opts:OptionsPattern[]] :=
    iOpenVDBFillWithBalls[vdb, n, {0.5vdb["VoxelSize"], \[Infinity]} -> $worldregime, opts]


iOpenVDBFillWithBalls[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBFillWithBalls, 1];


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
parseBallReturnType[___] = $Failed


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


(* ::Subsubsection::Closed:: *)
(*Messages*)


Options[mOpenVDBFillWithBalls] = Options[OpenVDBFillWithBalls];


mOpenVDBFillWithBalls[expr_, ___] /; messageScalarGridQ[expr, OpenVDBFillWithBalls] = $Failed;


mOpenVDBFillWithBalls[vdb_, expr_, rest___] /; !IntegerQ[expr] || !TrueQ[expr > 0] :=
    (
        Message[OpenVDBFillWithBalls::intpm, HoldForm[OpenVDBFillWithBalls[vdb, expr, rest]], 2];
        $Failed
    )


mOpenVDBFillWithBalls[vdb_, n_, rspec_List -> regime_, args___] := messageRegimeSpecQ[regime, OpenVDBFillWithBalls] || mOpenVDBFillWithBalls[vdb, n, rspec, args]


mOpenVDBFillWithBalls[vdb_, n_, rspec_, rest___] /; !MatchQ[rspec, {a_, b_} /; 0 <= a <= b] :=
    (
        Message[OpenVDBFillWithBalls::rspec, 3, HoldForm[OpenVDBFillWithBalls[vdb, n, rspec, rest]]];
        $Failed
    )


mOpenVDBFillWithBalls[args__, Longest[OptionsPattern[]]] :=
    (
        If[messageIsoValueQ[OptionValue["IsoValue"], OpenVDBFillWithBalls],
            Return[$Failed]
        ];

        If[parseBallReturnType[OptionValue["ReturnType"]] === $Failed,
            Message[OpenVDBFillWithBalls::rettype];
            Return[$Failed]
        ];

        If[parseBallSeedCount[OptionValue["SeedCount"]] === $Failed,
            Message[OpenVDBFillWithBalls::seedcnt];
            Return[$Failed]
        ];

        $Failed
    )


mOpenVDBFillWithBalls[___] = $Failed;


OpenVDBFillWithBalls::rspec = "`1` at position `2` should be a list of two increasing non\[Hyphen]negative radii.";
OpenVDBFillWithBalls::rettype = "The setting for \"ReturnType\" should be one of \"Regions\", \"PackedArray\", or Automatic.";
OpenVDBFillWithBalls::seedcnt = "The setting for \"SeedCount\" should either be a positive integer or Automatic.";


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*messageCPTFunction*)


Options[messageCPTFunction] = Options[OpenVDBMember];


messageCPTFunction[head_, expr_, ___] /; messageScalarGridQ[expr, head] = $Failed;


messageCPTFunction[head_, _, expr_, ___] /; messageCoordinateSpecQ[expr, head] = $Failed;


messageCPTFunction[head_, args__, Longest[OptionsPattern[]]] :=
    (
        If[messageIsoValueQ[OptionValue["IsoValue"], head],
            Return[$Failed]
        ];

        $Failed
    )


messageCPTFunction[___] = $Failed;
