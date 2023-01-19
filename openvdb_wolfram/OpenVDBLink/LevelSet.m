(* ::Package:: *)

(* ::Title:: *)
(*LevelSet*)


(* ::Subtitle:: *)
(*Create a level set representation of a region.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBLevelSet"]


OpenVDBLevelSet::usage = "OpenVDBLevelSet[reg] creates a signed distance level set representation of reg.";


(* ::Section:: *)
(*OpenVDBLevelSet*)


(* ::Subsection::Closed:: *)
(*OpenVDBLevelSet*)


Options[OpenVDBLevelSet] = {"Creator" :> $OpenVDBCreator, "Name" -> None, "ScalarType" -> "Float"};


OpenVDBLevelSet[args___] /; !CheckArgs[OpenVDBLevelSet[args], {1, 3}] = $Failed;


OpenVDBLevelSet[args___] :=
    With[{res = pOpenVDBLevelSet[args]},
        res /; res =!= $Failed
    ]


OpenVDBLevelSet[args___] := mOpenVDBLevelSet[args]


(* ::Subsection::Closed:: *)
(*pOpenVDBLevelSet*)


Options[pOpenVDBLevelSet] = Options[OpenVDBLevelSet];


pOpenVDBLevelSet[expr_, opts:OptionsPattern[]] := pOpenVDBLevelSet[expr, $OpenVDBSpacing, $OpenVDBHalfWidth, opts]


pOpenVDBLevelSet[expr_, spacing_?Positive, opts:OptionsPattern[]] := pOpenVDBLevelSet[expr, spacing, $OpenVDBHalfWidth, opts]


pOpenVDBLevelSet[expr_, spacing_?Positive, width_?Positive, OptionsPattern[]] :=
    Block[{reg, type, vdb},
        reg = processSDFInput[expr];
        type = OptionValue["ScalarType"];
        (
            vdb = iOpenVDBLevelSet[reg, spacing, width, type, True];
            (
                OpenVDBSetProperty[vdb, "Creator" -> OptionValue["Creator"]];
                OpenVDBSetProperty[vdb, "Name" -> OptionValue["Name"]];

                vdb

            ) /; OpenVDBScalarGridQ[vdb]

        ) /; reg =!= $Failed && validScalarTypeQ[type]
    ]


pOpenVDBLevelSet[___] = $Failed;


(* ::Subsection::Closed:: *)
(*iOpenVDBLevelSet*)


(* ::Subsubsection::Closed:: *)
(*Surface mesh / complex*)


iOpenVDBLevelSet[mr_?triangleSurfaceMeshQ, args__] :=
    triangleSurfaceComplexSignedDistanceField[MeshCoordinates[mr], Join @@ MeshCells[mr, 2, "Multicells" -> True][[All, 1]], mr["ConnectivityMatrix"[1, 2]], args]


iOpenVDBLevelSet[{coords_?coordinatesQ, pcells:_List|_Polygon}, args__] :=
    With[{cells = stripPolygonCells[pcells]},
        triangleSurfaceComplexSignedDistanceField[coords, cells, None, args] /; validTriangleCellsQ[cells, Length[coords]]
    ]


iOpenVDBLevelSet[mr_?surfaceMesh3DQ, args__] :=
    Block[{tri},
        tri = Quiet[laxMeshBlock @ Region`Mesh`TriangulateMeshCells[mr, MaxCellMeasure -> \[Infinity]]];

        iOpenVDBLevelSet[tri, args] /; triangleSurfaceMeshQ[tri]
    ]


(* ::Subsubsection::Closed:: *)
(*Surface with thickness*)


iOpenVDBLevelSet[{mr_MeshRegion?triangleSurfaceMeshQ, r_?Positive}, args__] :=
    thickSurfaceSignedDistanceField[MeshCoordinates[mr], Join @@ MeshCells[mr, 2, "Multicells" -> True][[All, 1]], r, args]


iOpenVDBLevelSet[{coords_?coordinatesQ, pcells:_List|_Polygon, r_?Positive}, args__] :=
    With[{cells = stripPolygonCells[pcells]},
        thickSurfaceSignedDistanceField[coords, cells, r, args] /; validTriangleCellsQ[cells, Length[coords]]
    ]


thickSurfaceSignedDistanceField[coords_, cells_, r_, spacing_, width_, type_, signedQ_] :=
    Block[{vdb = OpenVDBCreateGrid[spacing, type]},
        vdb["offsetSurfaceLevelSet"[coords, cells-1, r, spacing, width, signedQ]];

        vdb
    ]


(* ::Subsubsection::Closed:: *)
(*Tube regions*)


iOpenVDBLevelSet[capsule_CapsuleShape?ConstantRegionQ, args__] := iOpenVDBLevelSet[Tube @@ capsule, args]


iOpenVDBLevelSet[Tube[ptspec_, r_?Positive], args__] :=
    Block[{segdata = tubeSegmentData[ptspec]},
        tubeComplexSignedDistanceField[Sequence @@ segdata, r, args] /; segdata =!= $Failed
    ]


iOpenVDBLevelSet[tube_Tube, args__] :=
    Block[{mr = Quiet[DiscretizeGraphics[tube]]},
        iOpenVDBLevelSet[mr, args] /; MeshRegionQ[mr]
    ]


iOpenVDBLevelSet[{mr_MeshRegion?lineMesh3DQ, r_?Positive}, args__] :=
    tubeComplexSignedDistanceField[MeshCoordinates[mr], Join @@ MeshCells[mr, 1, "Multicells" -> True][[All, 1]], r, args]


iOpenVDBLevelSet[{coords_?coordinatesQ, lcells:_List|_Line, r_?Positive}, args__] :=
    With[{cells = stripLineCells[lcells]},
        tubeComplexSignedDistanceField[coords, cells, r, args] /; validLineCellsQ[cells, Length[coords]]
    ]


tubeComplexSignedDistanceField[coords_, cells_, r_, spacing_, width_, type_, signedQ_] :=
    Block[{vdb = OpenVDBCreateGrid[spacing, type], coords2, cells2},
        coords2 = Join[coords, coords[[cells[[All, 2]]]] + .0001];
        cells2 = Transpose[Append[Transpose[cells], Range[Length[coords]+1, Length[coords]+Length[cells]]]]-1;

        vdb["offsetSurfaceLevelSet"[coords2, cells2, r, spacing, width, signedQ]];

        vdb
    ]


(* ::Subsubsection::Closed:: *)
(*Torus*)


iOpenVDBLevelSet[torus:(Torus|FilledTorus)[___]?ConstantRegionQ, spacing_, args__] :=
    Block[{torusspec, c, rin, rout, rmid, rtube, n, pts},
        torusspec = torusData @@ torus;
        (
            {c, rin, rout} = torusspec;

            rmid = N[Mean[{rin, rout}]];
            rtube = 0.5(rout - rin);
            n = Ceiling[2\[Pi]*rmid/spacing];

            pts = Append[c[[3]]] /@ CirclePoints[c[[1 ;; 2]], rmid, n];
            AppendTo[pts, First[pts]];

            iOpenVDBLevelSet[Tube[pts, rtube], spacing, args]

        ) /; torusspec =!= $Failed
    ]


torusData[] = {{0, 0, 0}, 0.5, 1.0};
torusData[c_?VectorQ] := {c, 0.5, 1.0}
torusData[c_?VectorQ, {rin_?Positive, rout_?Positive}] /; rout > rin := {c, rin, rout}
torusData[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Ball*)


iOpenVDBLevelSet[ball:(Ball|Sphere)[___]?ConstantRegionQ, spacing_, width_, type_, signedQ_] /; RegionEmbeddingDimension[ball] == 3 :=
    Block[{ballspec, c, r, vdb},
        ballspec = singleBallData @@ ball;
        (
            {c, r} = ballspec;
            vdb = OpenVDBCreateGrid[spacing, type];
            vdb["ballLevelSet"[c, r, spacing, width, signedQ]];

            vdb

        ) /; ballspec =!= $Failed
    ]


singleBallData[] = {{0, 0, 0}, 1};
singleBallData[3] = {{0, 0, 0}, 1}
singleBallData[c_?VectorQ] := {c, 1}
singleBallData[c_?VectorQ, r_?Positive] := {c, r}
singleBallData[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*SphericalShell*)


iOpenVDBLevelSet[shell:HoldPattern[SphericalShell][___]?ConstantRegionQ, args___] /; RegionEmbeddingDimension[shell] == 3 :=
    Block[{shellspec, c, r1, r2, ballout, ballin},
        shellspec = sphericalShellData @@ shell;
        (
            {c, {r1, r2}} = shellspec;
            ballout = iOpenVDBLevelSet[Ball[c, r2], args];
            (
                ballin = iOpenVDBLevelSet[Ball[c, r1], args];

                OpenVDBDifferenceFrom[ballout, ballin] /; OpenVDBScalarGridQ[ballin]

            ) /; OpenVDBScalarGridQ[ballout]

        ) /; shellspec =!= $Failed
    ]


sphericalShellData[] = {{0, 0, 0}, {1/2, 1}};
sphericalShellData[{r1_, r2_}] := {{0, 0, 0}, {r1, r2}}
sphericalShellData[c_?VectorQ] := {c, {1/2, 1}}
sphericalShellData[c_?VectorQ, r_?Positive] := {c, {r/2, r}}
sphericalShellData[c_?VectorQ, {r1_, r2_}] := {c, {r1, r2}}
sphericalShellData[r_] := {{0, 0, 0}, {r/2, r}}
sphericalShellData[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*Cuboid*)


iOpenVDBLevelSet[cuboid_Cuboid?ConstantRegionQ, spacing_, width_, type_, signedQ_] :=
    Block[{bds, vdb},
        bds = RegionBounds[cuboid];
        (
            vdb = OpenVDBCreateGrid[spacing, type];
            vdb["cuboidLevelSet"[bds, spacing, width, signedQ]];

            vdb

        ) /; And @@ Less @@@ bds
    ]


iOpenVDBLevelSet[hex_Hexahedron?ConstantRegionQ, args__] /; Volume[hex] == Volume[BoundingRegion[hex]] := iOpenVDBLevelSet[BoundingRegion[hex], args]


(* ::Subsubsection::Closed:: *)
(*Special polyhedra*)


iOpenVDBLevelSet[poly_?specialPolyhedonQ, args__] :=
    With[{data = polyhedronTriangleData[poly]},
        (
            triangleSurfaceComplexSignedDistanceField[##, None, args, False]& @@ data

        ) /; data =!= $Failed
    ]


(* ::Subsubsection::Closed:: *)
(*EmptyRegion / FullRegion*)


iOpenVDBLevelSet[EmptyRegion[3], spacing_, width_, ___] := OpenVDBCreateGrid[spacing, "BackgroundValue" -> spacing*width, "GridClass" -> $gridLevelSet]


iOpenVDBLevelSet[FullRegion[3], spacing_, __] := OpenVDBCreateGrid[spacing, "BackgroundValue" -> -10^12., "GridClass" -> $gridLevelSet]


(* ::Subsubsection::Closed:: *)
(*BooleanRegion*)


(* ::Text:: *)
(*Would be nice to allow any Boolean function, but then we will need Boolean operations that don't clear the second input, which we don't have right now.*)


iOpenVDBLevelSet[reg:BooleanRegion[bfunc_, regs_]?ConstantRegionQ, args__] /; RegionEmbeddingDimension[reg] == 3 :=
    Block[{op},
        op = booleanHead[bfunc, Length[regs]];
        (
            If[op === Or,
                unionRegionSDFs[regs, args],
                intersectRegionSDFs[regs, args]
            ]

        ) /; op =!= $Failed
    ]


booleanHead[bfunc_, n_] :=
    Block[{vars, expr},
        vars = \[FormalX] /@ Range[n];
        expr = bfunc @@ vars;
        Which[
            expr === Or @@ vars, Or,
            expr === And @@ vars, And,
            True, $Failed
        ]
    ]


unionRegionSDFs[args__] := booleanRegionSDFs["gridUnion", args]
intersectRegionSDFs[args__] := booleanRegionSDFs["gridIntersection", args]
booleanRegionSDFs[boolVDB_, regs_, args__] :=
    Block[{vdb1, vdb2},
        vdb1 = iOpenVDBLevelSet[First[regs], args];
        If[vdb1 === $Failed,
            Return[$Failed]
        ];

        Do[
            vdb2 = iOpenVDBLevelSet[reg, args];
            If[vdb2 === $Failed,
                OpenVDBDeleteGrid[vdb1];
                Return[$Failed]
            ];
            vdb1[boolVDB[vdb2[[1]]]];,
            {reg, Rest[regs]}
        ];

        vdb1
    ]


(* ::Subsubsection::Closed:: *)
(*TransformedRegion*)


iOpenVDBLevelSet[treg:TransformedRegion[reg_, tfunc_TransformationFunction]?ConstantRegionQ, args__] /; RegionEmbeddingDimension[treg] == 3 :=
    With[{res = iOpenVDBLevelSet[reg, args]},
        OpenVDBTransform[res, tfunc] /; res =!= $Failed
    ]


(* ::Subsubsection::Closed:: *)
(*RegionBoundary*)


iOpenVDBLevelSet[RegionBoundary[reg_], args__, _] := iOpenVDBLevelSet[reg, args, False]


(* ::Subsubsection::Closed:: *)
(*General 3D region*)


iOpenVDBLevelSet[reg_?ConstantRegionQ, args__] /; RegionEmbeddingDimension[reg] == 3 :=
    Block[{bmr},
        bmr = Quiet[laxMeshBlock @ BoundaryDiscretizeRegion[reg]];

        iOpenVDBLevelSet[bmr, args] /; BoundaryMeshRegionQ[bmr]
    ]


(* ::Subsubsection::Closed:: *)
(*Generic behavior*)


iOpenVDBLevelSet[rspec_List, args__] /; VectorQ[rspec, !NumericQ[#]&] := unionRegionSDFs[rspec, args]


iOpenVDBLevelSet[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


SyntaxInformation[OpenVDBLevelSet] = {"ArgumentsPattern" -> {_, _., _., OptionsPattern[]}};


(* ::Subsection::Closed:: *)
(*Messages*)


Options[mOpenVDBLevelSet] = Options[OpenVDBLevelSet];


mOpenVDBLevelSet[expr_, ___] /; !validToLevelSetQ[expr] :=
    (
        Message[OpenVDBLevelSet::reg, expr, 1];
        $Failed
    )


mOpenVDBLevelSet[_, OptionsPattern[]] /; !TrueQ[$OpenVDBSpacing > 0] :=
    (
        Message[OpenVDBLevelSet::novoxsz];
        $Failed
    )


mOpenVDBLevelSet[_, _., OptionsPattern[]] /; !TrueQ[$OpenVDBHalfWidth > 0] :=
    (
        Message[OpenVDBLevelSet::nowidth];
        $Failed
    )


mOpenVDBLevelSet[_, vx_, ___] /; !TrueQ[vx > 0] && !OptionQ[vx] :=
    (
        Message[OpenVDBLevelSet::nonpos, vx, 2];
        $Failed
    )


mOpenVDBLevelSet[_, _, w_, ___] /; !TrueQ[w > 0] && !OptionQ[w] :=
    (
        Message[OpenVDBLevelSet::nonpos, w, 3];
        $Failed
    )


mOpenVDBLevelSet[__, OptionsPattern[]] :=
    Block[{validQ = validScalarTypeQ[OptionValue["ScalarType"]]},
        (
            Message[OpenVDBLevelSet::nonscalar, OptionValue["ScalarType"]];
            $Failed
        ) /; !validQ
    ]


mOpenVDBLevelSet[___] = $Failed;


OpenVDBLevelSet::reg = "`1` at position `2` is not a constant 3D region.";


OpenVDBLevelSet::novoxsz = "No grid spacing is provided since $OpenVDBSpacing is not a positive number.";
OpenVDBLevelSet::nowidth = "No half band width is provided since $OpenVDBHalfWidth is not a positive number."


OpenVDBLevelSet::nonpos = "`1` at position `2` is expected to be a positive number";


OpenVDBLevelSet::nonscalar = "`1` is not a valid setting for \"ScalarType\". Evaluate OpenVDBGridTypes[\"Scalar\"] for a list of valid types.";


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*Coordinate & cell utilities*)


validTriangleCellsQ[cells_, max_] := MatrixQ[cells, IntegerQ] && Length[cells[[1]]] == 3 && Min[cells] >= 1 && Max[cells] <= max
validLineCellsQ[cells_, max_] := MatrixQ[cells, IntegerQ] && Length[cells[[1]]] == 2 && Min[cells] >= 1 && Max[cells] <= max


stripPolygonCells[Polygon[cells_]] := cells
stripPolygonCells[pcells:{__Polygon}] :=
    With[{cells = pcells[[All, 1]]},
        Which[
            VectorQ[cells, VectorQ], cells,
            VectorQ[cells, MatrixQ], Join @@ cells,
            VectorQ[cells, ArrayQ[#, 1|2]&], Join[Join @@ Select[cells, MatrixQ], Select[cells, VectorQ]],
            True, $Failed
        ]
    ]
stripPolygonCells[expr_] := expr


stripLineCells[Line[cells_]] := cells
stripLineCells[lcells:{__Line}] :=
    With[{cells = lcells[[All, 1]]},
        Which[
            VectorQ[cells, VectorQ], cells,
            VectorQ[cells, MatrixQ], Join @@ cells,
            VectorQ[cells, ArrayQ[#, 1|2]&], Join[Join @@ Select[cells, MatrixQ], Select[cells, VectorQ]],
            True, $Failed
        ]
    ]
stripLineCells[{i1_Integer, i2_Integer}] := {{i1, i2}}
stripLineCells[expr_] := expr


(* ::Subsection::Closed:: *)
(*processSDFInput*)


processSDFInput[expr_] :=
    With[{res = Join @@ expandMultisetRegion /@ If[ListQ[expr], Identity, List][expr]},
        If[Length[res] == 1,
            First[res],
            res
        ]
    ];


expandMultisetRegion[(h:Ball | Sphere)[pts_?MatrixQ, r___]?ConstantRegionQ] /; Length[pts[[1]]] == 3 := h[#, r]& /@ pts


expandMultisetRegion[reg:(Cone | Cylinder)[pts_?ArrayQ, ___]?ConstantRegionQ] /; MatchQ[Dimensions[pts], {_, _, 3}] := Thread[reg]


expandMultisetRegion[reg:(Hexahedron | Prism | Pyramid | Tetrahedron)[pts_?ArrayQ]?ConstantRegionQ] /; MatchQ[Dimensions[pts], {_, _, 3}] := Thread[reg]


expandMultisetRegion[reg_] := {reg}


(* ::Subsection::Closed:: *)
(*tubeSegmentData*)


tubeSegmentData[pts_List] /; MatrixQ[pts, realQ] && Length[pts[[1]]] == 3 := {pts, Partition[Range[Length[pts]], 2, 1]}


tubeSegmentData[pts_List] /; VectorQ[pts, MatrixQ[#, realQ] && Length[#[[1]]] == 3&] :=
    {
        Join @@ pts,
        Join @@ Plus[Partition[Range[Length[#]], 2, 1]& /@ pts, Prepend[Most[Accumulate[Length /@ pts]], 0]]
    }


tubeSegmentData[pts_List] :=
    With[{data = tubeSegmentData /@ pts},
        (
            {
                Join @@ data[[All, 1]],
                Join @@ Plus[data[[All, 2]], Prepend[Most[Accumulate[Length /@ data[[All, 1]]]], 0]]
            }

        ) /; FreeQ[data, $Failed, {1}]
    ]


tubeSegmentData[Line[pts_]] := tubeSegmentData[pts]


tubeSegmentData[bc:(_BSplineCurve|_BezierCurve)] :=
    With[{mr = Quiet[DiscretizeGraphics[bc]]},
        {MeshCoordinates[mr], Join @@ MeshCells[mr, 1, "Multicell" -> True][[All, 1]]} /; MeshRegionQ[mr] && RegionDimension[mr] == 1
    ]


tubeSegmentData[___] = $Failed


(* ::Subsection::Closed:: *)
(*polyhedronTriangleData*)


specialPolyhedonQ[reg_] := TrueQ[specialPolyhedonHead[Head[reg]]] && ConstantRegionQ[reg] && RegionEmbeddingDimension[reg] === 3
specialPolyhedonHead[Cube] = True;
specialPolyhedonHead[Cuboid] = True;
specialPolyhedonHead[Dodecahedron] = True;
specialPolyhedonHead[Hexahedron] = True;
specialPolyhedonHead[Icosahedron] = True;
specialPolyhedonHead[Octahedron] = True;
specialPolyhedonHead[Parallelepiped] = True;
specialPolyhedonHead[Prism] = True;
specialPolyhedonHead[Pyramid] = True;
specialPolyhedonHead[Simplex] = True;
specialPolyhedonHead[Tetrahedron] = True;


polyhedronTriangleData[c_Cube] := cubeTriangleData @@ c
polyhedronTriangleData[c_Cuboid] := cuboidTriangleData @@ c
polyhedronTriangleData[d_Dodecahedron] := dodecahedronTriangleData @@ d
polyhedronTriangleData[h_Hexahedron] := hexahedronTriangleData @@ h
polyhedronTriangleData[i_Icosahedron] := icosahedronTriangleData @@ i
polyhedronTriangleData[o_Octahedron] := octahedronTriangleData @@ o
polyhedronTriangleData[p_Parallelepiped] := parallelepipedTriangleData @@ p
polyhedronTriangleData[p_Prism] := prismTriangleData @@ p
polyhedronTriangleData[p_Pyramid] := pyramidTriangleData @@ p
polyhedronTriangleData[s_Simplex] := simplexTriangleData @@ s
polyhedronTriangleData[t_Tetrahedron] := tetrahedronTriangleData @@ t
polyhedronTriangleData[___] = $Failed;


platonicSpecs[args___] :=
    Block[{data = {args}, center, angles, l, cuboidres},
        center = {0, 0, 0};
        angles = {0, 0};
        l = 1;

        Switch[Prepend[If[ListQ[#], Length[#], 0]& /@ data, Length[data]],
            {0}, Null,
            {1, 0}, {l} = data,
            {1, 2}, {angles} = data,
            {1, 3}, {center} = data,
            {2, 2, 0}, {angles, l} = data,
            {2, 3, 0}, {center, l} = data,
            {2, 3, 2}, {center, angles} = data,
            {3, __}, {center, angles, l} = data,
            _, Return[$Failed]
        ];

        {center, angles, l}
    ]


rotatePolyhedronCoordinates[coords_, {\[Theta]_, \[Phi]_}, center_] := (RotationTransform[\[Phi], {0, 1, 0}, center] @* RotationTransform[\[Theta], {0, 0, 1}, center])[coords]


cubeTriangleData[args___] :=
    Block[{center, angles, l, cuboidres},
        {center, angles, l} = platonicSpecs[args];

        cuboidres = cuboidTriangleData[center - 0.5l{1, 1, 1}, center + 0.5l{1, 1, 1}];
        If[angles =!= {0, 0},
            cuboidres[[1]] = rotatePolyhedronCoordinates[cuboidres[[1]], angles, center];
        ];

        cuboidres
    ]


cuboidTriangleData[l_] := cuboidTriangleData[l, l+1]
cuboidTriangleData[l_, u_] :=
    {
        Tuples[Transpose[{l, u}]],
        {{1,2,4},{1,4,3},{1,5,6},{1,6,2},{1,7,5},{1,3,7},{2,8,4},{2,6,8},{3,4,8},{3,8,7},{5,7,6},{6,7,8}}
    }


$dodecacoords = {{-1.3763819204711736, 0., 0.2628655560595668}, {1.3763819204711736, 0., -0.2628655560595668}, {-0.42532540417601994, -1.3090169943749475, 0.2628655560595668}, {-0.42532540417601994, 1.3090169943749475, 0.2628655560595668}, {1.1135163644116066, -0.8090169943749475, 0.2628655560595668}, {1.1135163644116066, 0.8090169943749475, 0.2628655560595668}, {-0.2628655560595668, -0.8090169943749475, 1.1135163644116066}, {-0.2628655560595668, 0.8090169943749475, 1.1135163644116066}, {-0.6881909602355868, -0.5, -1.1135163644116068}, {-0.6881909602355868, 0.5, -1.1135163644116068}, {0.6881909602355868, -0.5, 1.1135163644116066}, {0.6881909602355868, 0.5, 1.1135163644116066}, {0.85065080835204, 0., -1.1135163644116068}, {-1.1135163644116068, -0.8090169943749475, -0.2628655560595668}, {-1.1135163644116068, 0.8090169943749475, -0.2628655560595668}, {-0.8506508083520399, 0., 1.1135163644116066}, {0.2628655560595668, -0.8090169943749475, -1.1135163644116068}, {0.2628655560595668, 0.8090169943749475, -1.1135163644116068}, {0.42532540417601994, -1.3090169943749475, -0.2628655560595668}, {0.42532540417601994, 1.3090169943749475, -0.2628655560595668}};
dodecahedronTriangleData[args___] :=
    Block[{center, angles, l, dcoords, dcells},
        {center, angles, l} = platonicSpecs[args];

        dcoords = Transpose[Transpose[l*$dodecacoords] + center];
        If[angles =!= {0, 0},
            dcoords = rotatePolyhedronCoordinates[dcoords, angles, center];
        ];

        dcells = {{15,10,9},{15,9,14},{15,14,1},{2,6,12},{2,12,11},{2,11,5},{5,11,7},
            {5,7,3},{5,3,19},{11,12,8},{11,8,16},{11,16,7},{12,6,20},{12,20,4},{12,4,8},
            {6,2,13},{6,13,18},{6,18,20},{2,5,19},{2,19,17},{2,17,13},{4,20,18},{4,18,10},
            {4,10,15},{18,13,17},{18,17,9},{18,9,10},{17,19,3},{17,3,14},{17,14,9},{3,7,16},
            {3,16,1},{3,1,14},{16,8,4},{16,4,15},{16,15,1}};

        {dcoords, dcells}
    ]


hexahedronTriangleData[pts_] := {pts, {{1,3,2},{1,4,3},{1,6,5},{1,2,6},{1,5,8},{1,8,4},{3,4,8},{3,8,7},{2,3,7},{2,7,6},{5,6,8},{6,7,8}}}


$icosacoords = {{0., 0., -0.9510565162951536}, {0., 0., 0.9510565162951536}, {-0.85065080835204, 0., -0.42532540417601994}, {0.85065080835204, 0., 0.42532540417601994}, {0.6881909602355868, -0.5, -0.42532540417601994}, {0.6881909602355868, 0.5, -0.42532540417601994}, {-0.6881909602355868, -0.5, 0.42532540417601994}, {-0.6881909602355868, 0.5, 0.42532540417601994}, {-0.2628655560595668, -0.8090169943749475, -0.42532540417601994}, {-0.2628655560595668, 0.8090169943749475, -0.42532540417601994}, {0.2628655560595668, -0.8090169943749475, 0.42532540417601994}, {0.2628655560595668, 0.8090169943749475, 0.42532540417601994}};
icosahedronTriangleData[args___] :=
    Block[{center, angles, l, icoords, icells},
        {center, angles, l} = platonicSpecs[args];

        icoords = Transpose[Transpose[l*$icosacoords] + center];
        If[angles =!= {0, 0},
            icoords = rotatePolyhedronCoordinates[icoords, angles, center];
        ];

        icells = {{2,12,8},{2,8,7},{2,7,11},{2,11,4},{2,4,12},{5,9,1},{6,5,1},{10,6,1},
            {3,10,1},{9,3,1},{12,10,8},{8,3,7},{7,9,11},{11,5,4},{4,6,12},{5,11,9},
            {6,4,5},{10,12,6},{3,8,10},{9,7,3}};

        {icoords, icells}
    ]


$octacoords = {{0., 0.7071067811865475, 0.}, {0.7071067811865475, 0., 0.}, {0., -0.7071067811865475, 0.}, {-0.7071067811865475, 0., 0.}, {0., 0., 0.7071067811865475}, {0., 0., -0.7071067811865475}};
octahedronTriangleData[args___] :=
    Block[{center, angles, l, ocoords, ocells},
        {center, angles, l} = platonicSpecs[args];

        ocoords = Transpose[Transpose[l*$octacoords] + center];
        If[angles =!= {0, 0},
            ocoords = rotatePolyhedronCoordinates[ocoords, angles, center];
        ];

        ocells = {{5,2,1},{5,3,2},{5,4,3},{4,5,1},{2,6,1},{2,3,6},{4,6,3},{1,6,4}};

        {ocoords, ocells}
    ]


parallelepipedTriangleData[center_, {v1_, v2_, v3_}] :=
    {
        Transpose[Transpose[{{0,0,0}, v1, v2, v1+v2, v3, v1+v3, v2+v3, v1+v2+v3}] + center],
        {{1,2,4},{1,4,3},{1,5,6},{1,6,2},{1,7,5},{1,3,7},{2,8,4},{2,6,8},{3,4,8},{3,8,7},{5,7,6},{6,7,8}}
    }


prismTriangleData[pts_] := {pts, {{1,3,2},{4,5,6},{2,3,6},{2,6,5},{1,2,5},{1,5,4},{1,6,3},{1,4,6}}}


pyramidTriangleData[pts_] := {pts, {{1,2,5},{2,3,5},{3,4,5},{4,1,5},{1,3,2},{1,4,3}}}


simplexTriangleData[3] = simplexTriangleData[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]
simplexTriangleData[pts_] := tetrahedronTriangleData[pts]


$tetcoords = {{0., 0., 0.6123724356957945}, {-0.2886751345948129, -0.5, -0.20412414523193154}, {-0.2886751345948129, 0.5, -0.20412414523193154}, {0.5773502691896258, 0., -0.20412414523193154}};
tetrahedronTriangleData[pts_?MatrixQ] := {pts, {{1,2,4},{1,3,2},{1,4,3},{2,3,4}}}
tetrahedronTriangleData[args___] :=
    Block[{center, angles, l, tetres},
        {center, angles, l} = platonicSpecs[args];

        tetres = tetrahedronTriangleData[Transpose[Transpose[l*$tetcoords] + center]];
        If[angles =!= {0, 0},
            tetres[[1]] = rotatePolyhedronCoordinates[tetres[[1]], angles, center];
        ];

        tetres
    ]


(* ::Subsection::Closed:: *)
(*triangleSurfaceComplexSignedDistanceField*)


triangleSurfaceComplexSignedDistanceField[coords_, cells_, C12_, spacing_, width_, type_, signedQ_, nestingQ_:True] :=
    Block[{vdb, nestedcells, nestedvdb},
        vdb = OpenVDBCreateGrid[spacing, type];
        nestedcells = If[TrueQ[nestingQ] && ArrayQ[C12],
            nestedComponentHierarchy[coords, cells, C12],
            {cells}
        ];

        vdb["meshLevelSet"[coords, nestedcells[[1]] - 1, spacing, width, signedQ]];

        Do[
            nestedvdb = OpenVDBCreateGrid[spacing, type];
            nestedvdb["meshLevelSet"[coords, nestedcells[[i]] - 1, spacing, width, signedQ]];
            If[EvenQ[i],
                OpenVDBDifferenceFrom[vdb, nestedvdb],
                OpenVDBUnionTo[vdb, nestedvdb]
            ],
            {i, 2, Length[nestedcells]}
        ];

        vdb
    ]


nestedComponentHierarchy[coords_, cells_, C12_] :=
    Block[{C22, comps, conncells, n, adj, depths, depthmembers},
        C22 = triangleTriangleConnectivity[coords, cells, C12];
        comps = SparseArray`StronglyConnectedComponents[C22];

        (* only one component *)
        If[Length[comps] == 1, Return[{cells}]];

        conncells = cells[[#]]& /@ comps;

        n = Length[conncells];
        adj = boundaryNestingAdjacency[coords, Polygon /@ conncells];
        depths = nestingDepths[n, adj];

        (* multiple components, but none nested *)
        If[Max[depths] == 0, Return[{cells}]];

        depthmembers = GatherBy[SortBy[Transpose[{depths, Range[n]}], First], First][[All, All, 2]];

        (Join @@ conncells[[#]])& /@ depthmembers
    ]


triangleTriangleConnectivity[_, _, C12_SparseArray] := Transpose[C12] . C12


triangleTriangleConnectivity[coords_, cells_, _] :=
    With[{C12 = edgeTriangleAdjacencyMatrix[coords, cells]},
        Transpose[C12] . C12
    ]


(* ::Text:: *)
(*https://mathematica.stackexchange.com/a/160444/4346*)


getEdgesFromTriangles = Compile[{{f, _Integer, 1}},
    {
        Sort[{Compile`GetElement[f, 1], Compile`GetElement[f, 2]}],
        Sort[{Compile`GetElement[f, 2], Compile`GetElement[f, 3]}],
        Sort[{Compile`GetElement[f, 3], Compile`GetElement[f, 1]}]
    },
    RuntimeAttributes -> {Listable},
    Parallelization -> True
];
takeSortedThread = Compile[{{data, _Integer, 1}, {ran, _Integer, 1}},
   Sort[Part[data, ran[[1]] ;; ran[[2]]]],
    RuntimeAttributes -> {Listable},
    Parallelization -> True
   ];
extractIntegerFromSparseMatrix = Compile[
   {{vals, _Integer, 1}, {rp, _Integer, 1}, {ci, _Integer,
     1}, {background, _Integer},
    {i, _Integer}, {j, _Integer}},
   Block[{k},
    k = rp[[i]] + 1;
    While[k < rp[[i + 1]] + 1 && ci[[k]] != j, ++k];
    If[k == rp[[i + 1]] + 1, background, vals[[k]]]
    ],
    RuntimeAttributes -> {Listable},
    Parallelization -> True
   ];


edgeTriangleAdjacencyMatrix[coords_, cells_] :=
 Module[{edgesfrompolygons, edges, edgelookupcontainer,
    polyranges, polygonsneighedges, edgepolygonadjacencymatrix, acc},
  edgesfrompolygons = Flatten[getEdgesFromTriangles[cells], 1];
 edges = DeleteDuplicates[edgesfrompolygons];
  edgelookupcontainer =
   SparseArray[
    Rule[Join[edges, Transpose[Transpose[edges][[{2, 1}]]]],
     Join[Range[1, Length[edges]], Range[1, Length[edges]]]], {Length[coords], Length[coords]}];
  acc = Range[0, 3Length[cells], 3];
  polyranges = Transpose[{Most[acc] + 1, Rest[acc]}];
  polygonsneighedges = takeSortedThread[extractIntegerFromSparseMatrix[
      edgelookupcontainer["NonzeroValues"],
      edgelookupcontainer["RowPointers"],
      Flatten@edgelookupcontainer["ColumnIndices"],
      edgelookupcontainer["Background"],
      edgesfrompolygons[[All, 1]],
      edgesfrompolygons[[All, 2]]],
     polyranges];
  Transpose@With[{
      n = Length[edges], m = Length[cells],
      data = Flatten[polygonsneighedges]
      },
     SparseArray @@ {Automatic, {m, n},
       0, {1, {acc, Transpose[{data}]}, ConstantArray[1, Length[data]]}}
     ]
  ]


(* ::Subsection::Closed:: *)
(*Mesh utilities*)


(* ::Subsubsection::Closed:: *)
(*Q functions*)


meshQ[expr_] := MeshRegionQ[expr] || BoundaryMeshRegionQ[expr]


mesh3DQ[expr_] := (MeshRegionQ[expr] || BoundaryMeshRegionQ[expr]) && RegionEmbeddingDimension[expr] === 3


surfaceMesh3DQ[mr_] :=
    And[
        mesh3DQ[mr],
        BoundaryMeshRegionQ[mr] || RegionDimension[mr] === 2,
        (* has faces and all faces are triangles -- this allows for meshes with points, lines, etc, which will all be ignored *)
        !FreeQ[mr["MeshCellTypes"], {Polygon, {_, _}}]
    ]


triangleSurfaceMeshQ[mr_] := surfaceMesh3DQ[mr] && FreeQ[mr["MeshCellTypes"], {Polygon, {_, Except[3]}}]


lineMesh3DQ[mr_] := mesh3DQ[mr] && RegionDimension[mr] === 1


(* ::Subsubsection::Closed:: *)
(*laxMeshBlock*)


SetAttributes[laxMeshBlock, HoldFirst];


laxMeshBlock[code_] :=
    Block[{bmethod},
        Internal`WithLocalSettings[
            bmethod = OptionValue[BoundaryMeshRegion, Method];
            SetOptions[BoundaryMeshRegion, Method -> Join[{"CheckIntersections" -> False}, Replace[bmethod, Except[_List] -> {}, {0}]]];,

            code,

            SetOptions[BoundaryMeshRegion, Method -> bmethod];
        ]
    ]


(* ::Subsubsection::Closed:: *)
(*Nesting*)


(* ::Text:: *)
(*Uses the same idea as Region`Mesh`BoundaryNestingArrays, but only tests one point for crossing count instead of all points. This assumes no intersecting facets.*)


polygonBoundingBox[Polygon[coords_]] := CoordinateBoundingBox[coords]
polygonBoundingBox[data_List] := CoordinateBoundingBox[polygonBoundingBox /@ data]


polygonCoordinate[Polygon[data_]] := iPolygonCoordinate[data];
polygonCoordinate[data_List] := polygonCoordinate[First[data]];


iPolygonCoordinate[lis_List] :=
    If[Length[lis] == 3 && VectorQ[lis, NumericQ],
        lis,
        iPolygonCoordinate[lis[[1]]]
    ]


boundingBoxNesting[{min1_, max1_}, {min2_, max2_}] :=
    Module[{minless, maxless},

        If[Or @@ MapThread[Less, {max1, min2}],
            Return["Disjoint"]
        ];

        If[Or @@ MapThread[Less, {max2, min1}],
            Return["Disjoint"]
        ];

        minless = Union[MapThread[Less, {min1, min2}]];
        maxless = Union[MapThread[Less, {max1, max2}]];

        If[Length[minless] > 1 || Length[maxless] > 1 || minless === maxless,
            Return[Indeterminate]
        ];

        If[TrueQ[First[minless]],
            {"Inside", 2, 1},
            {"Inside", 1, 2}
        ]
    ]


pointInsideQ[polys_, pt_] := OddQ[Region`Mesh`CrossingCount[polys, pt]]


(* ::Text:: *)
(*Returns {{i1, j1}, {i2, j2}, ...} where the i1 component contains the j1 component, etc.*)


boundaryNestingAdjacency[coords_, cells_] :=
    Block[{np, polycomps, bb, pts, nesting},

        np = Length[cells];
        polycomps = Region`Mesh`ToCoordinates[cells, coords];
        bb = polygonBoundingBox /@ polycomps;
        pts = polygonCoordinate /@ polycomps;

        nesting = Reap[
            Do[
                Switch[
                    boundingBoxNesting[bb[[j]], bb[[i]]],
                    Indeterminate,
                        If[pointInsideQ[polycomps[[i]], pts[[j]]],
                            Sow[{i, j}],
                            If[pointInsideQ[polycomps[[j]], pts[[i]]],
                                Sow[{j, i}]
                            ]
                        ],
                    {"Inside", 1, 2},
                        If[pointInsideQ[polycomps[[i]], pts[[j]]],
                            Sow[{i, j}]
                        ],
                    {"Inside", 2, 1},
                        If[pointInsideQ[polycomps[[j]], pts[[i]]],
                            Sow[{j, i}]
                        ],
                    "Disjoint",
                        Null,
                    _,
                        Throw[0]
                ],
                {i, 1, np - 1},
                {j, i + 1, np}
            ]
        ][[-1]];

        Sort[Flatten[nesting, 1]]
]


nestingLevels[np_Integer, adj_] :=
    Block[{depths, roots, depthlist},

        (* The number of times a component appears in the second location is its depth in the inclusion tree. *)
        depths = SplitBy[SortBy[Tally[adj[[All, 2]]], Last], Last];

        roots = Complement[Range[np], adj[[All, 2]]];

        (* creates {{i01, i02, i03, ...}, {i11, i12, i13, ...}, {i21, i22, i23, ...}, ...}, where idj means the index has depth d. *)
        Prepend[depths[[All, All, 1]], roots]
    ]


nestingDepths[np_Integer, adj_] :=
    Block[{depthlist, df},
        depthlist = nestingLevels[np, adj];

        df[_] = 0;
        Do[Scan[(df[#] = i)&, depthlist[[i+1]]], {i, 0, Length[depthlist]-1}];

        df /@ Range[np]
    ]


(* ::Subsection::Closed:: *)
(*validToLevelSetQ*)


validToLevelSetQ[reg_?RegionQ] := ConstantRegionQ[reg] && RegionEmbeddingDimension[reg] === 3


validToLevelSetQ[{coords_?coordinatesQ, ocells:_List|_Polygon}] :=
    With[{cells = stripPolygonCells[ocells]},
        validTriangleCellsQ[cells, Length[coords]]
    ]


validToLevelSetQ[{coords_?coordinatesQ, ocells:_List|_Polygon|_Line, r_?Positive}] :=
    With[{cells = stripLineCells[stripPolygonCells[ocells]]},
        validTriangleCellsQ[cells, Length[coords]] || validLineCellsQ[cells, Length[coords]]
    ]


validToLevelSetQ[___] = False;


(* ::Subsection::Closed:: *)
(*validScalarTypeQ*)


validScalarTypeQ[type_] := KeyExistsQ[$GridClassData[$scalarType], type]


validScalarTypeQ[___] = $Failed;
