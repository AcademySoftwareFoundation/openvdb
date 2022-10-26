(* ::Package:: *)

(* ::Title:: *)
(*Mesh*)


(* ::Subtitle:: *)
(*Create a mesh representation of a level set or fog volume.*)


(* ::Text:: *)
(*Copyright Contributors to the OpenVDB Project*)
(*SPDX-License-Identifier: MPL-2.0*)


(* ::Section:: *)
(*Initialization & Usage*)


Package["OpenVDBLink`"]


PackageExport["OpenVDBMesh"]


OpenVDBMesh::usage = "OpenVDBMesh[expr] creates a mesh representation of an OpenVDB scalar grid.";


(* ::Section:: *)
(*OpenVDBMesh*)


(* ::Subsection::Closed:: *)
(*Main*)


Options[OpenVDBMesh] = {"Adaptivity" -> 0., "CloseBoundary" -> True, "IsoValue" -> Automatic, "ReturnQuads" -> False};


OpenVDBMesh[args___] /; !CheckArgs[OpenVDBMesh[args], {1, 3}] = $Failed;


OpenVDBMesh[args___] :=
    With[{res = iOpenVDBMesh[args]},
        res /; res =!= $Failed
    ]


OpenVDBMesh[args___] := mOpenVDBMesh[args]


(* ::Subsection::Closed:: *)
(*iOpenVDBMesh*)


Options[iOpenVDBMesh] = Options[OpenVDBMesh];


iOpenVDBMesh[expr_, opts:OptionsPattern[]] := iOpenVDBMesh[expr, Automatic, opts]


iOpenVDBMesh[vdb_?OpenVDBScalarGridQ, itype_, OptionsPattern[]] :=
    Block[{type, adaptivity, isovalue, quadQ, data, res},
        type = parseLevelSetMeshType[itype];
        (
            {adaptivity, isovalue, quadQ} = OptionValue[{"Adaptivity", "IsoValue", "ReturnQuads"}];

            adaptivity = Clip[adaptivity, {0., 1.}];
            isovalue = gridIsoValue[isovalue, vdb];
            quadQ = TrueQ[quadQ];
            (
                data = levelSetMeshData[vdb, isovalue, adaptivity, quadQ];
                (
                    res = constructLevelSetMesh[data, type];

                    res /; res =!= $Failed

                ) /; data =!= $Failed

            ) /; realQ[isovalue] && 0 <= adaptivity <= 1

        ) /; StringQ[type]
    ]


iOpenVDBMesh[vdb_, itype_, bds_List, opts:OptionsPattern[]] /; MatrixQ[bds, realQ] := iOpenVDBMesh[vdb, itype, bds -> $worldregime, opts]


iOpenVDBMesh[vdb_?OpenVDBScalarGridQ, itype_, bds_List -> regime_?regimeQ, opts:OptionsPattern[]] :=
    Block[{clipopts, clip},
        clipopts = FilterRules[{opts, Options[OpenVDBMesh]}, Options[OpenVDBClip]];

        clip = OpenVDBClip[vdb, bds -> regime, Sequence @@ clipopts];

        iOpenVDBMesh[clip, itype, opts] /; OpenVDBScalarGridQ[clip]
    ]


iOpenVDBMesh[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[iOpenVDBMesh, 1];


SyntaxInformation[OpenVDBMesh] = {"ArgumentsPattern" -> {_, _., _., OptionsPattern[]}};


addCodeCompletion[OpenVDBMesh][None, {"MeshRegion", "BoundaryMeshRegion", "ComplexData", "FaceData"}, None];


OpenVDBDefaultSpace[OpenVDBMesh] = $worldregime;


(* ::Subsection::Closed:: *)
(*Utilities*)


(* ::Subsubsection::Closed:: *)
(*parseLevelSetMeshType*)


parseLevelSetMeshType[Automatic] = "MeshRegion";
parseLevelSetMeshType["MeshRegion"] = "MeshRegion";
parseLevelSetMeshType[MeshRegion] = "MeshRegion";
parseLevelSetMeshType["BoundaryMeshRegion"] = "BoundaryMeshRegion";
parseLevelSetMeshType[BoundaryMeshRegion] = "BoundaryMeshRegion";
parseLevelSetMeshType["ComplexData"] = "ComplexData";
parseLevelSetMeshType["FaceData"] = "FaceData";
parseLevelSetMeshType[_] = $Failed;


(* ::Subsubsection::Closed:: *)
(*levelSetMeshData*)


levelSetMeshData[vdb_?emptyVDBQ, isovalue_, __] := makeFacelessRegion[vdb, isovalue]


levelSetMeshData[vdb_, isovalue_, adaptivity_, quadQ_] :=
    Block[{rawdata, doffset, coordlen, trilen, quadlen, coords, cells},
        rawdata = vdb["meshData"[isovalue, adaptivity, !quadQ]];
        (
            doffset = 3;
            {coordlen, trilen, quadlen} = {3, 3, 4} * rawdata[[1 ;; 3]];

            coords = Partition[rawdata[[doffset+1 ;; doffset+coordlen]], 3];

            cells = Reverse[Round[#]+1, {2}]& /@ {
                If[trilen > 0, Partition[rawdata[[doffset+coordlen+1 ;; doffset+coordlen+trilen]], 3], Nothing],
                If[quadlen > 0, Partition[rawdata[[doffset+coordlen+trilen+1 ;; -1]], 4], Nothing]
            };

            {coords, cells}

        ) /; ListQ[rawdata] && Length[rawdata] > 3
    ]


levelSetMeshData[vdb_, isovalue_, __] := makeFacelessRegion[vdb, isovalue]


makeFacelessRegion[vdb_, isovalue_] :=
    If[TrueQ[vdb["getBackgroundValue"[]] < isovalue],
        FullRegion[3],
        EmptyRegion[3]
    ]


(* ::Subsubsection::Closed:: *)
(*constructLevelSetMesh*)


constructLevelSetMesh[{coords_, cells:{__}}, type:("MeshRegion"|"BoundaryMeshRegion")] :=
    Block[{hasQuads, head, mr},
        hasQuads = Max[Length[#[[1]]]& /@ cells] == 4;
        head = If[type === "BoundaryMeshRegion", BoundaryMeshRegion, MeshRegion];

        mr = Quiet @ head[
            coords,
            Polygon /@ cells,
            Method -> {
                If[type === "BoundaryMeshRegion", "CheckIntersections" -> False, Nothing],
                If[hasQuads, "CoplanarityTolerance" -> 14, Nothing],
                "DeleteDuplicateCells" -> False,
                "DeleteDuplicateCoordinates" -> False,
                "EliminateUnusedCoordinates" -> False,
                "TJunction" -> False
            }
        ];

        mr /; RegionQ[mr]
    ]


constructLevelSetMesh[{coords_, cells_}, "ComplexData"] := {coords, cells}


constructLevelSetMesh[{coords_, cells_}, "FaceData"] := Partition[coords[[Flatten[#]]], Length[#[[1]]]]& /@ cells


constructLevelSetMesh[EmptyRegion[3], "MeshRegion"] = EmptyRegion[3];
constructLevelSetMesh[EmptyRegion[3], "BoundaryMeshRegion"] = EmptyRegion[3];
constructLevelSetMesh[EmptyRegion[3], "ComplexData"] = {{}, {}};
constructLevelSetMesh[EmptyRegion[3], "FaceData"] = {};


constructLevelSetMesh[FullRegion[3], "MeshRegion"] = EmptyRegion[3];
constructLevelSetMesh[FullRegion[3], "BoundaryMeshRegion"] = FullRegion[3];
constructLevelSetMesh[FullRegion[3], "ComplexData"] = {{}, {}};
constructLevelSetMesh[FullRegion[3], "FaceData"] = {};


constructLevelSetMesh[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Messages*)


Options[mOpenVDBMesh] = Options[OpenVDBMesh];


mOpenVDBMesh[expr_, ___] /; messageScalarGridQ[expr, OpenVDBMesh] = $Failed;


mOpenVDBMesh[_, type_, ___] /; parseLevelSetMeshType[type] === $Failed :=
    (
        Message[OpenVDBMesh::ret, type, 2];
        $Failed
    )


mOpenVDBMesh[_, _, bbox:Except[_?OptionQ], ___] /; message3DBBoxQ[bbox, OpenVDBMesh] = $Failed;


mOpenVDBMesh[__, OptionsPattern[]] :=
    (
        If[messageIsoValueQ[OptionValue["IsoValue"], OpenVDBMesh],
            Return[$Failed]
        ];

        If[!TrueQ[0 <= OptionValue["Adaptivity"] <= 1],
            Message[OpenVDBMesh::adapt];
            Return[$Failed]
        ];

        $Failed
    )


mOpenVDBMesh[___] = $Failed;


OpenVDBMesh::ret = "`1` at position `2` is not one of \"MeshRegion\", \"BoundaryMeshRegion\", \"ComplexData\", \"FaceData\", or Automatic.";


OpenVDBMesh::adapt = "The setting for \"Adaptivity\" must be a number between 0 and 1, inclusively.";
