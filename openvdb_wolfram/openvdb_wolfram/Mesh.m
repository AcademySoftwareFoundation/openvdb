(* ::Package:: *)

(* ::Title:: *)
(*Mesh*)


(* ::Subtitle:: *)
(*Create a mesh representation of a level set or fog volume.*)


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


OpenVDBMesh[expr_, opts:OptionsPattern[]] := OpenVDBMesh[expr, Automatic, opts]


OpenVDBMesh[vdb_?OpenVDBScalarGridQ, itype_, OptionsPattern[]] := 
	Block[{type, adaptivity, isovalue, quadQ, data, res},
		type = parseLevelSetMeshType[itype];
		(
			{adaptivity, isovalue, quadQ} = OptionValue[{"Adaptivity", "IsoValue", "ReturnQuads"}];
			
			adaptivity = If[realQ[#], #, 0.]&[Clip[adaptivity, {0., 1.}]];
			isovalue = gridIsoValue[isovalue, vdb];
			quadQ = TrueQ[quadQ];
			(
				data = levelSetMeshData[vdb, isovalue, adaptivity, quadQ];
				(
					res = constructLevelSetMesh[data, type];
					
					res /; res =!= $Failed
					
				) /; data =!= $Failed
				
			) /; realQ[isovalue]
			
		) /; StringQ[type]
	]


OpenVDBMesh[vdb_, itype_, bds_List, opts:OptionsPattern[]] /; MatrixQ[bds, realQ] := OpenVDBMesh[vdb, itype, bds -> $worldregime, opts]


OpenVDBMesh[vdb_?OpenVDBScalarGridQ, itype_, bds_List -> regime_?regimeQ, opts:OptionsPattern[]] := 
	Block[{clipopts, clip},
		clipopts = FilterRules[{opts, Options[OpenVDBMesh]}, Options[OpenVDBClip]];
		
		clip = OpenVDBClip[vdb, bds -> regime, Sequence @@ clipopts];
		
		OpenVDBMesh[clip, itype, opts] /; OpenVDBScalarGridQ[clip]
	]


OpenVDBMesh[___] = $Failed;


(* ::Subsection::Closed:: *)
(*Argument conform & completion*)


registerForLevelSet[OpenVDBMesh, 1];


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
