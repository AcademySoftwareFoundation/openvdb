BeginTestSection["Mesh Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`$OpenVDBHalfWidth = 3.; 
  vdb = OpenVDBLink`OpenVDBLevelSet[ExampleData[{"Geometry3D",  "Triceratops"},  
     "BoundaryMeshRegion"]]; OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBMesh"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBMesh]
	,
	"World"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBMesh]
	,
	{Protected,  ReadProtected}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBMesh]
	,
	{"Adaptivity" -> 0.,  "CloseBoundary" -> True, "IsoValue" -> Automatic, 
  "ReturnQuads" -> False}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBMesh]
	,
	{"ArgumentsPattern" -> {_,  _., _., OptionsPattern[]}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBMesh[],  OpenVDBLink`OpenVDBMesh["error"], 
  OpenVDBLink`OpenVDBMesh[vdb,  "error"], OpenVDBLink`OpenVDBMesh[vdb,  "ComplexData", 
   "error"], OpenVDBLink`OpenVDBMesh[vdb,  "ComplexData", {{0,  1},  {0,  1}, {0,  1}}, 
   "error"]}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBMesh::argb, OpenVDBMesh::scalargrid2, OpenVDBMesh::ret, OpenVDBMesh::bbox3d, OpenVDBMesh::nonopt}
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBMesh[OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ 
  {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", 
   "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, 
  $Failed, $Failed}
	,
	{OpenVDBMesh::scalargrid2, OpenVDBMesh::scalargrid2, OpenVDBMesh::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 8 *)
	OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`$OpenVDBHalfWidth = 3.; 
  bmr = ExampleData[{"Geometry3D",  "Triceratops"},  "BoundaryMeshRegion"]; 
  vdbempty = OpenVDBLink`OpenVDBCreateGrid[1.,  "Scalar"]; 
  vdb = OpenVDBLink`OpenVDBLevelSet[bmr]; fog = OpenVDBLink`OpenVDBFogVolume[vdb]; 
  {BoundaryMeshRegionQ[bmr],  OpenVDBLink`OpenVDBScalarGridQ[vdbempty], 
   OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True,  True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBMesh"]

VerificationTest[(* 9 *)
	OpenVDBLink`OpenVDBMesh[vdbempty]
	,
	EmptyRegion[3]	
]

VerificationTest[(* 10 *)
	(MeshCellCount[OpenVDBLink`OpenVDBMesh[#1],  2] & ) /@ {bmr,  vdb, fog}
	,
	{12244,  12244, 7620}	
]

VerificationTest[(* 11 *)
	(MeshCellCount[OpenVDBLink`OpenVDBMesh[#1,  "ReturnQuads" -> True, "IsoValue" -> 0.1, 
     "Adaptivity" -> 0.6],  2] & ) /@ {bmr,  vdb, fog}
	,
	{1539,  1539, 2246}	
]

VerificationTest[(* 12 *)
	MeshRegionQ[OpenVDBLink`OpenVDBMesh[vdb,  "MeshRegion", "ReturnQuads" -> True, 
   "Adaptivity" -> 1]]
	,
	True	
]

VerificationTest[(* 13 *)
	BoundaryMeshRegionQ[OpenVDBLink`OpenVDBMesh[vdb,  "BoundaryMeshRegion", 
   "ReturnQuads" -> True, "Adaptivity" -> 1]]
	,
	True	
]

VerificationTest[(* 14 *)
	MatchQ[OpenVDBLink`OpenVDBMesh[vdb,  "ComplexData", "ReturnQuads" -> True, 
   "Adaptivity" -> 1],  {(coords_)?MatrixQ,  {(tris_)?MatrixQ,  (quads_)?MatrixQ}} /; 
   Dimensions[coords][[2]] == Dimensions[tris][[2]] == 3 && 
    Dimensions[quads][[2]] == 4 && Developer`PackedArrayQ[coords,  Real] && 
    Developer`PackedArrayQ[tris,  Integer] && Developer`PackedArrayQ[quads,  
     Integer] && (Min[{tris,  quads}] > 0 && Max[{tris,  quads}] <= Length[coords])]
	,
	True	
]

VerificationTest[(* 15 *)
	MatchQ[OpenVDBLink`OpenVDBMesh[vdb,  "FaceData", "ReturnQuads" -> True, 
   "Adaptivity" -> 1],  {tris_List,  quads_List} /; 
   ArrayQ[tris,  3] && ArrayQ[quads,  3] && Dimensions[tris][[2 ;; -1]] == {3,  3} && 
    Dimensions[quads][[2 ;; -1]] == {4,  3} && Developer`PackedArrayQ[tris,  Real] && 
    Developer`PackedArrayQ[quads,  Real]]
	,
	True	
]

VerificationTest[(* 16 *)
	MeshCellCount[OpenVDBLink`OpenVDBMesh[vdb,  Automatic, {{-2,  2},  {-2,  2}, {-2,  2}} -> 
    "World"]]
	,
	{5124,  15366, 10244}	
]

VerificationTest[(* 17 *)
	MeshCellCount[OpenVDBLink`OpenVDBMesh[vdb,  Automatic, 
   {{0,  20},  {-10,  10}, {-20,  20}} -> "Index", "CloseBoundary" -> False]]
	,
	{2767,  8231, 5464}	
]

VerificationTest[(* 18 *)
	MeshCellCount[OpenVDBLink`OpenVDBMesh[fog,  Automatic, {{-2,  2},  {-2,  2}, {-2,  2}}]]
	,
	{3478,  10404, 6936}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 19 *)
	OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`$OpenVDBHalfWidth = 3.; 
  bmr = ExampleData[{"Geometry3D",  "Triceratops"},  "BoundaryMeshRegion"]; 
  vdbempty = OpenVDBLink`OpenVDBCreateGrid[1.,  "Double"]; 
  vdb = OpenVDBLink`OpenVDBLevelSet[bmr,  "ScalarType" -> "Double"]; 
  fog = OpenVDBLink`OpenVDBFogVolume[vdb]; {BoundaryMeshRegionQ[bmr],  
   OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb], 
   OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True,  True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBMesh"]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBMesh[vdbempty]
	,
	EmptyRegion[3]	
]

VerificationTest[(* 21 *)
	(MeshCellCount[OpenVDBLink`OpenVDBMesh[#1],  2] & ) /@ {bmr,  vdb, fog}
	,
	{12244,  12244, 7620}	
]

VerificationTest[(* 22 *)
	(MeshCellCount[OpenVDBLink`OpenVDBMesh[#1,  "ReturnQuads" -> True, "IsoValue" -> 0.1, 
     "Adaptivity" -> 0.6],  2] & ) /@ {bmr,  vdb, fog}
	,
	{1539,  1539, 2246}	
]

VerificationTest[(* 23 *)
	MeshRegionQ[OpenVDBLink`OpenVDBMesh[vdb,  "MeshRegion", "ReturnQuads" -> True, 
   "Adaptivity" -> 1]]
	,
	True	
]

VerificationTest[(* 24 *)
	BoundaryMeshRegionQ[OpenVDBLink`OpenVDBMesh[vdb,  "BoundaryMeshRegion", 
   "ReturnQuads" -> True, "Adaptivity" -> 1]]
	,
	True	
]

VerificationTest[(* 25 *)
	MatchQ[OpenVDBLink`OpenVDBMesh[vdb,  "ComplexData", "ReturnQuads" -> True, 
   "Adaptivity" -> 1],  {(coords_)?MatrixQ,  {(tris_)?MatrixQ,  (quads_)?MatrixQ}} /; 
   Dimensions[coords][[2]] == Dimensions[tris][[2]] == 3 && 
    Dimensions[quads][[2]] == 4 && Developer`PackedArrayQ[coords,  Real] && 
    Developer`PackedArrayQ[tris,  Integer] && Developer`PackedArrayQ[quads,  
     Integer] && (Min[{tris,  quads}] > 0 && Max[{tris,  quads}] <= Length[coords])]
	,
	True	
]

VerificationTest[(* 26 *)
	MatchQ[OpenVDBLink`OpenVDBMesh[vdb,  "FaceData", "ReturnQuads" -> True, 
   "Adaptivity" -> 1],  {tris_List,  quads_List} /; 
   ArrayQ[tris,  3] && ArrayQ[quads,  3] && Dimensions[tris][[2 ;; -1]] == {3,  3} && 
    Dimensions[quads][[2 ;; -1]] == {4,  3} && Developer`PackedArrayQ[tris,  Real] && 
    Developer`PackedArrayQ[quads,  Real]]
	,
	True	
]

VerificationTest[(* 27 *)
	MeshCellCount[OpenVDBLink`OpenVDBMesh[vdb,  Automatic, {{-2,  2},  {-2,  2}, {-2,  2}} -> 
    "World"]]
	,
	{5124,  15366, 10244}	
]

VerificationTest[(* 28 *)
	MeshCellCount[OpenVDBLink`OpenVDBMesh[vdb,  Automatic, 
   {{0,  20},  {-10,  10}, {-20,  20}} -> "Index", "CloseBoundary" -> False]]
	,
	{2767,  8231, 5464}	
]

VerificationTest[(* 29 *)
	MeshCellCount[OpenVDBLink`OpenVDBMesh[fog,  Automatic, {{-2,  2},  {-2,  2}, {-2,  2}}]]
	,
	{3478,  10404, 6936}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
