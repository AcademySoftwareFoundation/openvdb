BeginTestSection["CSG Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`$OpenVDBHalfWidth = 3.; bmr = ExampleData[{"Geometry3D",  "Triceratops"},  "BoundaryMeshRegion"]; bmr2 = ExampleData[{"Geometry3D",  "Triceratops"},  "BoundaryMeshRegion"]; {BoundaryMeshRegionQ[bmr],  BoundaryMeshRegionQ[bmr2]}
	,
	{True,  True}	
]

EndTestSection[]

BeginTestSection["OpenVDBUnion"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBUnion]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBUnion]
	,
	{Protected,  ReadProtected}	
	,
	{}
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBUnion]
	,
	{"Creator" -> Inherited,  "Name" -> Inherited}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBUnion]
	,
	{"ArgumentsPattern" -> {___,  OptionsPattern[]}}	
]

VerificationTest[(* 6 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBUnion[]]
	,
	True	
]

VerificationTest[(* 7 *)
	{OpenVDBLink`OpenVDBUnion["error"],  OpenVDBLink`OpenVDBUnion[bmr,  "error"], OpenVDBLink`OpenVDBUnion[bmr,  bmr2, "error"]}
	,
	{$Failed,  $Failed, $Failed}
	,
	{OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, General::stop}
]

VerificationTest[(* 8 *)
	(OpenVDBLink`OpenVDBUnion[OpenVDBLink`OpenVDBCreateGrid[1.,  #1],  OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBIntersection"]

VerificationTest[(* 9 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBIntersection]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 10 *)
	Attributes[OpenVDBLink`OpenVDBIntersection]
	,
	{Protected,  ReadProtected}	
	,
	{}
]

VerificationTest[(* 11 *)
	Options[OpenVDBLink`OpenVDBIntersection]
	,
	{"Creator" -> Inherited,  "Name" -> Inherited}	
]

VerificationTest[(* 12 *)
	SyntaxInformation[OpenVDBLink`OpenVDBIntersection]
	,
	{"ArgumentsPattern" -> {___,  OptionsPattern[]}}	
]

VerificationTest[(* 13 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBIntersection[]]
	,
	True	
]

VerificationTest[(* 14 *)
	{OpenVDBLink`OpenVDBIntersection["error"],  OpenVDBLink`OpenVDBIntersection[bmr,  "error"], OpenVDBLink`OpenVDBIntersection[bmr,  bmr2, "error"]}
	,
	{$Failed,  $Failed, $Failed}
	,
	{OpenVDBIntersection::scalargrid2, OpenVDBIntersection::scalargrid2, OpenVDBIntersection::scalargrid2, General::stop}
]

VerificationTest[(* 15 *)
	(OpenVDBLink`OpenVDBIntersection[OpenVDBLink`OpenVDBCreateGrid[1.,  #1],  OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBIntersection::scalargrid2, OpenVDBIntersection::scalargrid2, OpenVDBIntersection::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBDifference"]

VerificationTest[(* 16 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBDifference]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 17 *)
	Attributes[OpenVDBLink`OpenVDBDifference]
	,
	{Protected,  ReadProtected}	
	,
	{}
]

VerificationTest[(* 18 *)
	Options[OpenVDBLink`OpenVDBDifference]
	,
	{"Creator" -> Inherited,  "Name" -> Inherited}	
]

VerificationTest[(* 19 *)
	SyntaxInformation[OpenVDBLink`OpenVDBDifference]
	,
	{"ArgumentsPattern" -> {___,  OptionsPattern[]}}	
]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBDifference[]]
	,
	True	
]

VerificationTest[(* 21 *)
	{OpenVDBLink`OpenVDBDifference["error"],  OpenVDBLink`OpenVDBDifference[bmr,  "error"], OpenVDBLink`OpenVDBDifference[bmr,  bmr2, "error"]}
	,
	{$Failed,  $Failed, $Failed}
	,
	{OpenVDBDifference::scalargrid2, OpenVDBDifference::scalargrid2, OpenVDBDifference::scalargrid2, General::stop}
]

VerificationTest[(* 22 *)
	(OpenVDBLink`OpenVDBDifference[OpenVDBLink`OpenVDBCreateGrid[1.,  #1],  OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDifference::scalargrid2, OpenVDBDifference::scalargrid2, OpenVDBDifference::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBUnionTo"]

VerificationTest[(* 23 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBUnionTo]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 24 *)
	Attributes[OpenVDBLink`OpenVDBUnionTo]
	,
	{Protected,  ReadProtected}	
	,
	{}
]

VerificationTest[(* 25 *)
	Options[OpenVDBLink`OpenVDBUnionTo]
	,
	{"Creator" -> Inherited,  "Name" -> Inherited}	
]

VerificationTest[(* 26 *)
	SyntaxInformation[OpenVDBLink`OpenVDBUnionTo]
	,
	{"ArgumentsPattern" -> {_,  ___, OptionsPattern[]}}	
]

VerificationTest[(* 27 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBUnionTo[bmr]]
	,
	True	
]

VerificationTest[(* 28 *)
	{OpenVDBLink`OpenVDBUnionTo[],  OpenVDBLink`OpenVDBUnionTo["error"], OpenVDBLink`OpenVDBUnionTo[bmr,  "error"], OpenVDBLink`OpenVDBUnionTo[bmr,  bmr2, "error"]}
	,
	{$Failed,  $Failed, $Failed, $Failed}
	,
	{OpenVDBUnionTo::argm, OpenVDBUnionTo::scalargrid2, OpenVDBUnionTo::scalargrid2, OpenVDBUnionTo::scalargrid2, General::stop}
]

VerificationTest[(* 29 *)
	(OpenVDBLink`OpenVDBUnionTo[OpenVDBLink`OpenVDBCreateGrid[1.,  #1],  OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBUnionTo::scalargrid2, OpenVDBUnionTo::scalargrid2, OpenVDBUnionTo::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBIntersectWith"]

VerificationTest[(* 30 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBIntersectWith]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 31 *)
	Attributes[OpenVDBLink`OpenVDBIntersectWith]
	,
	{Protected,  ReadProtected}	
	,
	{}
]

VerificationTest[(* 32 *)
	Options[OpenVDBLink`OpenVDBIntersectWith]
	,
	{"Creator" -> Inherited,  "Name" -> Inherited}	
]

VerificationTest[(* 33 *)
	SyntaxInformation[OpenVDBLink`OpenVDBIntersectWith]
	,
	{"ArgumentsPattern" -> {_,  ___, OptionsPattern[]}}	
]

VerificationTest[(* 34 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBIntersectWith[bmr]]
	,
	True	
]

VerificationTest[(* 35 *)
	{OpenVDBLink`OpenVDBIntersectWith[],  OpenVDBLink`OpenVDBIntersectWith["error"], OpenVDBLink`OpenVDBIntersectWith[bmr,  "error"], OpenVDBLink`OpenVDBIntersectWith[bmr,  bmr2, "error"]}
	,
	{$Failed,  $Failed, $Failed, $Failed}
	,
	{OpenVDBIntersectWith::argm, OpenVDBIntersectWith::scalargrid2, OpenVDBIntersectWith::scalargrid2, OpenVDBIntersectWith::scalargrid2, General::stop}
]

VerificationTest[(* 36 *)
	(OpenVDBLink`OpenVDBIntersectWith[OpenVDBLink`OpenVDBCreateGrid[1.,  #1],  OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBIntersectWith::scalargrid2, OpenVDBIntersectWith::scalargrid2, OpenVDBIntersectWith::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBDifferenceFrom"]

VerificationTest[(* 37 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBDifferenceFrom]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 38 *)
	Attributes[OpenVDBLink`OpenVDBDifferenceFrom]
	,
	{Protected,  ReadProtected}	
	,
	{}
]

VerificationTest[(* 39 *)
	Options[OpenVDBLink`OpenVDBDifferenceFrom]
	,
	{"Creator" -> Inherited,  "Name" -> Inherited}	
]

VerificationTest[(* 40 *)
	SyntaxInformation[OpenVDBLink`OpenVDBDifferenceFrom]
	,
	{"ArgumentsPattern" -> {_,  ___, OptionsPattern[]}}	
]

VerificationTest[(* 41 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBDifferenceFrom[bmr]]
	,
	True	
]

VerificationTest[(* 42 *)
	{OpenVDBLink`OpenVDBDifferenceFrom[],  OpenVDBLink`OpenVDBDifferenceFrom["error"], OpenVDBLink`OpenVDBDifferenceFrom[bmr,  "error"], OpenVDBLink`OpenVDBDifferenceFrom[bmr,  bmr2, "error"]}
	,
	{$Failed,  $Failed, $Failed, $Failed}
	,
	{OpenVDBDifferenceFrom::argm, OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, General::stop}
]

VerificationTest[(* 43 *)
	(OpenVDBLink`OpenVDBDifferenceFrom[OpenVDBLink`OpenVDBCreateGrid[1.,  #1],  OpenVDBLink`OpenVDBCreateGrid[1.,  #1]] & ) /@ {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, OpenVDBUnion::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBClip"]

VerificationTest[(* 44 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBClip]
	,
	"World"	
]

VerificationTest[(* 45 *)
	Attributes[OpenVDBLink`OpenVDBClip]
	,
	{Protected,  ReadProtected}	
	,
	{}
]

VerificationTest[(* 46 *)
	Options[OpenVDBLink`OpenVDBClip]
	,
	{"CloseBoundary" -> True,  "Creator" -> Inherited, "Name" -> Inherited}	
]

VerificationTest[(* 47 *)
	SyntaxInformation[OpenVDBLink`OpenVDBClip]
	,
	{"ArgumentsPattern" -> {_,  _, OptionsPattern[]}}	
]

VerificationTest[(* 48 *)
	{OpenVDBLink`OpenVDBClip[],  OpenVDBLink`OpenVDBClip["error"], OpenVDBLink`OpenVDBClip[bmr,  "error"], OpenVDBLink`OpenVDBClip[bmr,  {{0,  1},  {0,  1}, {0,  1}} -> "error"], OpenVDBLink`OpenVDBClip[bmr,  {{0,  1},  {0,  1}, {0,  1}}, "error"]}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBClip::argrx, OpenVDBClip::argr, OpenVDBClip::bbox3d, OpenVDBClip::gridspace, OpenVDBClip::nonopt}
]

VerificationTest[(* 49 *)
	(OpenVDBLink`OpenVDBClip[OpenVDBLink`OpenVDBCreateGrid[1.,  #1],  {{0,  1},  {0,  1}, {0,  1}}] & ) /@ {"Int32",  "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed,  $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBClip::scalargrid2, OpenVDBClip::scalargrid2, OpenVDBClip::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 50 *)
	$propertyList = {"ActiveLeafVoxelCount",  "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "MinMaxValues", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}; OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`$OpenVDBHalfWidth = 3.; 
  bmr = ExampleData[{"Geometry3D",  "Triceratops"},  "MeshRegion"]; bmr2 = TransformedRegion[bmr,  TranslationTransform[{1,  1, 1}]]; bmr3 = TransformedRegion[bmr,  TranslationTransform[{2,  -1, 2}]]; vdb = OpenVDBLink`OpenVDBLevelSet[bmr,  "Creator" -> "initC", "Name" -> "initN"]; fog = OpenVDBLink`OpenVDBFogVolume[vdb]; 
  {MeshRegionQ[bmr],  MeshRegionQ[bmr2], MeshRegionQ[bmr3], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True,  True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBUnion"]

VerificationTest[(* 51 *)
	OpenVDBLink`OpenVDBUnion[OpenVDBLink`OpenVDBLevelSet[bmr,  0.2]][$propertyList]
	,
	{6987,  0, 6987, 0.6000000238418579, 20520, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-25,  19},  {-9,  9}, {-11,  12}}, {45,  19, 24}, {-0.5996429324150085,  0.5996281504631042}, True, 0.2, {{-5.,  3.8000000000000003},  {-1.8,  1.8}, {-2.2,  2.4000000000000004}}, {9.,  3.8000000000000003, 4.800000000000001}}	
]

VerificationTest[(* 52 *)
	OpenVDBLink`OpenVDBUnion[bmr,  bmr2][$propertyList]
	,
	{44609,  0, 44609, 0.30000001192092896, 201348, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  45},  {-16,  25}, {-19,  31}}, {94,  42, 51}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  4.5},  {-1.6,  2.5}, {-1.9000000000000001,  3.1}}, {9.4,  4.2, 5.1000000000000005}}	
]

VerificationTest[(* 53 *)
	OpenVDBLink`OpenVDBUnion[bmr,  bmr2, bmr3][$propertyList]
	,
	{67281,  0, 67281, 0.30000001192092896, 329888, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.2999959886074066,  0.29995036125183105}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}}	
]

VerificationTest[(* 54 *)
	OpenVDBLink`OpenVDBUnion[vdb,  bmr2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 55 *)
	OpenVDBLink`OpenVDBUnion[vdb,  bmr2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

EndTestSection[]

BeginTestSection["OpenVDBIntersection"]

VerificationTest[(* 56 *)
	OpenVDBLink`OpenVDBIntersection[OpenVDBLink`OpenVDBLevelSet[bmr,  0.2]][$propertyList]
	,
	{6987,  0, 6987, 0.6000000238418579, 20520, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-25,  19},  {-9,  9}, {-11,  12}}, {45,  19, 24}, {-0.5996429324150085,  0.5996281504631042}, True, 0.2, {{-5.,  3.8000000000000003},  {-1.8,  1.8}, {-2.2,  2.4000000000000004}}, {9.,  3.8000000000000003, 4.800000000000001}}	
]

VerificationTest[(* 57 *)
	OpenVDBLink`OpenVDBIntersection[bmr,  bmr2][$propertyList]
	,
	{7537,  0, 7537, 0.30000001192092896, 30240, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-20,  35},  {-5,  12}, {-8,  21}}, {56,  18, 30}, {-0.29793617129325867,  0.299932062625885}, True, 0.1, {{-2.,  3.5},  {-0.5,  1.2000000000000002}, {-0.8,  2.1}}, {5.6000000000000005,  1.8, 3.}}	
]

VerificationTest[(* 58 *)
	OpenVDBLink`OpenVDBIntersection[bmr,  bmr2, bmr3][$propertyList]
	,
	{656,  0, 656, 0.30000001192092896, 4256, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-2,  25},  {-2,  5}, {2,  20}}, {28,  8, 19}, {0.06343629211187363,  0.2996846139431}, True, 0.1, {{-0.2,  2.5},  {-0.2,  0.5}, {0.2,  2.}}, {2.8000000000000003,  0.8, 1.9000000000000001}}	
]

VerificationTest[(* 59 *)
	OpenVDBLink`OpenVDBIntersection[vdb,  bmr2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 60 *)
	OpenVDBLink`OpenVDBIntersection[vdb,  bmr2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

EndTestSection[]

BeginTestSection["OpenVDBDifference"]

VerificationTest[(* 61 *)
	OpenVDBLink`OpenVDBDifference[OpenVDBLink`OpenVDBLevelSet[bmr,  0.2]][$propertyList]
	,
	{6987,  0, 6987, 0.6000000238418579, 20520, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-25,  19},  {-9,  9}, {-11,  12}}, {45,  19, 24}, {-0.5996429324150085,  0.5996281504631042}, True, 0.2, {{-5.,  3.8000000000000003},  {-1.8,  1.8}, {-2.2,  2.4000000000000004}}, {9.,  3.8000000000000003, 4.800000000000001}}	
]

VerificationTest[(* 62 *)
	OpenVDBLink`OpenVDBDifference[bmr,  bmr2][$propertyList]
	,
	{26774,  0, 26774, 0.30000001192092896, 110208, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 63 *)
	OpenVDBLink`OpenVDBDifference[bmr,  bmr2, bmr3][$propertyList]
	,
	{27392,  0, 27392, 0.30000001192092896, 110208, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 64 *)
	OpenVDBLink`OpenVDBDifference[vdb,  bmr2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 65 *)
	OpenVDBLink`OpenVDBDifference[vdb,  bmr2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

EndTestSection[]

BeginTestSection["OpenVDBUnionTo"]

VerificationTest[(* 66 *)
	OpenVDBLink`OpenVDBUnionTo[OpenVDBLink`OpenVDBLevelSet[bmr,  0.2]][$propertyList]
	,
	{6987,  0, 6987, 0.6000000238418579, 20520, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-25,  19},  {-9,  9}, {-11,  12}}, {45,  19, 24}, {-0.5996429324150085,  0.5996281504631042}, True, 0.2, {{-5.,  3.8000000000000003},  {-1.8,  1.8}, {-2.2,  2.4000000000000004}}, {9.,  3.8000000000000003, 4.800000000000001}}	
]

VerificationTest[(* 67 *)
	OpenVDBLink`OpenVDBUnionTo[bmr,  bmr2][$propertyList]
	,
	{44609,  0, 44609, 0.30000001192092896, 201348, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  45},  {-16,  25}, {-19,  31}}, {94,  42, 51}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  4.5},  {-1.6,  2.5}, {-1.9000000000000001,  3.1}}, {9.4,  4.2, 5.1000000000000005}}	
]

VerificationTest[(* 68 *)
	OpenVDBLink`OpenVDBUnionTo[bmr,  bmr2, bmr3][$propertyList]
	,
	{67281,  0, 67281, 0.30000001192092896, 329888, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.2999959886074066,  0.29995036125183105}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}}	
]

VerificationTest[(* 69 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; OpenVDBLink`OpenVDBUnionTo[vdb2,  bmr2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 70 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; OpenVDBLink`OpenVDBUnionTo[vdb2,  bmr2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

VerificationTest[(* 71 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; vdb3 = OpenVDBLink`OpenVDBLevelSet[bmr2]; {OpenVDBLink`OpenVDBUnionTo[vdb2,  vdb3][$propertyList],  vdb3[$propertyList]}
	,
	{{44609,  0, 44609, 0.30000001192092896, 201348, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  45},  {-16,  25}, {-19,  31}}, {94,  42, 51}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  4.5},  {-1.6,  2.5}, {-1.9000000000000001,  3.1}}, {9.4,  4.2, 5.1000000000000005}},  
  {12000,  0, 12000, 0.30000001192092896, 55176, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-36,  39},  {-6,  15}, {-9,  23}}, {76,  22, 33}, {-0.29999592900276184,  0.299932062625885}, True, 0.1, {{-3.6,  3.9000000000000004},  {-0.6000000000000001,  1.5}, {-0.9,  2.3000000000000003}}, {7.6000000000000005,  2.2, 3.3000000000000003}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBIntersectWith"]

VerificationTest[(* 72 *)
	OpenVDBLink`OpenVDBIntersectWith[OpenVDBLink`OpenVDBLevelSet[bmr,  0.2]][$propertyList]
	,
	{6987,  0, 6987, 0.6000000238418579, 20520, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-25,  19},  {-9,  9}, {-11,  12}}, {45,  19, 24}, {-0.5996429324150085,  0.5996281504631042}, True, 0.2, {{-5.,  3.8000000000000003},  {-1.8,  1.8}, {-2.2,  2.4000000000000004}}, {9.,  3.8000000000000003, 4.800000000000001}}	
]

VerificationTest[(* 73 *)
	OpenVDBLink`OpenVDBIntersectWith[bmr,  bmr2][$propertyList]
	,
	{7537,  0, 7537, 0.30000001192092896, 30240, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-20,  35},  {-5,  12}, {-8,  21}}, {56,  18, 30}, {-0.29793617129325867,  0.299932062625885}, True, 0.1, {{-2.,  3.5},  {-0.5,  1.2000000000000002}, {-0.8,  2.1}}, {5.6000000000000005,  1.8, 3.}}	
]

VerificationTest[(* 74 *)
	OpenVDBLink`OpenVDBIntersectWith[bmr,  bmr2, bmr3][$propertyList]
	,
	{656,  0, 656, 0.30000001192092896, 4256, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-2,  25},  {-2,  5}, {2,  20}}, {28,  8, 19}, {0.06343629211187363,  0.2996846139431}, True, 0.1, {{-0.2,  2.5},  {-0.2,  0.5}, {0.2,  2.}}, {2.8000000000000003,  0.8, 1.9000000000000001}}	
]

VerificationTest[(* 75 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; OpenVDBLink`OpenVDBIntersectWith[vdb2,  bmr2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 76 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; OpenVDBLink`OpenVDBIntersectWith[vdb2,  bmr2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

VerificationTest[(* 77 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; vdb3 = OpenVDBLink`OpenVDBLevelSet[bmr2]; {OpenVDBLink`OpenVDBIntersectWith[vdb2,  vdb3][$propertyList],  vdb3[$propertyList]}
	,
	{{7537,  0, 7537, 0.30000001192092896, 30240, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-20,  35},  {-5,  12}, {-8,  21}}, {56,  18, 30}, {-0.29793617129325867,  0.299932062625885}, True, 0.1, {{-2.,  3.5},  {-0.5,  1.2000000000000002}, {-0.8,  2.1}}, {5.6000000000000005,  1.8, 3.}},  
  {12000,  0, 12000, 0.30000001192092896, 55176, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-36,  39},  {-6,  15}, {-9,  23}}, {76,  22, 33}, {-0.29999592900276184,  0.299932062625885}, True, 0.1, {{-3.6,  3.9000000000000004},  {-0.6000000000000001,  1.5}, {-0.9,  2.3000000000000003}}, {7.6000000000000005,  2.2, 3.3000000000000003}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDifferenceFrom"]

VerificationTest[(* 78 *)
	OpenVDBLink`OpenVDBDifferenceFrom[OpenVDBLink`OpenVDBLevelSet[bmr,  0.2]][$propertyList]
	,
	{6987,  0, 6987, 0.6000000238418579, 20520, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-25,  19},  {-9,  9}, {-11,  12}}, {45,  19, 24}, {-0.5996429324150085,  0.5996281504631042}, True, 0.2, {{-5.,  3.8000000000000003},  {-1.8,  1.8}, {-2.2,  2.4000000000000004}}, {9.,  3.8000000000000003, 4.800000000000001}}	
]

VerificationTest[(* 79 *)
	OpenVDBLink`OpenVDBDifferenceFrom[bmr,  bmr2][$propertyList]
	,
	{26774,  0, 26774, 0.30000001192092896, 110208, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 80 *)
	OpenVDBLink`OpenVDBDifferenceFrom[bmr,  bmr2, bmr3][$propertyList]
	,
	{27392,  0, 27392, 0.30000001192092896, 110208, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 81 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; OpenVDBLink`OpenVDBDifferenceFrom[vdb2,  bmr2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 82 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; OpenVDBLink`OpenVDBDifferenceFrom[vdb2,  bmr2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

VerificationTest[(* 83 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb]; vdb3 = OpenVDBLink`OpenVDBLevelSet[bmr2]; {OpenVDBLink`OpenVDBDifferenceFrom[vdb2,  vdb3][$propertyList],  vdb3[$propertyList]}
	,
	{{26774,  0, 26774, 0.30000001192092896, 110208, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}},  
  {26073,  0, 26073, 0.30000001192092896, 110208, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-38,  45},  {-6,  25}, {-9,  31}}, {84,  32, 41}, {-0.29999592900276184,  0.29995036125183105}, True, 0.1, {{-3.8000000000000003,  4.5},  {-0.6000000000000001,  2.5}, {-0.9,  3.1}}, {8.4,  3.2, 4.1000000000000005}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBClip"]

VerificationTest[(* 84 *)
	OpenVDBLink`OpenVDBClip[bmr,  {{-0.5,  1},  {0,  1}, {0,  1}}][$propertyList]
	,
	{3694,  0, 3694, 0.30000001192092896, 4200, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-7,  12},  {-2,  11}, {-2,  12}}, {20,  14, 15}, {-0.2994529902935028,  0.29942092299461365}, True, 0.1, {{-0.7000000000000001,  1.2000000000000002},  {-0.2,  1.1}, {-0.2,  1.2000000000000002}}, {2.,  1.4000000000000001, 1.5}}	
]

VerificationTest[(* 85 *)
	OpenVDBLink`OpenVDBClip[fog,  {{-0.5,  1},  {0,  1}, {0,  1}}][$propertyList]
	,
	{1444,  0, 1444, 0., 1584, False, "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-5,  10},  {0,  8}, {0,  10}}, {16,  9, 11}, {0.00011809170246124268,  1.}, True, 0.1, {{-0.5,  1.},  {0.,  0.8}, {0.,  1.}}, {1.6,  0.9, 1.1}}	
]

VerificationTest[(* 86 *)
	OpenVDBLink`OpenVDBClip[bmr,  {{-0.5,  1},  {0,  1}, {0,  1}} -> "World"][$propertyList]
	,
	{3694,  0, 3694, 0.30000001192092896, 4200, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-7,  12},  {-2,  11}, {-2,  12}}, {20,  14, 15}, {-0.2994529902935028,  0.29942092299461365}, True, 0.1, {{-0.7000000000000001,  1.2000000000000002},  {-0.2,  1.1}, {-0.2,  1.2000000000000002}}, {2.,  1.4000000000000001, 1.5}}	
]

VerificationTest[(* 87 *)
	OpenVDBLink`OpenVDBClip[bmr,  {{-20,  20},  {-30,  20}, {-40,  40}} -> "Index"][$propertyList]
	,
	{21703,  0, 21703, 0.30000001192092896, 59040, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-22,  22},  {-16,  15}, {-19,  21}}, {45,  32, 41}, {-0.2999959886074066,  0.299950510263443}, True, 0.1, {{-2.2,  2.2},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {4.5,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 88 *)
	OpenVDBLink`OpenVDBClip[vdb,  {{-20,  20},  {-30,  20}, {-40,  40}} -> "Index"][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 89 *)
	OpenVDBLink`OpenVDBClip[vdb,  {{-20,  20},  {-30,  20}, {-40,  40}} -> "Index", "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 90 *)
	$propertyList = {"ActiveLeafVoxelCount",  "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "MinMaxValues", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}; OpenVDBLink`$OpenVDBSpacing = 0.1; OpenVDBLink`$OpenVDBHalfWidth = 3.; 
  bmr = ExampleData[{"Geometry3D",  "Triceratops"},  "MeshRegion"]; bmr2 = TransformedRegion[bmr,  TranslationTransform[{1,  1, 1}]]; bmr3 = TransformedRegion[bmr,  TranslationTransform[{2,  -1, 2}]]; vdb1 = OpenVDBLink`OpenVDBLevelSet[bmr,  "Creator" -> "initC", "Name" -> "initN", "ScalarType" -> "Double"]; 
  vdb2 = OpenVDBLink`OpenVDBLevelSet[bmr2,  "ScalarType" -> "Double"]; vdb3 = OpenVDBLink`OpenVDBLevelSet[bmr3,  "ScalarType" -> "Double"]; fog = OpenVDBLink`OpenVDBFogVolume[vdb1]; {OpenVDBLink`OpenVDBScalarGridQ[vdb1],  OpenVDBLink`OpenVDBScalarGridQ[vdb2], OpenVDBLink`OpenVDBScalarGridQ[vdb3], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True,  True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBUnion"]

VerificationTest[(* 91 *)
	OpenVDBLink`OpenVDBUnion[vdb1][$propertyList]
	,
	{26073,  0, 26073, 0.30000000000000004, 110208, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 92 *)
	OpenVDBLink`OpenVDBUnion[vdb1,  vdb2][$propertyList]
	,
	{44609,  0, 44609, 0.30000000000000004, 201348, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  45},  {-16,  25}, {-19,  31}}, {94,  42, 51}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  4.5},  {-1.6,  2.5}, {-1.9000000000000001,  3.1}}, {9.4,  4.2, 5.1000000000000005}}	
]

VerificationTest[(* 93 *)
	OpenVDBLink`OpenVDBUnion[vdb1,  vdb2, vdb3][$propertyList]
	,
	{67281,  0, 67281, 0.30000000000000004, 329888, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.29999599125227844,  0.2999503432672819}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}}	
]

VerificationTest[(* 94 *)
	OpenVDBLink`OpenVDBUnion[vdb1,  vdb2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 95 *)
	OpenVDBLink`OpenVDBUnion[vdb1,  vdb2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

EndTestSection[]

BeginTestSection["OpenVDBIntersection"]

VerificationTest[(* 96 *)
	OpenVDBLink`OpenVDBIntersection[vdb1][$propertyList]
	,
	{26073,  0, 26073, 0.30000000000000004, 110208, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 97 *)
	OpenVDBLink`OpenVDBIntersection[vdb1,  vdb2][$propertyList]
	,
	{7537,  0, 7537, 0.30000000000000004, 30240, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-20,  35},  {-5,  12}, {-8,  21}}, {56,  18, 30}, {-0.2979361297880422,  0.299932051668202}, True, 0.1, {{-2.,  3.5},  {-0.5,  1.2000000000000002}, {-0.8,  2.1}}, {5.6000000000000005,  1.8, 3.}}	
]

VerificationTest[(* 98 *)
	OpenVDBLink`OpenVDBIntersection[vdb1,  vdb2, vdb3][$propertyList]
	,
	{656,  0, 656, 0.30000000000000004, 4256, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-2,  25},  {-2,  5}, {2,  20}}, {28,  8, 19}, {0.06343628790606619,  0.29968459473613956}, True, 0.1, {{-0.2,  2.5},  {-0.2,  0.5}, {0.2,  2.}}, {2.8000000000000003,  0.8, 1.9000000000000001}}	
]

VerificationTest[(* 99 *)
	OpenVDBLink`OpenVDBIntersection[vdb1,  vdb2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 100 *)
	OpenVDBLink`OpenVDBIntersection[vdb1,  vdb2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

EndTestSection[]

BeginTestSection["OpenVDBDifference"]

VerificationTest[(* 101 *)
	OpenVDBLink`OpenVDBDifference[vdb1][$propertyList]
	,
	{26073,  0, 26073, 0.30000000000000004, 110208, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 102 *)
	OpenVDBLink`OpenVDBDifference[vdb1,  vdb2][$propertyList]
	,
	{26774,  0, 26774, 0.30000000000000004, 110208, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 103 *)
	OpenVDBLink`OpenVDBDifference[vdb1,  vdb2, vdb3][$propertyList]
	,
	{27392,  0, 27392, 0.30000000000000004, 110208, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 104 *)
	OpenVDBLink`OpenVDBDifference[vdb1,  vdb2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 105 *)
	OpenVDBLink`OpenVDBDifference[vdb1,  vdb2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

EndTestSection[]

BeginTestSection["OpenVDBUnionTo"]

VerificationTest[(* 106 *)
	OpenVDBLink`OpenVDBUnionTo[vdb1][$propertyList]
	,
	{26073,  0, 26073, 0.30000000000000004, 110208, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  35},  {-16,  15}, {-19,  21}}, {84,  32, 41}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  3.5},  {-1.6,  1.5}, {-1.9000000000000001,  2.1}}, {8.4,  3.2, 4.1000000000000005}}	
]

VerificationTest[(* 107 *)
	OpenVDBLink`OpenVDBUnionTo[vdb1,  vdb2][$propertyList]
	,
	{44609,  0, 44609, 0.30000000000000004, 201348, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  45},  {-16,  25}, {-19,  31}}, {94,  42, 51}, {-0.29999599125227844,  0.29995050821018665}, True, 0.1, {{-4.800000000000001,  4.5},  {-1.6,  2.5}, {-1.9000000000000001,  3.1}}, {9.4,  4.2, 5.1000000000000005}}	
]

VerificationTest[(* 108 *)
	OpenVDBLink`OpenVDBUnionTo[vdb1,  vdb2, vdb3][$propertyList]
	,
	{67281,  0, 67281, 0.30000000000000004, 329888, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.29999599125227844,  0.2999503432672819}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}}	
]

VerificationTest[(* 109 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb1]; OpenVDBLink`OpenVDBUnionTo[vdb2,  vdb2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 110 *)
	vdbb = OpenVDBLink`OpenVDBCopyGrid[vdb1]; OpenVDBLink`OpenVDBUnionTo[vdb2,  vdbb, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

VerificationTest[(* 111 *)
	vdba = OpenVDBLink`OpenVDBCopyGrid[vdb1]; vdbb = OpenVDBLink`OpenVDBCopyGrid[vdb2]; {OpenVDBLink`OpenVDBUnionTo[vdba,  vdbb][$propertyList],  vdbb[$propertyList]}
	,
	{{67281,  0, 67281, 0.30000000000000004, 329888, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.29999599125227844,  0.2999503432672819}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}},  
  {67281,  0, 67281, 0.30000000000000004, 329888, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.29999599125227844,  0.2999503432672819}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBIntersectWith"]

VerificationTest[(* 112 *)
	OpenVDBLink`OpenVDBIntersectWith[vdb1][$propertyList]
	,
	{67281,  0, 67281, 0.30000000000000004, 329888, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.29999599125227844,  0.2999503432672819}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}}	
]

VerificationTest[(* 113 *)
	OpenVDBLink`OpenVDBIntersectWith[vdb1,  vdb2][$propertyList]
	,
	{67281,  0, 67281, 0.30000000000000004, 329888, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48,  55},  {-26,  25}, {-19,  41}}, {104,  52, 61}, {-0.29999599125227844,  0.2999503432672819}, True, 0.1, {{-4.800000000000001,  5.5},  {-2.6,  2.5}, {-1.9000000000000001,  4.1000000000000005}}, {10.4,  5.2, 6.1000000000000005}}	
]

VerificationTest[(* 114 *)
	OpenVDBLink`OpenVDBIntersectWith[vdb1,  vdb2, vdb3][$propertyList]
	,
	{7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {-0.2999200398082843,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}}	
]

VerificationTest[(* 115 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb1]; OpenVDBLink`OpenVDBIntersectWith[vdb2,  vdb2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 116 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb1]; OpenVDBLink`OpenVDBIntersectWith[vdb2,  vdb2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

VerificationTest[(* 117 *)
	vdba = OpenVDBLink`OpenVDBCopyGrid[vdb1]; vdbb = OpenVDBLink`OpenVDBCopyGrid[vdb2]; {OpenVDBLink`OpenVDBIntersectWith[vdba,  vdbb][$propertyList],  vdbb[$propertyList]}
	,
	{{7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {-0.2999200398082843,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}},  
  {7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {-0.2999200398082843,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBDifferenceFrom"]

VerificationTest[(* 118 *)
	OpenVDBLink`OpenVDBDifferenceFrom[vdb1][$propertyList]
	,
	{7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {-0.2999200398082843,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}}	
]

VerificationTest[(* 119 *)
	OpenVDBLink`OpenVDBDifferenceFrom[vdb1,  vdb2][$propertyList]
	,
	{7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {0.000030273042506981374,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}}	
]

VerificationTest[(* 120 *)
	OpenVDBLink`OpenVDBDifferenceFrom[vdb1,  vdb2, vdb3][$propertyList]
	,
	{7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {0.000030273042506981374,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}}	
]

VerificationTest[(* 121 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb1]; OpenVDBLink`OpenVDBDifferenceFrom[vdb2,  vdb2][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 122 *)
	vdb2 = OpenVDBLink`OpenVDBCopyGrid[vdb1]; OpenVDBLink`OpenVDBDifferenceFrom[vdb2,  vdb2, "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

VerificationTest[(* 123 *)
	vdba = OpenVDBLink`OpenVDBCopyGrid[vdb1]; vdbb = OpenVDBLink`OpenVDBCopyGrid[vdb2]; {OpenVDBLink`OpenVDBDifferenceFrom[vdba,  vdbb][$propertyList],  vdbb[$propertyList]}
	,
	{{7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {0.000030273042506981374,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}},  
  {7948,  0, 7948, 0.30000000000000004, 46376, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-24,  43},  {-16,  5}, {1,  31}}, {68,  22, 31}, {0.000030273042506981374,  0.2999321068403735}, True, 0.1, {{-2.4000000000000004,  4.3},  {-1.6,  0.5}, {0.1,  3.1}}, {6.800000000000001,  2.2, 3.1}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBClip"]

VerificationTest[(* 124 *)
	OpenVDBLink`OpenVDBClip[vdb1,  {{-0.5,  1},  {0,  1}, {0,  1}}][$propertyList]
	,
	{777,  0, 777, 0.30000000000000004, 1248, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-2,  10},  {-2,  5}, {1,  12}}, {13,  8, 12}, {0.00011350864244420722,  0.2998976303566372}, True, 0.1, {{-0.2,  1.},  {-0.2,  0.5}, {0.1,  1.2000000000000002}}, {1.3,  0.8, 1.2000000000000002}}	
]

VerificationTest[(* 125 *)
	OpenVDBLink`OpenVDBClip[fog,  {{-0.5,  1},  {0,  1}, {0,  1}}][$propertyList]
	,
	{1444,  0, 1444, 0., 1584, False, "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-5,  10},  {0,  8}, {0,  10}}, {16,  9, 11}, {0.00011808911465590229,  1.}, True, 0.1, {{-0.5,  1.},  {0.,  0.8}, {0.,  1.}}, {1.6,  0.9, 1.1}}	
]

VerificationTest[(* 126 *)
	OpenVDBLink`OpenVDBClip[vdb1,  {{-0.5,  1},  {0,  1}, {0,  1}} -> "World"][$propertyList]
	,
	{777,  0, 777, 0.30000000000000004, 1248, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-2,  10},  {-2,  5}, {1,  12}}, {13,  8, 12}, {0.00011350864244420722,  0.2998976303566372}, True, 0.1, {{-0.2,  1.},  {-0.2,  0.5}, {0.1,  1.2000000000000002}}, {1.3,  0.8, 1.2000000000000002}}	
]

VerificationTest[(* 127 *)
	OpenVDBLink`OpenVDBClip[vdb1,  {{-20,  20},  {-30,  20}, {-40,  40}} -> "Index"][$propertyList]
	,
	{5324,  0, 5324, 0.30000000000000004, 30690, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-22,  22},  {-16,  5}, {1,  31}}, {45,  22, 31}, {0.000030273042506981374,  0.2999321068403735}, True, 0.1, {{-2.2,  2.2},  {-1.6,  0.5}, {0.1,  3.1}}, {4.5,  2.2, 3.1}}	
]

VerificationTest[(* 128 *)
	OpenVDBLink`OpenVDBClip[vdb1,  {{-20,  20},  {-30,  20}, {-40,  40}} -> "Index"][{"Creator",  "Name"}]
	,
	{"initC",  "initN"}	
]

VerificationTest[(* 129 *)
	OpenVDBLink`OpenVDBClip[vdb1,  {{-20,  20},  {-30,  20}, {-40,  40}} -> "Index", "Creator" -> "test!", "Name" -> "test2!"][{"Creator",  "Name"}]
	,
	{"test!",  "test2!"}	
]

VerificationTest[(* 130 *)
	OpenVDBLink`OpenVDBClip[vdb1,  {{-20,  20},  {-30,  20}, {-40,  40}} -> "Index", "CloseBoundary" -> False][$propertyList]
	,
	{4705,  0, 4705, 0.30000000000000004, 27962, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-20,  20},  {-16,  5}, {1,  31}}, {41,  22, 31}, {0.000030273042506981374,  0.2999321068403735}, True, 0.1, {{-2.,  2.},  {-1.6,  0.5}, {0.1,  3.1}}, {4.1000000000000005,  2.2, 3.1}}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
