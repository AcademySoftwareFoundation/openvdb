BeginTestSection["Grid Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	$propertyList={"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox",     "IndexDimensions","MinMaxValues","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"};OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGrid"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBGrid]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBGrid]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBGrid]
	,
	{}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBGrid]
	,
	{}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBGrid[], OpenVDBLink`OpenVDBGrid[1.3], OpenVDBLink`OpenVDBGrid[1], OpenVDBLink`OpenVDBGrid[2, "error"], OpenVDBLink`OpenVDBGrid[3, "Vector", "error"]}
	,
	{OpenVDBLink`OpenVDBGrid[], OpenVDBLink`OpenVDBGrid[1.3], OpenVDBLink`OpenVDBGrid[1], OpenVDBLink`OpenVDBGrid[2, "error"], OpenVDBLink`OpenVDBGrid[3, "Vector", "error"]}	
]

EndTestSection[]

BeginTestSection["OpenVDBGridQ"]

VerificationTest[(* 7 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBGridQ]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 8 *)
	Attributes[OpenVDBLink`OpenVDBGridQ]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 9 *)
	Options[OpenVDBLink`OpenVDBGridQ]
	,
	{}	
]

VerificationTest[(* 10 *)
	SyntaxInformation[OpenVDBLink`OpenVDBGridQ]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 11 *)
	{OpenVDBLink`OpenVDBGridQ[], OpenVDBLink`OpenVDBGridQ["error"], OpenVDBLink`OpenVDBGridQ[OpenVDBLink`OpenVDBLevelSet[bmr], "error"]}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBGrids"]

VerificationTest[(* 12 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBGrids]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 13 *)
	Attributes[OpenVDBLink`OpenVDBGrids]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 14 *)
	Options[OpenVDBLink`OpenVDBGrids]
	,
	{}	
]

VerificationTest[(* 15 *)
	SyntaxInformation[OpenVDBLink`OpenVDBGrids]
	,
	{"ArgumentsPattern"->{_.}}	
]

VerificationTest[(* 16 *)
	{OpenVDBLink`OpenVDBGrids["error"], OpenVDBLink`OpenVDBGrids["Scalar", "error"]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBGrids::type, OpenVDBGrids::argt}
]

VerificationTest[(* 17 *)
	AssociationQ[OpenVDBLink`OpenVDBGrids[]]
	,
	True	
]

VerificationTest[(* 18 *)
	VectorQ[OpenVDBLink`OpenVDBGrids/@{Automatic, All, {"Float", "Vector"}}, AssociationQ]
	,
	True	
]

VerificationTest[(* 19 *)
	VectorQ[OpenVDBLink`OpenVDBGrids/@{"Int32", "Vector"}, ListQ]
	,
	True	
]

VerificationTest[(* 20 *)
	vdbball=OpenVDBLink`OpenVDBLevelSet[Ball[], 0.1, 3.];{OpenVDBLink`OpenVDBScalarGridQ[vdbball], ListQ[OpenVDBLink`OpenVDBGrids["Float"]], MemberQ[OpenVDBLink`OpenVDBGrids["Float"], vdbball]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBGridTypes"]

VerificationTest[(* 21 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBGridTypes]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 22 *)
	Attributes[OpenVDBLink`OpenVDBGridTypes]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 23 *)
	Options[OpenVDBLink`OpenVDBGridTypes]
	,
	{}	
]

VerificationTest[(* 24 *)
	SyntaxInformation[OpenVDBLink`OpenVDBGridTypes]
	,
	{"ArgumentsPattern"->{_.}}	
]

VerificationTest[(* 25 *)
	{OpenVDBLink`OpenVDBGridTypes["error"], OpenVDBLink`OpenVDBGridTypes["Scalar", "error"]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBGridTypes::type, OpenVDBGridTypes::argt}
]

VerificationTest[(* 26 *)
	OpenVDBLink`OpenVDBGridTypes[]
	,
	{"Scalar", "Vector", "Double", "Float", "Byte", "Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}	
]

VerificationTest[(* 27 *)
	OpenVDBLink`OpenVDBGridTypes["Scalar"]
	,
	{"Scalar", "Double", "Float"}	
]

VerificationTest[(* 28 *)
	OpenVDBLink`OpenVDBGridTypes["Integer"]
	,
	{"Byte", "Int32", "Int64", "UInt32"}	
]

VerificationTest[(* 29 *)
	OpenVDBLink`OpenVDBGridTypes["Vector"]
	,
	{"Vector", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S"}	
]

VerificationTest[(* 30 *)
	OpenVDBLink`OpenVDBGridTypes["Boolean"]
	,
	{"Boolean"}	
]

VerificationTest[(* 31 *)
	OpenVDBLink`OpenVDBGridTypes["Mask"]
	,
	{"Mask"}	
]

EndTestSection[]

BeginTestSection["OpenVDBScalarGridQ"]

VerificationTest[(* 32 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBScalarGridQ]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 33 *)
	Attributes[OpenVDBLink`OpenVDBScalarGridQ]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 34 *)
	Options[OpenVDBLink`OpenVDBScalarGridQ]
	,
	{}	
]

VerificationTest[(* 35 *)
	SyntaxInformation[OpenVDBLink`OpenVDBScalarGridQ]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 36 *)
	{OpenVDBLink`OpenVDBScalarGridQ[], OpenVDBLink`OpenVDBScalarGridQ["error"], OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Int32"], "error"]}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBIntegerGridQ"]

VerificationTest[(* 37 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBIntegerGridQ]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 38 *)
	Attributes[OpenVDBLink`OpenVDBIntegerGridQ]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 39 *)
	Options[OpenVDBLink`OpenVDBIntegerGridQ]
	,
	{}	
]

VerificationTest[(* 40 *)
	SyntaxInformation[OpenVDBLink`OpenVDBIntegerGridQ]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 41 *)
	{OpenVDBLink`OpenVDBIntegerGridQ[], OpenVDBLink`OpenVDBIntegerGridQ["error"], OpenVDBLink`OpenVDBIntegerGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Int32"], "error"]}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBVectorGridQ"]

VerificationTest[(* 42 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBVectorGridQ]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 43 *)
	Attributes[OpenVDBLink`OpenVDBVectorGridQ]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 44 *)
	Options[OpenVDBLink`OpenVDBVectorGridQ]
	,
	{}	
]

VerificationTest[(* 45 *)
	SyntaxInformation[OpenVDBLink`OpenVDBVectorGridQ]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 46 *)
	{OpenVDBLink`OpenVDBVectorGridQ[], OpenVDBLink`OpenVDBVectorGridQ["error"], OpenVDBLink`OpenVDBVectorGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Vector"], "error"]}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBBooleanGridQ"]

VerificationTest[(* 47 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBBooleanGridQ]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 48 *)
	Attributes[OpenVDBLink`OpenVDBBooleanGridQ]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 49 *)
	Options[OpenVDBLink`OpenVDBBooleanGridQ]
	,
	{}	
]

VerificationTest[(* 50 *)
	SyntaxInformation[OpenVDBLink`OpenVDBBooleanGridQ]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 51 *)
	{OpenVDBLink`OpenVDBBooleanGridQ[], OpenVDBLink`OpenVDBBooleanGridQ["error"], OpenVDBLink`OpenVDBBooleanGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"], "error"]}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBMaskGridQ"]

VerificationTest[(* 52 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBMaskGridQ]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 53 *)
	Attributes[OpenVDBLink`OpenVDBMaskGridQ]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 54 *)
	Options[OpenVDBLink`OpenVDBMaskGridQ]
	,
	{}	
]

VerificationTest[(* 55 *)
	SyntaxInformation[OpenVDBLink`OpenVDBMaskGridQ]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 56 *)
	{OpenVDBLink`OpenVDBMaskGridQ[], OpenVDBLink`OpenVDBMaskGridQ["error"], OpenVDBLink`OpenVDBMaskGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Mask"], "error"]}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBCreateGrid"]

VerificationTest[(* 57 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBCreateGrid]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 58 *)
	Attributes[OpenVDBLink`OpenVDBScalarGridQ]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 59 *)
	Options[OpenVDBLink`OpenVDBCreateGrid]
	,
	{"BackgroundValue"->Automatic, "Creator":>OpenVDBLink`$OpenVDBCreator, "GridClass"->None, "Name"->None}	
]

VerificationTest[(* 60 *)
	SyntaxInformation[OpenVDBLink`OpenVDBCreateGrid]
	,
	{"ArgumentsPattern"->{_., _., OptionsPattern[]}}	
]

VerificationTest[(* 61 *)
	{OpenVDBLink`OpenVDBCreateGrid["error"], OpenVDBLink`OpenVDBCreateGrid[1., "error"], OpenVDBLink`OpenVDBCreateGrid[1., "Scalar", "error"]}
	,
	{$Failed, $Failed, $Failed}
	,
	{OpenVDBCreateGrid::nonpos, OpenVDBCreateGrid::type, OpenVDBCreateGrid::nonopt}
]

EndTestSection[]

BeginTestSection["OpenVDBClearGrid"]

VerificationTest[(* 62 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBClearGrid]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 63 *)
	Attributes[OpenVDBLink`OpenVDBClearGrid]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 64 *)
	Options[OpenVDBLink`OpenVDBClearGrid]
	,
	{}	
]

VerificationTest[(* 65 *)
	SyntaxInformation[OpenVDBLink`OpenVDBClearGrid]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 66 *)
	{OpenVDBLink`OpenVDBClearGrid[], OpenVDBLink`OpenVDBClearGrid["error"], OpenVDBLink`OpenVDBClearGrid[OpenVDBLink`OpenVDBCreateGrid[1.], "error"]}
	,
	{$Failed, $Failed, $Failed}
	,
	{OpenVDBClearGrid::argx, OpenVDBClearGrid::grids, OpenVDBClearGrid::argx}
]

EndTestSection[]

BeginTestSection["OpenVDBCopyGrid"]

VerificationTest[(* 67 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBCopyGrid]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 68 *)
	Attributes[OpenVDBLink`OpenVDBCopyGrid]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 69 *)
	Options[OpenVDBLink`OpenVDBCopyGrid]
	,
	{"Creator"->Inherited, "Name"->Inherited}	
]

VerificationTest[(* 70 *)
	SyntaxInformation[OpenVDBLink`OpenVDBCopyGrid]
	,
	{"ArgumentsPattern"->{_, OptionsPattern[]}}	
]

VerificationTest[(* 71 *)
	{OpenVDBLink`OpenVDBCopyGrid[], OpenVDBLink`OpenVDBCopyGrid["error"], OpenVDBLink`OpenVDBCopyGrid[OpenVDBLink`OpenVDBCreateGrid[1.], "error"]}
	,
	{$Failed, $Failed, $Failed}
	,
	{OpenVDBCopyGrid::argx, OpenVDBCopyGrid::grid, OpenVDBCopyGrid::nonopt}
]

EndTestSection[]

BeginTestSection["$OpenVDBSpacing"]

VerificationTest[(* 72 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`$OpenVDBSpacing]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 73 *)
	Attributes[OpenVDBLink`$OpenVDBSpacing]
	,
	{}	
]

VerificationTest[(* 74 *)
	Options[OpenVDBLink`$OpenVDBSpacing]
	,
	{}	
]

VerificationTest[(* 75 *)
	SyntaxInformation[OpenVDBLink`$OpenVDBSpacing]
	,
	{}	
]

VerificationTest[(* 76 *)
	{OpenVDBLink`$OpenVDBSpacing=0.3, OpenVDBLink`$OpenVDBSpacing}
	,
	{0.3, 0.3}	
]

VerificationTest[(* 77 *)
	{OpenVDBLink`$OpenVDBSpacing=., OpenVDBLink`$OpenVDBSpacing}
	,
	{Null, OpenVDBLink`$OpenVDBSpacing}	
]

VerificationTest[(* 78 *)
	{OpenVDBLink`$OpenVDBSpacing=.;OpenVDBLink`$OpenVDBSpacing=x, OpenVDBLink`$OpenVDBSpacing}
	,
	{$Failed, OpenVDBLink`$OpenVDBSpacing}
	,
	{OpenVDBLink`$OpenVDBSpacing::setpos}
]

VerificationTest[(* 79 *)
	{OpenVDBLink`$OpenVDBSpacing=.;OpenVDBLink`$OpenVDBSpacing=0, OpenVDBLink`$OpenVDBSpacing}
	,
	{$Failed, OpenVDBLink`$OpenVDBSpacing}
	,
	{OpenVDBLink`$OpenVDBSpacing::setpos}
]

VerificationTest[(* 80 *)
	{OpenVDBLink`$OpenVDBSpacing=.;OpenVDBLink`$OpenVDBSpacing=-1., OpenVDBLink`$OpenVDBSpacing}
	,
	{$Failed, OpenVDBLink`$OpenVDBSpacing}
	,
	{OpenVDBLink`$OpenVDBSpacing::setpos}
]

VerificationTest[(* 81 *)
	{OpenVDBLink`$OpenVDBSpacing=.;OpenVDBLink`$OpenVDBSpacing=I, OpenVDBLink`$OpenVDBSpacing}
	,
	{$Failed, OpenVDBLink`$OpenVDBSpacing}
	,
	{OpenVDBLink`$OpenVDBSpacing::setpos}
]

EndTestSection[]

BeginTestSection["$OpenVDBHalfWidth"]

VerificationTest[(* 82 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`$OpenVDBHalfWidth]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 83 *)
	Attributes[OpenVDBLink`$OpenVDBHalfWidth]
	,
	{}	
]

VerificationTest[(* 84 *)
	Options[OpenVDBLink`$OpenVDBHalfWidth]
	,
	{}	
]

VerificationTest[(* 85 *)
	SyntaxInformation[OpenVDBLink`$OpenVDBHalfWidth]
	,
	{}	
]

VerificationTest[(* 86 *)
	{OpenVDBLink`$OpenVDBHalfWidth=3.5, OpenVDBLink`$OpenVDBHalfWidth}
	,
	{3.5, 3.5}	
]

VerificationTest[(* 87 *)
	{OpenVDBLink`$OpenVDBHalfWidth=., OpenVDBLink`$OpenVDBHalfWidth}
	,
	{Null, OpenVDBLink`$OpenVDBHalfWidth}	
]

VerificationTest[(* 88 *)
	{OpenVDBLink`$OpenVDBHalfWidth=.;OpenVDBLink`$OpenVDBHalfWidth=x, OpenVDBLink`$OpenVDBHalfWidth}
	,
	{$Failed, OpenVDBLink`$OpenVDBHalfWidth}
	,
	{OpenVDBLink`$OpenVDBHalfWidth::setpos}
]

VerificationTest[(* 89 *)
	{OpenVDBLink`$OpenVDBHalfWidth=.;OpenVDBLink`$OpenVDBHalfWidth=0.5, OpenVDBLink`$OpenVDBHalfWidth}
	,
	{0.5, 0.5}	
]

VerificationTest[(* 90 *)
	{OpenVDBLink`$OpenVDBHalfWidth=.;OpenVDBLink`$OpenVDBHalfWidth=-1., OpenVDBLink`$OpenVDBHalfWidth}
	,
	{$Failed, OpenVDBLink`$OpenVDBHalfWidth}
	,
	{OpenVDBLink`$OpenVDBHalfWidth::setpos}
]

VerificationTest[(* 91 *)
	{OpenVDBLink`$OpenVDBHalfWidth=.;OpenVDBLink`$OpenVDBHalfWidth=I, OpenVDBLink`$OpenVDBHalfWidth}
	,
	{$Failed, OpenVDBLink`$OpenVDBHalfWidth}
	,
	{Greater::nord, $OpenVDBHalfWidth::setpos}
]

VerificationTest[(* 92 *)
	{OpenVDBLink`$OpenVDBSpacing=0.1, OpenVDBLink`$OpenVDBHalfWidth=3.}
	,
	{0.1, 3.}	
]

EndTestSection[]

BeginTestSection["$OpenVDBCreator"]

VerificationTest[(* 93 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`$OpenVDBCreator]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 94 *)
	Attributes[OpenVDBLink`$OpenVDBCreator]
	,
	{}	
]

VerificationTest[(* 95 *)
	Options[OpenVDBLink`$OpenVDBCreator]
	,
	{}	
]

VerificationTest[(* 96 *)
	SyntaxInformation[OpenVDBLink`$OpenVDBCreator]
	,
	{}	
]

VerificationTest[(* 97 *)
	{OpenVDBLink`$OpenVDBCreator="Gauss", OpenVDBLink`$OpenVDBCreator}
	,
	{"Gauss", "Gauss"}	
]

VerificationTest[(* 98 *)
	{OpenVDBLink`$OpenVDBCreator=., OpenVDBLink`$OpenVDBCreator}
	,
	{Null, OpenVDBLink`$OpenVDBCreator}	
]

VerificationTest[(* 99 *)
	{OpenVDBLink`$OpenVDBCreator=x, OpenVDBLink`$OpenVDBCreator}
	,
	{$Failed, OpenVDBLink`$OpenVDBCreator}
	,
	{OpenVDBLink`$OpenVDBCreator::badset}
]

VerificationTest[(* 100 *)
	{OpenVDBLink`$OpenVDBCreator=.;OpenVDBLink`$OpenVDBCreator=2, OpenVDBLink`$OpenVDBCreator}
	,
	{$Failed, OpenVDBLink`$OpenVDBCreator}
	,
	{OpenVDBLink`$OpenVDBCreator::badset}
]

VerificationTest[(* 101 *)
	{OpenVDBLink`$OpenVDBCreator="", OpenVDBLink`$OpenVDBCreator}
	,
	{"", ""}	
]

EndTestSection[]

BeginTestSection["OpenVDBDefaultSpace"]

VerificationTest[(* 102 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBDefaultSpace]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 103 *)
	Attributes[OpenVDBLink`OpenVDBDefaultSpace]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 104 *)
	Options[OpenVDBLink`OpenVDBDefaultSpace]
	,
	{}	
]

VerificationTest[(* 105 *)
	SyntaxInformation[OpenVDBLink`OpenVDBDefaultSpace]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 106 *)
	{OpenVDBLink`OpenVDBDefaultSpace[], OpenVDBLink`OpenVDBDefaultSpace["error", "error"]}
	,
	{$Failed, $Failed}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 107 *)
	$propertyList={"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox",     "IndexDimensions","MinMaxValues","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"};OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBGrid"]

VerificationTest[(* 108 *)
	Head/@{vdbempty, vdb, fog}
	,
	{OpenVDBLink`OpenVDBGrid, OpenVDBLink`OpenVDBGrid, OpenVDBLink`OpenVDBGrid}	
]

VerificationTest[(* 109 *)
	(MatchQ[#1, OpenVDBLink`OpenVDBGrid[_Integer?Positive, "Float"]]&)/@{vdbempty, vdb, fog}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBGridQ"]

VerificationTest[(* 110 *)
	OpenVDBLink`OpenVDBGridQ/@{vdbempty, vdb, fog}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBScalarGridQ"]

VerificationTest[(* 111 *)
	OpenVDBLink`OpenVDBScalarGridQ/@{vdbempty, vdb, fog}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBIntegerGridQ"]

VerificationTest[(* 112 *)
	OpenVDBLink`OpenVDBIntegerGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBVectorGridQ"]

VerificationTest[(* 113 *)
	OpenVDBLink`OpenVDBVectorGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBBooleanGridQ"]

VerificationTest[(* 114 *)
	OpenVDBLink`OpenVDBBooleanGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBMaskGridQ"]

VerificationTest[(* 115 *)
	OpenVDBLink`OpenVDBMaskGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBCreateGrid"]

VerificationTest[(* 116 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"]]
	,
	True	
]

VerificationTest[(* 117 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBCreateGrid[0.4, "Float"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBClearGrid"]

VerificationTest[(* 118 *)
	vdb2=OpenVDBLink`OpenVDBLevelSet[Ball[]];{vdb2["ActiveVoxelCount"], OpenVDBLink`OpenVDBClearGrid[vdb2];vdb2["ActiveVoxelCount"]}
	,
	{7674, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBCopyGrid"]

VerificationTest[(* 119 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];vdb[$propertyList]===vdb2[$propertyList]&&ListQ[vdb[$propertyList]]
	,
	True	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 120 *)
	$propertyList={"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox",     "IndexDimensions","MinMaxValues","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"};OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Double"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBGrid"]

VerificationTest[(* 121 *)
	Head/@{vdbempty, vdb, fog}
	,
	{OpenVDBLink`OpenVDBGrid, OpenVDBLink`OpenVDBGrid, OpenVDBLink`OpenVDBGrid}	
]

VerificationTest[(* 122 *)
	(MatchQ[#1, OpenVDBLink`OpenVDBGrid[_Integer?Positive, "Double"]]&)/@{vdbempty, vdb, fog}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBGridQ"]

VerificationTest[(* 123 *)
	OpenVDBLink`OpenVDBGridQ/@{vdbempty, vdb, fog}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBScalarGridQ"]

VerificationTest[(* 124 *)
	OpenVDBLink`OpenVDBScalarGridQ/@{vdbempty, vdb, fog}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBIntegerGridQ"]

VerificationTest[(* 125 *)
	OpenVDBLink`OpenVDBIntegerGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBVectorGridQ"]

VerificationTest[(* 126 *)
	OpenVDBLink`OpenVDBVectorGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBBooleanGridQ"]

VerificationTest[(* 127 *)
	OpenVDBLink`OpenVDBBooleanGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBMaskGridQ"]

VerificationTest[(* 128 *)
	OpenVDBLink`OpenVDBMaskGridQ/@{vdbempty, vdb, fog}
	,
	{False, False, False}	
]

EndTestSection[]

BeginTestSection["OpenVDBCreateGrid"]

VerificationTest[(* 129 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"]]
	,
	True	
]

VerificationTest[(* 130 *)
	OpenVDBLink`OpenVDBScalarGridQ[OpenVDBLink`OpenVDBCreateGrid[0.4, "Float"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBClearGrid"]

VerificationTest[(* 131 *)
	vdb2=OpenVDBLink`OpenVDBLevelSet[Ball[]];{vdb2["ActiveVoxelCount"], OpenVDBLink`OpenVDBClearGrid[vdb2];vdb2["ActiveVoxelCount"]}
	,
	{7674, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBCopyGrid"]

VerificationTest[(* 132 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];vdb[$propertyList]===vdb2[$propertyList]&&ListQ[vdb[$propertyList]]
	,
	True	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Integer"]

BeginTestSection["Initialization"]

VerificationTest[(* 133 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Int64"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGrid"]

VerificationTest[(* 134 *)
	Head[vdb]
	,
	OpenVDBLink`OpenVDBGrid	
]

VerificationTest[(* 135 *)
	MatchQ[vdb, OpenVDBLink`OpenVDBGrid[_Integer?Positive, "Int64"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGridQ"]

VerificationTest[(* 136 *)
	OpenVDBLink`OpenVDBGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBScalarGridQ"]

VerificationTest[(* 137 *)
	OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBIntegerGridQ"]

VerificationTest[(* 138 *)
	OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBVectorGridQ"]

VerificationTest[(* 139 *)
	OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBBooleanGridQ"]

VerificationTest[(* 140 *)
	OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBMaskGridQ"]

VerificationTest[(* 141 *)
	OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBCreateGrid"]

VerificationTest[(* 142 *)
	(OpenVDBLink`OpenVDBIntegerGridQ[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32"}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBClearGrid"]

VerificationTest[(* 143 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb2["ActiveVoxelCount"], OpenVDBLink`OpenVDBClearGrid[vdb2];vdb2["ActiveVoxelCount"]}
	,
	{20, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBCopyGrid"]

VerificationTest[(* 144 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];vdb[$propertyList]===vdb2[$propertyList]&&ListQ[vdb[$propertyList]]
	,
	True	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Vector"]

BeginTestSection["Initialization"]

VerificationTest[(* 145 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Vec2D"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[{12 - i, 13 - i}, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[{12 - i, 13 - i}, {i, 10}]];OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGrid"]

VerificationTest[(* 146 *)
	Head[vdb]
	,
	OpenVDBLink`OpenVDBGrid	
]

VerificationTest[(* 147 *)
	MatchQ[vdb, OpenVDBLink`OpenVDBGrid[_Integer?Positive, "Vec2D"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGridQ"]

VerificationTest[(* 148 *)
	OpenVDBLink`OpenVDBGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBScalarGridQ"]

VerificationTest[(* 149 *)
	OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBIntegerGridQ"]

VerificationTest[(* 150 *)
	OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBVectorGridQ"]

VerificationTest[(* 151 *)
	OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBBooleanGridQ"]

VerificationTest[(* 152 *)
	OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBMaskGridQ"]

VerificationTest[(* 153 *)
	OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBCreateGrid"]

VerificationTest[(* 154 *)
	(OpenVDBLink`OpenVDBVectorGridQ[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S"}
	,
	{True, True, True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBClearGrid"]

VerificationTest[(* 155 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb2["ActiveVoxelCount"], OpenVDBLink`OpenVDBClearGrid[vdb2];vdb2["ActiveVoxelCount"]}
	,
	{20, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBCopyGrid"]

VerificationTest[(* 156 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];vdb[$propertyList]===vdb2[$propertyList]&&ListQ[vdb[$propertyList]]
	,
	True	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 157 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGrid"]

VerificationTest[(* 158 *)
	Head[vdb]
	,
	OpenVDBLink`OpenVDBGrid	
]

VerificationTest[(* 159 *)
	MatchQ[vdb, OpenVDBLink`OpenVDBGrid[_Integer?Positive, "Boolean"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGridQ"]

VerificationTest[(* 160 *)
	OpenVDBLink`OpenVDBGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBScalarGridQ"]

VerificationTest[(* 161 *)
	OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBIntegerGridQ"]

VerificationTest[(* 162 *)
	OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBVectorGridQ"]

VerificationTest[(* 163 *)
	OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBBooleanGridQ"]

VerificationTest[(* 164 *)
	OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBMaskGridQ"]

VerificationTest[(* 165 *)
	OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBCreateGrid"]

VerificationTest[(* 166 *)
	OpenVDBLink`OpenVDBBooleanGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBClearGrid"]

VerificationTest[(* 167 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb2["ActiveVoxelCount"], OpenVDBLink`OpenVDBClearGrid[vdb2];vdb2["ActiveVoxelCount"]}
	,
	{20, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBCopyGrid"]

VerificationTest[(* 168 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];vdb[$propertyList]===vdb2[$propertyList]&&ListQ[vdb[$propertyList]]
	,
	True	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 169 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGrid"]

VerificationTest[(* 170 *)
	Head[vdb]
	,
	OpenVDBLink`OpenVDBGrid	
]

VerificationTest[(* 171 *)
	MatchQ[vdb, OpenVDBLink`OpenVDBGrid[_Integer?Positive, "Mask"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBGridQ"]

VerificationTest[(* 172 *)
	OpenVDBLink`OpenVDBGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBScalarGridQ"]

VerificationTest[(* 173 *)
	OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBIntegerGridQ"]

VerificationTest[(* 174 *)
	OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBVectorGridQ"]

VerificationTest[(* 175 *)
	OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBBooleanGridQ"]

VerificationTest[(* 176 *)
	OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	False	
]

EndTestSection[]

BeginTestSection["OpenVDBMaskGridQ"]

VerificationTest[(* 177 *)
	OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBCreateGrid"]

VerificationTest[(* 178 *)
	OpenVDBLink`OpenVDBMaskGridQ[OpenVDBLink`OpenVDBCreateGrid[1., "Mask"]]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBClearGrid"]

VerificationTest[(* 179 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb2["ActiveVoxelCount"], OpenVDBLink`OpenVDBClearGrid[vdb2];vdb2["ActiveVoxelCount"]}
	,
	{12, 0}	
]

EndTestSection[]

BeginTestSection["OpenVDBCopyGrid"]

VerificationTest[(* 180 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];vdb[$propertyList]===vdb2[$propertyList]&&ListQ[vdb[$propertyList]]
	,
	True	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
