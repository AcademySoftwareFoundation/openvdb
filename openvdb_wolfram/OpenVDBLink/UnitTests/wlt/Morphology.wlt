BeginTestSection["Morphology Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;vdb=OpenVDBLink`OpenVDBLevelSet[ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"]];OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBResizeBandwidth"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBResizeBandwidth]
	,
	"Index"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBResizeBandwidth]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBResizeBandwidth]
	,
	{}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBResizeBandwidth]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBResizeBandwidth[], OpenVDBLink`OpenVDBResizeBandwidth["error"], OpenVDBLink`OpenVDBResizeBandwidth[vdb, "error"], OpenVDBLink`OpenVDBResizeBandwidth[vdb, 2, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBResizeBandwidth::argt, OpenVDBResizeBandwidth::scalargrid2, OpenVDBResizeBandwidth::nonneg, OpenVDBResizeBandwidth::argt}
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBResizeBandwidth[OpenVDBLink`OpenVDBCreateGrid[1., #1], 2]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBResizeBandwidth::scalargrid2, OpenVDBResizeBandwidth::scalargrid2, OpenVDBResizeBandwidth::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBDilation"]

VerificationTest[(* 8 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBDilation]
	,
	"World"	
]

VerificationTest[(* 9 *)
	Attributes[OpenVDBLink`OpenVDBDilation]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 10 *)
	Options[OpenVDBLink`OpenVDBDilation]
	,
	{}	
]

VerificationTest[(* 11 *)
	SyntaxInformation[OpenVDBLink`OpenVDBDilation]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 12 *)
	{OpenVDBLink`OpenVDBDilation[], OpenVDBLink`OpenVDBDilation["error"], OpenVDBLink`OpenVDBDilation[vdb, "error"], OpenVDBLink`OpenVDBDilation[vdb, 2->"Index", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDilation::argt, OpenVDBDilation::scalargrid2, OpenVDBDilation::nonneg, OpenVDBDilation::argt}
]

VerificationTest[(* 13 *)
	(OpenVDBLink`OpenVDBDilation[OpenVDBLink`OpenVDBCreateGrid[1., #1], 2->"Index"]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBDilation::scalargrid2, OpenVDBDilation::scalargrid2, OpenVDBDilation::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBErosion"]

VerificationTest[(* 14 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBErosion]
	,
	"World"	
]

VerificationTest[(* 15 *)
	Attributes[OpenVDBLink`OpenVDBErosion]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 16 *)
	Options[OpenVDBLink`OpenVDBErosion]
	,
	{}	
]

VerificationTest[(* 17 *)
	SyntaxInformation[OpenVDBLink`OpenVDBErosion]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 18 *)
	{OpenVDBLink`OpenVDBErosion[], OpenVDBLink`OpenVDBErosion["error"], OpenVDBLink`OpenVDBErosion[vdb, "error"], OpenVDBLink`OpenVDBErosion[vdb, 2->"Index", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBErosion::argt, OpenVDBErosion::scalargrid2, OpenVDBErosion::nonneg, OpenVDBErosion::argt}
]

VerificationTest[(* 19 *)
	(OpenVDBLink`OpenVDBErosion[OpenVDBLink`OpenVDBCreateGrid[1., #1], 2->"Index"]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBErosion::scalargrid2, OpenVDBErosion::scalargrid2, OpenVDBErosion::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBClosing"]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBClosing]
	,
	"World"	
]

VerificationTest[(* 21 *)
	Attributes[OpenVDBLink`OpenVDBClosing]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 22 *)
	Options[OpenVDBLink`OpenVDBClosing]
	,
	{}	
]

VerificationTest[(* 23 *)
	SyntaxInformation[OpenVDBLink`OpenVDBClosing]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 24 *)
	{OpenVDBLink`OpenVDBClosing[], OpenVDBLink`OpenVDBClosing["error"], OpenVDBLink`OpenVDBClosing[vdb, "error"], OpenVDBLink`OpenVDBClosing[vdb, 2->"Index", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBClosing::argt, OpenVDBClosing::scalargrid2, OpenVDBClosing::nonneg, OpenVDBClosing::argt}
]

VerificationTest[(* 25 *)
	(OpenVDBLink`OpenVDBClosing[OpenVDBLink`OpenVDBCreateGrid[1., #1], 2->"Index"]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBClosing::scalargrid2, OpenVDBClosing::scalargrid2, OpenVDBClosing::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBOpening"]

VerificationTest[(* 26 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBOpening]
	,
	"World"	
]

VerificationTest[(* 27 *)
	Attributes[OpenVDBLink`OpenVDBOpening]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 28 *)
	Options[OpenVDBLink`OpenVDBOpening]
	,
	{}	
]

VerificationTest[(* 29 *)
	SyntaxInformation[OpenVDBLink`OpenVDBOpening]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 30 *)
	{OpenVDBLink`OpenVDBOpening[], OpenVDBLink`OpenVDBOpening["error"], OpenVDBLink`OpenVDBOpening[vdb, "error"], OpenVDBLink`OpenVDBOpening[vdb, 2->"Index", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBOpening::argt, OpenVDBOpening::scalargrid2, OpenVDBOpening::nonneg, OpenVDBOpening::argt}
]

VerificationTest[(* 31 *)
	(OpenVDBLink`OpenVDBOpening[OpenVDBLink`OpenVDBCreateGrid[1., #1], 2->"Index"]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBOpening::scalargrid2, OpenVDBOpening::scalargrid2, OpenVDBOpening::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 32 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb],    OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBResizeBandwidth"]

VerificationTest[(* 33 *)
	OpenVDBLink`OpenVDBResizeBandwidth[bmr, 2]["ActiveVoxelCount"]
	,
	18602	
]

VerificationTest[(* 34 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBResizeBandwidth[vdb2, 2]["ActiveVoxelCount"]
	,
	18602	
]

VerificationTest[(* 35 *)
	OpenVDBLink`OpenVDBResizeBandwidth[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBResizeBandwidth::lvlsetgrid2}
]

VerificationTest[(* 36 *)
	OpenVDBLink`OpenVDBResizeBandwidth[bmr, 0.2->"World"]["ActiveVoxelCount"]
	,
	18602	
]

VerificationTest[(* 37 *)
	{OpenVDBLink`OpenVDBResizeBandwidth[bmr, 0.2], OpenVDBLink`OpenVDBResizeBandwidth[bmr, 0.51]["ActiveVoxelCount"]}
	,
	{$Failed, 11650}	
]

EndTestSection[]

BeginTestSection["OpenVDBDilation"]

VerificationTest[(* 38 *)
	OpenVDBLink`OpenVDBDilation[bmr, 0.2]["ActiveVoxelCount"]
	,
	37723	
]

VerificationTest[(* 39 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBDilation[vdb2, 0.2]["ActiveVoxelCount"]
	,
	37723	
]

VerificationTest[(* 40 *)
	OpenVDBLink`OpenVDBDilation[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBDilation::lvlsetgrid2}
]

VerificationTest[(* 41 *)
	OpenVDBLink`OpenVDBDilation[bmr, 0.2->"World"]["ActiveVoxelCount"]
	,
	37723	
]

VerificationTest[(* 42 *)
	OpenVDBLink`OpenVDBDilation[bmr, 1->"Index"]["ActiveVoxelCount"]
	,
	31851	
]

VerificationTest[(* 43 *)
	{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBDilation[bmr, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

BeginTestSection["OpenVDBErosion"]

VerificationTest[(* 44 *)
	OpenVDBLink`OpenVDBErosion[bmr, 0.2]["ActiveVoxelCount"]
	,
	14734	
]

VerificationTest[(* 45 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBErosion[vdb2, 0.2]["ActiveVoxelCount"]
	,
	14734	
]

VerificationTest[(* 46 *)
	OpenVDBLink`OpenVDBErosion[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBErosion::lvlsetgrid2}
]

VerificationTest[(* 47 *)
	OpenVDBLink`OpenVDBErosion[bmr, 0.2->"World"]["ActiveVoxelCount"]
	,
	14734	
]

VerificationTest[(* 48 *)
	OpenVDBLink`OpenVDBErosion[bmr, 1->"Index"]["ActiveVoxelCount"]
	,
	20219	
]

VerificationTest[(* 49 *)
	{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBErosion[bmr, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

BeginTestSection["OpenVDBClosing"]

VerificationTest[(* 50 *)
	OpenVDBLink`OpenVDBClosing[bmr, 0.2]["ActiveVoxelCount"]
	,
	25620	
]

VerificationTest[(* 51 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBClosing[vdb2, 0.2]["ActiveVoxelCount"]
	,
	25620	
]

VerificationTest[(* 52 *)
	OpenVDBLink`OpenVDBClosing[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBClosing::lvlsetgrid2}
]

VerificationTest[(* 53 *)
	OpenVDBLink`OpenVDBClosing[bmr, 0.2->"World"]["ActiveVoxelCount"]
	,
	25620	
]

VerificationTest[(* 54 *)
	OpenVDBLink`OpenVDBClosing[bmr, 1->"Index"]["ActiveVoxelCount"]
	,
	25659	
]

VerificationTest[(* 55 *)
	{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBClosing[bmr, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

BeginTestSection["OpenVDBOpening"]

VerificationTest[(* 56 *)
	OpenVDBLink`OpenVDBOpening[bmr, 0.2]["ActiveVoxelCount"]
	,
	22092	
]

VerificationTest[(* 57 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBOpening[vdb2, 0.2]["ActiveVoxelCount"]
	,
	22092	
]

VerificationTest[(* 58 *)
	OpenVDBLink`OpenVDBOpening[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBOpening::lvlsetgrid2}
]

VerificationTest[(* 59 *)
	OpenVDBLink`OpenVDBOpening[bmr, 0.2->"World"]["ActiveVoxelCount"]
	,
	22092	
]

VerificationTest[(* 60 *)
	OpenVDBLink`OpenVDBOpening[bmr, 1->"Index"]["ActiveVoxelCount"]
	,
	24742	
]

VerificationTest[(* 61 *)
	{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBOpening[bmr, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 62 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Double"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty],    OpenVDBLink`OpenVDBScalarGridQ[vdb],OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBResizeBandwidth"]

VerificationTest[(* 63 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBResizeBandwidth[vdb2, 2]["ActiveVoxelCount"]
	,
	18602	
]

VerificationTest[(* 64 *)
	OpenVDBLink`OpenVDBResizeBandwidth[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBResizeBandwidth::lvlsetgrid2}
]

VerificationTest[(* 65 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBResizeBandwidth[vdb2, 0.2->"World"]["ActiveVoxelCount"]
	,
	18602	
]

VerificationTest[(* 66 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];vdb3=OpenVDBLink`OpenVDBCopyGrid[vdb];{OpenVDBLink`OpenVDBResizeBandwidth[vdb2, 0.2], OpenVDBLink`OpenVDBResizeBandwidth[vdb3, 0.51]["ActiveVoxelCount"]}
	,
	{$Failed, 11650}	
]

EndTestSection[]

BeginTestSection["OpenVDBDilation"]

VerificationTest[(* 67 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBDilation[vdb2, 0.2]["ActiveVoxelCount"]
	,
	37723	
]

VerificationTest[(* 68 *)
	OpenVDBLink`OpenVDBDilation[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBDilation::lvlsetgrid2}
]

VerificationTest[(* 69 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBDilation[vdb2, 0.2->"World"]["ActiveVoxelCount"]
	,
	37723	
]

VerificationTest[(* 70 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBDilation[vdb2, 1->"Index"]["ActiveVoxelCount"]
	,
	31851	
]

VerificationTest[(* 71 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBDilation[vdb2, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

BeginTestSection["OpenVDBErosion"]

VerificationTest[(* 72 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBErosion[vdb2, 0.2]["ActiveVoxelCount"]
	,
	14734	
]

VerificationTest[(* 73 *)
	OpenVDBLink`OpenVDBErosion[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBErosion::lvlsetgrid2}
]

VerificationTest[(* 74 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBErosion[vdb2, 0.2->"World"]["ActiveVoxelCount"]
	,
	14734	
]

VerificationTest[(* 75 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBErosion[vdb2, 1->"Index"]["ActiveVoxelCount"]
	,
	20219	
]

VerificationTest[(* 76 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBErosion[vdb2, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

BeginTestSection["OpenVDBClosing"]

VerificationTest[(* 77 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBClosing[vdb2, 0.2]["ActiveVoxelCount"]
	,
	25620	
]

VerificationTest[(* 78 *)
	OpenVDBLink`OpenVDBClosing[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBClosing::lvlsetgrid2}
]

VerificationTest[(* 79 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBClosing[vdb2, 0.2->"World"]["ActiveVoxelCount"]
	,
	25620	
]

VerificationTest[(* 80 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBClosing[vdb2, 1->"Index"]["ActiveVoxelCount"]
	,
	25659	
]

VerificationTest[(* 81 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBClosing[vdb2, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

BeginTestSection["OpenVDBOpening"]

VerificationTest[(* 82 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBOpening[vdb2, 0.2]["ActiveVoxelCount"]
	,
	22092	
]

VerificationTest[(* 83 *)
	OpenVDBLink`OpenVDBOpening[fog, 0.2]
	,
	$Failed
	,
	{OpenVDBOpening::lvlsetgrid2}
]

VerificationTest[(* 84 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBOpening[vdb2, 0.2->"World"]["ActiveVoxelCount"]
	,
	22092	
]

VerificationTest[(* 85 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBOpening[vdb2, 1->"Index"]["ActiveVoxelCount"]
	,
	24742	
]

VerificationTest[(* 86 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBOpening[vdb2, 0]["ActiveVoxelCount"]}
	,
	{26073, 26073}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
