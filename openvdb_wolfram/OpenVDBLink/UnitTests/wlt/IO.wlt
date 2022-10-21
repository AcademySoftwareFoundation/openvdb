BeginTestSection["IO Tests"]

BeginTestSection["Generic"]

BeginTestSection["OpenVDBExport"]

VerificationTest[(* 1 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBExport]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 2 *)
	Attributes[OpenVDBLink`OpenVDBExport]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 3 *)
	Options[OpenVDBLink`OpenVDBExport]
	,
	{OverwriteTarget->False}	
]

VerificationTest[(* 4 *)
	SyntaxInformation[OpenVDBLink`OpenVDBExport]
	,
	{"ArgumentsPattern"->{_, _, OptionsPattern[]}}	
]

VerificationTest[(* 5 *)
	{OpenVDBLink`OpenVDBExport[], OpenVDBLink`OpenVDBExport["error"], OpenVDBLink`OpenVDBExport["file", "error"], OpenVDBLink`OpenVDBExport["error", bmr, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBExport::argt, OpenVDBExport::grid2, OpenVDBExport::nonopt}
]

EndTestSection[]

BeginTestSection["OpenVDBImport"]

VerificationTest[(* 6 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBImport]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 7 *)
	Attributes[OpenVDBLink`OpenVDBImport]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 8 *)
	Options[OpenVDBLink`OpenVDBImport]
	,
	{}	
]

VerificationTest[(* 9 *)
	SyntaxInformation[OpenVDBLink`OpenVDBImport]
	,
	{"ArgumentsPattern"->{_, _., _.}}	
]

VerificationTest[(* 10 *)
	{OpenVDBLink`OpenVDBImport[], OpenVDBLink`OpenVDBImport[FileNameJoin[{CreateUUID[], "file.vdb"}]], OpenVDBLink`OpenVDBImport["file", "error"], OpenVDBLink`OpenVDBImport["file", Automatic, "error"], OpenVDBLink`OpenVDBImport["file", Automatic, "Scalar", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBImport::argb, OpenVDBImport::nffil, OpenVDBImport::nffil, OpenVDBImport::nffil, General::stop, OpenVDBImport::argb}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 11 *)
	temporaryFile=Function[body, WithCleanup[filename=FileNameJoin[{$TemporaryDirectory, StringJoin[CreateUUID[], ".vdb"]}], body, DeleteFile[filename]], HoldFirst];OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb],    OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBExport"]

VerificationTest[(* 12 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, bmr];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

VerificationTest[(* 13 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

VerificationTest[(* 14 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, fog];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImport"]

VerificationTest[(* 15 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, bmr];vv=OpenVDBLink`OpenVDBImport[filename];vv[{"ActiveVoxelCount", "GridClass"}]]
	,
	{26073, "LevelSet"}	
]

VerificationTest[(* 16 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename];vv[{"ActiveVoxelCount", "GridClass"}]]
	,
	{26073, "LevelSet"}	
]

VerificationTest[(* 17 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, fog];vv=OpenVDBLink`OpenVDBImport[filename];vv[{"ActiveVoxelCount", "GridClass"}]]
	,
	{12052, "FogVolume"}	
]

VerificationTest[(* 18 *)
	temporaryFile[OpenVDBLink`OpenVDBSetProperty[vdb, "Name"->"Dino"];OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, "Dino"];vv[{"ActiveVoxelCount", "GridClass"}]]
	,
	{26073, "LevelSet"}	
]

VerificationTest[(* 19 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, bmr];OpenVDBLink`OpenVDBImport[filename, "Dino"]]
	,
	$Failed
	,
	{OpenVDBLink::error}
]

VerificationTest[(* 20 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, bmr];OpenVDBLink`OpenVDBImport[filename, Automatic, "Vector"]]
	,
	$Failed
	,
	{OpenVDBLink::error}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 21 *)
	temporaryFile=Function[body, WithCleanup[filename=FileNameJoin[{$TemporaryDirectory, StringJoin[CreateUUID[], ".vdb"]}], body, DeleteFile[filename]], HoldFirst];OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Double"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty],    OpenVDBLink`OpenVDBScalarGridQ[vdb],OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBExport"]

VerificationTest[(* 22 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

VerificationTest[(* 23 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, fog];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImport"]

VerificationTest[(* 24 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename];vv[{"ActiveVoxelCount", "GridClass"}]]
	,
	{26073, "LevelSet"}	
]

VerificationTest[(* 25 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, fog];vv=OpenVDBLink`OpenVDBImport[filename];vv[{"ActiveVoxelCount", "GridClass"}]]
	,
	{12052, "FogVolume"}	
]

VerificationTest[(* 26 *)
	temporaryFile[OpenVDBLink`OpenVDBSetProperty[vdb, "Name"->"Dino"];OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, "Dino"];vv[{"ActiveVoxelCount", "GridClass"}]]
	,
	{26073, "LevelSet"}	
]

VerificationTest[(* 27 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, "Dino_error"]]
	,
	$Failed
	,
	{OpenVDBLink::error}
]

VerificationTest[(* 28 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, Automatic, "Vector"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Integer"]

BeginTestSection["Initialization"]

VerificationTest[(* 29 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "UInt32"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBExport"]

VerificationTest[(* 30 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImport"]

VerificationTest[(* 31 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 32 *)
	temporaryFile[OpenVDBLink`OpenVDBSetProperty[vdb, "Name"->"Dino"];OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, "Dino"];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 33 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, None, "UInt32"];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 34 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, "Dino_error"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

VerificationTest[(* 35 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, None, "Int32"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Vector"]

BeginTestSection["Initialization"]

VerificationTest[(* 36 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Vec3S"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBExport"]

VerificationTest[(* 37 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImport"]

VerificationTest[(* 38 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 39 *)
	temporaryFile[OpenVDBLink`OpenVDBSetProperty[vdb, "Name"->"Dino"];OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, "Dino"];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 40 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, None, "Vec3S"];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 41 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, "Dino_error"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

VerificationTest[(* 42 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, None, "Int32"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 43 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBExport"]

VerificationTest[(* 44 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImport"]

VerificationTest[(* 45 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 46 *)
	temporaryFile[OpenVDBLink`OpenVDBSetProperty[vdb, "Name"->"Dino"];OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, "Dino"];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 47 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, None, "Boolean"];vv["ActiveVoxelCount"]]
	,
	20	
]

VerificationTest[(* 48 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, "Dino_error"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

VerificationTest[(* 49 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, None, "Vector"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 50 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBExport"]

VerificationTest[(* 51 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];{FileExistsQ[filename], FileByteCount[filename]>0}]
	,
	{True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBImport"]

VerificationTest[(* 52 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename];vv["ActiveVoxelCount"]]
	,
	12	
]

VerificationTest[(* 53 *)
	temporaryFile[OpenVDBLink`OpenVDBSetProperty[vdb, "Name"->"Dino"];OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, "Dino"];vv["ActiveVoxelCount"]]
	,
	12	
]

VerificationTest[(* 54 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];vv=OpenVDBLink`OpenVDBImport[filename, None, "Mask"];vv["ActiveVoxelCount"]]
	,
	12	
]

VerificationTest[(* 55 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, "Dino_error"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

VerificationTest[(* 56 *)
	temporaryFile[OpenVDBLink`OpenVDBExport[filename, vdb];OpenVDBLink`OpenVDBImport[filename, None, "Vector"]]
	,
	$Failed
	,
	{OpenVDBLink`OpenVDBLink::error}
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
