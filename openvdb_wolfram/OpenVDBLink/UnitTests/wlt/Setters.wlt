BeginTestSection["Setters Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;vdb=OpenVDBLink`OpenVDBLevelSet[ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"]];OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBSetProperty"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBSetProperty]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBSetProperty]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBSetProperty]
	,
	{}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBSetProperty]
	,
	{"ArgumentsPattern"->{_, _, _.}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBSetProperty[], OpenVDBLink`OpenVDBSetProperty["error"], OpenVDBLink`OpenVDBSetProperty[vdb, "error"], OpenVDBLink`OpenVDBSetProperty[vdb, "error", "error"], OpenVDBLink`OpenVDBSetProperty[vdb, "Name", "Dino", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBSetProperty::argt, OpenVDBSetProperty::argtu, OpenVDBSetProperty::spec, OpenVDBSetProperty::prop, OpenVDBSetProperty::argt}
]

VerificationTest[(* 7 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Name"->"Dino"], vdb["Name"]}
	,
	{"Dino", "Dino"}	
]

VerificationTest[(* 8 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Name", "Dino2"], vdb["Name"]}
	,
	{"Dino2", "Dino2"}	
]

VerificationTest[(* 9 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, {"Name", "Creator"}, {"Dino3", "Riemann"}], vdb[{"Name", "Creator"}]}
	,
	{{"Dino3", "Riemann"}, {"Dino3", "Riemann"}}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 10 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{MeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBSetProperty"]

VerificationTest[(* 11 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Description"->"hello!!"], vdb["Description"]}
	,
	{"hello!!", "hello!!"}	
]

VerificationTest[(* 12 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, {"BackgroundValue"->2, "Name"->"Dino", "VoxelSize"->3}], vdb[{"BackgroundValue", "Name", "VoxelSize"}]}
	,
	{{2., "Dino", 3.}, {2., "Dino", 3.}}	
]

VerificationTest[(* 13 *)
	{OpenVDBLink`OpenVDBSetProperty[fog, {"Creator"->"test", "Description"->"hello2!!"}], fog[{"Creator", "Description"}]}
	,
	{{"test", "hello2!!"}, {"test", "hello2!!"}}	
]

VerificationTest[(* 14 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "BackgroundValue"->{3}], vdb["BackgroundValue"]}
	,
	{$Failed, 2.}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 15 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{MeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBSetProperty"]

VerificationTest[(* 16 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Description"->"hello!!"], vdb["Description"]}
	,
	{"hello!!", "hello!!"}	
]

VerificationTest[(* 17 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, {"BackgroundValue"->2, "Name"->"Dino", "VoxelSize"->3}], vdb[{"BackgroundValue", "Name", "VoxelSize"}]}
	,
	{{2., "Dino", 3.}, {2., "Dino", 3.}}	
]

VerificationTest[(* 18 *)
	{OpenVDBLink`OpenVDBSetProperty[fog, {"Creator"->"test", "Description"->"hello2!!"}], fog[{"Creator", "Description"}]}
	,
	{{"test", "hello2!!"}, {"test", "hello2!!"}}	
]

VerificationTest[(* 19 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "BackgroundValue"->{3}], vdb["BackgroundValue"]}
	,
	{$Failed, 2.}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Integer"]

BeginTestSection["Initialization"]

VerificationTest[(* 20 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "UInt32"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBSetProperty"]

VerificationTest[(* 21 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Description"->"hello!!"], vdb["Description"]}
	,
	{"hello!!", "hello!!"}	
]

VerificationTest[(* 22 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, {"BackgroundValue"->2, "Name"->"Dino", "VoxelSize"->3}], vdb[{"BackgroundValue", "Name", "VoxelSize"}]}
	,
	{{2, "Dino", 3.}, {2, "Dino", 3.}}	
]

VerificationTest[(* 23 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "BackgroundValue"->{3}], vdb["BackgroundValue"]}
	,
	{$Failed, 2}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Vector"]

BeginTestSection["Initialization"]

VerificationTest[(* 24 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Vec3D"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBSetProperty"]

VerificationTest[(* 25 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Description"->"hello!!"], vdb["Description"]}
	,
	{"hello!!", "hello!!"}	
]

VerificationTest[(* 26 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, {"BackgroundValue"->{4, 2, 5}, "Name"->"Dino", "VoxelSize"->3}], vdb[{"BackgroundValue", "Name", "VoxelSize"}]}
	,
	{{{4., 2., 5.}, "Dino", 3.}, {{4., 2., 5.}, "Dino", 3.}}	
]

VerificationTest[(* 27 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "BackgroundValue"->{3}], vdb["BackgroundValue"]}
	,
	{$Failed, {4., 2., 5.}}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 28 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBSetProperty"]

VerificationTest[(* 29 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Description"->"hello!!"], vdb["Description"]}
	,
	{"hello!!", "hello!!"}	
]

VerificationTest[(* 30 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, {"BackgroundValue"->True, "Name"->"Dino", "VoxelSize"->3}], vdb[{"BackgroundValue", "Name", "VoxelSize"}]}
	,
	{{1, "Dino", 3.}, {1, "Dino", 3.}}	
]

VerificationTest[(* 31 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "BackgroundValue"->{3}], vdb["BackgroundValue"]}
	,
	{$Failed, 1}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 32 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBSetProperty"]

VerificationTest[(* 33 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, "Description"->"hello!!"], vdb["Description"]}
	,
	{"hello!!", "hello!!"}	
]

VerificationTest[(* 34 *)
	{OpenVDBLink`OpenVDBSetProperty[vdb, {"BackgroundValue"->1, "Name"->"Dino", "VoxelSize"->3}], vdb[{"BackgroundValue", "Name", "VoxelSize"}]}
	,
	{{$Failed, "Dino", 3.}, {Missing["NotApplicable"], "Dino", 3.}}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
