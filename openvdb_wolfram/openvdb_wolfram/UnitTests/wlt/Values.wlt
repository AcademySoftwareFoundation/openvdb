BeginTestSection["Value Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;vdb=OpenVDBLink`OpenVDBLevelSet[ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"]];OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBSetStates"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBSetStates]
	,
	"Index"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBSetStates]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBSetStates]
	,
	{}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBSetStates]
	,
	{"ArgumentsPattern"->{_, _, _}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBSetStates[], OpenVDBLink`OpenVDBSetStates["error"], OpenVDBLink`OpenVDBSetStates[vdb, "error"], OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, "error"], OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, 1, "error"], OpenVDBLink`OpenVDBSetStates[vdb, {{0, 0, 0}, {1, 1, 1}}, {1}]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBSetStates::argrx, OpenVDBSetStates::argr, OpenVDBSetStates::argrx, OpenVDBSetStates::argrx}
]

EndTestSection[]

BeginTestSection["OpenVDB*States"]

VerificationTest[(* 7 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBStates]
	,
	"Index"	
]

VerificationTest[(* 8 *)
	Attributes[OpenVDBLink`OpenVDBStates]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 9 *)
	Options[OpenVDBLink`OpenVDBStates]
	,
	{}	
]

VerificationTest[(* 10 *)
	SyntaxInformation[OpenVDBLink`OpenVDBStates]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 11 *)
	{OpenVDBLink`OpenVDBStates[], OpenVDBLink`OpenVDBStates["error"], OpenVDBLink`OpenVDBStates[vdb, "error"], OpenVDBLink`OpenVDBStates[vdb, {0, 0, 0}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBStates::argrx, OpenVDBStates::argr, OpenVDBStates::coord, OpenVDBStates::argrx}
]

EndTestSection[]

BeginTestSection["OpenVDBSetValues"]

VerificationTest[(* 12 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBSetValues]
	,
	"Index"	
]

VerificationTest[(* 13 *)
	Attributes[OpenVDBLink`OpenVDBSetValues]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 14 *)
	Options[OpenVDBLink`OpenVDBSetValues]
	,
	{}	
]

VerificationTest[(* 15 *)
	SyntaxInformation[OpenVDBLink`OpenVDBSetValues]
	,
	{"ArgumentsPattern"->{_, _, _}}	
]

VerificationTest[(* 16 *)
	{OpenVDBLink`OpenVDBSetValues[], OpenVDBLink`OpenVDBSetValues["error"], OpenVDBLink`OpenVDBSetValues[vdb, "error"], OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, "error"], OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, 1, "error"], OpenVDBLink`OpenVDBSetValues[vdb, {{0, 0, 0}, {1, 1, 1}}, {1}]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBSetValues::argrx, OpenVDBSetValues::argr, OpenVDBSetValues::argrx, OpenVDBSetValues::argrx}
]

EndTestSection[]

BeginTestSection["OpenVDB*Values"]

VerificationTest[(* 17 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBValues]
	,
	"Index"	
]

VerificationTest[(* 18 *)
	Attributes[OpenVDBLink`OpenVDBValues]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 19 *)
	Options[OpenVDBLink`OpenVDBValues]
	,
	{}	
]

VerificationTest[(* 20 *)
	SyntaxInformation[OpenVDBLink`OpenVDBValues]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 21 *)
	{OpenVDBLink`OpenVDBValues[], OpenVDBLink`OpenVDBValues["error"], OpenVDBLink`OpenVDBValues[vdb, "error"], OpenVDBLink`OpenVDBValues[vdb, {0, 0, 0}, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBValues::argrx, OpenVDBValues::argr, OpenVDBValues::coord, OpenVDBValues::argrx}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 22 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{MeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDB*States"]

VerificationTest[(* 23 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBStates[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 24 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{1, {1, 1, 1}}	
]

VerificationTest[(* 25 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0, 1, 1}], OpenVDBLink`OpenVDBStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0, 1, 1}, {0, 1, 1}}	
]

VerificationTest[(* 26 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 1}, {1, 1, 1}}	
]

VerificationTest[(* 27 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {6, 6, 6}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {6, 6, 6}]}
	,
	{$Failed, 1}	
]

EndTestSection[]

BeginTestSection["OpenVDB*Values"]

VerificationTest[(* 28 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBValues[vdb, {0, 0, 0}]}
	,
	{1., 1.}	
]

VerificationTest[(* 29 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 3], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{3., {3., 3., 3.}}	
]

VerificationTest[(* 30 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0.45, EulerGamma, 1.87}], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0.45, 0.5772156649015329, 1.87}, {0.44999998807907104, 0.5772156715393066, 1.8700000047683716}}	
]

VerificationTest[(* 31 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0., 1., 3.}, {3., 1., 3.}}	
]

VerificationTest[(* 32 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {6, 6, 6}, {0, 1, 3}], OpenVDBLink`OpenVDBValues[vdb, {6, 6, 6}]}
	,
	{$Failed, 3.}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 33 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{MeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDB*States"]

VerificationTest[(* 34 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBStates[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 35 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{1, {1, 1, 1}}	
]

VerificationTest[(* 36 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0, 1, 1}], OpenVDBLink`OpenVDBStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0, 1, 1}, {0, 1, 1}}	
]

VerificationTest[(* 37 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 1}, {1, 1, 1}}	
]

VerificationTest[(* 38 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {6, 6, 6}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {6, 6, 6}]}
	,
	{$Failed, 1}	
]

EndTestSection[]

BeginTestSection["OpenVDB*Values"]

VerificationTest[(* 39 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBValues[vdb, {0, 0, 0}]}
	,
	{1., 1.}	
]

VerificationTest[(* 40 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 3], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{3., {3., 3., 3.}}	
]

VerificationTest[(* 41 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0.45, EulerGamma, 1.87}], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0.45, 0.5772156649015329, 1.87}, {0.45, 0.5772156649015329, 1.87}}	
]

VerificationTest[(* 42 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0., 1., 3.}, {3., 1., 3.}}	
]

VerificationTest[(* 43 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {6, 6, 6}, {0, 1, 3}], OpenVDBLink`OpenVDBValues[vdb, {6, 6, 6}]}
	,
	{$Failed, 3.}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Integer"]

BeginTestSection["Initialization"]

VerificationTest[(* 44 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Int32"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDB*States"]

VerificationTest[(* 45 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBStates[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 46 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{1, {1, 1, 1}}	
]

VerificationTest[(* 47 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0, 1, 1}], OpenVDBLink`OpenVDBStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0, 1, 1}, {0, 1, 1}}	
]

VerificationTest[(* 48 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 1}, {1, 1, 1}}	
]

EndTestSection[]

BeginTestSection["OpenVDB*Values"]

VerificationTest[(* 49 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBValues[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 50 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 3], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{3, {3, 3, 3}}	
]

VerificationTest[(* 51 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0.45, EulerGamma, 1.87}], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{$Failed, {4, 5, 0}}	
]

VerificationTest[(* 52 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, N[{1, 2, 4}]], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{1., 2., 4.}, {1, 2, 4}}	
]

VerificationTest[(* 53 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 3}, {3, 1, 3}}	
]

VerificationTest[(* 54 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {6, 6, 6}, {0, 1, 3}], OpenVDBLink`OpenVDBValues[vdb, {6, 6, 6}]}
	,
	{$Failed, 3}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Vector"]

BeginTestSection["Initialization"]

VerificationTest[(* 55 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Vec3I"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDB*States"]

VerificationTest[(* 56 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBStates[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 57 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{1, {1, 1, 1}}	
]

VerificationTest[(* 58 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0, 1, 1}], OpenVDBLink`OpenVDBStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0, 1, 1}, {0, 1, 1}}	
]

VerificationTest[(* 59 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 1}, {1, 1, 1}}	
]

EndTestSection[]

BeginTestSection["OpenVDB*Values"]

VerificationTest[(* 60 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, {1, 1, 1}], OpenVDBLink`OpenVDBValues[vdb, {0, 0, 0}]}
	,
	{{1, 1, 1}, {1, 1, 1}}	
]

VerificationTest[(* 61 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, {4, 5, 3}], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{{4, 5, 3}, {{4, 5, 3}, {4, 5, 3}, {4, 5, 3}}}	
]

VerificationTest[(* 62 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, {{1, 2, 3}, {5, 6, 7}, {9, 8, 1}}], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{{{1, 2, 3}, {5, 6, 7}, {9, 8, 1}}, {{1, 2, 3}, {5, 6, 7}, {9, 8, 1}}}	
]

VerificationTest[(* 63 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0.45, EulerGamma, 1.87}], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{$Failed, {{8, 9, 4}, {7, 8, 5}, {0, 0, 0}}}	
]

VerificationTest[(* 64 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, N[{1, 2, 4}]], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{1., 2., 4.}, {{1, 2, 4}, {1, 2, 4}, {1, 2, 4}}}	
]

VerificationTest[(* 65 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 3}, {{0, 1, 3}, {0, 1, 3}, {0, 1, 3}}}	
]

VerificationTest[(* 66 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {6, 6, 6}, {0, 1, 3, 4}], OpenVDBLink`OpenVDBValues[vdb, {6, 6, 6}]}
	,
	{$Failed, {0, 1, 3}}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 67 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDB*States"]

VerificationTest[(* 68 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBStates[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 69 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{1, {1, 1, 1}}	
]

VerificationTest[(* 70 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0, 1, 1}], OpenVDBLink`OpenVDBStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0, 1, 1}, {0, 1, 1}}	
]

VerificationTest[(* 71 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 1}, {1, 1, 1}}	
]

EndTestSection[]

BeginTestSection["OpenVDB*Values"]

VerificationTest[(* 72 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBValues[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 73 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, {0, 1, 1}], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{{0, 1, 1}, {0, 1, 1}}	
]

VerificationTest[(* 74 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{1, {1, 1, 1}}	
]

VerificationTest[(* 75 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0.45, EulerGamma, 1.87}], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{$Failed, {1, 1, 0}}	
]

VerificationTest[(* 76 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, N[{1, 2, 4}]], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{1, 1, 1}, {1, 1, 1}}	
]

VerificationTest[(* 77 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {True, False, True}], OpenVDBLink`OpenVDBValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{1, 0, 1}, {1, 0, 1}}	
]

VerificationTest[(* 78 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {6, 6, 6}, {0, 1, 3, 4}], OpenVDBLink`OpenVDBValues[vdb, {6, 6, 6}]}
	,
	{$Failed, 1}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 79 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDB*States"]

VerificationTest[(* 80 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBStates[vdb, {0, 0, 0}]}
	,
	{1, 1}	
]

VerificationTest[(* 81 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBStates[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{1, {1, 1, 1}}	
]

VerificationTest[(* 82 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0, 1, 1}], OpenVDBLink`OpenVDBStates[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{{0, 1, 1}, {0, 1, 1}}	
]

VerificationTest[(* 83 *)
	{OpenVDBLink`OpenVDBSetStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {0, 1, 3}], OpenVDBLink`OpenVDBStates[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{{0, 1, 1}, {1, 1, 1}}	
]

EndTestSection[]

BeginTestSection["OpenVDB*Values"]

VerificationTest[(* 84 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {0, 0, 0}, 1], OpenVDBLink`OpenVDBValues[vdb, {0, 0, 0}]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBSetValues::nmsksupp, OpenVDBValues::nmsksupp}
]

VerificationTest[(* 85 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, {0, 1, 1}], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBSetValues::nmsksupp, OpenVDBValues::nmsksupp}
]

VerificationTest[(* 86 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}, 1], OpenVDBLink`OpenVDBValues[vdb, {{2, 2, 2}, {1, 1, 1}, {3, 3, 3}}]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBSetValues::nmsksupp, OpenVDBValues::nmsksupp}
]

VerificationTest[(* 87 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, {0.45, EulerGamma, 1.87}], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBSetValues::nmsksupp, OpenVDBValues::nmsksupp}
]

VerificationTest[(* 88 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}, N[{1, 2, 4}]], OpenVDBLink`OpenVDBValues[vdb, {{4, 4, 4}, {5, 5, 5}, {3, 2, 4}}]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBSetValues::nmsksupp, OpenVDBValues::nmsksupp}
]

VerificationTest[(* 89 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}, {True, False, True}], OpenVDBLink`OpenVDBValues[vdb, {{6, 6, 6}, {3, 5, 5}, {6, 6, 6}}]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBSetValues::nmsksupp, OpenVDBValues::nmsksupp}
]

VerificationTest[(* 90 *)
	{OpenVDBLink`OpenVDBSetValues[vdb, {6, 6, 6}, {0, 1, 3, 4}], OpenVDBLink`OpenVDBValues[vdb, {6, 6, 6}]}
	,
	{$Failed, $Failed}
	,
	{OpenVDBSetValues::nmsksupp, OpenVDBValues::nmsksupp}
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
