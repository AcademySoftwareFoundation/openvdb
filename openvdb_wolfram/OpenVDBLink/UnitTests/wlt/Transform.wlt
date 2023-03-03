BeginTestSection["Transform Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;vdb=OpenVDBLink`OpenVDBLevelSet[ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"]];OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBTransform"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBTransform]
	,
	"World"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBTransform]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBTransform]
	,
	{Resampling->Automatic}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBTransform]
	,
	{"ArgumentsPattern"->{_, _, OptionsPattern[]}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBTransform[], OpenVDBLink`OpenVDBTransform["error"], OpenVDBLink`OpenVDBTransform[vdb, "error"], OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[{1, 1, 2}], "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBTransform::argrx, OpenVDBTransform::argr, OpenVDBTransform::trans, OpenVDBTransform::nonopt}
]

VerificationTest[(* 7 *)
	{vdb["ActiveVoxelCount"], OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[{1, 1, 2}]]["ActiveVoxelCount"], vdb["ActiveVoxelCount"]}
	,
	{26073, 65292, 26073}	
]

EndTestSection[]

BeginTestSection["OpenVDBMultiply"]

VerificationTest[(* 8 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBMultiply]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 9 *)
	Attributes[OpenVDBLink`OpenVDBMultiply]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 10 *)
	Options[OpenVDBLink`OpenVDBMultiply]
	,
	{}	
]

VerificationTest[(* 11 *)
	SyntaxInformation[OpenVDBLink`OpenVDBMultiply]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 12 *)
	{OpenVDBLink`OpenVDBMultiply[], OpenVDBLink`OpenVDBMultiply["error"], OpenVDBLink`OpenVDBMultiply[vdb, "error"], OpenVDBLink`OpenVDBMultiply[vdb, 3, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBMultiply::argrx, OpenVDBMultiply::argr, OpenVDBMultiply::real, OpenVDBMultiply::argrx}
]

VerificationTest[(* 13 *)
	(OpenVDBLink`OpenVDBMultiply[OpenVDBLink`OpenVDBCreateGrid[1., #1], 3]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBMultiply::scalargrid2, OpenVDBMultiply::scalargrid2, OpenVDBMultiply::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBGammaAdjust"]

VerificationTest[(* 14 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBGammaAdjust]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 15 *)
	Attributes[OpenVDBLink`OpenVDBGammaAdjust]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 16 *)
	Options[OpenVDBLink`OpenVDBGammaAdjust]
	,
	{}	
]

VerificationTest[(* 17 *)
	SyntaxInformation[OpenVDBLink`OpenVDBGammaAdjust]
	,
	{"ArgumentsPattern"->{_, _}}	
]

VerificationTest[(* 18 *)
	{OpenVDBLink`OpenVDBGammaAdjust[], OpenVDBLink`OpenVDBGammaAdjust["error"], OpenVDBLink`OpenVDBGammaAdjust[vdb, "error"], OpenVDBLink`OpenVDBGammaAdjust[vdb, 3, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBGammaAdjust::argrx, OpenVDBGammaAdjust::argr, OpenVDBGammaAdjust::pos, OpenVDBGammaAdjust::argrx}
]

VerificationTest[(* 19 *)
	(OpenVDBLink`OpenVDBGammaAdjust[OpenVDBLink`OpenVDBCreateGrid[1., #1], 2]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBGammaAdjust::scalargrid2, OpenVDBGammaAdjust::scalargrid2, OpenVDBGammaAdjust::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 20 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{MeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBTransform"]

VerificationTest[(* 21 *)
	OpenVDBLink`OpenVDBTransform[bmr, ScalingTransform[1.125*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	46455	
]

VerificationTest[(* 22 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[1.125*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	46455	
]

VerificationTest[(* 23 *)
	OpenVDBLink`OpenVDBTransform[fog, RotationTransform[Pi/3, {1, 2, 3}]]["ActiveVoxelCount"]
	,
	15325	
]

VerificationTest[(* 24 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	4113	
]

VerificationTest[(* 25 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Nearest"]["ActiveVoxelCount"]
	,
	3254	
]

VerificationTest[(* 26 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Quadratic"]["ActiveVoxelCount"]
	,
	4992	
]

EndTestSection[]

BeginTestSection["OpenVDBMultiply"]

VerificationTest[(* 27 *)
	OpenVDBLink`OpenVDBMultiply[bmr, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0.30000001192092896, {-0.5999919772148132, 0.599901020526886}}	
]

VerificationTest[(* 28 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBMultiply[vdb2, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0.30000001192092896, {-0.5999919772148132, 0.599901020526886}}	
]

VerificationTest[(* 29 *)
	fog2=OpenVDBLink`OpenVDBCopyGrid[fog];OpenVDBLink`OpenVDBMultiply[fog2, 3][{"BackgroundValue", "MaxValue"}]
	,
	{0., 1.}	
]

EndTestSection[]

BeginTestSection["OpenVDBGammaAdjust"]

VerificationTest[(* 30 *)
	OpenVDBLink`OpenVDBGammaAdjust[bmr, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0., {1.0181959275712416*^-8, 1.}}	
]

VerificationTest[(* 31 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBGammaAdjust[vdb2, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0., {1.0181959275712416*^-8, 1.}}	
]

VerificationTest[(* 32 *)
	fog2=OpenVDBLink`OpenVDBCopyGrid[fog];OpenVDBLink`OpenVDBGammaAdjust[fog2, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0., {1.0181959275712416*^-8, 1.}}	
]

VerificationTest[(* 33 *)
	OpenVDBLink`OpenVDBGammaAdjust[fog, 0]
	,
	$Failed
	,
	{OpenVDBGammaAdjust::pos}
]

VerificationTest[(* 34 *)
	OpenVDBLink`OpenVDBGammaAdjust[fog, -1]
	,
	$Failed
	,
	{OpenVDBGammaAdjust::pos}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 35 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{MeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBTransform"]

VerificationTest[(* 36 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[1.125*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	46455	
]

VerificationTest[(* 37 *)
	OpenVDBLink`OpenVDBTransform[fog, RotationTransform[Pi/3, {1, 2, 3}]]["ActiveVoxelCount"]
	,
	15325	
]

VerificationTest[(* 38 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	4113	
]

VerificationTest[(* 39 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Nearest"]["ActiveVoxelCount"]
	,
	3254	
]

VerificationTest[(* 40 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Quadratic"]["ActiveVoxelCount"]
	,
	4992	
]

EndTestSection[]

BeginTestSection["OpenVDBMultiply"]

VerificationTest[(* 41 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBMultiply[vdb2, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0.30000000000000004, {-0.5999919825045569, 0.5999010164203733}}	
]

VerificationTest[(* 42 *)
	fog2=OpenVDBLink`OpenVDBCopyGrid[fog];OpenVDBLink`OpenVDBMultiply[fog2, 3][{"BackgroundValue", "MaxValue"}]
	,
	{0., 1.}	
]

EndTestSection[]

BeginTestSection["OpenVDBGammaAdjust"]

VerificationTest[(* 43 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBGammaAdjust[vdb2, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0., {1.0183166161797685*^-8, 1.}}	
]

VerificationTest[(* 44 *)
	fog2=OpenVDBLink`OpenVDBCopyGrid[fog];OpenVDBLink`OpenVDBGammaAdjust[fog2, 2][{"BackgroundValue", "MinMaxValues"}]
	,
	{0., {1.0183166161797685*^-8, 1.}}	
]

VerificationTest[(* 45 *)
	OpenVDBLink`OpenVDBGammaAdjust[fog, 0]
	,
	$Failed
	,
	{OpenVDBGammaAdjust::pos}
]

VerificationTest[(* 46 *)
	OpenVDBLink`OpenVDBGammaAdjust[fog, -1]
	,
	$Failed
	,
	{OpenVDBGammaAdjust::pos}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Integer"]

BeginTestSection["Initialization"]

VerificationTest[(* 47 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Int64"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBTransform"]

VerificationTest[(* 48 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[1.125*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	221	
]

VerificationTest[(* 49 *)
	OpenVDBLink`OpenVDBTransform[vdb, RotationTransform[Pi/3, {1, 2, 3}]]["ActiveVoxelCount"]
	,
	131	
]

VerificationTest[(* 50 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Nearest"]["ActiveVoxelCount"]
	,
	5	
]

VerificationTest[(* 51 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Quadratic"]["ActiveVoxelCount"]
	,
	52	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Vector"]

BeginTestSection["Initialization"]

VerificationTest[(* 52 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Vec2S"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[{12 - i, 13 - i}, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[{12 - i, 13 - i}, {i, 10}]];OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBTransform"]

VerificationTest[(* 53 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[1.125*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	221	
]

VerificationTest[(* 54 *)
	OpenVDBLink`OpenVDBTransform[vdb, RotationTransform[Pi/3, {1, 2, 3}]]["ActiveVoxelCount"]
	,
	131	
]

VerificationTest[(* 55 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Nearest"]["ActiveVoxelCount"]
	,
	5	
]

VerificationTest[(* 56 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Quadratic"]["ActiveVoxelCount"]
	,
	52	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 57 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBTransform"]

VerificationTest[(* 58 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[1.125*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	221	
]

VerificationTest[(* 59 *)
	OpenVDBLink`OpenVDBTransform[vdb, RotationTransform[Pi/3, {1, 2, 3}]]["ActiveVoxelCount"]
	,
	131	
]

VerificationTest[(* 60 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Nearest"]["ActiveVoxelCount"]
	,
	5	
]

VerificationTest[(* 61 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Quadratic"]["ActiveVoxelCount"]
	,
	52	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 62 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBTransform"]

VerificationTest[(* 63 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[1.125*{1, 1, 1}]]["ActiveVoxelCount"]
	,
	120	
]

VerificationTest[(* 64 *)
	OpenVDBLink`OpenVDBTransform[vdb, RotationTransform[Pi/3, {1, 2, 3}]]["ActiveVoxelCount"]
	,
	91	
]

VerificationTest[(* 65 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Nearest"]["ActiveVoxelCount"]
	,
	4	
]

VerificationTest[(* 66 *)
	OpenVDBLink`OpenVDBTransform[vdb, ScalingTransform[0.5*{1, 1, 1}], Resampling->"Quadratic"]["ActiveVoxelCount"]
	,
	4	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
