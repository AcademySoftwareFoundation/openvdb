BeginTestSection["Aggregate Data Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxelSliceTotals"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBActiveVoxelSliceTotals]
	,
	"Index"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBActiveVoxelSliceTotals]
	,
	{}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBActiveVoxelSliceTotals]
	,
	{}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBActiveVoxelSliceTotals]
	,
	{"ArgumentsPattern"->{_, _., _.}}	
]

VerificationTest[(* 6 *)
	{
OpenVDBLink`OpenVDBActiveVoxelSliceTotals[], 
OpenVDBLink`OpenVDBActiveVoxelSliceTotals["error"],
OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr, "error"],
OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr, {0, 3}->"error"],
OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr, {0, 3}, "error"],
OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr, {0, 3}, "Count", "error"]
}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

BeginTestSection["OpenVDBSlice"]

VerificationTest[(* 7 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBSlice]
	,
	"Index"	
]

VerificationTest[(* 8 *)
	Attributes[OpenVDBLink`OpenVDBSlice]
	,
	{}	
]

VerificationTest[(* 9 *)
	Options[OpenVDBLink`OpenVDBSlice]
	,
	{"MirrorSlice"->False}	
]

VerificationTest[(* 10 *)
	SyntaxInformation[OpenVDBLink`OpenVDBSlice]
	,
	{"ArgumentsPattern"->{_, _, _., OptionsPattern[]}}	
]

VerificationTest[(* 11 *)
	{
OpenVDBLink`OpenVDBSlice[], 
OpenVDBLink`OpenVDBSlice["error"],
OpenVDBLink`OpenVDBSlice[bmr, "error"],
OpenVDBLink`OpenVDBSlice[bmr, 3->"error"],
OpenVDBLink`OpenVDBSlice[bmr, 3, "error"],
OpenVDBLink`OpenVDBSlice[bmr, 3, Automatic, "error"]
}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

BeginTestSection["OpenVDBData"]

VerificationTest[(* 12 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBData]
	,
	"Index"	
]

VerificationTest[(* 13 *)
	Attributes[OpenVDBLink`OpenVDBData]
	,
	{}	
]

VerificationTest[(* 14 *)
	Options[OpenVDBLink`OpenVDBData]
	,
	{}	
]

VerificationTest[(* 15 *)
	SyntaxInformation[OpenVDBLink`OpenVDBData]
	,
	{"ArgumentsPattern"->{_, _.}}	
]

VerificationTest[(* 16 *)
	{
OpenVDBLink`OpenVDBData[], 
OpenVDBLink`OpenVDBData["error"],
OpenVDBLink`OpenVDBData[bmr, "error"],
OpenVDBLink`OpenVDBData[bmr, {{0, 1}, {0, 1}, {0, 1}}->"error"],
OpenVDBLink`OpenVDBData[bmr, Automatic, "error"]
}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveTiles"]

VerificationTest[(* 17 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBActiveTiles]
	,
	"Index"	
]

VerificationTest[(* 18 *)
	Attributes[OpenVDBLink`OpenVDBActiveTiles]
	,
	{}	
]

VerificationTest[(* 19 *)
	Options[OpenVDBLink`OpenVDBActiveTiles]
	,
	{"PartialOverlap"->True}	
]

VerificationTest[(* 20 *)
	SyntaxInformation[OpenVDBLink`OpenVDBActiveTiles]
	,
	{"ArgumentsPattern"->{_, _., OptionsPattern[]}}	
]

VerificationTest[(* 21 *)
	{
OpenVDBLink`OpenVDBActiveTiles[], 
OpenVDBLink`OpenVDBActiveTiles["error"],
OpenVDBLink`OpenVDBActiveTiles[bmr, "error"],
OpenVDBLink`OpenVDBActiveTiles[bmr, {{0, 1}, {0, 1}, {0, 1}}->"error"],
OpenVDBLink`OpenVDBActiveTiles[bmr, Automatic, "error"]
}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxels"]

VerificationTest[(* 22 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBActiveVoxels]
	,
	"Index"	
]

VerificationTest[(* 23 *)
	Attributes[OpenVDBLink`OpenVDBActiveVoxels]
	,
	{}	
]

VerificationTest[(* 24 *)
	Options[OpenVDBLink`OpenVDBActiveVoxels]
	,
	{}	
]

VerificationTest[(* 25 *)
	SyntaxInformation[OpenVDBLink`OpenVDBActiveVoxels]
	,
	{"ArgumentsPattern"->{_, _., _.}}	
]

VerificationTest[(* 26 *)
	{
OpenVDBLink`OpenVDBActiveVoxels[], 
OpenVDBLink`OpenVDBActiveVoxels["error"],
OpenVDBLink`OpenVDBActiveVoxels[bmr, "error"],
OpenVDBLink`OpenVDBActiveVoxels[bmr, {{0, 1}, {0, 1}, {0, 1}}->"error"],
OpenVDBLink`OpenVDBActiveVoxels[bmr, Automatic, "error"],
OpenVDBLink`OpenVDBActiveVoxels[bmr, Automatic, Automatic, "error"]
}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 27 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];fog=OpenVDBLink`OpenVDBFogVolume[OpenVDBLink`OpenVDBLevelSet[bmr, 0.065]];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxelSliceTotals"]

VerificationTest[(* 28 *)
	tots=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr];
Round[{Length[tots], Median[tots], Mean[tots]}, 0.001]
	,
	{41., 45.27, 45.129}	
]

VerificationTest[(* 29 *)
	tots=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr, Automatic, "Value"];
Round[{Length[tots], Median[tots], Mean[tots]}, 0.001]
	,
	{41., 45.27, 45.129}	
]

VerificationTest[(* 30 *)
	tots=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr, {0, 1}->"World"];
Round[{Length[tots], Median[tots], Mean[tots]}, 0.001]
	,
	{11., 31.168, 31.673000000000002}	
]

VerificationTest[(* 31 *)
	counts=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[bmr, {0, 1}->"World", "Count"]
	,
	{931, 951, 949, 970, 956, 944, 933, 918, 952, 959, 963}	
]

EndTestSection[]

BeginTestSection["OpenVDBSlice"]

VerificationTest[(* 32 *)
	slice=OpenVDBLink`OpenVDBSlice[bmr, 0];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {32, 84}, -0.30000001192092896, 0.30000001192092896, 416.4537309706211}	
]

VerificationTest[(* 33 *)
	slice=OpenVDBLink`OpenVDBSlice[bmr, 0.5->"World"];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {311, 831}, -0.30000001192092896, 0.30000001192092896, 77062.59436379373}	
]

VerificationTest[(* 34 *)
	slice=OpenVDBLink`OpenVDBSlice[bmr, 5->"Index", {{-40, 40}, {-40, 50}}];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {91, 81}, -0.30000001192092896, 0.30000001192092896, 1750.1216211095452}	
]

VerificationTest[(* 35 *)
	slice=OpenVDBLink`OpenVDBSlice[bmr, 5, {{-40, 40}, {-40, 50}}];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {91, 81}, -0.30000001192092896, 0.30000001192092896, 1750.1216211095452}	
]

EndTestSection[]

BeginTestSection["OpenVDBData"]

VerificationTest[(* 36 *)
	data=OpenVDBLink`OpenVDBData[bmr];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {41, 32, 84}, -0.30000001192092896, 0.30000001192092896, 24847.404193285853}	
]

VerificationTest[(* 37 *)
	data=OpenVDBLink`OpenVDBData[bmr, {{0.5, 1.0}, {-2.0, 1.0}, {-1.0, 0.0}}->"World"];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {11, 31, 6}, -0.30000001192092896, 0.30000001192092896, 167.6931715644896}	
]

VerificationTest[(* 38 *)
	data=OpenVDBLink`OpenVDBData[bmr, {{-40, 40}, {-40, 50}, {-10, 10}}->"Index"];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {21, 91, 81}, -0.30000001192092896, 0.30000001192092896, 39966.47893330082}	
]

VerificationTest[(* 39 *)
	data=OpenVDBLink`OpenVDBData[bmr, {{-40, 40}, {-40, 50}, {-10, 10}}];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {21, 91, 81}, -0.30000001192092896, 0.30000001192092896, 39966.47893330082}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveTiles"]

VerificationTest[(* 40 *)
	OpenVDBLink`OpenVDBActiveTiles[bmr]
	,
	{}	
]

VerificationTest[(* 41 *)
	OpenVDBLink`OpenVDBActiveTiles[fog]
	,
	{{{-24, -8, 0}, {-17, -1, 7}}, {{-8, -8, 8}, {-1, -1, 15}}, {{-24, 0, 0}, {-17, 7, 7}}, {{-16, 0, 8}, {-9, 7, 15}}, {{-8, 0, 8}, {-1, 7, 15}}, {{8, -8, 0}, {15, -1, 7}}, {{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 42 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{3, 40}, {0, 40}, {0, 40}}]
	,
	{{{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 43 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{0, 40}, {0, 40}, {0, 40}}, "PartialOverlap"->False]
	,
	{{{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 44 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{3, 40}, {0, 40}, {0, 40}}, "PartialOverlap"->False]
	,
	{{{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 45 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{3, 40}, {0, 40}, {0, 40}}->"Index"]
	,
	{{{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 46 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{-0.5, 1}, {0, 1}, {0, 1}}->"World"]
	,
	{{{-8, 0, 8}, {-1, 7, 15}}, {{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxels"]

VerificationTest[(* 47 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[bmr];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {84, 32, 41}, 26073, 0.0709662593281633, {47.66432708165535, 16.920684232731176, 21.702220688068117}}	
]

VerificationTest[(* 48 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[bmr, {{0, 40}, {0, 40}, {0, 40}}];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 4204, 0.053542955759930455, {16.422930542340627, 5.888201712654615, 9.418173168411037}}	
]

VerificationTest[(* 49 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[bmr, {{0, 40}, {0, 40}, {0, 40}}->"Index"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 4204, 0.053542955759930455, {16.422930542340627, 5.888201712654615, 9.418173168411037}}	
]

VerificationTest[(* 50 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[bmr, {{-0.5, 1.3}, {0, 1}, {-0.2, 0.7}}->"World"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {19, 11, 10}, 1210, -0.030239594995606044, {10.15206611570248, 8.00495867768595, 5.153719008264463}}	
]

VerificationTest[(* 51 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[bmr, Automatic, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {26073, 3}, {-1.3356729183446476, -0.07931576726882215, 1.7022206880681165}}	
]

VerificationTest[(* 52 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[bmr, {{0, 40}, {0, 40}, {0, 40}}, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {4204, 3}, {15.422930542340628, 4.888201712654615, 8.418173168411037}}	
]

VerificationTest[(* 53 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[bmr, {{0, 40}, {0, 40}, {0, 40}}, "Values"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, False, {4204}, 0.053542955759930455}	
]

VerificationTest[(* 54 *)
	OpenVDBLink`OpenVDBActiveVoxels[bmr, Automatic, "SparseArrayList"]
	,
	$Failed	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 55 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[OpenVDBLink`OpenVDBLevelSet[bmr, 0.065, "ScalarType"->"Double"]];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxelSliceTotals"]

VerificationTest[(* 56 *)
	tots=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb];
Round[{Length[tots], Median[tots], Mean[tots]}, 0.001]
	,
	{41., 45.27, 45.129}	
]

VerificationTest[(* 57 *)
	tots=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, Automatic, "Value"];
Round[{Length[tots], Median[tots], Mean[tots]}, 0.001]
	,
	{41., 45.27, 45.129}	
]

VerificationTest[(* 58 *)
	tots=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 1}->"World"];
Round[{Length[tots], Median[tots], Mean[tots]}, 0.001]
	,
	{11., 31.168, 31.673000000000002}	
]

VerificationTest[(* 59 *)
	counts=OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 1}->"World", "Count"]
	,
	{931, 951, 949, 970, 956, 944, 933, 918, 952, 959, 963}	
]

EndTestSection[]

BeginTestSection["OpenVDBSlice"]

VerificationTest[(* 60 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 0];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {32, 84}, -0.30000000000000004, 0.30000000000000004, 416.45371327429905}	
]

VerificationTest[(* 61 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 0.5->"World"];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {311, 831}, -0.30000000000000004, 0.30000000000000004, 77062.59130009596}	
]

VerificationTest[(* 62 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5->"Index", {{-40, 40}, {-40, 50}}];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {91, 81}, -0.30000000000000004, 0.30000000000000004, 1750.1215501539352}	
]

VerificationTest[(* 63 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5, {{-40, 40}, {-40, 50}}];
{Developer`PackedArrayQ[slice, Real], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {91, 81}, -0.30000000000000004, 0.30000000000000004, 1750.1215501539352}	
]

EndTestSection[]

BeginTestSection["OpenVDBData"]

VerificationTest[(* 64 *)
	data=OpenVDBLink`OpenVDBData[vdb];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {41, 32, 84}, -0.30000000000000004, 0.30000000000000004, 24847.40349374411}	
]

VerificationTest[(* 65 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0.5, 1.0}, {-2.0, 1.0}, {-1.0, 0.0}}->"World"];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {11, 31, 6}, -0.30000000000000004, 0.30000000000000004, 167.693155692996}	
]

VerificationTest[(* 66 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{-40, 40}, {-40, 50}, {-10, 10}}->"Index"];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {21, 91, 81}, -0.30000000000000004, 0.30000000000000004, 39966.477639134755}	
]

VerificationTest[(* 67 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{-40, 40}, {-40, 50}, {-10, 10}}];
{Developer`PackedArrayQ[data, Real], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {21, 91, 81}, -0.30000000000000004, 0.30000000000000004, 39966.477639134755}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveTiles"]

VerificationTest[(* 68 *)
	OpenVDBLink`OpenVDBActiveTiles[vdb]
	,
	{}	
]

VerificationTest[(* 69 *)
	OpenVDBLink`OpenVDBActiveTiles[fog]
	,
	{{{-24, -8, 0}, {-17, -1, 7}}, {{-8, -8, 8}, {-1, -1, 15}}, {{-24, 0, 0}, {-17, 7, 7}}, {{-16, 0, 8}, {-9, 7, 15}}, {{-8, 0, 8}, {-1, 7, 15}}, {{8, -8, 0}, {15, -1, 7}}, {{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 70 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{3, 40}, {0, 40}, {0, 40}}]
	,
	{{{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 71 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{0, 40}, {0, 40}, {0, 40}}, "PartialOverlap"->False]
	,
	{{{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 72 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{3, 40}, {0, 40}, {0, 40}}, "PartialOverlap"->False]
	,
	{{{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 73 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{3, 40}, {0, 40}, {0, 40}}->"Index"]
	,
	{{{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

VerificationTest[(* 74 *)
	OpenVDBLink`OpenVDBActiveTiles[fog, {{-0.5, 1}, {0, 1}, {0, 1}}->"World"]
	,
	{{{-8, 0, 8}, {-1, 7, 15}}, {{0, 0, 0}, {7, 7, 7}}, {{8, 0, 0}, {15, 7, 7}}}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxels"]

VerificationTest[(* 75 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {84, 32, 41}, 26073, 0.07096626754666166, {47.66432708165535, 16.920684232731176, 21.702220688068117}}	
]

VerificationTest[(* 76 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 4204, 0.053543058378818335, {16.422930542340627, 5.888201712654615, 9.418173168411037}}	
]

VerificationTest[(* 77 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}->"Index"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 4204, 0.053543058378818335, {16.422930542340627, 5.888201712654615, 9.418173168411037}}	
]

VerificationTest[(* 78 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{-0.5, 1.3}, {0, 1}, {-0.2, 0.7}}->"World"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {19, 11, 10}, 1210, -0.030239596325276886, {10.15206611570248, 8.00495867768595, 5.153719008264463}}	
]

VerificationTest[(* 79 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {26073, 3}, {-1.3356729183446476, -0.07931576726882215, 1.7022206880681165}}	
]

VerificationTest[(* 80 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {4204, 3}, {15.422930542340628, 4.888201712654615, 8.418173168411037}}	
]

VerificationTest[(* 81 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Values"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, False, {4204}, 0.053543058378818335}	
]

VerificationTest[(* 82 *)
	OpenVDBLink`OpenVDBActiveVoxels[bmr, Automatic, "SparseArrayList"]
	,
	$Failed	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Integer"]

BeginTestSection["Initialization"]

VerificationTest[(* 83 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Int32"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxelSliceTotals"]

VerificationTest[(* 84 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb]
	,
	{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}	
]

VerificationTest[(* 85 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, Automatic, "Value"]
	,
	{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}	
]

VerificationTest[(* 86 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 4}->"World"]
	,
	{0, 2, 4, 6, 8}	
]

VerificationTest[(* 87 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 4}->"World", "Count"]
	,
	{0, 2, 2, 2, 2}	
]

EndTestSection[]

BeginTestSection["OpenVDBSlice"]

VerificationTest[(* 88 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5];
{Developer`PackedArrayQ[slice, Integer], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {10, 10}, 0, 5, 10}	
]

VerificationTest[(* 89 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 4.5->"World"];
{Developer`PackedArrayQ[slice, Integer], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {10, 10}, 0, 4, 8}	
]

VerificationTest[(* 90 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5->"Index", {{0, 10}, {0, 8}}];
{Developer`PackedArrayQ[slice, Integer], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {9, 11}, 0, 5, 10}	
]

VerificationTest[(* 91 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5, {{0, 10}, {0, 8}}];
{Developer`PackedArrayQ[slice, Integer], MatrixQ[slice], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {9, 11}, 0, 5, 10}	
]

EndTestSection[]

BeginTestSection["OpenVDBData"]

VerificationTest[(* 92 *)
	data=OpenVDBLink`OpenVDBData[vdb];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {10, 10, 10}, 0, 10, 110}	
]

VerificationTest[(* 93 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0.5, 1.0}, {0, 3.0}, {0, 4.0}}->"World"];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {5, 4, 2}, 0, 1, 1}	
]

VerificationTest[(* 94 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0, 10}, {0, 8}, {0, 5}}->"Index"];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {6, 9, 11}, 0, 5, 27}	
]

VerificationTest[(* 95 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0, 10}, {0, 8}, {0, 5}}];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {6, 9, 11}, 0, 5, 27}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveTiles"]

VerificationTest[(* 96 *)
	OpenVDBLink`OpenVDBActiveTiles[vdb]
	,
	{}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxels"]

VerificationTest[(* 97 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {10, 10, 10}, 20, 5.5, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 98 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 20, 5.5, {6.5, 6.5, 6.5}}	
]

VerificationTest[(* 99 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}->"Index"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 20, 5.5, {6.5, 6.5, 6.5}}	
]

VerificationTest[(* 100 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{-0.5, 1.3}, {0, 1}, {-0.2, 0.7}}->"World"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {2, 2, 2}, 1, 1., {2., 2., 2.}}	
]

VerificationTest[(* 101 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {20, 3}, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 102 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {20, 3}, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 103 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Values"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, False, {20}, 5.5}	
]

VerificationTest[(* 104 *)
	OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "SparseArrayList"]
	,
	$Failed	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Vector"]

BeginTestSection["Initialization"]

VerificationTest[(* 105 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Vec2I"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[{12 - i, 13 - i}, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[{12 - i, 13 - i}, {i, 10}]];OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxelSliceTotals"]

VerificationTest[(* 106 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb]
	,
	{{22, 24}, {20, 22}, {18, 20}, {16, 18}, {14, 16}, {12, 14}, {10, 12}, {8, 10}, {6, 8}, {4, 6}}	
]

VerificationTest[(* 107 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, Automatic, "Value"]
	,
	{{22, 24}, {20, 22}, {18, 20}, {16, 18}, {14, 16}, {12, 14}, {10, 12}, {8, 10}, {6, 8}, {4, 6}}	
]

VerificationTest[(* 108 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 4}->"World"]
	,
	{{0, 0}, {22, 24}, {20, 22}, {18, 20}, {16, 18}}	
]

VerificationTest[(* 109 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 4}->"World", "Count"]
	,
	{0, 2, 2, 2, 2}	
]

EndTestSection[]

BeginTestSection["OpenVDBSlice"]

VerificationTest[(* 110 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {10, 10, 2}, 0, 8, {14, 16}}	
]

VerificationTest[(* 111 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 4.5->"World"];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {10, 10, 2}, 0, 9, {16, 18}}	
]

VerificationTest[(* 112 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5->"Index", {{0, 10}, {0, 8}}];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {9, 11, 2}, 0, 8, {14, 16}}	
]

VerificationTest[(* 113 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5, {{0, 10}, {0, 8}}];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, True, {9, 11, 2}, 0, 8, {14, 16}}	
]

EndTestSection[]

BeginTestSection["OpenVDBData"]

VerificationTest[(* 114 *)
	data=OpenVDBLink`OpenVDBData[vdb];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 4], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {10, 10, 10, 2}, 0, 12, {130, 150}}	
]

VerificationTest[(* 115 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0.5, 1.0}, {0, 3.0}, {0, 4.0}}->"World"];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 4], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {5, 4, 2, 2}, 0, 12, {11, 12}}	
]

VerificationTest[(* 116 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0, 10}, {0, 8}, {0, 5}}->"Index"];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 4], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {6, 9, 11, 2}, 0, 12, {69, 77}}	
]

VerificationTest[(* 117 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0, 10}, {0, 8}, {0, 5}}];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 4], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {6, 9, 11, 2}, 0, 12, {69, 77}}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveTiles"]

VerificationTest[(* 118 *)
	OpenVDBLink`OpenVDBActiveTiles[vdb]
	,
	{}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxels"]

VerificationTest[(* 119 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb];
{Head[#], ArrayQ[#, 3], Dimensions[#], Length[#["NonzeroValues"]], N[Mean[#["NonzeroValues"]]], N[Mean[#["NonzeroPositions"]]]}&/@vox
	,
	{{SparseArray, True, {10, 10, 10}, 20, 6.5, {5.5, 5.5, 5.5}}, {SparseArray, True, {10, 10, 10}, 20, 7.5, {5.5, 5.5, 5.5}}}	
]

VerificationTest[(* 120 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}];
{Head[#], ArrayQ[#, 3], Dimensions[#], Length[#["NonzeroValues"]], N[Mean[#["NonzeroValues"]]], N[Mean[#["NonzeroPositions"]]]}&/@vox
	,
	{{SparseArray, True, {41, 41, 41}, 20, 6.5, {6.5, 6.5, 6.5}}, {SparseArray, True, {41, 41, 41}, 20, 7.5, {6.5, 6.5, 6.5}}}	
]

VerificationTest[(* 121 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}->"Index"];
{Head[#], ArrayQ[#, 3], Dimensions[#], Length[#["NonzeroValues"]], N[Mean[#["NonzeroValues"]]], N[Mean[#["NonzeroPositions"]]]}&/@vox
	,
	{{SparseArray, True, {41, 41, 41}, 20, 6.5, {6.5, 6.5, 6.5}}, {SparseArray, True, {41, 41, 41}, 20, 7.5, {6.5, 6.5, 6.5}}}	
]

VerificationTest[(* 122 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{-0.5, 1.3}, {0, 1}, {-0.2, 0.7}}->"World"];
{Head[#], ArrayQ[#, 3], Dimensions[#], Length[#["NonzeroValues"]], N[Mean[#["NonzeroValues"]]], N[Mean[#["NonzeroPositions"]]]}&/@vox
	,
	{{SparseArray, True, {2, 2, 2}, 1, 11., {2., 2., 2.}}, {SparseArray, True, {2, 2, 2}, 1, 12., {2., 2., 2.}}}	
]

VerificationTest[(* 123 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {20, 3}, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 124 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {20, 3}, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 125 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Values"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {20, 2}, {6.5, 7.5}}	
]

VerificationTest[(* 126 *)
	OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "SparseArray"]
	,
	$Failed	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 127 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxelSliceTotals"]

VerificationTest[(* 128 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb]
	,
	$Failed	
]

VerificationTest[(* 129 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, Automatic, "Value"]
	,
	$Failed	
]

VerificationTest[(* 130 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 4}->"World", "Count"]
	,
	{0, 2, 2, 2, 2}	
]

EndTestSection[]

BeginTestSection["OpenVDBSlice"]

VerificationTest[(* 131 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, False, {10, 10}, 0, 1, 1}	
]

VerificationTest[(* 132 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 4.5->"World"];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, False, {10, 10}, 0, 1, 2}	
]

VerificationTest[(* 133 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5->"Index", {{0, 10}, {0, 8}}];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, False, {9, 11}, 0, 1, 1}	
]

VerificationTest[(* 134 *)
	slice=OpenVDBLink`OpenVDBSlice[vdb, 5, {{0, 10}, {0, 8}}];
{Developer`PackedArrayQ[slice, Integer], ArrayQ[slice, 3], Dimensions[slice], Min[slice], Max[slice], Total[slice, 2]}
	,
	{True, False, {9, 11}, 0, 1, 1}	
]

EndTestSection[]

BeginTestSection["OpenVDBData"]

VerificationTest[(* 135 *)
	data=OpenVDBLink`OpenVDBData[vdb];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {10, 10, 10}, 0, 1, 12}	
]

VerificationTest[(* 136 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0.5, 1.0}, {0, 3.0}, {0, 4.0}}->"World"];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {5, 4, 2}, 0, 1, 1}	
]

VerificationTest[(* 137 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0, 10}, {0, 8}, {0, 5}}->"Index"];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {6, 9, 11}, 0, 1, 5}	
]

VerificationTest[(* 138 *)
	data=OpenVDBLink`OpenVDBData[vdb, {{0, 10}, {0, 8}, {0, 5}}];
{Developer`PackedArrayQ[data, Integer], ArrayQ[data, 3], Dimensions[data], Min[data], Max[data], Total[data, 3]}
	,
	{True, True, {6, 9, 11}, 0, 1, 5}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveTiles"]

VerificationTest[(* 139 *)
	OpenVDBLink`OpenVDBActiveTiles[vdb]
	,
	{}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxels"]

VerificationTest[(* 140 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {10, 10, 10}, 20, 0.6, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 141 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 20, 0.6, {6.5, 6.5, 6.5}}	
]

VerificationTest[(* 142 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}->"Index"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 20, 0.6, {6.5, 6.5, 6.5}}	
]

VerificationTest[(* 143 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{-0.5, 1.3}, {0, 1}, {-0.2, 0.7}}->"World"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], N[Mean[vox["NonzeroValues"]]], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {2, 2, 2}, 1, 1., {2., 2., 2.}}	
]

VerificationTest[(* 144 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {20, 3}, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 145 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {20, 3}, {5.5, 5.5, 5.5}}	
]

VerificationTest[(* 146 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Values"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, False, {20}, 0.6}	
]

VerificationTest[(* 147 *)
	OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "SparseArrayList"]
	,
	$Failed	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 148 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxelSliceTotals"]

VerificationTest[(* 149 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb]
	,
	$Failed	
]

VerificationTest[(* 150 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, Automatic, "Value"]
	,
	$Failed	
]

VerificationTest[(* 151 *)
	OpenVDBLink`OpenVDBActiveVoxelSliceTotals[vdb, {0, 4}->"World", "Count"]
	,
	$Failed	
]

EndTestSection[]

BeginTestSection["OpenVDBSlice"]

VerificationTest[(* 152 *)
	OpenVDBLink`OpenVDBSlice[vdb, 5]
	,
	$Failed	
]

EndTestSection[]

BeginTestSection["OpenVDBData"]

VerificationTest[(* 153 *)
	OpenVDBLink`OpenVDBData[vdb]
	,
	$Failed	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveTiles"]

VerificationTest[(* 154 *)
	OpenVDBLink`OpenVDBActiveTiles[vdb]
	,
	{}	
]

EndTestSection[]

BeginTestSection["OpenVDBActiveVoxels"]

VerificationTest[(* 155 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], vox["NonzeroValues"], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {10, 10, 10}, 0, Pattern, {5.583333333333333, 5.583333333333333, 5.166666666666667}}	
]

VerificationTest[(* 156 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], vox["NonzeroValues"], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 0, Pattern, {6.583333333333333, 6.583333333333333, 6.166666666666667}}	
]

VerificationTest[(* 157 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}->"Index"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], vox["NonzeroValues"], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {41, 41, 41}, 0, Pattern, {6.583333333333333, 6.583333333333333, 6.166666666666667}}	
]

VerificationTest[(* 158 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{-0.5, 1.3}, {0, 1}, {-0.2, 0.7}}->"World"];
{Head[vox], ArrayQ[vox, 3], Dimensions[vox], Length[vox["NonzeroValues"]], vox["NonzeroValues"], N[Mean[vox["NonzeroPositions"]]]}
	,
	{SparseArray, True, {2, 2, 2}, 0, Pattern, {2., 2., 2.}}	
]

VerificationTest[(* 159 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {12, 3}, {5.583333333333333, 5.583333333333333, 5.166666666666667}}	
]

VerificationTest[(* 160 *)
	vox=OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Positions"];
{Head[vox], ArrayQ[vox, 2], Dimensions[vox], N[Mean[vox]]}
	,
	{List, True, {12, 3}, {5.583333333333333, 5.583333333333333, 5.166666666666667}}	
]

VerificationTest[(* 161 *)
	OpenVDBLink`OpenVDBActiveVoxels[vdb, {{0, 40}, {0, 40}, {0, 40}}, "Values"]
	,
	$Failed	
]

VerificationTest[(* 162 *)
	OpenVDBLink`OpenVDBActiveVoxels[vdb, Automatic, "SparseArrayList"]
	,
	$Failed	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
