BeginTestSection["Filter Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBFilter"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBFilter]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBFogVolume]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBFilter]
	,
	{}	
	,
	{}
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBFilter]
	,
	{"ArgumentsPattern"->{_, _, _.}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBFilter[], OpenVDBLink`OpenVDBFilter["error"], OpenVDBLink`OpenVDBFilter[bmr], OpenVDBLink`OpenVDBFilter[bmr, "error"], OpenVDBLink`OpenVDBFilter[bmr, {"Laplacian", 3}], OpenVDBLink`OpenVDBFilter[bmr, "Mean", "error"], OpenVDBLink`OpenVDBFilter[bmr, "Mean", -3]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBFilter::argt, OpenVDBFilter::argtu, OpenVDBFilter::argtu, OpenVDBFilter::filter, OpenVDBFilter::filter, OpenVDBFilter::intpm, OpenVDBFilter::intpm}
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBFilter[OpenVDBLink`OpenVDBCreateGrid[1., #1], "Mean"]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBFilter::scalargrid2, OpenVDBFilter::scalargrid2, OpenVDBFilter::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 8 *)
	$propertyList={"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox",     "IndexDimensions","MinMaxValues","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"};OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];MeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBFilter"]

VerificationTest[(* 9 *)
	OpenVDBLink`OpenVDBFilter[OpenVDBLink`OpenVDBFogVolume[bmr], "Mean"]
	,
	$Failed
	,
	{OpenVDBFilter::lvlsetgrid2}
]

VerificationTest[(* 10 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Mean"][$propertyList]
	,
	{25829, 0, 25829, 0.30000001192092896, 104160, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-15, 15}, {-19, 20}}, {84, 31, 40}, {-0.299991250038147, 0.2999999523162842}, True, 0.1, {{-4.800000000000001, 3.5}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.4, 3.1, 4.}}	
]

VerificationTest[(* 11 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Gaussian"][$propertyList]
	,
	{22492, 0, 22492, 0.30000001192092896, 84854, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-43, 33}, {-14, 14}, {-18, 19}}, {77, 29, 38}, {-0.29984864592552185, 0.29998090863227844}, True, 0.1, {{-4.3, 3.3000000000000003}, {-1.4000000000000001, 1.4000000000000001}, {-1.8, 1.9000000000000001}}, {7.7, 2.9000000000000004, 3.8000000000000003}}	
]

VerificationTest[(* 12 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Laplacian"][$propertyList]
	,
	{25744, 0, 25744, 0.30000001192092896, 104160, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-15, 15}, {-19, 20}}, {84, 31, 40}, {-0.2999761700630188, 0.29999420046806335}, True, 0.1, {{-4.800000000000001, 3.5}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.4, 3.1, 4.}}	
]

VerificationTest[(* 13 *)
	OpenVDBLink`OpenVDBFilter[bmr, "MeanCurvature"][$propertyList]
	,
	{25639, 0, 25639, 0.30000001192092896, 104160, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-15, 15}, {-19, 20}}, {84, 31, 40}, {-0.2999931871891022, 0.29997771978378296}, True, 0.1, {{-4.800000000000001, 3.5}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.4, 3.1, 4.}}	
]

VerificationTest[(* 14 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Median"][$propertyList]
	,
	{25458, 0, 25458, 0.30000001192092896, 104160, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-15, 15}, {-19, 20}}, {84, 31, 40}, {-0.29993146657943726, 0.29999688267707825}, True, 0.1, {{-4.800000000000001, 3.5}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.4, 3.1, 4.}}	
]

VerificationTest[(* 15 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Mean", 2][$propertyList]
	,
	{24918, 0, 24918, 0.30000001192092896, 101680, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-47, 34}, {-15, 15}, {-19, 20}}, {82, 31, 40}, {-0.29998451471328735, 0.29999208450317383}, True, 0.1, {{-4.7, 3.4000000000000004}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.200000000000001, 3.1, 4.}}	
]

VerificationTest[(* 16 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Gaussian", 2][$propertyList]
	,
	{18840, 0, 18840, 0.30000001192092896, 58625, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-35, 31}, {-12, 12}, {-15, 19}}, {67, 25, 35}, {-0.2999624013900757, 0.2999921441078186}, True, 0.1, {{-3.5, 3.1}, {-1.2000000000000002, 1.2000000000000002}, {-1.5, 1.9000000000000001}}, {6.7, 2.5, 3.5}}	
]

VerificationTest[(* 17 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Laplacian", 2][$propertyList]
	,
	{25256, 0, 25256, 0.30000001192092896, 104160, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-15, 15}, {-19, 20}}, {84, 31, 40}, {-0.2999767065048218, 0.2999972105026245}, True, 0.1, {{-4.800000000000001, 3.5}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.4, 3.1, 4.}}	
]

VerificationTest[(* 18 *)
	OpenVDBLink`OpenVDBFilter[bmr, "MeanCurvature", 2][$propertyList]
	,
	{25263, 0, 25263, 0.30000001192092896, 104160, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-15, 15}, {-19, 20}}, {84, 31, 40}, {-0.29997536540031433, 0.29999375343322754}, True, 0.1, {{-4.800000000000001, 3.5}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.4, 3.1, 4.}}	
]

VerificationTest[(* 19 *)
	OpenVDBLink`OpenVDBFilter[bmr, "Median", 2][$propertyList]
	,
	{24664, 0, 24664, 0.30000001192092896, 101680, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-47, 34}, {-15, 15}, {-19, 20}}, {82, 31, 40}, {-0.2999856770038605, 0.29998695850372314}, True, 0.1, {{-4.7, 3.4000000000000004}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.200000000000001, 3.1, 4.}}	
]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBFilter[bmr, {"Mean", 2}][$propertyList]
	,
	{26786, 0, 26786, 0.30000001192092896, 101680, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-47, 34}, {-15, 15}, {-19, 20}}, {82, 31, 40}, {-0.29997169971466064, 0.29999980330467224}, True, 0.1, {{-4.7, 3.4000000000000004}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.200000000000001, 3.1, 4.}}	
]

VerificationTest[(* 21 *)
	OpenVDBLink`OpenVDBFilter[bmr, {"Gaussian", 2}][$propertyList]
	,
	{19341, 0, 19341, 0.30000001192092896, 48800, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-30, 30}, {-12, 12}, {-12, 19}}, {61, 25, 32}, {-0.29999667406082153, 0.2999815046787262}, True, 0.1, {{-3., 3.}, {-1.2000000000000002, 1.2000000000000002}, {-1.2000000000000002, 1.9000000000000001}}, {6.1000000000000005, 2.5, 3.2}}	
]

VerificationTest[(* 22 *)
	OpenVDBLink`OpenVDBFilter[bmr, {"Median", 2}][$propertyList]
	,
	{24440, 0, 24440, 0.30000001192092896, 101680, False, "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-47, 34}, {-15, 15}, {-19, 20}}, {82, 31, 40}, {-0.2999989688396454, 0.29996877908706665}, True, 0.1, {{-4.7, 3.4000000000000004}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.200000000000001, 3.1, 4.}}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 23 *)
	$propertyList={"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox",     "IndexDimensions","MinMaxValues","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"};OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];MeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBFilter"]

VerificationTest[(* 24 *)
	OpenVDBLink`OpenVDBFilter[OpenVDBLink`OpenVDBFogVolume[vdb], "Mean"]
	,
	$Failed
	,
	{OpenVDBFilter::lvlsetgrid2}
]

VerificationTest[(* 25 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Mean"][$propertyList]
	,
	{25829, 0, 25829, 0.30000000000000004, 104160, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48, 35}, {-15, 15}, {-19, 20}}, {84, 31, 40}, {-0.2999912386442304, 0.29999993015807586}, True, 0.1, {{-4.800000000000001, 3.5}, {-1.5, 1.5}, {-1.9000000000000001, 2.}}, {8.4, 3.1, 4.}}	
]

VerificationTest[(* 26 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Gaussian"][$propertyList]
	,
	{21568, 0, 21568, 0.30000000000000004, 81548, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-41, 32}, {-14, 14}, {-18, 19}}, {74, 29, 38}, {-0.29994355630486597, 0.2999934886655026}, True, 0.1, {{-4.1000000000000005, 3.2}, {-1.4000000000000001, 1.4000000000000001}, {-1.8, 1.9000000000000001}}, {7.4, 2.9000000000000004, 3.8000000000000003}}	
]

VerificationTest[(* 27 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Laplacian"][$propertyList]
	,
	{20583, 0, 20583, 0.30000000000000004, 71928, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-39, 32}, {-13, 13}, {-17, 19}}, {72, 27, 37}, {-0.2999331703195918, 0.29997930395915146}, True, 0.1, {{-3.9000000000000004, 3.2}, {-1.3, 1.3}, {-1.7000000000000002, 1.9000000000000001}}, {7.2, 2.7, 3.7}}	
]

VerificationTest[(* 28 *)
	OpenVDBLink`OpenVDBFilter[vdb, "MeanCurvature"][$propertyList]
	,
	{19836, 0, 19836, 0.30000000000000004, 70929, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-38, 32}, {-13, 13}, {-17, 19}}, {71, 27, 37}, {-0.299966764506493, 0.299985965718042}, True, 0.1, {{-3.8000000000000003, 3.2}, {-1.3, 1.3}, {-1.7000000000000002, 1.9000000000000001}}, {7.1000000000000005, 2.7, 3.7}}	
]

VerificationTest[(* 29 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Median"][$propertyList]
	,
	{19163, 0, 19163, 0.30000000000000004, 66096, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-36, 31}, {-13, 13}, {-16, 19}}, {68, 27, 36}, {-0.2999681373324486, 0.2999869619312572}, True, 0.1, {{-3.6, 3.1}, {-1.3, 1.3}, {-1.6, 1.9000000000000001}}, {6.800000000000001, 2.7, 3.6}}	
]

VerificationTest[(* 30 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Mean", 2][$propertyList]
	,
	{17990, 0, 17990, 0.30000000000000004, 55250, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-33, 31}, {-12, 12}, {-14, 19}}, {65, 25, 34}, {-0.2998654133359068, 0.29999564031686216}, True, 0.1, {{-3.3000000000000003, 3.1}, {-1.2000000000000002, 1.2000000000000002}, {-1.4000000000000001, 1.9000000000000001}}, {6.5, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 31 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Gaussian", 2][$propertyList]
	,
	{15117, 0, 15117, 0.30000000000000004, 35420, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-26, 28}, {-11, 11}, {-9, 18}}, {55, 23, 28}, {-0.2999316267844502, 0.2999855883304682}, True, 0.1, {{-2.6, 2.8000000000000003}, {-1.1, 1.1}, {-0.9, 1.8}}, {5.5, 2.3000000000000003, 2.8000000000000003}}	
]

VerificationTest[(* 32 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Laplacian", 2][$propertyList]
	,
	{14592, 0, 14592, 0.30000000000000004, 34776, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-25, 28}, {-11, 11}, {-9, 18}}, {54, 23, 28}, {-0.2996867953729737, 0.29995669979135803}, True, 0.1, {{-2.5, 2.8000000000000003}, {-1.1, 1.1}, {-0.9, 1.8}}, {5.4, 2.3000000000000003, 2.8000000000000003}}	
]

VerificationTest[(* 33 *)
	OpenVDBLink`OpenVDBFilter[vdb, "MeanCurvature", 2][$propertyList]
	,
	{14322, 0, 14322, 0.30000000000000004, 34776, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-25, 28}, {-11, 11}, {-9, 18}}, {54, 23, 28}, {-0.29962514592760026, 0.2998941736270294}, True, 0.1, {{-2.5, 2.8000000000000003}, {-1.1, 1.1}, {-0.9, 1.8}}, {5.4, 2.3000000000000003, 2.8000000000000003}}	
]

VerificationTest[(* 34 *)
	OpenVDBLink`OpenVDBFilter[vdb, "Median", 2][$propertyList]
	,
	{14172, 0, 14172, 0.30000000000000004, 34132, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-25, 27}, {-11, 11}, {-9, 18}}, {53, 23, 28}, {-0.29995858307846673, 0.2999714684308279}, True, 0.1, {{-2.5, 2.7}, {-1.1, 1.1}, {-0.9, 1.8}}, {5.300000000000001, 2.3000000000000003, 2.8000000000000003}}	
]

VerificationTest[(* 35 *)
	OpenVDBLink`OpenVDBFilter[vdb, {"Mean", 2}][$propertyList]
	,
	{14943, 0, 14943, 0.30000000000000004, 34132, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-25, 27}, {-11, 11}, {-9, 18}}, {53, 23, 28}, {-0.299830334671523, 0.2999787093414914}, True, 0.1, {{-2.5, 2.7}, {-1.1, 1.1}, {-0.9, 1.8}}, {5.300000000000001, 2.3000000000000003, 2.8000000000000003}}	
]

VerificationTest[(* 36 *)
	OpenVDBLink`OpenVDBFilter[vdb, {"Gaussian", 2}][$propertyList]
	,
	{13639, 0, 13639, 0.30000000000000004, 28028, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-23, 25}, {-11, 10}, {-8, 17}}, {49, 22, 26}, {-0.2999987033762598, 0.29999435725242096}, True, 0.1, {{-2.3000000000000003, 2.5}, {-1.1, 1.}, {-0.8, 1.7000000000000002}}, {4.9, 2.2, 2.6}}	
]

VerificationTest[(* 37 *)
	OpenVDBLink`OpenVDBFilter[vdb, {"Median", 2}][$propertyList]
	,
	{11954, 0, 11954, 0.30000000000000004, 24675, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-22, 24}, {-10, 10}, {-7, 17}}, {47, 21, 25}, {-0.2999952119835797, 0.2999411013687887}, True, 0.1, {{-2.2, 2.4000000000000004}, {-1., 1.}, {-0.7000000000000001, 1.7000000000000002}}, {4.7, 2.1, 2.5}}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
