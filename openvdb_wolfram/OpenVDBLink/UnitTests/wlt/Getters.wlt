BeginTestSection["Getters Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBProperty"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBProperty]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBProperty]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBProperty]
	,
	{}	
	,
	{}
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBProperty]
	,
	{"ArgumentsPattern"->{_, _, _.}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBProperty[], OpenVDBLink`OpenVDBProperty["error"], OpenVDBLink`OpenVDBProperty[bmr, "error"], OpenVDBLink`OpenVDBProperty[bmr, "ActiveVoxelCount", "error"], OpenVDBLink`OpenVDBProperty[bmr, "ActiveVoxelCount", "RuleList", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBProperty::argt, OpenVDBProperty::argtu, OpenVDBProperty::prop, OpenVDBProperty::frmt, OpenVDBProperty::argt}
]

VerificationTest[(* 7 *)
	Head[OpenVDBLink`OpenVDBLevelSet[bmr]["PropertyValueGrid"]]
	,
	Grid	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 8 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBProperty"]

VerificationTest[(* 9 *)
	OpenVDBLink`OpenVDBProperty[vdb, "Properties"]===vdb["Properties"]
	,
	True	
]

VerificationTest[(* 10 *)
	vdb["Properties"]
	,
	{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "CreationDate", "Creator", "Description", "Empty", "ExpressionID", "GammaAdjustment", "GrayscaleWidth", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "LastModifiedDate", "MaxValue", "MemoryUsage", "MinValue", "MinMaxValues", "Name", "Properties", "PropertyValueGrid", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}	
]

VerificationTest[(* 11 *)
	OpenVDBLink`OpenVDBProperty[bmr, {"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment",    "GrayscaleWidth","GridClass","GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox",   "WorldDimensions"}]
	,
	{26073, 0, 26073, 0.30000001192092896, 110208, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-16, 15}, {-19, 21}}, {84, 32, 41}, 0.299950510263443, -0.2999959886074066, {-0.2999959886074066, 0.299950510263443}, Missing["NotAvailable"], True, 0.1, {{-4.800000000000001, 3.5}, {-1.6, 1.5}, {-1.9000000000000001, 2.1}}, {8.4, 3.2, 4.1000000000000005}}	
]

VerificationTest[(* 12 *)
	vdb[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{26073, 0, 26073, 0.30000001192092896, 110208, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], "LevelSet", "Tree_float_5_4_3", 3.0000001192092896, {{-48, 35}, {-16, 15}, {-19, 21}}, {84, 32, 41}, 0.299950510263443, -0.2999959886074066, {-0.2999959886074066, 0.299950510263443}, Missing["NotAvailable"], True, 0.1, {{-4.800000000000001, 3.5}, {-1.6, 1.5}, {-1.9000000000000001, 2.1}}, {8.4, 3.2, 4.1000000000000005}}	
]

VerificationTest[(* 13 *)
	fog[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{12052, 0, 12052, 0., 66300, Missing["NotAvailable"], Missing["NotAvailable"], False, 1., 3., "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, 1., 0.00010090569412568584, {0.00010090569412568584, 1.}, Missing["NotAvailable"], True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 14 *)
	{IntegerQ[vdb["ExpressionID"]], DateObjectQ[vdb["CreationDate"]], DateObjectQ[vdb["LastModifiedDate"]]}
	,
	{True, True, True}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 15 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdb],    OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBProperty"]

VerificationTest[(* 16 *)
	OpenVDBLink`OpenVDBProperty[vdb, "Properties"]===vdb["Properties"]
	,
	True	
]

VerificationTest[(* 17 *)
	vdb["Properties"]
	,
	{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "CreationDate", "Creator", "Description", "Empty", "ExpressionID", "GammaAdjustment", "GrayscaleWidth", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "LastModifiedDate", "MaxValue", "MemoryUsage", "MinValue", "MinMaxValues", "Name", "Properties", "PropertyValueGrid", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}	
]

VerificationTest[(* 18 *)
	OpenVDBLink`OpenVDBProperty[vdb, {"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment",    "GrayscaleWidth","GridClass","GridType","HalfWidth","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize",   "WorldBoundingBox","WorldDimensions"}]
	,
	{26073, 0, 26073, 0.30000000000000004, 110208, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, 3.0000000000000004, {{-48, 35}, {-16, 15}, {-19, 21}}, {84, 32, 41}, 0.29995050821018665, -0.29999599125227844, {-0.29999599125227844, 0.29995050821018665}, Missing["NotAvailable"], True, 0.1, {{-4.800000000000001, 3.5}, {-1.6, 1.5}, {-1.9000000000000001, 2.1}}, {8.4, 3.2, 4.1000000000000005}}	
]

VerificationTest[(* 19 *)
	vdb[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{26073, 0, 26073, 0.30000000000000004, 110208, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48, 35}, {-16, 15}, {-19, 21}}, {84, 32, 41}, 0.29995050821018665, -0.29999599125227844, {-0.29999599125227844, 0.29995050821018665}, Missing["NotAvailable"], True, 0.1, {{-4.800000000000001, 3.5}, {-1.6, 1.5}, {-1.9000000000000001, 2.1}}, {8.4, 3.2, 4.1000000000000005}}	
]

VerificationTest[(* 20 *)
	fog[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{12052, 0, 12052, 0., 66300, Missing["NotAvailable"], Missing["NotAvailable"], False, 1., 3., "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, 1., 0.00010091167505198635, {0.00010091167505198635, 1.}, Missing["NotAvailable"], True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 21 *)
	{IntegerQ[vdb["ExpressionID"]], DateObjectQ[vdb["CreationDate"]], DateObjectQ[vdb["LastModifiedDate"]]}
	,
	{True, True, True}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Integer"]

BeginTestSection["Initialization"]

VerificationTest[(* 22 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Int64"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Range[10]];OpenVDBLink`OpenVDBIntegerGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBProperty"]

VerificationTest[(* 23 *)
	OpenVDBLink`OpenVDBProperty[vdb, "Properties"]===vdb["Properties"]
	,
	True	
]

VerificationTest[(* 24 *)
	vdb["Properties"]
	,
	{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "CreationDate", "Creator", "Description", "Empty", "ExpressionID", "GammaAdjustment", "GrayscaleWidth", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "LastModifiedDate", "MaxValue", "MemoryUsage", "MinValue", "MinMaxValues", "Name", "Properties", "PropertyValueGrid", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}	
]

VerificationTest[(* 25 *)
	OpenVDBLink`OpenVDBProperty[vdb, {"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment",    "GrayscaleWidth","GridClass","GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox",   "WorldDimensions"}]
	,
	{20, 0, 20, 0, 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_int64_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, 10, 1, {1, 10}, Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 26 *)
	vdb[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{20, 0, 20, 0, 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_int64_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, 10, 1, {1, 10}, Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 27 *)
	{IntegerQ[vdb["ExpressionID"]], DateObjectQ[vdb["CreationDate"]], DateObjectQ[vdb["LastModifiedDate"]]}
	,
	{True, True, True}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Vector"]

BeginTestSection["Initialization"]

VerificationTest[(* 28 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Vec3S"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[{12 - i, 13 - i, i}, {i, 10}]];OpenVDBLink`OpenVDBVectorGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBProperty"]

VerificationTest[(* 29 *)
	OpenVDBLink`OpenVDBProperty[vdb, "Properties"]===vdb["Properties"]
	,
	True	
]

VerificationTest[(* 30 *)
	vdb["Properties"]
	,
	{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "CreationDate", "Creator", "Description", "Empty", "ExpressionID", "GammaAdjustment", "GrayscaleWidth", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "LastModifiedDate", "MaxValue", "MemoryUsage", "MinValue", "MinMaxValues", "Name", "Properties", "PropertyValueGrid", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}	
]

VerificationTest[(* 31 *)
	OpenVDBLink`OpenVDBProperty[vdb, {"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment",    "GrayscaleWidth","GridClass","GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox",   "WorldDimensions"}]
	,
	{20, 0, 20, {0., 0., 0.}, 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_vec3s_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, {11., 12., 1.}, {2., 3., 10.}, {{2., 3., 10.}, {11., 12., 1.}}, Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 32 *)
	vdb[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{20, 0, 20, {0., 0., 0.}, 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_vec3s_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, {11., 12., 1.}, {2., 3., 10.}, {{2., 3., 10.}, {11., 12., 1.}}, Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 33 *)
	{IntegerQ[vdb["ExpressionID"]], DateObjectQ[vdb["CreationDate"]], DateObjectQ[vdb["LastModifiedDate"]]}
	,
	{True, True, True}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Boolean"]

BeginTestSection["Initialization"]

VerificationTest[(* 34 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Boolean"];OpenVDBLink`OpenVDBSetValues[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetValues[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[EvenQ[i], {i, 10}]];OpenVDBLink`OpenVDBBooleanGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBProperty"]

VerificationTest[(* 35 *)
	OpenVDBLink`OpenVDBProperty[vdb, "Properties"]===vdb["Properties"]
	,
	True	
]

VerificationTest[(* 36 *)
	vdb["Properties"]
	,
	{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "CreationDate", "Creator", "Description", "Empty", "ExpressionID", "GammaAdjustment", "GrayscaleWidth", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "LastModifiedDate", "MaxValue", "MemoryUsage", "MinValue", "MinMaxValues", "Name", "Properties", "PropertyValueGrid", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}	
]

VerificationTest[(* 37 *)
	OpenVDBLink`OpenVDBProperty[vdb, {"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment",    "GrayscaleWidth","GridClass","GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox",   "WorldDimensions"}]
	,
	{20, 0, 20, 0, 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_bool_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, 1, 0, {0, 1}, Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 38 *)
	vdb[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{20, 0, 20, 0, 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_bool_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, 1, 0, {0, 1}, Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 39 *)
	{IntegerQ[vdb["ExpressionID"]], DateObjectQ[vdb["CreationDate"]], DateObjectQ[vdb["LastModifiedDate"]]}
	,
	{True, True, True}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Mask"]

BeginTestSection["Initialization"]

VerificationTest[(* 40 *)
	vdb=OpenVDBLink`OpenVDBCreateGrid[1., "Mask"];OpenVDBLink`OpenVDBSetStates[vdb, Table[{i, i, i}, {i, 10}], Table[Mod[i, 3], {i, 10}]];OpenVDBLink`OpenVDBSetStates[vdb, Table[{11 - i, 11 - i, i}, {i, 10}], Table[Mod[i, 2], {i, 10}]];OpenVDBLink`OpenVDBMaskGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBProperty"]

VerificationTest[(* 41 *)
	OpenVDBLink`OpenVDBProperty[vdb, "Properties"]===vdb["Properties"]
	,
	True	
]

VerificationTest[(* 42 *)
	vdb["Properties"]
	,
	{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "CreationDate", "Creator", "Description", "Empty", "ExpressionID", "GammaAdjustment", "GrayscaleWidth", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox", "IndexDimensions", "LastModifiedDate", "MaxValue", "MemoryUsage", "MinValue", "MinMaxValues", "Name", "Properties", "PropertyValueGrid", "UniformVoxels", "VoxelSize", "WorldBoundingBox", "WorldDimensions"}	
]

VerificationTest[(* 43 *)
	OpenVDBLink`OpenVDBProperty[vdb, {"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment",    "GrayscaleWidth","GridClass","GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox",   "WorldDimensions"}]
	,
	{12, 0, 12, Missing["NotApplicable"], 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_mask_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 44 *)
	vdb[{"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Creator", "Description", "Empty", "GammaAdjustment", "GrayscaleWidth", "GridClass",    "GridType","HalfWidth","IndexBoundingBox","IndexDimensions","MaxValue","MinValue","MinMaxValues","Name","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"}]
	,
	{12, 0, 12, Missing["NotApplicable"], 1000, Missing["NotAvailable"], Missing["NotAvailable"], False, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], "Tree_mask_5_4_3", Missing["NotApplicable"], {{1, 10}, {1, 10}, {1, 10}}, {10, 10, 10}, Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotApplicable"], Missing["NotAvailable"], True, 1., {{1., 10.}, {1., 10.}, {1., 10.}}, {10., 10., 10.}}	
]

VerificationTest[(* 45 *)
	{IntegerQ[vdb["ExpressionID"]], DateObjectQ[vdb["CreationDate"]], DateObjectQ[vdb["LastModifiedDate"]]}
	,
	{True, True, True}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
