BeginTestSection["FogVolume Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBFogVolume"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBFogVolume]
	,
	"Index"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBFogVolume]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBFogVolume]
	,
	{}	
	,
	{}
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBFogVolume]
	,
	{"ArgumentsPattern"->{_, _.}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBFogVolume[], OpenVDBLink`OpenVDBFogVolume["error"], OpenVDBLink`OpenVDBFogVolume[bmr, "error"], OpenVDBLink`OpenVDBFogVolume[bmr, 0, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBFogVolume::argt, OpenVDBFogVolume::scalargrid2, OpenVDBFogVolume::argt}
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBFogVolume[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBFogVolume::scalargrid2, OpenVDBFogVolume::scalargrid2, OpenVDBFogVolume::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBToFogVolume"]

VerificationTest[(* 8 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBToFogVolume]
	,
	"Index"	
]

VerificationTest[(* 9 *)
	Attributes[OpenVDBLink`OpenVDBToFogVolume]
	,
	{Protected, ReadProtected}	
	,
	{}
]

VerificationTest[(* 10 *)
	Options[OpenVDBLink`OpenVDBToFogVolume]
	,
	{}	
	,
	{}
]

VerificationTest[(* 11 *)
	SyntaxInformation[OpenVDBLink`OpenVDBToFogVolume]
	,
	{"ArgumentsPattern"->{_, _.}}	
]

VerificationTest[(* 12 *)
	{OpenVDBLink`OpenVDBToFogVolume[], OpenVDBLink`OpenVDBToFogVolume["error"], OpenVDBLink`OpenVDBToFogVolume[bmr, "error"], OpenVDBLink`OpenVDBToFogVolume[bmr, 0, "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBToFogVolume::argt, OpenVDBToFogVolume::scalargrid2, OpenVDBToFogVolume::argt}
]

VerificationTest[(* 13 *)
	(OpenVDBLink`OpenVDBToFogVolume[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBToFogVolume::scalargrid2, OpenVDBToFogVolume::scalargrid2, OpenVDBToFogVolume::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 14 *)
	$propertyList={"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox",     "IndexDimensions","MinMaxValues","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"};OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];MeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBFogVolume"]

VerificationTest[(* 15 *)
	OpenVDBLink`OpenVDBFogVolume[bmr][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00010090569412568584, 1.}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 16 *)
	OpenVDBLink`OpenVDBFogVolume[bmr, Automatic][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00010090569412568584, 1.}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 17 *)
	OpenVDBLink`OpenVDBFogVolume[bmr, 2][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00015135854482650757, 1.5}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 18 *)
	OpenVDBLink`OpenVDBFogVolume[bmr, 0.21->"World"][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.0001441509957658127, 1.4285714626312256}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 19 *)
	OpenVDBLink`OpenVDBFogVolume[bmr, 1->"Index"][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00030271708965301514, 3.}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 20 *)
	vdb=OpenVDBLink`OpenVDBLevelSet[bmr];OpenVDBLink`OpenVDBToFogVolume[vdb, 1->"Index"];vdb[$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_float_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00030271708965301514, 3.}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 21 *)
	$propertyList={"ActiveLeafVoxelCount", "ActiveTileCount", "ActiveVoxelCount", "BackgroundValue", "BoundingGridVoxelCount", "Empty", "GridClass", "GridType", "HalfWidth", "IndexBoundingBox",     "IndexDimensions","MinMaxValues","UniformVoxels","VoxelSize","WorldBoundingBox","WorldDimensions"};OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "MeshRegion"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];MeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBFogVolume"]

VerificationTest[(* 22 *)
	OpenVDBLink`OpenVDBFogVolume[vdb][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00010091167505198635, 1.}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 23 *)
	OpenVDBLink`OpenVDBFogVolume[vdb, Automatic][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00010091167505198635, 1.}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 24 *)
	OpenVDBLink`OpenVDBFogVolume[vdb, 2][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00015136751257797953, 1.5000000000000002}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 25 *)
	OpenVDBLink`OpenVDBFogVolume[vdb, 0.21->"World"][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00014415953578855194, 1.4285714285714288}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 26 *)
	OpenVDBLink`OpenVDBFogVolume[vdb, 1->"Index"][$propertyList]
	,
	{12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00030273502515595907, 3.0000000000000004}, True, 0.1, {{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}}, {7.800000000000001, 2.5, 3.4000000000000004}}	
]

VerificationTest[(* 27 *)
	vdb2=OpenVDBLink`OpenVDBCopyGrid[vdb];OpenVDBLink`OpenVDBToFogVolume[vdb2, 1->"Index"];{vdb[$propertyList], vdb2[$propertyList]}
	,
	{{26073, 0, 26073, 0.30000000000000004, 110208, False, "LevelSet", "Tree_double_5_4_3", 3.0000000000000004, {{-48, 35}, {-16, 15}, {-19, 21}}, {84, 32, 41},    {-0.29999599125227844, 0.29995050821018665},True,0.1,{{-4.800000000000001, 3.5}, {-1.6, 1.5}, {-1.9000000000000001, 2.1}},{8.4, 3.2, 4.1000000000000005}}, {12052, 0, 12052, 0., 66300, False, "FogVolume", "Tree_double_5_4_3", Missing["NotApplicable"], {{-45, 32}, {-12, 12}, {-16, 17}}, {78, 25, 34}, {0.00030273502515595907, 3.0000000000000004}, True,    0.1,{{-4.5, 3.2}, {-1.2000000000000002, 1.2000000000000002}, {-1.6, 1.7000000000000002}},{7.800000000000001, 2.5, 3.4000000000000004}}}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
