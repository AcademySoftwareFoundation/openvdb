BeginTestSection["Measure Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];BoundaryMeshRegionQ[bmr]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBArea"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBArea]
	,
	"World"	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBArea]
	,
	{}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBArea]
	,
	{}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBArea]
	,
	{"ArgumentsPattern"->{_, _.}}	
]

VerificationTest[(* 6 *)
	{
OpenVDBLink`OpenVDBArea[], 
OpenVDBLink`OpenVDBArea["error"],
OpenVDBLink`OpenVDBArea[bmr, "error"],
OpenVDBLink`OpenVDBArea[bmr, "World", "error"]
}
	,
	{$Failed, $Failed, $Failed, $Failed}	
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBArea[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

BeginTestSection["OpenVDBEulerCharacteristic"]

VerificationTest[(* 8 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBEulerCharacteristic]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 9 *)
	Attributes[OpenVDBLink`OpenVDBEulerCharacteristic]
	,
	{}	
]

VerificationTest[(* 10 *)
	Options[OpenVDBLink`OpenVDBEulerCharacteristic]
	,
	{}	
]

VerificationTest[(* 11 *)
	SyntaxInformation[OpenVDBLink`OpenVDBEulerCharacteristic]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 12 *)
	{
OpenVDBLink`OpenVDBEulerCharacteristic[], 
OpenVDBLink`OpenVDBEulerCharacteristic["error"],
OpenVDBLink`OpenVDBEulerCharacteristic[bmr, "error"]
}
	,
	{$Failed, $Failed, $Failed}	
]

VerificationTest[(* 13 *)
	(OpenVDBLink`OpenVDBEulerCharacteristic[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

BeginTestSection["OpenVDBGenus"]

VerificationTest[(* 14 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBGenus]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 15 *)
	Attributes[OpenVDBLink`OpenVDBGenus]
	,
	{}	
]

VerificationTest[(* 16 *)
	Options[OpenVDBLink`OpenVDBGenus]
	,
	{}	
]

VerificationTest[(* 17 *)
	SyntaxInformation[OpenVDBLink`OpenVDBGenus]
	,
	{"ArgumentsPattern"->{_}}	
]

VerificationTest[(* 18 *)
	{
OpenVDBLink`OpenVDBGenus[], 
OpenVDBLink`OpenVDBGenus["error"],
OpenVDBLink`OpenVDBGenus[bmr, "error"]
}
	,
	{$Failed, $Failed, $Failed}	
]

VerificationTest[(* 19 *)
	(OpenVDBLink`OpenVDBGenus[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

BeginTestSection["OpenVDBVolume"]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBVolume]
	,
	"World"	
]

VerificationTest[(* 21 *)
	Attributes[OpenVDBLink`OpenVDBVolume]
	,
	{}	
]

VerificationTest[(* 22 *)
	Options[OpenVDBLink`OpenVDBVolume]
	,
	{}	
]

VerificationTest[(* 23 *)
	SyntaxInformation[OpenVDBLink`OpenVDBVolume]
	,
	{"ArgumentsPattern"->{_, _.}}	
]

VerificationTest[(* 24 *)
	{
OpenVDBLink`OpenVDBVolume[], 
OpenVDBLink`OpenVDBVolume["error"],
OpenVDBLink`OpenVDBVolume[bmr, "error"],
OpenVDBLink`OpenVDBVolume[bmr, "World", "error"]
}
	,
	{$Failed, $Failed, $Failed, $Failed}	
]

VerificationTest[(* 25 *)
	(OpenVDBLink`OpenVDBVolume[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 26 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBArea"]

VerificationTest[(* 27 *)
	OpenVDBLink`OpenVDBArea/@{bmr, vdbempty, vdb, fog}
	,
	{41.20717458005259, Indeterminate, 41.20717458005259, Indeterminate}	
]

EndTestSection[]

BeginTestSection["OpenVDBEulerCharacteristic"]

VerificationTest[(* 28 *)
	OpenVDBLink`OpenVDBEulerCharacteristic/@{bmr, vdbempty, vdb, fog}
	,
	{11, Indeterminate, 11, Indeterminate}	
]

EndTestSection[]

BeginTestSection["OpenVDBGenus"]

VerificationTest[(* 29 *)
	OpenVDBLink`OpenVDBGenus/@{bmr, vdbempty, vdb, fog}
	,
	{-4, Indeterminate, -4, Indeterminate}	
]

EndTestSection[]

BeginTestSection["OpenVDBVolume"]

VerificationTest[(* 30 *)
	OpenVDBLink`OpenVDBVolume/@{bmr, vdbempty, vdb, fog}
	,
	{12.213059193871851, Indeterminate, 12.213059193871851, Indeterminate}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 31 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Double"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb], OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBArea"]

VerificationTest[(* 32 *)
	OpenVDBLink`OpenVDBArea/@{bmr, vdbempty, vdb, fog}
	,
	{41.20717458005259, Indeterminate, 41.20717496672823, Indeterminate}	
]

EndTestSection[]

BeginTestSection["OpenVDBEulerCharacteristic"]

VerificationTest[(* 33 *)
	OpenVDBLink`OpenVDBEulerCharacteristic/@{bmr, vdbempty, vdb, fog}
	,
	{11, Indeterminate, 11, Indeterminate}	
]

EndTestSection[]

BeginTestSection["OpenVDBGenus"]

VerificationTest[(* 34 *)
	OpenVDBLink`OpenVDBGenus/@{bmr, vdbempty, vdb, fog}
	,
	{-4, Indeterminate, -4, Indeterminate}	
]

EndTestSection[]

BeginTestSection["OpenVDBVolume"]

VerificationTest[(* 35 *)
	OpenVDBLink`OpenVDBVolume/@{bmr, vdbempty, vdb, fog}
	,
	{12.213059193871851, Indeterminate, 12.213059351349573, Indeterminate}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
