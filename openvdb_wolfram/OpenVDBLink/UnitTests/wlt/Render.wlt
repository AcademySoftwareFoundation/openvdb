BeginTestSection["Render Tests"]

BeginTestSection["Generic"]

BeginTestSection["Initialization"]

VerificationTest[(* 1 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;vdb=OpenVDBLink`OpenVDBLevelSet[ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"]];OpenVDBLink`OpenVDBScalarGridQ[vdb]
	,
	True	
]

EndTestSection[]

BeginTestSection["OpenVDBLevelSetViewer"]

VerificationTest[(* 2 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBLevelSetViewer]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 3 *)
	Attributes[OpenVDBLink`OpenVDBLevelSetViewer]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 4 *)
	Options[OpenVDBLink`OpenVDBLevelSetViewer]
	,
	{Background->Automatic, "ClosedClipping"->False, "FrameTranslation"->Automatic, ImageResolution->Automatic, "IsoValue"->0., "OrthographicFrame"->Automatic, PerformanceGoal:>$PerformanceGoal, ImageSize->Automatic, ViewAngle->Automatic, ViewCenter->Automatic, ViewPoint->{1.3, -2.4, 2.}, ViewProjection->Automatic, ViewRange->All, ViewVertical->{0, 0, 1}}	
]

VerificationTest[(* 5 *)
	SyntaxInformation[OpenVDBLink`OpenVDBLevelSetViewer]
	,
	{"ArgumentsPattern"->{_, _., OptionsPattern[]}}	
]

VerificationTest[(* 6 *)
	{OpenVDBLink`OpenVDBLevelSetViewer[], OpenVDBLink`OpenVDBLevelSetViewer["error"], OpenVDBLink`OpenVDBLevelSetViewer[vdb, "error"], OpenVDBLink`OpenVDBLevelSetViewer[vdb, "Soft", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBLevelSetViewer::argt, OpenVDBLevelSetViewer::scalargrid2, OpenVDBLevelSetViewer::shaderval, OpenVDBLevelSetViewer::nonopt}
]

VerificationTest[(* 7 *)
	(OpenVDBLink`OpenVDBLevelSetViewer[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBLevelSetViewer::scalargrid2, OpenVDBLevelSetViewer::scalargrid2, OpenVDBLevelSetViewer::scalargrid2, General::stop}
]

EndTestSection[]

BeginTestSection["OpenVDBLevelSetRender"]

VerificationTest[(* 8 *)
	OpenVDBLink`OpenVDBDefaultSpace[OpenVDBLink`OpenVDBLevelSetRender]
	,
	Missing["NotApplicable"]	
]

VerificationTest[(* 9 *)
	Attributes[OpenVDBLink`OpenVDBLevelSetRender]
	,
	{Protected, ReadProtected}	
]

VerificationTest[(* 10 *)
	Options[OpenVDBLink`OpenVDBLevelSetRender]
	,
	{Background->Automatic, "ClosedClipping"->False, "FrameTranslation"->Automatic, ImageResolution->Automatic, "IsoValue"->0., "OrthographicFrame"->Automatic, PerformanceGoal:>$PerformanceGoal, ImageSize->Automatic, ViewAngle->Automatic, ViewCenter->Automatic, ViewPoint->{1.3, -2.4, 2.}, ViewProjection->Automatic, ViewRange->All, ViewVertical->{0, 0, 1}}	
]

VerificationTest[(* 11 *)
	SyntaxInformation[OpenVDBLink`OpenVDBLevelSetRender]
	,
	{"ArgumentsPattern"->{_, _., OptionsPattern[]}}	
]

VerificationTest[(* 12 *)
	{OpenVDBLink`OpenVDBLevelSetRender[], OpenVDBLink`OpenVDBLevelSetRender["error"], OpenVDBLink`OpenVDBLevelSetRender[vdb, "error"], OpenVDBLink`OpenVDBLevelSetRender[vdb, "Soft", "error"]}
	,
	{$Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBLevelSetRender::argt, OpenVDBLevelSetRender::scalargrid2, OpenVDBLevelSetRender::shaderval, OpenVDBLevelSetRender::nonopt}
]

VerificationTest[(* 13 *)
	(OpenVDBLink`OpenVDBLevelSetRender[OpenVDBLink`OpenVDBCreateGrid[1., #1]]&)/@{"Int32", "Int64", "UInt32", "Vec2D", "Vec2I", "Vec2S", "Vec3D", "Vec3I", "Vec3S", "Boolean", "Mask"}
	,
	{$Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed, $Failed}
	,
	{OpenVDBLevelSetRender::scalargrid2, OpenVDBLevelSetRender::scalargrid2, OpenVDBLevelSetRender::scalargrid2, General::stop}
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Float"]

BeginTestSection["Initialization"]

VerificationTest[(* 14 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Scalar"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty], OpenVDBLink`OpenVDBScalarGridQ[vdb],    OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBLevelSetViewer"]

VerificationTest[(* 15 *)
	Head[OpenVDBLink`OpenVDBLevelSetViewer[vdb]]
	,
	DynamicModule	
]

VerificationTest[(* 16 *)
	OpenVDBLink`OpenVDBLevelSetViewer[fog]
	,
	$Failed
	,
	{OpenVDBLevelSetViewer::lvlsetgrid2}
]

VerificationTest[(* 17 *)
	OpenVDBLink`OpenVDBLevelSetViewer[vdbempty]
	,
	$Failed
	,
	{OpenVDBLevelSetViewer::lvlsetgrid2}
]

EndTestSection[]

BeginTestSection["OpenVDBLevelSetRender"]

VerificationTest[(* 18 *)
	ImageQ[OpenVDBLink`OpenVDBLevelSetRender[vdb]]
	,
	True	
]

VerificationTest[(* 19 *)
	OpenVDBLink`OpenVDBLevelSetRender[fog]
	,
	$Failed
	,
	{OpenVDBLevelSetRender::lvlsetgrid2}
]

VerificationTest[(* 20 *)
	OpenVDBLink`OpenVDBLevelSetRender[vdbempty]
	,
	$Failed
	,
	{OpenVDBLevelSetRender::lvlsetgrid2}
]

VerificationTest[(* 21 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, Background->Red, "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04, "OrthographicFrame"->Automatic,     PerformanceGoal->"Quality",ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->Automatic,ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9637830196855564, 0.02349328357738116, 0.03785025331238711}}	
]

VerificationTest[(* 22 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, {"Depth", Purple}, Background->Red, "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04, "OrthographicFrame"->Automatic,     PerformanceGoal->"Quality","ClosedClipping"->True,ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->Automatic,    ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9551883630046233, 0., 0.004016338707069198}}	
]

VerificationTest[(* 23 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, {"Matte", Purple, Green}, Background->Red, "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04, "OrthographicFrame"->Automatic,     PerformanceGoal->"Quality",ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->Automatic,ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9754757348417044, 0.00001487631242779759, 0.024310351755056272}}	
]

VerificationTest[(* 24 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, {"Matte", "Soft"}, Background->ColorData[112, 3], "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04,     "OrthographicFrame"->Automatic,PerformanceGoal->"Quality",ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->"Orthographic",    ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9754068654846159, 0.6310637327125854, 0.11472432831624234}}	
]

EndTestSection[]

EndTestSection[]

BeginTestSection["Double"]

BeginTestSection["Initialization"]

VerificationTest[(* 25 *)
	OpenVDBLink`$OpenVDBSpacing=0.1;OpenVDBLink`$OpenVDBHalfWidth=3.;bmr=ExampleData[{"Geometry3D", "Triceratops"}, "BoundaryMeshRegion"];vdbempty=OpenVDBLink`OpenVDBCreateGrid[1., "Double"];vdb=OpenVDBLink`OpenVDBLevelSet[bmr, "ScalarType"->"Double"];fog=OpenVDBLink`OpenVDBFogVolume[vdb];{BoundaryMeshRegionQ[bmr], OpenVDBLink`OpenVDBScalarGridQ[vdbempty],    OpenVDBLink`OpenVDBScalarGridQ[vdb],OpenVDBLink`OpenVDBScalarGridQ[fog]}
	,
	{True, True, True, True}	
]

EndTestSection[]

BeginTestSection["OpenVDBLevelSetViewer"]

VerificationTest[(* 26 *)
	Head[OpenVDBLink`OpenVDBLevelSetViewer[vdb]]
	,
	DynamicModule	
]

VerificationTest[(* 27 *)
	OpenVDBLink`OpenVDBLevelSetViewer[fog]
	,
	$Failed
	,
	{OpenVDBLevelSetViewer::lvlsetgrid2}
]

VerificationTest[(* 28 *)
	OpenVDBLink`OpenVDBLevelSetViewer[vdbempty]
	,
	$Failed
	,
	{OpenVDBLevelSetViewer::lvlsetgrid2}
]

EndTestSection[]

BeginTestSection["OpenVDBLevelSetRender"]

VerificationTest[(* 29 *)
	ImageQ[OpenVDBLink`OpenVDBLevelSetRender[vdb]]
	,
	True	
]

VerificationTest[(* 30 *)
	OpenVDBLink`OpenVDBLevelSetRender[fog]
	,
	$Failed
	,
	{OpenVDBLevelSetRender::lvlsetgrid2}
]

VerificationTest[(* 31 *)
	OpenVDBLink`OpenVDBLevelSetRender[vdbempty]
	,
	$Failed
	,
	{OpenVDBLevelSetRender::lvlsetgrid2}
]

VerificationTest[(* 32 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, Background->Red, "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04, "OrthographicFrame"->Automatic,     PerformanceGoal->"Quality",ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->Automatic,ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9637830196855564, 0.02349328357738116, 0.03785025331238711}}	
]

VerificationTest[(* 33 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, {"Depth", Purple}, Background->Red, "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04, "OrthographicFrame"->Automatic,     PerformanceGoal->"Quality","ClosedClipping"->True,ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->Automatic,    ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9551883630046233, 0., 0.004016338707069198}}	
]

VerificationTest[(* 34 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, {"Matte", Purple, Green}, Background->Red, "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04, "OrthographicFrame"->Automatic,     PerformanceGoal->"Quality",ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->Automatic,ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9754757348417044, 0.00001487631242779759, 0.024310351755056272}}	
]

VerificationTest[(* 35 *)
	im=OpenVDBLink`OpenVDBLevelSetRender[vdb, {"Matte", "Soft"}, Background->ColorData[112, 3], "FrameTranslation"->Automatic, ImageResolution->72, "IsoValue"->0.04,     "OrthographicFrame"->Automatic,PerformanceGoal->"Quality",ImageSize->243,ViewAngle->35*Degree,ViewCenter->Automatic,ViewPoint->{1., -2, 2},ViewProjection->"Orthographic",    ViewVertical->{0.1, -1, 0.45}];{ImageQ[im], ImageDimensions[im], Mean[im]}
	,
	{True, {243, 243}, {0.9754068654846159, 0.6310637327125854, 0.11472432831624234}}	
]

EndTestSection[]

EndTestSection[]

EndTestSection[]
